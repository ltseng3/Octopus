#!/usr/bin/env python3
"""Capture actual synchronized per-vehicle predictions from V2V4Real/OpenCOOD.

RUN THIS SCRIPT INSIDE THE OFFICIAL V2V4REAL ENVIRONMENT, with V2V4Real's
`opencood` importable and with the dataset and both model checkpoints installed.
Not executed as part of OCTOPUS's CPU-only unit tests: upstream model/CUDA
extensions and the full real-world dataset are not present in this release.

This script intentionally invokes each car's model separately (no fusion).
It then post-processes *that car only* into a common reference ego-LiDAR
frame, and exports the per-agent predictions, one shared GT, and a manifest.
The model outputs are genuine only if the user supplies trained checkpoints.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--edge-model-dir', required=True)
    parser.add_argument('--local-model-dir', required=True)
    parser.add_argument('--data-dir', required=True, help='OPV2V-format split, e.g. v2v4real/test')
    parser.add_argument('--dataset', default='V2V4Real', choices=['V2V4Real', 'OPV2V'])
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-frames', type=int, default=100)
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--stride', type=int, default=1, help='Record every Nth eligible frame')
    args = parser.parse_args()
    if Path(args.local_model_dir).resolve() == Path(args.edge_model_dir).resolve():
        parser.error('local and edge checkpoints must be distinct paths; separate executions of identical checkpoints are not independent evidence')
    if args.stride < 1:
        parser.error('--stride must be >=1')

    # Import heavy dependencies only inside the upstream environment.
    import numpy as np
    import torch
    import opencood.hypes_yaml.yaml_utils as yaml_utils
    from opencood.tools import train_utils
    from opencood.data_utils.datasets import build_dataset

    if args.max_frames < 1:
        parser.error('--max-frames must be >=1')
    device = torch.device('cuda' if args.device == 'cuda' or
                          (args.device == 'auto' and torch.cuda.is_available()) else 'cpu')
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('--device cuda requested but CUDA is unavailable')

    def load_ckpt(dirname):
        opt = argparse.Namespace(model_dir=str(Path(dirname).resolve()))
        hyp = yaml_utils.load_yaml(None, opt)
        hyp['validate_dir'] = str(Path(args.data_dir).resolve())
        hyp.setdefault('wild_setting', {})
        # Disable artificially introduced asynchronous and GPS noise in source
        # export: separate simulator sweeps should control these mechanisms.
        hyp['wild_setting'].update({'async': False, 'loc_err': False})
        model = train_utils.create_model(hyp)
        _, model = train_utils.load_saved_model(dirname, model)
        model = model.to(device)
        model.eval()
        return hyp, model

    edge_hyp, edge_model = load_ckpt(args.edge_model_dir)
    local_hyp, local_model = load_ckpt(args.local_model_dir)
    # Dataset preprocessing must be compatible with *both* checkpoint models.
    if edge_hyp['preprocess'] != local_hyp['preprocess'] or edge_hyp['postprocess'] != local_hyp['postprocess']:
        raise ValueError('Model preprocessing/postprocessing differs; run separate exports '
                         'and align their world-space boxes before import')
    if edge_hyp.get('fusion', {}).get('core_method', '').lower() not in ('latefusiondataset', ''):
        raise ValueError('This capture script targets LateFusionDataset / per-CAV postprocessing')

    try:
        dataset = build_dataset(edge_hyp, visualize=False, train=False, isSim=(args.dataset == 'OPV2V'))
    except TypeError:
        dataset = build_dataset(edge_hyp, visualize=False, train=False)
    if dataset.__class__.__name__ != 'LateFusionDataset':
        raise ValueError(f'Need LateFusionDataset, received {dataset.__class__.__name__}; '
                         'per-CAV independent outputs require late-fusion test preprocessing')

    dest = Path(args.output).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    out_file = dest / 'frames.jsonl'
    files_dir = dest / 'boxes'
    files_dir.mkdir(exist_ok=True)
    count = 0
    # Skip user-specified intraframe delay; timestamp mapping comes from the
    # actual dataset scenario database, not the dataloader's sequential index.
    with out_file.open('w', encoding='utf-8') as out, torch.no_grad():
        for idx in range(0, len(dataset), args.stride):
            if count >= args.max_frames:
                break
            scenario_idx = next(i for i, count_end in enumerate(dataset.len_record) if idx < count_end)
            scenario_database = dataset.scenario_database[scenario_idx]
            relative_idx = idx if scenario_idx == 0 else idx - dataset.len_record[scenario_idx-1]
            frame_id = dataset.return_timestamp_key(scenario_database, relative_idx)
            scene_id = Path(dataset.scenario_folders[scenario_idx]).name
            for cav_id, cav_info in scenario_database.items():
                if frame_id not in cav_info:
                    raise ValueError(f'Incomplete synchronization: {scene_id}/{frame_id} not present for CAV {cav_id}')
            raw_data = dataset.retrieve_base_data(idx)
            cav_ids = tuple(str(a) for a in raw_data)
            if len(cav_ids) < 2:
                continue
            ego_id = next(a for a in cav_ids if raw_data[a]['ego'])
            ref_pose = list(map(float, raw_data[ego_id]['params']['lidar_pose']))

            sample = dataset.collate_batch_test([dataset[idx]])
            sample = train_utils.to_device(sample, device)
            # Object detection annotations across all cars are de-duplicated
            # by the upstream postprocessor and referenced to the ego frame.
            gt = dataset.post_processor.generate_gt_bbx(sample)
            gt_array = (gt.detach().cpu().numpy() if gt is not None else
                        np.empty((0,8,3), dtype=np.float32))
            prefix = f'frame_{idx:06d}'
            gt_path = files_dir / f'{prefix}_gt.npy'
            np.save(gt_path, gt_array)
            agents = []
            for cav_id in cav_ids:
                sample_id = 'ego' if cav_id == ego_id else cav_id
                if sample_id not in sample:
                    raise RuntimeError(f'No preprocessed input for CAV {cav_id} in {scene_id}/{frame_id}')
                content = sample[sample_id]
                pred_files = {}
                for kind, model in (('local', local_model), ('edge', edge_model)):
                    prediction_output = model(content)
                    pred, _ = dataset.post_processor.post_process(sample, {sample_id: prediction_output})
                    pred_array = (pred.detach().cpu().numpy() if pred is not None else
                                  np.empty((0,8,3), dtype=np.float32))
                    pred_path = files_dir / f'{prefix}_{cav_id}_{kind}.npy'
                    np.save(pred_path, pred_array)
                    pred_files[f'{kind}_prediction'] = {
                        'path': str(pred_path.relative_to(dest)),
                        'coordinates': 'reference_lidar',
                    }
                agents.append({
                    'agent_id': cav_id,
                    'lidar_pose': list(map(float, raw_data[cav_id]['params']['lidar_pose'])),
                    **pred_files,
                })
            record = {
                'dataset': args.dataset, 'scene_id': scene_id, 'frame_id': str(frame_id),
                'reference_lidar_pose': ref_pose,
                'ground_truth': {'path': str(gt_path.relative_to(dest)), 'coordinates': 'reference_lidar'},
                'agents': agents,
            }
            out.write(json.dumps(record, sort_keys=True) + '\n')
            count += 1
            print(f'[{count}] {scene_id}/{frame_id}: {len(agents)} aligned vehicles')
    meta = {
        'source': 'real model outputs from official OPV2V-format dataset / V2V4Real codebase',
        'dataset': args.dataset, 'split': str(Path(args.data_dir).resolve()),
        'edge_checkpoint': str(Path(args.edge_model_dir).resolve()),
        'local_checkpoint': str(Path(args.local_model_dir).resolve()),
        'frames_exported': count, 'model_inference_device': str(device),
        'postprocessing': 'independent per-CAV no-fusion outputs in reference ego coordinate frame',
        'limitations': 'Requires external trained checkpoints and official V2V4Real environment',
    }
    (dest/'capture_metadata.json').write_text(json.dumps(meta, indent=2)+'\n')
    if count == 0:
        raise ValueError('No frames with >=2 synchronized CAVs found')
    print(f'Frames: {out_file}')
    print('Next: python -m octopus_sim import-opv2v '+str(out_file)+' --output outputs/empirical')


if __name__ == '__main__':
    main()
