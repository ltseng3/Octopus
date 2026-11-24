"""Strict importer of recorded OPV2V/V2V4Real multi-agent box predictions.

Inputs MUST contain actual model-generated per-agent prediction files for both
candidate locations. This module does not generate neural predictions.

The common decision is binary object presence in a world-coordinate circular ROI.
For each synchronized (scene, frame), ROI centers are calculated from the mean
reported vehicle sensor locations plus fixed offsets. Every candidate is tested
against the *same* world-space ROI and *same* ground truth, avoiding the common
error of comparing agent-specific boxes in unrelated coordinate systems.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any


class ImportFailure(ValueError):
    pass


@dataclass(frozen=True)
class ImportReport:
    source_frames: int
    valid_events: int
    positive_events: int
    negative_events: int
    skipped_outside_coverage: int
    input_files: int
    trace_path: Path
    provenance_path: Path
    report_path: Path


def _load_manifest(path: Path) -> list[dict]:
    frames: list[dict] = []
    seen: set[tuple[str, str]] = set()
    with path.open(encoding='utf-8') as f:
        for idx, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                frame = json.loads(line)
                if not isinstance(frame, dict):
                    raise ImportFailure('frame must be an object')
                pair = (frame['scene_id'], str(frame['frame_id']))
                if not isinstance(pair[0], str) or not pair[0] or not pair[1]:
                    raise ImportFailure('invalid scene or frame ID')
                if pair in seen:
                    raise ImportFailure(f'duplicate synchronized frame {pair}')
                seen.add(pair)
                frames.append(frame)
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise ImportFailure(f'{path}:{idx}: invalid frame: {exc}') from exc
    if not frames:
        raise ImportFailure('manifest has no synchronized frames')
    return frames


def _as_pose(value: Any, context: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != 6:
        raise ImportFailure(f'{context}: pose must be [x,y,z,roll,yaw,pitch]')
    try:
        pose = tuple(float(v) for v in value)
        if not all(math.isfinite(v) for v in pose):
            raise ValueError('nonfinite pose')
        return pose
    except (TypeError, ValueError) as exc:
        raise ImportFailure(f'{context}: invalid pose: {exc}') from exc


def _as_xy(item: Any, path: Path) -> list[tuple[float, float, float]]:
    """Parse numeric [N,8,3] corners, [N,4,2] corners or [N,7]/[N,3] centers.

    For 3D box corners, use the mean of the 8 corners. Shape checks are strict.
    Arrays in .npy files are read with allow_pickle=False (security boundary).
    """
    if hasattr(item, 'tolist'):
        item = item.tolist()
    if isinstance(item, dict):
        item = item.get('boxes')
    if not isinstance(item, list):
        raise ImportFailure(f'{path}: boxes must be a list or array')
    centers = []
    for i, box in enumerate(item):
        if not isinstance(box, list):
            raise ImportFailure(f'{path}: box {i} must be an array')
        if len(box) in (4, 8) and all(isinstance(x, list) for x in box):
            cols = len(box[0])
            if cols not in (2, 3) or any(len(x) != cols for x in box):
                raise ImportFailure(f'{path}: invalid box corner shape {i}')
            try:
                xyz = [sum(float(p[d]) for p in box) / len(box) for d in range(cols)]
            except (TypeError, ValueError) as exc:
                raise ImportFailure(f'{path}: invalid corner coordinate: {exc}') from exc
            if cols == 2:
                xyz.append(0.0)
        elif len(box) in (3, 7) and all(not isinstance(x, list) for x in box):
            try:
                xyz = [float(box[j]) for j in range(3)]
            except (TypeError, ValueError) as exc:
                raise ImportFailure(f'{path}: invalid box center: {exc}') from exc
        else:
            raise ImportFailure(f'{path}: unsupported box shape {i}; use [8,3], [4,2], [N,7] or [N,3]')
        if not all(math.isfinite(v) for v in xyz):
            raise ImportFailure(f'{path}: nonfinite box center')
        centers.append(tuple(xyz))
    return centers


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _read_boxes(spec: dict, *, manifest_dir: Path, context: str,
                files: dict[Path, str]) -> list[tuple[float, float, float]]:
    if not isinstance(spec, dict):
        raise ImportFailure(f'{context}: box source must be an object')
    target = spec.get('path')
    if not isinstance(target, str) or not target:
        raise ImportFailure(f'{context}: box path is required')
    path = (manifest_dir / target).resolve()
    if not path.is_file():
        raise ImportFailure(f'{context}: missing data file {path}')
    if path.suffix.lower() == '.npy':
        try:
            import numpy as np
        except ImportError as exc:
            raise ImportFailure('NumPy is required for .npy prediction exports') from exc
        try:
            obj = np.load(path, allow_pickle=False)
        except (ValueError, OSError) as exc:
            raise ImportFailure(f'{context}: cannot read .npy: {exc}') from exc
    elif path.suffix.lower() == '.json':
        try:
            obj = json.loads(path.read_text(encoding='utf-8'))
        except json.JSONDecodeError as exc:
            raise ImportFailure(f'{context}: invalid JSON: {exc}') from exc
    else:
        raise ImportFailure(f'{context}: supported boxes: .npy, .json')
    files[path] = _sha(path)
    return _as_xy(obj, path)


def _to_world(point: tuple[float, float, float], pose: tuple[float, ...]) -> tuple[float, float, float]:
    """OpenCOOD / CARLA x_to_world convention, [x,y,z,roll,yaw,pitch] degrees."""
    x, y, z, roll, yaw, pitch = pose
    cr, sr = math.cos(math.radians(roll)), math.sin(math.radians(roll))
    cy, sy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    cp, sp = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
    a, b, c = point
    return (
        x + (cp*cy)*a + (cy*sp*sr - sy*cr)*b + (-cy*sp*cr - sy*sr)*c,
        y + (sy*cp)*a + (sy*sp*sr + cy*cr)*b + (-sy*sp*cr + cy*sr)*c,
        z + sp*a - cp*sr*b + cp*cr*c,
    )


def _world_boxes(spec: dict, manifest_dir: Path, frame: dict, agent: dict | None,
                 files: dict[Path, str], context: str) -> list[tuple[float, float, float]]:
    boxes = _read_boxes(spec, manifest_dir=manifest_dir, context=context, files=files)
    coords = spec.get('coordinates')
    if coords == 'world':
        return boxes
    if coords == 'reference_lidar':
        pose = _as_pose(frame.get('reference_lidar_pose'), f'{context} reference_lidar_pose')
    elif coords == 'agent_lidar':
        if agent is None:
            raise ImportFailure(f'{context}: agent_lidar requires an agent')
        pose = _as_pose(agent.get('lidar_pose'), f'{context} agent_lidar_pose')
    else:
        raise ImportFailure(f'{context}: coordinates must be world/reference_lidar/agent_lidar')
    return [_to_world(p, pose) for p in boxes]


def _presence(boxes: list[tuple[float, float, float]], x: float, y: float, radius: float) -> int:
    return int(any((bx-x)**2 + (by-y)**2 <= radius*radius for bx, by, _ in boxes))


def import_box_manifest(manifest: str | Path, output_dir: str | Path, *,
                        radius_m: float = 5.0,
                        offsets_m: tuple[tuple[float, float], ...] = ((-12.0, 0.0), (0.0, 0.0), (12.0, 0.0)),
                        max_common_range_m: float = 45.0,
                        forbid_prediction_alias: bool = True) -> ImportReport:
    """Convert real per-agent model outputs to V1 strict binary trace replay.

    Entire inputs are validated before any output is written. Per-frame gt is
    shared; agents' predictions are independent *executions*, not assumed to be
    statistically independent. Result observations are keyed by frame/source.
    """
    if not all(math.isfinite(v) for v in (radius_m, max_common_range_m)) or radius_m <= 0 or max_common_range_m <= radius_m:
        raise ImportFailure('requires 0 < radius_m < max_common_range_m')
    if not offsets_m or any(len(p) != 2 or not all(math.isfinite(float(v)) for v in p) for p in offsets_m):
        raise ImportFailure('offsets_m must have finite XY offsets')
    path = Path(manifest).resolve()
    frames = _load_manifest(path)
    files: dict[Path, str] = {path: _sha(path)}
    records, provenance = [], []
    skipped = 0
    expected_agents: tuple[str, ...] | None = None
    for frame_slot, frame in enumerate(frames):
        scene, timestep = frame['scene_id'], str(frame['frame_id'])
        agents = frame.get('agents')
        if not isinstance(agents, list) or len(agents) < 2:
            raise ImportFailure(f'{scene}/{timestep}: need >=2 agents')
        agent_ids = tuple(a.get('agent_id') for a in agents)
        if any(not isinstance(a, str) or not a for a in agent_ids) or len(set(agent_ids)) != len(agent_ids):
            raise ImportFailure(f'{scene}/{timestep}: invalid/duplicate agent IDs')
        if expected_agents is None:
            expected_agents = agent_ids
        if set(agent_ids) != set(expected_agents):
            raise ImportFailure(f'{scene}/{timestep}: V1 requires same agent set across all frames')
        agents = sorted(agents, key=lambda a: expected_agents.index(a['agent_id']))
        poses = [_as_pose(a.get('lidar_pose'), f'{scene}/{timestep}/{a["agent_id"]}') for a in agents]
        ax = sum(p[0] for p in poses) / len(poses)
        ay = sum(p[1] for p in poses) / len(poses)
        gt_spec = frame.get('ground_truth')
        gt = _world_boxes(gt_spec, path.parent, frame, None, files, f'{scene}/{timestep}:ground_truth')
        candidates: dict[str, list[tuple[float,float,float]]] = {}
        paths_by_agent: dict[str, dict] = {}
        for agent in agents:
            who = agent['agent_id']
            paths_by_agent[who] = {}
            for kind in ('local', 'edge'):
                spec = agent.get(kind + '_prediction')
                boxes = _world_boxes(spec, path.parent, frame, agent, files, f'{scene}/{timestep}/{who}/{kind}')
                candidates[f'{kind}:{who}'] = boxes
                paths_by_agent[who][kind] = spec['path']
            if forbid_prediction_alias and (path.parent / paths_by_agent[who]['local']).resolve() == (path.parent / paths_by_agent[who]['edge']).resolve():
                raise ImportFailure(f'{scene}/{timestep}/{who}: local and edge predictions reference the same file; provide separate model outputs or explicitly allow aliases')
        for idx, (dx, dy) in enumerate(offsets_m):
            x, y = ax + float(dx), ay + float(dy)
            if any(math.hypot(x-p[0], y-p[1]) + radius_m > max_common_range_m for p in poses):
                skipped += 1
                continue
            eid = f'{scene}@{timestep}:roi_{idx:02d}'
            decision = {'event_id': eid, 'frame_id': f'{scene}@{timestep}',
                        'arrival_slot': frame_slot, 'truth': _presence(gt, x, y, radius_m),
                        'candidates': {k: _presence(v, x, y, radius_m) for k, v in candidates.items()}}
            records.append(decision)
            provenance.append({'event_id': eid, 'dataset': frame.get('dataset', 'unspecified'),
                               'scene_id': scene, 'frame_id': timestep,
                               'task': 'any_vehicle_center_in_world_xy_circle',
                               'roi': {'center_world_xy_m': [x, y], 'radius_m': radius_m},
                               'agents': list(expected_agents), 'model_prediction_sources': paths_by_agent,
                               'gt_source': gt_spec['path'],
                               'reference_frame': 'CARLA/OpenCOOD world; roll-yaw-pitch convention',
                               'label_source': 'recorded ground truth',
                               'prediction_source': 'recorded model output'})
    if not records:
        raise ImportFailure('no valid shared decision events remain after coverage filter')
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tp, pp, rp = out / 'trace.jsonl', out / 'provenance.jsonl', out / 'import_report.json'
    for dst, rows in ((tp, records), (pp, provenance)):
        dst.write_text(''.join(json.dumps(item, sort_keys=True) + '\n' for item in rows), encoding='utf-8')
    positive = sum(v['truth'] for v in records)
    report = {
        'dataset': sorted({f.get('dataset', 'unspecified') for f in frames}),
        'source_manifest': str(path), 'source_manifest_sha256': files[path],
        'source_frames': len(frames), 'agents': list(expected_agents),
        'valid_events': len(records), 'positive_events': positive,
        'negative_events': len(records) - positive,
        'skipped_outside_coverage': skipped,
        'input_file_hashes': {str(k): v for k,v in sorted(files.items())},
        'task': 'any vehicle center in a fixed radius in shared world XY coordinates',
        'selection': 'common coverage based on reported sensor poses',
        'ground_truth': 'shared annotated boxes transformed into world frame',
        'data_status': 'recorded predictions, not generated by this importer',
        'caveats': ['ROI labels are binary and do not constitute 3D object detection AP',
                    'Nearby frames may be correlated: use scene-level uncertainty intervals',
                    'Common sensor coverage does not guarantee real object visibility',
                    'Simulation execution latency/energy remain modeled separately',
                    'Frame-level projection assumes all region decisions use one complete frame output'],
    }
    rp.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    # Self-contained V1 configuration with explicit non-empirical resource costs.
    scenario = {
        'schema_version': 1,
        'scenario': {'name': 'imported_recorded_vision_boxes', 'agents': list(expected_agents),
                     'episodes': len(records), 'seed': 42,
                     'arrival_interval_ms': 250, 'deadline_ms': 200},
        'world': {'backend': 'trace', 'trace_file': 'trace.jsonl',
                  'execution_granularity': 'frame'},
        'policy': {'name': 'always_independent'},
    }
    (out / 'scenario.json').write_text(json.dumps(scenario, indent=2) + '\n', encoding='utf-8')
    return ImportReport(len(frames), len(records), positive, len(records)-positive, skipped,
                        len(files)-1, tp, pp, rp)
