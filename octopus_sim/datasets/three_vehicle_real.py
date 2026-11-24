"""Fail-closed three-*physical-vehicle* binary ROI replay importer.

This is NOT a LiDAR detector, box-projection tool, or data downloader. It takes
synchronized *already computed* three-vehicle ROI outputs with provenance,
checks their common-decision contract, and builds the existing frame-level
OCTOPUS trace. Nothing is inferred about a source's correctness from truth.

CooperScene uses vehicles 1,2,3 and RSU 0; the latter is NOT a third vehicle.
The original V2V4Real has only TWO physical vehicles and is rejected.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from octopus_sim.config import ExperimentConfig

VEHICLES = ('1', '2', '3')
SOURCE_URL = 'https://github.com/UCR-CISL/CooperScene'


class ThreeVehicleDataError(ValueError):
    pass


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ThreeVehicleDataError(message)


def _is_number(value: Any) -> bool:
    return type(value) in (float,int) and math.isfinite(value)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _binary(value: Any, field: str) -> int:
    _check(type(value) is int and value in (0, 1), field + ' must be integer 0 or 1')
    return value


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def read_protocol(manifest: Path, *, allow_fixture: bool = False) -> dict:
    m=_load_json(manifest)
    _check(isinstance(m,dict), 'Manifest must be an object')
    _check(m.get('dataset') == 'CooperScene',
           'Expected real three-vehicle CooperScene. Original V2V4Real only has two vehicles.')
    _check(m.get('split') in ('train','validate','test'), 'split must be train/validate/test')
    _check(m.get('vehicle_ids') == list(VEHICLES),
           'CooperScene physical vehicles must be exactly ["1","2","3"] (0 is roadside infrastructure)')
    _check(m.get('roi_protocol') == 'predeclared_world_xy_grid_v1',
           'Only fixed, predeclared world-XY ROI tasks supported; GT-selected ROIs forbidden')
    _check(m.get('coordinate_frame') == 'cooperscene_global_xy_m',
           'ROI must use explicitly registered CooperScene global XY coordinates')
    _check(m.get('prediction_origin') in ('real_detector_export','synthetic_fixture'),
           'prediction_origin must identify real_detector_export or synthetic_fixture')
    if m['prediction_origin']=='synthetic_fixture':
        _check(allow_fixture, 'Synthetic fixture rejected; pass allow_fixture=True only for tests')
    else:
        _check(not allow_fixture, 'Do not declare a real data export as a synthetic fixture')
    runs=m.get('physical_detector_runs')
    _check(isinstance(runs,dict) and set(runs)==set(VEHICLES) and
           all(isinstance(v,str) and v.strip() for v in runs.values()) and
           len(set(runs.values()))==3,
           'Need three distinct physically executed vehicle-detector run identifiers')
    qs=m.get('assumed_source_error')
    _check(isinstance(qs,dict) and set(qs)==set(VEHICLES) and
           all(_is_number(x) and 0 < x < .5 for x in qs.values()),
           'Must explicitly provide three assumed error rates strictly between 0 and 0.5')
    _check(m.get('error_estimator_status') in ('unfitted_assumption','fit_on_disjoint_training_split'),
           'Must disclose q source explicitly')
    if m['error_estimator_status'] == 'fit_on_disjoint_training_split':
        _check(m.get('error_fit_split')=='train' and m['split']!='train',
               'Estimated q must be fitted on disjoint train split')
    _check(_is_number(m.get('interval_ms')) and m['interval_ms']>0,
           'interval_ms must be finite and positive')
    # This only declares the run IDs, not independently proof of the detector
    # provenance; authors must keep the raw checkpoint/run manifest separately.
    return m




def _external_rows(path: Path, *, key_field: str, expected_agent: str | None = None) -> dict[tuple[str,str,str], int]:
    output = {}
    for lineno, line in enumerate(path.read_text(encoding='utf8').splitlines(), 1):
        if not line.strip():continue
        rec=json.loads(line)
        _check(isinstance(rec,dict),f'External export {path}:{lineno} must be JSON object')
        identity=tuple(rec.get(k) for k in ('take_id','frame_token','roi_id'))
        _check(len(identity)==3 and all(isinstance(v,str) and v for v in identity),
               f'External export {path}:{lineno} missing common frame/ROI identity')
        _check(identity not in output,f'Duplicate external source frame/ROI in {path}:{lineno}')
        if expected_agent is not None:
            _check(rec.get('vehicle_id') == expected_agent,
                   f'External source {path}:{lineno} mislabeled vehicle_id')
        output[identity]=_binary(rec.get(key_field),f'{path}:{lineno} {key_field}')
    return output


def _verify_real_source_support(m: dict, manifest: Path, rows: list[dict]) -> dict:
    """Require three independent per-vehicle prediction artifacts and GT source.

    Checks complete row-for-row identity and SHA-256 integrity. This guards
    accidental source aliasing; it does NOT attest that a detector truly ran.
    """
    def artifact(entry, name):
        _check(isinstance(entry,dict) and isinstance(entry.get('path'),str)
               and isinstance(entry.get('sha256'),str),f'Missing {name} export path/sha256')
        _check(re.fullmatch(r'[a-fA-F0-9]{64}',entry['sha256']) is not None,
               f'Invalid {name} SHA-256 digest')
        path=Path(entry['path'])
        if not path.is_absolute():path=manifest.parent/path
        path=path.resolve()
        _check(path.is_file(),f'Missing {name} artifact: {path}')
        _check(_sha(path)==entry['sha256'].lower(),f'SHA-256 mismatch in {name}')
        return path
    source_files=m.get('per_vehicle_prediction_exports')
    _check(isinstance(source_files,dict) and set(source_files)==set(VEHICLES),
           'Real detector export requires 3 individually hashed per-vehicle prediction files')
    paths={a:artifact(source_files[a],f'physical vehicle {a}') for a in VEHICLES}
    _check(len(set(paths.values()))==3,'Cannot alias one file as independent three-vehicle evidence')
    truth_file=artifact(m.get('ground_truth_export'),'shared ground truth')
    mapped={a:_external_rows(path,key_field='prediction',expected_agent=a)
            for a,path in paths.items()}
    gt=_external_rows(truth_file,key_field='truth')
    identities={tuple(r[k] for k in ('take_id','frame_token','roi_id')) for r in rows}
    for a in VEHICLES:
        _check(set(mapped[a])==identities,f'Physical vehicle {a} export has missing or extra ROI rows')
    _check(set(gt)==identities,'Shared ground truth export has missing or extra ROI rows')
    for r in rows:
        key=tuple(r[k] for k in ('take_id','frame_token','roi_id'))
        for a in VEHICLES:
            _check(mapped[a][key]==r['predictions'][a],
                   f'Merged output contradicts physical vehicle {a} artifact at {key}')
        _check(gt[key]==r['truth'],f'Merged labels contradict GT artifact at {key}')
    checkpoints=m.get('checkpoint_sha256')
    _check(isinstance(checkpoints,dict) and set(checkpoints)==set(VEHICLES) and
           all(isinstance(d,str) and re.fullmatch(r'[a-fA-F0-9]{64}',d) for d in checkpoints.values()),
           'Real detector results require source model checkpoint SHA-256 by vehicle')
    return {'per_vehicle_prediction_export_sha256':{a:_sha(p) for a,p in paths.items()},
            'ground_truth_export_sha256':_sha(truth_file),
            'checkpoint_sha256':checkpoints}


def convert_to_trace(source: Path, manifest: Path, output: Path, *,
                     allow_fixture: bool = False) -> dict:
    m=read_protocol(manifest,allow_fixture=allow_fixture)
    _check(source.is_file(),f'Predictions not found: {source}')
    original_sha=_sha(source)
    rows=[]; seen=set(); frames={}; roi_by_key={}; events_per_take=defaultdict(int)
    for line_no, line in enumerate(source.read_text(encoding='utf-8').splitlines(),1):
        if not line.strip():continue
        try:
            row=json.loads(line)
        except ValueError as e:
            raise ThreeVehicleDataError(f'Invalid input JSONL line {line_no}: {e}') from e
        _check(isinstance(row,dict),f'Line {line_no} must be an object')
        _check(row.get('split')==m['split'],f'Line {line_no} split mismatch')
        take=row.get('take_id'); frame=row.get('frame_token'); roi=row.get('roi_id')
        _check(all(isinstance(x,str) and x and '/' not in x and ':' not in x
                   for x in (take,frame,roi)),f'Line {line_no}: invalid take/frame/roi ID')
        key=(take,frame,roi)
        _check(key not in seen,f'Duplicate take/frame/ROI at line {line_no}')
        seen.add(key)
        truth=_binary(row.get('truth'),f'Line {line_no} truth')
        predictions=row.get('predictions')
        _check(isinstance(predictions,dict) and set(predictions)==set(VEHICLES),
               f'Line {line_no}: exactly three per-vehicle predictions required; no RSU or fusion alias')
        for a in VEHICLES:
            _binary(predictions[a],f'Line {line_no} prediction[{a}]')
        geometry=row.get('roi')
        _check(isinstance(geometry,dict) and geometry.get('frame')==m['coordinate_frame'],
               f'Line {line_no} ROI coordinate frame mismatch')
        center=geometry.get('center_xy_m'); radius=geometry.get('radius_m')
        _check(isinstance(center,list) and len(center)==2 and all(_is_number(v) for v in center)
               and _is_number(radius) and radius>0,
               f'Line {line_no}: ROI needs finite center_xy_m and positive radius_m')
        # Across the *same physical frame*, a given ROI identity cannot silently
        # change geometric meaning. Across moving frames it may be reprojected.
        roi_key=(take,frame,roi)
        geometry_tuple=(float(center[0]),float(center[1]),float(radius))
        _check(roi_key not in roi_by_key or roi_by_key[roi_key]==geometry_tuple,
               'ROI geometry changed within one physical frame')
        roi_by_key[roi_key]=geometry_tuple
        frames.setdefault((take,frame),[]).append(row)
        events_per_take[take]+=1
        rows.append(row)
    _check(rows,'No decision rows were provided')
    supporting=(_verify_real_source_support(m,manifest,rows)
                if m['prediction_origin']=='real_detector_export' else None)
    # Require every physical vehicle to be present on every common ROI. A
    # partial sample must not masquerade as a complete 3-CAV measurement.
    frame_sort=sorted(frames,key=lambda pair:(pair[0],0,int(pair[1])) if pair[1].isdigit() else (pair[0],1,pair[1]))
    slots={k:i for i,k in enumerate(frame_sort)}
    output.mkdir(parents=True,exist_ok=True)
    trace=[]
    for row in rows:
        take,token,roi=row['take_id'],row['frame_token'],row['roi_id']
        fid=f'CooperScene:{m["split"]}:{take}:{token}'
        eid=f'{fid}:{roi}'
        trace.append({'event_id':eid,'frame_id':fid,'arrival_slot':slots[(take,token)],
                      'truth':row['truth'],
                      'candidates':{f'local:{a}':row['predictions'][a] for a in VEHICLES}})
    trace.sort(key=lambda r:(r['arrival_slot'],r['event_id']))
    trace_path=output/'trace.jsonl'
    trace_path.write_text(''.join(json.dumps(r,sort_keys=True,separators=(',',':'))+'\n' for r in trace),encoding='utf8')
    template=ExperimentConfig.load(Path(__file__).resolve().parents[2]/'examples'/'f0'/'three_agent_f0.json')
    d=deepcopy(template.data)
    d['scenario'].update(name='CooperScene_three_physical_vehicle_ROI_F0',agents=list(VEHICLES),
                         episodes=len(trace),arrival_interval_ms=float(m['interval_ms']),
                         deadline_ms=float(m.get('deadline_ms',250.0)))
    d['world']={'backend':'trace','trace_file':'trace.jsonl','execution_granularity':'frame'}
    d['policy']={'name':'selective_fusion_f0'}
    d['coordination']['enabled']=False
    d['fusion_f0']['error_by_agent']={a:float(m['assumed_source_error'][a]) for a in VEHICLES}
    d['fusion_f0']['dependence_group_by_agent']={a:f'physical_vehicle_{a}' for a in VEHICLES}
    # The rate estimates and energy model remain assumptions, even though the
    # actual binary source predictions and truth may come from real detectors.
    if m['prediction_origin']=='real_detector_export':
        d['energy']['calibration']='hypothetical'
    scenario=ExperimentConfig.from_dict(d)
    (output/'scenario.json').write_text(json.dumps(scenario.data,indent=2,sort_keys=True)+'\n')
    # Path in generated scenario is RELATIVE to the output directory.
    provenance={'dataset':'CooperScene','split':m['split'],'physical_vehicles':list(VEHICLES),
                'physical_frames':len(frames),'decision_events':len(trace),
                'take_counts':dict(sorted(events_per_take.items())),
                'prediction_origin':m['prediction_origin'],
                'input_sha256':original_sha,'protocol_sha256':_sha(manifest),
                'verifiable_source_artifacts':supporting,
                'output_trace_sha256':_sha(trace_path),
                'source_repository':SOURCE_URL,
                'error_rate_origin':m['error_estimator_status'],
                'cost_status':'simulated_hypothetical_energy_and_service_times',
                'data_release_status':'upstream_data_not_redistributed',
                'limitations':[
                    'per-agent binary ROI predictions must be generated from real detectors externally',
                    'ground truth appears only in scoring, not in fusion observations',
                    'physical sensor lineages distinct; environmental errors may remain correlated',
                    'no calibrated network/energy model has been fitted to CooperScene',
                    'ROI-to-box projection and detector training not performed by this converter',
                ]}
    (output/'PROVENANCE.json').write_text(json.dumps(provenance,indent=2,sort_keys=True)+'\n')
    return provenance
