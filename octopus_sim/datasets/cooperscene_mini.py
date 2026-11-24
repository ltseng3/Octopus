"""CPU-only integrity inspection of the upstream CooperScene mini release.

This module verifies *presence, alignment and metadata structure*, not point
cloud geometry, detector predictions, sensor time calibration, or measurement
accuracy. Data are never downloaded or embedded by this module.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

SPLITS = ('train', 'validate', 'test')
VEHICLES = ('1','2','3')
AGENTS = ('0','1','2','3')
OFFICIAL_MINI_URL = 'https://data.ucr.edu/datasets/cooperscene/mini.zip'
UPSTREAM_REPO = 'https://github.com/UCR-CISL/CooperScene'
EXPECTED_MINI_FRAMES = {'train':120,'validate':30,'test':30}
EXCLUDED_YAML_SUFFIXES = ('_additional.yaml', '_separate.yaml')

class MiniDataError(ValueError):
    pass


def _regular_yaml(path: Path) -> bool:
    name=path.name
    return (name.endswith('.yaml') and name != 'data_protocol.yaml' and
            not name.startswith('data_protocol') and
            not any(name.endswith(s) for s in EXCLUDED_YAML_SUFFIXES))


def _pose_ok(pose: Any) -> bool:
    from math import isfinite
    if not isinstance(pose,(list,tuple)):
        return False
    if len(pose)==6:
        return all(type(x) in (int,float) and isfinite(x) for x in pose)
    if len(pose)==4:
        return (all(isinstance(row,(list,tuple)) and len(row)==4 for row in pose)
                and all(type(x) in (int,float) and isfinite(x) for row in pose for x in row))
    return False


def _pcd_header(path: Path) -> dict[str,Any]:
    """Header-only PCD check: avoids reading point clouds into RAM."""
    with path.open('rb') as handle:
        header=handle.read(131072)
    lines=[]
    for entry in header.splitlines():
        try:
            line=entry.decode('ascii')
        except UnicodeDecodeError:
            break
        lines.append(line)
        if line.startswith('DATA '):
            break
    fields={line.split(maxsplit=1)[0]:line.split(maxsplit=1)[1] for line in lines
            if line.strip() and not line.startswith('#') and len(line.split(maxsplit=1))==2}
    if 'DATA' not in fields or fields['DATA'] not in ('ascii','binary','binary_compressed'):
        raise MiniDataError(f'Invalid PCD header (no DATA mode): {path}')
    if 'FIELDS' not in fields or not {'x','y','z'}.issubset(set(fields['FIELDS'].split())):
        raise MiniDataError(f'PCD does not declare x,y,z: {path}')
    for key in ('POINTS','WIDTH','HEIGHT'):
        if key not in fields or not fields[key].strip().isdigit():
            raise MiniDataError(f'Invalid PCD {key}: {path}')
    return {'data_mode':fields['DATA'],'points':int(fields['POINTS']),
            'fields':fields['FIELDS'].split()}


def locate_root(candidate: Path) -> Path:
    """Accept the data root or a directory with one uniquely matching child."""
    candidate=Path(candidate).resolve()
    if all((candidate/s).is_dir() for s in SPLITS):
        return candidate
    choices=[]
    if candidate.is_dir():
        for sub in candidate.iterdir():
            if sub.is_dir() and all((sub/s).is_dir() for s in SPLITS):
                choices.append(sub)
        if len(choices)==0:
            for sub in candidate.iterdir():
                if sub.is_dir():
                    for child in sub.iterdir():
                        if child.is_dir() and all((child/s).is_dir() for s in SPLITS):
                            choices.append(child)
    if len(choices)!=1:
        raise MiniDataError('No unambiguous CooperScene root with train/validate/test: '
                            +str(candidate))
    return choices[0]


def scan_mini(root: Path, *, strict_mini: bool=False, parse_yaml: bool=True,
              check_pcd_headers: bool=True) -> dict[str,Any]:
    """Fail closed on missing raw triplets; warn about cross-split shared takes.

    No ground truth is used to choose decision ROIs. This scanner does not
    create an OCTOPUS trace because recorded model predictions do not exist.
    """
    root=locate_root(root)
    if parse_yaml:
        try:
            import yaml
        except ImportError as exc:
            raise MiniDataError('Install PyYAML to validate poses and labels') from exc
    warnings=[]
    takes_by_split={}
    results={}
    for split in SPLITS:
        take_dirs=sorted(p for p in (root/split).iterdir() if p.is_dir())
        if not take_dirs:
            raise MiniDataError(f'{split}: no take directories')
        takes_by_split[split]={p.name for p in take_dirs}
        complete=0; skipped=0; rows=[]
        file_count=Counter()
        for take in take_dirs:
            vehicle_frames={}
            agent_frames={}
            for agent in AGENTS:
                ad=take/agent
                if not ad.is_dir():
                    raise MiniDataError(f'{split}/{take.name}: missing expected agent {agent}')
                ys={p.stem:p for p in ad.iterdir() if p.is_file() and _regular_yaml(p)}
                pcds={p.stem:p for p in ad.glob('*.pcd')}
                bins={p.stem:p for p in ad.glob('*.bin')}
                for token in ys:
                    if token not in pcds and token not in bins:
                        raise MiniDataError(f'{split}/{take.name}/{agent}/{token}: YAML without point cloud')
                orphans=set(pcds)|set(bins)
                if orphans-set(ys):
                    raise MiniDataError(f'{split}/{take.name}/{agent}: unannotated point clouds: '
                                        +repr(sorted(orphans-set(ys))[:5]))
                if not ys:
                    raise MiniDataError(f'{split}/{take.name}/{agent}: no timestamped annotations')
                agent_frames[agent]=ys
                if agent in VEHICLES:
                    vehicle_frames[agent]=set(ys)
            # Verify synchronized vehicle snapshots; we do not infer alignment
            # from adjacent timestamps or interpolate a missing vehicle.
            sets=list(vehicle_frames.values())
            if any(s!=sets[0] for s in sets[1:]):
                details={a:len(s) for a,s in vehicle_frames.items()}
                raise MiniDataError(f'{split}/{take.name}: missing synchronized vehicle frames {details}')
            synchronized=sets[0]
            if synchronized != set(agent_frames['0']):
                raise MiniDataError(f'{split}/{take.name}: RSU/vehicle timestamp mismatch')
            for frame in sorted(synchronized):
                annotations=[]
                point_counts={}
                for agent in AGENTS:
                    ad=take/agent
                    y=agent_frames[agent][frame]
                    pcd=ad/(frame+'.pcd')
                    bin_file=ad/(frame+'.bin')
                    point_file=pcd if pcd.is_file() else bin_file
                    file_count['yaml']+=1
                    file_count[point_file.suffix.lower().lstrip('.')]+=1
                    if check_pcd_headers and point_file.suffix=='.pcd':
                        point_counts[agent]=_pcd_header(point_file)['points']
                    else:
                        point_counts[agent]=None
                    if parse_yaml:
                        # safe_load intentionally rejects arbitrary Python-object
                        # tags; never use yaml.unsafe_load on untrusted data.
                        try:
                            obj=yaml.safe_load(y.read_text(encoding='utf-8'))
                        except Exception as exc:
                            raise MiniDataError(f'Could not safely parse {y}: {exc}') from exc
                        if not isinstance(obj,dict):
                            raise MiniDataError(f'{y}: YAML must contain a mapping')
                        pose=obj.get('lidar_pose',obj.get('true_ego_pos'))
                        if not _pose_ok(pose):
                            raise MiniDataError(f'{y}: lidar_pose/true_ego_pos missing or malformed')
                        vehicles=obj.get('vehicles')
                        if not isinstance(vehicles,dict):
                            raise MiniDataError(f'{y}: vehicles annotations must be mapping')
                        annotations.append(len(vehicles))
                    else:
                        annotations.append(None)
                rows.append({'split':split,'take_id':take.name,'frame_token':frame,
                             'agents':list(AGENTS),'vehicle_ids':list(VEHICLES),
                             'reported_annotation_counts':annotations,
                             'pcd_points':point_counts})
                complete+=1
        results[split]={'take_count':len(take_dirs),'synchronized_frames':complete,
                        'file_counts':dict(file_count),'takes':sorted(takes_by_split[split]),
                        'frames':rows}
        if strict_mini and complete!=EXPECTED_MINI_FRAMES[split]:
            raise MiniDataError(f'{split}: expected {EXPECTED_MINI_FRAMES[split]} frames; saw {complete}')
    for ix,a in enumerate(SPLITS):
        for b in SPLITS[ix+1:]:
            common=takes_by_split[a]&takes_by_split[b]
            if common:
                warnings.append(f'{a} and {b} share take IDs {sorted(common)}; '
                                'mini train/validation/test may be temporally adjacent, '
                                'not statistically independent scene splits')
    warnings.append('Labels and PCD headers inspected, not detector outputs or 3D geometry')
    warnings.append('No real F0 R2/R3/accuracy results without separately executed detector outputs')
    return {'schema_version':1,'dataset':'CooperScene','origin':'local_mini_archive',
            'root':str(root),'official_download':OFFICIAL_MINI_URL,
            'license':'CC BY-NC-SA 4.0 (upstream dataset)',
            'status':'actual_raw_dataset_layout_checked_not_detector_inference',
            'parse_yaml':parse_yaml,'check_pcd_headers':check_pcd_headers,
            'splits':results,'total_frames':sum(x['synchronized_frames'] for x in results.values()),
            'warnings':warnings}


def summarize_report(report: dict[str,Any]) -> dict[str,Any]:
    return {k:v for k,v in report.items() if k!='splits'} | {
        'splits':{name:{k:v for k,v in section.items() if k!='frames'}
                  for name,section in report['splits'].items()}}


def write_reports(report: dict[str,Any], output: Path) -> None:
    import csv
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    (output/'mini_scan.json').write_text(json.dumps(summarize_report(report),indent=2,sort_keys=True)+'\n')
    with (output/'frame_index.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['split','take_id','frame_token','agent_ids',
                                            'annotation_counts_by_agent','pcd_points_by_agent'])
        writer.writeheader()
        for section in report['splits'].values():
            for row in section['frames']:
                writer.writerow({'split':row['split'],'take_id':row['take_id'],
                                 'frame_token':row['frame_token'],
                                 'agent_ids':','.join(row['agents']),
                                 'annotation_counts_by_agent':json.dumps(row['reported_annotation_counts']),
                                 'pcd_points_by_agent':json.dumps(row['pcd_points'])})
