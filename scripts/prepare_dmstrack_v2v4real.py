#!/usr/bin/env python3
"""V2V4Real-derived two-agent common-decision replay from public DMSTrack text.

The script DOES NOT train or run perception models; it uses publicly exported
predictions (ego CAV, CAV 1, CoBEVT, late-fusion) and shared ground-truth labels.

Source pinned to upstream commit d3b9949499c8e68ea33060873bd1cb95b6d4d323
https://github.com/eddyhkchiu/DMSTrack

Outputs a *new diagnostic task*, not V2V4Real's official AP/mAP benchmark.
KITTI tracking camera coordinates: X, Z are horizontal; Y is vertical.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
from collections import defaultdict
from typing import Callable

COMMIT = 'd3b9949499c8e68ea33060873bd1cb95b6d4d323'
BASE = f'https://raw.githubusercontent.com/eddyhkchiu/DMSTrack/{COMMIT}/'
SEQ_FRAMES = (147,114,144,198,180,310,304,221,375)
NEGATIVE_OFFSETS=((18,0),(-18,0),(0,18),(0,-18),(25,0),(-25,0),(0,25),(0,-25),(35,15))
GRID_CENTERS=tuple((x,z) for x in (-40,-20,0,20) for z in (-10,15,40))
COLUMNS = ['scene','frames','balanced_events','balanced_positives','ego_errors','agent1_errors',
           'cobevt_errors','late_fusion_errors','separate_joint_errors','fixed_events',
           'fixed_positives','fixed_ego_errors','fixed_agent1_errors','fixed_cobevt_errors',
           'fixed_late_fusion_errors','fixed_separate_joint_errors']


def source_paths(scene: str) -> dict[str,str]:
    root='AB3DMOT/data/v2v4real/detection/'
    nofusion=root+'multi_sensor_differentiable_kalman_filter_Car_val/'
    return {
        'gt':f'AB3DMOT/scripts/KITTI/v2v4real_val_label/{scene}.txt',
        'ego':f'{nofusion}ego/{scene}.txt',
        'agent1':f'{nofusion}1/{scene}.txt',
        'cobevt':f'{root}cobevt_Car_val/{scene}.txt',
        'late_fusion':f'{root}late_fusion_Car_val/{scene}.txt',
    }


def parse_boxes(text: str, *, ground_truth: bool, score_threshold:float = 0.30):
    """Map local frame index to XY-on-ground positions in camera (X,Z) plane."""
    out=defaultdict(list)
    for lineno,line in enumerate(text.splitlines(),1):
        if not line.strip():
            continue
        a=line.split() if ground_truth else [v.strip() for v in line.split(',')]
        if ground_truth:
            if len(a)<17 or a[2]!='Car':
                raise ValueError(f'GT row {lineno} has unsupported schema/class: {line[:120]}')
            x,z=float(a[13]),float(a[15])
        else:
            if len(a)<15 or a[1]!='2':
                raise ValueError(f'detection row {lineno} has unsupported schema/class: {line[:120]}')
            if float(a[6]) < score_threshold:
                continue
            x,z=float(a[10]),float(a[12])
        frame=int(a[0]);out[frame].append((x,z))
    return out


def present(points, x:float, z:float, radius:float=7.5) -> int:
    return int(any((px-x)**2 + (pz-z)**2 <= radius*radius for px,pz in points))


def decisions(maps:dict, frame:int, x:float,z:float, radius:float=7.5) -> dict:
    return {k:present(v.get(frame,()),x,z,radius) for k,v in maps.items()}


def choose_regions(gt:dict, frame:int, *, task:str='balanced', radius:float=7.5):
    """Generate decision tasks offline, from GT only; never use model outputs."""
    if task == 'fixed_grid':
        return [('grid_'+str(i).zfill(2), xy) for i,xy in enumerate(GRID_CENTERS)]
    objects=gt.get(frame,())
    if not objects:
        return []
    x,z=objects[frame % len(objects)]
    for dx,dz in NEGATIVE_OFFSETS:
        if not present(objects,x+dx,z+dz,radius):
            return [('positive',(x,z)),('negative',(x+dx,z+dz))]
    return []


def tally(event, counter):
    true=event['truth'];pred=event['predictions']
    counter['events']+=1
    counter['positives']+=true
    for method in ['ego','agent1','cobevt','late_fusion']:
        counter[method+'_errors']+=int(pred[method]!=true)
    counter['separate_joint_errors']+=int(pred['ego']!=true and pred['agent1']!=true)


def new_counter():
    c=defaultdict(int)
    return c


def analyze_scene(scene:str, nframes:int, maps:dict, radius:float=7.5):
    if max(maps['gt'])+1 != nframes:
        raise ValueError(f'{scene}: GT frame count differs from official seqmap')
    for name,frames in maps.items():
        if any(t<0 or t>=nframes for t in frames):
            raise ValueError(f'{scene}: {name} has out-of-range frame')
    balanced=new_counter();fixed=new_counter()
    emitted=[]
    for t in range(nframes):
        for _,(x,z) in choose_regions(maps['gt'],t,task='fixed_grid',radius=radius):
            tally({'truth':present(maps['gt'].get(t,()),x,z,radius),
                   'predictions':decisions({k:v for k,v in maps.items() if k!='gt'},t,x,z,radius)},fixed)
        for name,(x,z) in choose_regions(maps['gt'],t,task='balanced',radius=radius):
            pred=decisions({k:v for k,v in maps.items() if k!='gt'},t,x,z,radius)
            event={'scene':scene,'frame':t,'roi':name,'x':x,'z':z,
                   'truth':present(maps['gt'].get(t,()),x,z,radius),'predictions':pred}
            tally(event,balanced)
            emitted.append(event)
    row=dict(scene=scene,frames=nframes,
             balanced_events=balanced['events'],balanced_positives=balanced['positives'],
             ego_errors=balanced['ego_errors'],agent1_errors=balanced['agent1_errors'],
             cobevt_errors=balanced['cobevt_errors'],late_fusion_errors=balanced['late_fusion_errors'],
             separate_joint_errors=balanced['separate_joint_errors'],
             fixed_events=fixed['events'],fixed_positives=fixed['positives'],
             fixed_ego_errors=fixed['ego_errors'],fixed_agent1_errors=fixed['agent1_errors'],
             fixed_cobevt_errors=fixed['cobevt_errors'],fixed_late_fusion_errors=fixed['late_fusion_errors'],
             fixed_separate_joint_errors=fixed['separate_joint_errors'])
    return row,emitted


def prepare(fetch_text:Callable[[str],str], output:Path, *, threshold=0.30, radius=7.5):
    output.mkdir(parents=True,exist_ok=True)
    rows=[];trace=[];provenance=[];shas={};frame_offset=0
    for index,n in enumerate(SEQ_FRAMES):
        scene=f'{index:04d}';maps={}
        for method,path in source_paths(scene).items():
            data=fetch_text(path)
            shas[path]=hashlib.sha256(data.encode('utf-8')).hexdigest()
            maps[method]=parse_boxes(data,ground_truth=method=='gt',score_threshold=threshold)
        row,events=analyze_scene(scene,n,maps,radius=radius)
        rows.append(row)
        for event in events:
            eid=f'v2v4real_{scene}_{event["frame"]:04d}_{event["roi"]}'
            pred=event['predictions']
            # CRITICAL: edge source uses one cooperative late-fusion candidate.
            # edge:ego and edge:agent1 are aliases of the same already computed
            # prediction, NOT independent edge evidence; never benchmark with
            # always_independent edge policy as if these two were distinct.
            trace.append({'event_id':eid,'frame_id':f'v2v4real_{scene}_{event["frame"]:04d}',
                          'arrival_slot':frame_offset+event['frame'],
                          'truth':event['truth'],
                          'candidates':{'local:ego':pred['ego'],'local:agent1':pred['agent1'],
                                        'edge:ego':pred['late_fusion'],'edge:agent1':pred['late_fusion']}})
            provenance.append({'event_id':eid,'scene_id':scene,'frame_id':event['frame'],
                              'roi':event['roi'],'center_kitti_xz':[event['x'],event['z']],
                              'cobevt_candidate':pred['cobevt'],
                              'warning':'edge candidates alias one shared late-fusion output'})
        frame_offset+=n
    with (output/'sequence_counts.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=COLUMNS);w.writeheader();w.writerows(rows)
    for name,items in [('trace.jsonl',trace),('provenance.jsonl',provenance)]:
        with (output/name).open('w',encoding='utf8') as f:
            for obj in items:f.write(json.dumps(obj,separators=(',',':'))+'\n')
    scenario={
        'schema_version':1,
        'scenario':{'name':'v2v4real_public_dmstrack_diagnostic','seed':20261118,
                    'agents':['ego','agent1'],'episodes':len(trace),
                    'arrival_interval_ms':500,'deadline_ms':400},
        'world':{'backend':'trace','trace_file':'trace.jsonl',
                 'execution_granularity':'frame'},
        'policy':{'name':'mixed_local_shared','reuse_probability':0.5},
        'resources':{'edge_workers':4,'edge_inference_ms':30,'local_inference_ms':20,
                     'network_channels':4,'network_latency_ms':5,
                     'network_bandwidth_mbps':100,'observation_bytes':1024,
                     'result_bytes':128,'packet_loss':0}
    }
    (output/'scenario.json').write_text(json.dumps(scenario,indent=2)+'\n')
    (output/'source_manifest.json').write_text(json.dumps({
        'repository':'https://github.com/eddyhkchiu/DMSTrack',
        'commit':COMMIT,'detector_score_threshold':threshold,'radius_m':radius,
        'decision':'KITTI camera horizontal (X,Z) presence of any Car box center',
        'balanced_task':'one GT-anchored positive + nearby GT-negative per frame',
        'source_sha256':shas,'frames':sum(SEQ_FRAMES),'events':len(trace),
        'caveat':'recorded predictions are real; coordination and resource costs are simulated',
    },indent=2)+'\n')
    return rows


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--cache',type=Path,default=Path('data/dmstrack_public_text'))
    ap.add_argument('--offline',action='store_true',help='Require all pinned upstream files already in cache')
    ap.add_argument('--threshold',type=float,default=.30)
    ap.add_argument('--radius',type=float,default=7.5,help='Vehicle-center region radius in meters')
    ap.add_argument('--check',type=Path,help='Compare regenerated counts against reference CSV')
    args=ap.parse_args()
    if not 0<=args.threshold<=1:ap.error('threshold must be 0..1')
    if args.radius<=0:ap.error('radius must be positive')
    def fetch(path):
        dst=args.cache/path
        if dst.exists():return dst.read_text(encoding='utf8')
        if args.offline:raise FileNotFoundError(f'Offline cache miss: {dst}')
        import requests
        resp=requests.get(BASE+path,timeout=90,headers={'User-Agent':'OCTOPUS-Sim-Research-Reproduction/0.3'})
        resp.raise_for_status();text=resp.content.decode('utf8')
        dst.parent.mkdir(parents=True,exist_ok=True);dst.write_text(text,encoding='utf8')
        return text
    rows=prepare(fetch,args.output,threshold=args.threshold,radius=args.radius)
    if args.check:
        with args.check.open(newline='') as f: expected=list(csv.DictReader(f))
        if len(expected)!=len(rows):raise SystemExit('Mismatch: sequence count')
        for exp,actual in zip(expected,rows):
            for key in COLUMNS:
                if str(exp[key])!=str(actual[key]):
                    raise SystemExit(f'Validation mismatch {actual["scene"]} {key}: {actual[key]} vs {exp[key]}')
        print('All 9 sequence summaries EXACTLY MATCH the checked reference.')
    print(f'Wrote 9 sequences to {args.output}, {sum(x["balanced_events"] for x in rows)} events.')

if __name__=='__main__':main()
