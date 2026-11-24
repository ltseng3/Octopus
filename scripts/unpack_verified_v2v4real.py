#!/usr/bin/env python3
"""Expand source-verified, lossless per-event derived V2V4Real bit codes to a CPU replay.

Packed corpus contains only source-verified sequences; see README. Files contain
predictions derived from genuine public model exports, but not original 3D boxes.
No ROI coordinates are fabricated: use prepare_dmstrack_v2v4real.py to regenerate
full 3D-world provenance when internet/data are available.
"""
import argparse
import csv
import json
from pathlib import Path

ALPHABET='0123456789abcdefghijklmnopqrstuv'

def unpack(src:Path, reference:Path, out:Path):
    meta=json.loads(src.read_text(encoding='utf8'))
    with reference.open() as f: reference_rows={r['scene']:r for r in csv.DictReader(f)}
    trace=[];provenance=[];last_slot=0
    for scene,encoded in meta['scenes'].items():
        expected=reference_rows[scene]
        checksum=meta.get('fnv1a32',{}).get(scene)
        if checksum is not None:
            h=2166136261
            for char in encoded:
                h=((h^ord(char))*16777619)&0xffffffff
            if f'{h:08x}'!=checksum:
                raise ValueError(f'{scene} byte-level verification checksum mismatch')
        if len(encoded)%2:raise ValueError(f'{scene} encoded frame has odd number of digits')
        per_frame=[]
        for i in range(0,len(encoded),2):
            pair=[]
            for digit in encoded[i:i+2]:
                if digit=='.':continue
                if digit not in ALPHABET:raise ValueError('bad digit')
                value=ALPHABET.index(digit)
                pair.append({'truth':(value>>4)&1,'ego':(value>>3)&1,
                             'agent1':(value>>2)&1,'cobevt':(value>>1)&1,'late_fusion':value&1})
            if len(pair) not in (0,2):raise ValueError('incomplete positive/negative frame pair')
            per_frame.append(pair)
        vals=[v for pair in per_frame for v in pair]
        assert len(per_frame)==int(expected['frames'])
        assert len(vals)==int(expected['balanced_events'])
        assert sum(v['truth'] for v in vals)==int(expected['balanced_positives'])
        for method in ['ego','agent1','cobevt','late_fusion']:
            assert sum(v[method]!=v['truth'] for v in vals)==int(expected[method+'_errors']),method
        assert sum(v['ego']!=v['truth'] and v['agent1']!=v['truth'] for v in vals)==int(expected['separate_joint_errors'])
        for frame,pair in enumerate(per_frame):
            for idx,v in enumerate(pair):
                eid=f'v2v4real_{scene}_{frame:04d}_{"positive" if idx==0 else "negative"}'
                trace.append({'event_id':eid,'frame_id':f'v2v4real_{scene}_{frame:04d}',
                              'arrival_slot':last_slot+frame,'truth':v['truth'],
                    'candidates':{'local:ego':v['ego'],'local:agent1':v['agent1'],
                                  'edge:ego':v['late_fusion'],'edge:agent1':v['late_fusion']}})
                provenance.append({'event_id':eid,'scene_id':scene,'frame_id':frame,
                         'roi_kind':'positive' if idx==0 else 'negative',
                         'provenance_note':'Compressed true source outcome; ROI geometry recoverable from pinned text source',
                         'cobevt_prediction':v['cobevt']})
        last_slot+=len(per_frame)
    out.mkdir(parents=True,exist_ok=True)
    for name,items in [('trace.jsonl',trace),('provenance.jsonl',provenance)]:
        with (out/name).open('w') as f:
            for obj in items:f.write(json.dumps(obj,separators=(',',':'))+'\n')
    scenario={'schema_version':1,
        'scenario':{'name':'v2v4real_source_verified_packed_trace','seed':20261118,
                    'agents':['ego','agent1'],'episodes':len(trace),
                    'arrival_interval_ms':500,'deadline_ms':400},
        'world':{'backend':'trace','trace_file':'trace.jsonl',
                 'execution_granularity':'frame'},
        'policy':{'name':'mixed_local_shared','reuse_probability':0.5},
        'resources':{'edge_workers':4,'edge_inference_ms':30,'local_inference_ms':20,
                     'network_channels':4,'network_latency_ms':5,'network_bandwidth_mbps':100,
                     'observation_bytes':1024,'result_bytes':128,'packet_loss':0}}
    (out/'scenario.json').write_text(json.dumps(scenario,indent=2)+'\n')
    print(f'Verified and expanded {len(trace)} source-derived decision events across {len(meta["scenes"])} scenes')
    return len(trace)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    root=Path(__file__).resolve().parents[1]/'empirical/v2v4real'
    p.add_argument('--packed',type=Path,default=root/'packed_outcomes.json')
    p.add_argument('--reference',type=Path,default=root/'sequence_counts.csv')
    a=p.parse_args();unpack(a.packed,a.reference,a.output)
