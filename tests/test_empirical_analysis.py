from __future__ import annotations
import json
from pathlib import Path
import pytest
from octopus_sim.datasets.analysis import analyze_empirical, EmpiricalAnalysisError


def _records(d, prov, scene_count=4):
    d.mkdir()
    for name in ('share','independent'):
        pp=d/name;pp.mkdir()
        rows=[]
        for s in range(scene_count):
            for i in range(3):
                eid=f'scene{s}@frame{i}:roi0'
                rows.append({'event_id':eid,'ground_truth':1,
                             'all_wrong': bool((s+i)%2==0) if name=='share' else bool((s+i)%3==0),
                             'all_operational_fail':False,'complete':True})
        (pp/'scores.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
    prov.write_text(''.join(json.dumps({'event_id':f'scene{s}@frame{i}:roi0',
                                      'scene_id':f'scene{s}'})+'\n'
                            for s in range(scene_count) for i in range(3)))


def test_scene_clustered_deterministic(tmp_path):
    d=tmp_path/'result';prov=tmp_path/'prov.jsonl';_records(d,prov)
    a=analyze_empirical(d,prov,baseline='share',iterations=1000,seed=101)
    b=analyze_empirical(d,prov,baseline='share',iterations=1000,seed=101)
    assert a==b
    assert a['scene_count']==4 and a['event_count']==12
    assert a['policies']['independent']['scene_clustered_95_ci'] is not None
    diff=a['paired_differences']['independent']
    assert diff['vs']=='share'
    assert diff['scene_clustered_95_ci'][0] <= diff['difference_joint_all_error_rate'] <= diff['scene_clustered_95_ci'][1]


def test_one_scene_no_ci(tmp_path):
    d=tmp_path/'result';prov=tmp_path/'prov.jsonl';_records(d,prov,scene_count=1)
    a=analyze_empirical(d,prov,iterations=100)
    assert a['scene_count']==1
    assert a['policies']['share']['scene_clustered_95_ci'] is None
    assert 'warning' in a


def test_reject_mismatched_events_or_ground_truth(tmp_path):
    d=tmp_path/'result';prov=tmp_path/'prov.jsonl';_records(d,prov)
    f=d/'share'/'scores.jsonl'
    rows=[json.loads(x) for x in f.read_text().splitlines()]
    rows[0]['ground_truth']=0
    f.write_text(''.join(json.dumps(x)+'\n' for x in rows))
    with pytest.raises(EmpiricalAnalysisError,match='ground truth mismatch'):
        analyze_empirical(d,prov,iterations=100)


def test_cli_analysis(tmp_path):
    d=tmp_path/'result';prov=tmp_path/'prov.jsonl';_records(d,prov)
    from octopus_sim.cli import main
    out=tmp_path/'analysis.json'
    assert main(['analyze-empirical',str(d),str(prov),'--iterations','100', '--output',str(out)])==0
    assert json.loads(out.read_text())['scene_count']==4
