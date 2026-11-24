"""Fast reproducibility, no-leakage and online/offline agreement checks."""
from __future__ import annotations

import csv
import json
import hashlib
from pathlib import Path

import numpy as np
import pytest

from scripts.run_f0_cvc_parameter_study import (
    AGENTS, RHOS, TRUE_Q, generate_controlled_traces, summarize_local,
    fast_study_run, validate_online_subset, _frame_group_arrays, boot_ci,main,
)


def test_generator_is_reproducible_and_correlated_by_construction(tmp_path):
    a=generate_controlled_traces(tmp_path/'a',frames=65,rois=2,seed=92)
    b=generate_controlled_traces(tmp_path/'b',frames=65,rois=2,seed=92)
    assert set(a)==set(b)==set(RHOS)
    for rho in RHOS:
        assert hashlib.sha256(a[rho].read_bytes()).hexdigest()==hashlib.sha256(b[rho].read_bytes()).hexdigest()
    rows0=[json.loads(x) for x in a[0.0].read_text().splitlines()]
    rows1=[json.loads(x) for x in a[1.0].read_text().splitlines()]
    assert len(rows0)==len(rows1)==130
    for x,y in zip(rows0,rows1):
        assert x['truth']==y['truth']
        assert x['candidates']['local:a']==y['candidates']['local:a']
        assert x['candidates']['local:b']==y['candidates']['local:b']
        assert {y['candidates']['local:'+agent] for agent in ('c','d','e')}.__len__()==1
    assert len({x['frame_id'] for x in rows0})==65


def test_offline_lineage_and_local_decision_error_rates_are_well_formed(tmp_path):
    trace=generate_controlled_traces(tmp_path,frames=90,rois=2,seed=5701)[.75]
    baseline, local_scores=summarize_local(trace,90,2)
    _,fused_scores=fast_study_run(trace,method='lineage_aware',nframes=90,rois=2)
    assert [s['event_id'] for s in local_scores]==[s['event_id'] for s in fused_scores]
    frame_ids=[f'S{i:05d}' for i in range(90)]
    loc=_frame_group_arrays(local_scores,frame_ids)
    fus=_frame_group_arrays(fused_scores,frame_ids)
    assert loc.shape==fus.shape==(90,6)
    for scores in (local_scores,fused_scores):
        assert all(0<=s['error_count']<=5 for s in scores)
    for group in (loc,fus):
        assert np.all(group>=0) and np.all(group<=1)
        assert np.all(np.diff(group[:,:5],axis=1)<=0)


def test_actual_online_executor_matches_same_f0_predictions_and_joules(tmp_path):
    trace=generate_controlled_traces(tmp_path/'synthetic',frames=45,rois=2,seed=601)[.75]
    for method,cutoff in [('local_only',.9),('naive_majority',.9),
                          ('lineage_aware',.9),('false_independence',.9),
                          ('lineage_aware',.999)]:
        check=validate_online_subset(trace,method=method,frames=35,rois=2,
                                     cutoff=cutoff,tempdir=tmp_path/'checks')
        assert check['checked_decisions']==70
        assert check['verified_prediction_and_resource_fields']>=14


def test_ground_truth_flip_never_changes_decisions_or_cost(tmp_path):
    trace=generate_controlled_traces(tmp_path/'synthetic',frames=35,rois=2,seed=84)[.75]
    flipped=tmp_path/'flipped.jsonl'
    rows=[json.loads(x) for x in trace.read_text().splitlines()]
    for row in rows:row['truth']=1-row['truth']
    flipped.write_text(''.join(json.dumps(x)+'\n' for x in rows))
    first, a=fast_study_run(trace,method='lineage_aware',nframes=35,rois=2)
    second,b=fast_study_run(flipped,method='lineage_aware',nframes=35,rois=2)
    for k in ('share_roi_coverage','downlinked_frame_fraction',
              'changed_agent_decisions','incremental_energy_j','network_bytes'):
        assert first[k]==second[k],k
    for x,y in zip(a,b):
        assert x['event_id']==y['event_id']
        assert x['error_count']+y['error_count']==5


def test_physical_cost_sunk_and_energy_overhead(tmp_path):
    trace=generate_controlled_traces(tmp_path,frames=60,rois=2,seed=92)[.75]
    local,_=fast_study_run(trace,method='local_only',nframes=60,rois=2)
    conservative,_=fast_study_run(trace,method='lineage_aware',cutoff=.999,nframes=60,rois=2)
    eager,_=fast_study_run(trace,method='lineage_aware',cutoff=.55,nframes=60,rois=2)
    assert local['energy_per_frame_j']==pytest.approx(.400)
    assert conservative['share_roi_coverage']==0
    assert conservative['energy_per_frame_j']>local['energy_per_frame_j']
    assert eager['network_bytes']>=conservative['network_bytes']
    assert eager['local_compute_per_frame_j']==conservative['local_compute_per_frame_j']
    assert eager['f0_compute_per_frame_j']==conservative['f0_compute_per_frame_j']


def test_paired_cluster_bootstrap_keeps_frame_level_unit():
    arr=np.array([0,.5,1.0,.5])
    idx=np.array([[0,1,2,3],[3,2,1,0],[0,0,0,0]],dtype=int)
    lo,hi=boot_ci(arr,idx)
    assert lo==pytest.approx(.025)
    assert hi==pytest.approx(.5)


def test_script_figures_and_provenance_from_minimal_smoke(tmp_path):
    # Tests the CLI in a fresh, independent output directory; standard paper
    # settings are stored in the committed study outputs.
    import subprocess,sys
    script=Path(__file__).resolve().parents[1]/'scripts/run_f0_cvc_parameter_study.py'
    root=Path(__file__).resolve().parents[1]
    proc=subprocess.run([sys.executable,str(script),'--frames','30','--rois','2',
                         '--bootstrap','50','--online-check-frames','15',
                         '--skip-v2v','--output',str(tmp_path/'result')],
                         cwd=root,capture_output=True,text=True,timeout=150)
    assert proc.returncode==0,proc.stderr+'\n'+proc.stdout[-2000:]
    out=tmp_path/'result'
    assert len(list(out.glob('*.pdf')))==4
    assert len(list(out.glob('*.svg')))==4
    assert len(list(out.glob('*.png')))==4
    prov=json.loads((out/'STUDY_PROVENANCE.json').read_text())
    assert prov['synthetic']['frame_count']==30
    assert len(prov['online_validation'])==8
    with (out/'five_agent_policy_and_risk_grid.csv').open() as f:
        r=list(csv.DictReader(f))
    assert len(r)==27  # 5 rho * (local + 4 F0/vote methods), plus 2 trust variants
    assert all(0<=float(x['r_ge_5'])<=float(x['r_ge_3'])<=1 for x in r)
