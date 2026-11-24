"""End-to-end F0 over REAL OCTOPUS-Sim queues, frame cache and energy ledger."""
from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path

import pytest

from octopus_sim.config import ConfigurationError, ExperimentConfig
from octopus_sim.simulator import run_simulation
from octopus_sim.cli import save


def example_trace(tmp_path, *, agents=('a','b'), frames=2, rows=None):
    if rows is None:
        rows = [(('a',1),('b',1),1), (('a',1),('b',0),1), (('a',0),('b',0),0)]
    trace=tmp_path/'trace.jsonl'
    records=[]
    for f in range(frames):
        for r,values in enumerate(rows):
            observed=dict(values[:-1])
            if set(observed)!=set(agents):
                observed={a:observed.get(a, r%2) for a in agents}
            records.append({'event_id':f'frame{f}:roi{r}', 'frame_id':f'frame{f}',
                            'arrival_slot':f, 'truth':values[-1],
                            'candidates':{f'local:{a}':observed[a] for a in agents}})
    trace.write_text(''.join(json.dumps(x)+'\n' for x in records))
    return trace


def config(trace, *, agents=('a','b'), q=None, loss=0, deadline=200, edge_workers=1,
           age_ms=120, profiles=None, scenario_interval=500, groups=None, energy=True):
    if q is None:q={a:.06 for a in agents}
    prof=profiles or {'local_model':{'device':'local','compute_ms':20.,'output_bytes':1024},
                      'f0_decider':{'device':'edge','compute_ms':2.,'output_bytes':64}}
    radio={'mode':'per_bit','tx_j_per_bit':1.0e-7,'rx_j_per_bit':5.0e-8,
           'tx_incremental_w':0,'rx_incremental_w':0}
    d={'scenario':{'name':'f0_integration','seed':123,'agents':list(agents),
                   'episodes':1,'arrival_interval_ms':scenario_interval,'deadline_ms':deadline},
       'world':{'backend':'trace','trace_file':str(trace),'execution_granularity':'frame'},
       'policy':{'name':'selective_fusion_f0'},
       'resources':{'network_latency_ms':2,'network_bandwidth_mbps':8,'packet_loss':loss,
                    'network_channels':1,'edge_workers':edge_workers,'model_profiles':prof},
       'fusion_f0':{'error_by_agent':q, 'dependence_group_by_agent':groups or {},
                    'max_age_ms':age_ms},
       'energy':{'enabled':energy,'device_profiles':{
           'agent':{'baseline_w':2,'compute_incremental_w':4,'radio':radio},
           'edge':{'baseline_w':10,'compute_incremental_w':9,'radio':radio}}}}
    return ExperimentConfig.from_dict(d)


def test_same_frame_mixed_roi_fallback_keeps_private_local_answers(tmp_path):
    tr=example_trace(tmp_path,frames=2)
    r=run_simulation(config(tr),True)
    s=r.summary
    assert len(r.scores)==6
    assert s['physical_execution_batches']==2
    assert s['local_inference_executions']==4
    assert s['edge_inference_executions']==2
    assert s['f0_shared_roi_decisions']==4
    assert s['f0_keep_local_roi_decisions']==2
    assert s['f0_realized_agent_overrides']==8
    assert s['transmitted_bytes']==2*(2*1024+2*64)
    assert s['transmitted_bytes_by_role']=={'detection':4096,'result':256}
    assert len([x for x in r.inferences if x['model_id']=='f0_decider'])==2
    assert len({x['result_id'] for x in r.inferences})==6
    for frame in range(2):
        frame_scores=[x for x in r.scores if x['frame_id']==f'frame{frame}']
        shared0, fallback, shared2=frame_scores
        assert len(set(shared0['result_ids'].values()))==1
        assert all(':result:f0_decide' in x for x in shared0['result_ids'].values())
        assert fallback['predictions']=={'a':1,'b':0}
        assert fallback['result_ids']['a']!=fallback['result_ids']['b']
        assert all(':result:local:' in x for x in fallback['result_ids'].values())
        assert shared2['predictions']=={'a':0,'b':0}
    for f0_record in (x for x in r.inferences if x['model_id']=='f0_decider'):
        assert len(f0_record['parent_result_ids'])==2
        assert len(f0_record['predictions_by_event_id'])==2
        assert len(f0_record['decision_deliveries'])==4
        assert all(k.endswith(('roi0','roi2')) for k in f0_record['predictions_by_event_id'])
    assert s['frame_projection_cache_hits_by_model']=={'local_model':8,'f0_decider':4}


def test_all_fallback_performs_no_downlink_but_charges_detection_upload_and_edge_compute(tmp_path):
    trace=example_trace(tmp_path,frames=1,rows=[(('a',1),('b',0),1), (('a',0),('b',1),0)])
    s=run_simulation(config(trace),True)
    assert s.summary['f0_shared_roi_decisions']==0
    assert s.summary['f0_keep_local_roi_decisions']==2
    assert s.summary['transmitted_bytes_by_role']=={'detection':2048}
    assert s.summary['energy_compute_by_model_j']['f0_decider']==pytest.approx(.002*9)
    assert s.summary['energy_compute_by_model_j']['local_model']==pytest.approx(2*.02*4)
    assert s.summary['f0_realized_agent_overrides']==0
    assert all(x['result_ids']['a']!=x['result_ids']['b'] for x in s.scores)
    assert any(x['status']=='no_fused_roi_accepted' for x in s.operations)
    assert all(x['action']=='KEEP_LOCAL' for x in s.fusion_decisions)


def test_masking_and_fallback_in_one_frame_with_three_agents(tmp_path):
    rows=[(('a',1),('b',0),('c',1),1),(('a',1),('b',0),('c',0),0)]
    trace=example_trace(tmp_path,agents=('a','b','c'),frames=1,rows=rows)
    r=run_simulation(config(trace,agents=('a','b','c'),q={'a':.05,'b':.4,'c':.07}),True)
    assert r.summary['f0_masked_fusion_decisions']==1
    assert r.summary['f0_keep_local_roi_decisions']==1
    assert r.fusion_decisions[0]['action']=='MASK_AND_FUSE'
    assert 'b' in dict(r.fusion_decisions[0]['masked_agents'])
    assert r.fusion_decisions[1]['action']=='KEEP_LOCAL'
    assert r.scores[0]['predictions']=={'a':1,'b':1,'c':1}
    assert r.scores[1]['predictions']=={'a':1,'b':0,'c':0}
    assert r.summary['transmitted_bytes']==3*1024+3*64


def test_shared_lineage_conflict_masks_every_member_no_oracle(tmp_path):
    rows=[(('a',1),('b',0),('c',1),1)]
    trace=example_trace(tmp_path,agents=('a','b','c'),frames=1,rows=rows)
    c=config(trace,agents=('a','b','c'),q={'a':.05,'b':.07,'c':.06},groups={'a':'g','b':'g'})
    r=run_simulation(c)
    assert r.fusion_decisions[0]['action']=='KEEP_LOCAL'
    assert dict(r.fusion_decisions[0]['masked_agents'])=={'a':'dependence_conflict','b':'dependence_conflict'}
    assert r.summary['f0_realized_agent_overrides']==0


def test_truth_changes_do_not_change_actions_or_estimated_reliability(tmp_path):
    tr=example_trace(tmp_path,frames=2)
    c=config(tr)
    before=run_simulation(c)
    rows=[json.loads(x) for x in tr.read_text().splitlines()]
    for row in rows:row['truth']=1-row['truth']
    tr.write_text(''.join(json.dumps(x)+'\n' for x in rows))
    after=run_simulation(c)
    assert before.fusion_decisions==after.fusion_decisions
    assert before.inferences==after.inferences
    assert before.summary['transmitted_bytes']==after.summary['transmitted_bytes']
    assert before.scores[0]['ground_truth']!=after.scores[0]['ground_truth']


def test_one_lost_upload_falls_back_without_loss_of_local_inference(tmp_path):
    tr=example_trace(tmp_path,frames=1)
    r=run_simulation(config(tr,loss=1.0),True)
    assert r.summary['messages_lost']==2
    assert r.summary['edge_inference_executions']==0
    assert r.summary['local_inference_executions']==2
    assert r.summary['transmitted_bytes']==2*1024
    assert r.summary['f0_keep_local_roi_decisions']==3
    assert all('unavailable_edge_evidence' in x['fallback_reasons'] for x in r.fusion_decisions)
    assert r.summary['complete_event_rate']==1.0
    assert r.summary['energy_radio_incremental_j']>0
    assert r.summary['f0_realized_agent_overrides']==0


def test_lost_downlink_preserves_local_fallback(monkeypatch,tmp_path):
    import octopus_sim.execution_runtime as er
    original=er.uniform
    def losses(seed,batch,kind,stage,sender,receiver):
        if stage.startswith('f0_result_down:'):
            return 0.0  # packet_loss=0.5 => LOST
        return 1.0      # uploads succeed
    monkeypatch.setattr(er,'uniform',losses)
    tr=example_trace(tmp_path,frames=1)
    r=run_simulation(config(tr,loss=.5),True)
    assert r.summary['messages_lost']==2
    assert r.summary['f0_shared_roi_decisions']==2
    assert r.summary['f0_realized_agent_overrides']==0
    assert r.summary['complete_event_rate']==1.0
    assert all(':result:local:' in x for row in r.scores for x in row['result_ids'].values())
    assert r.summary['transmitted_bytes_by_role']=={'detection':2048,'result':128}


def test_deadline_clipping_preserves_post_window_energy(tmp_path):
    tr=example_trace(tmp_path,frames=1)
    r=run_simulation(config(tr,deadline=10),True)
    assert r.summary['complete_event_rate']==0
    assert r.summary['agent_deadline_miss_rate']==1
    assert r.summary['energy_post_window_incremental_j']>0
    assert r.summary['f0_realized_agent_overrides']==0
    assert r.summary['energy_total_j']>r.summary['energy_window_total_j']


@pytest.mark.parametrize('agents',(2,3,5))
def test_f0_runs_one_dag_per_frame_across_agent_counts(tmp_path,agents):
    names=tuple('a'+str(i) for i in range(agents))
    # all source labels consistent in first ROI; second ROI disparate
    data=[]
    for f in range(3):
        for j in range(2):
            data.append({'event_id':f'{f}:{j}','frame_id':f'f{f}','arrival_slot':f,'truth':int(j==0),
                         'candidates':{f'local:{a}':int(j==0) if j==0 else int(i%2) for i,a in enumerate(names)}})
    tr=tmp_path/'many.jsonl';tr.write_text(''.join(json.dumps(x)+'\n' for x in data))
    r=run_simulation(config(tr,agents=names,q={a:.08 for a in names}))
    assert len(r.scores)==6
    assert r.summary['local_inference_executions']==agents*3
    assert r.summary['edge_inference_executions']==3
    assert r.summary['f0_fusion_decisions']==6
    assert len({x['result_id'] for x in r.inferences})==3*(agents+1)
    assert r.summary['energy_total_j']>0


def test_provenance_and_cli_audit_persist(tmp_path):
    tr=example_trace(tmp_path,frames=1)
    r=run_simulation(config(tr))
    dest=tmp_path/'output'
    save(r,dest,tr)
    found=[json.loads(x) for x in (dest/'fusion_decisions.jsonl').read_text().splitlines()]
    assert len(found)==3
    assert all('ground_truth' not in x for x in found)
    assert all('input_result_ids' in x for x in found)
    assert json.loads((dest/'manifest.json').read_text())['octopus_version']=='0.7.1'
    assert len((dest/'scores.jsonl').read_text().splitlines())==3


def test_strict_configuration_and_p1_incompatibility(tmp_path):
    tr=example_trace(tmp_path,frames=1)
    d=deepcopy(config(tr).data)
    altered=[]
    x=deepcopy(d);x['fusion_f0']['error_by_agent']={'a':.1};altered.append(x)
    x=deepcopy(d);x['fusion_f0']['error_by_agent']['a']=0;altered.append(x)
    x=deepcopy(d);x['fusion_f0']['max_age_ms']=-1;altered.append(x)
    x=deepcopy(d);x['fusion_f0']['dependence_group_by_agent']={'nonagent':'g'};altered.append(x)
    x=deepcopy(d);x['resources']['model_profiles'].pop('f0_decider');altered.append(x)
    x=deepcopy(d);x['coordination']['enabled']=True;x['coordination']['candidate_ids']=['local_only'];altered.append(x)
    x=deepcopy(d);x['fusion_f0']['require_risk_bounds']=True;altered.append(x)
    x=deepcopy(d);x['world']['execution_granularity']='decision';altered.append(x)
    for bad in altered:
        with pytest.raises(ConfigurationError):ExperimentConfig.from_dict(bad)


def test_repeatable_runs_including_complete_f0_provenance(tmp_path):
    tr=example_trace(tmp_path,frames=3)
    cfg=config(tr)
    a=run_simulation(cfg)
    b=run_simulation(cfg)
    assert a.summary==b.summary and a.scores==b.scores
    assert a.inferences==b.inferences and a.fusion_decisions==b.fusion_decisions
    assert a.operations==b.operations


def test_injected_public_quality_estimator_and_non_oracle_boundaries(tmp_path):
    from octopus_fusion.estimators import QualityProxyEstimator
    tr=example_trace(tmp_path,frames=1)
    cfg=config(tr)
    data=deepcopy(cfg.data)
    data['fusion_f0']['quality_proxy_by_agent']={'a':.95,'b':.80}
    cfg=ExperimentConfig.from_dict(data)
    run=run_simulation(cfg,f0_error_estimator=QualityProxyEstimator())
    assert run.fusion_decisions
    assert all(d['estimator_id']=='uncalibrated_quality_proxy' for d in run.fusion_decisions)
    assert all('ground_truth' not in d for d in run.fusion_decisions)
    # A declared, visible quality proxy may alter decisions. Flipping only the
    # evaluation labels, which never enter FusionObservation, cannot.
    before=deepcopy(run.fusion_decisions)
    rows=[json.loads(x) for x in tr.read_text().splitlines()]
    for row in rows:row['truth']=1-row['truth']
    tr.write_text(''.join(json.dumps(x)+'\n' for x in rows))
    after=run_simulation(cfg,f0_error_estimator=QualityProxyEstimator())
    assert after.fusion_decisions==before
    wrong=deepcopy(data)
    wrong['fusion_f0']['quality_proxy_by_agent']={'a':.95}
    with pytest.raises(ConfigurationError,match='quality_proxy_by_agent'):
        ExperimentConfig.from_dict(wrong)


def test_actual_v2v4real_frame_level_replay_stays_consistent(root):
    f0=ExperimentConfig.load(root/'examples/f0/v2v4real_f0.json')
    baseline=ExperimentConfig.load(root/'examples/energy/v2v4real_0007_energy.json')
    local=run_simulation(baseline.with_policy('local_only'),False)
    integrated=run_simulation(f0,False)
    assert len(local.scores)==len(integrated.scores)==442
    assert integrated.summary['physical_execution_batches']==221
    assert integrated.summary['local_inference_executions']==442
    assert integrated.summary['edge_inference_executions']==221
    assert integrated.summary['compute_executions_by_model']=={'f0_decider':221,'local_model':442}
    assert len(integrated.fusion_decisions)==442
    assert integrated.summary['f0_shared_roi_decisions']>0
    assert integrated.summary['f0_keep_local_roi_decisions']>0
    assert integrated.summary['joint_all_error_count']==local.summary['joint_all_error_count']==4
    # With two agents, F0 accepts only agreement; it has no third source to
    # resolve disagreements. Thus ALL per-agent decisions match local-only.
    assert [s['predictions'] for s in local.scores]==[s['predictions'] for s in integrated.scores]
    assert integrated.summary['energy_workload_incremental_j'] > local.summary['energy_workload_incremental_j']
    assert integrated.summary['mean_delivered_latency_ms'] > local.summary['mean_delivered_latency_ms']
    assert len({r['result_id'] for r in integrated.inferences})==663


def test_partial_downlink_loss_causes_mixed_per_agent_results(monkeypatch,tmp_path):
    import octopus_sim.execution_runtime as er
    def loss_for_a(seed,batch,kind,stage,sender,recipient):
        return 0.0 if stage=='f0_result_down:a' else 1.0
    monkeypatch.setattr(er,'uniform',loss_for_a)
    tr=example_trace(tmp_path,frames=1)
    r=run_simulation(config(tr,loss=.5),True)
    assert r.summary['f0_shared_roi_decisions']==2
    assert r.summary['messages_lost']==1
    assert r.summary['f0_realized_agent_overrides']==2  # only agent b updated
    for event in (r.scores[0],r.scores[2]):
        assert ':result:local:a' in event['result_ids']['a']
        assert ':result:f0_decide' in event['result_ids']['b']
    assert r.summary['transmitted_bytes_by_role']=={'detection':2048,'result':128}


def test_energy_disabled_and_queued_frames_release_all_active_work(tmp_path):
    tr=example_trace(tmp_path,frames=3)
    c=config(tr,energy=False,scenario_interval=0,deadline=600,profiles={
        'local_model':{'device':'local','compute_ms':20,'output_bytes':1024},
        'f0_decider':{'device':'edge','compute_ms':100,'output_bytes':64}})
    from octopus_sim.simulator import Simulator
    sim=Simulator(c)
    r=sim.run()
    assert sim._active_batches==set()
    assert r.summary['local_queue_wait_ms']>0
    assert r.summary['edge_queue_wait_ms']>0
    assert r.summary['f0_fusion_decisions']==9
    assert not any(k.startswith('energy_') for k in r.summary)
    assert all(row['complete'] for row in r.scores)
