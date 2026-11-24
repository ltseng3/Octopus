"""P0 contract tests: complete work, causal ancestry, deterministic failure paths."""
from __future__ import annotations
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from octopus_sim.config import ConfigurationError, ExperimentConfig
from octopus_sim.execution import ExecutionGraph, Stage, compile_execution_graph
from octopus_sim.policies import BasicPolicy, CoordinatorView, Plan
from octopus_sim.simulator import run_simulation


PROFILES = {
    'local_model': {'device': 'local', 'compute_ms': 20, 'output_bytes': 4096},
    'edge_model': {'device': 'edge', 'compute_ms': 35, 'output_bytes': 64},
    'late_fusion': {'device': 'edge', 'compute_ms': 7, 'output_bytes': 128},
}


def cfg(policy='cooperative_late_fusion', *, n=2, episodes=3, **changes):
    data = {
        'scenario': {'name': 'p0_graph_test', 'agents': [f'a{i}' for i in range(n)],
                     'episodes': episodes, 'seed': 751, 'deadline_ms': 160,
                     'arrival_interval_ms': 300},
        'resources': {'model_profiles': deepcopy(PROFILES),
                      'edge_workers': 1, 'network_channels': 1,
                      'network_latency_ms': 4, 'network_bandwidth_mbps': 100,
                      'packet_loss': 0},
        'policy': {'name': policy, 'reuse_probability': 0.5},
    }
    for name, value in changes.items():
        if name == 'deadline_ms': data['scenario']['deadline_ms'] = value
        elif name == 'arrival_interval_ms': data['scenario']['arrival_interval_ms'] = value
        elif name == 'packet_loss': data['resources']['packet_loss'] = value
        elif name == 'bandwidth_mbps': data['resources']['network_bandwidth_mbps'] = value
        elif name == 'network_channels': data['resources']['network_channels'] = value
        elif name == 'edge_workers': data['resources']['edge_workers'] = value
        elif name == 'reuse_probability': data['policy']['reuse_probability'] = value
        else: raise AssertionError(name)
    return ExperimentConfig.from_dict(data)


def test_p0_exact_three_agent_work_and_wire_bytes():
    run = run_simulation(cfg(n=3, episodes=2))
    s = run.summary
    assert s['local_inference_executions'] == 6
    assert s['edge_inference_executions'] == 2
    assert s['compute_executions_by_model'] == {'late_fusion': 2, 'local_model': 6}
    assert s['compute_busy_ms_by_model'] == {'late_fusion': 14, 'local_model': 120}
    assert s['local_cpu_busy_ms'] == pytest.approx(120)
    assert s['edge_cpu_busy_ms'] == pytest.approx(14)
    assert s['transmitted_bytes_by_role'] == {'detection': 6*4096, 'result': 6*128}
    assert s['transmitted_bytes'] == 6*(4096+128)
    assert s['messages_attempted'] == 12
    assert len(run.operations) == 2*(3 + 3 + 1 + 3 + 3)
    assert all(x['status'] == 'completed' for x in run.operations)
    for scored in run.scores:
        ids = list(scored['result_ids'].values())
        assert None not in ids and len(set(ids)) == 1


def test_fusion_result_is_one_identity_with_complete_ancestry():
    run = run_simulation(cfg(n=3, episodes=1))
    assert len(run.inferences) == 4
    local = {r['source_agent']: r for r in run.inferences if r['model_id'] == 'local_model'}
    fused, = [r for r in run.inferences if r['model_id'] == 'late_fusion']
    assert set(local) == {'a0','a1','a2'}
    assert set(fused['parent_result_ids']) == {r['result_id'] for r in local.values()}
    assert set(fused['observation_ids']) == {f'event_000000:observation:a{i}' for i in range(3)}
    assert set(fused['consumers_requested']) == set(fused['consumers_delivered']) == set(local)
    assert not any(r['consumers_delivered'] for r in local.values())
    assert set(run.scores[0]['result_ids'].values()) == {fused['result_id']}


def test_fusion_will_not_run_until_every_uploaded_detection_arrives():
    run = run_simulation(cfg(n=3, episodes=1, bandwidth_mbps=2))
    op = {r['stage_id']:r for r in run.operations}
    fusion = op['late_fusion']
    uploads = [op[f'detection_upload:a{i}'] for i in range(3)]
    assert fusion['submitted_at_ms'] >= max(x['finished_at_ms'] for x in uploads)
    assert all(x['submitted_at_ms'] >= 20 for x in uploads)
    for i in range(3):
        delivery = op[f'deliver:a{i}']
        assert delivery['submitted_at_ms'] >= op[f'result_down:a{i}']['finished_at_ms']
    assert run.summary['link_queue_wait_ms'] > 0


def test_total_processing_work_not_one_fake_inference_job():
    r1 = run_simulation(cfg('local_only', episodes=2))
    r2 = run_simulation(cfg('cooperative_late_fusion', episodes=2))
    assert r1.summary['local_cpu_busy_ms'] == r2.summary['local_cpu_busy_ms'] == 80
    assert r1.summary['edge_cpu_busy_ms'] == 0
    assert r2.summary['edge_cpu_busy_ms'] == 14
    assert r1.summary['transmitted_bytes'] == 0
    assert r2.summary['transmitted_bytes'] == 2*2*(4096+128)
    assert r1.summary['local_inference_executions'] == r2.summary['local_inference_executions'] == 4


def test_loss_stops_fusion_but_does_not_erase_upstream_compute():
    run = run_simulation(cfg(n=3, episodes=1, packet_loss=1))
    assert run.summary['local_inference_executions'] == 3
    assert run.summary['edge_inference_executions'] == 0
    assert run.summary['messages_attempted'] == run.summary['messages_lost'] == 3
    assert run.summary['transmitted_bytes'] == 3*4096
    assert run.summary['complete_event_rate'] == 0
    op = {x['stage_id']:x['status'] for x in run.operations}
    assert all(op[f'detection_upload:a{i}'] == 'packet_lost' for i in range(3))
    assert op['late_fusion'] == 'dependency_failed'
    assert all(op[f'result_down:a{i}'] == 'dependency_failed' for i in range(3))
    assert not any(x['model_id'] == 'late_fusion' for x in run.inferences)


def test_deadline_does_not_cancel_or_retroactively_credit_late_fusion():
    run = run_simulation(cfg(episodes=1, deadline_ms=15))
    assert run.summary['complete_event_rate'] == 0
    assert run.summary['agent_deadline_miss_rate'] == 1
    assert run.summary['joint_all_error_rate'] == 0
    assert run.summary['edge_inference_executions'] == 1
    assert any(e['type']=='result_arrived_late' for e in run.events)
    assert not any(r['consumers_delivered'] for r in run.inferences)


def test_fusion_results_reproducible_and_counterfactual_matched():
    conf=cfg(episodes=12)
    a=run_simulation(conf)
    b=run_simulation(conf)
    assert a.summary==b.summary
    assert a.inferences==b.inferences
    assert a.operations==b.operations
    assert a.events==b.events
    mixed_local=run_simulation(cfg('mixed_local_fusion',episodes=12,reuse_probability=0))
    baseline=run_simulation(cfg('local_only',episodes=12))
    assert mixed_local.scores==baseline.scores
    mixed_fused=run_simulation(cfg('mixed_local_fusion',episodes=12,reuse_probability=1))
    assert mixed_fused.scores==a.scores
    assert mixed_fused.summary['transmitted_bytes']==a.summary['transmitted_bytes']


def test_same_trace_prediction_outcomes_used_for_fusion_and_local(tmp_path):
    event={'event_id':'f1','truth':1,'candidates':{
        'local:a0':0,'local:a1':1,'edge:a0':1,'edge:a1':1}}
    trace=tmp_path/'trace.jsonl';trace.write_text(json.dumps(event)+'\n')
    config=deepcopy(cfg(episodes=1).data)
    config['world']={'backend':'trace','trace_file':str(trace),
                     'fusion_candidate_key':'edge:a0'}
    fused=run_simulation(ExperimentConfig.from_dict(config))
    separate=run_simulation(ExperimentConfig.from_dict({**config,'policy':{'name':'local_only'}}))
    assert fused.scores[0]['predictions']=={'a0':1,'a1':1}
    assert separate.scores[0]['predictions']=={'a0':0,'a1':1}
    assert {r['source_agent']:r['prediction'] for r in fused.inferences
            if r['model_id']=='local_model'} == {'a0':0,'a1':1}


@pytest.mark.parametrize('model_id,profile', [
    ('late_fusion', {'device':'local','compute_ms':1,'output_bytes':128}),
    ('local_model', {'device':'edge','compute_ms':1,'output_bytes':1}),
    ('late_fusion', {'device':'edge','compute_ms':-1,'output_bytes':1}),
    ('late_fusion', {'device':'edge','compute_ms':1,'output_bytes':-1}),
    ('late_fusion', {'device':'edge','compute_ms':1,'output_bytes':1,'unknown':1}),
])
def test_profile_validation_errors(model_id,profile):
    data=deepcopy(cfg().data)
    data['resources']['model_profiles'][model_id]=profile
    with pytest.raises(ConfigurationError):
        ExperimentConfig.from_dict(data)


def test_missing_fusion_profile_fails_before_simulation():
    data=deepcopy(cfg().data)
    del data['resources']['model_profiles']['late_fusion']
    with pytest.raises(ConfigurationError,match='late_fusion'):
        ExperimentConfig.from_dict(data)


def test_graph_validator_rejects_cycle_and_missing_delivery():
    good=compile_execution_graph(
        BasicPolicy('cooperative_late_fusion').plan(CoordinatorView('id',('a0','a1'),300,.5,.05,.0)),
        ('a0','a1'),cfg().data['resources'])
    good.validate()
    bad=ExecutionGraph(tuple(s for s in good.stages if s.stage_id != 'deliver:a1'),good.agents,good.workflow)
    with pytest.raises(ValueError,match='Exactly one delivery'):
        bad.validate()
    stages=list(good.stages)
    first=stages[0]
    stages[0]=replace(first,dependencies=('late_fusion',))
    with pytest.raises(ValueError,match='Cycle'):
        ExecutionGraph(tuple(stages),good.agents,good.workflow).validate()


def test_legacy_plan_compiles_into_valid_dag(build):
    for name in ['local_only','always_share','always_independent','partial_share']:
        c=build(name,agents=('a','b','c'))
        v=CoordinatorView('event',('a','b','c'),125,.2,.06,0.0)
        graph=compile_execution_graph(BasicPolicy(name).plan(v),v.agents,c.data['resources'])
        graph.validate()
        assert sum(s.kind=='deliver' for s in graph.stages)==3


def test_real_p0_v2v4real_example_end_to_end(root):
    folder=root/'examples/v2v4real_real_seq0007'
    local=run_simulation(ExperimentConfig.load(folder/'scenario_p0_pipeline.json').with_policy('local_only'),
                         detailed_events=False)
    fusion=run_simulation(ExperimentConfig.load(folder/'scenario_p0_pipeline.json'),detailed_events=False)
    assert local.summary['joint_all_error_rate'] == pytest.approx(4/442)
    assert fusion.summary['joint_all_error_rate'] == pytest.approx(16/442)
    assert local.summary['local_inference_executions']==442
    assert fusion.summary['local_inference_executions']==442
    assert fusion.summary['edge_inference_executions']==221
    assert fusion.summary['transmitted_bytes_by_role']=={
        'detection':442*4096,'result':442*128}
    assert fusion.summary['compute_busy_ms_by_model']=={'late_fusion':221*6,'local_model':442*20}
    assert len(fusion.inferences)==3*221
    assert all(s['complete'] for s in fusion.scores)


def test_named_model_bindings_change_costs_without_editing_execution_code():
    data=deepcopy(cfg(n=3,episodes=2).data)
    data['resources']['model_bindings']={
        'local':'fast_detector','edge':'edge_model','fusion':'robust_fusion'}
    data['resources']['model_profiles']={
        'fast_detector':{'device':'local','compute_ms':11.5,'output_bytes':1536},
        'robust_fusion':{'device':'edge','compute_ms':9.25,'output_bytes':256},
    }
    sim=run_simulation(ExperimentConfig.from_dict(data),detailed_events=False)
    assert sim.summary['compute_executions_by_model']=={'fast_detector':6,'robust_fusion':2}
    assert sim.summary['compute_busy_ms_by_model']=={'fast_detector':69.,'robust_fusion':18.5}
    assert sim.summary['transmitted_bytes_by_role']=={'detection':9216,'result':1536}
    assert sim.summary['transmitted_bytes']==10752
    assert len([r for r in sim.inferences if r['model_id']=='robust_fusion'])==2
    assert sim.summary['complete_event_rate']==1


def test_trace_fusion_requires_explicit_candidate_selection(tmp_path):
    data=deepcopy(cfg(episodes=1).data)
    data['world']={'backend':'trace','trace_file':str(tmp_path/'data.jsonl')}
    with pytest.raises(ConfigurationError,match='fusion_candidate_key'):
        ExperimentConfig.from_dict(data)
    data['world']['fusion_candidate_key']='cooperative:fused'
    event={'event_id':'f1','truth':0,'candidates':{
        'local:a0':1,'local:a1':1,'edge:a0':1,'edge:a1':1}}
    trace=tmp_path/'data.jsonl';trace.write_text(json.dumps(event)+'\n')
    from octopus_sim.world import World,TraceError
    with pytest.raises(TraceError,match='cooperative:fused'):
        World(ExperimentConfig.from_dict(data).data,trace)
    event['candidates']['cooperative:fused']=0
    trace.write_text(json.dumps(event)+'\n')
    sim=run_simulation(ExperimentConfig.from_dict(data),detailed_events=False)
    assert sim.summary['joint_all_error_rate']==0
    fused,=[r for r in sim.inferences if r['model_id']=='late_fusion']
    assert fused['source_agent'] is None
    assert fused['source_agents']==['a0','a1']
    assert fused['candidate_key']=='cooperative:fused'
    assert fused['prediction']==0


@pytest.mark.parametrize('n', [2,3,5])
@pytest.mark.parametrize('loss', [0,1])
def test_multiagent_dag_bounds_and_conservation(n,loss):
    r=run_simulation(cfg(n=n,episodes=4,packet_loss=loss),detailed_events=True)
    assert r.summary['local_inference_executions']==n*4
    assert r.summary['edge_inference_executions']==(0 if loss else 4)
    assert r.summary['transmitted_bytes']==(4*n*4096 if loss else 4*n*(4096+128))
    assert r.summary['transmitted_bytes']==sum(r.summary['transmitted_bytes_by_role'].values())
    assert r.summary['local_cpu_busy_ms']==sum(r.summary['compute_busy_ms_by_model'].get(k,0)
                                                   for k in ['local_model'])
    assert r.summary['messages_attempted']==4*n*(1 if loss else 2)
    assert r.summary['agent_deadline_miss_rate']==(1 if loss else 0)
    assert len(r.operations)==len(set((r0['event_id'],r0['stage_id']) for r0 in r.operations))
    assert all(row['status'] in {'completed','packet_lost','dependency_failed'} for row in r.operations)


def test_nonfinite_resource_cost_rejected():
    for invalid in (float('inf'),float('-inf'),float('nan')):
        data=deepcopy(cfg().data)
        data['resources']['model_profiles']['late_fusion']['compute_ms']=invalid
        with pytest.raises(ConfigurationError):
            ExperimentConfig.from_dict(data)
