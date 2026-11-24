"""V0.7 contract tests: no oracle information, no candidate-execution side effects."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError
import json
import math

import pytest

from octopus_sim.config import ConfigurationError, ExperimentConfig
from octopus_sim.coordination import (
    AnalyticalResourceCostEstimator, CandidatePlan, CoordinatorObservation,
    QueueSnapshot, ResourceSnapshot, StandardCandidateProvider, SyntheticRiskEstimator,
)
from octopus_sim.policies import Group, Plan
from octopus_sim.execution import compile_execution_graph
from octopus_sim.simulator import Simulator, run_simulation
from octopus_sim.world import TraceError, World


def config(*, agents=('a','b','c'), policy='candidate_fixed', cid='local_only',
           episodes=1, interval=1000, deadline=200, enabled=True, energy=False):
    conf={
        'scenario': {'name':'v07_contract', 'agents':list(agents),'episodes':episodes,
                     'seed':4,'arrival_interval_ms':interval, 'deadline_ms':deadline},
        'policy': {'name':policy,'candidate_id':cid} if policy=='candidate_fixed' else {'name':policy},
        'coordination': {'enabled':enabled, 'visible_features':['visibility'],
                         'synthetic_features':{'visibility':{'kind':'uniform','low':0.0,'high':1.0}},
                         'risk_estimator':{'kind':'contextual','context_feature':'visibility',
                                           'context_slope':0.07}},
        'resources': {'edge_workers':2,'network_channels':2,'packet_loss':0.0,
                      'model_profiles':{
                          'local_model':{'device':'local','compute_ms':20.0,'output_bytes':512},
                          'edge_model':{'device':'edge','compute_ms':30.0,'output_bytes':128},
                          'late_fusion':{'device':'edge','compute_ms':8.0,'output_bytes':128}}},
    }
    if energy:
        radio={'mode':'per_bit', 'tx_j_per_bit':1e-7,'rx_j_per_bit':8e-8,
               'tx_incremental_w':0,'rx_incremental_w':0}
        conf['energy']={'enabled':True,'device_profiles':{
            'agent':{'baseline_w':1.0,'compute_incremental_w':4.0,'radio':radio},
            'edge':{'baseline_w':5.0,'compute_incremental_w':10.0,'radio':radio}}}
    return ExperimentConfig.from_dict(conf)


def test_candidate_fixed_runs_and_outputs_per_physical_batch():
    r=run_simulation(config(episodes=4))
    assert len(r.candidate_estimates)==4*7 # local, independent edge, 3 shared sources, partial, fusion
    assert all('truth' not in c and 'ground_truth' not in c for c in r.candidate_estimates)
    assert sum(c['selected'] for c in r.candidate_estimates)==4
    assert r.summary['local_inference_executions']==12
    assert r.summary['edge_inference_executions']==0
    assert len(r.scores)==4
    assert all(c['risk']['failure_threshold']==3 for c in r.candidate_estimates)
    assert all(c['cost']['ideal_deadline_feasible'] for c in r.candidate_estimates)


def test_isolating_legacy_comparison_preserves_outcomes_and_resources():
    a=config(policy='always_share',episodes=9,enabled=False,energy=True)
    b=config(policy='always_share',episodes=9,enabled=True,energy=True)
    ra,rb=run_simulation(a),run_simulation(b)
    assert ra.scores == rb.scores
    assert ra.inferences == rb.inferences
    for key,value in ra.summary.items():
        assert rb.summary[key]==value, key
    assert rb.candidate_estimates
    assert ra.candidate_estimates is None


def test_side_effect_free_analytic_estimation_and_exact_no_contention_latency_and_energy():
    cfg=config(agents=('a','b','c'),cid='cooperative_fusion',energy=True)
    sim=Simulator(cfg)
    before=(tuple(sim.edge_resource.available_at_us),tuple(sim.link.available_at_us),
            tuple(tuple(r.available_at_us) for r in sim.local_resources.values()),
            sim.energy.by_device.copy())
    result=sim.run()
    assert result.summary['local_inference_executions']==3
    assert result.summary['edge_inference_executions']==1
    after_evals=[c for c in result.candidate_estimates if c['candidate_id']=='cooperative_fusion']
    assert len(after_evals)==1
    predicted=after_evals[0]['cost']
    delivered=[lat for row in result.scores for lat in row['latency_ms'].values() if lat is not None]
    assert predicted['mean_delivery_latency_ms']==pytest.approx(sum(delivered)/len(delivered),abs=.001)
    assert predicted['completion_latency_ms']==pytest.approx(max(delivered),abs=.001)
    assert predicted['incremental_energy_j']==pytest.approx(result.summary['energy_workload_incremental_j'])
    # Only the selected graph can affect actual resource counts, not all candidates.
    assert sim.edge_resource.jobs==1
    assert sum(r.jobs for r in sim.local_resources.values())==3
    assert sim.link.jobs==6
    assert len(sim._active_batches)==0


def test_estimating_candidate_does_not_reserve_workers():
    sim=Simulator(config(episodes=1))
    snapshot=sim._resource_snapshot()
    batch=sim.world.batches[0]
    obs, candidates=sim._assess_candidates(batch, 200000, .17)
    assert sim._resource_snapshot()==snapshot
    assert sim.edge_resource.jobs==sim.link.jobs==0
    assert not sim.inferences and not sim._pending
    assert len(candidates)==7
    assert obs.resource_snapshot is snapshot or obs.resource_snapshot==snapshot


def test_coordinator_observation_is_recursively_readonly_and_no_private_truth():
    class Inspect:
        saw=False
        def select(self, observation, candidates):
            assert isinstance(observation,CoordinatorObservation)
            assert 'truth' not in observation.__dataclass_fields__
            assert 'candidates' not in observation.__dataclass_fields__
            assert 'world' not in observation.__dataclass_fields__
            assert 'event' not in observation.__dataclass_fields__
            assert 'error_count' not in observation.__dataclass_fields__
            assert 'visibility' in dict(observation.scene_features)
            with pytest.raises(FrozenInstanceError):
                observation.time_us=55
            with pytest.raises(TypeError):
                observation.scene_features['visibility']=.9
            with pytest.raises(FrozenInstanceError):
                observation.resource_snapshot.in_flight_workflows=200
            assert all(isinstance(x,tuple) for x in (observation.scene_features,
                       observation.completed_inferences, observation.resource_snapshot.local))
            self.saw=True
            return 'shared_edge:b'
    selector=Inspect()
    r=run_simulation(config(episodes=3), coordinator_selector=selector)
    assert selector.saw
    assert r.summary['edge_inference_executions']==3
    assert all(p['selected_candidate_id']=='shared_edge:b' for p in r.policy_plans)
    assert all(i['source_agent']=='b' for i in r.inferences)


def test_synthetic_features_do_not_read_truth_or_error_draws():
    cfg=config(episodes=10)
    sim=Simulator(cfg)
    values={b.batch_id: sim._public_features(b) for b in sim.world.batches}
    # Perturb world errors and all truths: public signals must be identical.
    modified=deepcopy(cfg.data)
    modified['world'].update(local_error=1.0,edge_error=1.0,local_correlation=1.0)
    sim2=Simulator(ExperimentConfig.from_dict(modified))
    assert values=={b.batch_id:sim2._public_features(b) for b in sim2.world.batches}


def test_risk_estimator_group_coupling_thresholds_and_misspecification():
    cfg=config(agents=('a','b','c'))
    sim=Simulator(cfg)
    obs,_=sim._assess_candidates(sim.world.batches[0],200000,.3)
    params={'kind':'constant','local_error':.2,'edge_error':.1,'fusion_error':.05,
            'correlation':0.0,'context_feature':None,'context_slope':0,'bias':0}
    estimator=SyntheticRiskEstimator(params)
    catalog={c.candidate_id:c for c in StandardCandidateProvider().generate(obs)}
    local=estimator.estimate(obs,catalog['local_only'],failure_threshold=3)
    shared=estimator.estimate(obs,catalog['shared_edge:a'],failure_threshold=3)
    partial=estimator.estimate(obs,catalog['partial_share'],failure_threshold=2)
    fused=estimator.estimate(obs,catalog['cooperative_fusion'],failure_threshold=3)
    assert local.all_agent_error_probability==pytest.approx(.2**3)
    assert shared.all_agent_error_probability==pytest.approx(.1)
    assert partial.at_least_m_error_probability==pytest.approx(.1)
    assert fused.all_agent_error_probability==pytest.approx(.05)
    params['kind']='misspecified';params['bias']=.9
    incorrect=SyntheticRiskEstimator(params).estimate(obs,catalog['local_only'],failure_threshold=3)
    assert incorrect.all_agent_error_probability==1


def test_queue_snapshot_is_immutable_and_changes_predicted_latency():
    base=config(agents=('a','b'),episodes=1,interval=1000)
    # By default a central edge service cannot magically read agent-local queues.
    assert Simulator(base)._resource_snapshot().local[0][1] is None
    modified=deepcopy(base.data)
    modified['coordination']['resource_visibility']='reported_all'
    sim=Simulator(ExperimentConfig.from_dict(modified))
    batch=sim.world.batches[0]
    obs,candidates=sim._assess_candidates(batch, 200000,.1)
    c=next(c for c in candidates if c.candidate_id=='local_only')
    snap=obs.resource_snapshot
    assert c.cost.completion_latency_ms==20
    assert isinstance(snap.edge.next_free_us,tuple)
    with pytest.raises(FrozenInstanceError):
        snap.edge.next_free_us=(500000,)
    # A queue commitment at the vehicle is explicitly visible to the predictor.
    sim.local_resources['a'].available_at_us[0]=100000
    obs2, cands2=sim._assess_candidates(batch, 200000,.1)
    c2=next(c for c in cands2 if c.candidate_id=='local_only')
    assert c2.cost.completion_latency_ms==120
    assert not sim.local_resources['b'].jobs


def _write_trace(tmp_path, *, differing=False, extra=False):
    t=tmp_path/'trace.jsonl'
    rows=[]
    for roi in range(2):
        row={'event_id':f'roi{roi}', 'frame_id':'scene0/frame0', 'arrival_slot':0,
             'truth':roi, 'public_features':{'visibility':.8 if (roi==1 and differing) else .5,
                                             'secret_not_allowlisted':999},
             'candidates':{'local:a':roi, 'local:b':roi,
                           'edge:a':roi, 'edge:b':roi, 'cooperative:fused':roi}}
        rows.append(row)
    t.write_text(''.join(json.dumps(x)+'\n' for x in rows))
    return t


def _trace_cfg(path, candidate_ids, selected='local_only'):
    return ExperimentConfig.from_dict({
        'scenario':{'agents':['a','b'],'episodes':2,'arrival_interval_ms':500,'deadline_ms':200},
        'world':{'backend':'trace','trace_file':str(path),'execution_granularity':'frame',
                 'fusion_candidate_key':'edge:a'},
        'resources':{'model_profiles':{
            'local_model':{'device':'local','compute_ms':20,'output_bytes':2048},
            'late_fusion':{'device':'edge','compute_ms':5,'output_bytes':128}}},
        'coordination':{'enabled':True,'visible_features':['visibility'],'candidate_ids':candidate_ids},
        'policy':{'name':'candidate_fixed','candidate_id':selected}
    })


def test_trace_allowlist_blocks_fusion_alias_and_public_feature_leak(tmp_path):
    trace=_write_trace(tmp_path)
    cfg=_trace_cfg(trace,['local_only','cooperative_fusion'])
    r=run_simulation(cfg)
    assert r.summary['episodes']==2
    assert r.summary['physical_execution_batches']==1
    assert r.summary['local_inference_executions']==2
    assert len(r.candidate_estimates)==2
    assert r.candidate_estimates[0]['public_scene_features']=={'visibility':.5}
    assert 'secret_not_allowlisted' not in str(r.candidate_estimates)
    assert r.candidate_estimates[0]['frame_id']=='scene0/frame0'
    bad=_trace_cfg(trace,['local_only','independent_edge'])
    with pytest.raises(ValueError, match='aliases recorded cooperative'):
        run_simulation(bad)
    bad2=_trace_cfg(trace,['local_only','shared_edge:a'])
    with pytest.raises(ValueError, match='aliases recorded cooperative'):
        run_simulation(bad2)


def test_trace_frame_feature_consistency_and_missing_features(tmp_path):
    trace=_write_trace(tmp_path,differing=True)
    cfg=_trace_cfg(trace,['local_only'])
    with pytest.raises(TraceError,match='inconsistent public features'):
        World(cfg.data,cfg.resolve_trace())
    text=trace.read_text()
    trace.write_text(text.replace('"visibility": 0.8', '"not_whitelisted": 0.8'))
    with pytest.raises(TraceError,match='Missing allowlisted'):
        World(cfg.data,cfg.resolve_trace())


def test_trace_requires_explicit_inventory():
    with pytest.raises(ConfigurationError,match='explicit candidate_ids'):
        ExperimentConfig.from_dict({'coordination':{'enabled':True},
                                    'world':{'backend':'trace','trace_file':'trace.jsonl'}})


def test_invalid_config_and_selected_candidate_rejected():
    bad=[
        {'coordination':{'enabled':True,'visible_features':['truth']}},
        {'coordination':{'enabled':True,'visible_features':['x']}},
        {'coordination':{'enabled':True,'visible_features':['x'],'synthetic_features':{'x':{'kind':'uniform','low':2,'high':1}}}},
        {'coordination':{'enabled':True,'risk_estimator':{'kind':'contextual','context_feature':'missing'}}},
        {'coordination':{'enabled':True,'candidate_ids':['local_only','local_only']}},
        {'coordination':{'enabled':False},'policy':{'name':'candidate_fixed','candidate_id':'local_only'}},
    ]
    for c in bad:
        with pytest.raises(ConfigurationError):
            ExperimentConfig.from_dict(c)
    cfg=config(cid='not_a_candidate')
    with pytest.raises(ValueError,match='unavailable candidate'):
        run_simulation(cfg)


def test_unknown_external_selector_and_risk_method_cannot_act_as_oracle():
    class BadSelector:
        def select(self, observation, candidates):
            return 'oracle_plan'
    with pytest.raises(ValueError,match='unavailable candidate'):
        run_simulation(config(),coordinator_selector=BadSelector())


def test_coordinator_candidate_output_survives_cli_roundtrip(tmp_path):
    from octopus_sim.cli import save
    cfg=config(episodes=2)
    r=run_simulation(cfg)
    save(r,tmp_path)
    rows=[json.loads(l) for l in (tmp_path/'candidates.jsonl').read_text().splitlines()]
    assert len(rows)==len(r.candidate_estimates)
    assert json.loads((tmp_path/'manifest.json').read_text())['coordination_contract']=='policy_observation_v1'
    assert rows[0]['risk']['estimator_id']=='synthetic_contextual'


def test_enabling_coordination_cannot_duplicate_frame_execution(tmp_path):
    trace=_write_trace(tmp_path)
    cfg=_trace_cfg(trace,['local_only','cooperative_fusion'],selected='cooperative_fusion')
    r=run_simulation(cfg)
    assert r.summary['local_inference_executions']==2
    assert r.summary['edge_inference_executions']==1
    assert r.summary['physical_execution_batches']==1
    assert len(r.scores)==2
    assert r.scores[0]['result_ids']['a']==r.scores[1]['result_ids']['a']
    assert len(r.candidate_estimates)==2  # candidate evaluated once per frame, not ROI


def test_five_agent_synthetic_threshold_risk_is_a_valid_joint_distribution():
    sim=Simulator(config(agents=('a','b','c','d','e'),episodes=1))
    obs,_=sim._assess_candidates(sim.world.batches[0],200000,.5)
    params={'kind':'constant','local_error':.2,'edge_error':.15,'fusion_error':.05,
            'correlation':.25,'context_feature':None,'context_slope':0,'bias':0}
    estimator=SyntheticRiskEstimator(params)
    local=next(c for c in StandardCandidateProvider().generate(obs) if c.candidate_id=='local_only')
    risks=[estimator.estimate(obs,local,failure_threshold=m).at_least_m_error_probability
           for m in range(1,6)]
    assert risks==sorted(risks,reverse=True)
    assert risks[-1]==pytest.approx(.25*.2+.75*.2**5)
    assert all(0<=p<=1 for p in risks)


def test_edge_history_is_visible_only_after_edge_compute_completed():
    class Collector:
        def __init__(self): self.observations=[]
        def select(self, observation, candidates):
            self.observations.append(observation)
            return 'shared_edge:a'
    select=Collector()
    run_simulation(config(episodes=3,interval=250),coordinator_selector=select)
    assert [len(x.completed_inferences) for x in select.observations]==[0,1,2]
    assert [x.completed_inferences[-1].source_agents for x in select.observations[1:]]==[('a',),('a',)]
    class LocalCollector(Collector):
        def select(self, observation, candidates):
            self.observations.append(observation)
            return 'local_only'
    loc=LocalCollector()
    run_simulation(config(episodes=3,interval=250),coordinator_selector=loc)
    assert all(not x.completed_inferences for x in loc.observations)


def test_unknown_agent_queues_are_not_leaked_when_edge_only():
    sim=Simulator(config(agents=('a','b')))
    sim.local_resources['a'].available_at_us[0]=1_000_000
    obs,candidates=sim._assess_candidates(sim.world.batches[0],2_000_000,.1)
    assert all(q is None for _,q in obs.resource_snapshot.local)
    local=next(c for c in candidates if c.candidate_id=='local_only')
    assert local.cost.completion_latency_ms==20  # optimistic free-queue approximation
    assert 'unknown_local_queues_treated_as_free_at_decision_time' in local.cost.assumptions


def test_trace_custom_provider_cannot_evade_candidate_allowlist(tmp_path):
    cfg=_trace_cfg(_write_trace(tmp_path), ['local_only'])
    class InvalidProvider:
        def generate(self, observation):
            return StandardCandidateProvider().generate(observation)
    with pytest.raises(ValueError,match='exceeded explicit provenance allowlist'):
        run_simulation(cfg,candidate_provider=InvalidProvider())


def test_custom_estimators_are_injected_without_world_access():
    class RiskSpy:
        calls=0
        def estimate(self,observation,candidate,*,failure_threshold):
            self.calls+=1
            assert not hasattr(observation,'truth')
            return SyntheticRiskEstimator({
                'kind':'constant','local_error':.01,'edge_error':.02,'fusion_error':.03,
                'correlation':0.0,'context_feature':None,'context_slope':0,'bias':0
            }).estimate(observation,candidate,failure_threshold=failure_threshold)
    class CostSpy:
        calls=0
        def __init__(self, cfg):
            self.engine=AnalyticalResourceCostEstimator(cfg.data['resources'],cfg.data['energy'])
        def estimate(self,observation,candidate,graph):
            self.calls+=1
            assert not hasattr(observation,'world')
            return self.engine.estimate(observation,candidate,graph)
    cfg=config(episodes=4)
    risk=RiskSpy();cost=CostSpy(cfg)
    r=run_simulation(cfg,risk_estimator=risk,resource_estimator=cost)
    assert risk.calls==cost.calls==len(r.candidate_estimates)
    assert r.candidate_estimates[0]['risk']['marginal_by_agent'][0][1]==pytest.approx(.01)


def test_failure_to_validate_candidate_is_atomic_and_does_not_execute():
    sim=Simulator(config(episodes=1))
    class DuplicateProvider:
        def generate(self,observation):
            c=StandardCandidateProvider().generate(observation)[0]
            return (c,c)
    sim.candidate_provider=DuplicateProvider()
    with pytest.raises(ValueError,match='duplicate IDs'):
        sim._assess_candidates(sim.world.batches[0],200000,.1)
    assert sim.candidate_estimates==[]
    assert sim.edge_resource.jobs==sim.link.jobs==0


def test_inflight_workflows_explicitly_flag_estimator_limitations():
    cfg=config(episodes=3,interval=0)
    r=run_simulation(cfg)
    snapshots={x['batch_id']:x['snapshot_in_flight_workflows'] for x in r.candidate_estimates}
    assert snapshots['event_000000']==0
    assert snapshots['event_000001']>=1
    assert all('no_unreserved_future_work_from_in_flight_graphs' in c['cost']['assumptions']
               for c in r.candidate_estimates)


def test_enabled_coordination_rejects_legacy_policy_outside_validated_catalog(tmp_path):
    cfg=_trace_cfg(_write_trace(tmp_path),['local_only','cooperative_fusion'])
    legacy=cfg.with_policy('always_independent')
    with pytest.raises(ValueError,match='not in the explicit coordinator candidate inventory'):
        run_simulation(legacy)
    good=cfg.with_policy('cooperative_late_fusion')
    result=run_simulation(good)
    assert all(x['selected_candidate_id']=='cooperative_fusion' for x in result.policy_plans)


def test_disabled_coordination_does_not_validate_legacy_extra_trace_fields(tmp_path):
    trace=_write_trace(tmp_path)
    data=[json.loads(x) for x in trace.read_text().splitlines()]
    for row in data:
        row['public_features']={'arbitrary_legacy_payload':{'unrecognized':True}}
    trace.write_text(''.join(json.dumps(row)+'\n' for row in data))
    cfg=ExperimentConfig.from_dict({
        'scenario':{'agents':['a','b']},
        'world':{'backend':'trace','trace_file':str(trace),'execution_granularity':'frame'}
    })
    w=World(cfg.data,cfg.resolve_trace())
    assert len(w.execution_batches())==1
    assert all(e.public_features==() for e in w.events)
