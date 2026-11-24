"""Three-agent correction, reliability and simultaneous-risk invariants."""
from __future__ import annotations
from itertools import product
from copy import deepcopy
import json
from pathlib import Path

import pytest

from octopus_fusion.contracts import AgentEvidence, FusionConfig, FusionObservation, ReliabilityEstimate, Action
from octopus_fusion.fusion import SelectiveFuser
from scripts.run_f0_three_agent_diagnostic import (
    AGENTS,ERROR_RATES, make_traces,config_for,run_method,summarize,frame_metrics,
)


def observation(bits):
    return FusionObservation('event0','frame0',10.,
        tuple(AgentEvidence(a,int(y),f'r:{a}',f'lineage:{a}',8.)
              for a,y in zip(AGENTS,bits)),.5)


def risk():
    return ReliabilityEstimate(tuple(zip(AGENTS,ERROR_RATES)),'configured_synthetic_marginals')


@pytest.mark.parametrize('cutoff',(.7,.9,.95))
@pytest.mark.parametrize('max_error',(.25,.49))
@pytest.mark.parametrize('truth',(0,1))
def test_three_agent_f0_preserves_majority_wrong_indicator_and_never_repairs_all_wrong(cutoff,max_error,truth):
    f=SelectiveFuser(FusionConfig(max_source_error=max_error,min_score_confidence=cutoff))
    for bits in product((0,1), repeat=3):
        d=f.decide(observation(bits),risk())
        original_wrong=sum(y!=truth for y in bits)  # evaluation truth is not given to F0
        new_wrong=sum(y!=truth for _,y in d.per_agent_decisions)
        assert (original_wrong>=2)==(new_wrong>=2)
        if original_wrong==3:
            assert new_wrong==3
        if d.shared:
            assert d.fused_prediction == int(sum(bits)>=2)
            assert new_wrong in (0,3)


def test_fault_repair_and_guard_against_wrongly_weighted_conflict():
    a=observation((1,1,0))
    f=SelectiveFuser(FusionConfig(min_score_confidence=.9,max_source_error=.25))
    decision=f.decide(a,risk())
    assert decision.action==Action.MASK_AND_FUSE
    assert decision.fused_prediction==1
    assert dict(decision.per_agent_decisions)==dict.fromkeys(AGENTS,1)
    assert ('c','too_unreliable') in decision.masked_agents
    assert decision.score_confidence>.98
    conflicting=f.decide(observation((0,1,1)),risk())
    assert conflicting.action==Action.KEEP_LOCAL
    assert conflicting.fused_prediction is None
    assert dict(conflicting.per_agent_decisions)=={'a':0,'b':1,'c':1}


def test_f0_guard_gates_overconfidence_but_source_mask_alone_changes_no_labels(tmp_path):
    t=make_traces(tmp_path,frames=50,rois=2,seed=60421)[.04]
    mask=run_method(t,'f0_mask_unreliable_c',50,2,60421)
    full=run_method(t,'f0_all_sources',50,2,60421)
    assert mask.scores==full.scores
    assert len(mask.fusion_decisions)==len(full.fusion_decisions)==100
    assert all(x['masked_agents'] for x in mask.fusion_decisions)
    assert all(not x['masked_agents'] for x in full.fusion_decisions)
    assert [x['action']!='KEEP_LOCAL' for x in mask.fusion_decisions]==[x['action']!='KEEP_LOCAL' for x in full.fusion_decisions]
    assert [x['fused_prediction'] for x in mask.fusion_decisions]==[x['fused_prediction'] for x in full.fusion_decisions]


def test_trace_data_generator_reproducibility_and_nominal_shock(tmp_path):
    p=make_traces(tmp_path/'a',frames=55,rois=2,seed=92)
    q=make_traces(tmp_path/'b',frames=55,rois=2,seed=92)
    assert {s: p[s].read_bytes() for s in p}=={s:q[s].read_bytes() for s in q}
    for s,trace in p.items():
        rows=[json.loads(x) for x in trace.read_text().splitlines()]
        assert len(rows)==110
        assert len({r['frame_id'] for r in rows})==55
        assert len({r['event_id'] for r in rows})==110
        assert all(set(f'local:{a}' for a in AGENTS).issubset(r['candidates']) for r in rows)


def test_simulator_caches_each_frame_and_records_real_corrected_and_harmed(tmp_path):
    t=make_traces(tmp_path,frames=45,rois=2,seed=60421)[.04]
    result=run_method(t,'f0_mask_unreliable_c',45,2,60421)
    assert result.summary['local_inference_executions']==3*45
    assert result.summary['edge_inference_executions']==45
    assert result.summary['physical_execution_batches']==45
    assert result.summary['agent_deadline_miss_rate']==0
    originals={};truth={}
    for line in t.read_text().splitlines():
        r=json.loads(line);originals[r['event_id']]={a:r['candidates'][f'local:{a}'] for a in AGENTS}
        truth[r['event_id']]=r['truth']
    row=summarize(result,originals,truth,'f0_mask_unreliable_c',.04,45,2)
    assert row['corrected_agent_labels']-row['harmed_agent_labels']==sum(
        (sum(originals[s['event_id']][a]!=truth[s['event_id']] for a in AGENTS)-s['error_count'])
        for s in result.scores)
    assert row['corrected_agent_labels']+row['harmed_agent_labels']==row['changed_agent_labels']
    assert row['energy_j_per_frame']>.24
    assert len(frame_metrics(result))==45


def test_f0_decision_and_actual_cost_does_not_use_ground_truth(tmp_path):
    p=make_traces(tmp_path,frames=40,rois=2,seed=141)[.04]
    q=tmp_path/'flipped.jsonl'
    rows=[json.loads(x) for x in p.read_text().splitlines()]
    for row in rows:row['truth']=1-row['truth']
    q.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    a=run_method(p,'f0_mask_unreliable_c',40,2,141)
    b=run_method(q,'f0_mask_unreliable_c',40,2,141)
    assert [(r['event_id'],r['predictions'],r['result_ids']) for r in a.scores]==[
        (r['event_id'],r['predictions'],r['result_ids']) for r in b.scores]
    assert a.summary['energy_workload_incremental_j']==pytest.approx(b.summary['energy_workload_incremental_j'])
    assert [r['error_count']+t['error_count'] for r,t in zip(a.scores,b.scores)]==[3]*80


def test_paired_policy_outputs_reproducible(tmp_path):
    trace=make_traces(tmp_path,frames=40,rois=2,seed=27)[0.0]
    a=run_method(trace,'f0_overtrust_c',40,2,27)
    b=run_method(trace,'f0_overtrust_c',40,2,27)
    assert a.summary==b.summary
    assert a.scores==b.scores
    assert a.fusion_decisions==b.fusion_decisions


def test_shared_lineage_is_not_counted_as_two_independent_supporters():
    ev=(AgentEvidence('a',1,'shared-physical-result','one-lineage',8.),
        AgentEvidence('b',1,'shared-physical-result','one-lineage',8.),
        AgentEvidence('c',0,'independent-result','independent-lineage',8.))
    obs=FusionObservation('event0','frame0',10.,ev,.5)
    decision=SelectiveFuser(FusionConfig(min_score_confidence=.7)).decide(
        obs,ReliabilityEstimate((('a',.06),('b',.06),('c',.12))))
    assert decision.action==Action.KEEP_LOCAL
    assert dict(decision.per_agent_decisions)=={'a':1,'b':1,'c':0}
    assert len(decision.used_groups)==2
    assert len(decision.supporting_groups)<2
