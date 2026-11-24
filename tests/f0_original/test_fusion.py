from dataclasses import replace, FrozenInstanceError
import math
import pytest
from octopus_fusion import (
    Action, AgentEvidence, FusionObservation, ReliabilityEstimate, RiskBounds,
    FusionConfig, SelectiveFuser, unweighted_majority, weighted_vote,
)


def obs(*items, prior=.5, as_of=100):
    return FusionObservation('scene1/roi1','scene1',as_of,tuple(
        AgentEvidence(id_,label,result,group,age, quality)
        for id_,label,result,group,age,quality in items
    ),prior)


def q(**rates):
    return ReliabilityEstimate(tuple(sorted(rates.items())),'synthetic_assumed_q')


def assert_local(out, observation):
    assert out.action == Action.KEEP_LOCAL
    assert dict(out.per_agent_decisions) == {e.agent_id:e.prediction for e in observation.evidence}
    assert out.fused_prediction is None


def test_healthy_consensus_shared(observation):
    est=q(A=.04,B=.06,C=.08)
    o=replace(observation,evidence=tuple(replace(e,prediction=1) for e in observation.evidence))
    out=SelectiveFuser().decide(o,est)
    assert out.action==Action.SHARE_FUSED
    assert set(dict(out.per_agent_decisions).values())=={1}
    assert set(out.supporting_groups)=={'A','B','C'}
    assert out.masked_agents==()
    assert out.score_confidence>.99


def test_high_error_agent_masked_without_copying_its_vote(observation,reliability):
    out=SelectiveFuser().decide(observation,reliability)
    assert out.action==Action.MASK_AND_FUSE
    assert out.fused_prediction==1
    assert dict(out.masked_agents)=={'B':'too_unreliable'}
    assert out.used_groups==('A','C')
    assert dict(out.per_agent_decisions)=={'A':1,'B':1,'C':1}


def test_low_quality_mask_alone_not_evidence_of_fault(observation,reliability):
    # The mask is triggered solely by the explicit estimated error rate;
    # changing hidden truth (not an input here) cannot alter the decision.
    a=SelectiveFuser().decide(observation,reliability)
    b=SelectiveFuser().decide(observation,reliability)
    assert a==b
    assert a.masked_agents==( ('B','too_unreliable'), )


def test_high_confidence_one_source_must_not_force_global_reuse():
    o=obs(('A',1,'rA','same',80,None),('B',1,'rA','same',80,None),('C',0,'rC','independent',80,None))
    out=SelectiveFuser().decide(o,q(A=.001,B=.001,C=.01))
    assert_local(out,o)
    assert 'insufficient_independent_support_for_winner' in out.fallback_reasons
    assert out.used_groups==('independent','same')


def test_duplicate_shared_result_contributes_only_one_group():
    o=obs(('A',1,'physical:r1','shared',80,None),('B',1,'physical:r1','shared',80,None),
          ('C',0,'physical:r2','unique',80,None))
    # Naive majority gets outvoted by duplicate evidence; safe F0 abstains.
    assert unweighted_majority(o)==1
    out=SelectiveFuser().decide(o,q(A=.01,B=.01,C=.11))
    assert_local(out,o)
    assert out.used_groups==('shared','unique')


def test_same_physical_result_cannot_claim_independence():
    with pytest.raises(ValueError,match='independent'):
        obs(('A',1,'same-result','A',80,None),('B',1,'same-result','B',80,None))


def test_same_physical_result_cannot_contradict_same_roi():
    with pytest.raises(ValueError,match='contradict'):
        obs(('A',0,'same-result','same',80,None),('B',1,'same-result','same',80,None))


def test_same_result_identity_can_project_different_rois():
    a=FusionObservation('frame0/roi0','frame0',100,(
        AgentEvidence('A',0,'frame0/rA','A',90),AgentEvidence('B',1,'frame0/rB','B',90)))
    b=FusionObservation('frame0/roi1','frame0',100,(
        AgentEvidence('A',1,'frame0/rA','A',90),AgentEvidence('B',0,'frame0/rB','B',90)))
    assert a.evidence[0].result_id==b.evidence[0].result_id
    assert a.evidence[0].prediction != b.evidence[0].prediction


def test_conflicting_dependent_group_entirely_masked():
    o=obs(('A',1,'rA','shared',80,None),('B',0,'rB','shared',80,None),
          ('C',1,'rC','unique',80,None))
    out=SelectiveFuser().decide(o,q(A=.01,B=.01,C=.01))
    assert_local(out,o)
    assert dict(out.masked_agents)=={'A':'dependence_conflict','B':'dependence_conflict'}
    assert out.used_groups==('unique',)


def test_conflicting_dependent_group_not_cherry_picked_if_one_bad():
    o=obs(('A',1,'rA','shared',80,None),('B',0,'rB','shared',80,None),
          ('C',1,'rC','unique',80,None))
    out=SelectiveFuser().decide(o,q(A=.01,B=.49,C=.01))
    assert_local(out,o)
    assert dict(out.masked_agents)['A']=='dependence_conflict'
    assert dict(out.masked_agents)['B']=='dependence_conflict'


def test_age_mask_and_future_rejected():
    o=obs(('A',1,'rA','A',0,None),('B',1,'rB','B',80,None),('C',1,'rC','C',90,None))
    out=SelectiveFuser(FusionConfig(max_age_ms=50)).decide(o,q(A=.02,B=.02,C=.02))
    assert out.action==Action.MASK_AND_FUSE
    assert dict(out.masked_agents)['A']=='stale'
    with pytest.raises(ValueError,match='future'):
        obs(('A',1,'rA','A',101,None),('B',1,'rB','B',80,None))


def test_both_agents_disagree_without_tie_breaker():
    o=obs(('A',0,'rA','A',95,None),('B',1,'rB','B',95,None))
    out=SelectiveFuser().decide(o,q(A=.03,B=.08))
    assert_local(out,o)
    assert 'insufficient_independent_support_for_winner' in out.fallback_reasons
    assert 'excessive_conflict' in out.fallback_reasons


def test_all_groups_agree_but_dependence_groups_fewer_than_minimum():
    o=obs(('A',1,'rA','g',95,None),('B',1,'rB','g',95,None),('C',1,'rC','g',95,None))
    out=SelectiveFuser().decide(o,q(A=.05,B=.06,C=.07))
    assert_local(out,o)
    assert 'insufficient_independent_groups' in out.fallback_reasons


def test_minimum_posterior_score_requirement():
    o=obs(('A',1,'rA','A',95,None),('B',1,'rB','B',95,None))
    out=SelectiveFuser(FusionConfig(min_score_confidence=.9999)).decide(o,q(A=.20,B=.21))
    assert_local(out,o)
    assert 'low_surrogate_confidence' in out.fallback_reasons


def test_strict_external_risk_bound_absent_falls_back():
    o=obs(('A',1,'rA','A',95,None),('B',1,'rB','B',95,None))
    cfg=FusionConfig(require_risk_bounds=True)
    out=SelectiveFuser(cfg).decide(o,q(A=.02,B=.02))
    assert_local(out,o)
    assert 'external_risk_bound_unavailable' in out.fallback_reasons


def test_external_shared_error_budget():
    o=obs(('A',1,'rA','A',95,None),('B',1,'rB','B',95,None))
    cfg=FusionConfig(max_shared_error_upper=.04)
    risk=RiskBounds(shared_error_upper=.05)
    out=SelectiveFuser(cfg).decide(o,q(A=.01,B=.02),risk_bounds=risk)
    assert_local(out,o)
    assert 'shared_risk_budget_exceeded' in out.fallback_reasons
    risk=RiskBounds(shared_error_upper=.03)
    assert SelectiveFuser(cfg).decide(o,q(A=.01,B=.02),risk_bounds=risk).shared


def test_relative_risk_guard_requires_local_LOWER_bound_not_upper():
    o=obs(('A',1,'rA','A',95,None),('B',1,'rB','B',95,None))
    cfg=FusionConfig(max_risk_increase_over_local=.01)
    out=SelectiveFuser(cfg).decide(o,q(A=.01,B=.02),risk_bounds=RiskBounds(shared_error_upper=.03))
    assert_local(out,o)
    assert 'local_risk_bound_unavailable' in out.fallback_reasons
    out=SelectiveFuser(cfg).decide(o,q(A=.01,B=.02),risk_bounds=RiskBounds(shared_error_upper=.03,local_joint_error_lower=.01))
    assert_local(out,o)
    assert 'risk_increase_budget_exceeded' in out.fallback_reasons
    out=SelectiveFuser(cfg).decide(o,q(A=.01,B=.02),risk_bounds=RiskBounds(shared_error_upper=.02,local_joint_error_lower=.02))
    assert out.shared


def test_input_order_invariance_and_immutability(observation,reliability):
    f=SelectiveFuser()
    original=f.decide(observation,reliability)
    perm=replace(observation,evidence=tuple(reversed(observation.evidence)))
    assert f.decide(perm,reliability)==original
    with pytest.raises(FrozenInstanceError):
        observation.as_of_ms=999
    with pytest.raises(FrozenInstanceError):
        original.action=Action.KEEP_LOCAL


def test_missing_estimate_rejected(observation):
    with pytest.raises(ValueError,match='exactly'):
        SelectiveFuser().decide(observation,q(A=.1,C=.1))


def test_wrong_probability_zero_one_nan_and_bool_rejected():
    for invalid in (0,1,float('nan'),float('inf'),True,-.1,1.1):
        with pytest.raises(ValueError):
            ReliabilityEstimate((('A',invalid),('B',.1)))


def test_uncalibrated_confidence_is_not_claimed_as_risk_bound():
    o=obs(('A',1,'rA','A',95,None),('B',1,'rB','B',95,None))
    out=SelectiveFuser().decide(o,q(A=.001,B=.001))
    assert out.score_confidence>.9999
    assert not hasattr(out,'calibrated_posterior')
    assert not hasattr(out,'guaranteed_joint_failure_probability')


def test_unweighted_and_naive_weighted_baselines_document_ties():
    o=obs(('A',1,'rA','A',95,None),('B',0,'rB','B',95,None))
    assert unweighted_majority(o) is None
    assert weighted_vote(o,q(A=.1,B=.1)) is None
    assert weighted_vote(o,q(A=.01,B=.4))==1


def test_invalid_configurations_rejected():
    for kw in ({'max_age_ms':-1}, {'dependence_discount':-1},
               {'min_supporting_groups':1}, {'max_conflict_fraction':2},
               {'min_score_confidence':1}, {'max_source_error':.6}):
        with pytest.raises(ValueError):FusionConfig(**kw)


def test_no_cross_frame_binary_cache():
    o1=obs(('A',1,'frame0/rA','A',95,None),('B',1,'frame0/rB','B',95,None))
    o2=replace(o1,frame_id='scene2',event_id='scene2/roi',evidence=tuple(
        replace(e,prediction=0,result_id='scene2/'+e.agent_id) for e in o1.evidence))
    f=SelectiveFuser()
    assert f.decide(o1,q(A=.02,B=.03)).fused_prediction==1
    assert f.decide(o2,q(A=.02,B=.03)).fused_prediction==0


def test_five_agents_and_single_bad_agent_masked():
    o=obs(*[(str(i),1 if i!=4 else 0,f'r{i}',f'g{i}',95,None) for i in range(5)])
    rel=q(**{str(i):(.40 if i==4 else .06) for i in range(5)})
    out=SelectiveFuser().decide(o,rel)
    assert out.action==Action.MASK_AND_FUSE
    assert len(out.supporting_groups)==4
    assert len(out.masked_agents)==1


def test_reliability_group_consensus_uses_conservative_error():
    o1=obs(('A',1,'shared','g',95,None),('C',1,'rC','C',95,None))
    o2=obs(('A',1,'shared','g',95,None),('B',1,'shared','g',95,None),('C',1,'rC','C',95,None))
    r1=SelectiveFuser().decide(o1,q(A=.20,C=.11))
    r2=SelectiveFuser().decide(o2,q(A=.20,B=.05,C=.11))
    assert r1.surrogate_score==pytest.approx(r2.surrogate_score)


def test_probability_prior_and_discounter_shift_score():
    o=obs(('A',1,'rA','A',95,None),('B',1,'rB','B',95,None),prior=.2)
    a=SelectiveFuser(FusionConfig(dependence_discount=1.)).decide(o,q(A=.1,B=.1))
    b=SelectiveFuser(FusionConfig(dependence_discount=.25)).decide(o,q(A=.1,B=.1))
    assert a.surrogate_score>b.surrogate_score


def test_offline_policy_cannot_mutate_observation():
    o=obs(('A',1,'rA','A',95,None),('B',1,'rB','B',95,None))
    before=repr(o)
    SelectiveFuser().decide(o,q(A=.1,B=.1))
    assert repr(o)==before


def test_extremely_small_error_rates_numerically_stable():
    o=obs(('A',1,'rA','A',95,None),('B',1,'rB','B',95,None))
    value=1e-320
    out=SelectiveFuser().decide(o,q(A=value,B=value))
    assert out.shared
    assert math.isfinite(out.surrogate_score)
    assert out.score_confidence==1


def test_discount_zero_cannot_imply_calibrated_fusion_confidence():
    o=obs(('A',1,'rA','A',95,None),('B',1,'rB','B',95,None))
    out=SelectiveFuser(FusionConfig(dependence_discount=0)).decide(o,q(A=.01,B=.01))
    assert_local(out,o)
    assert 'ambiguous_vote' in out.fallback_reasons


def test_invalid_decision_contract_refused():
    from octopus_fusion import FusionDecision
    with pytest.raises(ValueError,match='Local fallback'):
        FusionDecision('e','f',Action.KEEP_LOCAL,(('A',0),('B',1)),1,0,.5,
                       (),(),(),('test',),'x')
    with pytest.raises(ValueError,match='same fused'):
        FusionDecision('e','f',Action.SHARE_FUSED,(('A',0),('B',1)),1,0,.5,
                       (),(),(),(),'x')
    with pytest.raises(ValueError,match='explicitly masked'):
        FusionDecision('e','f',Action.MASK_AND_FUSE,(('A',1),('B',1)),1,0,.5,
                       (),(),(),(),'x')


def test_monotonic_weight_on_single_independent_source():
    o=obs(('A',1,'rA','A',95,None),('B',1,'rB','B',95,None))
    strong=SelectiveFuser().decide(o,q(A=.03,B=.2))
    weaker=SelectiveFuser().decide(o,q(A=.13,B=.2))
    assert strong.surrogate_score>weaker.surrogate_score
