"""Audit of paired three-agent frontier, extension gating, and physical-work costs."""
from __future__ import annotations
from dataclasses import FrozenInstanceError
from pathlib import Path
import csv
import json

import numpy as np
import pytest

from octopus_fusion.contracts import (AgentEvidence,FusionConfig,FusionObservation,
                                      ReliabilityEstimate,Action)
from octopus_fusion.fusion import SelectiveFuser
from scripts.run_three_agent_tradeoff_frontier import (
    AGENTS, P_TRUE, generate_arrays, make_fuser, observations, evaluate_offline,
    make_trace, evaluate_online, evaluate_deadline, bootstrap_pairs, main,
    DEFAULT_CONF, DiagnosticExpertOverride,
)


def test_paired_generator_reproducible_and_source_marginals():
    t1,p1=generate_arrays(1500,4,712,.18)
    t2,p2=generate_arrays(1500,4,712,.18)
    assert np.array_equal(t1,t2) and np.array_equal(p1,p2)
    actual=(p1!=t1[:,:,None]).mean(axis=(0,1))
    assert np.allclose(actual,P_TRUE,atol=.016),actual
    # A has independent error draws across shock probabilities; B/C retain
    # their marginals after conditional-rate adjustment.
    _,p3=generate_arrays(1500,4,712,.06)
    assert np.array_equal(p1[:,:,0],p3[:,:,0])
    err3=(p3!=t1[:,:,None]).mean(axis=(0,1))
    assert np.allclose(err3,P_TRUE,atol=.017),err3
    with pytest.raises(ValueError): generate_arrays(20,2,1,.40)


def test_three_agent_strict_f0_cannot_correct_majority_wrong_any_binary_pattern():
    from itertools import product
    q=ReliabilityEstimate(tuple(zip(AGENTS,P_TRUE)))
    strict=make_fuser('strict')
    for bits in product((0,1),repeat=3):
        obs=FusionObservation('roi','frame',20.,tuple(AgentEvidence(
            a,int(bit),f'result:{a}',a,15.) for a,bit in zip(AGENTS,bits)))
        d=strict.decide(obs,q)
        for truth in (0,1):
            k=sum(bit!=truth for bit in bits)
            new_k=sum(y!=truth for _,y in d.per_agent_decisions)
            assert (k>=2)==(new_k>=2)
            assert new_k>=k if k==3 else True


def test_diagnostic_expert_override_breaks_majority_only_when_all_other_checks_pass():
    q=ReliabilityEstimate((('a',.015),('b',.26),('c',.32)))
    obs=FusionObservation('roi','frame',20.,(
        AgentEvidence('a',1,'a','a',15.),AgentEvidence('b',0,'b','b',15.),
        AgentEvidence('c',0,'c','c',15.)))
    strict=make_fuser('strict').decide(obs,q)
    assert strict.action==Action.KEEP_LOCAL
    assert strict.fallback_reasons==('insufficient_independent_support_for_winner',)
    expert=make_fuser('expert').decide(obs,q)
    assert expert.shared and expert.fused_prediction==1
    assert all(y==1 for _,y in expert.per_agent_decisions)
    # Downweighting strong source enough removes override permission.
    qb=ReliabilityEstimate((('a',.06),('b',.26),('c',.32)))
    assert not make_fuser('expert').decide(obs,qb).shared
    # A source that is STALE cannot be promoted by the diagnostic extension.
    stale=FusionObservation('roi','frame',200.,(
        AgentEvidence('a',1,'a','a',0.),AgentEvidence('b',0,'b','b',198.),
        AgentEvidence('c',0,'c','c',198.)))
    assert not make_fuser('expert').decide(stale,q).shared


def test_expert_override_uses_no_ground_truth_and_keeps_original_f0_unchanged():
    truth,pred=generate_arrays(100,2,199,.18)
    original,_,acc,decisions=evaluate_offline(truth,pred,'expert')
    flipped,_,acc2,decisions2=evaluate_offline(1-truth,pred,'expert')
    assert np.array_equal(acc,acc2)
    assert np.array_equal(decisions,decisions2)
    assert original['share_roi_fraction']==flipped['share_roi_fraction']
    assert original['marginal_error']+flipped['marginal_error']==pytest.approx(1.)
    # Default F0 still requires two supporters (extension is a separate wrapper).
    assert SelectiveFuser().config.min_supporting_groups==2


def test_strict_r2_invariant_and_override_risk_tradeoff():
    t,p=generate_arrays(500,2,77109,.18)
    loc,_,_,_=evaluate_offline(t,p,'local')
    strict,_,_,_=evaluate_offline(t,p,'strict')
    expert,_,_,_=evaluate_offline(t,p,'expert')
    assert loc['r_ge_2']==strict['r_ge_2']
    assert strict['r_all_3']>=loc['r_all_3']
    assert expert['r_ge_2']<loc['r_ge_2']
    assert expert['r_all_3']>loc['r_all_3']


def test_miscalibration_can_reverse_majority_risk_gain():
    # Same underlying B/C correlated shocks, but A's *true* error is worse
    # than what the edge was told.
    good_t,good_p=generate_arrays(600,2,77109,.18,expert_true_error=.015)
    bad_t,bad_p=generate_arrays(600,2,77109,.18,expert_true_error=.45)
    good_local,_,_,_=evaluate_offline(good_t,good_p,'local')
    good_exp,_,_,_=evaluate_offline(good_t,good_p,'expert',assumed_a=.015)
    bad_local,_,_,_=evaluate_offline(bad_t,bad_p,'local')
    bad_exp,_,_,_=evaluate_offline(bad_t,bad_p,'expert',assumed_a=.015)
    assert good_exp['r_ge_2']<good_local['r_ge_2']
    assert bad_exp['r_ge_2']>bad_local['r_ge_2']
    assert bad_exp['r_all_3']>bad_local['r_all_3']


def test_online_offline_replay_matches_inference_identity_and_energy(tmp_path):
    t,p=generate_arrays(40,3,512,.18)
    trace=make_trace(tmp_path/'input.jsonl',t,p)
    for method in ('local','majority','strict','expert'):
        row=evaluate_online(trace,t,p,method,512)
        assert row['local_compute_jobs']==120
        assert row['edge_compute_jobs']==(0 if method=='local' else 40)
        assert row['energy_per_frame_j']>.23
        assert row['mean_delivered_latency_ms']>=20
        assert row['frame_with_downlink_fraction']==pytest.approx(
            row['actual_frame_downlink_fraction'])
    assert evaluate_online(trace,t,p,'local',512)['energy_per_frame_j'] < (
        evaluate_online(trace,t,p,'strict',512)['energy_per_frame_j'])


def test_deadline_cliff_preserves_local_decisions_and_sunk_energy(tmp_path):
    t,p=generate_arrays(36,2,512,.18)
    trace=make_trace(tmp_path/'input.jsonl',t,p)
    loc=evaluate_deadline(trace,t,p,'local',21,512)
    early=evaluate_deadline(trace,t,p,'expert',21,512)
    late=evaluate_deadline(trace,t,p,'expert',40,512)
    assert early['applied_agent_update_fraction']==0
    assert early['marginal_error']==pytest.approx(loc['marginal_error'])
    assert early['energy_per_frame_j']>loc['energy_per_frame_j']
    assert late['applied_agent_update_fraction']>=early['applied_agent_update_fraction']
    assert late['energy_per_frame_j']==pytest.approx(early['energy_per_frame_j'])
    assert late['agent_deadline_miss_rate']==0


def test_frame_level_output_cost_amortized_by_roi_count(tmp_path):
    t,p=generate_arrays(35,4,190,.18)
    energies={}
    for k in (1,2,4):
        trace=make_trace(tmp_path/f'rois{k}.jsonl',t[:,:k],p[:,:k])
        row=evaluate_online(trace,t[:,:k],p[:,:k],'strict',190,threshold=.999)
        energies[k]=row['energy_per_frame_j']
        assert row['share_roi_fraction']==0
        assert row['frame_with_downlink_fraction']==0
    assert energies[1]==pytest.approx(energies[2])==pytest.approx(energies[4])
    assert energies[4]/4==pytest.approx(energies[1]/4)


def test_paired_frame_bootstrap_reproducible_and_zero_delta():
    rng=np.random.default_rng(11)
    a=rng.integers(0,2,size=(60,3)).astype(float)
    b=a.copy();b[0,0]+=1
    x=bootstrap_pairs(a,b,112,200)
    y=bootstrap_pairs(a,b,112,200)
    assert all(np.array_equal(q,r) for q,r in zip(x,y))
    z=bootstrap_pairs(a,a,112,200)
    assert all(np.allclose(v,0) for v in z)


def test_analysis_cli_outputs_four_complete_figure_triplets_and_provenance(tmp_path):
    # Small reproducibility smoke in a fresh output directory, not paper stats.
    from subprocess import run
    import sys
    script=Path(__file__).resolve().parents[1]/'scripts/run_three_agent_tradeoff_frontier.py'
    proc=run([sys.executable,str(script),'--output',str(tmp_path),
              '--frames','105','--online-frames','100','--rois','2',
              '--bootstrap','110'],capture_output=True,text=True,check=True)
    assert 'Finished full-DAG simulation' in proc.stdout
    assert len(list(tmp_path.glob('*.pdf')))==4
    assert len(list(tmp_path.glob('*.svg')))==4
    assert len(list(tmp_path.glob('*.png')))==4
    with (tmp_path/'shock_and_policy_grid.csv').open(newline='') as f:
        assert len(list(csv.DictReader(f)))==20
    with (tmp_path/'deadline_reliability_energy_sweep.csv').open(newline='') as f:
        assert len(list(csv.DictReader(f)))==36
    prov=json.loads((tmp_path/'PROVENANCE.json').read_text())
    assert prov['agents']==3
    assert prov['online_frames']==100
    assert prov['energy'].startswith('hypothetical')
