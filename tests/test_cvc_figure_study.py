"""Statistical, provenance, resource, and reproducibility checks for CVC study."""
import json
import math
from pathlib import Path

import pytest

from scripts.run_cvc_figure_study import (
    SOURCE_CONFIG, SOURCE_SYNTHETIC, aggregate, bootstrap_aggregate,
    candidate_catalog, from_trace, load_scenes, modeled_energy_sensitivity,
    paired_failure_classes, queue_study, resource_endpoints,
    risk_estimator_prediction, main,
)
from octopus_sim.config import ExperimentConfig


def test_nine_scene_source_totals_and_matched_rates():
    scenes=load_scenes()
    assert len(scenes)==9
    assert sum(s['frames'] for s in scenes)==1993
    assert sum(s['balanced_events'] for s in scenes)==3986
    assert sum(s['separate_joint_errors'] for s in scenes)==105
    assert sum(s['late_fusion_errors'] for s in scenes)==194
    a=aggregate(scenes)
    assert math.isclose(a['local_joint'],105/3986)
    assert math.isclose(a['fused_joint'],194/3986)
    assert math.isclose(a['local_marginal'],(670+481)/(2*3986))
    assert a['joint_increase']>0
    assert a['marginal_benefit']>0
    assert sum(aggregate([s])['joint_increase']>0 for s in scenes)==8
    assert sum(aggregate([s])['joint_increase']==0 for s in scenes)==1


def test_source_integrity_fails_closed(tmp_path):
    p=tmp_path/'wrong.csv'
    source=Path(__file__).resolve().parents[1]/'empirical/v2v4real/sequence_counts.csv'
    data=source.read_text().replace('0007,221,442,221','0007,222,442,221')
    p.write_text(data)
    with pytest.raises(ValueError,match='Broken scene'):
        load_scenes(p)


def test_bootstrap_is_paired_scene_level_and_repeatable():
    scenes=load_scenes()
    b1=bootstrap_aggregate(scenes,repetitions=1600,seed=734)
    b2=bootstrap_aggregate(scenes,repetitions=1600,seed=734)
    assert (b1['joint_increase']==b2['joint_increase']).all()
    assert (b1['marginal_benefit']==b2['marginal_benefit']).all()
    assert (b1['joint_increase']==b1['fused_joint']-b1['local_joint']).all()
    assert (b1['marginal_benefit']==b1['local_marginal']-b1['fused_marginal']).all()
    assert .005<b1['joint_increase'].mean()<.05


def test_paired_events_keep_frame_id_and_distinct_roi_answers():
    events=from_trace()
    assert len(events)==442
    assert len({e['frame_id'] for e in events})==221
    assert all(sum(e['frame_id']==f for e in events)==2 for f in {r['frame_id'] for r in events})
    outcome=paired_failure_classes(events)
    assert outcome=={'both_success':426,'introduced_joint_failure':12,
                     'rescued_joint_failure':0,'both_fail':4}


def test_frame_level_executor_matches_energy_cost_anchors():
    c=ExperimentConfig.load(SOURCE_CONFIG)
    costs=resource_endpoints(c)
    local=costs['local_only'];fusion=costs['cooperative_fusion']
    assert local['local_detector_executions']==fusion['local_detector_executions']==442
    assert local['fusion_executions']==0
    assert fusion['fusion_executions']==221
    assert local['transmitted_bytes']==0
    assert fusion['transmitted_bytes']==1867008
    assert math.isclose(local['energy_per_frame_j'],.16,abs_tol=1e-8)
    assert math.isclose(fusion['energy_per_frame_j'],53.79302656/221,abs_tol=1e-8)
    assert math.isclose(local['mean_agent_latency_ms'],20,abs_tol=1e-7)
    assert math.isclose(fusion['mean_agent_latency_ms'],36.339,abs_tol=.01)


def test_energy_assumption_envelopes_do_not_claim_measured_precision():
    costs=resource_endpoints(ExperimentConfig.load(SOURCE_CONFIG))
    for k in (0,.25,.5,.75,1.):
        r=modeled_energy_sensitivity(costs,k)
        assert r['min_energy_per_frame_j']<=r['nominal_energy_per_frame_j']<=r['max_energy_per_frame_j']
        assert len(r['grid'])==9
        assert math.isclose(r['baseline_local_compute_j_per_frame'],.16,abs_tol=1e-8)
    assert math.isclose(modeled_energy_sensitivity(costs,0)['min_energy_per_frame_j'],.16,abs_tol=1e-8)
    assert modeled_energy_sensitivity(costs,1)['min_energy_per_frame_j']>.16


@pytest.mark.parametrize('agents',[2,3,5])
@pytest.mark.parametrize('rho',[0,.1,.5,1])
def test_synthetic_estimator_respects_common_shock_formula(agents,rho):
    p=.1
    actual=risk_estimator_prediction(agents,p,rho)
    assert math.isclose(actual,(1-rho)*p**agents+rho*p,rel_tol=0,abs_tol=1e-12)


def test_synthetic_candidate_catalog_exposes_all_available_plans():
    records=candidate_catalog(ExperimentConfig.load(SOURCE_SYNTHETIC))
    assert len(records)==7
    ids={r['candidate_id'] for r in records}
    assert 'local_only' in ids and 'cooperative_fusion' in ids
    assert 'independent_edge' in ids and 'partial_share' in ids
    assert len([x for x in ids if x.startswith('shared_edge:')])==3
    assert all(r['estimated_incremental_energy_j']>=0 for r in records)
    assert sum(r['selected'] for r in records)==1


def test_cost_forecasts_are_visibility_dependent_not_executions():
    c=ExperimentConfig.load(SOURCE_CONFIG)
    rows=queue_study(c,(10,500))
    assert len(rows)==8
    low=[r for r in rows if r['experiment']=='uncensored_forecast' and r['frame_interval_ms']==10]
    edge=next(r for r in low if r['resource_visibility']=='edge_only')
    reported=next(r for r in low if r['resource_visibility']=='reported_all')
    assert math.isclose(edge['realized_mean_latency_ms'],reported['realized_mean_latency_ms'],abs_tol=1e-7)
    assert edge['predicted_mean_latency_ms']<edge['realized_mean_latency_ms']*.1
    assert math.isclose(reported['predicted_mean_latency_ms'],reported['realized_mean_latency_ms'],abs_tol=1e-6)
    no_load=next(r for r in rows if r['experiment']=='uncensored_forecast' and
                 r['frame_interval_ms']==500 and r['resource_visibility']=='edge_only')
    assert math.isclose(no_load['predicted_mean_latency_ms'],no_load['realized_mean_latency_ms'],abs_tol=1e-6)


def test_generate_eight_vector_and_raster_figures_from_clean_inputs(tmp_path):
    pytest.importorskip('matplotlib')
    pytest.importorskip('numpy')
    main(['--output',str(tmp_path),'--bootstrap-repetitions','1100','--bootstrap-seed','123'])
    assert len(list(tmp_path.glob('*.png')))==8
    assert len(list(tmp_path.glob('*.pdf')))==8
    report=json.loads((tmp_path/'STUDY_REPORT.json').read_text())
    assert len(report['figures'])==8
    assert report['empirical_aggregate']['decisions']==3986
    assert report['paired_outcomes_0007']['introduced_joint_failure']==12
    assert report['research_status'].startswith('exploratory')
    for name in ('nine_scene_metrics.csv','queue_prediction_and_deadline_sweep.csv',
                 'synthetic_correlation_grid.csv','nine_scene_energy_assumption_sweep.csv',
                 'nine_scene_counterfactual_mixture.csv','candidate_estimate_catalog.csv'):
        assert (tmp_path/name).stat().st_size>100
