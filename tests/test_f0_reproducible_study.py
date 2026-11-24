"""Source-verified offline/online equality and reproducible analysis artifacts."""
from __future__ import annotations
from pathlib import Path
import csv
import json

import pytest

from octopus_sim.cli import save
from octopus_sim.config import ExperimentConfig
from octopus_sim.simulator import run_simulation
from octopus_fusion.fusion import SelectiveFuser
from octopus_fusion.contracts import FusionConfig
from octopus_fusion.v07_adapter import load_v07_local_outputs


def test_actual_v07_local_export_matches_online_f0_per_roi(root, tmp_path):
    cfg=ExperimentConfig.load(root/'examples/f0/v2v4real_f0.json')
    local=run_simulation(cfg.with_policy('local_only'),False)
    exported=tmp_path/'export'
    save(local,exported,cfg.resolve_trace())
    paired=load_v07_local_outputs(exported/'inferences.jsonl',
                                  exported/'scores.jsonl',
                                  cfg.data['fusion_f0']['error_by_agent'])
    fuser=SelectiveFuser(FusionConfig(min_score_confidence=.9))
    offline={case.observation.event_id:fuser.decide(case.observation,est)
             for case,est in paired}
    online=run_simulation(cfg,False)
    assert len(offline)==442==len(online.fusion_decisions)
    for decision in online.fusion_decisions:
        expected=offline[decision['event_id']]
        assert expected.action.value==decision['action']
        assert expected.per_agent_decisions==tuple(map(tuple,decision['per_agent_decisions']))
        assert expected.fused_prediction==decision['fused_prediction']
        assert expected.masked_agents==tuple(map(tuple,decision['masked_agents']))
        assert expected.fallback_reasons==tuple(decision['fallback_reasons'])
    assert online.summary['f0_realized_agent_overrides']==2*online.summary['f0_shared_roi_decisions']
    assert local.summary['joint_all_error_count']==online.summary['joint_all_error_count']==4


def test_synthetic_and_real_study_outputs_have_nonmisleading_provenance(root,tmp_path):
    from scripts.run_f0_integrated_study import main
    import sys
    args=['run_f0_integrated_study.py','--output',str(tmp_path),'--synthetic-events','100','--no-plot']
    previous=sys.argv
    try:
        sys.argv=args
        main()
    finally:
        sys.argv=previous
    records=list(csv.DictReader((tmp_path/'policy_comparison.csv').open()))
    assert len(records)==14
    assert sum(row['scenario']=='f0_v2v4real_seq0007' for row in records)==7
    for row in records:
        assert 0<=float(row['all_agent_wrong_rate'])<=1
        assert 0<=float(row['marginal_error_rate'])<=1
        assert float(row['incremental_energy_j'])>=0
    manifest=json.loads((tmp_path/'PROVENANCE.json').read_text())
    assert manifest['synthetic_events']==100
    assert manifest['real_replay'].startswith('one V2V4Real')
    assert 'hypothetical' in manifest['energy']


def test_plots_render_without_placeholder_metadata(root,tmp_path):
    pytest.importorskip('matplotlib')
    from scripts.run_f0_integrated_study import plot
    from scripts.run_f0_integrated_study import run_case
    cfg=ExperimentConfig.load(root/'examples/f0/v2v4real_f0.json')
    records=[run_case(cfg,'local_only'),run_case(cfg,'cooperative_late_fusion'),
             run_case(cfg,'selective_fusion_f0',.9)]
    out=tmp_path/'01.png'
    plot(records,out)
    assert out.stat().st_size>10_000
    assert (tmp_path/'02_f0_energy_latency.png').stat().st_size>10_000
