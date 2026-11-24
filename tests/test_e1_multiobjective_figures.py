"""Plausibility, identity and reproducibility checks for E1 multiobjective figures."""
from __future__ import annotations
from copy import deepcopy
import csv
import json
from pathlib import Path

import pytest

from octopus_sim.config import ExperimentConfig
from octopus_sim.simulator import run_simulation
from scripts.run_e1_multiobjective import (
    ROOT, bootstrap_ci, bootstrap_indices, evaluate_dataset,
    grouped_event_rates, main, sensitivity_energy,
)


def _v2v_config():
    return ExperimentConfig.load(ROOT / 'examples/energy/v2v4real_0007_energy.json')


def test_bundled_recorded_frame_pairs_are_verified():
    cfg=_v2v_config()
    out=run_simulation(cfg.with_policy('local_only'),detailed_events=False)
    joint,marginal=grouped_event_rates(out.scores,cfg,'v2v4real_0007')
    assert len(joint)==len(marginal)==221
    assert float(joint.sum())==4/2  # four binary decision events, two per frame
    assert out.summary['joint_all_error_count']==4
    assert joint.mean()==pytest.approx(4/442)
    assert marginal.mean()==pytest.approx(out.summary['marginal_error_rate_all_events'])


def test_temporal_bootstrap_preserves_consecutive_circular_blocks():
    idx=bootstrap_indices(31,120,5,17)
    assert idx.shape==(120,31)
    assert idx.min()>=0 and idx.max()<31
    for row in idx:
        for start in range(0,30,5):
            vals=row[start:start+5]
            assert all(int(y)==(int(x)+1)%31 for x,y in zip(vals,vals[1:]))
    with pytest.raises(ValueError): bootstrap_indices(31,20,0,1)


def test_bootstrap_reproducibility_and_bounds():
    import numpy as np
    one=bootstrap_indices(17,500,3,19)
    two=bootstrap_indices(17,500,3,19)
    assert np.array_equal(one,two)
    vals=np.array([0,1]*8+[0],dtype=float)
    lo,hi=bootstrap_ci(vals,one)
    assert 0<=lo<=vals.mean()<=hi<=1


def test_exact_reweighting_matches_actual_energy_simulator():
    cfg=_v2v_config()
    orig=run_simulation(cfg.with_policy('mixed_local_fusion',reuse_probability=.5),
                        detailed_events=False).summary
    for radio_factor,fusion_factor in ((.25,.5),(4,2),(1,1)):
        raw=deepcopy(cfg.data)
        raw['policy']={'name':'mixed_local_fusion','reuse_probability':.5}
        for profile in raw['energy']['device_profiles'].values():
            profile['radio']['tx_j_per_bit']*=radio_factor
            profile['radio']['rx_j_per_bit']*=radio_factor
        raw['energy']['model_incremental_w']['late_fusion']*=fusion_factor
        changed=run_simulation(ExperimentConfig.from_dict(raw,cfg.source_path),
                               detailed_events=False).summary
        assert sensitivity_energy(orig,radio_factor,fusion_factor)==pytest.approx(
            changed['energy_workload_incremental_j'],abs=1e-8)
        assert orig['joint_all_error_count']==changed['joint_all_error_count']
        assert orig['mean_delivered_latency_ms']==changed['mean_delivered_latency_ms']


def test_corrected_frame_level_energy_and_error_endpoints():
    cfg=_v2v_config()
    rows,grid=evaluate_dataset(cfg,'v2v4real_0007',200,12,17)
    assert len(rows)==5 and len(grid)==45
    baseline,shared=rows[0],rows[-1]
    assert baseline['joint_error_count']==4
    assert shared['joint_error_count']==16
    assert baseline['energy_incremental_total_j']==pytest.approx(35.36,abs=.0001)
    assert shared['energy_incremental_total_j']==pytest.approx(53.79302656,abs=.0001)
    assert baseline['latency_mean_ms']==pytest.approx(20.0)
    assert shared['latency_mean_ms']==pytest.approx(36.339,abs=1e-4)
    assert all(0<=r['joint_error_ci_low']<=r['joint_error_ci_high']<=1 for r in rows)
    assert all(r['energy_sensitivity_min_per_decision_j']-1e-10 <=r['energy_incremental_per_decision_j']
               <=r['energy_sensitivity_max_per_decision_j']+1e-10 for r in rows)
    assert all(r['deadline_miss_rate']==0 for r in rows)


def test_clean_plot_generation_and_output_counts(tmp_path: Path):
    main(['--output',str(tmp_path),'--synthetic-events','130','--bootstrap','100','--block-length','8'])
    pngs=sorted(tmp_path.glob('*.png'))
    assert len(pngs)==6
    assert all(f.read_bytes().startswith(b'\x89PNG\r\n\x1a\n') and f.stat().st_size>15_000 for f in pngs)
    with (tmp_path/'multiobjective_policy_metrics.csv').open() as f:
        rows=list(csv.DictReader(f))
    with (tmp_path/'energy_profile_sensitivity_grid.csv').open() as f:
        sensitivities=list(csv.DictReader(f))
    assert len(rows)==10
    assert len(sensitivities)==90
    assert sum(row['dataset']=='three_agent_synthetic' for row in rows)==5
    provenance=json.loads((tmp_path/'PROVENANCE.json').read_text())
    assert provenance['workloads']['v2v4real_0007']==442
    assert provenance['workloads']['three_agent_synthetic']==130
    assert provenance['energy_sensitivity']['not_a_probability_interval'] is True
