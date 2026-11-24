from __future__ import annotations
import csv
import json
from pathlib import Path
import pytest

from octopus_sim.config import ExperimentConfig
from scripts.run_frame_cache_study import main, run_study, ROOT, INTERVALS_MS


def test_load_sweep_fixed_outcome_endpoints_and_resource_reconciliation():
    cfg=ExperimentConfig.load(ROOT/'examples/energy/v2v4real_0007_energy.json')
    endpoints,stress=run_study(cfg)
    assert len(endpoints)==4
    assert len(stress)==4*len(INTERVALS_MS)
    for row in endpoints:
        assert row['decision_events']==442
        assert row['joint_error_count']==(4 if row['policy']=='local_only' else 16)
        assert row['agent_deadline_miss_rate']==0
    frame_local=next(r for r in endpoints if r['execution_granularity']=='frame' and r['policy']=='local_only')
    legacy_local=next(r for r in endpoints if r['execution_granularity']=='decision' and r['policy']=='local_only')
    assert frame_local['energy_incremental_j']==pytest.approx(legacy_local['energy_incremental_j']/2)
    assert frame_local['local_compute_jobs']==442
    assert legacy_local['local_compute_jobs']==884
    frame_fusion=next(r for r in endpoints if r['execution_granularity']=='frame' and r['policy']=='cooperative_late_fusion')
    legacy_fusion=next(r for r in endpoints if r['execution_granularity']=='decision' and r['policy']=='cooperative_late_fusion')
    assert frame_fusion['transmitted_bytes']*2==legacy_fusion['transmitted_bytes']
    assert frame_fusion['mean_delivered_latency_ms'] < legacy_fusion['mean_delivered_latency_ms']
    assert any(r['agent_deadline_miss_rate']>0 for r in stress)
    assert any(r['local_queue_wait_ms']>0 for r in stress)


def test_study_cli_writes_traceable_png_and_csv(tmp_path):
    main(['--output',str(tmp_path)])
    png=tmp_path/'01_latency_under_load_frame_vs_roi.png'
    assert png.read_bytes().startswith(b'\x89PNG')
    assert png.stat().st_size>20_000
    with (tmp_path/'modeled_latency_load_sweep.csv').open(newline='') as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==40
    prov=json.loads((tmp_path/'PROVENANCE.json').read_text())
    assert prov['recorded_decisions']==442
    assert prov['synchronized_frames']==221
