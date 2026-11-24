"""Reproduce V0.7 candidate-interface examples; no hardware inference required.

Outputs estimated (policy-visible) cost/risk beside realized (simulated) outcomes.
These are NOT calibration results or evidence of an optimized policy.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from octopus_sim.cli import save
from octopus_sim.config import ExperimentConfig
from octopus_sim.simulator import run_simulation

SOURCES = (
    'three_agent_contextual_local.json',
    'three_agent_contextual_fusion.json',
    'three_agent_reported_queues.json',
    'v2v4real_candidate_inventory.json',
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, default=ROOT/'samples'/'coordination_v07')
    args = ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    rows = []
    for filename in SOURCES:
        config = ExperimentConfig.load(ROOT/'examples'/'coordination'/filename)
        result = run_simulation(config, detailed_events=False)
        output = args.output/Path(filename).stem
        save(result,output,config.resolve_trace() if config.data['world']['backend']=='trace' else None)
        selected=[r for r in result.candidate_estimates if r['selected']]
        assert len(selected)==result.summary['physical_execution_batches']
        measured=result.summary
        rows.append({
            'scenario': measured['scenario'],
            'backend':config.data['world']['backend'],
            'selection': config.data['policy']['candidate_id'],
            'agent_count': len(config.data['scenario']['agents']),
            'decision_events': len(result.scores),
            'physical_execution_batches': measured['physical_execution_batches'],
            'mean_estimated_all_agent_error_probability':sum(x['risk']['all_agent_error_probability'] for x in selected)/len(selected),
            'observed_all_agent_error_rate':measured['joint_all_error_rate'],
            'mean_predicted_latency_ms_success_path':sum(x['cost']['mean_delivery_latency_ms'] for x in selected)/len(selected),
            'observed_mean_delivered_latency_ms':measured['mean_delivered_latency_ms'],
            'sum_predicted_incremental_energy_j_success_path':(
                sum(x['cost']['incremental_energy_j'] for x in selected)
                if all(x['cost']['incremental_energy_j'] is not None for x in selected) else None),
            'executed_incremental_energy_j':measured.get('energy_workload_incremental_j'),
            'simulated_edge_inference_jobs':measured['edge_inference_executions'],
            'simulated_local_inference_jobs':measured['local_inference_executions'],
            'packet_loss_model_probability':config.data['resources']['packet_loss'],
            'power_profile_label':config.data['energy']['calibration'],
        })
    with (args.output/'summary.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)
    (args.output/'protocol.json').write_text(json.dumps({
        'schema':'octopus_v07_coordination_diagnostics',
        'estimator_status':'synthetic_prior_not_empirically_fitted',
        'energy_status':'hypothetical_device_power_profiles',
        'resource_prediction':'deterministic successful path, committed reservations only',
        'risk_measurement':'event-level realized outcome; model-based estimators separate',
        'v2v4real_scope':'one validation scene 0007, 221 frames, 442 ROI labels',
    },indent=2)+'\n',encoding='utf-8')
    print(f'Wrote {args.output}/summary.csv for {len(rows)} example scenarios')


if __name__ == '__main__':
    main()
