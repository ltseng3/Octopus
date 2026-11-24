#!/usr/bin/env python3
"""Convert external three-real-CAV ROI exports and run OCTOPUS-Sim F0.

No raw sensor data are distributed here, and this command does not execute
CooperScene detectors or invent a third V2V4Real vehicle.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from octopus_sim.config import ExperimentConfig
from octopus_sim.datasets.three_vehicle_real import convert_to_trace, ThreeVehicleDataError
from octopus_sim.simulator import run_simulation
from octopus_fusion.fusion import unweighted_majority, weighted_vote
from octopus_fusion.contracts import Action, FusionDecision


class MajorityFuser:
    """Naive broadcast benchmark, intentionally ignores lineage and masks."""
    def decide(self, observation, reliability, *, risk_bounds=None):
        prediction=unweighted_majority(observation)
        local=tuple(sorted((e.agent_id,e.prediction) for e in observation.evidence))
        if prediction is None:
            return FusionDecision(observation.event_id,observation.frame_id,
                                  Action.KEEP_LOCAL,local,None,0.,.5,(),(),(),('tie',),'naive_majority')
        return FusionDecision(observation.event_id,observation.frame_id,Action.SHARE_FUSED,
                              tuple((a,prediction) for a,_ in local),prediction,0.,.5,
                              tuple(sorted({e.dependence_group for e in observation.evidence})),
                              tuple(sorted({e.dependence_group for e in observation.evidence if e.prediction==prediction})),
                              (),(),'naive_majority')


def perform_evaluation(scenario_file:Path, output:Path, *, allow_fixture=False,
                       thresholds=(.75,.85,.90,.95,.99)) -> list[dict]:
    source=scenario_file.parent
    meta=json.loads((source/'PROVENANCE.json').read_text())
    if meta['prediction_origin']=='synthetic_fixture' and not allow_fixture:
        raise ThreeVehicleDataError('Synthetic input fixture cannot be advertised as real-world measurement')
    if meta['physical_vehicles']!=['1','2','3']:
        raise ThreeVehicleDataError('Not three unique physical vehicles')
    from octopus_sim.datasets.three_vehicle_real import _sha
    if _sha(source/'trace.jsonl') != meta['output_trace_sha256']:
        raise ThreeVehicleDataError('Input trace hash does not match the conversion manifest')
    config=ExperimentConfig.load(scenario_file)
    labels={}
    truth={}
    for line in (source/'trace.jsonl').read_text().splitlines():
        r=json.loads(line)
        labels[r['event_id']]={a:r['candidates'][f'local:{a}'] for a in ('1','2','3')}
        truth[r['event_id']]=r['truth']
    output.mkdir(parents=True,exist_ok=True)
    runs=[('local_only',None)] + [('majority',None)] + [(f'f0_{t:.3f}',t) for t in thresholds]
    rows=[]
    for method,threshold in runs:
        d=deepcopy(config.data)
        d['policy']['name']='local_only' if method=='local_only' else 'selective_fusion_f0'
        if threshold is not None:
            d['fusion_f0']['min_score_confidence']=float(threshold)
        cfg=ExperimentConfig.from_dict(d,scenario_file)
        result=run_simulation(cfg,detailed_events=False,
                              **({'f0_fuser':MajorityFuser()} if method=='majority' else {}))
        n=len(result.scores)
        errors=[];corrected=0;harmed=0;changed=0
        for r in result.scores:
            original=labels[r['event_id']]
            for a,p in r['predictions'].items():
                if original[a]!=p:
                    changed+=1
                    if p==truth[r['event_id']]: corrected+=1
                    else:harmed+=1
            errors.append(r['error_count'])
        actions=result.fusion_decisions or []
        row={
            'dataset':'CooperScene' if meta['prediction_origin']=='real_detector_export' else 'SYNTHETIC_FIXTURE_NOT_REAL',
            'input_sha256':meta['input_sha256'],
            'method':method,'physical_frames':meta['physical_frames'], 'decision_events':n,
            'marginal_error':sum(errors)/(3*n),
            'r_ge_2':sum(e>=2 for e in errors)/n,
            'r_ge_3':sum(e==3 for e in errors)/n,
            'corrected_agent_decisions':corrected,'new_agent_errors':harmed,
            'changed_agent_decisions':changed,
            'accepted_roi_fraction':(sum(a['action']!='KEEP_LOCAL' for a in actions)/len(actions)) if actions else 0.,
            'simulated_incremental_energy_j':result.summary.get('energy_workload_incremental_j'),
            'mean_simulated_delivered_latency_ms':result.summary.get('mean_delivered_latency_ms'),
            'physical_local_jobs':result.summary['local_inference_executions'],
            'physical_edge_jobs':result.summary['edge_inference_executions'],
            'simulated_bytes':result.summary['transmitted_bytes'],
            'energy_provenance':'hypothetical_not_measured',
        }
        if row['physical_local_jobs'] != 3 * meta['physical_frames']:
            raise AssertionError('Unexpected physical-frame model execution count')
        if method != 'local_only' and row['physical_edge_jobs'] != meta['physical_frames']:
            raise AssertionError('F0 physical edge job count is not one per frame')
        rows.append(row)
    output_csv=output/'comparison.csv'
    with output_csv.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (output/'METHODS_AND_STATUS.json').write_text(json.dumps({
        'status':'real_per_vehicle_prediction_export' if meta['prediction_origin']=='real_detector_export' else 'synthetic_fixture_only',
        'source_input_hash':meta['input_sha256'],'physical_vehicles':3,
        'risk_definition':'R_m = event-level probability at least m of three agents wrong',
        'corrections':'changes in final agent labels relative to recorded local outputs',
        'paired_comparisons':True,'statistical_claim':'no confidence interval without multi-take hierarchy',
        'cost_origin':'simulated_hypothetical_profiles_no_hardware_calibration',
        'perception_metrics':'ROI presence, NOT 3D detection mAP',
        'not_performed':'real sensor model inference, physical communication or power measurement',
    },indent=2)+'\n')
    return rows


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare',help='Validate real exported source predictions and build F0 scenario')
    prep.add_argument('--source',type=Path,required=True)
    prep.add_argument('--manifest',type=Path,required=True)
    prep.add_argument('--output',type=Path,required=True)
    prep.add_argument('--allow-fixture',action='store_true')
    run=sub.add_parser('evaluate',help='Execute paired local/majority/F0 policies on same ROI events')
    run.add_argument('--scenario',type=Path,required=True)
    run.add_argument('--output',type=Path,required=True)
    run.add_argument('--allow-fixture',action='store_true')
    args=ap.parse_args()
    if args.command=='prepare':
        result=convert_to_trace(args.source,args.manifest,args.output,allow_fixture=args.allow_fixture)
        print(f'Prepared {result["physical_frames"]} physical frames and {result["decision_events"]} ROIs')
    else:
        rows=perform_evaluation(args.scenario,args.output,allow_fixture=args.allow_fixture)
        for r in rows:
            print(f'{r["method"]:16s} R2={r["r_ge_2"]:.4f} R3={r["r_ge_3"]:.4f} '
                  f'marginal={r["marginal_error"]:.4f} share={r["accepted_roi_fraction"]:.3f}')
        print('Outputs:',args.output)

if __name__=='__main__':main()
