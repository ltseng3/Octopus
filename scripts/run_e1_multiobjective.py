#!/usr/bin/env python3
"""Reproduce joint/marginal error vs energy and latency for OCTOPUS-Sim E1.

Scientific scope:
* Three-agent: synthetic errors (independent scene-event draws), 3,000 episodes by default.
* V2V4Real: published model predictions in one recorded sequence, 221 frames,
  with two ground-truth-defined common binary ROI events per frame.
  One frame-level inference is executed and transmitted once, then projected
  locally into both ROI decisions without additional inference/communication.
* Fusion modes are simulated counterfactual allocations; energy, network, and
  latency profiles are illustrative configurations, NOT hardware observations.
* Error intervals: event bootstrap (synthetic) and 12-frame circular moving-block
  bootstrap (V2V4Real) to respect frame pairing and some temporal dependence.
  The latter does NOT provide scene-generalizable confidence intervals.
* Horizontal energy ranges: nonprobabilistic sensitivity to specified radio
  energy and edge-fusion power multipliers, NOT confidence intervals.

Run from any directory:
    python scripts/run_e1_multiobjective.py --output samples/multiobjective_e1
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import csv
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from octopus_sim.config import ExperimentConfig
from octopus_sim.simulator import run_simulation

POLICIES = (0.0, 0.25, 0.5, 0.75, 1.0)
RADIO_MULTIPLIERS = (0.25, 1.0, 4.0)
FUSION_POWER_MULTIPLIERS = (0.5, 1.0, 2.0)


def make_variants(conf: ExperimentConfig) -> list[tuple[float, ExperimentConfig]]:
    return [(0.0, conf.with_policy('local_only'))] + [
        (k, conf.with_policy('mixed_local_fusion', reuse_probability=k))
        for k in POLICIES[1:]
    ]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError('Empty CSV')
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def grouped_event_rates(scores: list[dict[str, Any]], conf: ExperimentConfig,
                        dataset: str):
    """Make one bootstrap cluster per episode or per *physical* recorded frame.

    Return group-level means of all-wrong and team-average marginal error.
    All V2V4Real frames must have exactly two decision events; we deliberately
    fail closed if this assumption or aligned ordering ceases to hold.
    """
    import numpy as np
    if dataset == 'three_agent_synthetic':
        groups = [[s] for s in scores]
    else:
        trace_path = conf.resolve_trace()
        records = [json.loads(x) for x in trace_path.read_text(encoding='utf-8').splitlines() if x.strip()]
        if [r['event_id'] for r in records] != [s['event_id'] for s in scores]:
            raise ValueError('Score/trace identities are not aligned')
        frame_groups: dict[str, list[dict[str, Any]]] = {}
        for r, s in zip(records, scores):
            if 'frame_id' not in r or 'arrival_slot' not in r:
                raise ValueError('V2V4Real explicit frame identifier or arrival_slot missing')
            frame_groups.setdefault(r['frame_id'], []).append(s)
        if any(len(g) != 2 for g in frame_groups.values()):
            raise ValueError('Expected exactly two paired ROI events per recorded frame')
        groups = list(frame_groups.values())
    joint = np.array([sum(float(s['all_wrong']) for s in group)/len(group)
                      for group in groups], dtype=float)
    marginal = np.array([sum(sum(float(v is True) for v in s['wrong'].values())/len(s['wrong'])
                             for s in group)/len(group) for group in groups], dtype=float)
    if not (0 <= joint.mean() <= 1 and 0 <= marginal.mean() <= 1):
        raise AssertionError('Error rates outside [0,1]')
    return joint, marginal


def bootstrap_indices(n: int, iterations: int, block_length: int, seed: int):
    """Deterministic circular moving-block resampling, with event bootstrap L=1."""
    import numpy as np
    if n < 1 or iterations < 50 or not 1 <= block_length <= n:
        raise ValueError('Invalid bootstrap design')
    rng = np.random.default_rng(seed)
    k = (n + block_length - 1)//block_length
    starts = rng.integers(0, n, size=(iterations,k))
    offsets = np.arange(block_length, dtype=int)
    return ((starts[..., None] + offsets) % n).reshape(iterations, -1)[:, :n]


def bootstrap_ci(group_values, resample_indices) -> tuple[float,float]:
    import numpy as np
    arr = np.asarray(group_values, dtype=float)
    boot_means = arr[resample_indices].mean(axis=1)
    lo, hi = np.quantile(boot_means, [.025,.975])
    return float(lo), float(hi)


def sensitivity_energy(summary: dict[str, Any],
                       radio_factor: float, fusion_factor: float) -> float:
    """Exact analytical reweighting for the *per-bit, linear power* E1 profiles.

    Does not alter task graph or scheduler. Verified against rerunning the
    original simulation under modified configurations in the test suite.
    """
    radio = summary['energy_radio_incremental_j']
    fusion = summary['energy_compute_by_model_j'].get('late_fusion', 0.0)
    other = summary['energy_compute_incremental_j'] - fusion
    return other + radio_factor * radio + fusion_factor * fusion


def evaluate_dataset(conf: ExperimentConfig, dataset: str,
                     iterations: int, block_length: int,
                     bootstrap_seed: int) -> tuple[list[dict], list[dict]]:
    conf_energy = conf.data['energy']
    if not conf_energy['enabled']:
        raise ValueError('Analysis requires enabled energy accounting')
    if any(d['radio']['mode'] != 'per_bit'
           for d in conf_energy['device_profiles'].values()):
        raise ValueError('Reweighting assumes per-bit radio mode')
    if conf_energy['model_energy_j']:
        raise ValueError('Reweighting assumes incremental model power, not fixed joules')
    rows, sensitivity_rows = [], []
    resamples = None
    for prob, variant in make_variants(conf):
        outcome = run_simulation(variant, detailed_events=False)
        summ = outcome.summary
        joint, marginal = grouped_event_rates(outcome.scores, conf, dataset)
        if resamples is None:
            resamples = bootstrap_indices(len(joint), iterations, block_length, bootstrap_seed)
        if not (abs(float(joint.mean())-summ['joint_all_error_rate']) < 1e-11
                and abs(float(marginal.mean())-summ['marginal_error_rate_all_events']) < 1e-11):
            raise AssertionError('Bootstrap scoring differs from simulator scoring')
        jlo,jhi = bootstrap_ci(joint, resamples)
        mlo,mhi = bootstrap_ci(marginal, resamples)
        if not summ['complete_event_rate'] == 1.0:
            raise ValueError('This figure assumes complete decisions; use an operational-failure metric otherwise')
        energy_nominal = summ['energy_workload_incremental_j']
        values = []
        for r in RADIO_MULTIPLIERS:
            for f in FUSION_POWER_MULTIPLIERS:
                reweighted = sensitivity_energy(summ, r, f)
                values.append(reweighted)
                sensitivity_rows.append({
                    'dataset':dataset,'fusion_target_probability':prob,
                    'radio_energy_multiplier':r,'fusion_power_multiplier':f,
                    'energy_incremental_total_j':reweighted,
                    'energy_incremental_per_decision_j':reweighted/summ['episodes'],
                    'calibration':'hypothetical; fixed graph; linear reweighting',
                })
        elo,ehi = min(values),max(values)
        if not elo-1e-8 <= energy_nominal <= ehi+1e-8:
            raise AssertionError('Nominal energy outside sensitivity envelope')
        rows.append({
            'dataset':dataset,'fusion_target_probability':prob,
            'decision_events':summ['episodes'],'agents':summ['agents'],
            'resampling_clusters':len(joint),'bootstrap_block_length':block_length,
            'joint_error_count':summ['joint_all_error_count'],
            'joint_error_rate':summ['joint_all_error_rate'],
            'joint_error_ci_low':jlo,'joint_error_ci_high':jhi,
            'marginal_error_rate':summ['marginal_error_rate_all_events'],
            'marginal_error_ci_low':mlo,'marginal_error_ci_high':mhi,
            'energy_incremental_total_j':energy_nominal,
            'energy_incremental_per_decision_j':energy_nominal/summ['episodes'],
            'energy_sensitivity_min_per_decision_j':elo/summ['episodes'],
            'energy_sensitivity_max_per_decision_j':ehi/summ['episodes'],
            'energy_baseline_total_j':summ['energy_baseline_j'],
            'energy_total_j':summ['energy_total_j'],
            'latency_mean_ms':summ['mean_delivered_latency_ms'],
            'latency_p95_ms':summ['p95_delivered_latency_ms'],
            'deadline_miss_rate':summ['agent_deadline_miss_rate'],
            'complete_event_rate':summ['complete_event_rate'],
            'actual_shared_event_rate':summ['events_with_shared_inference_rate'],
            'local_jobs':summ['local_inference_executions'],
            'edge_jobs':summ['edge_inference_executions'],
            'bytes_transmitted':summ['transmitted_bytes'],
        })
    return rows, sensitivity_rows



def temporal_block_sensitivity(conf: ExperimentConfig, iterations: int,
                               bootstrap_seed: int) -> list[dict]:
    """Check how within-sequence joint-risk intervals change with block size."""
    rows = []
    for prob, variant in make_variants(conf):
        outcome = run_simulation(variant, detailed_events=False)
        joint, _ = grouped_event_rates(outcome.scores, conf, 'v2v4real_0007')
        for length in (6, 12, 24):
            idx = bootstrap_indices(len(joint), iterations, length, bootstrap_seed)
            low, high = bootstrap_ci(joint, idx)
            rows.append({
                'fusion_target_probability': prob,
                'block_length_frames': length,
                'joint_error_rate': outcome.summary['joint_all_error_rate'],
                'bootstrap_ci_low': low,
                'bootstrap_ci_high': high,
                'scope': 'one recorded scene; within-sequence sensitivity only',
            })
    return rows

def plot_tradeoff(rows: list[dict], dataset: str, kind: str, output: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    is_energy = kind == 'energy'
    xkey = 'energy_incremental_per_decision_j' if is_energy else 'latency_mean_ms'
    x = np.array([row[xkey] for row in rows])
    j = np.array([row['joint_error_rate'] for row in rows])*100
    m = np.array([row['marginal_error_rate'] for row in rows])*100
    e_low = np.array([row['energy_sensitivity_min_per_decision_j'] for row in rows])
    e_high = np.array([row['energy_sensitivity_max_per_decision_j'] for row in rows])
    figure, axes = plt.subplots(1,2,figsize=(12.0,5.0), layout='constrained', sharex=True)
    specs = [
        ('Joint failure: all agents wrong', 'joint_error_ci_low', 'joint_error_ci_high', j, '#BB553E'),
        ('Marginal error: average agent', 'marginal_error_ci_low', 'marginal_error_ci_high', m, '#287D83'),
    ]
    for ax,(label,low_key,high_key,y,color) in zip(axes,specs):
        yl=np.array([r[low_key] for r in rows])*100
        yh=np.array([r[high_key] for r in rows])*100
        yerr=np.vstack((np.maximum(0,y-yl),np.maximum(0,yh-y)))
        if is_energy:
            xerr=np.vstack((np.maximum(0,x-e_low),np.maximum(0,e_high-x)))
            ax.errorbar(x,y,yerr=yerr,xerr=xerr,fmt='o-',markersize=5,
                        color=color,ecolor=color,elinewidth=1,alpha=.9,capsize=3,
                        label='95% error interval; energy sensitivity span')
        else:
            ax.errorbar(x,y,yerr=yerr,fmt='o-',markersize=5,color=color,
                        ecolor=color,capsize=3,label='95% error interval')
        for idx,(vx,vy) in enumerate(zip(x,y)):
            offset=(5,10) if idx%2==0 else (5,-14)
            ax.annotate(f'{int(POLICIES[idx]*100)}%',(vx,vy),
                        textcoords='offset points',xytext=offset,fontsize=8,color=color)
        ax.set(title=label,ylabel='Decision-level error (%)',
               xlabel=('Modeled incremental energy (J / decision)' if is_energy
                      else 'Mean delivered agent latency (ms)'))
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.20)
    title=('V2V4Real · 2 agents · 221 frames / 442 decisions' if dataset=='v2v4real_0007'
           else f'Synthetic · 3 agents · {rows[0]["decision_events"]:,} independent events')
    figure.suptitle(f'{title}\n' + ('Reliability vs energy (assumed power profiles)' if is_energy
                                    else 'Reliability vs latency (assumed service times)'),fontsize=13)
    footer=('Horizontal bars = parameter envelope, not statistical CI. ' if is_energy else '')
    footer += ('Vertical intervals = 12-frame temporal block bootstrap; ONE sequence only.'
               if dataset=='v2v4real_0007' else 'Vertical intervals = event bootstrap (synthetic).')
    figure.supxlabel(footer,fontsize=8,color='#5D6571')
    figure.savefig(output,dpi=185,bbox_inches='tight',facecolor='white')
    plt.close(figure)


def plot_sensitivity(all_tables: dict[str,list[dict]], output: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.6),layout='constrained')
    for ax,(name,rows) in zip(axes,all_tables.items()):
        xs=np.array([100*r['fusion_target_probability'] for r in rows])
        lo=np.array([r['energy_sensitivity_min_per_decision_j'] for r in rows])
        hi=np.array([r['energy_sensitivity_max_per_decision_j'] for r in rows])
        nominal=np.array([r['energy_incremental_per_decision_j'] for r in rows])
        ax.fill_between(xs,lo,hi,alpha=.22,color='#377F98',label='Assumed parameter envelope')
        ax.plot(xs,nominal,'o-',color='#286B85',label='Default hypothetical profile',lw=2.0)
        ax.set(xlabel='Cooperative fusion selection probability (%)',
               ylabel='Incremental energy (J / decision)',
               title=('V2V4Real — two agents' if name=='v2v4real_0007' else 'Synthetic — three agents'))
        ax.set_ylim(bottom=0)
        ax.set_xticks(xs)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('Sensitivity to assumed fusion power ×{0.5, 1, 2} and radio energy ×{0.25, 1, 4}',fontsize=12)
    fig.savefig(output,dpi=185,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def plot_latency_percentiles(all_tables: dict[str,list[dict]], output: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11.3,4.5),layout='constrained')
    for ax,(name,rows) in zip(axes,all_tables.items()):
        x=[r['fusion_target_probability']*100 for r in rows]
        ax.plot(x,[r['latency_mean_ms'] for r in rows],marker='o',label='Mean delivered latency')
        ax.plot(x,[r['latency_p95_ms'] for r in rows],marker='s',linestyle='--',label='P95 delivered latency')
        ax.set(xlabel='Cooperative fusion selection probability (%)',ylabel='Modeled latency per agent (ms)',
               title=('V2V4Real — two agents' if name=='v2v4real_0007' else 'Synthetic — three agents'))
        ax.set_xticks(x)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('Latency as policies use multi-stage cooperative fusion more frequently',fontsize=12)
    fig.savefig(output,dpi=185,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'samples/multiobjective_e1')
    parser.add_argument('--synthetic-events',type=int,default=3000,
                        help='Independent synthetic events (up from the original 80-event demonstration)')
    parser.add_argument('--bootstrap',type=int,default=3000)
    parser.add_argument('--block-length',type=int,default=12)
    parser.add_argument('--bootstrap-seed',type=int,default=711910)
    args=parser.parse_args(argv)
    if args.synthetic_events < 2 or args.bootstrap < 50 or not 1<=args.block_length<=221:
        parser.error('Require >=2 synthetic events, >=50 bootstrap iterations and 1<=block-length<=221')
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    synth=ExperimentConfig.load(ROOT/'examples/energy/three_agent_per_bit.json')
    raw=deepcopy(synth.data)
    raw['scenario']['episodes']=args.synthetic_events
    synth=ExperimentConfig.from_dict(raw,synth.source_path)
    empirical=ExperimentConfig.load(ROOT/'examples/energy/v2v4real_0007_energy.json')
    configs={'v2v4real_0007':empirical,'three_agent_synthetic':synth}
    tables={}
    sensitivity=[]
    for index,(name,conf) in enumerate(configs.items()):
        records,profile_grid=evaluate_dataset(conf,name,args.bootstrap,
                                              args.block_length if name=='v2v4real_0007' else 1,
                                              args.bootstrap_seed+index)
        tables[name]=records
        sensitivity.extend(profile_grid)
        plot_tradeoff(records,name,'energy',out/f'0{1+index}_{name}_error_vs_energy.png')
        plot_tradeoff(records,name,'latency',out/f'0{3+index}_{name}_error_vs_latency.png')
    plot_sensitivity(tables,out/'05_energy_profile_sensitivity_bands.png')
    plot_latency_percentiles(tables,out/'06_latency_mean_and_p95.png')
    all_rows=[x for rows in tables.values() for x in rows]
    _write_csv(out/'multiobjective_policy_metrics.csv',all_rows)
    _write_csv(out/'energy_profile_sensitivity_grid.csv',sensitivity)
    _write_csv(out/'v2v4real_block_bootstrap_sensitivity.csv',
               temporal_block_sensitivity(empirical,args.bootstrap,args.bootstrap_seed))
    report={
        'analysis':'E1 multiobjective figure upgrade; no inference-model training or hardware energy measurement',
        'source_version':'OCTOPUS-Sim 0.6.0 frame-level execution; V2V4Real published prediction replay',
        'execution_granularity':'frame: one allocation and one physical DAG per recorded frame',
        'v2v4real_source_trace':'examples/v2v4real_real_seq0007/trace.jsonl (221 frames / 442 events)',
        'v2v4real_scope':'ONE driving sequence; bootstrap cannot generalize to unseen scenes',
        'prediction_provenance':'DMSTrack V2V4Real 0007 source-verified trace; sharing policy is counterfactual',
        'workloads':{'v2v4real_0007':442,'three_agent_synthetic':args.synthetic_events},
        'error_intervals':{'type':'percentile bootstrap','confidence':0.95,
                           'iterations':args.bootstrap,'seed':args.bootstrap_seed,
                           'v2v4real':'circular moving-block bootstrap over whole recorded frames',
                           'block_length_frames':args.block_length,
                           'synthetic':'resample independent simulated events'},
        'energy_sensitivity':{'not_a_probability_interval':True,
                              'method':'exact analytical reweighting of the same executed DAG under additive incremental power and per-bit radio model',
                              'radio_energy_multipliers':RADIO_MULTIPLIERS,
                              'fusion_compute_power_multipliers':FUSION_POWER_MULTIPLIERS,
                              'calibration':'hypothetical only'},
        'latency':'simulated mean and p95 per agent delivered result; configured service/network rates; no real latency traces',
        'warning':'Observed error vs simulated resources; fixed model outputs; no causal claim about effect of latency on model correctness',
    }
    (out/'PROVENANCE.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    for name,records in tables.items():
        print(name,'events=',records[0]['decision_events'])
        for r in records:
            print(f"  k={r['fusion_target_probability']:.2f}  joint={r['joint_error_rate']*100:.3f}% "
                  f"marginal={r['marginal_error_rate']*100:.3f}%  "
                  f"E={r['energy_incremental_per_decision_j']:.4f} J/event "
                  f"latency={r['latency_mean_ms']:.3f} ms "
                  f"p95={r['latency_p95_ms']:.3f} ms")
    print('Wrote figures, CSV and provenance to',out)

if __name__=='__main__':
    main()
