#!/usr/bin/env python3
"""Validate V2V4Real frame reuse and explore contention in a modeled edge system.

All detection outcomes are recorded; durations, radio parameters and offered
load are hypothetical. The decision-level mode is a historical *over-counting*
reference, NOT an alternative physical algorithm. This script does not train AI.

    python scripts/run_frame_cache_study.py --output samples/frame_cache_v06
"""
from __future__ import annotations
import argparse
import csv
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from octopus_sim.config import ExperimentConfig
from octopus_sim.simulator import run_simulation

INTERVALS_MS=(5,10,15,20,25,30,40,50,100,500)


def run_study(config: ExperimentConfig) -> tuple[list[dict], list[dict]]:
    """Two stable source outcome candidates, two execution granularities.

    Resource accounting is normalized by *physical frames* when comparing
    processing stages; decision-level error remains normalized by decision ROI.
    """
    endpoints=[]
    stress=[]
    for scope in ('decision','frame'):
        for policy in ('local_only','cooperative_late_fusion'):
            for interval in INTERVALS_MS:
                raw=deepcopy(config.data)
                raw['world']['execution_granularity']=scope
                raw['scenario']['arrival_interval_ms']=interval
                raw['policy']={'name':policy,'reuse_probability':.5}
                s=run_simulation(ExperimentConfig.from_dict(raw,config.source_path),False).summary
                row={
                    'execution_granularity':scope,
                    'policy':policy,
                    'frame_arrival_interval_ms':interval,
                    'physical_frames':221,
                    'decision_events':s['episodes'],
                    'joint_error_count':s['joint_all_error_count'],
                    'marginal_error_rate':s['marginal_error_rate_all_events'],
                    'local_compute_jobs':s['local_inference_executions'],
                    'edge_compute_jobs':s['edge_inference_executions'],
                    'transmitted_bytes':s['transmitted_bytes'],
                    'energy_incremental_j':s['energy_workload_incremental_j'],
                    'mean_delivered_latency_ms':s['mean_delivered_latency_ms'],
                    'p95_delivered_latency_ms':s['p95_delivered_latency_ms'],
                    'agent_deadline_miss_rate':s['agent_deadline_miss_rate'],
                    'complete_event_rate':s['complete_event_rate'],
                    'local_queue_wait_ms':s['local_queue_wait_ms'],
                    'edge_queue_wait_ms':s['edge_queue_wait_ms'],
                    'frame_projection_cache_hits':s['frame_projection_cache_hits'],
                }
                stress.append(row)
                if interval==500: endpoints.append(row)
    return endpoints,stress


def save_csv(path:Path,rows:list[dict]):
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def plot(stress:list[dict],output:Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12.0,4.8),layout='constrained')
    for scope in ('frame','decision'):
        for pol in ('local_only','cooperative_late_fusion'):
            sub=[x for x in stress if x['execution_granularity']==scope and x['policy']==pol]
            sub.sort(key=lambda x:x['frame_arrival_interval_ms'])
            key=f'{scope} / {"fusion" if pol=="cooperative_late_fusion" else "local"}'
            x=[r['frame_arrival_interval_ms'] for r in sub]
            y=[r['p95_delivered_latency_ms'] for r in sub]
            misses=[r['agent_deadline_miss_rate']*100 for r in sub]
            for ax,values in zip(axes,(y,misses)):
                ax.plot(x,values,marker='o',label=key)
    axes[0].set(ylabel='P95 delivered latency (ms)',title='Delivery latency of on-time results')
    axes[1].set(ylabel='Agent deadline misses (%)',title='Operational failure from overload')
    for ax in axes:
        ax.set_xscale('log')
        ax.set_xlabel('Frame arrival interval (ms); smaller = higher load')
        ax.set_xticks(INTERVALS_MS)
        ax.set_xticklabels([str(v) for v in INTERVALS_MS],rotation=40)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('V2V4Real-derived predictions · hypothetical resource-load sweep',fontsize=13)
    fig.savefig(output,dpi=180,bbox_inches='tight')
    plt.close(fig)


def main(argv:list[str]|None=None)->None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'samples/frame_cache_v06')
    args=parser.parse_args(argv)
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    config=ExperimentConfig.load(ROOT/'examples/energy/v2v4real_0007_energy.json')
    endpoints,stress=run_study(config)
    save_csv(out/'frame_granularity_endpoints.csv',endpoints)
    save_csv(out/'modeled_latency_load_sweep.csv',stress)
    plot(stress,out/'01_latency_under_load_frame_vs_roi.png')
    (out/'PROVENANCE.json').write_text(json.dumps({
        'source_trace':'examples/v2v4real_real_seq0007/trace.jsonl',
        'recorded_decisions':442,'synchronized_frames':221,
        'trace_origin':'public V2V4Real-derived recorded predictions; see root docs',
        'research_scope':'modeled timing and energy; recorded binary vision outcomes',
        'warning':'Decision-granularity execution is an intentional over-counting control, not a physical baseline',
        'sampling':'one source frame releases two region-level decisions; frames arrive at configured interval',
        'deadline_ms':config.data['scenario']['deadline_ms'],
        'plot_warning':'P95 delivered excludes missing results; see deadline-miss panel',
        'models':'per-frame detector and fusion are configured fixed-duration stages',
    },indent=2)+'\n')
    print('Output:',out)
    for row in endpoints: print(row['execution_granularity'],row['policy'],
        'jobs=',row['local_compute_jobs'],row['edge_compute_jobs'],
        'latency=',row['mean_delivered_latency_ms'],
        'energy=',row['energy_incremental_j'])

if __name__=='__main__':main()
