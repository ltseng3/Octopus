#!/usr/bin/env python3
"""Reproduce integrated F0 vs local and cooperative end-to-end results.

No training, hardware measurements, or real-time perception happens here.
The reported V2V4Real source replay comprises ONE driving sequence (0007).
The synthetic three-agent benchmark is not an empirical calibration test.
"""
from __future__ import annotations

from copy import deepcopy
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from octopus_sim.config import ExperimentConfig
from octopus_sim.simulator import run_simulation

THRESHOLDS=(0.70,0.80,0.90,0.95,0.98)


def run_case(base: ExperimentConfig, policy: str, cutoff:float|None=None)->dict:
    data=deepcopy(base.data)
    data['policy']={'name':policy, 'reuse_probability':.5}
    if cutoff is not None:data['fusion_f0']['min_score_confidence']=cutoff
    config=ExperimentConfig.from_dict(data,base.source_path)
    result=run_simulation(config,detailed_events=False)
    summ=result.summary
    return {
        'scenario':base.data['scenario']['name'],
        'policy':policy,
        'min_surrogate_confidence':cutoff,
        'decisions':len(result.scores),
        'physical_frames':summ['physical_execution_batches'],
        'all_agent_wrong_rate':summ['joint_all_error_rate'],
        'marginal_error_rate':summ['marginal_error_rate_all_events'],
        'complete_event_rate':summ['complete_event_rate'],
        'deadline_miss_rate':summ['agent_deadline_miss_rate'],
        'mean_delivered_latency_ms':summ['mean_delivered_latency_ms'],
        'p95_delivered_latency_ms':summ['p95_delivered_latency_ms'],
        'incremental_energy_j':summ.get('energy_workload_incremental_j'),
        'total_j_fixed_idle_plus_work':summ.get('energy_total_j'),
        'bytes_transmitted':summ['transmitted_bytes'],
        'local_compute_executions':summ['local_inference_executions'],
        'edge_compute_executions':summ['edge_inference_executions'],
        'f0_shared_coverage':summ.get('f0_shared_roi_coverage'),
        'f0_actual_agent_overrides':summ.get('f0_realized_agent_overrides'),
        'f0_fallback_count':summ.get('f0_keep_local_roi_decisions'),
        'f0_masked_count':summ.get('f0_masked_fusion_decisions'),
    }


def plot(rows:list[dict],out:Path)->None:
    """One label per distinct plotted location; grouped thresholds cannot overlap."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    def scenario_name(name):
        return 'V2V4Real · two agents' if 'v2v4real' in name else 'Synthetic · three agents'

    def name(row):
        return ('F0 ' + str(round(100*row['min_surrogate_confidence']))+'%' if
                row['policy']=='selective_fusion_f0' else
                'Local' if row['policy']=='local_only' else 'Cooperative')

    def render(file:Path, keyfunc, xlabel, ylabel, title):
        fig,axes=plt.subplots(1,2,figsize=(12.2,4.85),layout='constrained')
        for ax,scenario in zip(axes,dict.fromkeys(r['scenario'] for r in rows)):
            grouped={}
            for row in (r for r in rows if r['scenario']==scenario):
                x,y=keyfunc(row)
                grouped.setdefault((round(x,9),round(y,9)),[]).append(name(row))
            for index,((x,y),labels) in enumerate(grouped.items()):
                is_f0=any(label.startswith('F0') for label in labels)
                if 'Local' in labels:
                    color='#365D8B'
                elif 'Cooperative' in labels:
                    color='#CB6731'
                else:
                    color='#2F847B'
                ax.scatter(x,y,marker='s' if is_f0 else 'o',s=54,color=color,zorder=4)
                if len(labels)>2 and all(label.startswith('F0') for label in labels):
                    annotation=f'F0 {labels[0].split()[-1]}–{labels[-1].split()[-1]}'
                elif len(labels)>2:
                    annotation=' / '.join((labels[0],f'F0 {labels[1].split()[-1]}–{labels[-1].split()[-1]}'))
                elif len(labels)==2 and all(label.startswith('F0') for label in labels):
                    annotation=f'F0 {labels[0].split()[-1]}–{labels[-1].split()[-1]}'
                else:
                    annotation=' / '.join(labels)
                ax.annotate(annotation,(x,y),xytext=(5,7 if index%2==0 else -14),
                            textcoords='offset points',fontsize=8,color=color)
            ax.set(title=scenario_name(scenario),xlabel=xlabel,ylabel=ylabel)
            ax.grid(alpha=.23)
            ax.margins(x=.25,y=.2)
        fig.suptitle(title,fontsize=13)
        fig.savefig(file,dpi=180,bbox_inches='tight')
        plt.close(fig)

    render(out,lambda row:(row['marginal_error_rate']*100,row['all_agent_wrong_rate']*100),
           'Mean marginal agent error (%)','All-agent error (%)',
           'F0 vs baselines: accuracy and joint-failure tradeoff')
    render(out.with_name('02_f0_energy_latency.png'),
           lambda row:(row['incremental_energy_j']/row['decisions'],row['mean_delivered_latency_ms']),
           'Hypothetical incremental energy (J / decision)',
           'Modeled mean delivered latency (ms)',
           'F0 vs baselines: different inference workloads and energy assumptions')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'samples/f0_integrated')
    parser.add_argument('--synthetic-events',type=int,default=3000)
    parser.add_argument('--no-plot',action='store_true')
    args=parser.parse_args(argv)
    if args.synthetic_events<20:
        parser.error('synthetic-events must be >=20')
    configs=[]
    for name in ('v2v4real_f0.json','three_agent_f0.json'):
        c=ExperimentConfig.load(ROOT/'examples/f0'/name)
        if 'three_agent' in name:
            raw=deepcopy(c.data)
            raw['scenario']['episodes']=args.synthetic_events
            c=ExperimentConfig.from_dict(raw,c.source_path)
        configs.append(c)
    rows=[]
    for c in configs:
        rows.extend((run_case(c,'local_only'),run_case(c,'cooperative_late_fusion')))
        rows.extend(run_case(c,'selective_fusion_f0',cutoff) for cutoff in THRESHOLDS)
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    with (out/'policy_comparison.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)
    (out/'PROVENANCE.json').write_text(json.dumps({
        'source':'OCTOPUS-Sim v0.7.1 + OCTOPUS-Fusion F0 v0.1.0 integration',
        'real_replay':'one V2V4Real recorded-prediction sequence 0007 only',
        'frame_identity':'one local detector and edge F0 decision per physical frame',
        'error_rates':'F0 inputs are static hypothetical assumptions; not learned/calibrated',
        'energy':'hypothetical energy profiles, full executed-work accounting',
        'latency':'fixed configured durations / event-driven shared resource queues',
        'fusion':'F0 derives a binary vote from local predictions; cooperative baseline replays a different recorded fusion output',
        'interpretation':'not a controlled same-model or trained 3D detection comparison',
        'synthetic_events':args.synthetic_events,
        'thresholds':THRESHOLDS,
    },indent=2)+'\n')
    if not args.no_plot:
        plot(rows,out/'01_f0_joint_vs_marginal.png')
    for row in rows:
        print(f"{row['scenario']:35} {row['policy']:24} {str(row['min_surrogate_confidence']):>5} "
              f"joint={row['all_agent_wrong_rate']:.4%} "
              f"marginal={row['marginal_error_rate']:.4%} "
              f"energy={row['incremental_energy_j']:.2f}J "
              f"meanlat={row['mean_delivered_latency_ms']:.3f}ms")
    print('Wrote',out)

if __name__=='__main__':main()
