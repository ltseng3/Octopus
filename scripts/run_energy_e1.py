#!/usr/bin/env python3
"""Regenerate E1 hypothetical-energy experiments, tables and figures.

No model inference is executed. The V2V4Real example uses the bundled
source-verified sequence-0007 prediction trace, and energy is entirely
hypothetical. Run from any working directory:

    python scripts/run_energy_e1.py --output samples/energy_e1
"""
from __future__ import annotations
import argparse
import csv
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from octopus_sim.config import ExperimentConfig
from octopus_sim.simulator import run_simulation


def variants(conf: ExperimentConfig) -> list[tuple[str, ExperimentConfig]]:
    return ([('local_only', conf.with_policy('local_only'))]
            + [(f'fusion_{int(k*100)}pct', conf.with_policy('mixed_local_fusion', reuse_probability=k))
               for k in (.25,.50,.75,1.0)])


def run_rows(conf: ExperimentConfig, dataset: str) -> list[dict]:
    rows=[]
    for label, variant in variants(conf):
        r=run_simulation(variant, detailed_events=False);s=r.summary
        row=dict(dataset=dataset, policy=label, fusion_share_probability=(0 if label=='local_only' else int(label.split('_')[1].replace('pct',''))/100),
                 decision_events=s['episodes'], joint_error_rate=s['joint_all_error_rate'],
                 marginal_error_rate=s['marginal_error_rate_all_events'],
                 complete_event_rate=s['complete_event_rate'],
                 energy_baseline_j=s['energy_baseline_j'],
                 energy_incremental_j=s['energy_workload_incremental_j'],
                 energy_total_j=s['energy_total_j'],
                 energy_vehicle_total_j=s['energy_vehicle_total_j'],
                 energy_edge_total_j=s['energy_edge_total_j'],
                 energy_compute_j=s['energy_compute_incremental_j'],
                 energy_radio_j=s['energy_radio_incremental_j'],
                 energy_post_window_j=s['energy_post_window_incremental_j'],
                 energy_per_successful_team_decision_j=s['energy_incremental_j_per_successful_team_decision'],
                 local_jobs=s['local_inference_executions'],edge_jobs=s['edge_inference_executions'],
                 bytes=s['transmitted_bytes'],
                 latency_ms=s['mean_delivered_latency_ms'])
        rows.append(row)
    return rows


def write_csv(out: Path, name: str, rows: list[dict]) -> None:
    with (out/name).open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'samples/energy_e1')
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    configs={
        'three_agent_synthetic': ExperimentConfig.load(ROOT/'examples/energy/three_agent_per_bit.json'),
        'v2v4real_0007': ExperimentConfig.load(ROOT/'examples/energy/v2v4real_0007_energy.json'),
    }
    tables={name:run_rows(conf,name) for name,conf in configs.items()}
    allrows=[x for rs in tables.values() for x in rs];write_csv(out,'policy_energy_comparison.csv',allrows)
    # Comparison of illustrative radio profiles: keep the recorded vision outcomes fixed.
    sensitivity=[]
    c=configs['v2v4real_0007']
    for factor in (.25,1.0,4.0):
        for edge_power_factor in (.5,1.0,2.0):
            raw=deepcopy(c.data)
            for profile in raw['energy']['device_profiles'].values():
                rad=profile['radio']
                rad['tx_j_per_bit']*=factor
                rad['rx_j_per_bit']*=factor
            raw['energy']['model_incremental_w']['late_fusion']*=edge_power_factor
            for policy, prob in [('local_only',0),('mixed_local_fusion',1)]:
                raw['policy']={'name':policy,'reuse_probability':prob}
                s=run_simulation(ExperimentConfig.from_dict(raw,c.source_path),detailed_events=False).summary
                sensitivity.append(dict(radio_multiplier=factor,fusion_power_multiplier=edge_power_factor,
                                        policy=policy,incremental_energy_j=s['energy_workload_incremental_j'],
                                        joint_error_rate=s['joint_all_error_rate']))
    write_csv(out,'energy_sensitivity.csv',sensitivity)
    (out/'PROVENANCE.json').write_text(json.dumps({
        'status':'hypothetical energy profiles; no hardware measurements',
        'vision_trace':'bundled V2V4Real-derived seq0007; see original release and source SHA metadata',
        'risk_metric':'all-agents wrong on common binary ROI decision event',
        'energy_boundary':'fixed decision horizon baseline + all incremental work including drain',
        'comparison':{'three_agent':'synthetic', 'v2v4real_0007':'actual publicly released model predictions; counterfactual policies'},
        'config_paths':{name:str(conf.source_path.relative_to(ROOT)) for name,conf in configs.items()},
    },indent=2)+'\n',encoding='utf-8')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # One figure per plot; default matplotlib palette and styles.
    fig,ax=plt.subplots(figsize=(7.8,4.8),layout='constrained')
    for name,rows in tables.items():
        ax.plot([r['energy_incremental_j'] for r in rows],
                [100*r['joint_error_rate'] for r in rows], marker='o',label=name)
        for r in rows:
            if r['fusion_share_probability'] in (0, 1):
                ax.annotate(r['policy'].replace('_',' '),
                            (r['energy_incremental_j'],100*r['joint_error_rate']),
                            xytext=(3,5), textcoords='offset points',fontsize=8)
    ax.set(xlabel='Incremental modeled workload energy (J)', ylabel='Both/all agents wrong (%)',
           title='Collective risk vs modeled energy\nHypothetical profiles; no measured hardware energy')
    ax.legend();ax.grid(alpha=.25)
    fig.savefig(out/'01_risk_vs_incremental_energy.png',dpi=180)
    fig.savefig(out/'01_risk_vs_incremental_energy.pdf');plt.close(fig)

    for name,rows in tables.items():
        fig,ax=plt.subplots(figsize=(7.5,4.8),layout='constrained')
        labels=[r['policy'].replace('fusion_','fusion ') for r in rows]
        xs=list(range(len(rows)))
        comp=[r['energy_compute_j'] for r in rows]
        comm=[r['energy_radio_j'] for r in rows]
        ax.bar(xs,comp,label='Compute (incremental)')
        ax.bar(xs,comm,bottom=comp,label='Radio TX + RX (incremental)')
        ax.set_xticks(xs,labels,rotation=20,ha='right')
        ax.set(xlabel='Edge allocation policy',ylabel='Modeled incremental energy (J)',
               title=f'{name}: component energy\nIllustrative profiles; idle baseline excluded')
        ax.legend();ax.grid(axis='y',alpha=.2)
        fig.savefig(out/f'02_{name}_energy_breakdown.png',dpi=180)
        fig.savefig(out/f'02_{name}_energy_breakdown.pdf');plt.close(fig)

    fig,ax=plt.subplots(figsize=(7.4,4.8),layout='constrained')
    for scale in (.5,1.0,2.0):
        subset=[r for r in sensitivity if r['fusion_power_multiplier']==scale and r['policy']=='mixed_local_fusion']
        ax.plot([r['radio_multiplier'] for r in subset],[r['incremental_energy_j'] for r in subset],
                marker='o',label=f'Fusion power × {scale:g}')
    base=[r for r in sensitivity if r['fusion_power_multiplier']==1.0 and r['policy']=='local_only']
    ax.plot([r['radio_multiplier'] for r in base],[r['incremental_energy_j'] for r in base],
            linestyle='--',label='Local only')
    ax.set_xscale('log',base=2)
    ax.set(xlabel='Radio energy parameter multiplier',ylabel='Modeled incremental energy (J)',
           title='V2V4Real: sensitivity to assumed radio and fusion power\nIllustrative energy parameters only')
    ax.legend();ax.grid(alpha=.25)
    fig.savefig(out/'03_v2v4real_energy_sensitivity.png',dpi=180)
    fig.savefig(out/'03_v2v4real_energy_sensitivity.pdf');plt.close(fig)

    print('Wrote',out)
    for name,rows in tables.items():
        for r in (rows[0],rows[-1]):
            print(f"{name:22} {r['policy']:15} joint={r['joint_error_rate']:.4%} "
                  f"workload energy={r['energy_incremental_j']:.3f} J "
                  f"total energy={r['energy_total_j']:.3f} J")

if __name__=='__main__': main()
