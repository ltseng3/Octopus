"""Plot analytical two-agent reuse curve versus simulated Monte Carlo estimates.

Run: python scripts/validate_poster.py --episodes 6000 --output samples
Requires optional matplotlib for the PNG; CSV/JSON work without it.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from octopus_sim.config import ExperimentConfig
from octopus_sim.simulator import run_simulation
from octopus_sim.metrics import wilson


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=6000, help='Per independent replicate')
    ap.add_argument('--replicates', type=int, default=1)
    ap.add_argument('--output', default='samples')
    args = ap.parse_args()
    if args.episodes < 1 or args.replicates < 1:
        ap.error('Episodes and replicates must be positive')
    out=Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    base=ExperimentConfig.load(ROOT/'examples/two_agents_poster.json')
    f=.06
    e=.10
    ks=[0.0,.10,.25,.50,.75,1.0]
    rows=[]
    for k in ks:
        successes = total = edge_jobs = 0
        complete_total = 0.0
        for rep in range(args.replicates):
            cfg=base.with_policy('randomized_reuse', reuse_probability=k)
            cfg.data['scenario']['episodes']=args.episodes
            cfg.data['scenario']['seed']=base.data['scenario']['seed']+rep*7919
            result=run_simulation(cfg, detailed_events=False)
            successes += result.summary['joint_all_error_count']
            total += result.summary['episodes']
            edge_jobs += result.summary['edge_inference_executions']
            complete_total += result.summary['complete_event_rate']*result.summary['episodes']
        lower, upper = wilson(successes, total)
        rows.append({
            'reuse_probability':k,
            'theoretical_joint_error':f*f+k*(f-f*f),
            'observed_joint_error':successes/total,
            'wilson_lower':lower,
            'wilson_upper':upper,
            'edge_jobs_per_event':edge_jobs/total,
            'complete_event_rate':complete_total/total,
        })
        print(f'k={k:.2f}: observed J={rows[-1]["observed_joint_error"]:.5f}, '
              f'theory={rows[-1]["theoretical_joint_error"]:.5f}')
    with (out/'poster_validation.csv').open('w', newline='') as fcsv:
        writer=csv.DictWriter(fcsv, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    report={'episodes_per_replicate':args.episodes,
            'replicates':args.replicates,
            'events_per_policy':args.episodes*args.replicates,
            'expected_threshold':(e*e-f*f)/(f-f*f),
            'mean_absolute_deviation':sum(abs(r['observed_joint_error']-r['theoretical_joint_error']) for r in rows)/len(rows),
            'simulator_model':'binary common event, independent separately generated edge outcomes',
            'rows':rows}
    (out/'poster_validation.json').write_text(json.dumps(report, indent=2)+'\n')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib missing: CSV and JSON saved; figure skipped')
        return
    fig, ax=plt.subplots(figsize=(8,5))
    x=[r['reuse_probability'] for r in rows]
    y=[r['observed_joint_error']*100 for r in rows]
    error=[[100*(r['observed_joint_error']-r['wilson_lower']) for r in rows],
           [100*(r['wilson_upper']-r['observed_joint_error']) for r in rows]]
    ax.errorbar(x,y,yerr=error,fmt='o',capsize=4,label='Simulation (95% Wilson interval)')
    ax.plot([0,1],[100*f*f,100*f],label='Poster analytical model')
    ax.axhline(100*e*e,ls='--',label='Local independent joint-error baseline')
    ax.axvline(report['expected_threshold'],ls=':',label=f'Risk reversal k≈{report["expected_threshold"]:.3f}')
    ax.set_xlabel('Probability of fully reusing one edge inference (k)')
    ax.set_ylabel('P(both agents wrong), %')
    ax.set_title('Inference reuse and joint failure: theory versus CPU simulation')
    ax.set_xlim(-.03,1.03)
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(out/'poster_validation.png', dpi=180)
    plt.close(fig)
    print('Saved', out/'poster_validation.png')


if __name__=='__main__':
    main()
