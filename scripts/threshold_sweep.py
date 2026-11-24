"""Reproducible F0 threshold-sensitivity study (synthetic only).

Produces a policy-coverage vs both-error-metrics CSV and optional PNG figure.
The correlated *frame* is the resampling unit for pointwise bootstrap CIs.
No CPU-energy, network, latency or real V2V4Real measurements are inferred.
"""
from __future__ import annotations
import argparse
import csv
from collections import defaultdict
from pathlib import Path
import json
import random
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from octopus_fusion import SelectiveFuser, FusionConfig, QualityProxyEstimator
from octopus_fusion.synthetic import generate, ToySettings


def metric_rows(paired, *, min_confidence: float, max_source_error: float):
    model=SelectiveFuser(FusionConfig(min_score_confidence=min_confidence,
                                      max_source_error=max_source_error))
    per_frame=defaultdict(list)
    for case, rates in paired:
        result=model.decide(case.observation,rates)
        predictions=dict(result.per_agent_decisions)
        wrong=sum(value!=case.truth for value in predictions.values())
        per_frame[case.observation.frame_id].append((wrong/len(predictions),
                                                     float(wrong==len(predictions)),
                                                     float(result.shared),
                                                     float(result.shared and result.fused_prediction != case.truth)))
    frames=list(per_frame.values())
    def summary(chosen):
        cases=[item for group in chosen for item in group]
        n=len(cases)
        marginal=sum(item[0] for item in cases)/n
        joint=sum(item[1] for item in cases)/n
        shared=sum(item[2] for item in cases)/n
        return marginal,joint,shared
    return summary(frames),frames,summary


def sweep(*,frames=1500,seed=4701,bootstrap=300):
    if bootstrap < 30:raise ValueError('bootstrap must be >=30')
    cases=generate(ToySettings(frames=frames,seed=seed))
    estimator=QualityProxyEstimator()
    paired=tuple((case,estimator.estimate(case.observation)) for case in cases)
    rng=random.Random(7621)
    indices=[tuple(rng.randrange(frames) for _ in range(frames)) for _ in range(bootstrap)]
    records=[]
    for cutoff in (.70,.80,.90,.95,.98):
        for mask_cutoff in (.18,.25,.32):
            m,clusters,summarize=metric_rows(paired,min_confidence=cutoff,
                                              max_source_error=mask_cutoff)
            per_boot=[summarize([clusters[i] for i in draw]) for draw in indices]
            def bounds(k):
                vals=sorted(x[k] for x in per_boot)
                return vals[int(.025*(bootstrap-1))],vals[int(.975*(bootstrap-1))]
            r={
                'min_surrogate_confidence':cutoff, 'max_error_for_unmasked_input':mask_cutoff,
                'frame_clusters':frames, 'decision_events':len(cases),
                'marginal_error':m[0], 'joint_error':m[1], 'shared_coverage':m[2],
                'marginal_ci_lo':bounds(0)[0], 'marginal_ci_hi':bounds(0)[1],
                'joint_ci_lo':bounds(1)[0], 'joint_ci_hi':bounds(1)[1],
                'coverage_ci_lo':bounds(2)[0], 'coverage_ci_hi':bounds(2)[1],
            }
            records.append(r)
    return records


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,default=ROOT/'results'/'threshold_sweep')
    ap.add_argument('--frames',type=int,default=1500)
    ap.add_argument('--bootstrap',type=int,default=300)
    ap.add_argument('--no-plot',action='store_true')
    args=ap.parse_args(argv)
    results=sweep(frames=args.frames,bootstrap=args.bootstrap)
    args.out.mkdir(parents=True,exist_ok=True)
    with (args.out/'threshold_sweep.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(results[0]))
        w.writeheader();w.writerows(results)
    if not args.no_plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(10.5,4.8),layout='constrained')
        for cutoff in (.18,.25,.32):
            sub=[r for r in results if r['max_error_for_unmasked_input']==cutoff]
            sub=sorted(sub,key=lambda x:x['shared_coverage'])
            x=[100*r['shared_coverage'] for r in sub]
            axes[0].plot(x,[100*r['joint_error'] for r in sub],marker='o',label=f'mask max q={cutoff}')
            axes[1].plot(x,[100*r['marginal_error'] for r in sub],marker='o',label=f'mask max q={cutoff}')
        for ax in axes:
            ax.set(xlabel='Shared fusion acceptance coverage (%)')
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
        axes[0].set(ylabel='All-agent wrong (%)',title='Collective error vs acceptance')
        axes[1].set(ylabel='Average agent wrong (%)',title='Marginal error vs acceptance')
        fig.suptitle('F0 synthetic threshold sweep — assumed proxies, not real sensor calibration')
        fig.savefig(args.out/'threshold_tradeoff.png',dpi=165)
        plt.close(fig)
    (args.out/'PROVENANCE.json').write_text(json.dumps({
        'kind':'synthetic_only','frames':args.frames,'roi_events':2*args.frames,
        'bootstraps':args.bootstrap,'bootstrap_unit':'whole frame, two ROIs per frame',
        'risk_confidence':'uncalibrated surrogate evidence score',
        'source_data':'generated in scripts/threshold_sweep.py; no real detector predictions',
        'timing_energy':'not evaluated in F0'},indent=2)+'\n')
    print('Wrote',args.out)

if __name__=='__main__':main()
