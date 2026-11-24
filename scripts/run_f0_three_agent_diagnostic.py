#!/usr/bin/env python3
"""F0 three-agent diagnostic with paired synthetic source predictions.

Two physical frames/ROI questions are *not* separate physical detections:
one frame has 3 local inference outputs and 2 downstream binary ROI decisions.

Known marginal rates are supplied as synthetic model assumptions, not fitted
from realized errors. A common-mode error perturbation is separately varied.
Only actual local or F0 graphs run; dummy edge candidate keys are never used.

Example:
    python scripts/run_f0_three_agent_diagnostic.py --output samples/f0_three_agent
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from copy import deepcopy
import csv
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import numpy as np
from octopus_sim.config import ExperimentConfig
from octopus_sim.simulator import run_simulation
from octopus_fusion.contracts import Action,FusionDecision
from octopus_fusion.fusion import unweighted_majority, weighted_vote

AGENTS=('a','b','c')
ERROR_RATES=(.06,.12,.30)
SHOCKS=(0.0,.04)
METHODS=('local_only','naive_majority','naive_weighted',
         'f0_all_sources','f0_mask_unreliable_c','f0_overtrust_c')

class NaiveFusion:
    def __init__(self,method):
        self.method=method
    def decide(self,observation,reliability,*,risk_bounds=None):
        if risk_bounds is not None: raise ValueError('Naive baseline is not a risk-bound estimator')
        y=(unweighted_majority(observation) if self.method=='majority'
           else weighted_vote(observation,reliability))
        originals=tuple(sorted((e.agent_id,e.prediction) for e in observation.evidence))
        if y is None:
            return FusionDecision(observation.event_id,observation.frame_id,Action.KEEP_LOCAL,
                                  originals,None,0.,.5,(),(),(),('tie',),'naive_'+self.method)
        groups=tuple(sorted(set(e.dependence_group for e in observation.evidence)))
        return FusionDecision(observation.event_id,observation.frame_id,Action.SHARE_FUSED,
                              tuple((agent,y) for agent,_ in originals),y,0.,.5,
                              groups,tuple(e.dependence_group for e in observation.evidence if e.prediction==y),
                              (),(),'naive_'+self.method)


def make_traces(folder:Path,frames:int,rois:int,seed:int)->dict[float,Path]:
    if frames<20 or rois<1: raise ValueError('frames>=20, rois>=1 required')
    folder.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(seed)
    truth=rng.integers(0,2,size=(frames,rois),dtype=np.int8)
    # Exactly the same random variates for all common-shock parameter choices.
    independent=rng.random((frames,rois,3))
    common_active=rng.random((frames,rois))
    paths={}
    for s in SHOCKS:
        if s>min(ERROR_RATES):raise ValueError('Common shock exceeds a source marginal rate')
        p=folder/f'three_agents_common_shock_{int(s*1000):03d}.jsonl'
        # Bernoulli latent shock flips every agent; otherwise individual rates
        # are adjusted so Pr(E_i=1)=ERROR_RATES[i] exactly (in expectation).
        # This is a controlled correlation model, NOT the native OCTOPUS
        # synthetic world's local_correlation model.
        with p.open('w',encoding='utf-8') as f:
            for frame in range(frames):
                for roi in range(rois):
                    y=int(truth[frame,roi]); common=common_active[frame,roi]<s
                    preds={}
                    for i,a in enumerate(AGENTS):
                        conditional=(ERROR_RATES[i]-s)/(1-s)
                        err=common or (independent[frame,roi,i]<conditional)
                        preds[f'local:{a}']=y^int(err)
                    # Never execute independent edge operations against these
                    # stand-in values; they are trace-schema placeholders.
                    standins={f'edge:{a}':preds[f'local:{a}'] for a in AGENTS}
                    record={'event_id':f'frame_{frame:05d}:roi_{roi}',
                            'frame_id':f'frame_{frame:05d}','arrival_slot':frame,
                            'truth':y,'candidates':{**preds,**standins}}
                    f.write(json.dumps(record,separators=(',',':'),sort_keys=True)+'\n')
        paths[s]=p
    return paths


def config_for(trace:Path,method:str,frames:int,rois:int,seed:int):
    template=ExperimentConfig.load(ROOT/'examples/f0/three_agent_f0.json')
    d=deepcopy(template.data)
    d['scenario'].update(name='f0_three_agent_heterogeneous_error_diagnostic',seed=seed,
                         episodes=frames*rois, agents=list(AGENTS),
                         arrival_interval_ms=150,deadline_ms=140)
    d['world']={'backend':'trace','trace_file':str(trace.resolve()),'execution_granularity':'frame'}
    d['coordination']['enabled']=False
    d['policy']={'name':'local_only' if method=='local_only' else 'selective_fusion_f0',
                 'reuse_probability':.5}
    q={a:ERROR_RATES[i] for i,a in enumerate(AGENTS)}
    if method=='f0_overtrust_c':q['c']=.08
    d['fusion_f0'].update(error_by_agent=q, dependence_group_by_agent={a:a for a in AGENTS},
                         max_source_error=(.25 if method=='f0_mask_unreliable_c' else .49),
                         min_independent_groups=2,min_supporting_groups=2,
                         min_score_confidence=.90, max_conflict_fraction=.4,
                         dependence_discount=1.0,prior_positive=.5,max_age_ms=120.)
    d['resources']['packet_loss']=0.
    return ExperimentConfig.from_dict(d,template.source_path)


def run_method(trace:Path,method:str,frames:int,rois:int,seed:int):
    cfg=config_for(trace,method,frames,rois,seed)
    arg={}
    if method in ('naive_majority','naive_weighted'):
        arg['f0_fuser']=NaiveFusion(method.removeprefix('naive_'))
    result=run_simulation(cfg,detailed_events=False,**arg)
    s=result.summary
    if s['complete_event_rate']!=1 or s['agent_deadline_miss_rate']!=0:
        raise AssertionError('Study requires all results delivered on time')
    if s['local_inference_executions']!=3*frames or s['physical_execution_batches']!=frames:
        raise AssertionError('Frame-level execution identity violated')
    if method!='local_only' and s['edge_inference_executions']!=frames:
        raise AssertionError('F0 not executed once per frame')
    return result


def summarize(result,originals:dict[str,dict[str,int]],truth:dict[str,int],method:str,
              common:float,frames:int,rois:int)->dict:
    scores=result.scores
    per_event=[]
    corrected=harmed=changed=0
    for r in scores:
        eid=r['event_id']
        actual=r['predictions']; labels=originals[eid]
        for a in AGENTS:
            if actual[a]!=labels[a]:
                changed+=1
                if actual[a]==truth[eid]:corrected+=1
                else:harmed+=1
        per_event.append((int(r['error_count']>=2),int(r['error_count']==3),
                          r['error_count']/3))
    decisions=result.fusion_decisions or []
    shared=sum(d['action']!='KEEP_LOCAL' for d in decisions)
    masked=sum(bool(d['masked_agents']) for d in decisions)
    return {
        'shock_probability':common,'method':method,'agents':3,
        'physical_frames':frames,'decision_events':len(scores),
        'r_ge_2':sum(t[0] for t in per_event)/len(scores),
        'r_ge_3':sum(t[1] for t in per_event)/len(scores),
        'marginal_error':sum(t[2] for t in per_event)/len(scores),
        'accepted_roi_fraction':shared/len(scores) if decisions else 0.,
        'masked_roi_fraction':masked/len(scores) if decisions else 0.,
        'changed_agent_labels':changed,'corrected_agent_labels':corrected,
        'harmed_agent_labels':harmed,
        'energy_j_per_frame':result.summary['energy_workload_incremental_j']/frames,
        'mean_delivered_latency_ms':result.summary['mean_delivered_latency_ms'],
        'downlink_bytes':result.summary['transmitted_bytes']-3*frames*2048 if decisions else 0,
        'local_jobs':result.summary['local_inference_executions'],
        'edge_jobs':result.summary['edge_inference_executions'],
    }


def frame_metrics(result):
    """Each physical frame is the uncertainty unit; preserve both ROI errors."""
    frame_scores=defaultdict(list)
    for score in result.scores:
        frame_scores[score['event_id'].split(':roi_')[0]].append(score)
    return {frame:np.array([(sum(s['error_count']>=2 for s in ss)/len(ss)),
                            (sum(s['error_count']==3 for s in ss)/len(ss)),
                            (sum(s['error_count']/3 for s in ss)/len(ss))])
            for frame,ss in frame_scores.items()}


def boot_delta(a:np.ndarray,b:np.ndarray,indices:np.ndarray):
    delta=b-a
    means=delta[indices].mean(axis=1)
    return (float(delta.mean()),float(np.quantile(means,.025)),float(np.quantile(means,.975)))


def _write(path:Path,rows:list[dict]):
    if not rows:raise ValueError('empty rows')
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def plot(rows:list[dict],out:Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    colors={'local_only':'#64748B','naive_majority':'#E08A48','naive_weighted':'#8A68A8',
            'f0_all_sources':'#277D89','f0_mask_unreliable_c':'#2A8761',
            'f0_overtrust_c':'#C45155'}
    labels={'local_only':'Local only','naive_majority':'Majority vote',
            'naive_weighted':'Weighted (no guard)',
            'f0_all_sources':'F0 / three inputs',
            'f0_mask_unreliable_c':'F0 / mask C',
            'f0_overtrust_c':'F0 / overtrust C'}
    main=[x for x in rows if x['shock_probability']==.04]
    fig,axes=plt.subplots(1,2,figsize=(12.4,5),layout='constrained')
    for row in main:
        m=row['method'];c=colors[m]
        axes[0].scatter(100*row['marginal_error'],100*row['r_ge_3'],
                        s=110,color=c,marker='o' if m in ('local_only','naive_majority','naive_weighted') else 'D',label=labels[m],zorder=5)
        axes[1].scatter(100*row['marginal_error'],100*row['r_ge_2'],
                        s=110,color=c,marker='o' if m in ('local_only','naive_majority','naive_weighted') else 'D',label=labels[m],zorder=5)
    for ax in axes:
        ax.set(xlabel='Average individual-agent error (%)')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8,loc='best')
    axes[0].set(ylabel='All 3 agents wrong, R₃ (%)',title='Extreme shared failure')
    axes[1].set(ylabel='At least 2 of 3 wrong, R₂ (%)',title='Majority-level failure is unchanged')
    axes[1].set_ylim(0,10)
    axes[1].text(.05,.08, 'R₂ is invariant under three-agent\nmajority-preserving voting/fallback',
                 transform=axes[1].transAxes,fontsize=9,color='#485669')
    axes[0].annotate('F0 mask C = F0 all sources\n(same final decisions)',
        xy=(100*next(x for x in main if x['method']=='f0_mask_unreliable_c')['marginal_error'],
            100*next(x for x in main if x['method']=='f0_mask_unreliable_c')['r_ge_3']),
        xytext=(12,20),textcoords='offset points',fontsize=9,color='#256D58',
        arrowprops={'arrowstyle':'->','color':'#256D58'})
    fig.suptitle('Three agents · heterogeneous errors · common-shock probability 4%\n'
                 'Recorded predictions are synthetic; risk/energy are simulator diagnostics',fontsize=12)
    for ext in ('png','pdf'):
        fig.savefig(out/f'01_three_agent_error_tradeoff.{ext}',dpi=185,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12.4,4.9),layout='constrained')
    for idx,method in enumerate(METHODS):
        r=next(x for x in rows if x['method']==method and x['shock_probability']==.04)
        axes[0].bar(idx-.2,r['corrected_agent_labels'],width=.4,color=colors[method],
                    label='Corrected' if idx==0 else None)
        axes[0].bar(idx+.2,r['harmed_agent_labels'],width=.4,color='#4D5660',
                    label='Newly wrong' if idx==0 else None)
    axes[0].set_xticks(range(len(METHODS)),[labels[m] for m in METHODS],rotation=30,ha='right')
    axes[0].legend(fontsize=8)
    axes[0].set(ylabel='Agent-level decisions changed (count)',title='Recovered labels versus induced new errors')
    for r in main:
        ax=axes[1];ax.scatter(r['energy_j_per_frame'],100*r['marginal_error'],label=labels[r['method']],
                             s=90,color=colors[r['method']]);
    axes[1].set(xlabel='Modeled incremental energy (J per frame)',ylabel='Marginal agent error (%)',
                title='Correction comes after uploads are paid for')
    axes[1].grid(alpha=.2)
    fig.suptitle('Three-agent recovery and modeled resource cost · hypothetical power profiles',fontsize=12)
    for ext in ('png','pdf'):
        fig.savefig(out/f'02_three_agent_corrections_energy.{ext}',dpi=185,bbox_inches='tight')
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.6),layout='constrained')
    for method in METHODS:
        r=sorted((x for x in rows if x['method']==method),key=lambda x:x['shock_probability'])
        ax.plot([100*x['shock_probability'] for x in r],[100*x['r_ge_3'] for x in r],
                'o-',color=colors[method],label=labels[method])
    ax.set(xlabel='Synthetic common-error probability (%)',ylabel='All-three-agent error (%)',
           title='Correlated faults overwhelm even reliability-aware fusion')
    ax.grid(alpha=.2);ax.legend(ncol=2,fontsize=8)
    for ext in ('png','pdf'):
        fig.savefig(out/f'03_common_shock_sensitivity.{ext}',dpi=185,bbox_inches='tight')
    plt.close(fig)


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=ROOT/'samples/f0_three_agent')
    ap.add_argument('--frames',type=int,default=2000)
    ap.add_argument('--rois',type=int,default=2)
    ap.add_argument('--seed',type=int,default=60421)
    ap.add_argument('--bootstrap',type=int,default=2000)
    args=ap.parse_args(argv)
    if args.frames<20 or args.rois<1 or args.bootstrap<50:ap.error('invalid frames/rois/bootstrap')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    traces=make_traces(out/'synthetic_traces',args.frames,args.rois,args.seed)
    records=[];contrasts=[];frame_tables={}
    rng=np.random.default_rng(args.seed+9)
    bs=rng.integers(0,args.frames,size=(args.bootstrap,args.frames))
    for shock,trace in traces.items():
        original={};truth={}
        for line in trace.read_text().splitlines():
            r=json.loads(line);original[r['event_id']]={a:r['candidates'][f'local:{a}'] for a in AGENTS}
            truth[r['event_id']]=r['truth']
        for method in METHODS:
            run=run_method(trace,method,args.frames,args.rois,args.seed)
            records.append(summarize(run,original,truth,method,shock,args.frames,args.rois))
            frame_tables[(shock,method)]=frame_metrics(run)
        keys=sorted(frame_tables[(shock,'local_only')])
        source=np.array([frame_tables[(shock,'local_only')][k] for k in keys])
        for method in METHODS[1:]:
            other=np.array([frame_tables[(shock,method)][k] for k in keys])
            for index,label in enumerate(('r_ge_2','r_ge_3','marginal_error')):
                mean,lo,hi=boot_delta(source[:,index],other[:,index],bs)
                contrasts.append({'shock_probability':shock,'comparison':method+' - local_only',
                                  'metric':label,'difference_fraction':mean,
                                  'bootstrap_ci_low_fraction':lo,'bootstrap_ci_high_fraction':hi,
                                  'confidence':.95,'resampling_unit':'physical_frame'})
        print('shock=',shock,'completed all methods')
    _write(out/'three_agent_policy_comparison.csv',records)
    _write(out/'paired_frame_bootstrap.csv',contrasts)
    (out/'PROVENANCE.json').write_text(json.dumps({
        'method':'paired three-agent simulation with genuine physical DAG replay',
        'source':'synthetic trace generated by run_f0_three_agent_diagnostic.py',
        'seed':args.seed,'frames':args.frames,'rois_per_frame':args.rois,
        'marginal_source_error_rates':dict(zip(AGENTS,ERROR_RATES)),
        'common_shock_probabilities':SHOCKS,
        'shock_semantics':'with probability s all 3 wrong; otherwise each is independently wrong with residual (q_i-s)/(1-s)',
        'f0_estimates':'nominal true q_i parameters (known by construction), or deliberately incorrect C estimate',
        'models':'F0 is deterministic binary vote on given outputs; no V2V4Real or trained fusion models',
        'energy':'hypothetical device profiles, actual simulated DAG work counted',
        'bootstrap':{'iterations':args.bootstrap,'unit':'physical frame','paired':True,'confidence':.95},
        'trace_warning':'Edge prediction keys are placeholders not used by evaluated policies',
        'caveat':'One seed and synthetic rates; paired frame-bootstrap intervals are conditional on the toy generator'
    },indent=2)+'\n')
    plot(records,out)
    for r in records:
        print(f"{r['shock_probability']:.2f} {r['method']:<24} R2={r['r_ge_2']:.2%} R3={r['r_ge_3']:.2%} "
              f"marg={r['marginal_error']:.2%} corrected={r['corrected_agent_labels']} harmed={r['harmed_agent_labels']} "
              f"share={r['accepted_roi_fraction']:.2%} energy/frame={r['energy_j_per_frame']:.4f}")
    print('saved',out)

if __name__=='__main__':main()
