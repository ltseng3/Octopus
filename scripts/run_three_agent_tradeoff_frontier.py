#!/usr/bin/env python3
"""Paired three-agent F0 studies: minority rescue, calibration, and ROI/energy reuse.

Assumptions: recorded *synthetic* binary potential outcomes.  One independent
strong source A and two weaker sources B/C sharing a stochastic environmental
shock, but with DISTINCT physical result IDs.  Neither fuser sees truth.

Strict F0 is unchanged.  `DiagnosticExpertOverride` is an explicitly labelled
experimental POST-PROCESSING rule that relaxes one guard; not a claim of a new
validated risk estimator.  Energy comes from actual OCTOPUS-Sim DAG scheduling
for the selected policy/threshold experiments, using HYPOTHETICAL profiles.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import csv
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import numpy as np
from octopus_fusion.contracts import (Action, AgentEvidence, FusionConfig,
                                      FusionDecision, FusionObservation, ReliabilityEstimate)
from octopus_fusion.fusion import SelectiveFuser, unweighted_majority, weighted_vote
from octopus_sim.config import ExperimentConfig
from octopus_sim.simulator import run_simulation

AGENTS=('a','b','c')
P_TRUE=(.015,.26,.32)
DEFAULT_CONF=.90
SHOCKS=(.0,.06,.12,.18,.24)
ESTIMATED_A_GRID=(.003,.006,.010,.015,.020,.030,.045,.060,.100)
TRUE_A_GRID=(.015,.05,.10,.20,.30,.40,.45)
THRESHOLDS=(.80,.87,.90,.92,.94,.975,.985,.995,.999)
ROI_COUNTS=(1,2,4)
DEADLINES_MS=(21.,24.,26.,28.,30.,31.,31.5,32.,33.,35.,38.,45.)


class DiagnosticExpertOverride:
    """Experimental non-F0 extension: allow one trusted independent expert.

    Calls the unchanged F0 fuser first.  Only if its *sole* rejection reason is
    that the score winner has one supporter, and that supporter is the declared
    trusted expert A, may it share.  All other F0 checks remain mandatory.
    This is NOT a majority-safe F0 policy; it can propagate expert mistakes.
    """
    def __init__(self,base:SelectiveFuser, *, expert='a', expert_max_estimated_error=.05):
        self.base=base
        self.expert=expert
        self.max_q=expert_max_estimated_error

    def decide(self,observation,reliability,*,risk_bounds=None):
        decision=self.base.decide(observation,reliability,risk_bounds=risk_bounds)
        if decision.action!=Action.KEEP_LOCAL:
            return decision
        if tuple(decision.fallback_reasons)!=('insufficient_independent_support_for_winner',):
            return decision
        if len(decision.supporting_groups)!=1 or not decision.surrogate_score:
            return decision
        evidence={e.agent_id:e for e in observation.evidence}
        if (self.expert not in evidence or
                decision.supporting_groups[0]!=evidence[self.expert].dependence_group or
                dict(reliability.error_by_agent)[self.expert]>self.max_q or
                any(a==self.expert for a,_ in decision.masked_agents)):
            return decision
        label=int(decision.surrogate_score>0)
        if evidence[self.expert].prediction!=label:
            return decision
        return replace(decision,
                       action=(Action.MASK_AND_FUSE if decision.masked_agents else Action.SHARE_FUSED),
                       per_agent_decisions=tuple((a,label) for a,_ in decision.per_agent_decisions),
                       fused_prediction=label, fallback_reasons=(),
                       estimator_id=reliability.estimator_id+'+EXPERIMENTAL_single_expert')


class AlwaysVote:
    def __init__(self,kind:str): self.kind=kind
    def decide(self,observation,reliability,*,risk_bounds=None):
        if risk_bounds is not None:raise ValueError('Not a risk-bound estimator')
        label=(unweighted_majority(observation) if self.kind=='majority'
               else weighted_vote(observation,reliability))
        originals=tuple(sorted((e.agent_id,e.prediction) for e in observation.evidence))
        if label is None:
            return FusionDecision(observation.event_id,observation.frame_id,Action.KEEP_LOCAL,
                                 originals,None,0,.5,(),(),(),('ambiguous_vote',),'naive_'+self.kind)
        groups=tuple(sorted({e.dependence_group for e in observation.evidence}))
        return FusionDecision(observation.event_id,observation.frame_id,Action.SHARE_FUSED,
                              tuple((a,label) for a,_ in originals),label,0,.5,groups,
                              tuple(sorted({e.dependence_group for e in observation.evidence if e.prediction==label})),
                              (),(),'naive_'+self.kind)


def make_fuser(method:str,assumed_a:float=P_TRUE[0],threshold:float=DEFAULT_CONF):
    config=FusionConfig(max_source_error=.49,min_score_confidence=threshold,
                        min_independent_groups=2,min_supporting_groups=2,
                        max_conflict_fraction=.4,dependence_discount=1.,max_age_ms=120.)
    if method=='strict':return SelectiveFuser(config)
    if method=='expert':return DiagnosticExpertOverride(SelectiveFuser(config))
    if method=='majority':return AlwaysVote('majority')
    if method=='weighted':return AlwaysVote('weighted')
    raise ValueError(method)


def generate_arrays(frames:int,rois:int,seed:int,shock:float,expert_true_error=P_TRUE[0]):
    """Pair all shock/expert conditions using same independent latent uniforms."""
    if not (frames>0 and rois>0 and 0<=shock<=min(P_TRUE[1:])
            and 0<expert_true_error<.5): raise ValueError('invalid synthetic generator arguments')
    rng=np.random.default_rng(seed)
    truth=rng.integers(0,2,size=(frames,rois),dtype=np.int8)
    independent=rng.random((frames,rois,3))
    common=rng.random((frames,rois))<shock
    wrong=np.zeros((frames,rois,3),dtype=bool)
    wrong[:,:,0]=independent[:,:,0]<expert_true_error
    for i,q in ((1,P_TRUE[1]),(2,P_TRUE[2])):
        wrong[:,:,i]=common | (independent[:,:,i]<(q-shock)/(1-shock))
    pred=np.bitwise_xor(truth[:,:,None],wrong.astype(np.int8))
    return truth,pred


def make_trace(path:Path,truth:np.ndarray,pred:np.ndarray):
    path.parent.mkdir(parents=True,exist_ok=True)
    frames,rois=truth.shape
    with path.open('w',encoding='utf-8') as f:
        for frame in range(frames):
            for roi in range(rois):
                v={f'local:{a}':int(pred[frame,roi,i]) for i,a in enumerate(AGENTS)}
                # Edge:* values are placeholders to satisfy older trace schemas;
                # none of our methods ever executes independent-edge operations.
                v.update({f'edge:{a}':v[f'local:{a}'] for a in AGENTS})
                row={'event_id':f'frame_{frame:05d}:roi_{roi}',
                     'frame_id':f'frame_{frame:05d}','arrival_slot':frame,
                     'truth':int(truth[frame,roi]),'candidates':v}
                f.write(json.dumps(row,separators=(',',':'),sort_keys=True)+'\n')
    return path


def observations(pred:np.ndarray,assumed_a:float=P_TRUE[0]):
    """Public evidence only; no current truth or hidden error draws."""
    q=ReliabilityEstimate((('a',assumed_a),('b',P_TRUE[1]),('c',P_TRUE[2])),
                          'synthetic_assumed_q_not_hardware_calibrated')
    for frame in range(pred.shape[0]):
        for roi in range(pred.shape[1]):
            obs=FusionObservation(f'frame_{frame:05d}:roi_{roi}', f'frame_{frame:05d}',
                                  20.,tuple(AgentEvidence(
                                      a,int(pred[frame,roi,i]),f'result:frame{frame}:{a}',
                                      f'lineage:{a}',15.) for i,a in enumerate(AGENTS)), .5)
            yield frame,roi,obs,q


def evaluate_offline(truth,pred,method:str,*,assumed_a=P_TRUE[0],threshold=DEFAULT_CONF):
    if method=='local':
        new=pred.copy(); accepted=np.zeros(truth.shape,dtype=bool)
    else:
        fuser=make_fuser(method,assumed_a,threshold)
        new=np.empty_like(pred)
        accepted=np.zeros(truth.shape,dtype=bool)
        for f,r,obs,q in observations(pred,assumed_a):
            decision=fuser.decide(obs,q)
            new[f,r,:]=[dict(decision.per_agent_decisions)[a] for a in AGENTS]
            accepted[f,r]=decision.shared
    error=(new != truth[:,:,None])
    origerror=(pred != truth[:,:,None])
    # Return both scalar metrics and paired per-frame outcomes for cluster bootstrap.
    summary={
        'method':method,'assumed_expert_q':assumed_a,'confidence_threshold':threshold,
        'decision_events':int(truth.size),'physical_frames':int(truth.shape[0]),
        'marginal_error':float(error.mean()),
        'r_ge_2':float((error.sum(axis=-1)>=2).mean()),
        'r_all_3':float(error.all(axis=-1).mean()),
        'share_roi_fraction':float(accepted.mean()),
        'frame_with_downlink_fraction':float(accepted.any(axis=1).mean()),
        'corrected_labels':int(((origerror)&(~error)).sum()),
        'newly_wrong_labels':int(((~origerror)&error).sum()),
    }
    cluster=np.stack(((error.sum(axis=-1)>=2).mean(axis=1),
                      error.all(axis=-1).mean(axis=1),
                      error.mean(axis=(1,2))),axis=1)
    return summary,cluster,accepted,new


def config_for(trace:Path,method:str,frames:int,rois:int,seed:int,
               assumed_a=P_TRUE[0],threshold=DEFAULT_CONF):
    template=ExperimentConfig.load(ROOT/'examples/f0/three_agent_f0.json')
    data=deepcopy(template.data)
    data['scenario'].update(name='three_agent_expert_override_tradeoff',agents=list(AGENTS),
                            episodes=frames*rois,seed=seed,arrival_interval_ms=175,
                            deadline_ms=150)
    data['world']={'backend':'trace','trace_file':str(trace.resolve()),
                   'execution_granularity':'frame'}
    data['coordination']['enabled']=False
    data['policy']={'name':'local_only' if method=='local' else 'selective_fusion_f0','reuse_probability':.5}
    data['fusion_f0'].update(error_by_agent={'a':assumed_a,'b':P_TRUE[1],'c':P_TRUE[2]},
                             dependence_group_by_agent={a:a for a in AGENTS},
                             max_source_error=.49,max_age_ms=120.,
                             min_independent_groups=2,min_supporting_groups=2,
                             min_score_confidence=threshold,max_conflict_fraction=.4,
                             dependence_discount=1.,prior_positive=.5)
    data['resources']['packet_loss']=0
    return ExperimentConfig.from_dict(data,template.source_path)


def evaluate_online(trace:Path,truth,pred,method:str,seed:int,
                    *,assumed_a=P_TRUE[0],threshold=DEFAULT_CONF):
    frames,rois=truth.shape
    cfg=config_for(trace,method,frames,rois,seed,assumed_a,threshold)
    inject={} if method=='local' else {'f0_fuser':make_fuser(method,assumed_a,threshold)}
    result=run_simulation(cfg,detailed_events=False,**inject)
    if result.summary['complete_event_rate']!=1 or result.summary['agent_deadline_miss_rate']!=0:
        raise AssertionError('Unsafe study: missing/late decisions')
    if result.summary['local_inference_executions']!=3*frames or result.summary['physical_execution_batches']!=frames:
        raise AssertionError('Duplicated/omitted frame inference')
    if result.summary['edge_inference_executions']!=(0 if method=='local' else frames):
        raise AssertionError('Incorrect edge work')
    scores=np.zeros_like(pred)
    for score in result.scores:
        f,roi=map(int,(score['event_id'].split(':roi_')[0].split('_')[1],score['event_id'].split(':roi_')[1]))
        scores[f,roi,:]=[score['predictions'][a] for a in AGENTS]
    offline,_,acc,expected=evaluate_offline(truth,pred,method,assumed_a=assumed_a,threshold=threshold)
    if not np.array_equal(scores,expected):
        raise AssertionError(f'Online results differ from offline F0 for {method}')
    if method!='local':
        if len(result.fusion_decisions)!=int(truth.size):
            raise AssertionError('F0 must evaluate every ROI')
        observed={x['event_id']:x['action']!='KEEP_LOCAL' for x in result.fusion_decisions}
        if not all(observed[f'frame_{f:05d}:roi_{r}']==bool(acc[f,r]) for f in range(frames) for r in range(rois)):
            raise AssertionError('Online share actions differ')
    return {
        **offline,
        'energy_per_frame_j':result.summary['energy_workload_incremental_j']/frames,
        'mean_delivered_latency_ms':result.summary['mean_delivered_latency_ms'],
        'network_bytes':result.summary['transmitted_bytes'],
        'edge_compute_jobs':result.summary['edge_inference_executions'],
        'local_compute_jobs':result.summary['local_inference_executions'],
        'actual_frame_downlink_fraction':(
            sum(any(d['applied_agents'] for d in result.fusion_decisions[f*rois:(f+1)*rois])
                for f in range(frames))/frames if method!='local' else 0.0),
    }


def evaluate_deadline(trace,truth,pred,method:str,deadline_ms:float,seed:int):
    """Unlike ideal offline vote, reflects actual deadlines and partial delivery."""
    frames,rois=truth.shape
    cfg=config_for(trace,method,frames,rois,seed,threshold=DEFAULT_CONF)
    params=deepcopy(cfg.data)
    params['scenario']['deadline_ms']=deadline_ms
    cfg=ExperimentConfig.from_dict(params,cfg.source_path)
    addon={} if method=='local' else {'f0_fuser':make_fuser(method)}
    out=run_simulation(cfg,detailed_events=False,**addon)
    if out.summary['complete_event_rate']!=1:
        raise AssertionError('Deadline study requires timely LOCAL fallback for every agent')
    accepted=(sum(d['action']!='KEEP_LOCAL' for d in (out.fusion_decisions or [])))
    delivered=(sum(len(d['applied_agents']) for d in (out.fusion_decisions or [])))
    return {
        'method':method,'deadline_ms':deadline_ms,'frames':frames,'roi_events':int(truth.size),
        'marginal_error':out.summary['marginal_error_rate_all_events'],
        'r_ge_2':sum(score['error_count']>=2 for score in out.scores)/len(out.scores),
        'r_all_3':out.summary['joint_all_error_rate'],
        'accepted_roi_fraction':accepted/truth.size,
        'applied_agent_update_fraction':delivered/(3*truth.size),
        'mean_delivered_latency_ms':out.summary['mean_delivered_latency_ms'],
        'energy_per_frame_j':out.summary['energy_workload_incremental_j']/frames,
        'agent_deadline_miss_rate':out.summary['agent_deadline_miss_rate'],
        'downlink_bytes':out.summary['transmitted_bytes']-3*frames*2048 if method!='local' else 0,
    }


def write_csv(path:Path,rows:list[dict]):
    if not rows:raise ValueError('empty table')
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def bootstrap_pairs(original:np.ndarray,alternate:np.ndarray,seed:int,reps:int=2000):
    if original.shape!=alternate.shape or original.ndim!=2:raise ValueError('Unpaired bootstrap')
    rng=np.random.default_rng(seed)
    n=original.shape[0]
    offsets=alternate-original
    samples=[]
    # Avoid allocating a huge [reps,n,metrics] cube.
    for _ in range(reps):
        idx=rng.integers(0,n,size=n)
        samples.append(offsets[idx].mean(axis=0))
    vals=np.asarray(samples)
    return offsets.mean(axis=0),np.quantile(vals,.025,axis=0),np.quantile(vals,.975,axis=0)


def chart(rows_a,rows_b,rows_c,rows_d,intervals,out:Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    def save(fig,stem):
        for ext in ('png','pdf','svg'):
            fig.savefig(out/f'{stem}.{ext}',dpi=300,bbox_inches='tight',facecolor='white')
        plt.close(fig)
    gray='#65727E'; orange='#B96F3F'; blue='#207C8A'; green='#3B8B68'; red='#BB5353'
    # Figure 1: tradeoff of majority recovery and shared-failure amplification.
    fig,axes=plt.subplots(1,2,figsize=(12.4,4.7),layout='constrained')
    for method,label,c in [('local','Local','%s'%gray),('majority','Majority broadcast',orange),
                           ('strict','F0: two supporters',blue),('expert','Experimental expert override',red)]:
        items=sorted([r for r in rows_a if r['method']==method],key=lambda r:r['shock_probability'])
        x=[100*r['shock_probability'] for r in items]
        for ax,metric in ((axes[0],'r_ge_2'),(axes[1],'r_all_3')):
            y=np.array([100*r[metric] for r in items])
            lo=np.array([100*intervals[(r['shock_probability'],method,metric)]['low'] for r in items])
            hi=np.array([100*intervals[(r['shock_probability'],method,metric)]['high'] for r in items])
            ax.errorbar(x,y,yerr=np.vstack([np.maximum(0,y-lo),np.maximum(0,hi-y)]),
                        fmt='o-',label=label,color=c,capsize=2,elinewidth=1,alpha=.9)
    for ax in axes:
        ax.grid(alpha=.20);ax.set_xlabel('B-C common-shock probability (%)')
    axes[0].set_ylabel('At least two of three wrong, R2 (%)')
    axes[0].set_title('Recovering an incorrect majority')
    axes[1].set_ylabel('All three wrong, R3 (%)')
    axes[1].set_title('Cost: propagating expert errors')
    axes[0].legend(fontsize=8,loc='upper left')
    fig.suptitle('Three agents: minority-expert override trades majority recovery for extreme shared risk\n'
                 'A true error 1.5%, B 26%, C 32%; fixed marginal rates across shock settings',fontsize=12)
    save(fig,'01_majority_recovery_vs_shared_failure')

    # Figure 2: calibration sensitivity map (same latent noise, across true/assumed q_A).
    fig,axes=plt.subplots(1,2,figsize=(12.3,4.8),layout='constrained')
    true=sorted(set(r['true_expert_q'] for r in rows_b))
    assumed=sorted(set(r['assumed_expert_q'] for r in rows_b))
    for ax,metric,title in zip(axes,['r_ge_2_delta_pp','r_all_3_delta_pp'],
                               ['Change in R2 vs local (pp)','Change in R3 vs local (pp)']):
        vals=np.array([[next(z[metric] for z in rows_b if z['true_expert_q']==t and z['assumed_expert_q']==q)
                        for q in assumed] for t in true])
        vmax=max(1.0,float(np.abs(vals).max()))
        im=ax.imshow(vals,cmap='RdBu_r',vmin=-vmax,vmax=vmax,aspect='auto',origin='lower')
        ax.set_xticks(range(len(assumed)),[f'{q:.3f}' for q in assumed],rotation=45,ha='right')
        ax.set_yticks(range(len(true)),[f'{v:.3f}' for v in true])
        ax.set_xlabel('Edge-assumed error of A (q-hat)')
        ax.set_ylabel('Actual synthetic error rate of A')
        ax.set_title(title)
        fig.colorbar(im,ax=ax,shrink=.82,label='Difference in percentage points')
    fig.suptitle('Same observed B/C predictions and 18% common-shock process\n'
                 'Blue improves risk; red increases risk; estimates are not learned',fontsize=12)
    save(fig,'02_expert_miscalibration_failure_map')

    # Figure 3: frame-level caching and the "sunk energy" phenomenon.
    fig,axes=plt.subplots(1,3,figsize=(14.5,4.45),layout='constrained')
    baseline=min(r['energy_per_frame_j'] for r in rows_c)
    for n,c in zip(ROI_COUNTS,[blue,orange,green]):
        items=sorted([r for r in rows_c if r['roi_per_frame']==n],key=lambda r:r['threshold'])
        xx=[100*r['share_roi_fraction'] for r in items]
        axes[0].plot(xx,[100*r['frame_with_downlink_fraction'] for r in items],'-o',label=f'{n} ROI/frame',color=c)
        axes[1].plot(xx,[1000*(r['energy_per_frame_j']-baseline) for r in items],'-o',color=c)
        axes[2].plot(xx,[r['energy_per_frame_j']/n for r in items],'-o',color=c)
    axes[0].set_ylabel('Frames with >=1 downlink (%)')
    axes[0].set_title('One accepted ROI triggers frame result')
    axes[1].set_ylabel('Additional downlink energy (mJ/frame)')
    axes[1].set_title('Post-fusion savings are tiny')
    axes[2].set_ylabel('Modeled energy (J/ROI)')
    axes[2].set_title('Physical work amortized over ROIs')
    for ax in axes:
        ax.set_xlabel('Accepted ROI fraction (%)')
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Strict F0: confidence threshold x number of downstream ROI queries\n'
                 f'Sunk F0 cost = {baseline:.5f} J/frame; hypothetical power and fixed result payload',fontsize=12)
    save(fig,'03_roi_reuse_downlink_energy')

    # Figure 4: post-inference fallback cannot defeat a tight deadline.
    fig,axes=plt.subplots(1,3,figsize=(14.6,4.5),layout='constrained')
    for method,label,c in [('local','Local',gray),('strict','Strict F0',blue),
                           ('expert','Experimental expert override',red)]:
        items=sorted([r for r in rows_d if r['method']==method],key=lambda r:r['deadline_ms'])
        times=[r['deadline_ms'] for r in items]
        axes[0].plot(times,[100*r['marginal_error'] for r in items],'-o',color=c,label=label)
        axes[1].plot(times,[100*r['applied_agent_update_fraction'] for r in items],'-o',color=c)
        axes[2].plot(times,[r['energy_per_frame_j'] for r in items],'-o',color=c)
    axes[0].set(ylabel='Marginal error (%)',title='Correction requires timely delivery')
    axes[1].set(ylabel='Applied agent updates (%)',title='Post-fusion decisions become usable')
    axes[2].set(ylabel='Modeled incremental energy (J/frame)',title='Energy is spent even when too late')
    for ax in axes:
        ax.grid(alpha=.20)
        ax.set_xlabel('Decision deadline from frame arrival (ms)')
    axes[0].legend(fontsize=8)
    fig.suptitle('Three agents; 2 ROI/frame; fixed deterministic service times; zero packet loss\n'
                 'Discontinuities are a simulator scheduling effect, NOT a real hardware latency bound',fontsize=12)
    save(fig,'04_deadline_cliff_energy_without_recovery')


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=ROOT/'samples/f0_three_agent_extended')
    ap.add_argument('--frames',type=int,default=1600)
    ap.add_argument('--rois',type=int,default=2)
    ap.add_argument('--seed',type=int,default=77109)
    ap.add_argument('--bootstrap',type=int,default=2000)
    ap.add_argument('--online-frames',type=int,default=600,
                    help='Number of first frames used for full DAG energy/latency sweep')
    args=ap.parse_args(argv)
    if args.frames<100 or args.rois<2 or args.bootstrap<100 or not 100<=args.online_frames<=args.frames:
        ap.error('Require >=100 frames, >=2 ROI, >=100 bootstrap, online frames in [100, frames]')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    traces=out/'synthetic_traces';traces.mkdir(exist_ok=True)

    # Study A: common-mode errors with marginal rates held fixed.
    a=[];frame_cache={}
    for shock in SHOCKS:
        truth,pred=generate_arrays(args.frames,args.rois,args.seed,shock)
        trace=make_trace(traces/f'pairshock_{int(shock*100):02d}.jsonl',truth,pred)
        for method in ('local','majority','strict','expert'):
            row,clusters,_,_=evaluate_offline(truth,pred,method)
            row.update(shock_probability=shock,actual_expert_error=P_TRUE[0],
                       model='three_agent_BC_common_environment',estimate_status='configured_synthetic_assumption')
            a.append(row);frame_cache[(shock,method)]=clusters
    write_csv(out/'shock_and_policy_grid.csv',a)
    # Pointwise whole-frame bootstrap intervals for each plotted absolute risk;
    # paired delta intervals below answer the more important policy contrast.
    frame_indices=np.random.default_rng(args.seed+71).integers(
        0,args.frames,size=(args.bootstrap,args.frames))
    interval_data={};pointwise=[]
    for (shock_key,method),cluster in sorted(frame_cache.items()):
        draws=cluster[frame_indices].mean(axis=1)
        q=np.quantile(draws,[.025,.975],axis=0)
        for i,metric in enumerate(('r_ge_2','r_all_3','marginal_error')):
            interval_data[(shock_key,method,metric)]={'low':float(q[0,i]),'high':float(q[1,i])}
            pointwise.append({'shock_probability':shock_key,'method':method,
                              'metric':metric,'mean':float(cluster[:,i].mean()),
                              'ci_low':float(q[0,i]),'ci_high':float(q[1,i]),
                              'cluster_unit':'physical_frame','bootstrap_replicates':args.bootstrap})
    write_csv(out/'pointwise_frame_bootstrap.csv',pointwise)
    shock=.18
    original=frame_cache[(shock,'local')]
    boot=[]
    for method in ('majority','strict','expert'):
        d,lo,hi=bootstrap_pairs(original,frame_cache[(shock,method)],args.seed+123,args.bootstrap)
        for i,metric in enumerate(('r_ge_2','r_all_3','marginal_error')):
            boot.append({'shock_probability':shock,'comparison':method+' - local',
                         'metric':metric,'delta_percentage_points':d[i]*100,
                         'ci_low_pp':lo[i]*100,'ci_high_pp':hi[i]*100,
                         'unit':'physical_frame','replicates':args.bootstrap,'confidence':.95})
    write_csv(out/'paired_frame_bootstrap.csv',boot)

    # Study B: sweep true source quality separately from the edge's assumption.
    b=[]
    for p_true in TRUE_A_GRID:
        tr,pred=generate_arrays(args.frames,args.rois,args.seed,.18,expert_true_error=p_true)
        local,_,_,_=evaluate_offline(tr,pred,'local')
        for q_hat in ESTIMATED_A_GRID:
            row,_,_,_=evaluate_offline(tr,pred,'expert',assumed_a=q_hat)
            b.append({'true_expert_q':p_true,'assumed_expert_q':q_hat,
                      'r_ge_2_delta_pp':(row['r_ge_2']-local['r_ge_2'])*100,
                      'r_all_3_delta_pp':(row['r_all_3']-local['r_all_3'])*100,
                      'marginal_delta_pp':(row['marginal_error']-local['marginal_error'])*100,
                      'share_roi_fraction':row['share_roi_fraction']})
    write_csv(out/'expert_calibration_grid.csv',b)
    # Small independent-seed robustness audit; each seed has paired decisions
    # across strategies, but these are all draws from the SAME synthetic family.
    repeat=[]
    for trial in range(6):
        this_seed=args.seed+trial
        truth_i,pred_i=generate_arrays(args.frames,args.rois,this_seed,.18)
        local_i,_,_,_=evaluate_offline(truth_i,pred_i,'local')
        for method in ('strict','expert'):
            alt_i,_,_,_=evaluate_offline(truth_i,pred_i,method)
            repeat.append({'seed':this_seed,'method':method,'shock_probability':.18,
                           'delta_r2_pp':100*(alt_i['r_ge_2']-local_i['r_ge_2']),
                           'delta_r3_pp':100*(alt_i['r_all_3']-local_i['r_all_3']),
                           'delta_marginal_pp':100*(alt_i['marginal_error']-local_i['marginal_error'])})
    write_csv(out/'independent_seed_robustness.csv',repeat)

    # Study C: actual executor, one physical DAG per frame, one optional
    # downlink per agent/frame, independent of ROI count. Use common latent
    # frame/ROI draws so 1/2/4 ROI tests are nested paired prefixes.
    truth4,pred4=generate_arrays(args.online_frames,max(ROI_COUNTS),args.seed,.18)
    c=[]
    for roi in ROI_COUNTS:
        t=truth4[:,:roi].copy();p=pred4[:,:roi,:].copy()
        trace=make_trace(traces/f'ROI_{roi}_physical_{args.online_frames}.jsonl',t,p)
        for threshold in THRESHOLDS:
            r=evaluate_online(trace,t,p,'strict',args.seed,threshold=threshold)
            r.update(roi_per_frame=roi,threshold=threshold,shock_probability=.18)
            c.append(r)
        print('Finished full-DAG simulation for',roi,'ROIs per frame')
    write_csv(out/'roi_threshold_physical_cost_grid.csv',c)
    # Actual physical end-to-end A/B comparison for the minority override.
    t,p=generate_arrays(args.online_frames,args.rois,args.seed,.18)
    trace=make_trace(traces/'override_online_replay.jsonl',t,p)
    online=[]
    for method in ('local','majority','strict','expert'):
        online.append({'method':method,**evaluate_online(trace,t,p,method,args.seed)})
    write_csv(out/'offline_online_dag_crosscheck.csv',online)
    d=[]
    for deadline in DEADLINES_MS:
        for method in ('local','strict','expert'):
            d.append(evaluate_deadline(trace,t,p,method,deadline,args.seed))
    write_csv(out/'deadline_reliability_energy_sweep.csv',d)

    chart(a,b,c,d,interval_data,out)
    manifest={
        'dataset':'synthetic, NOT actual V2V4Real 3-agent capture',
        'agents':3,'physical_frames':args.frames,'roi_events_per_frame':args.rois,
        'offline_decision_events_per_shock':args.frames*args.rois,
        'online_frames':args.online_frames,'random_seed':args.seed,
        'true_marginal_error_rates':{'a':P_TRUE[0],'b':P_TRUE[1],'c':P_TRUE[2]},
        'shock_probabilities':SHOCKS,
        'generation':'A is independent with q_A. B,C common wrong shock p=s; otherwise their conditional Bernoulli rates (q_i-s)/(1-s).',
        'masking':'none in main grid; strict F0 accepts only if >=2 independent supporters',
        'expert_extension':'EXPERIMENTAL diagnostic postprocessing of strict F0; one highly trusted source can override mistaken majority; NOT standard F0',
        'expert_true_vs_assumed':'Synthetic population parameters are known by construction; calibration grid deliberately varies reported q_A away from true q_A.',
        'error_interval':'paired frame bootstrap over each frame and its complete ROI event set',
        'independent_seed_stress_check':'six independent synthetic draws from same generator; not domain shift',
        'energy':'hypothetical E1 profiles; validated against actual integrated simulator DAG; no hardware measurement',
        'roi_cost_model':'one physical local detector and F0 edge assessment per frame; ROI projections cost zero additional modeled inference',
        'packet_loss':0.0,'deadline_misses':'none for local fallback; late F0 updates are discarded in deadline sweep',
        'deadline_sweep_ms':DEADLINES_MS,
        'note_on_deadline_plot':'Deterministic fixed service profiles create sharp thresholds; no real-world latency distribution is implied',
        'source_script':'scripts/run_three_agent_tradeoff_frontier.py',
        'no_oracle':'policy input includes binary predictions, fixed assumed q, provenance and timestamps; never ground-truth labels',
    }
    (out/'PROVENANCE.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print('Wrote four paper-oriented plots and reproducible CSVs:',out)
    for method in ('local','majority','strict','expert'):
        row=next(r for r in a if r['method']==method and r['shock_probability']==.18)
        print(method, 'R2',round(100*row['r_ge_2'],3), 'R3',round(100*row['r_all_3'],3),
              'marg',round(100*row['marginal_error'],3), 'share',round(100*row['share_roi_fraction'],2))

if __name__=='__main__':main()
