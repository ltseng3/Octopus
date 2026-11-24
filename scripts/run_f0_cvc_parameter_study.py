#!/usr/bin/env python3
"""CVC study: lineage quality, failure definition, selectivity and sunk energy.

All controlled 5-agent results are from a generated *synthetic trace* with
explicitly specified heterogeneous Bernoulli errors, correlated clique errors,
and the actual F0 decision implementation in offline replay. Exact no-loss
per-frame physical costs are computed analytically and checked against online
OCTOPUS-Sim execution; optional complete online runs remain available.
V2V4Real uses one recorded trace only.
No device runtime/energy in this study is a hardware measurement.

Run: python scripts/run_f0_cvc_parameter_study.py --output samples/f0_cvc_parameter_study
Use --frames 50 --bootstrap 100 for a fast engineering smoke test (not paper figures).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import numpy as np
from octopus_sim.config import ExperimentConfig
from octopus_sim.simulator import run_simulation
from octopus_fusion.fusion import unweighted_majority, weighted_vote
from octopus_fusion.contracts import Action,FusionDecision,AgentEvidence,FusionObservation
from octopus_fusion.fusion import SelectiveFuser
from octopus_fusion.estimators import StaticErrorEstimator
from octopus_fusion.contracts import FusionConfig

AGENTS=('a','b','c','d','e')
TRUE_Q=(.055,.080,.180,.180,.180)
RHOS=(0.0,.25,.50,.75,1.0)
CUTOFFS=(.55,.70,.80,.90,.94,.97,.985,.995,.999)
DEFAULT_RHO=.75
BASE_SEED=5701
# Unobserved nuisance draws shared across rho: paired, not 5 independent datasets.


class NaiveVoteFuser:
    """Unprotected vote carried through the *same physical F0 execution DAG*.

    These are deliberate algorithmic baselines, not the separate pretrained
    cooperative-late-fusion model. No hidden truth is passed to this object.
    """
    def __init__(self, kind: str):
        if kind not in ('majority','weighted'):raise ValueError(kind)
        self.kind=kind

    def decide(self, observation, reliability, *, risk_bounds=None):
        if risk_bounds is not None:raise ValueError('Naive baseline does not use risk bounds')
        prediction=(unweighted_majority(observation) if self.kind=='majority' else
                    weighted_vote(observation,reliability))
        local=tuple(sorted((e.agent_id,e.prediction) for e in observation.evidence))
        if prediction is None:
            return FusionDecision(observation.event_id,observation.frame_id,Action.KEEP_LOCAL,
                                  local,None,0.0,.5,(),(),(),('vote_tie',),
                                  'naive_'+self.kind)
        groups=tuple(sorted({e.dependence_group for e in observation.evidence}))
        supporters=tuple(sorted({e.dependence_group for e in observation.evidence
                                 if e.prediction==prediction}))
        return FusionDecision(observation.event_id,observation.frame_id,Action.SHARE_FUSED,
                              tuple((a,prediction) for a,_ in local),prediction,
                              0.0,.5,groups,supporters,(),(),'naive_'+self.kind)


def generate_controlled_traces(out:Path,frames:int=2000,rois:int=2,seed:int=BASE_SEED):
    """Hold all latent draws fixed as clique-dependence rho varies.

    a,b independent; c,d,e each wrong with probability .18, with probability
    rho using the *same* Bernoulli wrong draw, else independent. Thus rho is a
    true-common-mode mixture parameter (not necessarily Pearson correlation).
    """
    if frames<2 or rois<1:raise ValueError('Need multiple frames and at least one ROI')
    out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(seed)
    truth=rng.integers(0,2,size=(frames,rois),dtype=np.int8)
    individual=rng.random((frames,rois,5))
    clique_mode=rng.random((frames,rois))
    clique_common=rng.random((frames,rois))
    paths={}
    for rho in RHOS:
        location=out/f'synthetic_five_agent_rho_{int(rho*100):03d}.jsonl'
        with location.open('w',encoding='utf-8') as f:
            for frame in range(frames):
                for region in range(rois):
                    y=int(truth[frame,region])
                    errors=[]
                    for i in range(5):
                        if i<2:bad=individual[frame,region,i]<TRUE_Q[i]
                        elif clique_mode[frame,region]<rho:bad=clique_common[frame,region]<TRUE_Q[i]
                        else:bad=individual[frame,region,i]<TRUE_Q[i]
                        errors.append(int(bool(bad)))
                    local={f'local:{a}':y^errors[i] for i,a in enumerate(AGENTS)}
                    # The original trace parser requires edge keys even for
                    # local-only execution. These pass-through placeholders are
                    # NEVER used for a named edge plan in this experiment.
                    placeholders={f'edge:{a}':local[f'local:{a}'] for a in AGENTS}
                    row={'event_id':f'S{frame:05d}:r{region}',
                         'frame_id':f'S{frame:05d}', 'arrival_slot':frame,
                         'truth':y,'candidates':{**local,**placeholders}}
                    f.write(json.dumps(row,separators=(',',':'),sort_keys=True)+'\n')
        paths[rho]=location
    return paths


def make_config(trace:Path,*, method:str,cutoff:float=.90,nframes:int=2000,rois:int=2,
                optimistic_clique:bool=False, max_source_error:float=.49, seed:int=BASE_SEED):
    original=ExperimentConfig.load(ROOT/'examples/f0/three_agent_f0.json')
    d=deepcopy(original.data)
    d['scenario'].update(name='cvc_f0_synthetic_five_agents',seed=seed,episodes=nframes*rois,
                         agents=list(AGENTS),arrival_interval_ms=250,deadline_ms=180)
    d['world']={'backend':'trace','trace_file':str(trace.resolve()),'execution_granularity':'frame'}
    d['policy']={'name':'local_only' if method=='local_only' else 'selective_fusion_f0',
                 'reuse_probability':.5}
    d['coordination']['enabled']=False
    d['resources'].update(network_channels=1,packet_loss=0.0,edge_workers=2)
    d['fusion_f0']['error_by_agent']={a:(.04 if optimistic_clique and i>=2 else TRUE_Q[i])
                                     for i,a in enumerate(AGENTS)}
    d['fusion_f0']['dependence_group_by_agent']=(
        {a:a for a in AGENTS} if method=='false_independence'
        else {'a':'a','b':'b','c':'clique','d':'clique','e':'clique'})
    d['fusion_f0'].update(min_score_confidence=cutoff,max_source_error=max_source_error,
                          min_supporting_groups=2,min_independent_groups=2,
                          max_conflict_fraction=.4,dependence_discount=1.0,
                          prior_positive=.5,max_age_ms=120.0)
    return ExperimentConfig.from_dict(d,original.source_path)


def _frame_group_arrays(scores:list[dict],frame_ids:list[str]):
    """Return one cluster per physical frame; never bootstrap individual ROI labels."""
    groups:dict[str,list[dict]]=defaultdict(list)
    for score in scores:
        groups[score['event_id'].split(':r')[0]].append(score)
    if set(groups)!=set(frame_ids):raise AssertionError('Missing or duplicated physical frame scores')
    n=len(scores)
    if n!=sum(len(g) for g in groups.values()):raise AssertionError('Lost decisions')
    per_frame=[]
    for fid in frame_ids:
        g=groups[fid]
        if len(g)!=len(groups[frame_ids[0]]):raise AssertionError('Unequal ROI count per frame')
        per_frame.append([[float(score['error_count']>=m) for m in range(1,len(AGENTS)+1)]+
                          [sum(v is True for v in score['wrong'].values())/len(AGENTS)] for score in g])
    return np.asarray(per_frame,dtype=float).mean(axis=1)


def _local_scores_from_trace(trace:Path):
    scores=[]
    for line in trace.read_text(encoding='utf-8').splitlines():
        if not line:continue
        r=json.loads(line)
        wrong={a:(r['candidates']['local:'+a]!=r['truth']) for a in AGENTS}
        scores.append({'event_id':r['event_id'],'wrong':wrong,
                       'all_wrong':all(wrong.values()),'error_count':sum(wrong.values())})
    return scores


def _write_csv(path:Path,records:list[dict]):
    if not records:raise ValueError('Cannot save empty CSV')
    with path.open('w',newline='',encoding='utf-8') as f:
        fields=sorted(set().union(*(record.keys() for record in records)))
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(records)


def study_run(trace:Path,*,method:str,cutoff:float=.90,nframes:int=2000,rois:int=2,
              optimistic_clique:bool=False,max_source_error:float=.49):
    conf=make_config(trace,method=method,cutoff=cutoff,nframes=nframes,rois=rois,
                     optimistic_clique=optimistic_clique,max_source_error=max_source_error)
    inject=NaiveVoteFuser(method.replace('naive_','')) if method in ('naive_majority','naive_weighted') else None
    res=run_simulation(conf,detailed_events=False,**({'f0_fuser':inject} if inject else {}))
    s=res.summary
    assert s['complete_event_rate']==1.0 and s['agent_deadline_miss_rate']==0.0
    assert s['physical_execution_batches']==nframes
    assert s['local_inference_executions']==nframes*len(AGENTS)
    if method!='local_only':
        assert s['edge_inference_executions']==nframes
        assert len(res.fusion_decisions)==nframes*rois
    # Distinguish applied result replacements from ACTUAL changed binary labels.
    local={x['event_id']:{a:x['wrong'][a] for a in AGENTS} for x in _local_scores_from_trace(trace)}
    changed=sum(sum(resrow['wrong'][a]!=local[resrow['event_id']][a] for a in AGENTS)
                for resrow in res.scores)
    if method=='local_only':
        shared=0.0; downlinked=0.0; applied=0;masked=0
    else:
        decisions=res.fusion_decisions
        shared=sum(x['action']!='KEEP_LOCAL' for x in decisions)/len(decisions)
        downlinked=len({x['frame_id'] for x in decisions if x['action']!='KEEP_LOCAL'})/nframes
        applied=s['f0_realized_agent_overrides']
        masked=s['f0_masked_fusion_decisions']
    total=s.get('energy_workload_incremental_j')
    if total is None:raise ValueError('Hypothetical energy fixture was disabled')
    breakdown=s.get('energy_compute_by_model_j',{})
    return {
        'model':'synthetic_five_agents', 'method':method,'cutoff':cutoff,
        'mask_threshold':max_source_error,'optimistic_clique':optimistic_clique,
        'frames':nframes,'rois_per_frame':rois,'decisions':len(res.scores),
        **{f'r_ge_{m}':s['tail_error_rates_by_m'][str(m)] for m in range(1,6)},
        'marginal_error':s['marginal_error_rate_all_events'],
        'share_roi_coverage':shared,'downlinked_frame_fraction':downlinked,
        'applied_result_replacements':applied,'changed_agent_decisions':changed,
        'changed_agent_decision_fraction':changed/(len(res.scores)*len(AGENTS)),
        'incremental_energy_j':total,'energy_per_frame_j':total/nframes,
        'local_compute_energy_j':breakdown.get('local_model',0.0),
        'edge_compute_energy_j':breakdown.get('f0_decider',0.0),
        'radio_energy_j':s['energy_radio_incremental_j'],
        'radio_per_frame_j':s['energy_radio_incremental_j']/nframes,
        'downlink_radio_per_frame_j':(
            s['transmitted_bytes_by_role'].get('result',0)*8*
            (conf.data['energy']['device_profiles']['edge']['radio']['tx_j_per_bit']+
             conf.data['energy']['device_profiles']['agent']['radio']['rx_j_per_bit'])/nframes),
        'f0_compute_per_frame_j':breakdown.get('f0_decider',0.0)/nframes,
        'local_compute_per_frame_j':breakdown.get('local_model',0.0)/nframes,
        'network_bytes':s['transmitted_bytes'],
        'mean_latency_ms':s['mean_delivered_latency_ms'],
        'p95_latency_ms':s['p95_delivered_latency_ms'],
        'compute_job_count':s['local_inference_executions']+s['edge_inference_executions'],
    }, res.scores



def fast_study_run(trace:Path,*,method:str,cutoff:float=.90,nframes:int=2000,rois:int=2,
                   optimistic_clique:bool=False,max_source_error:float=.49):
    """Actual F0 decision code over all ROI outcomes; exact L0 work projection.

    This is not the event-driven runtime: it is a deterministic replay of the
    SAME fuser. Cost estimates are based on the *fixed* no-loss, no-queueing,
    additive E1 profiles, independently checked via real GraphExecutor runs.
    """
    conf=make_config(trace,method=method,cutoff=cutoff,nframes=nframes,rois=rois,
                     optimistic_clique=optimistic_clique,max_source_error=max_source_error)
    f0=conf.data['fusion_f0'];resource=conf.data['resources'];energy=conf.data['energy']
    if not (resource['packet_loss']==0 and energy['enabled'] and
            all(p['radio']['mode']=='per_bit' for p in energy['device_profiles'].values())):
        raise ValueError('Fast physical cost projection requires zero packet loss and E1 per-bit radio')
    if energy['agent_overrides'] or energy['model_energy_j']:
        raise ValueError('Fast projection assumes no agent/fixed-joule power overrides')
    if (resource['model_profiles']['local_model']['compute_ms']!=20 or
        resource['model_profiles']['f0_decider']['compute_ms']!=1.0 or
        resource['network_latency_ms']!=5 or resource['network_channels']!=1):
        raise ValueError('Fast projection only validated for the pinned fixed-duration synthetic fixture')
    cfg=FusionConfig(**{k:v for k,v in f0.items() if k not in
        {'error_by_agent','dependence_group_by_agent','quality_proxy_by_agent','prior_positive'}})
    if cfg.max_age_ms < 12:
        raise ValueError('Offline projection assumes nonbinding age gate (>=12 ms)')
    fuser=NaiveVoteFuser(method[len('naive_'):]) if method.startswith('naive_') else SelectiveFuser(cfg)
    est=StaticErrorEstimator(tuple(sorted(f0['error_by_agent'].items())))
    scores=[];f0_decisions=[];frames_with_any=set();changed=0
    in_source=0
    for line in trace.read_text(encoding='utf-8').splitlines():
        if not line.strip():continue
        rec=json.loads(line);in_source+=1
        event_id=rec['event_id'];frame_id=rec['frame_id'];t=rec['truth']
        orig={a:rec['candidates']['local:'+a] for a in AGENTS}
        if method=='local_only':
            verdict=None;predictions=orig
        else:
            obs=FusionObservation(event_id,frame_id,as_of_ms=31.,
                evidence=tuple(AgentEvidence(a,orig[a],f'{frame_id}:local:{a}',
                                 f0['dependence_group_by_agent'].get(a,a),
                                 produced_at_ms=20.) for a in AGENTS),
                prior_positive=f0['prior_positive'])
            reliability=est.estimate(obs)
            verdict=fuser.decide(obs,reliability)
            f0_decisions.append(verdict)
            if verdict.shared:
                frames_with_any.add(frame_id)
            predictions=dict(verdict.per_agent_decisions)
        changed+=sum(orig[a]!=predictions[a] for a in AGENTS)
        wrong={a:(predictions[a]!=t) for a in AGENTS}
        scores.append({'event_id':event_id,'wrong':wrong,
                       'all_wrong':all(wrong.values()),'error_count':sum(wrong.values())})
    if in_source != nframes*rois:raise AssertionError('Trace source length differs from expected workload')
    # E1 additive incremental energy over *physical* operations; local and edge
    # baselines excluded (same fixed horizon, measured separately).
    m=resource['model_profiles']
    model_w=energy['model_incremental_w']
    agent=energy['device_profiles']['agent'];edge=energy['device_profiles']['edge']
    local_power=model_w.get('local_model',agent['compute_incremental_w'])
    edge_power=model_w.get('f0_decider',edge['compute_incremental_w'])
    local_j=len(AGENTS)*m['local_model']['compute_ms']/1000*local_power*nframes
    f0_j=(m['f0_decider']['compute_ms']/1000*edge_power*nframes if method!='local_only' else 0.)
    ul_bytes=nframes*len(AGENTS)*m['local_model']['output_bytes'] if method!='local_only' else 0
    dl_bytes=len(frames_with_any)*len(AGENTS)*m['f0_decider']['output_bytes'] if method!='local_only' else 0
    uplink_j=ul_bytes*8*(agent['radio']['tx_j_per_bit']+edge['radio']['rx_j_per_bit'])
    downlink_j=dl_bytes*8*(edge['radio']['tx_j_per_bit']+agent['radio']['rx_j_per_bit'])
    radio_j=uplink_j+downlink_j
    share=sum(x.shared for x in f0_decisions)/len(f0_decisions) if f0_decisions else 0.0
    applied=len(AGENTS)*sum(x.shared for x in f0_decisions)
    masked=sum(x.action==Action.MASK_AND_FUSE for x in f0_decisions)
    errs=np.fromiter((s['error_count'] for s in scores),dtype=np.int16,count=len(scores))
    no_fusion=method=='local_only'
    result={
        'model':'synthetic_five_agents','method':method,'cutoff':cutoff,
        'mask_threshold':max_source_error,'optimistic_clique':optimistic_clique,
        'frames':nframes,'rois_per_frame':rois,'decisions':len(scores),
        **{f'r_ge_{m}':float(np.mean(errs>=m)) for m in range(1,6)},
        'marginal_error':float(errs.mean()/5),
        'share_roi_coverage':share,'downlinked_frame_fraction':len(frames_with_any)/nframes,
        'applied_result_replacements':applied,'changed_agent_decisions':changed,
        'changed_agent_decision_fraction':changed/(len(scores)*len(AGENTS)),
        'incremental_energy_j':local_j+f0_j+radio_j,
        'energy_per_frame_j':(local_j+f0_j+radio_j)/nframes,
        'local_compute_energy_j':local_j,'edge_compute_energy_j':f0_j,
        'radio_energy_j':radio_j,'radio_per_frame_j':radio_j/nframes,
        'downlink_radio_per_frame_j':downlink_j/nframes,
        'f0_compute_per_frame_j':f0_j/nframes,'local_compute_per_frame_j':local_j/nframes,
        'network_bytes':ul_bytes+dl_bytes,
        'mean_latency_ms':None, 'p95_latency_ms':None,
        'compute_job_count':(len(AGENTS)+(0 if no_fusion else 1))*nframes,
        'execution_scope':'actual F0 vote on synthetic trace; analytical no-loss resource projection validated online',
    }
    return result,scores


def validate_online_subset(trace:Path,*,method:str,frames:int,rois:int,
                           cutoff:float=.90, optimistic_clique:bool=False,
                           max_source_error:float=.49, tempdir:Path):
    """Cross-check offline every ROI against the actual compiled GraphExecutor."""
    tempdir.mkdir(parents=True,exist_ok=True)
    excerpt=tempdir/f'online_check_{method}_{cutoff}.jsonl'
    with trace.open('r',encoding='utf-8') as inp, excerpt.open('w',encoding='utf-8') as dst:
        for i,line in enumerate(inp):
            if i==frames*rois:break
            dst.write(line)
    kw=dict(method=method,nframes=frames,rois=rois,cutoff=cutoff,
            optimistic_clique=optimistic_clique,max_source_error=max_source_error)
    offline,oscores=fast_study_run(excerpt,**kw)
    online,rscores=study_run(excerpt,**kw)
    if ([(r['event_id'],r['wrong']) for r in rscores] !=
        [(r['event_id'],r['wrong']) for r in oscores]):
        raise AssertionError(f'Offline/online F0 decision mismatch for {method}')
    for key in ('r_ge_1','r_ge_3','r_ge_5','marginal_error','share_roi_coverage',
                'downlinked_frame_fraction','changed_agent_decisions','network_bytes',
                'incremental_energy_j','local_compute_energy_j','edge_compute_energy_j',
                'radio_energy_j','downlink_radio_per_frame_j','compute_job_count'):
        if not np.isclose(online[key],offline[key],rtol=1e-10,atol=1e-9):
            raise AssertionError(f'Offline/online resource mismatch {method}: {key}: '
                                 f'{offline[key]} vs {online[key]}')
    if not (online['mean_latency_ms'] is not None and online['p95_latency_ms'] is not None):
        raise AssertionError('Online latency missing')
    return {'method':method,'cutoff':cutoff,'checked_frames':frames,
            'checked_decisions':frames*rois,'energy_j':online['incremental_energy_j'],
            'online_mean_latency_ms':online['mean_latency_ms'],
            'verified_prediction_and_resource_fields':14}

def summarize_local(trace:Path,nframes:int,rois:int):
    scores=_local_scores_from_trace(trace)
    a=np.asarray([int(s['error_count']>=m) for s in scores for m in range(1,6)]).reshape(-1,5)
    marg=sum(s['error_count'] for s in scores)/(len(scores)*5)
    return {'method':'local_only','frames':nframes,'rois_per_frame':rois,'decisions':len(scores),
            **{f'r_ge_{m}':float(a[:,m-1].mean()) for m in range(1,6)},
            'marginal_error':marg,'share_roi_coverage':0.0,
            'downlinked_frame_fraction':0.0,'applied_result_replacements':0,
            'changed_agent_decisions':0,'changed_agent_decision_fraction':0.0}, scores


def boot_ci(array:np.ndarray,indices:np.ndarray, *, lower=.025,upper=.975):
    if array.ndim!=1:raise ValueError('Expect one rate per frame')
    means=array[indices].mean(axis=1)
    return float(np.quantile(means,lower)),float(np.quantile(means,upper))


def _save_fig(fig,out:Path,name:str):
    import matplotlib.pyplot as plt
    for ext in ('png','pdf','svg'):
        fig.savefig(out/f'{name}.{ext}',dpi=300,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def make_plots(out:Path,all_rows:list[dict],rho_rows:list[dict],threshold_rows:list[dict],
               paired_rows:list[dict],empirical_rows:list[dict],nframes:int):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,
                         'axes.labelsize':8.5,'xtick.labelsize':7,'ytick.labelsize':7,
                         'legend.fontsize':7,'pdf.fonttype':42,'ps.fonttype':42,
                         'svg.fonttype':'none','lines.linewidth':1.7,'axes.spines.top':False,
                         'axes.spines.right':False,'figure.facecolor':'white'})
    colors={'local_only':'#4B5563','naive_majority':'#D97724',
            'naive_weighted':'#9C4C93','lineage_aware':'#177E89',
            'false_independence':'#C23A48','mask_clique':'#647A25',
            'optimistic_clique':'#A55B1A'}
    labels={'local_only':'Local only','naive_majority':'Naive majority',
            'naive_weighted':'Naive weighted','lineage_aware':'Lineage-aware F0',
            'false_independence':'F0: false independence','mask_clique':'Mask clique',
            'optimistic_clique':'Optimistic clique trust'}
    main={r['method']:r for r in all_rows if r['rho']==DEFAULT_RHO and r['experiment']=='lineage_sweep'}
    assert all(m in main for m in ('local_only','lineage_aware','false_independence','naive_majority'))
    # Figure 1: distinct system-risk definitions reverse baseline ranking.
    fig,axes=plt.subplots(1,2,figsize=(7.15,3.0),layout='constrained')
    methods=('local_only','naive_majority','naive_weighted','lineage_aware','false_independence')
    for method in methods:
        r=main[method];vals=[100*r[f'r_ge_{m}'] for m in range(1,6)]
        # A logarithmic axis cannot plot zero: leave it missing rather than
        # fabricate a tiny observed failure probability for visual appearance.
        axes[0].plot(range(1,6),[v if v>0 else np.nan for v in vals],
                     marker='o',markersize=3.5,color=colors[method],label=labels[method])
        axes[1].scatter(100*r['marginal_error'],100*r['r_ge_5'],s=64,
                        marker='s' if method=='lineage_aware' else 'o',
                        color=colors[method],edgecolor='white',linewidth=.6,zorder=3)
    axes[0].set_yscale('log');axes[0].set_xticks(range(1,6))
    axes[0].set_xlabel('At least $m$ of 5 agents wrong')
    axes[0].set_ylabel('Collective failure probability (%)')
    axes[0].legend(loc='lower left',frameon=False,fontsize=6.3)
    axes[0].set_title('(a) The failure definition matters')
    axes[1].set_xlabel('Average per-agent error (%)')
    axes[1].set_ylabel('All-five-agent error (%)')
    axes[1].set_title('(b) Marginal accuracy vs. shared failure')
    axes[1].text(.01,.03,'Colors/markers follow legend in (a)',transform=axes[1].transAxes,
                 fontsize=6.5,color='#475569')
    for ax in axes:ax.grid(alpha=.20)
    fig.suptitle('Five agents; one three-agent correlated clique; $\\rho=0.75$ (synthetic)',fontsize=9.3)
    _save_fig(fig,out,'01_risk_definition_reversal')
    # Figure 2: metadata-quality stress test with paired uncertainty intervals.
    fig,axes=plt.subplots(1,2,figsize=(7.15,3.0),layout='constrained')
    for ax,m in zip(axes,(3,5)):
        for method in ('local_only','naive_majority','lineage_aware','false_independence'):
            subset=sorted((r for r in rho_rows if r['method']==method),key=lambda x:x['rho'])
            xx=np.array([r['rho'] for r in subset]); yy=np.array([r[f'r_ge_{m}']*100 for r in subset])
            lo=np.array([r[f'r_ge_{m}_ci_low']*100 for r in subset])
            hi=np.array([r[f'r_ge_{m}_ci_high']*100 for r in subset])
            ax.plot(xx,yy,'o-',color=colors[method],label=labels[method],markersize=3)
            ax.fill_between(xx,lo,hi,color=colors[method],alpha=.11,linewidth=0)
        ax.set_xlabel('Common-shock mixture weight $\\rho$')
        ax.set_ylabel(f'Pr($\\geq${m} wrong) (%)')
        ax.grid(alpha=.22)
        ax.set_title('(a) Majority-level risk' if m==3 else '(b) Complete-team risk')
    axes[0].legend(loc='best',frameon=False,fontsize=6.4)
    axes[1].set_yscale('log')
    axes[1].set_ylim(.025,30)
    axes[1].text(.02,.025,'Zero observed local failures at low $\\rho$ omitted on log axis',
                 transform=axes[1].transAxes,fontsize=5.9,color='#475569')
    fig.suptitle('Correct dependence grouping vs. falsely independent votes (synthetic)',fontsize=9.3)
    _save_fig(fig,out,'02_lineage_quality_vs_correlation')
    # Figure 3: post-inference abstention vs irreversible cost.
    fig,axes=plt.subplots(1,2,figsize=(7.15,3.0),layout='constrained')
    subset=sorted(threshold_rows,key=lambda x:x['cutoff'])
    x=np.array([x['cutoff'] for x in subset]);cov=np.array([r['share_roi_coverage'] for r in subset])*100
    down=np.array([r['downlinked_frame_fraction'] for r in subset])*100
    changed=np.array([r['changed_agent_decision_fraction'] for r in subset])*100
    axes[0].plot(x,cov,'o-',label='Accepted ROI decisions',color='#177E89',markersize=3)
    axes[0].plot(x,down,'s--',label='Frames requiring downlink',color='#D97724',markersize=3)
    axes[0].plot(x,changed,'^-',label='Agent labels actually changed',color='#4B5563',markersize=3)
    axes[0].set(xlabel='Minimum surrogate confidence threshold',ylabel='Fraction (%)',title='(a) Rejected ROI does not imply saved downlink')
    axes[0].legend(loc='best',frameon=False,fontsize=6.2)
    xarr=np.array([r['cutoff'] for r in subset])
    # Display *incremental overhead beyond local-only* in millijoules. This
    # makes the tiny conditional-downlink saving visible without obscuring
    # the irrecoverable local detection/edge computation/uplink costs.
    fixed=np.array([1000*(r['f0_compute_per_frame_j'] +
             r['radio_per_frame_j']-r['downlink_radio_per_frame_j']) for r in subset])
    conditional=np.array([1000*r['downlink_radio_per_frame_j'] for r in subset])
    axes[1].fill_between(xarr,0,fixed,color='#64748B',alpha=.82,
                         label='Edge assessment + all uploads (sunk)')
    axes[1].fill_between(xarr,fixed,fixed+conditional,color='#D97724',alpha=.89,
                         label='Conditional downlink only')
    axes[1].plot(xarr,fixed+conditional,'o-',color='#177E89',ms=3)
    axes[1].set_ylim(0,27)
    axes[1].set(xlabel='Minimum surrogate confidence threshold',
                ylabel='Additional modeled energy (mJ/frame)',
                title='(b) Rejecting fusion barely reduces spent energy')
    axes[1].legend(loc='center right',frameon=False,fontsize=6.2)
    for ax in axes:ax.grid(alpha=.18)
    fig.suptitle('F0 decisions share a physical frame; two ROI decisions per frame',fontsize=9.3)
    _save_fig(fig,out,'03_selectivity_vs_sunk_energy')
    # Figure 4: calibrated/misreported probabilities, lineage and masking.
    fig,axes=plt.subplots(1,2,figsize=(7.15,3.5),layout='constrained',sharey=True)
    variants=[r for r in all_rows if r['rho']==DEFAULT_RHO and
              r['experiment'] in ('lineage_sweep','trust_perturbation')]
    ordered=['local_only','naive_majority','naive_weighted','false_independence',
             'optimistic_clique','lineage_aware','mask_clique']
    values={r['method']:r for r in variants};ordered=[x for x in ordered if x in values]
    short={'local_only':'Local', 'naive_majority':'Naive majority',
           'naive_weighted':'Weighted, no lineage','false_independence':'F0, false lineage',
           'optimistic_clique':'F0, optimistic trust','lineage_aware':'F0, correct lineage',
           'mask_clique':'F0, mask clique'}
    yy=np.arange(len(ordered))
    for ax,key,title in [(axes[0],'r_ge_3','(a) At least three wrong'),
                         (axes[1],'r_ge_5','(b) All five wrong')]:
        data=np.array([100*values[m][key] for m in ordered])
        ax.barh(yy,data,color=[colors[m] for m in ordered],height=.72,alpha=.91)
        ax.set_xlim(0,max(data)*1.18+0.3)
        ax.set_yticks(yy)
        if ax is axes[0]:
            ax.set_yticklabels([short[m] for m in ordered],fontsize=6.7)
        else:
            ax.tick_params(axis='y',labelleft=False)
        ax.set_xlabel('Failure probability (%)')
        ax.set_title(title)
        ax.grid(axis='x',alpha=.19)
        for j,value in enumerate(data):
            ax.text(value+.10,j,f'{value:.2f}',va='center',fontsize=6.6,color='#374151')
    axes[0].invert_yaxis()
    fig.suptitle('Bad trust/lineage metadata can erase fault-masking gains (synthetic)',fontsize=9.3)
    _save_fig(fig,out,'04_masking_and_miscalibration')
    if empirical_rows:
        # Figure 5: factual two-agent trace, where sharing agreements is not recovery.
        fig,axes=plt.subplots(1,2,figsize=(7.15,2.8),layout='constrained')
        by={r['method']:r for r in empirical_rows};keys=['local_only','f0','cooperative']
        xx=np.arange(3)
        colors2=['#4B5563','#177E89','#D97724']
        axes[0].bar(xx-.14,[100*by[k]['shared_roi_coverage'] for k in keys],width=.28,
                    color=colors2,alpha=.75,label='Fused-result usage')
        axes[0].bar(xx+.14,[100*by[k]['changed_agent_decision_fraction'] for k in keys],width=.28,
                    color=colors2,hatch='///',alpha=.8,label='Actual label changes')
        axes[0].set_xticks(xx);axes[0].set_xticklabels(['Local only','Selective F0','Recorded fusion'])
        axes[0].set(ylabel='Fraction (%)',title='(a) Sharing does not imply correction')
        axes[0].legend(loc='upper left',frameon=False,fontsize=6.4)
        for k,c in zip(keys,colors2):
            r=by[k]
            axes[1].scatter(r['energy_per_frame_j'],r['marginal_error']*100,
                            color=c,s=60,edgecolors='white',zorder=3)
            axes[1].annotate({'local_only':'Local','f0':'F0','cooperative':'Recorded fusion'}[k],
                             (r['energy_per_frame_j'],r['marginal_error']*100),
                             xytext=(5,5),textcoords='offset points',fontsize=7,color=c)
        axes[1].set(xlabel='Modeled incremental energy (J/frame)',ylabel='Mean agent error (%)',
                    title='(b) Pipeline choice is a different comparison')
        for ax in axes:ax.grid(alpha=.20)
        fig.suptitle('V2V4Real sequence 0007: 221 frames, 442 ROI decisions (one sequence)',fontsize=9.2)
        _save_fig(fig,out,'05_v2v4real_agreement_trap')


def empirical_v2v() -> list[dict]:
    base=ExperimentConfig.load(ROOT/'examples/f0/v2v4real_f0.json')
    reference={}
    rows=[]
    for method,policy in [('local_only','local_only'),('f0','selective_fusion_f0'),
                          ('cooperative','cooperative_late_fusion')]:
        raw=deepcopy(base.data);raw['policy']={'name':policy,'reuse_probability':.5}
        if method=='f0':raw['fusion_f0']['min_score_confidence']=.90
        cfg=ExperimentConfig.from_dict(raw,base.source_path)
        result=run_simulation(cfg,detailed_events=False)
        s=result.summary
        if not (s['episodes']==442 and s['physical_execution_batches']==221 and
                s['complete_event_rate']==1 and s['agent_deadline_miss_rate']==0):
            raise ValueError('Unexpected/incomplete V2V4Real replay')
        if method=='local_only':reference={q['event_id']:q['predictions'] for q in result.scores}
        changes=sum(sum(row['predictions'][agent]!=reference[row['event_id']][agent]
                        for agent in ('ego','agent1')) for row in result.scores)
        coverage=(0.0 if method=='local_only' else 1.0 if method=='cooperative' else s['f0_shared_roi_coverage'])
        rows.append({'model':'V2V4Real_0007','method':method,'frames':221,'decisions':442,
                     'shared_roi_coverage':coverage,
                     'changed_agent_decision_fraction':changes/(442*2),
                     'changed_agent_decisions':changes,
                     'applied_result_replacements':(s.get('f0_realized_agent_overrides') if method=='f0' else None),
                     'all_agent_error':s['joint_all_error_rate'],
                     'marginal_error':s['marginal_error_rate_all_events'],
                     'energy_per_frame_j':s['energy_workload_incremental_j']/221,
                     'mean_latency_ms':s['mean_delivered_latency_ms'],
                     'physical_local_jobs':s['local_inference_executions'],
                     'physical_edge_jobs':s['edge_inference_executions'],
                     'energy_scope':'hypothetical','risk_scope':'one recorded driving sequence',
                     'pipeline_caution':'recorded pretrained fusion differs from F0 vote'})
    return rows


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=ROOT/'samples/f0_cvc_parameter_study')
    ap.add_argument('--frames',type=int,default=2000)
    ap.add_argument('--rois',type=int,default=2)
    ap.add_argument('--bootstrap',type=int,default=2500)
    ap.add_argument('--seed',type=int,default=BASE_SEED)
    ap.add_argument('--no-plots',action='store_true')
    ap.add_argument('--skip-v2v',action='store_true')
    ap.add_argument('--online-check-frames',type=int,default=180)
    args=ap.parse_args(argv)
    if args.frames<20 or args.rois<1 or args.bootstrap<50 or args.online_check_frames<0 or args.online_check_frames>args.frames:
        ap.error('Require >=20 frames, >=1 ROI/frame, >=50 bootstrap draws')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    traces=generate_controlled_traces(out/'generated_traces',args.frames,args.rois,args.seed)
    fids=[f'S{i:05d}' for i in range(args.frames)]
    rng=np.random.default_rng(args.seed+109)
    indices=rng.integers(0,args.frames,size=(args.bootstrap,args.frames),dtype=np.int32)
    rates=[];rho_rows=[];threshold_rows=[];paired=[]
    cache={}
    # One full end-to-end *local-only* replay confirms local error metrics
    # and provides a physical workload energy baseline. The same original
    # truth/local error samples are used for every rho (by construction).
    local_cost, local_online=fast_study_run(traces[DEFAULT_RHO],method='local_only',
                                      nframes=args.frames,rois=args.rois)
    local_truth, local_offline=summarize_local(traces[DEFAULT_RHO],args.frames,args.rois)
    if ([(s['event_id'],s['wrong']) for s in local_online] !=
        [(s['event_id'],s['wrong']) for s in local_offline]):
        raise AssertionError('Offline local baseline differs from executed simulator')
    physical_fields=('incremental_energy_j','energy_per_frame_j','local_compute_energy_j',
                     'edge_compute_energy_j','radio_energy_j','radio_per_frame_j',
                     'f0_compute_per_frame_j','local_compute_per_frame_j','network_bytes',
                     'mean_latency_ms','p95_latency_ms','compute_job_count')
    print('Controlled synthetic frames:',args.frames,'ROI/frame:',args.rois,flush=True)
    for rho,trace in traces.items():
        local,locscores=summarize_local(trace,args.frames,args.rois)
        locarray=_frame_group_arrays(locscores,fids)
        cache[(rho,'local_only')]=locarray
        local.update({key:local_cost[key] for key in physical_fields})
        local.update(rho=rho,experiment='lineage_sweep')
        for m in range(1,6):
            lo,hi=boot_ci(locarray[:,m-1],indices)
            local[f'r_ge_{m}_ci_low']=lo;local[f'r_ge_{m}_ci_high']=hi
        rates.append(local);rho_rows.append(local)
        for method in ('naive_majority','naive_weighted','lineage_aware','false_independence'):
            record,scores=fast_study_run(trace,method=method,nframes=args.frames,rois=args.rois)
            recarray=_frame_group_arrays(scores,fids)
            cache[(rho,method)]=recarray
            record.update(rho=rho,experiment='lineage_sweep')
            for m in range(1,6):
                lo,hi=boot_ci(recarray[:,m-1],indices)
                record[f'r_ge_{m}_ci_low']=lo;record[f'r_ge_{m}_ci_high']=hi
            rates.append(record);rho_rows.append(record)
            if rho==DEFAULT_RHO:
                for m in (3,5):
                    diff=recarray[:,m-1]-locarray[:,m-1]
                    lo,hi=boot_ci(diff,indices)
                    paired.append({'scenario':'five_agent_correlated','rho':rho,'method_a':method,
                                   'method_b':'local_only','risk_definition':f'R_{m}',
                                   'paired_risk_difference':float(diff.mean()),
                                   'paired_cluster_bootstrap_low':lo,'paired_cluster_bootstrap_high':hi,
                                   'resampling_unit':'physical_frame'})
            print(f'  rho={rho:.2f} {method:20} R3={record["r_ge_3"]:.3%} '
                  f'R5={record["r_ge_5"]:.3%} marginal={record["marginal_error"]:.3%}',flush=True)
    mid_trace=traces[DEFAULT_RHO]
    for method,optimistic,mask in [('optimistic_clique',True,.49),('mask_clique',False,.15)]:
        rec,scores=fast_study_run(mid_trace,method=method,optimistic_clique=optimistic,
                             max_source_error=mask,nframes=args.frames,rois=args.rois)
        rec.update(rho=DEFAULT_RHO,experiment='trust_perturbation')
        rates.append(rec)
        recarray=_frame_group_arrays(scores,fids)
        for m in (3,5):
            diff=recarray[:,m-1]-cache[(DEFAULT_RHO,'lineage_aware')][:,m-1]
            lo,hi=boot_ci(diff,indices)
            paired.append({'scenario':'five_agent_correlated','rho':DEFAULT_RHO,
                           'method_a':method,'method_b':'lineage_aware','risk_definition':f'R_{m}',
                           'paired_risk_difference':float(diff.mean()),
                           'paired_cluster_bootstrap_low':lo,'paired_cluster_bootstrap_high':hi,
                           'resampling_unit':'physical_frame'})
    # Cost and coverage sweep: all points use same frames/errors (common random numbers).
    for cutoff in CUTOFFS:
        rec,scores=fast_study_run(mid_trace,method='lineage_aware',cutoff=cutoff,
                             nframes=args.frames,rois=args.rois)
        rec.update(rho=DEFAULT_RHO,experiment='selectivity_sweep')
        threshold_rows.append(rec)
    # Integration gate: spot-check full event-driven DAG, energy and predictions.
    online_checks=[]
    if args.online_check_frames:
        for method,cutoff,optimistic,mask in [
            ('local_only',.90,False,.49),('naive_majority',.90,False,.49),
            ('naive_weighted',.90,False,.49),('lineage_aware',.90,False,.49),
            ('false_independence',.90,False,.49),('optimistic_clique',.90,True,.49),
            ('mask_clique',.90,False,.15),('lineage_aware',.999,False,.49)]:
            check=validate_online_subset(mid_trace,method=method,frames=args.online_check_frames,
                rois=args.rois,cutoff=cutoff,optimistic_clique=optimistic,
                max_source_error=mask,tempdir=out/'online_validation_traces')
            online_checks.append(check)
        _write_csv(out/'online_executor_consistency.csv',online_checks)
    empirical=[] if args.skip_v2v else empirical_v2v()
    _write_csv(out/'five_agent_policy_and_risk_grid.csv',rates)
    _write_csv(out/'lineage_correlation_sweep.csv',rho_rows)
    _write_csv(out/'threshold_energy_coverage_sweep.csv',threshold_rows)
    _write_csv(out/'paired_frame_bootstrap_contrasts.csv',paired)
    if empirical:_write_csv(out/'v2v4real_agreement_control.csv',empirical)
    trace_digests={f'rho_{rho:.2f}':hashlib.sha256(path.read_bytes()).hexdigest()
                   for rho,path in traces.items()}
    report={
        'source_version':'OCTOPUS-Sim v0.7.1, OCTOPUS-Fusion F0 v0.1.0 integrated',
        'controlled_study_mode':'actual F0 per-ROI algorithm replay; deterministic analytical E1 physical cost projection; full online executor validated on first N frames per specified policy',
        'online_validation':online_checks,
        'purpose':'controlled sensitivity analysis, NOT evidence of measured deployed robustness',
        'synthetic':{'agents':AGENTS,'true_agent_error_probabilities':TRUE_Q,
                     'correlated_clique':['c','d','e'],'rho_mixture_weights':RHOS,
                     'frame_count':args.frames,'regions_per_frame':args.rois,
                     'seed':args.seed,'rng':'numpy Generator PCG64',
                     'common_random_numbers_across_rho':True,
                     'same_decision_truths_across_parameter_sweeps':True,
                     'per_agent_error_estimate':'true marginal q except explicitly labeled trust-misspecification scenarios',
                     'edge_placeholder_keys':'synthetic parser compatibility; no independent edge candidate used'},
        'statistical':{'unit':'physical_frame (all ROIs kept in one cluster)',
                       'method':'paired nonparametric percentile bootstrap, pointwise, not multiplicity-adjusted',
                       'replicates':args.bootstrap,'bootstrap_seed':args.seed+109,
                       'scene_generalization':'NOT claimed; synthetic frames are generated IID'},
        'resource':{'source':'hypothetical E1 power/timing profiles',
                    'one_local_compute_per_agent_per_frame':True,
                    'one_F0_edge_decision_per_frame':True,
                    'downlink':'one fixed payload per agent per frame if ANY ROI accepted',
                    'decision_override':'separate physical result ID replacement from actual changed binary label',
                    'timing':'250 ms frame interval, 180 ms deadline; all results required complete'},
        'v2v4real':{'sequence':'0007','frames':221,'decisions':442,
                    'scope':'single recorded sequence; no multi-scene uncertainty claim',
                    'model_pipeline_difference':'recorded cooperative model is NOT the F0 vote'},
        'hypotheses':['Dependence provenance influences systemic risk more than nominal vote count',
                      'Risk threshold m can reverse a policy ranking',
                      'Post-inference fallback has limited energy leverage after local compute/uploads',
                      'Two-agent agreement can cause share without any changed binary decision'],
        'trace_sha256':trace_digests,
    }
    (out/'STUDY_PROVENANCE.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    if not args.no_plots:make_plots(out,rates,rho_rows,threshold_rows,paired,empirical,args.frames)
    print('Study complete:',out, 'Figures=',0 if args.no_plots else (4 if args.skip_v2v else 5),
          'Synthetic F0 algorithm replays=',len(rates)-len(RHOS)+len(threshold_rows),flush=True)
    return out

if __name__=='__main__':main()
