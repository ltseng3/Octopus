#!/usr/bin/env python3
"""Reproducible CVC figure exploration using OCTOPUS-Sim v0.7.

Evidence is deliberately separated:
  EMPIRICAL: 9 V2V4Real validation sequences with public source-derived ROI counts.
  PAIRED: one source-verified sequence 0007 with 221 frames / 442 ROI decisions.
  PROJECTED: physical resource costs replayed using sequence 0007 with hypothetical
             hardware profiles, projected analytically to the 9-scene error curves.
  SYNTHETIC: V0.7 replaceable common-shock risk estimator, not empirical validation.
  DIAGNOSTIC: V0.7 queue-observability stress test under fixed synthetic timing.

This script neither trains a perception model nor implements a P1 optimizer.
Run: python scripts/run_cvc_figure_study.py --output samples/cvc_figure_study
"""
from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from octopus_sim.config import ExperimentConfig
from octopus_sim.coordination import (CandidatePlan, CoordinatorObservation,
    QueueSnapshot, ResourceSnapshot, StandardCandidateProvider, SyntheticRiskEstimator)
from octopus_sim.simulator import run_simulation

SOURCE_COUNTS = ROOT / 'empirical' / 'v2v4real' / 'sequence_counts.csv'
SOURCE_TRACE = ROOT / 'examples' / 'v2v4real_real_seq0007' / 'trace.jsonl'
SOURCE_CONFIG = ROOT / 'examples' / 'coordination' / 'v2v4real_candidate_inventory.json'
SOURCE_SYNTHETIC = ROOT / 'examples' / 'coordination' / 'three_agent_contextual_fusion.json'

B = '#255B77'  # local/separate
C = '#CB6E49'  # shared/fusion
G = '#34867A'  # individual error
DARK = '#203241'
MUTED = '#60717A'
LIGHT = '#E7EFF1'


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError('No data to export')
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_scenes(path: Path = SOURCE_COUNTS) -> list[dict[str, Any]]:
    result = []
    with path.open(encoding='utf-8', newline='') as f:
        for row in csv.DictReader(f):
            s = {'scene': row['scene']}
            for key in ('frames', 'balanced_events', 'balanced_positives',
                        'ego_errors', 'agent1_errors', 'late_fusion_errors',
                        'separate_joint_errors'):
                s[key] = int(row[key])
            if not (s['balanced_events'] == 2*s['frames'] == 2*s['balanced_positives']
                    and 0 <= s['separate_joint_errors'] <= min(s['ego_errors'], s['agent1_errors'])
                    and 0 <= s['late_fusion_errors'] <= s['balanced_events']):
                raise ValueError('Broken scene-level accounting: '+s['scene'])
            result.append(s)
    if not (len(result) == 9 and sum(s['frames'] for s in result) == 1993
            and sum(s['balanced_events'] for s in result) == 3986):
        raise ValueError('Expected 9 scenes, 1993 frames, and 3986 diagnostic decisions')
    return result


def aggregate(scenes: list[dict], indices: list[int] | None = None) -> dict[str, float]:
    subset = scenes if indices is None else [scenes[i] for i in indices]
    total = sum(s['balanced_events'] for s in subset)
    local_joint = sum(s['separate_joint_errors'] for s in subset) / total
    fused_joint = sum(s['late_fusion_errors'] for s in subset) / total
    local_marginal = sum(s['ego_errors'] + s['agent1_errors'] for s in subset) / (2 * total)
    fused_marginal = fused_joint   # one identical binary result to both agents
    return {'decisions':total, 'local_joint':local_joint, 'fused_joint':fused_joint,
            'local_marginal':local_marginal, 'fused_marginal':fused_marginal,
            'joint_increase':fused_joint-local_joint,
            'marginal_benefit':local_marginal-fused_marginal}


def bootstrap_aggregate(scenes: list[dict], repetitions: int = 20000,
                        seed: int = 20261119) -> dict[str, Any]:
    import numpy as np
    if repetitions < 1000 or len(scenes) != 9:
        raise ValueError('Use at least 1000 scene-cluster resamples over all nine scenes')
    rng = np.random.default_rng(seed)
    fields = ('balanced_events', 'separate_joint_errors', 'late_fusion_errors',
              'ego_errors', 'agent1_errors')
    ar = np.array([[s[key] for key in fields] for s in scenes],dtype=float)
    samples = rng.integers(0,len(scenes),size=(repetitions,len(scenes)))
    sums = ar[samples].sum(axis=1)
    n = sums[:,0]
    local_joint = sums[:,1]/n
    fusion_joint = sums[:,2]/n
    local_marg = (sums[:,3]+sums[:,4])/(2*n)
    fusion_marg = fusion_joint
    return {'local_joint':local_joint, 'fused_joint':fusion_joint,
            'local_marginal':local_marg, 'fused_marginal':fusion_marg,
            'joint_increase':fusion_joint-local_joint,
            'marginal_benefit':local_marg-fusion_marg,
            'repetitions':repetitions, 'seed':seed}


def ci(x) -> tuple[float,float]:
    import numpy as np
    return tuple(float(v) for v in np.quantile(x, [.025,.975]))


def from_trace(path: Path = SOURCE_TRACE):
    events = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if len(events) != 442 or len({r['frame_id'] for r in events}) != 221:
        raise ValueError('Unexpected source-verified sequence dimensions')
    for r in events:
        if not all(k in r['candidates'] for k in ('local:ego','local:agent1','edge:ego')):
            raise ValueError('Incomplete predictions')
    return events


def paired_failure_classes(events: list[dict]) -> dict[str,int]:
    classes = {'both_success':0, 'introduced_joint_failure':0,
               'rescued_joint_failure':0, 'both_fail':0}
    for r in events:
        t = r['truth'];p = r['candidates']
        local_joint = p['local:ego'] != t and p['local:agent1'] != t
        fused_joint = p['edge:ego'] != t
        if local_joint and fused_joint: classes['both_fail']+=1
        elif local_joint: classes['rescued_joint_failure']+=1
        elif fused_joint: classes['introduced_joint_failure']+=1
        else: classes['both_success']+=1
    if sum(classes.values()) != len(events): raise AssertionError('Lost paired outcomes')
    return classes


def _endpoint_run(config: ExperimentConfig, candidate: str, *,
                  arrival_ms: float=500, deadline_ms: float=400,
                  visibility: str='edge_only'):
    raw = deepcopy(config.data)
    raw['policy'] = {'name':'candidate_fixed','candidate_id':candidate}
    raw['scenario']['arrival_interval_ms'] = arrival_ms
    raw['scenario']['deadline_ms'] = deadline_ms
    raw['coordination']['resource_visibility'] = visibility
    c = ExperimentConfig.from_dict(raw, config.source_path)
    result = run_simulation(c, detailed_events=False)
    assert result.summary['physical_execution_batches'] == 221
    selected = [r for r in result.candidate_estimates if r['selected']]
    if len(selected) != 221:
        raise ValueError('Unexpected physical policy decisions')
    return result, selected


def resource_endpoints(config: ExperimentConfig) -> dict[str,dict]:
    metrics = {}
    for name in ('local_only','cooperative_fusion'):
        result, selected = _endpoint_run(config,name)
        summ = result.summary
        if (summ['complete_event_rate'] != 1 or summ['agent_deadline_miss_rate'] != 0
                or summ['episodes'] != 442):
            raise ValueError('The low-load projection requires complete on-time decisions')
        e = summ['energy_workload_incremental_j']
        cost = [r['cost'] for r in selected]
        pred_energy = sum(r['incremental_energy_j'] for r in cost)
        if not math.isclose(e,pred_energy,rel_tol=1e-9,abs_tol=1e-9):
            raise AssertionError('Predicted/realized low-load energy mismatch')
        pred_latency = sum(r['mean_delivery_latency_ms'] for r in cost)/len(cost)
        if not math.isclose(pred_latency,summ['mean_delivered_latency_ms'],abs_tol=1e-6):
            raise AssertionError('Predicted/realized low-load latency mismatch')
        metrics[name] = {'energy_per_frame_j':e/221,
            'energy_per_diagnostic_decision_j':e/442,
            'mean_agent_latency_ms':summ['mean_delivered_latency_ms'],
            'local_detector_executions':summ['local_inference_executions'],
            'fusion_executions':summ['edge_inference_executions'],
            'transmitted_bytes':summ['transmitted_bytes'],
            'energy_compute_j':summ['energy_compute_incremental_j'],
            'energy_radio_j':summ['energy_radio_incremental_j'],
            'energy_fusion_compute_j':summ['energy_compute_by_model_j'].get('late_fusion',0.0),
            'frames':221,'diagnostic_events':442,
            'device_profiles':'hypothetical',
        }
    if not (metrics['local_only']['local_detector_executions'] ==
            metrics['cooperative_fusion']['local_detector_executions'] == 442 and
            metrics['local_only']['fusion_executions'] == 0 and
            metrics['cooperative_fusion']['fusion_executions'] == 221):
        raise AssertionError('Frame-level physical work unexpectedly changed')
    return metrics


def risk_estimator_prediction(n: int, p: float, rho: float) -> float:
    """Run the real V0.7 constant SyntheticRiskEstimator, not a duplicated formula."""
    agents=tuple('agent_'+str(i) for i in range(n))
    snap=ResourceSnapshot(0,QueueSnapshot((0,)),
        tuple((a,QueueSnapshot((0,))) for a in agents),QueueSnapshot((0,)))
    observation=CoordinatorObservation('synthetic_risk',None,agents,10_000_000,0,
                                     (),(),snap,(),False,0.0)
    candidate=next(c for c in StandardCandidateProvider().generate(observation)
                   if c.candidate_id == 'local_only')
    model=SyntheticRiskEstimator({'kind':'constant','local_error':p,
        'edge_error':p,'fusion_error':p,'correlation':rho})
    return model.estimate(observation,candidate,failure_threshold=n).all_agent_error_probability


def queue_study(config: ExperimentConfig, intervals: tuple[int,...]=(5,10,15,20,25,30,50,100,500)):
    rows=[]
    for delta in intervals:
        runs={}
        for visibility in ('edge_only','reported_all'):
            result, selected = _endpoint_run(config,'cooperative_fusion',
                                             arrival_ms=delta,deadline_ms=20_000,
                                             visibility=visibility)
            s=result.summary
            if s['complete_event_rate'] != 1:
                raise ValueError('Forecast diagnostic must have uncensored 20-second deadlines')
            runs[visibility] = (s,selected)
            pred = sum(r['cost']['mean_delivery_latency_ms'] for r in selected)/len(selected)
            rows.append({'experiment':'uncensored_forecast',
               'frame_interval_ms':delta,'arrival_frames_per_sec':1000/delta,
               'resource_visibility':visibility,'predicted_mean_latency_ms':pred,
               'realized_mean_latency_ms':s['mean_delivered_latency_ms'],
               'realized_p95_latency_ms':s['p95_delivered_latency_ms'],
               'deadline_miss_rate':s['agent_deadline_miss_rate'],
               'mean_in_flight_workflows':sum(r['snapshot_in_flight_workflows'] for r in selected)/len(selected),
               'physical_frames':221,'energy_incremental_j':s['energy_workload_incremental_j']})
        if not math.isclose(runs['edge_only'][0]['mean_delivered_latency_ms'],
                            runs['reported_all'][0]['mean_delivered_latency_ms'],abs_tol=1e-7):
            raise AssertionError('Telemetry visibility changed execution')
        # Deadline-inclusive operational failure evaluated independently from
        # uncensored forecasting, with the SAME candidate and world predictions.
        missed = {}
        for choice in ('local_only','cooperative_fusion'):
            rr,_=_endpoint_run(config,choice,arrival_ms=delta,deadline_ms=400)
            ss=rr.summary
            missed[choice]=ss['agent_deadline_miss_rate']
            rows.append({'experiment':'operational_deadline',
                'frame_interval_ms':delta,'arrival_frames_per_sec':1000/delta,
                'resource_visibility':choice,'predicted_mean_latency_ms':'',
                'realized_mean_latency_ms':ss['mean_delivered_latency_ms'],
                'realized_p95_latency_ms':ss['p95_delivered_latency_ms'],
                'deadline_miss_rate':ss['agent_deadline_miss_rate'],
                'mean_in_flight_workflows':'', 'physical_frames':221,
                'energy_incremental_j':ss['energy_workload_incremental_j']})
    return rows


def candidate_catalog(config: ExperimentConfig) -> list[dict]:
    raw=deepcopy(config.data)
    raw['scenario']['episodes']=1
    raw['policy']={'name':'candidate_fixed','candidate_id':'local_only'}
    raw['coordination']['risk_estimator'] = {'kind':'constant',
        'local_error':0.10,'edge_error':0.06,'fusion_error':0.04,
        'correlation':0.18}
    raw['coordination']['candidate_ids'] = []
    r=run_simulation(ExperimentConfig.from_dict(raw,config.source_path),False)
    costs=r.candidate_estimates
    if len(costs) < 6: raise ValueError('Candidate inventory unexpectedly small')
    rows=[]
    for rec in costs:
        rows.append({'candidate_id':rec['candidate_id'],
            'estimated_all_agent_error_probability':rec['risk']['all_agent_error_probability'],
            'estimated_average_marginal_error_probability':sum(x[1] for x in rec['risk']['marginal_by_agent'])/3,
            'estimated_incremental_energy_j':rec['cost']['incremental_energy_j'],
            'estimated_mean_agent_latency_ms':rec['cost']['mean_delivery_latency_ms'],
            'estimated_bytes':rec['cost']['transfer_bytes'],
            'selected':rec['selected'],
            'assumptions':'synthetic model prior, not validated true risk'})
    return rows


def setup_plots():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
        'axes.titlesize':12,'axes.titleweight':'semibold','axes.labelsize':10,
        'figure.titlesize':14,'figure.titleweight':'bold',
        'axes.spines.top':False,'axes.spines.right':False,
        'savefig.facecolor':'white','pdf.fonttype':42,
        'lines.linewidth':2.0})
    return plt


def savefig(fig, path: Path):
    fig.savefig(path.with_suffix('.png'),dpi=220,bbox_inches='tight',facecolor='white')
    fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight',facecolor='white')
    import matplotlib.pyplot as plt
    plt.close(fig)


def plot_empirical_quadrant(plt, scenes, agg, boot, out):
    fig,ax=plt.subplots(figsize=(8.6,5.8),layout='constrained')
    for s in scenes:
        a=aggregate([s]);x=100*a['marginal_benefit'];y=100*a['joint_increase']
        ax.scatter(x,y,s=s['frames']*.55,color=B,alpha=.72,zorder=3,
                   edgecolor='white',linewidth=.7)
        offsets={'0004':(-10,9),'0005':(-15,-17),'0006':(7,10),'0007':(7,9),
                 '0008':(7,-15),'0000':(7,9),'0001':(7,-17),'0002':(7,9),'0003':(7,9)}
        dx,dy=offsets[s['scene']]
        ax.annotate(s['scene'],(x,y),xytext=(dx,dy),textcoords='offset points',
                    fontsize=9,color=DARK)
    x=100*agg['marginal_benefit'];y=100*agg['joint_increase']
    xe=ci(boot['marginal_benefit']);ye=ci(boot['joint_increase'])
    ax.errorbar(x,y,xerr=[[x-100*xe[0]],[100*xe[1]-x]],
                yerr=[[y-100*ye[0]],[100*ye[1]-y]],
                fmt='*',markersize=17,color=C,ecolor=C,capsize=5,elinewidth=1.7,
                zorder=8,label='9-scene weighted aggregate\n(95% scene bootstrap)')
    ax.axhline(0,lw=1,color='#88979C',ls='--')
    ax.axvline(0,lw=1,color='#88979C',ls='--')
    ax.grid(alpha=.13)
    ax.set(xlabel='Reduction in average individual error (percentage points)  →',
           ylabel='Increase in simultaneous-error risk (percentage points)  →',
           title='Accuracy gains do not guarantee collective reliability')
    ax.set_xlim(-.5,27)
    ax.set_ylim(-.6,6.35)
    ax.legend(loc='upper right',frameon=False,fontsize=9)
    fig.text(.5,-.025,'Recorded V2V4Real-derived predictions · 9 scenes · 1,993 frames / 3,986 ROI decisions',
             ha='center',fontsize=9,color=MUTED)
    savefig(fig,out/'01_empirical_accuracy_vs_joint_risk')


def plot_scene_heterogeneity(plt, scenes, agg, boot, out):
    ss=sorted(scenes,key=lambda s: aggregate([s])['joint_increase'],reverse=True)
    fig,ax=plt.subplots(figsize=(9.3,5.95),layout='constrained')
    for i,s in enumerate(ss):
        a=aggregate([s]);l=100*a['local_joint'];f=100*a['fused_joint']
        ax.plot([l,f],[i,i],color='#A6B4BA',lw=2,zorder=1)
        ax.scatter([l],[i],marker='o',color=B,s=65,zorder=3)
        ax.scatter([f],[i],marker='D',color=C,s=47,zorder=4)
        ax.text(9.6,i,f'{s["frames"]} frames',va='center',fontsize=8.6,color=MUTED)
    y=len(ss)+.55
    a=agg
    ax.plot([a['local_joint']*100,a['fused_joint']*100],[y,y],color='#647785',lw=2.8,zorder=2)
    ax.scatter([a['local_joint']*100],[y],s=90,color=B,zorder=4)
    ax.scatter([a['fused_joint']*100],[y],marker='D',s=90,color=C,zorder=4)
    ci_delta=ci(boot['joint_increase'])
    # Aggregate scene confidence interval lives in the footer rather than on the row.
    ax.set_yticks(list(range(len(ss)))+[y])
    ax.set_yticklabels([s['scene'] for s in ss]+['ALL 9'])
    ax.set_xlim(-.35,12)
    ax.set_ylim(y+1,-.75)
    ax.set_xlabel('Both agents wrong on a common ROI (%)')
    ax.set_title('Scene-to-scene variation in the cost of shared prediction errors')
    ax.grid(axis='x',alpha=.2)
    ax.scatter([],[],marker='o',color=B,label='Separate local predictions')
    ax.scatter([],[],marker='D',color=C,label='One shared late-fusion result')
    ax.legend(loc='upper center',bbox_to_anchor=(.55,1.03),frameon=False,
              fontsize=8.7,ncol=2)
    fig.text(.5,-.02,
             f'Weighted nine-scene Δ = +{100*a["joint_increase"]:.2f} pp '
             f'(95% scene CI: +{100*ci_delta[0]:.2f} to +{100*ci_delta[1]:.2f} pp) · '
             'Per-event variance unavailable for 8 scenes',
             ha='center',fontsize=8.5,color=MUTED)
    savefig(fig,out/'02_nine_scene_heterogeneity')


def mixture_quantities(agg, boot, k):
    import numpy as np
    local_joint=agg['local_joint'];fu_joint=agg['fused_joint']
    local_marg=agg['local_marginal'];fu_marg=agg['fused_marginal']
    qj=(1-k)*local_joint+k*fu_joint
    qm=(1-k)*local_marg+k*fu_marg
    bj=(1-k)*boot['local_joint'][:,None]+k*boot['fused_joint'][:,None]
    bm=(1-k)*boot['local_marginal'][:,None]+k*boot['fused_marginal'][:,None]
    return qj,qm,np.quantile(bj,[.025,.975],axis=0),np.quantile(bm,[.025,.975],axis=0)


def plot_projected_tradeoff(plt, agg, boot, costs, out):
    import numpy as np
    k=np.linspace(0,1,101)
    local=costs['local_only'];fusion=costs['cooperative_fusion']
    energy=(1-k)*local['energy_per_frame_j']+k*fusion['energy_per_frame_j']
    latency=(1-k)*local['mean_agent_latency_ms']+k*fusion['mean_agent_latency_ms']
    qj,qm,bj,bm=mixture_quantities(agg,boot,k)
    fig,axes=plt.subplots(1,2,figsize=(13,5.4),layout='constrained')
    for ax,x,key in ((axes[0],energy,'energy'),(axes[1],latency,'latency')):
        ax.plot(x,100*qm,color=G,lw=2.5,label='Average individual error')
        ax.fill_between(x,100*bm[0],100*bm[1],color=G,alpha=.13,linewidth=0)
        ax.plot(x,100*qj,color=C,lw=2.5,label='Both agents wrong')
        ax.fill_between(x,100*bj[0],100*bj[1],color=C,alpha=.14,linewidth=0)
        for p in [0,.25,.5,.75,1.]:
            i=round(p*100)
            ax.plot(x[i],100*qj[i],marker='o',color=C,markersize=5)
            if p in (0,.5,1):
                ax.annotate(f'{int(p*100)}% shared',(x[i],100*qj[i]),
                            xytext=(6,-14 if p==1 else 7),textcoords='offset points',
                            fontsize=8,color=C)
        ax.grid(alpha=.18)
        ax.set_ylabel('Diagnostic event error (%)')
        ax.set_ylim(bottom=0)
        if key == 'energy':
            ax.set(xlabel='Modeled incremental energy per physical frame (J)',
                   title='Reliability versus energy')
        else:
            ax.set(xlabel='Modeled mean delivery latency per agent (ms)',
                   title='Reliability versus latency')
    axes[0].legend(loc='upper right',frameon=False,fontsize=9)
    fig.suptitle('Nine-scene empirical outcomes × one illustrative frame-level resource configuration',fontsize=13)
    fig.text(.5,-.025,'Shading: 95% 9-scene cluster bootstrap for errors · Cost: fixed low-load V0.7 model projected from sequence 0007; not measured',
             ha='center',color=MUTED,fontsize=8.5)
    savefig(fig,out/'03_empirical_risk_and_modeled_cost')



def modeled_energy_sensitivity(costs: dict, k: float) -> dict:
    """Linearly reweight the hypothetical *per-bit-radio* and fusion-W profiles.

    Unlike a confidence interval, this is a bounded assumption grid. The
    executor uses fixed durations and packet loss 0 in this experiment.
    """
    if not 0 <= k <= 1: raise ValueError('Mixture probability must be in [0,1]')
    local=costs['local_only'];fusion=costs['cooperative_fusion']
    n=local['frames']
    if fusion['frames'] != n or not math.isclose(local['energy_radio_j'],0,abs_tol=1e-12):
        raise ValueError('Energy reweighting is only valid for this low-load per-bit configuration')
    fusion_model=fusion['energy_fusion_compute_j']
    fusion_radio=fusion['energy_radio_j']
    other_fusion=fusion['energy_compute_j']-fusion_model
    other_local=local['energy_compute_j']
    if not math.isclose(other_local,other_fusion,abs_tol=1e-7):
        raise ValueError('Unexpected local compute difference; cannot apply fixed-workflow scaling')
    values=[]
    for rf in (.25,1.,4.):
        for ff in (.5,1.,2.):
            amount=((1-k)*other_local+k*(other_fusion+rf*fusion_radio+ff*fusion_model))/n
            values.append({'fusion_probability':k,'radio_multiplier':rf,'fusion_power_multiplier':ff,
                           'incremental_energy_per_frame_j':amount})
    nominal=((1-k)*local['energy_per_frame_j']+k*fusion['energy_per_frame_j'])
    mini=min(r['incremental_energy_per_frame_j'] for r in values)
    maxi=max(r['incremental_energy_per_frame_j'] for r in values)
    if not mini-1e-12 <= nominal <= maxi+1e-12:
        raise AssertionError('Nominal energy outside perturbation envelope')
    return {'fusion_probability':k,'baseline_local_compute_j_per_frame':other_local/n,
            'nominal_energy_per_frame_j':nominal,'min_energy_per_frame_j':mini,
            'max_energy_per_frame_j':maxi,'fusion_compute_j_per_frame':fusion_model*k/n,
            'radio_j_per_frame':fusion_radio*k/n,'grid':values}


def plot_energy_sensitivity(plt,agg,boot,costs,out):
    import numpy as np
    ks=np.array([0,.25,.5,.75,1.])
    sens=[modeled_energy_sensitivity(costs,float(k)) for k in ks]
    joint=np.array([(1-k)*agg['local_joint']+k*agg['fused_joint'] for k in ks])*100
    lo=np.array([ci((1-k)*boot['local_joint']+k*boot['fused_joint'])[0] for k in ks])*100
    hi=np.array([ci((1-k)*boot['local_joint']+k*boot['fused_joint'])[1] for k in ks])*100
    energy=np.array([r['nominal_energy_per_frame_j'] for r in sens])
    e_min=np.array([r['min_energy_per_frame_j'] for r in sens])
    e_max=np.array([r['max_energy_per_frame_j'] for r in sens])
    fig,axes=plt.subplots(1,2,figsize=(12.5,4.9),layout='constrained')
    axes[0].errorbar(energy,joint,xerr=[energy-e_min,e_max-energy],
                     yerr=[joint-lo,hi-joint],fmt='o-',color=C,ecolor='#B58C7A',
                     capsize=3,elinewidth=1.4,markersize=6)
    for k,x,y in zip(ks,energy,joint):
        axes[0].annotate(f'{int(k*100)}%',(x,y),xytext=(5,8),
                         textcoords='offset points',fontsize=8.3,color=DARK)
    axes[0].set(xlabel='Modeled incremental energy per physical frame (J)',
                ylabel='Both-agent simultaneous-error rate (%)',
                title='Cost ranges are assumptions, not energy confidence intervals')
    axes[0].grid(alpha=.18);axes[0].set_ylim(bottom=0)
    w=.58
    x=np.arange(len(ks))
    b0=np.array([v['baseline_local_compute_j_per_frame'] for v in sens])*1000
    b1=np.array([v['fusion_compute_j_per_frame'] for v in sens])*1000
    b2=np.array([v['radio_j_per_frame'] for v in sens])*1000
    axes[1].bar(x,b0,width=w,color=B,label='Vehicle compute (unchanged)')
    axes[1].bar(x,b1,bottom=b0,width=w,color='#B7923D',label='Extra fusion compute')
    axes[1].bar(x,b2,bottom=b0+b1,width=w,color=G,label='Added radio TX/RX')
    axes[1].set_xticks(x,[f'{int(k*100)}%' for k in ks])
    axes[1].set(xlabel='Probability of frame-level cooperative fusion',
                ylabel='Nominal incremental energy (mJ / physical frame)',
                title='Why cooperative late fusion costs more here')
    axes[1].legend(frameon=False,fontsize=8,loc='upper left')
    axes[1].grid(axis='y',alpha=.18)
    fig.suptitle('Energy-model sensitivity for the nine-scene error projection',fontsize=13)
    fig.text(.5,-.03,'Horizontal spans: radio 0.25–4× and fusion power 0.5–2× · Vertical spans: 95% scene bootstrap · No hardware measurements',
             ha='center',fontsize=8.5,color=MUTED)
    savefig(fig,out/'08_energy_profile_sensitivity')
    return sens


def plot_paired_failures(plt, events, out):
    import numpy as np
    classes=paired_failure_classes(events)
    mat=np.array([[classes['both_success'],classes['introduced_joint_failure']],
                  [classes['rescued_joint_failure'],classes['both_fail']]])
    detail={'both_local_correct_fusion_wrong':0,
            'exactly_one_local_wrong_fusion_wrong':0,
            'both_local_wrong_fusion_wrong':0}
    for r in events:
        t=r['truth'];p=r['candidates'];
        if p['edge:ego']==t:continue
        wrong=int(p['local:ego']!=t)+int(p['local:agent1']!=t)
        key=('both_local_correct_fusion_wrong' if wrong==0 else
             'exactly_one_local_wrong_fusion_wrong' if wrong==1 else
             'both_local_wrong_fusion_wrong')
        detail[key]+=1
    fig,axes=plt.subplots(1,2,figsize=(11.6,4.8),layout='constrained')
    ax=axes[0]
    # Four mutually exclusive paired-outcome classes.
    from matplotlib.colors import ListedColormap
    ax.imshow([[0,1],[2,3]],cmap=ListedColormap(['#E5F0ED','#F7DFD5','#D7E8F0','#F1C2AF']),aspect='auto',vmin=0,vmax=3)
    ax.set_xticks([0,1],['Fusion correct','Fusion wrong'])
    ax.set_yticks([0,1],['Local pair: not both wrong','Local pair: both wrong'])
    ax.tick_params(length=0)
    for i in range(2):
        for j in range(2):
            ax.text(j,i,f'{mat[i,j]}',ha='center',va='center',fontsize=23,
                    fontweight='semibold',color=DARK)
    ax.set_title('Paired joint-error outcomes (442 ROIs)')
    ax=axes[1]
    labels=['Both local\ncorrect','One local\nwrong','Both local\nwrong']
    counts=[detail['both_local_correct_fusion_wrong'],
            detail['exactly_one_local_wrong_fusion_wrong'],
            detail['both_local_wrong_fusion_wrong']]
    bars=ax.bar(labels,counts,color=['#5F96A5','#D5A178',C],width=.62)
    ax.bar_label(bars,padding=3,fontsize=11)
    ax.set_ylim(0,max(counts)+3.5)
    ax.set_ylabel('Shared-fusion wrong decisions')
    ax.set_title('When a shared output is wrong')
    ax.grid(axis='y',alpha=.17)
    fig.suptitle('Sequence 0007: what changes when both agents reuse one fused output',fontsize=13)
    fig.text(.5,-.035,'Only one recorded sequence · matched outcomes, not an intervention or physical collision analysis',
             ha='center',color=MUTED,fontsize=9)
    savefig(fig,out/'04_paired_error_mechanism_0007')
    return detail


def plot_correlation_scaling(plt,out):
    import numpy as np
    rho=np.linspace(0,1,101)
    p=0.1
    fig,axes=plt.subplots(1,2,figsize=(12.1,4.9),layout='constrained')
    records=[]
    for n,col in ((2,B),(3,G),(5,C)):
        actual=np.array([risk_estimator_prediction(n,p,float(r)) for r in rho])
        expected=(1-rho)*p**n + rho*p
        if not np.allclose(actual,expected,atol=1e-12):
            raise AssertionError('Synthetic risk estimator violates common-shock model')
        axes[0].plot(rho,100*actual,color=col,label=f'{n} agents',lw=2.3)
        axes[1].plot(rho,actual/p**n,color=col,label=f'{n} agents',lw=2.3)
        for r in (0,.1,.25,.5,1.):
            value=risk_estimator_prediction(n,p,r)
            records.append({'agents':n,'marginal_error_probability':p,
                'common_shock_probability':r,'all_agent_error_probability':value,
                'independence_product':p**n,
                'risk_amplification_over_independence':value/p**n})
    axes[0].set(ylabel='All-agent simultaneous-error probability (%)',
                xlabel='Common-shock probability (ρ)',title='Collective failure, fixed 10% marginal error')
    axes[1].set(ylabel='True joint risk / independence estimate',
                xlabel='Common-shock probability (ρ)',title='Cost of assuming independence')
    axes[1].set_yscale('log')
    for ax in axes:ax.grid(alpha=.17);ax.legend(frameon=False,fontsize=9,loc='lower right')
    fig.suptitle('Synthetic common-shock dependence · analytical V0.7 estimator',fontsize=13)
    fig.text(.5,-.03,'Independent inference identities can still fail together; this model is illustrative and is not fitted to V2V4Real',
             color=MUTED,ha='center',fontsize=8.5)
    savefig(fig,out/'05_synthetic_team_size_vs_correlation')
    return records


def plot_queue_visibility(plt,rows,out):
    fig,ax=plt.subplots(1,2,figsize=(12.5,5.1),layout='constrained')
    forecasts=[r for r in rows if r['experiment']=='uncensored_forecast']
    for vis,col,label,ls in (('edge_only',C,'Edge-only: predicted','--'),
                             ('reported_all',G,'Fresh local telemetry: predicted',':')):
        rr=sorted((r for r in forecasts if r['resource_visibility']==vis),
                  key=lambda r:r['arrival_frames_per_sec'])
        x=[r['arrival_frames_per_sec'] for r in rr]
        ax[0].plot(x,[r['predicted_mean_latency_ms'] for r in rr],
                   color=col,marker='o',ms=4,ls=ls,label=label)
        if vis=='reported_all':
            ax[0].plot(x,[r['realized_mean_latency_ms'] for r in rr],
                       color=DARK,marker='s',markersize=4,alpha=.75,
                       label='Executed: both visibility settings')
    ax[0].set(xlabel='Offered physical frames / second',
              ylabel='Mean end-to-end latency (ms)',title='Prediction error when local queues are hidden')
    ax[0].grid(alpha=.19)
    ax[0].legend(loc='upper left',frameon=False,fontsize=8.2)
    deadlines=[r for r in rows if r['experiment']=='operational_deadline']
    for selection,col,name in (('local_only',B,'Local only'),('cooperative_fusion',C,'Cooperative fusion')):
        rr=sorted((r for r in deadlines if r['resource_visibility']==selection),
                  key=lambda r:r['arrival_frames_per_sec'])
        ax[1].plot([r['arrival_frames_per_sec'] for r in rr],
                   [100*r['deadline_miss_rate'] for r in rr],
                   marker='o',markersize=4,color=col,label=name)
    ax[1].set(xlabel='Offered physical frames / second',
              ylabel='Agent deadline-miss rate (%)',title='Operational failures with 400 ms deadline')
    ax[1].grid(alpha=.19);ax[1].legend(frameon=False,fontsize=9,loc='upper left')
    fig.suptitle('Model diagnostic: 221 frames, fixed 20 ms local service, hypothetical link and power',fontsize=13)
    fig.text(.5,-.03,'Left: 20-second uncensored deadline for comparable forecasts · Right: 400 ms deadline; conditional delivered latency omitted',
             ha='center',color=MUTED,fontsize=8.5)
    savefig(fig,out/'06_queue_visibility_and_deadlines')


def plot_candidate_catalog(plt,rows,out):
    import numpy as np
    fig,ax=plt.subplots(figsize=(8.6,5.7),layout='constrained')
    energy=np.array([r['estimated_incremental_energy_j'] for r in rows])
    risks=100*np.array([r['estimated_all_agent_error_probability'] for r in rows])
    latency=np.array([r['estimated_mean_agent_latency_ms'] for r in rows])
    # Collapse numerically identical forecasts for the three shared-edge
    # source choices. All candidates remain separate in the exported CSV.
    grouped = {}
    for row,e,q,lat in zip(rows,energy,risks,latency):
        key=(round(float(e),12),round(float(q),12),round(float(lat),12))
        grouped.setdefault(key, []).append(row['candidate_id'])
    lookup={'Separate local':B,'Independent edge':G,'Shared edge':C,
            'Partial sharing':'#9972A8','Late fusion':'#B7923D'}
    for (e,q,lat), names in grouped.items():
        name=names[0]
        if name=='local_only':cat='Separate local';label='local only'
        elif name=='independent_edge':cat='Independent edge';label='independent edge'
        elif name.startswith('shared_edge:'):
            cat='Shared edge';label=f'shared edge ({len(names)} sources)'
        elif name=='partial_share':cat='Partial sharing';label='partial share'
        else:cat='Late fusion';label='late fusion'
        ax.scatter(e,q,s=45+lat*2,facecolor=lookup[cat],edgecolor='white',lw=1,zorder=3)
        ax.annotate(label,(e,q),xytext=(6,7),textcoords='offset points',fontsize=8.2,color=DARK)
    ax.set(xlabel='Estimated incremental energy for one frame (J)',
           ylabel='Estimated probability all 3 agents are wrong (%)',
           title='A bounded plan catalog exposes competing systems choices')
    ax.grid(alpha=.16)
    ax.set_ylim(bottom=0)
    for cat,col in lookup.items():ax.scatter([],[],s=65,color=col,label=cat)
    ax.legend(frameon=False,fontsize=8,loc='upper right')
    fig.text(.5,-.02,'Point area scales with estimated mean latency; synthetic risk priors and hypothetical energy only — NOT evaluated policy quality',
             ha='center',fontsize=8.5,color=MUTED)
    savefig(fig,out/'07_v07_candidate_plan_landscape')


def main(argv: list[str] | None=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=ROOT/'samples'/'cvc_figure_study')
    ap.add_argument('--bootstrap-repetitions',type=int,default=20000)
    ap.add_argument('--bootstrap-seed',type=int,default=20261119)
    args=ap.parse_args(argv)
    out=args.output.resolve();out.mkdir(exist_ok=True,parents=True)
    plt=setup_plots()
    scenes=load_scenes();agg=aggregate(scenes)
    boot=bootstrap_aggregate(scenes,args.bootstrap_repetitions,args.bootstrap_seed)
    conf=ExperimentConfig.load(SOURCE_CONFIG)
    costs=resource_endpoints(conf)
    events=from_trace()
    paired=paired_failure_classes(events)
    if (paired['both_fail']+paired['rescued_joint_failure'] != 4 or
            paired['both_fail']+paired['introduced_joint_failure'] != 16):
        raise AssertionError('Real 0007 paired failure totals changed')
    plot_empirical_quadrant(plt,scenes,agg,boot,out)
    plot_scene_heterogeneity(plt,scenes,agg,boot,out)
    plot_projected_tradeoff(plt,agg,boot,costs,out)
    sensitivity=plot_energy_sensitivity(plt,agg,boot,costs,out)
    detail=plot_paired_failures(plt,events,out)
    synthetic=plot_correlation_scaling(plt,out)
    queue=queue_study(conf)
    plot_queue_visibility(plt,queue,out)
    candidates=candidate_catalog(ExperimentConfig.load(SOURCE_SYNTHETIC))
    plot_candidate_catalog(plt,candidates,out)
    write_csv(out/'nine_scene_metrics.csv',[{
        'scene':s['scene'],'frames':s['frames'],'decisions':s['balanced_events'],
        **aggregate([s])} for s in scenes])
    write_csv(out/'queue_prediction_and_deadline_sweep.csv',queue)
    write_csv(out/'synthetic_correlation_grid.csv',synthetic)
    write_csv(out/'candidate_estimate_catalog.csv',candidates)
    write_csv(out/'nine_scene_energy_assumption_sweep.csv',
              [v for record in sensitivity for v in record['grid']])
    k_values=(0,.25,.5,.75,1.)
    mixtures=[]
    for k in k_values:
        joint=(1-k)*agg['local_joint']+k*agg['fused_joint']
        marg=(1-k)*agg['local_marginal']+k*agg['fused_marginal']
        mixtures.append({'fusion_probability':k,
            'joint_error_rate':joint,'marginal_error_rate':marg,
            'joint_error_bootstrap_low':ci((1-k)*boot['local_joint']+k*boot['fused_joint'])[0],
            'joint_error_bootstrap_high':ci((1-k)*boot['local_joint']+k*boot['fused_joint'])[1],
            'marginal_error_bootstrap_low':ci((1-k)*boot['local_marginal']+k*boot['fused_marginal'])[0],
            'marginal_error_bootstrap_high':ci((1-k)*boot['local_marginal']+k*boot['fused_marginal'])[1],
            'projected_modeled_energy_per_frame_j':((1-k)*costs['local_only']['energy_per_frame_j']
                                                 +k*costs['cooperative_fusion']['energy_per_frame_j']),
            'projected_modeled_mean_delivery_latency_ms':(
                (1-k)*costs['local_only']['mean_agent_latency_ms']+k*costs['cooperative_fusion']['mean_agent_latency_ms']),
            'notes':'Expected frame-randomized mixture; errors from 9 scenes; cost from 0007 configuration'})
    write_csv(out/'nine_scene_counterfactual_mixture.csv',mixtures)
    report={
        'study':'OCTOPUS-Sim v0.7 CVC exploratory figures',
        'research_status':'exploratory, no P1 optimized policy, not official detection mAP',
        'source_revision':'DMSTrack commit d3b9949499c8e68ea33060873bd1cb95b6d4d323',
        'input_sha256':{str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest()
                        for p in (SOURCE_COUNTS,SOURCE_TRACE,SOURCE_CONFIG,SOURCE_SYNTHETIC)},
        'empirical_scope':'9 V2V4Real validation sequences; 1993 frames; 3986 two-ROI binary diagnostic decisions',
        'one_sequence_scope':'0007: 221 frames, 442 paired decision events; actual frame-level simulator replay',
        'empirical_aggregate':agg,
        'paired_outcomes_0007':paired,
        'fusion_wrong_breakdown_0007':detail,
        'scene_bootstrap':{'unit':'driving sequence (n=9), equal-probability cluster resampling',
            'repetitions':args.bootstrap_repetitions,'seed':args.bootstrap_seed,
            'joint_increase_95ci':ci(boot['joint_increase']),
            'marginal_benefit_95ci':ci(boot['marginal_benefit'])},
        'resource_endpoints_0007':costs,
        'energy_sensitivity':'radio 0.25x/1x/4x; edge fusion power 0.5x/1x/2x; not statistical CI',
        'resource_projection':'fixed no-contention per-frame costs from sequence 0007 reused analytically over 9-scene error counts; NOT 9-scene runtime simulation',
        'risk_estimation':'synthetic V0.7 common-shock model only; no empirically estimated scene features',
        'queue_diagnostic':'actual V0.7 fixed-candidate replay; assumed service/network and fresh local telemetry if enabled',
        'limits':[
            'ROI task and negative sampling are ground-truth-informed diagnostics, not causal scene-level detection benchmarks',
            'Nine scenes may not be statistically independent deployment environments',
            'Bootstrap captures between-sequence variation, not calibrated prediction uncertainty',
            'The shared late-fusion output is a different pipeline from separate local detection; resource differences are part of the comparison',
            'Energy and latency rely on hypothetical device profiles; no hardware readings or network traces',
            'Mixture curves are expected rates under randomized frame selection, not observed results of a learned coordinator',
            'Under load, predicted latency depends on coordinator-visible telemetry; neither forecast is an oracle',
            'No physical crash probabilities, production safety claims or P1 optimizer performance are presented',
        ],
        'figures':['01_empirical_accuracy_vs_joint_risk','02_nine_scene_heterogeneity',
                   '03_empirical_risk_and_modeled_cost','04_paired_error_mechanism_0007',
                   '05_synthetic_team_size_vs_correlation','06_queue_visibility_and_deadlines',
                   '07_v07_candidate_plan_landscape',
                   '08_energy_profile_sensitivity'],
    }
    (out/'STUDY_REPORT.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'joint_separate':agg['local_joint'],'joint_fusion':agg['fused_joint'],
        'avg_marginal_separate':agg['local_marginal'],'avg_marginal_fusion':agg['fused_marginal'],
        'delta_joint_ci':report['scene_bootstrap']['joint_increase_95ci'],
        'delta_marginal_ci':report['scene_bootstrap']['marginal_benefit_95ci'],
        'seq0007_paired':paired,'energy_j_per_frame':{
        k:v['energy_per_frame_j'] for k,v in costs.items()},
        'figures':len(report['figures']),'out':str(out)},indent=2))


if __name__ == '__main__':
    main()
