#!/usr/bin/env python3
"""Generate defensible exploratory plots from audited public V2V4Real-derived counts.

Source: eddyhkchiu/DMSTrack @ d3b9949499c8e68ea33060873bd1cb95b6d4d323
The producer prepare_from_public_repo.py can regenerate counts from original text.
No neural networks run here; resource cost is NOT measured.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
F=ROOT/'figures';F.mkdir(exist_ok=True)
s=pd.read_csv(ROOT/'sequence_counts.csv', dtype={'scene':str}); n=int(s.balanced_events.sum())
assert len(s)==9 and n==3986 and s.balanced_positives.sum()*2==n
baseline=s.separate_joint_errors.sum()/n
shared=s.late_fusion_errors.sum()/n
marginal={k:float(s[k].sum()/n) for k in ['ego_errors','agent1_errors','cobevt_errors','late_fusion_errors','separate_joint_errors']}
assert (s.late_fusion_errors>=s.separate_joint_errors).all()

rng=np.random.default_rng(20261118)
arrays=s[['balanced_events','ego_errors','agent1_errors','late_fusion_errors','separate_joint_errors']].to_numpy(dtype=float)
indices=rng.integers(0,len(s),size=(40000,len(s)))
boot=arrays[indices].sum(axis=1)
be=boot[:,1:]/boot[:,[0]]
# columns: ego, vehicle1, fused-late, separate-joint
kgrid=np.linspace(0,1,101)
line=(1-kgrid)*baseline+kgrid*shared
boot_line=(1-kgrid[None,:])*be[:,3,None]+kgrid[None,:]*be[:,2,None]
lo,hi=np.quantile(boot_line,[.025,.975],axis=0)

def save(name):
    plt.tight_layout()
    plt.savefig(F/name,dpi=210,bbox_inches='tight',facecolor='white')
    plt.savefig(F/name.replace('.png','.pdf'),bbox_inches='tight',facecolor='white')
    plt.close()

plt.figure(figsize=(9.5,5.5))
plt.plot(kgrid,100*line,lw=2.8,label='Both agents wrong: measured prediction counterfactual',color='#B4473C')
plt.fill_between(kgrid,100*lo,100*hi,color='#B4473C',alpha=.15,label='95% scene-cluster bootstrap interval')
plt.plot(kgrid,100*((1-kgrid)*marginal['ego_errors']+kgrid*shared),lw=1.9,ls='--',color='#2B6CB0',label='Vehicle A marginal error')
plt.plot(kgrid,100*((1-kgrid)*marginal['agent1_errors']+kgrid*shared),lw=1.9,ls=':',color='#528653',label='Vehicle B marginal error')
plt.scatter([0,1],[100*baseline,100*shared],color='#B4473C',s=40,zorder=5)
plt.xlabel('Probability of replacing separate predictions with one shared late-fusion result (κ)')
plt.ylabel('Error probability (%)')
plt.title('V2V4Real: sharing improves each agent, but increases joint error')
plt.legend(fontsize=8,loc='upper right');plt.grid(alpha=.18)
plt.text(.02,.05,'9 sequences · 1,993 frames · 3,986 balanced diagnostic events\nRecorded predictions; sharing is a counterfactual policy',transform=plt.gca().transAxes,fontsize=8,color='#666')
save('01_empirical_reuse_curve.png')

fig,ax=plt.subplots(figsize=(9.4,5.4))
x=np.arange(len(s));N=s.balanced_events.to_numpy()
ax.bar(x-.18,100*s.separate_joint_errors/N,width=.36,label='Separate vehicle predictions: both wrong',color='#326A96')
ax.bar(x+.18,100*s.late_fusion_errors/N,width=.36,label='Shared late fusion: both wrong',color='#D17A48')
ax.set_xticks(x);ax.set_xticklabels(s.scene);ax.set_ylabel('Joint-error rate (%)');ax.set_xlabel('V2V4Real validation sequence')
ax.set_title('Per-sequence joint failures (8 of 9 sequences show an increase)');ax.legend(fontsize=9)
ax.grid(axis='y',alpha=.2)
save('02_per_sequence_joint_error.png')

fig,ax=plt.subplots(figsize=(8.8,5.0))
labels=['Vehicle A\nseparate','Vehicle B\nseparate','CoBEVT\ncooperative','Late fusion\n(shared)','Both wrong\n(separate)','Both wrong\n(shared)']
values=[marginal['ego_errors'],marginal['agent1_errors'],marginal['cobevt_errors'],shared,baseline,shared]
colors=['#4378A4','#78A3BF','#A1A8AC','#D17A48','#4378A4','#D17A48']
ax.bar(range(6),np.array(values)*100,color=colors,width=.68)
ax.set_xticks(range(6));ax.set_xticklabels(labels,fontsize=9)
ax.set_ylim(0,21);ax.set_ylabel('Event error probability (%)');ax.set_title('Marginal accuracy does not determine collective reliability')
for i,v in enumerate(values):ax.text(i,v*100+.3,f'{100*v:.2f}%',ha='center',fontsize=9)
ax.grid(axis='y',alpha=.2)
save('03_accuracy_vs_joint_risk.png')

fg=s.fixed_events.sum();grid_q=s.fixed_separate_joint_errors.sum()/fg;grid_f=s.fixed_late_fusion_errors.sum()/fg
fig,ax=plt.subplots(figsize=(8.4,5.2))
categories=['Balanced near-object tasks\n(50% vehicle-present)','Fixed-grid tasks\n(16.9% vehicle-present)']
xi=np.arange(2)
ax.bar(xi-.18,[100*baseline,100*grid_q],width=.36,color='#326A96',label='Separate: both wrong')
ax.bar(xi+.18,[100*shared,100*grid_f],width=.36,color='#D17A48',label='Shared late fusion: both wrong')
ax.set_xticks(xi);ax.set_xticklabels(categories)
ax.set_ylabel('Joint-error rate (%)');ax.set_title('Sensitivity to the definition and prevalence of decision events')
ax.legend();ax.grid(axis='y',alpha=.2)
save('04_roi_task_sensitivity.png')

fig,ax=plt.subplots(figsize=(8.8,5.1))
ax.plot(2-kgrid,100*line,color='#B4473C',lw=2.8)
ax.fill_between(2-kgrid,100*lo,100*hi,color='#B4473C',alpha=.17)
ax.scatter([2,1],[100*baseline,100*shared],s=50,color='#B4473C')
ax.annotate('Separate results\n(two distinct outputs)',(2,100*baseline),xytext=(-120,-12),textcoords='offset points',fontsize=9)
ax.annotate('Shared late fusion\n(one reused output)',(1,100*shared),xytext=(10,10),textcoords='offset points',fontsize=9)
ax.set_xlabel('Expected distinct result identities delivered per event (2 − κ)')
ax.set_ylabel('Joint-error probability (%)');ax.set_title('Reuse-risk tradeoff: distinct answers, not measured compute cost')
ax.grid(alpha=.2)
ax.set_xlim(2.12,.88)
save('05_risk_vs_distinct_results.png')

bootdiff=be[:,2]-be[:,3]
report={
 'source':'https://github.com/eddyhkchiu/DMSTrack',
 'source_revision':'d3b9949499c8e68ea33060873bd1cb95b6d4d323',
 'dataset':'V2V4Real validation data republished as KITTI tracking/detection text by DMSTrack',
 'labels':'diagnostic vehicle-presence at radius 7.5m in shared KITTI camera x-z plane',
 'frame_count':int(s.frames.sum()),'scene_count':len(s), 'balanced_event_count':n,
 'ground_truth_positive_count':int(s.balanced_positives.sum()),
 'detector_score_threshold':0.3,
 'separate_error_count':int(s.separate_joint_errors.sum()),
 'separate_joint_error_rate':baseline,
 'shared_late_fusion_error_count':int(s.late_fusion_errors.sum()),
 'shared_joint_error_rate':shared,
 'relative_joint_risk':shared/baseline,
 'absolute_joint_risk_difference':shared-baseline,
 'cluster_bootstrap_95ci_absolute_difference':np.quantile(bootdiff,[.025,.975]).tolist(),
 'cluster_bootstrap_95ci_relative_ratio':np.quantile(be[:,2]/be[:,3],[.025,.975]).tolist(),
 'bootstrap_iterations':len(indices),'bootstrap_seed':20261118,
 'per_agent_error_rate':{'ego':marginal['ego_errors'],'agent_1':marginal['agent1_errors']},
 'cooperative_cobevt_error_rate':marginal['cobevt_errors'],
 'sequences_with_strict_joint_increase':int((s.late_fusion_errors>s.separate_joint_errors).sum()),
 'fixed_grid':{'events':int(fg),'positives':int(s.fixed_positives.sum()),'separate_joint_rate':grid_q,'shared_late_joint_rate':grid_f,
  'ego_marginal_rate':s.fixed_ego_errors.sum()/fg,'agent1_marginal_rate':s.fixed_agent1_errors.sum()/fg},
 'limitations':['Offline diagnostic tasks are sampled using ground truth to balance positive/negative examples',
 'This is a counterfactual sharing policy over recorded outputs, not observed network or CPU energy',
 'Cooperative late fusion has different computational requirements from individual no-fusion predictions',
 'Nine driving sequences may not represent independent deployment environments',
 'No actual physical collision or safety outcomes are evaluated']
}
(ROOT/'empirical_report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['frame_count','balanced_event_count','separate_joint_error_rate','shared_joint_error_rate','absolute_joint_risk_difference','cluster_bootstrap_95ci_absolute_difference']},indent=2))
print('Wrote',len(list(F.glob('*.png'))),'plots to',F)
