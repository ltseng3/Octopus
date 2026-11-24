"""Source-grounded CVC draft figures; never generates or estimates new model predictions."""
from pathlib import Path
import csv, json, collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.colors import TwoSlopeNorm

BASE=Path(__file__).resolve().parent
OUT=BASE/'figures';OUT.mkdir(parents=True,exist_ok=True)
DATA=BASE/'sources'
EXT=DATA
C={'local':'#477C91','majority':'#C67D36','strict':'#4F6EB5','expert':'#A44765','gray':'#666D75'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.0,'axes.titlesize':8.3,'axes.labelsize':8.1,'xtick.labelsize':7.4,'ytick.labelsize':7.4,'legend.fontsize':7.1,'figure.dpi':140,'savefig.dpi':300,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'axes.linewidth':0.6,'lines.linewidth':1.5,'grid.alpha':0.22})

def read(path):
 with open(path,newline='') as f:return [{k:float(v) if k not in ('scene','method','comparison','metric','model','estimate_status','unit','confidence') else v for k,v in row.items()} for row in csv.DictReader(f)]
def save(name):
 fig=plt.gcf();fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight',pad_inches=.05);fig.savefig(OUT/(name+'.png'),bbox_inches='tight',pad_inches=.05);plt.close(fig)

# FIG 1: raw nine sequence paired contrast, no independent-frame error bars
rs=read(DATA/'nine_scene_metrics.csv')
fig,ax=plt.subplots(figsize=(4.72,2.62))
for r in rs:
 x=100*r['marginal_benefit'];y=100*r['joint_increase']
 ax.scatter(x,y,s=18+0.12*r['frames'],color=C['local'],alpha=.86,edgecolors='white',linewidths=.4,zorder=3)
 ax.annotate(r['scene'],(x,y),xytext=(3,3),textcoords='offset points',fontsize=6.7)
ax.errorbar(9.571,2.233,xerr=[[9.571-5.36],[15.26-9.571]],yerr=[[2.233-.86],[3.61-2.233]],color=C['expert'],fmt='D',capsize=2.8,markersize=5.8,zorder=5,label='Nine-sequence aggregate')
ax.axhline(0,color='0.55',lw=.8,linestyle='--');ax.axvline(0,color='0.55',lw=.8,linestyle='--')
ax.set_xlim(-.9,25.9);ax.set_ylim(-.6,6.3)
ax.set_xlabel('Individual error reduction with shared fusion (pp)')
ax.set_ylabel('Increase in simultaneous-error risk (pp)')
ax.grid(lw=.45)
ax.legend(loc='upper right',frameon=False)
fig.subplots_adjust(left=.14,right=.99,top=.98,bottom=.24)
save('fig1_real_accuracy_joint_risk')

# FIG 2: sequence-0007 outcome and frame granularity
tr=DATA/'v2v4real_seq0007_trace.jsonl'
rows=[json.loads(s) for s in tr.read_text().splitlines() if s.strip()]
f=set(); count=collections.Counter()
for row in rows:
 a=row['candidates']['local:ego']; b=row['candidates']['local:agent1']; y=row['truth'];coop=row['candidates']['edge:ego'];
 if a==b:
  count['agree']+=1;count['agree_wrong']+=int(a!=y);count['agree_right']+=int(a==y)
 else:
  count['disagree']+=1;count['coop_wrong_when_disagree']+=int(coop!=y);count['coop_right_when_disagree']+=int(coop==y);f.add(row['frame_id'])
assert (count['agree'],count['disagree'],count['agree_right'],count['agree_wrong'],count['coop_right_when_disagree'],count['coop_wrong_when_disagree'],len(f)) == (250,192,246,4,182,10,183)
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(4.72,2.2),gridspec_kw={'width_ratios':[1,1.05]})
ax1.barh([1,0],[246,182],height=.58,color=C['local'],label='Correct')
ax1.barh([1,0],[4,10],left=[246,182],height=.58,color=C['expert'],label='Wrong')
ax1.set_yticks([1,0],['Locals agree','Locals disagree']);ax1.set_xlim(0,272);ax1.set_xlabel('ROI decisions')
ax1.text(124,1,'246 right',ha='center',va='center',fontsize=7.1,color='white')
ax1.text(91,0,'182 corrected by candidate',ha='center',va='center',fontsize=6.4,color='white')
ax1.text(253,1,'4',fontsize=6,va='center');ax1.text(195,0,'10',fontsize=6,va='center')
ax1.grid(axis='x',lw=.4)
ax2.bar([0,1],[192/442*100,183/221*100],width=.52,color=[C['local'],C['expert']])
for x,y in zip([0,1],[192/442*100,183/221*100]):ax2.text(x,y+2,f'{y:.1f}%',ha='center',fontsize=8)
ax2.set_xticks([0,1],['ROI disagreements','Frames needing\ncooperative candidate'],fontsize=7)
ax2.set_ylim(0,102);ax2.set_ylabel('Fraction (%)');ax2.grid(axis='y',lw=.4)
fig.subplots_adjust(left=.16,right=.99,top=.98,bottom=.27,wspace=.52)
save('fig2_disagreement_vs_frame_work')

# FIG 3: shocked source B/C majority recovery vs shared risk
shock=read(EXT/'shock_and_policy_grid.csv')
methods=['local','majority','strict','expert']; labels={'local':'Local','majority':'Majority sharing','strict':'F0 (two supporters)','expert':'Expert override*'}
fig,axs=plt.subplots(1,2,figsize=(4.72,2.55),sharex=True)
for ax,key,ttl in zip(axs,['r_ge_2','r_all_3'],['At least two agents wrong','All three agents wrong']):
 for m in methods:
  t=sorted([r for r in shock if r['method']==m],key=lambda r:r['shock_probability'])
  ax.plot([100*r['shock_probability'] for r in t],[100*r[key] for r in t],marker={'local':'o','majority':'s','strict':'D','expert':'^'}[m],ms=3.3,color=C[m],label=labels[m],lw=1.25)
 ax.set_title(ttl,pad=5);ax.set_xlabel('B/C common shock (%)');ax.set_ylabel('Error probability (%)');ax.grid(lw=.42);ax.set_xlim(-1,25)
axs[0].legend(fontsize=6.5,frameon=False,loc='upper left');axs[1].set_ylim(-1,27)
fig.subplots_adjust(left=.12,right=.985,top=.87,bottom=.19,wspace=.32)
save('fig3_three_agent_risk_tradeoff')

# FIG 4: calibration heatmap. rows: actual, columns assumed
cal=read(EXT/'expert_calibration_grid.csv')
trues=sorted({r['true_expert_q'] for r in cal});assumed=sorted({r['assumed_expert_q'] for r in cal})
fig,axs=plt.subplots(1,2,figsize=(4.72,2.60))
for ax,key,ttl in zip(axs,['r_ge_2_delta_pp','r_all_3_delta_pp'],['Change in R2 vs local','Change in R3 vs local']):
 Z=np.array([[next(r[key] for r in cal if r['true_expert_q']==t and r['assumed_expert_q']==q) for q in assumed] for t in trues]);
 im=ax.imshow(Z,cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-22,vcenter=0,vmax=39),origin='lower',aspect='auto')
 ax.set_title(ttl,pad=5);ax.set_xticks(range(len(assumed)),[f'{100*v:g}' for v in assumed],rotation=45,ha='right',fontsize=6.5);ax.set_yticks(range(len(trues)),[f'{100*v:g}' for v in trues],fontsize=6.7)
 ax.set_xlabel('Assumed A error (%)');ax.set_ylabel('True A error (%)')
 ax.plot([assumed.index(.015)],[trues.index(.015)],'kx',markersize=5)
cax=fig.add_axes([.87,.24,.022,.61]);cb=fig.colorbar(im,cax=cax);cb.set_label('Change (pp)',fontsize=7);cb.ax.tick_params(labelsize=6)
fig.subplots_adjust(left=.10,right=.85,top=.88,bottom=.24,wspace=.44)
save('fig4_expert_calibration')

# FIG5: ROI coverage vs frame downlink, energy threshold based on actual executor
rr=read(EXT/'roi_threshold_physical_cost_grid.csv')
from matplotlib.lines import Line2D
fig,axs=plt.subplots(1,2,figsize=(4.72,2.76),sharex=True)
colors={1:C['local'],2:C['majority'],4:C['strict']}
for K in [1,2,4]:
 t=sorted([r for r in rr if r['method']=='strict' and r['roi_per_frame']==K],key=lambda r:r['threshold'])
 x=[r['threshold'] for r in t]
 axs[0].plot(x,[100*r['share_roi_fraction'] for r in t],marker='o',ms=2.6,lw=1,ls='--',color=colors[K],label=f'{K} ROI accepted')
 axs[0].plot(x,[100*r['actual_frame_downlink_fraction'] for r in t],marker='s',ms=2.6,lw=1.4,color=colors[K],label=f'{K} ROI frame DL')
 axs[1].plot(x,[1000*(r['energy_per_frame_j']-.24) for r in t],marker='o',ms=2.6,lw=1.4,color=colors[K],label=f'{K} ROI/frame')
axs[0].set_ylabel('Accepted ROIs / downlink frames (%)');axs[0].set_ylim(-4,107)
axs[1].set_ylabel('Energy over local (mJ/frame)');axs[1].set_ylim(17.0,18.5)
for ax in axs:ax.set_xlabel('F0 acceptance threshold');ax.grid(lw=.45);ax.set_xlim(.79,1.005)
handles=[Line2D([0],[0],color=colors[k],lw=1.5,label=f'{k} ROI/frame') for k in [1,2,4]]
fig.legend(handles=handles,ncol=3,loc='upper center',frameon=False,fontsize=7,bbox_to_anchor=(.53,1.01))
axs[0].legend(handles=[Line2D([0],[0],color=C['gray'],ls='--',marker='o',ms=3,label='Accepted ROI'),Line2D([0],[0],color=C['gray'],ls='-',marker='s',ms=3,label='Frame DL')],fontsize=6.6,frameon=False,loc='lower left')
fig.subplots_adjust(left=.125,right=.985,top=.84,bottom=.19,wspace=.40)
save('fig5_sunk_cost_and_roi_reuse')

# FIG6: deadline and energy from actually simulated graph
rr=read(EXT/'deadline_reliability_energy_sweep.csv')
fig,axs=plt.subplots(1,2,figsize=(4.72,2.38),sharex=True)
for m in ['local','strict','expert']:
 t=sorted([r for r in rr if r['method']==m],key=lambda r:r['deadline_ms'])
 xx=[r['deadline_ms'] for r in t]
 axs[0].plot(xx,[100*r['marginal_error'] for r in t],color=C[m],lw=1.55,label=labels[m])
 axs[1].plot(xx,[1000*r['energy_per_frame_j'] for r in t],color=C[m],lw=1.55,label=labels[m])
axs[0].set_ylabel('Mean agent error (%)');axs[0].set_ylim(0,24)
axs[1].set_ylabel('Modeled energy (mJ/frame)');axs[1].set_ylim(234,264)
for ax in axs:
 ax.set_xlabel('Decision deadline after frame arrival (ms)');ax.grid(lw=.44);ax.axvline(31.5,ls=':',lw=.9,color=C['gray'])
axs[0].legend(fontsize=6.7,loc='upper right',frameon=False)
fig.subplots_adjust(left=.12,right=.985,top=.97,bottom=.22,wspace=.33)
save('fig6_deadline_cliff')

# Audit exact inputs
manifest={'figure_source':{'fig1':'nine_scene_metrics.csv, V0.7 figure study','fig2':'examples/v2v4real_real_seq0007/trace.jsonl','fig3':'shock_and_policy_grid.csv','fig4':'expert_calibration_grid.csv','fig5':'roi_threshold_physical_cost_grid.csv','fig6':'deadline_reliability_energy_sweep.csv'},'scene_0007_counts':dict(count),'disagreement_frames':len(f),'tools':'matplotlib 3.10; source-grounded plotting only'}
(OUT/'figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Wrote',len(list(OUT.glob('*.pdf'))),'PDF figures; 0007',count,'disagree frames',len(f))
