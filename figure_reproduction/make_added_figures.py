"""Additional paper diagrams and analytic/data-derived figures. No new detector outputs."""
from pathlib import Path
import csv, itertools, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Polygon, Circle, FancyArrowPatch
from matplotlib.lines import Line2D
P=Path(__file__).resolve().parent; D=P/'sources'; O=P/'figures'
C={'local':'#41778A','shared':'#C37937','f0':'#4D67AA','expert':'#AA4966','good':'#227C68','bad':'#B85046','ink':'#263746','gray':'#64748B','light':'#E9EFF3'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.titlesize':9,'axes.labelsize':8.5,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':7.5,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'grid.alpha':.23,'lines.linewidth':1.5})
def save(fig,name):
    fig.savefig(O/(name+'.pdf'),bbox_inches='tight',pad_inches=.06); plt.close(fig)
def rows(name):
    with (D/name).open() as f: return list(csv.DictReader(f))
def node(ax,x,y,w,h,txt,fc='white',ec=C['ink'],fs=8,weight='normal'):
    r=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.025,rounding_size=0.055',facecolor=fc,edgecolor=ec,lw=.9);ax.add_patch(r)
    ax.text(x+w/2,y+h/2,txt,ha='center',va='center',fontsize=fs,color=C['ink'],weight=weight)
def arr(ax,x1,y1,x2,y2,color=C['gray'],style='-',rad=0):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=8,color=color,lw=1.1,linestyle=style,connectionstyle=f'arc3,rad={rad}'))
# Intro diagram: one scene, complementary detection, separate evidence vs shared error.
fig,ax=plt.subplots(figsize=(4.72,3.55));ax.set_xlim(0,10);ax.set_ylim(0,7.1);ax.axis('off')
ax.text(.1,6.89,'(a) Complementary viewpoints',weight='bold',fontsize=9)
# top-view simplified road intersection
ax.add_patch(Rectangle((.1,3.65),5.2,2.98,facecolor='#F4F5F5',edgecolor='#D3DADF',lw=.7))
ax.add_patch(Rectangle((.1,4.56),5.2,.92,facecolor='#DEE3E7',edgecolor='none'))
ax.add_patch(Rectangle((2.25,3.65),1.2,2.98,facecolor='#DEE3E7',edgecolor='none'))
for x,y,w,h in [(2.25,4.0,1.2,.01),(.2,5.02,5,.01)]:ax.add_patch(Rectangle((x,y),w,h,fc='white',ec='none'))
# A bottom left, truck blocking diagonal, B upper left, target right
A=(.7,4.05);B=(1.05,5.94);T=(4.57,5.0)
ax.add_patch(Polygon([A,(4.9,4.6),(4.9,5.5)],closed=True,fc=C['local'],alpha=.12,ec='none'))
ax.add_patch(Polygon([B,(4.9,4.65),(4.9,5.4)],closed=True,fc=C['good'],alpha=.14,ec='none'))
ax.plot([A[0],T[0]],[A[1],T[1]],'--',color=C['bad'],lw=1.15)
ax.plot([B[0],T[0]],[B[1],T[1]],'-',color=C['good'],lw=1.3)
# occluder placed on A line near x2.7, y4.54; not B line y5.5
ax.add_patch(Rectangle((2.35,4.18),.95,.62,fc='#8D969C',ec='#535F67',lw=.7));ax.text(2.82,4.49,'Truck',ha='center',va='center',color='white',fontsize=7)
for (x,y),lab,col in [(A,'A',C['local']),(B,'B',C['good'])]:
    ax.add_patch(FancyBboxPatch((x-.28,y-.16),.56,.32,boxstyle='round,pad=.01,rounding_size=.06',fc=col,ec='white',lw=.7));ax.text(x,y,lab,ha='center',va='center',color='white',weight='bold',fontsize=7)
ax.add_patch(Rectangle((T[0]-.35,T[1]-.20),.70,.40,fc='#D5B578',ec='#8B662C',lw=.8));ax.text(T[0],T[1]+.36,'Car',ha='center',fontsize=7.5)
ax.text(.28,3.71,'A: occluded',color=C['bad'],fontsize=7.3);ax.text(.28,6.35,'B: visible',color=C['good'],fontsize=7.3)
node(ax,5.66,5.55,4.18,.78,'Local A: missed car',fc='#F8E8E5',ec=C['bad']);node(ax,5.66,4.53,4.18,.78,'Local B: detects car',fc='#E4F2ED',ec=C['good'])
ax.text(7.75,4.03,'A different view can recover a miss.',ha='center',fontsize=7.4)
ax.plot([.1,9.9],[3.42,3.42],color='#CBD5DC',lw=.75)
ax.text(.1,3.10,'(b) Redundancy depends on which predictions are retained',weight='bold',fontsize=9)
# Rows each same initial A/B, two cases
for x,txt,fc,ec in [(0.2,'A: miss','#F8E8E5',C['bad']),(2.05,'B: detect','#E4F2ED',C['good'])]:node(ax,x,1.93,1.58,.58,txt,fc,ec,7.4)
ax.text(1.88,2.68,'Separate outputs',ha='center',fontsize=8,weight='bold')
arr(ax,3.75,2.23,4.47,2.23)
node(ax,4.55,1.89,2.02,.68,'Shared result\n(misses car)','#F8E8E5',C['bad'],7.0)
arr(ax,6.61,2.24,7.03,2.24,C['bad'])
node(ax,7.1,1.95,2.65,.56,'A: miss   B: miss','#F8E8E5',C['bad'],7.6)
ax.text(8.43,2.68,'Both adopt',ha='center',fontsize=8,weight='bold')
ax.text(1.88,1.46,'One correct observer',ha='center',fontsize=7.8,color=C['good']);ax.text(8.43,1.46,'No correct observer',ha='center',fontsize=7.8,color=C['bad'])
arr(ax,5.56,1.83,5.56,1.18,C['good'],style='--');node(ax,3.62,.30,3.88,.76,'Keep B’s correct detection\nfor possible alerting','#E4F2ED',C['good'],7.1)
ax.text(7.70,.68,'Selective\nadoption',fontsize=7.2,va='center',color=C['good'])
save(fig,'fig_intro_complementarity')
# Architecture: generic vision path, no claim every existing model broadcasts.
fig,ax=plt.subplots(figsize=(4.72,3.55));ax.set_xlim(0,10);ax.set_ylim(0,7.25);ax.axis('off')
ax.add_patch(FancyBboxPatch((.12,3.08),9.76,3.99,boxstyle='round,pad=.025,rounding_size=.10',fc='#EFF3F6',ec='#B6C6D2',lw=.85))
ax.text(.38,6.81,'Edge service',fontsize=9,weight='bold',color=C['ink'])
node(ax,.48,5.62,2.70,.75,'Align views /\nassociate regions',fs=7.4);node(ax,3.65,5.62,2.70,.75,'Cooperative fusion\nor F0 label voting',fc='white',fs=7.1);node(ax,6.82,5.62,2.70,.75,'Selective\nadoption policy',fc='#E4F2ED',ec=C['good'],fs=7.4)
arr(ax,3.19,6.0,3.63,6.0);arr(ax,6.36,6.0,6.80,6.0,C['good'])
node(ax,3.65,4.17,2.70,.74,'Evidence metadata\nIDs, age, trust',fs=7.0);arr(ax,5,4.93,5,5.59);arr(ax,6.38,4.54,8.15,5.59,rad=-.15)
# sensor boxes + local detector + final paths
for x,lbl in [(0.35,'Vehicle A'),(6.85,'Vehicle B')]:
    ax.add_patch(FancyBboxPatch((x,.12),2.8,2.62,boxstyle='round,pad=.03,rounding_size=.07',fc='#F6F7F8',ec='#B6C6D2',lw=.8))
    ax.text(x+1.4,.23,lbl,ha='center',fontsize=7.5,weight='bold')
    node(ax,x+.22,1.90,2.36,.57,'Camera / LiDAR',fs=7.5);node(ax,x+.22,1.19,2.36,.51,'Local detector',fs=7.5);node(ax,x+.22,.54,2.36,.47,'Final prediction',fc='#E4F2ED',ec=C['good'],fs=7.4)
    arr(ax,x+1.4,1.89,x+1.4,1.71);arr(ax,x+1.4,1.18,x+1.4,1.02,C['local'])
# upload from local, left routes to edge align via x1.8; right via middle, separate downlink path
ax.plot([2.94,3.37,3.37,1.83,1.83],[1.45,1.45,3.57,3.57,5.35],color=C['local'],lw=1.1);arr(ax,1.83,5.35,1.83,5.60,C['local'])
ax.plot([6.81,6.60,6.60,2.8,2.8],[1.45,1.45,3.36,3.36,5.35],color=C['local'],lw=1.1);arr(ax,2.8,5.35,2.8,5.6,C['local'])
# adoption outputs run via right edge and central outside
ax.plot([8.17,8.17,9.91,9.91],[5.60,5.16,5.16,.78],color=C['good'],lw=1.15);arr(ax,9.91,.78,9.46,.78,C['good'])
ax.plot([7.58,7.58,6.45,6.45],[5.61,5.05,5.05,.78],color=C['good'],lw=1.15,ls='--');arr(ax,6.45,.78,2.96,.78,C['good'],style='--')
ax.text(5.03,2.90,'Uploads',ha='center',fontsize=7.3,color=C['local']);ax.text(5.10,.20,'Optional update to A',ha='center',fontsize=7.0,color=C['good']);ax.text(.5,3.81,'Frame features\nor detections',fontsize=7.4,color=C['local'])
save(fig,'fig_edge_architecture')
# Exact analytic example: weighted Bernoulli artifact sums.
parts=[(1,1,1,1,1),(2,2,1),(3,1,1),(5,)];q=.1
records=[]
for s in parts:
    pmf=np.zeros(6)
    for es in itertools.product((0,1),repeat=len(s)):
        pmf[sum(a*b for a,b in zip(s,es))]+=math.prod(q if b else 1-q for b in es)
    for m in range(1,6):records.append({'partition':'+'.join(map(str,s)),'jobs':len(s),'q':q,'m':m,'risk':sum(pmf[m:])})
with (D/'analytical_partition_risk.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=records[0]);w.writeheader();w.writerows(records)
fig,ax=plt.subplots(figsize=(4.72,2.56))
for s,col,mark in zip(parts,[C['local'],C['f0'],C['expert'],C['shared']],['o','s','D','^']):
    t=[r for r in records if r['partition']=='+'.join(map(str,s))];ax.plot([r['m'] for r in t],[r['risk']*100 for r in t],label=f"{t[0]['partition']} ({len(s)} jobs)",color=col,marker=mark,ms=4)
ax.set_yscale('log');ax.set_xticks(range(1,6));ax.set_xlabel('Collective failure threshold $m$');ax.set_ylabel('$R_m$ (%)');ax.grid(axis='y',which='both',lw=.4);ax.legend(frameon=False,ncol=2,loc='lower left',fontsize=7.2);ax.set_ylim(.0006,85);fig.tight_layout(pad=.5);save(fig,'fig_partition_analytical')
# RQ1/2 prevalence sensitivity, computed from class-conditional observed values.
r=rows('seq0007_class_policy.csv');p=np.linspace(0,1,401);fig,axs=plt.subplots(1,2,figsize=(4.72,2.35))
for meth,lab,col,ls in [('local','Local',C['local'],'-'),('shared','Late fusion',C['shared'],'-'),('OR','OR / disagreement gate',C['good'],'--'),('AND','AND',C['gray'],':')]:
    a=next(x for x in r if x['policy']==meth and x['truth']=='1');b=next(x for x in r if x['policy']==meth and x['truth']=='0')
    for ax,key in zip(axs,['marginal_error','joint_error']):ax.plot(p*100,100*(p*float(a[key])+(1-p)*float(b[key])),color=col,ls=ls,label=lab)
for ax,title in zip(axs,['Mean agent error','Both wrong']):
    ax.set_title(title);ax.set_xlabel('Positive-query prevalence (%)');ax.set_ylabel('Error (%)');ax.grid(lw=.45);ax.set_xlim(0,100)
axs[0].axvline(100*10/192,color=C['good'],lw=.8,ls=':');axs[0].annotate('OR / local\ncrossing: 5.2%',(100*10/192,3),xytext=(18,24),arrowprops={'arrowstyle':'-','lw':.7},fontsize=7)
# same scale? AND large overwhelms joint. omit AND from joint? use zoom inset? use both on 0-85, OR vs local small but readable. set second limit 0-6, omit AND line explicitly clipped? better separate AND from chart, use only local/shared/OR; AND table covers. 
for ax in axs:
    ax.lines[3].remove()
axs[0].set_ylim(0,46);axs[1].set_ylim(0,6);axs[0].legend(frameon=False,loc='upper left',fontsize=6.7);fig.tight_layout(pad=.55,w_pad=1.2);save(fig,'fig_prevalence_sensitivity')
# Five-agent study: correct grouping vs false independence at fixed marginal q.
r=rows('five_agent_lineage_sweep.csv');fig,axs=plt.subplots(1,2,figsize=(4.72,2.53))
methods=[('local_only','Local',C['local'],'o'),('naive_weighted','Weighted broadcast',C['shared'],'s'),('lineage_aware','Grouped F0',C['good'],'D'),('false_independence','Ungrouped F0',C['expert'],'^')]
for ax,key,title in zip(axs,['r_ge_3','r_ge_5'],['At least 3 wrong','All 5 wrong']):
    for name,lab,col,m in methods:
        t=sorted([x for x in r if x['method']==name],key=lambda x:float(x['rho']));ax.plot([float(x['rho']) for x in t],[float(x[key])*100 for x in t],color=col,marker=m,ms=3.5,label=lab)
    ax.set_title(title);ax.set_xlabel('Common-error mixture weight');ax.set_ylabel('Error probability (%)');ax.grid(lw=.4);ax.set_xticks([0,.25,.5,.75,1]);ax.set_xticklabels(['0','.25','.5','.75','1'])
axs[0].legend(frameon=False,fontsize=6.7,loc='upper left');axs[1].set_ylim(-.12,3.05);fig.tight_layout(pad=.6,w_pad=1.3);save(fig,'fig_five_agent_lineage')
# Prior E1 real replay actual frame randomized policies, energy per physical frame.
r=sorted([x for x in rows('multiobjective_policy_metrics.csv') if x['dataset']=='v2v4real_0007'],key=lambda x:float(x['fusion_target_probability']))
fig,axs=plt.subplots(1,2,figsize=(4.72,2.42));
for ax,k,title,col in zip(axs,['marginal_error','joint_error'],['Mean agent error','Both wrong'],[C['local'],C['expert']]):
    y=np.array([100*float(x[k+'_rate']) for x in r]);lo=np.array([100*float(x[k+'_ci_low']) for x in r]);hi=np.array([100*float(x[k+'_ci_high']) for x in r]);e=np.array([2*float(x['energy_incremental_per_decision_j']) for x in r]);elo=np.array([2*float(x['energy_sensitivity_min_per_decision_j']) for x in r]);ehi=np.array([2*float(x['energy_sensitivity_max_per_decision_j']) for x in r]);ax.errorbar(e,y,yerr=[y-lo,hi-y],xerr=[e-elo,ehi-e],color=col,marker='o',ms=3.5,lw=1.1,capsize=2)
    for x,v,row in zip(e,y,r):ax.annotate(f"{100*float(row['fusion_target_probability']):g}%",(x,v),xytext=(3,4),textcoords='offset points',fontsize=6.6)
    ax.set_title(title);ax.set_xlabel('Modeled energy (J/frame)');ax.set_ylabel('Error (%)');ax.grid(lw=.4);ax.set_xlim(.145,.365);ax.set_ylim(bottom=0)
fig.tight_layout(pad=.55,w_pad=1.2);save(fig,'fig_real_energy_tradeoff')
print('Generated six vector figures and exact partition CSV.')
