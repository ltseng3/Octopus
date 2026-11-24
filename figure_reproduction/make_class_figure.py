from pathlib import Path
import csv,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=Path(__file__).resolve().parent
rows=list(csv.DictReader((r/'sources/seq0007_class_policy.csv').open()))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'axes.linewidth':.65})
fig,axs=plt.subplots(1,2,figsize=(4.8,2.5),layout='constrained');names=['local','shared','OR','AND']; labels=['Local','Coop.','OR / gate','AND'];x=np.arange(4)
for ax,y,title in zip(axs,[1,0],['(a) Present-vehicle queries','(b) Negative queries']):
 vals=[next(v for v in rows if v['policy']==n and int(v['truth'])==y) for n in names]
 ax.bar(x-.18,[100*float(v['marginal_error']) for v in vals],width=.34,color='#527b9e',label='Mean agent error');ax.bar(x+.18,[100*float(v['joint_error']) for v in vals],width=.34,color='#b86b39',label='Both wrong')
 ax.set_xticks(x,labels,rotation=24,ha='right');ax.set_title(title,fontsize=8);ax.set_ylabel('Conditional error (%)');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
 # independent scales; make scales explicit
 ax.set_ylim(0,100 if y==1 else 7);ax.set_yticks([0,25,50,75,100] if y==1 else [0,2,4,6]);
axs[0].legend(frameon=False,loc='upper left',fontsize=7)
fig.savefig(r/'figures/fig2_class_specific_risk.pdf',bbox_inches='tight',pad_inches=.04);fig.savefig(r/'figures/fig2_class_specific_risk.png',dpi=300,bbox_inches='tight',pad_inches=.04)
