#!/usr/bin/env python3
"""Publication-style replots of *published CooperScene* detection benchmarks.

NOT original OCTOPUS measurements, NOT F0 results, NOT directly observed
joint-error probabilities. V+2V = three physical vehicles; V+V = two.
C-V2X mAP combines real sensor data with the authors' throughput-derived
asynchronous transmission model. Reported 'latency' is transfer delay under
that model, not measured end-to-end F0 latency.

Source: https://github.com/UCR-CISL/CooperScene#benchmark-result
Benchmark table (mAP at 3D IoU=0.5); retrieved 2026-11-20.
"""
from __future__ import annotations
import csv
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

CSV_PATH=ROOT/'samples'/'three_vehicle_real_published'/'cooperscene_published_v2v_v2v2.csv'


def records(path=CSV_PATH):
    with path.open(newline='',encoding='utf8') as f:
        rows=list(csv.DictReader(f))
    model_order=['V2VNet','V2X-ViT','V2VAM','CoBEVT','ERMVP','CoSDH']
    data={}
    for row in rows:
        for k in ('map_05_cv2x','map_05_unlimited','sharing_mb','published_transmission_latency_ms'):
            row[k]=float(row[k])
        row['vehicle_count']=int(row['vehicle_count'])
        data[(row['model'],row['vehicle_count'])]=row
    assert len(rows)==12 and set(data)=={(m,n) for m in model_order for n in (2,3)}
    for row in rows:
        assert 0<=row['map_05_cv2x']<=1 and 0<=row['map_05_unlimited']<=1
        assert row['sharing_mb']>0 and row['published_transmission_latency_ms']>0
    return model_order,data


def annotate(fig):
    fig.text(.5,.012,
        'Published CooperScene benchmarks (ECCV 2026); V+2V = 3 vehicles. '
        'Real sensor data + authors\' C-V2X simulation, NOT F0 results.',
        ha='center',fontsize=8,color='#50545A')


def make_plots(out:Path):
    out.mkdir(parents=True,exist_ok=True)
    models,d=records()
    plt.rcParams.update({'font.size':10,'font.family':'DejaVu Sans',
                         'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'svg.fonttype':'none'})
    blues='#367DA9'; amber='#CE8454';green='#339477';gray='#A4ADBA'
    x=np.arange(len(models))
    delta_real=[100*(d[m,3]['map_05_cv2x']-d[m,2]['map_05_cv2x']) for m in models]
    delta_ideal=[100*(d[m,3]['map_05_unlimited']-d[m,2]['map_05_unlimited']) for m in models]
    fig,ax=plt.subplots(figsize=(10.0,5.4))
    ax.axhline(0,color='#515B6B',linewidth=.9)
    bars1=ax.bar(x-.19,delta_ideal,width=.35,color=gray,label='Ideal sharing (unlimited)')
    bars2=ax.bar(x+.19,delta_real,width=.35,color=blues,label='C-V2X (throughput-constrained)')
    for bar,val in zip([*bars1,*bars2],[*delta_ideal,*delta_real]):
        ax.text(bar.get_x()+bar.get_width()/2,
                val+(1.1 if val>=0 else -1.1),f'{val:+.0f}',ha='center',
                va='bottom' if val>=0 else 'top',fontsize=8.6)
    ax.set_xticks(x,models)
    ax.set_ylabel('Change in detection mAP@IoU 0.5 (percentage points)')
    ax.set_title('A third real vehicle helps less under throughput constraints',loc='left',fontweight='bold')
    ax.set_ylim(-6,27)
    ax.legend(loc='upper left',ncol=2,frameon=False)
    ax.grid(axis='y',alpha=.20)
    fig.subplots_adjust(bottom=.15,top=.88,left=.105,right=.97)
    annotate(fig)
    save(fig,out/'01_third_vehicle_gain_ideal_vs_cv2x')

    fig,ax=plt.subplots(figsize=(9.8,5.5))
    for i,m in enumerate(models):
        c=blues if i%2==0 else green
        row=d[m,3]
        xmb=row['sharing_mb']
        ax.plot([xmb,xmb],[row['map_05_cv2x'],row['map_05_unlimited']],color=c,alpha=.58,linewidth=2.3)
        ax.scatter([xmb],[row['map_05_cv2x']],marker='o',s=58,color=c,zorder=4)
        ax.scatter([xmb],[row['map_05_unlimited']],marker='o',s=65,facecolors='white',edgecolors=c,linewidths=2,zorder=4)
        offsets={'CoSDH':(8,-3),'V2X-ViT':(-62,-15),'ERMVP':(-50,7),
                 'V2VAM':(9,9),'CoBEVT':(9,-3),'V2VNet':(8,-3)}
        ax.annotate(m,(xmb,row['map_05_cv2x']),xytext=offsets[m],
                    textcoords='offset points',fontsize=8.5)
    ax.scatter([],[],marker='o',s=58,color=blues,label='C-V2X-constrained mAP')
    ax.scatter([],[],marker='o',s=65,facecolors='white',edgecolors=blues,label='Ideal-sharing mAP')
    ax.set_xscale('log');ax.set_xlim(.07,50);ax.set_ylim(.15,.87)
    ax.set_xlabel('Published total shared feature data per frame (MB, log scale)')
    ax.set_ylabel('Three-vehicle detection mAP@IoU 0.5')
    ax.set_title('More transmitted features do not guarantee higher usable accuracy',loc='left',fontweight='bold')
    ax.legend(loc='lower left',frameon=False)
    ax.grid(axis='y',alpha=.20)
    fig.subplots_adjust(bottom=.15,top=.88,left=.105,right=.96)
    annotate(fig)
    save(fig,out/'02_three_vehicle_accuracy_vs_payload')

    fig,ax=plt.subplots(figsize=(9.9,5.4))
    for i,m in enumerate(models):
        p2=d[m,2];p3=d[m,3]
        xx=[p2['published_transmission_latency_ms']/1000,
            p3['published_transmission_latency_ms']/1000]
        yy=[p2['map_05_cv2x'],p3['map_05_cv2x']]
        c=[blues,green,amber,'#806AAE','#58737E','#B36D76'][i]
        ax.plot(xx,yy,marker='o',color=c,linewidth=1.7,markersize=5)
        offsets={'V2X-ViT':(-59,-10),'V2VAM':(7,9)}
        ax.annotate(m,(xx[1],yy[1]),xytext=offsets.get(m,(6,1)),
                    textcoords='offset points',fontsize=9,color=c)
    ax.set_xscale('log'); ax.set_xlim(.19,150);ax.set_ylim(.15,.69)
    ax.set_xlabel('Published C-V2X transmission-delay estimate (seconds, log scale)')
    ax.set_ylabel('C-V2X-constrained detection mAP@IoU 0.5')
    ax.set_title('A third vehicle trades detection accuracy for additional transfer delay',loc='left',fontweight='bold')
    ax.text(.03,.97,'Each segment: 2 vehicles to 3 vehicles\nDelay is an upstream modeled transfer metric, not measured F0 runtime',
            transform=ax.transAxes,va='top',fontsize=8.5,color='#535B63')
    ax.grid(axis='y',alpha=.20)
    fig.subplots_adjust(bottom=.15,top=.88,left=.105,right=.96)
    annotate(fig)
    save(fig,out/'03_adding_vehicle_network_delay')
    print('Replotted 12 published records (six methods × two vehicle counts) in',out)


def save(fig,stem):
    for ext in ('.pdf','.svg','.png'):
        fig.savefig(str(stem)+ext,dpi=250,bbox_inches='tight',facecolor='white')
    plt.close(fig)


if __name__=='__main__':
    from argparse import ArgumentParser
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'samples'/'three_vehicle_real_published')
    args=parser.parse_args();make_plots(args.output)
