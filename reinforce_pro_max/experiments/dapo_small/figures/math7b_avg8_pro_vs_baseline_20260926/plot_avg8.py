"""Plot both complete 200-step runs; trailing 10-step smoothing, no extrapolation."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

root = Path(__file__).resolve().parent
s = json.loads((root/'snapshot.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.labelsize':13,
                     'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig, ax = plt.subplots(figsize=(10.8,6.0))
styles = {'baseline':('Baseline','#526477','-'),'pro_only':('Pro-only','#0077B6','--')}
export=[]
for arm,(name,color,ls) in styles.items():
    rows=s['arms'][arm]['rows']; x=np.array([r['step'] for r in rows]); y=100*np.array([r['avg8'] for r in rows])
    smooth=np.array([y[max(0,i-9):i+1].mean() for i in range(len(y))])
    ax.plot(x,y,color=color,alpha=.24,lw=.9,zorder=2)
    ax.plot(x,smooth,color=color,ls=ls,lw=2.5,label=name,zorder=4)
    ax.scatter(x[-1],smooth[-1],color=color,s=26,zorder=5)
    export.extend({'arm':arm,'step':int(t),'avg8_percent':float(v),'trailing_10_mean_percent':float(z)} for t,v,z in zip(x,y,smooth))
    print(name,'last10',smooth[-1],'all',y.mean(),'range',y.min(),y.max())
ax.set(xlim=(1,202),ylim=(0,75),xlabel='Training step',ylabel='Online training avg@8 (%)')
ax.set_xticks([1,25,50,75,100,125,150,175,200]);ax.set_yticks(range(0,71,10))
ax.grid(axis='y',color='#E3E7EB',lw=.7);ax.set_axisbelow(True)
ax.legend(loc='lower right',frameon=True,framealpha=.96,edgecolor='#E1E5E9',fontsize=12)
fig.suptitle('Pro-only vs. Baseline · Qwen2.5-Math-7B',x=.105,ha='left',fontsize=18,fontweight='bold',y=.975)
fig.text(.105,.905,'Faint lines: per-step avg@8   ·   Bold lines: trailing 10-step mean',fontsize=11,color='#526477')
fig.text(.105,.046,'200 steps per run; 32 prompts × 8 responses per step. One training seed per arm.',fontsize=10,color='#526477')
fig.text(.105,.014,'Microbatch: Baseline 32; Pro-only 16. First 9 smoothed points use the available steps.',fontsize=9.5,color='#526477')
fig.subplots_adjust(left=.105,right=.975,bottom=.15,top=.86)
fig.savefig(root/'avg8_pro_vs_baseline.png',dpi=200)
fig.savefig(root/'avg8_pro_vs_baseline.pdf')
with (root/'avg8_pro_vs_baseline.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(export[0]));w.writeheader();w.writerows(export)
