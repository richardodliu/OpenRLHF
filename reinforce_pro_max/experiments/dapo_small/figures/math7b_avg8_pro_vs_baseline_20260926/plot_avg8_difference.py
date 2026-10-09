"""Full-history comparison with identical trailing means and symmetric difference axis."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

root=Path(__file__).resolve().parent
s=json.loads((root/'snapshot.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.labelsize':12,
                     'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
x=np.arange(1,201)
y={k:100*np.array([r['avg8'] for r in s['arms'][k]['rows']]) for k in ['baseline','pro_only']}
assert all([r['step'] for r in s['arms'][k]['rows']]==list(x) for k in y)
smooth=lambda v:np.array([v[max(0,i-9):i+1].mean() for i in range(len(v))])
b,p=smooth(y['baseline']),smooth(y['pro_only']);d=p-b
assert np.allclose(d,smooth(y['pro_only']-y['baseline']))
blue='#087FB5';gray='#687583';orange='#BA7C42'
fig,(ax,ad)=plt.subplots(2,1,figsize=(11.5,8),sharex=True,gridspec_kw={'height_ratios':[2.25,1],'hspace':.14})
for key,mean,color,name,ls in [('baseline',b,gray,'Baseline','-'),('pro_only',p,blue,'Pro-only','--')]:
    ax.plot(x,y[key],color=color,alpha=.13,lw=.85)
    ax.plot(x,mean,color=color,lw=2.4,ls=ls,label=name,zorder=4)
    ax.scatter(200,mean[-1],s=28,color=color,zorder=5)
ax.set(ylim=(0,65),ylabel='Online training avg@8 (%)')
ax.set_yticks(np.arange(0,61,10));ax.tick_params(axis='x',labelbottom=False)
ax.legend(loc='upper left',ncol=2,frameon=False,bbox_to_anchor=(0,1.025))
ax.axvspan(190.5,200,color=blue,alpha=.045,zorder=0)
ax.annotate('Steps 191–200 mean\nPro-only 44.41%\nBaseline 40.43%\nDifference +3.98 pp',
            xy=(199,(p[-1]+b[-1])/2),xytext=(.975,.96),textcoords='axes fraction',
            ha='right',va='top',fontsize=10.5,color='#233748',
            bbox={'boxstyle':'round,pad=.45','fc':'white','ec':'#D8E2E8','alpha':.97},
            arrowprops={'arrowstyle':'-','color':'#788B99','lw':.9},zorder=6)
# Positive and negative differences receive equal visual weight and symmetric limits.
ad.fill_between(x,0,d,where=d>=0,interpolate=True,color=blue,alpha=.20)
ad.fill_between(x,0,d,where=d<0,interpolate=True,color=orange,alpha=.20)
ad.plot(x,d,color='#344B5D',lw=1.6)
ad.axhline(0,color='#50606D',lw=1)
limit=max(5,5*np.ceil(np.max(np.abs(d))/5))
ad.set(ylim=(-limit,limit),xlim=(1,202),xlabel='Training step',ylabel='Pro − Baseline\n(percentage points)')
ad.yaxis.set_major_locator(MultipleLocator(10 if limit>=20 else 5))
ad.set_xticks([1,25,50,75,100,125,150,175,200])
ad.text(.015,.91,'Above zero: Pro-only higher',transform=ad.transAxes,color=blue,fontsize=10,va='top')
ad.text(.015,.09,'Below zero: Baseline higher',transform=ad.transAxes,color='#946036',fontsize=10,va='bottom')
ad.scatter(200,d[-1],color=blue,s=24,zorder=5)
for a in [ax,ad]:
    a.grid(axis='y',color='#E3E8EC',lw=.7);a.set_axisbelow(True)
fig.suptitle('Pro-only vs. Baseline · Qwen2.5-Math-7B',x=.105,ha='left',y=.975,fontsize=18,fontweight='bold')
fig.text(.105,.927,'Full 200-step history · Faint: raw values · Bold: trailing 10-step means',fontsize=11,color='#536777')
fig.text(.105,.069,'Lower panel: difference between the two trailing 10-step means; symmetric vertical scale.',fontsize=10,color='#536777')
fig.text(.105,.042,'One seed per arm; 32 prompts × 8 responses per step. Microbatch: Baseline 32; Pro-only 16.',fontsize=9.5,color='#536777')
fig.text(.105,.017,'First 9 smoothed points use available steps. Shaded endpoint window: steps 191–200.',fontsize=9.5,color='#536777')
fig.subplots_adjust(left=.105,right=.975,top=.88,bottom=.15)
for ext in ['png','pdf']:
    fig.savefig(root/f'avg8_full_and_difference.{ext}',dpi=200)
print('difference limits',-limit,limit,'endpoint',d[-1])
