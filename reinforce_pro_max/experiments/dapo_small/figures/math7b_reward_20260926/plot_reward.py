"""Render the captured 7B training reward, without loading models or running evaluation."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent
snapshot = json.loads((root / 'snapshot.json').read_text())
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12, 'axes.labelsize': 13,
                     'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
styles = {'baseline': ('Baseline', '#526477', '-'),
          'pro_only': ('Pro-only', '#0077B6', '--'),
          'max_only': ('Max-only', '#D97816', '-.')}
fig, ax = plt.subplots(figsize=(10.8, 6.0))
export = []
for arm, (name, color, line) in styles.items():
    rows = snapshot['arms'][arm]['rows']
    x = np.array([r['step'] for r in rows]); y = np.array([r['reward'] for r in rows])
    smoothed = np.array([y[max(0, i-19):i+1].mean() for i in range(len(y))])
    ax.plot(x, y, color=color, alpha=.14, linewidth=.8)
    ax.plot(x, smoothed, color=color, linestyle=line, linewidth=2.4,
            label=f'{name} ({len(rows)}/200 steps)')
    ax.scatter(x[-1], smoothed[-1], color=color, s=30, zorder=5)
    for step, reward, smooth in zip(x, y, smoothed):
        export.append({'arm': arm, 'step': int(step), 'reward': float(reward),
                       'trailing_20_mean': float(smooth)})
    print(name, 'steps',len(rows),'all_reward',float(y.mean()),'last20_reward',float(smoothed[-1]))
ax.axhline(0, color='#7C8792', linewidth=.8, alpha=.6)
ax.grid(axis='y', color='#E3E7EB', linewidth=.7)
ax.set_axisbelow(True)
ax.set(xlim=(1, 204), xlabel='Training step', ylabel='Mean training reward')
ax.set_xticks([1, 25, 50, 75, 100, 125, 150, 175, 200])
ax.legend(loc='lower right', frameon=True, framealpha=.95, edgecolor='#E1E5E9', fontsize=11)
fig.suptitle('Qwen2.5-Math-7B · Training reward', x=.105, ha='left', fontsize=18, fontweight='bold', y=.975)
fig.text(.105,.905,'Bold lines: trailing 20-step mean  ·  Faint lines: per-step reward',color='#526477',fontsize=11)
fig.text(.105,.045,'Reward: correct = 1; incorrect = −0.5; unextracted = −1.  One seed per arm.',fontsize=10,color='#526477')
fig.text(.105,.012,f"Microbatch: Baseline 32; Pro-only / Max-only 16.  Snapshot: {snapshot['captured_at'][:19].replace('T',' ')} UTC+8",fontsize=9,color='#526477')
fig.subplots_adjust(left=.105,right=.975,bottom=.15,top=.86)
fig.savefig(root/'training_reward.png',dpi=200)
fig.savefig(root/'training_reward.pdf')
with (root/'training_reward.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(export[0]));w.writeheader();w.writerows(export)
