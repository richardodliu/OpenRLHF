"""Reproduce paper figures from the completed 544-step study's exported curves."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DATA = Path(__file__).parent / 'results/math15b_544'
OUT = ROOT / 'tex/figures'
OUT.mkdir(exist_ok=True)
# Pairwise layout follows TRM, arXiv:2512.23075v4, Figures 1--3:
# mismatch on the left, score on the right, thin solid lines, bottom legend.
ARMS = [('baseline', 'Baseline', '#377EB8'),
        ('max_only', 'Max-only', '#E67E22'),
        ('pro_only', 'Pro-only', '#E67E22'),
        ('pro_max', 'Pro Max', '#E67E22')]
plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42,
                     'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.labelsize': 10, 'xtick.labelsize': 9,
                     'ytick.labelsize': 9, 'axes.linewidth': 0.65})
curves = {}
summary = {}
combined=[]
for arm,label,color in ARMS:
    train=list(csv.DictReader((DATA/arm/'training_avg8.csv').open()))
    policy=list(csv.DictReader((DATA/arm/'curve.csv').open()))
    assert len(train)==len(policy)==544
    assert [int(r['step']) for r in train]==list(range(1,545))
    assert [int(r['loss_call']) for r in policy]==list(range(1,545))
    assert all(int(r['rank_records'])==8 for r in policy)
    avg=np.array([float(r['training_avg_at_8']) for r in train])
    gap=np.array([float(r['ppl_gap_current_rollout']) for r in policy])
    for i,(a,g) in enumerate(zip(avg,gap),1):combined.append(dict(arm=arm,step=i,training_avg_at_8=a,ppl_gap=g))
    summary[arm]={'all_avg8':float(avg.mean()),'last50_avg8':float(avg[-50:].mean()),
                  'last50_ppl_gap':float(gap[-50:].mean()),
                  'token_weighted_prefix_rejection':sum((1-float(r['prefix_acceptance']))*int(r['active_tokens']) for r in policy)/sum(int(r['active_tokens']) for r in policy)}
    curves[arm] = (gap * 1000, avg * 100)

steps = np.arange(20, 545)
window = np.ones(20) / 20
smoothed = {arm: tuple(np.convolve(v, window, mode='valid') for v in values)
            for arm, values in curves.items()}
# Identical limits across pairs; show the entire available smoothed trajectory.
gap_max = max(float(values[0].max()) for values in smoothed.values())
gap_top = np.ceil(gap_max * 1.06 / 0.5) * 0.5
for arm, label, color in ARMS[1:]:
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0))
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.27, top=0.965, wspace=0.35)
    for name, legend, line_color in [('baseline', 'Baseline', ARMS[0][2]),
                                      (arm, label, color)]:
        for ax, values in zip(axes, smoothed[name]):
            ax.plot(steps, values, color=line_color, linewidth=1.15,
                    solid_capstyle='round', label=legend)
    for ax in axes:
        ax.set_xlim(0, 544)
        ax.set_xticks([0, 100, 200, 300, 400, 500])
        ax.set_xlabel('Training Step', labelpad=3)
        ax.grid(color='#D9DEE4', linewidth=0.55, alpha=0.75)
        ax.set_axisbelow(True)
        ax.tick_params(direction='out', length=3, width=0.6, pad=3)
        for spine in ax.spines.values():
            spine.set_color('#737373')
    axes[0].set_ylabel(r'Log Abs PPL Gap ($\times 10^{-3}$)', labelpad=5)
    axes[0].set_ylim(0, gap_top)
    axes[1].set_ylabel('Training avg@8 (%)', labelpad=5)
    axes[1].set_ylim(0, 40)
    axes[1].set_yticks([0, 10, 20, 30, 40])
    legend = fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center',
                       bbox_to_anchor=(0.54, 0.025), ncol=2, fontsize=10,
                       frameon=True, fancybox=False, edgecolor='#D9DEE4',
                       handlelength=2.6, columnspacing=2.0, borderpad=0.4)
    legend.get_frame().set_linewidth(0.5)
    fig.savefig(OUT / f'training_{arm}_vs_baseline.pdf',
                metadata={'CreationDate': None})
    # PNG previews stay outside the paper source tree.
    fig.savefig(Path('/tmp') / f'promax_training_{arm}_vs_baseline.png', dpi=200)
    plt.close(fig)
with (DATA/'paper_curves.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(combined[0]));w.writeheader();w.writerows(combined)
(DATA/'paper_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
