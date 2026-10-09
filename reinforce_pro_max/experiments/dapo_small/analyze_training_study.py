from pathlib import Path
import json,csv,collections,random,math
R=Path(__file__).resolve().parent;P=json.loads((R/'plan.json').read_text());names=['initial',*P['arms']];scores={};questions={};metrics={}
for name in names:
 e=R/name/'evaluation';f=e/'predictions.jsonl'
 if (e/'_SUCCESS').exists() and not (e/'SKIPPED.json').exists():
  rows=[json.loads(s) for s in f.open()];assert len(rows)==960
  groups=collections.defaultdict(list)
  for r in rows:groups[r['question_sha256']].append(r)
  assert len(groups)==30 and all(len(v)==32 for v in groups.values())
  questions[name]={k:sum(x['correctness'] for x in v)/32 for k,v in groups.items()}
  scores[name]={'avg_at_32':sum(questions[name].values())/30,'formatted_fraction':sum(r['formatted'] for r in rows)/960,'truncation_fraction':sum(r['finish_reason']=='length' for r in rows)/960}
 records=collections.defaultdict(list)
 for path in (R/name/'metrics').glob('policy.*.jsonl'):
  for s in path.open():
   try:r=json.loads(s)
   except json.JSONDecodeError:continue
   records[r['loss_call']].append(r)
 curve=[]
 for step,rs in sorted(records.items()):
  total=lambda k:sum(r[k] for r in rs)
  n=total('responses');tokens=total('active_tokens')
  curve.append({'loss_call':step,'rank_records':len(rs),'active_tokens':tokens,'ppl_gap_current_rollout':total('current_rollout_ppl_gap_sum')/n,'ppl_gap_cached_rollout':total('cached_rollout_ppl_gap_sum')/n,'prefix_acceptance':1-total('prefix_rejected')/tokens,'token_acceptance':1-total('token_rejected')/tokens,'retained_second_moment':total('retained_weight_sum_sq')/tokens,'unfiltered_second_moment':total('unfiltered_weight_sum_sq')/tokens,'lower_reentry_second_moment':total('lower_reentry_weight_sum_sq')/tokens})
 if curve:
  with (R/name/'curve.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(curve[0]));w.writeheader();w.writerows(curve)
  metrics[name]={'recorded_calls':len(curve),'all_calls_have_eight_ranks':all(r['rank_records']==8 for r in curve),'last':curve[-1]}
contrasts={}
for a,b in [('max_only','baseline'),('pro_only','baseline'),('pro_max','max_only'),('pro_max','pro_only'),('pro_max','baseline')]:
 if a in questions and b in questions:
  assert questions[a].keys()==questions[b].keys()
  vals=[questions[a][k]-questions[b][k] for k in sorted(questions[a])];rng=random.Random(20260925);boot=sorted(sum(rng.choices(vals,k=30))/30 for _ in range(5000));contrasts[a+' minus '+b]={'difference':sum(vals)/30,'question_cluster_bootstrap_95_percent':[boot[125],boot[4874]],'scope':'Question sampling only; one training seed.'}
result={'completed_scores':scores,'paired_contrasts':contrasts,'training_metrics':metrics,'scope':P['scope']};(R/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
text=['# TRM-inspired short experiment','',P['scope'],'','| Arm | AIME25 avg@32 | Format rate | Truncated |','|---|---:|---:|---:|']
for n in names:
 if n in scores:v=scores[n];text.append(f"| {n} | {100*v['avg_at_32']:.2f}% | {100*v['formatted_fraction']:.2f}% | {100*v['truncation_fraction']:.2f}% |")
 else:text.append(f'| {n} | Pending | Pending | Pending |')
text+=['','TRM-style PPL Gap is the response mean of the absolute mean token log-ratio.','Current/rollout and cached snapshot/rollout ratios are recorded separately.','Coefficient moments divide by all active tokens, including rejected tokens.','Curves and acceptance frequencies are observations; they do not establish global divergence assumptions.','All scores use 30 unique AIME25 questions, each sampled 32 times; avg@32 is not pass@32.','No held-out score selects thresholds or checkpoints.','']
(R/'report.md').write_text('\n'.join(text))

import subprocess,sys
subprocess.run([sys.executable, '/volume/pt-train/users/rbliu/github/OpenRLHF/reinforce_pro_max/experiments/dapo_small/summarize_training_avg8.py', str(R)], check=True)

if (R/'evaluation-policy.json').exists():
 result['training_avg8']=json.loads((R/'training_avg8_summary.json').read_text())
 result['evaluation_policy']=json.loads((R/'evaluation-policy.json').read_text())
 (R/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
 report=['# Online training avg@8','',result['training_avg8']['scope'],'','| Arm | Steps | Last 50 avg@8 | All steps avg@8 |','|---|---:|---:|---:|']
 for name,v in result['training_avg8']['arms'].items():
  report.append(f"| {name} | {v['recorded_steps']} | {100*v['last_50_steps_avg_at_8']:.2f}% | {100*v['all_recorded_steps_avg_at_8']:.2f}% |")
 report += ['', 'Future held-out evaluations were skipped by user request. Completed baseline AIME25 results remain in baseline/evaluation/results.json.']
 (R/'report.md').write_text('\n'.join(report)+'\n')
