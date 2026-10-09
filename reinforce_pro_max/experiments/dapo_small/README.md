# DAPO data and reward snapshot

This directory stores the dataset and reward function used for the small DAPO
experiment. It contains inputs and validation records, not experiment results.

## Current full-data run (2026-09-25)

Active run: `/volume/pt-train/users/rbliu/github/test/runs/promax_dapo_math15b_full_four_arm_20260925`.
The authoritative configuration is `study-full-plan.json`. The earlier 7B, 100-update
study and hardware timing checks are historical runs, not results from this study.

- Model: `/volume/pt-train/users/rbliu/model/Qwen2.5-Math-1.5B`.
- Sequential arms: Baseline, Max-only, Pro-only, Pro Max; 8 GPUs per arm; CUDA Graph enabled.
- All 17,381 eligible DAPO questions, plus 27 deterministic repeats to keep full batches:
  `train-full.jsonl` and `training-full-selection.json`; 17,408 rows, 544 updates per arm.
- Unchanged: 32 prompts/update, 8 responses/prompt, learning rate 3e-6, seed 42,
  3,072 generation tokens, 4,096 context, zero KL/entropy coefficient, single token IS ratio.
- Pro prefix geometric-mean bounds: [0.8, 1.25]. No PPO clipping or additional IS multiplier.
- Final-only checkpoint policy. Every arm starts from the initial model.
- Evaluate each final model and the initial model on the same local AIME25 960 rows,
  with the same reward file. The queue writes `status.json`, `summary.json`, and `analysis.json`.
- Launch a prepared run with `run_four_arm_study.py --run-dir <run directory>`;
  the runner reads the number of rows and updates from the plan.

## Files

| File | Contents |
| --- | --- |
| `distinct-prompts-with-rewards.parquet` | Full public deduplicated release: 17,398 exact-unique prompts and their original reward labels. |
| `train-3200.jsonl` | Fixed 3,200-example training subset with `question` and `label` fields. |
| `mathverify_dapo.py` | Reward function with final line-start `Answer:` section extraction. |
| `source.json` | Public source URL, pinned revision, size and SHA-256. |
| `training-selection.json` | Exclusions, zero-based source row indices, seed and training-file hash. |
| `reward-validation.json` | Validation results for this exact reward-file hash. |
| `SHA256SUMS` | Checksums of data, reward code and JSON records. |

## Dataset provenance and selection

The original dataset is [DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k).
The bundled Parquet is the public
[YouJiacheng deduplicated release](https://huggingface.co/datasets/YouJiacheng/DAPO-Math-17k-dedup),
pinned at revision `6e26a33abdabd3e6aaa1d742326b790758f7dbc5`.
Consult these source repositories for dataset attribution and licensing.
This is a pinned input for this experiment; it is not a claim that TRM used
this exact release or subset.

Deduplication here means exact prompt-text deduplication, not semantic
deduplication. All 17,398 bundled prompt/label pairs were found in the existing
local DAPO conversion, which contained 17,917 records. That conversion's SHA-256
was `b94a18b981121c5c9adcd22f86f3dbb7a6c2e241193e4e547f07075daec40b8b`.

For the training subset, seven source rows whose questions had conflicting
labels in that conversion were excluded. Ten further rows exceeded the
1,024-token prompt limit under the experiment's Qwen2.5-Math-7B preprocessing.
From the remaining 17,381 rows, `random.Random(42).sample(eligible, 3200)` selected
the subset, sorted by source row index before serialization. The manifest records
all selected and excluded indices; the bundled subset is the authoritative input.
Prompts, including their original `Answer:` instructions, and ground-truth labels
were preserved without rewriting.

The AIME25 overlap check removed only the known DAPO instruction wrapper and
compared NFKC/whitespace-normalized question text. It found no exact overlaps;
this is not a semantic contamination audit.

## Reward behavior

Relative to the original `mathverify_v3.py`, only `extract_answer` was changed:

1. If a line-start `Answer:` marker exists, restrict the passage to the text after
   its last occurrence (case-insensitive, with whitespace allowed).
2. Continue the existing boxed-answer extraction on that passage.
3. If no box is present but an `Answer:` section exists, use its first nonempty
   line after stripping surrounding whitespace.
4. Continue the original mathematical equivalence checking and reward logic.

Without an `Answer:` marker, the original boxed-answer behavior is retained.
The existing final-200-character extraction window is unchanged: an `Answer:`
marker outside that window is not seen by the extractor during reward scoring.
Correct, incorrect and unextractable answers still receive `1`, `-0.5` and `-1`,
respectively. For example, `\boxed{35}\nAnswer: \boxed{34}` is graded using `34`.

Dependencies are `torch`, `sympy` and `pylatexenc`. From the repository root,
the relevant training arguments are:

```bash
--prompt_data reinforce_pro_max/experiments/dapo_small/train-3200.jsonl \
--input_key question --label_key label \
--remote_rm_url reinforce_pro_max/experiments/dapo_small/mathverify_dapo.py
```

The recorded reward validation passed 22 targeted cases, five unchanged legacy
cases and correct-answer emission checks for all 3,200 training labels. These
checks validate parsing and grading behavior, not model performance.

Verify the archived bytes from this directory with `sha256sum -c SHA256SUMS`.

## Standalone AIME25 evaluation

`aime25.jsonl` contains the 960 AIME25 records used in this experiment: 30
questions repeated 32 times, preserving source messages, answers, order and
original row indices. `evaluation-source.json` records the LUFFY source commit
and checksums. LUFFY is the data source only; evaluation does not import its code
or require a checkout of that repository.

`evaluate_aime25.py` reads these local records and calls this directory's
`mathverify_dapo.py:reward_func` directly. Correctness is `reward == 1`, with the
same answer extraction, final-200-character window and mathematical grading as
training. Both `Answer:` and boxed answers are accepted. This replaces the
previously planned OAT grader; reported scores use this shared reward function.

Generation retains temperature 1.0, top-p 0.95, seed 42 plus the selected row's
position, context length 4,096 and generation cap 3,072. The source system message
is removed before applying the model's own chat template, as in the existing
evaluation setup. All 32 responses contribute to avg@32 (sample-mean accuracy),
not pass@32. The runner records data, reward, evaluator and model hashes and saves
generated text and individual scores before producing aggregate results.

In an environment with `vllm`, `transformers`, `torch`, `sympy` and `pylatexenc`:

```bash
python reinforce_pro_max/experiments/dapo_small/evaluate_aime25.py \
  --model /path/to/completed/model --output /path/to/evaluation
```

The script defaults to data and reward files beside itself. Add `--validate-only`
to check tokenization, repetition counts and grading without loading model weights
or generating responses. Standard evaluation uses four GPUs for tensor parallelism.

## Four-arm component comparison

`study-arms.json` is the authoritative current configuration. All four arms start
from Qwen2.5-Math-7B, use seed 42, and run 100 updates on the same training subset.
The shared objective is **unclipped token-level importance sampling**:

- Baseline: `advantage * ratio`.
- Pro: `causal_prefix_mask * advantage * ratio`.
- `ratio = exp(current_training_logprob - rollout_inference_logprob)` appears once.
- No PPO clipping, ICEPOP filtering or additional IS multiplier is used.

| Arm | Advantage processing | Trust-region mask |
| --- | --- | --- |
| baseline | RLOO + global normalization | None |
| max_only | Max sign-dependent scaling | None |
| pro_only | RLOO + global normalization | Causal prefix |
| pro_max | Max sign-dependent scaling | Causal prefix |

For valid token position t, Pro computes the geometric mean of the same ratio
from the first response token through t. It keeps the token when that value is
in `[0.8, 1.25]`, as explicitly selected by the user. The bounds are reciprocal
and symmetric in log space. The geometric-mean statistic follows the form of
[GSPO Eq. (7)](https://arxiv.org/html/2507.18071v2#S4.SS1), applied to causal prefixes;
GSPO's clipping widths are not adopted. Prefix accumulation uses float32. The mask is detached. Future tokens cannot affect earlier masks;
tokens may be accepted again after a rejected prefix. Padding masks still apply.
Loss reduction uses all valid response tokens, not only accepted tokens.

This follows TRM's masked IS surrogate structure, replacing its sequence mask
with a causal prefix mask. It is not an exact TRM reproduction: the mask criterion,
model, advantage processing and training budget differ. Max replaces the
baseline's global normalization with its own scaling, so the Max contrast measures
that complete advantage-processing change.

`training-source.tar.gz` contains the complete frozen training source for this
experiment, with no dependency on earlier run directories. `check_prefix_is.py`
checks loss values and gradients against the explicit formula. `study-plan.json`
records the complete fixed plan; its machine paths identify the training host.
`run_four_arm_study.py --run-dir /path/to/prepared/study` runs that plan.

Current run directory:
`/volume/pt-train/users/rbliu/github/test/runs/promax_dapo_fixed_20260925`.
All previous experimental run directories and superseded clipping/queue scripts
were removed at the user's request. Current results start from scratch.
All four arms use the same reward function and local AIME25 evaluation.
One seed and 100 updates per arm scope conclusions to this experiment.

### 2026-09-25：改为训练过程 avg@8

用户取消此后所有独立评测（Max-only、Pro-only、Pro Max 和 initial）；保留已完成 Baseline AIME25 结果。当前运行目录中的 `evaluation-policy.json` 控制跳过，训练配置与冻结训练源码不变。正在运行的旧 supervisor 不重启；其后续评测入口读取此配置并立即返回，不加载模型。兼容入口写 `SKIPPED.json`，其 `_SUCCESS` 内容明确为 `status=skipped, evaluated=false`，仅确认跳过命令成功，不能据此认定评测完成；分析程序显式排除这类目录。新启动的调度脚本直接跳过调用。历史评测输入哈希与新哈希记录在 `evaluation-policy-change.json`。

报告主指标改为 **online training avg@8**：每步32个训练 prompt，各采样8次，用相同奖励函数的二值 `score`（正确为1）计算每题采样正确率再平均。它是训练时当前策略的表现，不是最终模型在完整训练集上的重新评测，也不是 pass@8。`summarize_training_avg8.py RUN_DIR` 从已记录的全局 `score` 导出各组 `training_avg8.csv` 和根目录 `training_avg8_summary.json`。脚本核验本轮每步8个rank各32条响应的等权归约条件；不对不同微批次配置默认为精确样本均值。导出逐步曲线、前/后50步与全程平均，未满50步时使用已完成步数。未来评测跳过节点和最终分析会自动刷新；运行中也可只读日志重新汇总，不增加采样。

已完成 Baseline：544步，训练过程全程 avg@8 为24.0091%，最后50步为27.7344%；历史 AIME25 avg@32=5.625% 单独保留，不与训练指标混用。训练 reward、PPL Gap、mask 保留率和权重矩继续按原设置记录。

### 2026-09-25：停止 Max-only，切换 Pro-only

用户要求停止当前 Max-only，优先运行 Pro-only。Max-only 在第25步后停止，原始日志和指标保留，运行目录 `max_only/CANCELLED.json` 标记为用户取消，不计为完整实验。`queue-switch.json` 将后续队列设为 Pro-only → Pro Max，保留此前 Pro Max 授权；Baseline 不重跑。旧进程组退出并释放8卡后重新启动调度器。Pro-only 从相同初始1.5B模型开始，保持544步、CUDA Graph、区间[0.8,1.25]、单次current/rollout ratio、RLOO及全局归一化；后续独立评测继续跳过。

### 2026-09-25 最新队列：Pro-only → Max-only → Pro Max

用户重新指定上述顺序。当前 Pro-only 训练进程保持不变；停止的25步 Max-only 已移入运行目录 `archive/max_only_cancelled_step25`，正式 Max-only 将从同一初始模型重跑。调度器按 `queue-switch.json` 的列表顺序执行，而不是按原始 arms 字典顺序。由于运行中的旧调度器已读取旧队列，`queue-reload.json` 安排在 Pro-only 成功退出、写入训练退出码并汇总指标后交接调度器；仅替换父调度器，不终止训练进程。新调度器会跳过已成功完成的 Pro-only，接着启动 Max-only，再启动 Pro Max。交接逻辑已用隔离子进程验证锁释放和重新启动；无需用户手动操作。后续评测仍跳过。

### 2026-09-26：Qwen2.5-Math-7B 三组固定200步

用户确认使用之前的7B模型（非8B），顺序 Baseline → Pro-only → Max-only，不运行 Pro Max，不做独立评测。最终指令为每组固定200步，已取消中间讨论的 avg@8 达25%早停。新运行目录 `/volume/pt-train/users/rbliu/github/test/runs/promax_math7b_three_arm_200_20260926`。保持此前8卡/CUDA Graph、32 prompts × 8 responses、LR=3e-6、seed42、generation3072/context4096、阈值[0.8,1.25]、奖励函数和loss配置。`max_samples=6400` 对应200步；使用原固定训练文件的前6400条，三组相同，学习率调度总长度随200步预算计算。7B tokenizer核对最长prompt929，无超过1024的prompt。早停安装/取消发生于初始化、无已完成训练步；相应初始化日志归档，训练循环已恢复成上一轮同一源码。最终模型保存和avg@8/PPL Gap汇总保留。


### 论文逐方法对比图

运行 `python reinforce_pro_max/experiments/dapo_small/plot_paper_training.py`，从已完成的 `results/math15b_544/` 固定CSV生成三个独立矢量图：`tex/figures/training_{max_only,pro_only,pro_max}_vs_baseline.pdf`。参照 TRM v4 的左右指标布局、细实线和下方图例；每张只含 Baseline 和一个方法，左 PPL Gap、右训练avg@8，统一坐标范围与20步后向平滑。图表不包含进行中的7B数据；精确表格仍采用原始544步/最后50步数据。


### 7B 微批次调整（2026-09-26 16:13 北京时间）

用户要求将 micro_train_batch_size 从32降至16，Pro-only从头重启，随后Max-only同样使用16；全局train batch保持256，8卡梯度累积两次，其他启动参数不变。已完成Baseline保留microbatch32。失败Pro-only及旧计划/指纹保存在运行目录archive/pro_only_oom_micro32_20260926_161339；冻结计划指纹已更新。原实现的微批次token均值归约保留，因此记录此执行配置差异，不声称与原32微批次梯度严格相同。avg@8汇总按各组实际启动命令将微批次loss_call映射到更新步，验证8卡×2次×16响应，避免将200更新记成400步。合成检查覆盖32/16两种配置及未完成的下一步；原Baseline统计保持不变。
