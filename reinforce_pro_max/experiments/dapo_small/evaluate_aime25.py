"""Standalone AIME25 avg@32 evaluation using the unchanged training reward function."""
import argparse
import collections
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import importlib.util

HERE = Path(__file__).resolve().parent
MATH = ["aime25"]


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def atomic_json(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def atomic_rows(path, rows):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def init_grader(reward_file):
    global grader
    spec = importlib.util.spec_from_file_location("evaluation_reward", reward_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    grader = module.reward_func


def grade(row):
    row = dict(row)
    try:
        result = grader([row["generated_text"]], [""], [row["answer"]])
        reward = result["rewards"].item()
        row.update(reward=reward, correctness=bool(reward == 1.0),
                   formatted=bool(result["extra_logs"]["formats"].item()), grading_error=None)
    except Exception as exc:
        row.update(reward=None, correctness=None, formatted=None, grading_error=repr(exc))
    return row


def load_inputs(data, tokenizer):
    rows = []
    repetitions = collections.Counter()
    for record in map(json.loads, data.read_text().splitlines()):
        messages = record["messages"]
        assert messages[0]["role"] == "system"
        prompt = tokenizer.apply_chat_template(messages[1:], tokenize=False, add_generation_prompt=True)
        ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if len(ids) >= 4096:
            raise ValueError(f"Prompt cannot fit without truncation: {record['source_row_index']}")
        question_id = hashlib.sha256(messages[-1]["content"].encode()).hexdigest()
        repetition = repetitions[question_id]
        repetitions[question_id] += 1
        rows.append(dict(input_file=data.name, row_index=record["source_row_index"], data_source="aime25",
                         question_sha256=question_id, repetition_index=repetition,
                         prompt=prompt, prompt_tokens=len(ids), max_new_tokens=min(3072, 4096-len(ids)),
                         answer=record["answer"]))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=HERE / "aime25.jsonl")
    parser.add_argument("--reward-file", type=Path, default=HERE / "mathverify_dapo.py")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--suite", choices=["math"], default="math")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    import importlib.metadata
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    inputs = load_inputs(args.data, tokenizer)
    if args.suite == "math":
        inputs = [r for r in inputs if r["data_source"] in MATH]
    repeats = collections.Counter(r["question_sha256"] for r in inputs)
    assert len(inputs) == 960 and len(repeats) == 30 and set(repeats.values()) == {32}, repeats
    counts = dict(collections.Counter(row["data_source"] for row in inputs))
    config = dict(model=str(args.model), template="own", remove_system=True,
                  temperature=1.0, top_p=0.95, n=1, seed=42, sampling_seed_policy="42 + index in selected evaluation rows",
                  max_model_len=4096, max_tokens=3072,
                  long_prompt_policy="preserve full prompt; reduce output budget to fit 4096",
                  grader="training reward_func; correctness = reward == 1", tensor_parallel_size=4, suite=args.suite,
                  chunk_rows=256, rows=counts, total_rows=len(inputs),
                  shortened_output_budgets=sum(r["max_new_tokens"] < 3072 for r in inputs),
                  file_sha256={"data": digest(args.data), "reward": digest(args.reward_file)},
                  runner_sha256=digest(__file__),
                  versions={p: importlib.metadata.version(p) for p in
                            ["torch", "vllm", "transformers", "pylatexenc", "sympy"]})
    if args.validate_only:
        init_grader(args.reward_file)
        for text, answer, expected in [(r"\boxed{2}", "2", True), (r"\boxed{3}", "2", False),
                                       ("Answer: 2", "2", True), (r"Answer: \boxed{2}", "2", True), (r"\boxed{\frac{1}{2}}", "0.5", True)]:
            result = grade(dict(generated_text=text, answer=answer))
            assert result["correctness"] == expected, result
        for row in inputs:
            for text in ["Answer: " + row["answer"], "\\boxed{" + row["answer"] + "}"]:
                assert grade(dict(generated_text=text, answer=row["answer"]))["correctness"] is True
        atomic_json(args.output / "preflight.json", config)
        print(json.dumps(dict(status="validated", rows=counts, shortened_output_budgets=config["shortened_output_budgets"])))
        return
    # Only run on a completely saved model, never a partially written directory.
    index_path = args.model / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        for name in set(index["weight_map"].values()):
            assert (args.model / name).is_file(), name
    else:
        assert (args.model / "model.safetensors").is_file(), "No complete safetensors model"
    config["model_sha256"] = {p.name: digest(p) for p in sorted(args.model.iterdir()) if p.is_file()}
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == config, "Evaluation inputs changed; use a fresh output directory"
    else:
        atomic_json(manifest_path, config)
    chunks = args.output / "chunks"
    chunks.mkdir(exist_ok=True)
    llm = None
    with multiprocessing.get_context("spawn").Pool(8, initializer=init_grader, initargs=(str(args.reward_file),)) as pool:
        for start in range(0, len(inputs), 256):
            batch = inputs[start:start+256]
            decoded = chunks / f"{start:06d}.decoded.jsonl"
            scored = chunks / f"{start:06d}.scored.jsonl"
            if scored.exists():
                saved = [json.loads(line) for line in scored.open()]
                assert len(saved) == len(batch)
                assert [(r["input_file"], r["row_index"]) for r in saved] == [(r["input_file"], r["row_index"]) for r in batch]
                continue
            if decoded.exists():
                records = [json.loads(line) for line in decoded.open()]
                assert len(records) == len(batch)
            else:
                from vllm import LLM, SamplingParams
                if llm is None:
                    llm = LLM(model=str(args.model), tensor_parallel_size=4, max_model_len=4096,
                              gpu_memory_utilization=0.85, seed=42, max_num_seqs=128, enforce_eager=True)
                parameters = [SamplingParams(temperature=1., top_p=.95, max_tokens=r["max_new_tokens"],
                                             n=1, seed=42+start+i) for i, r in enumerate(batch)]
                outputs = llm.generate([r["prompt"] for r in batch], parameters)
                assert len(outputs) == len(batch)
                records = []
                for record, output in zip(batch, outputs):
                    result = output.outputs[0]
                    records.append(dict(record, generated_text=result.text, generated_tokens=len(result.token_ids),
                                        finish_reason=result.finish_reason, stop_reason=result.stop_reason))
                atomic_rows(decoded, records)
            results = pool.map_async(grade, records, chunksize=1).get(timeout=900)
            atomic_rows(scored, results)
            errors = [r for r in results if r["grading_error"]]
            print(json.dumps(dict(completed_rows=start+len(batch), total_rows=len(inputs), grading_errors=len(errors))), flush=True)
    all_rows = []
    for path in sorted(chunks.glob("*.scored.jsonl")):
        all_rows.extend(json.loads(line) for line in path.open())
    assert len(all_rows) == len(inputs)
    assert [(r["input_file"], r["row_index"]) for r in all_rows] == [(r["input_file"], r["row_index"]) for r in inputs]
    grouped = collections.defaultdict(list)
    for row in all_rows:
        grouped[row["data_source"]].append(row)
    summary = {}
    for name, group in grouped.items():
        errors = sum(r["grading_error"] is not None for r in group)
        summary[name] = dict(rows=len(group), unique_questions=len({r["question_sha256"] for r in group}),
                             correct=sum(r["correctness"] is True for r in group), grading_errors=errors,
                             accuracy=None if errors else sum(r["correctness"] for r in group)/len(group),
                             truncated=sum(r["finish_reason"] == "length" for r in group),
                             mean_generated_tokens=sum(r["generated_tokens"] for r in group)/len(group))
    error_count = sum(s["grading_errors"] for s in summary.values())
    report = dict(benchmarks=summary, total_rows=len(all_rows), grading_errors=error_count,
                  math_macro_accuracy=None if error_count else sum(summary[k]["accuracy"] for k in MATH)/len(MATH),

                  note="Sample-mean correctness; repeated rows are retained. Not pass@32. Generation cap differs from LUFFY default.")
    atomic_json(args.output / "results.json", report)
    atomic_rows(args.output / "predictions.jsonl", all_rows)
    if error_count:
        raise RuntimeError(f"{error_count} grading errors; results remain incomplete")
    atomic_json(args.output / "_SUCCESS", dict(rows=len(all_rows), results_sha256=digest(args.output / "results.json")))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
