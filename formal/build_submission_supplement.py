"""Build a portable, deterministic supplement from the current paper's proof closure.

Run from any directory with Python 3.10+. The output ZIP must not already exist.
This packages existing proofs; it neither changes their statements nor certifies
the correspondence between Lean definitions, manuscript prose, and Python code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path

import check_paper_coverage as coverage

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_FILES = (
    "openrlhf/models/loss.py",
    "openrlhf/models/utils.py",
    "openrlhf/trainer/ppo_utils/experience_maker.py",
    "openrlhf/trainer/ppo_utils/replay_buffer.py",
    "openrlhf/trainer/ray/ppo_actor.py",
)

README = """# REINFORCE Pro Max: mathematical supplement

This supplement contains the Lean proofs used by the accompanying anonymous
paper, their local dependencies, and the implementation files used to specify
the objective. Max assigns different scales to positive and negative advantages;
Pro recomputes a causal prefix mask from the current-to-rollout ratios and
retains a single token-level ratio. The mask is detached during differentiation.

## Reproduce the proofs

Install Lean's `elan` toolchain manager, Python 3.10 or later, and Git. The
`formal/lean-toolchain` file selects Lean 4.19.0; `lake-manifest.json` pins Mathlib
and its dependencies. From this extracted directory run:

```sh
python verify_manifest.py
cd formal
lake exe cache get
lake build PaperAudit
```

The first run downloads the pinned toolchain and dependency cache. No GPU,
training data, Python ML environment, or repository-specific path is needed.
On machines with limited resources, set `LEAN_NUM_THREADS=2` before the Lake
commands. The proof audit succeeds only if the imported project theorems depend
on `propext`, `Classical.choice`, and `Quot.sound` alone. A proof using `sorry`
or an additional axiom makes the audit fail. The audit's theorem count includes
supporting lemmas and is not the paper's contribution count.

## Find the relevant proofs

`statement-map.json` maps each explicit mathematical environment to its source
location, statement hash, review scope, and Lean evidence. It is an index for inspecting the
correspondence, not an automated equivalence proof. The source texts in
`tex/main/` accompany `paper.pdf`; they are supplied for inspection and do not
include the full LaTeX build environment.

| Mathematical role | Starting modules in `formal/` |
| --- | --- |
| Max scales, actual safeguards, and conditional ascent | AdaptiveMechanism, NormalizationSafeguards, BinaryAscent, PromptAscent |
| Current/rollout prefix mask and retained-weight moments | PrefixDefinition, PrefixGateMoments, PrefixGateExample |
| Conditional sequence TV, Adaptive and Mixed reward errors | TreeTV, TreeKL, AdaptiveBound |
| Actual objective, RLOO expectation, and EOS/token reduction | TwoPolicyObjective, TreeSurrogate, BatchDenominator, BatchReduction |
| Positive reward regime and failure boundary | Exact derivations in the paper; `checks/check_two_policy_examples.py` |

The population reward theorem has formal components for its pointwise algebra;
its full expectation argument is supplied in the manuscript. The rewritten
two-policy examples likewise have analytic proofs, supplemented by exact
rational checks on the parameter choices in `checks/check_two_policy_examples.py`.
Run those checks with `python checks/check_two_policy_examples.py`.

The Lean sources are unchanged copies of the paper's evidence modules and their
transitive local imports. Some supporting modules also contain earlier results
used by these proofs. `PaperAudit.lean` imports the evidence roots and checks
their axiom dependencies. The mathematical assumptions remain explicit in the
Lean theorem signatures and in the paper. Kernel checking verifies those
formal statements; the natural-language and implementation correspondence is
documented by the statement map and the accompanying sources.

`openrlhf/` contains five implementation reference files, not a stand-alone
training installation. They specify the signal construction, current/rollout `token_is` objective,
causal prefix mask, and loss reduction. Historical PPO branches remain available. The mathematical reproduction commands above do
not execute those files or establish empirical training performance. PolicyLoss
uses an active-token mean per invocation; dynamic batching aggregates these
means with response-count weights. Appendix D.2 gives the exact difference from
the full-batch token mean used by the population reward theorem. The identities
are in `BatchReduction.microbatch_token_gap` and
`BatchReduction.sample_weighted_response_means`. Existing source notices and the
Apache-2.0 license are retained in `LICENSE`.

`manifest.json` records payload hashes and dependency pins. Its own hash is not
self-referenced. `verify_manifest.py` checks the payload; generated `.lake`
build products are excluded from the manifest. The ZIP contains no toolchain,
dependency cache, training output, Git history, or local maintenance logs.
"""

VERIFY = '''"""Verify packaged sources before compiling; this is not a proof checker."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = json.loads((root / "manifest.json").read_text())
failures = []
for name, expected in manifest["files_sha256"].items():
    path = root / name
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        failures.append(name)
if failures:
    raise SystemExit("Missing or changed payload files: " + ", ".join(failures))
print(f"Verified {len(manifest['files_sha256'])} payload files. Run lake build PaperAudit to check proofs.")
'''


def encode_json(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def proof_closure(roots: list[str]) -> list[str]:
    pending = list(roots)
    seen: set[str] = set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        path = ROOT / "formal" / (name.replace(".", "/") + ".lean")
        source = path.read_text()
        seen.add(name)
        for line in re.findall(r"^import\s+([^\n]+)", source, re.M):
            for dependency in line.split("--", 1)[0].split():
                local = ROOT / "formal" / (dependency.replace(".", "/") + ".lean")
                if local.is_file():
                    pending.append(dependency)
                elif not dependency.startswith(("Mathlib", "Lean", "Std", "Batteries")):
                    raise ValueError(f"Unrecognized external import {dependency} in {name}")
    return sorted(seen)


def payload() -> dict[str, bytes]:
    if coverage.main():
        raise ValueError("Refresh and review the paper coverage ledger before packaging")
    ledger = json.loads((ROOT / "formal/PAPER_COVERAGE.json").read_text())
    roots = sorted({Path(e["file"]).stem for row in ledger["statements"] for e in row["lean_evidence"]})
    modules = proof_closure(roots)
    paths = [f"formal/{name}.lean" for name in modules]
    paths += ["formal/lean-toolchain", "formal/lake-manifest.json", "LICENSE", *REFERENCE_FILES]
    paths += sorted({row["source"] for row in ledger["statements"]})
    files = {name: (ROOT / name).read_bytes() for name in paths}
    files["checks/check_two_policy_examples.py"] = (
        ROOT / "reinforce_pro_max/check_two_policy_examples.py"
    ).read_bytes()
    files["paper.pdf"] = (ROOT / "iclr2027_submission.pdf").read_bytes()
    files["README.md"] = README.encode()
    files["verify_manifest.py"] = VERIFY.encode()
    fields = ("id", "source", "line", "kind", "label", "title", "statement_sha256",
              "review_status", "review_note", "lean_evidence")
    files["statement-map.json"] = encode_json([{k: row[k] for k in fields} for row in ledger["statements"]])
    audit = (ROOT / "formal/AxiomAudit.lean").read_text().split("run_cmd do\n", 1)[1]
    files["formal/PaperAudit.lean"] = (
        "".join(f"import {name}\n" for name in roots) + "\nrun_cmd do\n" + audit
    ).encode()
    config = '''name = "reinforce_promax_audit"
version = "0.1.0"
defaultTargets = ["PaperAudit"]

[[require]]
name = "mathlib"
git = "https://github.com/leanprover-community/mathlib4.git"
rev = "v4.19.0"
'''
    config += "".join(f'\n[[lean_lib]]\nname = "{name}"\n' for name in [*modules, "PaperAudit"])
    files["formal/lakefile.toml"] = config.encode()
    # Maintenance paths and private identity markers must not enter this package.
    for name, content in files.items():
        if name.endswith(".pdf"):
            continue
        if re.search(rb"/volume/|/home/|rbliu|183\.242\.150\.6", content):
            raise ValueError(f"Local identity or environment path in {name}; review before packaging")
    files["manifest.json"] = encode_json({
        "format_version": 1,
        "lean_toolchain": files["formal/lean-toolchain"].decode().strip(),
        "dependency_lock": json.loads(files["formal/lake-manifest.json"]),
        "evidence_roots": roots,
        "local_module_closure": modules,
        "explicit_statement_count": len(ledger["statements"]),
        "files_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in sorted(files.items())},
    })
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="New ZIP path; existing output is never overwritten")
    args = parser.parse_args()
    files = payload()
    # Stable timestamps, ordering, and permissions avoid leaking local filesystem metadata.
    with zipfile.ZipFile(args.output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, content)
    print(f"Wrote {len(files)} files to {args.output}")


if __name__ == "__main__":
    main()
