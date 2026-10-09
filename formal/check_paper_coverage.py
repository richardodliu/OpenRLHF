"""Check the freshness of a statement inventory, not mathematical correctness.

Scope: explicit theorem/lemma/proposition/corollary/remark/example/definition/
assumption environments in the active ICLR TeX input closure. Inline prose,
displays outside those environments, and proofs require separate review.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENTRIES = ("tex/iclr2027_submission.tex",)
KINDS = "theorem|lemma|proposition|corollary|remark|example|definition|assumption"
BLOCK = re.compile(r"\\begin\{(" + KINDS + r")\}(.*?)\\end\{\1\}", re.S)
LABEL = re.compile(r"\\label(?:\[[^\]]*\])?\{([^}]+)\}")
INPUT = re.compile(r"\\(?:input|include)\{([^}]+)\}")


def uncomment(text: str) -> str:
    return re.sub(r"(?<!\\)((?:\\\\)*)%[^\n]*", r"\1", text)


def collect() -> dict[str, dict]:
    files: dict[Path, set[str]] = {}
    for entry in ENTRIES:
        base = (ROOT / entry).parent
        pending = [ROOT / entry]
        seen: set[Path] = set()
        while pending:
            path = pending.pop().resolve()
            if path in seen:
                continue
            seen.add(path)
            files.setdefault(path, set()).add(entry)
            for name in INPUT.findall(uncomment(path.read_text())):
                target = base / name
                pending.append(target if target.suffix else target.with_suffix(".tex"))
    result = {}
    for path in sorted(files):
        text = uncomment(path.read_text())
        source = str(path.relative_to(ROOT))
        counts: Counter = Counter()
        for match in BLOCK.finditer(text):
            kind, body = match.groups()
            counts[kind] += 1
            labels = LABEL.findall(body)
            local_id = labels[0] if labels else f"{kind}:{counts[kind]}"
            key = f"{source}::{local_id}"
            if key in result:
                raise ValueError(f"Duplicate statement identity: {key}")
            title = re.match(r"\s*\[([^\]]*)\]", body)
            result[key] = {
                "source": source,
                "line": text.count("\n", 0, match.start()) + 1,
                "kind": kind,
                "label": labels[0] if labels else None,
                "title": title.group(1) if title else "",
                "entries": sorted(files[path]),
                "statement_sha256": hashlib.sha256(match.group(0).encode()).hexdigest(),
            }
    return result


def dependency_hashes() -> dict[str, str]:
    paths = set((ROOT / "formal").glob("*.lean"))
    paths.update((ROOT / "tex").glob("*.tex"))
    paths.update((ROOT / "tex/main").glob("*.tex"))
    paths.update((ROOT / "tex/iclr2027").glob("*.tex"))
    paths.add(ROOT / "openrlhf/trainer/ppo_utils/experience_maker.py")
    paths.add(ROOT / "openrlhf/models/loss.py")
    paths.add(ROOT / "openrlhf/models/utils.py")
    paths.add(ROOT / "openrlhf/trainer/ppo_utils/replay_buffer.py")
    paths.add(ROOT / "openrlhf/trainer/ray/ppo_actor.py")
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def main() -> int:
    ledger = json.loads((ROOT / "formal/PAPER_COVERAGE.json").read_text())
    current = collect()
    errors = []
    if ledger.get("dependencies_sha256") != dependency_hashes():
        errors.append("TeX, Lean, or linked implementation changed; review assumptions and proofs")
    recorded = {row["id"]: row for row in ledger["statements"]}
    if len(recorded) != len(ledger["statements"]):
        errors.append("Duplicate ledger identities")
    for key in sorted(current.keys() - recorded.keys()):
        errors.append(f"Unlisted statement: {key}")
    for key in sorted(recorded.keys() - current.keys()):
        errors.append(f"Removed/renamed statement: {key}")
    for key in current.keys() & recorded.keys():
        for field, value in current[key].items():
            if recorded[key].get(field) != value:
                errors.append(f"Stale {field}: {key}")
        row = recorded[key]
        if row["review_status"] == "reviewed_finite_scope" and not row.get("lean_evidence"):
            errors.append(f"Missing proof evidence: {key}")
        for evidence in row.get("lean_evidence", []):
            p = ROOT / evidence["file"]
            if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != evidence["file_sha256"]:
                errors.append(f"Changed Lean evidence: {key}: {p}")
            elif not re.search(r"\btheorem\s+" + re.escape(evidence["theorem"]) + r"\b", p.read_text()):
                errors.append(f"Missing theorem: {key}: {evidence['theorem']}")
    print("explicit_statement_blocks=", len(current))
    print("review_status=", dict(Counter(row["review_status"] for row in recorded.values())))
    print("This checks inventory freshness only, not semantic coverage, proof validity, or ICLR readiness.")
    for error in errors:
        print("ERROR:", error)
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
