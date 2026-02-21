#!/usr/bin/env python3
"""
Sanity-check structure of tex/literature.md.

We enforce a uniform per-entry template under:
  - "## Papers（文件夹内 `.tex`）"
  - "## Technical Reports（`.md`）"

Each entry ("### ...") must contain headings ("#### ...") in one of two forms:
  1) Required headings only (exact order)
  2) Required headings + optional OpenRLHF mapping (required order, optional last)

Run:
  python tex/check_literature_index.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys


REQUIRED_HEADINGS = [
    "问题设定",
    "承重公式",
    "算法步骤",
    "关键定理",
    "理论结论与证明过程入口",
]
OPTIONAL_HEADING = "与 OpenRLHF 映射（可选）"


@dataclass(frozen=True)
class Entry:
    section: str
    title: str
    start_line: int
    headings: list[tuple[str, int]]  # (heading, lineno)


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _find_h2(lines: list[str], exact: str) -> int:
    for i, line in enumerate(lines):
        if line.strip() == exact:
            return i
    return -1


def _parse_entries(lines: list[str], *, start_i: int, section_name: str) -> list[Entry]:
    entries: list[Entry] = []
    cur_title: str | None = None
    cur_start: int | None = None
    cur_headings: list[tuple[str, int]] = []

    def flush() -> None:
        nonlocal cur_title, cur_start, cur_headings
        if cur_title is None or cur_start is None:
            return
        entries.append(
            Entry(
                section=section_name,
                title=cur_title,
                start_line=cur_start,
                headings=list(cur_headings),
            )
        )
        cur_title = None
        cur_start = None
        cur_headings = []

    i = start_i + 1
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("## "):
            break

        if stripped.startswith("### "):
            flush()
            cur_title = stripped.removeprefix("### ").strip()
            cur_start = i + 1  # 1-based
            cur_headings = []
            i += 1
            continue

        if cur_title is not None and stripped.startswith("#### "):
            head = stripped.removeprefix("#### ").strip()
            cur_headings.append((head, i + 1))

        i += 1

    flush()
    return entries


def _validate_entry(e: Entry) -> list[str]:
    errs: list[str] = []
    heads = [h for (h, _lineno) in e.headings]

    if not heads:
        errs.append("missing all template headings (no '#### ...' found)")
        return errs

    unknown = [h for h in heads if h not in REQUIRED_HEADINGS and h != OPTIONAL_HEADING]
    if unknown:
        errs.append(f"unknown headings: {unknown}")

    # Duplicates
    seen: set[str] = set()
    dupes: list[str] = []
    for h in heads:
        if h in seen and h not in dupes:
            dupes.append(h)
        seen.add(h)
    if dupes:
        errs.append(f"duplicate headings: {dupes}")

    allowed_1 = REQUIRED_HEADINGS
    allowed_2 = REQUIRED_HEADINGS + [OPTIONAL_HEADING]
    if heads != allowed_1 and heads != allowed_2:
        errs.append(f"heading order must be exactly {allowed_1} (+ optional last); got {heads}")

    return errs


def main() -> int:
    path = Path("tex/literature.md")
    if not path.exists():
        print("error: tex/literature.md not found", file=sys.stderr)
        return 2

    lines = _read_lines(path)
    papers_i = _find_h2(lines, "## Papers（文件夹内 `.tex`）")
    reports_i = _find_h2(lines, "## Technical Reports（`.md`）")
    if papers_i < 0 or reports_i < 0:
        print("error: missing expected section headers in tex/literature.md", file=sys.stderr)
        print("  expected: '## Papers（文件夹内 `.tex`）' and '## Technical Reports（`.md`）'", file=sys.stderr)
        return 2

    entries = []
    entries.extend(_parse_entries(lines, start_i=papers_i, section_name="Papers"))
    entries.extend(_parse_entries(lines, start_i=reports_i, section_name="Technical Reports"))

    if not entries:
        print("error: no entries found under Papers/Technical Reports", file=sys.stderr)
        return 2

    bad = 0
    for e in entries:
        errs = _validate_entry(e)
        if not errs:
            continue
        bad += 1
        print(f"[{e.section}] '{e.title}' (starts at tex/literature.md:{e.start_line})")
        for msg in errs:
            print(f"  - {msg}")

    if bad:
        print(f"FAIL: {bad} / {len(entries)} entries violate the template", file=sys.stderr)
        return 1

    print(f"OK: {len(entries)} entries follow the template")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

