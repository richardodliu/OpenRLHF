#!/usr/bin/env python3
"""
Sanity-check structure of tex/literature.md.

We enforce a uniform per-entry template under:
  - "## Papers（文件夹内 `.tex`）"
  - "## Technical Reports（`.md`）"

Each entry ("### ...") must contain headings ("#### ...") in one of two forms:
  1) Required headings only (exact order)
  2) Required headings + optional OpenRLHF mapping (required order, optional last)

Additionally:
  - Math expressions must not be written as Markdown code spans (backticks).
  - Unicode math symbols that tend to break LaTeX/KaTeX rendering should be avoided
    (prefer LaTeX commands like \\pi, \\mu, \\sum, etc.).

Run:
  python tex/check_literature_index.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sys


REQUIRED_HEADINGS = [
    "问题设定",
    "承重公式",
    "算法步骤",
    "关键定理",
    "理论结论与证明过程入口",
]
OPTIONAL_HEADING = "与 OpenRLHF 映射（可选）"

# Targeted Unicode math symbols that should not appear verbatim in tex/literature.md.
# (This file is mostly Chinese; we keep this list tight to avoid false positives.)
FORBIDDEN_UNICODE_MATH = set("πμρθαβδεηχΔΣ∑∏≤≥≠∇∝≈∞→")

# Heuristic: if a code span contains any of these, it's almost certainly math.
MATH_IN_CODE_RE = re.compile(
    r"(\\(pi|mu|rho|theta|alpha|beta|delta|epsilon|eta|chi|sum|prod|exp|log|cdot|times|leq|geq|ne|approx|propto|infty|mathcal|bar|hat|tilde|Delta|Sigma|nabla)\b)"
    r"|\^"
    r"|(?<![=])=(?![=])"
    r"|\bD_[A-Za-z]+\b"
    r"|\bKL\s*\("
    r"|\bargmax_"
    r"|\bmin\s*\("
    r"|\bmax\s*\("
    r"|\bexp\s*\("
    r"|\blog\s*\("
)


def _parse_inline_code_spans(text: str) -> list[tuple[int, str]]:
    """Return (start_line, content) for each single-backtick inline code span."""
    spans: list[tuple[int, str]] = []
    in_code = False
    buf: list[str] = []
    line = 1
    start_line: int | None = None

    for ch in text:
        if ch == "\n":
            line += 1
        if ch == "`":
            if in_code:
                spans.append((start_line or line, "".join(buf)))
                buf.clear()
                in_code = False
                start_line = None
            else:
                in_code = True
                start_line = line
                buf.clear()
            continue
        if in_code:
            buf.append(ch)

    # If the file has an unmatched backtick, treat it as a failure signal.
    if in_code:
        spans.append((start_line or line, "<UNTERMINATED CODE SPAN>"))

    return spans


def _is_allowed_code_span(content: str) -> bool:
    c = content.strip()

    if c.startswith("--"):
        return True

    # File paths / repo references.
    if c.startswith(("tex/", "openrlhf/", "examples/")):
        return True
    if "openrlhf/" in c or "tex/" in c:
        return True
    if re.search(r"\.(py|tex|md|sh)(:\d+)?$", c):
        return True
    if re.search(r"\.(py|tex|md):\d+", c):
        return True

    # Markdown heading references.
    if c.startswith("#"):
        return True

    # TeX structural commands we mention as strings, plus custom macros that are not
    # intended to render in KaTeX.
    if c.startswith("\\"):
        if c.startswith(
            (
                "\\section",
                "\\subsection",
                "\\subsubsection",
                "\\paragraph",
                "\\label",
                "\\Cref",
                "\\cref",
                "\\ref",
                "\\begin",
                "\\end",
                "\\method",
            )
        ):
            return True

    # Python-ish comparisons.
    if "==" in c and ('"' in c or "'" in c):
        return True

    # Function / method names like Foo.bar() / Foo.bar(arg)
    if re.fullmatch(r"[A-Za-z0-9_]+\.[A-Za-z0-9_]+\(.*\)", c):
        return True

    return False


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

    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    # 1) Guardrail: avoid targeted unicode math symbols (prefer LaTeX commands).
    bad_unicode: list[tuple[int, str]] = []
    for i, line in enumerate(lines, start=1):
        for ch in FORBIDDEN_UNICODE_MATH:
            if ch in line:
                bad_unicode.append((i, ch))
    if bad_unicode:
        uniq = sorted({ch for (_i, ch) in bad_unicode})
        print("FAIL: tex/literature.md contains forbidden unicode math symbols:", file=sys.stderr)
        print(f"  symbols: {''.join(uniq)}", file=sys.stderr)
        for i, ch in bad_unicode[:30]:
            print(f"  - tex/literature.md:{i}: contains '{ch}'", file=sys.stderr)
        if len(bad_unicode) > 30:
            print("  ...", file=sys.stderr)
        return 1

    # 2) Guardrail: math expressions must not be written as markdown code spans.
    bad_spans: list[tuple[int, str]] = []
    for ln, content in _parse_inline_code_spans(text):
        if content == "<UNTERMINATED CODE SPAN>":
            bad_spans.append((ln, content))
            continue
        if _is_allowed_code_span(content):
            continue
        if MATH_IN_CODE_RE.search(content):
            bad_spans.append((ln, content))
    if bad_spans:
        print("FAIL: tex/literature.md contains math-like markdown code spans (use $...$ instead):", file=sys.stderr)
        for ln, content in bad_spans[:40]:
            print(f"  - tex/literature.md:{ln}: `{content}`", file=sys.stderr)
        if len(bad_spans) > 40:
            print("  ...", file=sys.stderr)
        return 1
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

    print(f"OK: {len(entries)} entries follow the template (and no math-in-code spans found)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
