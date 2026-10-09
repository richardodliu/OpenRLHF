"""Check paper source closure and optional build output, not scientific readiness."""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = re.compile(r"\\(?:input|include)\{([^}]+)\}")
LABEL = re.compile(r"\\label(?:\[[^\]]*\])?\{([^}]+)\}")
REFERENCE = re.compile(r"\\(?:[Cc]ref|ref|eqref|autoref)\*?\{([^}]+)\}")
CITATION = re.compile(r"\\(?:[Cc]ite\w*|nocite)\*?(?:\[[^\]]*\]){0,2}\{([^}]+)\}")
BIBLIOGRAPHY = re.compile(r"\\bibliography\{([^}]+)\}")


def uncomment(text: str) -> str:
    # An odd run of preceding backslashes escapes a percent sign.
    return re.sub(r"(?<!\\)((?:\\\\)*)%[^\n]*", r"\1", text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry", default="tex/iclr2027_submission.tex")
    parser.add_argument("--build-dir", type=Path, help="Optional directory containing this entry's PDF, log and aux")
    args = parser.parse_args()
    entry = (ROOT / args.entry).resolve()
    base = entry.parent
    pending = [entry]
    sources: dict[Path, str] = {}
    errors: list[str] = []
    while pending:
        path = pending.pop()
        if path in sources:
            continue
        if not path.is_file():
            errors.append(f"Missing TeX input: {path}")
            continue
        text = uncomment(path.read_text(encoding="utf-8"))
        sources[path] = text
        for name in INPUT.findall(text):
            target = base / name
            if not target.suffix:
                target = target.with_suffix(".tex")
            pending.append(target.resolve())

    joined = "\n".join(sources.values())
    labels = Counter(LABEL.findall(joined))
    refs = {item.strip() for group in REFERENCE.findall(joined) for item in group.split(",")}
    cites = {item.strip() for group in CITATION.findall(joined) for item in group.split(",")}
    for label, count in labels.items():
        if count > 1:
            errors.append(f"Duplicate label: {label} ({count})")
    errors.extend(f"Missing label: {label}" for label in sorted(refs - labels.keys()) if "#" not in label)

    bib_keys: set[str] = set()
    bib_paths: set[Path] = set()
    for group in BIBLIOGRAPHY.findall(joined):
        for name in group.split(","):
            path = (base / name.strip()).with_suffix(".bib")
            bib_paths.add(path)
            if not path.is_file():
                errors.append(f"Missing bibliography: {path}")
            else:
                bib_keys.update(re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", path.read_text(encoding="utf-8")))
    errors.extend(f"Missing citation: {key}" for key in sorted(cites - bib_keys - {"*"}) if "#" not in key)

    if args.build_dir:
        paths = {suffix: args.build_dir / f"{entry.stem}.{suffix}" for suffix in ("log", "aux", "pdf")}
        for path in paths.values():
            if not path.is_file():
                errors.append(f"Missing build artifact: {path}")
        if all(path.is_file() for path in paths.values()):
            log = paths["log"].read_text(encoding="utf-8", errors="replace")
            bad = re.compile(
                r"undefined|multiply defined|destination with the same identifier|"
                r"Rerun to|Please rerun|Label\(s\) may have changed|Overfull"
            )
            errors.extend(f"Build issue: {line.strip()}" for line in log.splitlines() if bad.search(line))
            if "Output written on" not in log:
                errors.append("Build log does not report a completed PDF")
            for path in set(sources) | bib_paths:
                if path.is_file() and path.stat().st_mtime_ns > paths["pdf"].stat().st_mtime_ns:
                    errors.append(f"Build PDF is older than source: {path}")
            if entry.name == "iclr2027_submission.tex":
                page_count = re.search(r"Output written on\s+.*?\((\d+)\s+pages?,", log, re.S)
                if page_count is None or not 1 <= int(page_count.group(1)) <= 30:
                    errors.append("Missing total page count or paper exceeds the user-specified 30-page limit")
                else:
                    print(f"total_pages={page_count.group(1)} limit=30")
                aux = paths["aux"].read_text(encoding="utf-8", errors="replace")
                boundary = re.search(r"\\newlabel\{sec:iclr-main-end\}\{\{[^}]*\}\{(\d+)\}", aux)
                if boundary is None or int(boundary.group(1)) != 9:
                    errors.append("Main text must end on page nine")
                else:
                    print(f"main_text_end_page={boundary.group(1)}")

    print(f"entry={entry.relative_to(ROOT) if entry.is_relative_to(ROOT) else entry}")
    print(f"tex_files={len(sources)} labels={len(labels)} cited_keys={len(cites)}")
    for error in errors:
        print(f"ERROR: {error}")
    print("This checks dependencies and references, not theorem correctness or submission readiness.")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
