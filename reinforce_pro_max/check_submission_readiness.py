"""Read-only checks for the current paper artifact and theorem inventory."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEX_MAIN = ROOT / "tex" / "main"
BUILD_LOG = Path("/tmp/openrlhf_paper_build/main.log")


def labeled_math_environments() -> list[tuple[str, str]]:
    text = "\n".join(p.read_text(encoding="utf-8") for p in TEX_MAIN.glob("*.tex"))
    pattern = re.compile(r"\\begin\{(theorem|proposition|corollary|lemma)\}(.*?)\\end\{\1\}", re.S)
    result = []
    for kind, body in pattern.findall(text):
        label = re.search(r"\\label\{([^}]+)\}", body)
        if label:
            result.append((kind, label.group(1)))
    return result


def main() -> int:
    if not BUILD_LOG.exists():
        raise SystemExit(f"missing build log: {BUILD_LOG}")
    log = BUILD_LOG.read_text(encoding="utf-8", errors="replace")
    output = re.search(r"Output written on .*? \((\d+) pages?,", log)
    page_markers = re.findall(r"\[(\d+)\]", log)
    appendix = re.search(r"main/appendix\.tex \[(\d+)\]", log)
    labels = labeled_math_environments()
    total_pages = int(output.group(1)) if output else None
    appendix_page = int(appendix.group(1)) if appendix else None
    print(f"pdf_pages={total_pages}")
    print(f"appendix_first_page={appendix_page}")
    print(f"main_pages_before_appendix={(appendix_page - 1) if appendix_page else None}")
    print(f"labeled_math_environments={len(labels)}")
    print(f"page_markers_seen={len(page_markers)}")
    if total_pages is None or appendix_page is None:
        return 2
    if len(labels) != 31:
        print("ERROR: expected 31 labeled mathematical environments")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
