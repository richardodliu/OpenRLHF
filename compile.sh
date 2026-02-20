#!/usr/bin/env bash
# LaTeX compilation script with BibTeX + auto git push.
# Full compilation: pdflatex -> bibtex -> pdflatex -> pdflatex (rerun if needed).
#
# Notes:
# - Default behavior is to commit and push all repo changes (not only `tex/`).
# - Set `PUSH=0` to disable git commit/push (compile-only dry run).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEX_DIR="${SCRIPT_DIR}/tex"
MAIN="main"

# Allow disabling push for local debugging.
PUSH="${PUSH:-1}"
COMMIT_MSG="${COMMIT_MSG:-update}"

# TeX cache paths (avoid permission errors when mktexpk tries writing under /root).
TEX_CACHE_ROOT="${TEX_CACHE_ROOT:-/tmp/openrlhf_tex_cache}"
export TEXMFVAR="${TEXMFVAR:-${TEX_CACHE_ROOT}/texmf-var}"
export TEXMFCONFIG="${TEXMFCONFIG:-${TEX_CACHE_ROOT}/texmf-config}"
export VARTEXFONTS="${VARTEXFONTS:-${TEX_CACHE_ROOT}/texfonts}"
mkdir -p "${TEXMFVAR}" "${TEXMFCONFIG}" "${VARTEXFONTS}"

# Build artifacts go to /tmp by default to avoid polluting the repo working tree.
BUILD_DIR="${BUILD_DIR:-/tmp/openrlhf_paper_build}"
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"

run_pdflatex() {
  local pass="$1"
  echo "[${pass}] pdflatex..."
  # -halt-on-error: fail fast.
  # -file-line-error: easier debugging.
  # -output-directory: keep aux/log out of the repo.
  pdflatex -interaction=nonstopmode -halt-on-error -file-line-error \
    -output-directory="${BUILD_DIR}" "${MAIN}.tex" \
    >"${BUILD_DIR}/pdflatex_${pass}.out" 2>&1
}

run_bibtex() {
  echo "[bibtex] bibtex..."
  # When using -output-directory, bibtex runs against the aux file in BUILD_DIR.
  # BIBINPUTS must include TEX_DIR so `main/reference.bib` can be found.
  (cd "${BUILD_DIR}" && BIBINPUTS=".:${TEX_DIR}:" bibtex "${MAIN}") \
    >"${BUILD_DIR}/bibtex.out" 2>&1
}

needs_rerun() {
  local log_file="${BUILD_DIR}/${MAIN}.log"
  grep -qF "Rerun to get cross-references right" "${log_file}" \
    || grep -qF "Rerun to get citations correct" "${log_file}" \
    || grep -qF "Rerun to get outlines right" "${log_file}"
}

fail_on_warnings() {
  local log_file="${BUILD_DIR}/${MAIN}.log"
  local found=0

  if grep -nE '^LaTeX Warning:' "${log_file}" >/dev/null; then
    echo "ERROR: LaTeX warnings found in ${log_file}:"
    grep -nE '^LaTeX Warning:' "${log_file}" || true
    found=1
  fi

  if grep -nE '^Package .* Warning:' "${log_file}" >/dev/null; then
    echo "ERROR: Package warnings found in ${log_file}:"
    grep -nE '^Package .* Warning:' "${log_file}" || true
    found=1
  fi

  if grep -nF 'Overfull \hbox' "${log_file}" >/dev/null; then
    echo "ERROR: Overfull \\hbox found in ${log_file}:"
    grep -nF 'Overfull \hbox' "${log_file}" || true
    found=1
  fi

  if grep -nF 'Underfull \hbox' "${log_file}" >/dev/null; then
    echo "ERROR: Underfull \\hbox found in ${log_file}:"
    grep -nF 'Underfull \hbox' "${log_file}" || true
    found=1
  fi

  if grep -nF 'pdfTeX warning' "${log_file}" >/dev/null; then
    echo "ERROR: pdfTeX warnings found in ${log_file}:"
    grep -nF 'pdfTeX warning' "${log_file}" || true
    found=1
  fi

  if grep -nF 'Warning--' "${BUILD_DIR}/bibtex.out" >/dev/null; then
    echo "ERROR: BibTeX warnings found in ${BUILD_DIR}/bibtex.out:"
    grep -nF 'Warning--' "${BUILD_DIR}/bibtex.out" || true
    found=1
  fi

  if [[ "${found}" -ne 0 ]]; then
    echo "Build failed due to warnings. See ${BUILD_DIR} for logs."
    exit 1
  fi
}

echo "=== LaTeX Compilation with BibTeX ==="
echo "TeX dir: ${TEX_DIR}"
echo "Build dir: ${BUILD_DIR}"
echo "TeX cache: ${TEX_CACHE_ROOT}"
echo ""

cd "${TEX_DIR}"

run_pdflatex 1
run_bibtex

# Run up to 4 passes post-bibtex to settle cross-references/outlines.
for pass in 2 3 4 5; do
  run_pdflatex "${pass}"
  if ! needs_rerun; then
    break
  fi
done

fail_on_warnings

cp -f "${BUILD_DIR}/${MAIN}.pdf" "${TEX_DIR}/${MAIN}.pdf"
echo ""
echo "Compilation successful! Output: ${TEX_DIR}/${MAIN}.pdf"

# Clean auxiliary files in the repo working tree (so `git add -A` does not
# accidentally commit build artifacts).
rm -f "${TEX_DIR}/${MAIN}".{aux,bbl,blg,out,toc,lof,lot,fls,fdb_latexmk,synctex.gz,log}

# Git commit + push from repo root (stage the entire repo, not just `tex/`).
if [[ "${PUSH}" == "1" ]]; then
  cd "${SCRIPT_DIR}"

  git add -A
  if git diff --cached --quiet; then
    echo "No changes to commit."
  else
    git commit -m "${COMMIT_MSG}"
  fi

  git push
else
  echo "PUSH=0: skip git commit/push."
fi
