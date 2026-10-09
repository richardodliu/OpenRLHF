#!/usr/bin/env bash
# Build and validate the anonymous ICLR submission; export only the PDF.
# Temporary artifacts are removed on success, failure and interruption.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEX_DIR="${REPO_DIR}/tex"
LATEX_ROOT=/volume/pt-train/users/rbliu/latex
if [[ -f "${LATEX_ROOT}/env.sh" ]]; then
    source "${LATEX_ROOT}/env.sh"
fi
BUILD_DIR="$(mktemp -d "${TMPDIR:-/tmp}/openrlhf_iclr.XXXXXX")"
MAIN=iclr2027_submission
EXPORT_TMP=""
cleanup() {
    local result=$?
    if [[ "${result}" -ne 0 && -f "${BUILD_DIR}/${MAIN}.log" ]]; then
        tail -n 40 "${BUILD_DIR}/${MAIN}.log" >&2
    fi
    [[ -z "${EXPORT_TMP}" ]] || rm -f -- "${EXPORT_TMP}"
    rm -rf -- "${BUILD_DIR}"
    return "${result}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
cd "${TEX_DIR}"

run_latex() {
    if ! pdflatex -interaction=nonstopmode -halt-on-error -file-line-error \
        -output-directory="${BUILD_DIR}" "${MAIN}.tex" >"${BUILD_DIR}/pass_$1.out" 2>&1; then
        tail -n 60 "${BUILD_DIR}/pass_$1.out"
        return 1
    fi
}

run_latex 1
if ! (cd "${BUILD_DIR}" && BIBINPUTS=".:${TEX_DIR}:" BSTINPUTS=".:${TEX_DIR}:" \
    bibtex "${MAIN}") >"${BUILD_DIR}/bibtex.out" 2>&1; then
    cat "${BUILD_DIR}/bibtex.out"
    exit 1
fi

# Settle forward references, bibliography, and PDF bookmarks.
for pass in 2 3 4 5; do
    run_latex "${pass}"
    if [[ "${pass}" -ge 3 ]] && ! grep -Eq 'Rerun to|Please rerun|Label\(s\) may have changed' "${BUILD_DIR}/${MAIN}.log"; then
        break
    fi
done

if grep -Eq 'undefined|multiply defined|destination with the same identifier|Rerun to|Please rerun|Label\(s\) may have changed|Overfull' \
    "${BUILD_DIR}/${MAIN}.log" || grep -q 'Warning--' "${BUILD_DIR}/bibtex.out"; then
    echo 'ERROR: unresolved references, bibliography/layout warnings, or an unsettled build.'
    exit 1
fi
MAIN_END_PAGE="$(sed -nE 's/^\\newlabel\{sec:iclr-main-end\}\{\{[^}]*\}\{([0-9]+)\}.*/\1/p' "${BUILD_DIR}/${MAIN}.aux")"
if ! [[ "${MAIN_END_PAGE}" =~ ^[1-9]$ ]]; then
    echo 'ERROR: missing main-text boundary or main text exceeds the nine-page limit.'
    exit 1
fi

python3 "${REPO_DIR}/reinforce_pro_max/check_submission_readiness.py" --build-dir "${BUILD_DIR}"
EXPORT_TMP="$(mktemp "${REPO_DIR}/.${MAIN}.XXXXXX.pdf")"
cp "${BUILD_DIR}/${MAIN}.pdf" "${EXPORT_TMP}"
chmod 644 "${EXPORT_TMP}"
mv -f -- "${EXPORT_TMP}" "${REPO_DIR}/${MAIN}.pdf"
EXPORT_TMP=""
rm -f -- "${TEX_DIR}/${MAIN}.pdf"
echo "Built ${REPO_DIR}/${MAIN}.pdf (main text ends on page ${MAIN_END_PAGE}); temporary artifacts removed on exit."
