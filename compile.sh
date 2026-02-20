#!/bin/bash
# LaTeX compilation script with bibtex
# Full compilation: pdflatex -> bibtex -> pdflatex -> pdflatex

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEX_DIR="$SCRIPT_DIR/tex"
MAIN="main"

cd "$TEX_DIR"

echo "=== LaTeX Compilation with BibTeX ==="
echo "Working directory: $TEX_DIR"
echo ""

echo "[1/4] First pdflatex pass..."
pdflatex -interaction=nonstopmode "$MAIN.tex" > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "ERROR: First pdflatex pass failed! See $MAIN.log"
    exit 1
fi

echo "[2/4] Running bibtex..."
bibtex "$MAIN" > /dev/null 2>&1

echo "[3/4] Second pdflatex pass..."
pdflatex -interaction=nonstopmode "$MAIN.tex" > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "ERROR: Second pdflatex pass failed! See $MAIN.log"
    exit 1
fi

echo "[4/4] Third pdflatex pass..."
pdflatex -interaction=nonstopmode "$MAIN.tex" > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "ERROR: Third pdflatex pass failed! See $MAIN.log"
    exit 1
fi

echo ""
echo "Compilation successful! Output: $TEX_DIR/$MAIN.pdf"

# Clean auxiliary files
rm -f "$MAIN".{aux,bbl,blg,out,toc,lof,lot,fls,fdb_latexmk,synctex.gz,log}

git add . && git commit -m "update" && git push