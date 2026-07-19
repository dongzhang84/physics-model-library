#!/usr/bin/env bash
# Build latex_en/main.tex. Usage: run  ./build.sh  in this directory.
# Requires XeLaTeX (not pdflatex) + BibTeX.

set -euo pipefail
cd "$(dirname "$0")"

if ! command -v xelatex >/dev/null 2>&1; then
    echo "Error: xelatex not found. Install MacTeX / TeX Live and retry." >&2
    exit 1
fi

echo "[1/4] xelatex pass 1 (generate .aux) ..."
xelatex -interaction=nonstopmode main.tex >/dev/null
echo "[2/4] bibtex (resolve references) ..."
bibtex main >/dev/null || true
echo "[3/4] xelatex pass 2 (write citations) ..."
xelatex -interaction=nonstopmode main.tex >/dev/null
echo "[4/4] xelatex pass 3 (resolve cross-references) ..."
xelatex -interaction=nonstopmode main.tex >/dev/null

echo "Done. Output: $(pwd)/main.pdf"
