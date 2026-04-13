#!/bin/zsh
set -euo pipefail

PAPER_DIR="/Users/savewind/Documents/gfkd/Metamaterials/IEEE_RadarConf"
PAPER_TEX="metasurface_isrj_parameter_estimation.tex"
TEXBIN="/Library/TeX/texbin"

if [[ -d "$TEXBIN" ]]; then
  export PATH="$TEXBIN:$PATH"
fi

LATEXMK_BIN="$(command -v latexmk || true)"

if [[ -z "$LATEXMK_BIN" ]]; then
  echo "Error: latexmk not found. Install MacTeX/TeX Live or add it to PATH." >&2
  exit 127
fi

cd "$PAPER_DIR"

echo "Compiling $PAPER_TEX..."
"$LATEXMK_BIN" -pdf -interaction=nonstopmode -halt-on-error "$PAPER_TEX"

echo
echo "Done."
echo "Output: $PAPER_DIR/metasurface_isrj_parameter_estimation.pdf"
