#!/usr/bin/env bash
# Convert docs/paper_draft.md to arxiv-style LaTeX (.tex) and optionally PDF.
#
# Uses pandoc to do the Markdown → LaTeX conversion. Pandoc is available
# on most platforms via Homebrew (mac) or apt (debian/ubuntu); install
# instructions are printed when missing. PDF compilation needs a LaTeX
# engine (xelatex by default, configurable via PDF_ENGINE); MacTeX or
# TeX Live cover that on mac / linux respectively.
#
# Output goes to release/paper/ (same convention as build_paper.sh) so
# the LaTeX + PDF live alongside the Markdown bundle.
#
# Usage:
#   bash scripts/build_paper_latex.sh
#
# Environment variables:
#   LATEX_OUT_DIR     release destination (default: release/paper)
#   PDF_ENGINE        LaTeX engine for PDF build (default: xelatex)
#   BUILD_PDF         set to "0" to skip PDF compilation
#
# What this script does NOT do:
#   * It doesn't try to produce a perfectly-arxiv-styled paper (two-column,
#     specific fonts, etc.). It produces a clean single-column article-
#     class PDF that compiles cleanly and is readable. Most arxiv papers
#     in alignment are single-column anyway; serious typesetting belongs
#     in a separate per-submission template.
#   * It doesn't regenerate figures. Run scripts/generate_paper_figures.py
#     first if your run JSONs have changed since docs/figures/ was
#     written.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

LATEX_OUT_DIR="${LATEX_OUT_DIR:-release/paper}"
PDF_ENGINE="${PDF_ENGINE:-xelatex}"
BUILD_PDF="${BUILD_PDF:-1}"

PAPER_SRC="docs/paper_draft.md"
FIG_SRC="docs/figures"

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

if ! command -v pandoc >/dev/null 2>&1; then
  echo "ERROR: pandoc not found." >&2
  echo "  Install:" >&2
  echo "    macOS:        brew install pandoc" >&2
  echo "    debian/ubuntu: sudo apt install pandoc" >&2
  echo "    or download from: https://pandoc.org/installing.html" >&2
  exit 2
fi

if [[ ! -f "$PAPER_SRC" ]]; then
  echo "ERROR: paper source not found at $PAPER_SRC" >&2
  exit 2
fi

if [[ ! -d "$FIG_SRC" ]]; then
  echo "ERROR: figures dir not found at $FIG_SRC" >&2
  echo "  Run: python scripts/generate_paper_figures.py ..." >&2
  exit 2
fi

# Verify all four expected figures exist.
missing=()
for fig in \
    fig1_refusal_forest.png \
    fig2_vea_inflation.png \
    fig3_qwen3_per_family.png \
    fig4_mediation_panels.png
do
  if [[ ! -f "$FIG_SRC/$fig" ]]; then
    missing+=("$fig")
  fi
done
if [[ ${#missing[@]} -gt 0 ]]; then
  echo "ERROR: missing figures in $FIG_SRC:" >&2
  for m in "${missing[@]}"; do echo "  - $m" >&2; done
  echo "Run: python scripts/generate_paper_figures.py ..." >&2
  exit 2
fi

# ---------------------------------------------------------------------------
# Prepare output directory + copy figures
# ---------------------------------------------------------------------------

mkdir -p "$LATEX_OUT_DIR/figures"
cp "$FIG_SRC"/*.png "$LATEX_OUT_DIR/figures/"

# ---------------------------------------------------------------------------
# Pandoc invocation: Markdown -> LaTeX
# ---------------------------------------------------------------------------
#
# Variables passed to the default pandoc template:
#   * documentclass=article    standard arxiv-style class
#   * fontsize=11pt            common alignment-paper size
#   * geometry=margin=1in      conservative margins
#   * linkcolor=blue            hyperlinks visible but not garish
#   * urlcolor=blue            same for URLs
#   * colorlinks=true          required for the above to take effect
#   * --resource-path           lets pandoc resolve relative figure paths

PANDOC_TEX_CMD=(
  pandoc "$PAPER_SRC"
  --output="$LATEX_OUT_DIR/paper.tex"
  --standalone
  --from=markdown
  --to=latex
  --resource-path="$REPO_ROOT/docs"
  --variable=documentclass:article
  --variable=fontsize:11pt
  --variable=geometry:margin=1in
  --variable=colorlinks:true
  --variable=linkcolor:blue
  --variable=urlcolor:blue
  --variable=toc-depth:2
)

echo "Running: ${PANDOC_TEX_CMD[*]}"
"${PANDOC_TEX_CMD[@]}"
echo "Wrote: $LATEX_OUT_DIR/paper.tex"

# ---------------------------------------------------------------------------
# Optional: pandoc -> PDF
# ---------------------------------------------------------------------------

if [[ "$BUILD_PDF" == "0" ]]; then
  echo "Skipping PDF build (BUILD_PDF=0)."
  exit 0
fi

if ! command -v "$PDF_ENGINE" >/dev/null 2>&1; then
  echo "WARN: PDF engine '$PDF_ENGINE' not found; skipping PDF build." >&2
  echo "  Install:" >&2
  echo "    macOS:         brew install --cask mactex-no-gui" >&2
  echo "    debian/ubuntu: sudo apt install texlive-xetex" >&2
  echo "  Or set PDF_ENGINE=pdflatex / lualatex if you have a different engine." >&2
  echo "Skipping. paper.tex was still written." >&2
  exit 0
fi

PANDOC_PDF_CMD=(
  pandoc "$PAPER_SRC"
  --output="$LATEX_OUT_DIR/paper.pdf"
  --pdf-engine="$PDF_ENGINE"
  --from=markdown
  --resource-path="$REPO_ROOT/docs"
  --variable=documentclass:article
  --variable=fontsize:11pt
  --variable=geometry:margin=1in
  --variable=colorlinks:true
  --variable=linkcolor:blue
  --variable=urlcolor:blue
)

echo "Running: ${PANDOC_PDF_CMD[*]}"
"${PANDOC_PDF_CMD[@]}"
echo "Wrote: $LATEX_OUT_DIR/paper.pdf"

echo
echo "LaTeX + PDF ready in $LATEX_OUT_DIR/"
