#!/usr/bin/env bash
# Package the paper into a self-contained release directory.
#
# Bundles docs/paper_draft.md + docs/figures/*.png + a README into
# release/paper/, with image references rewritten so the bundle works
# standalone (no repo paths required). Suitable for:
#
#   * sharing as a tarball (the script also produces release/paper.tar.gz)
#   * uploading to a preprint server
#   * sending to a reader for feedback
#
# Usage:
#   bash scripts/build_paper.sh
#
# Optional environment variables:
#   PAPER_OUT_DIR  release destination (default: release/paper)
#   PAPER_BUILD_TARBALL  set to "0" to skip the tarball step
#
# This script does NOT regenerate figures. Run
#   scripts/generate_paper_figures.py
# first if you've changed run JSONs since the figures in docs/figures/
# were produced.

set -euo pipefail

# Repo root (script lives in scripts/, so go up one level).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PAPER_OUT_DIR="${PAPER_OUT_DIR:-release/paper}"
PAPER_BUILD_TARBALL="${PAPER_BUILD_TARBALL:-1}"

PAPER_SRC="docs/paper_draft.md"
FIG_SRC="docs/figures"

if [[ ! -f "$PAPER_SRC" ]]; then
  echo "ERROR: paper source not found at $PAPER_SRC" >&2
  exit 2
fi

if [[ ! -d "$FIG_SRC" ]]; then
  echo "ERROR: figures dir not found at $FIG_SRC" >&2
  echo "  Run: python scripts/generate_paper_figures.py ..." >&2
  exit 2
fi

# Verify all five expected figures exist.
missing=()
for fig in \
    fig1_refusal_forest.png \
    fig2_vea_inflation.png \
    fig3_qwen3_per_family.png \
    fig4_mediation_panels.png \
    fig5_two_mechanism.png
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

# Reset the output dir.
rm -rf "$PAPER_OUT_DIR"
mkdir -p "$PAPER_OUT_DIR/figures"

# Copy figures.
cp "$FIG_SRC"/*.png "$PAPER_OUT_DIR/figures/"

# Copy paper. The image references in docs/paper_draft.md already use
# relative paths like figures/fig1_*.png, which work from the release
# dir too because we placed figures/ at the same level. No rewriting
# needed.
cp "$PAPER_SRC" "$PAPER_OUT_DIR/paper.md"

# Build a README for the release dir so a reader knows what they're
# looking at.
cat > "$PAPER_OUT_DIR/README.md" <<'EOF'
# Paper release bundle

This directory is a self-contained snapshot of the technical report
plus its figures, suitable for sharing or preprint upload.

## Contents

- `paper.md` — the full draft, Markdown, ~6,000 words.
- `figures/` — four PNG figures referenced from the paper.
- `README.md` — this file.

## How to read

Open `paper.md` in any Markdown viewer that supports inline images
(GitHub, Obsidian, VS Code preview, Typora, etc.). All four figure
references resolve to `figures/figN_*.png` in this directory.

## How this bundle was built

```bash
# 1. Regenerate figures (if run JSONs have changed since the last
#    bundle):
python scripts/generate_paper_figures.py \
  --cross-protocol-summary runs/cross-protocol-v6/cross_protocol_summary.json \
  --goodfire-summary runs/goodfire-mixed-n500/goodfire_vea_summary.json \
  --mediation-summary runs/goodfire-mixed-n500/vea_mediation_summary.json \
  --strict-mediation-summary runs/goodfire-mixed-n500/vea_mediation_summary.strict.json \
  --out-dir docs/figures/

# 2. Build the release bundle:
bash scripts/build_paper.sh
```

The release directory is regenerated from scratch on each build, so
edits should be made to `docs/paper_draft.md` and the figure-generation
inputs — not to files in this bundle directly.

## Versioning

The release reflects the state of `docs/paper_draft.md` at the
commit shown by `git rev-parse HEAD` at build time. See
`build_info.txt` for the exact commit SHA, branch, and build
timestamp.
EOF

# Record build provenance so a reader can trace the bundle back to
# a specific commit. Tolerant of repos with no commits yet (e.g. fresh
# clones or test fixtures) — falls back to a sentinel.
git_commit="$(git rev-parse HEAD 2>/dev/null || echo "(uncommitted)")"
git_branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "(detached)")"
git_dirty="$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
{
  echo "Bundle built at: $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  echo "Git commit:      $git_commit"
  echo "Git branch:      $git_branch"
  echo "Working tree:    $git_dirty file(s) modified vs HEAD"
} > "$PAPER_OUT_DIR/build_info.txt"

echo "Wrote: $PAPER_OUT_DIR/paper.md"
echo "Wrote: $PAPER_OUT_DIR/README.md"
echo "Wrote: $PAPER_OUT_DIR/build_info.txt"
echo "Wrote: $PAPER_OUT_DIR/figures/  ($(ls "$PAPER_OUT_DIR/figures" | wc -l | tr -d ' ') figures)"

if [[ "$PAPER_BUILD_TARBALL" == "1" ]]; then
  TARBALL="$(dirname "$PAPER_OUT_DIR")/paper.tar.gz"
  tar -czf "$TARBALL" -C "$(dirname "$PAPER_OUT_DIR")" "$(basename "$PAPER_OUT_DIR")"
  echo "Wrote: $TARBALL"
fi

echo
echo "Bundle ready in $PAPER_OUT_DIR/"
