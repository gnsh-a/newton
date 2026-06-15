#!/usr/bin/env bash
# Run the frozen sphere-on-box hydroelastic cross-check from a single scene.yaml.
#
# Two modes:
#   (default)  full report: drake_dump + newton_dump + compare + view_surface
#              -> out/diff.json, out/distributions.png, out/contact_surface_3d.png
#   --view     fast iterate-and-look loop: edit scene.yaml, then regenerate BOTH surface
#              dumps and open the rotatable PyVista window. Skips compare + the screenshot.
#
# Handles the two-env split: dumps + compare run in the newton-sap env
# (newton/CUDA + pydrake + matplotlib); view_surface runs in the scripts/hydro_compare
# uv env (pyvista, no newton). --view needs an X/VNC $DISPLAY.
#
#   Usage: run_all.sh [--view] [path/to/scene.yaml]
set -euo pipefail

MODE=full
SCENE=""
for arg in "$@"; do
  case "$arg" in
    --view|-v) MODE=view ;;
    *)         SCENE="$arg" ;;
  esac
done

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"   # ~/work/newton-sap
HC="$REPO/scripts/hydro_compare"
EXP="$HC/experiments"
FC="$EXP/frozen_compare"
SCENE="${SCENE:-$EXP/sphere_box.yaml}"

echo "== [dump 1/2] Drake (ground truth + surface mesh) =="
( cd "$REPO" && uv run --no-sync python "$FC/drake_dump.py" --scene "$SCENE" --mesh )

echo "== [dump 2/2] Newton (native SDF + surface mesh; needs CUDA) =="
( cd "$REPO" && uv run --no-sync python "$FC/newton_dump.py" --scene "$SCENE" --mesh )

if [ "$MODE" = view ]; then
  echo "== interactive surface viewer (rotatable; close window to exit) =="
  ( cd "$HC" && uv run python experiments/view_surface.py --scene "$SCENE" --interactive )
  exit 0
fi

echo "== compare -> diff.json + distributions.png =="
( cd "$REPO" && uv run --no-sync python "$FC/compare.py" --scene "$SCENE" )

echo "== 3D surface render -> contact_surface_3d.png (pyvista env) =="
( cd "$HC" && uv run python experiments/view_surface.py --scene "$SCENE" )

OUT="$(awk '/^output_dir:/{print $2}' "$SCENE")"
echo "== done -> $OUT/frozen_compare/{distributions.png, contact_surface_3d.png, diff.json} =="
