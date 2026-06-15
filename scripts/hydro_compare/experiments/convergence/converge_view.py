"""Interactive grid of a hydroelastic convergence sweep -- all levels at once.

Engine-agnostic viewer for BOTH studies: point --manifest at either out/drake_convergence/
(Drake tet levels) or out/newton_convergence/ (Newton voxel sizes). Loads the meshes dumped by
drake_converge_dump.py / newton_converge_dump.py and shows one contact surface per panel in a
linked-camera PyVista grid: pressure-colored on a SHARED scale with edges visible, so the
tessellation density is directly comparable across panels and the cameras orbit together. No
pydrake / CUDA -- re-view dumps without re-running either engine.

Run in the hydro_compare uv env (needs a $DISPLAY for the interactive window):
    cd ~/work/newton-sap/scripts/hydro_compare
    uv run python experiments/convergence/converge_view.py                        # Drake (default)
    uv run python experiments/convergence/converge_view.py --engine newton        # Newton
    uv run python experiments/convergence/converge_view.py --engine drake newton  # both, shared scale
    uv run python experiments/convergence/converge_view.py --save out.png         # headless render instead
"""
import argparse
import json
import math
import os

import numpy as np
import pyvista as pv

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # experiments/ (scene.py, view_surface.py)
import scene as scene_mod
from view_surface import _add_mesh, _drake_mesh


def _level_ok(lv, lmin, lmax):
    """L-based --lmin/--lmax filter; passes through levels with no L (Newton voxel sweep)."""
    L = lv.get("L")
    if L is None:
        return True
    return (lmin is None or L >= lmin) and (lmax is None or L <= lmax)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", nargs="+", choices=["drake", "newton"], default=["drake"],
                    help="which sweep(s) to view; resolves to out/<engine>_convergence/manifest.json "
                         "(default: drake). Pass both to view them together, one engine per row-block.")
    ap.add_argument("--manifest", nargs="+", default=None,
                    help="explicit manifest.json path(s), overriding --engine")
    ap.add_argument("--scene", default=scene_mod.default_scene_path())
    ap.add_argument("--lmin", type=int, default=None, help="only show levels >= this (Drake L only)")
    ap.add_argument("--lmax", type=int, default=None, help="only show levels <= this (Drake L only)")
    ap.add_argument("--save", default=None, help="render to this png/html headless instead of a window")
    ap.add_argument("--edges", choices=["on", "off"], default="on",
                    help="draw triangle wireframe (off keeps fine/high-L panels legible)")
    args = ap.parse_args()

    if args.manifest:
        manifests = args.manifest
    else:
        cfg = scene_mod.load_scene(args.scene)
        manifests = [os.path.join(scene_mod.experiment_dir(cfg, f"{e}_convergence"), "manifest.json")
                     for e in args.engine]

    # Load every manifest into an engine-block: its base dir, filtered levels, and engine tag.
    blocks = []
    vmax = 0.0
    for mpath in manifests:
        if not os.path.exists(mpath):
            raise SystemExit(f"no manifest at {mpath}; run the matching converge dump first")
        with open(mpath) as f:
            man = json.load(f)
        levels = [lv for lv in man["levels"] if _level_ok(lv, args.lmin, args.lmax)]
        if not levels:
            raise SystemExit(f"no levels match the --lmin/--lmax filter in {mpath}")
        blocks.append(dict(base=os.path.dirname(mpath), levels=levels,
                           engine=man.get("engine", "drake")))
        vmax = max(vmax, man["p_vmax_Pa"])
    clim = (0.0, vmax / 1.0e6)        # ONE pressure scale [MPa] shared across all engines

    # Layout: a fixed column count; each engine occupies its own contiguous block of rows.
    cols = min(4, max(len(b["levels"]) for b in blocks))
    rows_per = [math.ceil(len(b["levels"]) / cols) for b in blocks]
    grid_rows = sum(rows_per)
    row_off = [sum(rows_per[:k]) for k in range(len(blocks))]

    off = args.save is not None
    pl = pv.Plotter(shape=(grid_rows, cols), off_screen=off,
                    window_size=(420 * cols, 380 * grid_rows), border=True)
    first = True
    for b, r0 in zip(blocks, row_off):
        for i, lv in enumerate(b["levels"]):
            d = np.load(os.path.join(b["base"], lv["file"]))
            mesh = _drake_mesh(d["verts"], d["tris"], d["vp"])
            rr, cc = divmod(i, cols)
            label = lv.get("label", f"L{lv.get('L', '?')}")
            _add_mesh(pl, r0 + rr, cc, mesh,
                      f"{b['engine']} {label}: {lv['n_tris']} tris, edge={lv['median_edge']*1e3:.2f}mm",
                      clim, scalar_bar=first, show_edges=(args.edges == "on"))
            first = False
    pl.link_views()

    if args.save:
        if args.save.endswith(".html"):
            pl.export_html(args.save)
        else:
            pl.screenshot(args.save)
        print(f"[view] wrote {args.save}")
    else:
        engines = "+".join(b["engine"] for b in blocks)
        pl.show(title=f"hydroelastic convergence: {engines} (shared pressure scale)")


if __name__ == "__main__":
    main()
