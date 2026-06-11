"""Side-by-side 3D render of the ACTUAL contact-surface triangles from both engines (PyVista).

Drake : out/drake_surface.npz  (verts, tris, per-vertex pressure)  <- `drake_dump  --mesh`
Newton: out/newton_surface.npz (iso-surface triangle soup, per-face depth) <- `newton_dump --mesh`

Each surface is its engine's REAL mesh, face-colored by that engine's OWN native quantity
(pressure [MPa]). No cross-engine conversion. Rendered at TRUE scale (PyVista keeps equal
aspect, so the genuinely shallow patch looks nearly flat). 2x2 layout: top row = the two
contact-surface meshes (shared pressure scale); bottom row = area-weighted radial profile vs
the analytic reference, and its residual.

Run (this directory's uv env; run both dumps with --mesh first):
    cd ~/work/newton-sap/scripts/hydro_compare
    uv run python experiments/view_surface.py                # save out/contact_surface_3d.png
    uv run python experiments/view_surface.py --interactive  # rotatable VTK window on $DISPLAY
    uv run python experiments/view_surface.py --html         # standalone interactive .html
"""
import argparse
import math
import os

import numpy as np
import pyvista as pv

import scene as scene_mod


def _drake_refinement_level(resolution_hint, R):
    """Drake's quantized sphere refinement level L (geometry/proximity/make_sphere_mesh.h):
    L = max(0, ceil(log2(pi / asin(e / (2R)))) - 2). The tet mesh -- and hence the contact
    surface's granularity -- is pinned by L, not directly by the hint."""
    return max(0, math.ceil(math.log2(math.pi / math.asin(resolution_hint / (2.0 * R)))) - 2)


def _edge_stats(polys):
    """polys: (F,3,3) -> (min, median, mean, max) triangle edge length [m]."""
    e = np.concatenate([polys[:, 0] - polys[:, 1],
                        polys[:, 1] - polys[:, 2],
                        polys[:, 2] - polys[:, 0]])
    L = np.linalg.norm(e, axis=1)
    return float(L.min()), float(np.median(L)), float(L.mean()), float(L.max())


def _tri_area(polys):
    """polys: (F,3,3) world triangles -> (F,) areas [m^2]."""
    return 0.5 * np.linalg.norm(np.cross(polys[:, 1] - polys[:, 0],
                                         polys[:, 2] - polys[:, 0]), axis=1)


def _radial_profile(polys, fpress, center_xy, nbins, rmax):
    """Area-weighted mean pressure per radial bin (each face binned on its OWN mesh, no
    cross-mesh resampling). polys (F,3,3); fpress (F,) [MPa] -> (bin_centers, mean_p)."""
    r = np.linalg.norm(polys.mean(axis=1)[:, :2] - center_xy, axis=1)
    A = _tri_area(polys)
    edges = np.linspace(0.0, rmax, nbins + 1)
    idx = np.clip(np.digitize(r, edges) - 1, 0, nbins - 1)
    pbar = np.full(nbins, np.nan)
    for b in range(nbins):
        m = idx == b
        if m.any() and A[m].sum() > 0.0:
            pbar[b] = (A[m] * fpress[m]).sum() / A[m].sum()
    return 0.5 * (edges[:-1] + edges[1:]), pbar


def _l2_vs_reference(cfg, polys, fpress, center_xy):
    """Area-weighted L2 of per-face pressure [MPa] vs the reference [MPa], on the face's own mesh."""
    r = np.linalg.norm(polys.mean(axis=1)[:, :2] - center_xy, axis=1)
    A = _tri_area(polys)
    e = np.asarray(fpress, float) - scene_mod.reference_pressure(cfg, r) / 1.0e6
    return float(np.sqrt((A * e * e).sum() / A.sum()))


_SCALAR = "pressure [MPa]"


def _drake_mesh(verts, tris, vp):
    """Drake's NATIVE indexed surface with its per-VERTEX pressure field (smooth shading,
    no vertex->face averaging). vp [Pa] -> point scalars [MPa]."""
    f = tris.shape[0]
    faces = np.hstack([np.full((f, 1), 3, np.int64), tris.astype(np.int64)]).ravel()
    m = pv.PolyData(np.asarray(verts, float), faces)
    m.point_data[_SCALAR] = np.asarray(vp, float) / 1.0e6
    return m


def _newton_mesh(polys, fpress):
    """Newton's NATIVE triangle soup (one independent face/triangle) with its per-FACE
    pressure. fpress [Pa] -> cell scalars [MPa]."""
    f = polys.shape[0]
    verts = polys.reshape(-1, 3)
    faces = np.hstack([np.full((f, 1), 3, np.int64),
                       np.arange(3 * f, dtype=np.int64).reshape(f, 3)]).ravel()
    m = pv.PolyData(verts, faces)
    m.cell_data[_SCALAR] = np.asarray(fpress, float) / 1.0e6
    return m


def _add_mesh(pl, row, col, mesh, title, clim, *, scalar_bar):
    """Add one contact-surface mesh exactly as built (native point/cell scalars), true scale."""
    pl.subplot(row, col)
    pl.add_mesh(mesh, scalars=_SCALAR, cmap="viridis", clim=clim,
                show_edges=True, edge_color="black", line_width=0.3,
                show_scalar_bar=scalar_bar,
                scalar_bar_args=dict(title=_SCALAR, vertical=True,
                                     position_x=0.85, position_y=0.2, height=0.6))
    pl.add_text(title, font_size=9)
    pl.view_isometric()


def _finite(x, y):
    """Drop NaN-y bins so Chart2D doesn't choke."""
    m = np.isfinite(np.asarray(x, float)) & np.isfinite(np.asarray(y, float))
    return np.asarray(x, float)[m], np.asarray(y, float)[m]


def _add_chart(pl, row, col, series, x_label, y_label, *, zero_line=False, x_span=None):
    """series: list of (x, y, color, draw_line, draw_pts, label)."""
    pl.subplot(row, col)
    chart = pv.Chart2D()
    if zero_line and x_span is not None:
        chart.line(list(x_span), [0.0, 0.0], color="black", width=1.0)
    for x, y, color, draw_line, draw_pts, label in series:
        x, y = _finite(x, y)
        if not len(x):
            continue
        if draw_line:
            chart.line(x, y, color=color, width=2.0, label=label)
        if draw_pts:
            chart.scatter(x, y, color=color, size=7, label=None if draw_line else label)
    chart.x_label = x_label
    chart.y_label = y_label
    chart.legend_visible = True
    pl.add_chart(chart)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=scene_mod.default_scene_path())
    ap.add_argument("--interactive", action="store_true",
                    help="open a rotatable VTK window on $DISPLAY instead of saving a PNG")
    ap.add_argument("--html", action="store_true",
                    help="export a standalone interactive .html (meshes only) instead of a PNG")
    args = ap.parse_args()

    cfg = scene_mod.load_scene(args.scene)
    out = cfg.output_dir
    center_xy = np.asarray(cfg.box_center[:2], float)   # contact axis (x=y=0 here)

    # --- Drake: NATIVE indexed surface + per-vertex pressure field (no modification) ---
    d = np.load(os.path.join(out, "drake_surface.npz"))
    verts, tris, vp = d["verts"], d["tris"], d["vp"]
    dmesh = _drake_mesh(verts, tris, vp)
    L = _drake_refinement_level(cfg.resolution_hint, cfg.R)

    # --- Newton: NATIVE marching-cubes soup + per-face pressure (no modification) ---
    n = np.load(os.path.join(out, "newton_surface.npz"))
    soup, npress = n["tris"], n["pressure"] / 1.0e6     # per-face Newton pressure [MPa]
    nmesh = _newton_mesh(soup, n["pressure"])

    # Shared color scale so the two panels are visually comparable (mapping only, not a data
    # change). vmax from each engine's native field at its native granularity.
    vmax = float(max(vp.max(), npress.max() * 1.0e6) / 1.0e6)
    clim = (0.0, vmax)

    off_screen = not args.interactive
    pl = pv.Plotter(shape=(2, 2), off_screen=off_screen, window_size=(1500, 1100),
                    border=True)

    _add_mesh(pl, 0, 0, dmesh, f"Drake: {len(tris)} tris, L={L} (per-vertex field)", clim,
              scalar_bar=False)
    _add_mesh(pl, 0, 1, nmesh,
              f"Newton: {soup.shape[0]} tris, voxel={cfg.sdf_target_voxel_size * 1e3:.1f}mm (per-face)",
              clim, scalar_bar=True)
    pl.link_views((0, 1))   # the two meshes orbit together; charts are not cameras

    # --- Radial profile vs analytic reference (area-weighted bins, each on its OWN mesh).
    #     Binning needs a per-face value: Drake's is the centroid pressure = mean of its 3
    #     vertex values (exact for the linear field); Newton's is its native per-face pressure. ---
    dtri = verts[tris]                                  # (F,3,3) world triangles for binning
    dpress = vp[tris].mean(axis=1) / 1.0e6              # per-face Drake pressure [MPa] (analysis only)
    rmax = cfg.reference.patch_radius * 1.1
    rc_d, pbar_d = _radial_profile(dtri, dpress, center_xy, 14, rmax)
    rc_n, pbar_n = _radial_profile(soup, npress, center_xy, 14, rmax)
    rr = np.linspace(0.0, rmax, 200)
    pref = scene_mod.reference_pressure(cfg, rr) / 1.0e6
    dL2 = _l2_vs_reference(cfg, dtri, dpress, center_xy)
    nL2 = _l2_vs_reference(cfg, soup, npress, center_xy)
    _add_chart(pl, 1, 0, [
        (rr, pref, "black", True, False, "reference p*(r)"),
        (rc_d, pbar_d, "blue", True, True, "drake (binned)"),
        (rc_n, pbar_n, "red", True, True, "newton (binned)"),
    ], "patch radius r [m]", f"pressure [MPa]  (L2: drake {dL2:.3g}, newton {nL2:.3g})")

    # --- Residual of each binned profile vs reference: p_bar(r) - p*(r); flat-zero == exact ---
    pref_d = scene_mod.reference_pressure(cfg, rc_d) / 1.0e6
    pref_n = scene_mod.reference_pressure(cfg, rc_n) / 1.0e6
    _add_chart(pl, 1, 1, [
        (rc_d, pbar_d - pref_d, "blue", True, True, "drake - reference"),
        (rc_n, pbar_n - pref_n, "red", True, True, "newton - reference"),
    ], "patch radius r [m]", "pressure residual [MPa]", zero_line=True, x_span=(0.0, rmax))

    # --- measure-and-tune: edge-length stats so we can verify "about the same" ---
    ed, en = _edge_stats(dtri), _edge_stats(soup)
    print(f"[edges m] drake  n={len(tris):5d}  min/med/mean/max = "
          f"{ed[0]:.4g}/{ed[1]:.4g}/{ed[2]:.4g}/{ed[3]:.4g}")
    print(f"[edges m] newton n={soup.shape[0]:5d}  min/med/mean/max = "
          f"{en[0]:.4g}/{en[1]:.4g}/{en[2]:.4g}/{en[3]:.4g}")
    print(f"[edges m] median ratio newton/drake = {en[1] / ed[1]:.2f}")

    if args.interactive:
        pl.show(title="Contact-surface pressure (true scale)")
    elif args.html:
        # Interactive HTML can't carry Chart2D overlays -> export the two meshes only.
        hp = os.path.join(out, "contact_surface_3d.html")
        mp = pv.Plotter(shape=(1, 2), off_screen=True, window_size=(1500, 700), border=True)
        _add_mesh(mp, 0, 0, dmesh, f"Drake: {len(tris)} tris, L={L} (per-vertex field)", clim,
                  scalar_bar=False)
        _add_mesh(mp, 0, 1, nmesh, f"Newton: {soup.shape[0]} tris (per-face)", clim,
                  scalar_bar=True)
        mp.link_views()
        mp.export_html(hp)
        print(f"[view3d] Drake {len(tris)} tris | Newton {soup.shape[0]} tris -> {hp}")
    else:
        p = os.path.join(out, "contact_surface_3d.png")
        pl.screenshot(p)
        print(f"[view3d] Drake {len(tris)} tris | Newton {soup.shape[0]} tris -> {p}")


if __name__ == "__main__":
    main()
