"""Side-by-side 3D render of the ACTUAL contact-surface triangles from both engines.

Drake : out/drake_surface.npz  (verts, tris, per-vertex pressure)  <- `drake_dump  --mesh`
Newton: out/newton_surface.npz (iso-surface triangle soup, per-face depth) <- `newton_dump --mesh`

Each surface is its engine's REAL mesh, face-colored by that engine's OWN native quantity:
Drake = hydroelastic pressure [Pa]; Newton = per-face penetration [m]. No cross-engine
conversion. Rendered at TRUE scale (x:y:z to real proportions -- the patch is genuinely
shallow, so it looks nearly flat).

Run (the newton-sap uv env; run both dumps with --mesh first):
    cd ~/work/newton-sap
    uv run --no-sync python scripts/hydro_compare/experiments/view_surface.py [--interactive]
  Default saves experiments/out/contact_surface_3d.png (headless Agg); --interactive opens a
  rotatable TkAgg window on $DISPLAY instead.
"""
import argparse
import math
import os

import numpy as np
import matplotlib
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import scene as scene_mod

ELEV, AZIM = 30, -60


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


def _add_surface(ax, polys, face_vals, cmap, edge_lw):
    """polys: (F,3,3) world triangle verts; face_vals: (F,) scalar per triangle."""
    coll = Poly3DCollection(polys, cmap=cmap, edgecolor="k", linewidth=edge_lw)
    coll.set_array(np.asarray(face_vals))
    ax.add_collection3d(coll)
    pts = polys.reshape(-1, 3)
    ext = []
    for dim, setlim in enumerate((ax.set_xlim, ax.set_ylim, ax.set_zlim)):
        lo, hi = float(pts[:, dim].min()), float(pts[:, dim].max())
        setlim(lo, hi)
        ext.append(hi - lo)
    ax.set_box_aspect(ext)            # TRUE scale: axis lengths == real extents
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set(xlabel="x [m]", ylabel="y [m]", zlabel="z [m]")
    return coll


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=scene_mod.default_scene_path())
    ap.add_argument("--interactive", action="store_true",
                    help="open a rotatable window (TkAgg on $DISPLAY) instead of saving a PNG")
    args = ap.parse_args()

    if not args.interactive:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt   # after backend is chosen

    cfg = scene_mod.load_scene(args.scene)
    out = cfg.output_dir
    center_xy = np.asarray(cfg.box_center[:2], float)   # contact axis (x=y=0 here)
    fig = plt.figure(figsize=(15, 11))

    # --- Drake: actual ComputeContactSurfaces triangles, actual pressure field ---
    d = np.load(os.path.join(out, "drake_surface.npz"))
    verts, tris, vp = d["verts"], d["tris"], d["vp"]
    dtri = verts[tris]                                  # (F,3,3) world triangles
    dpress = vp[tris].mean(axis=1) / 1.0e6              # per-face Drake pressure [MPa]
    L = _drake_refinement_level(cfg.resolution_hint, cfg.R)
    axd = fig.add_subplot(2, 2, 1, projection="3d")
    cd = _add_surface(axd, dtri, dpress, "viridis", 0.3)
    axd.set_title(f"Drake: {len(tris)} tris, L={L} (pressure)")

    # --- Newton: actual marching-cubes triangles, colored by p = k_eff*|delta_total| (the
    #     equal-pressure surface value) so panels 1-2 share one pressure scale/colorbar ---
    n = np.load(os.path.join(out, "newton_surface.npz"))
    soup, npress = n["tris"], n["pressure"] / 1.0e6     # per-face Newton pressure [MPa]
    axn = fig.add_subplot(2, 2, 2, projection="3d")
    cn = _add_surface(axn, soup, npress, "viridis", 0.1)
    axn.set_title(f"Newton: {soup.shape[0]} tris, voxel={cfg.sdf_target_voxel_size * 1e3:.1f}mm (pressure)")

    vmax = float(max(dpress.max(), npress.max()))
    cd.set_clim(0.0, vmax)
    cn.set_clim(0.0, vmax)
    fig.colorbar(cn, ax=[axd, axn], shrink=0.6, pad=0.04, label="pressure [MPa]")

    # --- Error vs the analytic reference: area-weighted binned radial profiles (the patch is
    #     axisymmetric). Each engine is binned on its OWN mesh -- no cross-mesh resampling --
    #     and compared to the closed-form Winkler profile p*(r). The per-engine area-weighted
    #     L2 (vs reference) quantifies the error; the engine-to-engine gap is bounded by their sum. ---
    axe = fig.add_subplot(2, 2, 3)                            # 2D panel
    rmax = cfg.reference.patch_radius * 1.1
    rc_d, pbar_d = _radial_profile(dtri, dpress, center_xy, 14, rmax)
    rc_n, pbar_n = _radial_profile(soup, npress, center_xy, 14, rmax)
    rr = np.linspace(0.0, rmax, 200)
    axe.plot(rr, scene_mod.reference_pressure(cfg, rr) / 1.0e6, "k-", lw=1.5, label="reference p*(r)")
    axe.plot(rc_d, pbar_d, "o-", ms=4, label="drake (binned)")
    axe.plot(rc_n, pbar_n, "x--", ms=5, label="newton (binned)")
    dL2 = _l2_vs_reference(cfg, dtri, dpress, center_xy)
    nL2 = _l2_vs_reference(cfg, soup, npress, center_xy)
    axe.set(xlabel="patch radius r [m]", ylabel="pressure [MPa]",
            title=f"radial profile vs reference\narea-wtd L2: drake {dL2:.3g}, newton {nL2:.3g} MPa")
    axe.legend()

    # --- Panel 4: residual of each binned radial profile vs the analytic reference,
    #     p_bar(r) - p*(r). Flat at zero == exact; sign shows local over/under-pressure.
    #     Same area-weighted bins as panel 3, so it is the radius-resolved view of the L2. ---
    axr = fig.add_subplot(2, 2, 4)
    pref_d = scene_mod.reference_pressure(cfg, rc_d) / 1.0e6
    pref_n = scene_mod.reference_pressure(cfg, rc_n) / 1.0e6
    axr.axhline(0.0, color="k", lw=1)
    axr.plot(rc_d, pbar_d - pref_d, "o-", ms=4, label="drake - reference")
    axr.plot(rc_n, pbar_n - pref_n, "x--", ms=5, label="newton - reference")
    axr.set(xlabel="patch radius r [m]", ylabel="pressure residual [MPa]",
            title="residual vs reference  p̄(r) - p*(r)")
    axr.legend()

    fig.suptitle("Contact-surface pressure (panels 1-2: actual triangles, true scale); "
                 "panel 3: area-weighted radial profile vs reference; panel 4: residual vs reference")

    # --- measure-and-tune: edge-length stats so we can verify "about the same" ---
    ed, en = _edge_stats(verts[tris]), _edge_stats(soup)
    print(f"[edges m] drake  n={len(tris):5d}  min/med/mean/max = "
          f"{ed[0]:.4g}/{ed[1]:.4g}/{ed[2]:.4g}/{ed[3]:.4g}")
    print(f"[edges m] newton n={soup.shape[0]:5d}  min/med/mean/max = "
          f"{en[0]:.4g}/{en[1]:.4g}/{en[2]:.4g}/{en[3]:.4g}")
    print(f"[edges m] median ratio newton/drake = {en[1] / ed[1]:.2f}")

    if args.interactive:
        plt.show()
    else:
        p = os.path.join(out, "contact_surface_3d.png")
        fig.savefig(p, dpi=130, bbox_inches="tight")
        print(f"[view3d] Drake {len(tris)} tris | Newton {soup.shape[0]} tris -> {p}")


if __name__ == "__main__":
    main()
