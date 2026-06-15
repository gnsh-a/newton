"""Drake tet-resolution sweep: dump one contact-surface mesh per refinement level + metrics + plot.

Drake QUANTIZES the sphere tet mesh into refinement levels L = max(0, ceil(log2(pi/asin(e/2R))) - 2),
so a linear resolution_hint sweep is inert within a level. This steps L = lmin..lmax (inverting that
formula for one hint per level), builds the frozen sphere-on-box contact at each, and records:

  * scalar aggregates  : Fn, peak pressure p0_max, equivalent radius sqrt(A/pi)   (+ /Winkler ratios)
  * convergence        : successive relative change in Fn (mesh settled when small)
  * mesh descriptors   : triangle count, median edge length (the convergence x-axis)
  * pressure profile   : area-weighted radial p_bar(r) (axisymmetric 1-D distribution)

Physics is held fixed from scene.yaml; only the mesh refines. Pure pydrake -- no Newton/CUDA.
Run in the hydro_compare uv env:
    cd ~/work/newton-sap/scripts/hydro_compare
    uv run python experiments/convergence/drake_converge_dump.py [--lmin 1 --lmax 8] [--bins 14] [--scene path]
Writes out/drake_convergence/: drake_surf_L{n}.npz (per level), manifest.json, and one
publication figure drake_convergence.png with ALL non-surface metrics in a 2x3 grid.
Pair with converge_view.py for the interactive surface grid.
"""
import argparse
import json
import math
import os

import numpy as np

from pydrake.geometry import (
    AddCompliantHydroelasticProperties,
    Box,
    FramePoseVector,
    GeometryFrame,
    GeometryInstance,
    HydroelasticContactRepresentation,
    ProximityProperties,
    SceneGraph,
    Sphere,
)
from pydrake.math import RigidTransform

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # experiments/ (scene.py, view_surface.py)
import scene as scene_mod
from converge_common import _setup_style, _radial_profile, _plot, _profile_resolved


def _hint_for_level(L, R):
    """resolution_hint mid-band inside Drake sphere refinement level L.
    Inverts L = max(0, ceil(log2(pi/asin(e/2R))) - 2) at log2(...) = L + 1.5."""
    return 2.0 * R * math.sin(math.pi / (2.0 ** (L + 1.5)))


def _surface(cfg, hint, nbins):
    """Frozen scene at this hint -> dict of mesh + metrics, or None if no contact."""
    sg = SceneGraph()
    src = sg.RegisterSource("converge")
    f_s = sg.RegisterFrame(src, GeometryFrame("sphere_frame"))
    f_b = sg.RegisterFrame(src, GeometryFrame("box_frame"))
    g_s = sg.RegisterGeometry(src, f_s, GeometryInstance(RigidTransform(), Sphere(cfg.R), "sphere_geo"))
    counter_shape = Box(*cfg.box_full) if cfg.counter_type == "box" else Sphere(cfg.counter_radius)
    g_b = sg.RegisterGeometry(src, f_b, GeometryInstance(RigidTransform(), counter_shape, "counter_geo"))
    for gid, E in ((g_s, cfg.E_sphere), (g_b, cfg.E_counter)):
        props = ProximityProperties()
        AddCompliantHydroelasticProperties(hint, E, props)
        sg.AssignRole(src, gid, props)

    ctx = sg.CreateDefaultContext()
    poses = FramePoseVector()
    poses.set_value(f_s, RigidTransform(p=cfg.sphere_center))
    poses.set_value(f_b, RigidTransform(p=cfg.counter_center))
    sg.get_source_pose_port(src).FixValue(ctx, poses)

    surfaces = sg.get_query_output_port().Eval(ctx).ComputeContactSurfaces(
        HydroelasticContactRepresentation.kTriangle)
    if not surfaces:
        return None
    s = surfaces[0]
    field = s.tri_e_MN()

    # Per-face area, centroid, pressure (one pass) -> aggregates + radial profile.
    A_l, c_l, p_l = [], [], []
    for i in range(s.num_faces()):
        Ae = float(s.area(i))
        if Ae <= 1e-14:
            continue
        c = np.asarray(s.centroid(i)).reshape(3)
        A_l.append(Ae)
        c_l.append(c)
        p_l.append(float(field.EvaluateCartesian(i, c)))
    if not A_l:
        return None
    A = np.array(A_l)
    C = np.array(c_l)
    P = np.array(p_l)
    area = float(A.sum())
    rc, pbar = _radial_profile(C[:, :2], A, P, np.asarray(cfg.counter_center[:2]),
                               nbins, 1.1 * cfg.reference.patch_radius)

    mesh = s.tri_mesh_W()
    verts = np.array([np.asarray(v).reshape(3) for v in mesh.vertices()])
    tris = np.array([[t.vertex(0), t.vertex(1), t.vertex(2)] for t in mesh.triangles()], int)
    vp = np.array([float(field.EvaluateAtVertex(v)) for v in range(mesh.num_vertices())])
    p = verts[tris]
    edges = np.concatenate([p[:, 0] - p[:, 1], p[:, 1] - p[:, 2], p[:, 2] - p[:, 0]])

    return dict(
        verts=verts, tris=tris, vp=vp,
        n_tris=int(tris.shape[0]),
        median_edge=float(np.median(np.linalg.norm(edges, axis=1))),
        Fn=float((A * P).sum()),
        p0_max=float(P.max()),
        equiv_r=math.sqrt(area / math.pi),
        r_centers=rc, pbar=pbar,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=scene_mod.default_scene_path())
    ap.add_argument("--lmin", type=int, default=1, help="coarsest sphere refinement level")
    ap.add_argument("--lmax", type=int, default=8, help="finest sphere refinement level")
    ap.add_argument("--bins", type=int, default=14, help="radial bins for the pressure profile")
    args = ap.parse_args()
    cfg = scene_mod.load_scene(args.scene)
    out_dir = scene_mod.experiment_dir(cfg, "drake_convergence")
    ref = cfg.reference

    levels = []
    vmax = 0.0
    for L in range(args.lmin, args.lmax + 1):
        hint = _hint_for_level(L, cfg.R)
        res = _surface(cfg, hint, args.bins)
        if res is None:
            print(f"  L={L}  hint={hint*1e3:.3f} mm  -> no contact (skipped)")
            continue
        fname = f"drake_surf_L{L}.npz"
        np.savez(os.path.join(out_dir, fname), verts=res["verts"], tris=res["tris"], vp=res["vp"])
        vmax = max(vmax, float(res["vp"].max()))
        dFn = None if not levels else abs(res["Fn"] - levels[-1]["Fn"]) / res["Fn"]
        levels.append(dict(
            L=L, hint=hint, file=fname, label=f"L{L}",
            show_profile=_profile_resolved(res["pbar"]),
            n_tris=res["n_tris"], median_edge=res["median_edge"],
            Fn=res["Fn"], p0_max=res["p0_max"], equiv_r=res["equiv_r"], dFn=dFn,
            r_centers=res["r_centers"].tolist(),
            pbar=[None if np.isnan(v) else float(v) for v in res["pbar"]],
        ))

    if not levels:
        raise SystemExit("no contact at any level; widen --lmin/--lmax or check the scene")

    # --- Table ---
    hdr = (f"{'L':>2} {'hint[mm]':>9} {'edge[mm]':>9} {'tris':>7} {'Fn[N]':>9} "
           f"{'Fn/ref':>7} {'p0/ref':>7} {'eqr/ref':>8} {'dFn%':>7}")
    print(hdr); print("-" * len(hdr))
    for lv in levels:
        d = "" if lv["dFn"] is None else f"{lv['dFn']*100:>7.3f}"
        print(f"{lv['L']:>2} {lv['hint']*1e3:>9.3f} {lv['median_edge']*1e3:>9.3f} {lv['n_tris']:>7} "
              f"{lv['Fn']:>9.2f} {lv['Fn']/ref.Fn:>7.4f} {lv['p0_max']/ref.p0_max:>7.4f} "
              f"{lv['equiv_r']/ref.patch_radius:>8.4f} {d:>7}")

    # --- Manifest ---
    manifest = dict(
        scene=os.path.abspath(args.scene),
        reference=dict(Fn=ref.Fn, area=ref.area, p0_max=ref.p0_max, patch_radius=ref.patch_radius),
        p_vmax_Pa=vmax,
        bins=args.bins,
        levels=levels,
    )
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # --- Plot (publication-quality, png + vector pdf) ---
    _setup_style()
    plot_path = os.path.join(out_dir, "drake_convergence.png")
    _plot(levels, ref, cfg, plot_path)
    print(f"\n[dump] wrote {len(levels)} meshes + manifest.json + "
          f"drake_convergence.png to {out_dir}")


if __name__ == "__main__":
    main()
