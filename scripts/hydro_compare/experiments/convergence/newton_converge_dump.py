"""Newton SDF voxel-resolution sweep: dump one contact iso-surface per voxel size + metrics + plot.

Mirror of drake_converge_dump.py for Newton's native SDF hydroelastic pipeline. The refinement
knob here is ``sdf_target_voxel_size`` (continuous, unlike Drake's quantized tet levels): smaller
voxels -> finer SDF grid -> finer equal-pressure iso-surface. Physics is held fixed from
scene.yaml; only the voxel size changes. Records the SAME metrics as the Drake study against the
SAME Winkler reference, plotted on the SAME x-axis (median iso-surface triangle edge), so the two
sweeps are directly comparable.

  * scalar aggregates : Fn = sum(area_face * p_face), peak pressure, equiv radius sqrt(A/pi)
  * convergence       : successive relative change in Fn
  * mesh descriptors  : iso-surface triangle count, median edge (the convergence x-axis)
  * pressure profile  : area-weighted radial p_bar(r)

reduce_contacts AND pre-prune stay OFF so the FULL (undecimated) iso-surface is measured.

Run in the newton-sap uv env (requires a CUDA GPU; finalize() raises on CPU):
    cd ~/work/newton-sap
    uv run --no-sync python scripts/hydro_compare/experiments/convergence/newton_converge_dump.py \
        [--voxels 6 4 3 2 1.5 1 0.75] [--bins 14] [--scene path]
Writes out/newton_convergence/: newton_surf_v{mm}.npz (per voxel), manifest.json, and one
publication figure newton_convergence.png (2x3, all non-surface metrics).
Pair with converge_view.py --manifest .../newton_convergence/manifest.json for the surface grid.
"""
import argparse
import json
import math
import os

import numpy as np
import warp as wp

import newton
from newton.geometry import HydroelasticSDF

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # experiments/ (scene.py, view_surface.py)
import scene as scene_mod
from converge_common import _setup_style, _radial_profile, _plot, _profile_resolved


def _surface(cfg, voxel, nbins, device, box_voxel=None, rigid_contact_max=None):
    """Frozen sphere-on-box contact; sphere SDF at ``voxel`` -> dict of mesh + metrics, or None.

    box_voxel pins the box SDF resolution (default: same as the sphere). Pinning it coarse
    (a) keeps the sphere strictly finer so it ALWAYS hosts the iso-voxel grid / marching cubes
    (no silent host flip on realized-voxel-radius ties), and (b) costs no accuracy: the patch
    sits mid-face where the box SDF is linear and trilinear interpolation is exact (A/B
    verified: box 6mm vs 1mm at sphere 1mm -> Fn identical to 1e-5)."""
    def shape_cfg(kh, v):
        return newton.ModelBuilder.ShapeConfig(
            is_hydroelastic=True, kh=kh, sdf_target_voxel_size=v,
            sdf_narrow_band_range=tuple(cfg.sdf_narrow_band_range), gap=cfg.gap)

    builder = newton.ModelBuilder()
    body_counter = builder.add_body(xform=wp.transform(wp.vec3(*cfg.counter_center), wp.quat_identity()))
    body_sphere = builder.add_body(xform=wp.transform(wp.vec3(*cfg.sphere_center), wp.quat_identity()))
    if cfg.counter_type == "box":
        # box SDF pinned coarse (inert mid-face) so the sphere always hosts marching cubes
        hx, hy, hz = (v / 2.0 for v in cfg.box_full)
        builder.add_shape_box(body=body_counter, hx=hx, hy=hy, hz=hz,
                              cfg=shape_cfg(cfg.kh_box, box_voxel if box_voxel is not None else voxel))
    else:  # curved: second sphere; both bodies refine at the sweep voxel (no pinning)
        builder.add_shape_sphere(body=body_counter, radius=cfg.counter_radius,
                                 cfg=shape_cfg(cfg.kh_counter, voxel))
    builder.add_shape_sphere(body=body_sphere, radius=cfg.R, cfg=shape_cfg(cfg.kh_sphere, voxel))

    model = builder.finalize(device=device)
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    assert np.allclose(state.body_q.numpy()[:, 3:7], [0, 0, 0, 1], atol=1e-9), "identity rotation required"

    # Collision only (no solver). Full surface: reduce + pre-prune OFF, output_contact_surface ON.
    rcm = rigid_contact_max if rigid_contact_max is not None else cfg.rigid_contact_max
    pipeline = newton.CollisionPipeline(
        model, broad_phase="nxn", reduce_contacts=False, rigid_contact_max=rcm,
        sdf_hydroelastic_config=HydroelasticSDF.Config(reduce_contacts=False, output_contact_surface=True))
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)

    csd = pipeline.hydroelastic_sdf.get_contact_surface()
    nf = int(csd.face_contact_count.numpy()[0])
    if nf == 0:
        return None
    tris = csd.contact_surface_point.numpy()[:3 * nf].reshape(nf, 3, 3)   # 3 world verts/face
    fdepth = csd.contact_surface_depth.numpy()[:nf]                       # TOTAL overlap d_a+d_b
    fpair = csd.contact_surface_shape_pair.numpy()[:nf]                   # (shape_a, shape_b)
    kh_all = model.shape_material_kh.numpy()
    kh_a, kh_b = kh_all[fpair[:, 0]], kh_all[fpair[:, 1]]
    k_eff = kh_a * kh_b / (kh_a + kh_b)                                   # series stiffness (= kappa)
    P = k_eff * np.abs(fdepth)                                           # per-face pressure p = k_eff*|delta|

    # Per-face area + centroid from the vertex triples (Fn = int p dA via midpoint).
    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
    A = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    C = tris.mean(axis=1)
    keep = A > 1e-14
    if not keep.any():
        return None
    A, C, P, tris = A[keep], C[keep], P[keep], tris[keep]
    area = float(A.sum())
    rc, pbar = _radial_profile(C[:, :2], A, P, np.asarray(cfg.counter_center[:2]),
                               nbins, 1.1 * cfg.reference.patch_radius)

    # Non-indexed mesh for the viewer: 3 verts/face, per-vertex pressure = its (flat) face value.
    verts = tris.reshape(-1, 3)
    tri_idx = np.arange(verts.shape[0]).reshape(-1, 3)
    vp = np.repeat(P, 3)
    edges = np.concatenate([tris[:, 0] - tris[:, 1], tris[:, 1] - tris[:, 2], tris[:, 2] - tris[:, 0]])

    return dict(
        verts=verts, tris=tri_idx, vp=vp,
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
    ap.add_argument("--voxels", type=float, nargs="+", default=[6, 4, 3, 2, 1.5, 1, 0.75],
                    help="sphere SDF voxel sizes to sweep, in mm (coarse -> fine)")
    ap.add_argument("--box-voxel", type=float, default=6.0,
                    help="pinned box SDF voxel size in mm (keeps the sphere the MC host; "
                         "box resolution is inert mid-face). Pass 0 to refine the box too.")
    ap.add_argument("--bins", type=int, default=14, help="radial bins for the pressure profile")
    ap.add_argument("--rigid-contact-max", type=int, default=None,
                    help="override scene rigid_contact_max (raise for fine voxels that overflow 6000)")
    args = ap.parse_args()
    cfg = scene_mod.load_scene(args.scene)
    out_dir = scene_mod.experiment_dir(cfg, "newton_convergence")
    ref = cfg.reference

    wp.init()
    device = wp.get_device("cuda:0")
    assert device.is_cuda, "Newton hydroelastic SDF requires a CUDA device"

    box_voxel = args.box_voxel * 1e-3 if args.box_voxel > 0 else None
    levels = []
    vmax = 0.0
    for v_mm in args.voxels:
        voxel = v_mm * 1e-3
        res = _surface(cfg, voxel, args.bins, device, box_voxel=box_voxel,
                       rigid_contact_max=args.rigid_contact_max)
        if res is None:
            print(f"  voxel={v_mm:.3f} mm  -> no contact (skipped)")
            continue
        fname = f"newton_surf_v{v_mm:g}.npz"
        np.savez(os.path.join(out_dir, fname), verts=res["verts"], tris=res["tris"], vp=res["vp"])
        vmax = max(vmax, float(res["vp"].max()))
        dFn = None if not levels else abs(res["Fn"] - levels[-1]["Fn"]) / res["Fn"]
        levels.append(dict(
            voxel=voxel, file=fname, label=f"{v_mm:g}mm",
            show_profile=_profile_resolved(res["pbar"]),
            n_tris=res["n_tris"], median_edge=res["median_edge"],
            Fn=res["Fn"], p0_max=res["p0_max"], equiv_r=res["equiv_r"], dFn=dFn,
            r_centers=res["r_centers"].tolist(),
            pbar=[None if np.isnan(v) else float(v) for v in res["pbar"]],
        ))

    if not levels:
        raise SystemExit("no contact at any voxel size; coarsen --voxels or check the scene")

    # --- Table ---
    hdr = (f"{'voxel[mm]':>9} {'edge[mm]':>9} {'tris':>8} {'Fn[N]':>9} "
           f"{'Fn/ref':>7} {'p0/ref':>7} {'eqr/ref':>8} {'dFn%':>7}")
    print(hdr); print("-" * len(hdr))
    for lv in levels:
        d = "" if lv["dFn"] is None else f"{lv['dFn']*100:>7.3f}"
        print(f"{lv['voxel']*1e3:>9.3f} {lv['median_edge']*1e3:>9.3f} {lv['n_tris']:>8} "
              f"{lv['Fn']:>9.2f} {lv['Fn']/ref.Fn:>7.4f} {lv['p0_max']/ref.p0_max:>7.4f} "
              f"{lv['equiv_r']/ref.patch_radius:>8.4f} {d:>7}")

    # --- Manifest (same schema as the Drake study; engine tag distinguishes them) ---
    manifest = dict(
        engine="newton",
        box_voxel=box_voxel,
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
    plot_path = os.path.join(out_dir, "newton_convergence.png")
    _plot(levels, ref, cfg, plot_path)
    print(f"\n[dump] wrote {len(levels)} meshes + manifest.json + "
          f"newton_convergence.png to {out_dir}")


if __name__ == "__main__":
    main()
