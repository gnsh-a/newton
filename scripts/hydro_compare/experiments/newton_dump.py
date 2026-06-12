"""Dump Newton's SDF hydroelastic contacts for the same frozen compliant-compliant config.

Collision only -- no solver step. Requires a CUDA device (Newton's hydroelastic SDF path
raises at finalize() on CPU). Per-contact Fn_i is a reconstruction: with no solver run,
Newton only writes contact_stiffness = area*k_eff, and we reconstruct the normal force as
stiffness * penetration.

Run (the newton-sap uv env; requires a CUDA GPU -- finalize() raises on CPU):
    cd ~/work/newton-sap
    uv run --no-sync python scripts/hydro_compare/experiments/newton_dump.py [--mesh]
  Must run in the newton-sap uv env so `newton`/`warp` import. --mesh also writes
  experiments/out/newton_surface.npz (iso-surface triangles) for view_surface.py.
"""
import argparse
import json
import os

import numpy as np
import warp as wp

import newton
from newton.geometry import HydroelasticSDF

import scene as scene_mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=scene_mod.default_scene_path())
    ap.add_argument("--mesh", action="store_true",
                    help="also write out/newton_surface.npz (actual iso-surface triangles + per-face depth)")
    args = ap.parse_args()
    cfg = scene_mod.load_scene(args.scene)
    out_dir = scene_mod.experiment_dir(cfg, "frozen_compare")

    # --- CUDA device (hydroelastic SDF cannot run on CPU) ---
    wp.init()
    device = wp.get_device("cuda:0")
    assert device.is_cuda, "Newton hydroelastic SDF requires a CUDA device"

    # --- Build the scene: two compliant hydroelastic shapes at imposed poses ---
    def shape_cfg(kh):
        return newton.ModelBuilder.ShapeConfig(
            is_hydroelastic=True, kh=kh,
            sdf_target_voxel_size=cfg.sdf_target_voxel_size,
            sdf_narrow_band_range=tuple(cfg.sdf_narrow_band_range), gap=cfg.gap)

    builder = newton.ModelBuilder()
    body_box = builder.add_body(xform=wp.transform(wp.vec3(*cfg.box_center), wp.quat_identity()))
    body_sphere = builder.add_body(xform=wp.transform(wp.vec3(*cfg.sphere_center), wp.quat_identity()))
    hx, hy, hz = (v / 2.0 for v in cfg.box_full)
    shape_box = builder.add_shape_box(body=body_box, hx=hx, hy=hy, hz=hz, cfg=shape_cfg(cfg.kh_box))
    shape_sphere = builder.add_shape_sphere(body=body_sphere, radius=cfg.R, cfg=shape_cfg(cfg.kh_sphere))

    model = builder.finalize(device=device)
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)

    # --- Verify the frozen poses took effect (the translation-only world transform
    #     below is valid ONLY for identity rotation) ---
    body_q = state.body_q.numpy()
    assert abs(body_q[body_sphere][2] - cfg.sphere_center[2]) < 1e-9, "sphere pose not applied"
    assert np.allclose(body_q[:, 3:7], [0.0, 0.0, 0.0, 1.0], atol=1e-9), "identity rotation required"

    # --- Collision only (no solver). reduce_contacts=False keeps the full unreduced
    #     surface; rigid_contact_max is sized for that contact count ---
    pipeline = newton.CollisionPipeline(
        model, broad_phase="nxn", reduce_contacts=False,
        rigid_contact_max=cfg.rigid_contact_max,
        sdf_hydroelastic_config=HydroelasticSDF.Config(
            reduce_contacts=False, output_contact_surface=args.mesh))
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)

    # --- Optional: dump Newton's ACTUAL iso-surface triangles + per-face depth/pressure ---
    if args.mesh:
        csd = pipeline.hydroelastic_sdf.get_contact_surface()
        nf = int(csd.face_contact_count.numpy()[0])
        tris = csd.contact_surface_point.numpy()[:3 * nf].reshape(nf, 3, 3)  # 3 world verts/face
        fdepth = csd.contact_surface_depth.numpy()[:nf]                      # per-face TOTAL penetration
        # contact_surface_depth is now the TOTAL overlap sdf_a + sdf_b (= d_a + d_b), not one
        # body's phi_b. The shared equal-pressure surface value is therefore p = k_eff*|depth|
        # with k_eff the series stiffness k_a*k_b/(k_a+k_b) (== kappa); this equals k_b*d_b = k_a*d_a.
        fpair = csd.contact_surface_shape_pair.numpy()[:nf]                  # (shape_a, shape_b)
        kh_all = model.shape_material_kh.numpy()
        kh_a, kh_b = kh_all[fpair[:, 0]], kh_all[fpair[:, 1]]
        k_eff_face = kh_a * kh_b / (kh_a + kh_b)
        fpress = k_eff_face * np.abs(fdepth)                                 # p = k_eff * |delta_total|
        os.makedirs(out_dir, exist_ok=True)
        np.savez(os.path.join(out_dir, "newton_surface.npz"),
                 tris=tris, depth=fdepth, pressure=fpress)

    # --- Read per-contact arrays ---
    n = int(contacts.rigid_contact_count.numpy()[0])
    assert n < cfg.rigid_contact_max, \
        f"contact buffer truncated: count {n} >= rigid_contact_max {cfg.rigid_contact_max}"
    normal = contacts.rigid_contact_normal.numpy()[:n]      # shape0 -> shape1
    stiffness = contacts.rigid_contact_stiffness.numpy()[:n]
    p0 = contacts.rigid_contact_point0.numpy()[:n]          # child-body frame
    p1 = contacts.rigid_contact_point1.numpy()[:n]
    shape0 = contacts.rigid_contact_shape0.numpy()[:n]
    shape1 = contacts.rigid_contact_shape1.numpy()[:n]

    # --- Witness points -> world (identity rotation => translation-only). The exported
    #     contact_distance is now the TOTAL overlap -(d_a + d_b); witness points are split
    #     +/-0.5*contact_distance, so (p0w - p1w).(-n) recovers it directly (no /2). ---
    body_t = body_q[:, :3]
    shape_body = model.shape_body.numpy()
    p0w = p0 + body_t[shape_body[shape0]]
    p1w = p1 + body_t[shape_body[shape1]]
    depth_total = np.einsum("ij,ij->i", p0w - p1w, -normal)        # contact_distance = -(d_a+d_b)
    mask = depth_total < 0.0
    n_pen = int(mask.sum())
    assert n_pen > 0, (
        f"no penetrating contacts (raw count {n}); decrease mesh.target_edge "
        f"(currently {cfg.sdf_target_voxel_size}) or increase contact.penetration_x")
    normal, stiffness = normal[mask], stiffness[mask]
    p0w, p1w, depth_total = p0w[mask], p1w[mask], depth_total[mask]
    shape0, shape1 = shape0[mask], shape1[mask]

    # --- Per-contact quantities. depth_total is the TOTAL penetration d_a+d_b (signed, neg).
    #     The equal-pressure surface value is p = k_eff*|depth_total| (= k_b*d_b = k_a*d_a);
    #     this is the series law and is INDEPENDENT of which shape was finer-voxeled. Patch
    #     force is the hydroelastic integral int p dA = area*p = stiffness*|depth_total|.
    #     depth keeps the single-body phi_b (signed) for the 1-body diagnostic marker. ---
    kh = model.shape_material_kh.numpy()
    k_eff = (kh[shape0] * kh[shape1]) / (kh[shape0] + kh[shape1])   # series (= surface k_eff)
    area = stiffness / k_eff                                       # stiffness = area * k_eff
    pressure = k_eff * (-depth_total)                             # p = k_eff * |delta_total|
    depth = depth_total * kh[shape0] / (kh[shape0] + kh[shape1])   # single-body phi_b = d_total*k_a/(k_a+k_b)
    Fn_i = area * pressure                                         # int p dA on the patch
    point_W = 0.5 * (p0w + p1w)

    # --- Force-weighted aggregates; reduce to force ON THE SPHERE (normal is shape0->shape1) ---
    sign_sphere = np.where(shape0 == shape_sphere, -1.0, 1.0)[:, None]
    f_on_sphere = (Fn_i[:, None] * sign_sphere * normal).sum(0)
    Fn_total = float(Fn_i.sum())
    centroid = (Fn_i[:, None] * point_W).sum(0) / Fn_total

    # --- Write outputs: per-contact arrays (npz) + aggregates (json) ---
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "newton.npz"),
             centroid_W=point_W, area=area, n_hat=normal, depth=depth, depth_total=depth_total,
             stiffness=stiffness, pressure=pressure, Fn_i=Fn_i)
    meta = {
        "k_eff_mean": float(k_eff.mean()),
        "agg_Fn_total": Fn_total,
        "agg_net_force_on_sphere_W": f_on_sphere.tolist(),
        "agg_centroid_W": centroid.tolist(),
        "agg_total_area": float(area.sum()),
    }
    with open(os.path.join(out_dir, "newton.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
