"""Dump Drake's hydroelastic contact surface for one frozen compliant-compliant config.

Pure SceneGraph -- no MultibodyPlant, no solver, no dynamics. We register a compliant
sphere and a compliant box on SEPARATE frames (geometries on the same frame are
collision-filtered), impose their world poses via the source pose port, then evaluate
the QueryObject and ComputeContactSurfaces(kTriangle). The per-face SAP discrete-contact
inputs (g, k, phi0, fn0) are recomputed from PUBLIC accessors exactly per
multibody/plant/discrete_update_manager.cc:885-995, so no Drake patch is required.

Friction (AddContactMaterial/CoulombFriction) is intentionally omitted: it is read only
by MultibodyPlant force computation downstream and never by ComputeContactSurfaces, so
the ContactSurface produced here is identical with or without it.

Run (the dedicated uv project here; uses the pydrake PyPI wheel -- no Drake checkout):
    cd ~/work/newton-sap/scripts/hydro_compare      # uv project root (pyproject.toml lives here)
    uv run python experiments/drake_dump.py [--mesh]
  --mesh also writes experiments/out/drake_surface.npz (triangle mesh) for view_surface.py.
"""
import argparse
import json
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

import scene as scene_mod


def _make_props(resolution_hint, modulus):
    props = ProximityProperties()
    AddCompliantHydroelasticProperties(resolution_hint, modulus, props)
    return props


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=scene_mod.default_scene_path())
    ap.add_argument("--mesh", action="store_true",
                    help="also write out/drake_surface.npz (mesh + per-vertex pressure) for view_surface.py")
    args = ap.parse_args()
    cfg = scene_mod.load_scene(args.scene)

    # --- Build the scene: two compliant geometries on SEPARATE frames ---
    sg = SceneGraph()
    src = sg.RegisterSource("contact_surface_xcheck")
    # Separate frames: two geometries on the SAME frame would be collision-filtered.
    f_sphere = sg.RegisterFrame(src, GeometryFrame("sphere_frame"))
    f_box = sg.RegisterFrame(src, GeometryFrame("box_frame"))
    g_sphere = sg.RegisterGeometry(
        src, f_sphere, GeometryInstance(RigidTransform(), Sphere(cfg.R), "sphere_geo"))
    g_box = sg.RegisterGeometry(
        src, f_box, GeometryInstance(RigidTransform(), Box(*cfg.box_full), "box_geo"))
    sg.AssignRole(src, g_sphere, _make_props(cfg.resolution_hint, cfg.E_sphere))
    sg.AssignRole(src, g_box, _make_props(cfg.resolution_hint, cfg.E_box))

    # --- Impose the frozen world poses via the source pose port ---
    ctx = sg.CreateDefaultContext()
    poses = FramePoseVector()
    poses.set_value(id=f_sphere, value=RigidTransform(p=cfg.sphere_center))
    poses.set_value(id=f_box, value=RigidTransform(p=cfg.box_center))
    sg.get_source_pose_port(src).FixValue(ctx, poses)

    # --- Compute the hydroelastic contact surface (pure geometry query) ---
    query_object = sg.get_query_output_port().Eval(ctx)
    surfaces = query_object.ComputeContactSurfaces(
        HydroelasticContactRepresentation.kTriangle)
    assert len(surfaces) == 1, f"expected exactly 1 contact surface, got {len(surfaces)}"
    s = surfaces[0]
    assert s.HasGradE_M() and s.HasGradE_N(), \
        "need both pressure gradients (expected for compliant-compliant)"
    field = s.tri_e_MN()

    # --- Optional: dump the surface mesh (verts, tris, per-vertex pressure) for viz ---
    if args.mesh:
        mesh = s.tri_mesh_W()
        verts = np.array([np.asarray(v).reshape(3) for v in mesh.vertices()])
        tris = np.array([[t.vertex(0), t.vertex(1), t.vertex(2)]
                         for t in mesh.triangles()], dtype=int)
        vp = np.array([float(field.EvaluateAtVertex(v)) for v in range(mesh.num_vertices())])
        os.makedirs(cfg.output_dir, exist_ok=True)
        np.savez(os.path.join(cfg.output_dir, "drake_surface.npz"), verts=verts, tris=tris, vp=vp)

    # --- Per-face SAP inputs, faithful to discrete_update_manager.cc:885-995 ---
    rows = []
    for i in range(s.num_faces()):
        Ae = float(s.area(i))
        if Ae <= 1e-14:
            continue
        nhat = np.asarray(s.face_normal(i)).reshape(3)   # out of N into M
        c = np.asarray(s.centroid(i)).reshape(3)
        gM = float(np.asarray(s.EvaluateGradE_M_W(i)).reshape(3) @ nhat)
        gN = float(-(np.asarray(s.EvaluateGradE_N_W(i)).reshape(3) @ nhat))
        if gM < 1e-14 or gN < 1e-14:
            continue
        g = 1.0 / (1.0 / gM + 1.0 / gN)
        p0 = float(field.EvaluateCartesian(i, c))
        rows.append((c, Ae, nhat, p0, g, Ae * g, -p0 / g, Ae * p0))

    assert rows, "no valid contact faces produced"
    cols = list(zip(*rows))
    centroid_W = np.array(cols[0]); area = np.array(cols[1]); n_hat = np.array(cols[2])
    p0 = np.array(cols[3]); g = np.array(cols[4]); k = np.array(cols[5])
    phi0 = np.array(cols[6]); fn0 = np.array(cols[7])

    # --- Force-weighted aggregates (consumed by compare.py) ---
    Fn_total = float(fn0.sum())
    net_normal = (fn0[:, None] * n_hat).sum(0)
    centroid = (fn0[:, None] * centroid_W).sum(0) / Fn_total

    # --- Write outputs: per-face arrays (npz) + aggregates/ids (json) ---
    os.makedirs(cfg.output_dir, exist_ok=True)
    np.savez(os.path.join(cfg.output_dir, "drake.npz"),
             centroid_W=centroid_W, area=area, n_hat=n_hat, p0=p0,
             g=g, k=k, phi0=phi0, fn0=fn0)
    meta = {
        "id_M": s.id_M().get_value(),
        "id_sphere": g_sphere.get_value(),
        "agg_Fn_total": Fn_total,
        "agg_net_normal_W": net_normal.tolist(),  # sum(fn0 * n_hat); +n_hat is force into M
        "agg_centroid_W": centroid.tolist(),
        "agg_total_area": float(area.sum()),
    }
    with open(os.path.join(cfg.output_dir, "drake.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
