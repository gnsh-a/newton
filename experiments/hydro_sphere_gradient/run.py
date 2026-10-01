# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Full-patch equal-sphere experiment for issue #3503.

uv run --frozen python experiments/hydro_sphere_gradient/run.py

The projected variant uses exact radial sphere gradients on Newton's exported
contacts to isolate their normal stiffness response.
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton.geometry import HydroelasticSDF
from newton.solvers import SolverSemiImplicit

RADIUS = 0.05
KH = 1.0e6
MASS = 1.0
OVERLAP = 0.02
VOXEL = 0.0005
DT = 0.0005
DURATION = 0.6
VELOCITY = 0.02
CAPACITY = 200000
ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent.parent


def exact_force(overlap):
    """Return the exact equal-sphere hydroelastic force [N]."""
    return np.pi * KH * overlap**2 * (3.0 * RADIUS - overlap) / 12.0


def exact_stiffness(overlap):
    """Return the overlap derivative of the exact force [N/m]."""
    return np.pi * KH * overlap * (2.0 * RADIUS - overlap) / 4.0


@wp.kernel
def project_sphere_gradients(
    count: wp.array[wp.int32],
    shape0: wp.array[wp.int32],
    shape1: wp.array[wp.int32],
    shape_body: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    point0: wp.array[wp.vec3],
    point1: wp.array[wp.vec3],
    normal: wp.array[wp.vec3],
    stiffness: wp.array[wp.float32],
):
    """Correct equal-material sphere springs while preserving force and midpoint."""
    tid = wp.tid()
    if tid >= count[0]:
        return
    b0 = shape_body[shape0[tid]]
    b1 = shape_body[shape1[tid]]
    t0 = wp.transform_identity()
    t1 = wp.transform_identity()
    if b0 >= 0:
        t0 = body_q[b0]
    if b1 >= 0:
        t1 = body_q[b1]
    p0 = wp.transform_point(t0, point0[tid])
    p1 = wp.transform_point(t1, point1[tid])
    mid = 0.5 * (p0 + p1)
    n = normal[tid]
    ca = wp.dot(wp.normalize(mid - wp.transform_get_translation(t0)), n)
    cb = -wp.dot(wp.normalize(mid - wp.transform_get_translation(t1)), n)
    old_distance = wp.dot(p1 - p0, n)
    if ca > 0.0 and cb > 0.0 and old_distance < 0.0:
        factor = 2.0 * ca * cb / (ca + cb)
        # Equal linear pressure laws give k_current = area * kh / 2.
        # Preserve the current force and centre while changing its tangent.
        delta = old_distance * (1.0 / factor - 1.0)
        point0[tid] = wp.transform_point(wp.transform_inverse(t0), p0 - 0.5 * delta * n)
        point1[tid] = wp.transform_point(wp.transform_inverse(t1), p1 + 0.5 * delta * n)
        stiffness[tid] = stiffness[tid] * factor


class SphereScene:
    """Build an unreduced sphere patch with one world-fixed and one free sphere."""

    def __init__(self):
        self.initial_height = 2.0 * RADIUS - OVERLAP
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -exact_force(OVERLAP) / MASS))
        cfg = newton.ModelBuilder.ShapeConfig(
            is_hydroelastic=True,
            kh=KH,
            kd=0.0,
            kf=0.0,
            mu=0.0,
            margin=0.0,
            gap=0.0,
            sdf_target_voxel_size=VOXEL,
            sdf_narrow_band_range=(-0.025, 0.01),
        )
        builder.add_shape_sphere(body=-1, radius=RADIUS, cfg=cfg)
        inertia = 0.4 * MASS * RADIUS**2
        body = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, self.initial_height), wp.quat_identity()),
            mass=MASS,
            inertia=wp.mat33(inertia, 0.0, 0.0, 0.0, inertia, 0.0, 0.0, 0.0, inertia),
            lock_inertia=True,
        )
        builder.add_shape_sphere(body=body, radius=RADIUS, cfg=cfg)
        self.model = builder.finalize(device="cuda:0")
        self.state = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        self.pipeline = newton.CollisionPipeline(
            self.model,
            rigid_contact_max=CAPACITY,
            sdf_hydroelastic_config=HydroelasticSDF.Config(
                reduce_contacts=False,
                pre_prune_contacts=False,
                anchor_contact=False,
                output_contact_surface=False,
                mc_edge_clamp_min=0.0,
            ),
        )
        self.contacts = self.pipeline.contacts()

    def collide(self, projected=False):
        self.pipeline.collide(self.state, self.contacts)
        count = int(self.contacts.rigid_contact_count.numpy()[0])
        if count <= 0 or count >= CAPACITY:
            raise RuntimeError(f"Invalid contact count: {count}")
        if projected:
            wp.launch(
                project_sphere_gradients,
                dim=count,
                inputs=[
                    self.contacts.rigid_contact_count,
                    self.contacts.rigid_contact_shape0,
                    self.contacts.rigid_contact_shape1,
                    self.model.shape_body,
                    self.state.body_q,
                    self.contacts.rigid_contact_point0,
                    self.contacts.rigid_contact_point1,
                    self.contacts.rigid_contact_normal,
                    self.contacts.rigid_contact_stiffness,
                ],
                device=self.model.device,
            )
        return count

    def measure(self):
        count = int(self.contacts.rigid_contact_count.numpy()[0])
        c = self.contacts
        shape_body = self.model.shape_body.numpy()
        b0 = shape_body[c.rigid_contact_shape0.numpy()[:count]]
        b1 = shape_body[c.rigid_contact_shape1.numpy()[:count]]
        q = self.state.body_q.numpy()[0]
        np.testing.assert_allclose(q[3:], [0.0, 0.0, 0.0, 1.0], atol=1.0e-6)
        p0 = c.rigid_contact_point0.numpy()[:count].astype(np.float64)
        p1 = c.rigid_contact_point1.numpy()[:count].astype(np.float64)
        p0[b0 >= 0] += q[:3]
        p1[b1 >= 0] += q[:3]
        n = c.rigid_contact_normal.numpy()[:count].astype(np.float64)
        k = c.rigid_contact_stiffness.numpy()[:count].astype(np.float64)
        distance = np.einsum("ij,ij->i", p1 - p0, n)
        f = np.maximum(-k * distance, 0.0)
        direction = np.where(b1 >= 0, 1.0, -1.0)
        return {
            "contacts": count,
            "force_N": float(np.sum(f * n[:, 2] * direction)),
            "stiffness_N_m": float(np.sum(k * n[:, 2] ** 2)),
        }


def dynamic_case(refresh, projected):
    """Measure oscillation error against the exact-force discrete reference."""
    scene = SphereScene()
    solver = SolverSemiImplicit(scene.model, angular_damping=0.0)
    state_next = scene.model.state()
    velocity = scene.state.body_qd.numpy()
    velocity[0, 2] = VELOCITY
    scene.state.body_qd.assign(velocity)
    ref_x, ref_v = 0.0, VELOCITY
    positions = []
    lateral_drift = 0.0
    steps = round(DURATION / DT)
    for step in range(steps + 1):
        position = scene.state.body_q.numpy()[0, :3]
        positions.append([float(position[2] - scene.initial_height), ref_x])
        lateral_drift = max(lateral_drift, float(np.linalg.norm(position[:2])))
        if step == steps:
            break
        if step % refresh == 0:
            scene.collide(projected=projected)
        scene.state.clear_forces()
        solver.step(scene.state, state_next, None, scene.contacts, DT)
        scene.state, state_next = state_next, scene.state
        ref_v += DT * (exact_force(OVERLAP - ref_x) - exact_force(OVERLAP)) / MASS
        ref_x += DT * ref_v
    positions = np.asarray(positions)
    assert np.isfinite(positions).all()
    assert lateral_drift < 1.0e-5
    error = np.sqrt(np.mean((positions[:, 0] - positions[:, 1]) ** 2))
    error_percent = float(100 * error / np.max(np.abs(positions[:, 1])))
    return error_percent, positions[:, 0]


def main():
    """Run one static case and three contact-refresh comparisons."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results.json")
    args = parser.parse_args()
    if Path(newton.__file__).resolve().parent != REPO_ROOT / "newton":
        raise RuntimeError("Run with this checkout's uv environment")
    wp.init()
    if not wp.is_cuda_available():
        raise RuntimeError("CUDA is required for the texture-SDF collision pipeline")

    scene = SphereScene()
    scene.collide()
    original = scene.measure()
    scene.collide(projected=True)
    projected = scene.measure()
    reference = exact_stiffness(OVERLAP)
    np.testing.assert_allclose(projected["force_N"], original["force_N"], rtol=2.0e-5)
    np.testing.assert_allclose(original["force_N"], exact_force(OVERLAP), rtol=0.001)
    np.testing.assert_allclose(projected["stiffness_N_m"], reference, rtol=0.005)
    assert 0.115 < original["stiffness_N_m"] / reference - 1 < 0.135

    comparisons = []
    for refresh in (1, 8, 16):
        original_error, original_motion = dynamic_case(refresh, False)
        projected_error, projected_motion = dynamic_case(refresh, True)
        difference = float(np.max(np.abs(original_motion - projected_motion)))
        if refresh == 1:
            assert difference < 1.0e-6
        else:
            assert projected_error < original_error / 5
        comparisons.append(
            {
                "refresh_steps": refresh,
                "original_rms_error_percent": original_error,
                "projected_rms_error_percent": projected_error,
                "max_position_difference_m": difference,
            }
        )

    results = {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        "warp_version": wp.__version__,
        "device": wp.get_device("cuda:0").name,
        "setup": {
            "radius_m": RADIUS,
            "kh_Pa_m": KH,
            "mass_kg": MASS,
            "overlap_m": OVERLAP,
            "target_voxel_m": VOXEL,
            "dt_s": DT,
            "duration_s": DURATION,
            "initial_velocity_m_s": VELOCITY,
        },
        "exact_force_N": exact_force(OVERLAP),
        "exact_stiffness_N_m": reference,
        "original": original,
        "projected": projected,
        "motion": comparisons,
        "checks_passed": True,
    }
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    print(f"PASS. Results: {args.output}")


if __name__ == "__main__":
    main()
