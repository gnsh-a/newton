# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Hydroelastic Sphere Pair
#
# Minimal hydroelastic SDF sphere-pair demo with a reduce_contacts switch.
# The CSV trace exposes mode-dependent contact counts and dropper motion.
#
# Run modes:
#     python -m newton.examples hydro_sphere_pair                       # reduce on (default)
#     python -m newton.examples hydro_sphere_pair --no-reduce-contacts  # reduce off
#
# Command: python -m newton.examples hydro_sphere_pair
#
###########################################################################

import os

import numpy as np
import trimesh
import warp as wp

import newton
import newton.examples
from newton.geometry import HydroelasticSDF

SPHERE_RADIUS = 0.05
"""Radius of each icosphere [m]."""
INITIAL_GAP = 0.0
"""Initial clear-air gap between the two spheres before loading [m]."""
ICOSPHERE_SUBDIVISIONS = 3
"""Icosphere refinement level (3 -> 1280 triangles)."""
GRAVITY_Z = -2.0
"""Gentle gravity used to load the reduced sphere cap without leaving the SDF narrow band [m/s^2]."""

SDF_MAX_RESOLUTION = 64
"""Maximum SDF grid resolution along the longest mesh axis."""
SDF_NARROW_BAND_RANGE = (-0.005, 0.005)
"""SDF narrow-band range [m] used during baking."""
SDF_MARGIN = 0.005
"""Outward margin baked into the SDF surface [m]."""

KH = 1.0e9
"""Hydroelastic stiffness coefficient [Pa].

Note: Drake's ``examples/hydroelastic/two_spheres`` uses
``hydroelastic_modulus = 5e5 Pa``. Newton's ``kh`` enters the contact
constraint as ``contact_stiffness = area * k_eff`` (N/m) and is *not*
unit-equivalent to Drake's pressure-field modulus — setting ``kh = 5e5``
here produces a contact too weak to catch the dropper (depth saturates
the SDF narrow band and the body tunnels)."""
SPHERE_DENSITY = 191.0
"""Density [kg/m^3] chosen so a 0.05 m sphere weighs 0.1 kg, matching Drake."""

RIGID_CONTACT_MAX = 8192
"""Upper bound on rigid contacts allocated by the collision pipeline."""
MUJOCO_NCONMAX = 8192
"""MuJoCo contact buffer capacity (must be >= RIGID_CONTACT_MAX)."""
MUJOCO_NJMAX = 16384
"""MuJoCo constraint Jacobian row capacity (~ 3 * active contacts under elliptic cone)."""

OUTPUT_DIR = os.path.join("output", "hydro_sphere_pair")
"""Where CSV traces land, relative to the process cwd."""


def _build_icosphere_mesh(radius, subdivisions, sdf_resolution, narrow_band_range, margin):
    tm = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    vertices = np.asarray(tm.vertices, dtype=np.float32)
    indices = np.asarray(tm.faces, dtype=np.int32).flatten()
    mesh = newton.Mesh(vertices, indices)
    mesh.build_sdf(
        max_resolution=sdf_resolution,
        narrow_band_range=narrow_band_range,
        margin=margin,
    )
    return mesh


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.reduce_contacts = bool(args.reduce_contacts)
        # Close batch runs after the experiment window ends.
        self.auto_close_after_freeze = bool(getattr(args, "test", False)) or bool(getattr(args, "headless", False))
        self._viewer_closed = False

        self.fps = 120
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        # Long enough for the gentle contact load to settle.
        self.t_end = 1.2

        mesh = _build_icosphere_mesh(
            radius=SPHERE_RADIUS,
            subdivisions=ICOSPHERE_SUBDIVISIONS,
            sdf_resolution=SDF_MAX_RESOLUTION,
            narrow_band_range=SDF_NARROW_BAND_RANGE,
            margin=SDF_MARGIN,
        )

        shape_cfg = newton.ModelBuilder.ShapeConfig(
            margin=0.0,
            gap=0.001,
            mu=0.5,
            ke=1.0e7,
            kd=1.0e4,
            density=SPHERE_DENSITY,
            is_hydroelastic=True,
            kh=KH,
        )

        builder = newton.ModelBuilder(gravity=GRAVITY_Z)
        builder.default_shape_cfg.gap = 0.001

        builder.add_shape_mesh(
            body=-1,
            xform=wp.transform(
                wp.vec3(0.0, 0.0, SPHERE_RADIUS),
                wp.quat_identity(),
            ),
            mesh=mesh,
            cfg=shape_cfg,
            label="floor_sphere",
        )

        self.initial_dropper_z = 3.0 * SPHERE_RADIUS + INITIAL_GAP
        self.dropper_body = builder.add_body(
            xform=wp.transform(
                wp.vec3(0.0, 0.0, self.initial_dropper_z),
                wp.quat_identity(),
            ),
            label="dropper",
        )
        builder.add_shape_mesh(
            body=self.dropper_body,
            mesh=mesh,
            cfg=shape_cfg,
            label="dropper_sphere",
        )

        self.model = builder.finalize()

        # Increase buffers so reduce-off can keep dense face contacts.
        hydro_cfg = HydroelasticSDF.Config(
            output_contact_surface=True,
            reduce_contacts=self.reduce_contacts,
            buffer_mult_iso=4,
            buffer_mult_contact=4,
        )
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            sdf_hydroelastic_config=hydro_cfg,
            rigid_contact_max=RIGID_CONTACT_MAX,
            broad_phase="sap",
        )

        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            use_mujoco_contacts=False,
            solver="newton",
            integrator="implicitfast",
            cone="elliptic",
            njmax=MUJOCO_NJMAX,
            nconmax=MUJOCO_NCONMAX,
            iterations=15,
            ls_iterations=100,
            impratio=1.0,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.contacts = self.collision_pipeline.contacts()
        self.collision_pipeline.collide(self.state_0, self.contacts)

        self.viewer.set_model(self.model)
        self.viewer.show_contacts = True
        self.viewer.show_hydro_contact_surface = True
        self.viewer.set_camera(
            pos=wp.vec3(0.4, -0.4, 0.2),
            pitch=-15.0,
            yaw=135.0,
        )

        self.time_log: list[float] = []
        self.z_log: list[float] = []
        self.vz_log: list[float] = []
        self.depth_log: list[float] = []
        self.face_contact_count_log: list[int] = []
        self.rigid_contact_count_log: list[int] = []
        self.max_reduced_contacts = 0
        self.max_face_contacts = 0

        self._log_state()

        self.capture()

    def _log_state(self) -> None:
        body_q = self.state_0.body_q.numpy()[self.dropper_body]
        body_qd = self.state_0.body_qd.numpy()[self.dropper_body]
        z = float(body_q[2])
        # body_qd stores linear velocity first: (vx, vy, vz, wx, wy, wz).
        vz = float(body_qd[2])
        # Geometric overlap, not solver-resolved penetration.
        depth = max(0.0, 2.0 * SPHERE_RADIUS - abs(z - SPHERE_RADIUS))
        self.time_log.append(self.sim_time)
        self.z_log.append(z)
        self.vz_log.append(vz)
        self.depth_log.append(depth)

        # face_contact_count is post pre-prune when reduce_contacts=True.
        # rigid_contact_count is the solver input after reduction.
        rigid = int(self.contacts.rigid_contact_count.numpy()[0])
        self.rigid_contact_count_log.append(rigid)
        if self.collision_pipeline.hydroelastic_sdf is not None:
            surf = self.collision_pipeline.hydroelastic_sdf.get_contact_surface()
            face = int(surf.face_contact_count.numpy()[0])
        else:
            face = 0
        self.face_contact_count_log.append(face)

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
        # Refresh contacts so diagnostics match the advanced state.
        self.collision_pipeline.collide(self.state_0, self.contacts)

    def step(self):
        if self.sim_time >= self.t_end:
            if self.auto_close_after_freeze and not self._viewer_closed:
                self.viewer.close()
                self._viewer_closed = True
            return
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt
        self._log_state()

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        if self.collision_pipeline.hydroelastic_sdf is not None:
            self.viewer.log_hydro_contact_surface(
                self.collision_pipeline.hydroelastic_sdf.get_contact_surface(),
            )
        self.viewer.log_scalar("sim_time [s]", self.sim_time)
        if self.z_log:
            self.viewer.log_scalar("z_dropper [m]", self.z_log[-1])
            self.viewer.log_scalar("vz_dropper [m/s]", self.vz_log[-1])
            self.viewer.log_scalar("depth [m]", self.depth_log[-1])
            self.viewer.log_scalar("face_contact_count", self.face_contact_count_log[-1])
            self.viewer.log_scalar("rigid_contact_count", self.rigid_contact_count_log[-1])
        self.viewer.end_frame()

    def test_post_step(self):
        """Track per-step maxima for ``test_final`` assertions."""
        reduced = int(self.contacts.rigid_contact_count.numpy()[0])
        self.max_reduced_contacts = max(self.max_reduced_contacts, reduced)
        surf = self.collision_pipeline.hydroelastic_sdf.get_contact_surface()
        raw = int(surf.face_contact_count.numpy()[0])
        self.max_face_contacts = max(self.max_face_contacts, raw)

    def test_final(self):
        """Check that both contact modes generate stable, finite sphere-pair traces."""
        # Verify these checks when demo behavior changes.
        self._write_csv()

        zs = np.asarray(self.z_log)
        vzs = np.asarray(self.vz_log)
        assert np.all(np.isfinite(zs)), "z trace contains NaN/Inf"
        assert np.all(np.isfinite(vzs)), "vz trace contains NaN/Inf"

        min_z = float(zs.min())
        fall_threshold = self.initial_dropper_z - 0.001
        assert min_z < fall_threshold, (
            f"dropper never entered contact regime: min z={min_z:.4f} m, expected < {fall_threshold:.4f} m"
        )

        final_z = float(zs[-1])
        final_depth = float(self.depth_log[-1])
        final_face = int(self.face_contact_count_log[-1])
        final_rigid = int(self.rigid_contact_count_log[-1])
        assert final_z > 2.4 * SPHERE_RADIUS, (
            f"dropper fell through static sphere: final z={final_z:.4f} m, expected > {2.4 * SPHERE_RADIUS:.4f} m"
        )
        assert 0.0 < final_depth < SDF_NARROW_BAND_RANGE[1], (
            f"dropper ended outside the SDF narrow band: final depth={final_depth:.4f} m, "
            f"expected in (0, {SDF_NARROW_BAND_RANGE[1]:.4f}) m"
        )
        assert final_face > 0 and final_rigid > 0, (
            f"dropper ended without active contacts: final face={final_face}, final rigid={final_rigid}"
        )

        assert self.max_face_contacts > 0, (
            f"no hydroelastic face contacts were generated during the run (max_face_contacts={self.max_face_contacts})"
        )

        assert self.max_reduced_contacts > 0, (
            f"no reduced contacts reached the solver during the run (max_reduced_contacts={self.max_reduced_contacts})"
        )

        face = np.asarray(self.face_contact_count_log)
        rigid = np.asarray(self.rigid_contact_count_log)
        engaged = face > 0
        assert engaged.any(), "no frame ever generated a hydroelastic face contact"
        if self.reduce_contacts:
            ratios = rigid[engaged] / face[engaged]
            assert ratios.min() < 0.5, (
                f"reduce_contacts=True but minimum rigid/face ratio was {float(ratios.min()):.3f} "
                f"(>=0.5); reduction does not appear to be active"
            )
        else:
            equal = rigid[engaged] == face[engaged]
            assert equal.sum() >= max(1, int(engaged.sum() * 0.5)), (
                f"reduce_contacts=False but rigid == face held in only "
                f"{int(equal.sum())}/{int(engaged.sum())} engaged frames; "
                f"reduction may still be active"
            )

    def _write_csv(self) -> None:
        suffix = "reduce_on" if self.reduce_contacts else "reduce_off"
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        path = os.path.join(OUTPUT_DIR, f"hydro_sphere_pair_{suffix}.csv")
        with open(path, "w") as f:
            f.write("time_s,z_m,vz_m_per_s,depth_m,face_contact_count,rigid_contact_count\n")
            for t, z, vz, d, face, rigid in zip(
                self.time_log,
                self.z_log,
                self.vz_log,
                self.depth_log,
                self.face_contact_count_log,
                self.rigid_contact_count_log,
                strict=False,
            ):
                f.write(f"{t:.6f},{z:.6f},{vz:.6f},{d:.6f},{face},{rigid}\n")
        print(f"wrote {path} ({len(self.time_log)} samples)")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=160)
        import argparse  # noqa: PLC0415

        parser.add_argument(
            "--reduce-contacts",
            action=argparse.BooleanOptionalAction,
            default=True,
            help=(
                "Enable HydroelasticSDF.Config.reduce_contacts (default True, matching "
                "Newton's shipped examples). Use --no-reduce-contacts to keep all "
                "marching-cubes face contacts."
            ),
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
