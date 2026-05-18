# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Spinning Cylinder
#
# Measures how reduce_contacts affects yaw-spin decay for a flat cylinder
# on a hydroelastic plate. Torsional and rolling friction are zero, so yaw
# decay comes from sliding-friction moment arms.
#
# Uniform-pressure stop time:
#     t_stop = (3/4) * omega_0 * R / (mu * g)
#
# Run modes:
#     python -m newton.examples spinning_cylinder                       # reduce on (default)
#     python -m newton.examples spinning_cylinder --no-reduce-contacts  # reduce off
#
# Command: python -m newton.examples spinning_cylinder
#
###########################################################################

import math
import os

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.geometry import HydroelasticSDF

CYLINDER_RADIUS = 0.05
"""Disc radius R [m]."""
CYLINDER_HALF_HEIGHT = CYLINDER_RADIUS / 8.0
"""Half-height of the cylinder along its local Z axis [m] (total height = R/4)."""
CYLINDER_DENSITY = 191.0
"""Cylinder density [kg/m³] (matches the sphere-pair example)."""

PLATE_HALF_EXTENT = 0.20
"""Plate half-extent in X and Y [m]."""
PLATE_HALF_THICKNESS = 0.01
"""Plate half-thickness in Z [m]."""

INITIAL_OMEGA_Z = 60.0
"""Initial yaw rate ω₀ [rad/s] about the cylinder's vertical axis."""

MU_SLIDING = 0.2
"""Sliding friction coefficient (Coulomb). Picked low together with a
high ω₀ to stretch the decay regime: t_stop ∝ ω₀ / μ, so the run holds
~5x more decay than the original ω₀=30, μ=0.5 calibration."""
KH = 1.0e9
"""Hydroelastic contact stiffness [Pa·-style coefficient — see sphere_pair example]."""

SDF_MAX_RESOLUTION = 64
"""Maximum SDF grid resolution along the longest axis."""
SDF_NARROW_BAND_RANGE = (-0.005, 0.005)
"""SDF narrow-band range [m]."""

RIGID_CONTACT_MAX = 16384
"""Upper bound on rigid contacts allocated by the collision pipeline."""
MUJOCO_NCONMAX = 16384
"""MuJoCo contact buffer capacity (must be ≥ RIGID_CONTACT_MAX)."""
MUJOCO_NJMAX = 32768
"""MuJoCo constraint Jacobian row capacity (≈ 3·active contacts under elliptic cone)."""

GRAVITY = 9.81
"""Acceleration of gravity used for analytic predictions [m/s²]."""

OUTPUT_DIR = os.path.join("output", "spinning_cylinder")
"""Where CSV traces land, relative to the process cwd."""


def _analytic_t_stop(omega0: float, radius: float, mu: float) -> float:
    """Time for the cylinder to spin down to rest under uniform pressure friction."""
    return 0.75 * omega0 * radius / (mu * GRAVITY)


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.reduce_contacts = bool(args.reduce_contacts)
        # Close batch runs after the experiment window ends.
        self.auto_close_after_freeze = bool(getattr(args, "test", False)) or bool(getattr(args, "headless", False))
        self._viewer_closed = False

        self.fps = 240
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 8
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        self.t_stop = _analytic_t_stop(INITIAL_OMEGA_Z, CYLINDER_RADIUS, MU_SLIDING)
        self.omega0 = INITIAL_OMEGA_Z

        shape_cfg = newton.ModelBuilder.ShapeConfig(
            margin=0.0,
            gap=0.0005,
            mu=MU_SLIDING,
            mu_torsional=0.0,
            mu_rolling=0.0,
            ke=1.0e7,
            kd=1.0e5,
            density=CYLINDER_DENSITY,
            is_hydroelastic=True,
            kh=KH,
            sdf_max_resolution=SDF_MAX_RESOLUTION,
            sdf_narrow_band_range=SDF_NARROW_BAND_RANGE,
        )

        builder = newton.ModelBuilder()
        builder.default_shape_cfg.gap = 0.001

        builder.add_shape_box(
            body=-1,
            xform=wp.transform(
                wp.vec3(0.0, 0.0, -PLATE_HALF_THICKNESS),
                wp.quat_identity(),
            ),
            hx=PLATE_HALF_EXTENT,
            hy=PLATE_HALF_EXTENT,
            hz=PLATE_HALF_THICKNESS,
            cfg=shape_cfg,
            label="plate",
        )

        # Start inside the SDF narrow band to avoid a launch-from-rest bounce.
        cylinder_z = CYLINDER_HALF_HEIGHT - 0.0002
        self.cylinder_body = builder.add_body(
            xform=wp.transform(
                wp.vec3(0.0, 0.0, cylinder_z),
                wp.quat_identity(),
            ),
            label="cylinder",
        )
        builder.add_shape_cylinder(
            body=self.cylinder_body,
            radius=CYLINDER_RADIUS,
            half_height=CYLINDER_HALF_HEIGHT,
            cfg=shape_cfg,
            label="cylinder_shape",
        )

        # Visual-only spin markers.
        marker_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            has_shape_collision=False,
            is_hydroelastic=False,
        )
        bar_hx = CYLINDER_RADIUS * 0.95
        bar_hy = CYLINDER_RADIUS * 0.04
        bar_hz = 0.0005
        bar_z = CYLINDER_HALF_HEIGHT + bar_hz
        builder.add_shape_box(
            body=self.cylinder_body,
            xform=wp.transform(wp.vec3(0.0, 0.0, bar_z), wp.quat_identity()),
            hx=bar_hx,
            hy=bar_hy,
            hz=bar_hz,
            cfg=marker_cfg,
            color=(1.0, 0.2, 0.2),
            label="marker_x",
        )
        builder.add_shape_box(
            body=self.cylinder_body,
            xform=wp.transform(wp.vec3(0.0, 0.0, bar_z), wp.quat_identity()),
            hx=bar_hy,
            hy=bar_hx,
            hz=bar_hz,
            cfg=marker_cfg,
            color=(0.2, 0.4, 1.0),
            label="marker_y",
        )

        # Free-joint qd layout: (lin_x, lin_y, lin_z, ang_x, ang_y, ang_z).
        qd_start = builder.joint_qd_start[-1]
        builder.joint_qd[qd_start + 5] = INITIAL_OMEGA_Z

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
            pos=wp.vec3(0.35, -0.35, 0.18),
            pitch=-20.0,
            yaw=135.0,
        )

        self.omega_log: list[float] = []
        self.time_log: list[float] = []
        self.face_contact_count_log: list[int] = []
        self.rigid_contact_count_log: list[int] = []
        self.max_reduced_contacts = 0
        self.max_face_contacts = 0

        self._log_state()

        self.capture()

    def _log_state(self) -> None:
        body_qd = self.state_0.body_qd.numpy()
        omega_z = float(body_qd[self.cylinder_body, 5])
        self.omega_log.append(omega_z)
        self.time_log.append(self.sim_time)

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
        if self.sim_time >= 2.0 * self.t_stop:
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
        self.viewer.log_scalar("t / t_stop", self.sim_time / self.t_stop)
        if self.omega_log:
            self.viewer.log_scalar("omega_z [rad/s]", self.omega_log[-1])
            self.viewer.log_scalar("omega_z / omega_0", self.omega_log[-1] / self.omega0)
        self.viewer.end_frame()

    def test_post_step(self):
        reduced = int(self.contacts.rigid_contact_count.numpy()[0])
        self.max_reduced_contacts = max(self.max_reduced_contacts, reduced)
        if self.collision_pipeline.hydroelastic_sdf is not None:
            surf = self.collision_pipeline.hydroelastic_sdf.get_contact_surface()
            raw = int(surf.face_contact_count.numpy()[0])
            self.max_face_contacts = max(self.max_face_contacts, raw)

    def test_final(self):
        # Verify these checks when demo behavior changes.
        assert self.max_face_contacts > 0, (
            f"no hydroelastic face contacts were generated (max_face_contacts={self.max_face_contacts})"
        )
        assert self.max_reduced_contacts > 0, (
            f"no reduced contacts reached the solver (max_reduced_contacts={self.max_reduced_contacts})"
        )

        # Allow vertical drift from high kh without accepting outright flight.
        body_q = self.state_0.body_q.numpy()[self.cylinder_body]
        final_z = float(body_q[2])
        z_lo = CYLINDER_HALF_HEIGHT - 0.005
        z_hi = CYLINDER_HALF_HEIGHT + 0.05
        assert z_lo < final_z < z_hi, (
            f"cylinder drifted vertically: z={final_z:.4f} m, expected in ({z_lo:.4f}, {z_hi:.4f})"
        )

        quat = body_q[3:7]
        qx, qy, qz, qw = (float(quat[i]) for i in range(4))
        local_z_world = (
            2.0 * (qx * qz + qw * qy),
            2.0 * (qy * qz - qw * qx),
            1.0 - 2.0 * (qx * qx + qy * qy),
        )
        tilt = math.acos(max(-1.0, min(1.0, local_z_world[2])))
        # Loose enough for reduce-off contact wobble, strict enough for tip-over.
        assert tilt < math.radians(15.0), f"cylinder tipped: {math.degrees(tilt):.2f}° (limit 15°)"

        times = np.asarray(self.time_log)
        omegas = np.asarray(self.omega_log)
        assert np.all(np.isfinite(omegas)), "omega trace contains NaN/Inf"
        omega_abs = np.abs(omegas)

        # Bound rotational-energy growth while allowing solver noise.
        assert omega_abs.max() <= self.omega0 + 0.5, (
            f"|omega| exceeded omega_0: max(|omega|)={omega_abs.max():.3f}, omega_0={self.omega0:.3f}"
        )

        idx = int(np.argmin(np.abs(times - self.t_stop)))
        ratio = float(omega_abs[idx]) / self.omega0

        assert ratio < 0.90, (
            f"cylinder did not slow at all by t_stop={self.t_stop:.3f}s "
            f"(omega/omega0={ratio:.3f}). Hydroelastic friction is not engaging."
        )

        if not self.reduce_contacts:
            assert ratio < 0.20, (
                f"reduce_contacts=False: cylinder did not approach rest by t_stop. "
                f"omega/omega0 at t_stop={self.t_stop:.3f}s is {ratio:.3f}, expected < 0.20"
            )

        self._write_csv()

    def _write_csv(self) -> None:
        suffix = "reduce_on" if self.reduce_contacts else "reduce_off"
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        path = os.path.join(OUTPUT_DIR, f"spinning_cylinder_omega_{suffix}.csv")
        with open(path, "w") as f:
            f.write("time_s,omega_z_rad_per_s,omega_over_omega0,face_contact_count,rigid_contact_count\n")
            for t, w, face, rigid in zip(
                self.time_log,
                self.omega_log,
                self.face_contact_count_log,
                self.rigid_contact_count_log,
                strict=False,
            ):
                f.write(f"{t:.6f},{w:.6f},{w / self.omega0:.6f},{face},{rigid}\n")
        print(f"wrote {path} ({len(self.time_log)} samples)")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=560)
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
