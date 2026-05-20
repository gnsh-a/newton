# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Tipping Cube
#
# Compares reduce_contacts modes for a cube pushed at its top face on a
# hydroelastic plate. Two failure modes compete:
#
#     Slide:  F > μ · m · g
#     Tip:    F · L > m · g · (L/2)  →  F > m · g / 2
#
# With μ = 0.7, analytic static thresholds predict tipping before sliding.
# The demo records the first observed event and writes a CSV trace.
#
# Run modes:
#     python -m newton.examples tipping_cube              # reduce on (default)
#     python -m newton.examples tipping_cube_no_reduce    # reduce off
#
# Command: python -m newton.examples tipping_cube
#
###########################################################################

import math
import os

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.geometry import HydroelasticSDF

CUBE_HALF_EXTENT = 0.05
"""Cube half-extent [m]. Side length L = 2·CUBE_HALF_EXTENT = 0.10 m."""
CUBE_DENSITY = 800.0
"""Cube density [kg/m³]. With L=0.10 m gives m = rho·L³ = 0.8 kg, mg ≈ 7.85 N."""

PLATE_HALF_EXTENT = 0.25
"""Plate half-extent in X and Y [m]."""
PLATE_HALF_THICKNESS = 0.01
"""Plate half-thickness in Z [m]."""

MU_SLIDING = 0.7
"""Sliding friction coefficient. Picked > 1/2 so tipping wins analytically
(F_tip = m·g/2 ≈ 3.93 N before F_slide = μ·m·g ≈ 5.50 N)."""
KH = 1.0e9
"""Hydroelastic contact stiffness coefficient — same value as Demo 1."""

SDF_MAX_RESOLUTION = 64
"""Maximum SDF grid resolution along the longest axis."""
SDF_NARROW_BAND_RANGE = (-0.005, 0.005)
"""SDF narrow-band range [m]."""

RAMP_RATE_N_PER_S = 10.0
"""Linear ramp rate for the applied horizontal force [N/s]. Reaches 2·mg ≈ 15 N
in 1.5 s — comfortably past both analytic thresholds."""
EVENT_TILT_DEG = 10.0
"""Tilt threshold to declare a tip event [deg]."""
EVENT_SLIDE_MM = 5.0
"""Translation threshold to declare a slide event [mm] (= 5% of L)."""
EVENT_SLIDE_TILT_CEILING_DEG = 5.0
"""Tilt must stay below this to call the event a pure slide [deg]."""
SAFETY_CAP_SECONDS = 2.0
"""Hard cap on sim_time [s]: if no event fires by here, ramp is broken."""

RIGID_CONTACT_MAX = 16384
"""Upper bound on rigid contacts allocated by the collision pipeline."""
MUJOCO_NCONMAX = 16384
"""MuJoCo contact buffer capacity (must be ≥ RIGID_CONTACT_MAX)."""
MUJOCO_NJMAX = 32768
"""MuJoCo constraint Jacobian row capacity (≈ 3·active contacts under elliptic cone)."""

GRAVITY = 9.81
"""Acceleration of gravity used for analytic predictions [m/s²]."""

OUTPUT_DIR = os.path.join("output", "tipping_cube")
"""Where CSV traces land, relative to the process cwd."""


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.reduce_contacts = bool(args.reduce_contacts)
        self.pre_prune_contacts = self.reduce_contacts
        # Close batch runs after the experiment window ends.
        self.auto_close_after_freeze = bool(getattr(args, "test", False)) or bool(getattr(args, "headless", False))
        self._viewer_closed = False

        self.fps = 240
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 8
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        L = 2.0 * CUBE_HALF_EXTENT
        mass = CUBE_DENSITY * L * L * L
        self.cube_mass = mass
        self.weight_N = mass * GRAVITY
        self.F_tip_analytic = 0.5 * self.weight_N
        self.F_slide_analytic = MU_SLIDING * self.weight_N
        self.ramp_rate = RAMP_RATE_N_PER_S

        shape_cfg = newton.ModelBuilder.ShapeConfig(
            margin=0.0,
            gap=0.0005,
            mu=MU_SLIDING,
            mu_torsional=0.0,
            mu_rolling=0.0,
            ke=1.0e7,
            kd=1.0e5,
            density=CUBE_DENSITY,
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
            color=(0.58, 0.72, 0.78),
            label="plate",
        )

        # Start inside the SDF narrow band to avoid a launch-from-rest bounce.
        cube_z = CUBE_HALF_EXTENT - 0.0002
        self.cube_body = builder.add_body(
            xform=wp.transform(
                wp.vec3(0.0, 0.0, cube_z),
                wp.quat_identity(),
            ),
            label="cube",
        )
        builder.add_shape_box(
            body=self.cube_body,
            hx=CUBE_HALF_EXTENT,
            hy=CUBE_HALF_EXTENT,
            hz=CUBE_HALF_EXTENT,
            cfg=shape_cfg,
            color=(0.88, 0.52, 0.22),
            label="cube_shape",
        )

        # Free-joint qd layout: (lin_x, lin_y, lin_z, ang_x, ang_y, ang_z).
        self.cube_qd_start = int(builder.joint_qd_start[-1])

        self.model = builder.finalize()

        # Increase buffers so reduce-off can keep dense face contacts.
        hydro_cfg = HydroelasticSDF.Config(
            output_contact_surface=True,
            reduce_contacts=self.reduce_contacts,
            pre_prune_contacts=self.pre_prune_contacts,
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

        # Host scratch for the captured graph's control.joint_f buffer.
        # FREE joint_f layout: (fx, fy, fz, tau_x, tau_y, tau_z).
        joint_f_len = int(self.control.joint_f.shape[0])
        self._wrench_host = np.zeros(joint_f_len, dtype=np.float32)

        self.contacts = self.collision_pipeline.contacts()
        self.collision_pipeline.collide(self.state_0, self.contacts)

        self.viewer.set_model(self.model)
        self.viewer.show_contacts = True
        self.viewer.show_hydro_contact_surface = True
        self.viewer.show_collision = False
        self.viewer.show_triangles = False
        self.viewer.show_visual = True
        if hasattr(self.viewer, "renderer"):
            self.viewer.renderer.draw_wireframe = True
        self.viewer.set_camera(
            pos=wp.vec3(0.25, -0.45, 0.18),
            pitch=-15.0,
            yaw=120.0,
        )

        self.time_log: list[float] = []
        self.force_log: list[float] = []
        self.tilt_log: list[float] = []
        self.x_log_mm: list[float] = []
        self.face_contact_count_log: list[int] = []
        self.rigid_contact_count_log: list[int] = []
        self.max_reduced_contacts = 0
        self.max_face_contacts = 0

        body_q0 = self.state_0.body_q.numpy()[self.cube_body]
        self.cube_x0 = float(body_q0[0])

        self.event_type: str | None = None
        self.event_force_N: float | None = None
        self.event_sim_time: float | None = None

        self._log_state()

        self.capture()

    def _mode_suffix(self) -> str:
        if not self.reduce_contacts:
            return "reduce_off"
        return "reduce_on"

    def _apply_wrench(self) -> None:
        # Top-face force in +X; equivalent COM torque is +F * L / 2 about Y.
        F = self.ramp_rate * self.sim_time
        self._wrench_host[:] = 0.0
        self._wrench_host[self.cube_qd_start + 0] = F
        self._wrench_host[self.cube_qd_start + 4] = F * CUBE_HALF_EXTENT
        self.control.joint_f.assign(self._wrench_host)

    def _compute_tilt_and_x(self) -> tuple[float, float]:
        body_q = self.state_0.body_q.numpy()[self.cube_body]
        qx, qy, _qz, _qw = (float(body_q[3 + i]) for i in range(4))
        local_z_z = 1.0 - 2.0 * (qx * qx + qy * qy)
        local_z_z = max(-1.0, min(1.0, local_z_z))
        tilt_deg = math.degrees(math.acos(local_z_z))
        x_disp_mm = (float(body_q[0]) - self.cube_x0) * 1000.0
        return tilt_deg, x_disp_mm

    def _log_state(self) -> None:
        tilt_deg, x_disp_mm = self._compute_tilt_and_x()
        F = self.ramp_rate * self.sim_time
        self.time_log.append(self.sim_time)
        self.force_log.append(F)
        self.tilt_log.append(tilt_deg)
        self.x_log_mm.append(x_disp_mm)

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
        if self.event_type is not None or self.sim_time > SAFETY_CAP_SECONDS:
            if self.auto_close_after_freeze and not self._viewer_closed:
                self.viewer.close()
                self._viewer_closed = True
            return

        # Captured graph reads the control.joint_f values assigned here.
        self._apply_wrench()

        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt
        self._log_state()

        tilt_deg = self.tilt_log[-1]
        x_disp_mm = self.x_log_mm[-1]
        F_now = self.force_log[-1]

        tipped = tilt_deg > EVENT_TILT_DEG
        slid = abs(x_disp_mm) > EVENT_SLIDE_MM and tilt_deg < EVENT_SLIDE_TILT_CEILING_DEG
        if (tipped or slid) and self.event_type is None:
            self.event_type = "tip" if tipped else "slide"
            self.event_force_N = F_now
            self.event_sim_time = self.sim_time
            mode = self._mode_suffix()
            print(
                f"[{mode}] event={self.event_type} "
                f"F={F_now:.3f} N (analytic F_tip={self.F_tip_analytic:.3f}, "
                f"F_slide={self.F_slide_analytic:.3f}) "
                f"t={self.sim_time:.3f} s tilt={tilt_deg:.2f}° x={x_disp_mm:.2f} mm"
            )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        if self.collision_pipeline.hydroelastic_sdf is not None:
            self.viewer.log_hydro_contact_surface(
                self.collision_pipeline.hydroelastic_sdf.get_contact_surface(),
            )
        self.viewer.log_scalar("sim_time [s]", self.sim_time)
        if self.force_log:
            self.viewer.log_scalar("face_contact_count", self.face_contact_count_log[-1])
            self.viewer.log_scalar("rigid_contact_count", self.rigid_contact_count_log[-1])
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

        tilts = np.asarray(self.tilt_log)
        xs = np.asarray(self.x_log_mm)
        assert np.all(np.isfinite(tilts)), "tilt trace contains NaN/Inf"
        assert np.all(np.isfinite(xs)), "x_disp trace contains NaN/Inf"

        # Outcome and threshold depend on reduction mode.
        assert self.event_type is not None, (
            f"cube neither tipped nor slid by sim_time={self.sim_time:.3f} s "
            f"(max tilt={tilts.max():.2f}°, max |x|={np.abs(xs).max():.2f} mm). "
            f"Force ramp may be too low or contacts not engaging."
        )

        self._write_csv()

    def _write_csv(self) -> None:
        suffix = self._mode_suffix()
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        path = os.path.join(OUTPUT_DIR, f"tipping_cube_{suffix}.csv")
        with open(path, "w") as f:
            f.write("time_s,force_N,tilt_deg,x_mm,face_contact_count,rigid_contact_count\n")
            for t, F, tilt, x, face, rigid in zip(
                self.time_log,
                self.force_log,
                self.tilt_log,
                self.x_log_mm,
                self.face_contact_count_log,
                self.rigid_contact_count_log,
                strict=False,
            ):
                f.write(f"{t:.6f},{F:.6f},{tilt:.6f},{x:.6f},{face},{rigid}\n")
        print(f"wrote {path} ({len(self.time_log)} samples)")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=360)
        import argparse  # noqa: PLC0415

        parser.add_argument(
            "--reduce-contacts",
            action=argparse.BooleanOptionalAction,
            default=True,
            help=(
                "Enable HydroelasticSDF.Config.reduce_contacts (default True, matching "
                "Newton's shipped examples). Use --no-reduce-contacts to keep all "
                "marching-cubes face contacts and disable local pre-prune."
            ),
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
