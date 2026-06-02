# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Flat Sliding Block
#
# Viewer demo for H2: a hydroelastic cube slides on a flat hydroelastic plate.
# This is visual only; the headless experiment owns the CSV outputs.
#
# Run modes:
#     python -m newton.examples flat_sliding_block              # reduce on
#     python -m newton.examples flat_sliding_block_no_reduce    # reduce off
#
# Command: python -m newton.examples flat_sliding_block
#
###########################################################################

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.contacts import example_cube_on_plate as cube_on_plate
from newton.geometry import HydroelasticSDF

DEFAULT_INITIAL_SPEED = 0.4
DEFAULT_SIMULATION_TIME = 0.25
FRAME_FPS = 120
SIM_SUBSTEPS = 4
STOP_SPEED_M_PER_S = 0.005

RIGID_CONTACT_MAX = 131072
MUJOCO_NCONMAX = 131072
MUJOCO_NJMAX = 262144
BUFFER_MULT_ISO = 8
BUFFER_MULT_CONTACT = 16
BUFFER_FRACTION = 1.0


@dataclass(frozen=True)
class SceneConfig:
    """Scene constants loaded from the shared cube-on-plate config."""

    cube_half_extent: float
    cube_density: float
    plate_half_extent: float
    plate_half_thickness: float
    mu_sliding: float
    sdf_max_resolution: int
    sdf_narrow_band_range: tuple[float, float]
    kh: float


def _load_scene_config(config_path: str | Path) -> SceneConfig:
    constants, newton_constants = cube_on_plate._load_config(config_path)
    return SceneConfig(
        cube_half_extent=float(constants["CUBE_HALF_EXTENT"]),
        cube_density=float(constants["CUBE_DENSITY"]),
        plate_half_extent=float(constants["PLATE_HALF_EXTENT"]),
        plate_half_thickness=float(constants["PLATE_HALF_THICKNESS"]),
        mu_sliding=float(constants["MU_SLIDING"]),
        sdf_max_resolution=int(constants["SDF_MAX_RESOLUTION"]),
        sdf_narrow_band_range=tuple(float(x) for x in constants["SDF_NARROW_BAND_RANGE"]),
        kh=float(newton_constants["KH"]),
    )


def _cube_tilt_deg(body_q: np.ndarray) -> float:
    qx = float(body_q[3])
    qy = float(body_q[4])
    local_z_world_z = 1.0 - 2.0 * (qx * qx + qy * qy)
    return math.degrees(math.acos(max(-1.0, min(1.0, local_z_world_z))))


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.scene = _load_scene_config(args.config)
        self.initial_speed = float(args.initial_speed)
        if self.initial_speed <= 0.0:
            raise ValueError(f"initial_speed must be positive; got {self.initial_speed}")
        self.run_seconds = float(args.simulation_time)
        if self.run_seconds <= 0.0:
            raise ValueError(f"simulation_time must be positive; got {self.run_seconds}")

        self.reduce_contacts = bool(args.reduce_contacts)
        self.pre_prune_contacts = False
        self.auto_close_after_freeze = bool(getattr(args, "test", False)) or bool(getattr(args, "headless", False))
        self._viewer_closed = False

        self.frame_dt = 1.0 / FRAME_FPS
        self.sim_substeps = SIM_SUBSTEPS
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        if hasattr(self.viewer, "num_frames"):
            min_frames = math.ceil(self.run_seconds * FRAME_FPS) + 8
            self.viewer.num_frames = max(int(self.viewer.num_frames), min_frames)

        self.expected_stop_time = self.initial_speed / (self.scene.mu_sliding * cube_on_plate.GRAVITY)
        self.expected_stop_travel = (
            self.initial_speed * self.initial_speed / (2.0 * self.scene.mu_sliding * cube_on_plate.GRAVITY)
        )

        shape_cfg = newton.ModelBuilder.ShapeConfig(
            margin=0.0,
            gap=0.0005,
            mu=self.scene.mu_sliding,
            mu_torsional=0.0,
            mu_rolling=0.0,
            ke=1.0e7,
            kd=1.0e5,
            density=self.scene.cube_density,
            is_hydroelastic=True,
            kh=self.scene.kh,
            sdf_max_resolution=self.scene.sdf_max_resolution,
            sdf_narrow_band_range=self.scene.sdf_narrow_band_range,
        )

        builder = newton.ModelBuilder()
        builder.default_shape_cfg.gap = 0.001

        builder.add_shape_box(
            body=-1,
            xform=wp.transform(
                wp.vec3(0.0, 0.0, -self.scene.plate_half_thickness),
                wp.quat_identity(),
            ),
            hx=self.scene.plate_half_extent,
            hy=self.scene.plate_half_extent,
            hz=self.scene.plate_half_thickness,
            cfg=shape_cfg,
            color=(0.58, 0.72, 0.78),
            label="plate",
        )

        marker_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            has_shape_collision=False,
            is_hydroelastic=False,
        )
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(
                wp.vec3(self.expected_stop_travel, 0.0, 0.0015),
                wp.quat_identity(),
            ),
            hx=0.0015,
            hy=min(0.12, 0.5 * self.scene.plate_half_extent),
            hz=0.0015,
            cfg=marker_cfg,
            color=(0.05, 0.45, 0.18),
            label="coulomb_stop_marker",
        )

        self.cube_body = builder.add_body(
            xform=wp.transform(
                wp.vec3(0.0, 0.0, self.scene.cube_half_extent),
                wp.quat_identity(),
            ),
            label="cube",
        )
        cube_color = (0.95, 0.49, 0.13) if self.reduce_contacts else (0.20, 0.45, 0.95)
        builder.add_shape_box(
            body=self.cube_body,
            hx=self.scene.cube_half_extent,
            hy=self.scene.cube_half_extent,
            hz=self.scene.cube_half_extent,
            cfg=shape_cfg,
            color=cube_color,
            label="cube_shape",
        )
        builder.add_shape_box(
            body=self.cube_body,
            xform=wp.transform(
                wp.vec3(0.0, 0.0, self.scene.cube_half_extent + 0.0015),
                wp.quat_identity(),
            ),
            hx=0.8 * self.scene.cube_half_extent,
            hy=0.08 * self.scene.cube_half_extent,
            hz=0.0015,
            cfg=marker_cfg,
            color=(0.02, 0.02, 0.02),
            label="cube_top_direction_marker",
        )

        qd_start = builder.joint_qd_start[-1]
        builder.joint_qd[qd_start + 0] = self.initial_speed

        self.model = builder.finalize()

        hydro_cfg = HydroelasticSDF.Config(
            output_contact_surface=True,
            reduce_contacts=self.reduce_contacts,
            pre_prune_contacts=self.pre_prune_contacts,
            buffer_mult_iso=BUFFER_MULT_ISO,
            buffer_mult_contact=BUFFER_MULT_CONTACT,
            buffer_fraction=BUFFER_FRACTION,
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
        self.viewer.show_collision = False
        self.viewer.show_triangles = False
        self.viewer.show_visual = True
        if hasattr(self.viewer, "renderer"):
            self.viewer.renderer.draw_wireframe = True
        self.viewer.set_camera(
            pos=wp.vec3(0.22, -0.32, 0.16),
            pitch=-18.0,
            yaw=135.0,
        )

        self.time_log: list[float] = []
        self.x_log: list[float] = []
        self.y_log: list[float] = []
        self.speed_log: list[float] = []
        self.tilt_log: list[float] = []
        self.rigid_contact_count_log: list[int] = []
        self.face_contact_count_log: list[int] = []
        self.max_rigid_contacts = 0
        self.max_face_contacts = 0

        self._log_state()
        self.capture()

    def _mode_label(self) -> str:
        return "reduce on" if self.reduce_contacts else "reduce off"

    def _log_state(self) -> None:
        body_q = self.state_0.body_q.numpy()[self.cube_body]
        body_qd = self.state_0.body_qd.numpy()[self.cube_body]
        speed = math.hypot(float(body_qd[0]), float(body_qd[1]))
        rigid = int(self.contacts.rigid_contact_count.numpy()[0])
        if self.collision_pipeline.hydroelastic_sdf is not None:
            surf = self.collision_pipeline.hydroelastic_sdf.get_contact_surface()
            face = int(surf.face_contact_count.numpy()[0])
        else:
            face = 0

        self.time_log.append(self.sim_time)
        self.x_log.append(float(body_q[0]))
        self.y_log.append(float(body_q[1]))
        self.speed_log.append(speed)
        self.tilt_log.append(_cube_tilt_deg(body_q))
        self.rigid_contact_count_log.append(rigid)
        self.face_contact_count_log.append(face)
        self.max_rigid_contacts = max(self.max_rigid_contacts, rigid)
        self.max_face_contacts = max(self.max_face_contacts, face)

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
        self.collision_pipeline.collide(self.state_0, self.contacts)

    def step(self):
        if self.sim_time >= self.run_seconds:
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
        self.viewer.log_scalar("initial_speed [m/s]", self.initial_speed)
        self.viewer.log_scalar("reduce_contacts", 1.0 if self.reduce_contacts else 0.0)
        self.viewer.log_scalar("expected_stop_time [s]", self.expected_stop_time)
        self.viewer.log_scalar("expected_stop_travel [mm]", 1000.0 * self.expected_stop_travel)
        if self.speed_log:
            self.viewer.log_scalar("cube_speed [m/s]", self.speed_log[-1])
            self.viewer.log_scalar("cube_x [mm]", 1000.0 * self.x_log[-1])
            self.viewer.log_scalar("cube_y [mm]", 1000.0 * self.y_log[-1])
            self.viewer.log_scalar("cube_tilt [deg]", self.tilt_log[-1])
            self.viewer.log_scalar("face_contact_count", self.face_contact_count_log[-1])
            self.viewer.log_scalar("rigid_contact_count", self.rigid_contact_count_log[-1])
        self.viewer.end_frame()

    def test_post_step(self):
        self.max_rigid_contacts = max(self.max_rigid_contacts, int(self.contacts.rigid_contact_count.numpy()[0]))
        if self.collision_pipeline.hydroelastic_sdf is not None:
            surf = self.collision_pipeline.hydroelastic_sdf.get_contact_surface()
            self.max_face_contacts = max(self.max_face_contacts, int(surf.face_contact_count.numpy()[0]))

    def test_final(self):
        assert self.max_face_contacts > 0, "no hydroelastic face contacts were generated"
        assert self.max_rigid_contacts > 0, "no contacts reached the solver"
        assert np.all(np.isfinite(np.asarray(self.x_log))), "x trace contains NaN/Inf"
        assert np.all(np.isfinite(np.asarray(self.y_log))), "y trace contains NaN/Inf"
        assert np.all(np.isfinite(np.asarray(self.speed_log))), "speed trace contains NaN/Inf"

        final_speed = self.speed_log[-1]
        final_tilt = self.tilt_log[-1]
        forward_mm = 1000.0 * (self.x_log[-1] - self.x_log[0])
        lateral_mm = 1000.0 * (self.y_log[-1] - self.y_log[0])
        assert final_speed < 0.03, f"cube did not slow down enough: speed={final_speed:.4f} m/s"
        assert final_tilt < 0.5, f"cube tilted unexpectedly: tilt={final_tilt:.4f} deg"
        assert abs(lateral_mm) < 1.0, f"cube drifted laterally: y={lateral_mm:.4f} mm"
        assert forward_mm > 0.0, f"cube did not move forward: x={forward_mm:.4f} mm"

        if self.reduce_contacts:
            assert self.max_rigid_contacts < self.max_face_contacts, (
                f"reduced mode did not reduce contacts: face={self.max_face_contacts}, rigid={self.max_rigid_contacts}"
            )
        else:
            assert self.max_rigid_contacts > 1000, (
                f"reduce-off mode did not preserve dense contacts: max rigid={self.max_rigid_contacts}"
            )

        print(
            f"[flat_sliding_block {self._mode_label()}] "
            f"final_x={forward_mm:.3f} mm final_speed={final_speed:.5f} m/s "
            f"max_contacts={self.max_rigid_contacts}"
        )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=48)
        parser.add_argument(
            "--config",
            type=str,
            default=str(cube_on_plate.DEFAULT_CONFIG_PATH),
            help="Path to the shared cube-on-plate YAML config.",
        )
        parser.add_argument(
            "--initial-speed",
            type=float,
            default=DEFAULT_INITIAL_SPEED,
            help="Initial horizontal cube speed [m/s].",
        )
        parser.add_argument(
            "--simulation-time",
            type=float,
            default=DEFAULT_SIMULATION_TIME,
            help="Simulation time [s].",
        )
        parser.add_argument(
            "--reduce-contacts",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Enable hydroelastic contact reduction. Pre-prune remains off in both modes.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
