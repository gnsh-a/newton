# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Sliding Single Rib Cube
#
# Slides a hydroelastic cube across one shallow diagonal rib on a plate.
#
# Run modes:
#     python -m newton.examples sliding_single_rib_cube              # reduce on
#     python -m newton.examples sliding_single_rib_cube_global_only  # reduce on, no pre-prune
#     python -m newton.examples sliding_single_rib_cube_no_reduce    # reduce off
#
# Command: python -m newton.examples sliding_single_rib_cube
#
###########################################################################

import math
import os

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.geometry import HydroelasticSDF

CUBE_HALF_EXTENT = 0.035
CUBE_DENSITY = 700.0

PLATE_HALF_EXTENT = 0.24
PLATE_THICKNESS = 0.025
PLATE_GRID_RESOLUTION = 48

RIB_HEIGHT = 0.004
RIB_HALF_WIDTH = 0.028
RIB_SLOPE = 0.45
RIB_OFFSET = -0.005

MU_SLIDING = 0.45
KH = 1.0e9

SDF_MAX_RESOLUTION = 64
SDF_NARROW_BAND_RANGE = (-0.008, 0.008)
SDF_MARGIN = 0.008

INITIAL_X = -0.105
INITIAL_Y = -0.035
INITIAL_VX = 0.55

RIGID_CONTACT_MAX = 65536
MUJOCO_NCONMAX = 65536
MUJOCO_NJMAX = 196608

OUTPUT_DIR = os.path.join("output", "sliding_single_rib_cube")


def _single_rib_height(xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
    signed_dist = (yy - RIB_SLOPE * xx - RIB_OFFSET) / math.sqrt(1.0 + RIB_SLOPE * RIB_SLOPE)
    height = RIB_HEIGHT * np.maximum(0.0, 1.0 - np.abs(signed_dist) / RIB_HALF_WIDTH)
    return height.astype(np.float32)


def _build_single_rib_plate_mesh() -> newton.Mesh:
    coords = np.linspace(-PLATE_HALF_EXTENT, PLATE_HALF_EXTENT, PLATE_GRID_RESOLUTION + 1, dtype=np.float32)
    xx, yy = np.meshgrid(coords, coords, indexing="ij")
    top_z = _single_rib_height(xx, yy)
    bottom_z = np.full_like(top_z, -PLATE_THICKNESS, dtype=np.float32)

    top_vertices = np.stack((xx, yy, top_z), axis=-1).reshape(-1, 3)
    bottom_vertices = np.stack((xx, yy, bottom_z), axis=-1).reshape(-1, 3)
    vertices = np.vstack((top_vertices, bottom_vertices)).astype(np.float32)

    n = PLATE_GRID_RESOLUTION + 1
    bottom_offset = n * n

    def top_id(i: int, j: int) -> int:
        return i * n + j

    def bottom_id(i: int, j: int) -> int:
        return bottom_offset + i * n + j

    indices: list[int] = []
    for i in range(PLATE_GRID_RESOLUTION):
        for j in range(PLATE_GRID_RESOLUTION):
            t00 = top_id(i, j)
            t10 = top_id(i + 1, j)
            t01 = top_id(i, j + 1)
            t11 = top_id(i + 1, j + 1)
            indices.extend((t00, t10, t11, t00, t11, t01))

            b00 = bottom_id(i, j)
            b10 = bottom_id(i + 1, j)
            b01 = bottom_id(i, j + 1)
            b11 = bottom_id(i + 1, j + 1)
            indices.extend((b00, b01, b11, b00, b11, b10))

    for j in range(PLATE_GRID_RESOLUTION):
        t0 = top_id(0, j)
        t1 = top_id(0, j + 1)
        b0 = bottom_id(0, j)
        b1 = bottom_id(0, j + 1)
        indices.extend((b0, t0, t1, b0, t1, b1))

        t0 = top_id(PLATE_GRID_RESOLUTION, j)
        t1 = top_id(PLATE_GRID_RESOLUTION, j + 1)
        b0 = bottom_id(PLATE_GRID_RESOLUTION, j)
        b1 = bottom_id(PLATE_GRID_RESOLUTION, j + 1)
        indices.extend((b0, b1, t1, b0, t1, t0))

    for i in range(PLATE_GRID_RESOLUTION):
        t0 = top_id(i, 0)
        t1 = top_id(i + 1, 0)
        b0 = bottom_id(i, 0)
        b1 = bottom_id(i + 1, 0)
        indices.extend((b0, b1, t1, b0, t1, t0))

        t0 = top_id(i, PLATE_GRID_RESOLUTION)
        t1 = top_id(i + 1, PLATE_GRID_RESOLUTION)
        b0 = bottom_id(i, PLATE_GRID_RESOLUTION)
        b1 = bottom_id(i, PLATE_GRID_RESOLUTION)
        indices.extend((b0, t0, t1, b0, t1, b1))

    mesh = newton.Mesh(
        vertices,
        np.asarray(indices, dtype=np.int32),
        compute_inertia=False,
        color=(0.46, 0.63, 0.69),
    )
    mesh.build_sdf(
        max_resolution=SDF_MAX_RESOLUTION,
        narrow_band_range=SDF_NARROW_BAND_RANGE,
        margin=SDF_MARGIN,
    )
    return mesh


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.reduce_contacts = bool(args.reduce_contacts)
        self.pre_prune_contacts = bool(args.pre_prune_contacts)
        self.auto_close_after_freeze = bool(getattr(args, "test", False)) or bool(getattr(args, "headless", False))
        self._viewer_closed = False

        self.fps = 240
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 8
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        plate_cfg = newton.ModelBuilder.ShapeConfig(
            margin=0.0,
            gap=0.0005,
            mu=MU_SLIDING,
            mu_torsional=0.0,
            mu_rolling=0.0,
            ke=1.0e7,
            kd=1.0e5,
            density=0.0,
            is_hydroelastic=True,
            kh=KH,
        )
        cube_cfg = newton.ModelBuilder.ShapeConfig(
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

        plate_mesh = _build_single_rib_plate_mesh()
        builder.add_shape_mesh(
            body=-1,
            mesh=plate_mesh,
            cfg=plate_cfg,
            label="single_rib_plate",
        )

        cube_z = CUBE_HALF_EXTENT + 0.001
        self.cube_body = builder.add_body(
            xform=wp.transform(
                wp.vec3(INITIAL_X, INITIAL_Y, cube_z),
                wp.quat_identity(),
            ),
            label="cube",
        )
        builder.add_shape_box(
            body=self.cube_body,
            hx=CUBE_HALF_EXTENT,
            hy=CUBE_HALF_EXTENT,
            hz=CUBE_HALF_EXTENT,
            cfg=cube_cfg,
            color=(0.88, 0.52, 0.22),
            label="cube_shape",
        )

        qd_start = builder.joint_qd_start[-1]
        builder.joint_qd[qd_start + 0] = INITIAL_VX

        self.model = builder.finalize()

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
            pos=wp.vec3(0.20, -0.45, 0.20),
            pitch=-22.0,
            yaw=145.0,
        )

        self.time_log: list[float] = []
        self.x_log: list[float] = []
        self.y_log: list[float] = []
        self.z_log: list[float] = []
        self.yaw_log: list[float] = []
        self.vx_log: list[float] = []
        self.vy_log: list[float] = []
        self.omega_z_log: list[float] = []
        self.face_contact_count_log: list[int] = []
        self.rigid_contact_count_log: list[int] = []
        self.normal_lateral_avg_log: list[float] = []
        self.normal_lateral_max_log: list[float] = []
        self.normal_lateral_count_gt_0_2_log: list[int] = []
        self.max_reduced_contacts = 0
        self.max_face_contacts = 0

        self._log_state()

        self.capture()

    def _mode_suffix(self) -> str:
        if not self.reduce_contacts:
            return "reduce_off"
        if not self.pre_prune_contacts:
            return "global_only"
        return "reduce_on"

    def _compute_pose_and_velocity(self) -> tuple[float, float, float, float, float, float, float]:
        body_q = self.state_0.body_q.numpy()[self.cube_body]
        body_qd = self.state_0.body_qd.numpy()[self.cube_body]
        qx, qy, qz, qw = (float(body_q[3 + i]) for i in range(4))
        yaw = math.degrees(math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))
        return (
            float(body_q[0]),
            float(body_q[1]),
            float(body_q[2]),
            yaw,
            float(body_qd[0]),
            float(body_qd[1]),
            float(body_qd[5]),
        )

    def _log_state(self) -> None:
        x, y, z, yaw, vx, vy, omega_z = self._compute_pose_and_velocity()
        self.time_log.append(self.sim_time)
        self.x_log.append(x)
        self.y_log.append(y)
        self.z_log.append(z)
        self.yaw_log.append(yaw)
        self.vx_log.append(vx)
        self.vy_log.append(vy)
        self.omega_z_log.append(omega_z)

        rigid = int(self.contacts.rigid_contact_count.numpy()[0])
        self.rigid_contact_count_log.append(rigid)
        if self.collision_pipeline.hydroelastic_sdf is not None:
            surf = self.collision_pipeline.hydroelastic_sdf.get_contact_surface()
            face = int(surf.face_contact_count.numpy()[0])
        else:
            face = 0
        self.face_contact_count_log.append(face)

        n = min(rigid, self.contacts.rigid_contact_max)
        if n > 0:
            normals = self.contacts.rigid_contact_normal.numpy()[:n]
            lateral = np.linalg.norm(normals[:, :2], axis=1)
            lateral_avg = float(lateral.mean())
            lateral_max = float(lateral.max())
            lateral_count = int((lateral > 0.2).sum())
        else:
            lateral_avg = 0.0
            lateral_max = 0.0
            lateral_count = 0
        self.normal_lateral_avg_log.append(lateral_avg)
        self.normal_lateral_max_log.append(lateral_max)
        self.normal_lateral_count_gt_0_2_log.append(lateral_count)

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
        if self.sim_time >= self.sim_duration:
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

    @property
    def sim_duration(self) -> float:
        return 0.6

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        if self.collision_pipeline.hydroelastic_sdf is not None:
            self.viewer.log_hydro_contact_surface(
                self.collision_pipeline.hydroelastic_sdf.get_contact_surface(),
            )
        self.viewer.log_scalar("sim_time [s]", self.sim_time)
        if self.x_log:
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
            f"no contacts reached the solver (max_reduced_contacts={self.max_reduced_contacts})"
        )

        xs = np.asarray(self.x_log)
        ys = np.asarray(self.y_log)
        zs = np.asarray(self.z_log)
        yaws = np.asarray(self.yaw_log)
        assert np.all(np.isfinite(xs)), "x trace contains NaN/Inf"
        assert np.all(np.isfinite(ys)), "y trace contains NaN/Inf"
        assert np.all(np.isfinite(zs)), "z trace contains NaN/Inf"
        assert np.all(np.isfinite(yaws)), "yaw trace contains NaN/Inf"

        forward_mm = (xs[-1] - xs[0]) * 1000.0
        lateral_mm = (ys[-1] - ys[0]) * 1000.0
        yaw_span = float(np.max(np.abs(yaws - yaws[0])))
        assert forward_mm > 35.0, f"cube did not slide forward enough: {forward_mm:.2f} mm"
        assert np.min(zs) > 0.015, f"cube fell through the single-rib plate: min z={np.min(zs):.4f} m"
        assert max(self.normal_lateral_count_gt_0_2_log) > 0, "no oblique contact normals were observed"
        assert abs(lateral_mm) > 0.1 or yaw_span > 0.1, (
            f"rib response too small: lateral={lateral_mm:.3f} mm, yaw_span={yaw_span:.3f} deg"
        )

        if self.reduce_contacts:
            assert self.max_reduced_contacts < self.max_face_contacts, (
                f"reduced mode did not reduce contacts: face={self.max_face_contacts}, rigid={self.max_reduced_contacts}"
            )
        else:
            assert self.max_reduced_contacts > 1000, (
                f"no-reduce mode did not preserve dense contacts: max rigid={self.max_reduced_contacts}"
            )

        self._write_csv()

    def _write_csv(self) -> None:
        suffix = self._mode_suffix()
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        path = os.path.join(OUTPUT_DIR, f"sliding_single_rib_cube_{suffix}.csv")
        with open(path, "w") as f:
            f.write(
                "time_s,x_m,y_m,z_m,yaw_deg,vx_m_per_s,vy_m_per_s,omega_z_rad_per_s,"
                "face_contact_count,rigid_contact_count,normal_lateral_avg,normal_lateral_max,"
                "normal_lateral_count_gt_0_2\n"
            )
            for row in zip(
                self.time_log,
                self.x_log,
                self.y_log,
                self.z_log,
                self.yaw_log,
                self.vx_log,
                self.vy_log,
                self.omega_z_log,
                self.face_contact_count_log,
                self.rigid_contact_count_log,
                self.normal_lateral_avg_log,
                self.normal_lateral_max_log,
                self.normal_lateral_count_gt_0_2_log,
                strict=False,
            ):
                f.write(
                    f"{row[0]:.6f},{row[1]:.8f},{row[2]:.8f},{row[3]:.8f},{row[4]:.6f},"
                    f"{row[5]:.8f},{row[6]:.8f},{row[7]:.8f},{row[8]},{row[9]},"
                    f"{row[10]:.8f},{row[11]:.8f},{row[12]}\n"
                )
        print(
            f"[{suffix}] final y={(self.y_log[-1] - self.y_log[0]) * 1000.0:.3f} mm "
            f"yaw={self.yaw_log[-1] - self.yaw_log[0]:.3f} deg "
            f"max_contacts={self.max_reduced_contacts}"
        )
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
            help="Enable hydroelastic contact reduction.",
        )
        parser.add_argument(
            "--pre-prune-contacts",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Enable local pre-prune before global hydroelastic reduction.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
