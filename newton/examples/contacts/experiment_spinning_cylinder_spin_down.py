# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Pure spinning-cylinder contact-reduction experiment.
#
# Command:
#     python -m newton.examples.contacts.experiment_spinning_cylinder_spin_down
#
###########################################################################

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton.examples.contacts import example_spinning_cylinder as spinning
from newton.examples.contacts.experiment_cube_on_plate_settle import (
    BUFFER_FRACTION,
    BUFFER_MULT_CONTACT,
    BUFFER_MULT_ISO,
    MUJOCO_NCONMAX,
    MUJOCO_NJMAX,
    RIGID_CONTACT_MAX,
    BufferStats,
    _buffer_stats_with_capacities,
    _format_csv_value,
    _summary_stats,
)
from newton.geometry import HydroelasticSDF

DEFAULT_OUTPUT_DIR = Path("output") / "H4_spinning_cylinder_spin_down"
DEFAULT_INITIAL_OMEGAS = (15.0, 30.0, 60.0)
DEFAULT_SDF_MAX_RESOLUTION = 32

FRAME_FPS = 120
RUN_SECONDS = 1.35
SIM_SUBSTEPS = 4
STOP_OMEGA_RAD_PER_S = 0.5

TIMESERIES_CSV = "spin_down_timeseries.csv"
SUMMARY_CSV = "spin_down_summary.csv"

TIMESERIES_COLUMNS = (
    "time_s",
    "initial_omega_rad_per_s",
    "mode",
    "reduce_contacts",
    "pre_prune_contacts",
    "anchor_contact",
    "moment_matching",
    "face_contact_count",
    "rigid_contact_count",
    "cylinder_x_m",
    "cylinder_y_m",
    "cylinder_z_m",
    "cylinder_vx_m_per_s",
    "cylinder_vy_m_per_s",
    "cylinder_vz_m_per_s",
    "cylinder_omega_x_rad_per_s",
    "cylinder_omega_y_rad_per_s",
    "cylinder_omega_z_rad_per_s",
    "omega_over_omega0",
    "cylinder_tilt_deg",
    "cylinder_lateral_drift_m",
    "cylinder_penetration_depth_m",
    "solver_fx_N",
    "solver_fy_N",
    "solver_fz_N",
    "solver_tx_Nm",
    "solver_ty_Nm",
    "solver_tz_Nm",
    "solver_force_count",
    "buffer_overflow",
)

SUMMARY_COLUMNS = (
    "initial_omega_rad_per_s",
    "mode",
    "reduce_contacts",
    "pre_prune_contacts",
    "anchor_contact",
    "moment_matching",
    "frame_count",
    "stopped",
    "stop_omega_threshold_rad_per_s",
    "expected_uniform_torque_z_Nm",
    "expected_uniform_angular_accel_rad_per_s2",
    "expected_uniform_stop_time_s",
    "stop_time_s",
    "stop_time_error_s",
    "final_omega_z_rad_per_s",
    "final_omega_over_omega0",
    "final_tilt_deg",
    "final_lateral_drift_m",
    "final_penetration_depth_m",
    "max_penetration_depth_m",
    "mean_solver_tz_active_Nm",
    "mean_abs_solver_tz_active_Nm",
    "solver_yaw_impulse_Nms",
    "expected_yaw_impulse_to_stop_Nms",
    "mean_solver_force_count",
    "mean_rigid_contact_count",
    "mean_face_contact_count",
    "buffer_overflow",
    "max_hydro_broadphase_blocks",
    "hydro_broadphase_capacity",
    "max_hydro_iso_subblocks_l0",
    "hydro_iso_subblocks_l0_capacity",
    "max_hydro_iso_subblocks_l1",
    "hydro_iso_subblocks_l1_capacity",
    "max_hydro_iso_subblocks_l2",
    "hydro_iso_subblocks_l2_capacity",
    "max_hydro_iso_voxels",
    "hydro_iso_voxels_capacity",
    "max_face_contact_count",
    "face_contact_capacity",
    "max_rigid_contact_count",
    "rigid_contact_capacity",
    "max_reduction_hashtable_active",
    "reduction_hashtable_capacity",
    "max_reduction_hashtable_failures",
)


@dataclass(frozen=True)
class RunResult:
    """Output summary for one initial-spin/mode run."""

    initial_omega: float
    mode: str
    timeseries_path: Path
    summary_path: Path
    row_count: int
    buffer_overflow: bool


@dataclass(frozen=True)
class SpinModeConfig:
    """Contact-reduction mode for one spin-down run."""

    name: str
    reduce_contacts: bool
    pre_prune_contacts: bool = False
    anchor_contact: bool = False
    moment_matching: bool = False


@dataclass
class _RunData:
    initial_omega: float
    mode: str
    timeseries_rows: list[dict[str, object]]
    summary_row: dict[str, object]
    buffer_overflow: bool


@dataclass(frozen=True)
class _FrameData:
    timeseries_row: dict[str, object]
    buffer_stats: BufferStats


def _cylinder_mass() -> float:
    height = 2.0 * spinning.CYLINDER_HALF_HEIGHT
    return spinning.CYLINDER_DENSITY * math.pi * spinning.CYLINDER_RADIUS**2 * height


def _cylinder_izz() -> float:
    return 0.5 * _cylinder_mass() * spinning.CYLINDER_RADIUS**2


def _expected_uniform_torque_z(initial_omega: float) -> float:
    direction = 1.0 if initial_omega >= 0.0 else -1.0
    return (
        -(2.0 / 3.0) * spinning.MU_SLIDING * _cylinder_mass() * spinning.GRAVITY * spinning.CYLINDER_RADIUS * direction
    )


def _expected_uniform_stop_time(initial_omega: float) -> float:
    return 0.75 * abs(initial_omega) * spinning.CYLINDER_RADIUS / (spinning.MU_SLIDING * spinning.GRAVITY)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _stop_index(rows: list[dict[str, object]]) -> int | None:
    for idx, row in enumerate(rows):
        if abs(float(row["cylinder_omega_z_rad_per_s"])) <= STOP_OMEGA_RAD_PER_S:
            return idx
    return None


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format_csv_value(row[key]) for key in fieldnames})


def _cylinder_local_z(body_q: np.ndarray) -> tuple[float, float, float]:
    qx = float(body_q[3])
    qy = float(body_q[4])
    qz = float(body_q[5])
    qw = float(body_q[6])
    q_norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if q_norm > 0.0:
        qx /= q_norm
        qy /= q_norm
        qz /= q_norm
        qw /= q_norm
    return (
        2.0 * (qx * qz + qw * qy),
        2.0 * (qy * qz - qw * qx),
        1.0 - 2.0 * (qx * qx + qy * qy),
    )


def _cylinder_tilt_deg(body_q: np.ndarray) -> float:
    local_z = _cylinder_local_z(body_q)
    return math.degrees(math.acos(max(-1.0, min(1.0, local_z[2]))))


def _cylinder_signed_clearance_m(body_q: np.ndarray) -> float:
    local_z = _cylinder_local_z(body_q)
    axial_projection = spinning.CYLINDER_HALF_HEIGHT * abs(local_z[2])
    radial_projection = spinning.CYLINDER_RADIUS * math.sqrt(max(0.0, 1.0 - local_z[2] * local_z[2]))
    bottom_z = float(body_q[2]) - axial_projection - radial_projection
    return bottom_z


def _print_buffer_summary(initial_omega: float, mode: str, max_stats: BufferStats) -> None:
    print(f"[initial_omega={initial_omega:.6f} rad/s mode={mode}] buffer summary")
    for label, max_count, capacity, utilization in _summary_stats(max_stats):
        print(f"  {label}: max={max_count} capacity={capacity} utilization={utilization:.4f}")
    if max_stats.reduction_hashtable_failures:
        print(f"  reduction_hashtable_failures: max={max_stats.reduction_hashtable_failures}")


class _SpinDownRun:
    def __init__(
        self,
        *,
        initial_omega: float,
        initial_vx: float = 0.0,
        mode: SpinModeConfig,
        rigid_contact_max: int,
        nconmax: int,
        njmax: int,
        buffer_mult_iso: int,
        buffer_mult_contact: int,
        buffer_fraction: float,
        sdf_max_resolution: int,
        device: str | None,
    ) -> None:
        self.initial_omega = float(initial_omega)
        self.initial_vx = float(initial_vx)
        self.mode = mode

        shape_cfg = newton.ModelBuilder.ShapeConfig(
            margin=0.0,
            gap=0.0005,
            mu=spinning.MU_SLIDING,
            mu_torsional=0.0,
            mu_rolling=0.0,
            ke=1.0e7,
            kd=1.0e5,
            density=spinning.CYLINDER_DENSITY,
            is_hydroelastic=True,
            kh=spinning.KH,
            sdf_max_resolution=int(sdf_max_resolution),
            sdf_narrow_band_range=spinning.SDF_NARROW_BAND_RANGE,
        )

        builder = newton.ModelBuilder()
        builder.default_shape_cfg.gap = 0.001

        builder.add_shape_box(
            body=-1,
            xform=wp.transform(
                wp.vec3(0.0, 0.0, -spinning.PLATE_HALF_THICKNESS),
                wp.quat_identity(),
            ),
            hx=spinning.PLATE_HALF_EXTENT,
            hy=spinning.PLATE_HALF_EXTENT,
            hz=spinning.PLATE_HALF_THICKNESS,
            cfg=shape_cfg,
            label="plate",
        )

        cylinder_z = spinning.CYLINDER_HALF_HEIGHT - 0.0002
        self.cylinder_body = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, cylinder_z), wp.quat_identity()),
            label="cylinder",
        )
        self.cylinder_shape = builder.add_shape_cylinder(
            body=self.cylinder_body,
            radius=spinning.CYLINDER_RADIUS,
            half_height=spinning.CYLINDER_HALF_HEIGHT,
            cfg=shape_cfg,
            label="cylinder_shape",
        )

        qd_start = builder.joint_qd_start[-1]
        builder.joint_qd[qd_start + 0] = self.initial_vx
        builder.joint_qd[qd_start + 5] = self.initial_omega

        self.model = builder.finalize(device=device)
        self.model.request_contact_attributes("force")

        hydro_cfg = HydroelasticSDF.Config(
            output_contact_surface=True,
            reduce_contacts=mode.reduce_contacts,
            pre_prune_contacts=mode.pre_prune_contacts,
            anchor_contact=mode.anchor_contact,
            moment_matching=mode.moment_matching,
            buffer_mult_iso=int(buffer_mult_iso),
            buffer_mult_contact=int(buffer_mult_contact),
            buffer_fraction=float(buffer_fraction),
        )
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            sdf_hydroelastic_config=hydro_cfg,
            rigid_contact_max=int(rigid_contact_max),
            broad_phase="sap",
        )

        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            use_mujoco_contacts=False,
            solver="newton",
            integrator="implicitfast",
            cone="elliptic",
            njmax=int(njmax),
            nconmax=int(nconmax),
            iterations=15,
            ls_iterations=100,
            impratio=1.0,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.contacts = self.collision_pipeline.contacts()
        self.force_contacts = self.collision_pipeline.contacts()
        if self.force_contacts.force is None:
            raise RuntimeError("force contact buffer was not allocated")

        self._shape_body = self.model.shape_body.numpy()
        self._capacities = self._read_capacities()
        body_q = self.state_0.body_q.numpy()[self.cylinder_body]
        self._initial_x = float(body_q[0])
        self._initial_y = float(body_q[1])

    def _read_capacities(self) -> BufferStats:
        hydro = self.collision_pipeline.hydroelastic_sdf
        if hydro is None:
            raise RuntimeError("spin-down experiment requires hydroelastic contacts")
        reducer = hydro.contact_reduction.reducer
        return BufferStats(
            hydro_broadphase_capacity=hydro.max_num_blocks_broad,
            hydro_iso_subblocks_l0_capacity=hydro.iso_max_dims[0],
            hydro_iso_subblocks_l1_capacity=hydro.iso_max_dims[1],
            hydro_iso_subblocks_l2_capacity=hydro.iso_max_dims[2],
            hydro_iso_voxels_capacity=hydro.max_num_iso_voxels,
            face_contact_capacity=hydro.max_num_face_contacts,
            rigid_contact_capacity=self.contacts.rigid_contact_max,
            reduction_hashtable_capacity=reducer.hashtable.capacity,
        )

    def _read_buffer_stats(self) -> BufferStats:
        hydro = self.collision_pipeline.hydroelastic_sdf
        reducer = hydro.contact_reduction.reducer
        active_slots = reducer.hashtable.active_slots.numpy()
        capacity = reducer.hashtable.capacity
        return BufferStats(
            hydro_broadphase_blocks=int(hydro.block_broad_collide_count.numpy()[0]),
            hydro_iso_subblocks_l0=int(hydro.iso_buffer_counts[1].numpy()[0]),
            hydro_iso_subblocks_l1=int(hydro.iso_buffer_counts[2].numpy()[0]),
            hydro_iso_subblocks_l2=int(hydro.iso_buffer_counts[3].numpy()[0]),
            hydro_iso_voxels=int(hydro.iso_voxel_count.numpy()[0]),
            face_contact_count=int(hydro.contact_reduction.contact_count.numpy()[0]),
            rigid_contact_count=int(self.contacts.rigid_contact_count.numpy()[0]),
            reduction_hashtable_active=int(active_slots[capacity]) if self.mode.reduce_contacts else 0,
            reduction_hashtable_failures=int(reducer.ht_insert_failures.numpy()[0]) if self.mode.reduce_contacts else 0,
            hydro_broadphase_capacity=self._capacities.hydro_broadphase_capacity,
            hydro_iso_subblocks_l0_capacity=self._capacities.hydro_iso_subblocks_l0_capacity,
            hydro_iso_subblocks_l1_capacity=self._capacities.hydro_iso_subblocks_l1_capacity,
            hydro_iso_subblocks_l2_capacity=self._capacities.hydro_iso_subblocks_l2_capacity,
            hydro_iso_voxels_capacity=self._capacities.hydro_iso_voxels_capacity,
            face_contact_capacity=self._capacities.face_contact_capacity,
            rigid_contact_capacity=self._capacities.rigid_contact_capacity,
            reduction_hashtable_capacity=self._capacities.reduction_hashtable_capacity,
        )

    def _point_world(self, body_q: np.ndarray, body: int, local_point: np.ndarray) -> np.ndarray:
        if body < 0:
            return np.asarray(local_point, dtype=np.float64)

        q = body_q[body]
        pos = np.asarray(q[:3], dtype=np.float64)
        qvec = np.asarray(q[3:6], dtype=np.float64)
        qw = float(q[6])
        q_norm = math.sqrt(float(np.dot(qvec, qvec)) + qw * qw)
        if q_norm > 0.0:
            qvec = qvec / q_norm
            qw /= q_norm

        point = np.asarray(local_point, dtype=np.float64)
        t = 2.0 * np.cross(qvec, point)
        return pos + point + qw * t + np.cross(qvec, t)

    def _solver_wrench_on_cylinder(self) -> tuple[np.ndarray, np.ndarray, int]:
        n_force = int(self.force_contacts.rigid_contact_count.numpy()[0])
        if n_force <= 0:
            return np.zeros(3), np.zeros(3), 0

        body_q = self.state_0.body_q.numpy()
        cylinder_q = body_q[self.cylinder_body]
        cylinder_pos = np.asarray(cylinder_q[:3], dtype=np.float64)

        forces = self.force_contacts.force.numpy()[:n_force]
        shapes0 = self.force_contacts.rigid_contact_shape0.numpy()[:n_force]
        shapes1 = self.force_contacts.rigid_contact_shape1.numpy()[:n_force]
        points0 = self.force_contacts.rigid_contact_point0.numpy()[:n_force]
        points1 = self.force_contacts.rigid_contact_point1.numpy()[:n_force]

        force_sum = np.zeros(3, dtype=np.float64)
        torque_sum = np.zeros(3, dtype=np.float64)
        cylinder_force_count = 0
        for contact_idx in range(n_force):
            shape0 = int(shapes0[contact_idx])
            shape1 = int(shapes1[contact_idx])
            force_shape0 = np.asarray(forces[contact_idx, :3], dtype=np.float64)
            torque_shape0 = np.asarray(forces[contact_idx, 3:6], dtype=np.float64)
            if shape0 == self.cylinder_shape:
                body0 = int(self._shape_body[shape0])
                force = force_shape0
                point_world = self._point_world(body_q, body0, points0[contact_idx])
                torque = torque_shape0 + np.cross(point_world - cylinder_pos, force)
            elif shape1 == self.cylinder_shape:
                body1 = int(self._shape_body[shape1])
                force = -force_shape0
                point_world = self._point_world(body_q, body1, points1[contact_idx])
                torque = -torque_shape0 + np.cross(point_world - cylinder_pos, force)
            else:
                continue

            force_sum += force
            torque_sum += torque
            cylinder_force_count += 1

        return force_sum, torque_sum, cylinder_force_count

    def _timeseries_row(
        self,
        *,
        time_s: float,
        frame_stats: BufferStats,
        solver_force: np.ndarray,
        solver_torque: np.ndarray,
        solver_force_count: int,
    ) -> dict[str, object]:
        body_q = self.state_0.body_q.numpy()[self.cylinder_body]
        body_qd = self.state_0.body_qd.numpy()[self.cylinder_body]
        x = float(body_q[0])
        y = float(body_q[1])
        signed_clearance = _cylinder_signed_clearance_m(body_q)
        return {
            "time_s": time_s,
            "initial_omega_rad_per_s": self.initial_omega,
            "mode": self.mode.name,
            "reduce_contacts": self.mode.reduce_contacts,
            "pre_prune_contacts": self.mode.pre_prune_contacts,
            "anchor_contact": self.mode.anchor_contact,
            "moment_matching": self.mode.moment_matching,
            "face_contact_count": frame_stats.face_contact_count,
            "rigid_contact_count": frame_stats.rigid_contact_count,
            "cylinder_x_m": x,
            "cylinder_y_m": y,
            "cylinder_z_m": float(body_q[2]),
            "cylinder_vx_m_per_s": float(body_qd[0]),
            "cylinder_vy_m_per_s": float(body_qd[1]),
            "cylinder_vz_m_per_s": float(body_qd[2]),
            "cylinder_omega_x_rad_per_s": float(body_qd[3]),
            "cylinder_omega_y_rad_per_s": float(body_qd[4]),
            "cylinder_omega_z_rad_per_s": float(body_qd[5]),
            "omega_over_omega0": float(body_qd[5]) / self.initial_omega,
            "cylinder_tilt_deg": _cylinder_tilt_deg(body_q),
            "cylinder_lateral_drift_m": math.hypot(x - self._initial_x, y - self._initial_y),
            "cylinder_penetration_depth_m": max(0.0, -signed_clearance),
            "solver_fx_N": float(solver_force[0]),
            "solver_fy_N": float(solver_force[1]),
            "solver_fz_N": float(solver_force[2]),
            "solver_tx_Nm": float(solver_torque[0]),
            "solver_ty_Nm": float(solver_torque[1]),
            "solver_tz_Nm": float(solver_torque[2]),
            "solver_force_count": int(solver_force_count),
            "buffer_overflow": frame_stats.overflow,
        }

    def simulate_frame(self, *, frame_dt: float, sim_substeps: int, time_s: float) -> _FrameData:
        sim_dt = frame_dt / sim_substeps
        frame_stats = _buffer_stats_with_capacities(self._capacities)
        solver_force = np.zeros(3, dtype=np.float64)
        solver_torque = np.zeros(3, dtype=np.float64)
        solver_force_count = 0

        for _ in range(sim_substeps):
            self.collision_pipeline.collide(self.state_0, self.contacts)
            frame_stats.update_max(self._read_buffer_stats())
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.solver.update_contacts(self.force_contacts, self.state_0)
            solver_force, solver_torque, solver_force_count = self._solver_wrench_on_cylinder()

        return _FrameData(
            timeseries_row=self._timeseries_row(
                time_s=time_s,
                frame_stats=frame_stats,
                solver_force=solver_force,
                solver_torque=solver_torque,
                solver_force_count=solver_force_count,
            ),
            buffer_stats=frame_stats,
        )


def _summary_row(
    *,
    initial_omega: float,
    mode: SpinModeConfig,
    frame_dt: float,
    frame_count: int,
    max_stats: BufferStats,
    rows: list[dict[str, object]],
) -> dict[str, object]:
    stop_idx = _stop_index(rows)
    stop_row = rows[stop_idx] if stop_idx is not None else None
    final_row = rows[-1]
    expected_torque = _expected_uniform_torque_z(initial_omega)
    expected_accel = expected_torque / _cylinder_izz()
    expected_stop_time = _expected_uniform_stop_time(initial_omega)
    active_rows = rows[: stop_idx + 1] if stop_idx is not None else rows
    active_rows = [row for row in active_rows if int(row["solver_force_count"]) > 0]
    solver_yaw_impulse = sum(float(row["solver_tz_Nm"]) * frame_dt for row in rows)
    expected_yaw_impulse = -_cylinder_izz() * initial_omega
    return {
        "initial_omega_rad_per_s": initial_omega,
        "mode": mode.name,
        "reduce_contacts": mode.reduce_contacts,
        "pre_prune_contacts": mode.pre_prune_contacts,
        "anchor_contact": mode.anchor_contact,
        "moment_matching": mode.moment_matching,
        "frame_count": frame_count,
        "stopped": stop_row is not None,
        "stop_omega_threshold_rad_per_s": STOP_OMEGA_RAD_PER_S,
        "expected_uniform_torque_z_Nm": expected_torque,
        "expected_uniform_angular_accel_rad_per_s2": expected_accel,
        "expected_uniform_stop_time_s": expected_stop_time,
        "stop_time_s": float(stop_row["time_s"]) if stop_row is not None else float("nan"),
        "stop_time_error_s": float(stop_row["time_s"]) - expected_stop_time if stop_row is not None else float("nan"),
        "final_omega_z_rad_per_s": float(final_row["cylinder_omega_z_rad_per_s"]),
        "final_omega_over_omega0": float(final_row["omega_over_omega0"]),
        "final_tilt_deg": float(final_row["cylinder_tilt_deg"]),
        "final_lateral_drift_m": float(final_row["cylinder_lateral_drift_m"]),
        "final_penetration_depth_m": float(final_row["cylinder_penetration_depth_m"]),
        "max_penetration_depth_m": max(float(row["cylinder_penetration_depth_m"]) for row in rows),
        "mean_solver_tz_active_Nm": _mean([float(row["solver_tz_Nm"]) for row in active_rows]),
        "mean_abs_solver_tz_active_Nm": _mean([abs(float(row["solver_tz_Nm"])) for row in active_rows]),
        "solver_yaw_impulse_Nms": solver_yaw_impulse,
        "expected_yaw_impulse_to_stop_Nms": expected_yaw_impulse,
        "mean_solver_force_count": _mean([float(row["solver_force_count"]) for row in rows]),
        "mean_rigid_contact_count": _mean([float(row["rigid_contact_count"]) for row in rows]),
        "mean_face_contact_count": _mean([float(row["face_contact_count"]) for row in rows]),
        "buffer_overflow": max_stats.overflow,
        "max_hydro_broadphase_blocks": max_stats.hydro_broadphase_blocks,
        "hydro_broadphase_capacity": max_stats.hydro_broadphase_capacity,
        "max_hydro_iso_subblocks_l0": max_stats.hydro_iso_subblocks_l0,
        "hydro_iso_subblocks_l0_capacity": max_stats.hydro_iso_subblocks_l0_capacity,
        "max_hydro_iso_subblocks_l1": max_stats.hydro_iso_subblocks_l1,
        "hydro_iso_subblocks_l1_capacity": max_stats.hydro_iso_subblocks_l1_capacity,
        "max_hydro_iso_subblocks_l2": max_stats.hydro_iso_subblocks_l2,
        "hydro_iso_subblocks_l2_capacity": max_stats.hydro_iso_subblocks_l2_capacity,
        "max_hydro_iso_voxels": max_stats.hydro_iso_voxels,
        "hydro_iso_voxels_capacity": max_stats.hydro_iso_voxels_capacity,
        "max_face_contact_count": max_stats.face_contact_count,
        "face_contact_capacity": max_stats.face_contact_capacity,
        "max_rigid_contact_count": max_stats.rigid_contact_count,
        "rigid_contact_capacity": max_stats.rigid_contact_capacity,
        "max_reduction_hashtable_active": max_stats.reduction_hashtable_active,
        "reduction_hashtable_capacity": max_stats.reduction_hashtable_capacity,
        "max_reduction_hashtable_failures": max_stats.reduction_hashtable_failures,
    }


def run_single(
    *,
    initial_omega: float,
    mode: SpinModeConfig,
    simulation_time: float,
    frame_fps: int,
    sim_substeps: int,
    rigid_contact_max: int,
    nconmax: int,
    njmax: int,
    buffer_mult_iso: int,
    buffer_mult_contact: int,
    buffer_fraction: float,
    sdf_max_resolution: int,
    device: str | None = None,
    verbose: bool = True,
) -> _RunData:
    runner = _SpinDownRun(
        initial_omega=initial_omega,
        mode=mode,
        rigid_contact_max=rigid_contact_max,
        nconmax=nconmax,
        njmax=njmax,
        buffer_mult_iso=buffer_mult_iso,
        buffer_mult_contact=buffer_mult_contact,
        buffer_fraction=buffer_fraction,
        sdf_max_resolution=sdf_max_resolution,
        device=device,
    )

    frame_count = int(math.ceil(simulation_time * frame_fps))
    frame_dt = 1.0 / frame_fps
    rows: list[dict[str, object]] = []
    max_stats = _buffer_stats_with_capacities(runner._capacities)
    for frame_idx in range(frame_count):
        frame_data = runner.simulate_frame(
            frame_dt=frame_dt,
            sim_substeps=sim_substeps,
            time_s=(frame_idx + 1) * frame_dt,
        )
        rows.append(frame_data.timeseries_row)
        max_stats.update_max(frame_data.buffer_stats)

    summary_row = _summary_row(
        initial_omega=initial_omega,
        mode=mode,
        frame_dt=frame_dt,
        frame_count=len(rows),
        max_stats=max_stats,
        rows=rows,
    )
    if verbose:
        print(f"simulated initial_omega={initial_omega:.6f} rad/s mode={mode.name} ({len(rows)} samples)")
        _print_buffer_summary(initial_omega, mode.name, max_stats)

    if max_stats.overflow:
        raise RuntimeError(
            f"contact-buffer validity gate failed for initial_omega={initial_omega:.6f} rad/s mode={mode.name}"
        )

    return _RunData(
        initial_omega=initial_omega,
        mode=mode.name,
        timeseries_rows=rows,
        summary_row=summary_row,
        buffer_overflow=max_stats.overflow,
    )


def _comparison_status(summary_rows: list[dict[str, object]]) -> None:
    by_omega: dict[float, dict[str, dict[str, object]]] = {}
    for row in summary_rows:
        by_omega.setdefault(float(row["initial_omega_rad_per_s"]), {})[str(row["mode"])] = row

    for initial_omega in sorted(by_omega):
        reduced = by_omega[initial_omega].get("reduced")
        unreduced = by_omega[initial_omega].get("unreduced")
        if reduced is None or unreduced is None:
            continue
        count_ratio = float(reduced["mean_solver_force_count"]) / max(
            float(unreduced["mean_solver_force_count"]),
            1.0,
        )
        print(
            f"[omega0={initial_omega:.6f} rad/s] "
            f"expected_stop={float(unreduced['expected_uniform_stop_time_s']):.6g} s, "
            f"stop_time off/on={float(unreduced['stop_time_s']):.6g}/"
            f"{float(reduced['stop_time_s']):.6g} s, "
            f"mean |Tz| off/on={float(unreduced['mean_abs_solver_tz_active_Nm']):.6g}/"
            f"{float(reduced['mean_abs_solver_tz_active_Nm']):.6g} N*m, "
            f"force-count ratio={count_ratio:.5g}"
        )


def run_experiment(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    initial_omegas: tuple[float, ...] | list[float] = DEFAULT_INITIAL_OMEGAS,
    simulation_time: float = RUN_SECONDS,
    frame_fps: int = FRAME_FPS,
    sim_substeps: int = SIM_SUBSTEPS,
    rigid_contact_max: int = RIGID_CONTACT_MAX,
    nconmax: int = MUJOCO_NCONMAX,
    njmax: int = MUJOCO_NJMAX,
    buffer_mult_iso: int = BUFFER_MULT_ISO,
    buffer_mult_contact: int = BUFFER_MULT_CONTACT,
    buffer_fraction: float = BUFFER_FRACTION,
    sdf_max_resolution: int = DEFAULT_SDF_MAX_RESOLUTION,
    include_moment_matching: bool = False,
    device: str | None = None,
    verbose: bool = True,
) -> list[RunResult]:
    if simulation_time <= 0.0:
        raise ValueError(f"simulation_time must be positive; got {simulation_time}")
    if frame_fps <= 0:
        raise ValueError(f"frame_fps must be positive; got {frame_fps}")
    if sim_substeps <= 0:
        raise ValueError(f"sim_substeps must be positive; got {sim_substeps}")
    if device:
        wp.set_device(device)

    output_dir = Path(output_dir)
    timeseries_path = output_dir / TIMESERIES_CSV
    summary_path = output_dir / SUMMARY_CSV
    modes = [
        SpinModeConfig(name="unreduced", reduce_contacts=False),
        SpinModeConfig(name="reduced", reduce_contacts=True),
    ]
    if include_moment_matching:
        modes.append(
            SpinModeConfig(
                name="reduced_moment_matching",
                reduce_contacts=True,
                anchor_contact=True,
                moment_matching=True,
            )
        )
    run_data: list[_RunData] = []
    for initial_omega in initial_omegas:
        if float(initial_omega) <= 0.0:
            raise ValueError(f"initial omegas must be positive; got {initial_omega}")
        for mode in modes:
            run_data.append(
                run_single(
                    initial_omega=float(initial_omega),
                    mode=mode,
                    simulation_time=simulation_time,
                    frame_fps=frame_fps,
                    sim_substeps=sim_substeps,
                    rigid_contact_max=rigid_contact_max,
                    nconmax=nconmax,
                    njmax=njmax,
                    buffer_mult_iso=buffer_mult_iso,
                    buffer_mult_contact=buffer_mult_contact,
                    buffer_fraction=buffer_fraction,
                    sdf_max_resolution=sdf_max_resolution,
                    device=device,
                    verbose=verbose,
                )
            )

    timeseries_rows = [row for data in run_data for row in data.timeseries_rows]
    summary_rows = [data.summary_row for data in run_data]
    _write_csv(timeseries_path, timeseries_rows, TIMESERIES_COLUMNS)
    _write_csv(summary_path, summary_rows, SUMMARY_COLUMNS)

    if verbose:
        print(f"wrote {timeseries_path} ({len(timeseries_rows)} samples)")
        print(f"wrote {summary_path} ({len(summary_rows)} runs)")
        _comparison_status(summary_rows)

    return [
        RunResult(
            initial_omega=data.initial_omega,
            mode=data.mode,
            timeseries_path=timeseries_path,
            summary_path=summary_path,
            row_count=len(data.timeseries_rows),
            buffer_overflow=data.buffer_overflow,
        )
        for data in run_data
    ]


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Directory for CSV files.")
    parser.add_argument(
        "--initial-omegas",
        type=float,
        nargs="+",
        default=list(DEFAULT_INITIAL_OMEGAS),
        help="Initial yaw rates [rad/s] to sweep.",
    )
    parser.add_argument("--simulation-time", type=float, default=RUN_SECONDS, help="Simulation time per run [s].")
    parser.add_argument("--fps", type=int, default=FRAME_FPS, help="Logged frame rate [Hz].")
    parser.add_argument("--substeps", type=int, default=SIM_SUBSTEPS, help="Solver substeps per logged frame.")
    parser.add_argument("--device", type=str, default=None, help="Override the Warp device.")
    parser.add_argument("--quiet", action="store_true", help="Suppress Warp compilation and per-run summaries.")
    parser.add_argument("--rigid-contact-max", type=int, default=RIGID_CONTACT_MAX)
    parser.add_argument("--nconmax", type=int, default=MUJOCO_NCONMAX)
    parser.add_argument("--njmax", type=int, default=MUJOCO_NJMAX)
    parser.add_argument("--buffer-mult-iso", type=int, default=BUFFER_MULT_ISO)
    parser.add_argument("--buffer-mult-contact", type=int, default=BUFFER_MULT_CONTACT)
    parser.add_argument("--buffer-fraction", type=float, default=BUFFER_FRACTION)
    parser.add_argument(
        "--sdf-max-resolution",
        type=int,
        default=DEFAULT_SDF_MAX_RESOLUTION,
        help="Maximum SDF grid resolution along the cylinder or plate longest axis.",
    )
    parser.add_argument(
        "--include-moment-matching",
        action="store_true",
        help="Also run reduce on with anchor_contact=True and moment_matching=True.",
    )
    return parser


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    if args.quiet:
        wp.config.quiet = True
    run_experiment(
        output_dir=args.output_dir,
        initial_omegas=tuple(args.initial_omegas),
        simulation_time=args.simulation_time,
        frame_fps=args.fps,
        sim_substeps=args.substeps,
        rigid_contact_max=args.rigid_contact_max,
        nconmax=args.nconmax,
        njmax=args.njmax,
        buffer_mult_iso=args.buffer_mult_iso,
        buffer_mult_contact=args.buffer_mult_contact,
        buffer_fraction=args.buffer_fraction,
        sdf_max_resolution=args.sdf_max_resolution,
        include_moment_matching=args.include_moment_matching,
        device=args.device,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
