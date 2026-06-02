# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Cube-on-plate impact ring-down contact-reduction experiment.
#
# Command:
#     python -m newton.examples.contacts.experiment_cube_on_plate_impact
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
from newton.examples.contacts import example_cube_on_plate as cube_on_plate
from newton.examples.contacts.experiment_cube_on_plate_settle import (
    BUFFER_FRACTION,
    BUFFER_MULT_CONTACT,
    BUFFER_MULT_ISO,
    MUJOCO_NCONMAX,
    MUJOCO_NJMAX,
    RIGID_CONTACT_MAX,
    BufferStats,
    ModeConfig,
    SceneConfig,
    _buffer_stats_with_capacities,
    _cube_signed_clearance_m,
    _cube_tilt_deg,
    _format_csv_value,
    _load_scene_config,
    _summary_stats,
)
from newton.geometry import HydroelasticSDF

DEFAULT_OUTPUT_DIR = Path("output") / "H6_cube_on_plate_impact"
DEFAULT_DROP_HEIGHTS = (0.001, 0.0025, 0.005, 0.010, 0.020)

STEP_DT = 0.00025
RUN_SECONDS = 0.35
SETTLE_WINDOW_S = 0.05
SETTLE_FORCE_TOL = 0.02
SETTLE_VZ_TOL = 0.005
FINAL_TILT_LIMIT_DEG = 0.5
FINAL_DRIFT_LIMIT_M = 1.0e-4

TIMESERIES_CSV = "impact_timeseries.csv"
SUMMARY_CSV = "impact_summary.csv"

TIMESERIES_COLUMNS = (
    "time_s",
    "height_m",
    "mode",
    "reduce_contacts",
    "pre_prune_contacts",
    "cube_x_m",
    "cube_y_m",
    "cube_z_m",
    "cube_vz_m_per_s",
    "cube_tilt_deg",
    "cube_signed_clearance_m",
    "cube_penetration_depth_m",
    "solver_fx_N",
    "solver_fy_N",
    "solver_fz_N",
    "solver_tx_Nm",
    "solver_ty_Nm",
    "solver_tz_Nm",
    "solver_force_count",
    "rigid_contact_count",
    "face_contact_count",
    "buffer_overflow",
    "state_invalid",
)

SUMMARY_COLUMNS = (
    "height_m",
    "mode",
    "reduce_contacts",
    "pre_prune_contacts",
    "frame_count",
    "simulation_time_s",
    "step_dt_s",
    "cube_mass_kg",
    "cube_weight_N",
    "impact_velocity_m_per_s",
    "first_contact_time_s",
    "peak_solver_fz_N",
    "peak_solver_fz_over_weight",
    "time_to_peak_fz_s",
    "max_penetration_depth_m",
    "max_upward_rebound_velocity_m_per_s",
    "rebound_velocity_ratio",
    "settle_time_s",
    "post_settle_fz_rms_N",
    "normal_impulse_Ns",
    "final_tilt_deg",
    "final_drift_m",
    "mean_solver_force_count",
    "max_rigid_contact_count",
    "max_face_contact_count",
    "buffer_overflow",
    "state_invalid",
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
    "max_rigid_contact_count_buffer",
    "rigid_contact_capacity",
    "max_reduction_hashtable_active",
    "reduction_hashtable_capacity",
    "max_reduction_hashtable_failures",
    "valid_run",
)


@dataclass(frozen=True)
class RunResult:
    """Output summary for one height/mode run."""

    height: float
    mode: str
    timeseries_path: Path
    summary_path: Path
    row_count: int
    valid_run: bool


@dataclass
class _RunData:
    height: float
    mode: str
    timeseries_rows: list[dict[str, object]]
    summary_row: dict[str, object]
    valid_run: bool


@dataclass(frozen=True)
class _FrameData:
    timeseries_row: dict[str, object]
    buffer_stats: BufferStats


def _cube_mass(scene: SceneConfig) -> float:
    side = 2.0 * scene.cube_half_extent
    return scene.cube_density * side**3


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format_csv_value(row[key]) for key in fieldnames})


class _ImpactRun:
    def __init__(
        self,
        *,
        scene: SceneConfig,
        height: float,
        mode: ModeConfig,
        rigid_contact_max: int,
        nconmax: int,
        njmax: int,
        buffer_mult_iso: int,
        buffer_mult_contact: int,
        buffer_fraction: float,
        device: str | None,
    ) -> None:
        self.scene = scene
        self.height = float(height)
        self.mode = mode

        shape_cfg = newton.ModelBuilder.ShapeConfig(
            margin=0.0,
            gap=0.0005,
            mu=scene.mu_sliding,
            mu_torsional=0.0,
            mu_rolling=0.0,
            ke=1.0e7,
            kd=1.0e5,
            density=scene.cube_density,
            is_hydroelastic=True,
            kh=scene.kh,
            sdf_max_resolution=scene.sdf_max_resolution,
            sdf_narrow_band_range=scene.sdf_narrow_band_range,
        )

        builder = newton.ModelBuilder()
        builder.default_shape_cfg.gap = 0.001

        builder.add_shape_box(
            body=-1,
            xform=wp.transform(
                wp.vec3(0.0, 0.0, -scene.plate_half_thickness),
                wp.quat_identity(),
            ),
            hx=scene.plate_half_extent,
            hy=scene.plate_half_extent,
            hz=scene.plate_half_thickness,
            cfg=shape_cfg,
            label="plate",
        )

        self.cube_body = builder.add_body(
            xform=wp.transform(
                wp.vec3(0.0, 0.0, scene.cube_half_extent + self.height),
                wp.quat_identity(),
            ),
            label="cube",
        )
        self.cube_shape = builder.add_shape_box(
            body=self.cube_body,
            hx=scene.cube_half_extent,
            hy=scene.cube_half_extent,
            hz=scene.cube_half_extent,
            cfg=shape_cfg,
            label="cube_shape",
        )

        self.model = builder.finalize(device=device)
        self.model.request_contact_attributes("force")

        hydro_cfg = HydroelasticSDF.Config(
            output_contact_surface=True,
            reduce_contacts=mode.reduce_contacts,
            pre_prune_contacts=mode.pre_prune_contacts,
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

    def _read_capacities(self) -> BufferStats:
        hydro = self.collision_pipeline.hydroelastic_sdf
        if hydro is None:
            raise RuntimeError("cube-on-plate impact experiment requires hydroelastic contacts")
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

    def _solver_wrench_on_cube(self) -> tuple[np.ndarray, np.ndarray, int]:
        n_force = int(self.force_contacts.rigid_contact_count.numpy()[0])
        if n_force <= 0:
            return np.zeros(3), np.zeros(3), 0

        body_q = self.state_0.body_q.numpy()
        cube_q = body_q[self.cube_body]
        cube_pos = np.asarray(cube_q[:3], dtype=np.float64)
        forces = self.force_contacts.force.numpy()[:n_force]
        shapes0 = self.force_contacts.rigid_contact_shape0.numpy()[:n_force]
        shapes1 = self.force_contacts.rigid_contact_shape1.numpy()[:n_force]
        points0 = self.force_contacts.rigid_contact_point0.numpy()[:n_force]
        points1 = self.force_contacts.rigid_contact_point1.numpy()[:n_force]

        force_sum = np.zeros(3, dtype=np.float64)
        torque_sum = np.zeros(3, dtype=np.float64)
        cube_force_count = 0
        for contact_idx in range(n_force):
            shape0 = int(shapes0[contact_idx])
            shape1 = int(shapes1[contact_idx])
            force_shape0 = np.asarray(forces[contact_idx, :3], dtype=np.float64)
            torque_shape0 = np.asarray(forces[contact_idx, 3:6], dtype=np.float64)
            if shape0 == self.cube_shape:
                body0 = int(self._shape_body[shape0])
                force = force_shape0
                point_world = self._point_world(body_q, body0, points0[contact_idx])
                torque = torque_shape0 + np.cross(point_world - cube_pos, force)
            elif shape1 == self.cube_shape:
                body1 = int(self._shape_body[shape1])
                force = -force_shape0
                point_world = self._point_world(body_q, body1, points1[contact_idx])
                torque = -torque_shape0 + np.cross(point_world - cube_pos, force)
            else:
                continue
            force_sum += force
            torque_sum += torque
            cube_force_count += 1
        return force_sum, torque_sum, cube_force_count

    def _timeseries_row(
        self,
        *,
        time_s: float,
        frame_stats: BufferStats,
        solver_force: np.ndarray,
        solver_torque: np.ndarray,
        solver_force_count: int,
    ) -> dict[str, object]:
        body_q = self.state_0.body_q.numpy()[self.cube_body]
        body_qd = self.state_0.body_qd.numpy()[self.cube_body]
        signed_clearance = _cube_signed_clearance_m(body_q, self.scene.cube_half_extent)
        return {
            "time_s": time_s,
            "height_m": self.height,
            "mode": self.mode.name,
            "reduce_contacts": self.mode.reduce_contacts,
            "pre_prune_contacts": self.mode.pre_prune_contacts,
            "cube_x_m": float(body_q[0]),
            "cube_y_m": float(body_q[1]),
            "cube_z_m": float(body_q[2]),
            "cube_vz_m_per_s": float(body_qd[2]),
            "cube_tilt_deg": _cube_tilt_deg(body_q),
            "cube_signed_clearance_m": signed_clearance,
            "cube_penetration_depth_m": max(0.0, -signed_clearance),
            "solver_fx_N": float(solver_force[0]),
            "solver_fy_N": float(solver_force[1]),
            "solver_fz_N": float(solver_force[2]),
            "solver_tx_Nm": float(solver_torque[0]),
            "solver_ty_Nm": float(solver_torque[1]),
            "solver_tz_Nm": float(solver_torque[2]),
            "solver_force_count": int(solver_force_count),
            "rigid_contact_count": frame_stats.rigid_contact_count,
            "face_contact_count": frame_stats.face_contact_count,
            "buffer_overflow": frame_stats.overflow,
            "state_invalid": not bool(np.all(np.isfinite(body_q)) and np.all(np.isfinite(body_qd))),
        }

    def simulate_step(self, *, step_dt: float, time_s: float) -> _FrameData:
        self.collision_pipeline.collide(self.state_0, self.contacts)
        frame_stats = self._read_buffer_stats()
        self.state_0.clear_forces()
        self.solver.step(self.state_0, self.state_1, self.control, self.contacts, step_dt)
        self.state_0, self.state_1 = self.state_1, self.state_0
        self.solver.update_contacts(self.force_contacts, self.state_0)
        solver_force, solver_torque, solver_force_count = self._solver_wrench_on_cube()
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


def _first_contact_index(rows: list[dict[str, object]]) -> int | None:
    for idx, row in enumerate(rows):
        if (int(row["solver_force_count"]) > 0 and float(row["solver_fz_N"]) > 1.0e-9) or float(
            row["cube_penetration_depth_m"]
        ) > 0.0:
            return idx
    return None


def _settle_time(rows: list[dict[str, object]], *, start_idx: int, weight: float, step_dt: float) -> float:
    window_count = max(1, int(math.ceil(SETTLE_WINDOW_S / step_dt)))
    for idx in range(start_idx, max(start_idx, len(rows) - window_count + 1)):
        window = rows[idx : idx + window_count]
        if len(window) < window_count:
            break
        if all(
            abs(float(row["solver_fz_N"]) - weight) < SETTLE_FORCE_TOL * weight
            and abs(float(row["cube_vz_m_per_s"])) < SETTLE_VZ_TOL
            for row in window
        ):
            return float(rows[idx]["time_s"])
    return float("nan")


def _rms(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values) / len(values)) if values else float("nan")


def _summary_row(
    *,
    scene: SceneConfig,
    height: float,
    mode: ModeConfig,
    frame_count: int,
    max_stats: BufferStats,
    rows: list[dict[str, object]],
    step_dt: float,
) -> dict[str, object]:
    mass = _cube_mass(scene)
    weight = mass * cube_on_plate.GRAVITY
    first_contact_idx = _first_contact_index(rows)
    contact_rows = rows[first_contact_idx:] if first_contact_idx is not None else []
    first_contact_time = float(rows[first_contact_idx]["time_s"]) if first_contact_idx is not None else float("nan")
    impact_velocity = (
        abs(min(float(row["cube_vz_m_per_s"]) for row in rows[: first_contact_idx + 1]))
        if first_contact_idx is not None
        else float("nan")
    )
    peak_row = max(rows, key=lambda row: float(row["solver_fz_N"]))
    peak_fz = float(peak_row["solver_fz_N"])
    max_rebound_vz = max([float(row["cube_vz_m_per_s"]) for row in contact_rows], default=float("nan"))
    max_upward_rebound = max(0.0, max_rebound_vz) if math.isfinite(max_rebound_vz) else float("nan")
    rebound_ratio = max_upward_rebound / impact_velocity if impact_velocity and impact_velocity > 0.0 else float("nan")
    settle_time = (
        _settle_time(rows, start_idx=first_contact_idx, weight=weight, step_dt=step_dt)
        if first_contact_idx is not None
        else float("nan")
    )
    post_settle_rows = [row for row in rows if math.isfinite(settle_time) and float(row["time_s"]) >= settle_time]
    post_settle_fz_rms = _rms([float(row["solver_fz_N"]) - weight for row in post_settle_rows])
    normal_impulse = sum(max(0.0, float(row["solver_fz_N"])) * step_dt for row in contact_rows)
    final_row = rows[-1]
    final_drift = math.hypot(float(final_row.get("cube_x_m", 0.0)), float(final_row.get("cube_y_m", 0.0)))
    state_invalid = any(bool(row["state_invalid"]) for row in rows)
    valid_run = (
        not max_stats.overflow
        and not state_invalid
        and first_contact_idx is not None
        and peak_fz > 0.0
        and float(final_row["cube_tilt_deg"]) < FINAL_TILT_LIMIT_DEG
        and final_drift < FINAL_DRIFT_LIMIT_M
    )
    return {
        "height_m": height,
        "mode": mode.name,
        "reduce_contacts": mode.reduce_contacts,
        "pre_prune_contacts": mode.pre_prune_contacts,
        "frame_count": frame_count,
        "simulation_time_s": frame_count * step_dt,
        "step_dt_s": step_dt,
        "cube_mass_kg": mass,
        "cube_weight_N": weight,
        "impact_velocity_m_per_s": impact_velocity,
        "first_contact_time_s": first_contact_time,
        "peak_solver_fz_N": peak_fz,
        "peak_solver_fz_over_weight": peak_fz / weight,
        "time_to_peak_fz_s": float(peak_row["time_s"]) - first_contact_time
        if first_contact_idx is not None
        else float("nan"),
        "max_penetration_depth_m": max(float(row["cube_penetration_depth_m"]) for row in rows),
        "max_upward_rebound_velocity_m_per_s": max_upward_rebound,
        "rebound_velocity_ratio": rebound_ratio,
        "settle_time_s": settle_time,
        "post_settle_fz_rms_N": post_settle_fz_rms,
        "normal_impulse_Ns": normal_impulse,
        "final_tilt_deg": float(final_row["cube_tilt_deg"]),
        "final_drift_m": final_drift,
        "mean_solver_force_count": _mean([float(row["solver_force_count"]) for row in rows]),
        "max_rigid_contact_count": max(float(row["rigid_contact_count"]) for row in rows),
        "max_face_contact_count": max(float(row["face_contact_count"]) for row in rows),
        "buffer_overflow": max_stats.overflow,
        "state_invalid": state_invalid,
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
        "max_rigid_contact_count_buffer": max_stats.rigid_contact_count,
        "rigid_contact_capacity": max_stats.rigid_contact_capacity,
        "max_reduction_hashtable_active": max_stats.reduction_hashtable_active,
        "reduction_hashtable_capacity": max_stats.reduction_hashtable_capacity,
        "max_reduction_hashtable_failures": max_stats.reduction_hashtable_failures,
        "valid_run": valid_run,
    }


def _print_buffer_summary(height: float, mode: str, max_stats: BufferStats) -> None:
    print(f"[height={height:.6f} m mode={mode}] buffer summary")
    for label, max_count, capacity, utilization in _summary_stats(max_stats):
        print(f"  {label}: max={max_count} capacity={capacity} utilization={utilization:.4f}")
    if max_stats.reduction_hashtable_failures:
        print(f"  reduction_hashtable_failures: max={max_stats.reduction_hashtable_failures}")


def run_single(
    *,
    scene: SceneConfig,
    height: float,
    mode: ModeConfig,
    simulation_time: float,
    step_dt: float,
    rigid_contact_max: int,
    nconmax: int,
    njmax: int,
    buffer_mult_iso: int,
    buffer_mult_contact: int,
    buffer_fraction: float,
    device: str | None = None,
    verbose: bool = True,
) -> _RunData:
    runner = _ImpactRun(
        scene=scene,
        height=height,
        mode=mode,
        rigid_contact_max=rigid_contact_max,
        nconmax=nconmax,
        njmax=njmax,
        buffer_mult_iso=buffer_mult_iso,
        buffer_mult_contact=buffer_mult_contact,
        buffer_fraction=buffer_fraction,
        device=device,
    )
    frame_count = int(math.ceil(simulation_time / step_dt))
    rows: list[dict[str, object]] = []
    max_stats = _buffer_stats_with_capacities(runner._capacities)
    for frame_idx in range(frame_count):
        frame_data = runner.simulate_step(step_dt=step_dt, time_s=(frame_idx + 1) * step_dt)
        rows.append(frame_data.timeseries_row)
        max_stats.update_max(frame_data.buffer_stats)
    summary = _summary_row(
        scene=scene,
        height=height,
        mode=mode,
        frame_count=len(rows),
        max_stats=max_stats,
        rows=rows,
        step_dt=step_dt,
    )
    if verbose:
        print(
            f"simulated height={height:.6f} m mode={mode.name} "
            f"peak={float(summary['peak_solver_fz_N']):.6g} N valid={summary['valid_run']} ({len(rows)} samples)"
        )
        _print_buffer_summary(height, mode.name, max_stats)
    return _RunData(
        height=height,
        mode=mode.name,
        timeseries_rows=rows,
        summary_row=summary,
        valid_run=bool(summary["valid_run"]),
    )


def run_experiment(
    *,
    config_path: str | Path = cube_on_plate.DEFAULT_CONFIG_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    heights: tuple[float, ...] | list[float] = DEFAULT_DROP_HEIGHTS,
    simulation_time: float = RUN_SECONDS,
    step_dt: float = STEP_DT,
    rigid_contact_max: int = RIGID_CONTACT_MAX,
    nconmax: int = MUJOCO_NCONMAX,
    njmax: int = MUJOCO_NJMAX,
    buffer_mult_iso: int = BUFFER_MULT_ISO,
    buffer_mult_contact: int = BUFFER_MULT_CONTACT,
    buffer_fraction: float = BUFFER_FRACTION,
    device: str | None = None,
    verbose: bool = True,
) -> list[RunResult]:
    if simulation_time <= 0.0:
        raise ValueError(f"simulation_time must be positive; got {simulation_time}")
    if step_dt <= 0.0:
        raise ValueError(f"step_dt must be positive; got {step_dt}")
    if device:
        wp.set_device(device)

    scene = _load_scene_config(config_path)
    output_dir = Path(output_dir)
    timeseries_path = output_dir / TIMESERIES_CSV
    summary_path = output_dir / SUMMARY_CSV
    modes = (
        ModeConfig(name="unreduced", reduce_contacts=False, pre_prune_contacts=False),
        ModeConfig(name="reduced", reduce_contacts=True, pre_prune_contacts=False),
    )
    run_data: list[_RunData] = []
    for height in heights:
        if float(height) <= 0.0:
            raise ValueError(f"drop heights must be positive for impact; got {height}")
        for mode in modes:
            run_data.append(
                run_single(
                    scene=scene,
                    height=float(height),
                    mode=mode,
                    simulation_time=simulation_time,
                    step_dt=step_dt,
                    rigid_contact_max=rigid_contact_max,
                    nconmax=nconmax,
                    njmax=njmax,
                    buffer_mult_iso=buffer_mult_iso,
                    buffer_mult_contact=buffer_mult_contact,
                    buffer_fraction=buffer_fraction,
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
    return [
        RunResult(
            height=data.height,
            mode=data.mode,
            timeseries_path=timeseries_path,
            summary_path=summary_path,
            row_count=len(data.timeseries_rows),
            valid_run=data.valid_run,
        )
        for data in run_data
    ]


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=str, default=str(cube_on_plate.DEFAULT_CONFIG_PATH))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Directory for CSV files.")
    parser.add_argument(
        "--heights", type=float, nargs="+", default=list(DEFAULT_DROP_HEIGHTS), help="Drop heights [m]."
    )
    parser.add_argument("--simulation-time", type=float, default=RUN_SECONDS, help="Simulation time per run [s].")
    parser.add_argument("--step-dt", type=float, default=STEP_DT, help="Solver step and logging interval [s].")
    parser.add_argument("--device", type=str, default=None, help="Override the Warp device.")
    parser.add_argument("--quiet", action="store_true", help="Suppress Warp compilation and per-run summaries.")
    parser.add_argument("--rigid-contact-max", type=int, default=RIGID_CONTACT_MAX)
    parser.add_argument("--nconmax", type=int, default=MUJOCO_NCONMAX)
    parser.add_argument("--njmax", type=int, default=MUJOCO_NJMAX)
    parser.add_argument("--buffer-mult-iso", type=int, default=BUFFER_MULT_ISO)
    parser.add_argument("--buffer-mult-contact", type=int, default=BUFFER_MULT_CONTACT)
    parser.add_argument("--buffer-fraction", type=float, default=BUFFER_FRACTION)
    return parser


def main() -> None:
    args = create_parser().parse_args()
    if args.quiet:
        wp.config.quiet = True
    run_experiment(
        config_path=args.config,
        output_dir=args.output_dir,
        heights=tuple(args.heights),
        simulation_time=args.simulation_time,
        step_dt=args.step_dt,
        rigid_contact_max=args.rigid_contact_max,
        nconmax=args.nconmax,
        njmax=args.njmax,
        buffer_mult_iso=args.buffer_mult_iso,
        buffer_mult_contact=args.buffer_mult_contact,
        buffer_fraction=args.buffer_fraction,
        device=args.device,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
