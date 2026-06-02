# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Cube-on-plate tipping contact-reduction experiment.
#
# Command:
#     python -m newton.examples.contacts.experiment_cube_on_plate_tipping
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

DEFAULT_OUTPUT_DIR = Path("output") / "H3_cube_on_plate_tipping"
DEFAULT_TIP_MU_SLIDING = 0.7
DEFAULT_TIP_INITIAL_OVERLAP = 0.0002
DEFAULT_RAMP_RATE_N_PER_S = 5.0

FRAME_FPS = 60
RUN_SECONDS = 1.15
SIM_SUBSTEPS = 4
TIP_TILT_THRESHOLD_DEG = 10.0
SLIDE_THRESHOLD_M = 0.005
SLIDE_TILT_CEILING_DEG = 5.0

TIMESERIES_CSV = "tipping_timeseries.csv"
SUMMARY_CSV = "tipping_summary.csv"

TIMESERIES_COLUMNS = (
    "time_s",
    "mode",
    "reduce_contacts",
    "pre_prune_contacts",
    "applied_force_N",
    "applied_force_over_ftip",
    "cube_x_m",
    "cube_y_m",
    "cube_z_m",
    "cube_pitch_deg",
    "cube_tilt_deg",
    "cube_penetration_depth_m",
    "center_pressure_x_m",
    "center_pressure_x_over_half_extent",
    "solver_fx_N",
    "solver_fy_N",
    "solver_fz_N",
    "solver_tx_Nm",
    "solver_ty_Nm",
    "solver_tz_Nm",
    "solver_force_count",
    "face_contact_count",
    "rigid_contact_count",
    "buffer_overflow",
    "state_invalid",
)

SUMMARY_COLUMNS = (
    "mode",
    "reduce_contacts",
    "pre_prune_contacts",
    "frame_count",
    "mu_sliding",
    "cube_mass_kg",
    "cube_weight_N",
    "analytic_tip_force_N",
    "analytic_slide_force_N",
    "tip_tilt_threshold_deg",
    "slide_threshold_m",
    "event_type",
    "event_time_s",
    "event_force_N",
    "event_force_over_ftip",
    "tip_time_s",
    "tip_force_N",
    "slide_time_s",
    "slide_force_N",
    "pitch_at_0p25_ftip_deg",
    "pitch_at_0p50_ftip_deg",
    "pitch_at_0p75_ftip_deg",
    "pitch_at_0p90_ftip_deg",
    "cop_x_at_0p50_ftip_m",
    "cop_x_at_0p75_ftip_m",
    "cop_x_at_0p90_ftip_m",
    "final_pitch_deg",
    "final_tilt_deg",
    "final_x_m",
    "final_penetration_depth_m",
    "mean_solver_force_count",
    "mean_rigid_contact_count",
    "mean_face_contact_count",
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
    "max_face_contact_count",
    "face_contact_capacity",
    "max_rigid_contact_count",
    "rigid_contact_capacity",
    "max_reduction_hashtable_active",
    "reduction_hashtable_capacity",
    "max_reduction_hashtable_failures",
)


@dataclass(frozen=True)
class TipSceneConfig:
    """Cube-on-plate scene with tipping-specific settings."""

    base: SceneConfig
    mu_sliding: float
    tip_initial_overlap: float
    ramp_rate: float


@dataclass(frozen=True)
class RunResult:
    """Output summary for one tipping mode."""

    mode: str
    timeseries_path: Path
    summary_path: Path
    row_count: int
    buffer_overflow: bool


@dataclass
class _RunData:
    mode: str
    timeseries_rows: list[dict[str, object]]
    summary_row: dict[str, object]
    buffer_overflow: bool


@dataclass(frozen=True)
class _FrameData:
    timeseries_row: dict[str, object]
    buffer_stats: BufferStats


def _load_tip_scene_config(
    config_path: str | Path,
    *,
    mu_sliding: float,
    tip_initial_overlap: float | None,
    ramp_rate: float | None,
) -> TipSceneConfig:
    base = _load_scene_config(config_path)
    constants, _newton_constants = cube_on_plate._load_config(config_path)
    return TipSceneConfig(
        base=base,
        mu_sliding=float(mu_sliding),
        tip_initial_overlap=(
            float(tip_initial_overlap)
            if tip_initial_overlap is not None
            else float(constants.get("TIP_INITIAL_OVERLAP", DEFAULT_TIP_INITIAL_OVERLAP))
        ),
        ramp_rate=(
            float(ramp_rate)
            if ramp_rate is not None
            else float(constants.get("RAMP_RATE_N_PER_S", DEFAULT_RAMP_RATE_N_PER_S))
        ),
    )


def _cube_mass(scene: TipSceneConfig) -> float:
    side = 2.0 * scene.base.cube_half_extent
    return scene.base.cube_density * side**3


def _cube_pitch_deg(body_q: np.ndarray) -> float:
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
    local_z_x = 2.0 * (qx * qz + qw * qy)
    local_z_z = 1.0 - 2.0 * (qx * qx + qy * qy)
    return math.degrees(math.atan2(local_z_x, local_z_z))


def _center_pressure_x(solver_force: np.ndarray, solver_torque: np.ndarray) -> float:
    fz = float(solver_force[2])
    if abs(fz) <= 1.0e-12:
        return float("nan")
    return -float(solver_torque[1]) / fz


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format_csv_value(row[key]) for key in fieldnames})


class _TipRun:
    def __init__(
        self,
        *,
        scene: TipSceneConfig,
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
        self.mode = mode

        shape_cfg = newton.ModelBuilder.ShapeConfig(
            margin=0.0,
            gap=0.0005,
            mu=scene.mu_sliding,
            mu_torsional=0.0,
            mu_rolling=0.0,
            ke=1.0e7,
            kd=1.0e5,
            density=scene.base.cube_density,
            is_hydroelastic=True,
            kh=scene.base.kh,
            sdf_max_resolution=scene.base.sdf_max_resolution,
            sdf_narrow_band_range=scene.base.sdf_narrow_band_range,
        )

        builder = newton.ModelBuilder()
        builder.default_shape_cfg.gap = 0.001

        builder.add_shape_box(
            body=-1,
            xform=wp.transform(
                wp.vec3(0.0, 0.0, -scene.base.plate_half_thickness),
                wp.quat_identity(),
            ),
            hx=scene.base.plate_half_extent,
            hy=scene.base.plate_half_extent,
            hz=scene.base.plate_half_thickness,
            cfg=shape_cfg,
            label="plate",
        )

        cube_z = scene.base.cube_half_extent - scene.tip_initial_overlap
        self.cube_body = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, cube_z), wp.quat_identity()),
            label="cube",
        )
        self.cube_shape = builder.add_shape_box(
            body=self.cube_body,
            hx=scene.base.cube_half_extent,
            hy=scene.base.cube_half_extent,
            hz=scene.base.cube_half_extent,
            cfg=shape_cfg,
            label="cube_shape",
        )
        self.cube_qd_start = int(builder.joint_qd_start[-1])

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

        joint_f_len = int(self.control.joint_f.shape[0])
        self._wrench_host = np.zeros(joint_f_len, dtype=np.float32)
        self.contacts = self.collision_pipeline.contacts()
        self.force_contacts = self.collision_pipeline.contacts()
        if self.force_contacts.force is None:
            raise RuntimeError("force contact buffer was not allocated")
        self._shape_body = self.model.shape_body.numpy()
        self._capacities = self._read_capacities()

    def _read_capacities(self) -> BufferStats:
        hydro = self.collision_pipeline.hydroelastic_sdf
        if hydro is None:
            raise RuntimeError("cube-on-plate tipping experiment requires hydroelastic contacts")
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

    def _apply_top_force(self, force_N: float) -> None:
        self._wrench_host[:] = 0.0
        self._wrench_host[self.cube_qd_start + 0] = force_N
        self._wrench_host[self.cube_qd_start + 4] = force_N * self.scene.base.cube_half_extent
        self.control.joint_f.assign(self._wrench_host)

    def _timeseries_row(
        self,
        *,
        time_s: float,
        applied_force: float,
        frame_stats: BufferStats,
        solver_force: np.ndarray,
        solver_torque: np.ndarray,
        solver_force_count: int,
    ) -> dict[str, object]:
        body_q = self.state_0.body_q.numpy()[self.cube_body]
        body_qd = self.state_0.body_qd.numpy()[self.cube_body]
        signed_clearance = _cube_signed_clearance_m(body_q, self.scene.base.cube_half_extent)
        mass = _cube_mass(self.scene)
        f_tip = 0.5 * mass * cube_on_plate.GRAVITY
        cop_x = _center_pressure_x(solver_force, solver_torque)
        return {
            "time_s": time_s,
            "mode": self.mode.name,
            "reduce_contacts": self.mode.reduce_contacts,
            "pre_prune_contacts": self.mode.pre_prune_contacts,
            "applied_force_N": applied_force,
            "applied_force_over_ftip": applied_force / f_tip,
            "cube_x_m": float(body_q[0]),
            "cube_y_m": float(body_q[1]),
            "cube_z_m": float(body_q[2]),
            "cube_pitch_deg": _cube_pitch_deg(body_q),
            "cube_tilt_deg": _cube_tilt_deg(body_q),
            "cube_penetration_depth_m": max(0.0, -signed_clearance),
            "center_pressure_x_m": cop_x,
            "center_pressure_x_over_half_extent": cop_x / self.scene.base.cube_half_extent,
            "solver_fx_N": float(solver_force[0]),
            "solver_fy_N": float(solver_force[1]),
            "solver_fz_N": float(solver_force[2]),
            "solver_tx_Nm": float(solver_torque[0]),
            "solver_ty_Nm": float(solver_torque[1]),
            "solver_tz_Nm": float(solver_torque[2]),
            "solver_force_count": int(solver_force_count),
            "face_contact_count": frame_stats.face_contact_count,
            "rigid_contact_count": frame_stats.rigid_contact_count,
            "buffer_overflow": frame_stats.overflow,
            "state_invalid": not bool(np.all(np.isfinite(body_q)) and np.all(np.isfinite(body_qd))),
        }

    def simulate_frame(self, *, frame_dt: float, sim_substeps: int, frame_start_time: float) -> _FrameData:
        sim_dt = frame_dt / sim_substeps
        frame_stats = _buffer_stats_with_capacities(self._capacities)
        solver_force = np.zeros(3, dtype=np.float64)
        solver_torque = np.zeros(3, dtype=np.float64)
        solver_force_count = 0
        applied_force = 0.0
        for substep in range(sim_substeps):
            time_s = frame_start_time + substep * sim_dt
            applied_force = self.scene.ramp_rate * time_s
            self._apply_top_force(applied_force)
            self.collision_pipeline.collide(self.state_0, self.contacts)
            frame_stats.update_max(self._read_buffer_stats())
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.solver.update_contacts(self.force_contacts, self.state_0)
            solver_force, solver_torque, solver_force_count = self._solver_wrench_on_cube()

        time_s = frame_start_time + frame_dt
        return _FrameData(
            timeseries_row=self._timeseries_row(
                time_s=time_s,
                applied_force=self.scene.ramp_rate * time_s,
                frame_stats=frame_stats,
                solver_force=solver_force,
                solver_torque=solver_torque,
                solver_force_count=solver_force_count,
            ),
            buffer_stats=frame_stats,
        )


def _event_row(rows: list[dict[str, object]], *, kind: str) -> dict[str, object] | None:
    for row in rows:
        tilt = abs(float(row["cube_tilt_deg"]))
        x = abs(float(row["cube_x_m"]))
        if kind == "tip" and tilt >= TIP_TILT_THRESHOLD_DEG:
            return row
        if kind == "slide" and x >= SLIDE_THRESHOLD_M and tilt < SLIDE_TILT_CEILING_DEG:
            return row
    return None


def _nearest_by_force_ratio(rows: list[dict[str, object]], ratio: float) -> dict[str, object]:
    return min(rows, key=lambda row: abs(float(row["applied_force_over_ftip"]) - ratio))


def _summary_row(
    *,
    scene: TipSceneConfig,
    mode: ModeConfig,
    frame_count: int,
    max_stats: BufferStats,
    rows: list[dict[str, object]],
) -> dict[str, object]:
    mass = _cube_mass(scene)
    weight = mass * cube_on_plate.GRAVITY
    f_tip = 0.5 * weight
    f_slide = scene.mu_sliding * weight
    tip_row = _event_row(rows, kind="tip")
    slide_row = _event_row(rows, kind="slide")
    if tip_row is None and slide_row is None:
        event_type = "none"
        event_row = None
    elif slide_row is None or (tip_row is not None and float(tip_row["time_s"]) <= float(slide_row["time_s"])):
        event_type = "tip"
        event_row = tip_row
    else:
        event_type = "slide"
        event_row = slide_row
    final_row = rows[-1]
    state_invalid = any(bool(row["state_invalid"]) for row in rows)
    p25 = _nearest_by_force_ratio(rows, 0.25)
    p50 = _nearest_by_force_ratio(rows, 0.50)
    p75 = _nearest_by_force_ratio(rows, 0.75)
    p90 = _nearest_by_force_ratio(rows, 0.90)
    return {
        "mode": mode.name,
        "reduce_contacts": mode.reduce_contacts,
        "pre_prune_contacts": mode.pre_prune_contacts,
        "frame_count": frame_count,
        "mu_sliding": scene.mu_sliding,
        "cube_mass_kg": mass,
        "cube_weight_N": weight,
        "analytic_tip_force_N": f_tip,
        "analytic_slide_force_N": f_slide,
        "tip_tilt_threshold_deg": TIP_TILT_THRESHOLD_DEG,
        "slide_threshold_m": SLIDE_THRESHOLD_M,
        "event_type": event_type,
        "event_time_s": float(event_row["time_s"]) if event_row is not None else float("nan"),
        "event_force_N": float(event_row["applied_force_N"]) if event_row is not None else float("nan"),
        "event_force_over_ftip": float(event_row["applied_force_over_ftip"]) if event_row is not None else float("nan"),
        "tip_time_s": float(tip_row["time_s"]) if tip_row is not None else float("nan"),
        "tip_force_N": float(tip_row["applied_force_N"]) if tip_row is not None else float("nan"),
        "slide_time_s": float(slide_row["time_s"]) if slide_row is not None else float("nan"),
        "slide_force_N": float(slide_row["applied_force_N"]) if slide_row is not None else float("nan"),
        "pitch_at_0p25_ftip_deg": float(p25["cube_pitch_deg"]),
        "pitch_at_0p50_ftip_deg": float(p50["cube_pitch_deg"]),
        "pitch_at_0p75_ftip_deg": float(p75["cube_pitch_deg"]),
        "pitch_at_0p90_ftip_deg": float(p90["cube_pitch_deg"]),
        "cop_x_at_0p50_ftip_m": float(p50["center_pressure_x_m"]),
        "cop_x_at_0p75_ftip_m": float(p75["center_pressure_x_m"]),
        "cop_x_at_0p90_ftip_m": float(p90["center_pressure_x_m"]),
        "final_pitch_deg": float(final_row["cube_pitch_deg"]),
        "final_tilt_deg": float(final_row["cube_tilt_deg"]),
        "final_x_m": float(final_row["cube_x_m"]),
        "final_penetration_depth_m": float(final_row["cube_penetration_depth_m"]),
        "mean_solver_force_count": _mean([float(row["solver_force_count"]) for row in rows]),
        "mean_rigid_contact_count": _mean([float(row["rigid_contact_count"]) for row in rows]),
        "mean_face_contact_count": _mean([float(row["face_contact_count"]) for row in rows]),
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
        "max_face_contact_count": max_stats.face_contact_count,
        "face_contact_capacity": max_stats.face_contact_capacity,
        "max_rigid_contact_count": max_stats.rigid_contact_count,
        "rigid_contact_capacity": max_stats.rigid_contact_capacity,
        "max_reduction_hashtable_active": max_stats.reduction_hashtable_active,
        "reduction_hashtable_capacity": max_stats.reduction_hashtable_capacity,
        "max_reduction_hashtable_failures": max_stats.reduction_hashtable_failures,
    }


def _print_buffer_summary(mode: str, max_stats: BufferStats) -> None:
    print(f"[mode={mode}] buffer summary")
    for label, max_count, capacity, utilization in _summary_stats(max_stats):
        print(f"  {label}: max={max_count} capacity={capacity} utilization={utilization:.4f}")
    if max_stats.reduction_hashtable_failures:
        print(f"  reduction_hashtable_failures: max={max_stats.reduction_hashtable_failures}")


def run_single(
    *,
    scene: TipSceneConfig,
    mode: ModeConfig,
    simulation_time: float,
    frame_fps: int,
    sim_substeps: int,
    rigid_contact_max: int,
    nconmax: int,
    njmax: int,
    buffer_mult_iso: int,
    buffer_mult_contact: int,
    buffer_fraction: float,
    device: str | None = None,
    verbose: bool = True,
) -> _RunData:
    runner = _TipRun(
        scene=scene,
        mode=mode,
        rigid_contact_max=rigid_contact_max,
        nconmax=nconmax,
        njmax=njmax,
        buffer_mult_iso=buffer_mult_iso,
        buffer_mult_contact=buffer_mult_contact,
        buffer_fraction=buffer_fraction,
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
            frame_start_time=frame_idx * frame_dt,
        )
        rows.append(frame_data.timeseries_row)
        max_stats.update_max(frame_data.buffer_stats)
    summary = _summary_row(scene=scene, mode=mode, frame_count=len(rows), max_stats=max_stats, rows=rows)
    if verbose:
        print(
            f"simulated mode={mode.name} event={summary['event_type']} "
            f"force={float(summary['event_force_N']):.6g} N ({len(rows)} samples)"
        )
        _print_buffer_summary(mode.name, max_stats)
    if max_stats.overflow:
        raise RuntimeError(f"contact-buffer validity gate failed for mode={mode.name}")
    if bool(summary["state_invalid"]):
        raise RuntimeError(f"state-validity gate failed for mode={mode.name}")
    return _RunData(mode=mode.name, timeseries_rows=rows, summary_row=summary, buffer_overflow=max_stats.overflow)


def run_experiment(
    *,
    config_path: str | Path = cube_on_plate.DEFAULT_CONFIG_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    simulation_time: float = RUN_SECONDS,
    frame_fps: int = FRAME_FPS,
    sim_substeps: int = SIM_SUBSTEPS,
    mu_sliding: float = DEFAULT_TIP_MU_SLIDING,
    tip_initial_overlap: float | None = None,
    ramp_rate: float | None = None,
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
    if frame_fps <= 0:
        raise ValueError(f"frame_fps must be positive; got {frame_fps}")
    if sim_substeps <= 0:
        raise ValueError(f"sim_substeps must be positive; got {sim_substeps}")
    if mu_sliding <= 0.0:
        raise ValueError(f"mu_sliding must be positive; got {mu_sliding}")
    if device:
        wp.set_device(device)

    scene = _load_tip_scene_config(
        config_path,
        mu_sliding=mu_sliding,
        tip_initial_overlap=tip_initial_overlap,
        ramp_rate=ramp_rate,
    )
    output_dir = Path(output_dir)
    timeseries_path = output_dir / TIMESERIES_CSV
    summary_path = output_dir / SUMMARY_CSV
    modes = (
        ModeConfig(name="unreduced", reduce_contacts=False),
        ModeConfig(name="reduced", reduce_contacts=True, pre_prune_contacts=False),
    )
    run_data = [
        run_single(
            scene=scene,
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
            device=device,
            verbose=verbose,
        )
        for mode in modes
    ]
    timeseries_rows = [row for data in run_data for row in data.timeseries_rows]
    summary_rows = [data.summary_row for data in run_data]
    _write_csv(timeseries_path, timeseries_rows, TIMESERIES_COLUMNS)
    _write_csv(summary_path, summary_rows, SUMMARY_COLUMNS)
    if verbose:
        print(f"wrote {timeseries_path} ({len(timeseries_rows)} samples)")
        print(f"wrote {summary_path} ({len(summary_rows)} runs)")
    return [
        RunResult(
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
    parser.add_argument("--config", type=str, default=str(cube_on_plate.DEFAULT_CONFIG_PATH))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Directory for CSV files.")
    parser.add_argument("--simulation-time", type=float, default=RUN_SECONDS, help="Simulation time per run [s].")
    parser.add_argument("--fps", type=int, default=FRAME_FPS, help="Logged frame rate [Hz].")
    parser.add_argument("--substeps", type=int, default=SIM_SUBSTEPS, help="Solver substeps per logged frame.")
    parser.add_argument(
        "--mu-sliding",
        type=float,
        default=DEFAULT_TIP_MU_SLIDING,
        help="Sliding friction used for the tipping experiment.",
    )
    parser.add_argument("--tip-initial-overlap", type=float, default=None, help="Initial cube penetration [m].")
    parser.add_argument("--ramp-rate", type=float, default=None, help="Applied top-force ramp rate [N/s].")
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
        simulation_time=args.simulation_time,
        frame_fps=args.fps,
        sim_substeps=args.substeps,
        mu_sliding=args.mu_sliding,
        tip_initial_overlap=args.tip_initial_overlap,
        ramp_rate=args.ramp_rate,
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
