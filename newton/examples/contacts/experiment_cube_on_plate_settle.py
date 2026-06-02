# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Cube-on-plate settle contact-reduction experiment.
#
# Command:
#     python -m newton.examples.contacts.experiment_cube_on_plate_settle
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
from newton.geometry import HydroelasticSDF

DEFAULT_OUTPUT_DIR = Path("output") / "H1_cube_on_plate_settle"
DEFAULT_DROP_HEIGHTS = (0.0, 0.00025, 0.0005, 0.001, 0.0025, 0.005)

FRAME_FPS = 240
RUN_SECONDS = 1.0
SIM_SUBSTEPS = 8

RIGID_CONTACT_MAX = 131072
MUJOCO_NCONMAX = 131072
MUJOCO_NJMAX = 262144
BUFFER_MULT_ISO = 8
BUFFER_MULT_CONTACT = 16
BUFFER_FRACTION = 1.0

SETTLE_FINAL_TILT_DEG = 0.5
SETTLE_FINAL_DRIFT_M = 1.0e-4

TIMESERIES_CSV = "settle_timeseries.csv"
SUMMARY_CSV = "settle_summary.csv"
DEBUG_BUFFERS_CSV = "settle_debug_buffers.csv"

TIMESERIES_COLUMNS = (
    "time_s",
    "height_m",
    "mode",
    "reduce_contacts",
    "pre_prune_contacts",
    "rigid_contact_count",
    "cube_x_m",
    "cube_y_m",
    "cube_z_m",
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
    "buffer_overflow",
)

SUMMARY_COLUMNS = (
    "height_m",
    "mode",
    "reduce_contacts",
    "pre_prune_contacts",
    "frame_count",
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
    "final_cube_x_m",
    "final_cube_y_m",
    "final_cube_z_m",
    "final_cube_tilt_deg",
    "final_cube_signed_clearance_m",
    "final_cube_penetration_depth_m",
)

DEBUG_BUFFER_COLUMNS = (
    "time_s",
    "height_m",
    "mode",
    "reduce_contacts",
    "pre_prune_contacts",
    "hydro_broadphase_blocks",
    "hydro_broadphase_capacity",
    "hydro_broadphase_utilization",
    "hydro_iso_subblocks_l0",
    "hydro_iso_subblocks_l0_capacity",
    "hydro_iso_subblocks_l0_utilization",
    "hydro_iso_subblocks_l1",
    "hydro_iso_subblocks_l1_capacity",
    "hydro_iso_subblocks_l1_utilization",
    "hydro_iso_subblocks_l2",
    "hydro_iso_subblocks_l2_capacity",
    "hydro_iso_subblocks_l2_utilization",
    "hydro_iso_voxels",
    "hydro_iso_voxels_capacity",
    "hydro_iso_voxels_utilization",
    "face_contact_count",
    "face_contact_capacity",
    "face_contact_utilization",
    "rigid_contact_count",
    "rigid_contact_capacity",
    "rigid_contact_utilization",
    "reduction_hashtable_active",
    "reduction_hashtable_capacity",
    "reduction_hashtable_utilization",
    "reduction_hashtable_failures",
    "buffer_overflow",
)


@dataclass(frozen=True)
class SceneConfig:
    """Scene constants loaded from the shared cube-on-plate baseline config."""

    cube_half_extent: float
    cube_density: float
    plate_half_extent: float
    plate_half_thickness: float
    mu_sliding: float
    sdf_max_resolution: int
    sdf_narrow_band_range: tuple[float, float]
    kh: float


@dataclass(frozen=True)
class ModeConfig:
    """Contact-reduction mode for one experiment run."""

    name: str
    reduce_contacts: bool
    pre_prune_contacts: bool = False


@dataclass
class BufferStats:
    """Hydroelastic and solver contact buffer counts for one logged frame."""

    hydro_broadphase_blocks: int = 0
    hydro_iso_subblocks_l0: int = 0
    hydro_iso_subblocks_l1: int = 0
    hydro_iso_subblocks_l2: int = 0
    hydro_iso_voxels: int = 0
    face_contact_count: int = 0
    rigid_contact_count: int = 0
    reduction_hashtable_active: int = 0
    reduction_hashtable_failures: int = 0
    hydro_broadphase_capacity: int = 0
    hydro_iso_subblocks_l0_capacity: int = 0
    hydro_iso_subblocks_l1_capacity: int = 0
    hydro_iso_subblocks_l2_capacity: int = 0
    hydro_iso_voxels_capacity: int = 0
    face_contact_capacity: int = 0
    rigid_contact_capacity: int = 0
    reduction_hashtable_capacity: int = 0

    def update_max(self, other: BufferStats) -> None:
        self.hydro_broadphase_blocks = max(self.hydro_broadphase_blocks, other.hydro_broadphase_blocks)
        self.hydro_iso_subblocks_l0 = max(self.hydro_iso_subblocks_l0, other.hydro_iso_subblocks_l0)
        self.hydro_iso_subblocks_l1 = max(self.hydro_iso_subblocks_l1, other.hydro_iso_subblocks_l1)
        self.hydro_iso_subblocks_l2 = max(self.hydro_iso_subblocks_l2, other.hydro_iso_subblocks_l2)
        self.hydro_iso_voxels = max(self.hydro_iso_voxels, other.hydro_iso_voxels)
        self.face_contact_count = max(self.face_contact_count, other.face_contact_count)
        self.rigid_contact_count = max(self.rigid_contact_count, other.rigid_contact_count)
        self.reduction_hashtable_active = max(
            self.reduction_hashtable_active,
            other.reduction_hashtable_active,
        )
        self.reduction_hashtable_failures = max(
            self.reduction_hashtable_failures,
            other.reduction_hashtable_failures,
        )

    @property
    def overflow(self) -> bool:
        return (
            self.hydro_broadphase_blocks > self.hydro_broadphase_capacity
            or self.hydro_iso_subblocks_l0 > self.hydro_iso_subblocks_l0_capacity
            or self.hydro_iso_subblocks_l1 > self.hydro_iso_subblocks_l1_capacity
            or self.hydro_iso_subblocks_l2 > self.hydro_iso_subblocks_l2_capacity
            or self.hydro_iso_voxels > self.hydro_iso_voxels_capacity
            or self.face_contact_count > self.face_contact_capacity
            or self.rigid_contact_count > self.rigid_contact_capacity
            or self.reduction_hashtable_failures > 0
        )


@dataclass(frozen=True)
class RunResult:
    """Output summary for one height/mode run."""

    height: float
    mode: str
    timeseries_path: Path
    summary_path: Path
    debug_buffers_path: Path | None
    row_count: int
    buffer_overflow: bool


@dataclass
class _RunData:
    height: float
    mode: str
    timeseries_rows: list[dict[str, object]]
    summary_row: dict[str, object]
    debug_buffer_rows: list[dict[str, object]]
    buffer_overflow: bool


@dataclass
class _FrameData:
    timeseries_row: dict[str, object]
    debug_buffer_row: dict[str, object] | None
    buffer_stats: BufferStats


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


def _utilization(count: int, capacity: int) -> float:
    return float(count) / float(capacity) if capacity > 0 else 0.0


def _buffer_stats_with_capacities(capacities: BufferStats) -> BufferStats:
    return BufferStats(
        hydro_broadphase_capacity=capacities.hydro_broadphase_capacity,
        hydro_iso_subblocks_l0_capacity=capacities.hydro_iso_subblocks_l0_capacity,
        hydro_iso_subblocks_l1_capacity=capacities.hydro_iso_subblocks_l1_capacity,
        hydro_iso_subblocks_l2_capacity=capacities.hydro_iso_subblocks_l2_capacity,
        hydro_iso_voxels_capacity=capacities.hydro_iso_voxels_capacity,
        face_contact_capacity=capacities.face_contact_capacity,
        rigid_contact_capacity=capacities.rigid_contact_capacity,
        reduction_hashtable_capacity=capacities.reduction_hashtable_capacity,
    )


def _cube_tilt_deg(body_q: np.ndarray) -> float:
    qx = float(body_q[3])
    qy = float(body_q[4])
    local_z_z = max(-1.0, min(1.0, 1.0 - 2.0 * (qx * qx + qy * qy)))
    return math.degrees(math.acos(local_z_z))


def _cube_signed_clearance_m(body_q: np.ndarray, cube_half_extent: float) -> float:
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

    # The plate top is z=0. Project the cube's rotated half-extents onto the
    # world z axis to get the bottom-most point even when the cube is tilted.
    z_axis_x = 2.0 * (qx * qz - qy * qw)
    z_axis_y = 2.0 * (qy * qz + qx * qw)
    z_axis_z = 1.0 - 2.0 * (qx * qx + qy * qy)
    bottom_z = float(body_q[2]) - cube_half_extent * (abs(z_axis_x) + abs(z_axis_y) + abs(z_axis_z))
    return bottom_z


class _SettleRun:
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
        self.rigid_contact_max = int(rigid_contact_max)
        self.nconmax = int(nconmax)
        self.njmax = int(njmax)

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
            rigid_contact_max=self.rigid_contact_max,
            broad_phase="sap",
        )

        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            use_mujoco_contacts=False,
            solver="newton",
            integrator="implicitfast",
            cone="elliptic",
            njmax=self.njmax,
            nconmax=self.nconmax,
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
            raise RuntimeError("cube-on-plate experiment requires hydroelastic contacts")
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
        stats = BufferStats(
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
        return stats

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

        force_sum = np.zeros(3, dtype=np.float64)
        torque_sum = np.zeros(3, dtype=np.float64)
        cube_force_count = 0
        for contact_idx in range(n_force):
            shape0 = int(shapes0[contact_idx])
            shape1 = int(shapes1[contact_idx])
            force_shape0 = np.asarray(forces[contact_idx, :3], dtype=np.float64)
            torque_shape0 = np.asarray(forces[contact_idx, 3:6], dtype=np.float64)
            if shape0 == self.cube_shape:
                force = force_shape0
                torque = torque_shape0
            elif shape1 == self.cube_shape:
                body0 = int(self._shape_body[shape0])
                body0_pos = np.asarray(body_q[body0, :3], dtype=np.float64) if body0 >= 0 else np.zeros(3)
                force = -force_shape0
                torque = -torque_shape0 + np.cross(cube_pos - body0_pos, force_shape0)
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
        signed_clearance = _cube_signed_clearance_m(body_q, self.scene.cube_half_extent)
        return {
            "time_s": time_s,
            "height_m": self.height,
            "mode": self.mode.name,
            "reduce_contacts": self.mode.reduce_contacts,
            "pre_prune_contacts": self.mode.pre_prune_contacts,
            "rigid_contact_count": frame_stats.rigid_contact_count,
            "cube_x_m": float(body_q[0]),
            "cube_y_m": float(body_q[1]),
            "cube_z_m": float(body_q[2]),
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
            "buffer_overflow": frame_stats.overflow,
        }

    def _debug_buffer_row(self, *, time_s: float, frame_stats: BufferStats) -> dict[str, object]:
        return {
            "time_s": time_s,
            "height_m": self.height,
            "mode": self.mode.name,
            "reduce_contacts": self.mode.reduce_contacts,
            "pre_prune_contacts": self.mode.pre_prune_contacts,
            "hydro_broadphase_blocks": frame_stats.hydro_broadphase_blocks,
            "hydro_broadphase_capacity": frame_stats.hydro_broadphase_capacity,
            "hydro_broadphase_utilization": _utilization(
                frame_stats.hydro_broadphase_blocks,
                frame_stats.hydro_broadphase_capacity,
            ),
            "hydro_iso_subblocks_l0": frame_stats.hydro_iso_subblocks_l0,
            "hydro_iso_subblocks_l0_capacity": frame_stats.hydro_iso_subblocks_l0_capacity,
            "hydro_iso_subblocks_l0_utilization": _utilization(
                frame_stats.hydro_iso_subblocks_l0,
                frame_stats.hydro_iso_subblocks_l0_capacity,
            ),
            "hydro_iso_subblocks_l1": frame_stats.hydro_iso_subblocks_l1,
            "hydro_iso_subblocks_l1_capacity": frame_stats.hydro_iso_subblocks_l1_capacity,
            "hydro_iso_subblocks_l1_utilization": _utilization(
                frame_stats.hydro_iso_subblocks_l1,
                frame_stats.hydro_iso_subblocks_l1_capacity,
            ),
            "hydro_iso_subblocks_l2": frame_stats.hydro_iso_subblocks_l2,
            "hydro_iso_subblocks_l2_capacity": frame_stats.hydro_iso_subblocks_l2_capacity,
            "hydro_iso_subblocks_l2_utilization": _utilization(
                frame_stats.hydro_iso_subblocks_l2,
                frame_stats.hydro_iso_subblocks_l2_capacity,
            ),
            "hydro_iso_voxels": frame_stats.hydro_iso_voxels,
            "hydro_iso_voxels_capacity": frame_stats.hydro_iso_voxels_capacity,
            "hydro_iso_voxels_utilization": _utilization(
                frame_stats.hydro_iso_voxels,
                frame_stats.hydro_iso_voxels_capacity,
            ),
            "face_contact_count": frame_stats.face_contact_count,
            "face_contact_capacity": frame_stats.face_contact_capacity,
            "face_contact_utilization": _utilization(frame_stats.face_contact_count, frame_stats.face_contact_capacity),
            "rigid_contact_count": frame_stats.rigid_contact_count,
            "rigid_contact_capacity": frame_stats.rigid_contact_capacity,
            "rigid_contact_utilization": _utilization(
                frame_stats.rigid_contact_count,
                frame_stats.rigid_contact_capacity,
            ),
            "reduction_hashtable_active": frame_stats.reduction_hashtable_active,
            "reduction_hashtable_capacity": frame_stats.reduction_hashtable_capacity,
            "reduction_hashtable_utilization": _utilization(
                frame_stats.reduction_hashtable_active,
                frame_stats.reduction_hashtable_capacity,
            ),
            "reduction_hashtable_failures": frame_stats.reduction_hashtable_failures,
            "buffer_overflow": frame_stats.overflow,
        }

    def simulate_frame(self, *, frame_dt: float, sim_substeps: int, time_s: float, debug_buffers: bool) -> _FrameData:
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
            solver_force, solver_torque, solver_force_count = self._solver_wrench_on_cube()

        return _FrameData(
            timeseries_row=self._timeseries_row(
                time_s=time_s,
                frame_stats=frame_stats,
                solver_force=solver_force,
                solver_torque=solver_torque,
                solver_force_count=solver_force_count,
            ),
            debug_buffer_row=self._debug_buffer_row(time_s=time_s, frame_stats=frame_stats) if debug_buffers else None,
            buffer_stats=frame_stats,
        )


def _format_csv_value(value: object) -> object:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.10g}"
    return value


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format_csv_value(row[key]) for key in fieldnames})


def _summary_row(
    *,
    height: float,
    mode: ModeConfig,
    frame_count: int,
    max_stats: BufferStats,
    final_row: dict[str, object],
) -> dict[str, object]:
    return {
        "height_m": height,
        "mode": mode.name,
        "reduce_contacts": mode.reduce_contacts,
        "pre_prune_contacts": mode.pre_prune_contacts,
        "frame_count": frame_count,
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
        "final_cube_x_m": final_row["cube_x_m"],
        "final_cube_y_m": final_row["cube_y_m"],
        "final_cube_z_m": final_row["cube_z_m"],
        "final_cube_tilt_deg": final_row["cube_tilt_deg"],
        "final_cube_signed_clearance_m": final_row["cube_signed_clearance_m"],
        "final_cube_penetration_depth_m": final_row["cube_penetration_depth_m"],
    }


def _summary_stats(max_stats: BufferStats) -> list[tuple[str, int, int, float]]:
    stages = (
        ("broadphase_blocks", max_stats.hydro_broadphase_blocks, max_stats.hydro_broadphase_capacity),
        ("iso_subblocks_l0", max_stats.hydro_iso_subblocks_l0, max_stats.hydro_iso_subblocks_l0_capacity),
        ("iso_subblocks_l1", max_stats.hydro_iso_subblocks_l1, max_stats.hydro_iso_subblocks_l1_capacity),
        ("iso_subblocks_l2", max_stats.hydro_iso_subblocks_l2, max_stats.hydro_iso_subblocks_l2_capacity),
        ("iso_voxels", max_stats.hydro_iso_voxels, max_stats.hydro_iso_voxels_capacity),
        ("face_contacts", max_stats.face_contact_count, max_stats.face_contact_capacity),
        ("rigid_contacts", max_stats.rigid_contact_count, max_stats.rigid_contact_capacity),
        ("reduction_hashtable", max_stats.reduction_hashtable_active, max_stats.reduction_hashtable_capacity),
    )
    summary = []
    for label, max_count, capacity in stages:
        summary.append((label, max_count, capacity, _utilization(max_count, capacity)))
    return summary


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
    frame_fps: int,
    sim_substeps: int,
    rigid_contact_max: int,
    nconmax: int,
    njmax: int,
    buffer_mult_iso: int,
    buffer_mult_contact: int,
    buffer_fraction: float,
    debug_buffers: bool,
    device: str | None = None,
    verbose: bool = True,
) -> _RunData:
    runner = _SettleRun(
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

    frame_count = int(math.ceil(simulation_time * frame_fps))
    frame_dt = 1.0 / frame_fps
    timeseries_rows: list[dict[str, object]] = []
    debug_buffer_rows: list[dict[str, object]] = []
    max_stats = _buffer_stats_with_capacities(runner._capacities)
    for frame_idx in range(frame_count):
        frame_data = runner.simulate_frame(
            frame_dt=frame_dt,
            sim_substeps=sim_substeps,
            time_s=(frame_idx + 1) * frame_dt,
            debug_buffers=debug_buffers,
        )
        timeseries_rows.append(frame_data.timeseries_row)
        if frame_data.debug_buffer_row is not None:
            debug_buffer_rows.append(frame_data.debug_buffer_row)
        max_stats.update_max(frame_data.buffer_stats)

    buffer_overflow = max_stats.overflow
    summary_row = _summary_row(
        height=height,
        mode=mode,
        frame_count=len(timeseries_rows),
        max_stats=max_stats,
        final_row=timeseries_rows[-1],
    )
    if verbose:
        print(f"simulated height={height:.6f} m mode={mode.name} ({len(timeseries_rows)} samples)")
        _print_buffer_summary(height, mode.name, max_stats)

    if mode.name == "unreduced" and buffer_overflow:
        raise RuntimeError(f"unreduced run overflowed a buffer at height={height:.6f} m; CSV output is not valid")

    return _RunData(
        height=height,
        mode=mode.name,
        timeseries_rows=timeseries_rows,
        summary_row=summary_row,
        debug_buffer_rows=debug_buffer_rows,
        buffer_overflow=buffer_overflow,
    )


def run_experiment(
    *,
    config_path: str | Path = cube_on_plate.DEFAULT_CONFIG_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    heights: tuple[float, ...] | list[float] = DEFAULT_DROP_HEIGHTS,
    simulation_time: float = RUN_SECONDS,
    frame_fps: int = FRAME_FPS,
    sim_substeps: int = SIM_SUBSTEPS,
    rigid_contact_max: int = RIGID_CONTACT_MAX,
    nconmax: int = MUJOCO_NCONMAX,
    njmax: int = MUJOCO_NJMAX,
    buffer_mult_iso: int = BUFFER_MULT_ISO,
    buffer_mult_contact: int = BUFFER_MULT_CONTACT,
    buffer_fraction: float = BUFFER_FRACTION,
    debug_buffers: bool = False,
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

    scene = _load_scene_config(config_path)
    output_dir = Path(output_dir)
    timeseries_path = output_dir / TIMESERIES_CSV
    summary_path = output_dir / SUMMARY_CSV
    debug_buffers_path = output_dir / DEBUG_BUFFERS_CSV if debug_buffers else None
    modes = (
        ModeConfig(name="unreduced", reduce_contacts=False),
        ModeConfig(name="reduced", reduce_contacts=True),
    )
    run_data: list[_RunData] = []
    for height in heights:
        if float(height) < 0.0:
            raise ValueError(f"drop heights must be nonnegative; got {height}")
        for mode in modes:
            run_data.append(
                run_single(
                    scene=scene,
                    height=float(height),
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
                    debug_buffers=debug_buffers,
                    device=device,
                    verbose=verbose,
                )
            )

    timeseries_rows = [row for data in run_data for row in data.timeseries_rows]
    summary_rows = [data.summary_row for data in run_data]
    _write_csv(timeseries_path, timeseries_rows, TIMESERIES_COLUMNS)
    _write_csv(summary_path, summary_rows, SUMMARY_COLUMNS)
    if debug_buffers_path is not None:
        debug_rows = [row for data in run_data for row in data.debug_buffer_rows]
        _write_csv(debug_buffers_path, debug_rows, DEBUG_BUFFER_COLUMNS)

    if verbose:
        print(f"wrote {timeseries_path} ({len(timeseries_rows)} samples)")
        print(f"wrote {summary_path} ({len(summary_rows)} runs)")
        if debug_buffers_path is not None:
            print(f"wrote {debug_buffers_path}")

    results = [
        RunResult(
            height=data.height,
            mode=data.mode,
            timeseries_path=timeseries_path,
            summary_path=summary_path,
            debug_buffers_path=debug_buffers_path,
            row_count=len(data.timeseries_rows),
            buffer_overflow=data.buffer_overflow,
        )
        for data in run_data
    ]
    return results


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--config",
        type=str,
        default=str(cube_on_plate.DEFAULT_CONFIG_PATH),
        help="Path to the shared cube-on-plate YAML config.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for per-height/per-mode CSV files.",
    )
    parser.add_argument(
        "--heights",
        type=float,
        nargs="+",
        default=list(DEFAULT_DROP_HEIGHTS),
        help="Drop heights [m] to sweep.",
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
        "--debug-buffers",
        action="store_true",
        help=f"Write detailed per-frame buffer counters to {DEBUG_BUFFERS_CSV}.",
    )
    return parser


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    if args.quiet:
        wp.config.quiet = True
    run_experiment(
        config_path=args.config,
        output_dir=args.output_dir,
        heights=tuple(args.heights),
        simulation_time=args.simulation_time,
        frame_fps=args.fps,
        sim_substeps=args.substeps,
        rigid_contact_max=args.rigid_contact_max,
        nconmax=args.nconmax,
        njmax=args.njmax,
        buffer_mult_iso=args.buffer_mult_iso,
        buffer_mult_contact=args.buffer_mult_contact,
        buffer_fraction=args.buffer_fraction,
        debug_buffers=args.debug_buffers,
        device=args.device,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
