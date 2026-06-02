# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Flat sliding block contact-reduction experiment.
#
# Command:
#     python -m newton.examples.contacts.experiment_flat_sliding_block
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

DEFAULT_OUTPUT_DIR = Path("output") / "H2_flat_sliding_block"
DEFAULT_INITIAL_SPEEDS = (0.05, 0.1, 0.2, 0.4)

FRAME_FPS = 120
RUN_SECONDS = 0.25
SIM_SUBSTEPS = 4
STOP_SPEED_M_PER_S = 0.005

TIMESERIES_CSV = "sliding_timeseries.csv"
SUMMARY_CSV = "sliding_summary.csv"

TIMESERIES_COLUMNS = (
    "time_s",
    "initial_speed_m_per_s",
    "mode",
    "reduce_contacts",
    "pre_prune_contacts",
    "rigid_contact_count",
    "cube_x_m",
    "cube_y_m",
    "cube_z_m",
    "cube_vx_m_per_s",
    "cube_vy_m_per_s",
    "cube_vz_m_per_s",
    "cube_speed_m_per_s",
    "cube_tilt_deg",
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
    "initial_speed_m_per_s",
    "mode",
    "reduce_contacts",
    "pre_prune_contacts",
    "frame_count",
    "stopped",
    "stop_speed_threshold_m_per_s",
    "expected_coulomb_stop_time_s",
    "expected_coulomb_stop_travel_m",
    "stop_time_s",
    "stop_travel_m",
    "final_travel_m",
    "final_y_m",
    "final_speed_m_per_s",
    "final_tilt_deg",
    "final_penetration_depth_m",
    "max_penetration_depth_m",
    "solver_impulse_x_Ns",
    "solver_impulse_y_Ns",
    "solver_impulse_z_Ns",
    "horizontal_impulse_Ns",
    "mean_solver_force_count",
    "mean_rigid_contact_count",
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
    """Output summary for one initial-speed/mode run."""

    initial_speed: float
    mode: str
    timeseries_path: Path
    summary_path: Path
    row_count: int
    buffer_overflow: bool


@dataclass
class _RunData:
    initial_speed: float
    mode: str
    timeseries_rows: list[dict[str, object]]
    summary_row: dict[str, object]
    buffer_overflow: bool


@dataclass(frozen=True)
class _FrameData:
    timeseries_row: dict[str, object]
    buffer_stats: BufferStats


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _stop_index(rows: list[dict[str, object]]) -> int | None:
    for idx, row in enumerate(rows):
        if float(row["cube_speed_m_per_s"]) <= STOP_SPEED_M_PER_S:
            return idx
    return None


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format_csv_value(row[key]) for key in fieldnames})


def _print_buffer_summary(initial_speed: float, mode: str, max_stats: BufferStats) -> None:
    print(f"[initial_speed={initial_speed:.6f} m/s mode={mode}] buffer summary")
    for label, max_count, capacity, utilization in _summary_stats(max_stats):
        print(f"  {label}: max={max_count} capacity={capacity} utilization={utilization:.4f}")
    if max_stats.reduction_hashtable_failures:
        print(f"  reduction_hashtable_failures: max={max_stats.reduction_hashtable_failures}")


class _SlidingRun:
    def __init__(
        self,
        *,
        scene: SceneConfig,
        initial_speed: float,
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
        self.initial_speed = float(initial_speed)
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
                wp.vec3(0.0, 0.0, scene.cube_half_extent),
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
        qd_start = builder.joint_qd_start[-1]
        builder.joint_qd[qd_start + 0] = self.initial_speed

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
            raise RuntimeError("flat sliding block experiment requires hydroelastic contacts")
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
        body_qd = self.state_0.body_qd.numpy()[self.cube_body]
        signed_clearance = _cube_signed_clearance_m(body_q, self.scene.cube_half_extent)
        speed = math.hypot(float(body_qd[0]), float(body_qd[1]))
        return {
            "time_s": time_s,
            "initial_speed_m_per_s": self.initial_speed,
            "mode": self.mode.name,
            "reduce_contacts": self.mode.reduce_contacts,
            "pre_prune_contacts": self.mode.pre_prune_contacts,
            "rigid_contact_count": frame_stats.rigid_contact_count,
            "cube_x_m": float(body_q[0]),
            "cube_y_m": float(body_q[1]),
            "cube_z_m": float(body_q[2]),
            "cube_vx_m_per_s": float(body_qd[0]),
            "cube_vy_m_per_s": float(body_qd[1]),
            "cube_vz_m_per_s": float(body_qd[2]),
            "cube_speed_m_per_s": speed,
            "cube_tilt_deg": _cube_tilt_deg(body_q),
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


def _summary_row(
    *,
    scene: SceneConfig,
    initial_speed: float,
    mode: ModeConfig,
    frame_dt: float,
    frame_count: int,
    max_stats: BufferStats,
    rows: list[dict[str, object]],
) -> dict[str, object]:
    stop_idx = _stop_index(rows)
    stop_row = rows[stop_idx] if stop_idx is not None else None
    final_row = rows[-1]
    coulomb_deceleration = scene.mu_sliding * cube_on_plate.GRAVITY
    expected_stop_time = initial_speed / coulomb_deceleration if coulomb_deceleration > 0.0 else float("nan")
    expected_stop_travel = (
        initial_speed * initial_speed / (2.0 * coulomb_deceleration) if coulomb_deceleration > 0.0 else float("nan")
    )
    impulse_x = sum(float(row["solver_fx_N"]) * frame_dt for row in rows)
    impulse_y = sum(float(row["solver_fy_N"]) * frame_dt for row in rows)
    impulse_z = sum(float(row["solver_fz_N"]) * frame_dt for row in rows)
    return {
        "initial_speed_m_per_s": initial_speed,
        "mode": mode.name,
        "reduce_contacts": mode.reduce_contacts,
        "pre_prune_contacts": mode.pre_prune_contacts,
        "frame_count": frame_count,
        "stopped": stop_row is not None,
        "stop_speed_threshold_m_per_s": STOP_SPEED_M_PER_S,
        "expected_coulomb_stop_time_s": expected_stop_time,
        "expected_coulomb_stop_travel_m": expected_stop_travel,
        "stop_time_s": float(stop_row["time_s"]) if stop_row is not None else float("nan"),
        "stop_travel_m": float(stop_row["cube_x_m"]) if stop_row is not None else float("nan"),
        "final_travel_m": float(final_row["cube_x_m"]),
        "final_y_m": float(final_row["cube_y_m"]),
        "final_speed_m_per_s": float(final_row["cube_speed_m_per_s"]),
        "final_tilt_deg": float(final_row["cube_tilt_deg"]),
        "final_penetration_depth_m": float(final_row["cube_penetration_depth_m"]),
        "max_penetration_depth_m": max(float(row["cube_penetration_depth_m"]) for row in rows),
        "solver_impulse_x_Ns": impulse_x,
        "solver_impulse_y_Ns": impulse_y,
        "solver_impulse_z_Ns": impulse_z,
        "horizontal_impulse_Ns": math.hypot(impulse_x, impulse_y),
        "mean_solver_force_count": _mean([float(row["solver_force_count"]) for row in rows]),
        "mean_rigid_contact_count": _mean([float(row["rigid_contact_count"]) for row in rows]),
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
    scene: SceneConfig,
    initial_speed: float,
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
    runner = _SlidingRun(
        scene=scene,
        initial_speed=initial_speed,
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
            time_s=(frame_idx + 1) * frame_dt,
        )
        rows.append(frame_data.timeseries_row)
        max_stats.update_max(frame_data.buffer_stats)

    summary_row = _summary_row(
        scene=scene,
        initial_speed=initial_speed,
        mode=mode,
        frame_dt=frame_dt,
        frame_count=len(rows),
        max_stats=max_stats,
        rows=rows,
    )
    if verbose:
        print(f"simulated initial_speed={initial_speed:.6f} m/s mode={mode.name} ({len(rows)} samples)")
        _print_buffer_summary(initial_speed, mode.name, max_stats)

    if max_stats.overflow:
        raise RuntimeError(
            f"contact-buffer validity gate failed for initial_speed={initial_speed:.6f} m/s mode={mode.name}"
        )

    return _RunData(
        initial_speed=initial_speed,
        mode=mode.name,
        timeseries_rows=rows,
        summary_row=summary_row,
        buffer_overflow=max_stats.overflow,
    )


def _comparison_status(summary_rows: list[dict[str, object]]) -> None:
    by_speed: dict[float, dict[str, dict[str, object]]] = {}
    for row in summary_rows:
        by_speed.setdefault(float(row["initial_speed_m_per_s"]), {})[str(row["mode"])] = row

    for speed in sorted(by_speed):
        reduced = by_speed[speed].get("reduced")
        unreduced = by_speed[speed].get("unreduced")
        if reduced is None or unreduced is None:
            continue
        count_ratio = float(reduced["mean_solver_force_count"]) / max(
            float(unreduced["mean_solver_force_count"]),
            1.0,
        )
        print(
            f"[speed={speed:.6f} m/s] "
            f"stop_time off/on={float(unreduced['stop_time_s']):.6g}/{float(reduced['stop_time_s']):.6g} s, "
            f"travel off/on={float(unreduced['stop_travel_m']):.6g}/{float(reduced['stop_travel_m']):.6g} m, "
            f"impulse off/on={float(unreduced['horizontal_impulse_Ns']):.6g}/"
            f"{float(reduced['horizontal_impulse_Ns']):.6g} N*s, "
            f"force-count ratio={count_ratio:.5g}"
        )


def run_experiment(
    *,
    config_path: str | Path = cube_on_plate.DEFAULT_CONFIG_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    initial_speeds: tuple[float, ...] | list[float] = DEFAULT_INITIAL_SPEEDS,
    simulation_time: float = RUN_SECONDS,
    frame_fps: int = FRAME_FPS,
    sim_substeps: int = SIM_SUBSTEPS,
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
    if device:
        wp.set_device(device)

    scene = _load_scene_config(config_path)
    output_dir = Path(output_dir)
    timeseries_path = output_dir / TIMESERIES_CSV
    summary_path = output_dir / SUMMARY_CSV
    modes = (
        ModeConfig(name="unreduced", reduce_contacts=False),
        ModeConfig(name="reduced", reduce_contacts=True),
    )
    run_data: list[_RunData] = []
    for initial_speed in initial_speeds:
        if float(initial_speed) <= 0.0:
            raise ValueError(f"initial speeds must be positive; got {initial_speed}")
        for mode in modes:
            run_data.append(
                run_single(
                    scene=scene,
                    initial_speed=float(initial_speed),
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
            initial_speed=data.initial_speed,
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
    parser.add_argument(
        "--config",
        type=str,
        default=str(cube_on_plate.DEFAULT_CONFIG_PATH),
        help="Path to the shared cube-on-plate YAML config.",
    )
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Directory for CSV files.")
    parser.add_argument(
        "--initial-speeds",
        type=float,
        nargs="+",
        default=list(DEFAULT_INITIAL_SPEEDS),
        help="Initial horizontal speeds [m/s] to sweep.",
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
    return parser


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    if args.quiet:
        wp.config.quiet = True
    run_experiment(
        config_path=args.config,
        output_dir=args.output_dir,
        initial_speeds=tuple(args.initial_speeds),
        simulation_time=args.simulation_time,
        frame_fps=args.fps,
        sim_substeps=args.substeps,
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
