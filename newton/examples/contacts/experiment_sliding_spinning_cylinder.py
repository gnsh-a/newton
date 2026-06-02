# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Sliding-spinning cylinder contact-reduction experiment.
#
# Command:
#     python -m newton.examples.contacts.experiment_sliding_spinning_cylinder
#
###########################################################################

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import warp as wp

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
    _summary_stats,
)
from newton.examples.contacts.experiment_spinning_cylinder_spin_down import (
    DEFAULT_SDF_MAX_RESOLUTION,
    SpinModeConfig,
    _SpinDownRun,
    _write_csv,
)

DEFAULT_OUTPUT_DIR = Path("output") / "H5_sliding_spinning_cylinder"
DEFAULT_INITIAL_EPSILONS = (0.25, 0.5, 1.0, 2.0)
DEFAULT_INITIAL_OMEGA = 10.0

FRAME_FPS = 120
RUN_SECONDS = 0.8
SIM_SUBSTEPS = 4

STOP_SPEED_M_PER_S = 0.02
STOP_OMEGA_RAD_PER_S = 0.5
EPSILON_REFERENCE = 0.653

TIMESERIES_CSV = "sliding_spinning_timeseries.csv"
SUMMARY_CSV = "sliding_spinning_summary.csv"

TIMESERIES_COLUMNS = (
    "time_s",
    "initial_epsilon",
    "initial_speed_m_per_s",
    "initial_omega_rad_per_s",
    "mode",
    "reduce_contacts",
    "pre_prune_contacts",
    "face_contact_count",
    "rigid_contact_count",
    "cylinder_x_m",
    "cylinder_y_m",
    "cylinder_z_m",
    "cylinder_x_travel_m",
    "cylinder_y_drift_m",
    "cylinder_vx_m_per_s",
    "cylinder_vy_m_per_s",
    "cylinder_vz_m_per_s",
    "horizontal_speed_m_per_s",
    "cylinder_omega_z_rad_per_s",
    "omega_abs_rad_per_s",
    "spin_edge_speed_m_per_s",
    "epsilon",
    "cylinder_tilt_deg",
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
    "initial_epsilon",
    "initial_speed_m_per_s",
    "initial_omega_rad_per_s",
    "mode",
    "reduce_contacts",
    "pre_prune_contacts",
    "frame_count",
    "speed_stop_threshold_m_per_s",
    "spin_stop_threshold_rad_per_s",
    "speed_stopped",
    "spin_stopped",
    "coupled_stopped",
    "speed_stop_time_s",
    "spin_stop_time_s",
    "coupled_stop_time_s",
    "epsilon_reference",
    "late_epsilon",
    "late_epsilon_error",
    "final_epsilon",
    "final_speed_m_per_s",
    "final_omega_z_rad_per_s",
    "final_x_travel_m",
    "final_y_drift_m",
    "final_tilt_deg",
    "final_penetration_depth_m",
    "max_penetration_depth_m",
    "solver_impulse_x_Ns",
    "solver_impulse_y_Ns",
    "solver_impulse_z_Ns",
    "solver_horizontal_impulse_Ns",
    "solver_yaw_impulse_Nms",
    "mean_abs_solver_fx_active_N",
    "mean_abs_solver_tz_active_Nm",
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
    """Output summary for one sliding-spinning run."""

    initial_epsilon: float
    initial_speed: float
    initial_omega: float
    mode: str
    timeseries_path: Path
    summary_path: Path
    row_count: int
    buffer_overflow: bool


@dataclass
class _RunData:
    initial_epsilon: float
    initial_speed: float
    initial_omega: float
    mode: str
    timeseries_rows: list[dict[str, object]]
    summary_row: dict[str, object]
    buffer_overflow: bool


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _stop_index(rows: list[dict[str, object]], key: str, threshold: float) -> int | None:
    for idx, row in enumerate(rows):
        if abs(float(row[key])) <= threshold:
            return idx
    return None


def _coupled_stop_index(rows: list[dict[str, object]]) -> int | None:
    for idx, row in enumerate(rows):
        speed = float(row["horizontal_speed_m_per_s"])
        omega = abs(float(row["cylinder_omega_z_rad_per_s"]))
        if speed <= STOP_SPEED_M_PER_S and omega <= STOP_OMEGA_RAD_PER_S:
            return idx
    return None


def _late_epsilon(rows: list[dict[str, object]]) -> float:
    values = []
    for row in rows:
        speed = float(row["horizontal_speed_m_per_s"])
        omega = abs(float(row["cylinder_omega_z_rad_per_s"]))
        epsilon = float(row["epsilon"])
        if speed > STOP_SPEED_M_PER_S and omega > STOP_OMEGA_RAD_PER_S and math.isfinite(epsilon):
            values.append(epsilon)
    if not values:
        return float("nan")
    return median(values[-min(10, len(values)) :])


def _print_buffer_summary(initial_epsilon: float, mode: str, max_stats: BufferStats) -> None:
    print(f"[epsilon0={initial_epsilon:.6f} mode={mode}] buffer summary")
    for label, max_count, capacity, utilization in _summary_stats(max_stats):
        print(f"  {label}: max={max_count} capacity={capacity} utilization={utilization:.4f}")
    if max_stats.reduction_hashtable_failures:
        print(f"  reduction_hashtable_failures: max={max_stats.reduction_hashtable_failures}")


def _timeseries_row(
    *,
    base_row: dict[str, object],
    initial_epsilon: float,
    initial_speed: float,
    initial_omega: float,
    mode: SpinModeConfig,
) -> dict[str, object]:
    vx = float(base_row["cylinder_vx_m_per_s"])
    vy = float(base_row["cylinder_vy_m_per_s"])
    omega_z = float(base_row["cylinder_omega_z_rad_per_s"])
    speed = math.hypot(vx, vy)
    omega_abs = abs(omega_z)
    spin_edge_speed = spinning.CYLINDER_RADIUS * omega_abs
    epsilon = speed / spin_edge_speed if spin_edge_speed > 0.0 else float("inf")
    x = float(base_row["cylinder_x_m"])
    y = float(base_row["cylinder_y_m"])
    return {
        "time_s": base_row["time_s"],
        "initial_epsilon": initial_epsilon,
        "initial_speed_m_per_s": initial_speed,
        "initial_omega_rad_per_s": initial_omega,
        "mode": mode.name,
        "reduce_contacts": mode.reduce_contacts,
        "pre_prune_contacts": mode.pre_prune_contacts,
        "face_contact_count": base_row["face_contact_count"],
        "rigid_contact_count": base_row["rigid_contact_count"],
        "cylinder_x_m": x,
        "cylinder_y_m": y,
        "cylinder_z_m": base_row["cylinder_z_m"],
        "cylinder_x_travel_m": x,
        "cylinder_y_drift_m": y,
        "cylinder_vx_m_per_s": vx,
        "cylinder_vy_m_per_s": vy,
        "cylinder_vz_m_per_s": base_row["cylinder_vz_m_per_s"],
        "horizontal_speed_m_per_s": speed,
        "cylinder_omega_z_rad_per_s": omega_z,
        "omega_abs_rad_per_s": omega_abs,
        "spin_edge_speed_m_per_s": spin_edge_speed,
        "epsilon": epsilon,
        "cylinder_tilt_deg": base_row["cylinder_tilt_deg"],
        "cylinder_penetration_depth_m": base_row["cylinder_penetration_depth_m"],
        "solver_fx_N": base_row["solver_fx_N"],
        "solver_fy_N": base_row["solver_fy_N"],
        "solver_fz_N": base_row["solver_fz_N"],
        "solver_tx_Nm": base_row["solver_tx_Nm"],
        "solver_ty_Nm": base_row["solver_ty_Nm"],
        "solver_tz_Nm": base_row["solver_tz_Nm"],
        "solver_force_count": base_row["solver_force_count"],
        "buffer_overflow": base_row["buffer_overflow"],
    }


def _summary_row(
    *,
    initial_epsilon: float,
    initial_speed: float,
    initial_omega: float,
    mode: SpinModeConfig,
    frame_dt: float,
    frame_count: int,
    max_stats: BufferStats,
    rows: list[dict[str, object]],
) -> dict[str, object]:
    speed_stop_idx = _stop_index(rows, "horizontal_speed_m_per_s", STOP_SPEED_M_PER_S)
    spin_stop_idx = _stop_index(rows, "cylinder_omega_z_rad_per_s", STOP_OMEGA_RAD_PER_S)
    coupled_stop_idx = _coupled_stop_index(rows)
    final_row = rows[-1]
    active_rows = [row for row in rows if int(row["solver_force_count"]) > 0]
    solver_impulse_x = sum(float(row["solver_fx_N"]) * frame_dt for row in rows)
    solver_impulse_y = sum(float(row["solver_fy_N"]) * frame_dt for row in rows)
    solver_impulse_z = sum(float(row["solver_fz_N"]) * frame_dt for row in rows)
    solver_yaw_impulse = sum(float(row["solver_tz_Nm"]) * frame_dt for row in rows)
    late_epsilon = _late_epsilon(rows)
    return {
        "initial_epsilon": initial_epsilon,
        "initial_speed_m_per_s": initial_speed,
        "initial_omega_rad_per_s": initial_omega,
        "mode": mode.name,
        "reduce_contacts": mode.reduce_contacts,
        "pre_prune_contacts": mode.pre_prune_contacts,
        "frame_count": frame_count,
        "speed_stop_threshold_m_per_s": STOP_SPEED_M_PER_S,
        "spin_stop_threshold_rad_per_s": STOP_OMEGA_RAD_PER_S,
        "speed_stopped": speed_stop_idx is not None,
        "spin_stopped": spin_stop_idx is not None,
        "coupled_stopped": coupled_stop_idx is not None,
        "speed_stop_time_s": float(rows[speed_stop_idx]["time_s"]) if speed_stop_idx is not None else float("nan"),
        "spin_stop_time_s": float(rows[spin_stop_idx]["time_s"]) if spin_stop_idx is not None else float("nan"),
        "coupled_stop_time_s": float(rows[coupled_stop_idx]["time_s"])
        if coupled_stop_idx is not None
        else float("nan"),
        "epsilon_reference": EPSILON_REFERENCE,
        "late_epsilon": late_epsilon,
        "late_epsilon_error": late_epsilon - EPSILON_REFERENCE if math.isfinite(late_epsilon) else float("nan"),
        "final_epsilon": float(final_row["epsilon"]),
        "final_speed_m_per_s": float(final_row["horizontal_speed_m_per_s"]),
        "final_omega_z_rad_per_s": float(final_row["cylinder_omega_z_rad_per_s"]),
        "final_x_travel_m": float(final_row["cylinder_x_travel_m"]),
        "final_y_drift_m": float(final_row["cylinder_y_drift_m"]),
        "final_tilt_deg": float(final_row["cylinder_tilt_deg"]),
        "final_penetration_depth_m": float(final_row["cylinder_penetration_depth_m"]),
        "max_penetration_depth_m": max(float(row["cylinder_penetration_depth_m"]) for row in rows),
        "solver_impulse_x_Ns": solver_impulse_x,
        "solver_impulse_y_Ns": solver_impulse_y,
        "solver_impulse_z_Ns": solver_impulse_z,
        "solver_horizontal_impulse_Ns": math.hypot(solver_impulse_x, solver_impulse_y),
        "solver_yaw_impulse_Nms": solver_yaw_impulse,
        "mean_abs_solver_fx_active_N": _mean([abs(float(row["solver_fx_N"])) for row in active_rows]),
        "mean_abs_solver_tz_active_Nm": _mean([abs(float(row["solver_tz_Nm"])) for row in active_rows]),
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
    initial_epsilon: float,
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
    initial_speed = initial_epsilon * abs(initial_omega) * spinning.CYLINDER_RADIUS
    runner = _SpinDownRun(
        initial_omega=initial_omega,
        initial_vx=initial_speed,
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
        rows.append(
            _timeseries_row(
                base_row=frame_data.timeseries_row,
                initial_epsilon=initial_epsilon,
                initial_speed=initial_speed,
                initial_omega=initial_omega,
                mode=mode,
            )
        )
        max_stats.update_max(frame_data.buffer_stats)

    summary_row = _summary_row(
        initial_epsilon=initial_epsilon,
        initial_speed=initial_speed,
        initial_omega=initial_omega,
        mode=mode,
        frame_dt=frame_dt,
        frame_count=len(rows),
        max_stats=max_stats,
        rows=rows,
    )
    if verbose:
        print(f"simulated epsilon0={initial_epsilon:.6f} mode={mode.name} ({len(rows)} samples)")
        _print_buffer_summary(initial_epsilon, mode.name, max_stats)

    if max_stats.overflow:
        raise RuntimeError(f"contact-buffer validity gate failed for epsilon0={initial_epsilon:.6f} mode={mode.name}")

    return _RunData(
        initial_epsilon=initial_epsilon,
        initial_speed=initial_speed,
        initial_omega=initial_omega,
        mode=mode.name,
        timeseries_rows=rows,
        summary_row=summary_row,
        buffer_overflow=max_stats.overflow,
    )


def _comparison_status(summary_rows: list[dict[str, object]]) -> None:
    by_epsilon: dict[float, dict[str, dict[str, object]]] = {}
    for row in summary_rows:
        by_epsilon.setdefault(float(row["initial_epsilon"]), {})[str(row["mode"])] = row

    for epsilon0 in sorted(by_epsilon):
        reduced = by_epsilon[epsilon0].get("reduced")
        unreduced = by_epsilon[epsilon0].get("unreduced")
        if reduced is None or unreduced is None:
            continue
        count_ratio = float(reduced["mean_solver_force_count"]) / max(
            float(unreduced["mean_solver_force_count"]),
            1.0,
        )
        print(
            f"[epsilon0={epsilon0:.6f}] "
            f"late epsilon off/on={float(unreduced['late_epsilon']):.6g}/"
            f"{float(reduced['late_epsilon']):.6g}, "
            f"speed stop off/on={float(unreduced['speed_stop_time_s']):.6g}/"
            f"{float(reduced['speed_stop_time_s']):.6g} s, "
            f"spin stop off/on={float(unreduced['spin_stop_time_s']):.6g}/"
            f"{float(reduced['spin_stop_time_s']):.6g} s, "
            f"force-count ratio={count_ratio:.5g}"
        )


def run_experiment(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    initial_epsilons: tuple[float, ...] | list[float] = DEFAULT_INITIAL_EPSILONS,
    initial_omega: float = DEFAULT_INITIAL_OMEGA,
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
    device: str | None = None,
    verbose: bool = True,
) -> list[RunResult]:
    if simulation_time <= 0.0:
        raise ValueError(f"simulation_time must be positive; got {simulation_time}")
    if frame_fps <= 0:
        raise ValueError(f"frame_fps must be positive; got {frame_fps}")
    if sim_substeps <= 0:
        raise ValueError(f"sim_substeps must be positive; got {sim_substeps}")
    if initial_omega <= 0.0:
        raise ValueError(f"initial_omega must be positive; got {initial_omega}")
    if device:
        wp.set_device(device)

    output_dir = Path(output_dir)
    timeseries_path = output_dir / TIMESERIES_CSV
    summary_path = output_dir / SUMMARY_CSV
    modes = (
        SpinModeConfig(name="unreduced", reduce_contacts=False),
        SpinModeConfig(name="reduced", reduce_contacts=True),
    )
    run_data: list[_RunData] = []
    for initial_epsilon in initial_epsilons:
        if float(initial_epsilon) <= 0.0:
            raise ValueError(f"initial_epsilons must be positive; got {initial_epsilon}")
        for mode in modes:
            run_data.append(
                run_single(
                    initial_epsilon=float(initial_epsilon),
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
            initial_epsilon=data.initial_epsilon,
            initial_speed=data.initial_speed,
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
        "--initial-epsilons",
        type=float,
        nargs="+",
        default=list(DEFAULT_INITIAL_EPSILONS),
        help="Initial ratios v0 / (R * omega0).",
    )
    parser.add_argument("--initial-omega", type=float, default=DEFAULT_INITIAL_OMEGA, help="Initial yaw rate [rad/s].")
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
    return parser


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    if args.quiet:
        wp.config.quiet = True
    run_experiment(
        output_dir=args.output_dir,
        initial_epsilons=tuple(args.initial_epsilons),
        initial_omega=args.initial_omega,
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
        device=args.device,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
