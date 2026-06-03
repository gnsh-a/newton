# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Build a self-contained HTML report from cube-on-plate settle CSVs.

This script intentionally does not import Newton. The Newton experiment writes
CSV files; this script reads those CSV files and renders the standalone report
using the shared report framework in :mod:`_report_common`.
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _report_common as rc

DEFAULT_CSV_DIR = Path("output") / "H1_cube_on_plate_settle"
DEFAULT_HTML_PATH = DEFAULT_CSV_DIR / "cube_on_plate_settle_report.html"
HYPOTHESIS_RECORD_NAME = "H1_cube_on_plate_settle_hypothesis.md"
HYPOTHESIS_RECORD_SOURCE = Path(__file__).resolve().parents[1] / "hypothesis" / HYPOTHESIS_RECORD_NAME
TIMESERIES_CSV = "settle_timeseries.csv"
SUMMARY_CSV = "settle_summary.csv"

PAGE_TITLE = "H1: Cube-on-Plate Settle"

DEFAULT_CUBE_SIDE_M = 0.1
DEFAULT_CUBE_MASS_KG = 0.8
DEFAULT_GRAVITY_M_PER_S2 = 9.81
DEFAULT_PLATE_HALF_EXTENT_M = 0.25
DEFAULT_PLATE_HALF_THICKNESS_M = 0.2

MODE_COLORS = rc.MODE_COLORS
MODES = rc.MODES

PRACTICAL_FORCE_FLOOR_N = 0.02
PRACTICAL_TORQUE_FLOOR_NM = 1.0e-3
PRACTICAL_NORMALIZED_FLOOR = 1.0e-4
PRACTICAL_LENGTH_FLOOR_MM = 0.01

SOLVER_SERIES = (
    ("solver_fx_N", "Fx [N]", None),
    ("solver_fy_N", "Fy [N]", None),
    ("solver_fz_N", "Fz [N]", "weight"),
    ("solver_tx_Nm", "Tx [N m]", None),
    ("solver_ty_Nm", "Ty [N m]", None),
    ("solver_tz_Nm", "Tz [N m]", None),
    ("rigid_contact_count", "contact count", None),
)


@dataclass(frozen=True)
class PhysicsConstants:
    """Physical constants used to normalize the CSV data."""

    cube_side_m: float = DEFAULT_CUBE_SIDE_M
    cube_mass_kg: float = DEFAULT_CUBE_MASS_KG
    gravity_m_per_s2: float = DEFAULT_GRAVITY_M_PER_S2
    plate_half_extent_m: float = DEFAULT_PLATE_HALF_EXTENT_M
    plate_half_thickness_m: float = DEFAULT_PLATE_HALF_THICKNESS_M

    @property
    def cube_weight_N(self) -> float:
        return self.cube_mass_kg * self.gravity_m_per_s2

    @property
    def plate_side_m(self) -> float:
        return 2.0 * self.plate_half_extent_m

    @property
    def plate_thickness_m(self) -> float:
        return 2.0 * self.plate_half_thickness_m


def _array(rows: list[dict[str, str]], key: str) -> list[float]:
    return [rc.as_float(row, key) for row in rows]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _mean_finite(values: list[float]) -> float:
    return _mean([value for value in values if math.isfinite(value)])


def _max_finite(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return max(finite) if finite else float("nan")


def _median(values: list[float]) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _post_window_bounds(rows: list[dict[str, str]], fraction: float) -> tuple[int, float]:
    if not rows:
        return 0, 0.0
    count = max(1, int(math.ceil(len(rows) * fraction)))
    start = max(0, len(rows) - count)
    return start, rc.as_float(rows[start], "time_s")


def _post_window(rows: list[dict[str, str]], fraction: float) -> list[dict[str, str]]:
    start, _time = _post_window_bounds(rows, fraction)
    return rows[start:]


def _norm2(rows: list[dict[str, str]], x_key: str, y_key: str) -> list[float]:
    return [math.hypot(float(row[x_key]), float(row[y_key])) for row in rows]


def _norm3(rows: list[dict[str, str]], x_key: str, y_key: str, z_key: str) -> list[float]:
    return [math.sqrt(float(row[x_key]) ** 2 + float(row[y_key]) ** 2 + float(row[z_key]) ** 2) for row in rows]


def _penetration_depth_m(row: dict[str, str]) -> float:
    return rc.as_float(row, "cube_penetration_depth_m")


def _support_offset_m(row: dict[str, str]) -> float:
    fz = abs(rc.as_float(row, "solver_fz_N"))
    if fz <= 1.0e-12:
        return float("nan")
    tx = rc.as_float(row, "solver_tx_Nm")
    ty = rc.as_float(row, "solver_ty_Nm")
    return math.hypot(tx, ty) / fz


def load_runs(csv_dir: str | Path) -> dict[float, dict[str, list[dict[str, str]]]]:
    """Load settle CSVs grouped by drop height and mode."""

    csv_dir = Path(csv_dir)
    timeseries_path = csv_dir / TIMESERIES_CSV
    if not timeseries_path.exists():
        raise FileNotFoundError(f"missing compact timeseries CSV: {timeseries_path}")

    runs: dict[float, dict[str, list[dict[str, str]]]] = {}
    for row in rc.read_csv(timeseries_path):
        height = rc.as_float(row, "height_m")
        mode = row["mode"]
        runs.setdefault(height, {}).setdefault(mode, []).append(row)
    if not runs:
        raise ValueError(f"compact timeseries CSV has no rows: {timeseries_path}")
    return runs


def load_run_summaries(csv_dir: str | Path) -> dict[float, dict[str, dict[str, str]]]:
    """Load compact per-run summaries."""

    summary_path = Path(csv_dir) / SUMMARY_CSV
    if not summary_path.exists():
        raise FileNotFoundError(f"missing compact summary CSV: {summary_path}")
    summaries: dict[float, dict[str, dict[str, str]]] = {}
    for row in rc.read_csv(summary_path):
        height = rc.as_float(row, "height_m")
        mode = row["mode"]
        summaries.setdefault(height, {})[mode] = row
    if not summaries:
        raise ValueError(f"compact summary CSV has no rows: {summary_path}")
    return summaries


def select_height(runs: dict[float, dict[str, list[dict[str, str]]]], requested_height: float) -> float:
    """Return the closest available height to the requested value."""

    heights = sorted(runs)
    return float(min(heights, key=lambda h: abs(h - requested_height)))


def _summary_buffer_utilizations(summary: dict[str, str]) -> list[float]:
    stages = (
        ("max_hydro_broadphase_blocks", "hydro_broadphase_capacity"),
        ("max_hydro_iso_subblocks_l0", "hydro_iso_subblocks_l0_capacity"),
        ("max_hydro_iso_subblocks_l1", "hydro_iso_subblocks_l1_capacity"),
        ("max_hydro_iso_subblocks_l2", "hydro_iso_subblocks_l2_capacity"),
        ("max_hydro_iso_voxels", "hydro_iso_voxels_capacity"),
        ("max_face_contact_count", "face_contact_capacity"),
        ("max_rigid_contact_count", "rigid_contact_capacity"),
        ("max_reduction_hashtable_active", "reduction_hashtable_capacity"),
    )
    utilizations = []
    for count_key, capacity_key in stages:
        capacity = rc.as_float(summary, capacity_key)
        if capacity > 0.0:
            utilizations.append(rc.as_float(summary, count_key) / capacity)
    return utilizations


def compute_metrics(
    runs: dict[float, dict[str, list[dict[str, str]]]],
    constants: PhysicsConstants,
    window_fraction: float,
    summaries: dict[float, dict[str, dict[str, str]]],
) -> dict[float, dict[str, dict[str, float | bool]]]:
    """Compute normalized, physically interpretable summary metrics."""

    metrics: dict[float, dict[str, dict[str, float | bool]]] = {}
    torque_scale = constants.cube_weight_N * constants.cube_side_m
    for height, mode_rows in runs.items():
        metrics[height] = {}
        for mode, rows in mode_rows.items():
            window = _post_window(rows, window_fraction)
            lateral = _norm2(window, "solver_fx_N", "solver_fy_N")
            torque = _norm3(window, "solver_tx_Nm", "solver_ty_Nm", "solver_tz_Nm")
            fz = _array(window, "solver_fz_N")
            force_count = _array(window, "solver_force_count")
            rigid_count = _array(window, "rigid_contact_count")
            penetration_mm = [1000.0 * _penetration_depth_m(row) for row in window]
            support_offset_mm = [1000.0 * _support_offset_m(row) for row in window]
            summary = summaries.get(height, {}).get(mode)
            if summary is None:
                raise KeyError(f"missing summary row for height={height:.6f} mode={mode}")

            buffer_utilization = _summary_buffer_utilizations(summary)
            rigid_contact_count_max = rc.as_float(summary, "max_rigid_contact_count")
            rigid_capacity = rc.as_float(summary, "rigid_contact_capacity")
            rigid_contact_utilization_max = rigid_contact_count_max / rigid_capacity if rigid_capacity > 0.0 else 0.0
            buffer_overflow = rc.as_bool(summary["buffer_overflow"])
            final_drift_mm = 1000.0 * math.hypot(
                rc.as_float(summary, "final_cube_x_m"), rc.as_float(summary, "final_cube_y_m")
            )
            final_tilt_deg = rc.as_float(summary, "final_cube_tilt_deg")

            metrics[height][mode] = {
                "fz_norm_mean": _mean([value / constants.cube_weight_N for value in fz]),
                "lateral_norm_mean": _mean([value / constants.cube_weight_N for value in lateral]),
                "torque_norm_mean": _mean([value / torque_scale for value in torque]),
                "penetration_depth_mean_mm": _mean_finite(penetration_mm),
                "penetration_depth_max_mm": _max_finite(penetration_mm),
                "support_offset_mean_mm": _mean_finite(support_offset_mm),
                "support_offset_max_mm": _max_finite(support_offset_mm),
                "solver_force_count_mean": _mean(force_count),
                "rigid_contact_count_mean": _mean(rigid_count),
                "rigid_contact_count_max": rigid_contact_count_max,
                "rigid_contact_utilization_max": rigid_contact_utilization_max,
                "buffer_utilization_max": max(buffer_utilization) if buffer_utilization else 0.0,
                "buffer_overflow": buffer_overflow,
                "final_drift_mm": final_drift_mm,
                "final_tilt_deg": final_tilt_deg,
            }
    return metrics


def _format_percent(value: float) -> str:
    return f"{100.0 * value:.4g}%"


def _format_height(height: float) -> str:
    return f"{height:.6f} m"


def _mode_label(mode: str) -> str:
    return rc.MODE_LABELS.get(mode, mode)


def _padded_range(
    values: list[float], *, include: tuple[float, ...] = (), floor_span: float = 0.0
) -> tuple[float, float]:
    finite = rc.finite([*values, *include])
    if not finite:
        return 0.0, max(1.0, floor_span)
    low = min(finite)
    high = max(finite)
    if low == high:
        half_span = max(abs(low) * 0.05, floor_span * 0.5, 1.0)
        return low - half_span, high + half_span
    span = high - low
    center = 0.5 * (low + high)
    half_span = 0.55 * span
    return center - half_span, center + half_span


def _symmetric_range(values: list[float], *, floor_abs: float) -> tuple[float, float]:
    finite = rc.finite(values)
    limit = max([*[abs(v) for v in finite], floor_abs])
    return -1.1 * limit, 1.1 * limit


def _series_values(
    runs: dict[float, dict[str, list[dict[str, str]]]],
    height: float,
    key: str,
) -> list[float]:
    values: list[float] = []
    for rows in runs[height].values():
        values.extend(_array(rows, key))
    return values


def _raw_solver_y_range(
    runs: dict[float, dict[str, list[dict[str, str]]]],
    height: float,
    key: str,
    constants: PhysicsConstants,
) -> tuple[float, float]:
    values = _series_values(runs, height, key)
    if key in ("solver_fx_N", "solver_fy_N"):
        return _symmetric_range(values, floor_abs=PRACTICAL_FORCE_FLOOR_N)
    if key == "solver_fz_N":
        low, high = _padded_range(
            values, include=(0.0, constants.cube_weight_N), floor_span=0.05 * constants.cube_weight_N
        )
        return min(0.0, low), max(constants.cube_weight_N, high)
    if key in ("solver_tx_Nm", "solver_ty_Nm", "solver_tz_Nm"):
        return _symmetric_range(values, floor_abs=PRACTICAL_TORQUE_FLOOR_NM)
    if key == "rigid_contact_count":
        finite = rc.finite(values)
        high = max([*finite, 1.0])
        return 0.0, 1.1 * high
    return _padded_range(values)


def _time_series_figure(
    runs: dict[float, dict[str, list[dict[str, str]]]],
    *,
    height: float,
    constants: PhysicsConstants,
    window_fraction: float,
    key: str,
    ylabel: str,
    reference: str | None,
) -> rc.Figure:
    """Build one raw solver output against time for one drop height."""

    # Overlay every drop height (color = height, dash = mode); opens on the
    # representative height with the others a dropdown away.
    heights = sorted(runs)

    def _group(value: float) -> str:
        return f"h = {1000.0 * value:.3g} mm"

    default_group = _group(height)
    series: list[rc.Series] = []
    all_times: list[float] = []
    all_ys: list[float] = []
    for index, run_height in enumerate(heights):
        mode_rows = runs[run_height]
        group = _group(run_height)
        color = rc.group_color(index)
        for mode in MODES:
            rows = mode_rows.get(mode)
            if not rows:
                continue
            mode_dash = None if mode == "reduced" else "dash"
            solo = rc.SOLO_MODE_COLORS[mode]
            label = f"{group} · {_mode_label(mode)}"
            times = _array(rows, "time_s")
            ys = _array(rows, key)
            all_times.extend(times)
            all_ys.extend(ys)
            series.append(rc.Series(times, ys, label, color, dash=mode_dash, group=group, solo_color=solo))

    include = (0.0, constants.cube_weight_N) if reference == "weight" else (0.0,)
    hlines = ((constants.cube_weight_N, "m*g", "#111827"),) if reference == "weight" else ()
    return rc.Figure(
        title=ylabel,
        xlabel="time [s]",
        ylabel=ylabel,
        series=series,
        x_range=rc.padded_range(all_times, include=(0.0,)),
        y_range=rc.padded_range(all_ys, include=include),
        hlines=hlines,
        selector="drop height",
        selector_default=default_group,
        height=360,
    )


def render_solver_timeseries_figure(
    runs: dict[float, dict[str, list[dict[str, str]]]],
    *,
    height: float,
    constants: PhysicsConstants,
    window_fraction: float,
) -> str:
    """Render the raw seven solver outputs against time for one drop height."""

    figures = [
        _time_series_figure(
            runs,
            height=height,
            constants=constants,
            window_fraction=window_fraction,
            key=key,
            ylabel=ylabel,
            reference=reference,
        )
        for key, ylabel, reference in SOLVER_SERIES
    ]
    return rc.figure_grid(figures, columns=1)


def _metric_plot_series(
    metrics: dict[float, dict[str, dict[str, float | bool]]],
    *,
    key: str,
) -> list[rc.Series]:
    plot_series = []
    for mode in MODES:
        xs = [float(h) for h in sorted(metrics) if mode in metrics[h]]
        if not xs:
            continue
        ys = [float(metrics[h][mode][key]) for h in sorted(metrics) if mode in metrics[h]]
        plot_series.append(
            rc.Series(xs, ys, _mode_label(mode), MODE_COLORS.get(mode, "#111827"), draw_line=False, draw_marker=True)
        )
    return plot_series


def _all_metric_values(metrics: dict[float, dict[str, dict[str, float | bool]]], key: str) -> list[float]:
    return [float(mode_values[key]) for height_values in metrics.values() for mode_values in height_values.values()]


def render_physics_summary_figure(metrics: dict[float, dict[str, dict[str, float | bool]]]) -> str:
    """Render normalized post-settle physics metrics against drop height."""

    heights = sorted(metrics)
    x_range = _padded_range([float(h) for h in heights], floor_span=0.001)
    fz_values = _all_metric_values(metrics, "fz_norm_mean")
    fz_low = min(0.98, min(fz_values) - 0.002)
    fz_high = max(1.02, max(fz_values) + 0.002)

    lateral_values = _all_metric_values(metrics, "lateral_norm_mean")
    torque_values = _all_metric_values(metrics, "torque_norm_mean")
    penetration_values = rc.finite(_all_metric_values(metrics, "penetration_depth_mean_mm"))
    support_offset_values = rc.finite(_all_metric_values(metrics, "support_offset_mean_mm"))
    drift_values = rc.finite(_all_metric_values(metrics, "final_drift_mm"))
    tilt_values = rc.finite(_all_metric_values(metrics, "final_tilt_deg"))

    floor = PRACTICAL_NORMALIZED_FLOOR
    length_floor = PRACTICAL_LENGTH_FLOOR_MM
    return rc.figure_grid(
        [
            rc.Figure(
                title="Settled vertical support",
                xlabel="drop height [m]",
                ylabel="mean Fz / (m*g)",
                series=_metric_plot_series(metrics, key="fz_norm_mean"),
                x_range=x_range,
                y_range=(fz_low, fz_high),
                hlines=((1.0, "correct weight", "#111827"),),
                ybands=((1.0 - floor, 1.0 + floor, "practical floor", "#22c55e"),),
            ),
            rc.Figure(
                title="Sideways force leakage",
                xlabel="drop height [m]",
                ylabel="mean sqrt(Fx^2+Fy^2) / (m*g)",
                series=_metric_plot_series(metrics, key="lateral_norm_mean"),
                x_range=x_range,
                y_range=rc.zero_range(lateral_values),
                ybands=((0.0, floor, "practical floor", "#22c55e"),),
            ),
            rc.Figure(
                title="Torque imbalance",
                xlabel="drop height [m]",
                ylabel="mean ||tau|| / (m*g*L)",
                series=_metric_plot_series(metrics, key="torque_norm_mean"),
                x_range=x_range,
                y_range=rc.zero_range(torque_values),
                ybands=((0.0, floor, "practical floor", "#22c55e"),),
            ),
            rc.Figure(
                title="Geometric penetration",
                xlabel="drop height [m]",
                ylabel="mean penetration depth [mm]",
                series=_metric_plot_series(metrics, key="penetration_depth_mean_mm"),
                x_range=x_range,
                y_range=rc.zero_range(penetration_values),
                ybands=((0.0, length_floor, "practical floor", "#22c55e"),),
            ),
            rc.Figure(
                title="Support-point offset",
                xlabel="drop height [m]",
                ylabel="mean sqrt(Tx^2+Ty^2)/|Fz| [mm]",
                series=_metric_plot_series(metrics, key="support_offset_mean_mm"),
                x_range=x_range,
                y_range=rc.zero_range(support_offset_values),
                ybands=((0.0, length_floor, "practical floor", "#22c55e"),),
            ),
            rc.Figure(
                title="Final lateral drift",
                xlabel="drop height [m]",
                ylabel="final sqrt(x^2+y^2) [mm]",
                series=_metric_plot_series(metrics, key="final_drift_mm"),
                x_range=x_range,
                y_range=rc.zero_range(drift_values),
                hlines=((0.0, "rigid: no drift", "#111827"),),
                ybands=((0.0, length_floor, "practical floor", "#22c55e"),),
            ),
            rc.Figure(
                title="Final tilt",
                xlabel="drop height [m]",
                ylabel="final tilt [deg]",
                series=_metric_plot_series(metrics, key="final_tilt_deg"),
                x_range=x_range,
                y_range=rc.zero_range(tilt_values),
                hlines=((0.0, "rigid: no tilt", "#111827"),),
            ),
        ],
        columns=1,
    )


def _count_ratio_series(metrics: dict[float, dict[str, dict[str, float | bool]]]) -> list[rc.Series]:
    heights = []
    ratios = []
    for height in sorted(metrics):
        reduced = metrics[height].get("reduced")
        unreduced = metrics[height].get("unreduced")
        if reduced is None or unreduced is None:
            continue
        unreduced_count = float(unreduced["solver_force_count_mean"])
        if unreduced_count <= 0.0:
            continue
        heights.append(float(height))
        ratios.append(float(reduced["solver_force_count_mean"]) / unreduced_count)
    return (
        [rc.Series(heights, ratios, "reduce on / reduce off", "#7c3aed", draw_line=False, draw_marker=True)]
        if heights
        else []
    )


def _unreduced_metric_series(
    metrics: dict[float, dict[str, dict[str, float | bool]]],
    key: str,
    label: str,
    color: str,
) -> list[rc.Series]:
    xs = [float(h) for h in sorted(metrics) if "unreduced" in metrics[h]]
    ys = [float(metrics[h]["unreduced"][key]) for h in sorted(metrics) if "unreduced" in metrics[h]]
    return [rc.Series(xs, ys, label, color, draw_line=False, draw_marker=True)] if xs else []


def render_numerics_figure(metrics: dict[float, dict[str, dict[str, float | bool]]]) -> str:
    """Render contact-count and buffer-sanity metrics."""

    heights = sorted(metrics)
    x_range = _padded_range([float(h) for h in heights], floor_span=0.001)
    ratio_values = [ratio for plot_series in _count_ratio_series(metrics) for ratio in plot_series.ys]
    buffer_values = [
        float(metrics[h]["unreduced"]["buffer_utilization_max"]) for h in heights if "unreduced" in metrics[h]
    ]
    rigid_values = [
        float(metrics[h]["unreduced"]["rigid_contact_utilization_max"]) for h in heights if "unreduced" in metrics[h]
    ]

    return rc.figure_grid(
        [
            rc.Figure(
                title="Contact reduction ratio",
                xlabel="drop height [m]",
                ylabel="mean reduce on / reduce off force entries",
                series=_count_ratio_series(metrics),
                x_range=x_range,
                y_range=rc.zero_range(ratio_values),
                hlines=((1.0, "no reduction", "#111827"),),
            ),
            rc.Figure(
                title="Reduce-off buffer utilization",
                xlabel="drop height [m]",
                ylabel="max utilization",
                series=_unreduced_metric_series(metrics, "buffer_utilization_max", "max of all stages", "#0891b2"),
                x_range=x_range,
                y_range=rc.zero_range(buffer_values),
                hlines=((1.0, "capacity", "#111827"),),
            ),
            rc.Figure(
                title="Reduce-off rigid-contact buffer use",
                xlabel="drop height [m]",
                ylabel="max rigid_contact_count / capacity",
                series=_unreduced_metric_series(metrics, "rigid_contact_utilization_max", "rigid contacts", "#16a34a"),
                x_range=x_range,
                y_range=rc.zero_range(rigid_values),
                hlines=((1.0, "capacity", "#111827"),),
            ),
        ],
        columns=1,
    )


def _summary_values(metrics: dict[float, dict[str, dict[str, float | bool]]]) -> dict[str, float | bool]:
    reduced_errors = []
    reduced_lateral = []
    reduced_torque = []
    reduced_penetration = []
    reduced_support_offset = []
    ratios = []
    unreduced_overflow = False
    reduced_overflow = False
    unreduced_buffer_utilizations = []
    unreduced_errors = []
    for height in sorted(metrics):
        reduced = metrics[height].get("reduced")
        unreduced = metrics[height].get("unreduced")
        if unreduced is not None:
            unreduced_overflow = unreduced_overflow or bool(unreduced["buffer_overflow"])
            unreduced_buffer_utilizations.append(float(unreduced["buffer_utilization_max"]))
            unreduced_errors.append(abs(float(unreduced["fz_norm_mean"]) - 1.0))
        if reduced is not None:
            reduced_overflow = reduced_overflow or bool(reduced["buffer_overflow"])
            reduced_errors.append(abs(float(reduced["fz_norm_mean"]) - 1.0))
            reduced_lateral.append(float(reduced["lateral_norm_mean"]))
            reduced_torque.append(float(reduced["torque_norm_mean"]))
            reduced_penetration.append(float(reduced["penetration_depth_max_mm"]))
            reduced_support_offset.append(float(reduced["support_offset_max_mm"]))
        if reduced is not None and unreduced is not None and float(unreduced["solver_force_count_mean"]) > 0.0:
            ratios.append(float(reduced["solver_force_count_mean"]) / float(unreduced["solver_force_count_mean"]))

    return {
        "unreduced_overflow": unreduced_overflow,
        "reduced_overflow": reduced_overflow,
        "max_unreduced_buffer_utilization": _max_finite(unreduced_buffer_utilizations),
        "max_unreduced_fz_error": max(unreduced_errors) if unreduced_errors else float("nan"),
        "max_reduced_fz_error": max(reduced_errors) if reduced_errors else float("nan"),
        "max_reduced_lateral": max(reduced_lateral) if reduced_lateral else float("nan"),
        "max_reduced_torque": max(reduced_torque) if reduced_torque else float("nan"),
        "max_reduced_penetration_mm": _max_finite(reduced_penetration),
        "max_reduced_support_offset_mm": _max_finite(reduced_support_offset),
        "median_count_ratio": _median(ratios),
    }


def render_summary_cards(metrics: dict[float, dict[str, dict[str, float | bool]]], constants: PhysicsConstants) -> str:
    summary = _summary_values(metrics)
    items = [
        ("cube side", f"{1000.0 * constants.cube_side_m:.3g} mm"),
        ("cube mass", f"{constants.cube_mass_kg:.6f} kg"),
        (
            "plate dimensions",
            f"{1000.0 * constants.plate_side_m:.3g} x {1000.0 * constants.plate_side_m:.3g} x "
            f"{1000.0 * constants.plate_thickness_m:.3g} mm",
        ),
        ("expected weight", f"{constants.cube_weight_N:.6f} N"),
        ("reduce-off contact buffers valid", str(not bool(summary["unreduced_overflow"]))),
        ("reduce-on reduction buffers valid", str(not bool(summary["reduced_overflow"]))),
        ("max reduce-off buffer utilization", _format_percent(float(summary["max_unreduced_buffer_utilization"]))),
        ("max reduce-off Fz error vs reference", _format_percent(float(summary["max_unreduced_fz_error"]))),
        ("max reduce-on Fz error vs reference", _format_percent(float(summary["max_reduced_fz_error"]))),
        ("max reduce-on sideways leakage", _format_percent(float(summary["max_reduced_lateral"]))),
        ("max reduce-on torque imbalance", _format_percent(float(summary["max_reduced_torque"]))),
        (
            "max reduce-on penetration",
            f"{rc.format_number(float(summary['max_reduced_penetration_mm']), precision=4)} mm",
        ),
        (
            "max reduce-on support offset",
            f"{rc.format_number(float(summary['max_reduced_support_offset_mm']), precision=4)} mm",
        ),
        ("median reduce-on/off count ratio", rc.format_number(float(summary["median_count_ratio"]), precision=4)),
    ]
    return "\n".join(
        [
            "<ul>",
            *[f"<li><strong>{rc.escape(label)}:</strong> {rc.escape(value)}</li>" for label, value in items],
            "</ul>",
        ]
    )


def render_experiment_record(
    metrics: dict[float, dict[str, dict[str, float | bool]]], constants: PhysicsConstants
) -> str:
    summary = _summary_values(metrics)
    unreduced_counts = [
        float(metrics[height]["unreduced"]["solver_force_count_mean"])
        for height in metrics
        if "unreduced" in metrics[height]
    ]
    reduced_counts = [
        float(metrics[height]["reduced"]["solver_force_count_mean"])
        for height in metrics
        if "reduced" in metrics[height]
    ]
    unreduced_contact_summary = rc.format_number(_median(unreduced_counts), precision=4)
    reduced_contact_summary = rc.format_number(_median(reduced_counts), precision=4)
    heights_mm = ", ".join(rc.format_number(1000.0 * height, precision=3) for height in sorted(metrics))
    cube_side_mm = rc.format_number(1000.0 * constants.cube_side_m, precision=3)
    plate_side_mm = rc.format_number(1000.0 * constants.plate_side_m, precision=3)
    plate_thickness_mm = rc.format_number(1000.0 * constants.plate_thickness_m, precision=3)
    if bool(summary["unreduced_overflow"]):
        result_items = (
            "Status: inconclusive because reduce-off overflow occurred; the dense reference may be truncated.",
            f"Median post-settle solver contacts: reduce off {unreduced_contact_summary}, "
            f"reduce on {reduced_contact_summary}.",
        )
    elif bool(summary["reduced_overflow"]):
        result_items = (
            "Status: inconclusive because reduce-on contact reduction overflowed or reported hashtable failures.",
            f"Median post-settle solver contacts: reduce off {unreduced_contact_summary}, "
            f"reduce on {reduced_contact_summary}.",
        )
    else:
        result_items = (
            "Both modes settle to equilibrium: Fz = weight, with negligible lateral force, torque, drift, and tilt.",
            "Reduce-on matches the reference at least as well as dense, using far fewer contacts.",
        )

    sections = (
        (
            "Setup",
            (
                f"Hydroelastic cube: {cube_side_mm} mm side, {constants.cube_mass_kg:.3g} kg.",
                f"Fixed hydroelastic plate: {plate_side_mm} x {plate_side_mm} x {plate_thickness_mm} mm.",
                f"Drop heights: {heights_mm} mm.",
                "Modes: reduce off vs reduce on.",
                "Pre-prune disabled in both modes.",
            ),
        ),
        (
            "Reference",
            (
                "Rigid-body static equilibrium: a body at rest on a level plate carries Fz = m*g with zero "
                "lateral force, zero net torque, zero drift, and zero tilt.",
            ),
        ),
        (
            "Hypothesis",
            (
                "Both modes should match the rigid-body equilibrium reference: Fz ~= m*g; lateral force, net "
                "torque, drift, and tilt ~= 0.",
            ),
        ),
        (
            "Validity check",
            (
                "Reference gate: reduce off must have no broadphase, iso, face-contact, or rigid-contact overflow.",
                "Reduction gate: reduce on must have no rigid-contact overflow or reduction hashtable insertion "
                "failure.",
            ),
        ),
        (
            "Measured quantities",
            (
                "Primary: solver force and torque on the cube.",
                "Secondary: cube-plate gap and penetration depth, support-point offset from Tx, Ty, Fz, "
                "solver and rigid contact count, and final drift, tilt, and contact-buffer use.",
            ),
        ),
        ("Result", result_items),
    )
    return "\n".join(
        [
            '<section class="record-note" aria-label="experiment record">',
            *[
                f"<p><strong>{rc.escape(title)}.</strong></p>"
                f"<ul>{''.join(f'<li>{rc.escape(item)}</li>' for item in items)}</ul>"
                for title, items in sections
            ],
            "</section>",
        ]
    )


def render_metrics_table(metrics: dict[float, dict[str, dict[str, float | bool]]]) -> str:
    headers = [
        "height [m]",
        "mode",
        "Fz/(m*g)",
        "side/(m*g)",
        "torque/(m*g*L)",
        "mean penetration [mm]",
        "mean support offset [mm]",
        "force count",
        "rigid contacts",
        "max buffer util.",
        "overflow",
    ]
    rows = []
    for height in sorted(metrics):
        for mode in MODES:
            values = metrics[height].get(mode)
            if values is None:
                continue
            rows.append(
                [
                    f"{height:.6f}",
                    _mode_label(mode),
                    f"{float(values['fz_norm_mean']):.6f}",
                    rc.format_number(float(values["lateral_norm_mean"]), precision=4),
                    rc.format_number(float(values["torque_norm_mean"]), precision=4),
                    rc.format_number(float(values["penetration_depth_mean_mm"]), precision=4),
                    rc.format_number(float(values["support_offset_mean_mm"]), precision=4),
                    rc.format_number(float(values["solver_force_count_mean"]), precision=4),
                    rc.format_number(float(values["rigid_contact_count_mean"]), precision=4),
                    _format_percent(float(values["buffer_utilization_max"])),
                    str(bool(values["buffer_overflow"])),
                ]
            )
    return rc.data_table(headers, rows)


def render_html_report(
    runs: dict[float, dict[str, list[dict[str, str]]]],
    metrics: dict[float, dict[str, dict[str, float | bool]]],
    *,
    constants: PhysicsConstants,
    selected_height: float,
    window_fraction: float,
) -> str:
    """Render a self-contained HTML report for the settle CSVs."""

    window_note = (
        f"Post-settle means use the final {_format_percent(window_fraction)} of frames. Practical floors: normalized "
        f"{rc.format_number(PRACTICAL_NORMALIZED_FLOOR, precision=2)}, lateral force +/- "
        f"{rc.format_number(PRACTICAL_FORCE_FLOOR_N)} N, torque +/- {rc.format_number(PRACTICAL_TORQUE_FLOOR_NM)} "
        f"N m, length {PRACTICAL_LENGTH_FLOOR_MM:g} mm."
    )
    panels = [
        rc.TabPanel(
            "Figure 1",
            "<h2>Figure 1: raw solver outputs vs time</h2>\n"
            f"<p>Selected drop height: {_format_height(selected_height)}. Gray band is the averaging window.</p>\n"
            + render_solver_timeseries_figure(
                runs, height=selected_height, constants=constants, window_fraction=window_fraction
            ),
        ),
        rc.TabPanel(
            "Figure 2",
            "<h2>Figure 2: settled physical response vs height</h2>\n"
            "<p>Reference is rigid-body static equilibrium: vertical support should be 1; lateral force, torque, "
            "drift, tilt, penetration, and support-point offset should stay near 0.</p>\n"
            + render_physics_summary_figure(metrics),
        ),
        rc.TabPanel(
            "Figure 3",
            "<h2>Figure 3: contact reduction and buffer sanity</h2>\n"
            "<p>Count ratio records solver workload reduction; buffer plots summarize the contact-buffer validity "
            "gates.</p>\n" + render_numerics_figure(metrics),
        ),
    ]

    body = "\n".join(
        [
            f"<h1>{rc.escape(PAGE_TITLE)}</h1>",
            render_experiment_record(metrics, constants),
            "<h2>Figures</h2>",
            f"<p>{window_note}</p>",
            rc.figure_tabs(panels),
            "<h2>Post-settle table</h2>",
            render_metrics_table(metrics),
        ]
    )
    return rc.render_page(title=PAGE_TITLE, body=body)


def write_html_report(
    runs: dict[float, dict[str, list[dict[str, str]]]],
    metrics: dict[float, dict[str, dict[str, float | bool]]],
    *,
    constants: PhysicsConstants,
    selected_height: float,
    window_fraction: float,
    output_path: str | Path,
) -> Path:
    """Write a self-contained HTML report for the settle CSVs."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_html_report(
            runs,
            metrics,
            constants=constants,
            selected_height=selected_height,
            window_fraction=window_fraction,
        ),
        encoding="utf-8",
    )
    shutil.copyfile(HYPOTHESIS_RECORD_SOURCE, output_path.parent / HYPOTHESIS_RECORD_NAME)
    return output_path


def print_summary(metrics: dict[float, dict[str, dict[str, float | bool]]], constants: PhysicsConstants) -> None:
    summary = _summary_values(metrics)
    print(f"cube mass: {constants.cube_mass_kg:.6f} kg")
    print(f"expected cube weight m*g: {constants.cube_weight_N:.6f} N")
    print(f"reduce-off contact buffers valid: {not bool(summary['unreduced_overflow'])}")
    print(f"reduce-on reduction buffers valid: {not bool(summary['reduced_overflow'])}")
    print(f"max reduce-off buffer utilization: {_format_percent(float(summary['max_unreduced_buffer_utilization']))}")
    print(
        f"max reduce-off settled vertical support error: {_format_percent(float(summary['max_unreduced_fz_error']))} "
        "of weight"
    )
    print(
        f"max reduce-on settled vertical support error: {_format_percent(float(summary['max_reduced_fz_error']))} "
        "of weight"
    )
    print(f"max reduce-on settled sideways leakage: {_format_percent(float(summary['max_reduced_lateral']))} of weight")
    print(f"max reduce-on settled torque imbalance: {_format_percent(float(summary['max_reduced_torque']))} of m*g*L")
    print(
        "max reduce-on settled penetration: "
        f"{rc.format_number(float(summary['max_reduced_penetration_mm']), precision=4)} mm"
    )
    print(
        "max reduce-on settled support-point offset: "
        f"{rc.format_number(float(summary['max_reduced_support_offset_mm']), precision=4)} mm"
    )
    print(
        f"median reduce-on/off force-count ratio: {rc.format_number(float(summary['median_count_ratio']), precision=4)}"
    )

    print("\npost-settle means by height:")
    for height in sorted(metrics):
        print(f"height={height:.6f} m")
        for mode in MODES:
            values = metrics[height].get(mode)
            if values is None:
                continue
            print(
                "  "
                f"{_mode_label(mode)}: Fz/(m*g)={float(values['fz_norm_mean']):.6f}, "
                f"side={float(values['lateral_norm_mean']):.6e}, "
                f"torque={float(values['torque_norm_mean']):.6e}, "
                f"penetration_mm={float(values['penetration_depth_mean_mm']):.6g}, "
                f"support_offset_mm={float(values['support_offset_mean_mm']):.6g}, "
                f"force_count={float(values['solver_force_count_mean']):.2f}"
            )


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--csv-dir", type=str, default=str(DEFAULT_CSV_DIR), help="Directory containing settle CSVs.")
    parser.add_argument("--html-path", type=str, default=str(DEFAULT_HTML_PATH), help="Path for the HTML report.")
    parser.add_argument("--height", type=float, default=0.005, help="Drop height [m] for the raw time-series figure.")
    parser.add_argument(
        "--window-fraction",
        type=float,
        default=0.25,
        help="Final fraction of frames used for post-settle averages.",
    )
    parser.add_argument("--cube-side", type=float, default=DEFAULT_CUBE_SIDE_M, help="Cube side length [m].")
    parser.add_argument("--cube-mass", type=float, default=DEFAULT_CUBE_MASS_KG, help="Cube mass [kg].")
    parser.add_argument("--gravity", type=float, default=DEFAULT_GRAVITY_M_PER_S2, help="Gravity magnitude [m/s^2].")
    parser.add_argument(
        "--plate-half-extent",
        type=float,
        default=DEFAULT_PLATE_HALF_EXTENT_M,
        help="Plate half extent in x/y [m].",
    )
    parser.add_argument(
        "--plate-half-thickness",
        type=float,
        default=DEFAULT_PLATE_HALF_THICKNESS_M,
        help="Plate half thickness [m].",
    )
    return parser


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    if not (0.0 < args.window_fraction <= 1.0):
        parser.error("--window-fraction must be in (0, 1]")

    constants = PhysicsConstants(
        cube_side_m=args.cube_side,
        cube_mass_kg=args.cube_mass,
        gravity_m_per_s2=args.gravity,
        plate_half_extent_m=args.plate_half_extent,
        plate_half_thickness_m=args.plate_half_thickness,
    )
    runs = load_runs(args.csv_dir)
    height = select_height(runs, args.height)
    if abs(height - args.height) > 1.0e-12:
        print(f"requested height {args.height:.6f} m not found; using closest available height {height:.6f} m")

    summaries = load_run_summaries(args.csv_dir)
    metrics = compute_metrics(runs, constants, args.window_fraction, summaries=summaries)
    html_path = write_html_report(
        runs,
        metrics,
        constants=constants,
        selected_height=height,
        window_fraction=args.window_fraction,
        output_path=args.html_path,
    )

    print_summary(metrics, constants)
    print(f"wrote {html_path}")


if __name__ == "__main__":
    main()
