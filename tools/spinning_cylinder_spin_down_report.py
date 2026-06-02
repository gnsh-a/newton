# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Build a self-contained HTML report from spinning-cylinder spin-down CSVs.

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

DEFAULT_CSV_DIR = Path("output") / "H4_spinning_cylinder_spin_down"
DEFAULT_HTML_PATH = DEFAULT_CSV_DIR / "spinning_cylinder_spin_down_report.html"
HYPOTHESIS_RECORD_NAME = "H4_spinning_cylinder_spin_down_hypothesis.md"
HYPOTHESIS_RECORD_SOURCE = Path(__file__).resolve().parents[1] / "hypothesis" / HYPOTHESIS_RECORD_NAME
TIMESERIES_CSV = "spin_down_timeseries.csv"
SUMMARY_CSV = "spin_down_summary.csv"
VARIATIONS_DIR = DEFAULT_CSV_DIR / "variations"
DEFAULT_SDF_SWEEP_DIRS = (
    (16.0, VARIATIONS_DIR / "sdf16"),
    (24.0, VARIATIONS_DIR / "sdf24"),
    (32.0, VARIATIONS_DIR / "sdf32"),
    (48.0, VARIATIONS_DIR / "sdf48"),
)

PAGE_TITLE = "H4: Spinning Cylinder Yaw-Torque Contact Reduction"
OMEGA_COLORS = ("#2563eb", "#059669", "#f97316", "#7c3aed", "#dc2626", "#0891b2")


@dataclass(frozen=True)
class SdfSweepRow:
    """One SDF-resolution summary row for one yaw rate."""

    sdf_resolution: float
    initial_omega: float
    expected_stop_time: float
    expected_abs_torque: float
    reduce_off_stop_time: float
    reduce_on_stop_time: float
    reduce_off_abs_torque: float
    reduce_on_abs_torque: float
    reduce_off_contacts: float
    reduce_on_contacts: float
    reduce_off_stopped: bool
    reduce_on_stopped: bool
    reduce_off_valid: bool
    reduce_on_valid: bool


def load_timeseries(csv_dir: str | Path) -> dict[float, dict[str, list[dict[str, str]]]]:
    """Load time-series rows grouped by initial yaw rate and mode."""

    path = Path(csv_dir) / TIMESERIES_CSV
    if not path.exists():
        raise FileNotFoundError(f"missing time-series CSV: {path}")
    grouped: dict[float, dict[str, list[dict[str, str]]]] = {}
    for row in rc.read_csv(path):
        omega0 = rc.as_float(row, "initial_omega_rad_per_s")
        grouped.setdefault(omega0, {}).setdefault(row["mode"], []).append(row)
    for mode_rows in grouped.values():
        for rows in mode_rows.values():
            rows.sort(key=lambda row: rc.as_float(row, "time_s"))
    if not grouped:
        raise ValueError(f"time-series CSV has no rows: {path}")
    return grouped


def load_summaries(csv_dir: str | Path) -> dict[float, dict[str, dict[str, str]]]:
    """Load summary rows grouped by initial yaw rate and mode."""

    path = Path(csv_dir) / SUMMARY_CSV
    if not path.exists():
        raise FileNotFoundError(f"missing summary CSV: {path}")
    grouped: dict[float, dict[str, dict[str, str]]] = {}
    for row in rc.read_csv(path):
        omega0 = rc.as_float(row, "initial_omega_rad_per_s")
        grouped.setdefault(omega0, {})[row["mode"]] = row
    if not grouped:
        raise ValueError(f"summary CSV has no rows: {path}")
    return grouped


def load_sdf_sweep(
    sweep_dirs: tuple[tuple[float, str | Path], ...] | list[tuple[float, str | Path]],
) -> list[SdfSweepRow]:
    """Load optional SDF-resolution sweep summaries.

    Missing directories are ignored so the base report can still be generated
    from one experiment output folder.
    """

    rows: list[SdfSweepRow] = []
    for sdf_resolution, csv_dir in sweep_dirs:
        summary_path = Path(csv_dir) / SUMMARY_CSV
        if not summary_path.exists():
            continue
        summaries = load_summaries(csv_dir)
        for omega0 in sorted(summaries):
            off = summaries[omega0].get("unreduced")
            on = summaries[omega0].get("reduced")
            if off is None or on is None:
                continue
            off_contacts = rc.as_float(off, "mean_solver_force_count")
            on_contacts = rc.as_float(on, "mean_solver_force_count")
            rows.append(
                SdfSweepRow(
                    sdf_resolution=float(sdf_resolution),
                    initial_omega=omega0,
                    expected_stop_time=rc.as_float(off, "expected_uniform_stop_time_s"),
                    expected_abs_torque=abs(rc.as_float(off, "expected_uniform_torque_z_Nm")),
                    reduce_off_stop_time=rc.as_float(off, "stop_time_s"),
                    reduce_on_stop_time=rc.as_float(on, "stop_time_s"),
                    reduce_off_abs_torque=rc.as_float(off, "mean_abs_solver_tz_active_Nm"),
                    reduce_on_abs_torque=rc.as_float(on, "mean_abs_solver_tz_active_Nm"),
                    reduce_off_contacts=off_contacts,
                    reduce_on_contacts=on_contacts,
                    reduce_off_stopped=rc.as_bool(off["stopped"]),
                    reduce_on_stopped=rc.as_bool(on["stopped"]),
                    reduce_off_valid=(not rc.as_bool(off["buffer_overflow"])) and off_contacts > 0.0,
                    reduce_on_valid=(not rc.as_bool(on["buffer_overflow"])) and on_contacts > 0.0,
                )
            )
    return rows


def select_omega(runs: dict[float, dict[str, list[dict[str, str]]]], requested_omega: float | None) -> float:
    """Return the closest available yaw rate, defaulting to the fastest run."""

    omegas = sorted(runs)
    if not omegas:
        raise ValueError("no initial yaw rates available")
    if requested_omega is None:
        return float(omegas[-1])
    return float(min(omegas, key=lambda omega0: abs(omega0 - requested_omega)))


def _reference_omega_series(summary: dict[str, str], times: list[float]) -> list[float]:
    omega0 = rc.as_float(summary, "initial_omega_rad_per_s")
    alpha = rc.as_float(summary, "expected_uniform_angular_accel_rad_per_s2")
    return [max(omega0 + alpha * t, 0.0) / omega0 for t in times]


def _figure_spin_history(
    runs: dict[float, dict[str, list[dict[str, str]]]],
    summaries: dict[float, dict[str, dict[str, str]]],
    *,
    selected_omega: float,
) -> str:
    rows_by_mode = runs[selected_omega]
    reference_summary = summaries[selected_omega].get("unreduced") or next(iter(summaries[selected_omega].values()))
    reference_times: list[float] = []
    for mode in rc.MODES:
        if mode in rows_by_mode:
            reference_times = [rc.as_float(row, "time_s") for row in rows_by_mode[mode]]
            break

    omega_series: list[rc.Series] = [
        rc.Series(
            reference_times,
            _reference_omega_series(reference_summary, reference_times),
            "uniform-pressure reference",
            rc.REFERENCE_COLOR,
            dash="5 4",
        )
    ]
    torque_series: list[rc.Series] = [
        rc.Series(
            reference_times,
            [1000.0 * rc.as_float(reference_summary, "expected_uniform_torque_z_Nm") for _ in reference_times],
            "uniform-pressure reference",
            rc.REFERENCE_COLOR,
            dash="5 4",
        )
    ]

    for mode in rc.MODES:
        rows = rows_by_mode.get(mode)
        if not rows:
            continue
        times = [rc.as_float(row, "time_s") for row in rows]
        omega_series.append(
            rc.Series(
                times,
                [rc.as_float(row, "omega_over_omega0") for row in rows],
                rc.MODE_LABELS[mode],
                rc.MODE_COLORS[mode],
            )
        )
        torque_series.append(
            rc.Series(
                times,
                [1000.0 * rc.as_float(row, "solver_tz_Nm") for row in rows],
                rc.MODE_LABELS[mode],
                rc.MODE_COLORS[mode],
            )
        )

    all_times = [x for series in omega_series for x in series.xs]
    all_torques = [y for series in torque_series for y in series.ys]
    return rc.figure_grid(
        [
            rc.Figure(
                title=f"Spin decay, omega0 = {rc.format_number(selected_omega)} rad/s",
                xlabel="time [s]",
                ylabel="omega / omega0",
                series=omega_series,
                x_range=rc.padded_range(all_times, include=(0.0,), floor_span=0.05),
                y_range=rc.padded_range(
                    [y for series in omega_series for y in series.ys], include=(0.0, 1.0), floor_span=0.2
                ),
            ),
            rc.Figure(
                title="Solver yaw torque",
                xlabel="time [s]",
                ylabel="Tz [mN m]",
                series=torque_series,
                x_range=rc.padded_range(all_times, include=(0.0,), floor_span=0.05),
                y_range=rc.padded_range(all_torques, include=(0.0,), floor_span=0.5),
            ),
        ]
    )


def _figure_contact_history(
    runs: dict[float, dict[str, list[dict[str, str]]]],
    *,
    selected_omega: float,
) -> str:
    rows_by_mode = runs[selected_omega]
    contact_series = rc.mode_series(rows_by_mode, x_key="time_s", y_key="solver_force_count")
    fz_series = rc.mode_series(rows_by_mode, x_key="time_s", y_key="solver_fz_N")
    all_times = [x for series in contact_series for x in series.xs]
    all_contacts = [y for series in contact_series for y in series.ys]
    all_fz = [y for series in fz_series for y in series.ys]
    return rc.figure_grid(
        [
            rc.Figure(
                title="Solver contact count",
                xlabel="time [s]",
                ylabel="contacts",
                series=contact_series,
                x_range=rc.padded_range(all_times, include=(0.0,), floor_span=0.05),
                y_range=rc.padded_range(all_contacts, include=(0.0,), floor_span=50.0),
            ),
            rc.Figure(
                title="Solver vertical support force",
                xlabel="time [s]",
                ylabel="Fz [N]",
                series=fz_series,
                x_range=rc.padded_range(all_times, include=(0.0,), floor_span=0.05),
                y_range=rc.padded_range(all_fz, include=(0.0,), floor_span=0.05),
            ),
        ]
    )


def _summary_series(
    summaries: dict[float, dict[str, dict[str, str]]],
    *,
    mode: str,
    key: str,
    scale: float = 1.0,
) -> rc.Series:
    xs = [omega0 for omega0 in sorted(summaries) if mode in summaries[omega0]]
    ys = [scale * rc.as_float(summaries[omega0][mode], key) for omega0 in xs]
    return rc.Series(xs, ys, rc.MODE_LABELS[mode], rc.MODE_COLORS[mode], draw_marker=True)


def _reference_summary_series(
    summaries: dict[float, dict[str, dict[str, str]]],
    *,
    key: str,
    scale: float = 1.0,
    label: str = "uniform-pressure reference",
) -> rc.Series:
    xs = []
    ys = []
    for omega0 in sorted(summaries):
        row = summaries[omega0].get("unreduced") or next(iter(summaries[omega0].values()))
        xs.append(omega0)
        ys.append(scale * rc.as_float(row, key))
    return rc.Series(xs, ys, label, rc.REFERENCE_COLOR, draw_marker=True, dash="5 4")


def _figure_sweep_summary(summaries: dict[float, dict[str, dict[str, str]]]) -> str:
    omegas = sorted(summaries)
    x_range = rc.padded_range(omegas, include=(0.0,), floor_span=5.0)
    x_ticks = omegas if len(omegas) <= 6 else None

    stop_series = [
        _reference_summary_series(summaries, key="expected_uniform_stop_time_s"),
        _summary_series(summaries, mode="unreduced", key="stop_time_s"),
        _summary_series(summaries, mode="reduced", key="stop_time_s"),
    ]
    torque_series = [
        _reference_summary_series(summaries, key="expected_uniform_torque_z_Nm", scale=1000.0),
        _summary_series(summaries, mode="unreduced", key="mean_solver_tz_active_Nm", scale=1000.0),
        _summary_series(summaries, mode="reduced", key="mean_solver_tz_active_Nm", scale=1000.0),
    ]
    ratio_xs = []
    ratio_ys = []
    for omega0 in omegas:
        off = summaries[omega0].get("unreduced")
        on = summaries[omega0].get("reduced")
        if off is None or on is None:
            continue
        off_count = rc.as_float(off, "mean_solver_force_count")
        ratio_xs.append(omega0)
        ratio_ys.append(rc.as_float(on, "mean_solver_force_count") / off_count if off_count > 0.0 else float("nan"))
    ratio_series = [rc.Series(ratio_xs, ratio_ys, "reduce on / reduce off", "#059669", draw_marker=True)]

    return rc.figure_grid(
        [
            rc.Figure(
                title="Spin stop time",
                xlabel="initial yaw rate [rad/s]",
                ylabel="time [s]",
                series=stop_series,
                x_range=x_range,
                y_range=rc.padded_range(
                    [y for series in stop_series for y in series.ys], include=(0.0,), floor_span=0.05
                ),
                x_ticks=x_ticks,
            ),
            rc.Figure(
                title="Mean active yaw torque",
                xlabel="initial yaw rate [rad/s]",
                ylabel="Tz [mN m]",
                series=torque_series,
                x_range=x_range,
                y_range=rc.padded_range(
                    [y for series in torque_series for y in series.ys], include=(0.0,), floor_span=0.5
                ),
                x_ticks=x_ticks,
            ),
            rc.Figure(
                title="Contact count ratio",
                xlabel="initial yaw rate [rad/s]",
                ylabel="ratio",
                series=ratio_series,
                x_range=x_range,
                y_range=rc.padded_range(ratio_ys, include=(0.0,), floor_span=0.05),
                x_ticks=x_ticks,
            ),
        ]
    )


def _omega_color(omega0: float, omegas: list[float]) -> str:
    try:
        return OMEGA_COLORS[omegas.index(omega0) % len(OMEGA_COLORS)]
    except ValueError:
        return OMEGA_COLORS[0]


def _figure_sdf_sweep(sdf_sweep: list[SdfSweepRow]) -> str:
    if not sdf_sweep:
        return ""

    sdf_values = sorted({row.sdf_resolution for row in sdf_sweep})
    omega_values = sorted({row.initial_omega for row in sdf_sweep})
    by_omega: dict[float, dict[float, SdfSweepRow]] = {}
    for row in sdf_sweep:
        by_omega.setdefault(row.initial_omega, {})[row.sdf_resolution] = row

    stop_series = [
        rc.Series(sdf_values, [0.0 for _ in sdf_values], "uniform-pressure reference", rc.REFERENCE_COLOR, dash="5 4")
    ]
    torque_series = [
        rc.Series(sdf_values, [0.0 for _ in sdf_values], "uniform-pressure reference", rc.REFERENCE_COLOR, dash="5 4")
    ]
    ratio_series: list[rc.Series] = []

    for omega0 in omega_values:
        color = _omega_color(omega0, omega_values)
        omega_rows = by_omega[omega0]
        off_stop: list[float] = []
        on_stop: list[float] = []
        off_torque: list[float] = []
        on_torque: list[float] = []
        ratios: list[float] = []
        for sdf in sdf_values:
            row = omega_rows.get(sdf)
            if row is None:
                off_stop.append(float("nan"))
                on_stop.append(float("nan"))
                off_torque.append(float("nan"))
                on_torque.append(float("nan"))
                ratios.append(float("nan"))
                continue

            off_stop.append(
                row.reduce_off_stop_time / row.expected_stop_time - 1.0
                if row.reduce_off_valid and row.reduce_off_stopped and row.expected_stop_time > 0.0
                else float("nan")
            )
            on_stop.append(
                row.reduce_on_stop_time / row.expected_stop_time - 1.0
                if row.reduce_on_valid and row.reduce_on_stopped and row.expected_stop_time > 0.0
                else float("nan")
            )
            off_torque.append(
                row.reduce_off_abs_torque / row.expected_abs_torque - 1.0
                if row.reduce_off_valid and row.expected_abs_torque > 0.0
                else float("nan")
            )
            on_torque.append(
                row.reduce_on_abs_torque / row.expected_abs_torque - 1.0
                if row.reduce_on_valid and row.expected_abs_torque > 0.0
                else float("nan")
            )
            ratios.append(
                row.reduce_on_contacts / row.reduce_off_contacts
                if row.reduce_on_valid and row.reduce_off_valid and row.reduce_off_contacts > 0.0
                else float("nan")
            )

        stop_series.extend(
            [
                rc.Series(sdf_values, off_stop, f"omega0 {rc.format_number(omega0)} off", color, draw_marker=True),
                rc.Series(
                    sdf_values, on_stop, f"omega0 {rc.format_number(omega0)} on", color, draw_marker=True, dash="4 4"
                ),
            ]
        )
        torque_series.extend(
            [
                rc.Series(sdf_values, off_torque, f"omega0 {rc.format_number(omega0)} off", color, draw_marker=True),
                rc.Series(
                    sdf_values, on_torque, f"omega0 {rc.format_number(omega0)} on", color, draw_marker=True, dash="4 4"
                ),
            ]
        )
        ratio_series.append(
            rc.Series(sdf_values, ratios, f"omega0 {rc.format_number(omega0)}", color, draw_marker=True)
        )

    all_stop = [y for series in stop_series for y in series.ys]
    all_torque = [y for series in torque_series for y in series.ys]
    all_ratios = [y for series in ratio_series for y in series.ys]
    return rc.figure_grid(
        [
            rc.Figure(
                title="Stop-time error vs SDF resolution",
                xlabel="SDF max resolution",
                ylabel="relative error",
                series=stop_series,
                x_range=rc.padded_range(sdf_values, floor_span=4.0),
                y_range=rc.padded_range(all_stop, include=(0.0,), floor_span=0.1),
                x_ticks=sdf_values,
            ),
            rc.Figure(
                title="Mean yaw-torque error vs SDF resolution",
                xlabel="SDF max resolution",
                ylabel="relative error",
                series=torque_series,
                x_range=rc.padded_range(sdf_values, floor_span=4.0),
                y_range=rc.padded_range(all_torque, include=(0.0,), floor_span=0.2),
                x_ticks=sdf_values,
            ),
            rc.Figure(
                title="Contact-count ratio vs SDF resolution",
                xlabel="SDF max resolution",
                ylabel="reduce on / reduce off",
                series=ratio_series,
                x_range=rc.padded_range(sdf_values, floor_span=4.0),
                y_range=rc.padded_range(all_ratios, include=(0.0,), floor_span=0.02),
                x_ticks=sdf_values,
            ),
        ]
    )


def _result_bullets(summaries: dict[float, dict[str, dict[str, str]]]) -> list[str]:
    bullets = []
    valid = all(
        not rc.as_bool(row["buffer_overflow"]) for mode_rows in summaries.values() for row in mode_rows.values()
    )
    if valid:
        bullets.append("All reported runs passed the contact-buffer validity gate.")
    else:
        bullets.append("At least one reported run overflowed a contact buffer; treat that comparison as inconclusive.")

    for omega0 in sorted(summaries):
        off = summaries[omega0].get("unreduced")
        on = summaries[omega0].get("reduced")
        if off is None or on is None:
            continue
        expected_stop = rc.as_float(off, "expected_uniform_stop_time_s")
        off_stop = rc.as_float(off, "stop_time_s")
        on_stop = rc.as_float(on, "stop_time_s")
        off_torque = rc.as_float(off, "mean_abs_solver_tz_active_Nm")
        on_torque = rc.as_float(on, "mean_abs_solver_tz_active_Nm")
        expected_torque = abs(rc.as_float(off, "expected_uniform_torque_z_Nm"))
        off_count = rc.as_float(off, "mean_solver_force_count")
        on_count = rc.as_float(on, "mean_solver_force_count")
        ratio = on_count / off_count if off_count > 0.0 else float("nan")
        bullets.append(
            "omega0 = "
            f"{rc.format_number(omega0)} rad/s: stop time reference/off/on = "
            f"{rc.format_number(expected_stop)} / {rc.format_number(off_stop)} / {rc.format_number(on_stop)} s; "
            f"mean |Tz| reference/off/on = {rc.format_number(1000.0 * expected_torque)} / "
            f"{rc.format_number(1000.0 * off_torque)} / {rc.format_number(1000.0 * on_torque)} mN m; "
            f"contact ratio = {rc.format_percent(ratio)}."
        )

    return bullets


def _sdf_result_bullets(sdf_sweep: list[SdfSweepRow]) -> list[str]:
    if not sdf_sweep:
        return []

    bullets = []
    sdf_values = sorted({row.sdf_resolution for row in sdf_sweep})
    invalid_sdfs = []
    for sdf in sdf_values:
        rows = [row for row in sdf_sweep if row.sdf_resolution == sdf]
        if rows and all(not row.reduce_off_valid and not row.reduce_on_valid for row in rows):
            invalid_sdfs.append(sdf)
    if invalid_sdfs:
        bullets.append(
            "SDF "
            + ", ".join(rc.format_number(sdf) for sdf in invalid_sdfs)
            + " produced no loaded solver contacts in this setup, so those rows are invalid for H4 physics."
        )

    valid_rows = [row for row in sdf_sweep if row.reduce_off_valid and row.reduce_on_valid]
    ratios = [
        row.reduce_on_contacts / row.reduce_off_contacts
        for row in valid_rows
        if row.reduce_off_contacts > 0.0 and math.isfinite(row.reduce_on_contacts)
    ]
    if ratios:
        bullets.append(
            "Across valid SDF rows, reduce on used "
            f"{rc.format_percent(min(ratios))}-{rc.format_percent(max(ratios))} "
            "as many solver-force contacts as reduce off."
        )

    stopped_rows = [row for row in valid_rows if row.reduce_off_stopped and row.reduce_on_stopped]
    if stopped_rows:
        off_errors = [row.reduce_off_stop_time / row.expected_stop_time - 1.0 for row in stopped_rows]
        on_errors = [row.reduce_on_stop_time / row.expected_stop_time - 1.0 for row in stopped_rows]
        bullets.append(
            "For valid stopped runs, reduce-off stop-time error ranged from "
            f"{rc.format_percent(min(off_errors))} to {rc.format_percent(max(off_errors))}; "
            "reduce-on ranged from "
            f"{rc.format_percent(min(on_errors))} to {rc.format_percent(max(on_errors))}."
        )

    return bullets


def _build_html(
    *,
    runs: dict[float, dict[str, list[dict[str, str]]]],
    summaries: dict[float, dict[str, dict[str, str]]],
    sdf_sweep: list[SdfSweepRow],
    selected_omega: float,
    csv_dir: Path,
) -> str:
    omegas = ", ".join(rc.format_number(omega0) for omega0 in sorted(summaries))
    figure_4 = _figure_sdf_sweep(sdf_sweep)
    sdf_bullets = _sdf_result_bullets(sdf_sweep)

    panels = [
        rc.TabPanel(
            "Figure 1",
            "<p>Figure 1 checks the primary physics: spin decay and yaw torque versus time. If the reduced contacts "
            "lose patch lever arm, this is where the mismatch should appear first.</p>\n"
            + _figure_spin_history(runs, summaries, selected_omega=selected_omega),
        ),
        rc.TabPanel(
            "Figure 2",
            "<p>Figure 2 checks whether the contact set is loaded and whether reduction actually changes the solver "
            "contact count while maintaining vertical support.</p>\n"
            + _figure_contact_history(runs, selected_omega=selected_omega),
        ),
        rc.TabPanel(
            "Figure 3",
            "<p>Figure 3 summarizes stop time, mean active yaw torque, and contact-count ratio across the available "
            "initial yaw-rate sweep.</p>\n" + _figure_sweep_summary(summaries),
        ),
    ]
    if figure_4:
        panels.append(
            rc.TabPanel(
                "Figure 4",
                "<p>Figure 4 compares SDF resolution across the completed yaw-rate sweep. SDF 16 is retained as a "
                "failure case; SDF 64 is intentionally not included.</p>\n" + figure_4,
            )
        )

    sdf_result_section = f"<ul>\n{rc.bullet_list(sdf_bullets)}\n</ul>" if sdf_bullets else ""
    body = "\n".join(
        [
            f"<h1>{rc.escape(PAGE_TITLE)}</h1>",
            "<p>This experiment tests whether contact reduction preserves the yaw-friction torque of a flat spinning "
            "cylinder on a flat hydroelastic plate. Initial linear velocity is zero, and torsional and rolling "
            "friction are zero, so the measured spin decay must come from tangential solver forces distributed across "
            "the contact patch.</p>",
            "<p>The comparison is reduce off versus reduce on, with pre-prune off in both modes. The reference is the "
            "uniform-pressure disk estimate: <code>Tz = -(2/3) mu m g R sign(omega)</code> and "
            "<code>t_stop = (3/4) omega0 R / (mu g)</code>.</p>",
            f"<p>CSV data currently contains initial yaw rate(s): {omegas} rad/s. Figure time histories use "
            f"omega0 = {rc.format_number(selected_omega)} rad/s.</p>",
            "<h2>Measured Quantities</h2>",
            "<ul>",
            "<li>Solver yaw torque about the cylinder COM, computed from solver contact forces and contact-point "
            "lever arms.</li>",
            "<li>Yaw rate, normalized yaw rate, spin stop time, and integrated yaw impulse.</li>",
            "<li>Vertical support force, final drift, final tilt, and penetration depth.</li>",
            "<li>Solver contact count, raw face contact count, rigid contact count, and buffer validity flags.</li>",
            "</ul>",
            "<h2>Results</h2>",
            f"<ul>\n{rc.bullet_list(_result_bullets(summaries))}\n</ul>",
            sdf_result_section,
            "<h2>Figures</h2>",
            rc.figure_tabs(panels),
            f'<p class="meta">Generated from <code>{rc.escape(csv_dir / TIMESERIES_CSV)}</code> and '
            f"<code>{rc.escape(csv_dir / SUMMARY_CSV)}</code>.</p>",
        ]
    )
    return rc.render_page(title=PAGE_TITLE, body=body)


def write_html_report(
    *,
    csv_dir: str | Path = DEFAULT_CSV_DIR,
    output_path: str | Path = DEFAULT_HTML_PATH,
    selected_omega: float | None = None,
    sdf_sweep_dirs: tuple[tuple[float, str | Path], ...] | list[tuple[float, str | Path]] = DEFAULT_SDF_SWEEP_DIRS,
) -> Path:
    """Write a standalone HTML report and return its path."""

    csv_dir = Path(csv_dir)
    runs = load_timeseries(csv_dir)
    summaries = load_summaries(csv_dir)
    sdf_sweep = load_sdf_sweep(sdf_sweep_dirs)
    selected = select_omega(runs, selected_omega)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        _build_html(runs=runs, summaries=summaries, sdf_sweep=sdf_sweep, selected_omega=selected, csv_dir=csv_dir),
        encoding="utf-8",
    )
    shutil.copyfile(HYPOTHESIS_RECORD_SOURCE, output_path.parent / HYPOTHESIS_RECORD_NAME)
    return output_path


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--csv-dir", type=str, default=str(DEFAULT_CSV_DIR), help="Directory containing H4 CSV files.")
    parser.add_argument("--output", type=str, default=str(DEFAULT_HTML_PATH), help="HTML output path.")
    parser.add_argument("--selected-omega", type=float, default=None, help="Initial yaw rate for time-history figures.")
    parser.add_argument(
        "--no-sdf-sweep",
        action="store_true",
        help="Do not include the default SDF-resolution sweep figure.",
    )
    return parser


def main() -> None:
    args = create_parser().parse_args()
    output_path = write_html_report(
        csv_dir=args.csv_dir,
        output_path=args.output,
        selected_omega=args.selected_omega,
        sdf_sweep_dirs=() if args.no_sdf_sweep else DEFAULT_SDF_SWEEP_DIRS,
    )
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
