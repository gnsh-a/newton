# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Build a self-contained HTML report from sliding-spinning-cylinder CSVs.

This script intentionally does not import Newton. The Newton experiment writes
CSV files; this script reads those CSV files and renders the standalone report
using the shared report framework in :mod:`_report_common`.
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _report_common as rc

DEFAULT_CSV_DIR = Path("output") / "H5_sliding_spinning_cylinder"
DEFAULT_HTML_PATH = DEFAULT_CSV_DIR / "sliding_spinning_cylinder_report.html"
HYPOTHESIS_RECORD_NAME = "H5_sliding_spinning_cylinder_hypothesis.md"
HYPOTHESIS_RECORD_SOURCE = Path(__file__).resolve().parents[1] / "hypothesis" / HYPOTHESIS_RECORD_NAME
TIMESERIES_CSV = "sliding_spinning_timeseries.csv"
SUMMARY_CSV = "sliding_spinning_summary.csv"

PAGE_TITLE = "H5: Sliding-Spinning Cylinder"
SPEED_COLOR = "#0891b2"
SPIN_COLOR = "#7c3aed"
GRAVITY = 9.81
"""Gravitational acceleration [m/s^2]; matches the demo's GRAVITY."""


def load_timeseries(csv_dir: str | Path) -> dict[float, dict[str, list[dict[str, str]]]]:
    """Load time-series rows grouped by initial epsilon and mode."""

    path = Path(csv_dir) / TIMESERIES_CSV
    if not path.exists():
        raise FileNotFoundError(f"missing time-series CSV: {path}")
    grouped: dict[float, dict[str, list[dict[str, str]]]] = {}
    for row in rc.read_csv(path):
        epsilon0 = rc.as_float(row, "initial_epsilon")
        grouped.setdefault(epsilon0, {}).setdefault(row["mode"], []).append(row)
    for mode_rows in grouped.values():
        for rows in mode_rows.values():
            rows.sort(key=lambda row: rc.as_float(row, "time_s"))
    if not grouped:
        raise ValueError(f"time-series CSV has no rows: {path}")
    return grouped


def load_summaries(csv_dir: str | Path) -> dict[float, dict[str, dict[str, str]]]:
    """Load summary rows grouped by initial epsilon and mode."""

    path = Path(csv_dir) / SUMMARY_CSV
    if not path.exists():
        raise FileNotFoundError(f"missing summary CSV: {path}")
    grouped: dict[float, dict[str, dict[str, str]]] = {}
    for row in rc.read_csv(path):
        epsilon0 = rc.as_float(row, "initial_epsilon")
        grouped.setdefault(epsilon0, {})[row["mode"]] = row
    if not grouped:
        raise ValueError(f"summary CSV has no rows: {path}")
    return grouped


def select_epsilon(runs: dict[float, dict[str, list[dict[str, str]]]], requested_epsilon: float | None) -> float:
    """Return the closest available initial epsilon, defaulting to 1.0 if present."""

    epsilons = sorted(runs)
    if not epsilons:
        raise ValueError("no initial epsilons available")
    if requested_epsilon is None:
        return float(min(epsilons, key=lambda epsilon0: abs(epsilon0 - 1.0)))
    return float(min(epsilons, key=lambda epsilon0: abs(epsilon0 - requested_epsilon)))


def _figure_state_history(
    runs: dict[float, dict[str, list[dict[str, str]]]],
    *,
    selected_epsilon: float,
) -> str:
    # Overlay every eps0 (color = eps0, dash = mode, dotted = Farkas analytic);
    # opens on the representative eps0 with the others a dropdown away.
    epsilons = sorted(runs)
    default_group = f"eps0 = {rc.format_number(selected_epsilon)}"
    epsilon_series: list[rc.Series] = []
    speed_series: list[rc.Series] = []
    omega_series: list[rc.Series] = []
    all_times: list[float] = []
    for index, eps in enumerate(epsilons):
        rows_by_mode = runs[eps]
        group = f"eps0 = {rc.format_number(eps)}"
        color = rc.group_color(index)
        for mode in rc.MODES:
            rows = rows_by_mode.get(mode)
            if not rows:
                continue
            mode_dash = None if mode == "reduced" else "dash"
            solo = rc.SOLO_MODE_COLORS[mode]
            label = f"{group} · {rc.MODE_LABELS[mode]}"
            times = [rc.as_float(row, "time_s") for row in rows]
            all_times.extend(times)
            epsilon_series.append(
                rc.Series(
                    times,
                    [rc.as_float(row, "epsilon") for row in rows],
                    label,
                    color,
                    dash=mode_dash,
                    group=group,
                    solo_color=solo,
                )
            )
            speed_series.append(
                rc.Series(
                    times,
                    [rc.as_float(row, "horizontal_speed_m_per_s") for row in rows],
                    label,
                    color,
                    dash=mode_dash,
                    group=group,
                    solo_color=solo,
                )
            )
            omega_series.append(
                rc.Series(
                    times,
                    [rc.as_float(row, "cylinder_omega_z_rad_per_s") for row in rows],
                    label,
                    color,
                    dash=mode_dash,
                    group=group,
                    solo_color=solo,
                )
            )
        reference_rows = rows_by_mode.get("unreduced") or next(
            (rows for rows in (rows_by_mode.get(m) for m in rc.MODES) if rows), []
        )
        traj_t, traj_v, traj_w, traj_e = _farkas_state_trajectory(reference_rows)
        if traj_t:
            ref = rc.REFERENCE_COLOR
            epsilon_series.append(
                rc.Series(traj_t, traj_e, f"{group} · Farkas analytic", color, dash="dot", group=group, solo_color=ref)
            )
            speed_series.append(
                rc.Series(traj_t, traj_v, f"{group} · Farkas analytic", color, dash="dot", group=group, solo_color=ref)
            )
            omega_series.append(
                rc.Series(traj_t, traj_w, f"{group} · Farkas analytic", color, dash="dot", group=group, solo_color=ref)
            )

    time_range = rc.padded_range(all_times, include=(0.0,))

    def _hist(title: str, ylabel: str, series: list[rc.Series], y_range: tuple[float, float]) -> rc.Figure:
        return rc.Figure(
            title=title,
            xlabel="time [s]",
            ylabel=ylabel,
            series=series,
            x_range=time_range,
            y_range=y_range,
            selector="epsilon0",
            selector_default=default_group,
            height=360,
        )

    return rc.figure_grid(
        [
            _hist(
                "Coupling ratio",
                "epsilon = v / (R |omega|)",
                epsilon_series,
                rc.padded_range([y for series in epsilon_series for y in series.ys]),
            ),
            _hist(
                "Horizontal speed",
                "speed [m/s]",
                speed_series,
                rc.padded_range([y for series in speed_series for y in series.ys], include=(0.0,)),
            ),
            _hist(
                "Yaw rate",
                "omega_z [rad/s]",
                omega_series,
                rc.padded_range([y for series in omega_series for y in series.ys], include=(0.0,)),
            ),
        ]
    )


def _figure_solver_history(
    runs: dict[float, dict[str, list[dict[str, str]]]],
    *,
    selected_epsilon: float,
) -> str:
    epsilons = sorted(runs)
    default_group = f"eps0 = {rc.format_number(selected_epsilon)}"
    fx_series: list[rc.Series] = []
    tz_series: list[rc.Series] = []
    all_times: list[float] = []
    for index, eps in enumerate(epsilons):
        rows_by_mode = runs[eps]
        group = f"eps0 = {rc.format_number(eps)}"
        color = rc.group_color(index)
        for mode in rc.MODES:
            rows = rows_by_mode.get(mode)
            if not rows:
                continue
            mode_dash = None if mode == "reduced" else "dash"
            solo = rc.SOLO_MODE_COLORS[mode]
            label = f"{group} · {rc.MODE_LABELS[mode]}"
            times = [rc.as_float(row, "time_s") for row in rows]
            all_times.extend(times)
            fx_series.append(
                rc.Series(
                    times,
                    [rc.as_float(row, "solver_fx_N") for row in rows],
                    label,
                    color,
                    dash=mode_dash,
                    group=group,
                    solo_color=solo,
                )
            )
            tz_series.append(
                rc.Series(
                    times,
                    [1000.0 * rc.as_float(row, "solver_tz_Nm") for row in rows],
                    label,
                    color,
                    dash=mode_dash,
                    group=group,
                    solo_color=solo,
                )
            )

    time_range = rc.padded_range(all_times, include=(0.0,))

    def _hist(title: str, ylabel: str, series: list[rc.Series], y_range: tuple[float, float]) -> rc.Figure:
        return rc.Figure(
            title=title,
            xlabel="time [s]",
            ylabel=ylabel,
            series=series,
            x_range=time_range,
            y_range=y_range,
            selector="epsilon0",
            selector_default=default_group,
            height=360,
        )

    return rc.figure_grid(
        [
            _hist(
                "Solver force along sliding direction",
                "Fx [N]",
                fx_series,
                rc.padded_range([y for series in fx_series for y in series.ys], include=(0.0,)),
            ),
            _hist(
                "Solver yaw torque",
                "Tz [mN m]",
                tz_series,
                rc.padded_range([y for series in tz_series for y in series.ys], include=(0.0,)),
            ),
        ]
    )


def _farkas_factors(eps: float, *, n_r: int = 40, n_theta: int = 80) -> tuple[float, float]:
    """Farkas force F(eps) and torque T(eps) factors for a uniform-pressure disk.

    Computed by directly integrating the paper's Eq. (1) over the unit disk (the
    Coulomb traction direction -u/|u|, u = eps*e_v + e_w x r), rather than the
    elliptic-integral closed form. Limits: F(0)=0, F(inf)=1, T(0)=2/3, T(inf)=0.
    """
    f_acc = 0.0
    t_acc = 0.0
    for i in range(n_r):
        rho = (i + 0.5) / n_r
        for j in range(n_theta):
            theta = 2.0 * math.pi * (j + 0.5) / n_theta
            x = rho * math.cos(theta)
            y = rho * math.sin(theta)
            ux = eps - y
            uy = x
            speed = math.hypot(ux, uy)
            if speed < 1.0e-12:
                continue
            f_acc += (ux / speed) * rho
            t_acc += ((x * x + y * y - eps * y) / speed) * rho
    cell = (1.0 / n_r) * (2.0 * math.pi / n_theta) / math.pi
    return f_acc * cell, t_acc * cell


def _farkas_table(eps_max: float = 3.0, n: int = 140) -> tuple[list[float], list[float], list[float]]:
    """Tabulate F(eps), T(eps) on a uniform grid for fast lookup during integration."""
    grid = [eps_max * i / (n - 1) for i in range(n)]
    f_vals: list[float] = []
    t_vals: list[float] = []
    for e in grid:
        f_fac, t_fac = _farkas_factors(e)
        f_vals.append(f_fac)
        t_vals.append(t_fac)
    return grid, f_vals, t_vals


def _table_lookup(eps: float, grid: list[float], vals: list[float]) -> float:
    eps_max = grid[-1]
    if eps <= 0.0:
        return vals[0]
    if eps >= eps_max:
        return vals[-1]
    step = eps_max / (len(grid) - 1)
    j = int(eps / step)
    if j >= len(grid) - 1:
        return vals[-1]
    return vals[j] + (eps - grid[j]) / step * (vals[j + 1] - vals[j])


def _estimate_mu(rows: list[dict[str, str]]) -> float:
    """Recover the Coulomb mu from logged data: mu = |F_horiz| / (|Fz| * F(eps))."""
    samples = []
    for row in rows:
        eps = rc.as_float(row, "epsilon")
        fz = abs(rc.as_float(row, "solver_fz_N"))
        f_horiz = math.hypot(rc.as_float(row, "solver_fx_N"), rc.as_float(row, "solver_fy_N"))
        if not math.isfinite(eps) or eps <= 0.0 or eps > 3.0 or fz < 1.0e-3 or f_horiz < 1.0e-3:
            continue
        f_fac, _ = _farkas_factors(eps)
        if f_fac > 1.0e-3:
            samples.append(f_horiz / (fz * f_fac))
    if not samples:
        return float("nan")
    samples.sort()
    return samples[len(samples) // 2]


def _farkas_state_trajectory(
    rows: list[dict[str, str]],
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Integrate the Farkas ODEs (RK4) from the run's initial state.

    Constants come from the data -- radius R = spin_edge_speed/omega_abs, mu recovered
    from the friction ratio -- plus gravity. Returns (t, v, omega_z, epsilon).
    """
    if not rows:
        return [], [], [], []
    v = abs(rc.as_float(rows[0], "horizontal_speed_m_per_s"))
    omega0 = rc.as_float(rows[0], "cylinder_omega_z_rad_per_s")
    w = abs(omega0)
    omega_sign = 1.0 if omega0 >= 0.0 else -1.0
    radii = []
    for row in rows:
        om = abs(rc.as_float(row, "omega_abs_rad_per_s"))
        edge = rc.as_float(row, "spin_edge_speed_m_per_s")
        if om > 0.5 and edge > 0.0:
            radii.append(edge / om)
    mu = _estimate_mu(rows)
    t_end = rc.as_float(rows[-1], "time_s")
    if not radii or not math.isfinite(mu) or mu <= 0.0 or w <= 0.0 or t_end <= 0.0:
        return [], [], [], []
    radii.sort()
    radius = radii[len(radii) // 2]
    accel = mu * GRAVITY
    grid, f_grid, t_grid = _farkas_table()

    def deriv(vv: float, ww: float) -> tuple[float, float]:
        if vv <= 1.0e-9 and ww <= 1.0e-9:
            return 0.0, 0.0
        eps = vv / (radius * ww) if ww > 1.0e-9 else 3.0
        return -accel * _table_lookup(eps, grid, f_grid), -(2.0 * accel / radius) * _table_lookup(eps, grid, t_grid)

    n = 400
    dt = t_end / n
    v_init, w_init = v, w
    ts, vs, ws, es = [], [], [], []
    t = 0.0
    for _ in range(n + 1):
        ts.append(t)
        vs.append(v)
        ws.append(omega_sign * w)
        # eps = v/(R*w) is 0/0 near the simultaneous stop; mask the noisy tail.
        es.append(v / (radius * w) if (v > 0.03 * v_init and w > 0.03 * w_init) else float("nan"))
        k1v, k1w = deriv(v, w)
        k2v, k2w = deriv(v + 0.5 * dt * k1v, w + 0.5 * dt * k1w)
        k3v, k3w = deriv(v + 0.5 * dt * k2v, w + 0.5 * dt * k2w)
        k4v, k4w = deriv(v + dt * k3v, w + dt * k3w)
        v = max(v + dt / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v), 0.0)
        w = max(w + dt / 6.0 * (k1w + 2.0 * k2w + 2.0 * k3w + k4w), 0.0)
        t += dt
    return ts, vs, ws, es


def _figure_coupling(runs: dict[float, dict[str, list[dict[str, str]]]]) -> str:
    """Pointwise Farkas check: measured torque/force lever ratio vs T(eps)/F(eps).

    The radius is recovered from logged data (R = spin_edge_speed / omega_abs), and
    the mu*Fn factor cancels in the ratio, so no friction coefficient, mass, or
    geometry constant is hard-coded here.
    """
    coupling_series: list[rc.Series] = []
    eps_all: list[float] = []
    for mode in rc.MODES:
        xs: list[float] = []
        ys: list[float] = []
        for epsilon0 in sorted(runs):
            for row in runs[epsilon0].get(mode, []):
                eps = rc.as_float(row, "epsilon")
                omega_abs = rc.as_float(row, "omega_abs_rad_per_s")
                edge = rc.as_float(row, "spin_edge_speed_m_per_s")
                f_horiz = math.hypot(rc.as_float(row, "solver_fx_N"), rc.as_float(row, "solver_fy_N"))
                tz = abs(rc.as_float(row, "solver_tz_Nm"))
                if not math.isfinite(eps) or eps <= 0.0 or eps > 3.0:
                    continue
                if omega_abs < 0.5 or edge <= 0.0 or f_horiz < 1.0e-3:
                    continue
                radius = edge / omega_abs
                xs.append(eps)
                ys.append(tz / (radius * f_horiz))
                eps_all.append(eps)
        if xs:
            coupling_series.append(
                rc.Series(xs, ys, rc.MODE_LABELS[mode], rc.MODE_COLORS[mode], draw_line=False, draw_marker=True)
            )

    lo = min(eps_all) if eps_all else 0.2
    hi = max(eps_all) if eps_all else 2.0
    curve_eps = [lo + (hi - lo) * i / 59.0 for i in range(60)]
    ref_ys = []
    for e in curve_eps:
        f_fac, t_fac = _farkas_factors(e)
        ref_ys.append(t_fac / f_fac if f_fac > 1.0e-9 else float("nan"))
    reference = rc.Series(curve_eps, ref_ys, "Farkas T(eps)/F(eps)", rc.REFERENCE_COLOR, dash="5 4")
    all_y = [y for series in [reference, *coupling_series] for y in series.ys]
    return rc.figure_grid(
        [
            rc.Figure(
                title="Force-torque coupling vs analytic disk",
                xlabel="epsilon = v / (R |omega|)",
                ylabel="|Tz| / (R |F_horiz|)",
                series=[reference, *coupling_series],
                x_range=rc.padded_range(eps_all or [lo, hi], floor_span=0.2),
                y_range=rc.padded_range(all_y, include=(0.0,), floor_span=0.2),
            ),
        ]
    )


def _summary_series(
    summaries: dict[float, dict[str, dict[str, str]]],
    *,
    mode: str,
    key: str,
    scale: float = 1.0,
    label: str | None = None,
    color: str | None = None,
    dash: str | None = None,
) -> rc.Series:
    xs = [epsilon0 for epsilon0 in sorted(summaries) if mode in summaries[epsilon0]]
    ys = [scale * rc.as_float(summaries[epsilon0][mode], key) for epsilon0 in xs]
    return rc.Series(
        xs,
        ys,
        label or rc.MODE_LABELS[mode],
        color or rc.MODE_COLORS[mode],
        draw_marker=True,
        dash=dash,
    )


def _reference_series(
    summaries: dict[float, dict[str, dict[str, str]]],
    *,
    key: str,
    label: str,
) -> rc.Series:
    xs = []
    ys = []
    for epsilon0 in sorted(summaries):
        row = summaries[epsilon0].get("unreduced") or next(iter(summaries[epsilon0].values()))
        xs.append(epsilon0)
        ys.append(rc.as_float(row, key))
    return rc.Series(xs, ys, label, rc.REFERENCE_COLOR, draw_marker=False, dash="5 4")


def _figure_sweep_summary(summaries: dict[float, dict[str, dict[str, str]]]) -> str:
    epsilons = sorted(summaries)
    x_range = rc.padded_range(epsilons, floor_span=0.2)
    x_ticks = epsilons if len(epsilons) <= 8 else None

    late_epsilon_series = [
        _reference_series(summaries, key="epsilon_reference", label="Farkas attractor eps* = 0.653"),
        _summary_series(summaries, mode="unreduced", key="late_epsilon"),
        _summary_series(summaries, mode="reduced", key="late_epsilon"),
    ]
    stop_series = [
        _summary_series(
            summaries, mode="unreduced", key="speed_stop_time_s", label="speed stop, reduce off", color=SPEED_COLOR
        ),
        _summary_series(
            summaries,
            mode="reduced",
            key="speed_stop_time_s",
            label="speed stop, reduce on",
            color=SPEED_COLOR,
            dash="4 4",
        ),
        _summary_series(
            summaries, mode="unreduced", key="spin_stop_time_s", label="spin stop, reduce off", color=SPIN_COLOR
        ),
        _summary_series(
            summaries,
            mode="reduced",
            key="spin_stop_time_s",
            label="spin stop, reduce on",
            color=SPIN_COLOR,
            dash="4 4",
        ),
    ]

    final_speed_series = [
        _summary_series(summaries, mode="unreduced", key="final_speed_m_per_s"),
        _summary_series(summaries, mode="reduced", key="final_speed_m_per_s"),
    ]

    return rc.figure_grid(
        [
            rc.Figure(
                title="Late coupling ratio",
                xlabel="initial epsilon",
                ylabel="late epsilon",
                series=late_epsilon_series,
                x_range=x_range,
                y_range=rc.padded_range(
                    [y for series in late_epsilon_series for y in series.ys],
                    include=(0.653,),
                    floor_span=0.25,
                ),
                x_ticks=x_ticks,
            ),
            rc.Figure(
                title="Stop times",
                xlabel="initial epsilon",
                ylabel="time [s]",
                series=stop_series,
                x_range=x_range,
                y_range=rc.padded_range(
                    [y for series in stop_series for y in series.ys], include=(0.0,), floor_span=0.05
                ),
                x_ticks=x_ticks,
            ),
            rc.Figure(
                title="Final horizontal speed",
                xlabel="initial epsilon",
                ylabel="speed [m/s]",
                series=final_speed_series,
                x_range=x_range,
                y_range=rc.padded_range(
                    [y for series in final_speed_series for y in series.ys],
                    include=(0.0,),
                    floor_span=0.05,
                ),
                x_ticks=x_ticks,
            ),
        ]
    )


def _checks_table(summaries: dict[float, dict[str, dict[str, str]]]) -> str:
    """Figure 4: contact counts, buffer utilization, drift gate, and simultaneous-stop gate."""

    headers = [
        "eps0",
        "mode",
        "mean contacts",
        "max rigid / cap",
        "overflow",
        "hash fail",
        "drift [mm]",
        "coupled stop",
    ]
    rows = []
    for epsilon0 in sorted(summaries):
        for mode in rc.MODES:
            row = summaries[epsilon0].get(mode)
            if row is None:
                continue
            max_rigid = rc.format_number(rc.as_float(row, "max_rigid_contact_count"), precision=4)
            cap = rc.format_number(rc.as_float(row, "rigid_contact_capacity"), precision=4)
            rows.append(
                [
                    rc.format_number(epsilon0),
                    rc.MODE_LABELS.get(mode, mode),
                    rc.format_number(rc.as_float(row, "mean_solver_force_count"), precision=4),
                    f"{max_rigid} / {cap}",
                    row.get("buffer_overflow", "n/a"),
                    rc.format_number(rc.as_float(row, "max_reduction_hashtable_failures"), precision=4),
                    rc.format_number(1000.0 * rc.as_float(row, "final_y_drift_m")),
                    row.get("coupled_stopped", "n/a"),
                ]
            )
    return rc.data_table(headers, rows)


def _result_bullets(summaries: dict[float, dict[str, dict[str, str]]]) -> list[str]:
    bullets = []
    valid = all(
        not rc.as_bool(row["buffer_overflow"]) for mode_rows in summaries.values() for row in mode_rows.values()
    )
    if valid:
        bullets.append("All reported runs passed the contact-buffer validity gate.")
    else:
        bullets.append("At least one reported run overflowed a contact buffer; treat that comparison as inconclusive.")

    ratios = []
    for epsilon0 in sorted(summaries):
        off = summaries[epsilon0].get("unreduced")
        on = summaries[epsilon0].get("reduced")
        if off is None or on is None:
            continue
        off_count = rc.as_float(off, "mean_solver_force_count")
        on_count = rc.as_float(on, "mean_solver_force_count")
        if off_count > 0.0:
            ratios.append(on_count / off_count)
        bullets.append(
            f"eps0 {rc.format_number(epsilon0)}: late eps off/on = "
            f"{rc.format_number(rc.as_float(off, 'late_epsilon'))}/"
            f"{rc.format_number(rc.as_float(on, 'late_epsilon'))} (-> 0.653); "
            "speed/spin stop off = "
            f"{rc.format_number(rc.as_float(off, 'speed_stop_time_s'))}/"
            f"{rc.format_number(rc.as_float(off, 'spin_stop_time_s'))} s, on = "
            f"{rc.format_number(rc.as_float(on, 'speed_stop_time_s'))}/"
            f"{rc.format_number(rc.as_float(on, 'spin_stop_time_s'))} s."
        )

    if ratios:
        bullets.append(
            "Across this sweep, reduce on used "
            f"{rc.format_percent(min(ratios))}-{rc.format_percent(max(ratios))} as many solver contacts as reduce off."
        )

    return bullets


def _build_html(
    *,
    runs: dict[float, dict[str, list[dict[str, str]]]],
    summaries: dict[float, dict[str, dict[str, str]]],
    selected_epsilon: float,
    csv_dir: Path,
) -> str:
    epsilons = ", ".join(rc.format_number(epsilon0) for epsilon0 in sorted(summaries))
    reference_row = summaries[selected_epsilon].get("unreduced") or next(iter(summaries[selected_epsilon].values()))
    initial_omega = rc.as_float(reference_row, "initial_omega_rad_per_s")

    panels = [
        rc.TabPanel(
            "Figure 1",
            "<p>Figure 1: primary coupled state versus time &mdash; coupling ratio epsilon, horizontal speed, and yaw "
            "rate, each against the Farkas analytic trajectory (epsilon converging to 0.653).</p>\n"
            + _figure_state_history(runs, selected_epsilon=selected_epsilon),
        ),
        rc.TabPanel(
            "Figure 2",
            "<p>Figure 2: the solver forces behind that state (Fx, yaw torque) and the pointwise Farkas check &mdash; "
            "measured <code>|Tz|/(R|F_horiz|)</code> vs <code>T(eps)/F(eps)</code>. Radius is recovered from data and "
            "the friction/load factor cancels, so both modes should sit on the curve.</p>\n"
            + _figure_solver_history(runs, selected_epsilon=selected_epsilon)
            + _figure_coupling(runs),
        ),
        rc.TabPanel(
            "Figure 3",
            "<p>Figure 3: sweep across initial coupling ratios &mdash; late epsilon vs the 0.653 attractor, speed and "
            "spin stop times (simultaneous if coupled), and final speed.</p>\n" + _figure_sweep_summary(summaries),
        ),
        rc.TabPanel(
            "Figure 4",
            "<p>Figure 4: contact counts, buffer utilization, the drift gate, and the simultaneous-stop gate.</p>\n"
            + _checks_table(summaries),
        ),
    ]

    body = "\n".join(
        [
            f"<h1>{rc.escape(PAGE_TITLE)}</h1>",
            "<p>A flat cylinder both slides and yaw-spins on a flat hydroelastic plate. Torsional and rolling friction "
            "are zero, so translation and spin both decay through sliding friction over the same patch &mdash; the "
            "solver must get force and torque from one reduced contact set. Compare reduce off vs reduce on, pre-prune "
            "off in both.</p>",
            f"<p>Sweep variable <code>epsilon0 = v0 / (R * omega0)</code>; logged <code>epsilon = v / (R |omega_z|)</code>. "
            f"Values: {epsilons}; initial yaw rate {rc.format_number(initial_omega)} rad/s; time histories use "
            f"epsilon0 = {rc.format_number(selected_epsilon)}.</p>",
            "<h2>Reference</h2>",
            "<p>Uniform-pressure sliding-spinning disk (Farkas et al., PRL 90, 248302, 2003): "
            "<code>|F| = mu Fn F(eps)</code>, <code>|Tz| = mu Fn R T(eps)</code>. The ratio "
            "<code>|Tz|/(R|F_horiz|) = T(eps)/F(eps)</code> is friction- and load-independent. The disk has a "
            "universal terminal ratio <code>eps* ~ 0.653</code>, with sliding and spinning stopping together.</p>",
            "<h2>Measured Quantities</h2>",
            "<p>Primary (Figure 1, directly measured):</p>",
            "<ul>",
            "<li>Horizontal speed <code>v</code> and yaw rate <code>omega_z</code>.</li>",
            "<li>Coupling ratio <code>epsilon = v / (R |omega_z|)</code>.</li>",
            "</ul>",
            "<p>Secondary (derived / checks):</p>",
            "<ul>",
            "<li>Solver force <code>Fx</code>, yaw torque <code>Tz</code>, and the coupling ratio vs Farkas.</li>",
            "<li>Speed/spin stop times, late epsilon, final speed, drift, tilt, penetration.</li>",
            "<li>Solver, rigid, and raw face contact counts with buffer validity flags.</li>",
            "</ul>",
            "<h2>Results</h2>",
            f"<ul>\n{rc.bullet_list(_result_bullets(summaries))}\n</ul>",
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
    selected_epsilon: float | None = None,
) -> Path:
    """Write a standalone HTML report and return its path."""

    csv_dir = Path(csv_dir)
    runs = load_timeseries(csv_dir)
    summaries = load_summaries(csv_dir)
    selected = select_epsilon(runs, selected_epsilon)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        _build_html(runs=runs, summaries=summaries, selected_epsilon=selected, csv_dir=csv_dir),
        encoding="utf-8",
    )
    shutil.copyfile(HYPOTHESIS_RECORD_SOURCE, output_path.parent / HYPOTHESIS_RECORD_NAME)
    return output_path


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--csv-dir", type=str, default=str(DEFAULT_CSV_DIR), help="Directory containing H5 CSV files.")
    parser.add_argument("--output", type=str, default=str(DEFAULT_HTML_PATH), help="HTML output path.")
    parser.add_argument(
        "--selected-epsilon", type=float, default=None, help="Initial epsilon for time-history figures."
    )
    return parser


def main() -> None:
    args = create_parser().parse_args()
    output_path = write_html_report(
        csv_dir=args.csv_dir,
        output_path=args.output,
        selected_epsilon=args.selected_epsilon,
    )
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
