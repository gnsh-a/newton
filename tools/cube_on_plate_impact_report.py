# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Build a self-contained HTML report from cube-on-plate impact CSVs.

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

DEFAULT_CSV_DIR = Path("output") / "H6_cube_on_plate_impact"
DEFAULT_HTML_PATH = DEFAULT_CSV_DIR / "cube_on_plate_impact_report.html"
HYPOTHESIS_RECORD_NAME = "H6_cube_on_plate_impact_hypothesis.md"
HYPOTHESIS_RECORD_SOURCE = Path(__file__).resolve().parents[1] / "hypothesis" / HYPOTHESIS_RECORD_NAME
TIMESERIES_CSV = "impact_timeseries.csv"
SUMMARY_CSV = "impact_summary.csv"

PAGE_TITLE = "H6: Cube-on-Plate Impact"

# Ideal lumped vertical contact stiffness for the flat cube-on-plate, K = g_eff * A
# (VALIDATION.md "H6 stiffness levels", L1 ideal). Sets the damped-SDOF reference.
K_IDEAL_N_PER_M = 4.0e7


def load_timeseries(csv_dir: str | Path) -> dict[float, dict[str, list[dict[str, str]]]]:
    """Load time-series rows grouped by height and mode."""

    path = Path(csv_dir) / TIMESERIES_CSV
    if not path.exists():
        raise FileNotFoundError(f"missing time-series CSV: {path}")
    grouped: dict[float, dict[str, list[dict[str, str]]]] = {}
    for row in rc.read_csv(path):
        height = rc.as_float(row, "height_m")
        grouped.setdefault(height, {}).setdefault(row["mode"], []).append(row)
    for mode_rows in grouped.values():
        for rows in mode_rows.values():
            rows.sort(key=lambda row: rc.as_float(row, "time_s"))
    if not grouped:
        raise ValueError(f"no rows in {path}")
    return dict(sorted(grouped.items()))


def load_summaries(csv_dir: str | Path) -> dict[float, dict[str, dict[str, str]]]:
    """Load summary rows grouped by height and mode."""

    path = Path(csv_dir) / SUMMARY_CSV
    if not path.exists():
        raise FileNotFoundError(f"missing summary CSV: {path}")
    grouped: dict[float, dict[str, dict[str, str]]] = {}
    for row in rc.read_csv(path):
        grouped.setdefault(rc.as_float(row, "height_m"), {})[row["mode"]] = row
    if not grouped:
        raise ValueError(f"no rows in {path}")
    return dict(sorted(grouped.items()))


def select_height(runs: dict[float, dict[str, list[dict[str, str]]]], requested: float | None) -> float:
    """Select a representative height for time-history plots."""

    heights = sorted(runs)
    if requested is not None:
        return min(heights, key=lambda height: abs(height - requested))
    if 0.005 in runs:
        return 0.005
    return heights[len(heights) // 2]


def _impact_figure(
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    series: list[rc.Series],
    x_include: tuple[float, ...] = (),
    y_include: tuple[float, ...] = (),
    y_floor_span: float = 0.0,
    selector: str | None = None,
    selector_default: str | None = None,
    height: int = 300,
) -> rc.Figure:
    """Build a figure, computing axis ranges from the series and includes."""

    all_x = [x for item in series for x in item.xs]
    all_y = [y for item in series for y in item.ys]
    return rc.Figure(
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        series=series,
        x_range=rc.padded_range(all_x, include=x_include),
        y_range=rc.padded_range(all_y, include=y_include, floor_span=y_floor_span),
        selector=selector,
        selector_default=selector_default,
        height=height,
    )


def _summary_series(
    summaries: dict[float, dict[str, dict[str, str]]],
    key: str,
    *,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> list[rc.Series]:
    result: list[rc.Series] = []
    for mode in rc.MODES:
        xs: list[float] = []
        ys: list[float] = []
        for height, mode_rows in summaries.items():
            row = mode_rows.get(mode)
            if row is None:
                continue
            value = rc.as_float(row, key)
            if not math.isfinite(value):
                continue
            xs.append(height * scale_x)
            ys.append(value * scale_y)
        result.append(
            rc.Series(
                xs=xs,
                ys=ys,
                label=rc.MODE_LABELS[mode],
                color=rc.MODE_COLORS[mode],
                draw_line=False,
                draw_marker=True,
            )
        )
    return result


def _sdof_arc(
    *, impact_velocity: float, contact_time: float, weight: float, mass: float, n: int = 60
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Undamped damped-SDOF reference for the contact half-cycle (needs only K, m, v_imp).

    During contact a lumped mass on stiffness ``K_IDEAL`` follows
    ``z(t) = (v/omega_n) sin(omega_n t)`` with ``omega_n = sqrt(K/m)``; the realized
    (damped) response sits below this lossless envelope. Returns aligned
    ``(t, Fz/weight, penetration[mm], vz)`` over one half period from first contact.
    """

    if not (math.isfinite(impact_velocity) and impact_velocity > 0.0 and mass > 0.0 and weight > 0.0):
        return [], [], [], []
    omega_n = math.sqrt(K_IDEAL_N_PER_M / mass)
    half_period = math.pi / omega_n
    ts, fz, pen, vz = [], [], [], []
    for i in range(n + 1):
        tau = half_period * i / n
        depth = (impact_velocity / omega_n) * math.sin(omega_n * tau)
        ts.append(contact_time + tau)
        pen.append(1000.0 * depth)
        fz.append(K_IDEAL_N_PER_M * depth / weight)
        vz.append(-impact_velocity * math.cos(omega_n * tau))
    return ts, fz, pen, vz


def _time_history_figures(
    runs: dict[float, dict[str, list[dict[str, str]]]],
    summaries: dict[float, dict[str, dict[str, str]]],
    selected_height: float,
) -> str:
    weight = next(
        (rc.as_float(row, "cube_weight_N") for row in summaries.get(selected_height, {}).values()),
        float("nan"),
    )
    if not math.isfinite(weight) or weight <= 0.0:
        weight = 1.0

    def _group(height: float) -> str:
        return f"h = {1000.0 * height:.3g} mm"

    default_group = _group(selected_height)
    fz_series: list[rc.Series] = []
    pen_series: list[rc.Series] = []
    vz_series: list[rc.Series] = []
    # Overlay every drop height (color = height, dash = mode); opens on the
    # representative height with the others a dropdown away.
    for index, height in enumerate(sorted(runs)):
        mode_rows = runs[height]
        group = _group(height)
        color = rc.group_color(index)
        for mode in rc.MODES:
            rows = mode_rows.get(mode, [])
            if not rows:
                continue
            mode_dash = None if mode == "reduced" else "dash"
            solo = rc.SOLO_MODE_COLORS[mode]
            label = f"{group} · {rc.MODE_LABELS[mode]}"
            xs = [rc.as_float(row, "time_s") for row in rows]
            fz_series.append(
                rc.Series(
                    xs,
                    [rc.as_float(row, "solver_fz_N") / weight for row in rows],
                    label,
                    color,
                    dash=mode_dash,
                    group=group,
                    solo_color=solo,
                )
            )
            pen_series.append(
                rc.Series(
                    xs,
                    [1000.0 * rc.as_float(row, "cube_penetration_depth_m") for row in rows],
                    label,
                    color,
                    dash=mode_dash,
                    group=group,
                    solo_color=solo,
                )
            )
            vz_series.append(
                rc.Series(
                    xs,
                    [rc.as_float(row, "cube_vz_m_per_s") for row in rows],
                    label,
                    color,
                    dash=mode_dash,
                    group=group,
                    solo_color=solo,
                )
            )
        # Undamped SDOF reference arc for this height (color-matched, dotted).
        summary_ref = summaries.get(height, {}).get("unreduced") or next(iter(summaries.get(height, {}).values()), None)
        if summary_ref is not None:
            arc_t, arc_fz, arc_pen, arc_vz = _sdof_arc(
                impact_velocity=rc.as_float(summary_ref, "impact_velocity_m_per_s"),
                contact_time=rc.as_float(summary_ref, "first_contact_time_s"),
                weight=weight,
                mass=rc.as_float(summary_ref, "cube_mass_kg"),
            )
            if arc_t:
                ref = rc.REFERENCE_COLOR
                label = f"{group} · SDOF reference"
                fz_series.append(rc.Series(arc_t, arc_fz, label, color, dash="dot", group=group, solo_color=ref))
                pen_series.append(rc.Series(arc_t, arc_pen, label, color, dash="dot", group=group, solo_color=ref))
                vz_series.append(rc.Series(arc_t, arc_vz, label, color, dash="dot", group=group, solo_color=ref))

    def _hist(title: str, ylabel: str, series: list[rc.Series]) -> rc.Figure:
        return _impact_figure(
            title=title,
            xlabel="time [s]",
            ylabel=ylabel,
            series=series,
            y_include=(0.0,),
            selector="drop height",
            selector_default=default_group,
            height=360,
        )

    return rc.figure_grid(
        [
            _hist("Solver Fz history", "Fz / mg", fz_series),
            _hist("Compression history", "penetration [mm]", pen_series),
            _hist("Vertical velocity history", "vz [m/s]", vz_series),
        ]
    )


def _sdof_reference_series(summaries: dict[float, dict[str, dict[str, str]]], *, kind: str) -> rc.Series:
    """SDOF undamped peak reference over drop height (uses only K_IDEAL, m, measured v_imp).

    ``kind="peak_over_weight"`` -> F_peak/mg = v_imp*sqrt(K*m)/weight;
    ``kind="depth_mm"`` -> depth_max = v_imp*sqrt(m/K) in mm.
    """
    xs: list[float] = []
    ys: list[float] = []
    for height, mode_rows in summaries.items():
        row = mode_rows.get("unreduced") or next(iter(mode_rows.values()), None)
        if row is None:
            continue
        v = rc.as_float(row, "impact_velocity_m_per_s")
        m = rc.as_float(row, "cube_mass_kg")
        weight = rc.as_float(row, "cube_weight_N")
        if not (math.isfinite(v) and v > 0.0 and m > 0.0 and weight > 0.0):
            continue
        xs.append(1000.0 * height)
        if kind == "peak_over_weight":
            ys.append(v * math.sqrt(K_IDEAL_N_PER_M * m) / weight)
        else:
            ys.append(1000.0 * v * math.sqrt(m / K_IDEAL_N_PER_M))
    return rc.Series(xs, ys, "SDOF reference", rc.REFERENCE_COLOR, draw_marker=False, dash="5 4")


def _peak_response_figures(summaries: dict[float, dict[str, dict[str, str]]]) -> str:
    return rc.figure_grid(
        [
            _impact_figure(
                title="Peak solver force",
                xlabel="drop height [mm]",
                ylabel="peak Fz / mg",
                series=[
                    _sdof_reference_series(summaries, kind="peak_over_weight"),
                    *_summary_series(summaries, "peak_solver_fz_over_weight", scale_x=1000.0),
                ],
                y_include=(0.0,),
            ),
            _impact_figure(
                title="Maximum compression",
                xlabel="drop height [mm]",
                ylabel="penetration [mm]",
                series=[
                    _sdof_reference_series(summaries, kind="depth_mm"),
                    *_summary_series(summaries, "max_penetration_depth_m", scale_x=1000.0, scale_y=1000.0),
                ],
                y_include=(0.0,),
            ),
            _impact_figure(
                title="Time to peak force",
                xlabel="drop height [mm]",
                ylabel="time [ms]",
                series=_summary_series(summaries, "time_to_peak_fz_s", scale_x=1000.0, scale_y=1000.0),
                y_include=(0.0,),
            ),
        ]
    )


def _ring_down_figures(summaries: dict[float, dict[str, dict[str, str]]]) -> str:
    return rc.figure_grid(
        [
            _impact_figure(
                title="Rebound velocity ratio",
                xlabel="drop height [mm]",
                ylabel="rebound / impact",
                series=_summary_series(summaries, "rebound_velocity_ratio", scale_x=1000.0),
                y_include=(0.0,),
            ),
            _impact_figure(
                title="Settling time",
                xlabel="drop height [mm]",
                ylabel="time [ms]",
                series=_summary_series(summaries, "settle_time_s", scale_x=1000.0, scale_y=1000.0),
                y_include=(0.0,),
            ),
            _impact_figure(
                title="Post-settle force RMS",
                xlabel="drop height [mm]",
                ylabel="Fz RMS [N]",
                series=_summary_series(summaries, "post_settle_fz_rms_N", scale_x=1000.0),
                y_include=(0.0,),
            ),
        ]
    )


def _checks_table(summaries: dict[float, dict[str, dict[str, str]]]) -> str:
    """Figure 4: peak/penetration, contact counts, the dt-floor gate, and validity as a table.

    The dt-floor gate reports steps per contact half-period ``pi*sqrt(m/K_IDEAL)``; below a
    few steps the impact is dt-limited, not physics-limited, so the dynamic result is
    inconclusive regardless of the buffer-based ``valid`` flag.
    """

    headers = [
        "height [mm]",
        "mode",
        "peak Fz/mg",
        "max pen [mm]",
        "mean contacts",
        "max rigid / cap",
        "steps/half-T",
        "valid",
    ]
    rows = []
    for height, mode_rows in summaries.items():
        for mode in rc.MODES:
            row = mode_rows.get(mode)
            if row is None:
                continue
            mass = rc.as_float(row, "cube_mass_kg")
            step_dt = rc.as_float(row, "step_dt_s")
            half_period = math.pi * math.sqrt(mass / K_IDEAL_N_PER_M) if mass > 0.0 else float("nan")
            steps = half_period / step_dt if step_dt > 0.0 and math.isfinite(half_period) else float("nan")
            dt_ok = "ok" if math.isfinite(steps) and steps >= 5.0 else "dt-limited"
            max_rigid = rc.format_number(rc.as_float(row, "max_rigid_contact_count"), precision=4)
            cap = rc.format_number(rc.as_float(row, "rigid_contact_capacity"), precision=4)
            rows.append(
                [
                    f"{1000.0 * height:.3g}",
                    rc.MODE_LABELS[mode],
                    rc.format_number(rc.as_float(row, "peak_solver_fz_over_weight")),
                    f"{1000.0 * rc.as_float(row, 'max_penetration_depth_m'):.4g}",
                    rc.format_number(rc.as_float(row, "mean_solver_force_count"), precision=4),
                    f"{max_rigid} / {cap}",
                    f"{rc.format_number(steps)} ({dt_ok})",
                    row.get("valid_run", "n/a"),
                ]
            )
    return rc.data_table(headers, rows)


def _result_bullets(summaries: dict[float, dict[str, dict[str, str]]]) -> str:
    peak_ratios: list[float] = []
    contact_ratios: list[float] = []
    for mode_rows in summaries.values():
        off = mode_rows.get("unreduced")
        on = mode_rows.get("reduced")
        if off is None or on is None:
            continue
        off_peak = rc.as_float(off, "peak_solver_fz_over_weight")
        on_peak = rc.as_float(on, "peak_solver_fz_over_weight")
        if off_peak > 0.0:
            peak_ratios.append(on_peak / off_peak)
        off_contacts = rc.as_float(off, "mean_solver_force_count")
        on_contacts = rc.as_float(on, "mean_solver_force_count")
        if off_contacts > 0.0:
            contact_ratios.append(on_contacts / off_contacts)
    peak_text = rc.format_number(max(peak_ratios), precision=3) if peak_ratios else "nan"
    contact_text = (
        rc.format_percent(sum(contact_ratios) / len(contact_ratios), precision=3) if contact_ratios else "nan"
    )
    all_valid = all(rc.as_bool(row["valid_run"]) for mode_rows in summaries.values() for row in mode_rows.values())
    return (
        "<ul>"
        f"<li>Largest reduce-on / reduce-off peak-force ratio: <strong>{peak_text}</strong>.</li>"
        f"<li>Mean reduce-on contact-count ratio: <strong>{contact_text}</strong>.</li>"
        f"<li>All current rows valid: <strong>{str(all_valid).lower()}</strong>.</li>"
        "</ul>"
    )


def _run_settings_text(summaries: dict[float, dict[str, dict[str, str]]]) -> str:
    first_row = next((row for mode_rows in summaries.values() for row in mode_rows.values()), None)
    if first_row is None:
        return ""
    simulation_time = rc.as_float(first_row, "simulation_time_s") if "simulation_time_s" in first_row else float("nan")
    step_dt = rc.as_float(first_row, "step_dt_s") if "step_dt_s" in first_row else float("nan")
    if not (math.isfinite(simulation_time) and math.isfinite(step_dt)):
        return ""
    return (
        f"<p>Saved sweep settings: simulation time {rc.format_number(simulation_time)} s, "
        f"solver/log step {rc.format_number(1000.0 * step_dt)} ms.</p>"
    )


def _build_html(
    *,
    runs: dict[float, dict[str, list[dict[str, str]]]],
    summaries: dict[float, dict[str, dict[str, str]]],
    selected_height: float,
    csv_dir: Path,
) -> str:
    height_mm = 1000.0 * selected_height
    panels = [
        rc.TabPanel(
            "Figure 1",
            "<p>Figure 1: primary, directly-measured impact transient versus time &mdash; solver Fz/mg, compression, "
            f"and vertical velocity (selected height {height_mm:.4g} mm), each against the undamped SDOF reference "
            "arc.</p>\n" + _time_history_figures(runs, summaries, selected_height),
        ),
        rc.TabPanel(
            "Figure 2",
            "<p>Figure 2: peak response vs drop height &mdash; peak force and maximum compression against the SDOF "
            "targets <code>F_peak = v_imp*sqrt(K*m)</code> and <code>depth_max = v_imp*sqrt(m/K)</code>, plus time to "
            "peak.</p>\n" + _peak_response_figures(summaries),
        ),
        rc.TabPanel(
            "Figure 3",
            "<p>Figure 3: rebound and ring-down &mdash; whether the impact returns smoothly to static support after "
            "first contact.</p>\n" + _ring_down_figures(summaries),
        ),
        rc.TabPanel(
            "Figure 4",
            "<p>Figure 4: peak/penetration, contact counts, the dt-floor gate (steps per contact half-period), and "
            "validity. Fewer than ~5 steps per half-period means the impact is dt-limited and inconclusive.</p>\n"
            + _checks_table(summaries),
        ),
    ]

    body = "\n".join(
        [
            f"<h1>{rc.escape(PAGE_TITLE)}</h1>",
            '<p class="lede">Does contact reduction preserve vertical impact dynamics, not just the static support '
            "force? The H1 cube is dropped onto the H1 plate; reduce off vs reduce on are compared in force history, "
            "compression, rebound, and settling.</p>",
            "<h2>Reference</h2>",
            "<p>Damped single-DOF oscillator (VALIDATION.md): with lumped stiffness "
            f"<code>K = {rc.format_number(K_IDEAL_N_PER_M)} N/m</code> and mass m, "
            "<code>F_peak = v_imp*sqrt(K*m)</code>, <code>depth_max = v_imp*sqrt(m/K)</code>. The contact half-period "
            "is <code>pi*sqrt(m/K) ~ 0.44 ms</code>, so the step must resolve it; otherwise the impact is dt-limited "
            "and the dynamic comparison is inconclusive.</p>",
            "<h2>Measured Quantities</h2>",
            "<p>Primary (Figure 1, directly measured):</p>",
            "<ul>",
            "<li>Solver vertical force <code>Fz</code>, penetration depth, and vertical velocity.</li>",
            "</ul>",
            "<p>Secondary (derived / checks):</p>",
            "<ul>",
            "<li>Peak force and maximum compression vs the SDOF targets; time to peak.</li>",
            "<li>Rebound ratio, settle time, post-settle force RMS.</li>",
            "<li>Solver, rigid, and face contact counts; dt-floor gate and buffer validity.</li>",
            "</ul>",
            "<h2>Results</h2>",
            _run_settings_text(summaries),
            _result_bullets(summaries),
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
    selected_height: float | None = None,
) -> Path:
    """Write a standalone HTML report and return its path."""

    csv_dir = Path(csv_dir)
    runs = load_timeseries(csv_dir)
    summaries = load_summaries(csv_dir)
    selected = select_height(runs, selected_height)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        _build_html(runs=runs, summaries=summaries, selected_height=selected, csv_dir=csv_dir),
        encoding="utf-8",
    )
    shutil.copyfile(HYPOTHESIS_RECORD_SOURCE, output_path.parent / HYPOTHESIS_RECORD_NAME)
    return output_path


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--csv-dir", type=str, default=str(DEFAULT_CSV_DIR), help="Directory containing H6 CSV files.")
    parser.add_argument("--output", type=str, default=str(DEFAULT_HTML_PATH), help="HTML output path.")
    parser.add_argument("--height", type=float, default=None, help="Drop height [m] for Figure 1.")
    return parser


def main() -> None:
    args = create_parser().parse_args()
    path = write_html_report(csv_dir=args.csv_dir, output_path=args.output, selected_height=args.height)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
