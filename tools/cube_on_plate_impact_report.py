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

PAGE_TITLE = "H6: Cube-on-Plate Impact Ring-Down Contact Reduction"


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


def _time_history_figures(
    runs: dict[float, dict[str, list[dict[str, str]]]],
    summaries: dict[float, dict[str, dict[str, str]]],
    selected_height: float,
) -> str:
    mode_rows = runs[selected_height]
    weight = next(
        (rc.as_float(row, "cube_weight_N") for row in summaries.get(selected_height, {}).values()),
        float("nan"),
    )
    if not math.isfinite(weight) or weight <= 0.0:
        weight = 1.0
    fz_series: list[rc.Series] = []
    pen_series: list[rc.Series] = []
    vz_series: list[rc.Series] = []
    for mode in rc.MODES:
        rows = mode_rows.get(mode, [])
        xs = [rc.as_float(row, "time_s") for row in rows]
        fz_series.append(
            rc.Series(
                xs=xs,
                ys=[rc.as_float(row, "solver_fz_N") / weight for row in rows],
                label=rc.MODE_LABELS[mode],
                color=rc.MODE_COLORS[mode],
            )
        )
        pen_series.append(
            rc.Series(
                xs=xs,
                ys=[1000.0 * rc.as_float(row, "cube_penetration_depth_m") for row in rows],
                label=rc.MODE_LABELS[mode],
                color=rc.MODE_COLORS[mode],
            )
        )
        vz_series.append(
            rc.Series(
                xs=xs,
                ys=[rc.as_float(row, "cube_vz_m_per_s") for row in rows],
                label=rc.MODE_LABELS[mode],
                color=rc.MODE_COLORS[mode],
            )
        )
    return rc.figure_grid(
        [
            _impact_figure(
                title=f"Solver Fz history at h={1000.0 * selected_height:.3g} mm",
                xlabel="time [s]",
                ylabel="Fz / mg",
                series=fz_series,
                y_include=(0.0,),
            ),
            _impact_figure(
                title="Compression history",
                xlabel="time [s]",
                ylabel="penetration [mm]",
                series=pen_series,
                y_include=(0.0,),
            ),
            _impact_figure(
                title="Vertical velocity history",
                xlabel="time [s]",
                ylabel="vz [m/s]",
                series=vz_series,
                y_include=(0.0,),
            ),
        ]
    )


def _peak_response_figures(summaries: dict[float, dict[str, dict[str, str]]]) -> str:
    return rc.figure_grid(
        [
            _impact_figure(
                title="Peak solver force",
                xlabel="drop height [mm]",
                ylabel="peak Fz / mg",
                series=_summary_series(summaries, "peak_solver_fz_over_weight", scale_x=1000.0),
                y_include=(0.0,),
            ),
            _impact_figure(
                title="Maximum compression",
                xlabel="drop height [mm]",
                ylabel="penetration [mm]",
                series=_summary_series(summaries, "max_penetration_depth_m", scale_x=1000.0, scale_y=1000.0),
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


def _contact_validity_figures(summaries: dict[float, dict[str, dict[str, str]]]) -> str:
    contact_series = _summary_series(summaries, "mean_solver_force_count", scale_x=1000.0)
    scatter_series: list[rc.Series] = []
    for mode in rc.MODES:
        xs: list[float] = []
        ys: list[float] = []
        for mode_rows in summaries.values():
            row = mode_rows.get(mode)
            if row is None:
                continue
            xs.append(rc.as_float(row, "mean_solver_force_count"))
            ys.append(rc.as_float(row, "peak_solver_fz_over_weight"))
        scatter_series.append(
            rc.Series(
                xs=xs, ys=ys, label=rc.MODE_LABELS[mode], color=rc.MODE_COLORS[mode], draw_line=False, draw_marker=True
            )
        )
    validity_series: list[rc.Series] = []
    for mode in rc.MODES:
        xs = []
        ys = []
        for height, mode_rows in summaries.items():
            row = mode_rows.get(mode)
            if row is None:
                continue
            xs.append(1000.0 * height)
            ys.append(1.0 if rc.as_bool(row["valid_run"]) else 0.0)
        validity_series.append(
            rc.Series(
                xs=xs, ys=ys, label=rc.MODE_LABELS[mode], color=rc.MODE_COLORS[mode], draw_line=False, draw_marker=True
            )
        )
    return rc.figure_grid(
        [
            _impact_figure(
                title="Mean solver contact count",
                xlabel="drop height [mm]",
                ylabel="contacts",
                series=contact_series,
                y_include=(0.0,),
            ),
            _impact_figure(
                title="Peak force vs contact count",
                xlabel="mean solver contacts",
                ylabel="peak Fz / mg",
                series=scatter_series,
                x_include=(0.0,),
                y_include=(0.0,),
            ),
            _impact_figure(
                title="Validity flag",
                xlabel="drop height [mm]",
                ylabel="valid run",
                series=validity_series,
                y_include=(0.0, 1.0),
                y_floor_span=1.0,
            ),
        ]
    )


def _summary_table(summaries: dict[float, dict[str, dict[str, str]]]) -> str:
    headers = [
        "height [mm]",
        "mode",
        "peak Fz / mg",
        "max penetration [mm]",
        "rebound ratio",
        "mean solver contacts",
        "valid",
    ]
    rows = []
    for height, mode_rows in summaries.items():
        for mode in rc.MODES:
            row = mode_rows.get(mode)
            if row is None:
                continue
            rows.append(
                [
                    f"{1000.0 * height:.3g}",
                    rc.MODE_LABELS[mode],
                    rc.format_number(rc.as_float(row, "peak_solver_fz_over_weight")),
                    f"{1000.0 * rc.as_float(row, 'max_penetration_depth_m'):.4g}",
                    rc.format_number(rc.as_float(row, "rebound_velocity_ratio")),
                    rc.format_number(rc.as_float(row, "mean_solver_force_count")),
                    row["valid_run"],
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
            "<h2>Figure 1: impact time history</h2>\n"
            f"<p>Selected height: {height_mm:.4g} mm. These curves show the transient that static settle does not "
            "test.</p>\n" + _time_history_figures(runs, summaries, selected_height),
        ),
        rc.TabPanel(
            "Figure 2",
            "<h2>Figure 2: peak response vs drop height</h2>\n"
            "<p>Drop-height plots use points only because each height is a separate experiment condition.</p>\n"
            + _peak_response_figures(summaries),
        ),
        rc.TabPanel(
            "Figure 3",
            "<h2>Figure 3: rebound and ring-down</h2>\n"
            "<p>These metrics test whether the impact returns smoothly to static support after first contact.</p>\n"
            + _ring_down_figures(summaries),
        ),
        rc.TabPanel(
            "Figure 4",
            "<h2>Figure 4: contact reduction and validity</h2>\n"
            "<p>This figure connects the transient response to the number of solver contacts used by each mode.</p>\n"
            + _contact_validity_figures(summaries),
        ),
    ]

    body = "\n".join(
        [
            f"<h1>{rc.escape(PAGE_TITLE)}</h1>",
            '<p class="lede">This report checks whether contact reduction preserves vertical impact dynamics, not '
            "just the final static support force. The setup drops the H1 cube onto the H1 plate and compares solver "
            "force history, compression, rebound, and settling.</p>",
            "<h2>Hypothesis</h2>",
            "<p>Reduce on may preserve the settled resultant while changing transient compliance. A few reduced "
            "contacts can carry the same final support load as dense contact, but peak force and ring-down depend on "
            "how stiffness and damping are distributed during impact.</p>",
            "<h2>Measured quantities</h2>",
            "<ul>",
            "<li>Solver force and torque on the cube.</li>",
            "<li>Penetration depth, vertical velocity, final tilt, and drift.</li>",
            "<li>Peak solver force, time to peak, rebound ratio, settle time, and force RMS.</li>",
            "<li>Solver contact count and contact-buffer validity.</li>",
            "</ul>",
            "<h2>Current result</h2>",
            _run_settings_text(summaries),
            _result_bullets(summaries),
            "<h2>Figures</h2>",
            rc.figure_tabs(panels),
            "<h2>Summary Table</h2>",
            _summary_table(summaries),
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
