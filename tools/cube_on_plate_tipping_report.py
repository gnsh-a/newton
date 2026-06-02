# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Build a self-contained HTML report from cube-on-plate tipping CSVs.

This script intentionally does not import Newton. The Newton experiment writes
CSV files; this script reads those CSV files and renders the standalone report
using the shared report framework in :mod:`_report_common`.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _report_common as rc

DEFAULT_CSV_DIR = Path("output") / "H3_cube_on_plate_tipping"
DEFAULT_HTML_PATH = DEFAULT_CSV_DIR / "cube_on_plate_tipping_report.html"
HYPOTHESIS_RECORD_NAME = "H3_cube_on_plate_tipping_hypothesis.md"
HYPOTHESIS_RECORD_SOURCE = Path(__file__).resolve().parents[1] / "hypothesis" / HYPOTHESIS_RECORD_NAME
TIMESERIES_CSV = "tipping_timeseries.csv"
SUMMARY_CSV = "tipping_summary.csv"

PAGE_TITLE = "H3: Cube-on-Plate Tipping Contact Reduction"


def load_timeseries(csv_dir: str | Path) -> dict[str, list[dict[str, str]]]:
    """Load time-series rows grouped by mode and sorted by time."""

    path = Path(csv_dir) / TIMESERIES_CSV
    if not path.exists():
        raise FileNotFoundError(f"missing time-series CSV: {path}")
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rc.read_csv(path):
        grouped.setdefault(row["mode"], []).append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: rc.as_float(row, "time_s"))
    if not grouped:
        raise ValueError(f"time-series CSV has no rows: {path}")
    return grouped


def load_summaries(csv_dir: str | Path) -> dict[str, dict[str, str]]:
    """Load summary rows grouped by mode."""

    path = Path(csv_dir) / SUMMARY_CSV
    if not path.exists():
        raise FileNotFoundError(f"missing summary CSV: {path}")
    grouped = {row["mode"]: row for row in rc.read_csv(path)}
    if not grouped:
        raise ValueError(f"summary CSV has no rows: {path}")
    return grouped


def _figure_state(rows_by_mode: dict[str, list[dict[str, str]]]) -> str:
    pitch_series = rc.mode_series(rows_by_mode, x_key="applied_force_over_ftip", y_key="cube_pitch_deg")
    cop_series = rc.mode_series(
        rows_by_mode, x_key="applied_force_over_ftip", y_key="center_pressure_x_over_half_extent"
    )
    x_series = rc.mode_series(rows_by_mode, x_key="applied_force_over_ftip", y_key="cube_x_m", scale=1000.0)
    all_xs = [x for series in pitch_series for x in series.xs]
    x_range = rc.padded_range(all_xs, include=(0.0, 1.0), floor_span=0.2)
    front_edge = rc.Series([0.0, max(all_xs or [1.0])], [1.0, 1.0], "front edge", rc.REFERENCE_COLOR, dash="5 4")
    return rc.figure_grid(
        [
            rc.Figure(
                title="Cube pitch under ramped top force",
                xlabel="applied force / analytic tip force",
                ylabel="pitch [deg]",
                series=pitch_series,
                x_range=x_range,
                y_range=rc.padded_range(
                    [y for series in pitch_series for y in series.ys], include=(0.0, 10.0), floor_span=1.0
                ),
            ),
            rc.Figure(
                title="Center-pressure shift",
                xlabel="applied force / analytic tip force",
                ylabel="cop_x / half extent",
                series=[front_edge, *cop_series],
                x_range=x_range,
                y_range=rc.padded_range(
                    [y for series in cop_series for y in series.ys], include=(0.0, 1.0), floor_span=0.2
                ),
            ),
            rc.Figure(
                title="Horizontal drift",
                xlabel="applied force / analytic tip force",
                ylabel="x [mm]",
                series=x_series,
                x_range=x_range,
                y_range=rc.padded_range([y for series in x_series for y in series.ys], include=(0.0,), floor_span=1.0),
            ),
        ]
    )


def _figure_solver(rows_by_mode: dict[str, list[dict[str, str]]]) -> str:
    fz_series = rc.mode_series(rows_by_mode, x_key="applied_force_over_ftip", y_key="solver_fz_N")
    ty_series = rc.mode_series(rows_by_mode, x_key="applied_force_over_ftip", y_key="solver_ty_Nm", scale=1000.0)
    count_series = rc.mode_series(rows_by_mode, x_key="applied_force_over_ftip", y_key="solver_force_count")
    all_xs = [x for series in fz_series for x in series.xs]
    x_range = rc.padded_range(all_xs, include=(0.0, 1.0), floor_span=0.2)
    return rc.figure_grid(
        [
            rc.Figure(
                title="Solver vertical support force",
                xlabel="applied force / analytic tip force",
                ylabel="Fz [N]",
                series=fz_series,
                x_range=x_range,
                y_range=rc.padded_range([y for series in fz_series for y in series.ys], include=(0.0,), floor_span=0.5),
            ),
            rc.Figure(
                title="Solver pitch torque",
                xlabel="applied force / analytic tip force",
                ylabel="Ty [mN m]",
                series=ty_series,
                x_range=x_range,
                y_range=rc.padded_range(
                    [y for series in ty_series for y in series.ys], include=(0.0,), floor_span=10.0
                ),
            ),
            rc.Figure(
                title="Solver contact count",
                xlabel="applied force / analytic tip force",
                ylabel="contacts",
                series=count_series,
                x_range=x_range,
                y_range=rc.padded_range(
                    [y for series in count_series for y in series.ys], include=(0.0,), floor_span=50.0
                ),
            ),
        ]
    )


def _figure_summary(summaries: dict[str, dict[str, str]]) -> str:
    ratios = [0.25, 0.50, 0.75, 0.90]
    pitch_keys = (
        "pitch_at_0p25_ftip_deg",
        "pitch_at_0p50_ftip_deg",
        "pitch_at_0p75_ftip_deg",
        "pitch_at_0p90_ftip_deg",
    )
    pitch_series = []
    event_series = []
    contact_series = []
    for idx, mode in enumerate(rc.MODES):
        row = summaries.get(mode)
        if row is None:
            continue
        pitch_series.append(
            rc.Series(
                ratios,
                [rc.as_float(row, key) for key in pitch_keys],
                rc.MODE_LABELS[mode],
                rc.MODE_COLORS[mode],
                draw_marker=True,
            )
        )
        event_series.append(
            rc.Series(
                [float(idx)],
                [rc.as_float(row, "event_force_over_ftip")],
                rc.MODE_LABELS[mode],
                rc.MODE_COLORS[mode],
                draw_line=False,
                draw_marker=True,
            )
        )
        contact_series.append(
            rc.Series(
                [float(idx)],
                [rc.as_float(row, "mean_solver_force_count")],
                rc.MODE_LABELS[mode],
                rc.MODE_COLORS[mode],
                draw_line=False,
                draw_marker=True,
            )
        )

    analytic_tip = rc.Series([0.0, 1.0], [1.0, 1.0], "analytic tip", rc.REFERENCE_COLOR, dash="5 4")
    return rc.figure_grid(
        [
            rc.Figure(
                title="Pre-tip pitch samples",
                xlabel="applied force / analytic tip force",
                ylabel="pitch [deg]",
                series=pitch_series,
                x_range=(0.18, 0.97),
                y_range=rc.padded_range(
                    [y for series in pitch_series for y in series.ys], include=(0.0,), floor_span=1.0
                ),
                x_ticks=ratios,
            ),
            rc.Figure(
                title="First event force",
                xlabel="mode index: 0 off, 1 on",
                ylabel="force / F_tip",
                series=[analytic_tip, *event_series],
                x_range=(-0.25, 1.25),
                y_range=rc.padded_range(
                    [y for series in event_series for y in series.ys], include=(1.0,), floor_span=0.2
                ),
                x_ticks=[0.0, 1.0],
            ),
            rc.Figure(
                title="Mean solver contact count",
                xlabel="mode index: 0 off, 1 on",
                ylabel="contacts",
                series=contact_series,
                x_range=(-0.25, 1.25),
                y_range=rc.padded_range(
                    [y for series in contact_series for y in series.ys], include=(0.0,), floor_span=50.0
                ),
                x_ticks=[0.0, 1.0],
            ),
        ]
    )


def _result_bullets(summaries: dict[str, dict[str, str]]) -> list[str]:
    bullets = []
    valid = all(
        (not rc.as_bool(row["buffer_overflow"])) and (not rc.as_bool(row.get("state_invalid", "false")))
        for row in summaries.values()
    )
    bullets.append(
        "All reported runs passed the contact-buffer validity gate."
        if valid
        else "At least one run overflowed a contact buffer; treat that comparison as inconclusive."
    )
    off = summaries.get("unreduced")
    on = summaries.get("reduced")
    if off is not None and on is not None:
        off_count = rc.as_float(off, "mean_solver_force_count")
        on_count = rc.as_float(on, "mean_solver_force_count")
        ratio = on_count / off_count if off_count > 0.0 else float("nan")
        bullets.append(
            "First event off/on = "
            f"{off['event_type']} at {rc.format_number(rc.as_float(off, 'event_force_N'))} N / "
            f"{on['event_type']} at {rc.format_number(rc.as_float(on, 'event_force_N'))} N."
        )
        bullets.append(
            "Pre-tip pitch at 0.9 F_tip off/on = "
            f"{rc.format_number(rc.as_float(off, 'pitch_at_0p90_ftip_deg'))} / "
            f"{rc.format_number(rc.as_float(on, 'pitch_at_0p90_ftip_deg'))} deg."
        )
        bullets.append(
            "Mean solver contact count off/on = "
            f"{rc.format_number(off_count)} / {rc.format_number(on_count)} "
            f"({rc.format_percent(ratio)})."
        )
    return bullets


def _build_html(
    *,
    rows_by_mode: dict[str, list[dict[str, str]]],
    summaries: dict[str, dict[str, str]],
    csv_dir: Path,
) -> str:
    ref = summaries.get("unreduced") or next(iter(summaries.values()))
    mu = rc.format_number(rc.as_float(ref, "mu_sliding"))
    tabs = rc.figure_tabs(
        [
            rc.TabPanel(
                "Figure 1",
                "<p>Figure 1 checks the rigid-body state response and the inferred center-pressure motion as the "
                "top force approaches the analytic tip load.</p>\n" + _figure_state(rows_by_mode),
            ),
            rc.TabPanel(
                "Figure 2",
                "<p>Figure 2 checks the direct solver outputs behind the state response: vertical support, pitch "
                "torque, and solver contact count.</p>\n" + _figure_solver(rows_by_mode),
            ),
            rc.TabPanel(
                "Figure 3",
                "<p>Figure 3 summarizes pre-tip pitch samples, first event force, and the contact-count "
                "reduction.</p>\n" + _figure_summary(summaries),
            ),
        ]
    )
    body = "\n".join(
        [
            f"<h1>{rc.escape(PAGE_TITLE)}</h1>",
            "<p>This experiment uses the cube-on-plate demo geometry and applies a ramped horizontal force at the "
            "cube top face. The comparison is reduce off versus reduce on, with pre-prune off in both modes.</p>",
            f"<p>The run uses <code>mu = {mu}</code>, so the analytic tip force <code>m*g/2</code> is below the "
            "sliding force <code>mu*m*g</code>. The useful comparison is pitch and center-pressure shift before the "
            "event, plus event force if the cube tips or slides.</p>",
            "<h2>Measured Quantities</h2>",
            "<ul>",
            "<li>Cube pitch and horizontal drift versus applied force.</li>",
            "<li>Solver force and torque on the cube.</li>",
            "<li>Center-pressure shift computed from solver wrench: <code>cop_x = -Ty / Fz</code>.</li>",
            "<li>First tip or slide event force, and solver contact-count reduction.</li>",
            "</ul>",
            "<h2>Results</h2>",
            f"<ul>\n{rc.bullet_list(_result_bullets(summaries))}\n</ul>",
            "<h2>Figures</h2>",
            tabs,
            f'<p class="meta">Generated from <code>{rc.escape(csv_dir / TIMESERIES_CSV)}</code> and '
            f"<code>{rc.escape(csv_dir / SUMMARY_CSV)}</code>.</p>",
        ]
    )
    return rc.render_page(title=PAGE_TITLE, body=body)


def write_html_report(
    *,
    csv_dir: str | Path = DEFAULT_CSV_DIR,
    output_path: str | Path = DEFAULT_HTML_PATH,
) -> Path:
    """Write a standalone HTML report and return its path."""

    csv_dir = Path(csv_dir)
    rows_by_mode = load_timeseries(csv_dir)
    summaries = load_summaries(csv_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        _build_html(rows_by_mode=rows_by_mode, summaries=summaries, csv_dir=csv_dir), encoding="utf-8"
    )
    shutil.copyfile(HYPOTHESIS_RECORD_SOURCE, output_path.parent / HYPOTHESIS_RECORD_NAME)
    return output_path


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--csv-dir", type=str, default=str(DEFAULT_CSV_DIR), help="Directory containing H3 CSV files.")
    parser.add_argument("--output", type=str, default=str(DEFAULT_HTML_PATH), help="HTML output path.")
    return parser


def main() -> None:
    args = create_parser().parse_args()
    output_path = write_html_report(csv_dir=args.csv_dir, output_path=args.output)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
