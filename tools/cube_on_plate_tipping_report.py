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

PAGE_TITLE = "H3: Cube-on-Plate Tipping"


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


def _figure_primary(rows_by_mode: dict[str, list[dict[str, str]]], *, weight_n: float, ftip_n: float) -> str:
    """Figure 1: directly-measured primary quantities with their analytic references."""

    pitch_series = rc.mode_series(rows_by_mode, x_key="applied_force_over_ftip", y_key="cube_pitch_deg")
    fz_series = rc.mode_series(rows_by_mode, x_key="applied_force_over_ftip", y_key="solver_fz_N")
    fx_series = rc.mode_series(rows_by_mode, x_key="applied_force_over_ftip", y_key="solver_fx_N")
    all_xs = [x for series in pitch_series for x in series.xs]
    x_range = rc.padded_range(all_xs, include=(0.0, 1.0), floor_span=0.2)
    x_max = max(all_xs or [1.0])
    # Pre-slide static equilibrium: the contact friction reaction balances the applied
    # push, so solver_fx = -F_applied = -(F/Ftip) * Ftip.
    static_balance = rc.Series(
        [0.0, x_max], [0.0, -x_max * ftip_n], "static balance Fx = -F_applied", rc.REFERENCE_COLOR, dash="2 3"
    )
    tip_onset = ((1.0, "tip onset", rc.REFERENCE_COLOR),)
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
                hlines=((0.0, "rigid: zero pre-tip pitch", rc.REFERENCE_COLOR),),
                vlines=tip_onset,
            ),
            rc.Figure(
                title="Solver vertical support force",
                xlabel="applied force / analytic tip force",
                ylabel="Fz [N]",
                series=fz_series,
                x_range=x_range,
                y_range=rc.padded_range(
                    [y for series in fz_series for y in series.ys] + [weight_n], include=(0.0,), floor_span=0.5
                ),
                hlines=((weight_n, "weight m*g", rc.REFERENCE_COLOR),),
                vlines=tip_onset,
            ),
            rc.Figure(
                title="Solver horizontal reaction force",
                xlabel="applied force / analytic tip force",
                ylabel="Fx [N]",
                series=[static_balance, *fx_series],
                x_range=x_range,
                y_range=rc.padded_range(
                    [y for series in fx_series for y in series.ys] + [-x_max * ftip_n], include=(0.0,), floor_span=0.5
                ),
                vlines=tip_onset,
            ),
        ]
    )


def _figure_additional(rows_by_mode: dict[str, list[dict[str, str]]], summaries: dict[str, dict[str, str]]) -> str:
    """Figure 2: derived/secondary quantities — center-pressure shift, drift, event force."""

    # Physical center of pressure = normal-force offset, removing the base friction torque
    # (-Ty/Fz folds in friction and reads half the true offset): cop/he = (-Ty/Fz)/he - Fx/Fz.
    cop_series = []
    for mode in rc.MODES:
        rows = rows_by_mode.get(mode)
        if not rows:
            continue
        cop_xs = [rc.as_float(row, "applied_force_over_ftip") for row in rows]
        cop_ys = []
        for row in rows:
            fz = rc.as_float(row, "solver_fz_N")
            fx = rc.as_float(row, "solver_fx_N")
            cop_he = rc.as_float(row, "center_pressure_x_over_half_extent")
            cop_ys.append((cop_he - fx / fz) if fz else float("nan"))
        cop_series.append(rc.Series(cop_xs, cop_ys, rc.MODE_LABELS[mode], rc.MODE_COLORS[mode]))
    x_series = rc.mode_series(rows_by_mode, x_key="applied_force_over_ftip", y_key="cube_x_m", scale=1000.0)
    all_xs = [x for series in x_series for x in series.xs]
    x_range = rc.padded_range(all_xs, include=(0.0, 1.0), floor_span=0.2)
    analytic_cop = rc.Series([0.0, 1.0], [0.0, 1.0], "analytic cop = F/Ftip", rc.REFERENCE_COLOR, dash="2 3")

    event_series = []
    for idx, mode in enumerate(rc.MODES):
        row = summaries.get(mode)
        if row is None:
            continue
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
    return rc.figure_grid(
        [
            rc.Figure(
                title="Center-pressure shift (derived)",
                xlabel="applied force / analytic tip force",
                ylabel="cop_x / half extent",
                series=[analytic_cop, *cop_series],
                x_range=x_range,
                y_range=rc.padded_range(
                    [y for series in cop_series for y in series.ys], include=(0.0, 1.0), floor_span=0.2
                ),
                hlines=((1.0, "front edge", rc.REFERENCE_COLOR),),
                vlines=((1.0, "tip onset", rc.REFERENCE_COLOR),),
            ),
            rc.Figure(
                title="Horizontal drift",
                xlabel="applied force / analytic tip force",
                ylabel="x [mm]",
                series=x_series,
                x_range=x_range,
                y_range=rc.padded_range([y for series in x_series for y in series.ys], include=(0.0,), floor_span=1.0),
            ),
            rc.Figure(
                title="First event force",
                xlabel="mode index: 0 off, 1 on",
                ylabel="force / F_tip",
                series=event_series,
                x_range=(-0.25, 1.25),
                y_range=rc.padded_range(
                    [y for series in event_series for y in series.ys], include=(1.0,), floor_span=0.2
                ),
                x_ticks=[0.0, 1.0],
                hlines=((1.0, "analytic tip (F/Ftip = 1)", rc.REFERENCE_COLOR),),
            ),
        ]
    )


def _checks_table(summaries: dict[str, dict[str, str]]) -> str:
    """Figure 3: contact counts, buffer utilization, and validity gates as a table."""

    headers = [
        "mode",
        "mean solver contacts",
        "mean rigid contacts",
        "max rigid contacts",
        "rigid capacity",
        "overflow",
        "hash failures",
        "state valid",
    ]
    rows = []
    for mode in rc.MODES:
        row = summaries.get(mode)
        if row is None:
            continue
        rows.append(
            [
                rc.MODE_LABELS.get(mode, mode),
                rc.format_number(rc.as_float(row, "mean_solver_force_count"), precision=4),
                rc.format_number(rc.as_float(row, "mean_rigid_contact_count"), precision=4),
                rc.format_number(rc.as_float(row, "max_rigid_contact_count"), precision=4),
                rc.format_number(rc.as_float(row, "rigid_contact_capacity"), precision=4),
                row.get("buffer_overflow", "n/a"),
                rc.format_number(rc.as_float(row, "max_reduction_hashtable_failures"), precision=4),
                "yes" if not rc.as_bool(row.get("state_invalid", "false")) else "no",
            ]
        )
    return rc.data_table(headers, rows)


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
    weight_n = rc.as_float(ref, "cube_weight_N")
    ftip_n = rc.as_float(ref, "analytic_tip_force_N")
    tabs = rc.figure_tabs(
        [
            rc.TabPanel(
                "Figure 1",
                "<p>Figure 1: primary, directly-measured quantities versus applied force, each with its analytic "
                "reference &mdash; cube pitch (rigid: zero pre-tip pitch), solver vertical support (weight m*g), and "
                "solver horizontal reaction (static balance Fx = -F_applied). The vertical line marks the analytic "
                f"tip onset at F/Ftip = 1 (F = m*g/2 = {rc.format_number(ftip_n)} N), where the contact point reaches "
                "the front edge and the cube begins to tip.</p>\n"
                + _figure_primary(rows_by_mode, weight_n=weight_n, ftip_n=ftip_n),
            ),
            rc.TabPanel(
                "Figure 2",
                "<p>Figure 2: additional derived response &mdash; the center-pressure shift "
                "(cop_x/half_extent = F/Ftip, reaching the front edge at the tip-onset line), horizontal drift, and "
                "the first tip/slide event force. Note the recorded event sits just past F/Ftip = 1 because it is "
                "flagged at the 10 deg tilt threshold, a moment after the analytic onset.</p>\n"
                + _figure_additional(rows_by_mode, summaries),
            ),
            rc.TabPanel(
                "Figure 3",
                "<p>Figure 3: contact counts, buffer utilization, and validity gates.</p>\n" + _checks_table(summaries),
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
            "<h2>Reference</h2>",
            (
                "<p>Rigid-body quasi-static tipping, with no compliant model needed. With the push at the top "
                "face the cube tips about its front bottom edge at <code>F_tip = m*g/2</code>, and would slide "
                f"at <code>F_slide = mu*m*g</code> (here mu = {mu}, above 0.5, so it tips before it slides). "
                "Before tipping the cube is in static equilibrium, so the center of pressure (the normal-force "
                "offset, <code>-(Ty + h*Fx)/Fz</code>) grows linearly with the applied force, "
                "<code>cop_x/half_extent = F/F_tip</code>, reaching the front edge exactly at the tip force. A "
                "rigid cube has zero pre-tip pitch; any pitch before the tip is a contact-compliance effect.</p>"
            ),
            "<h2>Measured Quantities</h2>",
            "<p>Primary (Figure 1, directly measured):</p>",
            "<ul>",
            "<li>Cube pitch versus applied force.</li>",
            "<li>Solver vertical support force <code>Fz</code>.</li>",
            "<li>Solver horizontal reaction force <code>Fx</code>.</li>",
            "</ul>",
            "<p>Secondary (derived / checks):</p>",
            "<ul>",
            "<li>Center of pressure from the solver wrench with the base friction torque removed: "
            "<code>cop_x = -(Ty + h*Fx) / Fz</code>.</li>",
            "<li>Horizontal drift, and the first tip or slide event force.</li>",
            "<li>Solver, rigid, and raw face contact counts with buffer validity flags.</li>",
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
