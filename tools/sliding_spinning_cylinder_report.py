# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Build a self-contained HTML report from sliding-spinning-cylinder CSVs.

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

DEFAULT_CSV_DIR = Path("output") / "H5_sliding_spinning_cylinder"
DEFAULT_HTML_PATH = DEFAULT_CSV_DIR / "sliding_spinning_cylinder_report.html"
HYPOTHESIS_RECORD_NAME = "H5_sliding_spinning_cylinder_hypothesis.md"
HYPOTHESIS_RECORD_SOURCE = Path(__file__).resolve().parents[1] / "hypothesis" / HYPOTHESIS_RECORD_NAME
TIMESERIES_CSV = "sliding_spinning_timeseries.csv"
SUMMARY_CSV = "sliding_spinning_summary.csv"

PAGE_TITLE = "H5: Sliding-Spinning Cylinder Contact Reduction"
SPEED_COLOR = "#0891b2"
SPIN_COLOR = "#7c3aed"


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
    summaries: dict[float, dict[str, dict[str, str]]],
    *,
    selected_epsilon: float,
) -> str:
    rows_by_mode = runs[selected_epsilon]
    reference_row = summaries[selected_epsilon].get("unreduced") or next(iter(summaries[selected_epsilon].values()))
    epsilon_reference = rc.as_float(reference_row, "epsilon_reference")
    times: list[float] = []
    for mode in rc.MODES:
        rows = rows_by_mode.get(mode)
        if rows:
            times = [rc.as_float(row, "time_s") for row in rows]
            break

    epsilon_series = [
        rc.Series(times, [epsilon_reference for _ in times], "reference epsilon", rc.REFERENCE_COLOR, dash="5 4")
    ]
    epsilon_series.extend(rc.mode_series(rows_by_mode, x_key="time_s", y_key="epsilon"))
    speed_series = rc.mode_series(rows_by_mode, x_key="time_s", y_key="horizontal_speed_m_per_s")
    omega_series = rc.mode_series(rows_by_mode, x_key="time_s", y_key="cylinder_omega_z_rad_per_s")

    all_times = [x for series in epsilon_series for x in series.xs]
    return rc.figure_grid(
        [
            rc.Figure(
                title="Coupling ratio",
                xlabel="time [s]",
                ylabel="epsilon = v / (R |omega|)",
                series=epsilon_series,
                x_range=rc.padded_range(all_times, include=(0.0,), floor_span=0.05),
                y_range=rc.padded_range(
                    [y for series in epsilon_series for y in series.ys],
                    include=(epsilon_reference,),
                    floor_span=0.2,
                ),
            ),
            rc.Figure(
                title="Horizontal speed",
                xlabel="time [s]",
                ylabel="speed [m/s]",
                series=speed_series,
                x_range=rc.padded_range(all_times, include=(0.0,), floor_span=0.05),
                y_range=rc.padded_range(
                    [y for series in speed_series for y in series.ys], include=(0.0,), floor_span=0.05
                ),
            ),
            rc.Figure(
                title="Yaw rate",
                xlabel="time [s]",
                ylabel="omega_z [rad/s]",
                series=omega_series,
                x_range=rc.padded_range(all_times, include=(0.0,), floor_span=0.05),
                y_range=rc.padded_range(
                    [y for series in omega_series for y in series.ys], include=(0.0,), floor_span=1.0
                ),
            ),
        ]
    )


def _figure_solver_history(
    runs: dict[float, dict[str, list[dict[str, str]]]],
    *,
    selected_epsilon: float,
) -> str:
    rows_by_mode = runs[selected_epsilon]
    fx_series = rc.mode_series(rows_by_mode, x_key="time_s", y_key="solver_fx_N")
    tz_series = rc.mode_series(rows_by_mode, x_key="time_s", y_key="solver_tz_Nm", scale=1000.0)
    contact_series = rc.mode_series(rows_by_mode, x_key="time_s", y_key="solver_force_count")
    all_times = [x for series in fx_series for x in series.xs]
    return rc.figure_grid(
        [
            rc.Figure(
                title="Solver force along sliding direction",
                xlabel="time [s]",
                ylabel="Fx [N]",
                series=fx_series,
                x_range=rc.padded_range(all_times, include=(0.0,), floor_span=0.05),
                y_range=rc.padded_range(
                    [y for series in fx_series for y in series.ys], include=(0.0,), floor_span=0.05
                ),
            ),
            rc.Figure(
                title="Solver yaw torque",
                xlabel="time [s]",
                ylabel="Tz [mN m]",
                series=tz_series,
                x_range=rc.padded_range(all_times, include=(0.0,), floor_span=0.05),
                y_range=rc.padded_range([y for series in tz_series for y in series.ys], include=(0.0,), floor_span=0.5),
            ),
            rc.Figure(
                title="Solver contact count",
                xlabel="time [s]",
                ylabel="contacts",
                series=contact_series,
                x_range=rc.padded_range(all_times, include=(0.0,), floor_span=0.05),
                y_range=rc.padded_range(
                    [y for series in contact_series for y in series.ys],
                    include=(0.0,),
                    floor_span=50.0,
                ),
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
        _reference_series(summaries, key="epsilon_reference", label="reference epsilon"),
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
    ratio_xs = []
    ratio_ys = []
    for epsilon0 in epsilons:
        off = summaries[epsilon0].get("unreduced")
        on = summaries[epsilon0].get("reduced")
        if off is None or on is None:
            continue
        off_count = rc.as_float(off, "mean_solver_force_count")
        ratio_xs.append(epsilon0)
        ratio_ys.append(rc.as_float(on, "mean_solver_force_count") / off_count if off_count > 0.0 else float("nan"))
    ratio_series = [rc.Series(ratio_xs, ratio_ys, "reduce on / reduce off", "#059669", draw_marker=True)]

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
            rc.Figure(
                title="Contact count ratio",
                xlabel="initial epsilon",
                ylabel="ratio",
                series=ratio_series,
                x_range=x_range,
                y_range=rc.padded_range(ratio_ys, include=(0.0,), floor_span=0.05),
                x_ticks=x_ticks,
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
            "epsilon0 = "
            f"{rc.format_number(epsilon0)}: late epsilon off/on = "
            f"{rc.format_number(rc.as_float(off, 'late_epsilon'))} / "
            f"{rc.format_number(rc.as_float(on, 'late_epsilon'))}; "
            "speed stop off/on = "
            f"{rc.format_number(rc.as_float(off, 'speed_stop_time_s'))} / "
            f"{rc.format_number(rc.as_float(on, 'speed_stop_time_s'))} s; "
            "spin stop off/on = "
            f"{rc.format_number(rc.as_float(off, 'spin_stop_time_s'))} / "
            f"{rc.format_number(rc.as_float(on, 'spin_stop_time_s'))} s; "
            f"mean solver contacts off/on = {rc.format_number(off_count)} / {rc.format_number(on_count)}."
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
            "<p>Figure 1 checks the coupled state response. A mismatch here means the reduced contact set is changing "
            "how friction is split between translation and yaw spin.</p>\n"
            + _figure_state_history(runs, summaries, selected_epsilon=selected_epsilon),
        ),
        rc.TabPanel(
            "Figure 2",
            "<p>Figure 2 checks the direct solver outputs behind that state response: force along the initial sliding "
            "direction, yaw torque, and solver contact count.</p>\n"
            + _figure_solver_history(runs, selected_epsilon=selected_epsilon),
        ),
        rc.TabPanel(
            "Figure 3",
            "<p>Figure 3 summarizes the sweep across initial coupling ratios. It keeps the non-stopped cases visible "
            "through final horizontal speed instead of hiding them behind missing stop times.</p>\n"
            + _figure_sweep_summary(summaries),
        ),
    ]

    body = "\n".join(
        [
            f"<h1>{rc.escape(PAGE_TITLE)}</h1>",
            "<p>This experiment tests whether contact reduction preserves the coupled friction response of a flat "
            "cylinder that is both sliding and yaw-spinning on a flat hydroelastic plate. Torsional and rolling "
            "friction are zero, so translation and spin must both change through solver sliding-friction forces "
            "distributed over the same contact patch.</p>",
            "<p>The comparison is reduce off versus reduce on, with pre-prune off in both modes. The sweep variable is "
            "<code>epsilon0 = v0 / (R * omega0)</code>; the logged coupling ratio is "
            "<code>epsilon = v / (R * |omega_z|)</code>.</p>",
            f"<p>CSV data currently contains epsilon0 values: {epsilons}. Initial yaw rate is "
            f"{rc.format_number(initial_omega)} rad/s. Figure time histories use epsilon0 = "
            f"{rc.format_number(selected_epsilon)}.</p>",
            "<h2>Measured Quantities</h2>",
            "<ul>",
            "<li>Solver force and torque on the cylinder.</li>",
            "<li>Horizontal speed, yaw rate, and coupling ratio <code>epsilon</code>.</li>",
            "<li>Translation stop time, spin stop time, final speed, final drift, tilt, and penetration depth.</li>",
            "<li>Solver contact count, raw face contact count, rigid contact count, and contact-buffer validity "
            "flags.</li>",
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
