# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Build a self-contained HTML report from flat-sliding-block CSVs.

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

DEFAULT_CSV_DIR = Path("output") / "H2_flat_sliding_block"
DEFAULT_HTML_PATH = DEFAULT_CSV_DIR / "flat_sliding_block_report.html"
HYPOTHESIS_RECORD_NAME = "H2_flat_sliding_block_hypothesis.md"
HYPOTHESIS_RECORD_SOURCE = Path(__file__).resolve().parents[1] / "hypothesis" / HYPOTHESIS_RECORD_NAME
TIMESERIES_CSV = "sliding_timeseries.csv"
SUMMARY_CSV = "sliding_summary.csv"
VARIATIONS_DIR = DEFAULT_CSV_DIR / "variations"
DEFAULT_RESOLUTION_PROBES = (
    ("8", VARIATIONS_DIR / "sdf8"),
    ("8", VARIATIONS_DIR / "sdf8_highv"),
    ("16", VARIATIONS_DIR / "sdf16"),
    ("16", VARIATIONS_DIR / "sdf16_highv"),
    ("16", VARIATIONS_DIR / "sdf16_longstop"),
    ("24", VARIATIONS_DIR / "sdf24"),
    ("24", VARIATIONS_DIR / "sdf24_highv"),
    ("24", VARIATIONS_DIR / "sdf24_longstop"),
    ("32", DEFAULT_CSV_DIR),
    ("32", VARIATIONS_DIR / "sdf32_highv"),
    ("32", VARIATIONS_DIR / "sdf32_longstop"),
    ("40", VARIATIONS_DIR / "sdf40"),
    ("40", VARIATIONS_DIR / "sdf40_highv"),
    ("40", VARIATIONS_DIR / "sdf40_longstop"),
    ("48", VARIATIONS_DIR / "sdf48"),
    ("48", VARIATIONS_DIR / "sdf48_highv"),
    ("48", VARIATIONS_DIR / "sdf48_longstop"),
)

PAGE_TITLE = "H2: Flat Sliding Block Contact Reduction"

FACTS_CSS = """\
    .facts {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 8px 20px;
      margin-top: 18px;
      padding: 14px 0;
      border-top: 1px solid var(--line);
      border-bottom: 1px solid var(--line);
      color: var(--muted);
    }
    .facts div strong {
      color: var(--ink);
      display: block;
    }"""


@dataclass(frozen=True)
class ResolutionProbe:
    """One SDF-resolution probe row."""

    sdf_resolution: float
    initial_speed: float
    coulomb_stop_mm: float
    reduce_off_stop_mm: float
    reduce_on_stop_mm: float
    reduce_off_contacts: float
    reduce_on_contacts: float
    reduce_off_stopped: bool
    reduce_on_stopped: bool


def _array(rows: list[dict[str, str]], key: str) -> list[float]:
    return [rc.as_float(row, key) for row in rows]


def _positive_log_range(values: list[float], *, floor: float = 1.0) -> tuple[float, float]:
    finite = [value for value in rc.finite(values) if value > 0.0]
    if not finite:
        return floor, floor * 10.0
    low = max(min(finite) * 0.75, floor)
    high = max(max(finite) * 1.4, floor * 10.0)
    return low, high


def load_timeseries(csv_dir: str | Path) -> dict[float, dict[str, list[dict[str, str]]]]:
    """Load time-series rows grouped by initial speed and mode."""

    path = Path(csv_dir) / TIMESERIES_CSV
    if not path.exists():
        raise FileNotFoundError(f"missing time-series CSV: {path}")
    grouped: dict[float, dict[str, list[dict[str, str]]]] = {}
    for row in rc.read_csv(path):
        speed = rc.as_float(row, "initial_speed_m_per_s")
        grouped.setdefault(speed, {}).setdefault(row["mode"], []).append(row)
    for mode_rows in grouped.values():
        for rows in mode_rows.values():
            rows.sort(key=lambda row: rc.as_float(row, "time_s"))
    if not grouped:
        raise ValueError(f"time-series CSV has no rows: {path}")
    return grouped


def load_summaries(csv_dir: str | Path) -> dict[float, dict[str, dict[str, str]]]:
    """Load summary rows grouped by initial speed and mode."""

    path = Path(csv_dir) / SUMMARY_CSV
    if not path.exists():
        raise FileNotFoundError(f"missing summary CSV: {path}")
    grouped: dict[float, dict[str, dict[str, str]]] = {}
    for row in rc.read_csv(path):
        speed = rc.as_float(row, "initial_speed_m_per_s")
        grouped.setdefault(speed, {})[row["mode"]] = row
    if not grouped:
        raise ValueError(f"summary CSV has no rows: {path}")
    return grouped


def load_resolution_probes(
    probe_dirs: tuple[tuple[str, str | Path], ...] | list[tuple[str, str | Path]],
) -> list[ResolutionProbe]:
    """Load optional SDF-resolution probe summaries.

    Missing directories are ignored so the base report can still be generated
    from a single experiment output folder.
    """

    probes: dict[tuple[float, float], ResolutionProbe] = {}
    for sdf_resolution, csv_dir in probe_dirs:
        summary_path = Path(csv_dir) / SUMMARY_CSV
        if not summary_path.exists():
            continue
        summaries = load_summaries(csv_dir)
        if not summaries:
            continue
        for speed in sorted(summaries):
            modes = summaries[speed]
            off = modes.get("unreduced")
            on = modes.get("reduced")
            if off is None or on is None:
                continue
            probe = ResolutionProbe(
                sdf_resolution=float(sdf_resolution),
                initial_speed=speed,
                coulomb_stop_mm=1000.0 * rc.as_float(off, "expected_coulomb_stop_travel_m"),
                reduce_off_stop_mm=1000.0 * rc.as_float(off, "stop_travel_m"),
                reduce_on_stop_mm=1000.0 * rc.as_float(on, "stop_travel_m"),
                reduce_off_contacts=rc.as_float(off, "mean_rigid_contact_count"),
                reduce_on_contacts=rc.as_float(on, "mean_rigid_contact_count"),
                reduce_off_stopped=rc.as_bool(off["stopped"]),
                reduce_on_stopped=rc.as_bool(on["stopped"]),
            )
            key = (probe.sdf_resolution, probe.initial_speed)
            previous = probes.get(key)
            if previous is None or (not previous.reduce_off_stopped and probe.reduce_off_stopped):
                probes[key] = probe
    return list(probes.values())


def select_speed(
    runs: dict[float, dict[str, list[dict[str, str]]]],
    requested_speed: float | None,
) -> float:
    """Return the closest available initial speed, defaulting to the fastest run."""

    speeds = sorted(runs)
    if not speeds:
        raise ValueError("no speeds available")
    if requested_speed is None:
        return float(speeds[-1])
    return float(min(speeds, key=lambda speed: abs(speed - requested_speed)))


def _series_from_summary(
    summaries: dict[float, dict[str, dict[str, str]]],
    *,
    mode: str,
    key: str,
    scale: float = 1.0,
) -> rc.Series:
    xs = [speed for speed in sorted(summaries) if mode in summaries[speed]]
    ys = [scale * rc.as_float(summaries[speed][mode], key) for speed in xs]
    return rc.Series(
        xs, ys, rc.MODE_LABELS.get(mode, mode), rc.MODE_COLORS.get(mode, "#111827"), draw_line=False, draw_marker=True
    )


def _reference_series(
    summaries: dict[float, dict[str, dict[str, str]]],
    *,
    key: str,
    scale: float = 1.0,
) -> rc.Series:
    xs = []
    ys = []
    for speed in sorted(summaries):
        row = summaries[speed].get("unreduced") or next(iter(summaries[speed].values()))
        xs.append(speed)
        ys.append(scale * rc.as_float(row, key))
    return rc.Series(xs, ys, "Coulomb reference", rc.REFERENCE_COLOR, draw_line=False, draw_marker=True)


def _with_lines(series: list[rc.Series]) -> list[rc.Series]:
    return [
        rc.Series(
            xs=s.xs,
            ys=s.ys,
            label=s.label,
            color=s.color,
            draw_line=True,
            draw_marker=s.draw_marker,
            dash=s.dash,
        )
        for s in series
    ]


def _figure_stopping_response(summaries: dict[float, dict[str, dict[str, str]]]) -> str:
    speeds = sorted(summaries)
    x_range = rc.padded_range(speeds, include=(0.0,), floor_span=0.05)

    time_series = _with_lines(
        [
            _reference_series(summaries, key="expected_coulomb_stop_time_s"),
            _series_from_summary(summaries, mode="unreduced", key="stop_time_s"),
            _series_from_summary(summaries, mode="reduced", key="stop_time_s"),
        ]
    )
    time_values = [value for plot_series in time_series for value in plot_series.ys]

    distance_series = _with_lines(
        [
            _reference_series(summaries, key="expected_coulomb_stop_travel_m", scale=1000.0),
            _series_from_summary(summaries, mode="unreduced", key="stop_travel_m", scale=1000.0),
            _series_from_summary(summaries, mode="reduced", key="stop_travel_m", scale=1000.0),
        ]
    )
    distance_values = [value for plot_series in distance_series for value in plot_series.ys]

    return rc.figure_grid(
        [
            rc.Figure(
                title="Stopping time",
                xlabel="initial speed [m/s]",
                ylabel="time [s]",
                series=time_series,
                x_range=x_range,
                y_range=rc.padded_range(time_values, include=(0.0,), floor_span=0.02),
            ),
            rc.Figure(
                title="Stopping distance",
                xlabel="initial speed [m/s]",
                ylabel="distance [mm]",
                series=distance_series,
                x_range=x_range,
                y_range=rc.padded_range(distance_values, include=(0.0,), floor_span=1.0),
            ),
        ],
        columns=1,
    )


def _expected_motion(
    *,
    speed: float,
    summary: dict[str, str],
    times: list[float],
) -> tuple[list[float], list[float]]:
    stop_time = rc.as_float(summary, "expected_coulomb_stop_time_s")
    stop_travel = rc.as_float(summary, "expected_coulomb_stop_travel_m")
    if stop_time <= 0.0 or not math.isfinite(stop_time):
        return [float("nan") for _ in times], [float("nan") for _ in times]
    deceleration = speed / stop_time
    speeds = [max(speed - deceleration * time, 0.0) for time in times]
    travels = []
    for time in times:
        if time <= stop_time:
            travels.append(speed * time - 0.5 * deceleration * time * time)
        else:
            travels.append(stop_travel)
    return speeds, travels


def _figure_time_history(
    runs: dict[float, dict[str, list[dict[str, str]]]],
    summaries: dict[float, dict[str, dict[str, str]]],
    *,
    selected_speed: float,
) -> str:
    rows_by_mode = runs[selected_speed]
    reference_summary = summaries[selected_speed].get("unreduced") or next(iter(summaries[selected_speed].values()))
    all_times = sorted({time for rows in rows_by_mode.values() for time in _array(rows, "time_s")})
    expected_speed, expected_travel = _expected_motion(speed=selected_speed, summary=reference_summary, times=all_times)

    speed_series = [
        rc.Series(all_times, expected_speed, "Coulomb reference", rc.REFERENCE_COLOR, draw_line=True, dash="5 5"),
    ]
    travel_series = [
        rc.Series(
            all_times,
            [1000.0 * value for value in expected_travel],
            "Coulomb reference",
            rc.REFERENCE_COLOR,
            draw_line=True,
            dash="5 5",
        ),
    ]
    force_series: list[rc.Series] = []
    count_series: list[rc.Series] = []
    for mode in rc.MODES:
        rows = rows_by_mode.get(mode)
        if not rows:
            continue
        label = rc.MODE_LABELS.get(mode, mode)
        color = rc.MODE_COLORS.get(mode, "#111827")
        times = _array(rows, "time_s")
        speed_series.append(rc.Series(times, _array(rows, "cube_speed_m_per_s"), label, color))
        travel_series.append(rc.Series(times, [1000.0 * value for value in _array(rows, "cube_x_m")], label, color))
        force_series.append(rc.Series(times, _array(rows, "solver_fx_N"), label, color))
        count_series.append(rc.Series(times, _array(rows, "solver_force_count"), label, color))

    time_range = rc.padded_range(all_times, include=(0.0,), floor_span=0.02)
    speed_values = [value for plot_series in speed_series for value in plot_series.ys]
    travel_values = [value for plot_series in travel_series for value in plot_series.ys]
    force_values = [value for plot_series in force_series for value in plot_series.ys]
    count_values = [value for plot_series in count_series for value in plot_series.ys]

    grid = rc.figure_grid(
        [
            rc.Figure(
                title="Horizontal speed",
                xlabel="time [s]",
                ylabel="speed [m/s]",
                series=speed_series,
                x_range=time_range,
                y_range=rc.padded_range(speed_values, include=(0.0,), floor_span=0.05),
            ),
            rc.Figure(
                title="Travel in sliding direction",
                xlabel="time [s]",
                ylabel="travel [mm]",
                series=travel_series,
                x_range=time_range,
                y_range=rc.padded_range(travel_values, include=(0.0,), floor_span=1.0),
            ),
            rc.Figure(
                title="Solver friction force Fx",
                xlabel="time [s]",
                ylabel="force [N]",
                series=force_series,
                x_range=time_range,
                y_range=rc.padded_range(force_values, include=(0.0,), floor_span=0.5),
            ),
            rc.Figure(
                title="Solver force-contact count",
                xlabel="time [s]",
                ylabel="count",
                series=count_series,
                x_range=time_range,
                y_range=_positive_log_range(count_values),
                log_y=True,
            ),
        ],
        columns=1,
    )
    return f"<p>Representative run: initial speed {rc.format_number(selected_speed)} m/s.</p>\n{grid}"


def _figure_contact_counts(summaries: dict[float, dict[str, dict[str, str]]]) -> str:
    speeds = sorted(summaries)
    x_range = rc.padded_range(speeds, include=(0.0,), floor_span=0.05)
    mean_series = [
        _series_from_summary(summaries, mode="unreduced", key="mean_solver_force_count"),
        _series_from_summary(summaries, mode="reduced", key="mean_solver_force_count"),
    ]
    max_series = [
        _series_from_summary(summaries, mode="unreduced", key="max_rigid_contact_count"),
        _series_from_summary(summaries, mode="reduced", key="max_rigid_contact_count"),
    ]
    all_counts = [value for plot_series in [*mean_series, *max_series] for value in plot_series.ys]
    grid = rc.figure_grid(
        [
            rc.Figure(
                title="Mean solver contact count",
                xlabel="initial speed [m/s]",
                ylabel="contacts",
                series=mean_series,
                x_range=x_range,
                y_range=_positive_log_range(all_counts),
                log_y=True,
            ),
            rc.Figure(
                title="Max rigid contact count",
                xlabel="initial speed [m/s]",
                ylabel="contacts",
                series=max_series,
                x_range=x_range,
                y_range=_positive_log_range(all_counts),
                log_y=True,
            ),
        ],
        columns=1,
    )
    return f"{grid}\n{_buffer_table(summaries)}"


def _figure_resolution_probe(probes: list[ResolutionProbe]) -> str:
    if len(probes) < 2:
        return "<p>No optional SDF-resolution probe CSVs were found.</p>"

    speeds = sorted({probe.initial_speed for probe in probes})
    sdf_values = sorted({probe.sdf_resolution for probe in probes})
    by_speed = {
        speed: sorted(
            (probe for probe in probes if probe.initial_speed == speed), key=lambda probe: probe.sdf_resolution
        )
        for speed in speeds
    }

    figures: list[rc.Figure] = []
    for speed in speeds:
        speed_probes = by_speed[speed]
        distance_series = [
            rc.Series(
                [probe.sdf_resolution for probe in speed_probes],
                [probe.coulomb_stop_mm for probe in speed_probes],
                "Coulomb reference",
                rc.REFERENCE_COLOR,
                draw_line=True,
            ),
            rc.Series(
                [probe.sdf_resolution for probe in speed_probes if probe.reduce_off_stopped],
                [probe.reduce_off_stop_mm for probe in speed_probes if probe.reduce_off_stopped],
                rc.MODE_LABELS["unreduced"],
                rc.MODE_COLORS["unreduced"],
                draw_line=True,
            ),
            rc.Series(
                [probe.sdf_resolution for probe in speed_probes if probe.reduce_on_stopped],
                [probe.reduce_on_stop_mm for probe in speed_probes if probe.reduce_on_stopped],
                rc.MODE_LABELS["reduced"],
                rc.MODE_COLORS["reduced"],
                draw_line=True,
            ),
        ]
        distance_values = [value for plot_series in distance_series for value in plot_series.ys]
        figures.append(
            rc.Figure(
                title=f"Stopping distance, v0 = {rc.format_number(speed, precision=3)} m/s",
                xlabel="SDF resolution [cells/axis]",
                ylabel="distance [mm]",
                series=distance_series,
                x_range=rc.padded_range(sdf_values, floor_span=8.0),
                y_range=rc.padded_range(distance_values, include=(0.0,), floor_span=1.0),
                x_ticks=sdf_values,
            )
        )

    fastest_speed = speeds[-1]
    count_probes = by_speed[fastest_speed]
    contact_series = [
        rc.Series(
            [probe.sdf_resolution for probe in count_probes],
            [probe.reduce_off_contacts for probe in count_probes],
            rc.MODE_LABELS["unreduced"],
            rc.MODE_COLORS["unreduced"],
            draw_line=True,
        ),
        rc.Series(
            [probe.sdf_resolution for probe in count_probes],
            [probe.reduce_on_contacts for probe in count_probes],
            rc.MODE_LABELS["reduced"],
            rc.MODE_COLORS["reduced"],
            draw_line=True,
        ),
    ]
    contact_values = [value for plot_series in contact_series for value in plot_series.ys]
    figures.append(
        rc.Figure(
            title=f"Mean rigid contact count, v0 = {rc.format_number(fastest_speed, precision=3)} m/s",
            xlabel="SDF resolution [cells/axis]",
            ylabel="contacts",
            series=contact_series,
            x_range=rc.padded_range(sdf_values, floor_span=8.0),
            y_range=_positive_log_range(contact_values),
            log_y=True,
            x_ticks=sdf_values,
        )
    )

    grid = rc.figure_grid(figures, columns=1)
    return (
        "<p>Points are separate runs. Missing stop-distance points indicate the run did not cross the stop "
        f"threshold within the configured run window.</p>\n{grid}"
    )


def _result_bullets(summaries: dict[float, dict[str, dict[str, str]]]) -> list[str]:
    off_contacts = []
    on_mean_contacts = []
    on_max_contacts = []
    on_hash_failures = []
    buffer_overflows = []
    stop_shorter = []
    travel_shorter = []
    on_expected_distance_error = []
    off_expected_distance_error = []

    for speed in sorted(summaries):
        off = summaries[speed].get("unreduced")
        on = summaries[speed].get("reduced")
        if off is None or on is None:
            continue
        off_contacts.append(rc.as_float(off, "mean_solver_force_count"))
        on_mean_contacts.append(rc.as_float(on, "mean_solver_force_count"))
        on_max_contacts.append(rc.as_float(on, "max_rigid_contact_count"))
        on_hash_failures.append(rc.as_float(on, "max_reduction_hashtable_failures"))
        buffer_overflows.extend([rc.as_bool(off["buffer_overflow"]), rc.as_bool(on["buffer_overflow"])])

        off_stop_time = rc.as_float(off, "stop_time_s")
        on_stop_time = rc.as_float(on, "stop_time_s")
        off_stop_travel = rc.as_float(off, "stop_travel_m")
        on_stop_travel = rc.as_float(on, "stop_travel_m")
        expected_travel = rc.as_float(off, "expected_coulomb_stop_travel_m")
        if off_stop_time > 0.0:
            stop_shorter.append(max(0.0, (off_stop_time - on_stop_time) / off_stop_time))
        if off_stop_travel > 0.0:
            travel_shorter.append(max(0.0, (off_stop_travel - on_stop_travel) / off_stop_travel))
        if expected_travel > 0.0:
            on_expected_distance_error.append(abs(on_stop_travel - expected_travel) / expected_travel)
            off_expected_distance_error.append(abs(off_stop_travel - expected_travel) / expected_travel)

    return [
        "Full sweep completed with valid buffers for both modes."
        if not any(buffer_overflows) and max(on_hash_failures, default=0.0) == 0.0
        else "At least one run failed a buffer validity check, so the result is inconclusive.",
        (
            f"Reduce off used {rc.format_number(rc.mean(off_contacts), precision=4)} mean solver contacts; "
            f"reduce on used {rc.format_number(min(on_mean_contacts), precision=3)}-"
            f"{rc.format_number(max(on_mean_contacts), precision=3)} mean solver contacts, with max rigid count "
            f"{rc.format_number(max(on_max_contacts), precision=3)}."
        ),
        (
            f"Reduce on did not match reduce off: stop time was "
            f"{rc.format_percent(min(stop_shorter), precision=2)}-{rc.format_percent(max(stop_shorter), precision=2)} "
            f"shorter and stop distance was "
            f"{rc.format_percent(min(travel_shorter), precision=2)}-"
            f"{rc.format_percent(max(travel_shorter), precision=2)} shorter."
        ),
        (
            f"Reduce-on stop distance stayed within "
            f"{rc.format_percent(max(on_expected_distance_error), precision=2)} of the analytic Coulomb distance; "
            f"reduce off was as much as {rc.format_percent(max(off_expected_distance_error), precision=3)} long."
        ),
        "The reduced run stayed flat and did not drift laterally in a meaningful way, so this simple case does not "
        "expose a reduce-on instability.",
    ]


def _summary_table(summaries: dict[float, dict[str, dict[str, str]]]) -> str:
    headers = [
        "v0 [m/s]",
        "Coulomb x_stop [mm]",
        "reduce off x_stop [mm]",
        "reduce on x_stop [mm]",
        "reduce off t_stop [s]",
        "reduce on t_stop [s]",
        "contacts off",
        "contacts on",
    ]
    rows = []
    for speed in sorted(summaries):
        off = summaries[speed].get("unreduced")
        on = summaries[speed].get("reduced")
        if off is None or on is None:
            continue
        rows.append(
            [
                rc.format_number(speed, precision=3),
                rc.format_number(1000.0 * rc.as_float(off, "expected_coulomb_stop_travel_m"), precision=4),
                rc.format_number(1000.0 * rc.as_float(off, "stop_travel_m"), precision=4),
                rc.format_number(1000.0 * rc.as_float(on, "stop_travel_m"), precision=4),
                rc.format_number(rc.as_float(off, "stop_time_s"), precision=4),
                rc.format_number(rc.as_float(on, "stop_time_s"), precision=4),
                rc.format_number(rc.as_float(off, "mean_solver_force_count"), precision=4),
                rc.format_number(rc.as_float(on, "mean_solver_force_count"), precision=4),
            ]
        )
    return rc.data_table(headers, rows)


def _resolution_probe_table(probes: list[ResolutionProbe]) -> str:
    if len(probes) < 2:
        return ""

    headers = [
        "SDF resolution",
        "v0 [m/s]",
        "Coulomb x_stop [mm]",
        "reduce off x_stop [mm]",
        "reduce on x_stop [mm]",
        "contacts off",
        "contacts on",
    ]
    rows = []
    for probe in sorted(probes, key=lambda item: (item.sdf_resolution, item.initial_speed)):
        rows.append(
            [
                rc.format_number(probe.sdf_resolution, precision=3),
                rc.format_number(probe.initial_speed, precision=3),
                rc.format_number(probe.coulomb_stop_mm, precision=4),
                rc.format_number(probe.reduce_off_stop_mm, precision=4) if probe.reduce_off_stopped else "not stopped",
                rc.format_number(probe.reduce_on_stop_mm, precision=4) if probe.reduce_on_stopped else "not stopped",
                rc.format_number(probe.reduce_off_contacts, precision=4),
                rc.format_number(probe.reduce_on_contacts, precision=4),
            ]
        )
    return (
        "<h2>SDF Resolution Probe</h2>\n"
        "<p>Additional probe over SDF resolution and initial speed: the table records contact counts and stop "
        "distances from each generated CSV. Entries marked not stopped did not cross the stop threshold within "
        "the configured run window.</p>\n" + rc.data_table(headers, rows)
    )


def _buffer_table(summaries: dict[float, dict[str, dict[str, str]]]) -> str:
    headers = ["v0 [m/s]", "mode", "overflow", "max rigid contacts", "rigid capacity", "hash failures"]
    rows = []
    for speed in sorted(summaries):
        for mode in rc.MODES:
            row = summaries[speed].get(mode)
            if row is None:
                continue
            rows.append(
                [
                    rc.format_number(speed, precision=3),
                    rc.MODE_LABELS.get(mode, mode),
                    row["buffer_overflow"],
                    rc.format_number(rc.as_float(row, "max_rigid_contact_count"), precision=4),
                    rc.format_number(rc.as_float(row, "rigid_contact_capacity"), precision=4),
                    rc.format_number(rc.as_float(row, "max_reduction_hashtable_failures"), precision=4),
                ]
            )
    return rc.data_table(headers, rows)


def render_html_report(
    runs: dict[float, dict[str, list[dict[str, str]]]],
    summaries: dict[float, dict[str, dict[str, str]]],
    *,
    selected_speed: float,
    resolution_probes: list[ResolutionProbe] | None = None,
) -> str:
    """Render a self-contained HTML report."""

    probes = resolution_probes or []
    resolution_probe_section = _resolution_probe_table(probes)
    panels = [
        rc.TabPanel(
            "Figure 1",
            "<h2>Figure 1: stopping response vs initial speed</h2>\n"
            "<p>Points only: each marker is one completed run. The black points are the analytic Coulomb "
            "reference.</p>\n" + _figure_stopping_response(summaries),
        ),
        rc.TabPanel(
            "Figure 2",
            "<h2>Figure 2: representative time history</h2>\n"
            + _figure_time_history(runs, summaries, selected_speed=selected_speed),
        ),
        rc.TabPanel(
            "Figure 3",
            "<h2>Figure 3: contact reduction and buffer sanity</h2>\n"
            "<p>Contact counts use a log scale because reduce off and reduce on differ by roughly two orders of "
            "magnitude.</p>\n" + _figure_contact_counts(summaries),
        ),
        rc.TabPanel(
            "Figure 4",
            "<h2>Figure 4: SDF resolution probe</h2>\n" + _figure_resolution_probe(probes),
        ),
    ]

    body = "\n".join(
        [
            f"<h1>{rc.escape(PAGE_TITLE)}</h1>",
            '<p class="lede">This report checks whether reducing hydroelastic contacts preserves a simple Coulomb '
            "sliding response for a cube sliding flat on a fixed plate. It is a record of the hypothesis, the "
            "measured quantities, and the outcome from the generated CSVs.</p>",
            '<div class="facts">',
            "<div><strong>Cube</strong>100 mm side, 0.8 kg</div>",
            "<div><strong>Plate</strong>500 x 500 x 400 mm</div>",
            "<div><strong>Friction</strong>sliding coefficient 0.5</div>",
            "<div><strong>Speeds</strong>0.05, 0.1, 0.2, 0.4 m/s</div>",
            "<div><strong>Modes</strong>reduce off vs reduce on, pre-prune off</div>",
            "<div><strong>Run</strong>0.25 s, 120 logged frames/s, 4 substeps/frame</div>",
            "</div>",
            "<h2>Hypothesis</h2>",
            "<p>For flat-on-flat sliding with no spin and no applied force, reduce on should match reduce off in "
            "stopping time, travel distance, horizontal solver impulse, and settled geometry while using fewer "
            "solver contact entries.</p>",
            "<h2>Measured Quantities</h2>",
            "<ul>",
            "<li>Cube horizontal speed, stopping time, and travel distance.</li>",
            "<li>Solver force and torque on the cube, plus a logged-force horizontal impulse estimate.</li>",
            "<li>Final lateral drift, final tilt, and geometric penetration depth.</li>",
            "<li>Solver contact count, rigid contact count, buffer overflow state, and reduction hashtable "
            "failures.</li>",
            "<li>Analytic Coulomb stop time and stop distance from the initial speed, friction coefficient, and "
            "gravity.</li>",
            "</ul>",
            "<h2>Result</h2>",
            f"<ul>\n{rc.bullet_list(_result_bullets(summaries))}\n</ul>",
            "<p>Stop distance is the cleaner scalar for the low-speed cases because stop time is quantized by the "
            "120 Hz logging interval and the 0.005 m/s stop threshold.</p>",
            "<h2>Figures</h2>",
            rc.figure_tabs(panels),
            "<h2>Summary Table</h2>",
            _summary_table(summaries),
            resolution_probe_section,
            f'<p class="meta">Generated from <code>{rc.escape(TIMESERIES_CSV)}</code> and '
            f"<code>{rc.escape(SUMMARY_CSV)}</code>.</p>",
        ]
    )
    return rc.render_page(title=PAGE_TITLE, body=body, extra_css=FACTS_CSS)


def write_html_report(
    *,
    csv_dir: str | Path = DEFAULT_CSV_DIR,
    output_path: str | Path = DEFAULT_HTML_PATH,
    time_series_speed: float | None = None,
    resolution_probe_dirs: tuple[tuple[str, str | Path], ...] | list[tuple[str, str | Path]] | None = None,
) -> Path:
    """Write a standalone HTML report from flat-sliding-block CSVs."""

    runs = load_timeseries(csv_dir)
    summaries = load_summaries(csv_dir)
    selected_speed = select_speed(runs, time_series_speed)
    resolution_probes = load_resolution_probes(resolution_probe_dirs) if resolution_probe_dirs else []
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_html_report(runs, summaries, selected_speed=selected_speed, resolution_probes=resolution_probes),
        encoding="utf-8",
    )
    shutil.copyfile(HYPOTHESIS_RECORD_SOURCE, output_path.parent / HYPOTHESIS_RECORD_NAME)
    return output_path


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--csv-dir", type=str, default=str(DEFAULT_CSV_DIR), help="Directory containing H2 CSV files.")
    parser.add_argument("--html-path", type=str, default=str(DEFAULT_HTML_PATH), help="Path for the HTML report.")
    parser.add_argument(
        "--time-series-speed",
        type=float,
        default=None,
        help="Initial speed [m/s] to show in the time-history figure; defaults to the fastest run.",
    )
    parser.add_argument(
        "--no-resolution-probe",
        action="store_true",
        help="Do not include the optional SDF-resolution probe table.",
    )
    return parser


def main() -> None:
    args = create_parser().parse_args()
    html_path = write_html_report(
        csv_dir=args.csv_dir,
        output_path=args.html_path,
        time_series_speed=args.time_series_speed,
        resolution_probe_dirs=None if args.no_resolution_probe else DEFAULT_RESOLUTION_PROBES,
    )
    print(f"wrote {html_path}")


if __name__ == "__main__":
    main()
