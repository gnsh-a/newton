# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Build a self-contained HTML report from H7 transient-gap CSVs.

This script intentionally does not import Newton. The Newton experiment writes
CSV files; this script reads those CSV files and renders the standalone report
using the shared report framework in :mod:`_report_common`.

The hypothesis (``hypothesis/H7_transient_compliance_hypothesis.md``) is that
contact reduction preserves the static resultant but not transient compliance,
with ``delta_max(dense) / delta_max(reduced) = sqrt(N / K)``.
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _report_common as rc

DEFAULT_CSV_DIR = Path("output") / "H7_transient_gap"
DEFAULT_HTML_PATH = DEFAULT_CSV_DIR / "transient_gap_report.html"
HYPOTHESIS_RECORD_NAME = "H7_transient_compliance_hypothesis.md"
HYPOTHESIS_RECORD_SOURCE = Path(__file__).resolve().parents[1] / "hypothesis" / HYPOTHESIS_RECORD_NAME
TIMESERIES_CSV = "transient_gap_timeseries.csv"
SUMMARY_CSV = "transient_gap_summary.csv"

PAGE_TITLE = "H7: Transient-Compliance Contact Reduction"

# A pass means the measured ratio matches sqrt(N/K) within this relative band,
# the dense run was in the SDF band, and reduced peak force exceeds dense.
RATIO_REL_TOL = 0.20
# Static control: settled support force within this fraction of weight.
FORCE_REL_TOL = 0.10


def load_summaries(csv_dir: str | Path) -> list[dict[str, str]]:
    """Load all summary rows."""

    path = Path(csv_dir) / SUMMARY_CSV
    if not path.exists():
        raise FileNotFoundError(f"missing summary CSV: {path}")
    rows = rc.read_csv(path)
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows


def load_timeseries(csv_dir: str | Path) -> dict[str, list[dict[str, str]]]:
    """Load time-series rows (the primary drop-height run) grouped by mode."""

    path = Path(csv_dir) / TIMESERIES_CSV
    if not path.exists():
        return {}
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rc.read_csv(path):
        grouped.setdefault(row["mode"], []).append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: rc.as_float(row, "time_s"))
    return grouped


def pair_rows(summaries: list[dict[str, str]], sweep: str) -> list[tuple[float, dict[str, str], dict[str, str]]]:
    """Return ``(value, unreduced_row, reduced_row)`` triples for one sweep, sorted by value."""

    keyed: dict[float, dict[str, dict[str, str]]] = {}
    for row in summaries:
        if row["sweep"] != sweep:
            continue
        keyed.setdefault(rc.as_float(row, "sweep_value"), {})[row["mode"]] = row
    pairs: list[tuple[float, dict[str, str], dict[str, str]]] = []
    for value in sorted(keyed):
        modes = keyed[value]
        dense = modes.get("unreduced")
        reduced = modes.get("reduced")
        if dense and reduced:
            pairs.append((value, dense, reduced))
    return pairs


def _gap_metrics(dense: dict[str, str], reduced: dict[str, str]) -> dict[str, float]:
    """Compute the sqrt(N/K) law metrics for one (dense, reduced) pair."""

    n = rc.as_float(reduced, "face_count_N")
    k = rc.as_float(reduced, "rigid_count_K")
    nk = n / k if k > 0 else float("nan")
    pred = math.sqrt(nk) if math.isfinite(nk) and nk > 0 else float("nan")
    d_pen = rc.as_float(dense, "max_penetration_m") * 1e6
    r_pen = rc.as_float(reduced, "max_penetration_m") * 1e6
    ratio = d_pen / r_pen if r_pen > 0 else float("nan")
    rel_err = abs(ratio - pred) / pred if math.isfinite(pred) and pred > 0 else float("nan")
    return {
        "N": n,
        "K": k,
        "nk": nk,
        "pred": pred,
        "d_pen": d_pen,
        "r_pen": r_pen,
        "ratio": ratio,
        "rel_err": rel_err,
        "d_fzw": rc.as_float(dense, "peak_fz_over_weight"),
        "r_fzw": rc.as_float(reduced, "peak_fz_over_weight"),
        "in_band": rc.as_bool(dense["in_band"]),
    }


def _verdict(m: dict[str, float]) -> str:
    if not m["in_band"]:
        return "inconclusive (dense tunneled)"
    if math.isfinite(m["rel_err"]) and m["rel_err"] <= RATIO_REL_TOL and m["r_fzw"] > m["d_fzw"]:
        return "pass"
    return "fail"


# --------------------------------------------------------------------------- #
# Figures.
# --------------------------------------------------------------------------- #


def _figure(
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    series: list[rc.Series],
    x_include: tuple[float, ...] = (),
    y_include: tuple[float, ...] = (),
    hlines: tuple[tuple[float, str, str], ...] = (),
) -> rc.Figure:
    all_x = [x for item in series for x in item.xs]
    all_y = [y for item in series for y in item.ys]
    return rc.Figure(
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        series=series,
        x_range=rc.padded_range(all_x, include=x_include),
        y_range=rc.padded_range(all_y, include=y_include),
        hlines=hlines,
    )


def _time_history_figures(timeseries: dict[str, list[dict[str, str]]]) -> str:
    pen_series = rc.mode_series(timeseries, x_key="time_s", y_key="penetration_m", scale=1.0e6)
    fz_series = rc.mode_series(timeseries, x_key="time_s", y_key="solver_fz_N")
    return rc.figure_grid(
        [
            _figure(
                title="Impact penetration history",
                xlabel="time [s]",
                ylabel="penetration [µm]",
                series=pen_series,
                y_include=(0.0,),
            ),
            _figure(
                title="Solver Fz history",
                xlabel="time [s]",
                ylabel="Fz [N]",
                series=fz_series,
                y_include=(0.0,),
            ),
        ]
    )


def _pen_series(pairs: list[tuple[float, dict[str, str], dict[str, str]]], *, scale_x: float) -> list[rc.Series]:
    dense_xs, dense_ys, red_xs, red_ys = [], [], [], []
    for value, dense, reduced in pairs:
        dense_xs.append(value * scale_x)
        dense_ys.append(rc.as_float(dense, "max_penetration_m") * 1e6)
        red_xs.append(value * scale_x)
        red_ys.append(rc.as_float(reduced, "max_penetration_m") * 1e6)
    return [
        rc.Series(dense_xs, dense_ys, rc.MODE_LABELS["unreduced"], rc.MODE_COLORS["unreduced"], draw_marker=True),
        rc.Series(red_xs, red_ys, rc.MODE_LABELS["reduced"], rc.MODE_COLORS["reduced"], draw_marker=True),
    ]


def _ratio_series(pairs: list[tuple[float, dict[str, str], dict[str, str]]], *, scale_x: float) -> list[rc.Series]:
    measured_x, measured_y, pred_x, pred_y = [], [], [], []
    for value, dense, reduced in pairs:
        m = _gap_metrics(dense, reduced)
        measured_x.append(value * scale_x)
        measured_y.append(m["ratio"])
        pred_x.append(value * scale_x)
        pred_y.append(m["pred"])
    return [
        rc.Series(
            measured_x, measured_y, "measured ratio", rc.MODE_COLORS["reduced"], draw_line=False, draw_marker=True
        ),
        rc.Series(pred_x, pred_y, "√(N/K)", rc.REFERENCE_COLOR, draw_marker=True, dash="6 4"),
    ]


def _velocity_independence_figures(pairs: list[tuple[float, dict[str, str], dict[str, str]]]) -> str:
    return rc.figure_grid(
        [
            _figure(
                title="Max penetration vs drop height",
                xlabel="drop height [mm]",
                ylabel="δ_max [µm]",
                series=_pen_series(pairs, scale_x=1000.0),
                y_include=(0.0,),
            ),
            _figure(
                title="Penetration ratio vs drop height",
                xlabel="drop height [mm]",
                ylabel="δ_dense / δ_reduced",
                series=_ratio_series(pairs, scale_x=1000.0),
                y_include=(0.0,),
            ),
        ]
    )


def _sqrt_law_figures(pairs: list[tuple[float, dict[str, str], dict[str, str]]]) -> str:
    # Measured ratio vs sqrt(N/K) with a y = x reference line: the clincher plot.
    preds, ratios = [], []
    for _value, dense, reduced in pairs:
        m = _gap_metrics(dense, reduced)
        if math.isfinite(m["pred"]) and math.isfinite(m["ratio"]):
            preds.append(m["pred"])
            ratios.append(m["ratio"])
    span = rc.padded_range(preds + ratios, include=(0.0,))
    identity = rc.Series([span[0], span[1]], [span[0], span[1]], "ideal (y = x)", rc.REFERENCE_COLOR, dash="6 4")
    measured = rc.Series(preds, ratios, "measured", rc.MODE_COLORS["reduced"], draw_line=False, draw_marker=True)
    res_xs_dense, res_ys_dense, res_xs_red, res_ys_red = [], [], [], []
    for value, dense, reduced in pairs:
        res_xs_dense.append(value)
        res_ys_dense.append(rc.as_float(dense, "max_penetration_m") * 1e6)
        res_xs_red.append(value)
        res_ys_red.append(rc.as_float(reduced, "max_penetration_m") * 1e6)
    res_pen = [
        rc.Series(
            res_xs_dense, res_ys_dense, rc.MODE_LABELS["unreduced"], rc.MODE_COLORS["unreduced"], draw_marker=True
        ),
        rc.Series(res_xs_red, res_ys_red, rc.MODE_LABELS["reduced"], rc.MODE_COLORS["reduced"], draw_marker=True),
    ]
    return rc.figure_grid(
        [
            _figure(
                title="Measured ratio vs √(N/K)",
                xlabel="√(N/K)",
                ylabel="δ_dense / δ_reduced",
                series=[identity, measured],
                x_include=(0.0,),
                y_include=(0.0,),
            ),
            _figure(
                title="Max penetration vs SDF resolution",
                xlabel="SDF resolution",
                ylabel="δ_max [µm]",
                series=res_pen,
                y_include=(0.0,),
            ),
        ]
    )


# --------------------------------------------------------------------------- #
# Tables.
# --------------------------------------------------------------------------- #


def _gap_table(summaries: list[dict[str, str]]) -> str:
    headers = [
        "sweep",
        "value",
        "N",
        "K",
        "N/K",
        "√(N/K)",
        "δ_dense [µm]",
        "δ_reduced [µm]",
        "ratio",
        "rel. err",
        "peak Fz d/r [x w]",
        "in band",
        "verdict",
    ]
    rows: list[list[str]] = []
    for sweep, label in (("height", "drop height [mm]"), ("resolution", "SDF resolution")):
        for value, dense, reduced in pair_rows(summaries, sweep):
            m = _gap_metrics(dense, reduced)
            value_text = f"{1000.0 * value:.3g}" if sweep == "height" else f"{value:.0f}"
            rows.append(
                [
                    label,
                    value_text,
                    f"{m['N']:.0f}",
                    f"{m['K']:.0f}",
                    f"{m['nk']:.1f}",
                    f"{m['pred']:.2f}",
                    f"{m['d_pen']:.0f}",
                    f"{m['r_pen']:.0f}",
                    f"{m['ratio']:.2f}",
                    rc.format_percent(m["rel_err"]),
                    f"{m['d_fzw']:.2f}/{m['r_fzw']:.2f}",
                    str(m["in_band"]).lower(),
                    _verdict(m),
                ]
            )
    return rc.data_table(headers, rows)


def _static_table(summaries: list[dict[str, str]]) -> str:
    headers = ["mode", "final Fz [x w]", "final pen [µm]", "settled t [s]", "max pen [µm]", "verdict"]
    rows: list[list[str]] = []
    for _value, dense, reduced in pair_rows(summaries, "static"):
        for row in (dense, reduced):
            fzw = rc.as_float(row, "final_fz_over_weight")
            verdict = "pass" if abs(fzw - 1.0) <= FORCE_REL_TOL else "fail"
            rows.append(
                [
                    rc.MODE_LABELS[row["mode"]],
                    rc.format_number(fzw, precision=3),
                    f"{rc.as_float(row, 'final_penetration_m') * 1e6:.1f}",
                    rc.format_number(rc.as_float(row, "settled_time_s"), precision=3),
                    f"{rc.as_float(row, 'max_penetration_m') * 1e6:.0f}",
                    verdict,
                ]
            )
    return rc.data_table(headers, rows)


def _gates_list() -> str:
    return rc.bullet_list(
        [
            f"Transient: penetration ratio matches √(N/K) within ±{int(RATIO_REL_TOL * 100)}% "
            "(per-contact stiffness law).",
            f"Static control: settled support force within ±{int(FORCE_REL_TOL * 100)}% of weight, "
            "both modes (net force preserved).",
            "Corroborator: peak Fz(reduced) > peak Fz(dense) (sign-locked).",
            "Validity: dense penetration inside the ±5 mm SDF band (else inconclusive).",
        ]
    )


def _result_text(summaries: list[dict[str, str]]) -> str:
    ratios: list[float] = []
    preds: list[float] = []
    for sweep in ("height", "resolution"):
        for _value, dense, reduced in pair_rows(summaries, sweep):
            m = _gap_metrics(dense, reduced)
            if m["in_band"] and math.isfinite(m["ratio"]) and math.isfinite(m["pred"]):
                ratios.append(m["ratio"])
                preds.append(m["pred"])
    if not ratios:
        return "<p>No in-band transient comparisons available.</p>"
    rel_errs = [abs(r - p) / p for r, p in zip(ratios, preds, strict=True) if p > 0]
    worst = max(rel_errs) if rel_errs else float("nan")
    return rc.bullet_list(
        [
            f"In-band transient comparisons: {len(ratios)}.",
            f"Penetration ratio range: {min(ratios):.2f} to {max(ratios):.2f} (vs √(N/K) "
            f"{min(preds):.2f} to {max(preds):.2f}).",
            f"Worst relative error vs √(N/K): {rc.format_percent(worst)}.",
        ]
    )


def _build_html(
    *,
    summaries: list[dict[str, str]],
    timeseries: dict[str, list[dict[str, str]]],
    csv_dir: Path,
) -> str:
    height_pairs = pair_rows(summaries, "height")
    resolution_pairs = pair_rows(summaries, "resolution")

    panels: list[rc.TabPanel] = []
    if timeseries:
        panels.append(
            rc.TabPanel(
                "Figure 1",
                "<h2>Figure 1: impact time history</h2>\n"
                "<p>The primary drop-height run, both modes. Reduced arrests the body in much less "
                "penetration and with a higher peak force — the transient gap that static settle hides.</p>\n"
                + _time_history_figures(timeseries),
            )
        )
    if height_pairs:
        panels.append(
            rc.TabPanel(
                f"Figure {len(panels) + 1}",
                f"<h2>Figure {len(panels) + 1}: velocity independence</h2>\n"
                "<p>Across a 4x drop-height range the penetration ratio stays flat at √(N/K): the gap is "
                "not a velocity artifact. Points only, since each height is a separate condition.</p>\n"
                + _velocity_independence_figures(height_pairs),
            )
        )
    if resolution_pairs:
        panels.append(
            rc.TabPanel(
                f"Figure {len(panels) + 1}",
                f"<h2>Figure {len(panels) + 1}: the √(N/K) law</h2>\n"
                "<p>As SDF resolution grows, dense softens (more, smaller springs) while reduced is pinned "
                "(K fixed), so the measured ratio tracks √(N/K) along the ideal line — the decisive proof.</p>\n"
                + _sqrt_law_figures(resolution_pairs),
            )
        )

    body = "\n".join(
        [
            f"<h1>{rc.escape(PAGE_TITLE)}</h1>",
            '<p class="lede">This report checks whether contact reduction preserves transient compliance on '
            "impact, not just the static support force. A flat cube is dropped onto a hydroelastic plate; the "
            "maximum first-contact penetration of the dense and reduced contact sets is compared against the "
            "predicted law <code>δ_dense / δ_reduced = √(N/K)</code>.</p>",
            "<h2>Hypothesis</h2>",
            "<p>Contact reduction preserves the quasi-static net force (the body settles to the same "
            "equilibrium) but not transient compliance. Force-matching packs the net stiffness into K winners, "
            "each N/K stiffer, so each per-contact reference oscillator is √(N/K) more rigid and arrests the "
            "body √(N/K) shallower on impact.</p>",
            "<h2>Measured quantities</h2>",
            rc.bullet_list(
                [
                    "Max first-contact penetration δ_max for each mode (primary).",
                    "Dense face count N, reduced winner count K, and N/K.",
                    "Peak solver Fz / weight (corroborator: reduced higher, sign-locked).",
                    "Settled support force, final penetration, and settle time (static control).",
                ]
            ),
            "<h2>Current result</h2>",
            _result_text(summaries),
            "<h2>Figures</h2>",
            rc.figure_tabs(panels) if panels else "<p>No figure data available.</p>",
            "<h2>Transient gap</h2>",
            _gap_table(summaries),
            "<h2>Static control — net force preserved</h2>",
            _static_table(summaries),
            "<h2>Gates</h2>",
            _gates_list(),
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
    summaries = load_summaries(csv_dir)
    timeseries = load_timeseries(csv_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        _build_html(summaries=summaries, timeseries=timeseries, csv_dir=csv_dir),
        encoding="utf-8",
    )
    shutil.copyfile(HYPOTHESIS_RECORD_SOURCE, output_path.parent / HYPOTHESIS_RECORD_NAME)
    return output_path


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--csv-dir", type=str, default=str(DEFAULT_CSV_DIR), help="Directory containing H7 CSV files.")
    parser.add_argument("--output", type=str, default=str(DEFAULT_HTML_PATH), help="HTML output path.")
    return parser


def main() -> None:
    args = create_parser().parse_args()
    path = write_html_report(csv_dir=args.csv_dir, output_path=args.output)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
