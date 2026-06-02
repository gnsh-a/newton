# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared rendering framework for the hypothesis HTML reports.

The per-experiment report scripts (``tools/*_report.py``) read CSVs and render a
self-contained HTML page. Historically each script re-implemented the same CSV
helpers, axis math, SVG plotter, and page shell, which let the reports drift
apart visually. This module is the single source of truth for all of that.

The plotting model is a backend boundary: report scripts build :class:`Figure`
specifications (data, not markup) and call :func:`render_figure`. The current
backend emits inline SVG with no external dependencies, so reports stay tiny and
open offline. A different backend (e.g. Plotly) can be swapped in here without
touching the report scripts.

This module intentionally imports only the standard library.
"""

from __future__ import annotations

import csv
import html
import math
from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------- #
# Canonical reduce off/on palette, shared by every report.
# --------------------------------------------------------------------------- #

MODES: tuple[str, ...] = ("unreduced", "reduced")
MODE_LABELS: dict[str, str] = {"unreduced": "reduce off", "reduced": "reduce on"}
MODE_COLORS: dict[str, str] = {"unreduced": "#2563eb", "reduced": "#f97316"}
REFERENCE_COLOR = "#111827"

# --------------------------------------------------------------------------- #
# Plot model.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Series:
    """One plotted data series.

    Args:
        xs: X values.
        ys: Y values, paired with ``xs`` by index.
        label: Legend label.
        color: Stroke/marker color as a CSS color string.
        draw_line: Connect the points with a polyline.
        draw_marker: Draw a circular marker at each point.
        dash: SVG ``stroke-dasharray`` value, or ``None`` for a solid line.
    """

    xs: list[float]
    ys: list[float]
    label: str
    color: str
    draw_line: bool = True
    draw_marker: bool = False
    dash: str | None = None


@dataclass(frozen=True)
class Figure:
    """A single chart.

    Args:
        title: Chart title drawn above the plot area.
        xlabel: X-axis title.
        ylabel: Y-axis title.
        series: Data series, drawn in order.
        x_range: ``(low, high)`` data bounds for the x-axis.
        y_range: ``(low, high)`` data bounds for the y-axis.
        x_ticks: Explicit x tick positions, or ``None`` for evenly spaced ticks.
        log_y: Render the y-axis on a base-10 log scale.
        width: SVG viewbox width.
        height: SVG viewbox height.
        hlines: Dashed horizontal reference lines as ``(value, label, color)``,
            drawn with the label right-aligned at the line.
        xbands: Shaded vertical regions as ``(low, high, label, color)``.
        ybands: Shaded horizontal regions as ``(low, high, label, color)``.
    """

    title: str
    xlabel: str
    ylabel: str
    series: list[Series]
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    x_ticks: list[float] | None = None
    log_y: bool = False
    width: int = 760
    height: int = 300
    hlines: tuple[tuple[float, str, str], ...] = ()
    xbands: tuple[tuple[float, float, str, str], ...] = ()
    ybands: tuple[tuple[float, float, str, str], ...] = ()


# --------------------------------------------------------------------------- #
# CSV and value helpers.
# --------------------------------------------------------------------------- #


def read_csv(path: str | Path) -> list[dict[str, str]]:
    """Read a CSV file into a list of row dictionaries."""

    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(row: dict[str, str], key: str) -> float:
    """Return ``row[key]`` parsed as a float."""

    return float(row[key])


def as_bool(value: str) -> bool:
    """Parse a CSV truthy string (``"true"``/``"1"``/``"yes"``) into a bool."""

    return value.lower() in ("1", "true", "yes")


def finite(values: list[float]) -> list[float]:
    """Return only the finite entries of ``values`` as floats."""

    return [float(value) for value in values if math.isfinite(float(value))]


def mean(values: list[float]) -> float:
    """Return the arithmetic mean of the finite entries, or ``nan`` if none."""

    valid = finite(values)
    return sum(valid) / len(valid) if valid else float("nan")


# --------------------------------------------------------------------------- #
# Formatting helpers.
# --------------------------------------------------------------------------- #


def escape(value: object) -> str:
    """HTML-escape ``value`` (including quotes)."""

    return html.escape(str(value), quote=True)


def format_number(value: float, *, precision: int = 4) -> str:
    """Format a number compactly, switching to scientific notation at extremes."""

    if not math.isfinite(value):
        return "nan"
    if value == 0.0:
        return "0"
    abs_value = abs(value)
    if abs_value < 1.0e-3 or abs_value >= 1.0e4:
        return f"{value:.{precision}e}"
    return f"{value:.{precision}g}"


def format_percent(value: float, *, precision: int = 3) -> str:
    """Format a fraction as a percent string."""

    if not math.isfinite(value):
        return "nan"
    return f"{100.0 * value:.{precision}g}%"


# --------------------------------------------------------------------------- #
# Axis helpers.
# --------------------------------------------------------------------------- #


def padded_range(
    values: list[float],
    *,
    include: tuple[float, ...] = (),
    floor_span: float = 0.0,
) -> tuple[float, float]:
    """Compute a padded ``(low, high)`` range covering ``values`` and ``include``.

    Args:
        values: Data values to bound.
        include: Extra values the range must contain (e.g. a reference level).
        floor_span: Minimum span, used when the data is flat or nearly so.
    """

    valid = finite([*values, *include])
    if not valid:
        return 0.0, max(1.0, floor_span)
    low = min(valid)
    high = max(valid)
    if low == high:
        half_span = max(abs(low) * 0.05, floor_span * 0.5, 1.0e-9)
        return low - half_span, high + half_span
    span = max(high - low, floor_span)
    center = 0.5 * (low + high)
    half_span = 0.55 * span
    return center - half_span, center + half_span


def linear_ticks(low: float, high: float, count: int = 5) -> list[float]:
    """Return ``count`` evenly spaced tick positions in ``[low, high]``."""

    if count <= 1 or high <= low:
        return [low]
    return [low + (high - low) * i / (count - 1) for i in range(count)]


def log_ticks(low: float, high: float) -> list[float]:
    """Return base-10 decade tick positions spanning ``[low, high]``."""

    if high <= 0.0:
        return [1.0]
    if low <= 0.0:
        low = high / 1000.0
    lo_exp = math.floor(math.log10(low))
    hi_exp = math.ceil(math.log10(high))
    return [10.0**exp for exp in range(lo_exp, hi_exp + 1) if low <= 10.0**exp <= high]


# --------------------------------------------------------------------------- #
# Series construction.
# --------------------------------------------------------------------------- #


def mode_series(
    rows_by_mode: dict[str, list[dict[str, str]]],
    *,
    x_key: str,
    y_key: str,
    scale: float = 1.0,
    draw_line: bool = True,
    draw_marker: bool = False,
) -> list[Series]:
    """Build one :class:`Series` per mode using the canonical reduce off/on style.

    Args:
        rows_by_mode: Time-series rows grouped by mode.
        x_key: CSV column for the x values.
        y_key: CSV column for the y values.
        scale: Multiplier applied to every y value (e.g. ``1000`` for m to mm).
        draw_line: Connect points with a line.
        draw_marker: Draw a marker at each point.
    """

    series: list[Series] = []
    for mode in MODES:
        rows = rows_by_mode.get(mode)
        if not rows:
            continue
        series.append(
            Series(
                xs=[as_float(row, x_key) for row in rows],
                ys=[scale * as_float(row, y_key) for row in rows],
                label=MODE_LABELS[mode],
                color=MODE_COLORS[mode],
                draw_line=draw_line,
                draw_marker=draw_marker,
            )
        )
    return series


# --------------------------------------------------------------------------- #
# SVG rendering backend.
# --------------------------------------------------------------------------- #


def _svg_text(x: float, y: float, text: str, *, classes: str = "", anchor: str = "start") -> str:
    class_attr = f' class="{classes}"' if classes else ""
    return f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}"{class_attr}>{escape(text)}</text>'


def render_figure(fig: Figure) -> str:
    """Render a :class:`Figure` to a standalone inline ``<svg>`` element."""

    margin_left = 80
    margin_right = 28
    margin_top = 36
    margin_bottom = 56
    plot_width = fig.width - margin_left - margin_right
    plot_height = fig.height - margin_top - margin_bottom

    x_low, x_high = fig.x_range
    if x_high <= x_low:
        x_high = x_low + 1.0

    y_low, y_high = fig.y_range
    if fig.log_y:
        y_low = max(y_low, 1.0e-12)
        y_high = max(y_high, y_low * 10.0)
        y_low_t, y_high_t = math.log10(y_low), math.log10(y_high)
    else:
        y_low_t, y_high_t = y_low, y_high
    if y_high_t <= y_low_t:
        y_high_t = y_low_t + 1.0

    def sx(value: float) -> float:
        return margin_left + (value - x_low) / (x_high - x_low) * plot_width

    def sy(value: float) -> float:
        coord = math.log10(value) if fig.log_y else value
        return margin_top + plot_height - (coord - y_low_t) / (y_high_t - y_low_t) * plot_height

    def visible(x: float, y: float) -> bool:
        return math.isfinite(x) and math.isfinite(y) and (not fig.log_y or y > 0.0)

    parts = [
        f'<svg class="plot" viewBox="0 0 {fig.width} {fig.height}" role="img" aria-label="{escape(fig.title)}">',
        f"<title>{escape(fig.title)}</title>",
        f'<rect x="0" y="0" width="{fig.width}" height="{fig.height}" class="plot-shell"/>',
        f'<rect x="{margin_left}" y="{margin_top}" width="{plot_width}" height="{plot_height}" class="plot-area"/>',
    ]

    for band_low, band_high, _label, color in fig.ybands:
        clipped_low = max(min(band_low, band_high), y_low)
        clipped_high = min(max(band_low, band_high), y_high)
        if clipped_high > clipped_low:
            band_top = sy(clipped_high)
            band_bottom = sy(clipped_low)
            parts.append(
                f'<rect x="{margin_left:.2f}" y="{band_top:.2f}" width="{plot_width:.2f}" '
                f'height="{band_bottom - band_top:.2f}" fill="{color}" opacity="0.16"/>'
            )

    for band_low, band_high, _label, color in fig.xbands:
        clipped_low = max(min(band_low, band_high), x_low)
        clipped_high = min(max(band_low, band_high), x_high)
        if clipped_high > clipped_low:
            band_left = sx(clipped_low)
            band_right = sx(clipped_high)
            parts.append(
                f'<rect x="{band_left:.2f}" y="{margin_top:.2f}" width="{band_right - band_left:.2f}" '
                f'height="{plot_height:.2f}" fill="{color}" opacity="0.5"/>'
            )

    y_ticks = log_ticks(y_low, y_high) if fig.log_y else linear_ticks(y_low_t, y_high_t)
    for tick in y_ticks:
        if fig.log_y and tick <= 0.0:
            continue
        y = sy(tick)
        parts.append(
            f'<line x1="{margin_left:.2f}" y1="{y:.2f}" x2="{margin_left + plot_width:.2f}" '
            f'y2="{y:.2f}" class="grid-line"/>'
        )
        parts.append(
            _svg_text(margin_left - 10, y + 4, format_number(tick, precision=3), classes="axis-label", anchor="end")
        )

    x_ticks = fig.x_ticks if fig.x_ticks is not None else linear_ticks(x_low, x_high)
    for tick in x_ticks:
        x = sx(tick)
        parts.append(
            f'<line x1="{x:.2f}" y1="{margin_top + plot_height:.2f}" x2="{x:.2f}" '
            f'y2="{margin_top + plot_height + 5:.2f}" class="axis-line"/>'
        )
        parts.append(
            _svg_text(
                x,
                margin_top + plot_height + 22,
                format_number(tick, precision=3),
                classes="axis-label",
                anchor="middle",
            )
        )

    parts.append(
        f'<line x1="{margin_left:.2f}" y1="{margin_top:.2f}" x2="{margin_left:.2f}" '
        f'y2="{margin_top + plot_height:.2f}" class="axis-line"/>'
    )
    parts.append(
        f'<line x1="{margin_left:.2f}" y1="{margin_top + plot_height:.2f}" '
        f'x2="{margin_left + plot_width:.2f}" y2="{margin_top + plot_height:.2f}" class="axis-line"/>'
    )

    for value, _label, color in fig.hlines:
        if y_low <= value <= y_high:
            line_y = sy(value)
            parts.append(
                f'<line x1="{margin_left:.2f}" y1="{line_y:.2f}" x2="{margin_left + plot_width:.2f}" '
                f'y2="{line_y:.2f}" stroke="{color}" stroke-width="1.2" stroke-dasharray="5 5"/>'
            )

    for plot_series in fig.series:
        points = [(sx(x), sy(y)) for x, y in zip(plot_series.xs, plot_series.ys, strict=False) if visible(x, y)]
        if plot_series.draw_line and points:
            dash_attr = f' stroke-dasharray="{plot_series.dash}"' if plot_series.dash else ""
            point_attr = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
            parts.append(
                f'<polyline points="{point_attr}" fill="none" stroke="{plot_series.color}" '
                f'stroke-width="2.0" stroke-linecap="round" stroke-linejoin="round"{dash_attr}/>'
            )
        if plot_series.draw_marker:
            for x, y in points:
                parts.append(
                    f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.3" fill="{plot_series.color}" '
                    'stroke="#ffffff" stroke-width="1.1"/>'
                )

    parts.append(_svg_text(fig.width / 2, 23, fig.title, classes="plot-title", anchor="middle"))
    parts.append(_svg_text(fig.width / 2, fig.height - 13, fig.xlabel, classes="axis-title", anchor="middle"))
    parts.append(
        f'<text x="18" y="{margin_top + plot_height / 2:.2f}" '
        f'transform="rotate(-90 18 {margin_top + plot_height / 2:.2f})" '
        f'text-anchor="middle" class="axis-title">{escape(fig.ylabel)}</text>'
    )

    legend_x = margin_left + plot_width - 172
    legend_y = margin_top + 16
    for plot_series in fig.series:
        if plot_series.draw_line:
            dash_attr = f' stroke-dasharray="{plot_series.dash}"' if plot_series.dash else ""
            parts.append(
                f'<line x1="{legend_x:.2f}" y1="{legend_y:.2f}" x2="{legend_x + 20:.2f}" '
                f'y2="{legend_y:.2f}" stroke="{plot_series.color}" stroke-width="2.2"{dash_attr}/>'
            )
        if plot_series.draw_marker:
            parts.append(
                f'<circle cx="{legend_x + 10:.2f}" cy="{legend_y:.2f}" r="4.0" fill="{plot_series.color}" '
                'stroke="#ffffff" stroke-width="1.0"/>'
            )
        parts.append(_svg_text(legend_x + 26, legend_y + 4, plot_series.label, classes="legend-label"))
        legend_y += 18

    for value, label, color in fig.hlines:
        if not (y_low <= value <= y_high):
            continue
        parts.append(
            f'<line x1="{legend_x:.2f}" y1="{legend_y:.2f}" x2="{legend_x + 20:.2f}" '
            f'y2="{legend_y:.2f}" stroke="{color}" stroke-width="1.6" stroke-dasharray="5 5"/>'
        )
        parts.append(_svg_text(legend_x + 26, legend_y + 4, label, classes="legend-label"))
        legend_y += 18

    parts.append("</svg>")
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# HTML building blocks.
# --------------------------------------------------------------------------- #


def figure_grid(figures: list[Figure], *, columns: int = 1) -> str:
    """Render figures into a responsive grid.

    Args:
        figures: Figures to render in order.
        columns: Number of columns (1-3); collapses to one column on narrow
            screens.
    """

    cols_class = f" cols-{columns}" if columns > 1 else ""
    body = "\n".join(render_figure(fig) for fig in figures)
    return f'<div class="plot-grid{cols_class}">\n{body}\n</div>'


def bullet_list(items: list[str]) -> str:
    """Render ``items`` as escaped ``<li>`` rows (caller supplies the ``<ul>``)."""

    return "\n".join(f"<li>{escape(item)}</li>" for item in items)


def data_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a styled table from headers and pre-formatted string cells.

    Cell and header text is escaped, so callers pass already-formatted values
    (e.g. ``format_number(x)``), not raw HTML. The canonical theme right-aligns
    numeric columns and left-aligns the first column.

    Args:
        headers: Column header labels.
        rows: One list of cell strings per row, aligned with ``headers``.
    """

    head = "".join(f"<th>{escape(h)}</th>" for h in headers)
    body = [f"<tr>{''.join(f'<td>{escape(cell)}</td>' for cell in row)}</tr>" for row in rows]
    return f"<table>\n<thead><tr>{head}</tr></thead>\n<tbody>\n" + "\n".join(body) + "\n</tbody>\n</table>"


@dataclass(frozen=True)
class TabPanel:
    """One tab in a :func:`figure_tabs` group.

    Args:
        label: Tab button text.
        content: Raw inner HTML for the panel (already-rendered figures/prose).
    """

    label: str
    content: str


def figure_tabs(panels: list[TabPanel], *, group_id: str = "figs", aria_label: str = "Figures") -> str:
    """Render a tabbed figure switcher with self-contained, scoped JavaScript.

    The first panel is active by default. The script queries only within this
    group's wrapper, so multiple tab groups can coexist on one page.

    Args:
        panels: Tabs in display order.
        group_id: Unique DOM id for this tab group.
        aria_label: Accessible label for the tablist.
    """

    tabs = []
    sections = []
    for index, panel in enumerate(panels):
        active = index == 0
        tab_id = f"{group_id}-t{index + 1}"
        panel_id = f"{group_id}-p{index + 1}"
        tabs.append(
            f'<button class="tab-button" role="tab" type="button" '
            f'aria-selected="{"true" if active else "false"}" '
            f'aria-controls="{panel_id}" id="{tab_id}">{escape(panel.label)}</button>'
        )
        panel_class = "figure-panel active" if active else "figure-panel"
        sections.append(
            f'<section id="{panel_id}" class="{panel_class}" role="tabpanel" aria-labelledby="{tab_id}">\n'
            f"{panel.content}\n</section>"
        )

    script = (
        "<script>\n"
        "(() => {\n"
        f"  const group = document.getElementById({group_id!r});\n"
        "  const tabs = Array.from(group.querySelectorAll('.tab-button'));\n"
        "  const panels = Array.from(group.querySelectorAll('.figure-panel'));\n"
        "  for (const tab of tabs) {\n"
        "    tab.addEventListener('click', () => {\n"
        "      for (const item of tabs) item.setAttribute('aria-selected', String(item === tab));\n"
        "      for (const panel of panels)\n"
        "        panel.classList.toggle('active', panel.id === tab.getAttribute('aria-controls'));\n"
        "    });\n"
        "  }\n"
        "})();\n"
        "</script>"
    )

    return "\n".join(
        [
            f'<div class="figure-tabs-group" id="{group_id}">',
            f'<div class="tabs" role="tablist" aria-label="{escape(aria_label)}">',
            *tabs,
            "</div>",
            *sections,
            "</div>",
            script,
        ]
    )


# --------------------------------------------------------------------------- #
# Page shell and canonical theme.
# --------------------------------------------------------------------------- #

THEME_CSS = """\
    :root {
      color-scheme: light;
      --ink: #172033;
      --muted: #5f6b7a;
      --line: #d8dee8;
      --soft: #f6f8fb;
      --accent: #0f766e;
    }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: #ffffff;
    }
    main {
      max-width: 1180px;
      margin: 0 auto;
      padding: 34px 24px 48px;
    }
    h1 {
      margin: 0 0 10px;
      font-size: clamp(26px, 3.2vw, 42px);
      line-height: 1.08;
      letter-spacing: 0;
    }
    h2 {
      margin: 30px 0 10px;
      font-size: 19px;
      letter-spacing: 0;
    }
    p {
      max-width: 920px;
      margin: 0 0 12px;
      color: var(--muted);
      line-height: 1.55;
    }
    ul {
      margin: 8px 0 0 20px;
      padding: 0;
      color: var(--muted);
      line-height: 1.55;
    }
    .meta {
      margin-top: 14px;
      font-size: 13px;
      color: var(--muted);
    }
    table {
      border-collapse: collapse;
      margin: 8px 0 4px;
      font-size: 13px;
    }
    th, td {
      border: 1px solid var(--line);
      padding: 6px 10px;
      text-align: right;
    }
    th:first-child, td:first-child {
      text-align: left;
    }
    thead th {
      background: var(--soft);
    }
    .tabs {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 18px 0 16px;
      border-bottom: 1px solid var(--line);
      padding-bottom: 10px;
    }
    .tab-button {
      border: 1px solid var(--line);
      background: #ffffff;
      color: var(--ink);
      border-radius: 6px;
      padding: 8px 12px;
      font: inherit;
      cursor: pointer;
    }
    .tab-button[aria-selected="true"] {
      border-color: var(--accent);
      background: #ecfdf5;
      color: #064e3b;
    }
    .figure-panel {
      display: none;
    }
    .figure-panel.active {
      display: block;
    }
    .plot-grid {
      display: grid;
      gap: 16px;
      align-items: start;
      grid-template-columns: 1fr;
    }
    .plot-grid.cols-2 {
      grid-template-columns: repeat(2, 1fr);
    }
    .plot-grid.cols-3 {
      grid-template-columns: repeat(3, 1fr);
    }
    @media (max-width: 900px) {
      .plot-grid.cols-2,
      .plot-grid.cols-3 {
        grid-template-columns: 1fr;
      }
    }
    .plot {
      width: 100%;
      height: auto;
      display: block;
      overflow: visible;
    }
    .plot-shell {
      fill: #ffffff;
      stroke: var(--line);
      rx: 6;
    }
    .plot-area {
      fill: var(--soft);
    }
    .grid-line {
      stroke: #e7ebf2;
      stroke-width: 1;
    }
    .axis-line {
      stroke: #7b8794;
      stroke-width: 1;
    }
    .axis-label, .legend-label, .reference-label {
      fill: var(--muted);
      font-size: 12px;
    }
    .axis-title {
      fill: var(--ink);
      font-size: 13px;
      font-weight: 600;
    }
    .plot-title {
      fill: var(--ink);
      font-size: 14px;
      font-weight: 700;
    }
    code {
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 0.95em;
      color: #334155;
    }"""


def render_page(*, title: str, body: str, lang: str = "en", css: str | None = None, extra_css: str = "") -> str:
    """Wrap ``body`` in the canonical HTML document shell.

    Args:
        title: Page ``<title>`` and document language are escaped automatically.
        body: Inner HTML placed inside ``<main>``.
        lang: Document language attribute.
        css: Stylesheet to inline; defaults to :data:`THEME_CSS`.
        extra_css: Report-specific rules appended after the base stylesheet, for
            components the canonical theme does not cover (e.g. a setup grid).
    """

    stylesheet = THEME_CSS if css is None else css
    if extra_css:
        stylesheet = f"{stylesheet}\n{extra_css}"
    return (
        "<!doctype html>\n"
        f'<html lang="{escape(lang)}">\n'
        "<head>\n"
        '  <meta charset="utf-8">\n'
        '  <meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"  <title>{escape(title)}</title>\n"
        "  <style>\n"
        f"{stylesheet}\n"
        "  </style>\n"
        "</head>\n"
        "<body>\n"
        "<main>\n"
        f"{body}\n"
        "</main>\n"
        "</body>\n"
        "</html>\n"
    )
