# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared rendering framework for the hypothesis HTML reports.

The per-experiment report scripts (``tools/*_report.py``) read CSVs and render a
self-contained HTML page. Historically each script re-implemented the same CSV
helpers, axis math, plotting code, and page shell, which let the reports drift
apart visually. This module is the single source of truth for all of that.

The plotting model is a backend boundary: report scripts build :class:`Figure`
specifications (data, not markup) and call :func:`render_figure`. The backend
emits interactive `Plotly <https://plotly.com/javascript/>`_ charts; the page
shell loads ``plotly.js`` from its CDN, so a report needs a network connection
the first time it is opened. Swapping the backend never touches a report script.

This module intentionally imports only the standard library.
"""

from __future__ import annotations

import csv
import html
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path

# Pinned Plotly.js build loaded from the CDN by every rendered page.
PLOTLY_CDN_URL = "https://cdn.plot.ly/plotly-2.35.2.min.js"

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
        dash: Render the line dashed when set to any truthy value, else solid.
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
        x_ticks: Explicit x tick positions, or ``None`` for Plotly's auto ticks.
        log_y: Render the y-axis on a base-10 log scale.
        width: Retained for API compatibility; Plotly charts size responsively.
        height: Chart height in pixels.
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
        floor_span: Fallback span used only when the data (with ``include``) is
            perfectly flat. It is not a minimum for real data, so a small signal
            keeps its own scale and is not flattened by an arbitrary floor.
    """

    valid = finite([*values, *include])
    if not valid:
        return 0.0, max(1.0, floor_span)
    low = min(valid)
    high = max(valid)
    if low == high:
        half_span = max(abs(low) * 0.05, floor_span * 0.5, 1.0e-9)
        return low - half_span, high + half_span
    span = high - low
    center = 0.5 * (low + high)
    half_span = 0.55 * span
    return center - half_span, center + half_span


def zero_range(values: list[float], *, refs: tuple[float, ...] = ()) -> tuple[float, float]:
    """Adaptive y-range anchored at zero, for should-be-zero / non-negative metrics.

    The range always contains zero, adapts its top to the data (and any ``refs``,
    e.g. a reference-band ceiling), and adds a little head- and foot-room so a
    baseline at zero stays visible. Unlike :func:`padded_range` with a
    ``floor_span``, it imposes no fixed minimum span, so a small signal is not
    flattened by an arbitrary floor; a near-zero signal instead takes its scale
    from ``refs`` (the practical-floor band) and the framework's reference
    inclusion.

    Args:
        values: Data values.
        refs: Extra values the range must contain (e.g. a practical-floor band
            ceiling) so the reference stays visible when the data is near zero.
    """

    vals = finite([*values, *refs, 0.0])
    low, high = min(vals), max(vals)
    span = high - low
    if span <= 0.0:
        span = max(abs(high), 1.0e-9)
        high = low + span
    return (low - 0.06 * span, high + 0.15 * span)


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
# Plotly rendering backend.
# --------------------------------------------------------------------------- #

# Monotonic id source so every chart on a page gets a unique container.
_PLOT_IDS = itertools.count(1)

_BAND_LABEL_COLOR = "#5f6b7a"


def _clean(values: list[float], *, positive_only: bool = False) -> list[float | None]:
    """Map non-finite (and, for log axes, non-positive) values to ``None`` gaps."""

    cleaned: list[float | None] = []
    for value in values:
        number = float(value)
        if not math.isfinite(number) or (positive_only and number <= 0.0):
            cleaned.append(None)
        else:
            cleaned.append(number)
    return cleaned


def _plotly_traces(fig: Figure) -> list[dict]:
    """Build the Plotly trace list for ``fig``."""

    traces: list[dict] = []
    for plot_series in fig.series:
        modes = []
        if plot_series.draw_line:
            modes.append("lines")
        if plot_series.draw_marker:
            modes.append("markers")
        traces.append(
            {
                "type": "scatter",
                "x": _clean(plot_series.xs),
                "y": _clean(plot_series.ys, positive_only=fig.log_y),
                "mode": "+".join(modes) if modes else "markers",
                "name": plot_series.label,
                "line": {
                    "color": plot_series.color,
                    "width": 2.2,
                    "dash": "dash" if plot_series.dash else "solid",
                },
                "marker": {"color": plot_series.color, "size": 7, "line": {"color": "#ffffff", "width": 1}},
                "connectgaps": False,
            }
        )
    return traces


def _plotly_layout(fig: Figure) -> dict:
    """Build the Plotly layout (axes, reference shapes, bands) for ``fig``."""

    x_low, x_high = fig.x_range
    if x_high <= x_low:
        x_high = x_low + 1.0

    xaxis: dict = {
        "title": {"text": fig.xlabel},
        "range": [x_low, x_high],
        "zeroline": False,
        "gridcolor": "#e7ebf2",
        "linecolor": "#7b8794",
        "ticks": "outside",
        "tickcolor": "#7b8794",
    }
    if fig.x_ticks is not None:
        xaxis["tickmode"] = "array"
        xaxis["tickvals"] = list(fig.x_ticks)

    yaxis: dict = {
        "title": {"text": fig.ylabel},
        "zeroline": False,
        "gridcolor": "#e7ebf2",
        "linecolor": "#7b8794",
        "ticks": "outside",
        "tickcolor": "#7b8794",
    }
    y_low, y_high = fig.y_range
    # Keep reference *lines* in view by growing the range to fit them, so an
    # adaptive data-driven range never clips an hline the report drew. Shaded
    # bands are background tolerance zones, not values to compare against, so
    # they are allowed to clip: the data drives the scale and the band simply
    # fills whatever is visible (data entirely inside the band still reads as
    # "all within tolerance").
    ref_values = [v for v, _label, _color in fig.hlines]
    ref_values = [r for r in finite(ref_values) if not fig.log_y or r > 0.0]
    if ref_values:
        y_low = min(y_low, *ref_values)
        y_high = max(y_high, *ref_values)

    if fig.log_y:
        low = max(y_low, 1.0e-12)
        high = max(y_high, low * 10.0)
        yaxis["type"] = "log"
        yaxis["range"] = [math.log10(low), math.log10(high)]
    else:
        if y_high <= y_low:
            y_high = y_low + 1.0
        yaxis["range"] = [y_low, y_high]

    shapes: list[dict] = []
    annotations: list[dict] = []

    for band_low, band_high, label, color in fig.ybands:
        shapes.append(
            {
                "type": "rect",
                "xref": "paper",
                "x0": 0,
                "x1": 1,
                "yref": "y",
                "y0": band_low,
                "y1": band_high,
                "fillcolor": color,
                "opacity": 0.16,
                "line": {"width": 0},
                "layer": "below",
            }
        )
        if label:
            annotations.append(
                {
                    "xref": "paper",
                    "x": 0.01,
                    "yref": "y",
                    "y": 0.5 * (band_low + band_high),
                    "text": label,
                    "showarrow": False,
                    "xanchor": "left",
                    "font": {"size": 11, "color": _BAND_LABEL_COLOR},
                }
            )

    for band_low, band_high, label, color in fig.xbands:
        shapes.append(
            {
                "type": "rect",
                "yref": "paper",
                "y0": 0,
                "y1": 1,
                "xref": "x",
                "x0": band_low,
                "x1": band_high,
                "fillcolor": color,
                "opacity": 0.5,
                "line": {"width": 0},
                "layer": "below",
            }
        )
        if label:
            annotations.append(
                {
                    "yref": "paper",
                    "y": 0.98,
                    "xref": "x",
                    "x": 0.5 * (band_low + band_high),
                    "text": label,
                    "showarrow": False,
                    "yanchor": "top",
                    "font": {"size": 11, "color": _BAND_LABEL_COLOR},
                }
            )

    for value, label, color in fig.hlines:
        shapes.append(
            {
                "type": "line",
                "xref": "paper",
                "x0": 0,
                "x1": 1,
                "yref": "y",
                "y0": value,
                "y1": value,
                "line": {"color": color, "width": 1.3, "dash": "dash"},
            }
        )
        if label:
            annotations.append(
                {
                    "xref": "paper",
                    "x": 0.99,
                    "yref": "y",
                    "y": value,
                    "text": label,
                    "showarrow": False,
                    "xanchor": "right",
                    "yanchor": "bottom",
                    "font": {"size": 11, "color": color},
                }
            )

    return {
        "title": {"text": fig.title, "font": {"size": 15}, "x": 0.5, "xanchor": "center"},
        "xaxis": xaxis,
        "yaxis": yaxis,
        "shapes": shapes,
        "annotations": annotations,
        "legend": {
            "orientation": "v",
            "x": 1,
            "xanchor": "right",
            "y": 1,
            "yanchor": "top",
            "bgcolor": "rgba(255,255,255,0.72)",
            "bordercolor": "#d8dee8",
            "borderwidth": 1,
        },
        "margin": {"l": 66, "r": 24, "t": 48, "b": 54},
        "font": {
            "family": 'Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif',
            "color": "#172033",
            "size": 12,
        },
        "paper_bgcolor": "#ffffff",
        "plot_bgcolor": "#f6f8fb",
        "hovermode": "closest",
        "showlegend": True,
    }


def render_figure(fig: Figure) -> str:
    """Render a :class:`Figure` to a Plotly chart container plus its init script.

    The returned HTML is a sized ``<div>`` followed by a scoped ``<script>`` that
    calls ``Plotly.newPlot``. The page shell (:func:`render_page`) loads the
    ``plotly.js`` library once from the CDN.
    """

    div_id = f"plotfig-{next(_PLOT_IDS)}"
    payload = json.dumps(
        {
            "data": _plotly_traces(fig),
            "layout": _plotly_layout(fig),
            "config": {"responsive": True, "displaylogo": False, "displayModeBar": False},
        }
    )
    return (
        f'<div class="plotly-fig" id="{div_id}" style="height:{fig.height}px"></div>\n'
        f"<script>(function(){{var s={payload};"
        f"Plotly.newPlot({div_id!r},s.data,s.layout,s.config);}})();</script>"
    )


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
        "      if (window.Plotly) {\n"
        "        const shown = group.querySelector('.figure-panel.active');\n"
        "        if (shown)\n"
        "          shown.querySelectorAll('.js-plotly-plot').forEach((p) => window.Plotly.Plots.resize(p));\n"
        "      }\n"
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
    .plotly-fig {
      width: 100%;
      display: block;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #ffffff;
      overflow: hidden;
    }
    code {
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 0.95em;
      color: #334155;
    }"""


def render_page(
    *,
    title: str,
    body: str,
    lang: str = "en",
    css: str | None = None,
    extra_css: str = "",
    with_plotly: bool = True,
) -> str:
    """Wrap ``body`` in the canonical HTML document shell.

    Args:
        title: Page ``<title>`` and document language are escaped automatically.
        body: Inner HTML placed inside ``<main>``.
        lang: Document language attribute.
        css: Stylesheet to inline; defaults to :data:`THEME_CSS`.
        extra_css: Report-specific rules appended after the base stylesheet, for
            components the canonical theme does not cover (e.g. a setup grid).
        with_plotly: Load ``plotly.js`` from the CDN. Set to ``False`` for pages
            with no charts (e.g. a link index) to skip the unused download.
    """

    stylesheet = THEME_CSS if css is None else css
    if extra_css:
        stylesheet = f"{stylesheet}\n{extra_css}"
    plotly_tag = f'  <script src="{escape(PLOTLY_CDN_URL)}" charset="utf-8"></script>\n' if with_plotly else ""
    return (
        "<!doctype html>\n"
        f'<html lang="{escape(lang)}">\n'
        "<head>\n"
        '  <meta charset="utf-8">\n'
        '  <meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"  <title>{escape(title)}</title>\n"
        f"{plotly_tag}"
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
