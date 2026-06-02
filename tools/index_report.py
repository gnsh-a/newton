# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Build an index page that links to every hypothesis report.

This script imports each ``tools/<experiment>_report.py`` module to read its
canonical page title and default HTML path, then renders a single ``index.html``
listing the reports in hypothesis order. It uses the shared report framework in
:mod:`_report_common` so the index matches the reports' look.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _report_common as rc

DEFAULT_OUTPUT_ROOT = Path("output")
DEFAULT_HTML_PATH = DEFAULT_OUTPUT_ROOT / "index.html"
PAGE_TITLE = "Contact-Reduction Hypotheses"

# Report modules in hypothesis order, each with a one-line summary for the index.
REPORTS: tuple[tuple[str, str], ...] = (
    ("cube_on_plate_settle_report.py", "Static settle: support force equals weight, with no drift or tilt."),
    ("flat_sliding_block_report.py", "Sliding block stops at the rigid Coulomb time and distance."),
    ("cube_on_plate_tipping_report.py", "Tip-versus-slide threshold and the pre-tip tilt response."),
    ("spinning_cylinder_spin_down_report.py", "Pure-spin yaw torque (2/3 mu m g R) and linear spin-down."),
    ("sliding_spinning_cylinder_report.py", "Coupled slide and spin along the Farkas curve."),
    ("cube_on_plate_impact_report.py", "Impact ring-down: peak force, compression, rebound, and settle."),
    ("transient_gap_report.py", "Transient-compliance gap: delta_dense / delta_reduced = sqrt(N/K)."),
)

INDEX_CSS = """\
    .index-list {
      list-style: none;
      margin: 18px 0 0;
      padding: 0;
      max-width: 920px;
    }
    .index-item {
      display: flex;
      flex-direction: column;
      gap: 3px;
      padding: 14px 16px;
      margin-bottom: 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--soft);
    }
    .index-link {
      font-size: 16px;
      font-weight: 650;
      color: var(--accent);
      text-decoration: none;
    }
    .index-link:hover {
      text-decoration: underline;
    }
    .index-desc {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
    }"""


def _load_module(filename: str):
    """Import a sibling report module by filename."""

    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_entries(output_root: Path) -> list[tuple[str, str, str]]:
    """Return ``(title, href, description)`` per report, in hypothesis order.

    The href is the report's default HTML path made relative to ``output_root``
    so the index links resolve when both sit under the same output directory.
    """

    entries: list[tuple[str, str, str]] = []
    for filename, description in REPORTS:
        module = _load_module(filename)
        report_path = Path(module.DEFAULT_HTML_PATH)
        try:
            href = report_path.relative_to(output_root)
        except ValueError:
            href = report_path
        entries.append((module.PAGE_TITLE, str(href), description))
    return entries


def _body(entries: list[tuple[str, str, str]]) -> str:
    items = [
        f'<li class="index-item">'
        f'<a class="index-link" href="{rc.escape(href)}">{rc.escape(title)}</a>'
        f'<span class="index-desc">{rc.escape(description)}</span></li>'
        for title, href, description in entries
    ]
    return "\n".join(
        [
            f"<h1>{rc.escape(PAGE_TITLE)}</h1>",
            '<p class="lede">Each report compares hydroelastic contact with reduction off versus on for one '
            "hypothesis. Open a report to see its figures, tables, and pass/fail gates.</p>",
            '<ul class="index-list">',
            *items,
            "</ul>",
        ]
    )


def write_index(
    *,
    output_path: str | Path = DEFAULT_HTML_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> Path:
    """Write the index HTML and return its path."""

    entries = build_entries(Path(output_root))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        rc.render_page(title=PAGE_TITLE, body=_body(entries), extra_css=INDEX_CSS, with_plotly=False),
        encoding="utf-8",
    )
    return output_path


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output", type=str, default=str(DEFAULT_HTML_PATH), help="Index HTML output path.")
    parser.add_argument(
        "--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT), help="Root the report links are relative to."
    )
    return parser


def main() -> None:
    args = create_parser().parse_args()
    path = write_index(output_path=args.output, output_root=args.output_root)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
