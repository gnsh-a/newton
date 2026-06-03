# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import re
import sys
import unittest
from pathlib import Path


def _load_rc():
    path = Path(__file__).with_name("_report_common.py")
    spec = importlib.util.spec_from_file_location("_report_common", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _payload(html: str) -> dict:
    match = re.search(r"var s=(\{.*\});Plotly\.newPlot", html)
    assert match, "no Plotly payload found"
    return json.loads(match.group(1))


class TestGroupSelector(unittest.TestCase):
    def test_dropdown_filters_by_group(self):
        rc = _load_rc()
        series = [
            rc.Series([0, 1], [1, 2], "a off", rc.group_color(0), dash="dash", group="g=a"),
            rc.Series([0, 1], [2, 3], "a on", rc.group_color(0), group="g=a"),
            rc.Series([0, 1], [3, 4], "b off", rc.group_color(1), dash="dash", group="g=b"),
            rc.Series([0, 1], [4, 5], "b on", rc.group_color(1), group="g=b"),
        ]
        fig = rc.Figure(
            title="t",
            xlabel="x",
            ylabel="y",
            series=series,
            x_range=(0, 1),
            y_range=(0, 5),
            selector="group",
        )
        spec = _payload(rc.render_figure(fig))
        menus = spec["layout"]["updatemenus"]
        labels = [b["label"] for b in menus[0]["buttons"]]
        self.assertEqual(labels, ["All", "g=a", "g=b"])
        # "All" shows every trace; each group button shows exactly its two.
        self.assertEqual(menus[0]["buttons"][0]["args"][0]["visible"], [True, True, True, True])
        self.assertEqual(menus[0]["buttons"][1]["args"][0]["visible"], [True, True, False, False])
        self.assertEqual(menus[0]["buttons"][2]["args"][0]["visible"], [False, False, True, True])
        # selector label is shown.
        self.assertTrue(any(a.get("text") == "group" for a in spec["layout"]["annotations"]))

    def test_ungrouped_series_always_visible(self):
        rc = _load_rc()
        series = [
            rc.Series([0, 1], [1, 2], "ref", rc.REFERENCE_COLOR, dash="dot"),  # group=None
            rc.Series([0, 1], [2, 3], "a", rc.group_color(0), group="g=a"),
            rc.Series([0, 1], [3, 4], "b", rc.group_color(1), group="g=b"),
        ]
        fig = rc.Figure(
            title="t",
            xlabel="x",
            ylabel="y",
            series=series,
            x_range=(0, 1),
            y_range=(0, 5),
            selector="group",
        )
        spec = _payload(rc.render_figure(fig))
        button_a = spec["layout"]["updatemenus"][0]["buttons"][1]
        # Ungrouped reference (index 0) stays visible under a group selection.
        self.assertEqual(button_a["args"][0]["visible"], [True, True, False])

    def test_solo_color_recolors_on_selection(self):
        rc = _load_rc()
        series = [
            rc.Series([0, 1], [1, 2], "a off", rc.group_color(0), group="g=a", solo_color="#0000ff"),
            rc.Series([0, 1], [2, 3], "a on", rc.group_color(0), group="g=a", solo_color="#ff0000"),
            rc.Series([0, 1], [3, 4], "b off", rc.group_color(1), group="g=b", solo_color="#0000ff"),
            rc.Series([0, 1], [4, 5], "b on", rc.group_color(1), group="g=b", solo_color="#ff0000"),
        ]
        fig = rc.Figure(
            title="t",
            xlabel="x",
            ylabel="y",
            series=series,
            x_range=(0, 1),
            y_range=(0, 5),
            selector="group",
        )
        spec = _payload(rc.render_figure(fig))
        buttons = spec["layout"]["updatemenus"][0]["buttons"]
        # "All" restores per-group base colors.
        self.assertEqual(
            buttons[0]["args"][0]["line.color"],
            [rc.group_color(0), rc.group_color(0), rc.group_color(1), rc.group_color(1)],
        )
        # A single group recolors to the per-series solo colors.
        self.assertEqual(buttons[1]["args"][0]["line.color"], ["#0000ff", "#ff0000", "#0000ff", "#ff0000"])
        self.assertEqual(buttons[1]["args"][0]["marker.color"], ["#0000ff", "#ff0000", "#0000ff", "#ff0000"])
        # Default-on-a-group opens already recolored.
        fig_default = rc.Figure(
            title="t",
            xlabel="x",
            ylabel="y",
            series=series,
            x_range=(0, 1),
            y_range=(0, 5),
            selector="group",
            selector_default="g=a",
        )
        first_trace = _payload(rc.render_figure(fig_default))["data"][0]
        self.assertEqual(first_trace["line"]["color"], "#0000ff")

    def test_no_solo_color_leaves_colors_untouched(self):
        rc = _load_rc()
        series = [
            rc.Series([0, 1], [1, 2], "a", rc.group_color(0), group="g=a"),
            rc.Series([0, 1], [2, 3], "b", rc.group_color(1), group="g=b"),
        ]
        fig = rc.Figure(
            title="t",
            xlabel="x",
            ylabel="y",
            series=series,
            x_range=(0, 1),
            y_range=(0, 5),
            selector="group",
        )
        # No solo colors anywhere -> buttons only toggle visibility, never recolor.
        for button in _payload(rc.render_figure(fig))["layout"]["updatemenus"][0]["buttons"]:
            self.assertNotIn("line.color", button["args"][0])

    def test_no_selector_has_no_updatemenus(self):
        rc = _load_rc()
        fig = rc.Figure(
            title="t",
            xlabel="x",
            ylabel="y",
            series=[rc.Series([0, 1], [1, 2], "a", "#000")],
            x_range=(0, 1),
            y_range=(0, 2),
        )
        spec = _payload(rc.render_figure(fig))
        self.assertNotIn("updatemenus", spec["layout"])

    def test_dash_keyword_passthrough(self):
        rc = _load_rc()
        fig = rc.Figure(
            title="t",
            xlabel="x",
            ylabel="y",
            series=[
                rc.Series([0, 1], [1, 2], "dot", "#000", dash="dot"),
                rc.Series([0, 1], [1, 2], "legacy", "#000", dash="5 5"),
                rc.Series([0, 1], [1, 2], "solid", "#000"),
            ],
            x_range=(0, 1),
            y_range=(0, 2),
        )
        dashes = [t["line"]["dash"] for t in _payload(rc.render_figure(fig))["data"]]
        self.assertEqual(dashes, ["dot", "dash", "solid"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
