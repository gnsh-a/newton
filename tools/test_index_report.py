# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


def _load_index_tool():
    path = Path(__file__).with_name("index_report.py")
    spec = importlib.util.spec_from_file_location("index_report", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestIndexReport(unittest.TestCase):
    def test_index_lists_every_report(self):
        index_tool = _load_index_tool()
        entries = index_tool.build_entries(index_tool.DEFAULT_OUTPUT_ROOT)
        # One entry per registered report, all H1-H7 represented.
        self.assertEqual(len(entries), len(index_tool.REPORTS))

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            html_path = index_tool.write_index(
                output_path=output_dir / "index.html", output_root=index_tool.DEFAULT_OUTPUT_ROOT
            )
            self.assertTrue(html_path.exists())
            report = html_path.read_text(encoding="utf-8")

            self.assertIn("Contact-Reduction Hypotheses", report)
            # No charts on the index, so the Plotly library must not be loaded.
            self.assertNotIn(index_tool.rc.PLOTLY_CDN_URL, report)
            for title, href, _desc in entries:
                self.assertIn(title, report)
                self.assertIn(f'href="{href}"', report)
            # Every hypothesis number appears.
            for number in range(1, 8):
                self.assertIn(f"H{number}:", report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
