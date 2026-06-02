# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

TIMESERIES_COLUMNS = (
    "time_s",
    "mode",
    "applied_force_over_ftip",
    "cube_pitch_deg",
    "cube_x_m",
    "center_pressure_x_over_half_extent",
    "solver_fx_N",
    "solver_fz_N",
    "solver_ty_Nm",
    "solver_force_count",
)

SUMMARY_COLUMNS = (
    "mode",
    "buffer_overflow",
    "mu_sliding",
    "event_type",
    "event_force_N",
    "event_force_over_ftip",
    "pitch_at_0p25_ftip_deg",
    "pitch_at_0p50_ftip_deg",
    "pitch_at_0p75_ftip_deg",
    "pitch_at_0p90_ftip_deg",
    "mean_solver_force_count",
)


def _load_report_tool():
    path = Path(__file__).with_name("cube_on_plate_tipping_report.py")
    spec = importlib.util.spec_from_file_location("cube_on_plate_tipping_report", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _timeseries_rows(*, mode: str, force_count: int) -> list[dict[str, str]]:
    rows = []
    for frame in range(5):
        ratio = 0.25 * frame
        rows.append(
            {
                "time_s": str(0.1 * frame),
                "mode": mode,
                "applied_force_over_ftip": str(ratio),
                "cube_pitch_deg": str(8.0 * ratio),
                "cube_x_m": str(0.001 * ratio),
                "center_pressure_x_over_half_extent": str(0.2 + 0.6 * ratio),
                "solver_fx_N": str(-3.0 * ratio),
                "solver_fz_N": "7.85",
                "solver_ty_Nm": str(-0.02 * ratio),
                "solver_force_count": str(force_count),
            }
        )
    return rows


def _summary_row(*, mode: str, force_count: int) -> dict[str, str]:
    return {
        "mode": mode,
        "buffer_overflow": "false",
        "mu_sliding": "0.7",
        "event_type": "tip",
        "event_force_N": "4.2",
        "event_force_over_ftip": "1.07",
        "pitch_at_0p25_ftip_deg": "1.0",
        "pitch_at_0p50_ftip_deg": "2.0",
        "pitch_at_0p75_ftip_deg": "4.0",
        "pitch_at_0p90_ftip_deg": "7.0",
        "mean_solver_force_count": str(force_count),
    }


class TestCubeOnPlateTippingReport(unittest.TestCase):
    def test_html_report(self):
        report_tool = _load_report_tool()
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            rows = _timeseries_rows(mode="unreduced", force_count=2046) + _timeseries_rows(
                mode="reduced", force_count=34
            )
            summaries = [
                _summary_row(mode="unreduced", force_count=2046),
                _summary_row(mode="reduced", force_count=34),
            ]
            _write_csv(output_dir / report_tool.TIMESERIES_CSV, TIMESERIES_COLUMNS, rows)
            _write_csv(output_dir / report_tool.SUMMARY_CSV, SUMMARY_COLUMNS, summaries)

            html_path = report_tool.write_html_report(csv_dir=output_dir, output_path=output_dir / "report.html")
            self.assertTrue(html_path.exists())
            self.assertTrue((output_dir / report_tool.HYPOTHESIS_RECORD_NAME).exists())
            report = html_path.read_text(encoding="utf-8")
            self.assertIn("H3: Cube-on-Plate Tipping Contact Reduction", report)
            self.assertIn("Figure 1", report)
            self.assertIn("Figure 2", report)
            self.assertIn("Figure 3", report)
            self.assertIn("center-pressure", report)
            self.assertIn("reduce on", report)
            self.assertIn("Generated from", report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
