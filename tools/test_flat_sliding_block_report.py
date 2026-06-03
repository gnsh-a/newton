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
    "initial_speed_m_per_s",
    "mode",
    "cube_x_m",
    "cube_speed_m_per_s",
    "solver_fx_N",
    "solver_force_count",
)

SUMMARY_COLUMNS = (
    "initial_speed_m_per_s",
    "mode",
    "expected_coulomb_stop_time_s",
    "expected_coulomb_stop_travel_m",
    "stopped",
    "stop_time_s",
    "stop_travel_m",
    "final_y_m",
    "final_tilt_deg",
    "max_penetration_depth_m",
    "mean_solver_force_count",
    "max_rigid_contact_count",
    "rigid_contact_capacity",
    "buffer_overflow",
    "max_reduction_hashtable_failures",
)


def _load_report_tool():
    path = Path(__file__).with_name("flat_sliding_block_report.py")
    spec = importlib.util.spec_from_file_location("flat_sliding_block_report", path)
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


def _timeseries_rows(*, speed: float, mode: str, force_count: int) -> list[dict[str, str]]:
    rows = []
    for frame in range(4):
        time_s = 0.025 * (frame + 1)
        rows.append(
            {
                "time_s": str(time_s),
                "initial_speed_m_per_s": str(speed),
                "mode": mode,
                "cube_x_m": str(min(speed * time_s, speed * speed / 9.81)),
                "cube_speed_m_per_s": str(max(speed - 4.905 * time_s, 0.0)),
                "solver_fx_N": "-3.924",
                "solver_force_count": str(force_count),
            }
        )
    return rows


def _summary_row(*, speed: float, mode: str, stop_scale: float, force_count: int) -> dict[str, str]:
    expected_time = speed / 4.905
    expected_travel = speed * speed / (2.0 * 4.905)
    return {
        "initial_speed_m_per_s": str(speed),
        "mode": mode,
        "expected_coulomb_stop_time_s": str(expected_time),
        "expected_coulomb_stop_travel_m": str(expected_travel),
        "stopped": "true",
        "stop_time_s": str(expected_time * stop_scale),
        "stop_travel_m": str(expected_travel * stop_scale),
        "final_y_m": "0.0",
        "final_tilt_deg": "0.0",
        "max_penetration_depth_m": "0.0001",
        "mean_solver_force_count": str(force_count),
        "max_rigid_contact_count": str(force_count),
        "rigid_contact_capacity": "131072",
        "buffer_overflow": "false",
        "max_reduction_hashtable_failures": "0",
    }


class TestFlatSlidingBlockReport(unittest.TestCase):
    def test_html_report(self):
        report_tool = _load_report_tool()
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            timeseries_rows = (
                _timeseries_rows(speed=0.1, mode="unreduced", force_count=2046)
                + _timeseries_rows(speed=0.1, mode="reduced", force_count=32)
                + _timeseries_rows(speed=0.4, mode="unreduced", force_count=2046)
                + _timeseries_rows(speed=0.4, mode="reduced", force_count=28)
            )
            summary_rows = [
                _summary_row(speed=0.1, mode="unreduced", stop_scale=3.0, force_count=2046),
                _summary_row(speed=0.1, mode="reduced", stop_scale=1.05, force_count=32),
                _summary_row(speed=0.4, mode="unreduced", stop_scale=1.4, force_count=2046),
                _summary_row(speed=0.4, mode="reduced", stop_scale=0.98, force_count=28),
            ]
            _write_csv(output_dir / report_tool.TIMESERIES_CSV, TIMESERIES_COLUMNS, timeseries_rows)
            _write_csv(output_dir / report_tool.SUMMARY_CSV, SUMMARY_COLUMNS, summary_rows)

            html_path = report_tool.write_html_report(csv_dir=output_dir, output_path=output_dir / "report.html")
            self.assertTrue(html_path.exists())
            self.assertTrue((output_dir / report_tool.HYPOTHESIS_RECORD_NAME).exists())
            report = html_path.read_text(encoding="utf-8")
            self.assertIn("H2: Flat Sliding Block Contact Reduction", report)
            self.assertIn("Figure 1: primary measurables vs time", report)
            self.assertIn("Figure 2: additional response vs initial speed", report)
            self.assertIn("Figure 3: contact reduction and buffer sanity", report)
            self.assertIn("Figure 4: SDF resolution probe", report)
            self.assertIn("Coulomb reference", report)
            self.assertIn("reduce on", report)
            self.assertIn("Generated from", report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
