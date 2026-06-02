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
    "initial_epsilon",
    "mode",
    "epsilon",
    "horizontal_speed_m_per_s",
    "cylinder_omega_z_rad_per_s",
    "solver_fx_N",
    "solver_tz_Nm",
    "solver_force_count",
)

SUMMARY_COLUMNS = (
    "initial_epsilon",
    "initial_omega_rad_per_s",
    "mode",
    "epsilon_reference",
    "late_epsilon",
    "speed_stop_time_s",
    "spin_stop_time_s",
    "final_speed_m_per_s",
    "mean_solver_force_count",
    "buffer_overflow",
)


def _load_report_tool():
    path = Path(__file__).with_name("sliding_spinning_cylinder_report.py")
    spec = importlib.util.spec_from_file_location("sliding_spinning_cylinder_report", path)
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


def _timeseries_rows(*, epsilon0: float, mode: str, force_count: int, force_x: float, torque_z: float):
    rows = []
    for frame in range(5):
        time_s = 0.05 * frame
        rows.append(
            {
                "time_s": str(time_s),
                "initial_epsilon": str(epsilon0),
                "mode": mode,
                "epsilon": str(epsilon0 + 0.1 * frame),
                "horizontal_speed_m_per_s": str(max(0.25 - 0.05 * frame, 0.0)),
                "cylinder_omega_z_rad_per_s": str(max(10.0 - 2.0 * frame, 0.0)),
                "solver_fx_N": str(force_x),
                "solver_tz_Nm": str(torque_z),
                "solver_force_count": str(force_count),
            }
        )
    return rows


def _summary_row(*, epsilon0: float, mode: str, late_epsilon: float, force_count: int) -> dict[str, str]:
    return {
        "initial_epsilon": str(epsilon0),
        "initial_omega_rad_per_s": "10.0",
        "mode": mode,
        "epsilon_reference": "0.653",
        "late_epsilon": str(late_epsilon),
        "speed_stop_time_s": "0.2",
        "spin_stop_time_s": "0.19",
        "final_speed_m_per_s": "0.01",
        "mean_solver_force_count": str(force_count),
        "buffer_overflow": "false",
    }


class TestSlidingSpinningCylinderReport(unittest.TestCase):
    def test_html_report(self):
        report_tool = _load_report_tool()
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            timeseries_rows = (
                _timeseries_rows(epsilon0=0.5, mode="unreduced", force_count=1200, force_x=-0.04, torque_z=-0.0012)
                + _timeseries_rows(epsilon0=0.5, mode="reduced", force_count=70, force_x=-0.05, torque_z=-0.0015)
                + _timeseries_rows(epsilon0=1.0, mode="unreduced", force_count=1300, force_x=-0.06, torque_z=-0.0011)
                + _timeseries_rows(epsilon0=1.0, mode="reduced", force_count=75, force_x=-0.07, torque_z=-0.0016)
            )
            summary_rows = [
                _summary_row(epsilon0=0.5, mode="unreduced", late_epsilon=0.68, force_count=1200),
                _summary_row(epsilon0=0.5, mode="reduced", late_epsilon=0.82, force_count=70),
                _summary_row(epsilon0=1.0, mode="unreduced", late_epsilon=1.03, force_count=1300),
                _summary_row(epsilon0=1.0, mode="reduced", late_epsilon=1.37, force_count=75),
            ]
            _write_csv(output_dir / report_tool.TIMESERIES_CSV, TIMESERIES_COLUMNS, timeseries_rows)
            _write_csv(output_dir / report_tool.SUMMARY_CSV, SUMMARY_COLUMNS, summary_rows)

            html_path = report_tool.write_html_report(
                csv_dir=output_dir,
                output_path=output_dir / "report.html",
                selected_epsilon=1.0,
            )
            self.assertTrue(html_path.exists())
            self.assertTrue((output_dir / report_tool.HYPOTHESIS_RECORD_NAME).exists())
            report = html_path.read_text(encoding="utf-8")
            self.assertIn("H5: Sliding-Spinning Cylinder Contact Reduction", report)
            self.assertIn("Figure 1", report)
            self.assertIn("Figure 2", report)
            self.assertIn("Figure 3", report)
            self.assertIn("reference epsilon", report)
            self.assertIn("reduce on", report)
            self.assertIn("Generated from", report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
