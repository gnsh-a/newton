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
    "initial_omega_rad_per_s",
    "mode",
    "solver_force_count",
    "solver_fz_N",
    "solver_tz_Nm",
    "omega_over_omega0",
)

SUMMARY_COLUMNS = (
    "initial_omega_rad_per_s",
    "mode",
    "buffer_overflow",
    "stopped",
    "expected_uniform_torque_z_Nm",
    "expected_uniform_angular_accel_rad_per_s2",
    "expected_uniform_stop_time_s",
    "stop_time_s",
    "mean_solver_tz_active_Nm",
    "mean_abs_solver_tz_active_Nm",
    "mean_solver_force_count",
)


def _load_report_tool():
    path = Path(__file__).with_name("spinning_cylinder_spin_down_report.py")
    spec = importlib.util.spec_from_file_location("spinning_cylinder_spin_down_report", path)
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


def _timeseries_rows(*, omega0: float, mode: str, force_count: int, torque: float) -> list[dict[str, str]]:
    rows = []
    for frame in range(5):
        time_s = 0.05 * frame
        rows.append(
            {
                "time_s": str(time_s),
                "initial_omega_rad_per_s": str(omega0),
                "mode": mode,
                "solver_force_count": str(force_count),
                "solver_fz_N": "0.12",
                "solver_tz_Nm": str(torque),
                "omega_over_omega0": str(max(1.0 - 4.0 * time_s, 0.0)),
            }
        )
    return rows


def _summary_row(*, omega0: float, mode: str, force_count: int, stop_time: float, torque: float) -> dict[str, str]:
    return {
        "initial_omega_rad_per_s": str(omega0),
        "mode": mode,
        "buffer_overflow": "false",
        "stopped": "true",
        "expected_uniform_torque_z_Nm": "-0.0012",
        "expected_uniform_angular_accel_rad_per_s2": "-52.0",
        "expected_uniform_stop_time_s": "0.29",
        "stop_time_s": str(stop_time),
        "mean_solver_tz_active_Nm": str(torque),
        "mean_abs_solver_tz_active_Nm": str(abs(torque)),
        "mean_solver_force_count": str(force_count),
    }


class TestSpinningCylinderSpinDownReport(unittest.TestCase):
    def test_html_report(self):
        report_tool = _load_report_tool()
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            timeseries_rows = (
                _timeseries_rows(omega0=15.0, mode="unreduced", force_count=1400, torque=-0.0014)
                + _timeseries_rows(omega0=15.0, mode="reduced", force_count=70, torque=-0.0020)
                + _timeseries_rows(omega0=30.0, mode="unreduced", force_count=1400, torque=-0.0015)
                + _timeseries_rows(omega0=30.0, mode="reduced", force_count=72, torque=-0.0021)
            )
            summary_rows = [
                _summary_row(omega0=15.0, mode="unreduced", force_count=1400, stop_time=0.24, torque=-0.0014),
                _summary_row(omega0=15.0, mode="reduced", force_count=70, stop_time=0.22, torque=-0.0020),
                _summary_row(omega0=30.0, mode="unreduced", force_count=1400, stop_time=0.48, torque=-0.0015),
                _summary_row(omega0=30.0, mode="reduced", force_count=72, stop_time=0.43, torque=-0.0021),
            ]
            _write_csv(output_dir / report_tool.TIMESERIES_CSV, TIMESERIES_COLUMNS, timeseries_rows)
            _write_csv(output_dir / report_tool.SUMMARY_CSV, SUMMARY_COLUMNS, summary_rows)

            sdf24_dir = output_dir / "sdf24"
            sdf32_dir = output_dir / "sdf32"
            _write_csv(sdf24_dir / report_tool.SUMMARY_CSV, SUMMARY_COLUMNS, summary_rows)
            _write_csv(sdf32_dir / report_tool.SUMMARY_CSV, SUMMARY_COLUMNS, summary_rows)

            html_path = report_tool.write_html_report(
                csv_dir=output_dir,
                output_path=output_dir / "report.html",
                sdf_sweep_dirs=((24.0, sdf24_dir), (32.0, sdf32_dir)),
            )
            self.assertTrue(html_path.exists())
            self.assertTrue((output_dir / report_tool.HYPOTHESIS_RECORD_NAME).exists())
            report = html_path.read_text(encoding="utf-8")
            self.assertIn("H4: Spinning Cylinder Yaw-Torque Contact Reduction", report)
            self.assertIn("Figure 1", report)
            self.assertIn("Figure 2", report)
            self.assertIn("Figure 3", report)
            self.assertIn("Figure 4", report)
            self.assertIn("Stop-time error vs SDF resolution", report)
            self.assertIn("uniform-pressure reference", report)
            self.assertIn("reduce on", report)
            self.assertIn("Generated from", report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
