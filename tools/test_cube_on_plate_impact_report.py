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
    "height_m",
    "mode",
    "reduce_contacts",
    "pre_prune_contacts",
    "cube_x_m",
    "cube_y_m",
    "cube_z_m",
    "cube_vz_m_per_s",
    "cube_tilt_deg",
    "cube_signed_clearance_m",
    "cube_penetration_depth_m",
    "solver_fx_N",
    "solver_fy_N",
    "solver_fz_N",
    "solver_tx_Nm",
    "solver_ty_Nm",
    "solver_tz_Nm",
    "solver_force_count",
    "rigid_contact_count",
    "face_contact_count",
    "buffer_overflow",
    "state_invalid",
)

SUMMARY_COLUMNS = (
    "height_m",
    "mode",
    "reduce_contacts",
    "pre_prune_contacts",
    "frame_count",
    "simulation_time_s",
    "step_dt_s",
    "cube_mass_kg",
    "cube_weight_N",
    "impact_velocity_m_per_s",
    "first_contact_time_s",
    "peak_solver_fz_N",
    "peak_solver_fz_over_weight",
    "time_to_peak_fz_s",
    "max_penetration_depth_m",
    "max_upward_rebound_velocity_m_per_s",
    "rebound_velocity_ratio",
    "settle_time_s",
    "post_settle_fz_rms_N",
    "normal_impulse_Ns",
    "final_tilt_deg",
    "final_drift_m",
    "mean_solver_force_count",
    "max_rigid_contact_count",
    "rigid_contact_capacity",
    "max_face_contact_count",
    "buffer_overflow",
    "state_invalid",
    "valid_run",
)


def _load_report_tool():
    path = Path(__file__).with_name("cube_on_plate_impact_report.py")
    spec = importlib.util.spec_from_file_location("cube_on_plate_impact_report", path)
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


def _timeseries_rows(*, height: float, mode: str, force_count: int, peak: float) -> list[dict[str, str]]:
    rows = []
    for frame in range(6):
        time_s = 0.001 * (frame + 1)
        penetration = max(0.0, 0.0001 * (3 - abs(frame - 3)))
        fz = peak * max(0.0, 1.0 - abs(frame - 3) / 3.0)
        rows.append(
            {
                "time_s": str(time_s),
                "height_m": str(height),
                "mode": mode,
                "reduce_contacts": str(mode == "reduced").lower(),
                "pre_prune_contacts": "false",
                "cube_x_m": "0.0",
                "cube_y_m": "0.0",
                "cube_z_m": str(0.05 + height - penetration),
                "cube_vz_m_per_s": str(-0.1 + 0.04 * frame),
                "cube_tilt_deg": "0.0",
                "cube_signed_clearance_m": str(-penetration),
                "cube_penetration_depth_m": str(penetration),
                "solver_fx_N": "0.0",
                "solver_fy_N": "0.0",
                "solver_fz_N": str(fz),
                "solver_tx_Nm": "0.0",
                "solver_ty_Nm": "0.0",
                "solver_tz_Nm": "0.0",
                "solver_force_count": str(force_count if fz > 0.0 else 0),
                "rigid_contact_count": str(force_count if fz > 0.0 else 0),
                "face_contact_count": "2046",
                "buffer_overflow": "false",
                "state_invalid": "false",
            }
        )
    return rows


def _summary_row(*, height: float, mode: str, peak_ratio: float, contacts: int) -> dict[str, str]:
    weight = 7.848
    return {
        "height_m": str(height),
        "mode": mode,
        "reduce_contacts": str(mode == "reduced").lower(),
        "pre_prune_contacts": "false",
        "frame_count": "6",
        "simulation_time_s": "0.006",
        "step_dt_s": "0.001",
        "cube_mass_kg": "0.8",
        "cube_weight_N": str(weight),
        "impact_velocity_m_per_s": "0.14",
        "first_contact_time_s": "0.003",
        "peak_solver_fz_N": str(peak_ratio * weight),
        "peak_solver_fz_over_weight": str(peak_ratio),
        "time_to_peak_fz_s": "0.001",
        "max_penetration_depth_m": "0.0002",
        "max_upward_rebound_velocity_m_per_s": "0.02",
        "rebound_velocity_ratio": "0.14",
        "settle_time_s": "0.05",
        "post_settle_fz_rms_N": "0.1",
        "normal_impulse_Ns": "0.02",
        "final_tilt_deg": "0.0",
        "final_drift_m": "0.0",
        "mean_solver_force_count": str(contacts),
        "max_rigid_contact_count": str(contacts),
        "rigid_contact_capacity": "131072",
        "max_face_contact_count": "2046",
        "buffer_overflow": "false",
        "state_invalid": "false",
        "valid_run": "true",
    }


class TestCubeOnPlateImpactReport(unittest.TestCase):
    def test_html_report(self):
        report_tool = _load_report_tool()
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            rows = (
                _timeseries_rows(height=0.001, mode="unreduced", force_count=2046, peak=10.0)
                + _timeseries_rows(height=0.001, mode="reduced", force_count=34, peak=25.0)
                + _timeseries_rows(height=0.005, mode="unreduced", force_count=2046, peak=18.0)
                + _timeseries_rows(height=0.005, mode="reduced", force_count=32, peak=45.0)
            )
            summaries = [
                _summary_row(height=0.001, mode="unreduced", peak_ratio=1.3, contacts=2046),
                _summary_row(height=0.001, mode="reduced", peak_ratio=3.2, contacts=34),
                _summary_row(height=0.005, mode="unreduced", peak_ratio=2.1, contacts=2046),
                _summary_row(height=0.005, mode="reduced", peak_ratio=5.4, contacts=32),
            ]
            _write_csv(output_dir / report_tool.TIMESERIES_CSV, TIMESERIES_COLUMNS, rows)
            _write_csv(output_dir / report_tool.SUMMARY_CSV, SUMMARY_COLUMNS, summaries)

            html_path = report_tool.write_html_report(csv_dir=output_dir, output_path=output_dir / "report.html")
            self.assertTrue(html_path.exists())
            self.assertTrue((output_dir / report_tool.HYPOTHESIS_RECORD_NAME).exists())
            report = html_path.read_text(encoding="utf-8")
            self.assertIn("H6: Cube-on-Plate Impact", report)
            self.assertIn("Figure 1", report)
            self.assertIn("Figure 2", report)
            self.assertIn("Figure 3", report)
            self.assertIn("Figure 4", report)
            self.assertIn("SDOF reference", report)
            self.assertIn("steps/half-T", report)
            self.assertIn("Fz / mg", report)
            self.assertIn("reduce on", report)
            self.assertIn("Generated from", report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
