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
    "rigid_contact_count",
    "cube_x_m",
    "cube_y_m",
    "cube_z_m",
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
    "buffer_overflow",
)

SUMMARY_COLUMNS = (
    "height_m",
    "mode",
    "reduce_contacts",
    "pre_prune_contacts",
    "frame_count",
    "buffer_overflow",
    "max_hydro_broadphase_blocks",
    "hydro_broadphase_capacity",
    "max_hydro_iso_subblocks_l0",
    "hydro_iso_subblocks_l0_capacity",
    "max_hydro_iso_subblocks_l1",
    "hydro_iso_subblocks_l1_capacity",
    "max_hydro_iso_subblocks_l2",
    "hydro_iso_subblocks_l2_capacity",
    "max_hydro_iso_voxels",
    "hydro_iso_voxels_capacity",
    "max_face_contact_count",
    "face_contact_capacity",
    "max_rigid_contact_count",
    "rigid_contact_capacity",
    "max_reduction_hashtable_active",
    "reduction_hashtable_capacity",
    "max_reduction_hashtable_failures",
    "final_cube_x_m",
    "final_cube_y_m",
    "final_cube_z_m",
    "final_cube_tilt_deg",
    "final_cube_signed_clearance_m",
    "final_cube_penetration_depth_m",
)


def _load_report_tool():
    path = Path(__file__).with_name("cube_on_plate_settle_report.py")
    spec = importlib.util.spec_from_file_location("cube_on_plate_settle_report", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _synthetic_timeseries_rows(*, mode: str, height: float, mg: float, force_count: int) -> list[dict[str, str]]:
    rows = []
    for i in range(4):
        row = dict.fromkeys(TIMESERIES_COLUMNS, "0")
        row.update(
            {
                "time_s": str(0.25 * (i + 1)),
                "height_m": str(height),
                "mode": mode,
                "reduce_contacts": str(mode == "reduced").lower(),
                "pre_prune_contacts": "false",
                "cube_x_m": "0.0",
                "cube_y_m": "0.0",
                "cube_z_m": "0.04998",
                "cube_tilt_deg": "0.0",
                "cube_signed_clearance_m": "-0.00002",
                "cube_penetration_depth_m": "0.00002",
                "solver_fx_N": "0.0" if i < 2 else "0.1",
                "solver_fy_N": "0.0",
                "solver_fz_N": str(0.5 * mg if i < 2 else mg),
                "solver_tx_Nm": "0.0" if i < 2 else "0.001",
                "solver_ty_Nm": "0.0",
                "solver_tz_Nm": "0.0" if i < 2 else "0.01",
                "solver_force_count": str(force_count),
                "rigid_contact_count": str(force_count),
                "buffer_overflow": "false",
            }
        )
        rows.append(row)
    return rows


def _synthetic_summary_row(*, mode: str, height: float, force_count: int) -> dict[str, str]:
    row = dict.fromkeys(SUMMARY_COLUMNS, "0")
    row.update(
        {
            "height_m": str(height),
            "mode": mode,
            "reduce_contacts": str(mode == "reduced").lower(),
            "pre_prune_contacts": "false",
            "frame_count": "4",
            "buffer_overflow": "false",
            "max_hydro_broadphase_blocks": "1",
            "hydro_broadphase_capacity": "100",
            "max_hydro_iso_subblocks_l0": "2",
            "hydro_iso_subblocks_l0_capacity": "100",
            "max_hydro_iso_subblocks_l1": "3",
            "hydro_iso_subblocks_l1_capacity": "100",
            "max_hydro_iso_subblocks_l2": "4",
            "hydro_iso_subblocks_l2_capacity": "100",
            "max_hydro_iso_voxels": "5",
            "hydro_iso_voxels_capacity": "100",
            "max_face_contact_count": "6",
            "face_contact_capacity": "100",
            "max_rigid_contact_count": str(force_count),
            "rigid_contact_capacity": "100000",
            "max_reduction_hashtable_active": "8" if mode == "reduced" else "0",
            "reduction_hashtable_capacity": "100",
            "max_reduction_hashtable_failures": "0",
            "final_cube_x_m": "0.0",
            "final_cube_y_m": "0.0",
            "final_cube_z_m": "0.04998",
            "final_cube_tilt_deg": "0.0",
            "final_cube_signed_clearance_m": "-0.00002",
            "final_cube_penetration_depth_m": "0.00002",
        }
    )
    return row


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class TestCubeOnPlateSettleReport(unittest.TestCase):
    def test_report_metrics_and_html(self):
        report_tool = _load_report_tool()
        constants = report_tool.PhysicsConstants(cube_side_m=0.1, cube_mass_kg=0.8, gravity_m_per_s2=9.81)
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            timeseries_rows = _synthetic_timeseries_rows(
                mode="unreduced",
                height=0.005,
                mg=constants.cube_weight_N,
                force_count=2000,
            ) + _synthetic_timeseries_rows(
                mode="reduced",
                height=0.005,
                mg=constants.cube_weight_N,
                force_count=32,
            )
            summary_rows = [
                _synthetic_summary_row(mode="unreduced", height=0.005, force_count=2000),
                _synthetic_summary_row(mode="reduced", height=0.005, force_count=32),
            ]
            _write_csv(output_dir / report_tool.TIMESERIES_CSV, TIMESERIES_COLUMNS, timeseries_rows)
            _write_csv(output_dir / report_tool.SUMMARY_CSV, SUMMARY_COLUMNS, summary_rows)

            runs = report_tool.load_runs(output_dir)
            summaries = report_tool.load_run_summaries(output_dir)
            self.assertEqual(len(summaries[0.005]), 2)
            selected_height = report_tool.select_height(runs, 0.005)
            self.assertEqual(selected_height, 0.005)

            metrics = report_tool.compute_metrics(
                runs,
                constants,
                window_fraction=0.5,
                summaries=summaries,
            )
            reduced = metrics[0.005]["reduced"]
            self.assertAlmostEqual(float(reduced["fz_norm_mean"]), 1.0, places=6)
            self.assertGreater(float(reduced["lateral_norm_mean"]), 0.0)
            self.assertGreater(float(reduced["torque_norm_mean"]), 0.0)
            self.assertAlmostEqual(float(reduced["penetration_depth_mean_mm"]), 0.02, places=6)
            self.assertGreater(float(reduced["support_offset_mean_mm"]), 0.0)
            self.assertEqual(float(reduced["solver_force_count_mean"]), 32.0)

            html_path = report_tool.write_html_report(
                runs,
                metrics,
                constants=constants,
                selected_height=selected_height,
                window_fraction=0.5,
                output_path=output_dir / "report.html",
            )
            self.assertTrue(html_path.exists())
            self.assertTrue((output_dir / report_tool.HYPOTHESIS_RECORD_NAME).exists())
            report = html_path.read_text(encoding="utf-8")
            self.assertIn("Figure 1: raw solver outputs vs time", report)
            self.assertIn("Figure 2: settled physical response vs height", report)
            self.assertIn("Figure 3: contact reduction and buffer sanity", report)
            self.assertIn("Geometric penetration", report)
            self.assertIn("Support-point offset", report)
            self.assertIn("contact count", report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
