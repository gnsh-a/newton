# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import importlib.util
import math
import sys
import tempfile
import unittest
from pathlib import Path

SUMMARY_COLUMNS = (
    "sweep",
    "sweep_value",
    "mode",
    "reduce_contacts",
    "sdf_resolution",
    "drop_height_m",
    "impact_velocity_m_per_s",
    "face_count_N",
    "rigid_count_K",
    "N_over_K",
    "max_penetration_m",
    "final_penetration_m",
    "final_fz_N",
    "final_fz_over_weight",
    "settled_time_s",
    "peak_fz_N",
    "peak_fz_over_weight",
    "final_tilt_deg",
    "final_drift_m",
    "buffer_overflow",
    "state_invalid",
    "in_band",
    "valid",
)

TIMESERIES_COLUMNS = (
    "sweep_value",
    "mode",
    "time_s",
    "cube_z_m",
    "cube_vz_m_per_s",
    "penetration_m",
    "solver_fz_N",
    "rigid_contact_count",
    "face_contact_count",
)


def _load_report_tool():
    path = Path(__file__).with_name("transient_gap_report.py")
    spec = importlib.util.spec_from_file_location("transient_gap_report", path)
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


def _summary_row(
    *,
    sweep: str,
    sweep_value: float,
    mode: str,
    resolution: int,
    face_n: int,
    rigid_k: int,
    max_pen: float,
    peak_fzw: float,
    final_fzw: float,
) -> dict[str, str]:
    reduced = mode == "reduced"
    return {
        "sweep": sweep,
        "sweep_value": str(sweep_value),
        "mode": mode,
        "reduce_contacts": str(reduced).lower(),
        "sdf_resolution": str(resolution),
        "drop_height_m": str(sweep_value if sweep != "resolution" else 0.0005),
        "impact_velocity_m_per_s": "0.1",
        "face_count_N": str(face_n),
        "rigid_count_K": str(rigid_k),
        "N_over_K": str(face_n / rigid_k),
        "max_penetration_m": str(max_pen),
        "final_penetration_m": "2.0e-6",
        "final_fz_N": str(final_fzw * 7.848),
        "final_fz_over_weight": str(final_fzw),
        "settled_time_s": "0.05",
        "peak_fz_N": str(peak_fzw * 7.848),
        "peak_fz_over_weight": str(peak_fzw),
        "final_tilt_deg": "0.0",
        "final_drift_m": "0.0",
        "buffer_overflow": "false",
        "state_invalid": "false",
        "in_band": "true",
        "valid": "true",
    }


def _gap_pair(sweep: str, value: float, resolution: int, face_n: int, rigid_k: int) -> list[dict[str, str]]:
    pred = math.sqrt(face_n / rigid_k)
    reduced_pen = 3.0e-4
    dense_pen = reduced_pen * pred  # ratio matches sqrt(N/K)
    return [
        _summary_row(
            sweep=sweep,
            sweep_value=value,
            mode="unreduced",
            resolution=resolution,
            face_n=face_n,
            rigid_k=rigid_k,
            max_pen=dense_pen,
            peak_fzw=1.4,
            final_fzw=0.99,
        ),
        _summary_row(
            sweep=sweep,
            sweep_value=value,
            mode="reduced",
            resolution=resolution,
            face_n=face_n,
            rigid_k=rigid_k,
            max_pen=reduced_pen,
            peak_fzw=3.8,
            final_fzw=1.0,
        ),
    ]


def _timeseries_rows(mode: str, peak: float, depth: float) -> list[dict[str, str]]:
    rows = []
    for frame in range(6):
        pen = max(0.0, depth * (1.0 - abs(frame - 3) / 3.0))
        fz = peak * max(0.0, 1.0 - abs(frame - 3) / 3.0)
        rows.append(
            {
                "sweep_value": "0.0005",
                "mode": mode,
                "time_s": str(0.00025 * (frame + 1)),
                "cube_z_m": str(0.05 - pen),
                "cube_vz_m_per_s": str(-0.1 + 0.03 * frame),
                "penetration_m": str(pen),
                "solver_fz_N": str(fz),
                "rigid_contact_count": "34" if mode == "reduced" else "2046",
                "face_contact_count": "2046",
            }
        )
    return rows


class TestTransientGapReport(unittest.TestCase):
    def test_html_report(self):
        report_tool = _load_report_tool()
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            summaries: list[dict[str, str]] = []
            summaries += _gap_pair("height", 0.00025, 32, 2046, 34)
            summaries += _gap_pair("height", 0.0005, 32, 2046, 34)
            summaries += _gap_pair("resolution", 16, 16, 510, 34)
            summaries += _gap_pair("resolution", 32, 32, 2046, 34)
            summaries += [
                _summary_row(
                    sweep="static",
                    sweep_value=0.0001,
                    mode="unreduced",
                    resolution=32,
                    face_n=2046,
                    rigid_k=34,
                    max_pen=1.4e-3,
                    peak_fzw=1.0,
                    final_fzw=0.99,
                ),
                _summary_row(
                    sweep="static",
                    sweep_value=0.0001,
                    mode="reduced",
                    resolution=32,
                    face_n=2046,
                    rigid_k=34,
                    max_pen=1.7e-4,
                    peak_fzw=1.0,
                    final_fzw=1.0,
                ),
            ]
            ts_rows = _timeseries_rows("unreduced", peak=10.0, depth=3.0e-3) + _timeseries_rows(
                "reduced", peak=25.0, depth=3.6e-4
            )

            _write_csv(output_dir / report_tool.SUMMARY_CSV, SUMMARY_COLUMNS, summaries)
            _write_csv(output_dir / report_tool.TIMESERIES_CSV, TIMESERIES_COLUMNS, ts_rows)

            html_path = report_tool.write_html_report(csv_dir=output_dir, output_path=output_dir / "report.html")
            self.assertTrue(html_path.exists())
            self.assertTrue((output_dir / report_tool.HYPOTHESIS_RECORD_NAME).exists())
            report = html_path.read_text(encoding="utf-8")
            self.assertIn("H7: Transient-Compliance Contact Reduction", report)
            self.assertIn("Figure 1: impact time history", report)
            self.assertIn("velocity independence", report)
            self.assertIn("the √(N/K) law", report)
            self.assertIn("Transient gap", report)
            self.assertIn("Static control", report)
            self.assertIn("reduce on", report)
            self.assertIn("pass", report)
            self.assertIn("Generated from", report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
