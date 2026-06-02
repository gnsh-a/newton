# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import csv
import math
import tempfile
import unittest
from pathlib import Path

from newton.examples.contacts.experiment_cube_on_plate_settle import (
    SETTLE_FINAL_DRIFT_M,
    SETTLE_FINAL_TILT_DEG,
    SUMMARY_COLUMNS,
    SUMMARY_CSV,
    TIMESERIES_COLUMNS,
    TIMESERIES_CSV,
    run_experiment,
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _as_bool(value: str) -> bool:
    return value.lower() in ("1", "true", "yes")


def _max_force_norm(rows: list[dict[str, str]]) -> float:
    return max(
        math.sqrt(float(row["solver_fx_N"]) ** 2 + float(row["solver_fy_N"]) ** 2 + float(row["solver_fz_N"]) ** 2)
        for row in rows
    )


class TestCubeOnPlateSettleExperiment(unittest.TestCase):
    def test_cube_on_plate_settle_experiment(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            try:
                run_experiment(
                    output_dir=output_dir,
                    heights=(0.0, 0.00025),
                    simulation_time=0.1,
                    frame_fps=60,
                    sim_substeps=4,
                    verbose=False,
                )
            except ImportError as exc:
                self.skipTest(f"MuJoCo is not available: {exc}")

            timeseries_path = output_dir / TIMESERIES_CSV
            summary_path = output_dir / SUMMARY_CSV
            self.assertTrue(timeseries_path.exists())
            self.assertTrue(summary_path.exists())
            self.assertEqual(sorted(output_dir.glob("settle_h*_*.csv")), [])

            summary_rows = _read_csv(summary_path)
            self.assertEqual(len(summary_rows), 4)
            self.assertTrue(set(SUMMARY_COLUMNS).issubset(summary_rows[0].keys()))
            for row in summary_rows:
                if row["mode"] == "unreduced":
                    self.assertFalse(_as_bool(row["buffer_overflow"]))

            rows = _read_csv(timeseries_path)
            self.assertGreater(len(rows), 0)
            self.assertTrue(set(TIMESERIES_COLUMNS).issubset(rows[0].keys()))
            by_height_mode: dict[tuple[float, str], list[dict[str, str]]] = {}
            for row in rows:
                height = float(row["height_m"])
                mode = row["mode"]
                by_height_mode.setdefault((height, mode), []).append(row)

            self.assertEqual(
                set(by_height_mode),
                {(0.0, "unreduced"), (0.0, "reduced"), (0.00025, "unreduced"), (0.00025, "reduced")},
            )
            for (_height, mode), mode_rows in by_height_mode.items():
                self.assertGreater(_max_force_norm(mode_rows), 0.0)
                if mode == "unreduced":
                    self.assertFalse(any(_as_bool(row["buffer_overflow"]) for row in mode_rows))

                if mode == "reduced":
                    final = mode_rows[-1]
                    drift = math.hypot(float(final["cube_x_m"]), float(final["cube_y_m"]))
                    self.assertLess(float(final["cube_tilt_deg"]), SETTLE_FINAL_TILT_DEG)
                    self.assertLess(drift, SETTLE_FINAL_DRIFT_M)
                    self.assertGreaterEqual(float(final["cube_penetration_depth_m"]), 0.0)

            for height in (0.0, 0.00025):
                unreduced = by_height_mode[(height, "unreduced")]
                reduced = by_height_mode[(height, "reduced")]
                max_unreduced = max(int(row["rigid_contact_count"]) for row in unreduced)
                max_reduced = max(int(row["rigid_contact_count"]) for row in reduced)
                self.assertLess(max_reduced, max_unreduced)


if __name__ == "__main__":
    unittest.main(verbosity=2)
