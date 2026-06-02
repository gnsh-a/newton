# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import csv
import tempfile
import unittest
from pathlib import Path

from newton.examples.contacts.experiment_cube_on_plate_impact import (
    FINAL_TILT_LIMIT_DEG,
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


class TestCubeOnPlateImpactExperiment(unittest.TestCase):
    def test_cube_on_plate_impact_experiment(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            try:
                run_experiment(
                    output_dir=output_dir,
                    heights=(0.001, 0.0025),
                    simulation_time=0.04,
                    step_dt=0.001,
                    verbose=False,
                )
            except ImportError as exc:
                self.skipTest(f"MuJoCo is not available: {exc}")

            timeseries_path = output_dir / TIMESERIES_CSV
            summary_path = output_dir / SUMMARY_CSV
            self.assertTrue(timeseries_path.exists())
            self.assertTrue(summary_path.exists())

            rows = _read_csv(timeseries_path)
            self.assertGreater(len(rows), 0)
            self.assertTrue(set(TIMESERIES_COLUMNS).issubset(rows[0].keys()))
            by_height_mode: dict[tuple[float, str], list[dict[str, str]]] = {}
            for row in rows:
                by_height_mode.setdefault((float(row["height_m"]), row["mode"]), []).append(row)

            self.assertEqual(
                set(by_height_mode),
                {
                    (0.001, "unreduced"),
                    (0.001, "reduced"),
                    (0.0025, "unreduced"),
                    (0.0025, "reduced"),
                },
            )
            for mode_rows in by_height_mode.values():
                self.assertGreater(max(float(row["solver_fz_N"]) for row in mode_rows), 0.0)
                self.assertGreater(max(float(row["cube_penetration_depth_m"]) for row in mode_rows), 0.0)
                self.assertFalse(any(_as_bool(row["buffer_overflow"]) for row in mode_rows))
                self.assertFalse(any(_as_bool(row["state_invalid"]) for row in mode_rows))

            for height in (0.001, 0.0025):
                unreduced = by_height_mode[(height, "unreduced")]
                reduced = by_height_mode[(height, "reduced")]
                self.assertLess(
                    max(int(row["rigid_contact_count"]) for row in reduced),
                    max(int(row["rigid_contact_count"]) for row in unreduced),
                )

            summary_rows = _read_csv(summary_path)
            self.assertEqual(len(summary_rows), 4)
            self.assertTrue(set(SUMMARY_COLUMNS).issubset(summary_rows[0].keys()))
            for row in summary_rows:
                self.assertFalse(_as_bool(row["buffer_overflow"]))
                self.assertFalse(_as_bool(row["state_invalid"]))
                self.assertTrue(_as_bool(row["valid_run"]))
                self.assertGreater(float(row["peak_solver_fz_N"]), 0.0)
                self.assertGreater(float(row["max_penetration_depth_m"]), 0.0)
                self.assertLess(float(row["final_tilt_deg"]), FINAL_TILT_LIMIT_DEG)
                self.assertGreater(float(row["mean_solver_force_count"]), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
