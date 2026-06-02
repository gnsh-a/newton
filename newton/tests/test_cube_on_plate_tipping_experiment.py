# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import csv
import tempfile
import unittest
from pathlib import Path

from newton.examples.contacts.experiment_cube_on_plate_tipping import (
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


class TestCubeOnPlateTippingExperiment(unittest.TestCase):
    def test_cube_on_plate_tipping_experiment(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            try:
                run_experiment(
                    output_dir=output_dir,
                    simulation_time=0.2,
                    frame_fps=30,
                    sim_substeps=2,
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
            by_mode: dict[str, list[dict[str, str]]] = {}
            for row in rows:
                by_mode.setdefault(row["mode"], []).append(row)

            self.assertEqual(set(by_mode), {"unreduced", "reduced"})
            for mode_rows in by_mode.values():
                self.assertGreater(max(int(row["solver_force_count"]) for row in mode_rows), 0)
                self.assertFalse(any(_as_bool(row["buffer_overflow"]) for row in mode_rows))
                self.assertFalse(any(_as_bool(row["state_invalid"]) for row in mode_rows))
                self.assertGreater(float(mode_rows[-1]["applied_force_N"]), 0.0)

            unreduced_contacts = max(int(row["rigid_contact_count"]) for row in by_mode["unreduced"])
            reduced_contacts = max(int(row["rigid_contact_count"]) for row in by_mode["reduced"])
            self.assertLess(reduced_contacts, unreduced_contacts)

            summary_rows = _read_csv(summary_path)
            self.assertEqual(len(summary_rows), 2)
            self.assertTrue(set(SUMMARY_COLUMNS).issubset(summary_rows[0].keys()))
            for row in summary_rows:
                self.assertFalse(_as_bool(row["buffer_overflow"]))
                self.assertFalse(_as_bool(row["state_invalid"]))
                self.assertGreater(float(row["mean_solver_force_count"]), 0.0)
                self.assertEqual(row["mu_sliding"], "0.7")


if __name__ == "__main__":
    unittest.main(verbosity=2)
