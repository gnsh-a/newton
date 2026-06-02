# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import csv
import tempfile
import unittest
from pathlib import Path

from newton.examples.contacts.experiment_transient_gap import (
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


class TestTransientGapExperiment(unittest.TestCase):
    def test_transient_gap_experiment(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            try:
                run_experiment(
                    output_dir=output_dir,
                    heights=(0.0005,),
                    run_seconds=0.05,
                    step_dt=0.00025,
                    sweeps=("height",),
                    verbose=False,
                )
            except ImportError as exc:
                self.skipTest(f"MuJoCo is not available: {exc}")

            timeseries_path = output_dir / TIMESERIES_CSV
            summary_path = output_dir / SUMMARY_CSV
            self.assertTrue(timeseries_path.exists())
            self.assertTrue(summary_path.exists())

            ts_rows = _read_csv(timeseries_path)
            self.assertGreater(len(ts_rows), 0)
            self.assertTrue(set(TIMESERIES_COLUMNS).issubset(ts_rows[0].keys()))

            summary_rows = _read_csv(summary_path)
            self.assertEqual(len(summary_rows), 2)
            self.assertTrue(set(SUMMARY_COLUMNS).issubset(summary_rows[0].keys()))

            by_mode = {row["mode"]: row for row in summary_rows}
            self.assertEqual(set(by_mode), {"unreduced", "reduced"})
            for row in summary_rows:
                self.assertFalse(_as_bool(row["buffer_overflow"]))
                self.assertFalse(_as_bool(row["state_invalid"]))

            self.assertLess(
                int(by_mode["reduced"]["rigid_count_K"]),
                int(by_mode["unreduced"]["rigid_count_K"]),
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
