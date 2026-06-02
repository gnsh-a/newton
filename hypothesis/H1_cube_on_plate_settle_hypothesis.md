# H1: Cube-on-Plate Settle Contact Reduction

## Hypothesis

Contact reduction preserves the settled rigid-body response of a dense hydroelastic cube-on-plate contact while using far fewer solver contact entries.

The reduced run should match the dense reference in vertical support, lateral residual, torque residual, settled geometry, drift, and tilt.

## Setup

- Dynamic hydroelastic cube on a fixed hydroelastic plate.
- Cube side: 100 mm.
- Cube mass: 0.8 kg.
- Plate dimensions: 500 x 500 x 400 mm.
- Gravity: 9.81 m/s^2.
- Drop heights: 0, 0.25, 0.5, 1, 2.5, and 5 mm.
- Simulation: 1 s, 240 logged frames/s, 8 solver substeps per frame.

## Modes

- reduce off: dense contact reference.
- reduce on: contact reduction enabled.
- pre-prune: off in both modes.

## Measured Quantities

- Solver force and torque on the cube.
- Solver contact count and rigid contact count.
- Cube position, final drift, and final tilt.
- Cube-plate gap and penetration depth.
- Support offset from `sqrt(Tx^2 + Ty^2) / |Fz|`.
- Contact-buffer max counts and capacities.

## Validity Gates

- Reduce off is valid only if hydro broadphase, iso, face-contact, and rigid-contact buffers do not overflow.
- Reduce on is valid only if rigid-contact output does not overflow and reduction hashtable insertions do not fail.
- If either gate fails, the result is inconclusive rather than pass/fail.

## Output Files

- Experiment CSVs: `output/H1_cube_on_plate_settle/settle_timeseries.csv`
- Run summaries: `output/H1_cube_on_plate_settle/settle_summary.csv`
- Optional buffer debug CSV: `output/H1_cube_on_plate_settle/settle_debug_buffers.csv`
- HTML report: `output/H1_cube_on_plate_settle/cube_on_plate_settle_report.html`
- Hypothesis record copy: `output/H1_cube_on_plate_settle/H1_cube_on_plate_settle_hypothesis.md`

## Current Result

Passed for the current sweep.

- Reduce-off contact buffers valid: true.
- Reduce-on reduction buffers valid: true.
- Max reduce-off buffer utilization: 87.5%.
- Max reduce-on settled vertical support error: 0.002834% of weight.
- Max reduce-on settled sideways leakage: 0.0185% of weight.
- Max reduce-on settled torque imbalance: 0.009249% of `m*g*L`.
- Max reduce-on penetration: 0.001488 mm.
- Max reduce-on support offset: 0.02195 mm.
- Median reduce-on/off force-count ratio: 0.01611.

The evidence supports H1 for this settle sweep: contact reduction cuts solver contact entries from roughly 2046 to roughly 33 while preserving the measured settled response within practical scales.
