# H2: Flat Sliding Block Contact Reduction

## Hypothesis

Contact reduction preserves the basic Coulomb sliding response of a flat cube sliding on a flat plate.

For flat-on-flat sliding with no spin and no applied force, reduce on should match reduce off in stopping time, travel distance, horizontal impulse, and settled geometry while using fewer solver contact entries.

## Setup

- Dynamic hydroelastic cube on a fixed hydroelastic plate.
- Cube side: 100 mm.
- Cube mass: 0.8 kg.
- Plate dimensions: 500 x 500 x 400 mm.
- Sliding friction coefficient: 0.5.
- Initial horizontal speeds: 0.05, 0.1, 0.2, and 0.4 m/s.
- No initial spin.
- No applied external force.
- Simulation: 0.25 s, 120 logged frames/s, 4 solver substeps per frame.

## Modes

- reduce off: dense contact reference.
- reduce on: contact reduction enabled.
- pre-prune: off in both modes.

## Measured And Reference Quantities

- Solver force and torque on the cube.
- Horizontal velocity and stopping time.
- Travel distance in the sliding direction.
- Horizontal solver impulse.
- Final lateral drift, final tilt, and penetration depth.
- Solver contact count and rigid contact count.
- Contact-buffer max counts and capacities.
- Coulomb reference stop time and travel from `v0`, `mu`, and `g`.

## Validity Gates

- Reduce off is valid only if hydro broadphase, iso, face-contact, and rigid-contact buffers do not overflow.
- Reduce on is valid only if rigid-contact output does not overflow and reduction hashtable insertions do not fail.
- If either gate fails, the result is inconclusive rather than pass/fail.

## Output Files

- Experiment CSVs: `output/H2_flat_sliding_block/sliding_timeseries.csv`
- Run summaries: `output/H2_flat_sliding_block/sliding_summary.csv`
- HTML report: `output/H2_flat_sliding_block/flat_sliding_block_report.html`
- Hypothesis record copy: `output/H2_flat_sliding_block/H2_flat_sliding_block_hypothesis.md`

## Viewer

- Reduce on: `python -m newton.examples flat_sliding_block`
- Reduce off: `python -m newton.examples flat_sliding_block_no_reduce`

## Current Result

Full sweep completed with valid buffers for both modes.

- Dense/reduce-off produced 2046 solver contacts in every run.
- Reduce-on produced 26-32 mean solver contacts, with a max rigid contact count of 34 and no reduction hashtable failures.
- Reduce-on did not match reduce-off stop time or travel. Stop time was 54-83% shorter, and stop travel was 20-85% shorter.
- Compared with the analytic Coulomb stop distance, reduce-on was close for all tested speeds: 0.000277 vs 0.000255 m, 0.001032 vs 0.001019 m, 0.003964 vs 0.004077 m, and 0.015986 vs 0.016310 m.
- Reduce-off traveled farther than the Coulomb reference in all cases.
- Final lateral drift and tilt stayed tiny for reduce-on, so this test does not expose a reduced-contact instability.

Conclusion: the dense-reference equivalence hypothesis fails, but this simple flat sliding case does not show a reduce-on flaw. It suggests the dense unreduced contact set may be a poor physical reference for Coulomb sliding in this setup.

## SDF Resolution Observation

Increasing `SDF_MAX_RESOLUTION` increases the raw hydroelastic face-contact count in reduce-off mode. In the resolution sweep, reduce-off stop distance moved farther from the Coulomb reference as SDF resolution increased, while reduce-on stayed close to the reference over the same runs. At higher velocities and higher SDF resolutions, the short `0.25 s` runs did not always stop in reduce-off mode; longer `0.5 s` runs were needed to recover stop distances.

## Solver Time Constant Note

The MuJoCo `solref` time constant should be small relative to the physical stop time being measured. This is especially important at low initial speeds:

- `v0 = 0.05 m/s`: Coulomb stop time is about 0.010 s.
- `v0 = 0.10 m/s`: Coulomb stop time is about 0.020 s.

For these low-speed cases, a practical target is a contact time constant around `0.001-0.005 s`. A value around `0.01 s` is already comparable to the full stopping event, and values around `0.1 s` are too slow for Coulomb-like stop-distance measurements.

The direct knob to test is hydroelastic stiffness `KH`. Because contact time constant scales roughly as `1 / sqrt(contact stiffness)`, increasing `KH` reduces the time constant:

- `10x KH` gives about `3.2x` smaller time constant.
- `100x KH` gives about `10x` smaller time constant.

Suggested low-speed follow-up sweep:

- `KH = 1e9, 1e10, 5e10`.
- `SDF_MAX_RESOLUTION = 32` and `48`.
- `v0 = 0.05` and `0.10 m/s`.
- Run both reduce off and reduce on.

Check stop distance, penetration depth, contact count, and whether the solution develops jitter or instability. For reduce on, `KH = 1e10-2e10` is likely a reasonable first range; for high-SDF reduce off, much larger stiffness may be needed because each raw face contact has very small area and therefore very small per-contact stiffness.
