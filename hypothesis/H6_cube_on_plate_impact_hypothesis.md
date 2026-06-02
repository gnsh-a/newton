# H6: Cube-on-Plate Impact Ring-Down Contact Reduction

## Hypothesis

Contact reduction preserves static support, but may not preserve vertical impact and ring-down dynamics of the cube-on-plate contact.

The reduced contact set should be compared with the dense reference in peak solver force, maximum compression, rebound velocity, and settling time after a flat cube is dropped from rest onto the plate.

## First-Principles Model

During impact, the cube's vertical motion is governed by the contact force history:

- `m * z_ddot = Fz_contact - m * g`
- Dense hydroelastic contact is a distributed field of many small springs across the cube footprint.
- Reduced contact replaces that field with fewer solver contacts carrying redistributed stiffness.
- Static support can still be correct because the net vertical force is preserved.
- Transient response can differ because peak load, damping, compression, and rebound depend on how stiffness is distributed through time, not only on the final net force.

This is the first hypothesis in the sequence that directly targets dynamic compliance rather than static resultant force or quasi-static support moment.

## Setup

- Dynamic hydroelastic cube on a fixed hydroelastic plate.
- Cube side: 100 mm.
- Cube mass: 0.8 kg.
- Plate dimensions: 500 x 500 x 400 mm.
- Gravity: 9.81 m/s^2.
- Drop heights: 1, 2.5, 5, 10, and 20 mm.
- Initial velocity: 0 m/s.
- Initial tilt: 0 deg.
- No applied external force except gravity.
- Runner default simulation: 0.35 s, fixed 0.25 ms solver/log step.
- Current saved sweep: 0.12 s, fixed 1 ms solver/log step, used to keep the generated record practical while preserving the same height and mode sweep.

## Modes

- reduce off: dense contact reference.
- reduce on: contact reduction enabled.
- pre-prune: off in both modes.

## Measured Quantities

- Solver force and torque on the cube.
- Cube vertical position, vertical velocity, penetration depth, drift, and tilt.
- First contact time and impact velocity.
- Peak solver `Fz`, time to peak, and maximum penetration.
- Maximum upward rebound velocity and rebound velocity ratio.
- Settling time and post-settle force RMS.
- Normal solver impulse.
- Solver contact count, rigid contact count, face contact count, and buffer validity flags.

## Validity Gates

- Contact buffers must not overflow.
- Body state must remain finite.
- First contact must occur before the run ends.
- Solver force must become nonzero after contact.
- Final tilt must remain below 0.5 deg.
- Final horizontal drift must remain below 1e-4 m.

If a gate fails, that height/mode is inconclusive rather than pass/fail.

## Output Files

- Experiment timeseries: `output/H6_cube_on_plate_impact/impact_timeseries.csv`
- Run summaries: `output/H6_cube_on_plate_impact/impact_summary.csv`
- HTML report: `output/H6_cube_on_plate_impact/cube_on_plate_impact_report.html`
- Hypothesis record copy: `output/H6_cube_on_plate_impact/H6_cube_on_plate_impact_hypothesis.md`

## Expected Result

The useful signal is whether reduce on changes the impact transient while still settling to a physically reasonable static state. A difference in peak force, maximum penetration, rebound, or ring-down would show that contact reduction preserves the static resultant more reliably than the dynamic compliance distribution.
