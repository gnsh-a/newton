# H4: Spinning Cylinder Yaw-Torque Contact Reduction

## Hypothesis

Contact reduction preserves the yaw-friction torque of a flat spinning cylinder on a flat hydroelastic plate.

With no translational velocity and no torsional or rolling friction, the spin decay must come from sliding-friction forces distributed across the contact patch. Reduce on should match reduce off and the uniform-pressure disk reference in solver yaw torque, spin stop time, and spin decay while using fewer solver contact entries.

## Setup

- Dynamic hydroelastic cylinder on a fixed hydroelastic plate.
- Cylinder radius: 50 mm.
- Cylinder thickness: 12.5 mm.
- Cylinder mass: about 11.7 g.
- Plate dimensions: 400 x 400 x 20 mm.
- Sliding friction coefficient: 0.2.
- Torsional friction: 0.
- Rolling friction: 0.
- Initial yaw rates: 15, 30, and 60 rad/s.
- Initial linear velocity: 0 m/s.
- Experiment SDF max resolution: 32.
- Simulation: 1.35 s, 120 logged frames/s, 4 solver substeps per frame.

## Modes

- reduce off: dense contact reference.
- reduce on: contact reduction enabled.
- pre-prune: off in both modes.

## Measured And Reference Quantities

- Solver force and torque on the cylinder.
- Yaw rate `omega_z` and normalized yaw rate `omega_z / omega0`.
- Spin stop time.
- Integrated solver yaw impulse.
- Final lateral drift, tilt, and penetration depth.
- Solver contact count, rigid contact count, and raw face contact count.
- Contact-buffer overflow validity flags.
- Uniform-pressure disk reference:
  - `Tz = -(2/3) * mu * m * g * R * sign(omega)`.
  - `Izz = (1/2) * m * R^2`.
  - `t_stop = (3/4) * omega0 * R / (mu * g)`.

## Validity Gates

- Reduce off is valid only if hydro broadphase, iso, face-contact, and rigid-contact buffers do not overflow.
- Reduce on is valid only if rigid-contact output does not overflow and reduction hashtable insertions do not fail.
- If either gate fails, the result is inconclusive rather than pass/fail.

## Output Files

- Experiment CSVs: `output/H4_spinning_cylinder_spin_down/spin_down_timeseries.csv`
- Run summaries: `output/H4_spinning_cylinder_spin_down/spin_down_summary.csv`
- Hypothesis record copy: `output/H4_spinning_cylinder_spin_down/H4_spinning_cylinder_spin_down_hypothesis.md`

## Expected Result

This should be a stronger test than straight sliding because the measured response depends on the contact patch's friction moment, not only on net normal support.

Default reduce on may under-predict or add noise to `solver_tz_Nm` if the reduced points do not preserve enough footprint radius. If that happens, spin stop time will be longer than reduce off and farther from the uniform-pressure reference. If reduce on matches, then the default selected contacts are preserving enough patch-radius information for this simple spin case.

## H4b Moment-Matching Observation

For `SDF_MAX_RESOLUTION = 48` and `omega0 = 15 rad/s`, enabling `moment_matching=True` did not reduce the yaw-torque error. It increased mean yaw torque and made spin-down faster than default reduce on. This records the observation only; it does not change the H4 hypothesis.

## Viewer

- Reduce on: `python -m newton.examples spinning_cylinder`
- Reduce off: `python -m newton.examples spinning_cylinder_no_reduce`
