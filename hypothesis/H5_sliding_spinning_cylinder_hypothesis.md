# H5: Sliding-Spinning Cylinder Contact Reduction

## Hypothesis

Contact reduction preserves the coupled sliding and yaw-spinning friction response of a flat cylinder on a flat hydroelastic plate.

This is stronger than H2 and H4 separately: the solver must get both the net horizontal friction force and the yaw-friction torque from the same reduced contact set. Reduce on should match reduce off in the evolution of `epsilon = v / (R * |omega_z|)`, stop times, solver force, and solver yaw torque while using fewer solver contacts.

## Setup

- Dynamic hydroelastic cylinder on a fixed hydroelastic plate.
- Cylinder radius: 50 mm.
- Cylinder thickness: 12.5 mm.
- Plate dimensions: 400 x 400 x 20 mm.
- Sliding friction coefficient: 0.2.
- Torsional friction: 0.
- Rolling friction: 0.
- Initial yaw rate: 10 rad/s.
- Initial sliding speeds are selected through `epsilon0 = v0 / (R * omega0)`.
- Swept `epsilon0`: 0.25, 0.5, 1.0, and 2.0.
- Experiment SDF max resolution: 32.
- Simulation: 0.8 s, 120 logged frames/s, 4 solver substeps per frame.

## Modes

- reduce off: dense contact reference.
- reduce on: contact reduction enabled.
- pre-prune: off in both modes.

## Measured Quantities

- Solver force and torque on the cylinder.
- Horizontal speed `v`, yaw rate `omega_z`, and `epsilon = v / (R * |omega_z|)`.
- Translation stop time, spin stop time, and coupled stop time.
- Integrated solver horizontal impulse and yaw impulse.
- Final travel, lateral drift, tilt, and penetration depth.
- Solver contact count, rigid contact count, and raw face contact count.
- Contact-buffer overflow validity flags.

## Validity Gates

- Reduce off is valid only if hydro broadphase, iso, face-contact, and rigid-contact buffers do not overflow.
- Reduce on is valid only if rigid-contact output does not overflow and reduction hashtable insertions do not fail.
- If either gate fails, the result is inconclusive rather than pass/fail.

## Output Files

- Experiment timeseries: `output/H5_sliding_spinning_cylinder/sliding_spinning_timeseries.csv`
- Run summaries: `output/H5_sliding_spinning_cylinder/sliding_spinning_summary.csv`
- Hypothesis record copy: `output/H5_sliding_spinning_cylinder/H5_sliding_spinning_cylinder_hypothesis.md`

## Expected Result

If contact reduction preserves the contact-patch friction distribution, reduce on and reduce off should give similar `epsilon(t)`, stop times, horizontal friction impulse, and yaw impulse. A failure would appear as reduce on matching either translation or spin alone but not their coupled evolution.
