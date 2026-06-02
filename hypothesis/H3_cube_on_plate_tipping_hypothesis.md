# H3: Cube-on-Plate Tipping Contact Reduction

## Hypothesis

Contact reduction preserves the tipping response of the cube-on-plate demo under a ramped top-face force.

This is a cube version of the support-moment question. The test is not just whether the cube eventually tips; it compares reduce off and reduce on in pre-tip pitch, center-of-pressure shift, and tip-onset force while using the same cube and plate geometry as H1.

## Setup

- Dynamic hydroelastic cube on a fixed hydroelastic plate.
- Cube side: 100 mm.
- Cube mass: 0.8 kg.
- Plate dimensions: 500 x 500 x 400 mm.
- Initial cube overlap: 0.2 mm.
- Applied load: horizontal force at the cube top face, ramped in time.
- Tipping friction coefficient: 0.7, so the analytic tip threshold is below the sliding threshold.
- Analytic thresholds:
  - `F_tip = m * g / 2`.
  - `F_slide = mu * m * g`.
- Simulation: 1.15 s, 60 logged frames/s, 4 solver substeps per frame.

## Modes

- reduce off: dense contact reference.
- reduce on: contact reduction enabled.
- pre-prune: off in both modes.

## Measured Quantities

- Solver force and torque on the cube.
- Cube pitch, tilt, penetration depth, and horizontal drift.
- Center-pressure shift from solver wrench: `cop_x = -Ty / Fz`.
- Tip event force at 10 deg tilt.
- Slide event force at 5 mm horizontal motion with tilt below 5 deg.
- Solver contact count, rigid contact count, raw face contact count, and buffer validity flags.

## Validity Gates

- Both modes are valid only if contact buffers do not overflow.
- Body state must remain finite in both modes.
- A clean tipping result requires tip before slide in both modes.
- If sliding occurs first, the result is a mixed sliding/tipping case rather than a tipping-only conclusion.

## Output Files

- Experiment timeseries: `output/H3_cube_on_plate_tipping/tipping_timeseries.csv`
- Run summaries: `output/H3_cube_on_plate_tipping/tipping_summary.csv`
- HTML report: `output/H3_cube_on_plate_tipping/cube_on_plate_tipping_report.html`
- Hypothesis record copy: `output/H3_cube_on_plate_tipping/H3_cube_on_plate_tipping_hypothesis.md`

## Expected Result

Because the cube contact patch has clear corner and edge support points, reduce on may preserve the qualitative tipping event well. The useful signal is the quantitative difference in pre-tip pitch, center-pressure shift, and tip-onset force, not just whether both modes eventually tip.
