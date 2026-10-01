<!-- SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Sphere contact-gradient test

For [issue #3503](https://github.com/newton-physics/newton/issues/3503).
Compares the original contact springs with an analytical sphere-gradient
correction.
Production code is unchanged from baseline `7fa94b3e700f7cc7f0425e4dfeb33facc7463442`.

## Run

Requires an NVIDIA GPU with CUDA support. From the repository root:

```sh
uv sync --frozen
uv run --frozen python experiments/hydro_sphere_gradient/run.py
```

The script checks the results, prints `PASS`, and updates [results.json](results.json).
Use `--output /tmp/sphere-results.json` to keep the supplied results unchanged.
Do not use Python's `-O` option, which disables assertions.

## Setup

Two equal spheres: radius 50 mm, `kh = 1e6 Pa/m`, overlap 20 mm,
target voxel size 0.5 mm. The lower sphere is fixed; the upper has mass 1 kg.
Friction, damping, and contact reduction are disabled; the full patch is used.

The contact surface is flat, but the radial SDF gradients become oblique to
its normal, reaching 36.87 degrees at the rim. The correction changes stiffness
and witness separation together, preserving the current force and midpoint.
It uses exact sphere gradients, not texture-derived gradients.

## Results

| Formulation | Normal stiffness |
|---|---:|
| Exact analytical reference | 1256.64 N/m |
| Original | 1409.18 N/m |
| Projected correction | 1253.00 N/m |

The original stiffness is 12.14% too high; the corrected error is -0.29%.
Both give the same initial force, approximately 13.61110 N.

For motion, the upper sphere starts with an upward velocity of 0.02 m/s under
a constant balancing preload. `SolverSemiImplicit` runs for 0.6 s at a 0.5 ms
timestep. The reference uses the exact sphere force law and the same integrator.
Errors below are RMS displacement error divided by the reference amplitude:

| Refresh contacts | Original | Corrected |
|---|---:|---:|
| Every step | About 0.43% | About 0.43% |
| Every 8 steps | About 3.34% | About 0.43% |
| Every 16 steps | About 6.79% | About 0.46% |

**The motion improvement appears when contacts are reused, not with every-step
refresh.** Sixteen steps is only a test setting. The [code](run.py) contains
the analytical reference and checks; [results.json](results.json) has the
recorded values and run setup. Last digits can vary between GPU runs.

This is a hydroelastic pressure-field diagnostic, not a Hertz-contact model.
It does not validate contact reduction or friction.
