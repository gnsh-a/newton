# Drake vs Newton hydroelastic convergence comparison

Frozen sphere-on-box contact, refined per engine and compared on the same scene.
Outputs: `out/drake_convergence/` and `out/newton_convergence/` (meshes + `manifest.json` + figure).
View with `convergence/converge_view.py --engine drake|newton` (or `drake newton` for both).

## Scene (shared `scene.yaml`)

1. Sphere R = 0.05 m, box 0.2³ m, penetration x = 0.005 m.
2. Stiffness matched by `kh = E/H`: `kh_sphere = 2e8`, `kh_box = 1e9`, `kappa = 1.667e8`.
3. Winkler reference: `Fn = 654.5 N`, `p0 = 8.33e5 Pa`, `patch_radius = 22.36 mm`.

## Parity

1. Same scene, geometry, poses, stiffness, reference, metrics, and median-edge x-axis.
2. Only the refinement knob differs: Drake `resolution_hint` (tet level) vs Newton `sdf_target_voxel_size`.
3. Box is inert/pinned in both — Drake meshes it with a fixed medial-axis tet set that ignores
   `resolution_hint`; Newton pins it at 6 mm. Both studies are effectively sphere-only refinement.
4. Newton is the current **unpatched** code (iso-voxel cull bug present at fine resolution).

## Drake (8 tet levels)

| level | edge mm | tris |
|---|---|---|
| L1 | 6.39 | 24 |
| L2 | 6.24 | 80 |
| L3 | 5.08 | 152 |
| L4 | 3.19 | 504 |
| L5 | 1.72 | 1,728 |
| L6 | 0.87 | 6,360 |
| L7 | 0.44 | 24,120 |
| L8 | 0.22 | 94,032 |

Hole-free at every level (analytic linear field on a fitted tet mesh).

## Newton (8 voxel levels, box pinned 6 mm)

Run: `convergence/newton_converge_dump.py --voxels 3.0 1.4 0.75 0.7 0.55 0.36 0.25 0.18 --rigid-contact-max 1000000`

| voxel mm | edge mm | tris | holes | Fn/ref | note |
|---|---|---|---|---|---|
| 3.0 | 2.51 | 646 | 0 | 0.877 | ≈ Drake L4 |
| 1.4 | 1.39 | 1,596 | 0 | 0.871 | ≈ Drake L5 |
| 0.75 | 0.74 | 5,196 | 1 (big ring) | 0.775 | big hole, patch splits |
| 0.7 | 0.70 | 6,632 | 0 | 0.868 | ≈ Drake L6 |
| 0.55 | 0.54 | 10,878 | 0 | 0.868 | ~10k point |
| 0.36 | 0.36 | 24,618 | 10 | 0.867 | ≈ Drake L7 |
| 0.25 | 0.25 | 49,649 | 42 | 0.867 | ~50k point |
| 0.18 | 0.18 | 96,750 | 102 | 0.867 | ≈ Drake L8 |

## Triangle-count ↔ edge law

1. Both follow `N ∝ edge⁻²` (2-D patch); fitted slopes −1.96 (Drake), −1.91 (Newton).
2. `N·edge² ≈ C`: Drake C ≈ 4,900 mm², Newton C ≈ 3,350 mm² (same patch, so C is pure mesh density).
3. Density ratio Drake/Newton ≈ 1.46. Count-match rule: `newton_voxel ≈ 0.83 · drake_edge`.

## Count match (Drake L4–L8 vs Newton)

| Drake | Dtris | Newton voxel | Ntris | match |
|---|---|---|---|---|
| L4 | 504 | 3.0 mm | 646 | 128% |
| L5 | 1,728 | 1.4 mm | 1,596 | 92% |
| L6 | 6,360 | 0.7 mm | 6,632 | 104% |
| L7 | 24,120 | 0.36 mm | 24,618 | 102% |
| L8 | 94,032 | 0.18 mm | 96,750 | 103% |

1. L5–L8 match within ±8%; L4 is the closest reachable (Newton's ~208–646 tri grid floor).
2. Drake L1–L3 (24/80/152 tris) are below Newton's floor — unmatchable.

## Holes (Newton, unpatched cull)

1. Hole-free: all levels ≥ 1 mm voxel, plus 0.7 and 0.55 mm.
2. One big ring hole at 0.75 mm — splits the patch into 2 pieces, Fn drops to 0.775.
3. Many small pinholes below ~0.4 mm (10 → 42 → 102), growing with resolution; Fn stays ~0.867
   because each pinhole is tiny. Highly sensitive to exact voxel (0.18 vs 0.185 flips mild↔severe).
4. The cull patch (separate worktree) removes all of these; Drake has none by construction.

## Convergence result (same scene)

1. Drake → Fn/ref ≈ 0.940 (true Fn ~6% below Winkler at x/R = 0.1).
2. Newton → Fn/ref ≈ 0.868 (clean levels). ~7% below Drake — the open Fn-plateau gap.
3. p0 and equivalent radius agree to ~1% between engines; the gap is in total normal force.

## Drake gaps (cannot match)

Drake tet count ~quadruples per level, so 10k (L6→L7: 6,360→24,120) and 50k (L7→L8: 24,120→94,032)
fall in unreachable gaps. The 10k and 50k points are Newton-only.

## Reproduce

```bash
cd ~/work/newton-sap/scripts/hydro_compare
# Drake (in this uv env, pydrake)
uv run python experiments/convergence/drake_converge_dump.py
# Newton (run from repo root in the newton-sap uv env, needs CUDA)
cd ~/work/newton-sap
uv run --no-sync python scripts/hydro_compare/experiments/convergence/newton_converge_dump.py \
    --voxels 3.0 1.4 0.75 0.7 0.55 0.36 0.25 0.18 --rigid-contact-max 1000000
# visualize
cd ~/work/newton-sap/scripts/hydro_compare
uv run python experiments/convergence/converge_view.py --engine drake newton
```
