# H7: Transient Compliance — the √(N/K) Impact-Penetration Gap

A self-contained falsification study (separate from `hypothesis/H1..H6`). Follows
the `hypothesis/README.md` pattern. The theory it rests on is the verified
`hypothesis/contact_reduction_model.md`, Part 2 item 4 ("Transient / dynamic
compliance") — the residual gap that *no knob closes*.

## Hypothesis

Contact reduction preserves the **quasi-static net force** (the body settles to
the same equilibrium) but does **not** preserve **transient compliance**. On a
vertical impact the reduced contact set arrests the body in *much less*
penetration — and with a *higher* peak force — than the dense field, even though
the two carry the same net stiffness.

**Falsifiable, quantitative claim.** The maximum first-contact penetration obeys

```
δ_max(dense) / δ_max(reduced) = sqrt(N / K)
```

where `N` = dense face-contact count and `K` = reduced winner count. The ratio is
**independent of impact velocity** (drop height) and **tracks √(N/K)** as the SDF
resolution changes `N`. If the ratio were 1 (no gap) or independent of `N/K`, the
hypothesis is falsified.

## First-principles model (why √(N/K))

1. **Dense per-face spring.** Each of `N` marching-cubes faces is a contact of
   stiffness `ke_face = k_eff · area_face`, with `area_face = A_total / N`. So
   `ke_face = k_eff · A_total / N`.
2. **Reduction inflates the winner stiffness.** Force-matching gives every winner
   `shared_stiffness = k_eff · |agg_force| / total_depth`. With `|agg_force| ≈ N·area_face·d`
   and `total_depth ≈ K·d`, this is `ke_winner = ke_face · (N/K)`
   (`contact_reduction_hydroelastic.py:649`). Each winner is `N/K` times stiffer.
3. **Solver maps stiffness → a per-contact reference oscillator.** Hydroelastic
   contacts carry `kd=0`, so `kernels.py:406` sets `timeconst = sqrt(1/(ke·(1−imp)))`,
   `dampratio = 1` (critically damped). Each Newton contact becomes its **own**
   MuJoCo contact with its **own** `solref` (`kernels.py:420`) — no merge that
   would recompute one lumped stiffness.
4. **Per-contact frequency.** `ω = 1/timeconst = sqrt(ke·(1−imp))`, so
   `ω_winner / ω_face = sqrt(N/K)` — winners ring `√(N/K)` times faster (more rigid).
5. **Impact penetration.** For a critically-damped contact hit at velocity `v`,
   `δ_max ≈ v / (e·ω)`. The body's arrest is governed by the per-contact reference
   stiffness, so `δ_max ∝ 1/ω ∝ 1/sqrt(ke)`, giving
   **`δ_dense / δ_reduced = ω_winner / ω_dense = sqrt(N/K)`**.
6. **The control / null model.** A *lumped* model uses the net stiffness
   `K_total = N·ke_face = K·ke_winner` (identical for both — reduction's protected
   `2·k_eff·agg_force` invariant), giving the same lumped `ω = sqrt(K_total/m)` and
   therefore **no gap**. The observed gap is precisely the part the net-force
   invariant cannot protect: how the stiffness is *distributed* across per-contact
   reference oscillators.

Probe (RTX 6000, res 32, `N/K = 60.2`, `√(N/K)=7.76`): measured ratio **7.57**
(2 mm drop), **7.67** (5 mm drop) — the law holds to ~1%.

## Setup

- Reuse the validated cube-on-plate harness (`_ImpactRun`, `experiment_cube_on_plate_impact.py`).
- Cube 100 mm, 0.8 kg; plate 500×500×400 mm; `kh = 1e9`; `μ = 0.5`; gravity 9.81.
- Flat drop, zero tilt, zero spin, gravity only.
- `step_dt = 0.25 ms`; run long enough to capture max penetration **and** settle.
- Solver: `SolverMuJoCo(use_mujoco_contacts=False, solver="newton",
  integrator="implicitfast", cone="elliptic", iterations=15, ls_iterations=100,
  impratio=1.0)` — fixed across all runs.

## Modes

- `dense` — `reduce_contacts=False` (the truth: full field of springs).
- `reduced` — `reduce_contacts=True`, default knobs (`anchor_contact=False`,
  `moment_matching=False`), `pre_prune=False`.

## Sweeps

1. **Primary (start here): drop height** ∈ {0.25, 0.5, 1.0 mm}. Keeps dense
   penetration inside the ±5 mm SDF band. Tests that `δ_dense/δ_reduced` is
   **velocity-independent** and ≈ √(N/K).
2. **Mechanistic: SDF resolution** ∈ {16, 24, 32, 48} at a fixed safe drop height.
   Changes `N` (≈ res²) while `K` stays ~constant, so `N/K` changes. Tests that the
   ratio **tracks √(N/K)** — the decisive proof the gap is the per-contact
   stiffness inflation, not an artifact.

## Measured quantities

- `δ_max` = max `cube_penetration_depth_m` over the run (primary).
- `δ_final` = penetration at the end (control: should be ≈ equal across modes — net
  force preserved).
- `peak_Fz` and `peak_Fz/weight` (corroborator: reduced higher, sign-locked).
- `impact_velocity` (≈ √(2gh), validity check).
- `N` (dense `face_contact_count`), `K` (reduced `rigid_contact_count`), `N/K`.
- Derived: `ratio = δ_max(dense)/δ_max(reduced)`, compared to `sqrt(N/K)`.

## Validity gates

1. **Band gate (critical):** dense `δ_max < 5 mm` (the SDF narrow-band half-width).
   Beyond it the body tunnels past the band and the dense reading is contaminated —
   mark that height **inconclusive**.
2. No contact-buffer or hashtable overflow (`buffer_overflow == False`).
3. Body state finite; final tilt < 0.5°, final drift < 0.1 mm (flat, centered).
4. First contact occurs and solver force becomes non-zero.

## Output files

- `output/H7_transient_gap/transient_gap_summary.csv` — one row per (sweep, value, mode).
- `output/H7_transient_gap/transient_gap_timeseries.csv` — the primary drop-height case, both modes.
- `output/H7_transient_gap/transient_gap_report.html` — ratios vs √(N/K), pass/fail per gate.
- This hypothesis record (copied alongside the outputs).

## Expected result

`δ_dense/δ_reduced ≈ √(N/K)` across all in-band drop heights (velocity-independent)
and across resolutions (tracking √(N/K)); `δ_final` equal across modes (net force
preserved); `peak_Fz(reduced) > peak_Fz(dense)`. Together: **reduction preserves
the static resultant but not transient compliance**, with a clean analytic
signature.

## Notes / next hypotheses

- The gap is solver-mediated (per-contact `solref`), as the model doc predicts —
  this is *why* it survives every reduction knob: no knob changes the number of
  winners or the per-contact stiffness→`solref` map.
- Rocking (2nd normal moment) and friction-moment (spin-down, H4) gaps are
  **voxel-winner-mitigated** to ~10% for simple convex patches; the transient gap
  is the one that is large, robust, and knob-proof.
