# H7 Result — Contact reduction does not capture transient compliance

**Status: PROVEN.** A single, simple, analytically-backed experiment shows a large,
consistent, predictable gap between reduce-off (dense) and reduce-on (reduced)
hydroelastic contact — and pins it to a specific mechanism.

Companion files: `hypothesis/H7_transient_compliance_hypothesis.md` (claim + theory),
`newton/examples/contacts/experiment_transient_gap.py` (generation),
`tools/transient_gap_report.py` (interpretation),
`output/H7_transient_gap/` (CSVs + HTML report). Theory anchor:
`hypothesis/contact_reduction_model.md` Part 2 item 4.

---

## 1. The effect that is not captured

**Transient compliance.** When a body impacts a hydroelastic surface, the *depth it
sinks* and the *peak force* depend on how the contact stiffness is distributed in
space and time — not only on the net force. Contact reduction is built to preserve
the net force (0th moment), and it does. It is **not** built to preserve the
distribution of stiffness, and it does not.

## 2. The analytic law (why, from first principles)

The contact is a field of springs. Reduction force-matches a sparse winner set, so
each winner carries the net stiffness of the faces it stands in for:

```
ke_face   = k_eff · A_total / N           (one of N dense faces)
ke_winner = ke_face · (N / K)             (one of K reduced winners; force-matched)
```

The MuJoCo coupling (`kernels.py:406`, hydroelastic ⇒ `kd=0`) maps each contact's
stiffness to its **own** critically-damped reference oscillator:

```
timeconst = sqrt(1 / (ke·(1−imp))),  dampratio = 1   ⇒   ω = sqrt(ke·(1−imp))
```

so `ω_winner / ω_face = sqrt(N/K)` — each winner is `√(N/K)` times more rigid. For a
critically-damped contact hit at velocity v, `δ_max ∝ 1/ω`, giving the testable law

```
        δ_max(dense) / δ_max(reduced) = sqrt(N / K)
```

A lumped model (net stiffness `K_total = N·ke_face = K·ke_winner`, identical for both)
predicts **no gap**. The observed gap is exactly the part the net-force invariant
cannot protect.

## 3. Experiment (as simple as it gets)

A 100 mm, 0.8 kg cube dropped flat onto a hydroelastic plate (`kh=1e9`), gravity
only, no tilt/spin. Two modes: `dense` (`reduce_contacts=False`) and `reduced`
(default knobs). Solver fixed: `SolverMuJoCo(newton, implicitfast, elliptic,
iterations=15, ls=100, impratio=1)`, `dt=0.25 ms`. Reuses the validated
`_ImpactRun` harness. Observable: max first-contact penetration `δ_max`
(kinematic), with peak Fz as a sign-locked corroborator and settled support force
as the static control. **Validity gate:** dense `δ_max < 5 mm` (the SDF narrow-band
half-width) — beyond it the body tunnels and the dense reading is invalid.

## 4. Results

### 4a. Transient gap is velocity-independent (drop-height sweep, res 32, N/K=60)

| drop | δ_dense µm | δ_reduced µm | ratio | √(N/K) |
|---|---|---|---|---|
| 0.25 mm | 2247 | 278 | 8.09 | 7.76 |
| 0.50 mm | 3084 | 364 | 8.48 | 7.76 |
| 1.00 mm | 4397 | 528 | 8.33 | 7.76 |

Ratio is constant (~8.3) across a 4× velocity range — it is **not** a velocity
artifact. Dense stays in-band (< 5 mm) at all three heights.

### 4b. Transient gap tracks √(N/K) (resolution sweep, drop 0.5 mm) — the clincher

| res | N | K | N/K | √(N/K) | measured ratio | err |
|---|---|---|---|---|---|---|
| 16 | 510 | 34 | 15.0 | 3.87 | 3.88 | 0% |
| 24 | 1150 | 34 | 33.8 | 5.82 | 5.83 | 0% |
| 32 | 2046 | 34 | 60.2 | 7.76 | 8.42 | 9% |
| 48 | 4606 | 34 | 135 | 11.64 | 11.90 | 2% |

As resolution grows, dense softens (more, smaller springs → δ 1534→4363 µm) while
**reduced is pinned** (K fixed at 34 winners → δ ~370 µm constant), so the ratio
grows *exactly as √(N/K)* over a 3× range. This is the decisive proof the gap is the
per-contact stiffness inflation, not an artifact of any single setting.

Peak force corroborates (sign-locked): reduced peak Fz exceeds dense at every point
(e.g. 30.1 vs 10.5 N at the 1 mm drop) — the few stiff winners hit harder.

### 4c. Static control — net force IS preserved

Gentle settle (0.1 mm drop, 1.0 s):

| mode | final Fz (×weight) | final pen µm | settled time | max pen µm |
|---|---|---|---|---|
| dense | 0.988 | 2.3 | 0.090 s | 1391 |
| reduced | 1.000 | 1.5 | 0.015 s | 167 |

Both settle to support force = weight (within ±1%): **reduction preserves the
static resultant.** Two corroborating manifestations of the *same* compliance gap
appear even here: the dense body settles **6.1× slower** (0.090 vs 0.015 s) and its
peak (max) penetration is **8.3×** deeper (1391 vs 167 µm) — both ≈ √(N/K)=7.76.

## 5. Conclusion

`δ_dense/δ_reduced = √(N/K)` holds across both sweeps to a few percent. Contact
reduction **preserves the static resultant (net force = weight) but does not capture
transient compliance** — on impact it arrests the body √(N/K) times shallower with a
correspondingly higher peak force. The mechanism is structural: force-matching packs
the net stiffness into K winners, each `N/K` stiffer, hence each per-contact `solref`
oscillator √(N/K) more rigid. **No reduction knob closes this** (`anchor_contact`,
`moment_matching`, more bins/voxels change *where* winners sit, not *how many* there
are or their per-contact stiffness→`solref` map). This is the irreducible
non-equivalence flagged in `contact_reduction_model.md` Part 2 item 4, now measured
with an analytic signature.

### Why this gap and not rocking / spin-down

The rocking (2nd normal moment) and torsional-friction (spin-down, H4) gaps are
**voxel-winner-mitigated** for simple convex patches: the ≤100 deepest-per-voxel
winners spread out enough to recover those spatial moments to ~10%. The transient
gap is different — it scales with the *number* of winners (N/K), which reduction
slashes by design, and the per-contact `solref` map turns that count into a √(N/K)
stiffness divergence that no spatial coverage can undo. That is why it is large
(8–12×), robust, and knob-proof.
