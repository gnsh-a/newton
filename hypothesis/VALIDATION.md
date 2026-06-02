# Validation References for Contact-Reduction Hypotheses

This document defines the analytic / closed-form reference each hypothesis (H1-H6)
should be validated against, and why. It is the reference spec the experiment
scripts and reports should follow.

## Core principle

The hydroelastic "bed of springs" contact model is the **system under test (SUT)**,
not the reference. For most of these regimes the correct ground truth is plain
**rigid-body physics** - parameter-free, exact, and cheaper. The compliant
(spring) model is only *required* where the measured quantity does not exist in
the rigid limit.

- Validate the solver by checking it **reproduces the rigid-body result** wherever
  the response is quasi-static or kinematic (final rest state, stop distance, tip
  threshold, spin-down, terminal slide/spin ratio).
- Carry the spring (compliant) reference only for **H6** (required) and as a
  second-order add-on for **H3** pre-tip and **H4/H5** rim pressure.

## Master validation table

| H | Primary closed-form reference | Spring model | Validate / gate |
|---|---|---|---|
| H1 settle | Rigid static equilibrium: rest at N=mg, drift=tilt=0 | Optional (transient only) | Support = mg; drift ~ 0, tilt ~ 0 |
| H2 slide | Rigid Coulomb (= Farkas eps->inf, F=1): t=v0/(mu*g), d=v0^2/(2*mu*g) | No | Stop time and travel; gate: solref time const << t_stop |
| H3 tip | Rigid onset F_tip=m*g/2, F_slide=mu*m*g; rigid edge-pivot; rigid COP shift cop_x/half_extent=F/F_tip | Pre-tip pitch only: Winkler K_theta=k_eff*I | Tip force, tip-before-slide (mu>0.5); COP tracks F/F_tip to the front edge; pre-tip pitch valid for e<L/6 |
| H4 spin | Farkas eps=0 endpoint: T(0)=2/3 => Tz=(2/3)*mu*m*g*R, t=(3/4)*w0*R/(mu*g), w linear | No | Tz, t_stop, linear w-decay; gate: drift ~ 0 (eps=0 unstable) |
| H5 slide+spin | Farkas curve F(eps), T(eps) (elliptic); attractor eps0 ~ 0.653 | No | Pointwise \|F\|/(mu*m*g) vs F(eps), \|T\|/(mu*m*g*R) vs T(eps); eps->0.653; simultaneous stop; gate: uniform pressure (cylinder drift) |
| H6 impact | Rigid gives only impulse J=m*v*(1+e), v_reb=e*v_imp | Required: damped SDOF | Peak F, depth_max, ring-down, settle; 3 stiffness levels + dt-floor gate; e = lower bound |

## Closed-form appendix

### Farkas sliding-spinning disk (H4, H5)

Friction force `F = -mu*Fn * F(eps) * e_v` and yaw torque `T = -mu*Fn*R * T(eps) * e_w`,
with `eps = v/(R*w)` and `Fn` the normal load. `F(eps)`, `T(eps)` are the dimensionless
force and torque factors for a uniform-pressure disk. Evaluate them by integrating the
Coulomb traction direction over the unit disk (paper Eq. 1) -- this is unambiguous and
matches the paper's Fig. 2:

```
u(x, y) = (eps - y, x)             # local slip direction; e_v = x_hat, spin about +z
F(eps) = (1/pi) * integral_unit_disk (eps - y)        / |u| dA
T(eps) = (1/pi) * integral_unit_disk (x^2 + y^2 - eps*y) / |u| dA

verified anchors: F(0)=0, F(1)=8/(3*pi)~=0.849, F(inf)=1
                  T(0)=2/3, T(1)=8/(9*pi)~=0.283, T(inf)=0

equations of motion:  m*dv/dt = -mu*Fn*F(eps);  (1/2)*m*R^2*dw/dt = -mu*Fn*R*T(eps)
```

NOTE: the paper also prints an elliptic-integral closed form, but the literal expression
does not reproduce these limits under the standard `K(k)`/`E(k)` convention (`F` would
diverge as `eps->0` instead of going to 0). Use the direct integral above.

The ratio is friction- and load-independent: `|T|/(R*|F|) = T(eps)/F(eps)` -- this is the
H5 pointwise check. Stable attractor `eps0 ~ 0.653`; sliding and spinning stop together.

- **H4** is the pure-spin endpoint (`eps = 0`): `F(0)=0` (no induced translation),
  `T(0)=2/3` => `Tz = (2/3)*mu*m*g*R`. The elliptic curves are not needed; only the
  endpoint value. Note `eps=0` is an **unstable** fixed point (`f(eps) ~ eps/4`), so
  any asymmetry kicks the disk off pure spin - this makes "zero lateral drift" a
  real validity gate, not a formality.
- **H5** lives on the interior of the curve and converges to `eps0 ~ 0.653`. Validate
  the **instantaneous** normalized force/torque pointwise against `F(eps)`/`T(eps)`
  along the whole run (no time integration needed for this test), in addition to the
  terminal `eps*` and simultaneous-stop checks.

### Rigid center-of-pressure shift (H3, rigid statics)

Before tipping the cube is in static equilibrium, so the physical center of pressure
(the normal-force offset) grows linearly with the applied force and reaches the front
edge exactly at the tip force:

```
cop_x / half_extent = F / F_tip
```

This is rigid statics, not a compliance effect. Compute it from the solver wrench with
the base friction torque removed: `cop_x = -(Ty + h*Fx) / Fz`. The raw `-Ty/Fz` folds in
the friction torque and reads only half the true offset.

### Winkler torsional spring (H3 pre-tip pitch)

A rigid cube has zero pre-tip pitch; the small pre-tip tilt is a compliance effect. A
rigid flat patch on a Winkler foundation of modulus `k_a` [Pa/m = N/m^3] tilted by small
angle `theta` about its centroidal axis:

```
M = k_a * I_area * theta,   I_area = integral(x^2 dA)   (square side L: I = L^4/12)
K_theta = k_a * I_area,   with k_a = k_eff (effective hydroelastic gradient)
```

Valid only in the small-angle, full-contact (no-uplift) regime, i.e. eccentricity
within the kern (`e < L/6` for a square). Beyond uplift the moment-rotation response
is nonlinear and this over-predicts the moment.

### Damped single-DOF oscillator (H6)

```
omega_n = sqrt(K/m),   zeta = d / (2*sqrt(K*m)),   omega_d = omega_n*sqrt(1-zeta^2)
depth_max = v_imp*sqrt(m/K)            (undamped energy balance)
F_peak    = K*depth_max = v_imp*sqrt(K*m)
settle    ~ 4 / (zeta*omega_n)         (2% envelope)
e (restitution) = exp(-zeta*pi/sqrt(1-zeta^2))   -- LOWER BOUND only
```

The restitution formula is exact only under the half-damped-period contact
assumption, which produces an unphysical tensile force before separation. Newton's
positive-part clamp `f_n = (-k*phi - d*v_n)_+` ends contact when the force vanishes,
so the **realized rebound is larger and velocity-dependent** - treat the formula as a
lower bound.

## H6 stiffness levels + dt gate

There are three distinct stiffnesses; a closed-form is only a fair anchor at the
matching level.

| Level | Value | Check |
|---|---|---|
| L1 ideal | K = g_eff * A = 4e7 N/m | physics target |
| L2 as-fed | dense: sum(area_i * k_eff) = A*k_eff;  reduced: sum(k_eff*\|F\|/depth) | does reduction conserve sum(C)? |
| L3 realized | post-solref (~1e3 dense, ~5e4 reduced; fit from F_peak/depth_max) | vs L1; if timeconst = sqrt(1/ke) is clamped to 2*dt => inconclusive |

**dt-floor validity gate:** if a contact's `timeconst` hits the MuJoCo `2*dt` floor,
that contact is dt-limited, not physics-limited - mark the dynamic (H6) or transient
(H1) result inconclusive. Re-run H6 at a `dt` small enough to resolve the contact
half-period (geometric: `pi*sqrt(m/K) ~ 0.44 ms` for K=4e7; the saved sweep used
`dt = 1 ms`, which cannot resolve it).

## How Newton applies the stiffness (wiring)

Both cube-on-plate scripts construct the solver with `use_mujoco_contacts=False`, so
**MuJoCo native contact detection is OFF**; all contacts come from the hydroelastic
SDF collision pipeline and are fed in as external constraints.

- The per-contact normal stiffness is the **hydroelastic** value
  `c_stiffness = area * k_eff`, where `k_eff = k_a*k_b/(k_a+k_b)` is the series
  combination (the paper's effective gradient `g_eff`). For reduced/aggregated
  contacts it is `k_eff * |agg_force| / total_depth`.
- In the conversion kernel, when `rigid_contact_stiffness > 0` (always true for
  hydroelastic contacts) this **overrides** the geom-level `solref` derived from the
  ShapeConfig `ke`/`kd`. So `ke=1e7`/`kd=1e5` are **fallback only** (non-hydroelastic
  or zero-stiffness contacts). The hydroelastic stiffness governs.
- The contact then becomes a regularized, mass-normalized, implicitly-integrated
  MuJoCo constraint via `solref = (timeconst, dampratio)` with
  `timeconst = sqrt(1/contact_ke)` (floored at `2*dt`). This is why the realized
  effective stiffness (L3) sits far below the ideal lump (L1), and why dense and
  reduced - which compute `c_stiffness` by different formulas - differ dynamically.

Key files: `newton/examples/contacts/example_cube_on_plate.py` (solver config),
`newton/_src/solvers/mujoco/kernels.py` (`convert_newton_contacts_to_mjwarp_kernel`,
the stiffness override), `newton/_src/geometry/contact_reduction_hydroelastic.py`
(`c_stiffness`, `k_eff`).

## Unifying structure

1. **H2 and H4 are the two endpoints of one Farkas curve**: `eps=inf` -> `F=1`
   (pure Coulomb slide, H2); `eps=0` -> `T=2/3` (pure spin, H4). At each end the
   curve collapses to a single constant, so both reduce to rigid-body physics. **H5
   is the curve's interior** and needs the full elliptic `F(eps)`/`T(eps)`.
2. **H1** is rigid statics; **H3** is rigid threshold/dynamics plus a compliance
   micro-term (pre-tip tilt); **H6** is the only experiment that genuinely requires
   the spring model (peak force / compression / ring-down do not exist in the rigid
   limit).
3. The bed-of-springs appears in the *reference* only for H6 (required) and as a
   second-order add-on for H3 pre-tip; everywhere else the anchor is parameter-free
   rigid-body physics.

## Verification status

All references were checked against primary literature (18 confirmed, 5 with caveat,
1 refuted). Material corrections folded into the table above:

- **REFUTED:** "uniform pressure is exact for flat hydroelastic contact." Drake's
  field is linear in distance-to-nearest-face; uniform only over the central face,
  non-uniform within ~one half-extent of the rim. The `(2/3)*mu*m*g*R` torque and
  `eps* ~ 0.653` still hold for the cylinder geometry, just not by a
  "uniform-penetration => uniform-pressure" argument (Drake source; Farkas p.4).
- **CAVEAT:** the H6 linear-dashpot restitution `e=exp(-zeta*pi/sqrt(1-zeta^2))` is a
  lower bound because Newton's `(.)_+` clamp raises the realized rebound
  (Schwager-Poschel 2008; Hunt-Crossley 1975).
- **CAVEAT:** H3 Winkler torsional spring valid only within the kern (`e < L/6`).
- **CAVEAT:** H5 universality of `eps* ~ 0.653` and simultaneous stop hold for the
  uniform/homogeneous disk only.

## References

1. Farkas, Bartels, Unger, Wolf, "Frictional Coupling between Sliding and Spinning
   Motion", Phys. Rev. Lett. 90, 248302 (2003). arXiv:physics/0210024.
   DOI 10.1103/PhysRevLett.90.248302. -- F(eps), T(eps), eps0 ~ 0.653, T(0)=2/3,
   simultaneous stop, uniform-pressure / cylinder-drift caveat.
2. Weidman & Malhotra, "On the terminal motion of sliding spinning disks with uniform
   Coulomb friction", Physica D 233(1):1-13 (2007). DOI 10.1016/j.physd.2007.06.012.
   -- clean eps* = 0.653 (disk), 1.0 (ring).
3. Goyal, Ruina, Papadopoulos, "Planar sliding with dry friction, Parts 1 & 2", Wear
   143 (1991). -- foundational planar-sliding limit-surface results.
4. K. L. Johnson, "Contact Mechanics", Cambridge Univ. Press, 1985.
   DOI 10.1017/CBO9781139171731. -- elastic/Winkler foundation; impact energy method.
5. Schwager & Poschel, "Coefficient of restitution and linear dashpot model
   revisited", Granular Matter 10:23-33 (2008). DOI 10.1007/s10035-007-0065-z.
6. Hunt & Crossley, "Coefficient of Restitution Interpreted as Damping in
   Vibroimpact", J. Appl. Mech. 42(2):440-445 (1975). DOI 10.1115/1.3423596.
7. Girgin, "Simplified formulations for the determination of rotational spring
   constants in rigid spread footings resting on tensionless soil", J. Civil Eng.
   Manag. 23(4):464-474 (2017). DOI 10.3846/13923730.2016.1210218. -- K_r = K_s * I.
8. Halliday, Resnick, Walker, "Fundamentals of Physics", Ch. 6 (Coulomb friction).
   Purdue ME270 "Tipping vs. slipping" (F_tip = m*g/2).
9. Moore et al., "Mechanics Map" 6.6 "Disc Friction" ((2/3)*mu*P*R).
10. Masterjohn, Guoy, Shepherd, Castro, "Velocity Level Approximation of Pressure
    Field Contact Patches", IROS 2022. arXiv:2110.04157. -- hydroelastic force law
    (Eqs. 15-16, 23). Elandt et al., IROS 2019, arXiv:1904.11433 (elastic-foundation
    basis). Drake "Estimation of Hydroelastic Parameters" doc; Drake source
    `geometry/proximity/make_box_field.cc`.
11. MuJoCo docs, solver parameters (solref/solimp):
    https://mujoco.readthedocs.io/en/stable/modeling.html. Newton issue #2009
    (effective ke from solref).
