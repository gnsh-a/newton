"""Compare the Drake and Newton frozen-contact dumps against each other and the reference.

Compares only mesh-invariant aggregates (the tet and marching-cubes meshes are NOT
expected to match) plus two physics checks:
  A) Drake effective gradient g  vs  Newton k_eff   (should both equal kappa)
  B) Drake phi0  vs  Newton delta_total (both TOTAL penetration; Newton's recovered from phi_b)
Writes out/diff.json and out/distributions.png. No mesh resampling.

Run (the newton-sap uv env -- needs numpy+matplotlib+pyyaml; run both dumps first):
    cd ~/work/newton-sap
    uv run --no-sync python scripts/hydro_compare/experiments/compare.py
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # experiments/ (scene.py)
import scene as scene_mod


def _load(out_dir, engine):
    npz = dict(np.load(os.path.join(out_dir, f"{engine}.npz")))
    with open(os.path.join(out_dir, f"{engine}.json")) as f:
        return npz, json.load(f)


def _unit(v):
    v = np.asarray(v, float)
    nrm = np.linalg.norm(v)
    return v / nrm if nrm > 0 else v


def _patch_radius(centroid_W, center_xy):
    c = np.asarray(centroid_W)
    return np.linalg.norm(c[:, :2] - np.asarray(center_xy)[:2], axis=1)


def _field_error_vs_reference(cfg, centroid_W, area, p, center_xy):
    """Area-weighted L2 and L-inf of a per-element pressure field [Pa] vs the analytic reference,
    integrated on the element's OWN mesh (no cross-mesh interpolation, area-weighted so sliver
    faces don't dominate). L2 = sqrt(sum A_i e_i^2 / sum A_i); L-inf = max|e_i|."""
    r = np.linalg.norm(np.asarray(centroid_W)[:, :2] - np.asarray(center_xy), axis=1)
    e = np.asarray(p, float) - scene_mod.reference_pressure(cfg, r)
    A = np.asarray(area, float)
    return float(np.sqrt((A * e * e).sum() / A.sum())), float(np.abs(e).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=scene_mod.default_scene_path())
    cfg = scene_mod.load_scene(ap.parse_args().scene)
    out = scene_mod.experiment_dir(cfg, "frozen_compare")

    d, dm = _load(out, "drake")
    n, nm = _load(out, "newton")

    # Reduce both engines to "force on the sphere", in world. Drake's agg_net_normal_W
    # is sum(fn0 * n_hat) with +n_hat pointing into M; map M/N to the sphere via ids.
    drake_force = np.array(dm["agg_net_normal_W"])
    f_sphere_drake = drake_force if dm["id_M"] == dm["id_sphere"] else -drake_force
    f_sphere_newton = np.array(nm["agg_net_force_on_sphere_W"])

    cen_d, cen_n = np.array(dm["agg_centroid_W"]), np.array(nm["agg_centroid_W"])
    g = d["g"]

    # Reference-native area-weighted field error (option 1) + mesh-independent invariants.
    center_xy = np.asarray(cfg.box_center[:2], float)
    dL2, dLinf = _field_error_vs_reference(cfg, d["centroid_W"], d["area"], d["p0"], center_xy)
    nL2, nLinf = _field_error_vs_reference(cfg, n["centroid_W"], n["area"], n["pressure"], center_xy)
    p0max = cfg.reference.p0_max

    diff = {
        "x": cfg.x,
        "Fn_total": {
            "drake": dm["agg_Fn_total"], "newton": nm["agg_Fn_total"], "reference": cfg.reference.Fn,
            "drake/reference": dm["agg_Fn_total"] / cfg.reference.Fn,
            "newton/reference": nm["agg_Fn_total"] / cfg.reference.Fn,
            "newton/drake": nm["agg_Fn_total"] / dm["agg_Fn_total"],
        },
        "force_on_sphere": {
            "drake": f_sphere_drake.tolist(), "newton": f_sphere_newton.tolist(),
            "angle_deg": float(np.degrees(np.arccos(
                np.clip(_unit(f_sphere_drake) @ _unit(f_sphere_newton), -1.0, 1.0)))),
        },
        "centroid_W": {
            "drake": cen_d.tolist(), "newton": cen_n.tolist(),
            "distance_m": float(np.linalg.norm(cen_d - cen_n)),
        },
        "total_area": {
            "drake": dm["agg_total_area"], "newton": nm["agg_total_area"],
            "reference": cfg.reference.area,
            "newton/drake": nm["agg_total_area"] / dm["agg_total_area"],
        },
        "checkA_stiffness_Pa_per_m": {
            "drake_g_mean": float(g.mean()), "drake_g_min": float(g.min()),
            "drake_g_max": float(g.max()), "newton_k_eff": nm["k_eff_mean"],
            "kappa": cfg.kappa,
        },
        "checkB_penetration_m": {
            "drake_phi0_fwmean": float((d["fn0"] * d["phi0"]).sum() / d["fn0"].sum()),
            "newton_total_depth_fwmean": float((n["Fn_i"] * n["depth_total"]).sum() / n["Fn_i"].sum()),
        },
        "checkC_field_error_vs_reference_Pa": {
            "drake_L2": dL2, "drake_Linf": dLinf, "drake_L2_rel": dL2 / p0max,
            "newton_L2": nL2, "newton_Linf": nLinf, "newton_L2_rel": nL2 / p0max,
            "engine_gap_L2_bound": dL2 + nL2,   # triangle-ineq. bound on ||p_drake - p_newton||
        },
        "invariants": {
            "p_max_Pa": {"drake": float(d["p0"].max()),
                         "newton": float(n["pressure"].max()), "reference": p0max},
            "equiv_radius_m": {"drake": float(np.sqrt(d["area"].sum() / np.pi)),
                               "newton": float(np.sqrt(n["area"].sum() / np.pi)),
                               "reference": cfg.reference.patch_radius},
            "cop_radial_offset_m": {"drake": float(np.linalg.norm(cen_d[:2] - center_xy)),
                                    "newton": float(np.linalg.norm(cen_n[:2] - center_xy))},
        },
    }
    b = diff["checkB_penetration_m"]
    b["newton/drake"] = b["newton_total_depth_fwmean"] / b["drake_phi0_fwmean"]

    with open(os.path.join(out, "diff.json"), "w") as f:
        json.dump(diff, f, indent=2)
    print(json.dumps(diff, indent=2))

    rd = _patch_radius(d["centroid_W"], cen_d)
    rn = _patch_radius(n["centroid_W"], cen_n)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    rr = np.linspace(0.0, cfg.reference.patch_radius, 200)
    axs[0].plot(rr, scene_mod.reference_pressure(cfg, rr) / 1.0e6, "k-", lw=1, label="reference p*(r)")
    axs[0].scatter(rd, d["p0"] / 1.0e6, s=8, label="drake p0")
    axs[0].scatter(rn, n["pressure"] / 1.0e6, s=8, marker="x", label="newton p=kh_b*|depth|")
    axs[0].set(xlabel="patch radius [m]", ylabel="pressure [MPa]", title="pressure")
    axs[1].scatter(rd, d["k"], s=8, label="drake k")
    axs[1].scatter(rn, n["stiffness"], s=8, marker="x", label="newton stiffness")
    axs[1].set(xlabel="patch radius [m]", ylabel="k [N/m]", title="stiffness")
    # Panel 2 is the factor-of-2 detector: total penetration must stay UNDER the imposed
    # geometric interference delta_geom(r) = x - r^2/2R (ceiling = x at the apex). If Newton's
    # contact_distance = 2*depth were mishandled (no /2), |delta_total| would ~double and breach
    # the x = 5 mm ceiling -- physically impossible. Plotted in mm, magnitudes.
    rr2 = np.linspace(0.0, cfg.reference.patch_radius, 200)
    geom_mm = np.maximum(0.0, cfg.x - rr2 * rr2 / (2.0 * cfg.R)) * 1.0e3
    axs[2].plot(rr2, geom_mm, "k-", lw=1, label="imposed delta_geom(r)")
    axs[2].axhline(cfg.x * 1.0e3, color="r", ls=":", lw=1, label=f"ceiling x = {cfg.x * 1e3:.0f} mm")
    axs[2].scatter(rd, -d["phi0"] * 1.0e3, s=8, label="drake |phi0| (total)")
    axs[2].scatter(rn, -n["depth_total"] * 1.0e3, s=8, marker="x", label="newton |delta_total|")
    axs[2].scatter(rn, -n["depth"] * 1.0e3, s=6, marker="+", alpha=0.5, label="newton |depth| (1 body)")
    axs[2].set(xlabel="patch radius [m]", ylabel="penetration [mm]",
               title="penetration vs imposed (2x detector)")
    for a in axs:
        a.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "distributions.png"), dpi=120)
    print(f"[compare] wrote {out}/diff.json and {out}/distributions.png")


if __name__ == "__main__":
    main()
