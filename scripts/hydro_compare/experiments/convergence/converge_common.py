"""Shared metrics + plotting for the Drake and Newton hydroelastic convergence sweeps.

Pure numpy/matplotlib/scene -- no pydrake or newton imports -- so it loads in BOTH uv envs
(the Drake dump runs in hydro_compare/newton-sap with pydrake; the Newton dump runs in
newton-sap with CUDA). Each dump builds a list of per-resolution ``level`` dicts with the
SAME keys and hands them here, so the two studies produce format-identical figures on the
same Winkler reference and the same x-axis (median iso-surface triangle edge).

Expected per-level keys: median_edge, Fn, p0_max, equiv_r, dFn, n_tris, r_centers, pbar,
                         label (legend text), show_profile (include in the p(r) panel).
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # experiments/ (scene.py, view_surface.py)
import scene as scene_mod


def _radial_profile(centroid_xy, area, pressure, center_xy, nbins, rmax):
    """Area-weighted mean pressure per radial bin (axisymmetric collapse to 1-D).
    Returns (bin_centers, p_bar); empty bins are NaN."""
    r = np.linalg.norm(centroid_xy - center_xy, axis=1)
    edges = np.linspace(0.0, rmax, nbins + 1)
    idx = np.clip(np.digitize(r, edges) - 1, 0, nbins - 1)
    pbar = np.full(nbins, np.nan)
    for b in range(nbins):
        m = idx == b
        if m.any() and area[m].sum() > 0.0:
            pbar[b] = (area[m] * pressure[m]).sum() / area[m].sum()
    return 0.5 * (edges[:-1] + edges[1:]), pbar


def _profile_resolved(pbar):
    """A radial profile is resolved enough to plot when its peak (innermost) bin is filled
    and at most one bin (the outermost, beyond the patch) is empty -- the data-driven version
    of "skip the coarse under-sampled curves" (matches the empirical L>=5 / edge<=~1.8mm cut)."""
    filled = int(np.sum(~np.isnan(pbar)))
    return bool((not np.isnan(pbar[0])) and filled >= len(pbar) - 2)


def _setup_style():
    """Publication defaults: larger consistent fonts, subtle grid, vector-friendly text."""
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.titleweight": "bold",
        "legend.fontsize": 9,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.6,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.6,
        "lines.markersize": 5,
        "legend.frameon": False,
        "pdf.fonttype": 42,      # embed TrueType (editable text in the PDF)
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
    })


def _plot(levels, ref, cfg, path):
    """One publication figure, all non-surface metrics (2x3):
      top    -- ratios/Winkler, successive ΔFn, mesh growth (the convergence story)
      bottom -- absolute Fn, absolute p0_max & equiv_r, radial pressure profile.
    Saves <path>.png (300 dpi)."""
    edge = np.array([lv["median_edge"] for lv in levels]) * 1e3
    fn = np.array([lv["Fn"] for lv in levels])
    p0 = np.array([lv["p0_max"] for lv in levels]) / 1e6
    eqr = np.array([lv["equiv_r"] for lv in levels]) * 1e3
    ntris = np.array([lv["n_tris"] for lv in levels])
    cmap = plt.get_cmap("viridis")
    lvl_color = [cmap(j / max(1, len(levels) - 1)) for j in range(len(levels))]
    fig, axs = plt.subplots(2, 3, figsize=(15, 8.5))

    # (a) aggregates / Winkler vs edge -- convergence quality (→1)
    series = (("Fn", r"$F_n$", "o", "tab:blue"),
              ("p0_max", r"$p_0^{\max}$", "s", "tab:orange"),
              ("equiv_r", r"$r_\mathrm{eq}$", "^", "tab:green"))
    for key, lab, mk, col in series:
        rref = dict(Fn=ref.Fn, p0_max=ref.p0_max, equiv_r=ref.patch_radius)[key]
        axs[0, 0].plot(edge, np.array([lv[key] for lv in levels]) / rref, mk + "-", color=col, label=lab)
    axs[0, 0].axhline(1.0, color="k", ls=":", lw=1, label="Winkler")
    axs[0, 0].set(xlabel="mesh resolution [mm]", ylabel="value / Winkler",
                  title="(a) accuracy vs Winkler")
    axs[0, 0].invert_xaxis(); axs[0, 0].legend()

    # (b) successive relative change in Fn -- self-convergence (→0)
    dfn = [lv["dFn"] for lv in levels if lv["dFn"] is not None]
    if dfn:
        axs[0, 1].semilogy(edge[1:], dfn, "s-", color="tab:red")
    axs[0, 1].set(xlabel="mesh resolution [mm]", ylabel=r"$|\Delta F_n|\,/\,F_n$",
                  title="(b) successive change")
    axs[0, 1].invert_xaxis()

    # (c) mesh growth -- triangle count vs edge (cost of refinement)
    axs[0, 2].loglog(edge, ntris, "o-", color="tab:purple")
    axs[0, 2].set(xlabel="mesh resolution [mm]", ylabel="triangle count",
                  title="(c) mesh growth")
    axs[0, 2].invert_xaxis()

    # (d) absolute normal force + Winkler target
    axs[1, 0].plot(edge, fn, "o-", color="tab:blue", label=r"$F_n$")
    axs[1, 0].axhline(ref.Fn, color="k", ls=":", lw=1, label="Winkler")
    axs[1, 0].set(xlabel="mesh resolution [mm]", ylabel=r"$F_n$ [N]",
                  title="(d) normal force")
    axs[1, 0].invert_xaxis(); axs[1, 0].legend()

    # (e) absolute peak pressure (left) + equiv radius (right), twin-axis, + Winkler targets
    ax = axs[1, 1]
    l1 = ax.plot(edge, p0, "o-", color="tab:orange", label=r"$p_0^{\max}$")
    ax.axhline(ref.p0_max / 1e6, color="tab:orange", ls=":", lw=1)
    ax.set(xlabel="mesh resolution [mm]", ylabel=r"$p_0^{\max}$ [MPa]",
           title="(e) peak pressure & patch radius")
    ax.invert_xaxis()
    ax2 = ax.twinx()
    ax2.grid(False)
    l2 = ax2.plot(edge, eqr, "^-", color="tab:green", label=r"$r_\mathrm{eq}$")
    ax2.axhline(ref.patch_radius * 1e3, color="tab:green", ls=":", lw=1)
    ax2.set_ylabel(r"$r_\mathrm{eq}$ [mm]")
    ax.legend(l1 + l2, [h.get_label() for h in l1 + l2], loc="lower left")

    # (f) radial pressure profile p_bar(r); skip under-resolved coarse levels (viridis: dark→bright) + Winkler
    nbins = len(levels[0]["pbar"])
    for j, lv in enumerate(levels):
        if not lv["show_profile"]:
            continue
        axs[1, 2].plot(np.array(lv["r_centers"]) * 1e3,
                       np.array([np.nan if v is None else v for v in lv["pbar"]]) / 1e6,
                       "o-", ms=3, color=lvl_color[j], label=lv["label"])
    rr = np.linspace(0.0, 1.1 * ref.patch_radius, 200)
    axs[1, 2].plot(rr * 1e3, scene_mod.reference_pressure(cfg, rr) / 1e6,
                   "k--", lw=1.5, label=r"Winkler $p^*(r)$")
    axs[1, 2].set(xlabel="patch radius $r$ [mm]", ylabel=r"$\bar p(r)$ [MPa]",
                  title=f"(f) radial pressure profile ({nbins} bins)")
    axs[1, 2].legend(fontsize=7, ncol=2)

    fig.tight_layout()
    fig.savefig(path, dpi=300)
