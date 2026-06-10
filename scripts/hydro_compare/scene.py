"""Shared scene config for the Drake vs Newton frozen-contact cross-check.

`scene.yaml` holds only physical inputs. This module is the SINGLE source of truth
for the derived quantities both engines need:
  * matched-modulus mapping  kh = E / H   (H = Drake foundation depth)
  * the frozen world poses (box at its center, sphere lowered by the penetration x)
  * the analytic elastic-foundation reference (Fn, contact area, peak pressure)

Loading uses PyYAML, available in both envs (the hydro_compare uv project and the
newton-sap env).

Not run directly -- imported by drake_dump.py / compare.py / view_surface.py (run via
`uv run` in the hydro_compare project) and by newton_dump.py (run via `uv run` in the
newton-sap env, needs CUDA). See each script's header for its exact run command.
"""
import math
import os
from types import SimpleNamespace


def _load_raw(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def default_scene_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "scene.yaml")


def load_scene(path=None):
    raw = _load_raw(path or default_scene_path())

    R = float(raw["sphere"]["radius"])
    E_sphere = float(raw["sphere"]["hydroelastic_modulus"])
    box_full = [float(v) for v in raw["box"]["full_size"]]
    E_box = float(raw["box"]["hydroelastic_modulus"])
    box_center = [float(v) for v in raw["box"]["center"]]
    x = float(raw["contact"]["penetration_x"])
    target_edge = float(raw["mesh"]["target_edge"])          # Newton voxel = this
    drake_scale = float(raw["mesh"].get("drake_scale", 1.0))    # Drake hint = scale * target
    newton_scale = float(raw["mesh"].get("newton_scale", 1.0))  # Newton voxel = scale * target

    # Drake elastic-foundation depths: sphere -> R; box -> min half-size.
    H_sphere = R
    H_box = min(box_full) / 2.0

    # Matched modulus: kh = E / H  (units N/m^3 = Pa/m), shared by both engines.
    kh_sphere = E_sphere / H_sphere
    kh_box = E_box / H_box
    # Effective normal gradient / per-area stiffness (springs in series).
    kappa = 1.0 / (1.0 / kh_sphere + 1.0 / kh_box)

    # Frozen poses: box at its center; sphere lowered so its lowest point sits x below
    # the box top face (i.e. an interpenetration of exactly x).
    box_top_z = box_center[2] + box_full[2] / 2.0
    sphere_center = [box_center[0], box_center[1], box_top_z + R - x]

    # Analytic reference (Winkler / elastic foundation, small x):
    #   Fn = kappa * pi * R * x^2 ,  area ~= 2*pi*R*x ,  p0_max = kappa * x.
    reference = SimpleNamespace(
        Fn=kappa * math.pi * R * x * x,
        area=2.0 * math.pi * R * x,
        p0_max=kappa * x,
        patch_radius=math.sqrt(2.0 * R * x),
    )

    return SimpleNamespace(
        raw=raw,
        R=R, E_sphere=E_sphere, E_box=E_box,
        box_full=box_full, box_center=box_center, x=x,
        H_sphere=H_sphere, H_box=H_box,
        kh_sphere=kh_sphere, kh_box=kh_box, kappa=kappa,
        box_top_z=box_top_z, sphere_center=sphere_center,
        friction=float(raw["material"]["friction"]),
        margin=float(raw["material"]["margin"]),
        target_edge=target_edge,
        drake_scale=drake_scale,
        resolution_hint=target_edge * drake_scale,  # Drake: ~1.3x finer than Newton at equal L
        sdf_target_voxel_size=target_edge * newton_scale,  # Newton: tuned to Drake's pinned median
        representation=str(raw["drake"]["representation"]),
        sdf_narrow_band_range=[float(v) for v in raw["newton"]["sdf_narrow_band_range"]],
        gap=float(raw["newton"]["gap"]),
        rigid_contact_max=int(raw["newton"]["rigid_contact_max"]),
        output_dir=str(raw["output_dir"]),
        reference=reference,
    )


def reference_pressure(cfg, r):
    """Winkler / elastic-foundation pressure profile p*(r) [Pa] for the frozen sphere-on-box:
    p*(r) = kappa * max(0, x - r^2/(2R)), peak kappa*x at r=0, zero at r = sqrt(2Rx). The
    common analytic reference for area-weighted field-error metrics (each engine's field is
    compared to this at its OWN sample points -- no cross-mesh interpolation)."""
    import numpy as np
    r = np.asarray(r, float)
    return cfg.kappa * np.maximum(0.0, cfg.x - r * r / (2.0 * cfg.R))
