"""Shared scene config for the Drake vs Newton frozen-contact cross-check.

`scene.yaml` holds only physical inputs. This module is the SINGLE source of truth
for the derived quantities both engines need:
  * matched-modulus mapping  kh = E / H   (H = Drake foundation depth)
  * the frozen world poses (counter body at its center, upper sphere lowered by x)
  * the analytic elastic-foundation reference (Fn, contact area, peak pressure)

The counter body is either a flat box (sphere-on-flat, contact patch is a flat disc)
or a second sphere (sphere-on-sphere, the contact patch is a curved cap). The ONLY
physical change between them is the effective radius R_eff used by the reference:
  * box    -> R_eff = R          (counter radius is infinite / flat)
  * sphere -> R_eff = R*R2/(R+R2)  (series of the two curvatures)

Loading uses PyYAML, available in both envs (the hydro_compare uv project and the
newton-sap env).

Not run directly -- imported by frozen_compare/*.py and convergence/*.py.
"""
import math
import os
from types import SimpleNamespace


def _load_raw(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def default_scene_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "sphere_box.yaml")


def experiment_dir(cfg, name):
    """Per-experiment output subfolder under ``cfg.output_dir`` (created if missing)."""
    path = os.path.join(cfg.output_dir, name)
    os.makedirs(path, exist_ok=True)
    return path


def load_scene(path=None):
    raw = _load_raw(path or default_scene_path())

    R = float(raw["sphere"]["radius"])
    E_sphere = float(raw["sphere"]["hydroelastic_modulus"])
    x = float(raw["contact"]["penetration_x"])
    target_edge = float(raw["mesh"]["target_edge"])          # Newton voxel = this
    drake_scale = float(raw["mesh"].get("drake_scale", 1.0))    # Drake hint = scale * target
    newton_scale = float(raw["mesh"].get("newton_scale", 1.0))  # Newton voxel = scale * target

    # Sphere elastic-foundation depth -> R. Matched modulus kh = E / H (N/m^3).
    H_sphere = R
    kh_sphere = E_sphere / H_sphere

    # Counter body: flat box, or a second sphere (curved). box-counter fields stay None for
    # sphere-counter and vice versa, so each engine's dump builds only what its scene defines.
    box_full = box_center = box_top_z = H_box = kh_box = E_box = None
    counter_radius = E_counter = H_counter = None
    if "box" in raw:
        counter_type = "box"
        box_full = [float(v) for v in raw["box"]["full_size"]]
        E_box = float(raw["box"]["hydroelastic_modulus"])
        box_center = [float(v) for v in raw["box"]["center"]]
        H_box = min(box_full) / 2.0                          # box foundation depth -> min half-size
        kh_box = E_box / H_box
        kh_counter = kh_box
        E_counter = E_box
        R_eff = R                                            # flat counter -> infinite radius
        box_top_z = box_center[2] + box_full[2] / 2.0
        counter_center = box_center
        # upper sphere lowered so its lowest point sits x below the box top face
        sphere_center = [box_center[0], box_center[1], box_top_z + R - x]
    elif "counter" in raw and str(raw["counter"]["type"]) == "sphere":
        counter_type = "sphere"
        counter_radius = float(raw["counter"]["radius"])
        E_counter = float(raw["counter"]["hydroelastic_modulus"])
        counter_center = [float(v) for v in raw["counter"]["center"]]
        H_counter = counter_radius                           # sphere foundation depth -> its radius
        kh_counter = E_counter / H_counter
        R_eff = R * counter_radius / (R + counter_radius)    # series of the two curvatures
        # upper sphere lowered along z so the two surfaces overlap by exactly x
        sphere_center = [counter_center[0], counter_center[1],
                         counter_center[2] + counter_radius + R - x]
    else:
        raise ValueError("scene must define a 'box' or a 'counter: {type: sphere, ...}' body")

    # Effective normal gradient / per-area stiffness (springs in series).
    kappa = 1.0 / (1.0 / kh_sphere + 1.0 / kh_counter)

    # Analytic reference (Winkler / elastic foundation, small x): only R -> R_eff changes
    # between flat and curved.
    #   Fn = kappa * pi * R_eff * x^2 ,  area ~= 2*pi*R_eff*x ,  p0_max = kappa * x.
    reference = SimpleNamespace(
        Fn=kappa * math.pi * R_eff * x * x,
        area=2.0 * math.pi * R_eff * x,
        p0_max=kappa * x,
        patch_radius=math.sqrt(2.0 * R_eff * x),
    )

    return SimpleNamespace(
        raw=raw,
        counter_type=counter_type,
        R=R, E_sphere=E_sphere, R_eff=R_eff, x=x,
        # box-counter fields (None when counter is a sphere)
        box_full=box_full, box_center=box_center, E_box=E_box, H_box=H_box,
        kh_box=kh_box, box_top_z=box_top_z,
        # sphere-counter fields (None when counter is a box)
        counter_radius=counter_radius, counter_center=counter_center,
        E_counter=E_counter, H_counter=H_counter,
        # shared
        H_sphere=H_sphere, kh_sphere=kh_sphere, kh_counter=kh_counter, kappa=kappa,
        sphere_center=sphere_center,
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
    """Winkler / elastic-foundation pressure profile p*(r) [Pa]:
    p*(r) = kappa * max(0, x - r^2/(2 R_eff)), peak kappa*x at r=0, zero at r = sqrt(2 R_eff x)."""
    import numpy as np
    r = np.asarray(r, float)
    return cfg.kappa * np.maximum(0.0, cfg.x - r * r / (2.0 * cfg.R_eff))
