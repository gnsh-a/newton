# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run the Drake Hydro + Drake SAP sphere-box reference from the Newton YAML scene."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

CSV_FIELDS = (
    "time",
    "x",
    "y",
    "z",
    "qx",
    "qy",
    "qz",
    "qw",
    "wx",
    "wy",
    "wz",
    "vx",
    "vy",
    "vz",
    "angular_speed",
    "linear_speed",
    "point_contacts",
    "hydro_contacts",
    "contact_force_x",
    "contact_force_y",
    "contact_force_z",
    "contact_area",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_scene_path() -> Path:
    return _repo_root() / "newton" / "examples" / "assets" / "sphere_box_hydro.yaml"


def _default_output_path() -> Path:
    return Path(__file__).resolve().parent / "out" / "run_sphere_box" / "drake_sphere_box_reference.csv"


def _load_scene(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in scene YAML: {path}")
    return data


def _import_pydrake():
    try:
        from pydrake.geometry import (  # noqa: PLC0415
            AddCompliantHydroelasticProperties,
            AddContactMaterial,
            Box,
            ProximityProperties,
            Sphere,
        )
        from pydrake.math import RigidTransform  # noqa: PLC0415
        from pydrake.multibody.plant import (  # noqa: PLC0415
            AddMultibodyPlant,
            CoulombFriction,
            MultibodyPlantConfig,
        )
        from pydrake.multibody.tree import SpatialInertia, UnitInertia  # noqa: PLC0415
        from pydrake.systems.analysis import Simulator  # noqa: PLC0415
        from pydrake.systems.framework import DiagramBuilder  # noqa: PLC0415
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("pydrake"):
            raise SystemExit(
                "pydrake is not installed in this environment. Run with the main checkout env, e.g.\n"
                "  cd /home/ganesharivoli/work/newton-gnsh\n"
                "  UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python "
                "/home/ganesharivoli/work/newton-gnsh/.claude/worktrees/sap-hydro-smoke/"
                "scripts/hydro_compare/drake_sphere_box_reference.py"
            ) from exc
        raise

    return {
        "AddCompliantHydroelasticProperties": AddCompliantHydroelasticProperties,
        "AddContactMaterial": AddContactMaterial,
        "AddMultibodyPlant": AddMultibodyPlant,
        "Box": Box,
        "CoulombFriction": CoulombFriction,
        "DiagramBuilder": DiagramBuilder,
        "MultibodyPlantConfig": MultibodyPlantConfig,
        "ProximityProperties": ProximityProperties,
        "RigidTransform": RigidTransform,
        "Simulator": Simulator,
        "SpatialInertia": SpatialInertia,
        "Sphere": Sphere,
        "UnitInertia": UnitInertia,
    }


def _make_proximity_properties(
    drake: dict[str, Any],
    *,
    hydroelastic_modulus: float,
    resolution_hint: float,
    friction: float,
    dissipation: float | None,
):
    properties = drake["ProximityProperties"]()
    drake["AddCompliantHydroelasticProperties"](
        resolution_hint=resolution_hint,
        hydroelastic_modulus=hydroelastic_modulus,
        properties=properties,
    )
    drake["AddContactMaterial"](
        properties=properties,
        dissipation=dissipation,
        friction=drake["CoulombFriction"](
            static_friction=friction,
            dynamic_friction=friction,
        ),
    )
    return properties


def _build_diagram(scene: dict[str, Any], args: argparse.Namespace, drake: dict[str, Any]):
    radius = float(scene["sphere"]["radius"])
    sphere_pos = [float(v) for v in scene["sphere"]["initial_position"]]
    sphere_modulus = float(scene["sphere"]["hydroelastic_modulus"])
    sphere_density = float(scene["sphere"].get("density", args.density))
    box_full = [float(v) for v in scene["box"]["full_size"]]
    box_modulus = float(scene["box"]["hydroelastic_modulus"])
    friction = float(scene["material"]["friction"])
    resolution_hint = float(scene["mesh"]["sdf_target_voxel_size"])

    frame_dt = 1.0 / float(args.fps)
    sim_dt = frame_dt / int(args.substeps)

    builder = drake["DiagramBuilder"]()
    plant_config = drake["MultibodyPlantConfig"](time_step=sim_dt)
    plant_config.contact_model = args.contact_model
    plant_config.discrete_contact_approximation = args.discrete_contact_approximation
    plant_config.contact_surface_representation = args.contact_surface_representation
    plant, _scene_graph = drake["AddMultibodyPlant"](plant_config, builder)

    mass = sphere_density * (4.0 / 3.0) * math.pi * radius**3
    inertia = drake["SpatialInertia"](
        mass=mass,
        p_PScm_E=[0.0, 0.0, 0.0],
        G_SP_E=drake["UnitInertia"].SolidSphere(radius),
    )
    sphere_body = plant.AddRigidBody("sphere", inertia)

    sphere_props = _make_proximity_properties(
        drake,
        hydroelastic_modulus=sphere_modulus,
        resolution_hint=resolution_hint,
        friction=friction,
        dissipation=args.dissipation,
    )
    box_props = _make_proximity_properties(
        drake,
        hydroelastic_modulus=box_modulus,
        resolution_hint=resolution_hint,
        friction=friction,
        dissipation=args.dissipation,
    )

    X_identity = drake["RigidTransform"]()
    plant.RegisterCollisionGeometry(
        sphere_body,
        X_identity,
        drake["Sphere"](radius),
        "sphere_collision",
        sphere_props,
    )
    plant.RegisterVisualGeometry(
        sphere_body,
        X_identity,
        drake["Sphere"](radius),
        "sphere_visual",
        [0.2, 0.45, 1.0, 1.0],
    )
    plant.RegisterCollisionGeometry(
        plant.world_body(),
        X_identity,
        drake["Box"](*box_full),
        "box_collision",
        box_props,
    )
    plant.RegisterVisualGeometry(
        plant.world_body(),
        X_identity,
        drake["Box"](*box_full),
        "box_visual",
        [0.5, 0.5, 0.5, 1.0],
    )

    plant.Finalize()
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyMutableContextFromRoot(context)
    plant.SetFreeBodyPose(plant_context, sphere_body, drake["RigidTransform"](sphere_pos))

    return diagram, plant, sphere_body, context, frame_dt, sim_dt, mass


def _sample_row(plant, plant_context, sphere_body, time: float) -> dict[str, float]:
    pose = plant.EvalBodyPoseInWorld(plant_context, sphere_body)
    velocity = plant.EvalBodySpatialVelocityInWorld(plant_context, sphere_body)
    contacts = plant.get_contact_results_output_port().Eval(plant_context)

    translation = pose.translation()
    quat = pose.rotation().ToQuaternion()
    angular = np.asarray(velocity.rotational(), dtype=float)
    linear = np.asarray(velocity.translational(), dtype=float)

    total_force = np.zeros(3, dtype=float)
    contact_area = 0.0
    for i in range(contacts.num_hydroelastic_contacts()):
        info = contacts.hydroelastic_contact_info(i)
        total_force += np.asarray(info.F_Ac_W().translational(), dtype=float)
        contact_area += float(info.contact_surface().total_area())
    for i in range(contacts.num_point_pair_contacts()):
        total_force += np.asarray(contacts.point_pair_contact_info(i).contact_force(), dtype=float)

    return {
        "time": float(time),
        "x": float(translation[0]),
        "y": float(translation[1]),
        "z": float(translation[2]),
        "qx": float(quat.x()),
        "qy": float(quat.y()),
        "qz": float(quat.z()),
        "qw": float(quat.w()),
        "wx": float(angular[0]),
        "wy": float(angular[1]),
        "wz": float(angular[2]),
        "vx": float(linear[0]),
        "vy": float(linear[1]),
        "vz": float(linear[2]),
        "angular_speed": float(np.linalg.norm(angular)),
        "linear_speed": float(np.linalg.norm(linear)),
        "point_contacts": int(contacts.num_point_pair_contacts()),
        "hydro_contacts": int(contacts.num_hydroelastic_contacts()),
        "contact_force_x": float(total_force[0]),
        "contact_force_y": float(total_force[1]),
        "contact_force_z": float(total_force[2]),
        "contact_area": float(contact_area),
    }


def _write_rows(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    scene = _load_scene(args.scene)
    drake = _import_pydrake()
    diagram, plant, sphere_body, context, frame_dt, sim_dt, mass = _build_diagram(scene, args, drake)

    simulator = drake["Simulator"](diagram, context)
    simulator.set_target_realtime_rate(0.0)

    rows: list[dict[str, float]] = []
    rows.append(_sample_row(plant, plant.GetMyContextFromRoot(simulator.get_context()), sphere_body, 0.0))
    for frame in range(1, int(args.num_frames) + 1):
        time = frame * frame_dt
        simulator.AdvanceTo(time)
        plant_context = plant.GetMyContextFromRoot(simulator.get_context())
        rows.append(_sample_row(plant, plant_context, sphere_body, time))

    _write_rows(args.output, rows)

    final = rows[-1]
    first_contact_time = next(
        (row["time"] for row in rows if int(row["point_contacts"]) + int(row["hydro_contacts"]) > 0),
        None,
    )
    summary = {
        "output": str(args.output),
        "scene": str(args.scene),
        "case": "Drake Hydro + Drake SAP",
        "contact_source": "Drake Hydro",
        "solver": "Drake SAP",
        "num_frames": int(args.num_frames),
        "frame_dt": frame_dt,
        "sim_dt": sim_dt,
        "sphere_mass": mass,
        "first_contact_time": first_contact_time,
        "final_z": final["z"],
        "final_linear_speed": final["linear_speed"],
        "final_angular_speed": final["angular_speed"],
        "final_hydro_contacts": int(final["hydro_contacts"]),
        "final_point_contacts": int(final["point_contacts"]),
    }
    if not args.quiet:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=_default_scene_path(), help="Scene YAML path.")
    parser.add_argument("--output", type=Path, default=_default_output_path(), help="Output CSV path.")
    parser.add_argument("--num-frames", type=int, default=600, help="Number of 120 Hz frames to simulate.")
    parser.add_argument("--fps", type=float, default=120.0, help="Output frame rate.")
    parser.add_argument("--substeps", type=int, default=4, help="Discrete Drake steps per output frame.")
    parser.add_argument("--density", type=float, default=1000.0, help="Fallback sphere density [kg/m^3].")
    parser.add_argument(
        "--dissipation",
        type=float,
        default=None,
        help="Optional Drake contact material dissipation. Omitted by default.",
    )
    parser.add_argument(
        "--contact-model",
        choices=["hydroelastic", "hydroelastic_with_fallback"],
        default="hydroelastic_with_fallback",
        help="Drake contact model.",
    )
    parser.add_argument(
        "--discrete-contact-approximation",
        choices=["sap", "lagged", "similar", "tamsi"],
        default="sap",
        help="Drake discrete contact approximation.",
    )
    parser.add_argument(
        "--contact-surface-representation",
        choices=["polygon", "triangle"],
        default="polygon",
        help="Drake Hydro contact surface representation.",
    )
    parser.add_argument("--quiet", action="store_true", help="Do not print the JSON summary.")
    return parser


def main() -> None:
    args = create_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
