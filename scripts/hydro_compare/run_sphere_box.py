# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run the sphere-box hydro scene with selectable hydro and solver backends."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp
import yaml

import newton
from newton.geometry import HydroelasticSDF

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


def _default_output_path(hydro_pipeline: str, solver: str) -> Path:
    out_dir = Path(__file__).resolve().parent / "out" / "run_sphere_box"
    names = {
        ("newton", "sap_warp"): "newton_sap_sphere_box.csv",
        ("newton", "mujoco"): "newton_mujoco_sphere_box.csv",
        ("drake", "sap_warp"): "drake_hydro_sap_warp_sphere_box.csv",
        ("drake", "mujoco"): "drake_hydro_newton_mujoco_sphere_box.csv",
    }
    return out_dir / names[(hydro_pipeline, solver)]


def _load_scene(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in scene YAML: {path}")
    return data


def _write_rows(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _scene_resolution_hint(scene: dict[str, Any], args: argparse.Namespace) -> float:
    if args.drake_resolution_hint is not None:
        return float(args.drake_resolution_hint)
    return float(scene["mesh"]["sdf_target_voxel_size"])


def _import_sap_warp() -> tuple[Any, Any, Any, Any]:
    # Keep sap_warp local so MuJoCo-only runs do not require that checkout.
    sap_warp_root = os.environ.get("SAP_WARP_ROOT", os.path.expanduser("~/work/sap_warp"))
    if sap_warp_root not in sys.path:
        sys.path.insert(0, sap_warp_root)

    from sim.sap_runtime import sap_control_from_newton, sap_model_from_newton, sap_state_from_newton  # noqa: PLC0415
    from sim.solver_sap import SolverSAP  # noqa: PLC0415

    return sap_control_from_newton, sap_model_from_newton, sap_state_from_newton, SolverSAP


def _build_model(scene: dict[str, Any]) -> tuple[Any, int]:
    radius = float(scene["sphere"]["radius"])
    sphere_pos = [float(v) for v in scene["sphere"]["initial_position"]]
    sphere_modulus = float(scene["sphere"]["hydroelastic_modulus"])
    box_full = [float(v) for v in scene["box"]["full_size"]]
    box_modulus = float(scene["box"]["hydroelastic_modulus"])
    friction = float(scene["material"]["friction"])
    voxel = float(scene["mesh"]["sdf_target_voxel_size"])
    narrow_band = tuple(float(v) for v in scene["mesh"]["sdf_narrow_band_range"])

    kh_sphere = sphere_modulus / radius
    kh_box = box_modulus / (min(box_full) / 2.0)

    def hydro_cfg(kh: float) -> newton.ModelBuilder.ShapeConfig:
        return newton.ModelBuilder.ShapeConfig(
            is_hydroelastic=True,
            kh=kh,
            sdf_target_voxel_size=voxel,
            sdf_narrow_band_range=narrow_band,
            mu=friction,
            gap=0.0,
        )

    builder = newton.ModelBuilder()
    hx, hy, hz = (v / 2.0 for v in box_full)
    builder.add_shape_box(body=-1, hx=hx, hy=hy, hz=hz, cfg=hydro_cfg(kh_box), label="box")
    sphere_body = builder.add_body(
        xform=wp.transform(wp.vec3(*sphere_pos), wp.quat_identity()),
        label="sphere",
    )
    builder.add_shape_sphere(sphere_body, radius=radius, cfg=hydro_cfg(kh_sphere), label="sphere")
    return builder.finalize(), sphere_body


def _default_collision_pipeline(model: Any, contact_cap: int, args: argparse.Namespace) -> Any:
    return newton.CollisionPipeline(
        model,
        reduce_contacts=False,
        rigid_contact_max=contact_cap,
        broad_phase=args.broad_phase,
        sdf_hydroelastic_config=HydroelasticSDF.Config(
            reduce_contacts=False,
            output_contact_surface=False,
        ),
    )


class _SphereBoxRunBase:
    def __init__(self, scene: dict[str, Any], args: argparse.Namespace, collision_pipeline_factory: Any | None = None):
        self.args = args
        self.fps = float(args.fps)
        self.frame_dt = 1.0 / self.fps
        self.substeps = int(args.substeps)
        self.sim_dt = self.frame_dt / self.substeps
        self.sim_time = 0.0
        self.last_contact_count = 0

        self.model, self.sphere_body = _build_model(scene)
        self.contact_cap = int(args.contact_cap)
        if collision_pipeline_factory is None:
            self.collision_pipeline = _default_collision_pipeline(self.model, self.contact_cap, args)
        else:
            self.collision_pipeline = collision_pipeline_factory(self.model, self.contact_cap, args)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.contacts = self.collision_pipeline.contacts()
        self.collision_pipeline.collide(self.state_0, self.contacts)
        self._update_contact_count()

    def _update_contact_count(self) -> None:
        self.last_contact_count = int(self.contacts.rigid_contact_count.numpy()[0])

    def sample_row(self, time: float) -> dict[str, float]:
        q = np.asarray(self.state_0.body_q.numpy()[self.sphere_body], dtype=float)
        qd = np.asarray(self.state_0.body_qd.numpy()[self.sphere_body], dtype=float)
        angular = qd[0:3]
        linear = qd[3:6]
        return {
            "time": float(time),
            "x": float(q[0]),
            "y": float(q[1]),
            "z": float(q[2]),
            "qx": float(q[3]),
            "qy": float(q[4]),
            "qz": float(q[5]),
            "qw": float(q[6]),
            "wx": float(angular[0]),
            "wy": float(angular[1]),
            "wz": float(angular[2]),
            "vx": float(linear[0]),
            "vy": float(linear[1]),
            "vz": float(linear[2]),
            "angular_speed": float(np.linalg.norm(angular)),
            "linear_speed": float(np.linalg.norm(linear)),
            "point_contacts": 0,
            "hydro_contacts": int(self.last_contact_count),
            "contact_force_x": float("nan"),
            "contact_force_y": float("nan"),
            "contact_force_z": float("nan"),
            "contact_area": float("nan"),
        }


class SphereBoxSapRun(_SphereBoxRunBase):
    """Small runner for the sphere-box scene with sap_warp SAP."""

    def __init__(self, scene: dict[str, Any], args: argparse.Namespace, collision_pipeline_factory: Any | None = None):
        sap_control_from_newton, sap_model_from_newton, sap_state_from_newton, SolverSAP = _import_sap_warp()

        super().__init__(scene, args, collision_pipeline_factory)
        self.sap_model = sap_model_from_newton(self.model)
        self.solver = SolverSAP(
            self.sap_model,
            max_rigid_contact=self.contact_cap,
            max_iterations=int(args.solver_iterations),
            contact_beta=args.contact_beta,
            contact_sigma=args.contact_sigma,
            contact_preset_variant=args.contact_preset,
            contact_weight_mode=args.contact_weight_mode,
            line_search_variant=args.line_search,
            contact_tau_d=args.contact_tau_d,
        )
        self.sap_state_0 = sap_state_from_newton(self.state_0)
        self.sap_state_1 = sap_state_from_newton(self.state_1)
        self.sap_control = sap_control_from_newton(self.control)

    def step_frame(self) -> None:
        for _ in range(self.substeps):
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self._update_contact_count()
            self.solver.step(
                self.sap_state_0,
                self.sap_state_1,
                self.sap_control,
                self.contacts,
                self.sim_dt,
            )
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.sap_state_0, self.sap_state_1 = self.sap_state_1, self.sap_state_0
            self.sim_time += self.sim_dt


class SphereBoxMuJoCoRun(_SphereBoxRunBase):
    """Small runner for the sphere-box scene with Newton MuJoCo."""

    def __init__(self, scene: dict[str, Any], args: argparse.Namespace, collision_pipeline_factory: Any | None = None):
        self.collide_every = int(args.collide_every)
        super().__init__(scene, args, collision_pipeline_factory)
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            use_mujoco_contacts=False,
            solver=args.solver,
            integrator=args.integrator,
            cone=args.cone,
            njmax=self.contact_cap,
            nconmax=self.contact_cap,
            iterations=int(args.solver_iterations),
            ls_iterations=int(args.line_search_iterations),
            impratio=float(args.impratio),
        )

    def step_frame(self) -> None:
        for substep in range(self.substeps):
            if substep % self.collide_every == 0:
                self.collision_pipeline.collide(self.state_0, self.contacts)
                self._update_contact_count()
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.sim_time += self.sim_dt


def _collision_pipeline_factory(
    scene: dict[str, Any], args: argparse.Namespace
) -> tuple[Callable[[Any, int, argparse.Namespace], Any] | None, float | None]:
    if args.hydro_pipeline == "newton":
        return None, None

    from drake_pipeline import DrakeContactPipeline  # noqa: PLC0415

    resolution_hint = _scene_resolution_hint(scene, args)

    def make_drake_pipeline(model: Any, contact_cap: int, _args: argparse.Namespace) -> DrakeContactPipeline:
        return DrakeContactPipeline(
            model,
            resolution_hint=resolution_hint,
            rigid_contact_max=contact_cap,
            contact_surface_representation=args.drake_contact_surface_representation,
        )

    return make_drake_pipeline, resolution_hint


def _runner_args(args: argparse.Namespace) -> argparse.Namespace:
    runner_args = argparse.Namespace(**vars(args))
    runner_args.solver_backend = args.solver
    runner_args.output = args.output or _default_output_path(args.hydro_pipeline, args.solver)
    runner_args.solver_iterations = int(args.solver_iterations or (100 if args.solver == "sap_warp" else 15))
    if args.solver == "mujoco":
        runner_args.solver = args.mujoco_solver
    return runner_args


def _summary(
    rows: list[dict[str, float]],
    runner: Any,
    args: argparse.Namespace,
    resolution_hint: float | None,
) -> dict[str, Any]:
    final = rows[-1]
    first_contact_time = next(
        (row["time"] for row in rows if int(row["point_contacts"]) + int(row["hydro_contacts"]) > 0),
        None,
    )
    final_lateral = math.hypot(final["x"], final["y"])
    max_lateral = max(math.hypot(row["x"], row["y"]) for row in rows)
    max_hydro_contacts = max(int(row["hydro_contacts"]) for row in rows)

    solver_backend = getattr(args, "solver_backend", args.solver)
    hydro_label = "Newton Hydro" if args.hydro_pipeline == "newton" else "Drake Hydro"
    solver_label = "sap_warp SAP" if solver_backend == "sap_warp" else "Newton MuJoCo"
    summary = {
        "output": str(args.output),
        "scene": str(args.scene),
        "case": f"{hydro_label} + {solver_label}",
        "hydro_pipeline": args.hydro_pipeline,
        "solver": solver_backend,
        "contact_cap": int(args.contact_cap),
        "num_frames": int(args.num_frames),
        "frame_dt": runner.frame_dt,
        "sim_dt": runner.sim_dt,
        "first_contact_time": first_contact_time,
        "final_x": final["x"],
        "final_y": final["y"],
        "final_z": final["z"],
        "final_lateral": final_lateral,
        "max_lateral": max_lateral,
        "final_linear_speed": final["linear_speed"],
        "final_angular_speed": final["angular_speed"],
        "final_hydro_contacts": int(final["hydro_contacts"]),
        "max_hydro_contacts": max_hydro_contacts,
    }
    if args.hydro_pipeline == "drake":
        summary["drake_contact_surface_representation"] = args.drake_contact_surface_representation
        summary["drake_resolution_hint"] = resolution_hint
    if solver_backend == "sap_warp":
        summary["sap_iterations"] = int(args.solver_iterations)
        summary["contact_tau_d"] = args.contact_tau_d
        summary["contact_preset"] = args.contact_preset
        summary["line_search"] = args.line_search
    else:
        summary["mujoco_solver"] = args.mujoco_solver
        summary["mujoco_iterations"] = int(args.solver_iterations)
        summary["line_search_iterations"] = int(args.line_search_iterations)
    return summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    runner_args = _runner_args(args)
    runner_cls = SphereBoxSapRun if args.solver == "sap_warp" else SphereBoxMuJoCoRun
    scene = _load_scene(runner_args.scene)
    collision_pipeline_factory, resolution_hint = _collision_pipeline_factory(scene, runner_args)

    runner = runner_cls(scene, runner_args, collision_pipeline_factory=collision_pipeline_factory)
    rows = [runner.sample_row(0.0)]
    for frame in range(1, int(runner_args.num_frames) + 1):
        runner.step_frame()
        rows.append(runner.sample_row(frame * runner.frame_dt))

    _write_rows(runner_args.output, rows)
    summary = _summary(rows, runner, runner_args, resolution_hint)
    if not runner_args.quiet:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hydro-pipeline", choices=["newton", "drake"], default="newton", help="Hydro contact producer."
    )
    parser.add_argument(
        "--solver", choices=["sap_warp", "mujoco"], default="sap_warp", help="Newton-side solver backend."
    )
    parser.add_argument("--scene", type=Path, default=_default_scene_path(), help="Scene YAML path.")
    parser.add_argument("--output", type=Path, default=None, help="Output CSV path.")
    parser.add_argument("--num-frames", type=int, default=600, help="Number of output frames.")
    parser.add_argument("--fps", type=float, default=120.0, help="Output frame rate.")
    parser.add_argument("--substeps", type=int, default=4, help="Simulation substeps per output frame.")
    parser.add_argument("--contact-cap", type=int, default=6000, help="Maximum rigid contacts.")
    parser.add_argument("--broad-phase", choices=["nxn", "sap"], default="nxn", help="Newton broad phase.")

    parser.add_argument(
        "--solver-iterations",
        type=int,
        default=None,
        help="Solver iterations. Defaults to 100 for sap_warp and 15 for MuJoCo.",
    )

    parser.add_argument("--contact-beta", type=float, default=1.0, help="sap_warp normal regularization beta.")
    parser.add_argument("--contact-sigma", type=float, default=1.0e-3, help="sap_warp tangential regularization sigma.")
    parser.add_argument(
        "--contact-preset",
        choices=["approx32", "approx64", "drake"],
        default="drake",
        help="sap_warp contact preset.",
    )
    parser.add_argument(
        "--contact-weight-mode",
        choices=["body_inertia", "diag_delassus"],
        default=None,
        help="Override sap_warp contact weight mode. Defaults to the selected preset.",
    )
    parser.add_argument(
        "--line-search",
        choices=["monotone_decay", "armijo_decay", "exact_root"],
        default="exact_root",
        help="sap_warp line search.",
    )
    parser.add_argument(
        "--contact-tau-d",
        type=float,
        default=0.1,
        help="sap_warp per-shape contact relaxation/dissipation time scale.",
    )

    parser.add_argument("--collide-every", type=int, default=1, help="MuJoCo collision update interval in substeps.")
    parser.add_argument("--mujoco-solver", default="newton", help="MuJoCo solver.")
    parser.add_argument("--integrator", default="implicitfast", help="MuJoCo integrator.")
    parser.add_argument("--cone", default="elliptic", help="MuJoCo cone model.")
    parser.add_argument("--line-search-iterations", type=int, default=100, help="MuJoCo line-search iterations.")
    parser.add_argument("--impratio", type=float, default=1.0, help="MuJoCo impedance ratio.")

    parser.add_argument(
        "--drake-resolution-hint",
        type=float,
        default=None,
        help="Drake Hydro resolution_hint [m]. Defaults to the YAML SDF target voxel size.",
    )
    parser.add_argument(
        "--drake-contact-surface-representation",
        choices=["triangle", "polygon"],
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
