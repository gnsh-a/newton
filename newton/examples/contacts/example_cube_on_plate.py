# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Cube on Plate
#
# One demo, two scenarios over the same hydroelastic cube + plate scene:
#
#   --scenario tip     (default)
#       Cube starts pre-penetrated by TIP_INITIAL_OVERLAP and a horizontal
#       force is applied at the top face, ramped at RAMP_RATE_N_PER_S N/s.
#       Past F = m·g/2 the cube tips analytically; past F = μ·m·g it would
#       slide. The demo probes rotation under quasi-static torque.
#
#   --scenario settle
#       Cube starts SETTLE_DROP_HEIGHT above the plate and falls under
#       gravity only. The demo probes translation under impact and the
#       resulting steady-state penetration δ_eq = m·g / (k_eff · L²).
#
# Scene and physics constants are loaded from a YAML config. The bundled
# ``configs/cube_on_plate_baseline.yaml`` is used by default. The top-level
# ``constants`` mapping is shared with the Drake demo; the top-level
# ``newton`` mapping contains Newton-specific parameters. A ``drake`` mapping
# may be present so the same file can be passed to both demos, but this
# example ignores it.
#
# Run modes:
#     python -m newton.examples cube_on_plate
#     python -m newton.examples cube_on_plate --simulation-time 2.0
#     python -m newton.examples cube_on_plate --scenario settle --config ...
#     python -m newton.examples cube_on_plate --no-reduce-contacts --config ...
#
# Command: python -m newton.examples cube_on_plate
#
###########################################################################

import math
from pathlib import Path

import numpy as np
import warp as wp
import yaml

import newton
import newton.examples
from newton.geometry import HydroelasticSDF

SCENARIO_TIP = "tip"
SCENARIO_SETTLE = "settle"
DEFAULT_CONFIG_PATH = Path(__file__).parent / "configs" / "cube_on_plate_baseline.yaml"
FRAME_FPS = 240
FRAME_COUNT_MARGIN = 20
RUN_SECONDS = 1.0
"""Default hard cap on sim_time [s] for both scenarios."""

# Scene and physics constants must be supplied by the YAML config
# (``--config``). They are set as module globals by ``_apply_config`` before
# ``Example.__init__`` reads them.
REQUIRED_CONSTANTS = (
    "CUBE_HALF_EXTENT",
    "CUBE_DENSITY",
    "PLATE_HALF_EXTENT",
    "PLATE_HALF_THICKNESS",
    "MU_SLIDING",
    "SDF_MAX_RESOLUTION",
    "SDF_NARROW_BAND_RANGE",
    "RAMP_RATE_N_PER_S",
    "TIP_INITIAL_OVERLAP",
    "SETTLE_DROP_HEIGHT",
)
REQUIRED_NEWTON = ("KH",)

RIGID_CONTACT_MAX = 16384
"""Upper bound on rigid contacts allocated by the collision pipeline."""
MUJOCO_NCONMAX = 16384
"""MuJoCo contact buffer capacity (must be ≥ RIGID_CONTACT_MAX)."""
MUJOCO_NJMAX = 32768
"""MuJoCo constraint Jacobian row capacity (≈ 3·active contacts under elliptic cone)."""

GRAVITY = 9.81
"""Acceleration of gravity used for analytic predictions [m/s²]."""

TIP_FINAL_TILT_DEG = 30.0
"""Tip test: cube must have tipped past this angle by RUN_SECONDS."""
SETTLE_FINAL_TILT_DEG = 0.5
"""Settle test: cube tilt must be below this at RUN_SECONDS."""
SETTLE_FINAL_DRIFT_M = 1.0e-4
"""Settle test: cube lateral drift |x|, |y| must be below this at RUN_SECONDS."""


def _load_config(config_path: str | Path) -> tuple[dict[str, object], dict[str, object]]:
    path = Path(config_path).resolve()
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise TypeError(f"config {path} must be a YAML mapping")
    constants = cfg.get("constants", {})
    if not isinstance(constants, dict):
        raise TypeError(f"config {path} constants must be a mapping")
    newton_constants = cfg.get("newton", {})
    if not isinstance(newton_constants, dict):
        raise TypeError(f"config {path} newton must be a mapping")

    required = set(REQUIRED_CONSTANTS)
    missing = required - set(constants)
    if missing:
        raise KeyError(f"config {path} is missing required constants keys: {sorted(missing)}")
    unknown = set(constants) - required
    if unknown:
        raise KeyError(
            f"config {path} has unknown constants keys: {sorted(unknown)}; expected one of {sorted(required)}."
        )

    required_newton = set(REQUIRED_NEWTON)
    missing_newton = required_newton - set(newton_constants)
    if missing_newton:
        raise KeyError(f"config {path} is missing required newton keys: {sorted(missing_newton)}")
    unknown_newton = set(newton_constants) - required_newton
    if unknown_newton:
        raise KeyError(
            f"config {path} has unknown newton keys: {sorted(unknown_newton)}; "
            f"expected one of {sorted(required_newton)}."
        )
    return constants, newton_constants


def _apply_config(config_path: str | Path) -> None:
    """Load a YAML config and bind its ``constants`` mapping to this module.

    Every name in ``REQUIRED_CONSTANTS`` must appear in ``constants`` and
    every name in ``REQUIRED_NEWTON`` must appear in ``newton``; missing keys
    raise. Unknown keys raise. There are no source-level defaults - the YAML
    is the sole source of truth for the listed names.
    """
    constants, newton_constants = _load_config(config_path)
    for key, value in constants.items():
        globals()[key] = tuple(value) if isinstance(value, list) else value
    for key, value in newton_constants.items():
        globals()[key] = tuple(value) if isinstance(value, list) else value


class Example:
    def __init__(self, viewer, args):
        _apply_config(args.config)

        self.viewer = viewer
        self.scenario = str(args.scenario)
        if self.scenario not in (SCENARIO_TIP, SCENARIO_SETTLE):
            raise ValueError(f"unknown scenario: {self.scenario!r}")
        self.run_seconds = float(args.simulation_time)
        if self.run_seconds <= 0.0:
            raise ValueError(f"simulation_time must be positive; got {self.run_seconds}")
        self.reduce_contacts = bool(args.reduce_contacts)
        self.pre_prune_contacts = self.reduce_contacts
        self.auto_close_after_freeze = bool(getattr(args, "test", False)) or bool(getattr(args, "headless", False))
        self._viewer_closed = False

        self.fps = FRAME_FPS
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 8
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        if hasattr(self.viewer, "num_frames"):
            min_frames = math.ceil(self.run_seconds * self.fps) + FRAME_COUNT_MARGIN
            self.viewer.num_frames = max(int(self.viewer.num_frames), min_frames)

        L = 2.0 * CUBE_HALF_EXTENT
        mass = CUBE_DENSITY * L * L * L
        self.cube_mass = mass
        self.weight_N = mass * GRAVITY
        self.ramp_rate = RAMP_RATE_N_PER_S

        shape_cfg = newton.ModelBuilder.ShapeConfig(
            margin=0.0,
            gap=0.0005,
            mu=MU_SLIDING,
            mu_torsional=0.0,
            mu_rolling=0.0,
            ke=1.0e7,
            kd=1.0e5,
            density=CUBE_DENSITY,
            is_hydroelastic=True,
            kh=KH,
            sdf_max_resolution=SDF_MAX_RESOLUTION,
            sdf_narrow_band_range=SDF_NARROW_BAND_RANGE,
        )

        builder = newton.ModelBuilder()
        builder.default_shape_cfg.gap = 0.001

        builder.add_shape_box(
            body=-1,
            xform=wp.transform(
                wp.vec3(0.0, 0.0, -PLATE_HALF_THICKNESS),
                wp.quat_identity(),
            ),
            hx=PLATE_HALF_EXTENT,
            hy=PLATE_HALF_EXTENT,
            hz=PLATE_HALF_THICKNESS,
            cfg=shape_cfg,
            color=(0.58, 0.72, 0.78),
            label="plate",
        )

        # Branch 1: initial cube z.
        if self.scenario == SCENARIO_TIP:
            cube_z = CUBE_HALF_EXTENT - TIP_INITIAL_OVERLAP
        else:
            cube_z = CUBE_HALF_EXTENT + SETTLE_DROP_HEIGHT

        self.cube_body = builder.add_body(
            xform=wp.transform(
                wp.vec3(0.0, 0.0, cube_z),
                wp.quat_identity(),
            ),
            label="cube",
        )
        builder.add_shape_box(
            body=self.cube_body,
            hx=CUBE_HALF_EXTENT,
            hy=CUBE_HALF_EXTENT,
            hz=CUBE_HALF_EXTENT,
            cfg=shape_cfg,
            color=(0.88, 0.52, 0.22),
            label="cube_shape",
        )

        # Free-joint qd layout: (lin_x, lin_y, lin_z, ang_x, ang_y, ang_z).
        self.cube_qd_start = int(builder.joint_qd_start[-1])

        self.model = builder.finalize()

        # Increase buffers so reduce-off can keep dense face contacts.
        hydro_cfg = HydroelasticSDF.Config(
            output_contact_surface=True,
            reduce_contacts=self.reduce_contacts,
            pre_prune_contacts=self.pre_prune_contacts,
            buffer_mult_iso=4,
            buffer_mult_contact=4,
        )
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            sdf_hydroelastic_config=hydro_cfg,
            rigid_contact_max=RIGID_CONTACT_MAX,
            broad_phase="sap",
        )

        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            use_mujoco_contacts=False,
            solver="newton",
            integrator="implicitfast",
            cone="elliptic",
            njmax=MUJOCO_NJMAX,
            nconmax=MUJOCO_NCONMAX,
            iterations=15,
            ls_iterations=100,
            impratio=1.0,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # Host scratch for the captured graph's control.joint_f buffer.
        # FREE joint_f layout: (fx, fy, fz, tau_x, tau_y, tau_z).
        joint_f_len = int(self.control.joint_f.shape[0])
        self._wrench_host = np.zeros(joint_f_len, dtype=np.float32)

        self.contacts = self.collision_pipeline.contacts()
        self.collision_pipeline.collide(self.state_0, self.contacts)

        self.viewer.set_model(self.model)
        self.viewer.show_contacts = True
        self.viewer.show_hydro_contact_surface = True
        self.viewer.show_collision = False
        self.viewer.show_triangles = False
        self.viewer.show_visual = True
        if hasattr(self.viewer, "renderer"):
            self.viewer.renderer.draw_wireframe = True
        self.viewer.set_camera(
            pos=wp.vec3(0.25, -0.45, 0.18),
            pitch=-15.0,
            yaw=120.0,
        )

        self.capture()

    def _apply_wrench(self) -> None:
        # Branch 2: tip ramps a top-face force; settle applies no wrench.
        self._wrench_host[:] = 0.0
        if self.scenario == SCENARIO_TIP:
            F = self.ramp_rate * self.sim_time
            self._wrench_host[self.cube_qd_start + 0] = F
            self._wrench_host[self.cube_qd_start + 4] = F * CUBE_HALF_EXTENT
        self.control.joint_f.assign(self._wrench_host)

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
        # Refresh contacts so the viewer reflects the advanced state.
        self.collision_pipeline.collide(self.state_0, self.contacts)

    def step(self):
        if self.sim_time > self.run_seconds:
            if self.auto_close_after_freeze and not self._viewer_closed:
                self.viewer.close()
                self._viewer_closed = True
            return

        # Captured graph reads the control.joint_f values assigned here.
        self._apply_wrench()

        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        if self.collision_pipeline.hydroelastic_sdf is not None:
            self.viewer.log_hydro_contact_surface(
                self.collision_pipeline.hydroelastic_sdf.get_contact_surface(),
            )
        self.viewer.end_frame()

    def test_final(self):
        body_q = self.state_0.body_q.numpy()[self.cube_body]
        x = float(body_q[0])
        y = float(body_q[1])
        z = float(body_q[2])
        qx = float(body_q[3])
        qy = float(body_q[4])
        local_z_z = max(-1.0, min(1.0, 1.0 - 2.0 * (qx * qx + qy * qy)))
        tilt_deg = math.degrees(math.acos(local_z_z))

        assert math.isfinite(x) and math.isfinite(y) and math.isfinite(z), (
            f"cube position contains NaN/Inf: ({x}, {y}, {z})"
        )
        assert math.isfinite(tilt_deg), f"cube tilt contains NaN/Inf: {tilt_deg}"

        # Branch 3: scenario-specific final-state assertion.
        if self.scenario == SCENARIO_TIP:
            assert tilt_deg > TIP_FINAL_TILT_DEG, (
                f"cube did not tip by t={self.sim_time:.3f} s (tilt={tilt_deg:.2f}°, threshold={TIP_FINAL_TILT_DEG}°)"
            )
        else:
            assert tilt_deg < SETTLE_FINAL_TILT_DEG, (
                f"cube did not stay upright after settle (tilt={tilt_deg:.4f}°, threshold={SETTLE_FINAL_TILT_DEG}°)"
            )
            assert abs(x) < SETTLE_FINAL_DRIFT_M and abs(y) < SETTLE_FINAL_DRIFT_M, (
                f"cube drifted laterally during settle: "
                f"x={x * 1000:.4f} mm, y={y * 1000:.4f} mm "
                f"(threshold={SETTLE_FINAL_DRIFT_M * 1000:.4f} mm)"
            )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=math.ceil(RUN_SECONDS * FRAME_FPS) + FRAME_COUNT_MARGIN)
        import argparse  # noqa: PLC0415

        parser.add_argument(
            "--scenario",
            choices=[SCENARIO_TIP, SCENARIO_SETTLE],
            default=SCENARIO_TIP,
            help=(
                "Which physical scenario to run. 'tip': ramped horizontal force at the "
                "top face, cube starts pre-penetrated. 'settle': no applied force, cube "
                "starts above the plate and falls under gravity."
            ),
        )
        parser.add_argument(
            "--reduce-contacts",
            action=argparse.BooleanOptionalAction,
            default=True,
            help=(
                "Enable HydroelasticSDF.Config.reduce_contacts (default True). Use "
                "--no-reduce-contacts to keep all marching-cubes face contacts and "
                "disable local pre-prune."
            ),
        )
        parser.add_argument(
            "--config",
            type=str,
            default=str(DEFAULT_CONFIG_PATH),
            help="Path to a YAML config supplying every scene and physics constant. Defaults to the bundled baseline.",
        )
        parser.add_argument(
            "--simulation-time",
            type=float,
            default=RUN_SECONDS,
            help="Simulated duration in seconds before physics freezes.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
