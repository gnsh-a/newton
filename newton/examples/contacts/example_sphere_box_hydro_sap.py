# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Sphere-Box Hydroelastic Drop (SAP solver smoke test)
#
# Same scene as example_sphere_box_hydro.py, but stepped with the external
# sap_warp SolverSAP (github.com/sap-sim/sap_warp) instead of SolverMuJoCo.
# The native Newton SDF hydroelastic pipeline produces the contacts; SAP
# consumes the Newton Contacts buffer directly.
#
# Smoke test only: this validates the API/contact plumbing and stability for
# hydroelastic contacts consumed by SolverSAP.
#
# Differences from the MuJoCo example:
#   1. sap_model/state/control wrap the Newton objects (zero-copy aliasing);
#      newton and SAP states are swapped in lockstep.
#   2. No CUDA-graph capture (run eager) to keep the first integration simple.
#
# Command: python newton/examples/contacts/example_sphere_box_hydro_sap.py --viewer null
#
###########################################################################

import os
import sys

import numpy as np
import warp as wp
import yaml

import newton
import newton.examples
from newton.geometry import HydroelasticSDF

# Make the external sap_warp package importable (clone-and-run repo).
SAP_WARP_ROOT = os.environ.get("SAP_WARP_ROOT", os.path.expanduser("~/work/sap_warp"))
if SAP_WARP_ROOT not in sys.path:
    sys.path.insert(0, SAP_WARP_ROOT)
from sim.sap_runtime import sap_control_from_newton, sap_model_from_newton, sap_state_from_newton
from sim.solver_sap import SolverSAP


class Example:
    def __init__(self, viewer, args):
        self.fps = 120
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.collide_every = 2

        self.viewer = viewer

        # --- Physical inputs (pure SI) read from the asset yaml ---
        with open(newton.examples.get_asset("sphere_box_hydro.yaml")) as f:
            scene = yaml.safe_load(f)
        radius = float(scene["sphere"]["radius"])
        e_sphere = float(scene["sphere"]["hydroelastic_modulus"])
        sphere_pos = [float(v) for v in scene["sphere"]["initial_position"]]
        box_full = [float(v) for v in scene["box"]["full_size"]]
        e_box = float(scene["box"]["hydroelastic_modulus"])
        friction = float(scene["material"]["friction"])
        voxel = float(scene["mesh"]["sdf_target_voxel_size"])
        narrow_band = tuple(float(v) for v in scene["mesh"]["sdf_narrow_band_range"])

        # Matched-modulus mapping kh = E / H.
        kh_sphere = e_sphere / radius
        kh_box = e_box / (min(box_full) / 2.0)

        self.sphere_start = sphere_pos
        self.reduce_contacts = (args.reduce or "off") == "on"
        self.quiet = args.quiet

        def hydro_cfg(kh):
            return newton.ModelBuilder.ShapeConfig(
                is_hydroelastic=True,
                kh=kh,
                sdf_target_voxel_size=voxel,
                sdf_narrow_band_range=narrow_band,
                mu=friction,
                gap=0.0,
            )

        builder = newton.ModelBuilder()

        # Fixed compliant box: static world shape (body=-1) centered at the origin.
        hx, hy, hz = (v / 2.0 for v in box_full)
        builder.add_shape_box(body=-1, hx=hx, hy=hy, hz=hz, cfg=hydro_cfg(kh_box), label="box")

        # Compliant sphere on a free body, started above the box so it falls.
        self.sphere_body = builder.add_body(
            xform=wp.transform(wp.vec3(*self.sphere_start), wp.quat_identity()),
            label="sphere",
        )
        builder.add_shape_sphere(self.sphere_body, radius=radius, cfg=hydro_cfg(kh_sphere), label="sphere")

        self.model = builder.finalize()

        contact_cap = 6000
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            reduce_contacts=self.reduce_contacts,
            rigid_contact_max=contact_cap,
            broad_phase="nxn",
            sdf_hydroelastic_config=HydroelasticSDF.Config(
                reduce_contacts=self.reduce_contacts,
                output_contact_surface=hasattr(viewer, "renderer"),
            ),
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # --- SAP solver (external sap_warp) ---
        self.sap_model = sap_model_from_newton(self.model)
        self.solver = SolverSAP(
            self.sap_model,
            max_rigid_contact=contact_cap,
            max_iterations=int(args.solver_iterations),
            contact_preset_variant=(args.contact_preset or None),
        )
        # Zero-copy wrappers over the Newton state/control arrays.
        self.sap_state_0 = sap_state_from_newton(self.state_0)
        self.sap_state_1 = sap_state_from_newton(self.state_1)
        self.sap_control = sap_control_from_newton(self.control)

        self.contacts = self.collision_pipeline.contacts()
        self.collision_pipeline.collide(self.state_0, self.contacts)

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(0.4, -0.4, 0.35), pitch=-20.0, yaw=135.0)
        if hasattr(self.viewer, "renderer"):
            self.viewer.show_hydro_contact_surface = True

        # Run eager (no CUDA-graph capture) for the first SAP integration.
        self.graph = None

    def simulate(self):
        for sub in range(self.sim_substeps):
            if sub % self.collide_every == 0:
                self.collision_pipeline.collide(self.state_0, self.contacts)
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.sap_state_0, self.sap_state_1, self.sap_control, self.contacts, self.sim_dt)
            # Swap newton and SAP states in lockstep (SAP states alias newton arrays).
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.sap_state_0, self.sap_state_1 = self.sap_state_1, self.sap_state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt
        dbg = getattr(self, "_dbg_frames", 0)
        if not self.quiet and dbg < 25:
            z = float(self.state_0.body_q.numpy()[self.sphere_body][2])
            cc = int(self.contacts.rigid_contact_count.numpy()[0])
            print(
                f"[dbg] frame={dbg:3d} z={z:+.5f} contacts.count={cc} "
                f"sap_last={self.solver.last_contact_count} conv={self.solver.last_converged}"
            )
            self._dbg_frames = dbg + 1

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        hydro_sdf = getattr(self.collision_pipeline, "hydroelastic_sdf", None)
        self.viewer.log_hydro_contact_surface(
            hydro_sdf.get_contact_surface() if hydro_sdf is not None else None,
            penetrating_only=True,
        )
        self.viewer.end_frame()

    def test_final(self):
        q = self.state_0.body_q.numpy()[self.sphere_body]
        qd = self.state_0.body_qd.numpy()[self.sphere_body]
        pos = q[:3]
        lin_vel = qd[3:6]  # Newton body_qd = [angular(3), linear(3)]
        speed = float(np.linalg.norm(lin_vel))
        z = float(pos[2])
        rest_z = 0.1 + 0.05  # box top + sphere radius
        print(f"[test_final] sphere pos={pos}  lin_vel={lin_vel}")
        print(
            f"[test_final] z={z:.5f} (expected rest ~{rest_z:.3f})  speed={speed:.5f}  "
            f"last_contacts={self.solver.last_contact_count}  converged={self.solver.last_converged}"
        )
        assert np.all(np.isfinite(q)) and np.all(np.isfinite(qd)), "non-finite state (blew up)"
        assert 0.12 < z < 0.16, f"sphere not resting on box top (z={z:.4f})"
        assert speed < 0.05, f"sphere not settled (speed={speed:.4f})"
        print("[test_final] PASS: sphere settled on box via SAP + hydroelastic contacts")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=600)
        parser.add_argument(
            "--reduce",
            choices=["on", "off"],
            default=None,
            help="Native contact reduction. Defaults to off.",
        )
        parser.add_argument(
            "--solver-iterations",
            type=int,
            default=30,
            help="SAP solver max iterations per step.",
        )
        parser.add_argument(
            "--contact-preset",
            type=str,
            default="",
            help="SolverSAP contact_preset_variant (e.g. 'drake'); empty = solver default.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)

    newton.examples.run(Example(viewer, args), args)
