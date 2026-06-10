# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Sphere-Box Hydroelastic Drop
#
# A compliant sphere is dropped onto a fixed, stiffer compliant box and comes
# to rest through SDF-based hydroelastic contact, solved with SolverMuJoCo.
# Physical inputs are read from assets/sphere_box_hydro.yaml; the matched
# modulus kh = E / H is derived here.
#
# Command: python -m newton.examples sphere_box_hydro
#
###########################################################################

import warp as wp
import yaml

import newton
import newton.examples
from newton.geometry import HydroelasticSDF


class Example:
    def __init__(self, viewer, args):
        self.fps = 120
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps
        # MuJoCo is most stable when contacts are refreshed every few substeps.
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

        # Matched-modulus mapping kh = E / H (H = elastic-foundation depth):
        # sphere -> radius, box -> min half-size.
        kh_sphere = e_sphere / radius
        kh_box = e_box / (min(box_full) / 2.0)

        self.sphere_start = sphere_pos

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

        # Fixed compliant box: a static world shape (body=-1) centered at the origin.
        hx, hy, hz = (v / 2.0 for v in box_full)
        builder.add_shape_box(body=-1, hx=hx, hy=hy, hz=hz, cfg=hydro_cfg(kh_box), label="box")

        # Compliant sphere on a free body, started above the box so it falls.
        self.sphere_body = builder.add_body(
            xform=wp.transform(wp.vec3(*self.sphere_start), wp.quat_identity()),
            label="sphere",
        )
        builder.add_shape_sphere(self.sphere_body, radius=radius, cfg=hydro_cfg(kh_sphere), label="sphere")

        self.model = builder.finalize()

        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            reduce_contacts=True,
            rigid_contact_max=6000,
            broad_phase="nxn",
            # Output the contact-surface triangles so the viewer can draw the isosurface.
            sdf_hydroelastic_config=HydroelasticSDF.Config(
                output_contact_surface=hasattr(viewer, "renderer"),
            ),
        )

        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            use_mujoco_contacts=False,
            solver="newton",
            integrator="implicitfast",
            cone="elliptic",
            njmax=6000,
            nconmax=6000,
            iterations=15,
            ls_iterations=100,
            impratio=1.0,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.contacts = self.collision_pipeline.contacts()
        self.collision_pipeline.collide(self.state_0, self.contacts)

        self.viewer.set_model(self.model)
        self.viewer.set_camera(
            pos=wp.vec3(0.4, -0.4, 0.35),
            pitch=-20.0,
            yaw=135.0,
        )
        # Draw the hydroelastic contact surface (toggle in the GL viewer's UI).
        if hasattr(self.viewer, "renderer"):
            self.viewer.show_hydro_contact_surface = True

        self.capture()

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for sub in range(self.sim_substeps):
            if sub % self.collide_every == 0:
                self.collision_pipeline.collide(self.state_0, self.contacts)
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        # Draws the contact-surface isosurface; no-op when show_hydro_contact_surface is False.
        self.viewer.log_hydro_contact_surface(
            self.collision_pipeline.hydroelastic_sdf.get_contact_surface()
            if self.collision_pipeline.hydroelastic_sdf is not None
            else None,
            penetrating_only=True,
        )
        self.viewer.end_frame()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        # Default to ~5 s of simulation (120 fps).
        parser.set_defaults(num_frames=600)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)

    newton.examples.run(Example(viewer, args), args)
