# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Hydroelastic Sphere Pair
#
# Minimal demonstration of the hydroelastic SDF contact pipeline.
#
# A static icosphere sits on the ground; a second icosphere falls onto it
# under gravity.  Both shapes are flagged hydroelastic, so contact runs
# through the full SDF pipeline: broadphase -> octree iso-voxels ->
# marching cubes contact surface -> contact generation -> reduction ->
# MuJoCo solver.  The marching-cubes contact polygon is rendered each
# frame via ``viewer.log_hydro_contact_surface``.
#
# Command: python -m newton.examples hydro_sphere_pair
#
###########################################################################

import numpy as np
import trimesh
import warp as wp

import newton
import newton.examples
from newton.geometry import HydroelasticSDF

# Sphere geometry and scene layout
SPHERE_RADIUS = 0.05
"""Radius of each icosphere [m]."""
INITIAL_GAP = 0.05
"""Initial clear-air gap between the two spheres before the drop [m]."""
ICOSPHERE_SUBDIVISIONS = 3
"""Icosphere refinement level (3 -> 1280 triangles)."""

# SDF baking parameters
SDF_MAX_RESOLUTION = 64
"""Maximum SDF grid resolution along the longest mesh axis."""
SDF_NARROW_BAND_RANGE = (-0.005, 0.005)
"""SDF narrow-band range [m] used during baking."""
SDF_MARGIN = 0.005
"""Outward margin baked into the SDF surface [m]."""

# Hydroelastic material
KH = 1.0e9
"""Hydroelastic stiffness coefficient [Pa].

Note: Drake's ``examples/hydroelastic/two_spheres`` uses
``hydroelastic_modulus = 5e5 Pa``. Newton's ``kh`` enters the contact
constraint as ``contact_stiffness = area * k_eff`` (N/m) and is *not*
unit-equivalent to Drake's pressure-field modulus — setting ``kh = 5e5``
here produces a contact too weak to catch the dropper (depth saturates
the SDF narrow band and the body tunnels)."""
SPHERE_DENSITY = 191.0
"""Density [kg/m^3] chosen so a 0.05 m sphere weighs 0.1 kg, matching Drake."""

# Pipeline allocation
RIGID_CONTACT_MAX = 1024
"""Upper bound on rigid contacts allocated by the collision pipeline."""


def _build_icosphere_mesh(radius, subdivisions, sdf_resolution, narrow_band_range, margin):
    tm = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    vertices = np.asarray(tm.vertices, dtype=np.float32)
    indices = np.asarray(tm.faces, dtype=np.int32).flatten()
    mesh = newton.Mesh(vertices, indices)
    mesh.build_sdf(
        max_resolution=sdf_resolution,
        narrow_band_range=narrow_band_range,
        margin=margin,
    )
    return mesh


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer

        # Timing
        self.fps = 120
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        # Shared icosphere mesh (baked SDF) reused by both shapes.
        mesh = _build_icosphere_mesh(
            radius=SPHERE_RADIUS,
            subdivisions=ICOSPHERE_SUBDIVISIONS,
            sdf_resolution=SDF_MAX_RESOLUTION,
            narrow_band_range=SDF_NARROW_BAND_RANGE,
            margin=SDF_MARGIN,
        )

        shape_cfg = newton.ModelBuilder.ShapeConfig(
            margin=0.0,
            gap=0.001,
            mu=0.5,
            ke=1.0e7,
            kd=1.0e4,
            density=SPHERE_DENSITY,
            is_hydroelastic=True,
            kh=KH,
        )

        builder = newton.ModelBuilder()
        builder.default_shape_cfg.gap = 0.001

        # Static floor sphere: attached to body=-1 (worldbody), no DoF and
        # no joint required to hold it in place.
        builder.add_shape_mesh(
            body=-1,
            xform=wp.transform(
                wp.vec3(0.0, 0.0, SPHERE_RADIUS),
                wp.quat_identity(),
            ),
            mesh=mesh,
            cfg=shape_cfg,
            label="floor_sphere",
        )

        # Falling sphere: free 6-DoF rigid body starting above the floor
        # sphere by INITIAL_GAP of clear space.
        self.initial_dropper_z = 3.0 * SPHERE_RADIUS + INITIAL_GAP
        self.dropper_body = builder.add_body(
            xform=wp.transform(
                wp.vec3(0.0, 0.0, self.initial_dropper_z),
                wp.quat_identity(),
            ),
            label="dropper",
        )
        builder.add_shape_mesh(
            body=self.dropper_body,
            mesh=mesh,
            cfg=shape_cfg,
            label="dropper_sphere",
        )

        self.model = builder.finalize()

        # Collision pipeline with the marching-cubes contact polygon
        # exposed so the viewer can render it each frame.
        hydro_cfg = HydroelasticSDF.Config(
            output_contact_surface=True,
            reduce_contacts=True,
        )
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            sdf_hydroelastic_config=hydro_cfg,
            rigid_contact_max=RIGID_CONTACT_MAX,
            broad_phase="sap",
        )

        # MuJoCo solver driving Newton's hydroelastic contacts.
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            use_mujoco_contacts=False,
            solver="newton",
            integrator="implicitfast",
            cone="elliptic",
            njmax=200,
            nconmax=200,
            iterations=15,
            ls_iterations=100,
            impratio=1.0,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # Allocate contact buffers and do one initial collide pass so the
        # buffers are populated before the first step.
        self.contacts = self.collision_pipeline.contacts()
        self.collision_pipeline.collide(self.state_0, self.contacts)

        # Viewer setup. ``show_hydro_contact_surface`` is False on the base
        # viewer by default; flip it on so the marching-cubes contact patch
        # rendered by ``log_hydro_contact_surface`` is actually visible.
        self.viewer.set_model(self.model)
        self.viewer.show_hydro_contact_surface = True
        self.viewer.set_camera(
            pos=wp.vec3(0.4, -0.4, 0.2),
            pitch=-15.0,
            yaw=135.0,
        )

        self.max_reduced_contacts = 0
        self.max_face_contacts = 0

        self.capture()

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
        if self.collision_pipeline.hydroelastic_sdf is not None:
            self.viewer.log_hydro_contact_surface(
                self.collision_pipeline.hydroelastic_sdf.get_contact_surface(),
            )
        self.viewer.end_frame()

    def test_post_step(self):
        """Track per-step maxima for ``test_final`` assertions."""
        reduced = int(self.contacts.rigid_contact_count.numpy()[0])
        self.max_reduced_contacts = max(self.max_reduced_contacts, reduced)
        surf = self.collision_pipeline.hydroelastic_sdf.get_contact_surface()
        raw = int(surf.face_contact_count.numpy()[0])
        self.max_face_contacts = max(self.max_face_contacts, raw)

    def test_final(self):
        """Verify the dropper fell, was stopped by hydroelastic contact, and the pipeline produced contacts."""
        final_z = float(self.state_0.body_q.numpy()[self.dropper_body, 2])

        fall_threshold = self.initial_dropper_z - INITIAL_GAP + 0.005
        assert final_z < fall_threshold, (
            f"dropper did not fall through initial gap: final z={final_z:.4f} m, expected < {fall_threshold:.4f} m"
        )

        assert final_z > SPHERE_RADIUS, (
            f"dropper tunneled through static sphere: final z={final_z:.4f} m, expected > {SPHERE_RADIUS:.4f} m"
        )

        assert self.max_face_contacts > 0, (
            f"no hydroelastic face contacts were generated during the run (max_face_contacts={self.max_face_contacts})"
        )

        assert self.max_reduced_contacts > 0, (
            f"no reduced contacts reached the solver during the run (max_reduced_contacts={self.max_reduced_contacts})"
        )

    @staticmethod
    def create_parser():
        return newton.examples.create_parser()


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
