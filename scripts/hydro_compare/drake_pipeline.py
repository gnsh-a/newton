# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Drake Hydro contact producer that fills a Newton ``Contacts`` buffer."""

from __future__ import annotations

from typing import Any

import numpy as np

import newton
from newton import GeoType

_SLIVER_AREA = 1.0e-14
_GRAD_EPS = 1.0e-14


def _import_pydrake() -> dict[str, Any]:
    try:
        from pydrake.common.eigen_geometry import Quaternion  # noqa: PLC0415
        from pydrake.geometry import (  # noqa: PLC0415
            AddCompliantHydroelasticProperties,
            Box,
            FramePoseVector,
            GeometryFrame,
            GeometryInstance,
            HydroelasticContactRepresentation,
            ProximityProperties,
            SceneGraph,
            Sphere,
        )
        from pydrake.math import RigidTransform, RotationMatrix  # noqa: PLC0415
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("pydrake"):
            raise SystemExit(
                "pydrake is not installed in this environment. Run with the main checkout env, e.g.\n"
                "  UV_CACHE_DIR=/tmp/uv-cache uv run --project /home/ganesharivoli/work/newton-gnsh "
                "--no-sync python scripts/hydro_compare/drake_hydro_sap_warp_sphere_box.py"
            ) from exc
        raise

    return {
        "AddCompliantHydroelasticProperties": AddCompliantHydroelasticProperties,
        "Box": Box,
        "FramePoseVector": FramePoseVector,
        "GeometryFrame": GeometryFrame,
        "GeometryInstance": GeometryInstance,
        "HydroelasticContactRepresentation": HydroelasticContactRepresentation,
        "ProximityProperties": ProximityProperties,
        "Quaternion": Quaternion,
        "RigidTransform": RigidTransform,
        "RotationMatrix": RotationMatrix,
        "SceneGraph": SceneGraph,
        "Sphere": Sphere,
    }


def _rigid_transform(t7: np.ndarray, drake: dict[str, Any]):
    """Convert Newton ``[px, py, pz, qx, qy, qz, qw]`` to Drake ``RigidTransform``."""
    p = np.asarray(t7[:3], dtype=float)
    q = np.asarray(t7[3:7], dtype=float)
    norm = np.linalg.norm(q)
    q = q / norm if norm > 0.0 else np.array([0.0, 0.0, 0.0, 1.0])
    quat = drake["Quaternion"](w=q[3], x=q[0], y=q[1], z=q[2])
    return drake["RigidTransform"](drake["RotationMatrix"](quat), p)


def _foundation_depth(geo_type: int, scale: np.ndarray) -> float:
    """Return the Drake elastic-foundation depth represented by a Newton shape."""
    if geo_type == int(GeoType.SPHERE):
        return float(scale[0])
    if geo_type == int(GeoType.BOX):
        return float(min(scale[0], scale[1], scale[2]))
    raise NotImplementedError(f"DrakeContactPipeline supports SPHERE/BOX only, got GeoType={geo_type}")


def _drake_shape(geo_type: int, scale: np.ndarray, drake: dict[str, Any]):
    if geo_type == int(GeoType.SPHERE):
        return drake["Sphere"](float(scale[0]))
    if geo_type == int(GeoType.BOX):
        return drake["Box"](2.0 * float(scale[0]), 2.0 * float(scale[1]), 2.0 * float(scale[2]))
    raise NotImplementedError(f"DrakeContactPipeline supports SPHERE/BOX only, got GeoType={geo_type}")


def _hydro_representation(value: str, drake: dict[str, Any]):
    mode = str(value).strip().lower()
    representation = drake["HydroelasticContactRepresentation"]
    if mode == "triangle":
        return representation.kTriangle
    if mode == "polygon":
        return representation.kPolygon
    raise ValueError(f"Unsupported Drake Hydro contact surface representation {value!r}.")


class DrakeContactPipeline:
    """Mirror Newton sphere/box shapes into Drake SceneGraph and emit Newton contacts.

    The output buffer matches the subset consumed by ``sap_warp`` SAP and
    ``SolverMuJoCo(use_mujoco_contacts=False)``: shape ids, body-frame witness
    points, normal, margins, and per-face hydro stiffness.
    """

    def __init__(
        self,
        model: Any,
        *,
        resolution_hint: float,
        rigid_contact_max: int = 6000,
        contact_surface_representation: str = "polygon",
    ):
        self.model = model
        self.device = model.device
        self.rigid_contact_max = int(rigid_contact_max)
        self._drake = _import_pydrake()
        self._representation = _hydro_representation(contact_surface_representation, self._drake)

        self._shape_type = model.shape_type.numpy()
        self._shape_scale = model.shape_scale.numpy()
        self._shape_body = model.shape_body.numpy()
        self._shape_kh = model.shape_material_kh.numpy()
        self._shape_margin = model.shape_margin.numpy()
        shape_xform = model.shape_transform.numpy()
        self.shape_count = int(model.shape_count)

        self._shape_to_body = [_rigid_transform(shape_xform[i], self._drake) for i in range(self.shape_count)]

        self._scene_graph = self._drake["SceneGraph"]()
        self._source = self._scene_graph.RegisterSource("newton_drake_pipeline")
        self._frame = []
        self._geom_to_shape: dict[int, int] = {}
        for shape_id in range(self.shape_count):
            geo_type = int(self._shape_type[shape_id])
            scale = self._shape_scale[shape_id]
            frame_id = self._scene_graph.RegisterFrame(self._source, self._drake["GeometryFrame"](f"shape_{shape_id}"))
            geometry_id = self._scene_graph.RegisterGeometry(
                self._source,
                frame_id,
                self._drake["GeometryInstance"](
                    self._drake["RigidTransform"](),
                    _drake_shape(geo_type, scale, self._drake),
                    f"geo_{shape_id}",
                ),
            )
            modulus = float(self._shape_kh[shape_id]) * _foundation_depth(geo_type, scale)
            properties = self._drake["ProximityProperties"]()
            self._drake["AddCompliantHydroelasticProperties"](float(resolution_hint), modulus, properties)
            self._scene_graph.AssignRole(self._source, geometry_id, properties)
            self._frame.append(frame_id)
            self._geom_to_shape[geometry_id.get_value()] = shape_id

        self._context = self._scene_graph.CreateDefaultContext()
        self.last_faces: dict[str, np.ndarray] | None = None

    def contacts(self):
        """Allocate a Newton ``Contacts`` buffer with per-contact stiffness arrays."""
        return newton.Contacts(
            self.rigid_contact_max,
            0,
            per_contact_shape_properties=True,
            device=self.device,
        )

    def collide(self, state: Any, contacts: Any, capture_faces: bool = False):
        """Pose Drake from ``state.body_q`` and fill ``contacts`` with per-face hydro contacts."""
        contacts.clear()
        self.last_faces = None

        body_q = state.body_q.numpy()

        def body_pose(body_id: int):
            return self._drake["RigidTransform"]() if body_id < 0 else _rigid_transform(body_q[body_id], self._drake)

        body_poses = {int(body_id): body_pose(int(body_id)) for body_id in np.unique(self._shape_body)}

        poses = self._drake["FramePoseVector"]()
        for shape_id in range(self.shape_count):
            body_id = int(self._shape_body[shape_id])
            poses.set_value(self._frame[shape_id], body_poses[body_id].multiply(self._shape_to_body[shape_id]))
        self._scene_graph.get_source_pose_port(self._source).FixValue(self._context, poses)

        query = self._scene_graph.get_query_output_port().Eval(self._context)
        surfaces = query.ComputeContactSurfaces(self._representation)

        shape0_list: list[int] = []
        shape1_list: list[int] = []
        normal_list: list[np.ndarray] = []
        point0_list: list[np.ndarray] = []
        point1_list: list[np.ndarray] = []
        margin0_list: list[float] = []
        margin1_list: list[float] = []
        stiffness_list: list[float] = []
        phi0_list: list[float] = []
        grad_list: list[float] = []
        area_list: list[float] = []
        pressure_list: list[float] = []
        centroid_list: list[np.ndarray] = []

        for surface in surfaces:
            field = surface.tri_e_MN() if surface.is_triangle() else surface.poly_e_MN()
            shape_m = self._geom_to_shape[surface.id_M().get_value()]
            shape_n = self._geom_to_shape[surface.id_N().get_value()]
            body_m = int(self._shape_body[shape_m])
            body_n = int(self._shape_body[shape_n])
            x_body_world_m = body_poses[body_m].inverse()
            x_body_world_n = body_poses[body_n].inverse()

            for face_id in range(surface.num_faces()):
                area = float(surface.area(face_id))
                if area <= _SLIVER_AREA:
                    continue

                normal = np.asarray(surface.face_normal(face_id), dtype=float).reshape(3)
                grad_m = float(np.asarray(surface.EvaluateGradE_M_W(face_id), dtype=float).reshape(3) @ normal)
                grad_n = float(-(np.asarray(surface.EvaluateGradE_N_W(face_id), dtype=float).reshape(3) @ normal))
                if grad_m < _GRAD_EPS or grad_n < _GRAD_EPS:
                    continue

                grad = 1.0 / (1.0 / grad_m + 1.0 / grad_n)
                centroid = np.asarray(surface.centroid(face_id), dtype=float).reshape(3)
                pressure = float(field.EvaluateCartesian(face_id, centroid))
                phi0 = -pressure / grad

                point0_world = centroid - 0.5 * phi0 * normal
                point1_world = centroid + 0.5 * phi0 * normal

                shape0_list.append(shape_n)
                shape1_list.append(shape_m)
                normal_list.append(normal)
                point0_list.append(np.asarray(x_body_world_n.multiply(point0_world), dtype=float).reshape(3))
                point1_list.append(np.asarray(x_body_world_m.multiply(point1_world), dtype=float).reshape(3))
                margin0_list.append(float(self._shape_margin[shape_n]))
                margin1_list.append(float(self._shape_margin[shape_m]))
                stiffness_list.append(area * grad)
                phi0_list.append(phi0)
                grad_list.append(grad)
                area_list.append(area)
                pressure_list.append(pressure)
                centroid_list.append(centroid)

        contact_count = len(shape0_list)
        if contact_count > self.rigid_contact_max:
            print(
                f"[DrakeContactPipeline] {contact_count} contacts exceed rigid_contact_max "
                f"{self.rigid_contact_max}; truncating."
            )
            contact_count = self.rigid_contact_max

        shape0 = np.asarray(shape0_list[:contact_count], dtype=np.int32)
        shape1 = np.asarray(shape1_list[:contact_count], dtype=np.int32)
        normal = np.asarray(normal_list[:contact_count], dtype=float).reshape(-1, 3)

        if capture_faces:
            self.last_faces = {
                "phi0": np.asarray(phi0_list[:contact_count], dtype=float),
                "g": np.asarray(grad_list[:contact_count], dtype=float),
                "area": np.asarray(area_list[:contact_count], dtype=float),
                "p0": np.asarray(pressure_list[:contact_count], dtype=float),
                "centroid_w": np.asarray(centroid_list[:contact_count], dtype=float).reshape(-1, 3),
                "normal": normal,
                "shape0": shape0,
                "shape1": shape1,
            }

        if contact_count > 0:
            contacts.rigid_contact_shape0[:contact_count].assign(shape0)
            contacts.rigid_contact_shape1[:contact_count].assign(shape1)
            contacts.rigid_contact_normal[:contact_count].assign(normal.astype(np.float32))
            contacts.rigid_contact_point0[:contact_count].assign(
                np.asarray(point0_list[:contact_count], dtype=np.float32)
            )
            contacts.rigid_contact_point1[:contact_count].assign(
                np.asarray(point1_list[:contact_count], dtype=np.float32)
            )
            contacts.rigid_contact_margin0[:contact_count].assign(
                np.asarray(margin0_list[:contact_count], dtype=np.float32)
            )
            contacts.rigid_contact_margin1[:contact_count].assign(
                np.asarray(margin1_list[:contact_count], dtype=np.float32)
            )
            if contacts.rigid_contact_stiffness is not None:
                contacts.rigid_contact_stiffness[:contact_count].assign(
                    np.asarray(stiffness_list[:contact_count], dtype=np.float32)
                )
                contacts.rigid_contact_friction[:contact_count].assign(np.ones(contact_count, dtype=np.float32))

        contacts.rigid_contact_count.assign(np.array([contact_count], dtype=np.int32))
        return contacts
