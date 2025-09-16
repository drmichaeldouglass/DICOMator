"""Mesh voxelization helpers used by the DICOMator add-on."""
from __future__ import annotations

import math
from typing import Callable, Iterable, Optional, Sequence, Tuple

import bpy
import numpy as np
from bpy.types import Object
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from .constants import (
    AIR_DENSITY,
    DEFAULT_DENSITY,
    MAX_HU_VALUE,
    MIN_HU_VALUE,
)

VectorLike = Sequence[float]
VoxelSize = Sequence[float]


def _bvh_from_object(obj: Object, depsgraph: Optional[bpy.types.Depsgraph] = None, *, apply_modifiers: bool = False) -> BVHTree:
    """Build a BVH tree in world space for ``obj``."""
    if apply_modifiers and depsgraph is not None:
        obj_eval = obj.evaluated_get(depsgraph)
        mesh = obj_eval.to_mesh(preserve_all_data_layers=False, depsgraph=depsgraph)
        try:
            verts_world = [obj_eval.matrix_world @ vert.co for vert in mesh.vertices]
            polygons = [list(poly.vertices) for poly in mesh.polygons]
        finally:
            obj_eval.to_mesh_clear()
    else:
        mesh = obj.data
        verts_world = [obj.matrix_world @ vert.co for vert in mesh.vertices]
        polygons = [list(poly.vertices) for poly in mesh.polygons]
    return BVHTree.FromPolygons(verts_world, polygons)


def _bvh_from_object_base(obj: Object) -> BVHTree:
    """Build a BVH from the object's base mesh without evaluating modifiers."""
    mesh = obj.data
    verts_world = [obj.matrix_world @ vert.co for vert in mesh.vertices]
    polygons = [list(poly.vertices) for poly in mesh.polygons]
    return BVHTree.FromPolygons(verts_world, polygons)


def voxelize_mesh(obj: Object, voxel_size: VectorLike | float = 1.0, padding: int = 1) -> Tuple[np.ndarray, Vector, Tuple[int, int, int]]:
    """Voxelize ``obj`` by casting vertical rays."""
    if isinstance(voxel_size, Iterable) and len(voxel_size) == 3:
        vx, vy, vz = (float(component) for component in voxel_size)
    else:
        vx = vy = vz = float(voxel_size)

    bvh = _bvh_from_object_base(obj)

    bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    min_x = min(corner.x for corner in bbox_corners) - padding * vx
    max_x = max(corner.x for corner in bbox_corners) + padding * vx
    min_y = min(corner.y for corner in bbox_corners) - padding * vy
    max_y = max(corner.y for corner in bbox_corners) + padding * vy
    min_z = min(corner.z for corner in bbox_corners) - padding * vz
    max_z = max(corner.z for corner in bbox_corners) + padding * vz

    width = max(1, int(math.ceil((max_x - min_x) / vx)))
    height = max(1, int(math.ceil((max_y - min_y) / vy)))
    depth = max(1, int(math.ceil((max_z - min_z) / vz)))

    voxel_array = np.zeros((width, height, depth), dtype=np.uint8)
    origin = Vector((min_x, min_y, min_z))

    xs = min_x + (np.arange(width) + 0.5) * vx
    ys = min_y + (np.arange(height) + 0.5) * vy

    z0_center = min_z + 0.5 * vz
    inv_dz = 1.0 / vz
    ray_dir = Vector((0.0, 0.0, 1.0))
    max_dist = (max_z - min_z) + 4.0 * vz
    epsilon = 1e-6

    total_columns = width * height
    processed_columns = 0

    for ix in range(width):
        x_world = float(xs[ix])
        for iy in range(height):
            y_world = float(ys[iy])
            origin_ray = Vector((x_world, y_world, min_z - 2.0 * vz))
            hits_z: list[float] = []

            while True:
                location, _normal, _face_index, _distance = bvh.ray_cast(origin_ray, ray_dir, max_dist)
                if location is None:
                    break
                hits_z.append(location.z)
                origin_ray = Vector((location.x, location.y, location.z + epsilon))

            if hits_z:
                hits_z.sort()
                for start in range(0, len(hits_z) - 1, 2):
                    lower = hits_z[start]
                    upper = hits_z[start + 1]
                    start_idx = int(math.ceil((lower - z0_center) * inv_dz))
                    end_idx = int(math.floor((upper - z0_center) * inv_dz))
                    if end_idx >= start_idx:
                        s = max(0, start_idx)
                        e = min(depth - 1, end_idx)
                        if e >= s:
                            voxel_array[ix, iy, s:e + 1] = 1

            processed_columns += 1
            if processed_columns % 10000 == 0:
                print(f"  processed columns: {processed_columns}/{total_columns} ({(processed_columns/total_columns)*100:.1f}%)")

    print(f"Voxelization complete. Filled voxels: {int(voxel_array.sum())}/{voxel_array.size}")
    return voxel_array, origin, (width, height, depth)


def voxelize_objects_to_hu(
    objects: Sequence[Object],
    voxel_size: VoxelSize | float = 1.0,
    padding: int = 1,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    bbox_override: Optional[Tuple[float, float, float, float, float, float]] = None,
    *,
    apply_modifiers: bool = False,
    depsgraph: Optional[bpy.types.Depsgraph] = None,
) -> Tuple[np.ndarray, Vector, Tuple[int, int, int]]:
    """Voxelize multiple objects into a single HU grid."""
    if not objects:
        raise ValueError("No objects provided for voxelization")

    if apply_modifiers and depsgraph is None:
        depsgraph = bpy.context.evaluated_depsgraph_get()

    if isinstance(voxel_size, Iterable) and len(voxel_size) == 3:
        vx, vy, vz = (float(component) for component in voxel_size)
    else:
        vx = vy = vz = float(voxel_size)

    if bbox_override is not None:
        min_x, max_x, min_y, max_y, min_z, max_z = bbox_override
    else:
        min_x = min_y = min_z = float('inf')
        max_x = max_y = max_z = float('-inf')
        for obj in objects:
            if apply_modifiers and depsgraph is not None:
                obj_eval = obj.evaluated_get(depsgraph)
                mesh = obj_eval.to_mesh(preserve_all_data_layers=False, depsgraph=depsgraph)
                try:
                    for vertex in mesh.vertices:
                        world_vertex = obj_eval.matrix_world @ vertex.co
                        min_x = min(min_x, world_vertex.x)
                        max_x = max(max_x, world_vertex.x)
                        min_y = min(min_y, world_vertex.y)
                        max_y = max(max_y, world_vertex.y)
                        min_z = min(min_z, world_vertex.z)
                        max_z = max(max_z, world_vertex.z)
                finally:
                    obj_eval.to_mesh_clear()
            else:
                for corner in obj.bound_box:
                    world_corner = obj.matrix_world @ Vector(corner)
                    min_x = min(min_x, world_corner.x)
                    max_x = max(max_x, world_corner.x)
                    min_y = min(min_y, world_corner.y)
                    max_y = max(max_y, world_corner.y)
                    min_z = min(min_z, world_corner.z)
                    max_z = max(max_z, world_corner.z)
        min_x -= padding * vx
        max_x += padding * vx
        min_y -= padding * vy
        max_y += padding * vy
        min_z -= padding * vz
        max_z += padding * vz

    width = max(1, int(math.ceil((max_x - min_x) / vx)))
    height = max(1, int(math.ceil((max_y - min_y) / vy)))
    depth = max(1, int(math.ceil((max_z - min_z) / vz)))
    origin = Vector((min_x, min_y, min_z))

    print(f"Voxelizing {len(objects)} objects into {width}x{height}x{depth} HU grid...")

    hu_array = np.full((width, height, depth), int(AIR_DENSITY), dtype=np.int16)

    xs = min_x + (np.arange(width) + 0.5) * vx
    ys = min_y + (np.arange(height) + 0.5) * vy
    z0_center = min_z + 0.5 * vz
    inv_dz = 1.0 / vz
    ray_dir = Vector((0.0, 0.0, 1.0))
    max_dist = (max_z - min_z) + 4.0 * vz
    epsilon = 1e-6

    sorted_objects = sorted(objects, key=lambda obj: obj.name.casefold())
    object_data = []
    for obj in sorted_objects:
        bvh = _bvh_from_object(obj, depsgraph=depsgraph, apply_modifiers=apply_modifiers)
        hu_value = float(getattr(obj, 'dicomator_hu', DEFAULT_DENSITY))
        hu_value = max(MIN_HU_VALUE, min(MAX_HU_VALUE, hu_value))
        object_data.append((bvh, np.int16(hu_value), obj.name))

    total_columns = width * height
    processed_columns = 0

    for ix in range(width):
        x_world = float(xs[ix])
        for iy in range(height):
            y_world = float(ys[iy])
            for bvh, hu_value, _name in object_data:
                origin_ray = Vector((x_world, y_world, min_z - 2.0 * vz))
                hits_z: list[float] = []
                while True:
                    location, _normal, _face_index, _distance = bvh.ray_cast(origin_ray, ray_dir, max_dist)
                    if location is None:
                        break
                    hits_z.append(location.z)
                    origin_ray = Vector((location.x, location.y, location.z + epsilon))
                if hits_z:
                    hits_z.sort()
                    for start in range(0, len(hits_z) - 1, 2):
                        lower = hits_z[start]
                        upper = hits_z[start + 1]
                        start_idx = int(math.ceil((lower - z0_center) * inv_dz))
                        end_idx = int(math.floor((upper - z0_center) * inv_dz))
                        if end_idx >= start_idx:
                            s = max(0, start_idx)
                            e = min(depth - 1, end_idx)
                            if e >= s:
                                hu_array[ix, iy, s:e + 1] = hu_value
            processed_columns += 1
            if progress_callback and (processed_columns % 5000) == 0:
                progress_callback(processed_columns, total_columns)
            if processed_columns % 20000 == 0:
                print(f"  processed columns: {processed_columns}/{total_columns} ({(processed_columns/total_columns)*100:.1f}%)")

    print("Voxelization complete (multi-object HU grid).")
    return hu_array, origin, (width, height, depth)


__all__ = [
    "voxelize_mesh",
    "voxelize_objects_to_hu",
]
