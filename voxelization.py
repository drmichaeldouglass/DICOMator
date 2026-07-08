"""Mesh voxelization helpers used by the DICOMator add-on.

All voxelizers share one ray-casting core (:func:`_voxelize_objects_iter`)
that fills axis-aligned voxel columns by casting +Z rays through each mesh
and pairing entry/exit hits. The core is a generator yielding
``(processed_columns, total_columns)`` so callers (e.g. the modal export
operator) can keep the Blender UI responsive; the blocking wrappers drive
the generator to completion and forward progress to an optional callback.
"""
from __future__ import annotations

import math
from typing import Callable, Generator, Iterable, Optional, Sequence, Tuple

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
ProgressCallback = Optional[Callable[[int, int], None]]
Bounds = Tuple[float, float, float, float, float, float]
VoxelizeResult = Tuple[np.ndarray, Vector, Tuple[int, int, int]]
VoxelizeGenerator = Generator[Tuple[int, int], None, VoxelizeResult]

#: Ray hits closer together than this distance (metres) are merged before
#: entry/exit pairing. A ray grazing an edge shared by two faces reports the
#: same surface twice; without merging, the duplicate flips the inside/outside
#: parity and incorrectly fills the remainder of the voxel column.
_HIT_MERGE_TOLERANCE_M = 1e-5

#: Number of voxel columns processed between progress yields.
_PROGRESS_CHUNK = 2048


def _resolve_voxel_size(voxel_size: VoxelSize | float) -> Tuple[float, float, float]:
    """Return ``(vx, vy, vz)`` in metres from a scalar or 3-sequence."""
    if isinstance(voxel_size, Iterable):
        components = [float(component) for component in voxel_size]
        if len(components) != 3:
            raise ValueError("voxel_size must be a scalar or a 3-component sequence")
        return components[0], components[1], components[2]
    return (float(voxel_size),) * 3


def _object_geometry(
    obj: Object,
    depsgraph: Optional[bpy.types.Depsgraph] = None,
    *,
    apply_modifiers: bool = False,
) -> Optional[Tuple[BVHTree, Tuple[float, float, float, float]]]:
    """Build a world-space BVH for ``obj`` plus its world-space XY extent.

    Returns ``None`` when the (evaluated) mesh has no vertices or faces.
    """
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
    if not verts_world or not polygons:
        return None
    min_x = min(vert.x for vert in verts_world)
    max_x = max(vert.x for vert in verts_world)
    min_y = min(vert.y for vert in verts_world)
    max_y = max(vert.y for vert in verts_world)
    return BVHTree.FromPolygons(verts_world, polygons), (min_x, max_x, min_y, max_y)


def _objects_world_bounds(
    objects: Sequence[Object],
    depsgraph: Optional[bpy.types.Depsgraph],
    *,
    apply_modifiers: bool,
) -> Bounds:
    """Return the combined world-space bounds of ``objects``."""
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
    return min_x, max_x, min_y, max_y, min_z, max_z


def _voxelize_objects_iter(
    objects: Sequence[Object],
    voxel_size: VoxelSize | float,
    padding: int,
    bbox_override: Optional[Bounds],
    *,
    apply_modifiers: bool,
    depsgraph: Optional[bpy.types.Depsgraph],
    value_for_object: Callable[[Object], float],
    dtype: np.dtype,
    background_value: float,
    accumulate: bool,
    label: str,
    messages: Optional[list[str]] = None,
) -> VoxelizeGenerator:
    """Shared ray-casting voxelizer.

    Fills a ``(width, height, depth)`` grid of ``dtype`` initialized to
    ``background_value``. Meshes are processed in alphabetical name order;
    when ``accumulate`` is False the alphabetically last mesh wins any
    overlapping voxels, when True the per-object values are summed.

    When ``messages`` is provided, human-readable warnings about skipped
    objects are appended to it so callers can surface them in the UI.
    """
    if not objects:
        raise ValueError(f"No objects provided for {label} voxelization")

    if apply_modifiers and depsgraph is None:
        depsgraph = bpy.context.evaluated_depsgraph_get()

    vx, vy, vz = _resolve_voxel_size(voxel_size)

    if bbox_override is not None:
        min_x, max_x, min_y, max_y, min_z, max_z = bbox_override
    else:
        min_x, max_x, min_y, max_y, min_z, max_z = _objects_world_bounds(
            objects, depsgraph, apply_modifiers=apply_modifiers
        )
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

    grid = np.full((width, height, depth), background_value, dtype=dtype)

    sorted_objects = sorted(
        objects,
        key=lambda obj: (obj.name.casefold(), obj.name),
    )
    def _skip(reason: str) -> None:
        print(reason)
        if messages is not None:
            messages.append(reason)

    skipped_names: list[str] = []
    object_data: list[tuple[BVHTree, float, int, int, int, int]] = []
    for obj in sorted_objects:
        geometry = _object_geometry(obj, depsgraph=depsgraph, apply_modifiers=apply_modifiers)
        if geometry is None:
            skipped_names.append(obj.name)
            _skip(f"Skipped '{obj.name}' during {label} voxelization: mesh has no faces")
            continue
        bvh, (obj_min_x, obj_max_x, obj_min_y, obj_max_y) = geometry
        # Rays outside the object's XY footprint cannot intersect it, so only
        # the covered column range (plus one voxel of slack) is visited. For
        # small objects inside a large grid this skips almost all columns.
        ix0 = max(0, int(math.floor((obj_min_x - min_x) / vx)) - 1)
        ix1 = min(width - 1, int(math.ceil((obj_max_x - min_x) / vx)) + 1)
        iy0 = max(0, int(math.floor((obj_min_y - min_y) / vy)) - 1)
        iy1 = min(height - 1, int(math.ceil((obj_max_y - min_y) / vy)) + 1)
        if ix1 < ix0 or iy1 < iy0:
            skipped_names.append(obj.name)
            _skip(f"Skipped '{obj.name}' during {label} voxelization: outside the voxel grid")
            continue
        object_data.append((bvh, float(value_for_object(obj)), ix0, ix1, iy0, iy1))

    if not object_data:
        raise ValueError(
            f"No voxelizable {label} objects: all selected meshes were skipped "
            f"({', '.join(skipped_names)})"
        )

    print(f"Voxelizing {len(object_data)} object(s) into {width}x{height}x{depth} {label} grid...")

    xs = min_x + (np.arange(width) + 0.5) * vx
    ys = min_y + (np.arange(height) + 0.5) * vy
    z0_center = min_z + 0.5 * vz
    inv_dz = 1.0 / vz
    ray_dir = Vector((0.0, 0.0, 1.0))
    ray_start_z = min_z - 2.0 * vz
    max_dist = (max_z - min_z) + 4.0 * vz
    epsilon = 1e-6

    total_columns = max(
        1,
        sum((ix1 - ix0 + 1) * (iy1 - iy0 + 1) for _bvh, _value, ix0, ix1, iy0, iy1 in object_data),
    )
    processed = 0

    for bvh, value, ix0, ix1, iy0, iy1 in object_data:
        for ix in range(ix0, ix1 + 1):
            x_world = float(xs[ix])
            for iy in range(iy0, iy1 + 1):
                y_world = float(ys[iy])
                origin_ray = Vector((x_world, y_world, ray_start_z))
                hits_z: list[float] = []
                while True:
                    location, _normal, _face_index, _distance = bvh.ray_cast(origin_ray, ray_dir, max_dist)
                    if location is None:
                        break
                    hits_z.append(location.z)
                    origin_ray = Vector((location.x, location.y, location.z + epsilon))

                if hits_z:
                    hits_z.sort()
                    merged: list[float] = []
                    for hit_z in hits_z:
                        if not merged or (hit_z - merged[-1]) > _HIT_MERGE_TOLERANCE_M:
                            merged.append(hit_z)
                    for start in range(0, len(merged) - 1, 2):
                        lower = merged[start]
                        upper = merged[start + 1]
                        start_idx = int(math.ceil((lower - z0_center) * inv_dz))
                        end_idx = int(math.floor((upper - z0_center) * inv_dz))
                        if end_idx >= start_idx:
                            s = max(0, start_idx)
                            e = min(depth - 1, end_idx)
                            if e >= s:
                                if accumulate:
                                    grid[ix, iy, s:e + 1] += value
                                else:
                                    grid[ix, iy, s:e + 1] = value

                processed += 1
                if processed % _PROGRESS_CHUNK == 0:
                    yield processed, total_columns

    print(f"Voxelization complete ({label} grid).")
    yield total_columns, total_columns
    return grid, origin, (width, height, depth)


def _drive(generator: VoxelizeGenerator, progress_callback: ProgressCallback) -> VoxelizeResult:
    """Run a voxelize generator to completion, forwarding progress."""
    while True:
        try:
            current, total = next(generator)
        except StopIteration as stop:
            return stop.value
        if progress_callback:
            progress_callback(current, total)


def voxelize_objects_to_hu_iter(
    objects: Sequence[Object],
    voxel_size: VoxelSize | float = 1.0,
    padding: int = 1,
    bbox_override: Optional[Bounds] = None,
    *,
    apply_modifiers: bool = False,
    depsgraph: Optional[bpy.types.Depsgraph] = None,
    background_value: float = AIR_DENSITY,
    messages: Optional[list[str]] = None,
) -> VoxelizeGenerator:
    """Generator variant of :func:`voxelize_objects_to_hu`.

    ``background_value`` fills voxels not covered by any mesh. CT exports use
    air (-1000 HU); MR exports should pass 0 (signal void) instead.
    ``messages`` collects skipped-object warnings for the caller's UI.
    """
    def hu_for_object(obj: Object) -> float:
        hu_value = float(getattr(obj, "dicomator_hu", DEFAULT_DENSITY))
        return max(MIN_HU_VALUE, min(MAX_HU_VALUE, hu_value))

    return _voxelize_objects_iter(
        objects,
        voxel_size,
        padding,
        bbox_override,
        apply_modifiers=apply_modifiers,
        depsgraph=depsgraph,
        value_for_object=hu_for_object,
        dtype=np.int16,
        background_value=float(background_value),
        accumulate=False,
        label="HU",
        messages=messages,
    )


def voxelize_objects_to_hu(
    objects: Sequence[Object],
    voxel_size: VoxelSize | float = 1.0,
    padding: int = 1,
    progress_callback: ProgressCallback = None,
    bbox_override: Optional[Bounds] = None,
    *,
    apply_modifiers: bool = False,
    depsgraph: Optional[bpy.types.Depsgraph] = None,
    background_value: float = AIR_DENSITY,
) -> VoxelizeResult:
    """Voxelize multiple objects into a single intensity grid.

    Overlapping solids resolve deterministically: meshes are processed in
    alphabetical order by name and the alphabetically last mesh wins any
    conflicting voxels.
    """
    return _drive(
        voxelize_objects_to_hu_iter(
            objects,
            voxel_size,
            padding,
            bbox_override,
            apply_modifiers=apply_modifiers,
            depsgraph=depsgraph,
            background_value=background_value,
        ),
        progress_callback,
    )


def voxelize_objects_to_dose_iter(
    objects: Sequence[Object],
    voxel_size: VoxelSize | float = 1.0,
    padding: int = 1,
    bbox_override: Optional[Bounds] = None,
    *,
    apply_modifiers: bool = False,
    depsgraph: Optional[bpy.types.Depsgraph] = None,
    accumulate: bool = True,
    messages: Optional[list[str]] = None,
) -> VoxelizeGenerator:
    """Generator variant of :func:`voxelize_objects_to_dose`."""
    def dose_for_object(obj: Object) -> float:
        # Clamp dose to non-negative values; negative dose has no physical meaning.
        return max(0.0, float(getattr(obj, "dicomator_dose", 0.0)))

    return _voxelize_objects_iter(
        objects,
        voxel_size,
        padding,
        bbox_override,
        apply_modifiers=apply_modifiers,
        depsgraph=depsgraph,
        value_for_object=dose_for_object,
        dtype=np.float32,
        background_value=0.0,
        accumulate=accumulate,
        label="dose",
        messages=messages,
    )


def voxelize_objects_to_dose(
    objects: Sequence[Object],
    voxel_size: VoxelSize | float = 1.0,
    padding: int = 1,
    progress_callback: ProgressCallback = None,
    bbox_override: Optional[Bounds] = None,
    *,
    apply_modifiers: bool = False,
    depsgraph: Optional[bpy.types.Depsgraph] = None,
    accumulate: bool = True,
) -> VoxelizeResult:
    """Voxelize multiple objects into a dose grid (Gy).

    Reads ``obj.dicomator_dose`` (float, Gy) per object and returns a
    ``float32`` array (background = 0.0 Gy). When ``accumulate`` is True
    (default) overlapping dose volumes sum, which matches how physical dose
    from multiple sources combines; when False the alphabetically last mesh
    overwrites earlier assignments wherever voxels overlap.
    """
    return _drive(
        voxelize_objects_to_dose_iter(
            objects,
            voxel_size,
            padding,
            bbox_override,
            apply_modifiers=apply_modifiers,
            depsgraph=depsgraph,
            accumulate=accumulate,
        ),
        progress_callback,
    )


def voxelize_mesh(
    obj: Object,
    voxel_size: VectorLike | float = 1.0,
    padding: int = 1,
) -> VoxelizeResult:
    """Voxelize a single object's base mesh into a binary occupancy grid."""
    return _drive(
        _voxelize_objects_iter(
            [obj],
            voxel_size,
            padding,
            None,
            apply_modifiers=False,
            depsgraph=None,
            value_for_object=lambda _obj: 1.0,
            dtype=np.uint8,
            background_value=0.0,
            accumulate=False,
            label="occupancy",
        ),
        None,
    )


__all__ = [
    "voxelize_mesh",
    "voxelize_objects_to_hu",
    "voxelize_objects_to_hu_iter",
    "voxelize_objects_to_dose",
    "voxelize_objects_to_dose_iter",
]
