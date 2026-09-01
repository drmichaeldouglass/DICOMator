"""Mesh voxelization helpers used by the DICOMator add-on.

All voxelizers share one ray-casting core (:func:`_voxelize_objects_iter`)
that fills axis-aligned voxel columns by casting +Z rays through each mesh
and pairing entry/exit hits. The core is a generator yielding
``(processed_columns, total_columns)`` so callers (e.g. the modal export
operator) can keep the Blender UI responsive; the blocking wrappers drive
the generator to completion and forward progress to an optional callback.
"""
from __future__ import annotations

import logging
import math
from typing import Callable, Generator, Optional, Sequence, Tuple

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
    resolve_positive_voxel_size,
)

LOGGER = logging.getLogger(__name__)

VectorLike = Sequence[float]
VoxelSize = Sequence[float]
ProgressCallback = Optional[Callable[[int, int], None]]
Bounds = Tuple[float, float, float, float, float, float]
VoxelizeResult = Tuple[np.ndarray, Vector, Tuple[int, int, int]]
VoxelizeGenerator = Generator[Tuple[int, int], None, VoxelizeResult]
#: ``(bvh, world_bounds)`` pair produced by :func:`_object_geometry` /
#: :func:`prepare_object_geometry_iter`.
PreparedGeometry = Tuple[BVHTree, Bounds]

#: Ray hits closer together than this distance (metres) are merged before
#: entry/exit pairing. A ray grazing an edge shared by two faces reports the
#: same surface twice; without merging, the duplicate flips the inside/outside
#: parity and incorrectly fills the remainder of the voxel column.
_HIT_MERGE_TOLERANCE_M = 1e-5

#: Number of voxel columns processed between progress yields.
_PROGRESS_CHUNK = 2048

#: Hard cap on surface crossings recorded for one voxel column. A closed mesh
#: produces a handful; a runaway count means the marching ray stopped making
#: progress, so the column is abandoned rather than looping forever.
_MAX_COLUMN_HITS = 4096

#: Smallest absolute distance (metres) a restarted ray is pushed past the face
#: it just hit.
_RAY_RESTART_EPSILON_M = 1e-6


def restart_z_past_hit(hit_z: float) -> float:
    """Return a Z just past ``hit_z`` so a restarted +Z ray clears that face.

    ``mathutils.Vector`` stores single-precision floats, whose spacing already
    swallows a fixed 1e-6 m nudge once ``|z|`` reaches 32 metres: the restarted
    ray would round straight back onto the face it just hit and report the same
    crossing forever. Scaling the step with the coordinate magnitude keeps it
    ahead of the surface anywhere in the scene, while staying well below the
    finest voxel spacing the add-on offers (0.1 mm).
    """

    return hit_z + max(_RAY_RESTART_EPSILON_M, abs(hit_z) * 1e-6)


def _resolve_voxel_size(voxel_size: VoxelSize | float) -> Tuple[float, float, float]:
    """Return ``(vx, vy, vz)`` in metres from a scalar or 3-sequence."""
    return resolve_positive_voxel_size(voxel_size)


def _object_priority_key(obj: Object) -> tuple[int, str, str]:
    """Sort low-priority objects first so higher priorities overwrite them."""

    return (
        int(getattr(obj, "dicomator_priority", 0)),
        obj.name.casefold(),
        obj.name,
    )


def _world_vertex_array(mesh: bpy.types.Mesh, matrix_world) -> np.ndarray:
    """Return the mesh vertices as an ``(N, 3)`` world-space float64 array.

    Uses ``foreach_get`` plus one NumPy matrix multiply instead of a Python
    loop over vertices, which is dramatically faster on dense meshes. The
    returned buffer is caller-owned, so it stays valid after
    ``to_mesh_clear()``.

    The read buffer must be ``float32``: ``foreach_get`` only takes its bulk
    C copy path when the buffer's format matches the RNA property's raw type,
    and ``MeshVertex.co`` is a float32 vector. A float64 buffer silently falls
    back to a per-vertex Python loop, which is the very cost this helper
    exists to avoid. Blender stores the coordinates as float32 anyway, so
    widening afterwards loses nothing.
    """
    count = len(mesh.vertices)
    coords = np.empty(count * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", coords)
    matrix = np.array(matrix_world, dtype=np.float64)
    local = coords.reshape(count, 3).astype(np.float64)
    return local @ matrix[:3, :3].T + matrix[:3, 3]


def _mesh_polygon_indices(mesh: bpy.types.Mesh) -> list[list[int]]:
    """Return per-polygon vertex index lists via ``foreach_get`` batch reads."""
    poly_count = len(mesh.polygons)
    if poly_count == 0:
        return []
    loop_start = np.empty(poly_count, dtype=np.int32)
    mesh.polygons.foreach_get("loop_start", loop_start)
    loop_verts = np.empty(len(mesh.loops), dtype=np.int32)
    mesh.loops.foreach_get("vertex_index", loop_verts)
    return [segment.tolist() for segment in np.split(loop_verts, loop_start[1:])]


def _object_geometry(
    obj: Object,
    depsgraph: Optional[bpy.types.Depsgraph] = None,
    *,
    apply_modifiers: bool = False,
) -> Optional[PreparedGeometry]:
    """Build a world-space BVH for ``obj`` plus its world-space bounds.

    Returns ``(bvh, (min_x, max_x, min_y, max_y, min_z, max_z))``, or ``None``
    when the (evaluated) mesh has no vertices or faces.
    """
    if apply_modifiers and depsgraph is not None:
        obj_eval = obj.evaluated_get(depsgraph)
        mesh = obj_eval.to_mesh(preserve_all_data_layers=False, depsgraph=depsgraph)
        try:
            verts_world = _world_vertex_array(mesh, obj_eval.matrix_world)
            polygons = _mesh_polygon_indices(mesh)
        finally:
            obj_eval.to_mesh_clear()
    else:
        mesh = obj.data
        verts_world = _world_vertex_array(mesh, obj.matrix_world)
        polygons = _mesh_polygon_indices(mesh)
    if verts_world.size == 0 or not polygons:
        return None
    mins = verts_world.min(axis=0)
    maxs = verts_world.max(axis=0)
    bounds: Bounds = (
        float(mins[0]), float(maxs[0]),
        float(mins[1]), float(maxs[1]),
        float(mins[2]), float(maxs[2]),
    )
    return BVHTree.FromPolygons(verts_world.tolist(), polygons), bounds


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
                verts_world = _world_vertex_array(mesh, obj_eval.matrix_world)
            finally:
                obj_eval.to_mesh_clear()
            if verts_world.size:
                mins = verts_world.min(axis=0)
                maxs = verts_world.max(axis=0)
                min_x = min(min_x, float(mins[0]))
                max_x = max(max_x, float(maxs[0]))
                min_y = min(min_y, float(mins[1]))
                max_y = max(max_y, float(maxs[1]))
                min_z = min(min_z, float(mins[2]))
                max_z = max(max_z, float(maxs[2]))
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
    prepared: Optional[dict[str, PreparedGeometry]] = None,
) -> VoxelizeGenerator:
    """Shared ray-casting voxelizer.

    Fills a ``(width, height, depth)`` grid of ``dtype`` initialized to
    ``background_value``. Meshes are processed by ``dicomator_priority`` then
    name; when ``accumulate`` is False the highest-priority mesh wins any
    overlapping voxels, when True the per-object values are summed.

    When ``messages`` is provided, human-readable warnings about skipped
    objects are appended to it so callers can surface them in the UI.

    ``prepared`` optionally supplies pre-built world-space geometry (from
    :func:`prepare_object_geometry_iter`) keyed by object name, avoiding a
    second mesh evaluation and BVH build; objects missing from the cache are
    treated as skipped (they had no usable geometry when prepared).
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
        # Objects that evaluate to an empty mesh contribute no corners, so the
        # running extremes stay at +/-inf. Report that directly instead of
        # letting math.ceil() below fail with 'cannot convert float infinity
        # to integer', which tells the user nothing about their scene.
        if not all(
            math.isfinite(value)
            for value in (min_x, max_x, min_y, max_y, min_z, max_z)
        ):
            raise ValueError(
                f"No voxelizable {label} objects: none of the selected meshes "
                "have any vertices (a modifier that empties the mesh will do "
                "this)"
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

    # NumPy casts a float into an integer grid by truncating toward zero, so an
    # integer-valued grid has to be given values that are already whole. Without
    # this a mesh set to -75.6 HU would be stored as -75 and one at 50.7 HU as
    # 50: a sub-HU error whose sign follows the tissue rather than cancelling
    # out. Rounding here (half-to-even, matching ``artifacts.py``) keeps every
    # voxel at the nearest value the user actually asked for.
    stores_integers = np.issubdtype(np.dtype(dtype), np.integer)

    def _grid_value(value: float) -> float:
        return float(round(float(value))) if stores_integers else float(value)

    grid = np.full((width, height, depth), _grid_value(background_value), dtype=dtype)

    sorted_objects = sorted(objects, key=_object_priority_key)
    def _skip(reason: str) -> None:
        LOGGER.warning(reason)
        if messages is not None:
            messages.append(reason)

    skipped_names: list[str] = []
    object_data: list[tuple[str, BVHTree, float, int, int, int, int]] = []
    for obj in sorted_objects:
        if prepared is not None:
            geometry = prepared.get(obj.name)
        else:
            geometry = _object_geometry(obj, depsgraph=depsgraph, apply_modifiers=apply_modifiers)
        if geometry is None:
            skipped_names.append(obj.name)
            _skip(f"Skipped '{obj.name}' during {label} voxelization: mesh has no faces")
            continue
        bvh, (obj_min_x, obj_max_x, obj_min_y, obj_max_y, _obj_min_z, _obj_max_z) = geometry
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
        object_data.append((obj.name, bvh, _grid_value(value_for_object(obj)), ix0, ix1, iy0, iy1))

    if not object_data:
        raise ValueError(
            f"No voxelizable {label} objects: all selected meshes were skipped "
            f"({', '.join(skipped_names)})"
        )

    LOGGER.info(
        "Voxelizing %d object(s) into a %dx%dx%d %s grid",
        len(object_data), width, height, depth, label,
    )

    xs = [float(min_x + (index + 0.5) * vx) for index in range(width)]
    ys = [float(min_y + (index + 0.5) * vy) for index in range(height)]
    z0_center = min_z + 0.5 * vz
    inv_dz = 1.0 / vz
    ray_dir = Vector((0.0, 0.0, 1.0))
    ray_start_z = min_z - 2.0 * vz
    max_dist = (max_z - min_z) + 4.0 * vz

    total_columns = max(
        1,
        sum(
            (ix1 - ix0 + 1) * (iy1 - iy0 + 1)
            for _name, _bvh, _value, ix0, ix1, iy0, iy1 in object_data
        ),
    )
    processed = 0

    for object_name, bvh, value, ix0, ix1, iy0, iy1 in object_data:
        ray_cast = bvh.ray_cast
        odd_columns = 0
        recovered_columns = 0
        stalled_rays = 0

        def _merged_hits(x_world: float, y_world: float) -> list[float]:
            nonlocal stalled_rays
            origin_ray = Vector((x_world, y_world, ray_start_z))
            hits_z: list[float] = []
            while True:
                location, _normal, _face_index, _distance = ray_cast(
                    origin_ray, ray_dir, max_dist
                )
                if location is None:
                    break
                hits_z.append(location.z)
                if len(hits_z) >= _MAX_COLUMN_HITS:
                    stalled_rays += 1
                    break
                next_origin = Vector(
                    (location.x, location.y, restart_z_past_hit(float(location.z)))
                )
                if next_origin.z <= origin_ray.z:
                    # The restart point rounded back onto the face just hit, so
                    # the next cast would report the same crossing forever.
                    stalled_rays += 1
                    break
                origin_ray = next_origin
            hits_z.sort()
            merged: list[float] = []
            for hit_z in hits_z:
                if not merged or (hit_z - merged[-1]) > _HIT_MERGE_TOLERANCE_M:
                    merged.append(hit_z)
            return merged

        for ix in range(ix0, ix1 + 1):
            x_world = xs[ix]
            for iy in range(iy0, iy1 + 1):
                y_world = ys[iy]
                merged = _merged_hits(x_world, y_world)
                if len(merged) % 2:
                    odd_columns += 1
                    # Retry edge/vertex-grazing rays with a small deterministic
                    # sub-voxel offset. The first even result preserves stable
                    # output while avoiding arbitrary unpaired surface loss.
                    jittered_hits = None
                    for x_fraction, y_fraction in (
                        (0.173, 0.271),
                        (-0.173, 0.271),
                        (0.173, -0.271),
                        (-0.173, -0.271),
                    ):
                        candidate = _merged_hits(
                            x_world + x_fraction * vx,
                            y_world + y_fraction * vy,
                        )
                        if candidate and len(candidate) % 2 == 0:
                            jittered_hits = candidate
                            break
                    if jittered_hits is not None:
                        merged = jittered_hits
                        recovered_columns += 1

                if merged:
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

        if stalled_rays:
            _skip(
                f"'{object_name}' abandoned {stalled_rays} ray(s) during {label} "
                "voxelization after they stopped advancing past a surface; the "
                "affected columns may be incompletely filled (check for "
                "coincident faces or geometry far from the world origin)"
            )

        if odd_columns:
            unresolved = odd_columns - recovered_columns
            warning = (
                f"'{object_name}' produced {odd_columns} odd ray-intersection "
                f"column(s) during {label} voxelization; {recovered_columns} "
                f"recovered with deterministic sub-voxel rays"
            )
            if unresolved:
                warning += f", {unresolved} remained ambiguous and may need mesh repair"
            _skip(warning)

    LOGGER.info("Voxelization complete (%s grid)", label)
    yield total_columns, total_columns
    return grid, origin, (width, height, depth)


def prepare_object_geometry_iter(
    objects: Sequence[Object],
    depsgraph: Optional[bpy.types.Depsgraph],
    *,
    apply_modifiers: bool,
) -> Generator[Tuple[int, int], None, dict[str, PreparedGeometry]]:
    """Evaluate each object's world-space BVH and bounds exactly once.

    Yields ``(objects_done, total_objects)`` and returns a dict keyed by
    object name; objects without usable geometry are omitted. The cache is
    only valid for the frame/depsgraph it was built with — callers must not
    reuse it across animation frames.
    """
    prepared: dict[str, PreparedGeometry] = {}
    total = max(1, len(objects))
    for index, obj in enumerate(objects, start=1):
        geometry = _object_geometry(obj, depsgraph=depsgraph, apply_modifiers=apply_modifiers)
        if geometry is not None:
            prepared[obj.name] = geometry
        yield index, total
    return prepared


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
    prepared: Optional[dict[str, PreparedGeometry]] = None,
) -> VoxelizeGenerator:
    """Generator variant of :func:`voxelize_objects_to_hu`.

    ``background_value`` fills voxels not covered by any mesh. CT exports use
    air (-1000 HU); MR exports should pass 0 (signal void) instead.
    ``messages`` collects skipped-object warnings for the caller's UI;
    ``prepared`` reuses geometry from :func:`prepare_object_geometry_iter`.
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
        prepared=prepared,
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

    Overlapping solids resolve deterministically: meshes are processed by
    ``dicomator_priority`` and then by name, so the highest-priority mesh wins
    conflicting voxels and names break ties.
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
    prepared: Optional[dict[str, PreparedGeometry]] = None,
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
        prepared=prepared,
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
    from multiple sources combines; when False the highest-priority mesh
    overwrites earlier assignments wherever voxels overlap, with names used
    to break equal-priority ties.
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
    "prepare_object_geometry_iter",
    "restart_z_past_hit",
    "voxelize_mesh",
    "voxelize_objects_to_hu",
    "voxelize_objects_to_hu_iter",
    "voxelize_objects_to_dose",
    "voxelize_objects_to_dose_iter",
]
