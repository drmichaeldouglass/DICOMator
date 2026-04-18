"""Digital reconstructed radiograph (DRR) generation helpers."""
from __future__ import annotations

import math
from typing import Callable, Optional, Sequence

import bpy
import numpy as np
from mathutils import Vector

ProgressCallback = Optional[Callable[[int, int], None]]


def resolve_drr_detector_size(scene: bpy.types.Scene, resolution_scale: float = 1.0) -> tuple[int, int]:
    """Return the DRR detector size in pixels from the active render settings."""

    render = scene.render
    render_scale = float(render.resolution_percentage) / 100.0
    scale = max(0.1, float(resolution_scale))
    width = max(1, int(round(float(render.resolution_x) * render_scale * scale)))
    height = max(1, int(round(float(render.resolution_y) * render_scale * scale)))
    return width, height


def _camera_frame_corners(scene: bpy.types.Scene, camera_obj: bpy.types.Object) -> tuple[Vector, Vector, Vector, Vector]:
    """Return the detector plane corners in camera-local coordinates."""

    frame = [Vector(point) for point in camera_obj.data.view_frame(scene=scene)]
    min_x = min(point.x for point in frame)
    max_x = max(point.x for point in frame)
    min_y = min(point.y for point in frame)
    max_y = max(point.y for point in frame)
    mean_z = sum(point.z for point in frame) / max(1, len(frame))
    bottom_left = Vector((min_x, min_y, mean_z))
    bottom_right = Vector((max_x, min_y, mean_z))
    top_left = Vector((min_x, max_y, mean_z))
    top_right = Vector((max_x, max_y, mean_z))
    return bottom_left, bottom_right, top_left, top_right


def _ray_box_intersections(
    origins: np.ndarray,
    directions: np.ndarray,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Intersect a batch of rays with an axis-aligned bounding box."""

    safe_directions = np.where(np.abs(directions) < 1e-8, 1e-8, directions)
    t0 = (bounds_min - origins) / safe_directions
    t1 = (bounds_max - origins) / safe_directions
    t_min = np.minimum(t0, t1)
    t_max = np.maximum(t0, t1)
    entry = np.max(t_min, axis=1)
    exit_ = np.min(t_max, axis=1)
    valid = exit_ > np.maximum(entry, 0.0)
    entry = np.where(valid, np.maximum(entry, 0.0), 0.0).astype(np.float32, copy=False)
    exit_ = np.where(valid, exit_, 0.0).astype(np.float32, copy=False)
    return entry, exit_, valid


def _normalize_projection(line_integrals: np.ndarray) -> np.ndarray:
    """Map line integrals into a 16-bit display-ready DRR image."""

    transmission = np.exp(-line_integrals.astype(np.float32, copy=False))
    radiograph = 1.0 - transmission

    if not np.any(radiograph > 0.0):
        return np.zeros(radiograph.shape, dtype=np.uint16)

    low = float(np.percentile(radiograph, 1.0))
    high = float(np.percentile(radiograph, 99.5))
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        low = float(radiograph.min())
        high = float(radiograph.max())

    scale = max(high - low, 1e-6)
    normalized = np.clip((radiograph - low) / scale, 0.0, 1.0)
    return np.round(normalized * 65535.0).astype(np.uint16, copy=False)


def generate_drr_from_hu_volume(
    hu_volume: np.ndarray,
    voxel_size: Sequence[float] | float,
    origin: Vector,
    scene: bpy.types.Scene,
    camera_obj: bpy.types.Object,
    *,
    resolution_scale: float = 1.0,
    progress_callback: ProgressCallback = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Project ``hu_volume`` into a DRR using the active camera geometry."""

    if camera_obj is None or camera_obj.type != 'CAMERA':
        raise ValueError("Scene must have an active camera for DRR export")
    if hu_volume.ndim != 3:
        raise ValueError("DRR generation requires a 3D HU volume")

    if isinstance(voxel_size, Sequence) and len(voxel_size) == 3:
        vx, vy, vz = (float(component) for component in voxel_size)
    else:
        vx = vy = vz = float(voxel_size)

    detector_width, detector_height = resolve_drr_detector_size(scene, resolution_scale=resolution_scale)
    bounds_min = np.array((float(origin.x), float(origin.y), float(origin.z)), dtype=np.float32)
    bounds_max = bounds_min + np.array(
        (
            float(hu_volume.shape[0]) * vx,
            float(hu_volume.shape[1]) * vy,
            float(hu_volume.shape[2]) * vz,
        ),
        dtype=np.float32,
    )

    # Approximate linear attenuation from CT HU using mu/mu_water = 1 + HU / 1000.
    attenuation_volume = np.maximum(0.0, 1.0 + (hu_volume.astype(np.float32) / 1000.0))
    step_size = max(1e-5, min(vx, vy, vz))

    bottom_left, bottom_right, top_left, top_right = _camera_frame_corners(scene, camera_obj)
    frame_width_m = float((bottom_right - bottom_left).length)
    frame_height_m = float((top_left - bottom_left).length)

    camera_origin = np.array(camera_obj.matrix_world.translation, dtype=np.float32)
    rotation = np.array(camera_obj.matrix_world.to_3x3(), dtype=np.float32)
    local_bottom_left = np.array(bottom_left, dtype=np.float32)
    local_bottom_right = np.array(bottom_right, dtype=np.float32)
    local_top_left = np.array(top_left, dtype=np.float32)
    local_top_right = np.array(top_right, dtype=np.float32)

    is_orthographic = str(getattr(camera_obj.data, "type", "PERSP")).upper() == "ORTHO"
    orthographic_direction = rotation @ np.array((0.0, 0.0, -1.0), dtype=np.float32)
    orthographic_direction /= max(np.linalg.norm(orthographic_direction), 1e-8)

    line_integrals = np.zeros((detector_height, detector_width), dtype=np.float32)
    rays_per_chunk_target = 4096
    rows_per_chunk = max(1, min(detector_height, rays_per_chunk_target // max(1, detector_width)))
    sample_block = 256

    pixel_u = (np.arange(detector_width, dtype=np.float32) + 0.5) / float(detector_width)

    for row_start in range(0, detector_height, rows_per_chunk):
        row_end = min(detector_height, row_start + rows_per_chunk)
        # Row 0 must map to the top of the camera frame so the rendered image
        # matches the ImagePositionPatient/column-direction metadata written
        # below (which anchor the image at top_left).
        pixel_v = 1.0 - (np.arange(row_start, row_end, dtype=np.float32) + 0.5) / float(detector_height)
        uu, vv = np.meshgrid(pixel_u, pixel_v, indexing='xy')

        bottom_edge = local_bottom_left[None, None, :] + (local_bottom_right - local_bottom_left)[None, None, :] * uu[:, :, None]
        top_edge = local_top_left[None, None, :] + (local_top_right - local_top_left)[None, None, :] * uu[:, :, None]
        detector_points_local = bottom_edge + (top_edge - bottom_edge) * vv[:, :, None]
        ray_count = detector_points_local.shape[0] * detector_points_local.shape[1]
        detector_points_local = detector_points_local.reshape(ray_count, 3).astype(np.float32, copy=False)

        if is_orthographic:
            origins = detector_points_local @ rotation.T + camera_origin[None, :]
            directions = np.repeat(orthographic_direction[None, :], ray_count, axis=0)
        else:
            origins = np.repeat(camera_origin[None, :], ray_count, axis=0)
            directions = detector_points_local @ rotation.T
            norms = np.linalg.norm(directions, axis=1, keepdims=True)
            directions = directions / np.maximum(norms, 1e-8)

        entry_t, exit_t, valid = _ray_box_intersections(origins, directions, bounds_min, bounds_max)
        chunk_integrals = np.zeros(ray_count, dtype=np.float32)

        if np.any(valid):
            path_lengths = exit_t[valid] - entry_t[valid]
            max_samples = int(math.ceil(float(path_lengths.max()) / step_size))

            for sample_start in range(0, max_samples, sample_block):
                sample_end = min(max_samples, sample_start + sample_block)
                sample_offsets = (np.arange(sample_start, sample_end, dtype=np.float32) + 0.5) * step_size
                t_values = entry_t[None, :] + sample_offsets[:, None]
                active_samples = valid[None, :] & (t_values < exit_t[None, :])
                if not np.any(active_samples):
                    continue

                sample_positions = origins[None, :, :] + directions[None, :, :] * t_values[:, :, None]
                ix = np.floor((sample_positions[:, :, 0] - bounds_min[0]) / vx).astype(np.int32, copy=False)
                iy = np.floor((sample_positions[:, :, 1] - bounds_min[1]) / vy).astype(np.int32, copy=False)
                iz = np.floor((sample_positions[:, :, 2] - bounds_min[2]) / vz).astype(np.int32, copy=False)

                inside = (
                    active_samples
                    & (ix >= 0)
                    & (ix < hu_volume.shape[0])
                    & (iy >= 0)
                    & (iy < hu_volume.shape[1])
                    & (iz >= 0)
                    & (iz < hu_volume.shape[2])
                )
                if not np.any(inside):
                    continue

                attenuation_samples = np.zeros(t_values.shape, dtype=np.float32)
                attenuation_samples[inside] = attenuation_volume[ix[inside], iy[inside], iz[inside]]
                chunk_integrals += np.sum(attenuation_samples, axis=0, dtype=np.float32) * step_size

        line_integrals[row_start:row_end, :] = chunk_integrals.reshape(row_end - row_start, detector_width)

        if progress_callback is not None:
            progress_callback(row_end, detector_height)

    projection_image = _normalize_projection(line_integrals)

    detector_origin_world = camera_obj.matrix_world @ top_left
    row_direction_world = (camera_obj.matrix_world.to_3x3() @ (top_right - top_left)).normalized()
    column_direction_world = (camera_obj.matrix_world.to_3x3() @ (bottom_left - top_left)).normalized()

    metadata = {
        "detector_size": (detector_width, detector_height),
        "pixel_spacing_mm": (
            (frame_height_m / max(1, detector_height)) * 1000.0,
            (frame_width_m / max(1, detector_width)) * 1000.0,
        ),
        "image_position_patient": (
            float(detector_origin_world.x * 1000.0),
            float(detector_origin_world.y * 1000.0),
            float(detector_origin_world.z * 1000.0),
        ),
        "image_orientation_patient": (
            float(row_direction_world.x),
            float(row_direction_world.y),
            float(row_direction_world.z),
            float(column_direction_world.x),
            float(column_direction_world.y),
            float(column_direction_world.z),
        ),
    }
    return projection_image, metadata


__all__ = ["generate_drr_from_hu_volume", "resolve_drr_detector_size"]
