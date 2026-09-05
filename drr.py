"""Digital reconstructed radiograph (DRR) generation helpers."""
from __future__ import annotations

import math
from typing import Callable, Generator, Optional, Sequence

import bpy
import numpy as np
from mathutils import Vector

from .constants import resolve_positive_voxel_size, validate_numeric_array

ProgressCallback = Optional[Callable[[int, int], None]]
DRRResult = tuple[np.ndarray, dict[str, object]]
DRRGenerator = Generator[tuple[int, int], None, DRRResult]


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

    # Parallel rays impose no distance limit when inside the axis slab, and
    # cannot hit it when outside. Replacing zero with a small positive value
    # tilts the ray, losing rays on the upper face and admitting outside rays.
    parallel = directions == 0.0
    safe_directions = np.where(parallel, 1.0, directions)
    t0 = (bounds_min - origins) / safe_directions
    t1 = (bounds_max - origins) / safe_directions
    t_min = np.where(parallel, -np.inf, np.minimum(t0, t1))
    t_max = np.where(parallel, np.inf, np.maximum(t0, t1))
    entry = np.max(t_min, axis=1)
    exit_ = np.min(t_max, axis=1)
    outside_parallel = np.any(
        parallel & ((origins < bounds_min) | (origins > bounds_max)), axis=1
    )
    valid = (
        ~outside_parallel
        & np.any(~parallel, axis=1)
        & (exit_ > np.maximum(entry, 0.0))
    )
    entry = np.where(valid, np.maximum(entry, 0.0), 0.0).astype(np.float32, copy=False)
    exit_ = np.where(valid, exit_, 0.0).astype(np.float32, copy=False)
    return entry, exit_, valid


def _normalize_projection(line_integrals: np.ndarray, fixed: bool = False) -> np.ndarray:
    """Map line integrals into a 16-bit display-ready DRR image.

    When ``fixed`` is True the physical absorption fraction ``1 - exp(-L)``
    is mapped directly onto the uint16 range without percentile stretching,
    so intensities stay comparable across 4D phases.
    """

    transmission = np.exp(-line_integrals.astype(np.float32, copy=False))
    radiograph = 1.0 - transmission

    if fixed:
        return np.round(np.clip(radiograph, 0.0, 1.0) * 65535.0).astype(np.uint16, copy=False)

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


def _hu_to_linear_attenuation(
    hu_volume: np.ndarray,
    water_attenuation_coefficient_m_inv: float,
) -> np.ndarray:
    """Convert HU to a non-negative linear attenuation coefficient in m^-1."""

    mu_water = float(water_attenuation_coefficient_m_inv)
    if not np.isfinite(mu_water) or mu_water <= 0.0:
        raise ValueError("water_attenuation_coefficient_m_inv must be finite and greater than zero")
    return np.maximum(
        0.0,
        np.float32(mu_water) * (1.0 + (hu_volume.astype(np.float32) / 1000.0)),
    ).astype(np.float32, copy=False)


def generate_drr_from_hu_volume(
    hu_volume: np.ndarray,
    voxel_size: Sequence[float] | float,
    origin: Vector,
    scene: bpy.types.Scene,
    camera_obj: bpy.types.Object,
    *,
    resolution_scale: float = 1.0,
    progress_callback: ProgressCallback = None,
    fixed_normalization: bool = False,
    water_attenuation_coefficient_m_inv: float = 20.0,
) -> DRRResult:
    """Project ``hu_volume`` into a DRR using the active camera geometry.

    Blocking wrapper around :func:`generate_drr_from_hu_volume_iter`.
    """
    generator = generate_drr_from_hu_volume_iter(
        hu_volume,
        voxel_size,
        origin,
        scene,
        camera_obj,
        resolution_scale=resolution_scale,
        fixed_normalization=fixed_normalization,
        water_attenuation_coefficient_m_inv=water_attenuation_coefficient_m_inv,
    )
    while True:
        try:
            current, total = next(generator)
        except StopIteration as stop:
            return stop.value
        if progress_callback:
            progress_callback(current, total)


def generate_drr_from_hu_volume_iter(
    hu_volume: np.ndarray,
    voxel_size: Sequence[float] | float,
    origin: Vector,
    scene: bpy.types.Scene,
    camera_obj: bpy.types.Object,
    *,
    resolution_scale: float = 1.0,
    fixed_normalization: bool = False,
    water_attenuation_coefficient_m_inv: float = 20.0,
) -> DRRGenerator:
    """Generator variant: yields ``(rows_done, total_rows)`` per detector chunk."""

    if camera_obj is None or camera_obj.type != 'CAMERA':
        raise ValueError("Scene must have an active camera for DRR export")
    hu_volume = validate_numeric_array(hu_volume, name="hu_volume", ndim=3)
    vx, vy, vz = resolve_positive_voxel_size(voxel_size)
    mu_water = float(water_attenuation_coefficient_m_inv)

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

    # Convert CT HU to a true linear attenuation coefficient in m^-1 using
    # mu = mu_water * (1 + HU / 1000). The configurable monoenergetic water
    # coefficient keeps the Beer-Lambert line integral dimensionless.
    attenuation_volume = _hu_to_linear_attenuation(hu_volume, mu_water)
    step_size = max(1e-5, min(vx, vy, vz))

    bottom_left, bottom_right, top_left, top_right = _camera_frame_corners(scene, camera_obj)

    camera_matrix = camera_obj.matrix_world
    camera_rotation = camera_matrix.to_3x3()
    camera_origin = np.array(camera_matrix.translation, dtype=np.float32)
    rotation = np.array(camera_rotation, dtype=np.float32)

    # Measure the detector in world space. ``view_frame`` reports camera-local
    # corners, but both the rays cast below and ImagePositionPatient are built
    # through matrix_world, which also carries the camera object's scale.
    # Taking the extents from the unscaled local corners would report a
    # PixelSpacing that disagrees with the geometry actually projected.
    frame_width_m = float((camera_rotation @ (bottom_right - bottom_left)).length)
    frame_height_m = float((camera_rotation @ (top_left - bottom_left)).length)

    local_bottom_left = np.array(bottom_left, dtype=np.float32)
    local_bottom_right = np.array(bottom_right, dtype=np.float32)
    local_top_left = np.array(top_left, dtype=np.float32)
    local_top_right = np.array(top_right, dtype=np.float32)

    is_orthographic = str(getattr(camera_obj.data, "type", "PERSP")).upper() == "ORTHO"
    orthographic_direction = rotation @ np.array((0.0, 0.0, -1.0), dtype=np.float32)
    orthographic_direction /= max(np.linalg.norm(orthographic_direction), 1e-8)

    # Blender's view frame sits one unit in front of the camera, so parallel
    # rays launched from it would start *inside* (or past) a grid placed closer
    # than that; the entry distance is clamped to zero, silently dropping the
    # part of the volume nearest the camera. Parallel rays carry no perspective,
    # so the origins can simply slide back along the view direction until the
    # whole grid is ahead of them. The projection of the nearest bounding-box
    # corner onto the view direction is the axis-wise minimum of the two bounds.
    nearest_corner_projection = float(
        np.sum(
            np.minimum(
                orthographic_direction * bounds_min,
                orthographic_direction * bounds_max,
            )
        )
    )

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
            back_off = origins @ orthographic_direction - nearest_corner_projection
            if np.any(back_off > 0.0):
                # Only rays that already overshot the grid are moved; the rest
                # keep their exact origins. Sliding along the ray leaves the
                # sampled positions unchanged, so this costs no extra samples.
                back_off = np.maximum(back_off, 0.0) + np.float32(step_size)
                origins = origins - orthographic_direction[None, :] * back_off[:, None]
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
                sample_offsets = np.arange(sample_start, sample_end, dtype=np.float32) * step_size
                # Integrate the final, possibly shorter segment using its own
                # midpoint and length. A fixed midpoint/full-step weight can
                # drop short paths entirely or overestimate their attenuation.
                segment_lengths = np.clip(
                    (exit_t - entry_t)[None, :] - sample_offsets[:, None],
                    0.0,
                    step_size,
                )
                t_values = entry_t[None, :] + sample_offsets[:, None] + 0.5 * segment_lengths
                active_samples = valid[None, :] & (segment_lengths > 0.0)
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
                attenuation_samples *= segment_lengths
                chunk_integrals += np.sum(attenuation_samples, axis=0, dtype=np.float32)

        line_integrals[row_start:row_end, :] = chunk_integrals.reshape(row_end - row_start, detector_width)

        yield row_end, detector_height

    projection_image = _normalize_projection(line_integrals, fixed=fixed_normalization)

    row_direction_world = (camera_rotation @ (top_right - top_left)).normalized()
    column_direction_world = (camera_rotation @ (bottom_left - top_left)).normalized()

    pixel_spacing_mm = None
    image_position_patient = None
    image_orientation_patient = None
    if is_orthographic:
        row_step_local = (top_right - top_left) / float(detector_width)
        column_step_local = (bottom_left - top_left) / float(detector_height)
        first_pixel_center_local = top_left + 0.5 * row_step_local + 0.5 * column_step_local
        first_pixel_center_world = camera_matrix @ first_pixel_center_local
        pixel_spacing_mm = (
            (frame_height_m / float(detector_height)) * 1000.0,
            (frame_width_m / float(detector_width)) * 1000.0,
        )
        image_position_patient = (
            float(first_pixel_center_world.x * 1000.0),
            float(first_pixel_center_world.y * 1000.0),
            float(first_pixel_center_world.z * 1000.0),
        )
        image_orientation_patient = (
            float(row_direction_world.x),
            float(row_direction_world.y),
            float(row_direction_world.z),
            float(column_direction_world.x),
            float(column_direction_world.y),
            float(column_direction_world.z),
        )

    metadata = {
        "detector_size": (detector_width, detector_height),
        "pixel_spacing_mm": pixel_spacing_mm,
        "image_position_patient": image_position_patient,
        "image_orientation_patient": image_orientation_patient,
        "spatial_geometry_valid": bool(is_orthographic),
        "water_attenuation_coefficient_m_inv": mu_water,
    }
    return projection_image, metadata


__all__ = [
    "generate_drr_from_hu_volume",
    "generate_drr_from_hu_volume_iter",
    "resolve_drr_detector_size",
]
