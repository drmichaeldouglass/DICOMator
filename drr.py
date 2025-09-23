"""Digital Radiograph Reconstruction (DRR) helpers."""
from __future__ import annotations

import math
from typing import Sequence

import bpy
import numpy as np
from bpy.types import Object
from mathutils import Vector

from .voxelization import voxelize_objects_to_hu


def _trilinear_sample(volume: np.ndarray, x: float, y: float, z: float) -> float:
    """Sample ``volume`` at fractional index coordinates using trilinear interpolation."""

    width, height, depth = volume.shape
    if x < 0.0 or y < 0.0 or z < 0.0 or x > (width - 1) or y > (height - 1) or z > (depth - 1):
        return -1000.0

    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    z0 = int(math.floor(z))
    x1 = min(x0 + 1, width - 1)
    y1 = min(y0 + 1, height - 1)
    z1 = min(z0 + 1, depth - 1)

    dx = x - x0
    dy = y - y0
    dz = z - z0

    c000 = float(volume[x0, y0, z0])
    c100 = float(volume[x1, y0, z0])
    c010 = float(volume[x0, y1, z0])
    c110 = float(volume[x1, y1, z0])
    c001 = float(volume[x0, y0, z1])
    c101 = float(volume[x1, y0, z1])
    c011 = float(volume[x0, y1, z1])
    c111 = float(volume[x1, y1, z1])

    c00 = (c000 * (1.0 - dx)) + (c100 * dx)
    c10 = (c010 * (1.0 - dx)) + (c110 * dx)
    c01 = (c001 * (1.0 - dx)) + (c101 * dx)
    c11 = (c011 * (1.0 - dx)) + (c111 * dx)
    c0 = (c00 * (1.0 - dy)) + (c10 * dy)
    c1 = (c01 * (1.0 - dy)) + (c11 * dy)
    return (c0 * (1.0 - dz)) + (c1 * dz)


def generate_drr_image(
    objects: Sequence[Object],
    lateral_resolution_mm: float,
    axial_resolution_mm: float,
    angle_degrees: float,
    *,
    padding: int = 1,
    apply_modifiers: bool = False,
    depsgraph: bpy.types.Depsgraph | None = None,
) -> np.ndarray:
    """Generate a normalized DRR image from the provided ``objects``."""

    if not objects:
        raise ValueError("No mesh objects provided for DRR generation")

    if lateral_resolution_mm <= 0.0 or axial_resolution_mm <= 0.0:
        raise ValueError("Voxel resolution must be positive")

    vx = float(lateral_resolution_mm) * 0.001
    vy = float(lateral_resolution_mm) * 0.001
    vz = float(axial_resolution_mm) * 0.001

    hu_grid, origin, (width, height, depth) = voxelize_objects_to_hu(
        objects,
        voxel_size=(vx, vy, vz),
        padding=padding,
        apply_modifiers=apply_modifiers,
        depsgraph=depsgraph,
    )

    if width <= 0 or height <= 0 or depth <= 0:
        raise ValueError("Empty HU grid returned from voxelization")

    theta = math.radians(angle_degrees)
    dir_vec = Vector((math.cos(theta), math.sin(theta), 0.0))
    if dir_vec.length == 0.0:
        dir_vec = Vector((1.0, 0.0, 0.0))
    dir_vec.normalize()

    perp_vec = Vector((-dir_vec.y, dir_vec.x, 0.0))
    if perp_vec.length == 0.0:
        perp_vec = Vector((0.0, 1.0, 0.0))
    perp_vec.normalize()

    up_vec = Vector((0.0, 0.0, 1.0))

    dims = Vector((width * vx, height * vy, depth * vz))
    corners: list[Vector] = []
    for dx in (0.0, dims.x):
        for dy in (0.0, dims.y):
            for dz in (0.0, dims.z):
                corners.append(origin + Vector((dx, dy, dz)))

    min_u = min(perp_vec.dot(corner) for corner in corners)
    max_u = max(perp_vec.dot(corner) for corner in corners)
    min_dir = min(dir_vec.dot(corner) for corner in corners)
    max_dir = max(dir_vec.dot(corner) for corner in corners)
    min_v = min(up_vec.dot(corner) for corner in corners)
    max_v = max(up_vec.dot(corner) for corner in corners)

    u_spacing = vx
    v_spacing = vz
    num_u = max(1, int(math.ceil((max_u - min_u) / u_spacing)))
    num_v = max(1, int(math.ceil((max_v - min_v) / v_spacing)))

    u_coords = min_u + (np.arange(num_u) + 0.5) * u_spacing
    v_coords = min_v + (np.arange(num_v) + 0.5) * v_spacing

    step = min(vx, vy, vz)
    if step <= 0.0:
        raise ValueError("Computed invalid ray marching step size")

    t_start = min_dir - step
    t_end = max_dir + step
    num_steps = max(1, int(math.ceil((t_end - t_start) / step)))
    t_values = t_start + (np.arange(num_steps) + 0.5) * step

    origin_x = float(origin.x)
    origin_y = float(origin.y)
    origin_z = float(origin.z)
    dir_x, dir_y, dir_z = dir_vec.x, dir_vec.y, dir_vec.z
    perp_x, perp_y, perp_z = perp_vec.x, perp_vec.y, perp_vec.z
    up_x, up_y, up_z = up_vec.x, up_vec.y, up_vec.z

    max_local_x = dims.x
    max_local_y = dims.y
    max_local_z = dims.z
    eps = 1e-6

    line_integrals = np.zeros((num_v, num_u), dtype=np.float32)
    for v_idx, v_val in enumerate(v_coords):
        for u_idx, u_val in enumerate(u_coords):
            accum = 0.0
            for sample_t in t_values:
                px = (dir_x * sample_t) + (perp_x * u_val) + (up_x * v_val)
                py = (dir_y * sample_t) + (perp_y * u_val) + (up_y * v_val)
                pz = (dir_z * sample_t) + (perp_z * u_val) + (up_z * v_val)

                local_x = px - origin_x
                local_y = py - origin_y
                local_z = pz - origin_z

                if (
                    -eps <= local_x <= (max_local_x + eps)
                    and -eps <= local_y <= (max_local_y + eps)
                    and -eps <= local_z <= (max_local_z + eps)
                ):
                    ix = (local_x / vx) - 0.5
                    iy = (local_y / vy) - 0.5
                    iz = (local_z / vz) - 0.5
                    sample_hu = _trilinear_sample(hu_grid, ix, iy, iz)
                    attenuation = max(0.0, (sample_hu + 1000.0) * 0.001)
                    accum += attenuation * (step * 1000.0)
            line_integrals[v_idx, u_idx] = accum

    max_val = float(line_integrals.max())
    if max_val > 0.0:
        normalized = line_integrals / max_val
    else:
        normalized = line_integrals

    image = 1.0 - normalized
    np.clip(image, 0.0, 1.0, out=image)
    return image


__all__ = ["generate_drr_image"]
