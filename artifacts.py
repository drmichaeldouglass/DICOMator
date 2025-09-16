"""Utilities for adding synthetic CT acquisition artifacts to voxel volumes."""
from __future__ import annotations

import math

import numpy as np

from .constants import MAX_HU_VALUE, MIN_HU_VALUE

GeneratorLike = np.random.Generator | int | None


def _get_generator(rng: GeneratorLike) -> np.random.Generator:
    """Return a :class:`numpy.random.Generator` from *rng* or create a new one."""
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def _moving_average_along_axis(array: np.ndarray, kernel_size: int, axis: int) -> np.ndarray:
    """Apply a 1D moving average with ``kernel_size`` along ``axis``."""
    if kernel_size <= 1:
        return array
    pad = kernel_size // 2
    pad_width: list[tuple[int, int]] = [(0, 0)] * array.ndim
    pad_width[axis] = (pad, pad)
    padded = np.pad(array, pad_width=pad_width, mode="edge")
    cumsum = np.cumsum(padded, axis=axis, dtype=np.float32)

    upper_slice: list[slice] = [slice(None)] * array.ndim
    lower_slice: list[slice] = [slice(None)] * array.ndim
    upper_slice[axis] = slice(kernel_size, None)
    lower_slice[axis] = slice(None, -kernel_size)

    window_sums = cumsum[tuple(upper_slice)] - cumsum[tuple(lower_slice)]
    return window_sums / float(kernel_size)


def add_gaussian_noise(hu_array: np.ndarray, std_hu: float, rng: GeneratorLike = None) -> np.ndarray:
    """Add zero-mean Gaussian noise in HU to ``hu_array``.

    Parameters
    ----------
    hu_array:
        Input HU volume with shape ``(width, height, depth)``.
    std_hu:
        Standard deviation of the Gaussian noise in HU. Values ``<= 0`` return
        the original volume (converted to ``int16`` if necessary).
    rng:
        Optional seed or :class:`numpy.random.Generator` for reproducible noise.

    Returns
    -------
    numpy.ndarray
        The noisy HU volume as ``int16``.
    """

    if std_hu <= 0.0:
        return hu_array.astype(np.int16, copy=False)

    generator = _get_generator(rng)
    noisy = hu_array.astype(np.float32, copy=True)

    # Draw a Gaussian-distributed perturbation for each voxel.
    noise = generator.normal(0.0, float(std_hu), noisy.shape).astype(np.float32, copy=False)

    # Apply the perturbation and clamp to valid HU bounds.
    noisy += noise
    noisy = np.clip(noisy, MIN_HU_VALUE, MAX_HU_VALUE)
    return noisy.astype(np.int16, copy=False)


def apply_partial_volume_effect(
    hu_array: np.ndarray,
    kernel_size: int = 3,
    iterations: int = 1,
    mix: float = 1.0,
) -> np.ndarray:
    """Blur sharp object boundaries to approximate partial volume averaging.

    Parameters
    ----------
    hu_array:
        Input HU volume.
    kernel_size:
        Size of the moving-average kernel along each axis. Must be an odd
        integer greater than or equal to 1.
    iterations:
        Number of times the smoothing pass is repeated. Higher values produce
        a stronger blending between adjacent materials.
    mix:
        Blend factor between the smoothed result and the original volume.
        ``mix=1`` keeps only the smoothed data, while ``mix=0`` returns the
        original array unchanged.

    Returns
    -------
    numpy.ndarray
        Volume with softened boundaries as ``int16``.
    """

    if kernel_size <= 1 or iterations <= 0:
        return hu_array.astype(np.int16, copy=False)
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be an odd integer >= 1")

    mix = float(np.clip(mix, 0.0, 1.0))
    source = hu_array.astype(np.float32, copy=False)
    result = source.copy()

    # Sequentially blur along each axis to mimic volumetric averaging.
    for _ in range(iterations):
        for axis in range(result.ndim):
            result = _moving_average_along_axis(result, kernel_size, axis)

    if mix < 1.0:
        # Blend with the original data to retain some sharpness if requested.
        result = mix * result + (1.0 - mix) * source

    result = np.clip(np.round(result), MIN_HU_VALUE, MAX_HU_VALUE)
    return result.astype(np.int16, copy=False)


def add_metal_artifacts(
    hu_array: np.ndarray,
    intensity: float = 400.0,
    density_threshold: float = 2000.0,
    num_streaks: int = 10,
    falloff: float = 6.0,
    rng: GeneratorLike = None,
) -> np.ndarray:
    """Inject streak-like artifacts originating from high-density voxels.

    Parameters
    ----------
    hu_array:
        Input HU volume.
    intensity:
        Base amplitude of the streaks in HU.
    density_threshold:
        Voxels with HU above this value are treated as metal sources.
    num_streaks:
        Number of radial streaks to synthesize for each affected slice. When set
        to ``0`` or negative, the number of streaks scales with the amount of
        detected metal voxels.
    falloff:
        Controls how quickly the streaks decay away from the streak direction.
        Higher values localize the artifact closer to the metal object.
    rng:
        Optional seed or generator for deterministic results.

    Returns
    -------
    numpy.ndarray
        Volume containing the simulated metal artifacts as ``int16``.
    """

    generator = _get_generator(rng)
    result = hu_array.astype(np.float32, copy=True)
    width, height, depth = result.shape

    # Pre-compute a normalized coordinate grid for the slice plane.
    x_coords = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    y_coords = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    base_xx, base_yy = np.meshgrid(x_coords, y_coords, indexing="ij")

    for iz in range(depth):
        source_mask = hu_array[:, :, iz] >= density_threshold
        if not np.any(source_mask):
            continue

        points = np.argwhere(source_mask)
        centroid_x = float(np.mean(x_coords[points[:, 0]]))
        centroid_y = float(np.mean(y_coords[points[:, 1]]))

        # Express coordinates relative to the metal centroid to drive radial streaks.
        rel_x = base_xx - centroid_x
        rel_y = base_yy - centroid_y
        radius = np.sqrt(rel_x**2 + rel_y**2) + 1e-3

        slice_artifact = np.zeros_like(result[:, :, iz], dtype=np.float32)
        streak_count = int(num_streaks if num_streaks > 0 else max(6, points.shape[0] // 25))

        for _ in range(streak_count):
            angle = generator.uniform(0.0, math.pi)
            line = rel_x * math.cos(angle) + rel_y * math.sin(angle)
            width_scale = generator.uniform(0.02, 0.08)
            profile = np.exp(-falloff * np.abs(line) / width_scale)
            modulation = np.cos(radius * generator.uniform(4.0, 9.0) + generator.uniform(-math.pi, math.pi))
            decay = 1.0 / (1.0 + 6.0 * radius)
            amplitude = generator.uniform(0.5, 1.0) * intensity * generator.choice([-1.0, 1.0])
            slice_artifact += amplitude * profile * modulation * decay

        # Limit extreme values and only keep streaks that extend beyond the metal
        slice_artifact *= np.exp(-1.5 * radius)
        result[:, :, iz] = np.clip(result[:, :, iz] + slice_artifact, MIN_HU_VALUE, MAX_HU_VALUE)

    return result.astype(np.int16, copy=False)


def add_ring_artifacts(
    hu_array: np.ndarray,
    ring_intensity: float = 80.0,
    num_rings: tuple[int, int] = (4, 7),
    jitter: float = 0.02,
    rng: GeneratorLike = None,
) -> np.ndarray:
    """Add circular banding artifacts similar to detector gain errors.

    Parameters
    ----------
    hu_array:
        Input HU volume.
    ring_intensity:
        Maximum amplitude in HU applied to the generated rings.
    num_rings:
        Range ``(min_rings, max_rings)`` that controls the number of rings in
        the base pattern.
    jitter:
        Standard deviation of a low-amplitude speckle field mixed with the
        rings to avoid perfectly smooth structures.
    rng:
        Optional seed or generator for deterministic behaviour.

    Returns
    -------
    numpy.ndarray
        Volume containing ring artifacts as ``int16``.
    """

    if num_rings[0] <= 0 or num_rings[1] < num_rings[0]:
        raise ValueError("num_rings must define a positive inclusive range")

    generator = _get_generator(rng)
    result = hu_array.astype(np.float32, copy=True)
    width, height, depth = result.shape

    # Normalized radial coordinate system to anchor concentric bands.
    x_coords = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    y_coords = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    xx, yy = np.meshgrid(x_coords, y_coords, indexing="ij")
    radius = np.sqrt(xx**2 + yy**2)

    ring_count = int(generator.integers(num_rings[0], num_rings[1] + 1))
    base_pattern = np.zeros_like(radius, dtype=np.float32)
    for _ in range(ring_count):
        r0 = generator.uniform(0.05, 0.95)
        thickness = generator.uniform(0.01, 0.04)
        amplitude = generator.uniform(0.4, 1.0) * ring_intensity * generator.choice([-1.0, 1.0])
        base_pattern += amplitude * np.exp(-0.5 * ((radius - r0) / thickness) ** 2)

    base_pattern *= np.exp(-0.5 * radius**2)

    if jitter > 0.0:
        jitter_field = generator.normal(0.0, jitter, size=radius.shape).astype(np.float32, copy=False)
    else:
        jitter_field = 0.0

    for iz in range(depth):
        scale = generator.uniform(0.85, 1.15)
        offset = generator.normal(0.0, ring_intensity * 0.05)
        slice_pattern = scale * base_pattern + offset
        if isinstance(jitter_field, np.ndarray):
            slice_pattern += ring_intensity * 0.1 * jitter_field
        result[:, :, iz] = np.clip(result[:, :, iz] + slice_pattern, MIN_HU_VALUE, MAX_HU_VALUE)

    return result.astype(np.int16, copy=False)


def add_motion_artifact(
    hu_array: np.ndarray,
    blur_size: int = 9,
    severity: float = 0.5,
    axis: int = 0,
    rng: GeneratorLike = None,
) -> np.ndarray:
    """Simulate in-plane patient motion as directional blurring and ghosting.

    Parameters
    ----------
    hu_array:
        Input HU volume.
    blur_size:
        Length of the motion blur kernel (must be odd). Larger values smear the
        image more strongly along the selected axis.
    severity:
        Blend factor between the original slice and the blurred/ghosted version.
        ``0`` disables the artifact, while ``1`` keeps only the motion-blurred
        slice.
    axis:
        Axis within each slice along which the motion blur is applied. ``0``
        blurs along the x-axis (first dimension), ``1`` along the y-axis.
    rng:
        Optional seed or generator controlling random sub-voxel jitter.

    Returns
    -------
    numpy.ndarray
        Volume with simulated motion artifacts as ``int16``.
    """

    if blur_size <= 1 or severity <= 0.0:
        return hu_array.astype(np.int16, copy=False)
    if blur_size % 2 == 0:
        raise ValueError("blur_size must be an odd integer >= 1")

    severity = float(np.clip(severity, 0.0, 1.0))
    generator = _get_generator(rng)

    result = hu_array.astype(np.float32, copy=True)
    depth = result.shape[2]

    for iz in range(depth):
        slice_view = result[:, :, iz]

        # Blur along the motion axis to emulate patient displacement.
        blurred = _moving_average_along_axis(slice_view, blur_size, axis)

        # Create a light ghosted duplicate shifted in the motion direction.
        ghost_shift = generator.uniform(-1.0, 1.0)
        ghost = 0.5 * np.roll(slice_view, int(math.copysign(1, ghost_shift) or 1), axis=axis)
        ghost += 0.5 * np.roll(slice_view, -int(math.copysign(1, ghost_shift) or 1), axis=axis)

        slice_view = (1.0 - severity) * slice_view + severity * (0.7 * blurred + 0.3 * ghost)
        result[:, :, iz] = slice_view

    result = np.clip(np.round(result), MIN_HU_VALUE, MAX_HU_VALUE)
    return result.astype(np.int16, copy=False)


def add_poisson_noise(hu_array: np.ndarray, scale: float = 150.0, rng: GeneratorLike = None) -> np.ndarray:
    """Approximate quantum noise by sampling from a Poisson distribution.

    Parameters
    ----------
    hu_array:
        Input HU volume.
    scale:
        Conversion factor from HU to pseudo photon counts. Larger values reduce
        the amount of noise. ``scale <= 0`` disables the artifact.
    rng:
        Optional seed or generator for deterministic noise.

    Returns
    -------
    numpy.ndarray
        Volume with Poisson-distributed fluctuations as ``int16``.
    """

    if scale <= 0.0:
        return hu_array.astype(np.int16, copy=False)

    generator = _get_generator(rng)
    source = hu_array.astype(np.float32, copy=True)

    # Shift to a strictly positive range to interpret HU as pseudo counts.
    shifted = np.clip(source - MIN_HU_VALUE, 0.0, None)

    counts = shifted / float(scale)

    # Sample Poisson-distributed photon counts and transform back to HU.
    noisy_counts = generator.poisson(counts).astype(np.float32, copy=False)
    noisy = noisy_counts * float(scale) + MIN_HU_VALUE

    mask_zero = shifted == 0.0
    noisy[mask_zero] = source[mask_zero]

    noisy = np.clip(np.round(noisy), MIN_HU_VALUE, MAX_HU_VALUE)
    return noisy.astype(np.int16, copy=False)


__all__ = [
    "add_gaussian_noise",
    "apply_partial_volume_effect",
    "add_metal_artifacts",
    "add_ring_artifacts",
    "add_motion_artifact",
    "add_poisson_noise",
]
