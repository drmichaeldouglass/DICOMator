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
    """Apply a 1D moving average with ``kernel_size`` along ``axis``.

    Uses edge-padded 1D convolution so the output length along ``axis`` always
    matches the input length and avoids off-by-one bugs.
    """
    if kernel_size <= 1:
        return array
    k = int(kernel_size)
    if k < 1:
        return array

    arr = array.astype(np.float32, copy=False)
    kernel = np.ones(k, dtype=np.float32) / float(k)
    return _convolve_along_axis(arr, kernel, axis)


def _convolve_along_axis(array: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    """Convolve ``array`` with a 1D kernel along one axis using edge padding."""

    arr = array.astype(np.float32, copy=False)
    kernel = kernel.astype(np.float32, copy=False)
    if kernel.size <= 1:
        return arr

    pad = int(kernel.size // 2)
    moved = np.moveaxis(arr, axis, -1)

    def _conv_row(v: np.ndarray) -> np.ndarray:
        padded = np.pad(v, (pad, pad), mode="edge")
        return np.convolve(padded, kernel, mode="valid")

    smoothed = np.apply_along_axis(_conv_row, -1, moved)
    if smoothed.shape != moved.shape:
        smoothed = _ensure_shape_like(smoothed, moved.shape)

    return np.moveaxis(smoothed, -1, axis)


def _gaussian_kernel(kernel_size: int, sigma: float | None = None) -> np.ndarray:
    """Return a normalized odd-length Gaussian kernel."""

    if kernel_size <= 1:
        return np.ones(1, dtype=np.float32)
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be an odd integer >= 1")

    radius = kernel_size // 2
    if sigma is None:
        sigma = max(0.5, float(kernel_size - 1) / 4.0)
    coords = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (coords / float(sigma)) ** 2)
    kernel /= float(np.sum(kernel))
    return kernel.astype(np.float32, copy=False)


def _gaussian_blur(array: np.ndarray, kernel_size: int, sigma: float | None = None) -> np.ndarray:
    """Apply a separable Gaussian blur to ``array``."""

    kernel = _gaussian_kernel(kernel_size, sigma=sigma)
    result = array.astype(np.float32, copy=False)
    for axis in range(result.ndim):
        result = _convolve_along_axis(result, kernel, axis)
    return result


def _ensure_shape_like(arr: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    """Return a view/array with shape exactly equal to target_shape.

    If arr is larger along an axis it is center-cropped; if smaller it is
    edge-padded. This keeps simple smoothing/resampling operations robust
    to occasional off-by-one differences.
    """
    if arr.shape == target_shape:
        return arr
    res = arr
    # Crop larger axes first (to avoid repeated padding on cropped data).
    for axis in range(res.ndim):
        curr = res.shape[axis] if axis < res.ndim else 1
        target = target_shape[axis] if axis < len(target_shape) else 1
        if curr > target:
            start = (curr - target) // 2
            end = start + target
            slc = [slice(None)] * res.ndim
            slc[axis] = slice(start, end)
            res = res[tuple(slc)]
    # Then pad smaller axes.
    if res.shape != target_shape:
        pad_width = []
        for axis in range(res.ndim):
            curr = res.shape[axis]
            target = target_shape[axis]
            if curr < target:
                before = (target - curr) // 2
                after = target - curr - before
                pad_width.append((before, after))
            else:
                pad_width.append((0, 0))
        res = np.pad(res, pad_width=pad_width, mode="edge")
    return res


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
    """Blur sharp object boundaries to approximate scanner partial volume.

    Parameters
    ----------
    hu_array:
        Input HU volume.
    kernel_size:
        Support of the scanner point-spread kernel. Must be an odd integer
        greater than or equal to 1.
    iterations:
        Number of point-spread passes. Higher values produce stronger blending
        between adjacent materials.
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

    # CT/MR partial volume is closer to a scanner point-spread function than a
    # box average. A separable Gaussian is a lightweight approximation.
    for _ in range(iterations):
        result = _gaussian_blur(result, kernel_size)

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
        source_slice = hu_array[:, :, iz].astype(np.float32, copy=False)
        source_mask = source_slice >= density_threshold
        if not np.any(source_mask):
            continue

        metal_excess = np.clip(source_slice - float(density_threshold), 0.0, None)
        weights = metal_excess + source_mask.astype(np.float32)
        weight_sum = float(np.sum(weights))
        if weight_sum <= 0.0:
            continue

        centroid_x = float(np.sum(base_xx * weights) / weight_sum)
        centroid_y = float(np.sum(base_yy * weights) / weight_sum)
        metal_fraction = float(np.mean(source_mask))
        mean_excess = float(np.mean(metal_excess[source_mask]))
        source_strength = np.clip(mean_excess / 1200.0 + 10.0 * metal_fraction, 0.25, 3.0)

        # Express coordinates relative to the metal centroid. This is an image-
        # space approximation of projection paths crossing a high-attenuation
        # object, producing photon-starvation shadows plus beam-hardening bands.
        rel_x = base_xx - centroid_x
        rel_y = base_yy - centroid_y
        radius = np.sqrt(rel_x**2 + rel_y**2) + 1e-3

        slice_artifact = np.zeros_like(result[:, :, iz], dtype=np.float32)
        metal_voxels = int(np.count_nonzero(source_mask))
        streak_count = int(num_streaks if num_streaks > 0 else max(6, min(32, metal_voxels // 25)))

        for _ in range(streak_count):
            angle = generator.uniform(0.0, math.pi)
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            parallel = rel_x * cos_a + rel_y * sin_a
            perpendicular = -rel_x * sin_a + rel_y * cos_a
            line_sigma = generator.uniform(0.018, 0.055) / max(0.5, float(falloff) / 6.0)
            profile = np.exp(-0.5 * (perpendicular / line_sigma) ** 2)
            envelope = 1.0 / (1.0 + 1.5 * np.abs(parallel) + 2.5 * radius)
            phase = generator.uniform(-math.pi, math.pi)
            banding = np.cos(8.0 * parallel + phase)
            amplitude = generator.uniform(0.7, 1.2) * float(intensity) * float(source_strength)

            starvation_shadow = -0.65 * profile * envelope
            beam_hardening_band = 0.35 * profile * banding * envelope
            slice_artifact += amplitude * (starvation_shadow + beam_hardening_band)

        halo_kernel = max(3, min(21, (min(width, height) // 10) * 2 + 1))
        halo = _gaussian_blur(source_mask.astype(np.float32), halo_kernel)
        halo_max = float(np.max(halo))
        if halo_max > 0.0:
            halo /= halo_max
            slice_artifact -= 0.12 * float(intensity) * float(source_strength) * halo

        result[:, :, iz] = np.clip(result[:, :, iz] + slice_artifact, MIN_HU_VALUE, MAX_HU_VALUE)

    return result.astype(np.int16, copy=False)


def add_ring_artifacts(
    hu_array: np.ndarray,
    ring_intensity: float = 80.0,
    ring_radius: float | None = None,
    thickness: float = 0.02,
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
    ring_radius:
        Relative radius of the ring (0 = center, 1 = edge). When ``None``, a
        small random cluster of detector-channel rings is generated.
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

    # Validate user-provided ring parameters.
    if thickness <= 0.0:
        raise ValueError("thickness must be a positive number")
    if ring_radius is not None and not (0.0 <= ring_radius <= 1.0):
        raise ValueError("ring_radius must be between 0.0 and 1.0 (relative radius) or None")

    generator = _get_generator(rng)
    result = hu_array.astype(np.float32, copy=True)

    # Use explicit first-two axes from the volume to derive slice dimensions.
    # This guarantees the radial grid shape matches result[:, :, iz] exactly.
    if result.ndim < 3 or result.shape[2] == 0:
        return result.astype(np.int16, copy=False)
    rows = int(result.shape[0])
    cols = int(result.shape[1])
    depth = int(result.shape[2])

    r_idx, c_idx = np.indices((rows, cols), dtype=np.float32)
    x = (c_idx / max(1, cols - 1)) * 2.0 - 1.0 if cols > 1 else c_idx * 0.0
    y = (r_idx / max(1, rows - 1)) * 2.0 - 1.0 if rows > 1 else r_idx * 0.0

    # A persistent detector-channel calibration error reconstructs as a signed
    # ring. When no radius is specified, synthesize a small cluster of rings.
    ring_specs: list[tuple[float, float]] = []
    if ring_radius is None:
        for _ in range(int(generator.integers(2, 5))):
            ring_specs.append((float(generator.uniform(0.08, 0.95)), float(generator.choice([-1.0, 1.0]))))
    else:
        ring_specs.append((float(ring_radius), float(generator.choice([-1.0, 1.0]))))

    center_x = float(generator.normal(0.0, 0.015))
    center_y = float(generator.normal(0.0, 0.015))
    radius = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

    base_pattern = np.zeros((rows, cols), dtype=np.float32)
    t = float(thickness)
    for r0, sign in ring_specs:
        channel_width = t * float(generator.uniform(0.75, 1.35))
        amplitude = sign * float(ring_intensity) * float(generator.uniform(0.55, 1.0))
        ring = np.exp(-0.5 * ((radius - r0) / channel_width) ** 2).astype(np.float32, copy=False)
        base_pattern += amplitude * ring

    # Rings weaken near the edge because fewer reconstructed pixels sample the
    # same faulty detector channel across the rotation.
    base_pattern *= np.exp(-0.35 * radius**2)

    if jitter > 0.0:
        jitter_field = generator.normal(0.0, jitter, size=radius.shape).astype(np.float32, copy=False)
    else:
        jitter_field = 0.0

    # Apply the same base pattern to every slice, allowing small per-slice
    # scale/offset variations so the effect isn't perfectly identical slice-to-slice.
    for iz in range(depth):
        # Slow axial modulation mimics channel drift while preserving continuity.
        scale = generator.uniform(0.97, 1.03)
        offset = float(generator.normal(0.0, abs(ring_intensity) * 0.01))
        slice_pattern = scale * base_pattern + offset
        if isinstance(jitter_field, np.ndarray):
            slice_pattern += abs(ring_intensity) * 0.1 * jitter_field
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
        Optional seed or generator for deterministic ghost strength variation.

    Returns
    -------
    numpy.ndarray
        Volume with simulated motion artifacts as ``int16``.
    """

    if blur_size <= 1 or severity <= 0.0:
        return hu_array.astype(np.int16, copy=False)
    if blur_size % 2 == 0:
        raise ValueError("blur_size must be an odd integer >= 1")
    if axis not in (0, 1):
        raise ValueError("axis must be 0 (x) or 1 (y) for in-plane motion blur")

    severity = float(np.clip(severity, 0.0, 1.0))

    generator = _get_generator(rng)
    result = hu_array.astype(np.float32, copy=True)
    depth = result.shape[2]
    half_width = blur_size // 2
    offsets = np.arange(-half_width, half_width + 1, dtype=np.int32)
    sigma = max(1.0, float(blur_size) / 4.0)
    weights = np.exp(-0.5 * (offsets.astype(np.float32) / sigma) ** 2)
    weights /= float(np.sum(weights))
    ghost_shift = max(1, blur_size // 3)

    for iz in range(depth):
        slice_view = result[:, :, iz]

        # Average several displaced positions. This better represents object
        # motion during acquisition than a flat box blur.
        blurred = np.zeros_like(slice_view, dtype=np.float32)
        for offset, weight in zip(offsets, weights, strict=False):
            blurred += float(weight) * np.roll(slice_view, int(offset), axis=axis)

        # Residual ghost edges approximate projection inconsistency.
        ghost_weight = float(generator.uniform(0.15, 0.35))
        ghost = 0.5 * np.roll(slice_view, ghost_shift, axis=axis)
        ghost += 0.5 * np.roll(slice_view, -ghost_shift, axis=axis)

        slice_view = (1.0 - severity) * slice_view + severity * ((1.0 - ghost_weight) * blurred + ghost_weight * ghost)
        result[:, :, iz] = slice_view

    result = np.clip(np.round(result), MIN_HU_VALUE, MAX_HU_VALUE)
    return result.astype(np.int16, copy=False)


def add_poisson_noise(hu_array: np.ndarray, scale: float = 150.0, rng: GeneratorLike = None) -> np.ndarray:
    """Approximate CT quantum noise from transmitted photon statistics.

    Parameters
    ----------
    hu_array:
        Input HU volume.
    scale:
        Relative incident photon fluence. Larger values reduce the amount of
        noise. ``scale <= 0`` disables the artifact.
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

    # Convert HU to an approximate attenuation scale: air ~ 0, water ~ 1,
    # cortical bone > 1. The exact path length is unknown post-reconstruction,
    # so this models the local noise trend rather than a full projection.
    relative_attenuation = np.clip((source + 1000.0) / 1000.0, 0.0, 4.0)
    attenuation_coefficient = 0.35
    incident_photons = max(1.0, float(scale)) * 100.0
    expected_counts = incident_photons * np.exp(-attenuation_coefficient * relative_attenuation)
    expected_counts = np.clip(expected_counts, 1.0, None)

    measured_counts = generator.poisson(expected_counts).astype(np.float32, copy=False)
    measured_counts = np.clip(measured_counts, 1.0, None)
    log_error = -np.log(measured_counts / expected_counts)
    noise_hu = (1000.0 / attenuation_coefficient) * log_error
    noisy = source + noise_hu

    noisy = np.clip(np.round(noisy), MIN_HU_VALUE, MAX_HU_VALUE)
    return noisy.astype(np.int16, copy=False)


def add_bias_field_shading(
    intensity_array: np.ndarray,
    strength: float = 0.25,
    scale: float = 0.3,
    rng: GeneratorLike = None,
) -> np.ndarray:
    """Apply a smooth intensity bias field to mimic MRI coil shading.

    Parameters
    ----------
    intensity_array:
        Input volume whose intensities should be modulated.
    strength:
        Fractional amplitude of the multiplicative bias (0 disables the effect).
    scale:
        Relative smoothing window (0-1) controlling how quickly the bias varies.
        Larger values produce slower spatial variation.
    rng:
        Optional seed or :class:`numpy.random.Generator` for deterministic bias.

    Returns
    -------
    numpy.ndarray
        Volume with low-frequency multiplicative shading, preserving dtype.
    """

    if strength <= 0.0:
        return intensity_array

    generator = _get_generator(rng)
    source = intensity_array.astype(np.float32, copy=False)

    if source.size == 0:
        return intensity_array

    axes = [np.linspace(-1.0, 1.0, size, dtype=np.float32) for size in source.shape]
    coords = np.meshgrid(*axes, indexing="ij")
    center = [float(generator.normal(0.0, 0.25)) for _ in source.shape]
    coil_width = max(0.35, float(scale) * 1.5)
    radius_sq = np.zeros(source.shape, dtype=np.float32)
    for coord, center_value in zip(coords, center, strict=False):
        radius_sq += (coord - center_value) ** 2
    coil_field = np.exp(-0.5 * radius_sq / (coil_width**2)).astype(np.float32, copy=False)

    random_field = generator.normal(0.0, 1.0, size=source.shape).astype(np.float32, copy=False)
    for axis, axis_len in enumerate(source.shape):
        window = max(3, int(round(scale * max(1, axis_len))))
        if window % 2 == 0:
            window += 1
        random_field = _moving_average_along_axis(random_field, window, axis)

    field = 0.75 * coil_field + 0.25 * random_field
    field -= float(np.mean(field))
    max_abs = float(np.max(np.abs(field)))
    if max_abs > 0.0:
        field /= max_abs
    bias = np.clip(1.0 + float(strength) * field, 0.05, None)

    biased = source * bias
    if np.issubdtype(intensity_array.dtype, np.integer):
        info = np.iinfo(intensity_array.dtype)
        biased = np.clip(np.round(biased), info.min, info.max)
    return biased.astype(intensity_array.dtype, copy=False)


__all__ = [
    "add_gaussian_noise",
    "apply_partial_volume_effect",
    "add_metal_artifacts",
    "add_ring_artifacts",
    "add_motion_artifact",
    "add_poisson_noise",
    "add_bias_field_shading",
]
