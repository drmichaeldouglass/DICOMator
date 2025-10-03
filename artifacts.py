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

    Reimplemented to use 1D convolution (mode='same') so the output length
    along ``axis`` always matches the input length and avoids off-by-one bugs.
    """
    if kernel_size <= 1:
        return array
    k = int(kernel_size)
    if k < 1:
        return array

    # Ensure float computation and build normalized kernel.
    arr = array.astype(np.float32, copy=False)
    kernel = np.ones(k, dtype=np.float32) / float(k)

    # Move the smoothing axis to the last dimension so we can convolve each 1D vector
    # independently and preserve the shape using mode='same'.
    moved = np.moveaxis(arr, axis, -1)

    # Convolve along the last axis for every subarray.
    # Using np.apply_along_axis keeps the original shape (mode='same').
    def _conv_row(v: np.ndarray) -> np.ndarray:
        return np.convolve(v, kernel, mode="same")

    smoothed = np.apply_along_axis(_conv_row, -1, moved)

    return np.moveaxis(smoothed, -1, axis)


# New helper: ensure an array matches a target shape by center-cropping or edge-padding.
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

    # Validate user-provided ring parameters. A single ring will be created.
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
    if cols > 1:
        x = (c_idx / (cols - 1)) * 2.0 - 1.0
    else:
        x = c_idx * 0.0
    if rows > 1:
        y = (r_idx / (rows - 1)) * 2.0 - 1.0
    else:
        y = r_idx * 0.0

    radius = np.sqrt(x**2 + y**2)

    # Build a single ring pattern. If the user did not specify a radius, pick
    # a reasonable default location inside the detector area.
    r0 = float(ring_radius) if ring_radius is not None else float(generator.uniform(0.05, 0.95))
    t = float(thickness)
    # Use the user-provided ring_intensity directly as the ring amplitude. This
    # lets the caller precisely control how many HU the ring will add.
    amplitude = float(ring_intensity)

    base_pattern = amplitude * np.exp(-0.5 * ((radius - r0) / t) ** 2).astype(np.float32, copy=False)
    # Slightly attenuate rings toward the edges to mimic typical detector falloff.
    base_pattern *= np.exp(-0.5 * radius**2)

    if jitter > 0.0:
        jitter_field = generator.normal(0.0, jitter, size=radius.shape).astype(np.float32, copy=False)
    else:
        jitter_field = 0.0

    # Apply the same base pattern to every slice, allowing small per-slice
    # scale/offset variations so the effect isn't perfectly identical slice-to-slice.
    for iz in range(depth):
        # small per-slice modulation preserves the user-selected intensity while
        # still producing realistic slice-to-slice variation
        scale = generator.uniform(0.98, 1.02)
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

        # Ensure the blurred result has exactly the same shape as the source slice.
        if blurred.shape != slice_view.shape:
            blurred = _ensure_shape_like(blurred, slice_view.shape)

        # Create a light ghosted duplicate shifted in the motion direction.
        ghost_shift = generator.uniform(-1.0, 1.0)
        ghost = 0.5 * np.roll(slice_view, int(math.copysign(1, ghost_shift) or 1), axis=axis)
        ghost += 0.5 * np.roll(slice_view, -int(math.copysign(1, ghost_shift) or 1), axis=axis)

        # Combine original, blurred and ghost components.
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

    field = generator.normal(0.0, 1.0, size=source.shape).astype(np.float32, copy=False)

    # Smooth the random field along each axis to produce a gentle bias pattern.
    for axis, axis_len in enumerate(source.shape):
        window = max(3, int(round(scale * max(1, axis_len))))
        if window % 2 == 0:
            window += 1
        field = _moving_average_along_axis(field, window, axis)

    field -= float(np.mean(field))
    max_abs = float(np.max(np.abs(field)))
    if max_abs > 0.0:
        field /= max_abs
    bias = 1.0 + float(strength) * field

    biased = source * bias
    min_val = float(np.min(source))
    max_val = float(np.max(source))
    if max_val > min_val:
        biased = np.clip(biased, min_val, max_val)
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
