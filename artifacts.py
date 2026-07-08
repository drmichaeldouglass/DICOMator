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


def _remap_bilinear(
    image: np.ndarray,
    coord0: np.ndarray,
    coord1: np.ndarray,
    fill: float = 0.0,
) -> np.ndarray:
    """Sample ``image`` at floating-point coordinates using bilinear interpolation.

    ``coord0``/``coord1`` give, for every output pixel, the source position in
    the first/second axis of ``image``. Positions whose 2x2 support falls
    outside the image are set to ``fill``. This is a lightweight, NumPy-only
    stand-in for ``scipy.ndimage.map_coordinates`` (SciPy is not bundled with
    Blender's Python).
    """

    n0, n1 = image.shape
    x0 = np.floor(coord0).astype(np.int64)
    x1 = np.floor(coord1).astype(np.int64)
    f0 = (coord0 - x0).astype(np.float32)
    f1 = (coord1 - x1).astype(np.float32)

    valid = (x0 >= 0) & (x0 < n0 - 1) & (x1 >= 0) & (x1 < n1 - 1)
    x0c = np.clip(x0, 0, n0 - 2)
    x1c = np.clip(x1, 0, n1 - 2)

    img = image.astype(np.float32, copy=False)
    v00 = img[x0c, x1c]
    v01 = img[x0c, x1c + 1]
    v10 = img[x0c + 1, x1c]
    v11 = img[x0c + 1, x1c + 1]
    top = v00 * (1.0 - f1) + v01 * f1
    bot = v10 * (1.0 - f1) + v11 * f1
    out = top * (1.0 - f0) + bot * f0
    return np.where(valid, out, np.float32(fill)).astype(np.float32)


def _rotate_bilinear(image: np.ndarray, angle_rad: float, fill: float = 0.0) -> np.ndarray:
    """Rotate a 2D image about its centre by ``angle_rad`` (bilinear, NumPy-only)."""

    n0, n1 = image.shape
    c0 = (n0 - 1) / 2.0
    c1 = (n1 - 1) / 2.0
    o0, o1 = np.indices((n0, n1), dtype=np.float32)
    r0 = o0 - c0
    r1 = o1 - c1
    ca = math.cos(angle_rad)
    sa = math.sin(angle_rad)
    # Inverse map: fetch the source pixel that rotates onto each output pixel.
    src0 = c0 + ca * r0 + sa * r1
    src1 = c1 - sa * r0 + ca * r1
    return _remap_bilinear(image, src0, src1, fill)


def _resize_bilinear(image: np.ndarray, out_shape: tuple[int, int]) -> np.ndarray:
    """Resample a 2D image to ``out_shape`` using bilinear interpolation."""

    n0, n1 = image.shape
    m0, m1 = out_shape
    if (n0, n1) == (m0, m1):
        return image.astype(np.float32, copy=False)
    # Map output pixel centres back to source coordinates (align corners).
    s0 = (np.arange(m0, dtype=np.float32) * (max(1, n0 - 1) / max(1, m0 - 1)))
    s1 = (np.arange(m1, dtype=np.float32) * (max(1, n1 - 1) / max(1, m1 - 1)))
    coord0 = np.repeat(s0[:, None], m1, axis=1)
    coord1 = np.repeat(s1[None, :], m0, axis=0)
    return _remap_bilinear(image, coord0, coord1, fill=0.0)


def _smooth_random_field(shape: tuple[int, ...], scale: float, rng: np.random.Generator) -> np.ndarray:
    """Return a smooth, zero-centred random field in roughly ``[-1, 1]``.

    Used to model spatially slowly-varying physical quantities such as the
    static-field (B0) inhomogeneity that drives MRI geometric distortion.
    """

    field = rng.normal(0.0, 1.0, size=shape).astype(np.float32, copy=False)
    for axis, axis_len in enumerate(shape):
        window = max(3, int(round(float(scale) * max(1, axis_len))))
        if window % 2 == 0:
            window += 1
        field = _moving_average_along_axis(field, window, axis)
    field -= float(np.mean(field))
    max_abs = float(np.max(np.abs(field)))
    if max_abs > 0.0:
        field /= max_abs
    return field.astype(np.float32, copy=False)


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
    num_streaks: int = 60,
    falloff: float = 6.0,
    rng: GeneratorLike = None,
) -> np.ndarray:
    """Inject streak artifacts from high-density voxels using a projection model.

    Rather than drawing streaks at random image-space angles, this generates the
    artifact in the projection (sinogram) domain, which is where CT metal
    artifacts physically originate:

    1. Voxels above ``density_threshold`` form a metal attenuation map.
    2. The map is forward-projected (a discrete Radon transform) over a set of
       view angles, giving the metal path length seen by each detector reading.
    3. Two non-linear detector errors are modelled per view:
       * **Photon starvation** - very few photons survive the long, dense path,
         so the log-converted signal is dominated by noise. Modelled as a
         path-length-weighted random error that grows super-linearly with path.
       * **Beam hardening** - the polychromatic beam is preferentially hardened
         along dense paths, so the reconstructed value is under-estimated. This
         produces the dark bands that connect pairs of dense objects.
    4. The zero-mean view errors are unfiltered back-projected, which smears each
       inconsistent view across the image as a streak. Superimposing the views
       reproduces the classic star pattern that is tangent to the metal and the
       dark bands between separate implants.

    Parameters
    ----------
    hu_array:
        Input HU volume with shape ``(width, height, depth)``.
    intensity:
        Target amplitude of the streaks in HU (robust peak of the artifact).
    density_threshold:
        Voxels with HU above this value are treated as metal sources.
    num_streaks:
        Number of projection view angles distributed over 180 degrees. More
        views produce a denser, finer star pattern. Values ``<= 0`` fall back to
        an automatic count. (Kept named ``num_streaks`` for backward
        compatibility with existing callers/UI.)
    falloff:
        Controls how quickly streaks decay with distance from the metal. Higher
        values localise the artifact closer to the dense object.
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

    # Cap the working resolution of the (relatively expensive) rotation-based
    # projection so large grids stay responsive; the artifact is resampled back
    # to full resolution afterwards. Streaks are low-frequency, so this is a
    # good approximation.
    work_cap = 256
    num_views = int(num_streaks) if int(num_streaks) > 0 else 60
    num_views = int(np.clip(num_views, 16, 360))
    # ``falloff`` sharpens the radial decay of the back-projected streaks.
    decay = float(np.clip(falloff, 0.1, 24.0)) / 6.0

    for iz in range(depth):
        source_slice = hu_array[:, :, iz].astype(np.float32, copy=False)
        source_mask = source_slice >= density_threshold
        if not np.any(source_mask):
            continue

        # Metal attenuation excess above the threshold, normalised so a typical
        # metal path integrates to O(1) regardless of the HU units in use.
        metal_excess = np.clip(source_slice - float(density_threshold), 0.0, None)
        norm = float(np.percentile(metal_excess[source_mask], 90)) or 1.0
        attn = (metal_excess / norm).astype(np.float32, copy=False)

        # Optionally downsample the attenuation map for the projection loop.
        scale = 1.0
        if max(width, height) > work_cap:
            scale = work_cap / float(max(width, height))
        w0 = max(8, int(round(width * scale)))
        h0 = max(8, int(round(height * scale)))
        attn_small = _resize_bilinear(attn, (w0, h0)) if (w0, h0) != (width, height) else attn

        angles = np.linspace(0.0, math.pi, num_views, endpoint=False)
        back_proj = np.zeros((w0, h0), dtype=np.float32)
        for angle in angles:
            rotated = _rotate_bilinear(attn_small, float(angle))
            # Line integral of metal along the rotated projection direction.
            path = rotated.sum(axis=0)

            # Beam hardening: dark (negative) error growing with path length.
            beam_hardening = -(path ** 2)
            # Photon starvation: log-domain noise amplified along dense paths.
            starvation = generator.normal(0.0, 1.0, size=path.shape).astype(np.float32)
            starvation *= path * np.sqrt(1.0 + path)
            view_error = beam_hardening + 0.6 * starvation

            # Remove the DC term so the streaks are signed and do not bias HU.
            view_error -= float(np.mean(view_error))

            smear = np.broadcast_to(view_error, (w0, h0))
            back_proj += _rotate_bilinear(smear, -float(angle))

        back_proj /= float(num_views)

        # Radial decay away from the metal so distant streaks fade, controlled by
        # ``falloff``. Distance is measured from the metal centroid. Axis 0 of
        # the working slice is the volume's X/width axis, axis 1 is Y/height.
        g0, g1 = np.indices((w0, h0), dtype=np.float32)
        mask_small = _resize_bilinear(source_mask.astype(np.float32), (w0, h0)) > 0.25
        if np.any(mask_small):
            c0 = float(np.mean(g0[mask_small]))
            c1 = float(np.mean(g1[mask_small]))
        else:
            c0, c1 = (w0 - 1) / 2.0, (h0 - 1) / 2.0
        rad = np.sqrt((g0 - c0) ** 2 + (g1 - c1) ** 2)
        rad_norm = rad / (0.5 * float(math.hypot(w0, h0)) + 1e-3)
        back_proj *= np.exp(-decay * rad_norm).astype(np.float32)

        # Scale so the robust peak of the streak field reaches ``intensity`` HU.
        peak = float(np.percentile(np.abs(back_proj), 99.5))
        if peak <= 0.0:
            continue
        slice_artifact = (float(intensity) / peak) * back_proj

        if (w0, h0) != (width, height):
            slice_artifact = _resize_bilinear(slice_artifact, (width, height))

        # Broad beam-hardening shading immediately around the dense object.
        halo_kernel = max(3, min(21, (min(width, height) // 10) * 2 + 1))
        halo = _gaussian_blur(source_mask.astype(np.float32), halo_kernel)
        halo_max = float(np.max(halo))
        if halo_max > 0.0:
            halo /= halo_max
            slice_artifact -= 0.15 * float(intensity) * halo

        # Do not let streaks corrupt the metal voxels themselves.
        slice_artifact[source_mask] = 0.0

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
    thickness:
        Relative radial thickness of each ring (Gaussian width before the
        per-ring random variation). Must be strictly positive.
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
    # The volume is ordered (X, Y, Z), so axis 0 is the image X/width axis and
    # axis 1 is Y/height; the radial pattern is symmetric, so only the shape
    # match with result[:, :, iz] matters.
    if result.ndim < 3 or result.shape[2] == 0:
        return result.astype(np.int16, copy=False)
    size0 = int(result.shape[0])
    size1 = int(result.shape[1])
    depth = int(result.shape[2])

    idx0, idx1 = np.indices((size0, size1), dtype=np.float32)
    norm1 = (idx1 / max(1, size1 - 1)) * 2.0 - 1.0 if size1 > 1 else idx1 * 0.0
    norm0 = (idx0 / max(1, size0 - 1)) * 2.0 - 1.0 if size0 > 1 else idx0 * 0.0

    # A persistent detector-channel calibration error reconstructs as a signed
    # ring. When no radius is specified, synthesize a small cluster of rings.
    ring_specs: list[tuple[float, float]] = []
    if ring_radius is None:
        for _ in range(int(generator.integers(2, 5))):
            ring_specs.append((float(generator.uniform(0.08, 0.95)), float(generator.choice([-1.0, 1.0]))))
    else:
        ring_specs.append((float(ring_radius), float(generator.choice([-1.0, 1.0]))))

    center1 = float(generator.normal(0.0, 0.015))
    center0 = float(generator.normal(0.0, 0.015))
    radius = np.sqrt((norm1 - center1) ** 2 + (norm0 - center0) ** 2)

    base_pattern = np.zeros((size0, size1), dtype=np.float32)
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


def add_rician_noise(
    intensity_array: np.ndarray,
    sigma: float,
    rng: GeneratorLike = None,
) -> np.ndarray:
    """Add Rician-distributed noise appropriate for MR magnitude images.

    An MR magnitude image is the modulus of a complex signal whose real and
    imaginary channels each carry independent zero-mean Gaussian noise of
    standard deviation ``sigma``. The magnitude therefore follows a Rician
    distribution: at high signal it looks Gaussian, but as the true signal
    approaches zero the noise mean rises to ``sigma * sqrt(pi/2)`` and the
    background never reaches zero. This elevated noise floor is the reason plain
    additive Gaussian noise is unphysical for MR, especially in dark regions.

    Parameters
    ----------
    intensity_array:
        Input magnitude volume. Values are treated as non-negative signal.
    sigma:
        Standard deviation of the underlying complex-channel Gaussian noise, in
        image-intensity units. ``sigma <= 0`` returns the input unchanged.
    rng:
        Optional seed or :class:`numpy.random.Generator` for reproducibility.

    Returns
    -------
    numpy.ndarray
        Noisy magnitude volume, preserving the input dtype.
    """

    if sigma <= 0.0:
        return intensity_array

    generator = _get_generator(rng)
    signal = np.clip(intensity_array.astype(np.float32, copy=False), 0.0, None)

    real = signal + generator.normal(0.0, float(sigma), signal.shape).astype(np.float32, copy=False)
    imag = generator.normal(0.0, float(sigma), signal.shape).astype(np.float32, copy=False)
    magnitude = np.sqrt(real * real + imag * imag)

    if np.issubdtype(intensity_array.dtype, np.integer):
        info = np.iinfo(intensity_array.dtype)
        magnitude = np.clip(np.round(magnitude), max(0, info.min), info.max)
    return magnitude.astype(intensity_array.dtype, copy=False)


def add_mri_geometric_distortion(
    intensity_array: np.ndarray,
    gradient_strength: float = 0.05,
    b0_strength: float = 3.0,
    b0_scale: float = 0.35,
    readout_axis: int = 1,
    rng: GeneratorLike = None,
) -> np.ndarray:
    """Warp an MR volume to mimic geometric distortion.

    Two dominant, physically distinct mechanisms are modelled, both applied
    in-plane (per axial slice) where their clinical effect is greatest:

    * **Gradient non-linearity** - the spatial encoding gradients are only
      linear near iso-centre, so voxels are increasingly mis-mapped towards the
      edges of the field of view. This is modelled as a radial polynomial
      displacement (barrel/pincushion warp) that grows with the square of the
      distance from the image centre.
    * **B0 (static field) inhomogeneity** - off-resonance spins are mis-assigned
      along the frequency-encode (readout) direction by an amount proportional
      to the local field offset divided by the receiver bandwidth. This is
      modelled as a smooth off-resonance field that shifts voxels only along the
      readout axis, reproducing the characteristic local stretching/compression
      near air-tissue interfaces.

    Parameters
    ----------
    intensity_array:
        Input volume with shape ``(width, height, depth)``.
    gradient_strength:
        Radial gradient-non-linearity coefficient. Positive values push the
        periphery outward (pincushion), negative inward (barrel). Roughly the
        fractional displacement at the FOV corner. ``0`` disables it.
    b0_strength:
        Peak B0 off-resonance displacement in **voxels** along the readout axis.
        ``0`` disables it.
    b0_scale:
        Relative spatial smoothness of the off-resonance field (0-1); larger
        values vary more slowly across the volume.
    readout_axis:
        In-plane axis of the frequency-encode direction: ``0`` (x) or ``1`` (y).
    rng:
        Optional seed or generator for the off-resonance field.

    Returns
    -------
    numpy.ndarray
        Geometrically distorted volume, preserving the input dtype.
    """

    if intensity_array.ndim != 3:
        return intensity_array
    if abs(gradient_strength) <= 0.0 and b0_strength <= 0.0:
        return intensity_array
    if readout_axis not in (0, 1):
        raise ValueError("readout_axis must be 0 (x) or 1 (y)")

    generator = _get_generator(rng)
    source = intensity_array.astype(np.float32, copy=False)
    width, height, depth = source.shape

    c0 = (width - 1) / 2.0
    c1 = (height - 1) / 2.0
    o0, o1 = np.indices((width, height), dtype=np.float32)
    rel0 = o0 - c0
    rel1 = o1 - c1
    r_max_sq = (c0 * c0 + c1 * c1) + 1e-3
    rho_sq = (rel0 * rel0 + rel1 * rel1) / r_max_sq

    # Inverse-map radial factor: output pixel at radius r samples the source at
    # ``(1 + k*rho^2)`` times its radius, warping the periphery.
    grad_factor = (1.0 + float(gradient_strength) * rho_sq).astype(np.float32)
    base0 = c0 + grad_factor * rel0
    base1 = c1 + grad_factor * rel1

    if b0_strength > 0.0:
        b0_field = _smooth_random_field((width, height, depth), float(b0_scale), generator)
    else:
        b0_field = None

    result = np.empty_like(source)
    for iz in range(depth):
        coord0 = base0.copy()
        coord1 = base1.copy()
        if b0_field is not None:
            shift = float(b0_strength) * b0_field[:, :, iz]
            if readout_axis == 0:
                coord0 = coord0 + shift
            else:
                coord1 = coord1 + shift
        result[:, :, iz] = _remap_bilinear(source[:, :, iz], coord0, coord1, fill=0.0)

    if np.issubdtype(intensity_array.dtype, np.integer):
        info = np.iinfo(intensity_array.dtype)
        result = np.clip(np.round(result), info.min, info.max)
    return result.astype(intensity_array.dtype, copy=False)


def add_gibbs_ringing(
    intensity_array: np.ndarray,
    strength: float = 0.6,
    truncation: float = 0.2,
) -> np.ndarray:
    """Add Gibbs (truncation) ringing from finite k-space sampling.

    MR images are reconstructed from a finite region of k-space, so sharp edges
    are effectively convolved with a sinc, producing parallel bright/dark ripples
    that run alongside high-contrast boundaries (e.g. skull/CSF, fat/muscle).
    This is reproduced faithfully by transforming each slice to the frequency
    domain, discarding the outer fraction of k-space, and transforming back.

    Parameters
    ----------
    intensity_array:
        Input volume with shape ``(width, height, depth)``.
    strength:
        Blend factor (0-1) between the original slice and the truncated
        reconstruction. ``0`` disables the effect.
    truncation:
        Fraction of the k-space extent removed from each in-plane edge
        (0-0.49). Larger values give coarser, stronger ringing.

    Returns
    -------
    numpy.ndarray
        Volume with truncation ringing, preserving the input dtype.
    """

    if strength <= 0.0 or truncation <= 0.0:
        return intensity_array
    if intensity_array.ndim != 3:
        return intensity_array

    strength = float(np.clip(strength, 0.0, 1.0))
    truncation = float(np.clip(truncation, 0.0, 0.49))
    source = intensity_array.astype(np.float32, copy=False)
    width, height, depth = source.shape

    # Build a centred rectangular k-space mask keeping the inner (1-2*trunc) band
    # along each in-plane axis. The hard edge is what generates the ringing.
    keep0 = max(1, int(round(width * (1.0 - 2.0 * truncation))))
    keep1 = max(1, int(round(height * (1.0 - 2.0 * truncation))))
    mask = np.zeros((width, height), dtype=np.float32)
    lo0 = (width - keep0) // 2
    lo1 = (height - keep1) // 2
    mask[lo0:lo0 + keep0, lo1:lo1 + keep1] = 1.0

    result = source.copy()
    for iz in range(depth):
        spectrum = np.fft.fftshift(np.fft.fft2(source[:, :, iz]))
        truncated = np.fft.ifft2(np.fft.ifftshift(spectrum * mask))
        # MR images are magnitude data, so take the modulus of the result.
        truncated = np.abs(truncated).astype(np.float32)
        result[:, :, iz] = (1.0 - strength) * source[:, :, iz] + strength * truncated

    if np.issubdtype(intensity_array.dtype, np.integer):
        info = np.iinfo(intensity_array.dtype)
        result = np.clip(np.round(result), info.min, info.max)
    return result.astype(intensity_array.dtype, copy=False)


__all__ = [
    "add_gaussian_noise",
    "apply_partial_volume_effect",
    "add_metal_artifacts",
    "add_ring_artifacts",
    "add_motion_artifact",
    "add_poisson_noise",
    "add_bias_field_shading",
    "add_rician_noise",
    "add_mri_geometric_distortion",
    "add_gibbs_ringing",
]
