"""Invariant, statistical, and golden-seed tests for artifacts.py."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conftest import load_module

artifacts = load_module("artifacts")
constants = load_module("constants")

MIN_HU = constants.MIN_HU_VALUE
MAX_HU = constants.MAX_HU_VALUE

GOLDEN_PATH = Path(__file__).parent / "data" / "golden_artifacts.npz"


def _volume(shape=(24, 20, 6), seed=7, low=-500, high=1500) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(low, high, size=shape, dtype=np.int16)


def _metal_volume(shape=(24, 20, 6)) -> np.ndarray:
    vol = _volume(shape)
    vol[8:12, 8:12, :] = 3000
    return vol


INT16_FUNCS = [
    ("gaussian", lambda v, rng: artifacts.add_gaussian_noise(v, 40.0, rng=rng)),
    ("partial_volume", lambda v, rng: artifacts.apply_partial_volume_effect(v, kernel_size=3)),
    ("metal", lambda v, rng: artifacts.add_metal_artifacts(v, density_threshold=2000.0, rng=rng)),
    ("ring", lambda v, rng: artifacts.add_ring_artifacts(v, ring_intensity=80.0, rng=rng)),
    ("motion", lambda v, rng: artifacts.add_motion_artifact(v, blur_size=5, severity=0.5, rng=rng)),
    ("poisson", lambda v, rng: artifacts.add_poisson_noise(v, scale=150.0, rng=rng)),
]

DTYPE_PRESERVING_FUNCS = [
    ("bias", lambda v, rng: artifacts.add_bias_field_shading(v, strength=0.25, rng=rng)),
    ("rician", lambda v, rng: artifacts.add_rician_noise(v, sigma=20.0, rng=rng)),
    ("distortion", lambda v, rng: artifacts.add_mri_geometric_distortion(v, gradient_strength=0.05, b0_strength=2.0, rng=rng)),
    ("gibbs", lambda v, rng: artifacts.add_gibbs_ringing(v, strength=0.6, truncation=0.2)),
]


@pytest.mark.parametrize("name,func", INT16_FUNCS, ids=[n for n, _ in INT16_FUNCS])
def test_int16_funcs_shape_dtype_and_bounds(name, func):
    volume = _metal_volume() if name == "metal" else _volume()
    out = func(volume, np.random.default_rng(0))
    assert out.shape == volume.shape
    assert out.dtype == np.int16
    assert int(out.min()) >= MIN_HU
    assert int(out.max()) <= MAX_HU


@pytest.mark.parametrize("name,func", DTYPE_PRESERVING_FUNCS, ids=[n for n, _ in DTYPE_PRESERVING_FUNCS])
@pytest.mark.parametrize("dtype", [np.int16, np.uint16, np.float32])
def test_dtype_preserving_funcs(name, func, dtype):
    volume = np.clip(_volume(low=0, high=250), 0, None).astype(dtype)
    out = func(volume, np.random.default_rng(0))
    assert out.shape == volume.shape
    assert out.dtype == dtype


def test_disable_paths_return_input_unchanged():
    volume = _volume()
    assert np.array_equal(artifacts.add_gaussian_noise(volume, 0.0), volume)
    assert np.array_equal(artifacts.apply_partial_volume_effect(volume, kernel_size=1), volume)
    assert np.array_equal(artifacts.add_motion_artifact(volume, blur_size=1), volume)
    assert np.array_equal(artifacts.add_motion_artifact(volume, severity=0.0), volume)
    assert np.array_equal(artifacts.add_poisson_noise(volume, scale=0.0), volume)
    assert artifacts.add_bias_field_shading(volume, strength=0.0) is volume
    assert artifacts.add_rician_noise(volume, sigma=0.0) is volume
    assert artifacts.add_mri_geometric_distortion(volume, gradient_strength=0.0, b0_strength=0.0) is volume
    assert artifacts.add_gibbs_ringing(volume, strength=0.0) is volume
    # No voxels above the metal threshold: volume passes through numerically.
    assert np.array_equal(artifacts.add_metal_artifacts(volume, density_threshold=5000.0, rng=1), volume)


def test_validation_errors():
    volume = _volume()
    with pytest.raises(ValueError):
        artifacts.apply_partial_volume_effect(volume, kernel_size=4)
    with pytest.raises(ValueError):
        artifacts.add_motion_artifact(volume, blur_size=4)
    with pytest.raises(ValueError):
        artifacts.add_motion_artifact(volume, axis=2)
    with pytest.raises(ValueError):
        artifacts.add_ring_artifacts(volume, thickness=0.0)
    with pytest.raises(ValueError):
        artifacts.add_ring_artifacts(volume, ring_radius=1.5)
    with pytest.raises(ValueError):
        artifacts.add_mri_geometric_distortion(volume, gradient_strength=0.05, readout_axis=2)


@pytest.mark.parametrize(
    "name,func",
    INT16_FUNCS + DTYPE_PRESERVING_FUNCS,
    ids=[n for n, _ in INT16_FUNCS + DTYPE_PRESERVING_FUNCS],
)
def test_determinism_per_seed(name, func):
    volume = _metal_volume() if name == "metal" else _volume()
    out_a = func(volume, np.random.default_rng(42))
    out_b = func(volume, np.random.default_rng(42))
    assert np.array_equal(out_a, out_b)


def test_gaussian_noise_statistics():
    volume = np.zeros((32, 32, 8), dtype=np.int16)
    out = artifacts.add_gaussian_noise(volume, 50.0, rng=np.random.default_rng(3))
    measured = float(np.std(out.astype(np.float64)))
    assert abs(measured - 50.0) / 50.0 < 0.05


def test_rician_noise_statistics_on_zero_signal():
    volume = np.zeros((32, 32, 8), dtype=np.float32)
    sigma = 20.0
    out = artifacts.add_rician_noise(volume, sigma, rng=np.random.default_rng(3))
    assert float(out.min()) >= 0.0
    expected_mean = sigma * np.sqrt(np.pi / 2.0)
    measured_mean = float(np.mean(out))
    assert abs(measured_mean - expected_mean) / expected_mean < 0.1


def test_poisson_noise_decreases_with_scale():
    volume = np.zeros((32, 32, 8), dtype=np.int16)
    noisy_low = artifacts.add_poisson_noise(volume, scale=50.0, rng=np.random.default_rng(3))
    noisy_high = artifacts.add_poisson_noise(volume, scale=5000.0, rng=np.random.default_rng(3))
    assert float(np.std(noisy_high.astype(np.float64))) < float(np.std(noisy_low.astype(np.float64)))


def test_partial_volume_softens_edges():
    volume = np.full((16, 16, 4), -1000, dtype=np.int16)
    volume[8:, :, :] = 1000
    out = artifacts.apply_partial_volume_effect(volume, kernel_size=5)
    # The boundary rows must now hold intermediate values.
    boundary = out[7:9, :, :]
    assert np.any((boundary > -900) & (boundary < 900))


# ---------------------------------------------------------------------------
# Golden-seed regression captures. Generated once (see tests/data) so that the
# vectorized rewrites of these artifact paths remain bit-exact.
# ---------------------------------------------------------------------------

def _golden_inputs() -> np.ndarray:
    return _volume(shape=(24, 20, 6), seed=11, low=0, high=1200)


def _golden_outputs() -> dict[str, np.ndarray]:
    volume = _golden_inputs()
    return {
        "motion": artifacts.add_motion_artifact(
            volume, blur_size=5, severity=0.6, axis=1, rng=np.random.default_rng(5)
        ),
        "gibbs": artifacts.add_gibbs_ringing(volume, strength=0.7, truncation=0.25),
        "distortion": artifacts.add_mri_geometric_distortion(
            volume, gradient_strength=0.06, b0_strength=2.5, b0_scale=0.3, rng=np.random.default_rng(5)
        ),
    }


def test_golden_seed_regressions():
    if not GOLDEN_PATH.exists():
        pytest.skip("golden capture file not generated")
    golden = np.load(GOLDEN_PATH)
    outputs = _golden_outputs()
    # Motion deliberately changed from circular to physical padding, so its
    # legacy capture remains in the archive for history but is not compared.
    for key, actual in outputs.items():
        if key == "motion":
            continue
        assert np.array_equal(actual, golden[key]), f"golden mismatch for {key!r}"


def test_bilinear_remap_preserves_last_row_and_column():
    image = np.arange(12, dtype=np.float32).reshape(3, 4)
    coord0, coord1 = np.indices(image.shape, dtype=np.float32)
    out = artifacts._remap_bilinear(image, coord0, coord1, fill=-99.0)
    np.testing.assert_array_equal(out, image)


@pytest.mark.parametrize("shape", [(1, 5), (5, 1), (1, 1)])
def test_bilinear_remap_supports_singleton_dimensions(shape):
    image = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    coord0, coord1 = np.indices(shape, dtype=np.float32)
    out = artifacts._remap_bilinear(image, coord0, coord1, fill=-99.0)
    np.testing.assert_array_equal(out, image)


def test_motion_does_not_wrap_anatomy_across_image_edges():
    volume = np.full((9, 7, 2), -1000, dtype=np.int16)
    volume[0, 3, :] = 3000
    out = artifacts.add_motion_artifact(
        volume,
        blur_size=5,
        severity=1.0,
        axis=0,
        rng=np.random.default_rng(4),
        fill_value=-1000.0,
    )
    assert np.all(out[-1, :, :] == -1000)


def test_gradient_only_distortion_matches_per_slice_remap():
    """The batched gather used when no B0 field is present must equal the
    per-slice bilinear remap it replaced."""
    volume = _volume(shape=(20, 16, 5), seed=3, low=0, high=300).astype(np.float32)
    out = artifacts.add_mri_geometric_distortion(
        volume, gradient_strength=0.08, b0_strength=0.0
    )

    width, height, depth = volume.shape
    c0 = (width - 1) / 2.0
    c1 = (height - 1) / 2.0
    o0, o1 = np.indices((width, height), dtype=np.float32)
    rel0 = o0 - c0
    rel1 = o1 - c1
    r_max_sq = (c0 * c0 + c1 * c1) + 1e-3
    rho_sq = (rel0 * rel0 + rel1 * rel1) / r_max_sq
    grad_factor = (1.0 - 0.08 * rho_sq).astype(np.float32)
    base0 = c0 + grad_factor * rel0
    base1 = c1 + grad_factor * rel1
    jacobian = artifacts._inverse_map_jacobian(base0, base1)
    expected = np.empty_like(volume)
    for iz in range(depth):
        expected[:, :, iz] = artifacts._remap_bilinear(volume[:, :, iz], base0, base1, fill=0.0) * jacobian

    np.testing.assert_array_equal(out, expected)


def test_ring_artifact_statistics():
    """Ring artifacts are checked statistically: the vectorized rewrite draws
    per-slice randoms in batches, which changes the RNG stream order."""
    volume = np.zeros((48, 40, 6), dtype=np.int16)
    out = artifacts.add_ring_artifacts(
        volume, ring_intensity=100.0, ring_radius=0.5, thickness=0.05, jitter=0.0,
        rng=np.random.default_rng(9),
    )
    added = out.astype(np.float64)
    peak = float(np.abs(added).max())
    # The ring must be visible (a solid fraction of ring_intensity) and bounded
    # by the requested amplitude plus the per-slice modulation headroom.
    assert 30.0 <= peak <= 110.0
    # The pattern is shared across slices: per-slice peaks stay similar.
    slice_peaks = np.abs(added).max(axis=(0, 1))
    assert slice_peaks.min() > 0.5 * slice_peaks.max()


@pytest.mark.parametrize("background_hu", [-800, -75, 50, 400, 1100])
def test_gaussian_noise_has_no_truncation_bias(background_hu):
    """The int16 cast must round, not truncate toward zero.

    Truncation shifts every voxel about half a HU toward 0, which turns
    nominally zero-mean noise into a signed offset whose direction flips
    with the sign of the tissue's HU.
    """

    volume = np.full((64, 64, 16), background_hu, dtype=np.int16)
    out = artifacts.add_gaussian_noise(volume, 20.0, rng=np.random.default_rng(11))
    bias = float(np.mean(out.astype(np.float64))) - background_hu
    assert abs(bias) < 0.1, f"mean shifted by {bias:+.3f} HU"


@pytest.mark.parametrize("background_hu", [75, 800])
def test_ring_artifacts_round_symmetrically_about_zero(background_hu):
    """Mirrored backgrounds must pick up the same ring pattern.

    The ring field depends only on the seed, so ``mean(out) - background``
    has to agree for +B and -B. Truncation toward zero pulls the positive
    case down and the negative case up, splitting the two by about 1 HU.
    """

    shape = (48, 48, 8)
    above = artifacts.add_ring_artifacts(
        np.full(shape, background_hu, dtype=np.int16),
        ring_intensity=80.0,
        ring_radius=0.5,
        jitter=0.0,
        rng=np.random.default_rng(11),
    )
    below = artifacts.add_ring_artifacts(
        np.full(shape, -background_hu, dtype=np.int16),
        ring_intensity=80.0,
        ring_radius=0.5,
        jitter=0.0,
        rng=np.random.default_rng(11),
    )
    shift_above = float(np.mean(above.astype(np.float64))) - background_hu
    shift_below = float(np.mean(below.astype(np.float64))) + background_hu
    assert abs(shift_above - shift_below) < 0.05


def test_metal_artifacts_round_symmetrically_about_zero():
    """The streak field is seed-driven, so mirrored backgrounds must match."""

    shape = (48, 48, 4)
    metal = (slice(22, 26), slice(22, 26), slice(None))

    def run(background_hu: int) -> float:
        volume = np.full(shape, background_hu, dtype=np.int16)
        volume[metal] = 3000
        out = artifacts.add_metal_artifacts(
            volume,
            intensity=200.0,
            density_threshold=2000.0,
            rng=np.random.default_rng(11),
        )
        soft = volume != 3000
        return float(np.mean(out[soft].astype(np.float64))) - background_hu

    assert abs(run(75) - run(-75)) < 0.05


@pytest.mark.parametrize("shape,out_shape", [((1, 6), (8, 8)), ((6, 1), (8, 8)), ((1, 1), (4, 5))])
def test_resize_bilinear_repeats_a_degenerate_source_axis(shape, out_shape):
    """A length-1 source axis must broadcast, not fall outside the image.

    ``add_metal_artifacts`` resamples every slice to at least 8x8 for the
    projection loop, so a one-voxel-thin grid hits this path; stepping along
    an axis with no span pushed every sample but the first out of bounds,
    where it was replaced by the zero fill.
    """

    image = np.arange(1, np.prod(shape) + 1, dtype=np.float32).reshape(shape)
    out = artifacts._resize_bilinear(image, out_shape)
    assert out.shape == out_shape
    assert float(out.min()) > 0.0
    if shape[0] == 1:
        # Every output row must reproduce the single source row.
        np.testing.assert_allclose(out, np.repeat(out[:1, :], out_shape[0], axis=0), rtol=1e-6)
    if shape[1] == 1:
        np.testing.assert_allclose(out, np.repeat(out[:, :1], out_shape[1], axis=1), rtol=1e-6)


def test_metal_streaks_do_not_depend_on_slice_orientation():
    """Transposing a wide slice must transpose its streak field.

    Rotating inside a slice-sized canvas cropped the corners, so a laterally
    placed implant disappeared from the near-vertical views of a wide slice
    (but not of the same slice stored tall), and the streak pattern depended
    on which way the grid happened to be longer.
    """
    volume = np.zeros((200, 80, 1), dtype=np.int16)
    volume[28:33, 38:43, 0] = 3000
    kwargs = dict(intensity=400.0, density_threshold=2000.0, num_streaks=64, falloff=0.1)
    wide = artifacts.add_metal_artifacts(volume, rng=np.random.default_rng(0), **kwargs)
    tall = artifacts.add_metal_artifacts(
        np.ascontiguousarray(volume.transpose(1, 0, 2)), rng=np.random.default_rng(0), **kwargs
    )
    wide = wide[:, :, 0].astype(np.float32)
    tall = tall[:, :, 0].astype(np.float32).T
    # Photon-starvation noise and the view sampling are not transpose
    # symmetric, so allow a modest residual; the cropped canvas gave a
    # difference larger than the streaks themselves.
    assert np.abs(wide - tall).mean() < 0.5 * np.abs(wide).mean()


def test_metal_streak_strength_scales_with_metal_amount():
    """Slices with less, or less dense, metal get weaker streaks."""
    volume = np.zeros((48, 48, 2), dtype=np.int16)
    volume[12:36, 12:36, 0] = 3071   # large dense implant
    volume[24, 24, 1] = 2001         # single voxel barely over threshold
    out = artifacts.add_metal_artifacts(
        volume, intensity=400.0, density_threshold=2000.0, num_streaks=32,
        rng=np.random.default_rng(1),
    )
    strong = np.abs(out[:, :, 0].astype(np.float32))
    strong[12:36, 12:36] = 0.0
    weak = np.abs(out[:, :, 1].astype(np.float32))
    weak[24, 24] = 0.0
    assert np.percentile(strong, 99.5) > 100.0
    assert weak.max() < 1.0


def test_ring_artifacts_are_circular_on_non_square_slices():
    """Detector-channel rings are circles about the centre, not ellipses."""
    volume = np.zeros((201, 81, 1), dtype=np.int16)
    out = artifacts.add_ring_artifacts(
        volume, ring_intensity=100.0, ring_radius=0.3, thickness=0.02, jitter=0.0,
        rng=np.random.default_rng(2),
    ).astype(np.float32)[:, :, 0]
    centre0, centre1 = 100, 40
    along_x = np.abs(out[centre0:, centre1])
    along_y = np.abs(out[centre0, centre1:])
    # Allow for the small random centre jitter.
    assert abs(int(np.argmax(along_x)) - int(np.argmax(along_y))) <= 2


def test_positive_gradient_nonlinearity_is_pincushion():
    """k > 0 must push peripheral content outward, as documented in the UI."""
    volume = np.zeros((81, 81, 1), dtype=np.float32)
    volume[40 + 30, 40, 0] = 1000.0
    out = artifacts.add_mri_geometric_distortion(volume, gradient_strength=0.2, b0_strength=0.0)
    assert int(np.argmax(out[:, 40, 0])) > 70
    barrel = artifacts.add_mri_geometric_distortion(volume, gradient_strength=-0.2, b0_strength=0.0)
    assert int(np.argmax(barrel[:, 40, 0])) < 70


def test_gibbs_mask_keeps_dc_term_for_even_widths():
    """The kept k-space band is centred on DC even at the maximum truncation."""
    volume = np.full((64, 64, 1), 100, dtype=np.int16)
    out = artifacts.add_gibbs_ringing(volume, strength=1.0, truncation=0.49)
    np.testing.assert_array_equal(out, volume)


def test_gibbs_ringing_preserves_symmetry_for_odd_band_on_even_width():
    """An off-centre band would skew a symmetric object's ringing."""
    volume = np.zeros((64, 64, 1), dtype=np.float32)
    volume[24:40, 24:40, 0] = 100.0
    out = artifacts.add_gibbs_ringing(volume, strength=1.0, truncation=0.21)[:, :, 0]
    # The square is symmetric about the 31.5 midline, and so is the DC-centred
    # band (up to the lone Nyquist-side bin, which a centred band never keeps).
    np.testing.assert_allclose(out, out[::-1, :], atol=1e-3)


def test_poisson_noise_is_unbiased_at_low_photon_scale():
    """-log(N/lambda) is biased high for Poisson counts; the correction keeps
    quantum noise zero-mean even at the lowest photon scale."""
    volume = np.zeros((96, 96, 12), dtype=np.int16)  # water
    out = artifacts.add_poisson_noise(volume, scale=1.0, rng=np.random.default_rng(3))
    noise = out.astype(np.float64)
    # Standard error of the mean is ~0.3 HU here; the uncorrected bias was ~20 HU.
    assert abs(noise.mean()) < 2.0
    assert noise.std() > 100.0


def test_b0_distortion_conserves_signal_and_piles_up():
    """Off-resonance distortion moves signal, it does not create or destroy it.

    Without the Jacobian, a compressed region kept its intensity while losing
    area, so the total MR signal of an object changed with the field map.
    """
    x = np.arange(64, dtype=np.float32)
    blob = np.exp(-(((x[:, None] - 32) ** 2) + ((x[None, :] - 32) ** 2)) / (2 * 8.0**2))
    volume = np.repeat((1000.0 * blob)[:, :, None], 3, axis=2).astype(np.float32)
    out = artifacts.add_mri_geometric_distortion(
        volume, gradient_strength=0.0, b0_strength=4.0, b0_scale=0.3, rng=np.random.default_rng(8)
    )
    # Conserved to ~0.02% with the Jacobian; ~1.5% lost without it.
    np.testing.assert_allclose(out.sum(axis=(0, 1)), volume.sum(axis=(0, 1)), rtol=0.002)
    # Compressed regions brighten beyond the undistorted peak.
    assert out.max() > volume.max() * 1.02


def test_inverse_map_jacobian_of_uniform_scaling():
    """Sampling the source at twice the radius shows 4x the area per pixel."""
    o0, o1 = np.indices((9, 7), dtype=np.float32)
    jacobian = artifacts._inverse_map_jacobian(2.0 * o0, 2.0 * o1)
    np.testing.assert_allclose(jacobian, 4.0)
    single_column = artifacts._inverse_map_jacobian(2.0 * o0[:, :1], o1[:, :1])
    np.testing.assert_allclose(single_column, 2.0)
