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
    for key, expected in golden.items():
        assert np.array_equal(outputs[key], expected), f"golden mismatch for {key!r}"


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
    grad_factor = (1.0 + 0.08 * rho_sq).astype(np.float32)
    base0 = c0 + grad_factor * rel0
    base1 = c1 + grad_factor * rel1
    expected = np.empty_like(volume)
    for iz in range(depth):
        expected[:, :, iz] = artifacts._remap_bilinear(volume[:, :, iz], base0, base1, fill=0.0)

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
