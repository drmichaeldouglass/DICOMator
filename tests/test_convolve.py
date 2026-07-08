"""Exact-equality safety net for the 1D convolution helpers in artifacts.py."""
from __future__ import annotations

import numpy as np
import pytest

from conftest import load_module

artifacts = load_module("artifacts")


def _brute_force_convolve(array: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    arr = array.astype(np.float32, copy=False)
    kernel = kernel.astype(np.float32, copy=False)
    pad = int(kernel.size // 2)
    moved = np.ascontiguousarray(np.moveaxis(arr, axis, -1))
    length = moved.shape[-1]
    flat = moved.reshape(-1, length)
    out_flat = np.empty_like(flat)
    for i in range(flat.shape[0]):
        padded = np.pad(flat[i], (pad, pad), mode="edge")
        row = np.convolve(padded, kernel, mode="valid")
        out_flat[i] = row[:length]
    return np.moveaxis(out_flat.reshape(moved.shape), -1, axis)


@pytest.mark.parametrize("shape", [(17,), (9, 13), (8, 7, 6)])
@pytest.mark.parametrize("kernel_size", [2, 3, 5, 7])
def test_convolve_matches_brute_force(shape, kernel_size):
    rng = np.random.default_rng(1)
    array = rng.normal(0.0, 100.0, size=shape).astype(np.float32)
    kernel = rng.uniform(0.1, 1.0, size=kernel_size).astype(np.float32)
    kernel /= kernel.sum()
    for axis in range(len(shape)):
        got = artifacts._convolve_along_axis(array, kernel, axis)
        expected = _brute_force_convolve(array, kernel, axis)
        assert got.shape == array.shape
        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-3)


def test_convolve_kernel_of_one_is_identity():
    array = np.arange(24, dtype=np.float32).reshape(4, 6)
    out = artifacts._convolve_along_axis(array, np.ones(1, dtype=np.float32), 0)
    np.testing.assert_array_equal(out, array)


def test_moving_average_of_constant_is_constant():
    array = np.full((10, 12, 4), 37.0, dtype=np.float32)
    out = artifacts._moving_average_along_axis(array, 5, axis=1)
    np.testing.assert_allclose(out, array, rtol=1e-6)


def test_gaussian_kernel_normalized_and_symmetric():
    kernel = artifacts._gaussian_kernel(7)
    assert kernel.shape == (7,)
    np.testing.assert_allclose(float(kernel.sum()), 1.0, rtol=1e-6)
    np.testing.assert_allclose(kernel, kernel[::-1], rtol=1e-6)
    with pytest.raises(ValueError):
        artifacts._gaussian_kernel(6)
