"""Tests for the pure-NumPy math helpers in drr.py."""
from __future__ import annotations

import numpy as np

from conftest import load_module

drr = load_module("drr")

BOX_MIN = np.array([0.0, 0.0, 0.0], dtype=np.float32)
BOX_MAX = np.array([1.0, 1.0, 1.0], dtype=np.float32)


def _intersect(origin, direction):
    origins = np.asarray([origin], dtype=np.float32)
    directions = np.asarray([direction], dtype=np.float32)
    entry, exit_, valid = drr._ray_box_intersections(origins, directions, BOX_MIN, BOX_MAX)
    return float(entry[0]), float(exit_[0]), bool(valid[0])


def test_ray_hits_box():
    entry, exit_, valid = _intersect((-2.0, 0.5, 0.5), (1.0, 0.0, 0.0))
    assert valid
    np.testing.assert_allclose(entry, 2.0, atol=1e-5)
    np.testing.assert_allclose(exit_, 3.0, atol=1e-5)


def test_ray_misses_box():
    entry, exit_, valid = _intersect((-2.0, 5.0, 5.0), (1.0, 0.0, 0.0))
    assert not valid
    assert entry == 0.0
    assert exit_ == 0.0


def test_ray_origin_inside_box_clamps_entry_to_zero():
    entry, exit_, valid = _intersect((0.5, 0.5, 0.5), (1.0, 0.0, 0.0))
    assert valid
    assert entry == 0.0
    np.testing.assert_allclose(exit_, 0.5, atol=1e-5)


def test_axis_parallel_ray_with_zero_components():
    entry, exit_, valid = _intersect((0.5, 0.5, -2.0), (0.0, 0.0, 1.0))
    assert valid
    np.testing.assert_allclose(entry, 2.0, atol=1e-4)
    np.testing.assert_allclose(exit_, 3.0, atol=1e-4)


def test_ray_pointing_away_is_invalid():
    entry, exit_, valid = _intersect((-2.0, 0.5, 0.5), (-1.0, 0.0, 0.0))
    assert not valid


def test_normalize_projection_zeros():
    out = drr._normalize_projection(np.zeros((8, 10), dtype=np.float32))
    assert out.dtype == np.uint16
    assert not np.any(out)


def test_normalize_projection_fixed_maps_absorption_directly():
    integrals = np.full((4, 4), -float(np.log(0.75)), dtype=np.float32)  # 1-exp(-L) = 0.25
    out = drr._normalize_projection(integrals, fixed=True)
    assert out.dtype == np.uint16
    expected = int(np.round(0.25 * 65535.0))
    assert np.all(np.abs(out.astype(np.int64) - expected) <= 1)


def test_normalize_projection_percentile_spans_uint16_range():
    rng = np.random.default_rng(2)
    integrals = rng.uniform(0.05, 4.0, size=(64, 64)).astype(np.float32)
    out = drr._normalize_projection(integrals)
    assert out.dtype == np.uint16
    assert int(out.min()) == 0
    assert int(out.max()) == 65535
