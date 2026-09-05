"""Tests for the pure-NumPy math helpers in drr.py."""
from __future__ import annotations

import numpy as np
import pytest

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


@pytest.mark.parametrize("x", [0.0, 1.0])
def test_parallel_ray_on_box_face_keeps_its_full_path(x):
    entry, exit_, valid = _intersect((x, 0.5, -2.0), (0.0, 0.0, 1.0))
    assert valid
    np.testing.assert_allclose([entry, exit_], [2.0, 3.0])


def test_parallel_ray_just_outside_box_is_not_tilted_into_it():
    _entry, _exit, valid = _intersect((-1e-9, 0.5, -2.0), (0.0, 0.0, 1.0))
    assert not valid


def test_small_negative_direction_keeps_its_sign():
    origins = np.array([[1e-9, 0.5, 0.0]])
    directions = np.array([[-1e-9, 0.0, 1.0]])
    entry, exit_, valid = drr._ray_box_intersections(
        origins, directions, BOX_MIN, np.array([1.0, 1.0, 10.0])
    )
    assert valid[0]
    np.testing.assert_allclose([entry[0], exit_[0]], [0.0, 1.0])


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


def test_hu_to_linear_attenuation_uses_physical_water_scale():
    hu = np.array([-1000.0, 0.0, 1000.0], dtype=np.float32)
    attenuation = drr._hu_to_linear_attenuation(hu, 20.0)
    np.testing.assert_allclose(attenuation, [0.0, 20.0, 40.0], atol=1e-6)
    assert attenuation.dtype == np.float32


def test_hu_to_linear_attenuation_rejects_invalid_water_coefficient():
    hu = np.zeros((2, 2), dtype=np.float32)
    for value in (0.0, -1.0, np.nan, np.inf):
        try:
            drr._hu_to_linear_attenuation(hu, value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected invalid water coefficient {value!r} to fail")
