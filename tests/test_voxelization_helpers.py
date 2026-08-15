"""Pure helper tests for voxel spacing, overlap ordering, and ray restarts."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from conftest import load_module

voxelization = load_module("voxelization")


@pytest.mark.parametrize("value", [0.0, -0.001, (0.001, 0.0, 0.001)])
def test_voxel_size_must_be_positive(value):
    with pytest.raises(ValueError, match="greater than zero"):
        voxelization._resolve_voxel_size(value)


def test_overlap_priority_sorts_highest_last():
    objects = [
        SimpleNamespace(name="Bone", dicomator_priority=10),
        SimpleNamespace(name="Soft", dicomator_priority=0),
        SimpleNamespace(name="Air", dicomator_priority=-10),
    ]
    ordered = sorted(objects, key=voxelization._object_priority_key)
    assert [obj.name for obj in ordered] == ["Air", "Soft", "Bone"]


@pytest.mark.parametrize(
    "hit_z",
    [0.0, 1e-4, 0.5, -0.5, 1.0, 31.9, -31.9, 32.0, -32.0, 1000.0, -1000.0, 1e6],
)
def test_restarted_ray_clears_the_face_in_single_precision(hit_z):
    """The nudge must survive the float32 rounding mathutils applies.

    A fixed 1e-6 m step vanishes once the float32 spacing swallows it (from
    |z| = 32 m upward), leaving the restarted ray exactly on the face it just
    hit and the marching loop reporting that crossing forever.
    """

    restarted = voxelization.restart_z_past_hit(hit_z)
    assert restarted > hit_z
    # mathutils.Vector stores single precision, so the advance has to remain
    # visible after the narrowing conversion.
    assert np.float32(restarted) > np.float32(hit_z)


def test_restart_step_stays_sub_voxel_at_human_scale():
    """The nudge must not skip past genuinely nearby surfaces."""

    # 0.1 mm is the finest voxel spacing the UI allows.
    assert voxelization.restart_z_past_hit(0.25) - 0.25 < 1e-4
    assert voxelization.restart_z_past_hit(-0.25) + 0.25 < 1e-4
