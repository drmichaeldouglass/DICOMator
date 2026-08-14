"""Pure helper tests for voxel spacing and overlap ordering."""
from __future__ import annotations

from types import SimpleNamespace

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
