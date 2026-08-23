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


class _EmptyVertices:
    """Vertex collection of an object whose modifiers emptied the mesh."""

    def __len__(self) -> int:
        return 0

    def foreach_get(self, attribute, buffer) -> None:
        pass


class _EmptyMesh:
    vertices = _EmptyVertices()
    polygons = _EmptyVertices()
    loops = _EmptyVertices()


class _IdentityMatrix:
    def __matmul__(self, other):
        return other

    def __array__(self, dtype=None, copy=None):
        return np.eye(4, dtype=dtype or float)


class _EmptyObject:
    name = "Emptied"
    data = _EmptyMesh()
    matrix_world = _IdentityMatrix()
    bound_box = ()
    dicomator_hu = 0.0
    dicomator_priority = 0


def test_empty_geometry_reports_a_usable_error_not_an_overflow():
    """Meshes with no vertices leave the bounds at +/-inf.

    ``math.ceil(inf)`` then raises ``OverflowError: cannot convert float
    infinity to integer``, which surfaces in the UI as an export failure that
    says nothing about the scene. The voxelizer has to name the real problem.
    """

    generator = voxelization.voxelize_objects_to_hu_iter(
        [_EmptyObject()], voxel_size=0.002, padding=1
    )
    with pytest.raises(ValueError, match="no voxelizable|any vertices"):
        voxelization._drive(generator, None)


class _RecordingVertices:
    def __init__(self, coordinates):
        self._coordinates = list(coordinates)
        self.requested_dtype = None

    def __len__(self) -> int:
        return len(self._coordinates) // 3

    def foreach_get(self, attribute, buffer) -> None:
        assert attribute == "co"
        self.requested_dtype = buffer.dtype
        buffer[:] = self._coordinates


def test_vertex_read_uses_a_float32_buffer():
    """``foreach_get`` only bulk-copies when the buffer matches the RNA type.

    ``MeshVertex.co`` is float32, so a float64 buffer silently drops Blender
    into a per-vertex Python loop -- the exact cost this helper exists to
    avoid.
    """

    vertices = _RecordingVertices([1.0, 2.0, 3.0, -4.0, -5.0, -6.0])
    mesh = SimpleNamespace(vertices=vertices)
    world = voxelization._world_vertex_array(mesh, _IdentityMatrix())

    assert vertices.requested_dtype == np.float32
    assert world.dtype == np.float64
    np.testing.assert_allclose(world, [[1.0, 2.0, 3.0], [-4.0, -5.0, -6.0]])
