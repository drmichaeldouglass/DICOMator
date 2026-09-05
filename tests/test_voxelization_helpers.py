"""Pure helper tests for voxel spacing, overlap ordering, and ray restarts."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from mathutils import Vector

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


class _SlabBVH:
    """Ray-cast stand-in for a solid slab spanning the whole XY plane."""

    def __init__(self, z_min: float, z_max: float):
        self._faces = (float(z_min), float(z_max))

    def ray_cast(self, origin, direction, max_dist):
        for face_z in self._faces:
            if face_z > origin.z and (face_z - origin.z) <= max_dist:
                return Vector((origin.x, origin.y, face_z)), None, 0, face_z - origin.z
        return None, None, None, None


@pytest.mark.parametrize("kind", ["hu", "dose"])
@pytest.mark.parametrize("slab_bounds", [(-0.1, 0.1), (-0.1, 0.003), (-0.1, -0.02)])
def test_cropped_grid_preserves_surface_pairing(kind, slab_bounds):
    """The crop may start inside a solid, but the ray must start outside it."""
    lower, upper = slab_bounds
    obj = SimpleNamespace(name="Slab", dicomator_hu=250.0, dicomator_dose=2.0)
    factory = getattr(voxelization, f"voxelize_objects_to_{kind}_iter")
    grid, _origin, _dims = voxelization._drive(
        factory(
            [obj], voxel_size=0.001, padding=0,
            bbox_override=(0.0, 0.002, 0.0, 0.002, 0.0, 0.005),
            prepared={"Slab": (
                _SlabBVH(lower, upper),
                (0.0, 0.002, 0.0, 0.002, lower, upper),
            )},
        ),
        None,
    )
    z_centers = (np.arange(5) + 0.5) * 0.001
    background, value = (-1000, 250) if kind == "hu" else (0, 2)
    expected = np.where((z_centers >= lower) & (z_centers <= upper), value, background)
    np.testing.assert_array_equal(grid, np.broadcast_to(expected, grid.shape))


def _voxelize_slab(hu_value: float) -> np.ndarray:
    """Fill a small grid from one slab mesh carrying ``hu_value``."""

    obj = SimpleNamespace(name="Slab", dicomator_hu=hu_value, dicomator_priority=0)
    bounds = (0.0, 0.005, 0.0, 0.005, 0.001, 0.004)
    grid, _origin, _dims = voxelization._drive(
        voxelization.voxelize_objects_to_hu_iter(
            [obj],
            voxel_size=(0.001, 0.001, 0.001),
            padding=0,
            bbox_override=(0.0, 0.005, 0.0, 0.005, 0.0, 0.005),
            prepared={"Slab": (_SlabBVH(0.001, 0.004), bounds)},
        ),
        None,
    )
    return grid


@pytest.mark.parametrize(
    ("hu_value", "expected"),
    [(50.7, 51), (-75.6, -76), (-0.9, -1), (300.4, 300), (1100.0, 1100)],
)
def test_fractional_hu_is_rounded_not_truncated(hu_value, expected):
    """A fractional HU must reach the grid as its nearest integer.

    ``dicomator_hu`` is a float property, so dragging the slider stores values
    such as -75.6. NumPy casts a float into an int16 grid by truncating toward
    zero, which would store -75 here and 50 for 50.7: an error of up to 1 HU
    whose sign follows the tissue instead of cancelling out.
    """

    grid = _voxelize_slab(hu_value)
    filled = np.unique(grid[grid != voxelization.AIR_DENSITY])

    assert filled.tolist() == [expected]


def test_air_background_is_written_exactly():
    """Voxels no mesh covers stay at the air value the caller asked for."""

    grid = _voxelize_slab(300.0)

    assert grid[0, 0, 0] == voxelization.AIR_DENSITY
    assert grid.dtype == np.int16
