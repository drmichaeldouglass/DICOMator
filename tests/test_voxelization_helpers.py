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


def test_bounds_without_modifiers_come_from_the_base_mesh():
    """With modifiers off the grid must enclose ``obj.data``, which is what
    gets ray-cast. ``obj.bound_box`` describes the evaluated mesh instead
    (smaller under a Boolean/Mask modifier, unset if never evaluated)."""

    class _NoBoundBox:
        name = "Masked"
        data = SimpleNamespace(
            vertices=_RecordingVertices([-0.1, -0.2, -0.3, 0.1, 0.2, 0.3])
        )
        matrix_world = _IdentityMatrix()

        @property
        def bound_box(self):
            raise AssertionError("bound_box describes the evaluated mesh")

    bounds = voxelization._objects_world_bounds([_NoBoundBox()], None, apply_modifiers=False)
    np.testing.assert_allclose(bounds, (-0.1, 0.1, -0.2, 0.2, -0.3, 0.3), atol=1e-7)


def test_value_for_object_overrides_and_is_clamped():
    """The DRR of an MR export re-voxelizes with CT numbers via this hook."""
    obj = SimpleNamespace(name="Slab", dicomator_hu=150.0, dicomator_priority=0)
    bounds = (0.0, 0.005, 0.0, 0.005, 0.001, 0.004)

    def voxelize(value_for_object):
        grid, _origin, _dims = voxelization._drive(
            voxelization.voxelize_objects_to_hu_iter(
                [obj], voxel_size=0.001, padding=0,
                bbox_override=(0.0, 0.005, 0.0, 0.005, 0.0, 0.005),
                prepared={"Slab": (_SlabBVH(0.001, 0.004), bounds)},
                value_for_object=value_for_object,
            ),
            None,
        )
        return grid

    assert int(voxelize(None).max()) == 150
    assert int(voxelize(lambda _obj: 1100.0).max()) == 1100
    assert int(voxelize(lambda _obj: 99999.0).max()) == voxelization.MAX_HU_VALUE


# ---------------------------------------------------------------------------
# Multi-shell meshes (several closed surfaces in one object)
# ---------------------------------------------------------------------------


class _TriangleBVH:
    """Brute-force ray/triangle stand-in for ``BVHTree.FromPolygons``."""

    def __init__(self, vertices, polygons):
        verts = np.asarray(vertices, dtype=np.float64)
        triangles = []
        for poly in polygons:
            for k in range(1, len(poly) - 1):
                triangles.append((verts[poly[0]], verts[poly[k]], verts[poly[k + 1]]))
        self._triangles = np.asarray(triangles)

    @classmethod
    def FromPolygons(cls, vertices, polygons):  # noqa: N802 - mirrors Blender's API
        return cls(vertices, polygons)

    def ray_cast(self, origin, direction, max_dist):
        o = np.array(tuple(origin), dtype=np.float64)
        d = np.array(tuple(direction), dtype=np.float64)
        a, b, c = self._triangles[:, 0], self._triangles[:, 1], self._triangles[:, 2]
        e1, e2 = b - a, c - a
        p = np.cross(d, e2)
        det = np.einsum("ij,ij->i", e1, p)
        ok = np.abs(det) > 1e-12
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        t_vec = o - a
        u = np.einsum("ij,ij->i", t_vec, p) * inv
        q = np.cross(t_vec, e1)
        v = (q @ d) * inv
        t = np.einsum("ij,ij->i", e2, q) * inv
        hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t >= 0) & (t <= max_dist)
        if not np.any(hit):
            return None, None, None, None
        index = int(np.flatnonzero(hit)[np.argmin(t[hit])])
        location = o + t[index] * d
        return Vector(location), None, index, float(t[index])


def _box(lo, hi, offset=0):
    """Closed axis-aligned box: 8 vertices, 6 quads (indices offset)."""
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    verts = [
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
    ]
    quads = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    return verts, [[offset + i for i in quad] for quad in quads]


class _Collection:
    def __init__(self, attribute, values, dtype):
        self._attribute = attribute
        self._values = np.asarray(values, dtype=dtype)

    def __len__(self):
        return len(self._values) // 3 if self._attribute == "co" else len(self._values)

    def foreach_get(self, attribute, buffer):
        assert attribute == self._attribute
        buffer[:] = self._values


def _mesh_object(name, boxes, hu=500.0):
    verts, polys = [], []
    for lo, hi in boxes:
        box_verts, box_polys = _box(lo, hi, offset=len(verts))
        verts.extend(box_verts)
        polys.extend(box_polys)
    loop_start = np.cumsum([0] + [len(p) for p in polys[:-1]])
    loop_verts = [i for poly in polys for i in poly]
    data = SimpleNamespace(
        vertices=_Collection("co", np.ravel(verts), np.float32),
        polygons=_Collection("loop_start", loop_start, np.int32),
        loops=_Collection("vertex_index", loop_verts, np.int32),
    )
    return SimpleNamespace(
        name=name, data=data, matrix_world=_IdentityMatrix(),
        dicomator_hu=hu, dicomator_dose=2.0, dicomator_priority=0,
    )


def _voxelize_boxes(monkeypatch, boxes, kind="hu"):
    monkeypatch.setattr(voxelization, "BVHTree", _TriangleBVH)
    obj = _mesh_object("Joined", boxes)
    factory = getattr(voxelization, f"voxelize_objects_to_{kind}_iter")
    grid, _origin, _dims = voxelization._drive(
        factory([obj], voxel_size=0.01, padding=0, bbox_override=(0.0, 0.1, 0.0, 0.1, 0.0, 0.1)),
        None,
    )
    return grid


def _box_mask(lo, hi):
    # Box faces sit between voxel centres, so the comparison is unambiguous.
    centres = (np.arange(10) + 0.5) * 0.01
    inside = [(centres >= lo[axis]) & (centres <= hi[axis]) for axis in range(3)]
    return inside[0][:, None, None] & inside[1][None, :, None] & inside[2][None, None, :]


def test_overlapping_shells_in_one_object_fill_their_overlap(monkeypatch):
    """Two organs joined into one object overlap; even/odd pairing emptied
    the shared region (in, in, out, out -> filled, empty, filled)."""
    a = ((0.01, 0.01, 0.01), (0.06, 0.06, 0.06))
    b = ((0.03, 0.03, 0.03), (0.09, 0.09, 0.09))
    grid = _voxelize_boxes(monkeypatch, [a, b])
    expected = _box_mask(*a) | _box_mask(*b)
    np.testing.assert_array_equal(grid == 500, expected)


def test_nested_shell_still_makes_a_hollow_object(monkeypatch):
    """A shell inside another shell of the same object is a cavity."""
    outer = ((0.01, 0.01, 0.01), (0.09, 0.09, 0.09))
    inner = ((0.03, 0.03, 0.03), (0.07, 0.07, 0.07))
    grid = _voxelize_boxes(monkeypatch, [outer, inner])
    expected = _box_mask(*outer) & ~_box_mask(*inner)
    np.testing.assert_array_equal(grid == 500, expected)


def test_solid_inside_a_cavity_is_filled_again(monkeypatch):
    """Nesting depth alternates solid/cavity/solid, like even/odd pairing."""
    outer = ((0.01, 0.01, 0.01), (0.09, 0.09, 0.09))
    cavity = ((0.02, 0.02, 0.02), (0.08, 0.08, 0.08))
    core = ((0.04, 0.04, 0.04), (0.06, 0.06, 0.06))
    grid = _voxelize_boxes(monkeypatch, [outer, cavity, core])
    expected = (_box_mask(*outer) & ~_box_mask(*cavity)) | _box_mask(*core)
    np.testing.assert_array_equal(grid == 500, expected)


def test_overlapping_dose_shells_count_the_object_once(monkeypatch):
    """Inside one object the dose is its value once, even where shells overlap."""
    a = ((0.01, 0.01, 0.01), (0.06, 0.06, 0.06))
    b = ((0.03, 0.03, 0.03), (0.09, 0.09, 0.09))
    grid = _voxelize_boxes(monkeypatch, [a, b], kind="dose")
    assert float(grid.max()) == pytest.approx(2.0)


def test_polygon_islands_groups_connected_faces():
    polys = [[0, 1, 2], [2, 3, 0], [4, 5, 6], [6, 7, 4], [8, 9, 10]]
    islands = sorted(sorted(island.tolist()) for island in voxelization._polygon_islands(polys, 11))
    assert islands == [[0, 1], [2, 3], [4]]
