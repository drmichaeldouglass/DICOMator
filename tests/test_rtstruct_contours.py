"""Plane selection tests for the RT Structure contour extractor.

``_extract_contours_iter`` copies and bisects the whole mesh once per Z
plane, so planes that cannot possibly intersect the structure are pure waste
on a grid that is much taller than the ROI. These tests pin the skipping
behaviour and confirm the emitted contour dictionary is unchanged by it.
"""
from __future__ import annotations

import types

import pytest

from conftest import load_module

rtstruct_export = load_module("rtstruct_export")


class _FakeVert:
    def __init__(self, z: float):
        self.co = types.SimpleNamespace(x=0.0, y=0.0, z=float(z))


class _FakeBMesh:
    def __init__(self, z_values):
        self.verts = [_FakeVert(z) for z in z_values]
        self.freed = False

    def free(self) -> None:
        self.freed = True


def _drive(generator):
    while True:
        try:
            next(generator)
        except StopIteration as stop:
            return stop.value


@pytest.fixture
def patched(monkeypatch):
    """Replace the bmesh-dependent helpers with recording fakes."""

    state = types.SimpleNamespace(bisected=[], mesh=None)

    def fake_world_bmesh(obj, depsgraph, *, apply_modifiers):
        state.mesh = _FakeBMesh(obj.z_values)
        return state.mesh

    def fake_slice_plane(bm_base, z_m):
        state.bisected.append(float(z_m))
        return [[(0.0, 0.0, float(z_m))] * 4], 0

    monkeypatch.setattr(rtstruct_export, "_world_bmesh", fake_world_bmesh)
    monkeypatch.setattr(rtstruct_export, "_slice_plane", fake_slice_plane)
    return state


def test_planes_outside_the_mesh_are_not_bisected(patched):
    """A 3 cm ROI in a 30 cm grid must not pay for the other 27 cm."""

    obj = types.SimpleNamespace(name="GTV", z_values=[0.100, 0.130])
    z_positions = [0.001 + i * 0.002 for i in range(150)]

    contours, dropped = _drive(
        rtstruct_export._extract_contours_iter(obj, z_positions, None, apply_modifiers=False)
    )

    # Every requested plane still appears in the result, so downstream code
    # sees exactly what it did before.
    assert sorted(contours) == pytest.approx(sorted(float(z) for z in z_positions))
    assert dropped == 0

    assert patched.bisected, "planes inside the mesh must still be bisected"
    assert all(0.100 <= z <= 0.130 for z in patched.bisected)
    assert len(patched.bisected) < len(z_positions) // 4
    # Planes outside the mesh carry no contours; inside planes keep theirs.
    assert contours[z_positions[0]] == []
    assert contours[patched.bisected[0]]


def test_every_plane_is_bisected_when_the_mesh_spans_the_grid(patched):
    obj = types.SimpleNamespace(name="Body", z_values=[-1.0, 1.0])
    z_positions = [0.001 + i * 0.002 for i in range(20)]

    _drive(rtstruct_export._extract_contours_iter(obj, z_positions, None, apply_modifiers=False))

    assert patched.bisected == pytest.approx(z_positions)


def test_planes_within_the_bisect_tolerance_are_kept(patched):
    """bisect_plane snaps near-plane vertices, so the bounds need padding."""

    tolerance = rtstruct_export._BISECT_TOLERANCE_M
    obj = types.SimpleNamespace(name="Flat", z_values=[0.05, 0.05])
    z_positions = [0.05 - tolerance / 2.0, 0.05, 0.05 + tolerance / 2.0, 0.05 + 10 * tolerance]

    _drive(rtstruct_export._extract_contours_iter(obj, z_positions, None, apply_modifiers=False))

    assert patched.bisected == pytest.approx(z_positions[:3])


def test_mesh_is_freed_even_when_no_plane_intersects(patched):
    obj = types.SimpleNamespace(name="Away", z_values=[5.0, 5.1])
    _drive(rtstruct_export._extract_contours_iter(obj, [0.0, 0.1], None, apply_modifiers=False))

    assert patched.bisected == []
    assert patched.mesh.freed


def test_a_mesh_without_vertices_yields_empty_contours(patched):
    obj = types.SimpleNamespace(name="Nothing", z_values=[])
    contours, dropped = _drive(
        rtstruct_export._extract_contours_iter(obj, [0.0, 0.1], None, apply_modifiers=False)
    )

    assert contours == {0.0: [], 0.1: []}
    assert dropped == 0
    assert patched.bisected == []
