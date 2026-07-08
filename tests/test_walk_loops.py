"""Graph-walk tests for rtstruct_export._walk_loops using fake bmesh objects.

``_walk_loops`` currently returns just the loop list; after the hardening
rewrite it returns ``(loops, dropped_edge_count)``. ``_walk`` normalizes both
forms so these tests stay valid across the change; dropped-count assertions
are skipped while the old API is in place.
"""
from __future__ import annotations

import types

import pytest

from conftest import load_module

rtstruct_export = load_module("rtstruct_export")


class FakeVert:
    def __init__(self, x: float, y: float, z: float = 0.0):
        self.co = types.SimpleNamespace(x=float(x), y=float(y), z=float(z))


class FakeEdge:
    def __init__(self, a: FakeVert, b: FakeVert):
        self.verts = (a, b)


def _chain(points, closed=True):
    verts = [FakeVert(x, y) for x, y in points]
    edges = [FakeEdge(verts[i], verts[i + 1]) for i in range(len(verts) - 1)]
    if closed:
        edges.append(FakeEdge(verts[-1], verts[0]))
    return verts, edges


def _walk(edges):
    result = rtstruct_export._walk_loops(edges)
    if isinstance(result, tuple):
        return result
    return result, None


def _loop_points(loop):
    return {(round(x, 6), round(y, 6)) for x, y, _z in loop}


def test_empty_input():
    loops, dropped = _walk([])
    assert loops == []
    if dropped is not None:
        assert dropped == 0


def test_single_square():
    square = [(0, 0), (1, 0), (1, 1), (0, 1)]
    _, edges = _chain(square)
    loops, dropped = _walk(edges)
    assert len(loops) == 1
    assert len(loops[0]) == 4
    assert _loop_points(loops[0]) == {(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)}
    if dropped is not None:
        assert dropped == 0


def test_two_disjoint_triangles():
    _, edges_a = _chain([(0, 0), (1, 0), (0.5, 1)])
    _, edges_b = _chain([(5, 5), (6, 5), (5.5, 6)])
    loops, dropped = _walk(edges_a + edges_b)
    assert len(loops) == 2
    assert sorted(len(loop) for loop in loops) == [3, 3]
    if dropped is not None:
        assert dropped == 0


def test_open_chain_is_discarded():
    _, edges = _chain([(0, 0), (1, 0), (2, 0), (3, 0)], closed=False)
    loops, dropped = _walk(edges)
    assert loops == []
    if dropped is not None:
        assert dropped == 3


@pytest.mark.xfail(
    strict=False,
    reason="greedy neighbors[0] walk may lose the loop depending on start vertex; fixed by the edge-consumption rewrite",
)
def test_t_branch_keeps_loop_and_drops_dangling_edge():
    verts, edges = _chain([(0, 0), (1, 0), (1, 1), (0, 1)])
    dangler = FakeVert(2, 0)
    edges.append(FakeEdge(verts[1], dangler))
    loops, dropped = _walk(edges)
    assert len(loops) == 1
    assert len(loops[0]) == 4
    if dropped is not None:
        assert dropped == 1


@pytest.mark.xfail(
    strict=False,
    reason="greedy neighbors[0] walk drops the second lobe at a degree-4 crossing; fixed by the edge-consumption rewrite",
)
def test_figure_eight_recovers_both_lobes():
    """Two loops sharing one degree-4 vertex must both be recovered."""
    shared = FakeVert(0, 0)
    a1 = FakeVert(1, 1)
    a2 = FakeVert(2, 0)
    a3 = FakeVert(1, -1)
    b1 = FakeVert(-1, 1)
    b2 = FakeVert(-2, 0)
    b3 = FakeVert(-1, -1)
    edges = [
        FakeEdge(shared, a1),
        FakeEdge(a1, a2),
        FakeEdge(a2, a3),
        FakeEdge(a3, shared),
        FakeEdge(shared, b1),
        FakeEdge(b1, b2),
        FakeEdge(b2, b3),
        FakeEdge(b3, shared),
    ]
    loops, dropped = _walk(edges)
    assert len(loops) == 2
    assert sorted(len(loop) for loop in loops) == [4, 4]
    assert sum(len(loop) for loop in loops) == len(edges)
    if dropped is not None:
        assert dropped == 0
