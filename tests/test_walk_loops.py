"""Graph-walk tests for rtstruct_export._walk_loops using fake bmesh objects."""
from __future__ import annotations

import types

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


def test_t_branch_keeps_loop_and_drops_dangling_edge():
    verts, edges = _chain([(0, 0), (1, 0), (1, 1), (0, 1)])
    dangler = FakeVert(2, 0)
    edges.append(FakeEdge(verts[1], dangler))
    loops, dropped = _walk(edges)
    assert len(loops) == 1
    assert len(loops[0]) == 4
    if dropped is not None:
        assert dropped == 1


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


def _polygon_area(loop):
    points = [(x, y) for x, y, _z in loop]
    return 0.5 * abs(sum(
        x0 * y1 - x1 * y0
        for (x0, y0), (x1, y1) in zip(points, points[1:] + points[:1])
    ))


def test_two_loops_joined_by_a_bridge_are_both_recovered():
    """A branch attached at both ends is not a dangling chain, so peeling
    leaves it; the walk used to wander across it and discard both squares."""
    import random

    a, edges_a = _chain([(0, 0), (1, 0), (1, 1), (0, 1)])
    b, edges_b = _chain([(3, 0), (4, 0), (4, 1), (3, 1)])
    edges = edges_a + edges_b + [FakeEdge(a[1], b[0])]
    for seed in range(200):
        order = random.Random(seed).sample(edges, len(edges))
        loops, dropped = _walk(order)
        assert sorted(_polygon_area(loop) for loop in loops) == [1.0, 1.0], seed
        assert dropped == 1


def test_duplicate_edge_pair_is_not_a_contour():
    a, b = FakeVert(0, 0), FakeVert(1, 0)
    loops, dropped = _walk([FakeEdge(a, b), FakeEdge(a, b)])
    assert loops == [] and dropped == 2


class _Face:
    def __init__(self, *verts):
        self.verts = verts


def test_edges_inside_a_cap_on_the_slice_plane_are_not_outline():
    """A triangulated square cap resting on the slice plane: the diagonal has
    both faces in the plane and must not reach the loop walk, where it could
    turn the square cross-section into one of its triangles."""
    corners = [FakeVert(0, 0), FakeVert(2, 0), FakeVert(2, 2), FakeVert(0, 2)]
    below = [FakeVert(v.co.x, v.co.y, -1.0) for v in corners]
    cap_a = _Face(corners[0], corners[1], corners[2])
    cap_b = _Face(corners[0], corners[2], corners[3])
    outline = []
    for index in range(4):
        nxt = (index + 1) % 4
        edge = FakeEdge(corners[index], corners[nxt])
        wall = _Face(corners[index], corners[nxt], below[nxt], below[index])
        edge.link_faces = [wall, cap_a if index < 2 else cap_b]
        outline.append(edge)
    diagonal = FakeEdge(corners[0], corners[2])
    diagonal.link_faces = [cap_a, cap_b]
    loose = FakeEdge(corners[1], corners[3])
    loose.link_faces = []

    kept = rtstruct_export._outline_edges([diagonal, loose] + outline, 0.0)
    assert kept == outline
    loops, dropped = _walk(kept)
    assert [_polygon_area(loop) for loop in loops] == [4.0] and dropped == 0
