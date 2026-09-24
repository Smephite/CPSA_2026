import numpy as np
import pytest

from utils.geometry import convex_hull, distance_to_hull, point_in_polygon, point_segment_distance

SQUARE = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)


def test_hull_drops_interior_points():
    hull = convex_hull(np.vstack([SQUARE, [[5, 5], [2, 8]]]))
    assert sorted(map(tuple, hull)) == sorted(map(tuple, SQUARE))


def test_hull_of_two_points_is_the_segment():
    assert len(convex_hull([[0, 0], [3, 4], [0, 0]])) == 2


@pytest.mark.parametrize("p, inside", [((5, 5), True), ((11, 5), False), ((-1, -1), False)])
def test_point_in_polygon(p, inside):
    assert point_in_polygon(p, SQUARE) == inside


@pytest.mark.parametrize("p, d", [((5, 5), 0.0), ((13, 5), 3.0), ((13, 14), 5.0)])
def test_distance_to_hull(p, d):
    assert distance_to_hull(p, SQUARE) == pytest.approx(d)


def test_distance_to_degenerate_hulls():
    assert distance_to_hull((3, 4), [[0, 0]]) == pytest.approx(5.0)
    assert distance_to_hull((5, 3), [[0, 0], [10, 0]]) == pytest.approx(3.0)
    assert distance_to_hull((5, 3), []) == float("inf")


def test_point_segment_distance_clamps_to_ends():
    assert point_segment_distance((-3, 4), (0, 0), (10, 0)) == pytest.approx(5.0)
    assert point_segment_distance((2, 2), (1, 1), (1, 1)) == pytest.approx(np.sqrt(2))


def test_hull_distances_matches_scalar_version():
    from utils.geometry import hull_distances
    rng = np.random.default_rng(0)
    hull = convex_hull(rng.uniform(0, 100, (12, 2)))
    pts = rng.uniform(-50, 150, (200, 2))
    vec = hull_distances(pts, hull)
    ref = [0.0 if point_in_polygon(p, hull) else min(point_segment_distance(p, hull[i], hull[(i + 1) % len(hull)])
                                                     for i in range(len(hull))) for p in pts]
    assert np.allclose(vec, ref)
    assert np.allclose(hull_distances(pts, hull[::-1]), ref)          # either winding order
