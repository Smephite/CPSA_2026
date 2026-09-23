"""Small 2-D geometry helpers (numpy only)."""
import numpy as np


def convex_hull(points):
    """Andrew's monotone chain. points (N, 2) -> hull (M, 2), counter-clockwise, M <= N."""
    pts = np.unique(np.asarray(points, dtype=np.float64), axis=0)
    if len(pts) <= 2:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def point_segment_distance(p, a, b):
    p, a, b = (np.asarray(v, dtype=np.float64) for v in (p, a, b))
    ab = b - a
    denom = float(ab @ ab)
    t = 0.0 if denom == 0 else float(np.clip((p - a) @ ab / denom, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def point_in_polygon(p, poly):
    """Ray casting; poly (M, 2) with M >= 3."""
    x, y = float(p[0]), float(p[1])
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xin = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < xin:
                inside = not inside
    return inside


def distance_to_hull(p, hull):
    """0 inside the hull, else distance to its boundary. Works for 1- and 2-point 'hulls' too."""
    hull = np.asarray(hull, dtype=np.float64)
    if len(hull) == 0:
        return float("inf")
    if len(hull) == 1:
        return float(np.linalg.norm(np.asarray(p) - hull[0]))
    if len(hull) >= 3 and point_in_polygon(p, hull):
        return 0.0
    n = len(hull)
    segs = range(n) if n >= 3 else range(1)
    return min(point_segment_distance(p, hull[i], hull[(i + 1) % n]) for i in segs)


def dilate_polygon(hull, margin, n_arc=6):
    """Approximate outline of hull dilated by `margin` (for drawing only)."""
    hull = np.asarray(hull, dtype=np.float64)
    if len(hull) == 0:
        return hull
    angles = np.linspace(0, 2 * np.pi, n_arc * 4, endpoint=False)
    ring = np.stack([np.cos(angles), np.sin(angles)], axis=1) * margin
    return convex_hull((hull[:, None, :] + ring[None, :, :]).reshape(-1, 2))


def expand_box(box, mx, my, width, height):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return np.array([max(0.0, x1 - w * mx), max(0.0, y1 - h * my),
                     min(float(width), x2 + w * mx), min(float(height), y2 + h * my)])


def box_center(box):
    return np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0])


def box_foot(box):
    return np.array([(box[0] + box[2]) / 2.0, box[3]])
