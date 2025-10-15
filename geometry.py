from typing import Iterable, List, Optional, Sequence, Tuple

Point = Tuple[float, float]
Segment = Tuple[Point, Point]


def _cross(o: Point, a: Point, b: Point) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def convex_hull(points: Iterable[Point]) -> List[Point]:
    pts = sorted(set(points))
    if len(pts) <= 1:
        return pts
    lower: List[Point] = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: List[Point] = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _on_segment(a: Point, b: Point, c: Point) -> bool:
    return (
        min(a[0], b[0]) - 1e-9 <= c[0] <= max(a[0], b[0]) + 1e-9
        and min(a[1], b[1]) - 1e-9 <= c[1] <= max(a[1], b[1]) + 1e-9
    )


def _orientation(a: Point, b: Point, c: Point) -> float:
    return _cross(a, b, c)


def segments_intersect(s1: Segment, s2: Segment) -> bool:
    (p1, p2), (q1, q2) = s1, s2
    o1 = _orientation(p1, p2, q1)
    o2 = _orientation(p1, p2, q2)
    o3 = _orientation(q1, q2, p1)
    o4 = _orientation(q1, q2, p2)
    if (o1 * o2 < 0) and (o3 * o4 < 0):
        return True
    if abs(o1) < 1e-9 and _on_segment(p1, p2, q1):
        return True
    if abs(o2) < 1e-9 and _on_segment(p1, p2, q2):
        return True
    if abs(o3) < 1e-9 and _on_segment(q1, q2, p1):
        return True
    if abs(o4) < 1e-9 and _on_segment(q1, q2, p2):
        return True
    return False


def intersection_point(s1: Segment, s2: Segment) -> Optional[Point]:
    if not segments_intersect(s1, s2):
        return None
    (p1, p2), (q1, q2) = s1, s2
    xdiff = (p1[0] - p2[0], q1[0] - q2[0])
    ydiff = (p1[1] - p2[1], q1[1] - q2[1])

    def _det(a: Point, b: Point) -> float:
        return a[0] * b[1] - a[1] * b[0]

    div = _det(xdiff, ydiff)
    if abs(div) < 1e-9:
        candidates = [p for p in (p1, p2, q1, q2) if _on_segment(p1, p2, p) and _on_segment(q1, q2, p)]
        return candidates[0] if candidates else None
    d = (_det(p1, p2), _det(q1, q2))
    x = _det(d, xdiff) / div
    y = _det(d, ydiff) / div
    pt = (x, y)
    if _on_segment(p1, p2, pt) and _on_segment(q1, q2, pt):
        return pt
    return None


def hull_edges(hull: Sequence[Point]) -> List[Segment]:
    if len(hull) < 2:
        return []
    if len(hull) == 2:
        return [(hull[0], hull[1])]
    return [(hull[i], hull[(i + 1) % len(hull)]) for i in range(len(hull))]
