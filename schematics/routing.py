"""Grid-based Manhattan routing for schematic wires.

All routing happens on an 8-pixel grid. The caller supplies two endpoints
(typically component ports) plus the full component list; the router picks
between two strategies:

1. **Midpoint Z-route** — a single centred bend. Used when the straight
   three-segment path does not cross any other component's bbox. This is
   fast and produces visually balanced wires.
2. **A* search** — full orthogonal shortest-path, used only when the Z
   route is blocked. A bend penalty keeps detours simple (one or two
   bends where possible).

Public API
----------
    get_ports(component)                          -> dict[str, Point]
    port_outward(side)                            -> Direction
    select_port_pair(from_comp, to_comp)          -> ((Point, side), (Point, side))
    nearest_port(component, target)               -> (Point, side)
    build_blocked_cells(components, ...)          -> set[Point]
    is_blocked(point, blocked_cells)              -> bool
    simplify_path(path)                           -> list[Point]
    route_line(start, end, components, ...)       -> list[Point]   # main entry
    astar_route(start, end, components, ...)      -> list[Point]   # low-level

All coordinates are integer pixels; grid-aligned points are multiples of
``GRID``.
"""

from __future__ import annotations

import heapq
from typing import Iterable

# --- Tunables ---------------------------------------------------------------

GRID = 8           # pixels per grid cell
BEND_PENALTY = 5   # extra cost incurred when the route changes direction
OBSTACLE_PAD = 3   # pixels added around every component bbox before rasterising
SEARCH_MARGIN = 10 # grid cells of slack around the start/end bbox

Point = tuple[int, int]
Direction = tuple[int, int]

_DIRECTIONS: tuple[Direction, ...] = ((GRID, 0), (-GRID, 0), (0, GRID), (0, -GRID))


# --- Grid helpers -----------------------------------------------------------

def _snap(value: float, grid: int = GRID) -> int:
    """Round ``value`` to the nearest multiple of ``grid``."""
    return int(round(value / grid) * grid)


def snap_point(point: tuple[float, float], grid: int = GRID) -> Point:
    return (_snap(point[0], grid), _snap(point[1], grid))


def _manhattan(a: Point, b: Point) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# --- Ports ------------------------------------------------------------------

# Outward direction vectors for each named port side.
_PORT_OUTWARD: dict[str, Direction] = {
    "left":   (-GRID, 0),
    "right":  (GRID, 0),
    "top":    (0, -GRID),
    "bottom": (0, GRID),
}


def get_ports(component) -> dict[str, Point]:
    """Return the four orthogonal connection ports of a component.

    Ports are the edge midpoints of the bounding box:
        left   = (xmin, center_y)
        right  = (xmax, center_y)
        top    = (center_x, ymin)
        bottom = (center_x, ymax)

    All four points are snapped to the routing grid so that A* can
    terminate on them exactly.
    """
    cx = (component.xmin + component.xmax) / 2.0
    cy = (component.ymin + component.ymax) / 2.0
    return {
        "left":   snap_point((component.xmin, cy)),
        "right":  snap_point((component.xmax, cy)),
        "top":    snap_point((cx, component.ymin)),
        "bottom": snap_point((cx, component.ymax)),
    }


def port_outward(side: str) -> Direction:
    """Unit-grid step vector pointing away from the component at ``side``."""
    return _PORT_OUTWARD[side]


def select_port_pair(
    from_comp, to_comp
) -> tuple[tuple[Point, str], tuple[Point, str]]:
    """Pick the ((from_port, from_side), (to_port, to_side)) pair with
    minimum Manhattan distance."""
    best: tuple[tuple[Point, str], tuple[Point, str]] | None = None
    best_d = float("inf")
    for side_from, p_from in get_ports(from_comp).items():
        for side_to, p_to in get_ports(to_comp).items():
            d = _manhattan(p_from, p_to)
            if d < best_d:
                best_d = d
                best = ((p_from, side_from), (p_to, side_to))
    assert best is not None
    return best


def nearest_port(component, target: Point) -> tuple[Point, str]:
    """Return the (port, side) closest (Manhattan) to ``target``."""
    target_snapped = snap_point(target)
    return min(
        get_ports(component).items(),
        key=lambda kv: _manhattan(kv[1], target_snapped),
    )[::-1]  # items() yields (side, point); we want (point, side)


# --- Obstacle map -----------------------------------------------------------

def build_blocked_cells(
    components: Iterable,
    pad: int = OBSTACLE_PAD,
    exclude_ids: set[int] | None = None,
) -> set[Point]:
    """Rasterise every component bbox (inflated by ``pad`` px) onto the grid.

    Components whose id is in ``exclude_ids`` are skipped — typically the
    route's own endpoints so A* can freely enter/exit on their ports.

    Returns the set of grid points that a route must avoid.
    """
    exclude_ids = exclude_ids or set()
    blocked: set[Point] = set()
    for comp in components:
        if comp.id in exclude_ids:
            continue
        x0 = _snap(comp.xmin - pad)
        x1 = _snap(comp.xmax + pad)
        y0 = _snap(comp.ymin - pad)
        y1 = _snap(comp.ymax + pad)
        for gx in range(x0, x1 + GRID, GRID):
            for gy in range(y0, y1 + GRID, GRID):
                blocked.add((gx, gy))
    return blocked


def is_blocked(point: Point, blocked: set[Point]) -> bool:
    """Whether the grid point ``point`` lies inside any obstacle."""
    return point in blocked


# --- Path post-processing ---------------------------------------------------

def simplify_path(path: list[Point]) -> list[Point]:
    """Collapse runs of collinear points down to their endpoints.

    Keeps the first point, every bend, and the last point.
    """
    if len(path) < 3:
        return list(path)

    out = [path[0]]
    for i in range(1, len(path) - 1):
        ax, ay = out[-1]
        bx, by = path[i]
        cx, cy = path[i + 1]
        # On an orthogonal grid, b is collinear with a and c iff they all
        # share either an x or a y coordinate.
        if ax == bx == cx or ay == by == cy:
            continue
        out.append(path[i])
    out.append(path[-1])
    return out


# --- Straight-Z fast path ---------------------------------------------------

def _z_route(
    start: Point,
    end: Point,
    start_dir: Direction | None,
) -> list[Point]:
    """Three-segment orthogonal route with a single mid-span bend.

    Follows ``start_dir``'s axis first (if given), otherwise the dominant axis
    of the displacement. Bend is placed at the midpoint so the two long runs
    are balanced and the short segment stays well away from either endpoint.
    """
    sx, sy = start
    ex, ey = end
    if start_dir is not None:
        horiz_first = start_dir[0] != 0
    else:
        horiz_first = abs(ex - sx) >= abs(ey - sy)

    if horiz_first:
        mid = _snap((sx + ex) / 2)
        return [(sx, sy), (mid, sy), (mid, ey), (ex, ey)]
    mid = _snap((sy + ey) / 2)
    return [(sx, sy), (sx, mid), (ex, mid), (ex, ey)]


def _segment_hits_blocked(p: Point, q: Point, blocked: set[Point]) -> bool:
    """Whether the axis-aligned segment from ``p`` to ``q`` crosses any
    blocked grid cell. Both endpoints are excluded from the check."""
    px, py = p
    qx, qy = q
    if px == qx:
        lo, hi = sorted((py, qy))
        for y in range(lo + GRID, hi, GRID):
            if (px, y) in blocked:
                return True
    elif py == qy:
        lo, hi = sorted((px, qx))
        for x in range(lo + GRID, hi, GRID):
            if (x, py) in blocked:
                return True
    else:
        # Diagonal — shouldn't happen on an orthogonal grid; treat as blocked.
        return True
    return False


def _path_is_clear(path: list[Point], blocked: set[Point]) -> bool:
    for a, b in zip(path, path[1:]):
        if _segment_hits_blocked(a, b, blocked):
            return False
    return True


# --- A* ---------------------------------------------------------------------

def astar_route(
    start: Point,
    end: Point,
    components: Iterable,
    exclude_ids: set[int] | None = None,
    start_dir: Direction | None = None,
    end_dir: Direction | None = None,
    margin_cells: int = SEARCH_MARGIN,
) -> list[Point]:
    """Find a minimum-cost orthogonal route from ``start`` to ``end``.

    Parameters
    ----------
    start, end
        Route endpoints; will be snapped to the grid internally.
    components
        Iterable of components whose bounding boxes block the grid.
    exclude_ids
        Component ids to skip when building the obstacle map — typically the
        source and target components, so A* can enter/exit on their ports
        without being walled in by the component's own inflated bbox.
    start_dir
        Preferred outward direction from ``start``. Seeding the search with
        this "incoming" direction means a perpendicular exit is free while a
        parallel exit pays the bend penalty, biasing the route to leave the
        port cleanly.
    end_dir
        Preferred inward direction at ``end`` (a vector pointing from the
        free space into the port). Steps ending at ``end`` that do not match
        ``end_dir`` pay the bend penalty for the same reason.

    Cost model: 1 per grid step, plus ``BEND_PENALTY`` when the step direction
    differs from the previous step. Path state is ``(point, incoming_direction)``
    so the penalty only applies on actual direction changes.

    Falls back to a naive L-route when A* cannot reach the end.
    """
    start = snap_point(start)
    end = snap_point(end)

    if start == end:
        return [start]

    blocked = build_blocked_cells(components, exclude_ids=exclude_ids)
    # Always ensure the endpoints themselves are passable even if their
    # components weren't excluded (e.g. dangling wires).
    blocked.discard(start)
    blocked.discard(end)

    margin = margin_cells * GRID
    xmin = min(start[0], end[0]) - margin
    xmax = max(start[0], end[0]) + margin
    ymin = min(start[1], end[1]) - margin
    ymax = max(start[1], end[1]) + margin

    def in_bounds(p: Point) -> bool:
        return xmin <= p[0] <= xmax and ymin <= p[1] <= ymax

    # State = (point, incoming_direction). Direction is part of the state so
    # that turning costs are charged correctly: two distinct paths reaching
    # the same cell from different directions are not collapsed.
    State = tuple[Point, Direction | None]
    start_state: State = (start, start_dir)

    g_score: dict[State, int] = {start_state: 0}
    came_from: dict[State, State] = {}

    counter = 0
    heap: list[tuple[int, int, int, Point, Direction | None]] = [
        (_manhattan(start, end), counter, 0, start, start_dir)
    ]

    while heap:
        _, _, g, cur, last_dir = heapq.heappop(heap)
        state: State = (cur, last_dir)
        if g > g_score.get(state, g):
            continue
        if cur == end:
            return simplify_path(_reconstruct(came_from, state))

        for dxy in _DIRECTIONS:
            nxt = (cur[0] + dxy[0], cur[1] + dxy[1])
            if not in_bounds(nxt):
                continue
            if nxt != end and nxt in blocked:
                continue

            step = 1
            if last_dir is not None and dxy != last_dir:
                step += BEND_PENALTY
            # Arriving at the goal from a non-preferred direction also bends.
            if nxt == end and end_dir is not None and dxy != end_dir:
                step += BEND_PENALTY

            ng = g + step
            nstate: State = (nxt, dxy)
            if ng < g_score.get(nstate, 1 << 30):
                g_score[nstate] = ng
                came_from[nstate] = state
                counter += 1
                heapq.heappush(
                    heap,
                    (ng + _manhattan(nxt, end), counter, ng, nxt, dxy),
                )

    # No path found — fall back to a straight L so the wire still renders.
    return simplify_path([start, (end[0], start[1]), end])


def route_line(
    start: Point,
    end: Point,
    components: Iterable,
    exclude_ids: set[int] | None = None,
    start_dir: Direction | None = None,
    end_dir: Direction | None = None,
) -> list[Point]:
    """Compute a clean orthogonal route from ``start`` to ``end``.

    Strategy:
      1. Try a midpoint Z-route. If no other component's bbox intersects it,
         use it — this produces visually balanced wires with a single bend
         placed halfway across the run.
      2. Otherwise fall back to A* with obstacle avoidance.

    Parameters mirror :func:`astar_route`. ``components`` is the full
    schematic component list; ``exclude_ids`` are ignored for clearance
    checks (typically the route's own endpoints).
    """
    start_s = snap_point(start)
    end_s = snap_point(end)

    if start_s == end_s:
        return [start_s]

    blocked = build_blocked_cells(components, exclude_ids=exclude_ids)
    blocked.discard(start_s)
    blocked.discard(end_s)

    z = _z_route(start_s, end_s, start_dir)
    if _path_is_clear(z, blocked):
        return simplify_path(z)

    return astar_route(
        start_s,
        end_s,
        components,
        exclude_ids=exclude_ids,
        start_dir=start_dir,
        end_dir=end_dir,
    )


def _reconstruct(
    came_from: dict[tuple[Point, Direction | None], tuple[Point, Direction | None]],
    state: tuple[Point, Direction | None],
) -> list[Point]:
    path = [state[0]]
    while state in came_from:
        state = came_from[state]
        path.append(state[0])
    path.reverse()
    return path
