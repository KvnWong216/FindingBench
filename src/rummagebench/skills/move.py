"""MOVE: signed relative base translation (protocol §12.1, §12.3).

Deterministic, controller-free base motion along the CURRENT heading.
Before execution the swept path is sampled at <= translation_sample_cm and
every sample's base footprint is checked against world collision geometry:
any collision rejects the whole motion (UNSAFE) with zero partial execution.
"""

from __future__ import annotations

import math

import numpy as np

from rummagebench.core.scenario import AnchorSpec

MAX_FLOOR_HEIGHT_DIFFERENCE_M = 0.30  # strict: exactly 30 cm is rejected


def floor_height_allowed(surface_z, base_z):
    difference = abs(float(surface_z) - float(base_z))
    return math.isfinite(difference) and difference < MAX_FLOOR_HEIGHT_DIFFERENCE_M


def is_navigation_floor(backend, name, info=None):
    """Low surfaces are terrain by actual height, independent of category."""
    if name in getattr(backend, 'robot_entity_names', lambda:set())():
        return False
    if getattr(backend, 'is_holding', lambda _:False)(name):
        return False
    box = backend.entity_aabb(name)
    if box is None:
        return False
    values = np.asarray(box, dtype=float)
    if values.shape != (2,3) or not np.isfinite(values).all():
        return False
    z = getattr(backend, 'robot_pose', lambda:([0,0,0],[0,0,0,1]))()[0][2]
    return bool(np.all(values[1]>=values[0]) and floor_height_allowed(values[1,2], z))


def yaw_from_quat(q: list[float]) -> float:
    """Z-Y-X intrinsic yaw for a quaternion [x, y, z, w]."""
    x, y, z, w = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_from_yaw(yaw: float) -> list[float]:
    return [0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)]


def footprint_corners(x: float, y: float, yaw: float, half_extent: float) -> list[tuple[float, float]]:
    c, s = math.cos(yaw), math.sin(yaw)
    return [
        (x + c * dx - s * dy, y + s * dx + c * dy)
        for dx, dy in (
            (-half_extent, -half_extent), (half_extent, -half_extent),
            (half_extent, half_extent), (-half_extent, half_extent),
        )
    ]


def _aabb_overlaps_footprint(aabb, corners, margin: float) -> bool:
    (lo, hi) = aabb
    # Geometry below the step-over threshold is traversable, regardless of label.
    if lo[2] > 0.5 or hi[2] < MAX_FLOOR_HEIGHT_DIFFERENCE_M:
        return False
    # Keep the actual rotated base rectangle. Its enclosing world AABB
    # contains empty corners and is only a broad-phase bound.
    obstacle = [
        (lo[0] - margin, lo[1] - margin),
        (hi[0] + margin, lo[1] - margin),
        (hi[0] + margin, hi[1] + margin),
        (lo[0] - margin, hi[1] + margin),
    ]
    axes = [(1.0, 0.0), (0.0, 1.0)]
    for i in (0, 1):
        a, b = corners[i], corners[i + 1]
        axes.append((-(b[1] - a[1]), b[0] - a[0]))
    for ax, ay in axes:
        base_projection = [x * ax + y * ay for x, y in corners]
        obstacle_projection = [x * ax + y * ay for x, y in obstacle]
        # Touching remains a collision, including the full safety margin.
        if (max(base_projection) < min(obstacle_projection) - 1e-12
                or max(obstacle_projection) < min(base_projection) - 1e-12):
            return False
    return True


def base_pose_collision_free(backend, x: float, y: float, yaw: float,
                             half_extent: float, margin: float,
                             skip_entities: set[str] = frozenset()) -> bool:
    """Conservative footprint check against world obstacle AABBs."""
    corners = footprint_corners(x, y, yaw, half_extent)
    controlled = getattr(backend, "robot_entity_names", lambda: set())()
    for name in backend.entity_names():
        if name in skip_entities or name in controlled:
            continue
        info = backend.describe_entity(name)
        if info is None:
            continue
        # Movable furniture still obstructs a teleporting base. Only bodies
        # carried by this robot are excluded from the external obstacle set.
        if getattr(backend, "is_holding", lambda _: False)(name):
            continue
        if any(_aabb_overlaps_footprint(aabb, corners, margin) for aabb in navigation_aabbs(backend,name)):
            return False
    return True


def navigation_aabbs(backend, name):
    """One shared obstacle representation for probing and public execution.

    Specialized backends may conservatively decompose architectural meshes
    and express z relative to their actual floor. Legacy backends retain
    their existing whole-entity bounds. Unavailable specialized geometry
    raises instead of disappearing from the obstacle set.
    """
    info = backend.describe_entity(name)
    if is_navigation_floor(backend, name, info):
        return []
    provider=getattr(backend,'navigation_obstacle_aabbs',None)
    if provider is not None:return provider(name)
    box=backend.entity_aabb(name)
    return [] if box is None else [box]


def move_target_pose(robot_pose, distance_m: float) -> tuple[list[float], list[float]]:
    """Final pose after signed translation along the current heading."""
    (x, y, z), q = robot_pose
    yaw = yaw_from_quat(q)
    nx = x + distance_m * math.cos(yaw)
    ny = y + distance_m * math.sin(yaw)
    return [nx, ny, z], list(q)


def turn_target_pose(robot_pose, angle_deg: float) -> tuple[list[float], list[float]]:
    (x, y, z), q = robot_pose
    yaw = yaw_from_quat(q) + math.radians(angle_deg)
    return [x, y, z], quat_from_yaw(yaw)


def execute_move(backend, robot_pose, distance_m: float, half_extent: float,
                 sample_m: float, margin: float) -> tuple[bool, list[float] | None]:
    """Swept-collision MOVE. Returns (safe, final_pose_or_None)."""
    (x0, y0, z0), q = robot_pose
    yaw = yaw_from_quat(q)
    steps = max(1, int(abs(distance_m) / max(sample_m, 1e-6)) + 1)
    for i in range(1, steps + 1):
        f = i / steps
        xi = x0 + distance_m * math.cos(yaw) * f
        yi = y0 + distance_m * math.sin(yaw) * f
        if not base_pose_collision_free(backend, xi, yi, yaw, half_extent, margin):
            return False, None
    target, tq = move_target_pose(robot_pose, distance_m)
    backend.teleport_robot(AnchorSpec(position=target, orientation=tq))
    return True, target


def execute_turn(backend, robot_pose, angle_deg: float, half_extent: float,
                 sample_deg: float, margin: float) -> tuple[bool, list[float] | None]:
    """Swept-collision TURN (position unchanged, yaw interpolated)."""
    (x, y, z), q = robot_pose
    yaw0 = yaw_from_quat(q)
    steps = max(1, int(abs(angle_deg) / max(sample_deg, 1e-6)) + 1)
    for i in range(1, steps + 1):
        yi = yaw0 + math.radians(angle_deg) * i / steps
        if not base_pose_collision_free(backend, x, y, yi, half_extent, margin):
            return False, None
    target, tq = turn_target_pose(robot_pose, angle_deg)
    backend.teleport_robot(AnchorSpec(position=target, orientation=tq))
    return True, target


def plan_bev_path(checker, start, goal, yaw, floor_boxes, resolution=0.1,
                  max_expansions=100000):
    """Deterministic 8-connected A*; fixed-heading footprint, 5cm edge checks.

    floor_boxes are same-level support envelopes. No snapping of the requested
    destination. Map and path are evaluator-private. A failed finite search is
    not a proof that no continuous-space route exists.
    """
    import heapq
    from functools import lru_cache

    start, goal = np.asarray(start, float), np.asarray(goal, float)
    if not floor_boxes:
        return None, "NAV_SUPPORT_UNAVAILABLE"
    lo = np.min([b[0][:2] for b in floor_boxes], axis=0)
    hi = np.max([b[1][:2] for b in floor_boxes], axis=0)
    def free(p):
        # Whole footprint must remain on the union of same-level floor bounds.
        corners = footprint_corners(*p, yaw, checker.half)
        probes = [p, *corners]
        if not all(any(b[0][0] <= x <= b[1][0] and b[0][1] <= y <= b[1][1]
                       for b in floor_boxes) for x, y in probes):
            return False
        return checker.free(float(p[0]), float(p[1]), yaw)
    def edge(a, b):
        n = max(1, math.ceil(float(np.linalg.norm(b-a)) / 0.05))
        return all(free(a+(b-a)*i/n) for i in range(n+1))
    if not free(start): return None, "NAV_START_BLOCKED"
    if not free(goal): return None, "NAV_GOAL_BLOCKED"
    # Grid anchored at the exact start; goal connected only by a checked edge.
    def point(k): return start + np.asarray(k)*resolution
    @lru_cache(None)
    def valid(k):
        p=point(k)
        return bool(np.all(p >= lo) and np.all(p <= hi) and free(p))
    root=(0,0); costs={root:0.}; parent={}; closed=set()
    queue=[(float(np.linalg.norm(goal-start)),0.,root)]
    directions=((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1))
    while queue:
        _,g,k=heapq.heappop(queue)
        if k in closed: continue
        closed.add(k)
        if len(closed)>max_expansions: return None,"NAV_SEARCH_LIMIT"
        p=point(k)
        if np.linalg.norm(goal-p)<=resolution*1.5 and edge(p,goal):
            keys=[k]
            while keys[-1] in parent: keys.append(parent[keys[-1]])
            path=[point(n).tolist() for n in reversed(keys)]
            if np.linalg.norm(goal-p)>1e-9: path.append(goal.tolist())
            return path,None
        for dx,dy in directions:
            nxt=(k[0]+dx,k[1]+dy)
            if nxt in closed or not valid(nxt): continue
            # Prevent diagonal corner cutting even where endpoints are free.
            if dx and dy and (not valid((k[0]+dx,k[1])) or not valid((k[0],k[1]+dy))): continue
            q=point(nxt)
            ng=g+resolution*math.hypot(dx,dy)
            if ng>=costs.get(nxt,float('inf')) or not edge(p,q): continue
            costs[nxt]=ng;parent[nxt]=k
            heapq.heappush(queue,(ng+float(np.linalg.norm(goal-q)),ng,nxt))
    return None,"NAV_NO_PATH"


def execute_move_to(backend, surface, half_extent, margin=0.03):
    """Oracle arrival after BEV A*: fixed heading; no trajectory controller."""
    from rummagebench.authoring.tasks.navigation import FootprintChecker
    pose=backend.robot_pose(); position,orientation=pose
    if not floor_height_allowed(surface[2], position[2]):
        return False,"NAV_DIFFERENT_LEVEL",{}
    floors = same_level_floors(backend, position[2])
    checker=FootprintChecker.from_backend(backend,half_extent,margin)
    path,reason=plan_bev_path(checker,position[:2],surface[:2],yaw_from_quat(orientation),floors)
    evidence={'planner':'bev_astar_fixed_heading_v1','resolution_m':0.1,
              'execution':'oracle_endpoint','support':'same_level_floor_aabb_envelopes',
              'path_xy':path,'heading_preserved':True}
    evidence['floor_height_limit_m'] = MAX_FLOOR_HEIGHT_DIFFERENCE_M
    if path is None:
        if reason in ("NAV_GOAL_BLOCKED", "NAV_START_BLOCKED"):
            xy = surface[:2] if reason == "NAV_GOAL_BLOCKED" else position[:2]
            corners = footprint_corners(*xy, yaw_from_quat(orientation), half_extent)
            controlled = getattr(backend, 'robot_entity_names', lambda:set())()
            evidence['blocking_geometry'] = [
                {'entity':name, 'base_relative_aabb':box}
                for name in backend.entity_names()
                if name not in controlled and not backend.is_holding(name)
                for box in navigation_aabbs(backend,name)
                if _aabb_overlaps_footprint(box,corners,margin)]
        return False,reason,evidence
    evidence['distance_m']=sum(math.dist(a,b) for a,b in zip(path,path[1:]))
    arrive = getattr(backend, 'navigate_oracle_endpoint', backend.teleport_robot)
    arrive(AnchorSpec(position=[float(surface[0]),float(surface[1]),position[2]],orientation=list(orientation)))
    actual=backend.robot_pose()
    if math.dist(actual[0][:2],surface[:2])>0.02:
        raise RuntimeError('MOVE_TO endpoint postcondition failed')
    return True,None,evidence


# ---------------------------------------------------------------- object-directed MOVE_TO
# 2026-10-26 (user decision, AsgardBench-style automatic positioning): a
# MOVE_TO click on a VISIBLE non-floor entity places the robot in front of
# it, facing it. The stand pose is chosen deterministically from the clicked
# surface point and public geometry only (no skill-feasibility lookahead);
# GRASP / PLACE / OPEN feasibility is still judged afterwards by the unchanged
# skill gates.

STAND_OFFSETS_M = tuple(round(0.45 + 0.05 * i, 2) for i in range(16))   # base centre to AABB edge
STAND_SPACING_M = 0.20
STAND_MAX_PLANS = 12
NOT_STAND_TARGETS = {"walls", "ceilings", "floors", "paver", "carpet", "rug"}


def stand_ring(aabb, offsets=STAND_OFFSETS_M, spacing=STAND_SPACING_M):
    """(x, y, yaw, offset): rectangles around the XY AABB, each pose facing
    the nearest AABB point (same construction as the route oracle's ring)."""
    (lx, ly), (hx, hy) = aabb[0][:2], aabb[1][:2]
    out = []
    for d in offsets:
        x0, y0, x1, y1 = lx - d, ly - d, hx + d, hy + d
        per = 2 * ((x1 - x0) + (y1 - y0))
        n = max(8, int(per / spacing))
        for i in range(n):
            s = per * i / n
            if s < x1 - x0:
                x, y = x0 + s, y0
            elif s < (x1 - x0) + (y1 - y0):
                x, y = x1, y0 + s - (x1 - x0)
            elif s < 2 * (x1 - x0) + (y1 - y0):
                x, y = x1 - (s - (x1 - x0) - (y1 - y0)), y1
            else:
                x, y = x0, y1 - (s - 2 * (x1 - x0) - (y1 - y0))
            nx, ny = min(max(x, lx), hx), min(max(y, ly), hy)
            out.append((x, y, math.atan2(ny - y, nx - x), d))
    return out


def _turn_free(checker, x, y, yaw0, yaw1, step_deg=10.0):
    delta = (yaw1 - yaw0 + math.pi) % (2 * math.pi) - math.pi
    n = max(1, math.ceil(abs(math.degrees(delta)) / step_deg))
    return all(checker.free(x, y, yaw0 + delta * i / n) for i in range(n + 1))


OBJECT_REACH_M = 0.9   # preferred base-to-object distance (median of feasible GRASP poses, 2026-10-27)


def same_level_floors(backend, z):
    floors = []
    for name in backend.entity_names():
        info = backend.describe_entity(name)
        if not is_navigation_floor(backend, name, info):
            continue
        box = backend.entity_aabb(name)
        if box is not None and floor_height_allowed(box[1][2], z):
            floors.append(box)
    return floors


def stand_candidates(backend, entity, half_extent, margin=0.03, yaw=None, offsets=STAND_OFFSETS_M):
    """(checker, ring) for an entity: ring poses facing it, footprint-free at
    their own heading, with nothing but the entity and its support (furniture
    whose footprint overlaps it, e.g. the table under a cup) between the
    footprint's front edge and the entity. None when the entity has no AABB."""
    from rummagebench.authoring.tasks.navigation import FootprintChecker
    aabb = backend.entity_aabb(entity)
    if aabb is None:
        return None, None
    from rummagebench.authoring.tasks.navigation import Z_HIGH, Z_LOW
    checker = FootprintChecker.from_backend(backend, half_extent, margin)
    lo2, hi2 = np.asarray(aabb[0][:2], float), np.asarray(aabb[1][:2], float)
    # obstacle boxes WITH their owner, same set and z band as the checker
    controlled = backend.robot_entity_names()
    others_lo, others_hi = [], []
    for name in backend.entity_names():
        info = backend.describe_entity(name)
        if name == entity or name in controlled or info is None or backend.is_holding(name):
            continue
        for lo, hi in navigation_aabbs(backend, name):
            if lo[2] > Z_HIGH or hi[2] < Z_LOW:
                continue
            overlaps = lo[0] < hi2[0] and hi[0] > lo2[0] and lo[1] < hi2[1] and hi[1] > lo2[1]
            if overlaps and info.category not in NOT_STAND_TARGETS | {"door"}:
                continue   # support under / around the entity (never a wall or door)
            others_lo.append(np.asarray(lo[:2], float) - margin)
            others_hi.append(np.asarray(hi[:2], float) + margin)
    others_lo = np.asarray(others_lo).reshape(-1, 2)
    others_hi = np.asarray(others_hi).reshape(-1, 2)

    def clear_reach(x, y):
        nx, ny = min(max(x, lo2[0]), hi2[0]), min(max(y, lo2[1]), hi2[1])
        dist = math.hypot(nx - x, ny - y)
        if dist <= half_extent:
            return True
        for t in np.linspace(half_extent / dist, 1.0, max(2, int((dist - half_extent) / 0.05) + 1)):
            p = (x + (nx - x) * t, y + (ny - y) * t)
            if np.any(np.all(others_lo <= p, 1) & np.all(others_hi >= p, 1)):
                return False
        return True

    # yaw=None: free at the pose's own (facing) heading; else at the given heading
    ring = [c for c in stand_ring(aabb, offsets) if checker.free(c[0], c[1], c[2] if yaw is None else yaw)
            and clear_reach(c[0], c[1])]
    return checker, ring


def select_stand_pose(backend, entity, surface, half_extent, margin=0.03):
    """Stand pose for an object-directed MOVE_TO, or (None, reason, evidence).

    Candidates: stand_candidates(). Furniture (fixed base): ordered by
    distance to the clicked surface point, so the clicked side wins. Movable
    objects: about OBJECT_REACH_M from the object, nearest the robot first.
    The first one reachable by the fixed-heading BEV planner plus an in-place
    turn (at the end, or at the start) is chosen."""
    position, orientation = backend.robot_pose()
    yaw0 = yaw_from_quat(orientation)
    checker, ring = stand_candidates(backend, entity, half_extent, margin)
    if checker is None:
        return None, "NAV_TARGET_GEOMETRY_UNAVAILABLE", {}
    floors = same_level_floors(backend, position[2])
    here = np.asarray(position[:2], float)
    info = backend.describe_entity(entity)
    if info is not None and not info.fixed_base:
        ring.sort(key=lambda c: (round(abs(c[3] - OBJECT_REACH_M), 1), math.dist(c[:2], here)))
    else:
        ring.sort(key=lambda c: (math.dist(c[:2], surface[:2]), c[3]))
    evidence = {'planner': 'stand_pose_v1', 'clicked_entity': entity, 'free_candidates': len(ring)}
    for x, y, yaw, offset in ring[:STAND_MAX_PLANS]:
        if math.dist(here, (x, y)) < 0.02:
            path, order = [here.tolist()], 'turn_only'
            ok = _turn_free(checker, x, y, yaw0, yaw)
        else:
            path, _ = plan_bev_path(checker, here, (x, y), yaw0, floors)
            order = 'move_then_turn'
            ok = path is not None and _turn_free(checker, x, y, yaw0, yaw)
            if not ok and _turn_free(checker, *here, yaw0, yaw):
                path, _ = plan_bev_path(checker, here, (x, y), yaw, floors)
                order, ok = 'turn_then_move', path is not None
        if ok:
            evidence.update(stand_pose=[round(x, 3), round(y, 3), round(yaw, 4)], offset_m=offset,
                            order=order, path_xy=path)
            return (x, y, yaw), None, evidence
    return None, "NAV_NO_STAND_POSE", evidence


def execute_move_to_entity(backend, entity, surface, half_extent, margin=0.03):
    """Object-directed MOVE_TO: oracle arrival at the stand pose, facing the entity."""
    position, _ = backend.robot_pose()
    pose, reason, evidence = select_stand_pose(backend, entity, surface, half_extent, margin)
    if pose is None:
        return False, reason, evidence
    path = evidence.get('path_xy') or []
    evidence['distance_m'] = sum(math.dist(a, b) for a, b in zip(path, path[1:]))
    evidence['execution'] = 'oracle_endpoint'
    anchor = AnchorSpec(position=[float(pose[0]), float(pose[1]), position[2]], orientation=quat_from_yaw(pose[2]))
    arrive = getattr(backend, 'navigate_oracle_pose', backend.teleport_robot)
    arrive(anchor)
    actual = backend.robot_pose()
    if math.dist(actual[0][:2], pose[:2]) > 0.02:
        raise RuntimeError('MOVE_TO stand pose postcondition failed')
    return True, None, evidence
