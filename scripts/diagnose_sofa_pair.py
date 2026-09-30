"""Evaluator-only static gate reproduction; no state changes.

Historical mesh/distance experiments remain in server evidence; this is the
final bounded diagnostic used at handoff, not a navigation policy.

audit() reproduces the MOVE50 gate along the current heading AND the TURN
sweep (both directions, 5 deg samples) so a rejected TURN yields the exact
blocking pair without consuming planning steps.
"""

def _first_move_blocks(b, yaw, xy0):
    import math
    from rummagebench.skills.move import footprint_corners, _aabb_overlaps_footprint
    hits = []
    for i in range(1, 12):
        distance = .5 * i / 11
        xy = [xy0[0] + distance * math.cos(yaw), xy0[1] + distance * math.sin(yaw)]
        corners = footprint_corners(xy[0], xy[1], yaw, .4)
        for name in b.entity_names():
            if name in b.robot_entity_names() or b.is_holding(name):
                continue
            if b.describe_entity(name) is None:
                continue
            box = b.entity_aabb(name)
            if box is not None and _aabb_overlaps_footprint(box, corners, .03):
                hits.append({"distance_m": distance, "entity": name, "aabb": box,
                             "base_corners": corners})
        if hits:
            break
    return hits


def _first_turn_blocks(b, angle_deg):
    import math
    from rummagebench.skills.move import yaw_from_quat, quat_from_yaw, footprint_corners, _aabb_overlaps_footprint
    p, q = b.robot_pose()
    yaw0 = yaw_from_quat(q)
    steps = max(1, int(abs(angle_deg) / 5.0))
    hits = []
    for i in range(1, steps + 1):
        yaw = yaw0 + math.radians(angle_deg) * i / steps
        corners = footprint_corners(p[0], p[1], yaw, .4)
        for name in b.entity_names():
            if name in b.robot_entity_names() or b.is_holding(name):
                continue
            if b.describe_entity(name) is None:
                continue
            box = b.entity_aabb(name)
            if box is not None and _aabb_overlaps_footprint(box, corners, .03):
                hits.append({"angle_deg": abs(angle_deg) * i / steps,
                             "entity": name, "aabb": box, "base_corners": corners,
                             "yaw_sample_deg": math.degrees(yaw)})
        if hits:
            break
    return hits


def _fan(b, span_deg=90.0, step_deg=15.0, max_distance=1.6, sample=0.1):
    import math
    from rummagebench.skills.move import yaw_from_quat, footprint_corners, _aabb_overlaps_footprint
    p, q = b.robot_pose()
    yaw0 = yaw_from_quat(q)
    out = []
    for offset in range(int(-span_deg), int(span_deg) + 1, int(step_deg)):
        yaw = yaw0 + math.radians(offset)
        clear = 0.0
        blocker = None
        n = int(max_distance / sample)
        for i in range(1, n + 1):
            d = sample * i
            xy = [p[0] + d * math.cos(yaw), p[1] + d * math.sin(yaw)]
            corners = footprint_corners(xy[0], xy[1], yaw, .4)
            hit = None
            for name in b.entity_names():
                if name in b.robot_entity_names() or b.is_holding(name):
                    continue
                if b.describe_entity(name) is None:
                    continue
                box = b.entity_aabb(name)
                if box is not None and _aabb_overlaps_footprint(box, corners, .03):
                    hit = name
                    break
            if hit:
                blocker = {"entity": hit, "distance_m": d, "aabb": [list(box[0]), list(box[1])]}
                break
            clear = d
        out.append({"rel_deg": offset, "clear_m": clear, "blocker": blocker})
    return out

def _target_probe(b):
    pose = aabb = None
    try:
        pose = b.entity_pose6d("target_knife")
        aabb = b.entity_aabb("target_knife")
    except Exception as e:
        return {"error": repr(e)}
    return {"pose": None if pose is None else [list(pose.position), list(pose.orientation)],
            "aabb": None if aabb is None else [list(aabb[0]), list(aabb[1])]}


def _knife_pixel_probe(b):
    try:
        frame = b.capture_visual_frame()
        seg = frame.instance_segmentation
        labels = frame.meta.get("instance_labels", {})
        knife_ids = [k for k, v in labels.items() if v == "target_knife"]
        if not knife_ids:
            return {"knife_ids": [], "note": "no knife instance in labels", "meta_bridge": frame.meta.get("bridge"), "unsupported": frame.meta.get("unsupported_reason")}
        out = {}
        for k in knife_ids:
            key = int(k) if seg.dtype.kind not in "USO" else k
            ys, xs = (seg == key).nonzero()
            out[str(k)] = {"pixels": int(len(ys)), "rows": [int(ys.min()), int(ys.max())] if len(ys) else None, "cols": [int(xs.min()), int(xs.max())] if len(ys) else None}
        return {"knife_ids": out, "shape": [int(seg.shape[0]), int(seg.shape[1])]}
    except Exception as e:
        return {"error": repr(e)}


def audit(b):
    import math
    from rummagebench.skills.move import yaw_from_quat
    p, q = b.robot_pose()
    yaw = yaw_from_quat(q)
    return {"pose": [p, q], "query": "MOVE50",
            "first_blocks": _first_move_blocks(b, yaw, p),
            "turn_left_180": _first_turn_blocks(b, 180.0),
            "turn_right_180": _first_turn_blocks(b, -180.0),
            "fan": _fan(b), "fan_full": _fan(b, span_deg=180.0, step_deg=15.0, max_distance=1.2),
            "target_probe": _target_probe(b),
            "knife_pixels": _knife_pixel_probe(b),
            "pose_after": b.robot_pose()}
