"""Evaluator-only static MOVE50 gate reproduction; no state changes.

Historical mesh/distance experiments remain in server evidence; this is the
final bounded diagnostic used at handoff, not a navigation policy.
"""

def audit(b):
    import math
    from rummagebench.skills.move import yaw_from_quat, footprint_corners, _aabb_overlaps_footprint
    p,q=b.robot_pose();yaw=yaw_from_quat(q);hits=[]
    for i in range(1,12):
        distance=.5*i/11
        xy=[p[0]+distance*math.cos(yaw),p[1]+distance*math.sin(yaw)]
        corners=footprint_corners(*xy,yaw,.4)
        for name in b.entity_names():
            if name in b.robot_entity_names() or b.is_holding(name):continue
            if b.describe_entity(name) is None:continue
            box=b.entity_aabb(name)
            if box is not None and _aabb_overlaps_footprint(box,corners,.03):
                hits.append({"distance_m":distance,"entity":name,"aabb":box,"base_corners":corners})
        if hits:break
    return {"pose":[p,q],"query":"MOVE50","first_blocks":hits,"pose_after":b.robot_pose()}
