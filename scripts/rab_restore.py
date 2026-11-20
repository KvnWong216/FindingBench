"""Continuous restoration score from native settled world state."""
from __future__ import annotations

import math


def rotation_difference(a, b):
    norm_a = math.sqrt(sum(x*x for x in a))
    norm_b = math.sqrt(sum(x*x for x in b))
    if not norm_a or not norm_b:
        raise ValueError('Invalid orientation quaternion')
    cosine = abs(sum(x*y for x,y in zip(a,b,strict=True)))/(norm_a*norm_b)
    return 2*math.acos(min(1.,cosine))


def score_restoration(reference, current, *, position_scale_m=.05, angle_scale_rad=.35,
                      slide_scale_m=.04):
    """Score returned objects and closed drawers separately and jointly.

    Current body positions use native xyzw poses and the observed support path.
    Drawer contents use the position relative to their drawer. Delivery
    targets are excluded because the task requires moving them.
    """
    by_body = {}
    by_joint = {}
    for name, initial in reference['bodies'].items():
        if initial['target']:
            continue
        final = current.get('bodies',{}).get(name)
        if final is None or final.get('support')!=initial['support']:
            by_body[name] = 0.
            continue
        position_key = 'drawer_relative_xyz_m' if 'drawer_relative_xyz_m' in initial else None
        if position_key and position_key not in final:
            by_body[name] = 0.
            continue
        a = initial[position_key] if position_key else initial['pose'][:3]
        b = final[position_key] if position_key else final['pose'][:3]
        distance = math.dist(a,b)
        angle = rotation_difference(initial['pose'][3:],final['pose'][3:])
        by_body[name] = math.exp(-.5*((distance/position_scale_m)**2+(angle/angle_scale_rad)**2))
    for name, initial in reference['articulations_m'].items():
        final = current.get('articulations_m',{}).get(name)
        by_joint[name] = 0. if final is None else math.exp(-.5*((final-initial)/slide_scale_m)**2)
    values = list(by_body.values())+list(by_joint.values())
    return dict(score=sum(values)/len(values) if values else None,bodies=by_body,
                articulations=by_joint,scales=dict(position_m=position_scale_m,
                    angle_rad=angle_scale_rad,slide_m=slide_scale_m))
