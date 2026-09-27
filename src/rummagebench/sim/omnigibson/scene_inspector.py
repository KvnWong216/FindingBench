"""Scene inspection: enumerate rooms, objects, openable containers, receptacles.

Used by `python -m rummagebench.authoring.inspect` and scripts/inspect_scene.py
to author scenarios against real BEHAVIOR assets.
"""

from __future__ import annotations

from typing import Any


def available_scenes() -> list[str]:
    from omnigibson.utils.asset_utils import get_available_behavior_1k_scenes

    return list(get_available_behavior_1k_scenes())


def inspect_scene(scene_model: str, include_robot: bool = True) -> dict[str, Any]:
    """Load one interactive scene and return a structured inventory."""
    from omnigibson.macros import gm

    apply_headless_defaults()

    import omnigibson as og
    from omnigibson.envs import Environment
    from omnigibson.object_states import Open

    config = {
        "env": {"action_frequency": 30, "physics_frequency": 120},
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": scene_model,
            "include_robots": False,
        },
        "robots": (
            [
                {
                    "model": "r1pro",
                    "obs_modalities": [],
                    "action_type": "continuous",
                    "grasping_mode": "sticky",
                }
            ]
            if include_robot
            else []
        ),
        "objects": [],
        "task": {"type": "DummyTask"},
    }
    env = Environment(configs=config)
    scene = env.scene

    result: dict[str, Any] = {
        "scene_model": scene_model,
        "data_path": str(gm.DATA_PATH),
        "rooms": {},
        "objects": [],
        "openable_containers": [],
        "receptacles": [],
    }

    seg_map = scene.seg_map
    room_sem_to_ins = dict(seg_map.room_sem_name_to_ins_name)
    result["rooms"] = room_sem_to_ins

    # map instance -> semantic type
    ins_to_sem = {
        ins: sem
        for sem, instances in room_sem_to_ins.items()
        for ins in (instances if isinstance(instances, list) else [instances])
    }

    for obj in scene.objects:
        rooms = list(getattr(obj, "in_rooms", None) or [])
        pos, quat = obj.get_position_orientation()
        to_np = lambda t: t.detach().cpu().numpy().tolist() if hasattr(t, "detach") else list(t)
        entry = {
            "name": obj.name,
            "category": obj.category,
            "model": getattr(obj, "model", None),
            "fixed_base": bool(getattr(obj, "fixed_base", False)),
            "rooms": rooms,
            "room_types": sorted({ins_to_sem.get(r, "?") for r in rooms}),
            "position": [round(v, 4) for v in to_np(pos)],
            "orientation": [round(v, 4) for v in to_np(quat)],
            "openable": "openable" in (obj.abilities or {}),
        }
        result["objects"].append(entry)
        if entry["openable"]:
            has_open = obj.states.get(Open) is not None
            if has_open:
                result["openable_containers"].append(entry["name"])
        abilities = obj.abilities or {}
        if "fillable" in abilities or "inside" in abilities:
            result["receptacles"].append(entry["name"])

    result["robot_pose"] = None
    if scene.robots:
        robot = scene.robots[0]
        pos, quat = robot.get_position_orientation()
        to_np = lambda t: t.detach().cpu().numpy().tolist() if hasattr(t, "detach") else list(t)
        result["robot_pose"] = {"position": to_np(pos), "orientation": to_np(quat)}

    og.shutdown()
    return result


def apply_headless_defaults() -> None:
    import os

    os.environ.setdefault("OMNIGIBSON_HEADLESS", "True")
    os.environ.setdefault("OMNIGIBSON_NO_OMNI_LOGS", "True")


def format_report(result: dict[str, Any]) -> str:
    lines = [f"scene: {result['scene_model']}", f"data_path: {result['data_path']}"]
    lines.append("\n== rooms (type -> instances) ==")
    for sem, instances in result["rooms"].items():
        lines.append(f"  {sem}: {instances}")
    lines.append(f"\n== objects ({len(result['objects'])}) ==")
    by_room: dict[str, list[dict]] = {}
    for obj in result["objects"]:
        key = ",".join(obj["room_types"]) or "?"
        by_room.setdefault(key, []).append(obj)
    for room_type, objs in sorted(by_room.items()):
        lines.append(f"  [{room_type}] ({len(objs)} objects)")
        for obj in objs:
            flags = []
            if obj["openable"]:
                flags.append("openable")
            if obj["fixed_base"]:
                flags.append("fixed")
            flag_s = f" ({','.join(flags)})" if flags else ""
            lines.append(
                f"    - {obj['name']} cat={obj['category']} model={obj['model']} "
                f"pos={obj['position']}{flag_s}"
            )
    lines.append("\n== openable containers ==")
    lines.append("  " + ", ".join(result["openable_containers"]) or "  (none)")
    lines.append("\n== candidate receptacles ==")
    lines.append("  " + ", ".join(result["receptacles"]) or "  (none)")
    if result["robot_pose"]:
        lines.append(f"\nrobot pose: {result['robot_pose']}")
    return "\n".join(lines)
