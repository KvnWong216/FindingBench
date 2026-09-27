"""Scenario/scene inspection CLI:

    python -m rummagebench.authoring.inspect --scene Beechwood_0_int
    python -m rummagebench.authoring.inspect --list-scenes
"""

from __future__ import annotations

import argparse
import json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="inspect BEHAVIOR scenes for authoring")
    parser.add_argument("--scene", type=str, default=None, help="scene model to load")
    parser.add_argument(
        "--list-scenes", action="store_true", help="list installed BEHAVIOR scenes"
    )
    parser.add_argument("--no-robot", action="store_true")
    parser.add_argument("--json", action="store_true", help="emit raw JSON")
    args = parser.parse_args(argv)

    from rummagebench.sim.omnigibson.scene_inspector import (
        apply_headless_defaults,
    )

    apply_headless_defaults()

    if args.list_scenes:
        from rummagebench.sim.omnigibson.scene_inspector import available_scenes

        scenes = available_scenes()
        print(json.dumps(scenes, indent=2) if args.json else "\n".join(scenes))
        return 0

    if not args.scene:
        parser.error("either --scene or --list-scenes is required")

    from rummagebench.sim.omnigibson.scene_inspector import format_report, inspect_scene

    result = inspect_scene(args.scene, include_robot=not args.no_robot)
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(format_report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
