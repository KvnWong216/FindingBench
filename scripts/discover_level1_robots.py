"""§5/rev§2: discover candidate robots from the installed OG registry.

Introspects the installed OmniGibson robots module for REAL robot classes
and emits a candidate skeleton for configs/level1/robot_pool_v1.yaml.
Mobile-manipulator candidates (arm + mobile base + camera) are preferred;
GPU verification (URDF export, camera identity, MOVE/TURN/OBSERVE validity)
is a separate stage and never assumed here.

    PYTHONPATH=src python scripts/discover_level1_robots.py [--out configs/level1/robot_pool_v1.yaml]

Importing omnigibson requires the behavior environment; this script refuses
to invent robots that the installed package does not expose.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]


def discover() -> dict[str, dict]:
    try:
        import omnigibson.robots as og_robots
        from omnigibson.robots.robot import Robot
    except Exception as e:  # pragma: no cover - environment specific
        raise SystemExit(f"omnigibson import failed ({e}); run inside the "
                         "behavior environment")
    candidates: dict[str, dict] = {}
    for name in sorted(dir(og_robots)):
        cls = getattr(og_robots, name)
        if not isinstance(cls, type) or not issubclass(cls, Robot) or cls is Robot:
            continue
        lowered = name.lower()
        try:
            has_arm = bool(getattr(cls, "arm_names", None) is not None)
        except Exception:
            has_arm = False
        sample = {"og_robot_class": name,
                  "is_manipulator": has_arm,
                  "module": cls.__module__}
        # heuristic classification; GPU verification decides eligibility
        sample["likely_mobile_manipulator"] = any(
            k in lowered for k in ("fetch", "tiago", "r1", "stretch", "vector"))
        candidates[lowered] = sample
    return candidates


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="configs/level1/robot_pool_v1.yaml")
    args = parser.parse_args()
    candidates = discover()
    mobile = {k: v for k, v in candidates.items()
              if v.get("likely_mobile_manipulator")}
    header = "\n".join([
        "# §5/rev§2: filled by scripts/discover_level1_robots.py from the",
        "# installed OG registry. status: candidate until GPU verification",
        "# (URDF export, camera, MOVE/TURN/OBSERVE) passes. Never invent",
        "# unavailable robots; production_ready stays false until five real",
        "# mobile manipulators are verified (revision §2).",
        "",
    ])
    doc = {
        "robot_count": 5,
        "production_ready": False,
        "robots": {k: {"og_robot_class": v["og_robot_class"],
                       "status": "candidate",
                       "notes": f"discovered module {v['module']}"}
                   for k, v in sorted(mobile.items())},
    }
    payload = header + yaml.safe_dump(doc, sort_keys=False)
    out = Path(args.out)
    if not out.is_absolute():
        out = REPO / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(payload, encoding="utf-8")
    print(f"discovered {len(mobile)} mobile-manipulator candidates "
          f"({len(candidates)} robot classes total) -> {out}")
    print("candidates:", ", ".join(sorted(mobile)) or "NONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
