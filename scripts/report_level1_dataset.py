"""§47: dataset statistics report (registry-driven, CPU only).

    PYTHONPATH=src python scripts/report_level1_dataset.py
"""
from __future__ import annotations

from pathlib import Path

from rummagebench.authoring.level1.registry import DatasetRegistry

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    registry_path = REPO / "build" / "level1" / "dataset_registry.json"
    if not registry_path.exists():
        raise SystemExit("dataset registry not found; run generation first")
    registry = DatasetRegistry.load(registry_path)
    accepted = registry.accepted_environments()
    by_paradigm: dict[str, int] = {}
    for e in accepted:
        by_paradigm[e.paradigm] = by_paradigm.get(e.paradigm, 0) + 1
    rejections: dict[str, int] = {}
    for r in registry.rejections:
        code = r.get("code") or "UNKNOWN"
        rejections[code] = rejections.get(code, 0) + 1
    report = {
        "accepted_environments": len(accepted),
        "by_paradigm": by_paradigm,
        "episodes": sum(len(e.robot_ids) for e in accepted),
        "rejection_histogram": rejections,
        "invalid_runs": len(registry.invalid_runs),
        "registry_hash": registry.hash(),
        "reserve": sorted(e.environment_id for e in registry.entries.values()
                          if e.split == "morphology_stress_reserve"),
    }
    for e in accepted:
        for robot_id, ok in sorted(e.per_robot_certification.items()):
            report.setdefault("per_robot_pass", {}).setdefault(robot_id, 0)
            report["per_robot_pass"][robot_id] += int(bool(ok))
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
