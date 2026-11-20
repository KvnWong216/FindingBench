"""§36/§44: dataset registry — accepted/rejected/reserve records, hashes,
and §36/rev§1: split by ENVIRONMENT id (all five robots of one environment
share the split), stratified by paradigm, deterministic."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from rummagebench.authoring.level1.snapshot import canonical_hash

RESERVE = "morphology_stress_reserve"


@dataclass
class RegistryEntry:
    environment_id: str
    environment_seed: int
    paradigm: str
    support_model: str
    support_height: float | None = None
    support_size: list[float] | None = None
    container_model: str | None = None
    container_size: list[float] | None = None
    target_category: str | None = None
    target_model: str | None = None
    object_count: int | None = None
    categories: list[str] = field(default_factory=list)
    models: list[str] = field(default_factory=list)
    visibility_ratio: float | None = None
    packing_fraction: float | None = None
    contact_count: int | None = None
    support_depth: int | None = None
    appearance_seed: int | None = None
    canonical_manifest_hash: str | None = None
    robot_ids: list[str] = field(default_factory=list)
    per_robot_certification: dict[str, bool] = field(default_factory=dict)
    split: str | None = None
    accepted: bool = False
    failing_robot: str | None = None
    rejection_code: str | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


class DatasetRegistry:
    def __init__(self) -> None:
        self.entries: dict[str, RegistryEntry] = {}
        self.rejections: list[dict] = []
        self.invalid_runs: list[dict] = []

    def accept(self, entry: RegistryEntry, split: str) -> None:
        entry.accepted = True
        entry.split = split
        self.entries[entry.environment_id] = entry

    def reject(self, environment_id: str, environment_seed: int, paradigm: str,
               code: str, failing_robot: str | None = None,
               details: dict | None = None) -> None:
        self.rejections.append({
            "environment_id": environment_id,
            "environment_seed": environment_seed,
            "paradigm": paradigm,
            "code": code,
            "failing_robot": failing_robot,
            "details": details or {},
        })

    def record_invalid_run(self, environment_id: str, robot_id: str,
                           stage: str, error: str) -> None:
        """§46: simulator crashes are infrastructure, not scene semantics."""
        self.invalid_runs.append({"environment_id": environment_id,
                                  "robot_id": robot_id, "stage": stage,
                                  "error": error})

    def reserve(self, entry: RegistryEntry, failing_robot: str,
                code: str) -> None:
        """§30/rev§17: partially certified environments go to the
        morphology_stress_reserve with the failing robot recorded."""
        entry.accepted = False
        entry.split = RESERVE
        entry.failing_robot = failing_robot
        entry.rejection_code = code
        self.entries[entry.environment_id] = entry

    def accepted_environments(self) -> list[RegistryEntry]:
        return [e for e in self.entries.values() if e.accepted]

    def accepted_count(self, paradigm: str | None = None) -> int:
        return sum(1 for e in self.accepted_environments()
                   if paradigm is None or e.paradigm == paradigm)

    def all_robot_pairs(self) -> list[tuple[str, str]]:
        return [(e.environment_id, r) for e in self.accepted_environments()
                for r in e.robot_ids]

    def hash(self) -> str:
        return canonical_hash({
            "accepted": [e.to_dict() for e in self.accepted_environments()],
            "rejections": self.rejections,
            "invalid_runs": self.invalid_runs,
        })

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        doc = {"hash": self.hash(),
               "entries": [e.to_dict() for e in self.entries.values()],
               "rejections": self.rejections,
               "invalid_runs": self.invalid_runs}
        path.write_text(json.dumps(doc, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "DatasetRegistry":
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        registry = cls()
        import dataclasses

        names = {f.name for f in dataclasses.fields(RegistryEntry)}
        for d in doc.get("entries", []):
            entry = RegistryEntry(**{k: v for k, v in d.items() if k in names})
            registry.entries[entry.environment_id] = entry
        registry.rejections = doc.get("rejections", [])
        registry.invalid_runs = doc.get("invalid_runs", [])
        return registry
