"""TaskPlan: the deterministic planner output.

A TaskPlan is pure data. Its canonical serialisation (sorted-key compact
JSON) is what the determinism gate compares byte for byte, and plan_hash is
derived from it. Provenance records every input hash — the same seed under
changed slots/priors is NOT the same episode.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

GENERATOR_VERSION = "tasks-factory-1"


@dataclass
class PlacedObject:
    entity: str
    category: str
    model: str
    slot_id: str
    role: str  # target | target_slot | other_container | other_surface


@dataclass
class TaskPlan:
    tier: str
    tier_version: int
    mode: str
    seed: int
    episode_index: int
    scene: str
    room: str
    room_type: str
    robot_id: str
    target: PlacedObject
    distractors: list[PlacedObject]
    start_pose: str
    start: dict[str, list[float]]  # anchor dict: position, orientation
    instruction: str
    # search-region facts the planner designed for; RE-DERIVED by the
    # certificate from the assets, never trusted as-is
    search_rooms: list[str]
    candidate_slots: list[str]
    expected: dict[str, int]
    provenance: dict[str, Any] = field(default_factory=dict)
    # landmark search region named by the instruction;
    # None = the whole room(s) in search_rooms
    search_region: Optional[str] = None

    @property
    def objects(self) -> list[PlacedObject]:
        return [self.target] + list(self.distractors)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TaskPlan":
        d = dict(d)
        d["target"] = PlacedObject(**d["target"])
        d["distractors"] = [PlacedObject(**o) for o in d["distractors"]]
        return cls(**d)

    def canonical_bytes(self) -> bytes:
        return canonical_json(self.to_dict())

    @property
    def plan_hash(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def identity_bytes(self) -> bytes:
        """Episode identity: content + input hashes + robot, WITHOUT the
        generator commit (same inputs at another commit = same episode)."""
        d = self.to_dict()
        prov = d.pop("provenance") or {}
        d["inputs"] = prov.get("inputs")
        d["robot_provenance"] = prov.get("robot")
        return canonical_json(d)

    @property
    def task_id(self) -> str:
        """Opaque episode id (ids carry no semantics)."""
        return "fb_" + hashlib.sha256(self.identity_bytes()).hexdigest()[:16]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.canonical_bytes())

    @classmethod
    def load(cls, path: Path) -> "TaskPlan":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: str | Path) -> Optional[str]:
    p = Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def git_state(repo: Path) -> dict[str, Any]:
    """Commit + dirty flag; never raises (a tarball checkout has no git)."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True,
                                capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "src", "assets"],
                                    cwd=repo, check=True, capture_output=True,
                                    text=True).stdout.strip())
        return {"git_commit": commit, "git_dirty": dirty}
    except Exception:
        return {"git_commit": None, "git_dirty": None}


def provenance(*, generator: dict[str, Any], inputs: dict[str, Optional[str]],
               environment: dict[str, Any], robot: dict[str, Any],
               seed: int, episode_index: int) -> dict[str, Any]:
    """Input-hash record. scenario_hash is added by the compiler."""
    missing = [k for k, v in inputs.items() if not v]
    if missing:
        raise ValueError(f"provenance input hashes missing: {missing}")
    return {
        "generator": {"version": GENERATOR_VERSION, **generator},
        "inputs": dict(sorted(inputs.items())),
        "environment": environment,
        "robot": robot,
        "episode": {"seed": int(seed), "episode_index": int(episode_index)},
    }
