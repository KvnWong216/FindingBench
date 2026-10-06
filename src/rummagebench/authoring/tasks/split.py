"""Release split policy.

dev  : may share scenes (agent debugging / prompt development).
test : held-out SCENES by default. If too few scenes exist the policy may
       fall back to held-out landmark-region / room families, and the
       release metadata says so explicitly.
Every episode also carries its object-generalization keys so later
held-out-category / model / receptacle splits need no regeneration.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from rummagebench.authoring.tasks.plan import TaskPlan


@dataclass
class EpisodeRecord:
    task_id: str
    scene: str
    room: str
    room_type: str
    search_family: str  # region id, or "<scene>/<room>" for room-scoped episodes
    target_category: str
    target_model: str
    distractor_models: list[str]
    slot_type: str
    mode: str

    @classmethod
    def from_plan(cls, plan: TaskPlan, slot_type: str) -> "EpisodeRecord":
        return cls(task_id=plan.task_id, scene=plan.scene, room=plan.room,
                   room_type=plan.room_type,
                   search_family=plan.search_region or f"{plan.scene}/{plan.room}",
                   target_category=plan.target.category,
                   target_model=f"{plan.target.category}/{plan.target.model}",
                   distractor_models=sorted(f"{o.category}/{o.model}"
                                            for o in plan.distractors),
                   slot_type=slot_type, mode=plan.mode)


@dataclass
class Split:
    policy: str  # held_out_scene | held_out_family
    dev: list[str]
    test: list[str]
    test_scenes: list[str]
    test_families: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"policy": self.policy, "test_scenes": self.test_scenes,
                "test_families": self.test_families, "notes": self.notes,
                "dev": sorted(self.dev), "test": sorted(self.test)}


def make_split(records: Iterable[EpisodeRecord], test_scenes: list[str],
               min_scenes_for_scene_split: int = 3,
               test_families: Optional[list[str]] = None) -> Split:
    records = sorted(records, key=lambda r: r.task_id)
    scenes = sorted({r.scene for r in records})
    if len(scenes) >= min_scenes_for_scene_split and test_scenes:
        unknown = set(test_scenes) - set(scenes)
        if unknown:
            raise ValueError(f"test scenes without episodes: {sorted(unknown)}")
        if set(test_scenes) == set(scenes):
            raise ValueError("held-out scenes cannot be every scene")
        test = [r.task_id for r in records if r.scene in test_scenes]
        dev = [r.task_id for r in records if r.scene not in test_scenes]
        return Split("held_out_scene", dev, test, sorted(test_scenes))
    if not test_families:
        raise ValueError(f"only {len(scenes)} scene(s): a held-out-family fallback "
                         "needs explicit test_families")
    test = [r.task_id for r in records if r.search_family in test_families]
    dev = [r.task_id for r in records if r.search_family not in test_families]
    return Split("held_out_family", dev, test, [], sorted(test_families),
                 notes=[f"FALLBACK: only {len(scenes)} scene(s); test holds out "
                        "landmark-region / room families, NOT scenes"])


def leakage_violations(split: Split, records: Iterable[EpisodeRecord]) -> list[str]:
    by_id = {r.task_id: r for r in records}
    out = []
    if set(split.dev) & set(split.test):
        out.append("episode in both dev and test")
    dev = [by_id[t] for t in split.dev]
    test = [by_id[t] for t in split.test]
    if split.policy == "held_out_scene":
        leaked = {r.scene for r in dev} & {r.scene for r in test}
        if leaked:
            out.append(f"test scenes appear in dev: {sorted(leaked)}")
    else:
        leaked = {r.search_family for r in dev} & {r.search_family for r in test}
        if leaked:
            out.append(f"test families appear in dev: {sorted(leaked)}")
    if len({r.task_id for r in records}) != len(by_id):
        out.append("duplicate task ids")
    return out


def generalization_stats(records: Iterable[EpisodeRecord], ids: Iterable[str]) -> dict:
    by_id = {r.task_id: r for r in records}
    sel = [by_id[t] for t in ids]
    return {
        "episodes": len(sel),
        "target_category": dict(Counter(r.target_category for r in sel)),
        "target_model": len({r.target_model for r in sel}),
        "distractor_models": len({m for r in sel for m in r.distractor_models}),
        "slot_type": dict(Counter(r.slot_type for r in sel)),
        "room_type": dict(Counter(r.room_type for r in sel)),
        "mode": dict(Counter(r.mode for r in sel)),
    }
