"""SlotEmbodimentOverlay: robot-dependent slot facts, one file per (scene, robot)
(assets/tasks/embodiment/<scene>__<robot>.yaml).

Per slot: an evaluator-private navigation anchor and probed feasibility
flags. Per room: robot start poses. A flag of None means "not probed";
planning only accepts overlays with status "probed".
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

OVERLAY_VERSION = 1


@dataclass(frozen=True)
class Pose2:
    position: tuple[float, float, float]
    orientation: tuple[float, float, float, float]  # xyzw

    @property
    def xy(self) -> tuple[float, float]:
        return (self.position[0], self.position[1])

    @property
    def yaw(self) -> float:
        import math
        x, y, z, w = self.orientation
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def to_anchor(self) -> dict[str, list[float]]:
        return {"position": list(self.position), "orientation": list(self.orientation)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Pose2":
        return cls(tuple(float(v) for v in d["position"]),
                   tuple(float(v) for v in d["orientation"]))


@dataclass
class StartPose:
    name: str
    pose: Pose2
    # slot ids visible from this pose (probe output); None = not probed
    visible_slots: Optional[list[str]] = None


@dataclass
class SlotInteraction:
    navigation_anchor: Pose2
    open_feasible: Optional[bool] = None
    grasp_region_feasible: Optional[bool] = None
    probed: bool = False
    note: str = ""
    # raw probe measurements (placement, GRASP-from, open/closed visibility
    # of the probe object) — audit trail, not read by planning
    evidence: dict[str, Any] = field(default_factory=dict)
    # slot-level pose where the revealed slot content is visible AND
    # graspable (may differ from the furniture's OPEN pose); None = none
    reveal_anchor: Optional[Pose2] = None

    @property
    def open_causal(self) -> Optional[bool]:
        """Probe evidence that OPEN is a causal reveal for this slot:
        the probe object was invisible AND ungraspable with the furniture
        closed. None = not measured (unprobed, or not an OPEN slot)."""
        ev = self.evidence or {}
        vis = (ev.get("closed_visibility") or {}).get("visible_px")
        grasp = ev.get("closed_graspable")
        if vis is None or grasp is None:
            return None
        return vis == 0 and grasp is False

    @property
    def reveal_visibility(self) -> Optional[float]:
        """Probe object's visibility ratio from the anchor with the slot
        revealed (open); None = not measured, 0.0 = out of view."""
        m = (self.evidence or {}).get("open_visibility")
        if not m:
            return None
        return 0.0 if m.get("ratio") is None else float(m["ratio"])

    def usable(self, requires_open: bool) -> bool:
        """Embodiment can serve this slot (every needed flag probed True)."""
        flags = [self.grasp_region_feasible]
        if requires_open:
            flags.append(self.open_feasible)
        return all(f is True for f in flags)


@dataclass
class SlotEmbodimentOverlay:
    scene: str
    robot_id: str
    urdf_path: str
    urdf_hash: Optional[str]
    status: str  # probed | partial
    source: str
    robot_config: dict[str, Any]
    feasibility: dict[str, Any]
    start_poses: dict[str, list[StartPose]]  # room -> poses
    slots: dict[str, SlotInteraction]  # slot_id -> interaction
    version: int = OVERLAY_VERSION
    notes: list[str] = field(default_factory=list)

    @property
    def probed(self) -> bool:
        return self.status == "probed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene": self.scene, "version": self.version,
            "robot": {"id": self.robot_id, "urdf_path": self.urdf_path,
                      "urdf_hash": self.urdf_hash},
            "status": self.status, "source": self.source, "notes": list(self.notes),
            "robot_config": self.robot_config, "feasibility": self.feasibility,
            "start_poses": {room: [{"name": p.name, **p.pose.to_anchor(),
                                    "visible_slots": p.visible_slots}
                                   for p in poses]
                            for room, poses in sorted(self.start_poses.items())},
            "slots": {sid: {"navigation_anchor": s.navigation_anchor.to_anchor(),
                            "open_feasible": s.open_feasible,
                            "grasp_region_feasible": s.grasp_region_feasible,
                            "probed": s.probed, "note": s.note,
                            **({"reveal_anchor": s.reveal_anchor.to_anchor()}
                               if s.reveal_anchor is not None else {}),
                            **({"evidence": s.evidence} if s.evidence else {})}
                      for sid, s in sorted(self.slots.items())},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SlotEmbodimentOverlay":
        starts = {room: [StartPose(p["name"], Pose2.from_dict(p), p.get("visible_slots"))
                         for p in poses]
                  for room, poses in (d.get("start_poses") or {}).items()}
        slots = {sid: SlotInteraction(Pose2.from_dict(s["navigation_anchor"]),
                                      s.get("open_feasible"),
                                      s.get("grasp_region_feasible"),
                                      bool(s.get("probed", False)), s.get("note", ""),
                                      dict(s.get("evidence") or {}),
                                      Pose2.from_dict(s["reveal_anchor"])
                                      if s.get("reveal_anchor") else None)
                 for sid, s in (d.get("slots") or {}).items()}
        r = d["robot"]
        return cls(scene=d["scene"], robot_id=r["id"], urdf_path=r["urdf_path"],
                   urdf_hash=r.get("urdf_hash"), status=d["status"],
                   source=d.get("source", ""), robot_config=dict(d["robot_config"]),
                   feasibility=dict(d.get("feasibility") or {}),
                   start_poses=starts, slots=slots,
                   version=int(d.get("version", OVERLAY_VERSION)),
                   notes=list(d.get("notes") or []))

    def save(self, path: Path, header: str = "") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(header + yaml.safe_dump(self.to_dict(), sort_keys=False, width=100),
                        encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "SlotEmbodimentOverlay":
        return cls.from_dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def overlay_path(assets_dir: Path, scene: str, robot_id: str) -> Path:
    return Path(assets_dir) / "embodiment" / f"{scene}__{robot_id}.yaml"


def sha256_file(path: str | Path) -> Optional[str]:
    p = Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
