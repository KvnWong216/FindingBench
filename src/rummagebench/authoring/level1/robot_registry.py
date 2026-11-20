"""§5/rev§2: robot eligibility registry.

A Level-1 robot must be a REAL MOBILE MANIPULATOR verified against the
installed OG assets: simulator articulation, controlled arm, RGB sensor,
body collision, URDF/kinematics export, supported holding, MOVE/TURN and
OBSERVE validity. Profiles come from configs/level1/robot_pool_v1.yaml,
which scripts/discover_level1_robots.py fills from the installed OG robot
registry. Unverified entries never enter production episodes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from rummagebench.authoring.level1.config import load_robot_pool
from rummagebench.authoring.tasks.plan import canonical_json, sha256_bytes, sha256_path

REQUIRED_CAPABILITIES = (
    "simulator_asset", "urdf_export", "controlled_arm", "eef_identity",
    "rgb_sensor", "body_collision", "supported_holding",
    "move_turn_valid", "observe_valid",
)


def runtime_protocol_hash() -> str:
    """Bind default public configuration and skill/camera implementation.

    Custom protocol configurations require separate verification; this
    registry's production gate covers the default public protocol only.
    """
    from rummagebench.core.public_types import AgentProtocolConfig

    root = Path(__file__).resolve().parents[2]
    sources = ["core/public_types.py", "core/visual_session.py",
               "sim/omnigibson/observation.py"]
    sources += [f"skills/{name}.py" for name in ("move", "observe", "open", "close", "grasp", "place", "report_done")]
    return sha256_bytes(canonical_json({"configuration": AgentProtocolConfig().model_dump(mode="json"),
                                       "sources": {p: sha256_path(root / p) for p in sources}}))


@dataclass
class RobotProfile:
    robot_id: str
    og_robot_class: str
    status: str = "candidate"            # candidate | gpu_verified | rejected
    base_link: str | None = None
    eef_link: str | None = None
    arm_joints: list[str] = field(default_factory=list)
    camera_sensor: str | None = None
    nominal_camera_height_m: float | None = None
    base_footprint_m: list[float] | None = None
    workspace_summary: str | None = None
    urdf_source: str | None = None
    missing_capabilities: list[str] = field(default_factory=list)
    notes: str | None = None
    evidence_path: str | None = None
    evidence_sha256: str | None = None

    def verified(self) -> bool:
        if (self.status != "gpu_verified" or self.missing_capabilities
                or not self.evidence_path or not self.evidence_sha256
                or not self.camera_sensor or not self.urdf_source):
            return False
        try:
            if sha256_path(self.evidence_path) != self.evidence_sha256:
                return False
            evidence = json.loads(Path(self.evidence_path).read_text())
            urdf_hash = sha256_path(self.urdf_source)
            return (bool(urdf_hash) and evidence.get("urdf_sha256") == urdf_hash
                    and evidence.get("robot_id") == self.robot_id
                    and evidence.get("camera_sensor") == self.camera_sensor
                    and evidence.get("protocol_sha256") == runtime_protocol_hash()
                    and all(evidence.get("capabilities", {}).get(cap) is True
                            for cap in REQUIRED_CAPABILITIES))
        except (OSError, ValueError, TypeError, AttributeError):
            return False


def load_profiles(path=None) -> dict[str, RobotProfile]:
    doc = load_robot_pool(path)
    profiles: dict[str, RobotProfile] = {}
    for robot_id, entry in (doc.get("robots") or {}).items():
        profiles[robot_id] = RobotProfile(
            robot_id=robot_id,
            og_robot_class=entry.get("og_robot_class", robot_id),
            status=entry.get("status", "candidate"),
            base_link=entry.get("base_link"),
            eef_link=entry.get("eef_link"),
            arm_joints=list(entry.get("arm_joints") or []),
            camera_sensor=entry.get("camera"),
            nominal_camera_height_m=entry.get("nominal_camera_height_m"),
            base_footprint_m=entry.get("base_footprint_m"),
            workspace_summary=entry.get("workspace_summary"),
            urdf_source=entry.get("urdf_source"),
            missing_capabilities=list(entry.get("missing_capabilities") or []),
            notes=entry.get("notes"),
            evidence_path=entry.get("evidence_path"),
            evidence_sha256=entry.get("evidence_sha256"),
        )
    return profiles


def verified_profiles(path=None) -> dict[str, RobotProfile]:
    """Only GPU-verified mobile manipulators may enter production episodes."""
    return {rid: p for rid, p in load_profiles(path).items() if p.verified()}


def production_ready(path=None, robot_count: int = 5) -> tuple[bool, list[str]]:
    verified = verified_profiles(path)
    return (len(verified) >= robot_count,
            sorted(k for k, p in load_profiles(path).items()
                   if not p.verified()))


def skeleton_entry(robot_id: str, og_robot_class: str,
                   extra: dict[str, Any] | None = None) -> dict[str, Any]:
    entry = {"og_robot_class": og_robot_class, "status": "candidate"}
    if extra:
        entry.update(extra)
    return entry
