"""Episode certification (§3): every benchmark episode is certified solvable
by the oracle semantic planner BEFORE it enters a benchmark split.

    certify_episode(scenario, session) -> EpisodeCertificate

The certificate records the exact oracle depth, the optimal semantic plan,
and the provenance hashes (scenario YAML, robot URDF) so a split can be
audited. Episodes the oracle proves UNSOLVABLE_FOR_EMBODIMENT are excluded
from the standard solvable split and may be retained in a separate
embodiment_stress split — they must never contaminate task-success
evaluation.

Remediation R11: the NAV-based BFS depth is stored as ``oracle_depth`` /
``oracle_symbolic_depth`` under an explicit ``action_space_id``; the
public MOVE/TURN/point/REPORT_DONE protocol has its own ``public_witness_steps``
(filled by the public witness when certified) and ``public_optimal_steps``
(null until actually proven — the NAV depth must never be presented as
public-protocol optimality).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from rummagebench.core.scenario import ScenarioSpec
from rummagebench.evaluation.oracle_planner import (
    OraclePlanResult,
    initial_oracle_state,
    solve_full_information,
)

CERTIFICATION_VERSION = "cert-v1"
# the symbolic NAV-based BFS action space (legacy oracle certification)
ORACLE_SYMBOLIC_ACTION_SPACE_ID = "oracle_nav_symbolic_v1"


@dataclass
class EpisodeCertificate:
    episode_id: str
    solvable: bool
    oracle_depth: int | None
    oracle_plan: list[dict[str, Any]]
    target_container: str | None
    robot_variant: str
    action_interface_mode: str
    certification_version: str
    scenario_hash: str
    robot_urdf_hash: str | None
    expanded_states: int = 0
    visited_states: int = 0
    reason: str | None = None
    # replay verification: the certified plan executed through the
    # production BenchmarkSession reaches the task goal
    plan_replay_status: str | None = None  # SUCCESS | <failure status> | None
    plan_replay_steps: int | None = None
    # remediation R11: oracle depth and public-protocol step counts are
    # separate quantities over different action spaces; never conflate them
    action_space_id: str = ORACLE_SYMBOLIC_ACTION_SPACE_ID
    oracle_symbolic_depth: int | None = None  # mirrors oracle_depth
    public_witness_steps: int | None = None  # public protocol witness (certified)
    public_optimal_steps: int | None = None  # null unless actually proven
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def scenario_hash(scenario: ScenarioSpec) -> str:
    """Stable hash of the canonical scenario content."""
    import yaml

    canonical = yaml.safe_dump(scenario.model_dump(mode="json"), sort_keys=True)
    return hashlib.sha1(canonical.encode()).hexdigest()[:12]


def robot_urdf_hash(scenario: ScenarioSpec) -> str | None:
    if scenario.robot.kinematics is None:
        return None
    from rummagebench.robots.model_loader import resolve_urdf_path

    try:
        urdf = resolve_urdf_path(scenario.robot.kinematics.urdf_path)
    except Exception:
        return None
    return hashlib.sha1(urdf.read_bytes()).hexdigest()[:12]


def certify_episode(
    scenario: ScenarioSpec,
    session,
    robot_variant: str = "default",
) -> EpisodeCertificate:
    """Certify one episode with the oracle planner bound to a LIVE session
    (production backend + feasibility validator)."""
    from rummagebench.evaluation.oracle_planner import (
        OracleFeasibilityModel,
        initial_oracle_state,
    )

    model = OracleFeasibilityModel(session._backend, scenario, session._feasibility)
    model.bind_robot(session.robot)
    initial = initial_oracle_state(scenario)
    result: OraclePlanResult = solve_full_information(scenario, initial, model)

    target_container = None
    for p in scenario.placements:
        if p.entity == scenario.target.entity:
            target_container = p.receptacle
            break

    return EpisodeCertificate(
        episode_id=scenario.id,
        solvable=result.solvable,
        oracle_depth=result.depth,
        oracle_plan=result.actions,
        target_container=target_container,
        robot_variant=robot_variant,
        action_interface_mode=scenario.action_interface.mode,
        certification_version=CERTIFICATION_VERSION,
        scenario_hash=scenario_hash(scenario),
        robot_urdf_hash=robot_urdf_hash(scenario),
        expanded_states=result.expanded_states,
        visited_states=result.visited_states,
        reason=result.reason,
        action_space_id=ORACLE_SYMBOLIC_ACTION_SPACE_ID,
        oracle_symbolic_depth=result.depth,
        # public-protocol step counts are separate certificates (R11): the
        # NAV BFS depth is not a public-protocol witness, let alone an
        # optimal-step proof — both stay null until actually certified
        public_witness_steps=None,
        public_optimal_steps=None,
        extra={
            "oracle_min_steps_applied": (
                result.depth if result.solvable else None
            ),
        },
    )


def save_certificate(certificate: EpisodeCertificate, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{certificate.episode_id}.json"
    path.write_text(certificate.to_json(), encoding="utf-8")
    return path


def load_certificate(path: str | Path) -> EpisodeCertificate:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return EpisodeCertificate(**data)


def depth_statistics(certificates: list[EpisodeCertificate]) -> dict[str, Any]:
    """Depth distribution over SOLVABLE certificates (§10)."""
    depths = sorted(c.oracle_depth for c in certificates if c.solvable and c.oracle_depth)
    if not depths:
        return {"depth_histogram": {}, "min_depth": None, "max_depth": None,
                "mean_depth": None, "num_solvable": 0}
    histogram: dict[str, int] = {}
    for d in depths:
        histogram[str(d)] = histogram.get(str(d), 0) + 1
    return {
        "depth_histogram": histogram,
        "min_depth": depths[0],
        "max_depth": depths[-1],
        "mean_depth": round(sum(depths) / len(depths), 3),
        "num_solvable": len(depths),
    }
