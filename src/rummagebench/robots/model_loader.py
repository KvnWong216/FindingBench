"""Robot model loading: build the feasibility kinematics backend from the
scenario robot spec.

Backend selection (fail loudly, never silently degrade):

- ``feasibility.backend: pinocchio`` (production default) requires
  ``robot.kinematics.urdf_path``; the URDF must come from the robot asset
  (OmniGibson articulation export via scripts/export_robot_kinematics.py or a
  vendor URDF). Any load failure raises FeasibilityBackendError.
- ``feasibility.backend: proxy`` selects the reach-radius solver. It exists
  for unit tests and coarse prefilters ONLY and is never auto-selected.

reach_radius / z_min / z_max stay capability *data* on RobotEmbodiment but
cannot authorize an interaction in the pinocchio backend: feasibility is
derived from URDF geometry, joint limits, IK and configuration-space
collision.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from rummagebench.core.errors import FeasibilityBackendError

if TYPE_CHECKING:  # pragma: no cover
    from rummagebench.core.scenario import FeasibilitySpec, KinematicsSpec, RobotSpec

logger = logging.getLogger(__name__)

# link-name substrings treated as gripper/finger class when no explicit list
# is configured (used by the allowed-collision matrix)
_GRIPPER_NAME_HINTS = ("finger", "gripper", "hand", "eef")


def resolve_urdf_path(raw: str, scenario_dir: Path | None = None) -> Path:
    """Resolve a kinematics urdf_path.

    Order: absolute -> {RUMMAGEBENCH_ROOT} prefix -> scenario-relative ->
    repo-root relative. The resolved file must exist, otherwise raise
    (no silent fallbacks).
    """
    raw_expanded = os.path.expandvars(raw)
    if "{RUMMAGEBENCH_ROOT}" in raw_expanded:
        raw_expanded = raw_expanded.replace("{RUMMAGEBENCH_ROOT}", str(repo_root()))
    path = Path(raw_expanded)
    if path.is_absolute():
        if not path.is_file():
            raise FeasibilityBackendError(
                f"robot.kinematics.urdf_path not found: {path}"
            )
        return path
    candidates: list[Path] = []
    if scenario_dir is not None:
        candidates.append(scenario_dir / path)
    candidates.append(repo_root() / path)
    for cand in candidates:
        if cand.is_file():
            return cand
    tried = ", ".join(str(c) for c in candidates)
    raise FeasibilityBackendError(
        f"robot.kinematics.urdf_path {raw!r} not found (tried: {tried}). "
        "Export it with scripts/export_robot_kinematics.py on the simulator host."
    )


def repo_root() -> Path:
    """Repository root: $RUMMAGEBENCH_ROOT if set, else the package's parent."""
    env = os.environ.get("RUMMAGEBENCH_ROOT")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[3]


def load_kinematics_backend(
    robot_spec: "RobotSpec",
    feasibility_spec: "FeasibilitySpec",
    scenario_dir: Path | None = None,
):
    """Build the kinematics backend requested by the scenario config.

    Returns an object implementing solve_ik()/forward_kinematics() — either a
    PinocchioKinematics (URDF-backed) or, when explicitly configured, the
    legacy reach-radius proxy for tests.
    """
    if feasibility_spec.backend == "proxy":
        if feasibility_spec.backend == "proxy" and robot_spec.kinematics is not None:
            logger.warning(
                "feasibility.backend=proxy with a kinematics block present: "
                "the URDF will be IGNORED. This is test-only configuration."
            )
        from rummagebench.feasibility.ik_solver import ReachabilityIKSolver

        return ReachabilityIKSolver()

    if feasibility_spec.backend == "pinocchio":
        kin = robot_spec.kinematics
        if kin is None:
            raise FeasibilityBackendError(
                "feasibility.backend=pinocchio requires robot.kinematics "
                "(urdf_path, base_link, end_effector_link). Add the block to "
                "the scenario YAML or switch tests to backend: proxy."
            )
        try:
            import pinocchio  # noqa: F401
        except ImportError as e:
            raise FeasibilityBackendError(
                "feasibility.backend=pinocchio requires the 'pin' package "
                "(pip install pin). Refusing to fall back to the reach-radius "
                "proxy."
            ) from e
        urdf = resolve_urdf_path(kin.urdf_path, scenario_dir)
        from rummagebench.feasibility.pinocchio_solver import PinocchioKinematics

        return PinocchioKinematics(
            urdf_path=str(urdf),
            base_link=kin.base_link,
            eef_link=kin.end_effector_link,
            controlled_joints=kin.controlled_joints,
            gripper_links=kin.gripper_links,
            pos_tol=feasibility_spec.ik_pos_tol,
            rot_tol=feasibility_spec.ik_rot_tol,
            max_iters=feasibility_spec.ik_max_iters,
            restarts=feasibility_spec.ik_restarts,
            seed=kin.ik_seed,
            collision_padding=feasibility_spec.collision_padding,
        )

    raise FeasibilityBackendError(
        f"unknown feasibility.backend {feasibility_spec.backend!r}"
    )


def classify_robot_link(link_name: str, eef_link: str, gripper_links: list[str] | None) -> str:
    """Allowed-collision-matrix link class: 'gripper' or 'arm'.

    A link is gripper-class when explicitly listed, when it is the configured
    end-effector link, when it is a URDF-child of the EEF link (fingers hang
    below the tool frame), or when its name matches a gripper hint. Everything
    else is arm-class (subject to forbidden contact with foreign bodies).
    """
    if gripper_links and link_name in gripper_links:
        return "gripper"
    if link_name == eef_link:
        return "gripper"
    lowered = link_name.lower()
    if any(h in lowered for h in _GRIPPER_NAME_HINTS):
        return "gripper"
    return "arm"
