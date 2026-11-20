"""Dynamic skill grounding: regenerate the available action space each step.

    A_t = Ground(RobotGeometry, ObjectInterface, WorldGeometry, WorldState_t)

The grounder is explicitly layered (no duplicated logic):

    semantic_candidates(robot, world)   semantic + state-valid skills for
                                        VISIBLE objects
    physical_filter(candidates)         IK + configuration-space collision
                                        + state preconditions

Two agent-facing protocols (scenario.action_interface.mode):

    admissible: Observation.available_skills <- physical_filter(candidates)
    candidate:  Observation.candidate_skills <- candidates (labels only; the
                feasibility oracle answers attempts with structured step
                feedback — never with scores or IK/collision metadata)

Visible-object discipline: candidates are generated ONLY for
backend.visible_entities(); contents of closed containers never leak into
the agent-facing action space, while oracle entity grounding remains
available for action execution and evaluator-side tracing.
"""

from __future__ import annotations

from dataclasses import dataclass

import logging

from rummagebench.core.scenario import ScenarioSpec
from rummagebench.core.types import TargetKind
from rummagebench.object_interface.base import SkillCandidate, build_object_interface
from rummagebench.robots.robot import RobotEmbodiment
from rummagebench.sim.base import ResolvedTarget, SimBackend
from rummagebench.validation.feasibility import FeasibilityValidator

logger = logging.getLogger(__name__)


@dataclass
class CandidateVerdict:
    """Evaluator-side grounding result for one semantic candidate."""

    candidate: SkillCandidate
    feasible: bool | None  # None when the mode skips the physical filter
    reason: str  # AVAILABLE | UNREACHABLE | COLLISION | INVALID_STATE
    details: dict


class SkillGrounder:
    """Enumerates the skills that exist for the robot right now.

    This is the benchmark's core mechanism, the Embodied Action Grounding
    Engine: the admissible action space is not fixed by the skill library but
    derived per step from robot geometry x object interface x world state.
    """

    def __init__(
        self,
        backend: SimBackend,
        scenario: ScenarioSpec,
        feasibility: FeasibilityValidator | None = None,
    ):
        self._backend = backend
        self._scenario = scenario
        if feasibility is None:
            from rummagebench.core.session import _build_feasibility

            feasibility = _build_feasibility(backend, scenario)
        self._feasibility = feasibility
        # state-fingerprint cache: identical (visible objects, semantic state,
        # robot pose bucket) reuses the previous grounding verbatim — grounding
        # is a pure function of that state, so caching is semantics-preserving
        self._cache: dict[str, tuple] = {}

    def _fingerprint(self) -> str:
        import hashlib
        import json as _json

        backend = self._backend
        pos, _ = backend.robot_pose()
        bucket = [round(v / 0.05) for v in pos]
        opens = tuple(sorted(e for e in backend.entity_names() if backend.is_open(e)))
        payload = _json.dumps(
            {
                "visible": sorted(backend.visible_entities()),
                "opens": opens,
                "held": getattr(self._world_ref, "held_object", None),
                "base": bucket,
            },
            sort_keys=True,
        )
        return hashlib.sha1(payload.encode()).hexdigest()

    def bind_world(self, world) -> None:
        """Bind the benchmark world-state object (for the fingerprint)."""
        self._world_ref = world

    # ------------------------------------------------------------- layering

    def semantic_candidates(self, robot: RobotEmbodiment, world) -> list[SkillCandidate]:
        """Layer 1: semantic + state-valid skills over VISIBLE objects.

        NAV is a global skill over build-time-verified anchors. Entity skills
        are proposed by the object adapters exclusively for entities that are
        currently observable — closed-container contents are excluded here,
        so they can never leak into any agent-facing protocol.
        """
        candidates: list[SkillCandidate] = []
        for anchor in self._scenario.anchors:
            candidates.append(SkillCandidate(skill="NAV", target=anchor))

        for name in self._backend.visible_entities():
            info = self._backend.describe_entity(name)
            if info is None:
                continue
            try:
                obj = build_object_interface(info, self._backend)
                candidates.extend(obj.available_skills(robot, world))
            except Exception as e:
                # one corrupted entity (PhysX execution noise) degrades only
                # itself, never the whole grounding pass
                logger.warning("skill proposal failed for %s: %s", name, e)
        return candidates

    def physical_filter(
        self, candidates: list[SkillCandidate], robot: RobotEmbodiment, world,
        collect: list[CandidateVerdict] | None = None,
    ) -> list[SkillCandidate]:
        """Layer 2: keep candidates with at least one feasible interaction
        configuration (real IK + configuration-space collision)."""
        kept: list[SkillCandidate] = []
        for cand in candidates:
            try:
                resolved, verdict = self._evaluate(cand, robot, world)
            except Exception as e:
                logger.warning(
                    "feasibility evaluation failed for %s(%s): %s",
                    cand.skill, cand.target, e,
                )
                if collect is not None:
                    collect.append(CandidateVerdict(
                        candidate=cand, feasible=None, reason="EVALUATION_ERROR",
                        details={"error": str(e)},
                    ))
                continue
            if collect is not None:
                collect.append(CandidateVerdict(
                    candidate=cand,
                    feasible=verdict.feasible,
                    reason=self._reason_of(verdict),
                    details=verdict.details,
                ))
            if verdict.feasible:
                kept.append(cand)
        return kept

    def _evaluate(self, cand: SkillCandidate, robot: RobotEmbodiment, world):
        if cand.skill == "NAV":
            resolved = ResolvedTarget(
                kind=TargetKind.PLACE, place=cand.target,
                anchor=self._scenario.anchors.get(cand.target),
            )
            verdict = self._feasibility.check(cand.skill, resolved, robot, world)
            return resolved, verdict
        info = self._backend.describe_entity(cand.target)
        resolved = ResolvedTarget(kind=TargetKind.ENTITY, entity=cand.target, info=info)
        verdict = self._feasibility.check(cand.skill, resolved, robot, world)
        return resolved, verdict

    @staticmethod
    def _reason_of(verdict) -> str:
        return verdict.reason if not verdict.feasible else "AVAILABLE"

    # ------------------------------------------------------------- protocols

    def ground(self, robot: RobotEmbodiment, world) -> list[str]:
        """Agent-facing action space per the configured protocol."""
        mode = self._scenario.action_interface.mode
        candidates = self.semantic_candidates(robot, world)
        if mode == "candidate":
            return sorted({c.label() for c in candidates})
        kept = self.physical_filter(candidates, robot, world)
        return sorted({c.label() for c in kept})

    def ground_with_verdicts(self, robot: RobotEmbodiment, world):
        """Full evaluator-side grounding: candidates + per-candidate verdicts
        (physical filter ALWAYS applied here regardless of the agent-facing
        protocol — this is the oracle view for traces, metrics and graphs).
        Cached per world-state fingerprint."""
        self._world_ref = world
        key = self._fingerprint()
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        candidates = self.semantic_candidates(robot, world)
        verdicts: list[CandidateVerdict] = []
        kept = self.physical_filter(candidates, robot, world, collect=verdicts)
        result = (candidates, kept, verdicts)
        self._cache[key] = result
        if len(self._cache) > 64:
            self._cache.pop(next(iter(self._cache)))
        return result

    def candidate_labels(self, robot: RobotEmbodiment, world) -> list[str]:
        return sorted({c.label() for c in self.semantic_candidates(robot, world)})

    def admissible_labels(self, robot: RobotEmbodiment, world) -> list[str]:
        return sorted({c.label() for c in self.physical_filter(
            self.semantic_candidates(robot, world), robot, world
        )})


# Paper-facing alias: the grounder is the Embodied Action Grounding Engine
# (EAGE) described in the benchmark design notes.
EmbodiedActionGroundingEngine = SkillGrounder
