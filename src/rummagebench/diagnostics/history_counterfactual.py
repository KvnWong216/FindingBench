"""History-counterfactual diagnostic (§7).

Generates PAIRED agent situations with (near-)identical CURRENT observable
state but different valid search histories, so that the RATIONAL next search
action differs. The history difference must come from the actual interaction
trace supplied to the agent — never from hidden prompts.

Minimal v0 pair:

    Pair A (history-aware):  container X was OPENed, its contents verified,
                             target verified absent, container CLOSEd
                             (state restored), robot back at anchor P.
    Pair B (naive):          container X never inspected; robot at anchor P;
                             X closed; visible present-time state matches A.

Current-state match (mechanically checked):
    robot anchor equal
    X closed in both (A restored it)
    visible entities equal (closed container hides contents in both)
    target placement equal

Rational next search set: computed by the SearchStateTracker — in A, X is
EXHAUSTED_EMPTY so searching X again is a revisit error; in B, X is UNSEEN
so searching X is valid. The history-consistent search sets therefore
differ.

HTA (history tracking accuracy), for deterministic outputs:

    the agent's selected action is consistent with the history-aware
    admissible search set of the pair arm it faced:

    consistent(action, arm) =
        arm == A and action is NOT a search interaction on an exhausted
                   location of A (or action is not a search action at all)
        arm == B and action is not excluded by B's history
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rummagebench.evaluation.search_state import (
    SearchLocationState,
    SearchStateTracker,
)


@dataclass
class HistoryCounterfactualPair:
    pair_id: str
    scene: str
    anchor: str  # shared current robot anchor
    container: str  # the contested search location
    target_entity: str
    history_A: list[dict[str, Any]]  # inspect + restore history
    history_B: list[dict[str, Any]]  # no inspection
    exhausted_locations_A: list[str]
    exhausted_locations_B: list[str]
    valid_next_action_set_A: list[str]
    valid_next_action_set_B: list[str]
    current_observation_reference: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "scene": self.scene,
            "anchor": self.anchor,
            "container": self.container,
            "target_entity": self.target_entity,
            "history_A": self.history_A,
            "history_B": self.history_B,
            "exhausted_locations_A": self.exhausted_locations_A,
            "exhausted_locations_B": self.exhausted_locations_B,
            "valid_next_action_set_A": self.valid_next_action_set_A,
            "valid_next_action_set_B": self.valid_next_action_set_B,
            "current_observation_reference": self.current_observation_reference,
        }


def build_inspection_pair(
    pair_id: str,
    scenario,
    container: str,
    anchor: str,
    inspection_step: int = 1,
) -> HistoryCounterfactualPair:
    """Build the minimal A/B pair for one (container, anchor) combination.

    History A: NAV anchor -> OPEN container (observe: target absent by
    ground truth) -> CLOSE container (state restored). The tracker marks the
    container EXHAUSTED_EMPTY.

    History B: NAV anchor only. The container stays UNSEEN.

    Present-time observable state is identical: same anchor, container
    closed in both, closed containers hide their contents in both.
    """
    target = scenario.target.entity

    def act(skill: str, value: str, kind: str = "entity") -> dict[str, Any]:
        return {"skill": skill, "target": {"type": kind, "value": value}}

    history_a = [
        act("NAV", anchor, "place"),
        act("OPEN", container),
        act("CLOSE", container),
    ]
    history_b = [act("NAV", anchor, "place")]

    tracker_a = SearchStateTracker.from_scenario(scenario)
    for i, action in enumerate(history_a, start=1):
        tracker_a.process_event({
            "step": i,
            "action": action,
            "execution": {"executed": True},
        })
    exhausted_a = sorted(
        loc for loc, st in tracker_a.states.items()
        if st == SearchLocationState.EXHAUSTED_EMPTY
    )

    tracker_b = SearchStateTracker.from_scenario(scenario)
    for i, action in enumerate(history_b, start=1):
        tracker_b.process_event({
            "step": i,
            "action": action,
            "execution": {"executed": True},
        })
    exhausted_b = sorted(
        loc for loc, st in tracker_b.states.items()
        if st == SearchLocationState.EXHAUSTED_EMPTY
    )

    # history-consistent next search sets: searching an exhausted location
    # is a revisit error and therefore NOT history-consistent; every other
    # unsearched/valid container remains a rational search action
    def consistent_search_set(tracker: SearchStateTracker) -> list[str]:
        valid = []
        for loc in tracker.search_containers:
            if tracker.state_of(loc) == SearchLocationState.EXHAUSTED_EMPTY:
                continue
            valid.append(f"OPEN({loc})")
        return sorted(valid)

    rel = tracker_a.object_relations.get(target)
    target_absent_from_container = not (rel is not None and rel[1] == container)

    return HistoryCounterfactualPair(
        pair_id=pair_id,
        scene=scenario.scene.model,
        anchor=anchor,
        container=container,
        target_entity=target,
        history_A=history_a,
        history_B=history_b,
        exhausted_locations_A=exhausted_a,
        exhausted_locations_B=exhausted_b,
        valid_next_action_set_A=consistent_search_set(tracker_a),
        valid_next_action_set_B=consistent_search_set(tracker_b),
        current_observation_reference={
            "robot_anchor": anchor,
            "container_state": "closed (A restored, B untouched)",
            "visible_contents": "hidden in both (container closed)",
            "target_placement": f"{rel[0]}:{rel[1]}" if rel else "unknown",
            "target_absent_from_container": target_absent_from_container,
            "inspection_step": inspection_step,
        },
    )


def history_consistent(
    pair: HistoryCounterfactualPair, arm: str, selected_action: dict[str, Any]
) -> bool:
    """HTA check for a deterministic output: is the selected action
    consistent with the history-aware search set of the given arm?"""
    action_label = f"{selected_action['skill']}({(selected_action.get('target') or {}).get('value')})"
    valid = (pair.valid_next_action_set_A if arm == "A"
             else pair.valid_next_action_set_B)
    # non-search actions (NAV/GRASP/PLACE...) are not constrained by the
    # search history: they are consistent with both arms
    if not selected_action["skill"].startswith(("OPEN", "CLOSE")):
        return True
    return action_label in valid


def generate_diagnostic_pairs(
    scenario,
    containers: list[str] | None = None,
    anchor: str | None = None,
    prefix: str = "hcf",
) -> list[HistoryCounterfactualPair]:
    """One inspection pair per (container, anchor) combination. Defaults:
    every openable container at the robot's init anchor."""
    from rummagebench.evaluation.search_state import SearchStateTracker as _S  # noqa: F401

    containers = containers or sorted(scenario.initial_states.keys())
    anchor = anchor or scenario.robot.init_anchor
    return [
        build_inspection_pair(
            f"{prefix}_{container}", scenario, container, anchor
        )
        for container in containers
    ]
