"""Search-state tracking (§5): mechanically defined revisit errors.

A search location (container) moves through explicit states:

    UNSEEN               never interacted with
    PARTIALLY_OBSERVED   interacted but contents not fully verified (v0:
                         reserved for future sub-container reasoning)
    EXHAUSTED_EMPTY      observed sufficiently for the benchmark observation
                         protocol, target verified ABSENT, no hidden
                         sub-container remains unobserved
    TARGET_FOUND         the target was found at this location

For v0, OPEN-ing a location reveals its contents: if the target is inside
(ground truth from the scenario placements) the location becomes
TARGET_FOUND; otherwise EXHAUSTED_EMPTY (Beechwood containers have no
hidden sub-containers, so condition 3 of exhaustion is trivially met; the
class structure keeps the extension point).

A REVISIT ERROR is counted only when ALL hold:
    1. the location was EXHAUSTED_EMPTY;
    2. the agent initiates a NEW search interaction at that location
       (OPEN on the container — mechanically requires CLOSE first, since
       OPEN on an open container is INVALID_STATE);
    3. no world transition since exhaustion invalidated the conclusion
       (v0: object relations are static within an episode, so exhaustion
       stands; the tracker re-checks relations and would invalidate).

NAV to the same anchor twice is NOT a search interaction and is never a
revisit error by itself.

RER = exhausted_location_revisits / search_interactions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SearchLocationState(str, Enum):
    UNSEEN = "UNSEEN"
    PARTIALLY_OBSERVED = "PARTIALLY_OBSERVED"
    EXHAUSTED_EMPTY = "EXHAUSTED_EMPTY"
    TARGET_FOUND = "TARGET_FOUND"


SEARCH_SKILLS = {"OPEN"}  # interactions that reveal a location's contents


@dataclass
class RevisitEvent:
    location: str
    first_exhausted_step: int
    revisit_step: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": "exhausted_location_revisit",
            "location": self.location,
            "first_exhausted_step": self.first_exhausted_step,
            "revisit_step": self.revisit_step,
        }


@dataclass
class SearchStateTracker:
    """Post-hoc search-state analysis over an episode event stream."""

    target_entity: str
    # ground truth: entity -> (relation, receptacle) from the scenario
    object_relations: dict[str, tuple[str, str]]
    search_containers: list[str]  # locations that can be searched (openable)
    states: dict[str, SearchLocationState] = field(default_factory=dict)
    first_exhausted: dict[str, int] = field(default_factory=dict)
    exhausted_location_revisits: list[RevisitEvent] = field(default_factory=list)
    search_interactions: int = 0
    target_found_location: str | None = None
    # relations may change within an episode (GRASP/PLACE); a change at an
    # exhausted location invalidates the conclusion
    _relations_snapshot: frozenset | None = None

    @classmethod
    def from_scenario(cls, scenario) -> "SearchStateTracker":
        """Search locations = every container that can hold the target:
        initial-state keys (door state forced) + placement receptacles."""
        containers = sorted(
            set(scenario.initial_states.keys())
            | {p.receptacle for p in scenario.placements}
        )
        relations = {p.entity: (p.relation, p.receptacle)
                     for p in scenario.placements}
        return cls(
            target_entity=scenario.target.entity,
            object_relations=relations,
            search_containers=containers,
        )

    def state_of(self, location: str) -> SearchLocationState:
        return self.states.get(location, SearchLocationState.UNSEEN)

    def _relations_changed(self, current_relations: frozenset) -> bool:
        if self._relations_snapshot is None:
            self._relations_snapshot = current_relations
            return False
        if current_relations != self._relations_snapshot:
            self._relations_snapshot = current_relations
            return True
        return False

    def process_event(self, event: dict[str, Any],
                      current_relations: frozenset | None = None) -> dict[str, Any] | None:
        """Feed one JSONL event; returns a revisit-event dict when a revisit
        error is detected, else None."""
        action = event.get("action", {})
        skill = action.get("skill")
        target = (action.get("target") or {}).get("value")
        step = event.get("step", 0)
        executed = event.get("execution", {}).get("executed", False)

        if skill not in SEARCH_SKILLS or target is None:
            return None

        relations = current_relations if current_relations is not None else frozenset()
        invalidated = self._relations_changed(relations) if current_relations is not None else False

        if target not in self.search_containers:
            return None  # not a search location (v0: openable containers)

        self.search_interactions += 1
        state = self.state_of(target)

        # revisit error: interacting with an EXHAUSTED_EMPTY location, and no
        # world transition has invalidated that conclusion
        if state == SearchLocationState.EXHAUSTED_EMPTY and executed and not invalidated:
            revisit = RevisitEvent(
                location=target,
                first_exhausted_step=self.first_exhausted.get(target, -1),
                revisit_step=step,
            )
            self.exhausted_location_revisits.append(revisit)
            return revisit.to_dict()

        if not executed:
            return None  # failed attempts don't update the search state

        # the location was observed: target found or verified absent
        rel = self.object_relations.get(self.target_entity)
        if rel is not None and rel[1] == target:
            self.states[target] = SearchLocationState.TARGET_FOUND
            self.target_found_location = target
        else:
            # EXHAUSTED_EMPTY requires: (1) observed per the protocol (OPEN
            # reveals the container interior — v0 observation contract);
            # (2) target verified absent (ground truth above); (3) no hidden
            # sub-containers remain (v0: Beechwood containers are flat)
            self.states[target] = SearchLocationState.EXHAUSTED_EMPTY
            if target not in self.first_exhausted:
                self.first_exhausted[target] = step
        return None

    def revisit_error_rate(self) -> float | None:
        if self.search_interactions == 0:
            return None
        return round(len(self.exhausted_location_revisits) / self.search_interactions, 3)

    def summary(self) -> dict[str, Any]:
        return {
            "location_states": {k: v.value for k, v in self.states.items()},
            "search_interactions": self.search_interactions,
            "exhausted_location_revisits": len(self.exhausted_location_revisits),
            "revisit_error_rate": self.revisit_error_rate(),
            "revisit_events": [r.to_dict() for r in self.exhausted_location_revisits],
            "target_found_location": self.target_found_location,
        }


def track_search_states(events: list[dict[str, Any]], scenario) -> SearchStateTracker:
    """Run the tracker over a full episode event stream."""
    tracker = SearchStateTracker.from_scenario(scenario)
    for event in events:
        tracker.process_event(event)
    return tracker
