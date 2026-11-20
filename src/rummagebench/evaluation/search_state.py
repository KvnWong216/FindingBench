"""Search-state tracking (§5): mechanically defined revisit errors.

A search location (container) moves through explicit states:

    UNSEEN               never interacted with
    PARTIALLY_OBSERVED   interacted (OPEN executed) but the interior has NOT
                         been covered by a delivered frame (remediation R09:
                         an OPEN only OPENS — it does not certify that the
                         agent saw anything)
    EXHAUSTED_EMPTY      the interior was covered by a delivered frame
                         (evidence recorded with its frame/event source) and
                         the ground truth places the target elsewhere
    TARGET_FOUND         the interior was covered by a delivered frame and
                         the ground truth places the target here

Remediation R09: state upgrades away from PARTIALLY_OBSERVED REQUIRE
observation evidence from delivered frames (`record_observation`, with the
source frame id / event id and the evidence source label). Ground truth
alone NEVER flips a state — "the target is not in the container" is a
scorer-side conclusion that may only be drawn once the delivered frames
could actually have shown the interior. Without such evidence the location
stays PARTIALLY_OBSERVED (unknown), never EXHAUSTED_EMPTY.

Relation changes (executed GRASP/PLACE — objects move) INVALIDATE
exhaustion conclusions: every EXHAUSTED_EMPTY reverts to
PARTIALLY_OBSERVED and the reversion is recorded. Exhaustion conclusions
survive only while the object relations that grounded them are unchanged.

A REVISIT ERROR is counted only when ALL hold:
    1. the location is EXHAUSTED_EMPTY (evidence-backed);
    2. the agent initiates a NEW search interaction at that location;
    3. no relation change since exhaustion invalidated the conclusion.

NAV to the same anchor twice is NOT a search interaction.

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


SEARCH_SKILLS = {"OPEN"}  # interactions that CAN reveal a location's contents
# skills whose committed execution moves objects and thereby invalidates
# exhaustion conclusions drawn from earlier frames
RELATION_CHANGING_SKILLS = {"GRASP", "PLACE"}


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
class ObservationRecord:
    """One accepted observation-evidence record with its source (R09)."""

    location: str
    step: int
    covered: bool
    frame_id: str | None = None
    event_id: str | None = None
    source: str = "delivered_frame"  # delivered_frame | synthetic_replay | ...

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": "location_observation",
            "location": self.location,
            "step": self.step,
            "covered": self.covered,
            "frame_id": self.frame_id,
            "event_id": self.event_id,
            "source": self.source,
        }


def _event_view(event: dict[str, Any]) -> tuple[str | None, str | None, bool, int]:
    """Normalize one event row from either session mode (R09): oracle rows
    carry action/validation/execution, visual rows carry parsed_action /
    raw_action / binding + executed."""
    if "action" in event:
        action = event.get("action") or {}
        return (
            action.get("skill"),
            (action.get("target") or {}).get("value"),
            bool((event.get("execution") or {}).get("executed", False)),
            int(event.get("step", 0) or 0),
        )
    parsed = event.get("parsed_action") or {}
    raw = event.get("raw_action")
    skill = parsed.get("skill") or (
        raw.get("skill") if isinstance(raw, dict) else None)
    target = event.get("selected_entity") or event.get("binding", {}).get(
        "selected_entity")
    return (
        skill,
        target,
        bool(event.get("executed", False)),
        int(event.get("planning_step", 0) or 0),
    )


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
    # delivered-frame observation evidence per location (R09): the last
    # accepted record with its frame/event source
    observations: dict[str, ObservationRecord] = field(default_factory=dict)
    # exhaustion conclusions reverted by relation changes
    invalidated_observations: list[dict[str, Any]] = field(default_factory=list)
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

    def _invalidate_exhaustions(self, step: int) -> None:
        """Relation change: exhaustion conclusions drawn from earlier frames
        no longer hold — revert them to PARTIALLY_OBSERVED (R09)."""
        for loc, st in list(self.states.items()):
            if st == SearchLocationState.EXHAUSTED_EMPTY:
                self.invalidated_observations.append({
                    "event": "exhaustion_invalidated",
                    "location": loc,
                    "step": step,
                    "first_exhausted_step": self.first_exhausted.pop(loc, None),
                })
                self.states[loc] = SearchLocationState.PARTIALLY_OBSERVED

    def process_event(self, event: dict[str, Any],
                      current_relations: frozenset | None = None) -> dict[str, Any] | None:
        """Feed one JSONL event; returns a revisit-event dict when a revisit
        error is detected, else None."""
        skill, target, executed, step = _event_view(event)

        relations = current_relations if current_relations is not None else frozenset()
        invalidated = False
        if current_relations is not None:
            invalidated = self._relations_changed(relations)
        if invalidated and executed:
            # objects moved: earlier exhaustion conclusions are stale
            self._invalidate_exhaustions(step)

        if skill not in SEARCH_SKILLS or target is None:
            return None
        if target not in self.search_containers:
            return None  # not a search location (v0: openable containers)

        self.search_interactions += 1
        state = self.state_of(target)

        # revisit error: interacting with an EXHAUSTED_EMPTY location, and no
        # relation change has invalidated that conclusion
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

        # R09: an executed OPEN only OPENS the location. Whether the agent
        # could actually SEE the interior is decided by delivered-frame
        # evidence (record_observation), never by this event alone and never
        # by ground truth.
        if state == SearchLocationState.UNSEEN:
            self.states[target] = SearchLocationState.PARTIALLY_OBSERVED
        return None

    def record_observation(self, location: str, step: int, covered: bool,
                           frame_id: str | None = None,
                           event_id: str | None = None,
                           source: str = "delivered_frame") -> dict[str, Any] | None:
        """Accept one observation-evidence record for a location (R09).

        ``covered`` asserts the delivered frame(s) actually showed the
        location's interior — the predefined v0 coverage condition. Only a
        covered observation may upgrade the state, and the upgrade uses the
        ground truth as the SCORER's knowledge (target present ->
        TARGET_FOUND, absent -> EXHAUSTED_EMPTY). Non-covered observations
        are recorded for audit but never upgrade anything; without evidence
        the location keeps the unknown PARTIALLY_OBSERVED state.
        """
        rec = ObservationRecord(location=location, step=step, covered=covered,
                                frame_id=frame_id, event_id=event_id,
                                source=source)
        state = self.state_of(location)
        if state in (SearchLocationState.EXHAUSTED_EMPTY,
                     SearchLocationState.TARGET_FOUND):
            # already concluded and not invalidated since: keep the record
            # that established the conclusion (the concluding evidence)
            return None
        self.observations[location] = rec
        if not covered:
            return None
        rel = self.object_relations.get(self.target_entity)
        if rel is not None and rel[1] == location:
            self.states[location] = SearchLocationState.TARGET_FOUND
            self.target_found_location = location
        else:
            self.states[location] = SearchLocationState.EXHAUSTED_EMPTY
            if location not in self.first_exhausted:
                self.first_exhausted[location] = step
        return rec.to_dict()

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
            "observations": {k: v.to_dict() for k, v in self.observations.items()},
            "invalidated_observations": self.invalidated_observations,
        }


def track_search_states(events: list[dict[str, Any]], scenario) -> SearchStateTracker:
    """Run the tracker over a full episode event stream (either session
    mode).

    Relation changes are derived from the stream itself: every EXECUTED
    GRASP/PLACE advances the relation epoch (the tracker invalidates stale
    exhaustion conclusions on change). Observation evidence is read from
    ``observed_evidence`` records the visual session emits for committed
    interactions (entity -> visible pixel count in the post-action delivered
    frame); a search location present with a positive pixel count counts as
    a covered observation, sourced by the event's own frame id.
    """
    tracker = SearchStateTracker.from_scenario(scenario)
    epoch: list[tuple[str, str]] = []
    for i, event in enumerate(events):
        skill, target, executed, step = _event_view(event)
        if executed and skill in RELATION_CHANGING_SKILLS and target:
            epoch.append((skill, target))
        tracker.process_event(event, current_relations=frozenset(epoch))
        if not executed or target is None:
            continue
        evidence = event.get("observed_evidence")
        if isinstance(evidence, dict) and target in evidence:
            count = evidence.get(target)
            try:
                covered = count is not None and int(count) > 0
            except (TypeError, ValueError):
                covered = False
            tracker.record_observation(
                target, step=step, covered=covered,
                frame_id=event.get("frame_id_after"),
                event_id=f"{event.get('event_type', 'action')}@{step}",
                source="delivered_frame",
            )
    return tracker
