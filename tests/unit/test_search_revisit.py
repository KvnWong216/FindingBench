"""§11 E/G tests: search-revisit semantics (RER) and the
history-counterfactual diagnostic (HTA)."""

from pathlib import Path

from rummagebench.core.scenario import load_scenario
from rummagebench.diagnostics.history_counterfactual import (
    build_inspection_pair,
    generate_diagnostic_pairs,
    history_consistent,
)
from rummagebench.evaluation.search_state import (
    SearchLocationState,
    SearchStateTracker,
    track_search_states,
)

FIXTURE = Path(__file__).parent / "fixtures" / "mini.yaml"


def _scenario():
    return load_scenario(FIXTURE)


def _ev(step, skill, value, executed=True):
    return {
        "step": step,
        "action": {"skill": skill, "target": {"type": "entity", "value": value}},
        "execution": {"executed": executed},
    }


# ---------------------------------------------------------------------------
# E. revisit semantics
# ---------------------------------------------------------------------------


def test_exhausted_reopen_is_revisit_error():
    """OPEN A -> observe empty (target elsewhere) -> CLOSE -> OPEN A again
    => the second OPEN is an exhausted-location revisit error."""
    scenario = _scenario()
    tracker = SearchStateTracker.from_scenario(scenario)
    # ground truth: knife is in cabinet_B, so cabinet_A opens empty
    events = [
        _ev(1, "NAV", "kitchen"),
        _ev(2, "OPEN", "cabinet_A"),
        _ev(3, "CLOSE", "cabinet_A"),
        _ev(4, "OPEN", "cabinet_A"),  # revisit of an exhausted location
    ]
    revisit_dicts = [tracker.process_event(e) for e in events]
    revisit_dicts = [r for r in revisit_dicts if r is not None]
    assert tracker.state_of("cabinet_A") == SearchLocationState.EXHAUSTED_EMPTY
    assert len(tracker.exhausted_location_revisits) == 1
    assert revisit_dicts[0]["location"] == "cabinet_A"
    assert revisit_dicts[0]["first_exhausted_step"] == 2
    assert revisit_dicts[0]["revisit_step"] == 4
    assert tracker.revisit_error_rate() == round(1 / 2, 3)  # 2 search interactions


def test_open_target_container_is_found_not_exhausted():
    scenario = _scenario()
    tracker = SearchStateTracker.from_scenario(scenario)
    events = [
        _ev(1, "NAV", "kitchen"),
        _ev(2, "OPEN", "cabinet_B"),  # knife IS in cabinet_B
    ]
    for e in events:
        tracker.process_event(e)
    assert tracker.state_of("cabinet_B") == SearchLocationState.TARGET_FOUND
    assert tracker.exhausted_location_revisits == []
    # re-OPENing the TARGET location is not an exhausted revisit (it holds
    # the target — the tracker state is TARGET_FOUND)
    tracker.process_event(_ev(3, "CLOSE", "cabinet_B"))
    tracker.process_event(_ev(4, "OPEN", "cabinet_B"))
    assert tracker.exhausted_location_revisits == []


def test_repeated_nav_is_not_a_revisit_error():
    """NAV to the same anchor twice is locomotion, not a search interaction:
    never counted as a revisit error."""
    scenario = _scenario()
    tracker = SearchStateTracker.from_scenario(scenario)
    events = [
        {"step": 1, "action": {"skill": "NAV",
                               "target": {"type": "place", "value": "kitchen"}},
         "execution": {"executed": True}},
        {"step": 2, "action": {"skill": "NAV",
                               "target": {"type": "place", "value": "kitchen"}},
         "execution": {"executed": True}},
    ]
    for e in events:
        assert tracker.process_event(e) is None
    assert tracker.search_interactions == 0
    assert tracker.exhausted_location_revisits == []


def test_failed_open_does_not_update_search_state():
    scenario = _scenario()
    tracker = SearchStateTracker.from_scenario(scenario)
    tracker.process_event(_ev(1, "OPEN", "cabinet_A", executed=False))
    assert tracker.state_of("cabinet_A") == SearchLocationState.UNSEEN
    assert tracker.search_interactions == 1  # attempt counted


def test_rer_via_track_search_states():
    scenario = _scenario()
    events = [
        _ev(1, "NAV", "kitchen"),
        _ev(2, "OPEN", "cabinet_A"),
        _ev(3, "CLOSE", "cabinet_A"),
        _ev(4, "OPEN", "cabinet_A"),
        _ev(5, "OPEN", "cabinet_B"),
    ]
    tracker = track_search_states(events, scenario)
    # 3 search interactions (2xA cabinet_A, 1x cabinet_B), 1 revisit
    assert tracker.search_interactions == 3
    assert len(tracker.exhausted_location_revisits) == 1
    assert tracker.revisit_error_rate() == round(1 / 3, 3)


# ---------------------------------------------------------------------------
# G. history-counterfactual
# ---------------------------------------------------------------------------


def test_history_pair_current_state_matches():
    """Mechanical preconditions: same anchor, container closed in both arms,
    target placement equal, histories differ."""
    scenario = _scenario()
    pair = build_inspection_pair("hcf_cabinet_A", scenario, "cabinet_A", "kitchen")
    assert pair.history_A != pair.history_B
    assert pair.history_A[0] == pair.history_B[0]  # same NAV
    assert pair.anchor == "kitchen"
    # container closed in both arms: A restored it, B never touched it
    opens_a = [a for a in pair.history_A if a["skill"] == "OPEN"]
    closes_a = [a for a in pair.history_A if a["skill"] == "CLOSE"]
    assert len(opens_a) == len(closes_a) == 1
    opens_b = [a for a in pair.history_B if a["skill"] == "OPEN"]
    assert opens_b == []
    assert pair.current_observation_reference["container_state"].startswith("closed")


def test_history_pair_rational_search_sets_differ():
    """A: cabinet_A exhausted -> searching it again is NOT history-consistent.
    B: cabinet_A unseen -> searching it IS valid. The sets differ."""
    scenario = _scenario()
    pair = build_inspection_pair("hcf_cabinet_A", scenario, "cabinet_A", "kitchen")
    assert "cabinet_A" in pair.exhausted_locations_A
    assert "cabinet_A" not in pair.exhausted_locations_B
    assert f"OPEN(cabinet_A)" not in pair.valid_next_action_set_A
    assert f"OPEN(cabinet_A)" in pair.valid_next_action_set_B


def test_history_pair_target_found_does_not_exhaust():
    """If the TARGET is in the inspected container, the container becomes
    TARGET_FOUND (not exhausted) and the history difference vanishes for the
    search-decision purpose — the diagnostic only applies to empty arms."""
    scenario = _scenario()
    # knife IS in cabinet_B per mini.yaml ground truth
    pair = build_inspection_pair("hcf_cabinet_B", scenario, "cabinet_B", "kitchen")
    assert "cabinet_B" not in pair.exhausted_locations_A
    # both arms consider searching cabinet_B valid (it holds the target)
    assert "OPEN(cabinet_B)" in pair.valid_next_action_set_A
    assert "OPEN(cabinet_B)" in pair.valid_next_action_set_B


def test_hta_consistency_check():
    scenario = _scenario()
    pair = build_inspection_pair("hcf_cabinet_A", scenario, "cabinet_A", "kitchen")
    # arm A: re-searching the exhausted container is history-INconsistent
    assert not history_consistent(
        pair, "A", {"skill": "OPEN", "target": {"type": "entity", "value": "cabinet_A"}}
    )
    # arm B: searching the unseen container is consistent
    assert history_consistent(
        pair, "B", {"skill": "OPEN", "target": {"type": "entity", "value": "cabinet_A"}}
    )
    # non-search actions are unconstrained by the search history
    assert history_consistent(
        pair, "A", {"skill": "NAV", "target": {"type": "place", "value": "kitchen"}}
    )


def test_generate_diagnostic_pairs_covers_containers():
    scenario = _scenario()
    pairs = generate_diagnostic_pairs(scenario)
    assert len(pairs) == len(scenario.initial_states)
    for pair in pairs:
        assert pair.container in scenario.initial_states
        assert pair.history_A and pair.history_B
