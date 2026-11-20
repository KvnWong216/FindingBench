"""Model-agnostic policy adapter (no simulator)."""
from rummagebench.agents.policy_adapter import PolicyAgent, load_agent, parse_action


def test_parse_action_variants():
    assert parse_action({"skill": "REPORT_DONE"}) == {"skill": "REPORT_DONE"}
    assert parse_action('{"skill": "MOVE", "distance_cm": 50}') == {"skill": "MOVE", "distance_cm": 50}
    assert parse_action('I will turn.\n```json\n{"skill": "TURN", "angle_deg": 30}\n```') == {
        "skill": "TURN", "angle_deg": 30}
    assert parse_action("no idea") == {}
    assert parse_action(None) is None


def test_agent_passes_only_public_observation_and_history():
    seen = []

    def policy(obs, history):
        seen.append((obs, history))
        return {"skill": "VIEW", "yaw": 0}

    agent = PolicyAgent(policy)
    agent.reset("x")
    agent.act({"instruction": "x", "feedback": None})
    agent.act({"instruction": "x", "feedback": {"code": "EXECUTED"}})
    assert seen[0][1] == []
    assert seen[1][1] == [{"action": {"skill": "VIEW", "yaw": 0}, "feedback": {"code": "EXECUTED"}}]


def test_load_reference_policy():
    agent = load_agent("rummagebench.agents.policy_adapter:make_turn_in_place_policy")
    agent.reset("x")
    actions = [agent.act({"feedback": None}) for _ in range(5)]
    assert [a["skill"] for a in actions] == ["VIEW"] * 4 + ["REPORT_DONE"]
