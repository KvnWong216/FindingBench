"""Synthetic fixtures ONLY: no simulator execution or generated dataset claims."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from rummagebench.authoring.tasks import household_dataset as hd
from rummagebench.authoring.tasks.household import family_members
from rummagebench.authoring.tasks.household_search import count_public_trace, reference_order, summarize_reference_search, POLICY_VERSION
from rummagebench.authoring.tasks.household_state import load_frozen_member, restore_frozen_member, scenario_state_identity
from rummagebench.authoring.tasks.plan import canonical_json, sha256_path, sha256_bytes
from rummagebench.core.scenario import load_scenario, SceneSpec

from test_infinigen_pipeline import usd_bundle, make_spec, attach_synthetic_probe_receipt, REPO


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value))


def signed(value, key):
    result = deepcopy(value)
    result.pop(key, None)
    result[key] = sha256_bytes(canonical_json(result))
    return result


@pytest.fixture
def compiled_family(usd_bundle, tmp_path, monkeypatch):
    from rummagebench.authoring.tasks.household_binding import bind_family
    from rummagebench.authoring.tasks.embodiment_slots import SlotEmbodimentOverlay, SlotInteraction, StartPose, Pose2
    from rummagebench.sim.omnigibson.infinigen_adapter import prepare_assembly, scene_slots_from_assembly
    scene, solve, assets, bindings = usd_bundle
    monkeypatch.setattr(hd,"validate_authoring_spec",lambda spec: None)  # synthetic fixture policy only
    assembly = tmp_path / "assembly"
    # Explicit mock admission ONLY for exercising subsequent contracts. It
    # cannot appear in the real CLI or production inventory counts.
    admission = {"receipt": str(tmp_path / "synthetic_admission_TEST_ONLY.json"),
                 "receipt_sha256": "unit_fixture_only", "measurement": {"candidate_mapping": {b["location_id"]:b["source_id"] for b in bindings}},
                 "solve_state_sha256": sha256_path(solve), "scope": "synthetic_unit_fixture_only"}
    monkeypatch.setattr(hd,"verify_generation_admission",lambda *args: admission)
    prepare_assembly(make_spec(), scene, solve, bindings, assets, assembly, generation_receipt=Path(admission["receipt"]))
    slots = scene_slots_from_assembly(assembly / "assembly.json")
    urdf = tmp_path / "unit_robot.urdf"
    urdf.write_text('<robot name="synthetic_unit_only"><link name="base_link"/><link name="right_eef_link"/></robot>')
    pose = Pose2((1., -2., .0053), (0., 0., 0., 1.))
    evidence = {"closed_visibility": {"visible_px": 0}, "closed_graspable": False, "open_visibility": {"ratio": .8}}
    overlay = SlotEmbodimentOverlay(slots.scene, "r1pro", str(urdf), sha256_path(urdf), "probed", "synthetic_unit_only",
        yaml.safe_load((REPO / "configs/tasks/robots/r1pro.yaml").read_text())["robot"], {},
        {"kitchen_0": [StartPose("common_start", pose)]},
        {s.slot_id: SlotInteraction(pose, True, True, True, evidence=evidence,
            reveal_anchor=Pose2((i + .2, -2., .0053), (0., 0., 0., 1.))) for i, s in enumerate(slots.slots)},
        notes=[f"assembly_manifest_sha256:{sha256_path(assembly / 'assembly.json')}"])
    attach_synthetic_probe_receipt(overlay, assembly / "assembly.json", assets, tmp_path)
    op = tmp_path / "unit_overlay.yaml"
    overlay.save(op)
    root = tmp_path / "family"
    family = bind_family(family_members(make_spec()), assembly / "assembly.json", op, root)
    return root, family, assembly


@pytest.fixture
def frozen_family(compiled_family, tmp_path):
    root, family, assembly = compiled_family
    frozen = tmp_path / "frozen_unit_only"
    records = {"unit_only": {"pose": [0, 0, 0]}}
    write_json(frozen / "background_records.json", records)
    members = []
    for episode in family["task_ids"]:
        scenario = load_scenario(root / "candidates" / f"{episode}.yaml")
        state_ref = f"states/{episode}.pt"
        state = frozen / state_ref
        state.parent.mkdir(parents=True, exist_ok=True)
        state.write_bytes(b"NOT A SIMULATOR SNAPSHOT; synthetic unit fixture only")
        member = {"version": 1, "status": "frozen_member", "scenario_id": episode,
                  "scenario_sha256": scenario_state_identity(scenario), "assembly_manifest_sha256": family["assembly_manifest_sha256"],
                  "runtime_topology": [], "parking": [100., 100., 10.], "state": state_ref,
                  "background_records": "background_records.json", "builder": {"placements": [{"verified": True}]},
                  "artifacts": {state_ref: sha256_path(state), "background_records.json": sha256_path(frozen / "background_records.json")}}
        member = signed(member, "manifest_sha256")
        ref = f"members/{episode}.json"
        write_json(frozen / ref, member)
        members.append({"scenario_id": episode, "background_match": True, "restore_repeats": 2, "member_manifest": ref})
    # Simulated protocol receipt exclusively inside this test's tmp_path.
    receipt = {"version": 1, "status": "pending_simulation_acceptance", "gpu_executed": True,
               "assembly_manifest_sha256": family["assembly_manifest_sha256"],
               "background_and_members": "frozen_and_restore_checked", "members": members,
               "artifacts": {str(p.relative_to(frozen)): sha256_path(p) for p in frozen.rglob("*") if p.is_file()}}
    write_json(frozen / "family_preflight.json", signed(receipt, "receipt_sha256"))
    return root, family, frozen


@pytest.fixture
def validation_request(frozen_family, tmp_path):
    root, family, frozen = frozen_family
    output = tmp_path / "validation_unit_only"
    report = hd.prepare_family_validation(root, frozen, output, "/unit/simulator/python", REPO / "scripts/tasks/household_pipeline.py")
    return root, family, frozen, output, report


def trace(skills):
    result = []
    for i, skill in enumerate(skills, 1):
        action = {"skill": skill}
        if skill == "MOVE": action["distance_cm"] = 10.
        elif skill == "TURN": action["angle_deg"] = 15.
        elif skill != "REPORT_DONE": action["point"] = {"frame_id": "unit_frame", "x": .5, "y": .5}
        result.append({"action": action, "feedback": "EXECUTED", "step": i, "status": "SUCCESS" if skill == "REPORT_DONE" else "RUNNING"})
    return result


def write_unit_member_results(validation_request):
    """Exercise validation logic with explicit test-only simulated measurements."""
    from rummagebench.authoring.tasks.search_certificate import SearchStructureCertificate, SimulatorEvidence
    from rummagebench.sim.omnigibson.infinigen_adapter import scene_slots_from_assembly
    from rummagebench.authoring.tasks.embodiment_slots import SlotEmbodimentOverlay
    root, family, frozen, output, _ = validation_request
    request = hd.load_validation_request(output / "request.json")
    _, specs, plans = hd.load_compiled_family(root)
    slots = scene_slots_from_assembly(Path(request["inputs"]["assembly"]))
    overlay = SlotEmbodimentOverlay.load(root / "overlay.json")
    slot_order = reference_order(next(iter(plans.values())).start["position"][:2],
                                  {sid: list(overlay.slots[sid].navigation_anchor.xy) for sid in plans[family["task_ids"][0]].candidate_slots})
    order = [specs[episode].target_location for sid in slot_order for episode in family["task_ids"] if plans[episode].target.slot_id == sid]
    for episode in family["task_ids"]:
        certificate = SearchStructureCertificate("easy", "in_container", 1, 2, 1, 1, 0, 0, "cabinet_interior", 0, 3,
                                                  candidate_slots=slot_order, rooms=["kitchen_0"])
        ev = SimulatorEvidence(builder_ok=True, placements_verified=True, settle_ok=True, start_visibility=0.,
            closed_visibility=0., pre_reveal_graspable=False, post_reveal_graspable=True, revealed_visibility=.8,
            reachable_candidate_slot_count=2, rearrangement_depth=0, oracle_solvable=True, replay_success=True,
            certified_execution_steps=4)
        certificate.attach_evidence(ev)
        index = order.index(specs[episode].target_location)
        actions = trace(["MOVE", "TURN"] + ["OBSERVE", "OPEN"] * (index + 1) + ["GRASP", "REPORT_DONE"])
        record = {"version": 1, "episode_id": episode, "request_sha256": request["request_sha256"],
                  "gpu_executed": True, "status": "certified", "error": None, "search_certificate": certificate.to_dict(),
                  "checks": {c: "passed" for c in request["inputs"]["required_checks"]},
                  "reference_search": {"policy": POLICY_VERSION, "gpu_executed": True, "candidate_order": order,
                    "visited_slots": order[:index + 1], "trace": actions, "status": "SUCCESS", "closed_observe_bypass": False},
                  "artifacts": {}}
        write_json(output / "results" / episode / "result.json", signed(record, "receipt_sha256"))
    return output / "request.json", output / "results"


def test_compile_reveal_anchors_are_complete_and_target_independent(compiled_family):
    root, family, _ = compiled_family
    scenarios = [load_scenario(root / "candidates" / f"{e}.yaml") for e in family["task_ids"]]
    assert scenarios[0].anchors == scenarios[1].anchors
    assert len([name for name in scenarios[0].anchors if name.endswith("__reveal")]) == 2
    assert scenarios[0].placements != scenarios[1].placements
    assert hd.load_compiled_family(root)[0]["split_assignment"]["split_seed"] == 17


@pytest.mark.parametrize("corruption", ["plan", "scenario", "extra", "spec", "overlay"])
def test_compiled_family_integrity_rejects_mutations(compiled_family, corruption):
    root, family, _ = compiled_family
    episode = family["task_ids"][0]
    if corruption == "extra": (root / "candidates" / "fb_extra.yaml").write_text("unit_only")
    else:
        path = {"plan": root / "plans" / f"{episode}.json", "scenario": root / "candidates" / f"{episode}.yaml",
                "spec": root / "specs" / f"{family['specification_ids'][0]}.json", "overlay": root / "overlay.json"}[corruption]
        path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError): hd.load_compiled_family(root)


def test_frozen_manifest_binding_and_topology_guard(frozen_family, monkeypatch):
    root, family, frozen = frozen_family
    episode = family["task_ids"][0]
    scenario = load_scenario(root / "candidates" / f"{episode}.yaml")
    path = frozen / "members" / f"{episode}.json"
    assert load_frozen_member(path, scenario)["scenario_id"] == episode
    scenario.scene.frozen_member_manifest = str(path)
    assert load_frozen_member(path, scenario)["scenario_id"] == episode
    backend = SimpleNamespace(_env=SimpleNamespace(scene=SimpleNamespace(objects=[SimpleNamespace(name="unexpected", links={})])))
    # If torch gets imported or a synthetic pickle gets read, this test fails.
    from rummagebench.sim.omnigibson import state_io
    monkeypatch.setattr(state_io, "load_snapshot", lambda p: pytest.fail("pickle read before checking serialized topology"))
    with pytest.raises(ValueError, match="TOPOLOGY_OR_ORDER"): restore_frozen_member(backend, path, scenario)
    scenario.instruction = "changed"
    with pytest.raises(ValueError, match="SCENARIO_OR_HASH"): load_frozen_member(path, scenario)


@pytest.mark.parametrize("artifact", ["state", "background_records"])
def test_frozen_state_files_protected(frozen_family, artifact):
    root, family, frozen = frozen_family
    e = family["task_ids"][0]
    path = frozen / "members" / f"{e}.json"
    member = json.loads(path.read_text())
    (frozen / member[artifact]).write_bytes(b"changed unit fixture")
    with pytest.raises(ValueError, match="FILE_MISSING_OR_CHANGED"):
        load_frozen_member(path, load_scenario(root / "candidates" / f"{e}.yaml"))


def test_request_is_pending_and_missing_members_block_publication(validation_request, tmp_path):
    _, family, _, output, report = validation_request
    assert report["counts"]["evaluation_ready_episodes"] == 0 and report["gpu_started"] is False
    request = hd.load_validation_request(output / "request.json")
    assert request["gpu_executed"] is False
    assert len(report["commands"]) == len(family["task_ids"])
    collected = hd.collect_family_validation(output / "request.json", output / "results")
    assert collected["status"] == "pending_simulation" and len(collected["pending_checks"]) == 2
    with pytest.raises(ValueError, match="PUBLICATION_BLOCKED"):
        hd.publish_family(output / "request.json", output / "results", tmp_path / "must_not_publish")
    assert not (tmp_path / "must_not_publish").exists()


@pytest.mark.parametrize("corruption", ["pending", "failed", "missing", "gate", "order", "extra", "hash"])
def test_family_validation_is_all_or_nothing(validation_request, corruption):
    request, results = write_unit_member_results(validation_request)
    episode = validation_request[1]["task_ids"][0]
    path = results / episode / "result.json"
    record = json.loads(path.read_text())
    if corruption == "extra": write_json(results / "fb_extra" / "result.json", record)
    elif corruption == "missing": path.unlink()
    else:
        if corruption in ("pending", "failed"): record["checks"]["door_sweep"] = corruption
        if corruption == "gate": record["search_certificate"]["evidence"]["closed_visibility"] = .1
        if corruption == "order": record["reference_search"]["candidate_order"].reverse()
        if corruption == "hash": record["gpu_executed"] = False
        write_json(path, signed(record, "receipt_sha256"))
    if corruption in ("extra", "hash"):
        with pytest.raises(ValueError): hd.collect_family_validation(request, results)
    else:
        collected = hd.collect_family_validation(request, results)
        assert collected["status"] != "validated" and collected["counts"]["evaluation_ready_episodes"] == 0


def test_publish_local_dataset_revalidates_and_protects_public_boundary(validation_request, tmp_path):
    request, results = write_unit_member_results(validation_request)
    output = tmp_path / "synthetic_dataset_TEST_ONLY"
    doc = hd.publish_family(request, results, output)
    assert doc["status"] == "validated"
    assert hd.load_dataset(output) == doc
    assert hd.publish_family(request, results, output) == doc
    public = json.loads((output / "public_tasks.json").read_text())
    assert all(set(task) == {"episode_id", "split", "instruction", "max_planning_steps"} for task in public["tasks"])
    assert "search_target" not in json.dumps(public) and "storage_0" not in json.dumps(public)
    for row in doc["episodes"]:
        assert load_scenario(output / row["scenario"]).scene.frozen_member_manifest is not None
    (output / "public_tasks.json").write_text('{}')
    with pytest.raises(ValueError): hd.load_dataset(output)


def test_reference_counts_actual_move_turn_observe_and_report_including_failures():
    rows = trace(["MOVE", "TURN", "OBSERVE", "OPEN", "GRASP", "PLACE", "GRASP", "REPORT_DONE"])
    rows[3]["feedback"] = "UNSAFE"
    metrics = count_public_trace(rows)
    assert (metrics["spatial"], metrics["operations"], metrics["steps"]) == (2, 6, 8)
    assert metrics["skill_calls"]["GRASP"] == 2
    rows[0]["action"] = {"skill": "NAV"}
    with pytest.raises(ValueError): count_public_trace(rows)


def test_reference_cost_is_asymmetric_measured_and_not_an_optimum():
    order = reference_order([0., 0.], {"a": [2., 0.], "b": [1., 0.]})
    assert order == ["b", "a"]
    runs = {k: {"policy": POLICY_VERSION, "gpu_executed": True, "candidate_order": order,
               "visited_slots": order[:i + 1], "status": "SUCCESS", "trace": trace(["MOVE", "TURN"] + ["OPEN", "OBSERVE"] * (i + 1) + ["GRASP", "REPORT_DONE"])}
            for i, k in enumerate(order)}
    intervals = {"spatial": {"low": [0, 8], "high": [8, None]}, "operations": {"low": [0, 6], "high": [6, None]}}
    doc = summarize_reference_search(runs, {"a": .8, "b": .2}, intervals)
    assert doc["expected"] == {"spatial": 2., "operations": pytest.approx(5.6)}
    assert doc["status"] == "measured_reference" and doc["real_scene_optimality_proved"] is False
    runs["a"]["closed_observe_bypass"] = True
    assert summarize_reference_search(runs, {"a": .8, "b": .2}, intervals)["status"] == "rejected"


def test_empty_inventory_does_not_count_configs_or_arbitrary_json(tmp_path):
    write_json(tmp_path / "fake.json", {"gpu_executed": True, "count": 10000})
    report = hd.pipeline_inventory(tmp_path)
    assert set(report["counts"].values()) == {0}
    assert report["gpu_started_by_this_command"] is False


def test_behavior_cannot_use_household_snapshot():
    with pytest.raises(ValueError): SceneSpec(model="behavior", frozen_member_manifest="unit_only.json")


@pytest.mark.parametrize("rejection", ["mismatch", "failed", "missing"])
def test_generation_rejection_cannot_be_bypassed_by_assembly(tmp_path, monkeypatch, rejection):
    from rummagebench.authoring.tasks import infinigen
    solve = tmp_path / "solve_state.json"
    solve.write_text('synthetic unit solve state only')
    receipt = tmp_path / "generation.json"
    receipt.write_text('synthetic unit receipt only')
    folder = tmp_path / "geometry"
    inspection = {"completed": [{"folder": str(folder), "measurement": {"proxy_filter_passed": rejection != "failed"},
                                  "artifacts": {str(folder / "solve_state.json"): sha256_path(solve) if rejection != "mismatch" else "changed"}}]}
    if rejection == "missing": inspection["completed"] = []
    monkeypatch.setattr(infinigen,"inspect_geometry_receipt",lambda *args: inspection)
    with pytest.raises(ValueError, match="ACCEPTED_COMPLETE_GENERATION"):
        hd.verify_generation_admission(make_spec(),receipt,solve)


def test_generation_admission_selects_matching_successful_retry(tmp_path,monkeypatch):
    from rummagebench.authoring.tasks import infinigen
    solve = tmp_path / "solve.json"; solve.write_text('unit fixture solve')
    receipt = tmp_path / "generation.json"; receipt.write_text('unit fixture invocation')
    old, last = tmp_path / "attempt_0/geometry", tmp_path / "attempt_1/geometry"
    inspection = {"completed": [
        {"folder": str(old), "measurement": {"proxy_filter_passed":False}, "artifacts": {str(old / "solve_state.json"):"failed-old-attempt"}},
        {"folder": str(last), "measurement": {"proxy_filter_passed":True}, "artifacts": {str(last / "solve_state.json"):sha256_path(solve)}}]}
    monkeypatch.setattr(infinigen,"inspect_geometry_receipt",lambda *args: inspection)
    admitted = hd.verify_generation_admission(make_spec(),receipt,solve)
    assert admitted["solve_state_sha256"] == sha256_path(solve)
    assert admitted["receipt_sha256"] == sha256_path(receipt)


def test_legacy_oracle_adaptation_does_not_change_public_completion(compiled_family, monkeypatch):
    from rummagebench.authoring.tasks.certify import TaskCertifier, EpisodeCertification
    from rummagebench.core import session as session_module
    from rummagebench.evaluation import certification as certificate_module
    root, family, _ = compiled_family
    scenario = load_scenario(root / "candidates" / f"{family['task_ids'][0]}.yaml")
    original = scenario.model_copy(deep=True)
    observed = []
    class UnitSession:
        _planning_step = 1
        def __init__(self, backend, doc):
            observed.append(doc)
        def reset(self): pass
        def act(self, action): pass
        def status(self): return SimpleNamespace(value="SUCCESS")
    result = SimpleNamespace(solvable=True, oracle_depth=1, oracle_plan=[{"skill":"GRASP","target":{"type":"entity","value":scenario.target.entity}}],
        plan_replay_status=None, plan_replay_steps=None, to_dict=lambda: {"solvable":True})
    monkeypatch.setattr(session_module, "BenchmarkSession", UnitSession)
    def certify(doc, session):
        assert doc.termination.succeed_when_holding_target is True
        assert scenario.termination.succeed_when_holding_target is False
        return result
    monkeypatch.setattr(certificate_module, "certify_episode", certify)
    certifier = TaskCertifier(None,None,None)
    monkeypatch.setattr(certifier, "_register_view_anchor", lambda *args: None)
    out = EpisodeCertification()
    certifier._oracle(SimpleNamespace(), scenario, out)
    assert scenario == original and observed[0] != scenario
    assert out.evidence.replay_success is True
    assert "public_REPORT_DONE" in out.oracle_certificate["scope"]
