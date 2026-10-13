"""CPU audit regressions; all fixture assets/evidence are synthetic tests only."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from rummagebench.authoring.level1.robot_inventory import discover_installed_robots
from rummagebench.authoring.level1.robot_registry import REQUIRED_CAPABILITIES, RobotProfile, production_ready, runtime_protocol_hash
from rummagebench.authoring.level1.snapshot import write_environment_snapshot
from rummagebench.authoring.tasks import gates
from rummagebench.authoring.tasks.plan import sha256_path
from rummagebench.authoring.tasks.split import EpisodeRecord, Split, leakage_violations

REPO = Path(__file__).resolve().parents[2]


def test_unreachable_code_is_a_supported_rejection():
    assert gates.Rejection(gates.START_UNREACHABLE, 'no public route').to_dict()['code'] == gates.START_UNREACHABLE


def test_duplicate_record_ids_are_detected_for_list_and_iterator():
    r = EpisodeRecord('id', 'scene', 'room', 'kitchen', 'family', 'bowl', 'bowl/m', [], 'drawer', 'in_container')
    split = Split('held_out_scene', ['id'], [], [])
    for records in ([r, r], iter([r, r])):
        assert 'duplicate task ids' in leakage_violations(split, records)
    split.dev += ['missing', 'id']
    violations = leakage_violations(split, [r])
    assert any('unknown task ids' in v for v in violations)
    assert 'duplicate ids in dev' in violations


def test_frozen_bundle_cannot_be_partially_overwritten(tmp_path):
    out = tmp_path / 'frozen'
    write_environment_snapshot(out, {'a.json': {'a': 1}, 'b.json': {'b': 2}})
    before = (out / 'a.json').read_bytes()
    with pytest.raises(FileExistsError):
        write_environment_snapshot(out, {'a.json': {'a': 3}, 'c.json': {'c': 4}})
    assert (out / 'a.json').read_bytes() == before and not (out / 'c.json').exists()
    write_environment_snapshot(out, {'a.json': {'a': 1}, 'b.json': {'b': 2}})


def test_freeze_rejects_invalid_artifacts_before_publishing(tmp_path):
    for name in ('../outside.json', '/absolute.json'):
        with pytest.raises(ValueError):
            write_environment_snapshot(tmp_path / 'frozen', {name: {}})
    with pytest.raises(ValueError):
        write_environment_snapshot(tmp_path / 'frozen', {'good.json': {}, 'bad.json': {'nan': float('nan')}})
    assert not (tmp_path / 'frozen').exists()


def test_status_label_alone_never_verifies_robot_and_pending_is_complete(tmp_path):
    assert not RobotProfile('fixture', 'Robot', status='gpu_verified').verified()
    path = tmp_path / 'pool.yaml'
    path.write_text(yaml.safe_dump({'robots': {'fixture': {'status': 'gpu_verified'}}}))
    ready, pending = production_ready(path, robot_count=1)
    assert not ready and pending == ['fixture']


def test_robot_evidence_requires_all_checks_and_current_urdf(tmp_path):
    urdf = tmp_path / 'fixture.urdf'; urdf.write_text('synthetic unit test only')
    evidence = tmp_path / 'evidence.json'
    doc = {'robot_id': 'fixture', 'camera_sensor': 'test_head', 'protocol_sha256': runtime_protocol_hash(),
           'urdf_sha256': sha256_path(urdf), 'capabilities': {c: True for c in REQUIRED_CAPABILITIES}}
    evidence.write_text(json.dumps(doc))
    profile = RobotProfile('fixture', 'Robot', status='gpu_verified', camera_sensor='test_head',
                           urdf_source=str(urdf), evidence_path=str(evidence), evidence_sha256=sha256_path(evidence))
    assert profile.verified()
    for cap in REQUIRED_CAPABILITIES:
        changed = deepcopy(doc); del changed['capabilities'][cap]
        evidence.write_text(json.dumps(changed)); profile.evidence_sha256 = sha256_path(evidence)
        assert not profile.verified()
    evidence.write_text(json.dumps(doc)); profile.evidence_sha256 = sha256_path(evidence)
    urdf.write_text('different synthetic model')
    assert not profile.verified()


def test_static_robot_inventory_uses_yaml_and_usda_without_og_import(tmp_path):
    root = tmp_path / 'models' / 'r1pro'; (root / 'usd').mkdir(parents=True)
    (root / 'r1pro.yaml').write_text(yaml.safe_dump({'manipulation': {'arm_joint_names': {'right': ['j']},
                                                                   'eef_link_names': {'right': 'eef'}},
                                                  'locomotion': {'base_joint_names': ['x', 'y', 'yaw']}}))
    (root / 'usd' / 'fixture.usda').write_text('synthetic unit test asset only')
    inventory = discover_installed_robots(tmp_path)
    assert inventory['r1pro']['status'] == 'candidate'
    assert inventory['fetch']['status'] == 'unavailable'
    assert inventory['r1pro']['camera_verification'] == 'pending'


def test_cpu_pipeline_reports_missing_sources_and_zero_generated_families():
    result = subprocess.run([sys.executable, str(REPO / 'scripts/tasks/household_pipeline.py'), 'cpu'],
                            cwd=REPO, capture_output=True, text=True)
    # Missing historical cloud sources remain visible. The explicitly
    # requested Infinigen route can proceed, with physical acceptance pending.
    doc = json.loads(result.stdout)
    if doc['missing_baseline_files']:
        assert result.returncode == 0 and doc['status'] == 'pending_simulation'
    assert doc['counts']['generated_family_candidates_this_run'] == 0
    assert doc['counts']['simulation_validated_families_this_run'] == 0


def test_legacy_promoter_rejects_household_family_before_release_mutation(tmp_path, monkeypatch):
    import importlib
    monkeypatch.syspath_prepend(str(REPO / 'scripts/tasks'))
    promoter = importlib.import_module('promote_tasks')
    from test_task_planner_easy import build_ctx
    from rummagebench.authoring.tasks.planner import plan_episode
    from rummagebench.authoring.tasks.search_certificate import SimulatorEvidence
    ctx = build_ctx(wide_room=False)
    outcome = plan_episode(ctx, 7, 0)
    cert = outcome.certificate
    cert.attach_evidence(SimulatorEvidence(builder_ok=True, placements_verified=True, settle_ok=True,
                         start_visibility=0., closed_visibility=0., pre_reveal_graspable=False,
                         revealed_visibility=1., post_reveal_graspable=True, oracle_solvable=True,
                         replay_success=True, certified_execution_steps=8))
    assert gates.certify(cert, ctx.tier) == []
    src = tmp_path / 'build'; (src / 'plans').mkdir(parents=True); (src / 'candidates').mkdir()
    candidate = src / 'candidates/id.yaml'; candidate.write_text('scene:\n  model: synthetic_unit_test_only\n')
    plan = outcome.plan.to_dict(); plan['provenance']['family_id'] = 'synthetic_family'
    (src / 'plans/id.json').write_text(json.dumps(plan))
    records = {'id': {'status': 'certified', 'search_certificate': cert.to_dict(),
                      'provenance': {'certifier': {'scenario_sha256': sha256_path(candidate)}}}}
    with pytest.raises(SystemExit, match='single-episode certificate'):
        promoter.preflight_episodes(src, {'id': 'dev'}, records, 'easy_v1')
    assert set(src.iterdir()) == {src / 'plans', src / 'candidates'}
