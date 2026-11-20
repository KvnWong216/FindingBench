import copy
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from relocate_frozen_member import digest, relocate
from rummagebench.authoring.tasks.plan import canonical_json


def fixture(tmp_path):
    old=tmp_path/'old/asset_X_01';old.mkdir(parents=True)
    runtime=tmp_path/'runtime';(runtime/'src').mkdir(parents=True)
    source=b'original native source';(runtime/'src/code.py').write_bytes(source)
    expected=hashlib.sha256(source).hexdigest()
    recipe=tmp_path/'scenario.yaml';recipe.write_text('id: asset_X_01\ninstruction: Find the soap.\n')
    state=old/'state.pt';state.write_bytes(b'unchanged checkpoint, never deserialize in authoring')
    frozen=dict(version=3,status='frozen_member',scenario_id='asset_X_01',
        scenario_sha256='original canonical scenario identity',
        dependencies={str(tmp_path/'mutable/src/code.py'):expected},
        state='asset_X_01/state.pt',artifacts={'asset_X_01/state.pt':digest(state)})
    frozen['manifest_sha256']=hashlib.sha256(canonical_json(frozen)).hexdigest()
    manifest=old/'frozen_member.private.json';manifest.write_text(json.dumps(frozen))
    (old/'play.yaml').write_text(yaml.safe_dump(dict(id='asset_X_01',scene=dict(frozen_member_manifest=str(manifest)))))
    report=dict(status='initialization_validated',fresh_native_setup=True,scenario_sha256=digest(recipe),
        runtime_sources={'src/code.py':expected})
    (old/'validation.json').write_text(json.dumps(report))
    report.update(status='restore_validated',frozen_manifest_sha256=digest(manifest))
    (old/'restore_validation.json').write_text(json.dumps(report))
    task=dict(id='asset_X_01',parent_task_id='X',scenario=str(recipe),play_scenario=str(old/'play.yaml'))
    return task,old,runtime,frozen


def test_relocation_keeps_checkpoint_and_initialization_proof_but_requires_new_reload(tmp_path):
    task,old,runtime,frozen=fixture(tmp_path)
    result=relocate(task,tmp_path/'catalogue',tmp_path/'output',runtime)
    dest=tmp_path/'output/asset_X_01'
    assert (dest/'state.pt').read_bytes()==(old/'state.pt').read_bytes()
    assert (dest/'validation.json').read_bytes()==(old/'validation.json').read_bytes()
    assert not (dest/'restore_validation.json').exists()
    assert Path(result['scenario']).read_bytes()==Path(task['scenario']).read_bytes()
    moved=json.loads((dest/'frozen_member.private.json').read_text())
    assert moved['artifacts']==frozen['artifacts'] and moved['scenario_sha256']==frozen['scenario_sha256']
    expected=copy.deepcopy(frozen);expected['dependencies']={str(runtime/'src/code.py'):next(iter(frozen['dependencies'].values()))}
    expected['manifest_sha256']=hashlib.sha256(canonical_json({k:v for k,v in expected.items() if k!='manifest_sha256'})).hexdigest()
    assert moved==expected
    certificate=json.loads((dest/'relocation.private.json').read_text())
    assert certificate['fresh_native_reload_required'] and certificate['initialization_report_unchanged']
    assert certificate['scenario_bytes_unchanged'] and certificate['checkpoint_artifacts_unchanged']
    with pytest.raises(ValueError,match='ALREADY_EXISTS'):relocate(task,tmp_path/'catalogue',tmp_path/'output',runtime)


@pytest.mark.parametrize('changed',['runtime','state','scene_dependency','restore_manifest'])
def test_relocation_rejects_nonidentical_sources_or_artifacts(tmp_path,changed):
    task,old,runtime,frozen=fixture(tmp_path)
    if changed=='runtime':(runtime/'src/code.py').write_text('different implementation')
    if changed=='state':(old/'state.pt').write_bytes(b'different checkpoint')
    if changed=='restore_manifest':
        p=old/'restore_validation.json';d=json.loads(p.read_text());d['frozen_manifest_sha256']='changed';p.write_text(json.dumps(d))
    if changed=='scene_dependency':
        path=tmp_path/'clearance.json';path.write_text('changed scene')
        frozen['dependencies'][str(path)]='0'*64
        frozen['manifest_sha256']=hashlib.sha256(canonical_json({k:v for k,v in frozen.items() if k!='manifest_sha256'})).hexdigest()
        manifest=old/'frozen_member.private.json';manifest.write_text(json.dumps(frozen))
        p=old/'restore_validation.json';d=json.loads(p.read_text());d['frozen_manifest_sha256']=digest(manifest);p.write_text(json.dumps(d))
    with pytest.raises(ValueError):relocate(task,tmp_path/'catalogue',tmp_path/'output',runtime)
    assert not (tmp_path/'output').exists()
