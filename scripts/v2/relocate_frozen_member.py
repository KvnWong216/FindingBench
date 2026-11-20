"""Relocate an unchanged native checkpoint to byte-identical frozen sources.

The historical initialization report is retained verbatim. A new independent
native reload is mandatory before the relocated member can be published.
"""
from pathlib import Path
import copy
import hashlib
import json
import shutil

import yaml

from rummagebench.authoring.tasks.plan import canonical_json


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relocate(task, catalogue, output, runtime):
    catalogue,output,runtime=map(lambda p:Path(p).resolve(),(catalogue,output,runtime))
    original=Path(task['play_scenario']).resolve().parent
    recipe=Path(task['scenario']).resolve()
    report=json.loads((original/'validation.json').read_text())
    restored=json.loads((original/'restore_validation.json').read_text())
    manifest_path=original/'frozen_member.private.json'
    frozen=json.loads(manifest_path.read_text())
    if (report.get('status')!='initialization_validated' or not report.get('fresh_native_setup')
            or restored.get('status')!='restore_validated' or not restored.get('fresh_native_setup')
            or report.get('scenario_sha256')!=digest(recipe)
            or restored.get('scenario_sha256')!=digest(recipe)
            or restored.get('frozen_manifest_sha256')!=digest(manifest_path)
            or not report.get('runtime_sources')
            or report['runtime_sources']!=restored.get('runtime_sources')
            or report.get('runtime_environment',{})!=restored.get('runtime_environment',{})):
        raise ValueError('ORIGINAL_NATIVE_PAIR_NOT_VERIFIED')
    if (frozen.get('version')!=3 or frozen.get('status')!='frozen_member'
            or frozen.get('manifest_sha256')!=hashlib.sha256(canonical_json(
                {k:v for k,v in frozen.items() if k!='manifest_sha256'})).hexdigest()):
        raise ValueError('ORIGINAL_FROZEN_MANIFEST_NOT_VERIFIED')
    sources=report['runtime_sources']
    for name,expected in sources.items():
        path=(runtime/name).resolve()
        if not path.is_relative_to(runtime) or not path.is_file() or digest(path)!=expected:
            raise ValueError('RECONSTRUCTED_RUNTIME_SOURCE_CHANGED:'+name)
    moved=copy.deepcopy(frozen);dependencies={};relocations=[];seen=set()
    for name,expected in frozen['dependencies'].items():
        suffix=next((n for n,h in sources.items() if expected==h and
                     Path(name).as_posix().endswith('/'+n)),None)
        replacement=runtime/suffix if suffix else Path(name)
        if not replacement.is_file() or digest(replacement)!=expected:
            raise ValueError('NONIDENTICAL_FROZEN_DEPENDENCY:'+name)
        dependencies[str(replacement)]=expected
        if suffix:
            seen.add(suffix);relocations.append(dict(before=name,after=str(replacement),sha256=expected))
    if seen!=set(sources):raise ValueError('FROZEN_RUNTIME_DEPENDENCY_LEDGER_INCOMPLETE')
    moved['dependencies']=dependencies
    moved['manifest_sha256']=hashlib.sha256(canonical_json(
        {k:v for k,v in moved.items() if k!='manifest_sha256'})).hexdigest()
    tid=task['id'];dest=output/tid;target=catalogue/tid
    if dest.exists() or target.exists():raise ValueError('RELOCATION_DESTINATION_ALREADY_EXISTS')
    # Validate every immutable checkpoint artifact before creating any output.
    artifacts=[]
    for name,expected in frozen['artifacts'].items():
        path=(original.parent/name).resolve();relative=Path(name)
        if (not path.is_relative_to(original) or relative.parts[0]!=tid
                or not path.is_file() or digest(path)!=expected):
            raise ValueError('ORIGINAL_CHECKPOINT_ARTIFACT_CHANGED:'+name)
        artifacts.append((path,output/relative,expected))
    if frozen['state'] not in frozen['artifacts']:raise ValueError('FROZEN_STATE_NOT_HASHED')
    old_play=yaml.safe_load(Path(task['play_scenario']).read_text())
    if Path(old_play['scene']['frozen_member_manifest']).resolve()!=manifest_path:
        raise ValueError('ORIGINAL_PLAY_MANIFEST_MISMATCH')
    dest.mkdir(parents=True);target.mkdir(parents=True)
    for path,new_path,expected in artifacts:
        new_path.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,new_path)
        if digest(new_path)!=expected:raise ValueError('RELOCATED_CHECKPOINT_ARTIFACT_CHANGED')
    for name in ('restoration_baseline.private.json','initial_panorama.png','geometry.private.json'):
        if (original/name).exists() and not (dest/name).exists():shutil.copy2(original/name,dest/name)
    shutil.copy2(recipe,target/'scenario.yaml')
    shutil.copy2(original/'validation.json',dest/'validation.json')
    (dest/'frozen_member.private.json').write_text(json.dumps(moved,indent=2))
    play=copy.deepcopy(old_play);play['scene']['frozen_member_manifest']=str(dest/'frozen_member.private.json')
    (dest/'play.yaml').write_text(yaml.safe_dump(play,sort_keys=False))
    certificate=dict(method='byte_identical_frozen_dependency_relocation',
        original_scenario=str(recipe),original_manifest=str(manifest_path),
        original_manifest_sha256=digest(manifest_path),
        original_initialization_report=str(original/'validation.json'),
        original_initialization_report_sha256=digest(original/'validation.json'),
        original_reload_report=str(original/'restore_validation.json'),
        original_reload_report_sha256=digest(original/'restore_validation.json'),
        relocated_initialization_report_sha256=digest(dest/'validation.json'),
        relocated_manifest_sha256=digest(dest/'frozen_member.private.json'),
        relocations=relocations,checkpoint_artifacts_unchanged=True,
        scenario_bytes_unchanged=digest(target/'scenario.yaml')==digest(recipe),
        initialization_report_unchanged=digest(dest/'validation.json')==digest(original/'validation.json'),
        fresh_native_reload_required=True,full_skill_completion='unverified')
    (dest/'relocation.private.json').write_text(json.dumps(certificate,indent=2))
    return dict(task,scenario=str(target/'scenario.yaml'),runtime_root=str(runtime),
                runtime_environment=report.get('runtime_environment',{}))
