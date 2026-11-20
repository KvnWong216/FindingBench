"""Expose verified RAB resources as archives arrive, before the full install."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def provision(repo, run, destination):
    sys.path.insert(0, str(repo / 'src'))
    from robotagentbench.resources import read_manifest, _verify_file
    from robotagentbench.installation import original_library_alias, _canonical_source
    manifest = read_manifest(repo / 'resources/assets-0.1.7.json')
    destination.mkdir(parents=True, exist_ok=True)
    objects = destination / 'content'
    objects.mkdir(exist_ok=True)
    receipts = destination / 'verified'
    receipts.mkdir(exist_ok=True)
    for name, record in manifest['archives'].items():
        archive = run / 'downloads' / name
        receipt = receipts / record['sha256']
        if receipt.exists() or not archive.is_file():
            continue
        _verify_file(archive, record)
        seen = set()
        with tarfile.open(archive, 'r|gz') as source:
            for member in source:
                expected = record['members'].get(member.name)
                if (expected is None or not member.isfile() or member.name in seen
                        or member.size != expected['bytes']):
                    raise ValueError('Unexpected RAB archive member: ' + member.name)
                target = objects / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                temp = target.with_suffix(target.suffix + '.partial')
                with source.extractfile(member) as stream, temp.open('wb') as output:
                    shutil.copyfileobj(stream, output, 1024**2)
                _verify_file(temp, expected)
                os.replace(temp, target)
                seen.add(member.name)
        if seen != record['members'].keys():
            raise ValueError('Missing RAB archive members: ' + name)
        receipt.write_text(name)
    index_file = objects / manifest['index']
    if not index_file.exists():
        return {}, len(list(receipts.iterdir()))
    index = json.loads(index_file.read_text())
    location_map = {}
    paths = {}
    for location, sha in index['locations'].items():
        path = objects / index['objects'][sha]['path']
        if not path.is_file():
            continue
        targets = [location]
        alias = original_library_alias(location)
        if alias:
            targets.append(alias)
        if location.startswith('profiles/room/'):
            mapped = _canonical_source(location.removeprefix('profiles/room/'))
            targets.append('room/' + mapped)
        if location.startswith('libraries/generic_current/'):
            mapped = _canonical_source(location.removeprefix('libraries/generic_current/'))
            targets.append('room/' + mapped)
        if location.startswith('libraries/current_assets/'):
            targets.append('room/' + location.removeprefix('libraries/current_assets/'))
        for relative in targets:
            # Profile resources take precedence over library aliases.
            priority = (2 if location.startswith('profiles/room/') else 1)
            if relative not in paths or priority > paths[relative][0]:
                paths[relative] = priority, path
        location_map[location] = str(destination / location)
    for relative, (_, path) in paths.items():
        target = destination / relative
        if target.exists() and target.samefile(path):
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix(target.suffix + '.link')
        temp.unlink(missing_ok=True)
        os.link(path, temp)
        os.replace(temp, target)
    return location_map, len(list(receipts.iterdir()))


def namespace(root):
    from rab_assets import AssetNamespace
    old = '/home/lingjun/robot_agent_arena'
    current = '/workspace/lukun/robot_agent_arena'
    mounts = {
        '/mnt/agent-bench-data/lingjun': root / 'libraries/original_absolute/mnt/agent-bench-data/lingjun',
        '/job': root,
        'D:/00_Sale/TableBasse/TableBasse/PBR_Textures':
            root / 'room/assets/house_scene_l1/assets_living_room/coffee_table/PBR_Textures',
    }
    for prefix in (old, current):
        mounts[prefix + '/api-protocol'] = root / 'libraries/current_assets'
        mounts[prefix + '/api-protocol/assets'] = root / 'room/assets'
        mounts[prefix + '/api-protocol-st18-20261013-frozen4'] = root / 'libraries/frozen_assets'
        mounts[prefix + '/short_tasks/agent_probe_v1/deployment_generic_20261010/source'] = root / 'room'
        for path in ('layout_task_activation_v1', 'openend/tasks/all_objects_integration'):
            mounts[prefix + '/' + path] = root / 'room' / path
    return AssetNamespace(mounts)
