"""Checks for route geometry and the new release boundary (no Isaac import)."""
import importlib.util
from pathlib import Path
import sys

import numpy as np

scripts=Path(__file__).resolve().parents[2]/'scripts'
if str(scripts) not in sys.path:
    sys.path.insert(0,str(scripts))
from rab_skills import Skills
from rab_spectator_data import current, digest, save


def test_route_can_go_around_a_wall():
    class Motion:
        @staticmethod
        def clear_base(p,q):
            for v in np.linspace(p,q,35):
                if abs(v[0]-1)<.25 and abs(v[1])<.55:
                    return False
            return True
    start=np.array([0.,0.,0.])
    end=np.array([2.,0.,0.])
    assert not Motion.clear_base(start,end)
    path=Skills.route(Motion(),start,[end])
    assert len(path)>=2 and np.array_equal(path[-1],end)
    assert all(Motion.clear_base(a,b) for a,b in zip([start]+path,path))


def test_select_active_overlay_and_reject_changed_usd(tmp_path):
    import pytest
    root=tmp_path/'release'
    save(root/'combined_manifest.json',{})
    base=tmp_path/'scene.usda'
    base.write_text('base')
    old=dict(id='rab_F001',scene=str(base),scene_sha256=digest(base))
    episode=tmp_path/'original.json'
    receipt=tmp_path/'original_receipt.json'
    save(episode,old)
    save(receipt,{'old':True})
    task=dict(id='rab_F001',episode_path=str(episode),physics_receipt=str(receipt))
    overlays=tmp_path/'overlays'
    got,_,member,rev=current(root,task,overlays)
    assert got==old and not rev
    p=member/'revisions'/'one'
    p.mkdir(parents=True)
    scene=p/'scene.usda'
    scene.write_text('changed')
    new=dict(old,scene=str(scene),scene_sha256=digest(scene),base_scene_sha256=digest(base))
    save(p/'episode.json',new)
    save(p/'physics.json',{'fresh':True})
    save(member/'current.json',dict(revision='one',episode_sha256=digest(p/'episode.json'),
                                   physics_sha256=digest(p/'physics.json')))
    assert current(root,task,overlays)[:2] == (new,{'fresh':True})
    scene.write_text('tampered')
    with pytest.raises(ValueError,match='does not match'):
        current(root,task,overlays)
