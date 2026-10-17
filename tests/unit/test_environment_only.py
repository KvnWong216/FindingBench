"""Synthetic interface tests only; no GPU or physical readiness claims."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import pytest


@pytest.mark.parametrize('target_present',[True,False])
def test_environment_reset_and_checkpoint_never_submit_actions(tmp_path,monkeypatch,target_present):
    repo=Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(repo/'scripts/tasks'))
    spec=importlib.util.spec_from_file_location('unit_environment_cli',repo/'scripts/tasks/behavior_pipeline.py')
    cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
    from rummagebench.core import visual_session
    from rummagebench.sim.omnigibson import state_io,probe_support
    from rummagebench.authoring.layout import indoor_realization
    calls=[]
    class Session:
        def __init__(self,*a):pass
        def reset(self):calls.append('reset')
        def step(self,*a):pytest.fail('environment production must not submit any action')
        def _capture_current(self):return NS(rgb=np.zeros((4,4,3),dtype=np.uint8))
    monkeypatch.setattr(visual_session,'VisualProtocolSession',Session)
    monkeypatch.setattr(state_io,'save_snapshot',lambda path,*a,**k:path.write_bytes(b'synthetic state'))
    monkeypatch.setattr(state_io,'load_snapshot',lambda path:('unit-state',{'format':'unit'}))
    monkeypatch.setattr(probe_support,'relation_holds',lambda *a:target_present)
    def verify(*a,**k):
        assert k['require_matched_background'] is False
        calls.append('physical_background')
    monkeypatch.setattr(indoor_realization,'verify_after_target',verify)
    backend=NS(capture_observe_state=lambda:'unit-state',restore_observe_state=lambda s:calls.append('restore'),
        validate_physics_state=lambda:calls.append('physics'),_validate_spawn_clearance=lambda:calls.append('spawn'))
    scenario=NS(target=NS(entity='unit-target'),placements=[NS(receptacle='unit-container')])
    (tmp_path/'background_realization.json').write_text('{}')
    if not target_present:
        with pytest.raises(ValueError,match='TARGET_RELATION_LOST'):cli.check_environment(backend,scenario,{'manifest_sha256':'unit'},tmp_path)
        assert not (tmp_path/'environment_start.png').exists()
    else:
        result=cli.check_environment(backend,scenario,{'manifest_sha256':'unit'},tmp_path)
        assert result['status']=='environment_ready' and result['policy_executed'] is False
        assert calls==['reset','physical_background','restore','physics','spawn']
