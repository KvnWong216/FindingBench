"""Probe diagnostics must report failure and close the backend reliably."""
import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
import pytest


@pytest.mark.parametrize("supported, expected", [(True, 0), (False, 3)])
@pytest.mark.parametrize("background", [None, "positive_inf", "nan", "foreground_inf"])
def test_probe_reports_and_closes(monkeypatch, capsys, supported, expected, background):
    if background in ("nan", "foreground_inf"):
        expected = 3
    scenario_module = types.ModuleType("rummagebench.core.scenario")
    scenario_module.load_scenario = lambda _: types.SimpleNamespace(
        robot=types.SimpleNamespace(obs_modalities=[], include_sensor_names=None))
    backend_module = types.ModuleType("rummagebench.sim.omnigibson.backend")
    closed = []
    validations = []

    class Backend:
        def __init__(self, seed):
            self._sim = types.SimpleNamespace(render=lambda: None)
            self._instance_labels = {"1": "object"} if supported else {}
        def setup(self, scenario):
            pass
        def capture_visual_frame(self):
            depth=np.ones((2, 2)) if supported else np.full((2, 2), np.nan)
            seg=np.ones((2, 2)) if supported else np.zeros((2, 2))
            if background:
                depth[0,0] = np.nan if background == "nan" else np.inf
                if background != "foreground_inf": seg[0,0] = 0
            return types.SimpleNamespace(
                depth=depth, instance_segmentation=seg,
                camera_intrinsics=np.eye(3), camera_extrinsics=np.eye(4),
                image_width=2, image_height=2, depth_convention="z_depth",
                meta={"bridge": "segmentation" if supported else "unsupported"})
        def validate_physics_state(self):
            validations.append(True)
        def close(self):
            closed.append(True)

    backend_module.OmniGibsonBackend = Backend
    monkeypatch.setitem(sys.modules, scenario_module.__name__, scenario_module)
    monkeypatch.setitem(sys.modules, backend_module.__name__, backend_module)
    monkeypatch.setattr(sys, "argv", ["probe_sensors", "--frames", "1", "--warmup", "0"])
    spec = importlib.util.spec_from_file_location(
        "probe_sensors_under_test", Path(__file__).parents[2] / "scripts/probe_sensors.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main() == expected
    assert closed == [True]
    assert validations == [True]
    assert ("frame 0: OK" if expected == 0 else "frame 0: BAD") in capsys.readouterr().out

    def invalid_world(self):
        raise RuntimeError("nonfinite world")
    with monkeypatch.context() as scope:
        scope.setattr(Backend, "validate_physics_state", invalid_world)
        assert module.main() == 3
        assert "CAPTURE_FAIL nonfinite world" in capsys.readouterr().out
    assert closed == [True, True]

    def fail_setup(self, scenario):
        raise RuntimeError("setup failure")
    monkeypatch.setattr(Backend, "setup", fail_setup)
    with pytest.raises(RuntimeError, match="setup failure"):
        module.main()
    assert closed == [True, True, True]


@pytest.mark.parametrize("report_ok, child_exit, expected", [
    (False, 0, 3), (True, 0, 0), (None, 0, 1), (True, -11, 139)])
def test_supervisor_rejects_native_false_success(monkeypatch, tmp_path, report_ok, child_exit, expected):
    import json
    spec = importlib.util.spec_from_file_location(
        "probe_supervisor_under_test", Path(__file__).parents[2] / "scripts/probe_sensors.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    def child(command, check):
        if report_ok is not None:
            Path(command[-1]).write_text(json.dumps({"ok": report_ok}))
        return types.SimpleNamespace(returncode=child_exit)
    monkeypatch.setattr(module.subprocess, "run", child)
    out = tmp_path / "report.json"
    assert module.supervise(["--modalities", "rgb", "--out", str(out)]) == expected
    assert out.exists() == (report_ok is not None and child_exit == 0)
