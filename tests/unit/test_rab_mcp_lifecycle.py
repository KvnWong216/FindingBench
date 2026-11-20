"""Unit tests for the RAB MCP lifecycle and protocol contract (findingbench_rab_pinhole_v1)."""
from __future__ import annotations

import base64
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from rummagebench.adapters import rab_mcp_server


def _sample_png_bytes(color=(12, 34, 56), size=(64, 48)) -> bytes:
    arr = np.full((size[1], size[0], 3), color, dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


FAKE_WORKER_SCRIPT = """
import argparse, base64, io, json, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit
from uuid import uuid4
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, required=True)
parser.add_argument("--task", required=True)
parser.add_argument("--run-output", required=True)
parser.add_argument("--mode", default="normal")
args = parser.parse_args()

if args.mode == "hang_startup":
    time.sleep(30)
    sys.exit(0)
if args.mode == "crash_startup":
    sys.exit(2)

def make_png_b64(step, w=64, h=48):
    arr = np.full((h, w, 3), (step * 17) % 255, dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

STATE = {
    "frame_id": "frame_0000",
    "step": 0,
    "horizon": 120,
    "done": False,
    "status": "RUNNING",
    "feedback": None,
    "inspection": None,
}

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def reply(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if urlsplit(self.path).path in ("/observation", "/state"):
            obs = {
                "instruction": f"Test instruction for {args.task}",
                "frame_id": STATE["frame_id"],
                "image_png_b64": make_png_b64(STATE["step"]),
                "planning_step": STATE["step"],
                "max_planning_steps": STATE["horizon"],
                "skill_library": ["MOVE_TO","TURN","OPEN","CLOSE","GRASP","PLACE","VIEW","CHECK","REPORT_DONE"],
                "feedback": STATE["feedback"],
                "inspection_view": STATE["inspection"],
                "private_secret_do_not_leak": "SECRET_ORACLE_PATH",
            }
            return self.reply({"ready": True, "observation": obs, "result": None, "error": None})
        return self.reply({"error": "not found"}, 404)

    def do_POST(self):
        size = int(self.headers.get("Content-Length", 0))
        act = json.loads(self.rfile.read(size))
        if STATE["done"]:
            return self.reply({
                "id": uuid4().hex,
                "feedback": {"code": "INVALID_ACTION", "message": "Terminated"},
                "terminal": True,
                "planning_step": STATE["step"],
                "episode_status": STATE["status"],
            })
        skill = act.get("skill")
        STATE["inspection"] = None
        code = "EXECUTED"
        extras = {}
        if skill == "VIEW":
            pass
        elif skill == "REPORT_DONE":
            STATE["done"] = True
            STATE["status"] = "FAIL_FALSE_COMPLETION"
            extras = {"success": False, "restoration": 0.91}
        elif skill in ("CHECK", "OPEN", "CLOSE", "GRASP", "PLACE", "MOVE_TO"):
            pt = act.get("point") or {}
            if pt.get("frame_id") != STATE["frame_id"]:
                code = "INVALID_ACTION"
            elif skill == "CHECK":
                STATE["inspection"] = {
                    "frame_id": "check_frame_999",
                    "image_png_b64": make_png_b64(200, 72, 45),
                    "width": 720,
                    "height": 450,
                    "source": "isolated_object_multiview",
                }
        else:
            code = "INVALID_ACTION"
        STATE["step"] += 1
        STATE["frame_id"] = f"frame_{STATE['step']:04d}"
        STATE["feedback"] = {"code": code, "message": code}
        return self.reply({
            "id": uuid4().hex,
            "feedback": STATE["feedback"],
            "terminal": STATE["done"],
            "planning_step": STATE["step"],
            "episode_status": STATE["status"],
            **extras,
        })

ThreadingHTTPServer(("127.0.0.1", args.port), H).serve_forever()
"""


def test_rab_task_resolution_blocks_arbitrary_paths():
    root = Path(__file__).resolve().parents[2]
    release = root / "runs/rab_dense_serial_20261119"
    for bad in ("../F251", "/etc/passwd", "F251.json", "scenarios/F251/scenario.yaml", ""):
        with pytest.raises(ValueError):
            rab_mcp_server._resolve_rab_task(release, bad)
    resolved = rab_mcp_server._resolve_rab_task(release, "F251")
    assert resolved["task"]["id"] == "rab_F251"
    assert len(resolved["manifest_sha256"]) == 64


def test_rab_worker_handle_lifecycle_and_protocol(tmp_path):
    root = Path(__file__).resolve().parents[2]
    release = root / "runs/rab_dense_serial_20261119"
    fake_script = tmp_path / "fake_worker.py"
    fake_script.write_text(FAKE_WORKER_SCRIPT)

    handle = rab_mcp_server.RabWorkerHandle(
        release_dir=release,
        out_root=tmp_path / "out",
        gpu_id=0,
        startup_timeout_s=10.0,
        request_timeout_s=5.0,
        worker_command=[sys.executable, "-u", str(fake_script)],
    )
    try:
        # 1. Seed rejection
        with pytest.raises(ValueError, match="seed"):
            handle.reset("F251", seed=42)

        # 2. Reset F251
        reset_payload, png0 = handle.reset("F251")
        assert reset_payload["ok"] is True
        assert reset_payload["task_id"] == "rab_F251"
        assert reset_payload["episode_status"] == "RUNNING"
        ep1 = reset_payload["episode_id"]
        obs0 = reset_payload["reset"]
        assert "private_secret_do_not_leak" not in obs0
        assert "image_png_b64" not in obs0
        assert obs0["planning_step"] == 0
        assert png0.startswith(b"\x89PNG")
        tmpdir1 = handle.tmpdir
        pid1 = handle.process.pid
        assert tmpdir1.exists()

        # 3. Observe does not advance step or frame_id
        obs_payload, png0_again, check_png = handle.observe()
        assert obs_payload["observation"]["frame_id"] == obs0["frame_id"]
        assert obs_payload["observation"]["planning_step"] == 0
        assert png0_again == png0
        assert check_png is None

        # 4. CHECK produces auxiliary PNG and non-selectable frame_id
        act_check, scene_png1, check_png1 = handle.act(
            {"skill": "CHECK", "point": {"frame_id": obs0["frame_id"], "x": 0.4, "y": 0.5}}
        )
        assert act_check["feedback"]["code"] == "EXECUTED"
        assert act_check["planning_step"] == 1
        assert check_png1 is not None and check_png1.startswith(b"\x89PNG")
        check_fid = act_check["observation"]["inspection_view"]["frame_id"]
        assert act_check["observation"]["inspection_view"]["point_selectable"] is False

        # 5. Using CHECK frame_id or stale frame_id for a point action is rejected as INVALID_ACTION and charges 1 step
        act_stale, _, _ = handle.act(
            {"skill": "OPEN", "point": {"frame_id": check_fid, "x": 0.5, "y": 0.5}}
        )
        assert act_stale["feedback"]["code"] == "INVALID_ACTION"
        assert act_stale["planning_step"] == 2

        # 6. REPORT_DONE terminates and strips private success/restoration from public payload
        act_done, _, _ = handle.act({"skill": "REPORT_DONE"})
        assert act_done["terminal"] is True
        assert act_done["episode_status"] == "FAIL_FALSE_COMPLETION"
        assert act_done["planning_step"] == 3
        assert "success" not in act_done and "restoration" not in act_done

        # 7. Post-termination action is rejected without advancing planning_step
        act_after, _, _ = handle.act({"skill": "VIEW", "yaw": 0, "pitch": 0, "zoom": 1})
        assert act_after["feedback"]["code"] == "INVALID_ACTION"
        assert act_after["planning_step"] == 3
        assert act_after["terminal"] is True

        # 8. Same-ID and cross-ID reset cleans up old process & tmpdir and starts fresh
        reset2, _ = handle.reset("F262")
        assert reset2["task_id"] == "rab_F262"
        assert reset2["episode_id"] != ep1
        assert reset2["reset"]["planning_step"] == 0
        assert not tmpdir1.exists()
        assert handle.process.pid != pid1
    finally:
        handle.stop()


def test_rab_worker_startup_failure_and_hang_cleanup(tmp_path):
    root = Path(__file__).resolve().parents[2]
    release = root / "runs/rab_dense_serial_20261119"
    fake_script = tmp_path / "fake_worker.py"
    fake_script.write_text(FAKE_WORKER_SCRIPT)

    # Crash on startup
    handle_crash = rab_mcp_server.RabWorkerHandle(
        release_dir=release,
        out_root=tmp_path / "out_crash",
        gpu_id=0,
        startup_timeout_s=5.0,
        worker_command=[sys.executable, "-u", str(fake_script), "--mode", "crash_startup"],
    )
    with pytest.raises(RuntimeError, match="INFRA_WORKER_EXITED_DURING_STARTUP"):
        handle_crash.reset("F251")
    assert handle_crash.tmpdir is None

    # Hang on startup -> timeout -> killed and tmpdir cleaned
    handle_hang = rab_mcp_server.RabWorkerHandle(
        release_dir=release,
        out_root=tmp_path / "out_hang",
        gpu_id=0,
        startup_timeout_s=1.2,
        worker_command=[sys.executable, "-u", str(fake_script), "--mode", "hang_startup"],
    )
    with pytest.raises(RuntimeError, match="INFRA_WORKER_STARTUP_TIMEOUT"):
        handle_hang.reset("F251")
    assert handle_hang.process is None
    assert handle_hang.tmpdir is None
