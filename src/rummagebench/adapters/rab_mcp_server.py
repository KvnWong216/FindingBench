"""RAB MCP adapter (stdio) for findingbench_rab_pinhole_v1.

Forwards MCP tool calls (reset_episode, observe, act, episode_status) to a
managed localhost RAB HTTP skill worker process (`scripts/spectator.py --skills`).

Cardinal invariants:
- The stdio server process NEVER imports Kit, OmniGibson, or pxr.
- Stdout of the stdio server is exclusively MCP JSON-RPC.
- Each `reset_episode(scenario_id)` terminates any existing worker, allocates
  a fresh temporary directory (`TMPDIR`), resolves `scenario_id` strictly
  against the frozen RAB manifest (`combined_manifest.json`), and launches a
  fresh native worker process bound to an empty editor-revisions overlay so
  authoritative scenes are never contaminated by interactive edits.
- Private evaluation signals (`success`, `restoration`, `private_error`,
  `measurements`, `physx_steps`, scene paths, object IDs) are written to the
  evaluator-private episode run directory and NEVER returned across the MCP
  public boundary.
- Primary scene RGB and optional `CHECK` multiview cards are returned as native
  MCP `ImageContent` (`image/png`) blocks alongside structured JSON metadata.
"""

from __future__ import annotations

import atexit
import base64
import hashlib
import json
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = "findingbench_rab_pinhole_v1"
PUBLIC_OBSERVATION_KEYS = (
    "instruction",
    "frame_id",
    "planning_step",
    "max_planning_steps",
    "skill_library",
    "embodiment_spec",
    "feedback",
)


def _repo_root() -> Path:
    root = os.environ.get("RUMMAGEBENCH_ROOT")
    return Path(root).resolve() if root else Path(__file__).resolve().parents[3]


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _resolve_rab_task(release_dir: Path, query: str) -> dict[str, Any]:
    """Resolve a task ID strictly from the frozen RAB release manifest.
    Refuses path traversal or arbitrary file paths."""
    q = str(query).strip()
    if not q or "/" in q or "\\" in q or ".." in q or q.endswith(".json") or q.endswith(".yaml"):
        raise ValueError(f"Invalid RAB task identifier: {query!r}")
    sys.path.insert(0, str(_repo_root() / "scripts"))
    from rab_spectator_data import digest, load, select

    manifest_path = release_dir / "combined_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"RAB release manifest not found: {manifest_path}")
    empty_edits = release_dir / "_empty_edits_probe"
    task = select(release_dir, q, base=empty_edits)
    episode = load(task["episode_path"])
    if digest(episode["scene"]) != episode["scene_sha256"]:
        raise ValueError(f"RAB source USD SHA256 mismatch for {task['id']}")
    return {
        "task": task,
        "episode": episode,
        "manifest_path": str(manifest_path),
        "manifest_sha256": digest(manifest_path),
        "episode_sha256": digest(task["episode_path"]),
        "physics_receipt_sha256": digest(task["physics_receipt"]),
    }


def _strip_public_observation(obs: dict[str, Any]) -> tuple[dict[str, Any], bytes, dict[str, Any] | None, bytes | None]:
    """Separate public observation metadata from binary PNG bytes (scene + optional CHECK)."""
    if not isinstance(obs, dict) or not obs.get("image_png_b64"):
        raise RuntimeError("Worker returned observation without primary RGB image")
    scene_png = base64.b64decode(obs["image_png_b64"])
    public_meta: dict[str, Any] = {}
    for k in PUBLIC_OBSERVATION_KEYS:
        if k == "embodiment_spec" and obs.get(k) is None:
            continue
        public_meta[k] = obs.get(k)
    public_meta["protocol"] = PROTOCOL_VERSION

    insp = obs.get("inspection_view")
    check_meta = None
    check_png = None
    if isinstance(insp, dict) and insp.get("image_png_b64"):
        check_png = base64.b64decode(insp["image_png_b64"])
        check_meta = {
            "frame_id": insp.get("frame_id"),
            "width": insp.get("width", 720),
            "height": insp.get("height", 450),
            "source": insp.get("source", "isolated_object_multiview"),
            "point_selectable": False,
        }
    public_meta["inspection_view"] = check_meta
    return public_meta, scene_png, check_meta, check_png


class RabWorkerHandle:
    """Manages one dedicated RAB HTTP skill worker subprocess for one episode."""

    def __init__(
        self,
        *,
        release_dir: Path,
        out_root: Path,
        gpu_id: int = 0,
        startup_timeout_s: float = 300.0,
        request_timeout_s: float | None = None,
        worker_command: list[str] | None = None,
    ) -> None:
        self.release_dir = Path(release_dir).resolve()
        self.out_root = Path(out_root).resolve()
        self.gpu_id = int(gpu_id)
        self.startup_timeout_s = float(startup_timeout_s)
        if request_timeout_s is None:
            is_replay = os.environ.get("FINDINGBENCH_RAB_MOTION", "discrete").strip().lower() == "replay"
            request_timeout_s = 450.0 if is_replay else 220.0
        self.request_timeout_s = float(request_timeout_s)
        self.worker_command = worker_command

        self.episode_uuid: str | None = None
        self.task_id: str | None = None
        self.parent_task_id: str | None = None
        self.port: int | None = None
        self.process: subprocess.Popen | None = None
        self.tmpdir: Path | None = None
        self.episode_dir: Path | None = None
        self.log_file = None
        self.episode_status: str = "NO_EPISODE"
        self.planning_step: int = 0
        self.max_planning_steps: int = 120
        self.last_public_obs: dict[str, Any] | None = None
        self.last_scene_png: bytes | None = None
        self.last_check_png: bytes | None = None

    def stop(self) -> None:
        proc = self.process
        self.process = None
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=10)
            except Exception:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=5)
                except Exception:
                    pass
        if self.log_file is not None:
            try:
                self.log_file.close()
            except Exception:
                pass
            self.log_file = None
        if self.tmpdir is not None and self.tmpdir.exists():
            shutil.rmtree(self.tmpdir, ignore_errors=True)
            self.tmpdir = None

    def _http_get(self, route: str, timeout: float | None = None) -> dict[str, Any]:
        url = f"http://127.0.0.1:{self.port}{route}"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout or self.request_timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _http_post(self, route: str, payload: dict[str, Any], timeout: float | None = None) -> tuple[int, dict[str, Any]]:
        url = f"http://127.0.0.1:{self.port}{route}"
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.request_timeout_s) as resp:
                return resp.status, json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8", errors="replace")
            try:
                return e.code, json.loads(raw)
            except Exception:
                return e.code, {"error": raw}

    def reset(self, scenario_id: str, seed: int | None = None) -> tuple[dict[str, Any], bytes]:
        if seed is not None:
            raise ValueError(
                "RAB native PhysX worker uses deterministic initial USD snapshot poses and does not accept a runtime random seed parameter; omit `seed`."
            )
        self.stop()
        resolved = _resolve_rab_task(self.release_dir, scenario_id)
        task = resolved["task"]
        episode = resolved["episode"]

        self.episode_uuid = uuid4().hex
        self.task_id = task["id"]
        self.parent_task_id = task["parent_task_id"]
        self.port = _free_local_port()

        ts = time.strftime("%Y%m%d_%H%M%S")
        self.episode_dir = self.out_root / f"{ts}_{self.task_id}_{self.episode_uuid[:8]}"
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        skill_output_dir = self.episode_dir / "worker_trace"
        empty_edits_dir = self.episode_dir / "empty_rab_edits"
        empty_edits_dir.mkdir(parents=True, exist_ok=True)

        self.tmpdir = Path(tempfile.mkdtemp(prefix=f"rab_mcp_{self.task_id}_"))

        private_manifest = {
            "schema": "findingbench.rab_mcp_episode.v1",
            "episode_id": self.episode_uuid,
            "task_id": self.task_id,
            "parent_task_id": self.parent_task_id,
            "layout_id": episode["layout_id"],
            "protocol": PROTOCOL_VERSION,
            "release_manifest": resolved["manifest_path"],
            "release_manifest_sha256": resolved["manifest_sha256"],
            "episode_path": task["episode_path"],
            "episode_sha256": resolved["episode_sha256"],
            "scene_path": episode["scene"],
            "scene_sha256": episode["scene_sha256"],
            "physics_receipt_path": task["physics_receipt"],
            "physics_receipt_sha256": resolved["physics_receipt_sha256"],
            "executor": "rab_physx_pinhole_v1",
            "robot_embodiment": os.environ.get("FINDINGBENCH_RAB_ROBOT", "unitree_g1"),
            "embodiment_prior": os.environ.get("FINDINGBENCH_RAB_EMBODIMENT_PRIOR", "text"),
            "ik_backend": os.environ.get("FINDINGBENCH_RAB_IK_BACKEND", "auto"),
            "feedback_mode": os.environ.get("FINDINGBENCH_RAB_FEEDBACK_MODE", "four_class_v1"),
            "rab_skills_sha256": _digest(_repo_root() / "scripts/rab_skills.py"),
            "rab_edit_sha256": _digest(_repo_root() / "scripts/rab_edit.py"),
            "rab_embodiment_sha256": _digest(_repo_root() / "scripts/rab_embodiment.py"),
            "rab_curobo_sha256": _digest(_repo_root() / "scripts/rab_curobo.py"),
            "horizon": 120,
            "determinism_policy": "frozen_usd_scene_and_physx_60hz_cpu_dynamics_no_runtime_seed",
            "gpu_id": self.gpu_id,
            "started_at": time.time(),
        }
        (self.episode_dir / "private_manifest.json").write_text(
            json.dumps(private_manifest, indent=2, ensure_ascii=False) + "\n"
        )

        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        env["RUMMAGEBENCH_MCP_GPU"] = str(self.gpu_id)
        env["OMNI_KIT_ACCEPT_EULA"] = "YES"
        env["PXR_WORK_THREAD_LIMIT"] = "4"
        env["TMPDIR"] = str(self.tmpdir)
        env["TEMP"] = str(self.tmpdir)
        env["TMP"] = str(self.tmpdir)
        env["PYTHONPATH"] = f"{_repo_root() / 'src'}:{_repo_root() / 'scripts'}"

        if self.worker_command is not None:
            cmd = [
                *self.worker_command,
                "--port",
                str(self.port),
                "--task",
                self.task_id,
                "--run-output",
                str(skill_output_dir),
            ]
        else:
            cmd = [
                sys.executable,
                "-u",
                str(_repo_root() / "scripts/spectator.py"),
                "--skills",
                "--task",
                self.task_id,
                "--asset-run",
                str(self.release_dir),
                "--rab-edits",
                str(empty_edits_dir),
                "--host",
                "127.0.0.1",
                "--port",
                str(self.port),
                "--run-output",
                str(skill_output_dir),
            ]

        self.log_file = (self.episode_dir / "worker.log").open("w", encoding="utf-8")
        t0 = time.monotonic()
        self.process = subprocess.Popen(
            cmd,
            cwd=str(_repo_root()),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        # Wait for HTTP readiness
        deadline = time.monotonic() + self.startup_timeout_s
        last_err = None
        raw_obs = None
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                self.stop()
                raise RuntimeError(
                    f"INFRA_WORKER_EXITED_DURING_STARTUP (rc={self.process.returncode if self.process else 'unknown'}); see {self.episode_dir / 'worker.log'}"
                )
            try:
                data = self._http_get("/observation", timeout=5.0)
                if data.get("error"):
                    self.stop()
                    raise RuntimeError(f"INFRA_WORKER_RUNTIME_ERROR: {data['error']}")
                if data.get("ready") and isinstance(data.get("observation"), dict):
                    raw_obs = data["observation"]
                    break
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
                last_err = str(e)
            time.sleep(0.35)

        if raw_obs is None:
            self.stop()
            raise RuntimeError(
                f"INFRA_WORKER_STARTUP_TIMEOUT after {self.startup_timeout_s:.1f}s (last_err={last_err}); see {self.episode_dir / 'worker.log'}"
            )

        load_s = time.monotonic() - t0
        public_meta, scene_png, _, _ = _strip_public_observation(raw_obs)
        self.episode_status = "RUNNING"
        self.planning_step = int(public_meta["planning_step"])
        self.max_planning_steps = int(public_meta["max_planning_steps"])
        self.last_public_obs = public_meta
        self.last_scene_png = scene_png
        self.last_check_png = None

        reset_payload = {
            "ok": True,
            "episode_id": self.episode_uuid,
            "task_id": self.task_id,
            "episode_status": self.episode_status,
            "reset": public_meta,
        }
        self._append_private_event(
            {
                "event": "reset_episode",
                "episode_id": self.episode_uuid,
                "task_id": self.task_id,
                "load_wall_s": round(load_s, 4),
                "frame_id": public_meta["frame_id"],
                "scene_png_sha256": hashlib.sha256(scene_png).hexdigest(),
            }
        )
        return reset_payload, scene_png

    def _check_alive(self) -> None:
        if self.process is None or self.episode_uuid is None:
            raise RuntimeError("NO_ACTIVE_EPISODE: call reset_episode first")
        if self.process.poll() is not None:
            rc = self.process.returncode
            self.stop()
            raise RuntimeError(f"INFRA_WORKER_CRASHED (rc={rc}); see {self.episode_dir / 'worker.log'}")

    def observe(self) -> tuple[dict[str, Any], bytes, bytes | None]:
        if self.episode_uuid is not None and self.episode_status != "RUNNING" and self.last_public_obs is not None and self.last_scene_png is not None:
            payload = {
                "ok": True,
                "episode_id": self.episode_uuid,
                "task_id": self.task_id,
                "episode_status": self.episode_status,
                "observation": self.last_public_obs,
            }
            return payload, self.last_scene_png, self.last_check_png
        self._check_alive()
        data = self._http_get("/observation")
        if data.get("error"):
            raise RuntimeError(f"INFRA_WORKER_RUNTIME_ERROR: {data['error']}")
        if not data.get("ready") or not isinstance(data.get("observation"), dict):
            raise RuntimeError("INFRA_WORKER_NOT_READY")
        public_meta, scene_png, _, check_png = _strip_public_observation(data["observation"])
        self.last_public_obs = public_meta
        self.last_scene_png = scene_png
        self.last_check_png = check_png
        payload = {
            "ok": True,
            "episode_id": self.episode_uuid,
            "task_id": self.task_id,
            "episode_status": self.episode_status,
            "observation": public_meta,
        }
        return payload, scene_png, check_png

    def act(self, action: dict[str, Any]) -> tuple[dict[str, Any], bytes, bytes | None]:
        if self.episode_uuid is not None and self.episode_status != "RUNNING" and self.last_public_obs is not None and self.last_scene_png is not None:
            if not isinstance(action, dict):
                raise ValueError("action must be a JSON object")
            feedback = {
                "code": "INVALID_ACTION",
                "message": "The episode has already terminated; call reset_episode to start a new episode.",
            }
            payload = {
                "ok": True,
                "episode_id": self.episode_uuid,
                "task_id": self.task_id,
                "episode_status": self.episode_status,
                "terminal": True,
                "planning_step": self.planning_step,
                "feedback": feedback,
                "observation": self.last_public_obs,
            }
            self._append_private_event(
                {
                    "event": "post_termination_rejected",
                    "planning_step": self.planning_step,
                    "raw_action": action,
                    "episode_status": self.episode_status,
                }
            )
            return payload, self.last_scene_png, None

        self._check_alive()
        if not isinstance(action, dict):
            raise ValueError("action must be a JSON object")

        t0 = time.monotonic()
        code, res = self._http_post("/action", action)
        step_wall_s = time.monotonic() - t0
        if code == 504:
            self.stop()
            raise RuntimeError("INFRA_ACTION_TIMEOUT: worker exceeded action deadline")
        if code != 200 or "feedback" not in res:
            raise RuntimeError(f"INFRA_WORKER_HTTP_{code}: {res.get('error', res)}")

        obs_resp = self._http_get("/observation")
        if not obs_resp.get("ready") or not isinstance(obs_resp.get("observation"), dict):
            raise RuntimeError("INFRA_OBSERVATION_MISSING_AFTER_ACTION")

        public_meta, scene_png, _, check_png = _strip_public_observation(obs_resp["observation"])
        self.planning_step = int(public_meta["planning_step"])
        self.max_planning_steps = int(public_meta["max_planning_steps"])

        # Determine canonical public episode_status without leaking private success/restoration fields
        if "episode_status" in res:
            self.episode_status = str(res["episode_status"])
        elif res.get("terminal"):
            if action.get("skill") == "REPORT_DONE":
                self.episode_status = "SUCCESS" if res.get("success") else "FAIL_FALSE_COMPLETION"
            else:
                self.episode_status = "FAIL_MAX_STEPS"
        else:
            self.episode_status = "RUNNING"

        self.last_public_obs = public_meta
        self.last_scene_png = scene_png
        self.last_check_png = check_png

        public_result = {
            "ok": True,
            "episode_id": self.episode_uuid,
            "task_id": self.task_id,
            "episode_status": self.episode_status,
            "terminal": self.episode_status != "RUNNING",
            "planning_step": self.planning_step,
            "feedback": res["feedback"],
            "observation": public_meta,
        }
        self._append_private_event(
            {
                "event": "act",
                "planning_step": self.planning_step,
                "raw_action": action,
                "public_feedback": res["feedback"],
                "episode_status": self.episode_status,
                "terminal": self.episode_status != "RUNNING",
                "private_success": res.get("success"),
                "private_restoration": res.get("restoration"),
                "step_wall_s": round(step_wall_s, 4),
                "frame_id": public_meta["frame_id"],
                "scene_png_sha256": hashlib.sha256(scene_png).hexdigest(),
                "check_png_sha256": hashlib.sha256(check_png).hexdigest() if check_png else None,
            }
        )
        if self.episode_status != "RUNNING" and os.environ.get("FINDINGBENCH_RAB_STOP_ON_TERMINAL", "").lower() in ("1", "true", "yes"):
            self.stop()
        return public_result, scene_png, check_png

    def status(self) -> dict[str, Any]:
        if self.episode_uuid is None:
            return {"ok": True, "status": "NO_EPISODE"}
        if self.episode_status != "RUNNING":
            return {
                "ok": True,
                "episode_id": self.episode_uuid,
                "task_id": self.task_id,
                "status": self.episode_status,
                "terminal": True,
                "planning_step": self.planning_step,
                "max_planning_steps": self.max_planning_steps,
                "protocol": PROTOCOL_VERSION,
            }
        if self.process is None:
            return {"ok": True, "status": "NO_EPISODE"}
        self._check_alive()
        return {
            "ok": True,
            "episode_id": self.episode_uuid,
            "task_id": self.task_id,
            "status": self.episode_status,
            "terminal": False,
            "planning_step": self.planning_step,
            "max_planning_steps": self.max_planning_steps,
            "protocol": PROTOCOL_VERSION,
        }

    def _append_private_event(self, record: dict[str, Any]) -> None:
        if self.episode_dir is None:
            return
        record = dict(record, timestamp=time.time())
        with (self.episode_dir / "mcp_private_events.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


_RAB_HANDLE: RabWorkerHandle | None = None


def _get_handle() -> RabWorkerHandle:
    global _RAB_HANDLE
    if _RAB_HANDLE is None:
        root = _repo_root()
        release = Path(os.environ.get("FINDINGBENCH_RAB_RELEASE", root / "runs/rab_lived_in_20261120"))
        out_root = Path(os.environ.get("FINDINGBENCH_RAB_MCP_OUT", root / "runs/rab_mcp_sessions"))
        gpu_id = int(os.environ.get("RUMMAGEBENCH_MCP_GPU", os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or "0"))
        is_replay = os.environ.get("FINDINGBENCH_RAB_MOTION", "discrete").strip().lower() == "replay"
        startup_timeout = float(os.environ.get("FINDINGBENCH_RAB_STARTUP_TIMEOUT", "300"))
        request_timeout = float(os.environ.get("FINDINGBENCH_RAB_REQUEST_TIMEOUT", "450" if is_replay else "220"))
        worker_cmd_env = os.environ.get("FINDINGBENCH_RAB_FAKE_WORKER_CMD")
        worker_cmd = json.loads(worker_cmd_env) if worker_cmd_env else None
        _RAB_HANDLE = RabWorkerHandle(
            release_dir=release,
            out_root=out_root,
            gpu_id=gpu_id,
            startup_timeout_s=startup_timeout,
            request_timeout_s=request_timeout,
            worker_command=worker_cmd,
        )
        atexit.register(_stop_rab_worker)
    return _RAB_HANDLE


def _stop_rab_worker() -> None:
    global _RAB_HANDLE
    if _RAB_HANDLE is not None:
        _RAB_HANDLE.stop()
        _RAB_HANDLE = None


def create_rab_server():
    from mcp.server.fastmcp import FastMCP, Image
    from mcp.types import TextContent

    mcp = FastMCP("findingbench-rab")

    @mcp.tool(structured_output=False)
    def reset_episode(scenario_id: str, seed: int | None = None) -> list[Any]:
        """Reset (or create) a RAB benchmark episode for the given task ID (e.g. 'F251' or 'rab_F251').
        Returns structured JSON metadata and the initial primary RGB observation as a native PNG ImageContent."""
        payload, scene_png = _get_handle().reset(scenario_id, seed=seed)
        return [
            TextContent(type="text", text=json.dumps(payload, ensure_ascii=False)),
            Image(data=scene_png, format="png").to_image_content(),
        ]

    @mcp.tool(structured_output=False)
    def observe() -> list[Any]:
        """Get the current public observation without advancing physics or charging planning steps.
        Returns JSON metadata plus native PNG ImageContent (and a second PNG ImageContent if a CHECK card is active)."""
        payload, scene_png, check_png = _get_handle().observe()
        blocks: list[Any] = [
            TextContent(type="text", text=json.dumps(payload, ensure_ascii=False)),
            Image(data=scene_png, format="png").to_image_content(),
        ]
        if check_png is not None:
            blocks.append(Image(data=check_png, format="png").to_image_content())
        return blocks

    @mcp.tool(structured_output=False)
    def act(action: dict[str, Any]) -> list[Any]:
        """Submit one canonical findingbench_rab_pinhole_v1 skill action:
        - MOVE_TO: {"skill": "MOVE_TO", "point": {"frame_id": "<current>", "x": 0.5, "y": 0.7}}
        - TURN: {"skill": "TURN", "direction": "left" | "right" | "back"}
        - VIEW: {"skill": "VIEW", "yaw": 0.0, "pitch": -12.0, "zoom": 1.0}
        - CHECK: {"skill": "CHECK", "point": {"frame_id": "<current>", "x": 0.4, "y": 0.5}}
        - OPEN / CLOSE / GRASP / PLACE: {"skill": "<SKILL>", "point": {"frame_id": "<current>", "x": 0.5, "y": 0.5}}
        - REPORT_DONE: {"skill": "REPORT_DONE"}
        Coordinates are normalized [0, 1] from top-left on the current scene frame_id. CHECK multiview frames cannot be used as point-action frame_ids."""
        payload, scene_png, check_png = _get_handle().act(action)
        blocks: list[Any] = [
            TextContent(type="text", text=json.dumps(payload, ensure_ascii=False)),
            Image(data=scene_png, format="png").to_image_content(),
        ]
        if check_png is not None:
            blocks.append(Image(data=check_png, format="png").to_image_content())
        return blocks

    @mcp.tool(structured_output=False)
    def episode_status() -> dict[str, Any]:
        """Get the public episode status (NO_EPISODE, RUNNING, SUCCESS, FAIL_FALSE_COMPLETION, FAIL_MAX_STEPS)."""
        return _get_handle().status()

    return mcp


def main() -> int:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    server = create_rab_server()
    try:
        server.run(transport="stdio")
    finally:
        _stop_rab_worker()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
