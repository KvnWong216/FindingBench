"""MCP adapter (stdio). Contains ZERO benchmark logic (Architecture Rule 5).

Each tool forwards into BenchmarkSession and returns canonical results. The
same JSON action schema as the Python API/CLI is used.

The simulator runs in a dedicated worker PROCESS: Kit's internal asyncio loop
conflicts with the MCP stdio server's anyio loop if they share a process, and
keeping the adapter process simulator-free is cleaner anyway. The worker owns
the BenchmarkSession; the server only shuttles dicts over a pipe.

Run: python -m rummagebench.cli serve-mcp
"""

from __future__ import annotations

import base64
import io
import logging
import multiprocessing as mp
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_WORKER = None  # (process, connection) started on first reset_episode


def _scenario_path(scenario_id: str) -> Path:
    root = os.environ.get("RUMMAGEBENCH_ROOT")
    base = Path(root) if root else Path(__file__).resolve().parents[3]
    return base / "scenarios" / scenario_id / "scenario.yaml"


def _rgb_to_png_b64(rgb) -> str | None:
    if rgb is None:
        return None
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _obs_payload(obs) -> dict:
    return {
        "instruction": obs.instruction,
        "planning_step": obs.planning_step,
        "max_planning_steps": obs.max_planning_steps,
        "previous_action_result": obs.previous_action_result,
        "image_png_b64": _rgb_to_png_b64(obs.rgb),
    }


def _worker_main(conn, scenario_path: str, gpu_id: int) -> None:
    """Simulator-owner process. Imports omnigibson; the MCP server never does."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("OMNIGIBSON_HEADLESS", "True")
    os.environ.setdefault("OMNIGIBSON_NO_OMNI_LOGS", "True")
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

    try:
        from rummagebench.adapters.python_api import create_session

        session = create_session(scenario_path)
        obs = session.reset()
        conn.send({"ok": True, **_obs_payload(obs)})
    except Exception as e:
        conn.send({"ok": False, "error": f"{type(e).__name__}: {e}"})
        return

    while True:
        try:
            cmd, payload = conn.recv()
        except (EOFError, KeyboardInterrupt):
            return
        try:
            if cmd == "observe":
                conn.send({"ok": True, **_obs_payload(session.observe())})
            elif cmd == "act":
                result = session.act(payload)
                conn.send(
                    {
                        "ok": True,
                        "episode_status": result.episode_status.value,
                        "planning_step": result.planning_step,
                        "action": result.action,
                        "executed": result.executed,
                        "failure_reason": result.failure_reason.value,
                        "events": result.events,
                        "previous_action_result": result.observation.previous_action_result,
                        "image_png_b64": _rgb_to_png_b64(result.observation.rgb),
                    }
                )
            elif cmd == "status":
                conn.send({"ok": True, "status": session.status().value})
            elif cmd == "exit":
                conn.send({"ok": True})
                return
            else:
                conn.send({"ok": False, "error": f"unknown command {cmd!r}"})
        except Exception as e:
            conn.send({"ok": False, "error": f"{type(e).__name__}: {e}"})


def _ensure_worker(scenario_id: str, gpu_id: int | None = None) -> dict:
    """(Re)start the worker and reset the episode. Returns the reset payload."""
    global _WORKER
    _stop_worker()
    path = str(_scenario_path(scenario_id))
    gpu = gpu_id if gpu_id is not None else int(os.environ.get("RUMMAGEBENCH_MCP_GPU", "4"))

    parent_conn, child_conn = mp.Pipe()
    proc = mp.Process(target=_worker_main, args=(child_conn, path, gpu), daemon=True)
    proc.start()
    child_conn.close()
    result = parent_conn.recv()
    if not result.get("ok"):
        proc.join(timeout=30)
        raise RuntimeError(f"worker failed to start: {result.get('error')}")
    _WORKER = (proc, parent_conn)
    return result


def _stop_worker() -> None:
    global _WORKER
    if _WORKER is not None:
        proc, conn = _WORKER
        try:
            conn.send(("exit", None))
            proc.join(timeout=20)
        except Exception:
            pass
        if proc.is_alive():
            proc.terminate()
        _WORKER = None


def _call_worker(cmd: str, payload=None) -> dict:
    if _WORKER is None:
        raise RuntimeError("no episode; call reset_episode first")
    _, conn = _WORKER
    conn.send((cmd, payload))
    return conn.recv()


def create_server():
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("rummagebench")

    @mcp.tool()
    def reset_episode(scenario_id: str) -> dict:
        """Reset (or create) a benchmark episode for the given scenario id."""
        return _ensure_worker(scenario_id)

    @mcp.tool()
    def observe() -> dict:
        """Get the current observation (instruction, step budget, head RGB)."""
        return _call_worker("observe")

    @mcp.tool()
    def act(action: dict) -> dict:
        """Submit one canonical action, e.g.
        {"skill": "NAV", "target": {"type": "place", "value": "kitchen"}}"""
        return _call_worker("act", action)

    @mcp.tool()
    def episode_status() -> dict:
        """Get the current episode status."""
        if _WORKER is None:
            return {"status": "NO_EPISODE"}
        return _call_worker("status")

    return mcp


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass  # already set
    server = create_server()
    try:
        server.run(transport="stdio")
    finally:
        _stop_worker()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
