"""Run FindingBench RAB v1 evaluation strictly through the stdio MCP contract.

Driven by a committed JSON config file (`configs/benchmarks/eval_config.json`) so
that OpenResearch (`orx`) experiment nodes share the exact same `--run-command`
and vary only committed configuration/code on each `orx/<slug>` branch.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import ImageContent, TextContent

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from rummagebench.agents.policy_adapter import PolicyInfrastructureError, parse_action  # noqa: E402

RAB_PINHOLE_PROMPT_V1 = """You control a mobile manipulation robot in a household room using its current pinhole RGB camera image and public feedback.
Complete the user's household instruction, then call REPORT_DONE. Return exactly ONE JSON action object per turn.
Never invent object IDs, world coordinates, or hidden state.

PUBLIC SKILL LIBRARY (9 skills, protocol: findingbench_rab_pinhole_v1):
1. VIEW: {"skill":"VIEW","yaw":0.0,"pitch":-20.0,"zoom":1.0}
   - Changes the head camera viewing direction without moving the robot base.
   - yaw in [-180, 180] degrees (positive = left, negative = right, relative to robot base heading).
   - pitch in [-89, 89] degrees (negative = look down at table/drawers/floor, positive = look up).
   - zoom in [1.0, 4.0] (1.0 = default 78-deg HFOV; larger zoom inspects a narrower region).
2. TURN: {"skill":"TURN","direction":"left"}
   - Rotates the robot base in place; direction must be "left" (90 deg), "right" (-90 deg), or "back" (180 deg).
3. MOVE_TO: {"skill":"MOVE_TO","point":{"frame_id":"CURRENT_FRAME_ID","x":0.5,"y":0.5}}
   - Navigates the robot base to a visible floor point or in front of a clicked piece of furniture/object.
4. OPEN: {"skill":"OPEN","point":{"frame_id":"CURRENT_FRAME_ID","x":0.5,"y":0.5}}
   - Opens a closed drawer or refrigerator door at the clicked handle/front surface (must be within reach; MOVE_TO closer first if needed).
5. CLOSE: {"skill":"CLOSE","point":{"frame_id":"CURRENT_FRAME_ID","x":0.5,"y":0.5}}
   - Closes an open drawer or refrigerator door at the clicked surface.
6. CHECK: {"skill":"CHECK","point":{"frame_id":"CURRENT_FRAME_ID","x":0.5,"y":0.5}}
   - Renders a 4-view isolated exterior inspection card (720x450) of a visible movable object so you can verify packaging/identity.
   - IMPORTANT: The CHECK inspection image is NOT a scene camera and its frame_id cannot be used for point actions. Always use the current scene observation's frame_id for point actions.
7. GRASP: {"skill":"GRASP","point":{"frame_id":"CURRENT_FRAME_ID","x":0.5,"y":0.5}}
   - Grasps a visible movable object within reach after passing kinematic reachability (IK) and collision checks.
8. PLACE: {"skill":"PLACE","point":{"frame_id":"CURRENT_FRAME_ID","x":0.5,"y":0.5}}
   - Places the currently held object onto the clicked horizontal support surface within reach.
9. REPORT_DONE: {"skill":"REPORT_DONE"}
   - Terminates the episode and triggers task evaluation.

COORDINATE & FEEDBACK RULES:
- Image coordinates x, y are normalized in [0.0, 1.0] with origin (0,0) at the top-left of the current pinhole RGB image.
- Every point action (MOVE_TO, OPEN, CLOSE, CHECK, GRASP, PLACE) MUST use the exact `frame_id` of the current scene observation.
- Every action (including VIEW and rejected actions) costs 1 planning step.
- Public feedback codes: EXECUTED, INVALID_ACTION, OUT_OF_CAPABILITY, UNSAFE.
"""

DIAGNOSTIC_RESTORE_ADDENDUM = """
DIAGNOSTIC RESTORATION & SAFETY REMINDER:
- When searching inside drawers or refrigerator doors, close any opened drawer or door (using CLOSE) after retrieving the target object or before leaving the container.
- If a movable distractor blocks access, place it back on a valid support surface before calling REPORT_DONE.
"""


def request_json(url: str, body: dict | None = None, timeout: int = 300) -> dict:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.load(resp)
    except urllib.error.HTTPError as err:
        detail = err.read(8192).decode("utf-8", errors="replace")
        raise PolicyInfrastructureError(f"HTTP {err.code}: {detail}") from err
    except (OSError, ValueError) as err:
        raise PolicyInfrastructureError(f"{type(err).__name__}: {err}") from err
    if not isinstance(result, dict) or result.get("error"):
        raise PolicyInfrastructureError(str(result))
    return result


def ensure_vlm_server(model_key: str, endpoint: str, profile: dict) -> subprocess.Popen | None:
    try:
        health = request_json(endpoint + "/health", timeout=5)
        if health.get("model_id") == profile["model_id"]:
            return None
    except Exception:
        pass
    port = int(endpoint.rsplit(":", 1)[-1].split("/")[0])
    env_name = profile.get("environment", "ygbench-vlm")
    py_bin = Path(f"/home/ubuntu/miniforge3/envs/{env_name}/bin/python")
    server_script = ROOT / "scripts" / ("serve_extended_vlm.py" if env_name != "ygbench-vlm" else "serve_local_vlm.py")
    cmd = [str(py_bin), "-u", str(server_script), "--model", model_key, "--port", str(port)]
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="0",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    deadline = time.monotonic() + 240.0
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            out = proc.stdout.read() if proc.stdout else ""
            raise RuntimeError(f"VLM server exited early with code {proc.returncode}:\n{out}")
        try:
            health = request_json(endpoint + "/health", timeout=3)
            if health.get("model_id") == profile["model_id"]:
                return proc
        except Exception:
            time.sleep(2.0)
    proc.terminate()
    raise TimeoutError(f"Timed out waiting for VLM server {model_key} on {endpoint}")


def parse_mcp_result(result) -> tuple[dict, list[bytes]]:
    if getattr(result, "isError", False):
        texts = [c.text for c in result.content if isinstance(c, TextContent)]
        raise RuntimeError("MCP tool error: " + "\n".join(texts))
    payload = None
    images: list[bytes] = []
    for item in result.content:
        if isinstance(item, TextContent):
            payload = json.loads(item.text)
        elif isinstance(item, ImageContent):
            images.append(base64.b64decode(item.data))
    if payload is None:
        raise RuntimeError("MCP response missing TextContent JSON payload")
    return payload, images


def build_pinhole_prompt(obs: dict, history: list[dict], history_steps: int, prompt_mode: str) -> str:
    header = RAB_PINHOLE_PROMPT_V1
    if prompt_mode == "diagnostic_explicit_restore":
        header += DIAGNOSTIC_RESTORE_ADDENDUM
    public_obs = {
        "instruction": obs["instruction"],
        "frame_id": obs["frame_id"],
        "planning_step": obs["planning_step"],
        "max_planning_steps": obs["max_planning_steps"],
        "skill_library": obs["skill_library"],
        "feedback": obs.get("feedback"),
        "inspection_view": obs.get("inspection_view"),
    }
    trimmed = history[-history_steps:] if history_steps > 0 else []
    return (
        header
        + "\nCurrent observation:\n"
        + json.dumps(public_obs, ensure_ascii=False)
        + f"\nPrevious actions and public feedback (last {len(trimmed)} of {len(history)} steps, oldest first):\n"
        + json.dumps(trimmed, ensure_ascii=False)
        + '\nChoose your next action now. Return only one JSON object with top-level "skill":\n'
    )


def classify_episode_failure(ep_record: dict) -> dict:
    """Two-level failure attribution from public actions + private simulator logs."""
    if ep_record.get("status") == "SUCCESS":
        return {"high_level_category": "NONE_SUCCESS", "low_level_category": "NONE_SUCCESS"}
    actions = ep_record.get("actions", [])
    private_actions = ep_record.get("private_actions", [])
    codes = Counter(a.get("feedback", {}).get("code") for a in actions)
    skills = Counter((a.get("action") or {}).get("skill") for a in actions)

    # Low-level attribution
    low_cat = "NONE_OBSERVED"
    for pa in private_actions:
        res_code = pa.get("result", {}).get("feedback", {}).get("code")
        perr = str(pa.get("private_error") or "")
        sk = (pa.get("raw") or {}).get("skill")
        if res_code in ("OUT_OF_CAPABILITY", "UNSAFE"):
            if sk in ("MOVE_TO", "TURN") or "NAV_" in perr or "TURN_" in perr:
                low_cat = "L1_NAV_CLEARANCE_OR_ROUTE_REFUSAL"
            elif sk in ("OPEN", "CLOSE"):
                low_cat = "L2_ARTICULATION_REACH_OR_SWEEP_REFUSAL"
            elif sk == "GRASP":
                low_cat = "L3_GRASP_IK_OR_PATH_REFUSAL"
            elif sk == "PLACE":
                low_cat = "L4_PLACE_SETTLE_OR_IK_REFUSAL"
            else:
                low_cat = "L5_OTHER_PHYSICAL_REFUSAL"
            break

    # High-level attribution
    if ep_record.get("status") == "FAIL_FALSE_COMPLETION":
        if skills.get("GRASP", 0) == 0:
            high_cat = "H1_PREMATURE_REPORT_DONE_WITHOUT_GRASP"
        elif skills.get("PLACE", 0) == 0:
            high_cat = "H2_REPORT_DONE_WITHOUT_DELIVERY"
        else:
            high_cat = "H3_WRONG_TARGET_OR_WRONG_DESTINATION"
    else:
        if skills.get("MOVE_TO", 0) == 0 and skills.get("OPEN", 0) == 0:
            high_cat = "H4_VIEW_ONLY_OR_NO_EXPLORATION"
        elif codes.get("INVALID_ACTION", 0) >= max(3, len(actions) // 2):
            high_cat = "H5_SCHEMA_OR_STALE_FRAME_ERRORS"
        elif skills.get("GRASP", 0) == 0:
            high_cat = "H6_SEARCH_EXHAUSTED_TARGET_NOT_GRASPED"
        else:
            high_cat = "H7_GRASPED_BUT_DELIVERY_INCOMPLETE"

    return {"high_level_category": high_cat, "low_level_category": low_cat}


async def run_evaluation(cfg: dict) -> dict:
    out_dir = ROOT / cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    sessions_dir = out_dir / "mcp_sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    profiles = json.loads((ROOT / "configs/models/local_vlm.json").read_text())["models"]
    model_key = cfg["model"]
    profile = profiles[model_key]
    endpoint = cfg["endpoint"].rstrip("/")

    vlm_proc = None
    if cfg.get("auto_serve_vlm", True):
        print(f"[eval] Ensuring local VLM server for {model_key} at {endpoint}...", flush=True)
        vlm_proc = ensure_vlm_server(model_key, endpoint, profile)

    try:
        health = request_json(endpoint + "/health", timeout=10)
        print(f"[eval] Connected to VLM server: {health.get('model_id')} on {health.get('gpu')}", flush=True)

        server_params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "rummagebench.cli", "serve-mcp", "--backend", "rab"],
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": "0",
                "RUMMAGEBENCH_MCP_GPU": "0",
                "RUMMAGEBENCH_RAB_LOG_DIR": str(sessions_dir),
                "PYTHONPATH": str(ROOT / "src") + os.pathsep + str(ROOT / "scripts"),
            },
        )

        episode_results = []
        t_eval_start = time.monotonic()

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                for task_id in cfg["episodes"]:
                    ep_dir = out_dir / "episodes" / task_id
                    ep_dir.mkdir(parents=True, exist_ok=True)
                    print(f"[eval] Starting episode {task_id} (budget={cfg['max_planning_steps']} steps)...", flush=True)

                    t0 = time.monotonic()
                    reset_payload, reset_imgs = parse_mcp_result(
                        await session.call_tool(
                            "reset_episode",
                            {"scenario_id": task_id, "seed": int(cfg["seed"]), "max_planning_steps": int(cfg["max_planning_steps"])},
                        )
                    )
                    reset_wall_s = time.monotonic() - t0
                    episode_id = reset_payload["episode_id"]
                    obs = reset_payload["observation"]
                    curr_images = reset_imgs
                    (ep_dir / "step_0000.png").write_bytes(curr_images[0])

                    history: list[dict] = []
                    action_logs: list[dict] = []
                    model_wall_total_s = 0.0
                    tool_wall_total_s = 0.0
                    input_tokens_total = 0
                    output_tokens_total = 0
                    terminal = False
                    episode_status = "RUNNING"

                    while not terminal:
                        step_idx = obs["planning_step"]
                        prompt = build_pinhole_prompt(
                            obs,
                            history,
                            int(cfg.get("history_steps", 60)),
                            cfg.get("prompt_mode", "default_natural"),
                        )
                        img_b64 = base64.b64encode(curr_images[0]).decode("ascii")
                        images_payload = [
                            {
                                "label": f"Image 1: Current pinhole RGB camera view; frame_id={obs['frame_id']}",
                                "image_png_b64": img_b64,
                            }
                        ]
                        if len(curr_images) > 1 and obs.get("inspection_view"):
                            images_payload.append(
                                {
                                    "label": "Image 2: Isolated 4-view CHECK inspection sheet (not point-selectable)",
                                    "image_png_b64": base64.b64encode(curr_images[1]).decode("ascii"),
                                }
                            )

                        t_m0 = time.monotonic()
                        gen = request_json(
                            endpoint + "/generate",
                            {
                                "model": profile["model_id"],
                                "prompt": prompt,
                                "image_png_b64": img_b64,
                                "images": images_payload,
                                "max_new_tokens": int(cfg.get("max_new_tokens", 512)),
                                "seed": int(cfg.get("seed", 0)),
                            },
                            timeout=300,
                        )
                        m_dt = time.monotonic() - t_m0
                        model_wall_total_s += m_dt
                        input_tokens_total += int(gen.get("input_tokens", 0))
                        output_tokens_total += int(gen.get("output_tokens", 0))

                        raw_text = gen.get("text", "")
                        parsed_action = parse_action(raw_text)
                        if not isinstance(parsed_action, dict) or "skill" not in parsed_action:
                            # Malformed output is submitted as an invalid skill so the step is charged fairly
                            parsed_action = {"skill": "VIEW", "yaw": 999.0, "pitch": 0.0}

                        t_a0 = time.monotonic()
                        act_payload, act_imgs = parse_mcp_result(
                            await session.call_tool("act", {"action": parsed_action})
                        )
                        a_dt = time.monotonic() - t_a0
                        tool_wall_total_s += a_dt

                        feedback = act_payload["feedback"]
                        terminal = bool(act_payload["terminal"])
                        episode_status = act_payload["episode_status"]
                        obs = act_payload["observation"]
                        curr_images = act_imgs

                        (ep_dir / f"step_{obs['planning_step']:04d}.png").write_bytes(curr_images[0])
                        if len(curr_images) > 1:
                            (ep_dir / f"step_{obs['planning_step']:04d}_check.png").write_bytes(curr_images[1])

                        history.append({"action": parsed_action, "feedback": feedback})
                        step_record = {
                            "step": obs["planning_step"],
                            "raw_model_text": raw_text,
                            "action": parsed_action,
                            "feedback": feedback,
                            "episode_status": episode_status,
                            "terminal": terminal,
                            "model_wall_s": round(m_dt, 4),
                            "tool_wall_s": round(a_dt, 4),
                            "input_tokens": gen.get("input_tokens"),
                            "output_tokens": gen.get("output_tokens"),
                        }
                        action_logs.append(step_record)
                        print(
                            f"  [{task_id} step {obs['planning_step']:02d}] {parsed_action.get('skill')} -> {feedback['code']} | status={episode_status} (model={m_dt:.2f}s, sim={a_dt:.2f}s)",
                            flush=True,
                        )

                    # Read private simulator logs & restoration score from session log dir
                    private_dir = sessions_dir / f"{task_id}_{episode_id}" / "actions"
                    private_actions = []
                    restoration_score = None
                    if private_dir.is_dir():
                        for pfile in sorted(private_dir.glob("*.json")):
                            pa = json.loads(pfile.read_text())
                            private_actions.append(pa)
                            if "restoration" in pa.get("result", {}):
                                restoration_score = pa["result"]["restoration"]

                    # Compute restoration if episode ended via FAIL_MAX_STEPS without REPORT_DONE
                    ep_total_wall_s = reset_wall_s + model_wall_total_s + tool_wall_total_s
                    ep_summary = {
                        "task_id": task_id,
                        "episode_id": episode_id,
                        "instruction": obs["instruction"],
                        "valid": True,
                        "status": episode_status,
                        "success": episode_status == "SUCCESS",
                        "planning_steps_used": obs["planning_step"],
                        "max_planning_steps": obs["max_planning_steps"],
                        "restoration_score": restoration_score,
                        "timing": {
                            "reset_wall_s": round(reset_wall_s, 4),
                            "model_wall_s": round(model_wall_total_s, 4),
                            "tool_sim_wall_s": round(tool_wall_total_s, 4),
                            "episode_total_wall_s": round(ep_total_wall_s, 4),
                        },
                        "tokens": {
                            "input_tokens": input_tokens_total,
                            "output_tokens": output_tokens_total,
                        },
                        "feedback_counts": dict(Counter(a["feedback"]["code"] for a in action_logs)),
                        "skill_counts": dict(Counter((a["action"] or {}).get("skill") for a in action_logs)),
                        "actions": action_logs,
                        "private_actions": private_actions,
                    }
                    ep_summary["failure_taxonomy"] = classify_episode_failure(ep_summary)
                    (ep_dir / "episode_report.json").write_text(json.dumps(ep_summary, indent=2, ensure_ascii=False) + "\n")
                    episode_results.append(ep_summary)

        # Aggregate benchmark v1 metrics
        valid_eps = [e for e in episode_results if e["valid"]]
        succ_eps = [e for e in valid_eps if e["success"]]
        rest_scores = [e["restoration_score"] for e in valid_eps if e["restoration_score"] is not None]
        sr = len(succ_eps) / len(valid_eps) if valid_eps else 0.0
        st_wall_s = (
            sum(e["timing"]["episode_total_wall_s"] for e in succ_eps) / len(succ_eps)
            if succ_eps
            else None
        )
        st_steps = (
            sum(e["planning_steps_used"] for e in succ_eps) / len(succ_eps)
            if succ_eps
            else None
        )
        mean_restoration = sum(rest_scores) / len(rest_scores) if rest_scores else None

        summary = {
            "schema": "findingbench.rab_v1_eval_summary.v1",
            "experiment_id": cfg["experiment_id"],
            "description": cfg.get("description", ""),
            "protocol": "findingbench_rab_pinhole_v1",
            "model": model_key,
            "model_id": profile["model_id"],
            "revision": profile["revision"],
            "prompt_mode": cfg.get("prompt_mode", "default_natural"),
            "history_steps": cfg.get("history_steps", 60),
            "max_planning_steps": cfg["max_planning_steps"],
            "seed": cfg["seed"],
            "metrics": {
                "attempted_episodes": len(episode_results),
                "valid_episodes": len(valid_eps),
                "invalid_infrastructure_episodes": len(episode_results) - len(valid_eps),
                "success_count": len(succ_eps),
                "SR": round(sr, 4),
                "ST_wall_s": round(st_wall_s, 4) if st_wall_s is not None else None,
                "ST_steps": round(st_steps, 4) if st_steps is not None else None,
                "mean_steps_all_valid": round(sum(e["planning_steps_used"] for e in valid_eps) / len(valid_eps), 4) if valid_eps else None,
                "mean_wall_s_all_valid": round(sum(e["timing"]["episode_total_wall_s"] for e in valid_eps) / len(valid_eps), 4) if valid_eps else None,
                "mean_model_wall_s": round(sum(e["timing"]["model_wall_s"] for e in valid_eps) / len(valid_eps), 4) if valid_eps else None,
                "mean_tool_sim_wall_s": round(sum(e["timing"]["tool_sim_wall_s"] for e in valid_eps) / len(valid_eps), 4) if valid_eps else None,
                "Restoration_mean": round(mean_restoration, 4) if mean_restoration is not None else None,
            },
            "failure_taxonomy_counts": {
                "high_level": dict(Counter(e["failure_taxonomy"]["high_level_category"] for e in valid_eps)),
                "low_level": dict(Counter(e["failure_taxonomy"]["low_level_category"] for e in valid_eps)),
            },
            "episodes": [
                {k: v for k, v in e.items() if k not in ("actions", "private_actions")}
                for e in episode_results
            ],
            "total_eval_wall_s": round(time.monotonic() - t_eval_start, 3),
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
        print("\n=== EVALUATION SUMMARY ===", flush=True)
        print(json.dumps(summary["metrics"], indent=2, ensure_ascii=False), flush=True)
        print("Failure Taxonomy:", json.dumps(summary["failure_taxonomy_counts"], indent=2, ensure_ascii=False), flush=True)
        return summary
    finally:
        if vlm_proc is not None:
            print("[eval] Terminating auto-spawned VLM server...", flush=True)
            vlm_proc.terminate()
            try:
                vlm_proc.wait(timeout=10)
            except Exception:
                vlm_proc.kill()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/benchmarks/eval_config.json")
    args = parser.parse_args()
    cfg = json.loads(args.config.resolve().read_text())
    asyncio.run(run_evaluation(cfg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
