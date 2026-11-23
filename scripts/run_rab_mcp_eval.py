"""Run FindingBench RAB v1 evaluation strictly through the stdio MCP contract.

Supports full-scale 1,000-episode evaluation with:
- Unitree G1 (Dex1-1 2-finger parallel gripper default) / Dex3 / AgiBot G1 switching
- Static Embodiment Prior (`embodiment_spec`) forwarding in public observations
- NVLabs cuRoboV2-Fast GPU IK + proprioceptive_v2 physical feedback
- Both local HTTP VLM servers (`api_type="local_vlm"`) and OpenAI-compatible
  `/v1/chat/completions` endpoints (`api_type="openai_chat"`, e.g. vLLM / SGLang /
  OpenRouter / LiteLLM / OpenAI / Gemini / Anthropic gateways)
- `--episodes all` + `--shard idx/total` + `--resume` checkpointing
- Per-episode crash isolation + automatic MCP session restart
- Automatic v2 metrics aggregation (`SR`, `ST`, `EFF_phys`, `EFF_all`, `LAT`,
  `RES`, `RSER`, Layer A 5-axis capabilities, Layer B terminal attribution)
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
import traceback
import urllib.error
import urllib.request

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import ImageContent, TextContent

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from rummagebench.agents.policy_adapter import PolicyInfrastructureError, parse_action  # noqa: E402
from rab_agent_memory import (  # noqa: E402
    build_dynamic_step_prompt,
    build_static_episode_prefix,
    build_visual_memory_strip,
    parse_memory_and_action,
    resize_scene_image,
    short_action_signature,
)

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


def request_json(url: str, body: dict | None = None, *, headers: dict | None = None, timeout: int = 300) -> dict:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req_headers = {"Content-Type": "application/json", **(headers or {})}
    req = urllib.request.Request(url, data=data, headers=req_headers)
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


def call_model_endpoint(
    api_type: str,
    endpoint: str,
    model_id: str,
    prompt: str,
    images_payload: list[dict],
    *,
    static_prefix: str | None = None,
    enable_cache_control: bool = False,
    max_new_tokens: int = 384,
    seed: int = 0,
    temperature: float = 0.0,
    api_key: str | None = None,
    timeout: int = 300,
) -> dict:
    """Unified VLM query supporting local `/generate` and prefix-cache-aligned `/v1/chat/completions`."""
    if api_type == "openai_chat":
        messages: list[dict] = []
        if static_prefix:
            if enable_cache_control:
                messages.append({
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": static_prefix,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                })
            else:
                messages.append({"role": "system", "content": static_prefix})
        content = [{"type": "text", "text": prompt}]
        for im in images_payload:
            content.append({"type": "text", "text": im["label"]})
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{im['image_png_b64']}"},
            })
        messages.append({"role": "user", "content": content})
        url = endpoint if endpoint.endswith("/chat/completions") else (endpoint.rstrip("/") + "/v1/chat/completions")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        body = {
            "model": model_id,
            "messages": messages,
            "max_tokens": int(max_new_tokens),
            "temperature": float(temperature),
            "seed": int(seed),
        }
        res = request_json(url, body, headers=headers, timeout=timeout)
        choice = (res.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        text = msg.get("content") or ""
        if isinstance(text, list):
            text = "\n".join(part.get("text", "") for part in text if isinstance(part, dict))
        usage = res.get("usage") or {}
        pt_details = usage.get("prompt_tokens_details") or {}
        cached_tokens = int(
            pt_details.get("cached_tokens")
            or usage.get("prompt_cache_hit_tokens")
            or usage.get("cache_read_input_tokens")
            or 0
        )
        return {
            "text": str(text),
            "input_tokens": int(usage.get("prompt_tokens", 0)),
            "cached_input_tokens": cached_tokens,
            "output_tokens": int(usage.get("completion_tokens", 0)),
        }

    full_prompt = f"{static_prefix}\n\n{prompt}" if static_prefix else prompt
    return request_json(
        endpoint.rstrip("/") + "/generate",
        {
            "model": model_id,
            "prompt": full_prompt,
            "image_png_b64": images_payload[0]["image_png_b64"],
            "images": images_payload,
            "max_new_tokens": int(max_new_tokens),
            "seed": int(seed),
        },
        timeout=timeout,
    )


def ensure_vlm_server(model_key: str, endpoint: str, profile: dict, *, gpu: str = "0") -> subprocess.Popen | None:
    try:
        health = request_json(endpoint + "/health", timeout=5)
        if health.get("model_id") == profile["model_id"]:
            return None
    except Exception:
        pass
    port = int(endpoint.rsplit(":", 1)[-1].split("/")[0])
    env_name = profile.get("environment", "ygbench-vlm")
    conda_roots = [
        Path(os.environ["CONDA_EXE"]).resolve().parents[1] if os.environ.get("CONDA_EXE") else None,
        Path.home() / "miniforge3",
        Path.home() / "miniconda3",
        Path("/home/ubuntu/miniforge3"),
    ]
    py_bin = next(
        (r / "envs" / env_name / "bin" / "python" for r in conda_roots if r and (r / "envs" / env_name / "bin" / "python").is_file()),
        Path(sys.executable),
    )
    server_script = ROOT / "scripts" / ("serve_extended_vlm.py" if env_name != "ygbench-vlm" else "serve_local_vlm.py")
    cmd = [str(py_bin), "-u", str(server_script), "--model", model_key, "--port", str(port)]
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=str(gpu),
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
    if obs.get("embodiment_spec") is not None:
        public_obs["embodiment_spec"] = obs["embodiment_spec"]
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


def resolve_episode_list(cfg: dict, release_dir: Path, shard: str | None = None) -> list[str]:
    raw_eps = cfg["episodes"]
    if raw_eps == "all":
        manifest = json.loads((release_dir / "combined_manifest.json").read_text())
        episodes = [t["id"] for t in manifest["tasks"]]
    elif raw_eps == "core200":
        core_path = ROOT / "configs/benchmarks/rab_v1_core200.json"
        episodes = list(json.loads(core_path.read_text())["episodes"])
    elif isinstance(raw_eps, str) and raw_eps.startswith("@"):
        sub_path = (ROOT / raw_eps[1:]).resolve() if not Path(raw_eps[1:]).is_absolute() else Path(raw_eps[1:])
        episodes = list(json.loads(sub_path.read_text())["episodes"])
    else:
        episodes = list(raw_eps)
    sl = cfg.get("episode_slice")
    if isinstance(sl, list) and len(sl) == 2:
        episodes = episodes[int(sl[0]) : int(sl[1])]
    if shard:
        idx_s, tot_s = shard.split("/", 1)
        idx, tot = int(idx_s), int(tot_s)
        if not (0 <= idx < tot):
            raise ValueError(f"Invalid --shard {shard!r}; expected 0 <= idx < total")
        episodes = [ep for i, ep in enumerate(episodes) if i % tot == idx]
    return episodes


async def _run_single_episode(
    session: ClientSession,
    task_id: str,
    cfg: dict,
    *,
    api_type: str,
    endpoint: str,
    model_id: str,
    api_key: str | None,
    ep_dir: Path,
    sessions_dir: Path,
) -> dict:
    step_budget = int(cfg["max_planning_steps"])
    memory_mode = str(cfg.get("memory_mode", "scratchpad_visual_k2"))
    prompt_mode = str(cfg.get("prompt_mode", "default_natural"))
    history_steps = int(cfg.get("history_steps", 120))
    max_memory_chars = int(cfg.get("max_memory_chars", 600))
    visual_memory_frames = int(cfg.get("visual_memory_frames", 2))
    scene_image_max_width = int(cfg.get("scene_image_max_width", 960))
    enable_cache_control = bool(cfg.get("enable_cache_control", False))

    ep_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    reset_payload, reset_imgs = parse_mcp_result(
        await session.call_tool("reset_episode", {"scenario_id": task_id})
    )
    reset_wall_s = time.monotonic() - t0
    episode_id = reset_payload["episode_id"]
    obs = reset_payload.get("reset") or reset_payload["observation"]
    curr_images = reset_imgs
    (ep_dir / "step_0000.png").write_bytes(curr_images[0])

    # Build the 100% step-invariant prefix ONCE per episode for prefix-cache alignment
    static_prefix: str | None = None
    static_prefix_sha256: str | None = None
    if memory_mode != "legacy_json":
        static_prefix = build_static_episode_prefix(
            obs["instruction"],
            int(obs["max_planning_steps"]),
            obs.get("embodiment_spec"),
            prompt_mode=prompt_mode,
            memory_mode=memory_mode,
        )
        static_prefix_sha256 = hashlib.sha256(static_prefix.encode("utf-8")).hexdigest()

    history: list[dict] = []
    visual_ring: list[tuple[int, str, bytes]] = []
    working_memory = ""
    action_logs: list[dict] = []
    model_wall_total_s = 0.0
    tool_wall_total_s = 0.0
    input_tokens_total = 0
    cached_input_tokens_total = 0
    output_tokens_total = 0
    terminal = False
    episode_status = "RUNNING"

    while not terminal:
        visual_strip_bytes: bytes | None = None
        if memory_mode == "scratchpad_visual_k2" and visual_ring and visual_memory_frames > 0:
            visual_strip_bytes = build_visual_memory_strip(
                visual_ring,
                max_frames=visual_memory_frames,
            )

        if memory_mode == "legacy_json":
            prompt = build_pinhole_prompt(obs, history, history_steps, prompt_mode)
        else:
            prompt = build_dynamic_step_prompt(
                obs,
                history,
                working_memory,
                memory_mode=memory_mode,
                history_steps=history_steps,
                has_visual_strip=visual_strip_bytes is not None,
            )

        primary_png = resize_scene_image(curr_images[0], max_width=scene_image_max_width)
        images_payload = [
            {
                "label": (
                    f"Image 1: Current pinhole RGB camera view "
                    f"(SELECTABLE: point.frame_id={obs['frame_id']}, x/y normalized in [0,1])"
                ),
                "image_png_b64": base64.b64encode(primary_png).decode("ascii"),
            }
        ]
        if len(curr_images) > 1 and obs.get("inspection_view"):
            images_payload.append(
                {
                    "label": (
                        f"Image {len(images_payload) + 1}: Isolated 4-view CHECK inspection sheet "
                        "(NOT point-selectable)"
                    ),
                    "image_png_b64": base64.b64encode(curr_images[1]).decode("ascii"),
                }
            )
        if visual_strip_bytes is not None:
            images_payload.append(
                {
                    "label": (
                        f"Image {len(images_payload) + 1}: Short-term visual memory strip of previous views "
                        "(spatial context ONLY — NOT point-selectable)"
                    ),
                    "image_png_b64": base64.b64encode(visual_strip_bytes).decode("ascii"),
                }
            )

        prev_step_idx = int(obs["planning_step"])
        prev_scene_png = curr_images[0]

        t_m0 = time.monotonic()
        gen = call_model_endpoint(
            api_type,
            endpoint,
            model_id,
            prompt,
            images_payload,
            static_prefix=static_prefix,
            enable_cache_control=enable_cache_control,
            max_new_tokens=int(cfg.get("max_new_tokens", 384)),
            seed=int(cfg.get("seed", 0)),
            temperature=float(cfg.get("temperature", 0.0)),
            api_key=api_key,
            timeout=int(cfg.get("model_timeout_s", 300)),
        )
        m_dt = time.monotonic() - t_m0
        model_wall_total_s += m_dt
        input_tokens_total += int(gen.get("input_tokens", 0))
        cached_input_tokens_total += int(gen.get("cached_input_tokens", 0))
        output_tokens_total += int(gen.get("output_tokens", 0))

        raw_text = gen.get("text", "")
        if memory_mode == "legacy_json":
            parsed_action = parse_action(raw_text)
        else:
            parsed_action, working_memory = parse_memory_and_action(
                raw_text,
                working_memory,
                max_memory_chars=max_memory_chars,
            )
        if not isinstance(parsed_action, dict) or "skill" not in parsed_action:
            parsed_action = {"skill": "VIEW", "yaw": 999.0, "pitch": 0.0}

        # Push pre-action frame + chosen action tag into the visual memory ring
        visual_ring.append((prev_step_idx, short_action_signature(parsed_action), prev_scene_png))
        if len(visual_ring) > max(4, visual_memory_frames + 1):
            visual_ring = visual_ring[-max(4, visual_memory_frames + 1) :]

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
            "working_memory": working_memory,
            "action": parsed_action,
            "feedback": feedback,
            "episode_status": episode_status,
            "terminal": terminal,
            "model_wall_s": round(m_dt, 4),
            "tool_wall_s": round(a_dt, 4),
            "input_tokens": gen.get("input_tokens"),
            "cached_input_tokens": gen.get("cached_input_tokens", 0),
            "output_tokens": gen.get("output_tokens"),
        }
        action_logs.append(step_record)
        fb_tag = feedback.get("detail_code") or feedback["code"]
        print(
            f"  [{task_id} step {obs['planning_step']:02d}/{step_budget}] "
            f"{parsed_action.get('skill')} -> {fb_tag} | status={episode_status} "
            f"(think={m_dt:.2f}s, sim={a_dt:.2f}s)",
            flush=True,
        )
        if not terminal and obs["planning_step"] >= step_budget:
            terminal = True
            episode_status = "FAIL_MAX_STEPS"

    matches = sorted(sessions_dir.glob(f"*_{task_id}_{episode_id[:8]}"))
    private_dir = (matches[-1] / "worker_trace" / "actions") if matches else None
    private_actions = []
    restoration_score = None
    if private_dir and private_dir.is_dir():
        for pfile in sorted(private_dir.glob("*.json")):
            pa = json.loads(pfile.read_text())
            private_actions.append(pa)
            if "restoration" in pa.get("result", {}):
                restoration_score = pa["result"]["restoration"]
            elif "restoration" in pa:
                restoration_score = pa["restoration"]

    ep_total_wall_s = reset_wall_s + model_wall_total_s + tool_wall_total_s
    ep_summary = {
        "task_id": task_id,
        "episode_id": episode_id,
        "instruction": obs["instruction"],
        "memory_mode": memory_mode,
        "static_prefix_sha256": static_prefix_sha256,
        "valid": True,
        "status": episode_status,
        "success": episode_status == "SUCCESS",
        "planning_steps_used": obs["planning_step"],
        "max_planning_steps": obs["max_planning_steps"],
        "restoration_score": restoration_score,
        "timing": {
            "reset_wall_s": round(reset_wall_s, 4),
            "model_wall_s": round(model_wall_total_s, 4),
            "mean_inter_step_lat_s": round(model_wall_total_s / max(1, len(action_logs)), 4),
            "tool_sim_wall_s": round(tool_wall_total_s, 4),
            "episode_total_wall_s": round(ep_total_wall_s, 4),
        },
        "tokens": {
            "input_tokens": input_tokens_total,
            "cached_input_tokens": cached_input_tokens_total,
            "uncached_input_tokens": max(0, input_tokens_total - cached_input_tokens_total),
            "output_tokens": output_tokens_total,
        },
        "feedback_counts": dict(Counter(a["feedback"]["code"] for a in action_logs)),
        "skill_counts": dict(Counter((a["action"] or {}).get("skill") for a in action_logs)),
        "actions": action_logs,
        "private_actions": private_actions,
    }
    ep_summary["failure_taxonomy"] = classify_episode_failure(ep_summary)
    (ep_dir / "episode_report.json").write_text(json.dumps(ep_summary, indent=2, ensure_ascii=False) + "\n")
    return ep_summary


async def run_evaluation(cfg: dict, *, resume: bool = False, shard: str | None = None) -> dict:
    default_root = Path("/home/ubuntu/ygwang/FindingBench")
    fallback_root = default_root.resolve() if default_root.exists() else ROOT
    workspace_root = Path(os.environ.get("RUMMAGEBENCH_ROOT", str(fallback_root))).resolve()
    asset_root = workspace_root if not (ROOT / "runs/rab_lived_in_20261120").exists() else ROOT
    release_dir = Path(cfg.get("release_dir") or (asset_root / "runs/rab_lived_in_20261120")).resolve()
    out_dir = asset_root / cfg["out_dir"]
    if shard:
        out_dir = out_dir / f"shard_{shard.replace('/', '_of_')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    sessions_dir = out_dir / "mcp_sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    api_type = cfg.get("api_type", "local_vlm")
    model_key = cfg["model"]
    endpoint = cfg["endpoint"].rstrip("/")
    api_key = os.environ.get(cfg.get("api_key_env", "OPENAI_API_KEY"))

    vlm_proc = None
    sim_gpu = str(cfg.get("sim_gpu", os.environ.get("RUMMAGEBENCH_MCP_GPU", "0")))
    vlm_gpu = str(cfg.get("vlm_gpu", sim_gpu))

    if api_type == "local_vlm":
        profiles = json.loads((ROOT / "configs/models/local_vlm.json").read_text())["models"]
        profile = profiles[model_key]
        model_id = profile["model_id"]
        revision = profile.get("revision", "local")
        if cfg.get("auto_serve_vlm", True):
            print(f"[eval] Ensuring local VLM server for {model_key} at {endpoint}...", flush=True)
            vlm_proc = ensure_vlm_server(model_key, endpoint, profile, gpu=vlm_gpu)
        health = request_json(endpoint + "/health", timeout=10)
        print(f"[eval] Connected to local VLM server: {health.get('model_id')} on {health.get('gpu')}", flush=True)
    else:
        model_id = cfg.get("model_id", model_key)
        revision = cfg.get("revision", "api")
        print(f"[eval] Using OpenAI-compatible chat endpoint: {endpoint} (model={model_id})", flush=True)

    robot_emb = cfg.get("robot_embodiment", os.environ.get("FINDINGBENCH_RAB_ROBOT", "unitree_g1"))
    emb_prior = cfg.get("embodiment_prior", os.environ.get("FINDINGBENCH_RAB_EMBODIMENT_PRIOR", "text"))
    ik_backend = cfg.get("ik_backend", os.environ.get("FINDINGBENCH_RAB_IK_BACKEND", "auto"))
    fb_mode = cfg.get("feedback_mode", os.environ.get("FINDINGBENCH_RAB_FEEDBACK_MODE", "proprioceptive_v2"))
    motion_mode = cfg.get("motion_mode", os.environ.get("FINDINGBENCH_RAB_MOTION", "discrete"))

    server_params = StdioServerParameters(
        command=str(ROOT / "mcp/run_findingbench_rab_mcp.sh"),
        args=[],
        cwd=str(ROOT),
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": sim_gpu,
            "RUMMAGEBENCH_MCP_GPU": sim_gpu,
            "RUMMAGEBENCH_ROOT": str(asset_root),
            "FINDINGBENCH_RAB_RELEASE": str(release_dir),
            "FINDINGBENCH_RAB_MCP_OUT": str(sessions_dir),
            "FINDINGBENCH_RAB_ROBOT": robot_emb,
            "FINDINGBENCH_RAB_EMBODIMENT_PRIOR": emb_prior,
            "FINDINGBENCH_RAB_IK_BACKEND": ik_backend,
            "FINDINGBENCH_RAB_FEEDBACK_MODE": fb_mode,
            "FINDINGBENCH_RAB_MOTION": motion_mode,
            "FINDINGBENCH_RAB_HOT_RELOAD": os.environ.get("FINDINGBENCH_RAB_HOT_RELOAD", "1"),
            "FINDINGBENCH_RAB_STOP_ON_TERMINAL": os.environ.get("FINDINGBENCH_RAB_STOP_ON_TERMINAL", "0"),
            "PYTHONPATH": str(ROOT / "src") + os.pathsep + str(ROOT / "scripts"),
        },
    )

    episodes = resolve_episode_list(cfg, release_dir, shard=shard)
    use_resume = bool(resume or cfg.get("resume", False))
    step_budget = int(cfg["max_planning_steps"])

    try:
        episode_results: list[dict] = []
        pending: list[str] = []
        for task_id in episodes:
            rep_path = out_dir / "episodes" / task_id / "episode_report.json"
            if use_resume and rep_path.is_file():
                try:
                    cached = json.loads(rep_path.read_text())
                    if cached.get("valid"):
                        print(f"[eval] Resuming: skipping completed {task_id} ({cached.get('status')})", flush=True)
                        episode_results.append(cached)
                        continue
                except Exception:
                    pass
            pending.append(task_id)

        t_eval_start = time.monotonic()
        idx_p = 0
        while idx_p < len(pending):
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    while idx_p < len(pending):
                        task_id = pending[idx_p]
                        ep_dir = out_dir / "episodes" / task_id
                        print(
                            f"[eval] [{len(episode_results) + 1}/{len(episodes)}] Starting {task_id} "
                            f"(budget={step_budget}, robot={robot_emb}, ik={ik_backend}, fb={fb_mode})...",
                            flush=True,
                        )
                        try:
                            ep_summary = await _run_single_episode(
                                session,
                                task_id,
                                cfg,
                                api_type=api_type,
                                endpoint=endpoint,
                                model_id=model_id,
                                api_key=api_key,
                                ep_dir=ep_dir,
                                sessions_dir=sessions_dir,
                            )
                            episode_results.append(ep_summary)
                            idx_p += 1
                        except Exception as exc:
                            err_trace = traceback.format_exc()
                            print(f"[eval] INFRA ERROR on {task_id}, restarting MCP session:\n{err_trace}", flush=True)
                            fail_rec = {
                                "task_id": task_id,
                                "valid": False,
                                "status": "INFRA_ERROR",
                                "success": False,
                                "error": str(exc),
                                "traceback": err_trace,
                            }
                            ep_dir.mkdir(parents=True, exist_ok=True)
                            (ep_dir / "episode_report.json").write_text(json.dumps(fail_rec, indent=2) + "\n")
                            episode_results.append(fail_rec)
                            idx_p += 1
                            break  # Re-open a fresh MCP stdio session for remaining tasks

        valid_eps = [e for e in episode_results if e.get("valid")]
        succ_eps = [e for e in valid_eps if e.get("success")]
        rest_scores = [e["restoration_score"] for e in valid_eps if e.get("restoration_score") is not None]
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

        # Compute full v2 deterministic metrics (SR, ST, EFF_phys, LAT, RES, RSER, Layer A/B)
        import rab_eval_metrics as M

        ep_dirs = M.find_episodes(sessions_dir)
        v2_report = M.aggregate(M.load_episode(d) for d in ep_dirs) if ep_dirs else None
        if v2_report is not None:
            (out_dir / "metrics_v2.json").write_text(json.dumps(v2_report, indent=2, ensure_ascii=False) + "\n")
            (out_dir / "metrics_v2.md").write_text(M.to_markdown({model_key: v2_report}))

        total_in_tok = sum(int(e.get("tokens", {}).get("input_tokens", 0)) for e in valid_eps)
        total_cached_tok = sum(int(e.get("tokens", {}).get("cached_input_tokens", 0)) for e in valid_eps)
        total_out_tok = sum(int(e.get("tokens", {}).get("output_tokens", 0)) for e in valid_eps)

        summary = {
            "schema": "findingbench.rab_v1_eval_summary.v2",
            "experiment_id": cfg["experiment_id"],
            "description": cfg.get("description", ""),
            "protocol": "findingbench_rab_pinhole_v1",
            "robot_embodiment": robot_emb,
            "embodiment_prior": emb_prior,
            "ik_backend": ik_backend,
            "feedback_mode": fb_mode,
            "motion_mode": motion_mode,
            "memory_mode": cfg.get("memory_mode", "scratchpad_visual_k2"),
            "scene_image_max_width": int(cfg.get("scene_image_max_width", 960)),
            "model": model_key,
            "model_id": model_id,
            "revision": revision,
            "prompt_mode": cfg.get("prompt_mode", "default_natural"),
            "history_steps": cfg.get("history_steps", 120),
            "max_planning_steps": step_budget,
            "seed": cfg["seed"],
            "metrics": {
                "attempted_episodes": len(episode_results),
                "valid_episodes": len(valid_eps),
                "invalid_infrastructure_episodes": len(episode_results) - len(valid_eps),
                "success_count": len(succ_eps),
                "SR": round(sr, 4),
                "ST_wall_s": round(st_wall_s, 4) if st_wall_s is not None else None,
                "ST_steps": round(st_steps, 4) if st_steps is not None else None,
                "EFF_phys": v2_report["core"]["EFF_phys"] if v2_report else None,
                "EFF_all": v2_report["core"]["EFF_all"] if v2_report else None,
                "LAT_s_per_step": v2_report["core"]["LAT"] if v2_report else None,
                "RSER": v2_report["core"]["RSER"] if v2_report else None,
                "Restoration_mean": round(mean_restoration, 4) if mean_restoration is not None else None,
                "total_input_tokens": total_in_tok,
                "total_cached_input_tokens": total_cached_tok,
                "total_output_tokens": total_out_tok,
            },
            "metrics_v2": v2_report,
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
    parser.add_argument("--resume", action="store_true", help="Skip already completed valid episodes")
    parser.add_argument("--shard", type=str, default=None, help="Optional shard spec i/N (e.g. 0/4)")
    args = parser.parse_args()
    cfg = json.loads(args.config.resolve().read_text())
    asyncio.run(run_evaluation(cfg, resume=args.resume, shard=args.shard))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
