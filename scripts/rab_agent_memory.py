"""3-Tier Self-Maintained Agentic Memory & Prefix-Cache-Aligned Prompt Builder.

Designed for FindingBench RAB v1 (`findingbench_rab_pinhole_v1`) to cut VLM
token consumption by ~20x-25x on 1,000-episode evaluations while preserving
100% genuine, zero-cheating agent memory:

1. **Prefix-Cache-Aligned Static System Prefix**:
   `PUBLIC_SKILL_LIBRARY + COORDINATE_RULES + EMBODIMENT_SPEC + EPISODE_INSTRUCTION`
   is 100% byte-identical across every planning step `0..T` of an episode so
   OpenAI / Gemini / Anthropic / DeepSeek / vLLM / SGLang prefix caches hit at
   75%-90% discount on every step after step 0.
2. **Tier 1 — Model-Authored Working Scratchpad (`working_memory`)**:
   The evaluated VLM outputs `{"memory": "...", "action": {...}}` (backward
   compatible with bare `{"skill": ...}`). The harness feeds back ONLY the
   model's own previous `"memory"` string (capped at `max_memory_chars=600`),
   injecting zero privileged simulator state.
3. **Tier 2 — Complete Compact Action & Proprioceptive Feedback Trace**:
   Lossless 1-line-per-step chronological log (~14 tokens/step) omitting stale
   historical `frame_id` hashes (saving ~85% history tokens and preventing
   stale-`frame_id` copy-paste errors).
4. **Tier 3 — Short-Term Visual Memory Strip ($O(1)$ Vision Tokens)**:
   Primary pinhole view resized to `960x540` (configurable, preserving `[0,1]`
   normalized coordinates) + a single stitched `768x238` side-by-side thumbnail
   strip of the previous `K=2` frames (`t-2`, `t-1`).
"""
from __future__ import annotations

import io
import json
from typing import Any, Sequence

from PIL import Image, ImageDraw

from rummagebench.agents.policy_adapter import parse_action

VALID_MEMORY_MODES = (
    "scratchpad_visual_k2",  # Full 3-tier memory: scratchpad + compact trace + K=2 visual strip (default)
    "scratchpad_only",       # Text-only 2-tier memory: scratchpad + compact trace, primary image only
    "history_only",          # Compact action/feedback trace only, bare {"skill": ...} output
    "stateless",             # Zero cross-step memory ablation baseline
    "legacy_json",           # Original unsplit prompt + verbose JSON history (for exact backward repro)
)

RAB_PINHOLE_STATIC_RULES = """You control a mobile manipulation robot in a household room using its current pinhole RGB camera image and public physical feedback.
Complete the user's household instruction, then call REPORT_DONE.
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
   - Opens a closed drawer or refrigerator door at the clicked handle/front surface (MOVE_TO closer first if out of reach).
5. CLOSE: {"skill":"CLOSE","point":{"frame_id":"CURRENT_FRAME_ID","x":0.5,"y":0.5}}
   - Closes an open drawer or refrigerator door at the clicked surface.
6. CHECK: {"skill":"CHECK","point":{"frame_id":"CURRENT_FRAME_ID","x":0.5,"y":0.5}}
   - Renders a 4-view isolated exterior inspection card (720x450) of a visible movable object to verify packaging/identity.
   - IMPORTANT: Neither the CHECK inspection card nor any visual memory strip is point-selectable. Always use the current primary camera view's `selectable_frame_id` for point actions.
7. GRASP: {"skill":"GRASP","point":{"frame_id":"CURRENT_FRAME_ID","x":0.5,"y":0.5}}
   - Grasps a visible movable object within reach after passing kinematic reachability (IK) and collision checks.
8. PLACE: {"skill":"PLACE","point":{"frame_id":"CURRENT_FRAME_ID","x":0.5,"y":0.5}}
   - Places the currently held object onto the clicked horizontal support surface within reach.
9. REPORT_DONE: {"skill":"REPORT_DONE"}
   - Terminates the episode and triggers task evaluation.

COORDINATE & FEEDBACK RULES:
- Image coordinates x, y are normalized in [0.0, 1.0] with origin (0,0) at the top-left of Image 1 (current pinhole RGB camera view).
- Every point action (MOVE_TO, OPEN, CLOSE, CHECK, GRASP, PLACE) MUST set `point.frame_id` to the exact `selectable_frame_id` of the current step.
- Every action (including VIEW and rejected actions) costs 1 planning step.
- Public feedback codes: EXECUTED, INVALID_ACTION, OUT_OF_CAPABILITY, UNSAFE."""

DIAGNOSTIC_RESTORE_ADDENDUM = """

DIAGNOSTIC RESTORATION & SAFETY REMINDER:
- When searching inside drawers or refrigerator doors, close any opened drawer or door (using CLOSE) after retrieving the target object or before leaving the container.
- If a movable distractor blocks access, place it back on a valid support surface before calling REPORT_DONE."""

SCRATCHPAD_OUTPUT_CONTRACT = """

SELF-MAINTAINED WORKING MEMORY & OUTPUT SCHEMA:
Return exactly ONE JSON object per turn with two top-level keys:
{"memory": "<concise note <= 500 chars: (1) target & delivery goal, (2) containers/zones already checked & findings, (3) held object & any open drawer/door to close, (4) next subgoal>", "action": {"skill": "...", ...}}
Your own "memory" text is echoed back to you on the next step so you maintain state across views and steps."""

BARE_ACTION_OUTPUT_CONTRACT = """

OUTPUT SCHEMA:
Return exactly ONE JSON action object per turn with top-level key "skill" (e.g. {"skill":"VIEW","yaw":45.0,"pitch":-15.0,"zoom":1.0})."""


def _coerce_memory_text(val: Any, max_chars: int) -> str:
    if val is None:
        return ""
    if isinstance(val, str):
        text = " ".join(val.split())
    else:
        text = json.dumps(val, ensure_ascii=False, separators=(",", ":"))
    if len(text) > max_chars:
        return text[: max_chars - 3] + "..."
    return text


def parse_memory_and_action(
    raw: Any,
    prev_memory: str = "",
    *,
    max_memory_chars: int = 600,
) -> tuple[dict[str, Any], str]:
    """Parse model output into `(public_action_dict, next_working_memory)`.

    Supports:
    1. `{"memory": "...", "action": {"skill": ...}}`
    2. `{"memory": "...", "skill": ..., ...}` (strips `"memory"` so strict PublicAction validation passes)
    3. Bare `{"skill": ..., ...}` (retains `prev_memory` unchanged)
    """
    parsed = parse_action(raw)
    if not isinstance(parsed, dict) or not parsed:
        return {}, prev_memory

    next_memory = prev_memory
    if "memory" in parsed:
        next_memory = _coerce_memory_text(parsed.get("memory"), max_memory_chars)

    if isinstance(parsed.get("action"), dict):
        action_dict = dict(parsed["action"])
        action_dict.pop("memory", None)
        return action_dict, next_memory

    if "skill" in parsed:
        action_dict = {k: v for k, v in parsed.items() if k != "memory"}
        return action_dict, next_memory

    return {}, next_memory


def short_action_signature(action: dict[str, Any] | None) -> str:
    if not isinstance(action, dict) or "skill" not in action:
        return "INVALID()"
    sk = str(action.get("skill"))
    if sk == "VIEW":
        yaw = float(action.get("yaw", 0.0))
        pitch = float(action.get("pitch", 0.0))
        zoom = float(action.get("zoom", 1.0))
        return f"VIEW(yaw={yaw:+.0f},pitch={pitch:+.0f},zoom={zoom:.1f})"
    if sk == "TURN":
        return f"TURN({action.get('direction', '?')})"
    if sk == "REPORT_DONE":
        return "REPORT_DONE()"
    pt = action.get("point")
    if isinstance(pt, dict) and "x" in pt and "y" in pt:
        try:
            return f"{sk}({float(pt['x']):.3f},{float(pt['y']):.3f})"
        except (TypeError, ValueError):
            pass
    return f"{sk}()"


def format_compact_feedback(feedback: dict[str, Any] | None) -> str:
    if not isinstance(feedback, dict):
        return "NONE"
    code = str(feedback.get("code", "UNKNOWN"))
    detail = feedback.get("detail_code")
    head = f"{code}[{detail}]" if detail and detail != code else code
    if code == "EXECUTED":
        return head
    diag = feedback.get("diagnostics")
    extras: list[str] = []
    if isinstance(diag, dict):
        for k, v in list(diag.items())[:4]:
            if isinstance(v, float):
                extras.append(f"{k}={v:.3g}")
            elif isinstance(v, (int, str, bool)):
                extras.append(f"{k}={v}")
    if extras:
        return f"{head} ({', '.join(extras)})"
    msg = str(feedback.get("remediation") or feedback.get("message") or "").strip()
    if msg:
        msg = " ".join(msg.split())
        if len(msg) > 96:
            msg = msg[:93] + "..."
        return f"{head} ({msg})"
    return head


def format_compact_history(history: Sequence[dict[str, Any]], *, max_steps: int = 120) -> str:
    """Format chronological `(action, feedback)` pairs into 1-line-per-step text."""
    if not history or max_steps <= 0:
        return "(none — episode start)"
    total = len(history)
    start_idx = max(0, total - max_steps)
    lines: list[str] = []
    for idx in range(start_idx, total):
        item = history[idx]
        sig = short_action_signature(item.get("action"))
        fb = format_compact_feedback(item.get("feedback"))
        lines.append(f"#{idx + 1:02d} {sig} -> {fb}")
    return "\n".join(lines)


def resize_scene_image(png_bytes: bytes, *, max_width: int = 960) -> bytes:
    """Resize primary scene RGB frame while preserving aspect ratio and `[0,1]` coords."""
    if max_width <= 0:
        return png_bytes
    with Image.open(io.BytesIO(png_bytes)) as img:
        w, h = img.size
        if w <= max_width:
            return png_bytes
        new_h = max(1, int(round(h * (max_width / float(w)))))
        resized = img.convert("RGB").resize((max_width, new_h), Image.Resampling.BILINEAR)
        buf = io.BytesIO()
        resized.save(buf, format="PNG", optimize=False)
        return buf.getvalue()


def build_visual_memory_strip(
    prev_frames: Sequence[tuple[int, str, bytes]],
    *,
    max_frames: int = 2,
    panel_size: tuple[int, int] = (384, 216),
) -> bytes | None:
    """Stitch the last `K` observations (`t-2`, `t-1`) into one compact labeled RGB strip."""
    if not prev_frames or max_frames <= 0:
        return None
    selected = list(prev_frames[-max_frames:])
    pw, ph = int(panel_size[0]), int(panel_size[1])
    banner_h = 22
    gap = 4
    n = len(selected)
    total_w = n * pw + (n - 1) * gap
    total_h = ph + banner_h
    canvas = Image.new("RGB", (total_w, total_h), color=(24, 26, 30))
    draw = ImageDraw.Draw(canvas)

    for i, (step_idx, tag, raw_png) in enumerate(selected):
        x0 = i * (pw + gap)
        with Image.open(io.BytesIO(raw_png)) as im:
            thumb = im.convert("RGB").resize((pw, ph), Image.Resampling.BILINEAR)
        canvas.paste(thumb, (x0, banner_h))
        rel = i - n  # e.g. -2, -1
        header = f"t{rel} | step {step_idx:02d}: {tag}"[:48]
        draw.text((x0 + 6, 4), header, fill=(235, 240, 245))

    buf = io.BytesIO()
    canvas.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def build_static_episode_prefix(
    instruction: str,
    max_planning_steps: int,
    embodiment_spec: dict[str, Any] | None,
    *,
    prompt_mode: str = "default_natural",
    memory_mode: str = "scratchpad_visual_k2",
) -> str:
    """Build the 100% step-invariant prefix for an episode (maximizes prefix-cache hits)."""
    if memory_mode not in VALID_MEMORY_MODES:
        raise ValueError(f"Unknown memory_mode={memory_mode!r}; expected one of {VALID_MEMORY_MODES}")
    parts = [RAB_PINHOLE_STATIC_RULES]
    if prompt_mode == "diagnostic_explicit_restore":
        parts.append(DIAGNOSTIC_RESTORE_ADDENDUM)
    if memory_mode in ("scratchpad_visual_k2", "scratchpad_only"):
        parts.append(SCRATCHPAD_OUTPUT_CONTRACT)
    else:
        parts.append(BARE_ACTION_OUTPUT_CONTRACT)

    if embodiment_spec is not None:
        parts.append(
            "\n\nROBOT EMBODIMENT SPECIFICATION (constant for this episode):\n"
            + json.dumps(embodiment_spec, ensure_ascii=False, separators=(",", ":"))
        )
    parts.append(
        "\n\nEPISODE TASK (constant for this episode):\n"
        + json.dumps(
            {
                "instruction": instruction,
                "max_planning_steps": int(max_planning_steps),
            },
            ensure_ascii=False,
        )
    )
    return "".join(parts)


def build_dynamic_step_prompt(
    obs: dict[str, Any],
    history: Sequence[dict[str, Any]],
    working_memory: str = "",
    *,
    memory_mode: str = "scratchpad_visual_k2",
    history_steps: int = 120,
    has_visual_strip: bool = False,
) -> str:
    """Build the compact per-step dynamic user turn."""
    sections: list[str] = []
    use_scratchpad = memory_mode in ("scratchpad_visual_k2", "scratchpad_only")

    if use_scratchpad:
        mem_text = working_memory.strip() or "(empty — initialize your search plan and state in \"memory\")"
        sections.append(f"YOUR WORKING MEMORY (from your previous step):\n{mem_text}")

    if memory_mode != "stateless":
        trace_text = format_compact_history(history, max_steps=history_steps)
        sections.append(
            f"COMPLETE ACTION & FEEDBACK TRACE ({len(history)} steps elapsed, oldest first):\n{trace_text}"
        )

    step_state: dict[str, Any] = {
        "planning_step": int(obs["planning_step"]),
        "remaining_steps": int(obs["max_planning_steps"]) - int(obs["planning_step"]),
        "selectable_frame_id": obs["frame_id"],
        "last_feedback": obs.get("feedback"),
        "check_inspection_attached": bool(obs.get("inspection_view")),
        "visual_memory_strip_attached": bool(has_visual_strip),
    }
    sections.append("CURRENT STEP STATE:\n" + json.dumps(step_state, ensure_ascii=False))

    if use_scratchpad:
        sections.append(
            'Choose your next action now. Return ONLY one JSON object: {"memory":"...","action":{"skill":...}}'
        )
    else:
        sections.append(
            'Choose your next action now. Return ONLY one JSON action object with top-level "skill":'
        )
    return "\n\n".join(sections)
