"""Unit tests for 3-tier self-maintained agentic memory, prefix-cache prompt alignment,
and FindingBench-Core-200 stratified split."""
from __future__ import annotations

import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from pydantic import TypeAdapter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from rab_agent_memory import (
    build_dynamic_step_prompt,
    build_static_episode_prefix,
    build_visual_memory_strip,
    format_compact_history,
    parse_memory_and_action,
    resize_scene_image,
)
from rab_embodiment import UNITREE_G1_CONFIG
from rummagebench.core.public_types import PublicAction
from run_rab_mcp_eval import build_pinhole_prompt, resolve_episode_list


def _make_png(w: int = 1280, h: int = 720, color: tuple[int, int, int] = (40, 90, 140)) -> bytes:
    arr = np.full((h, w, 3), color, dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def test_parse_memory_and_action_envelopes_and_strict_schema():
    adapter = TypeAdapter(PublicAction)

    # 1. Canonical {"memory": ..., "action": {...}} wrapped in <think> + code fence
    raw1 = (
        "<think>Let me scan the kitchen counter first.</think>\n"
        '```json\n'
        '{"memory": "Goal: knife -> counter. Checked: none. Next: scan right.", '
        '"action": {"skill": "VIEW", "yaw": -35.0, "pitch": -15.0, "zoom": 1.0}}\n'
        '```'
    )
    act1, mem1 = parse_memory_and_action(raw1, prev_memory="")
    assert mem1 == "Goal: knife -> counter. Checked: none. Next: scan right."
    validated1 = adapter.validate_python(act1)
    assert validated1.skill == "VIEW" and validated1.yaw == -35.0

    # 2. Flat {"memory": ..., "skill": ...} — must strip "memory" so strict PublicAction passes
    raw2 = '{"memory": "Saw drawer handle at (0.48,0.62). Moving closer.", "skill": "MOVE_TO", "point": {"frame_id": "f01", "x": 0.48, "y": 0.62}}'
    act2, mem2 = parse_memory_and_action(raw2, prev_memory=mem1)
    assert "memory" not in act2
    assert "Saw drawer handle" in mem2
    validated2 = adapter.validate_python(act2)
    assert validated2.skill == "MOVE_TO"

    # 3. Legacy bare {"skill": ...} retains previous memory unchanged
    raw3 = '{"skill": "TURN", "direction": "left"}'
    act3, mem3 = parse_memory_and_action(raw3, prev_memory=mem2)
    assert mem3 == mem2
    assert adapter.validate_python(act3).skill == "TURN"

    # 4. Unparseable output returns empty action {} without erasing previous memory
    act4, mem4 = parse_memory_and_action("truncated {invalid json", prev_memory=mem2)
    assert act4 == {}
    assert mem4 == mem2

    # 5. Oversized memory is capped cleanly at max_memory_chars
    long_mem = "x" * 1200
    _, mem5 = parse_memory_and_action({"memory": long_mem, "action": {"skill": "REPORT_DONE"}}, max_memory_chars=600)
    assert len(mem5) == 600 and mem5.endswith("...")


def test_prefix_invariance_and_zero_stale_frame_id_leakage():
    spec = UNITREE_G1_CONFIG.public_spec
    p0 = build_static_episode_prefix(
        "Find the coke can in the living room and put it on the tray.",
        50,
        spec,
        memory_mode="scratchpad_visual_k2",
    )
    p1 = build_static_episode_prefix(
        "Find the coke can in the living room and put it on the tray.",
        50,
        spec,
        memory_mode="scratchpad_visual_k2",
    )
    assert hashlib.sha256(p0.encode()).hexdigest() == hashlib.sha256(p1.encode()).hexdigest()
    assert "unitree_g1_dex1" in p0

    # Simulate 35 historical steps with distinct 32-hex frame_ids
    history = []
    for i in range(35):
        stale_fid = f"stale_hex_frame_{i:04d}_abcdef123456"
        history.append({
            "action": {"skill": "OPEN", "point": {"frame_id": stale_fid, "x": 0.45, "y": 0.58}},
            "feedback": {
                "code": "OUT_OF_CAPABILITY",
                "detail_code": "ARTICULATION_TOO_FAR",
                "message": "The clicked drawer or door is beyond manipulation reach; move closer before opening.",
                "diagnostics": {"distance_m": 1.04, "reach_limit_m": 0.88},
            },
        })

    obs = {
        "instruction": "Find the coke can in the living room and put it on the tray.",
        "frame_id": "current_live_frame_9999",
        "planning_step": 35,
        "max_planning_steps": 50,
        "skill_library": ["VIEW", "MOVE_TO", "OPEN"],
        "embodiment_spec": spec,
        "feedback": history[-1]["feedback"],
        "inspection_view": None,
    }
    dyn = build_dynamic_step_prompt(
        obs,
        history,
        working_memory="Checked coffee table left drawer. Next: move closer to right drawer.",
        memory_mode="scratchpad_visual_k2",
        has_visual_strip=True,
    )
    # Current frame_id is present; NO stale historical frame_ids appear anywhere in dynamic prompt!
    assert "current_live_frame_9999" in dyn
    assert "stale_hex_frame_" not in dyn
    assert "Checked coffee table left drawer." in dyn
    assert "OUT_OF_CAPABILITY[ARTICULATION_TOO_FAR] (distance_m=1.04, reach_limit_m=0.88)" in dyn

    # Compare dynamic uncached prompt size vs legacy unsplit JSON prompt at step 35
    legacy = build_pinhole_prompt(obs, history, 60, "default_natural")
    assert len(dyn) < 0.30 * len(legacy)


def test_visual_memory_strip_o1_bounds_and_resize():
    raw_hd = _make_png(1280, 720)
    resized = resize_scene_image(raw_hd, max_width=960)
    with Image.open(io.BytesIO(resized)) as im:
        assert im.size == (960, 540)

    assert build_visual_memory_strip([]) is None

    ring_1 = [(0, "VIEW(yaw=+30,pitch=-12,zoom=1.0)", raw_hd)]
    strip_1 = build_visual_memory_strip(ring_1, max_frames=2, panel_size=(384, 216))
    assert strip_1 is not None
    with Image.open(io.BytesIO(strip_1)) as im1:
        assert im1.size == (384, 238)

    # Even after 10 steps, K=2 strip keeps exact constant width (2 * 384 + 4 = 772)
    ring_10 = [(i, f"MOVE_TO(0.500,0.{60+i:02d}0)", raw_hd) for i in range(10)]
    strip_2 = build_visual_memory_strip(ring_10, max_frames=2, panel_size=(384, 216))
    assert strip_2 is not None
    with Image.open(io.BytesIO(strip_2)) as im2:
        assert im2.size == (772, 238)


def test_core200_stratified_subset_integrity():
    core_path = ROOT / "configs/benchmarks/rab_v1_core200.json"
    assert core_path.is_file()
    data = json.loads(core_path.read_text())
    eps = data["episodes"]
    assert len(eps) == 200
    assert len(set(eps)) == 200
    assert data["layouts_covered"] == 50

    rel = ROOT / "runs/rab_lived_in_20261120"
    resolved = resolve_episode_list({"episodes": "core200"}, rel)
    assert resolved == eps
    sh0 = resolve_episode_list({"episodes": "core200"}, rel, shard="0/4")
    assert len(sh0) == 50
