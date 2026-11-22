#!/usr/bin/env python3
"""Real stdio MCP acceptance test for findingbench_rab_pinhole_v1 (--backend rab).

Exercises the complete MCP handshake against the real Isaac Sim / PhysX RAB skill
worker on GPU 0:
1. List tools -> verify {reset_episode, observe, act, episode_status}
2. reset_episode(F251) -> verify public schema (9 skills, protocol, no leaked private fields)
   and native PNG ImageContent.
3. observe() -> verify idempotence (planning_step=0, identical frame_id and PNG SHA256).
4. act(VIEW, yaw=-27.5, pitch=-60, zoom=1.0) -> EXECUTED, planning_step=1, new frame_id.
5. act(CHECK, point=(0.42, 0.48)) -> EXECUTED, planning_step=2, 2 ImageContent blocks
   (primary scene PNG + 720x450 CHECK 4-view PNG).
6. Stale frame / CHECK frame negative control -> act(GRASP, frame_id=check_frame_id)
   -> INVALID_ACTION, planning_step=3, episode_status=RUNNING.
7. Current frame valid action -> act(GRASP, point=(0.42, 0.48)) -> EXECUTED, planning_step=4.
8. Current frame valid action -> act(PLACE, point=(0.51, 0.65)) -> EXECUTED, planning_step=5.
9. act(REPORT_DONE) -> terminal=True, episode_status in {SUCCESS, FAIL_FALSE_COMPLETION},
   planning_step=6, no private success/restoration leaked in public text.
10. Post-termination action guard -> act(VIEW) after terminal -> INVALID_ACTION,
    planning_step remains 6 (no budget charged).
11. Same-ID reset_episode(F251) -> fresh episode_id, planning_step=0.
12. Cross-ID reset_episode(F262) -> task_id='rab_F262', VIEW + OPEN refrigerator door -> EXECUTED.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from PIL import Image

EXPECTED_SKILLS = {
    "MOVE_TO",
    "TURN",
    "OPEN",
    "CLOSE",
    "GRASP",
    "PLACE",
    "VIEW",
    "CHECK",
    "REPORT_DONE",
}
EXPECTED_OBS_KEYS = {
    "instruction",
    "frame_id",
    "planning_step",
    "max_planning_steps",
    "skill_library",
    "feedback",
    "inspection_view",
    "protocol",
}


def _parse_mcp_response(response):
    assert not response.isError, f"MCP tool call returned error: {response}"
    texts = [c.text for c in response.content if c.type == "text"]
    images = [c for c in response.content if c.type == "image"]
    assert len(texts) == 1, f"Expected 1 TextContent, got {len(texts)}"
    payload = json.loads(texts[0])
    assert payload.get("ok", True), f"Tool payload reported failure: {payload}"
    decoded_images = []
    for img_block in images:
        assert img_block.mimeType == "image/png", f"Expected image/png, got {img_block.mimeType}"
        raw = base64.b64decode(img_block.data)
        assert raw.startswith(b"\x89PNG"), "ImageContent is not valid PNG bytes"
        im = Image.open(io.BytesIO(raw))
        decoded_images.append(
            {
                "bytes": raw,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "width": im.width,
                "height": im.height,
            }
        )
    return payload, decoded_images


async def run(args: argparse.Namespace) -> int:
    root = Path(__file__).resolve().parents[1]
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img_dir = out_path.parent / (out_path.stem + "_images")
    img_dir.mkdir(parents=True, exist_ok=True)
    session_out_dir = out_path.parent / (out_path.stem + "_sessions")
    session_out_dir.mkdir(parents=True, exist_ok=True)

    gpu = os.environ.get("RUMMAGEBENCH_MCP_GPU", "0")
    env = dict(
        os.environ,
        PYTHONPATH=f"{root / 'src'}:{root / 'scripts'}",
        PYTHONUNBUFFERED="1",
        CUDA_VISIBLE_DEVICES=gpu,
        RUMMAGEBENCH_MCP_GPU=gpu,
        FINDINGBENCH_RAB_MCP_OUT=str(session_out_dir),
    )

    report = {
        "schema": "findingbench.rab_mcp_acceptance.v1",
        "protocol": "findingbench_rab_pinhole_v1",
        "ok": False,
        "gpu": gpu,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "checks": [],
        "episodes": [],
    }

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "rummagebench.cli", "serve-mcp", "--backend", "rab"],
        cwd=str(root),
        env=env,
    )

    t_start = time.monotonic()
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # 1. List tools
            tools_res = await session.list_tools()
            tool_names = {t.name for t in tools_res.tools}
            assert {"reset_episode", "observe", "act", "episode_status"} <= tool_names
            report["checks"].append({"check": "list_tools", "tools": sorted(tool_names)})

            # 2. Initial episode_status == NO_EPISODE
            st0, _ = _parse_mcp_response(await session.call_tool("episode_status", {}))
            assert st0["status"] == "NO_EPISODE"
            report["checks"].append({"check": "initial_no_episode", "status": st0["status"]})

            # 3. reset_episode(F251)
            t0 = time.monotonic()
            res1, imgs1 = _parse_mcp_response(await session.call_tool("reset_episode", {"scenario_id": args.scenario}))
            reset_s = time.monotonic() - t0
            obs = res1["reset"]
            assert set(obs.keys()) - {"embodiment_spec"} == EXPECTED_OBS_KEYS, f"Unexpected keys: {set(obs.keys())}"
            assert set(obs["skill_library"]) == EXPECTED_SKILLS
            assert obs["planning_step"] == 0 and obs["max_planning_steps"] == 120
            assert obs["protocol"] == "findingbench_rab_pinhole_v1"
            assert len(imgs1) == 1 and imgs1[0]["width"] == 1280 and imgs1[0]["height"] == 720
            (img_dir / "F251_step0_reset.png").write_bytes(imgs1[0]["bytes"])
            ep1_id = res1["episode_id"]
            report["checks"].append(
                {
                    "check": "reset_F251_schema_and_native_png",
                    "episode_id": ep1_id,
                    "task_id": res1["task_id"],
                    "reset_wall_s": round(reset_s, 3),
                    "frame_id": obs["frame_id"],
                    "png_sha256": imgs1[0]["sha256"],
                    "size": [imgs1[0]["width"], imgs1[0]["height"]],
                }
            )

            # 4. observe() idempotence
            obs_again, imgs_again = _parse_mcp_response(await session.call_tool("observe", {}))
            assert obs_again["observation"] == obs
            assert len(imgs_again) == 1 and imgs_again[0]["sha256"] == imgs1[0]["sha256"]
            report["checks"].append({"check": "observe_idempotent_no_step_or_physics_advance"})

            is_unitree = str((obs.get("embodiment_spec") or {}).get("embodiment_id", "")).startswith("unitree_g1") or os.environ.get("FINDINGBENCH_RAB_ROBOT", "unitree_g1").startswith("unitree")
            view_cfg = {"yaw": 0.0, "pitch": -18.0, "zoom": 1.0} if is_unitree else {"yaw": -27.5, "pitch": -60.0, "zoom": 1.0}
            can_xy = (0.60, 0.62) if is_unitree else (0.42, 0.48)
            place_xy = (0.67, 0.77) if is_unitree else (0.51, 0.65)

            # 5. act(VIEW)
            act_view, imgs_view = _parse_mcp_response(
                await session.call_tool(
                    "act",
                    {"action": {"skill": "VIEW", **view_cfg}},
                )
            )
            assert act_view["feedback"]["code"] == "EXECUTED" and act_view["planning_step"] == 1
            assert act_view["episode_status"] == "RUNNING"
            frame1 = act_view["observation"]["frame_id"]
            assert frame1 != obs["frame_id"]
            (img_dir / "F251_step1_view.png").write_bytes(imgs_view[0]["bytes"])
            report["checks"].append(
                {
                    "check": "act_VIEW",
                    "planning_step": act_view["planning_step"],
                    "frame_id": frame1,
                    "png_sha256": imgs_view[0]["sha256"],
                }
            )

            # 6. act(CHECK) on visible can
            act_check, imgs_check = _parse_mcp_response(
                await session.call_tool(
                    "act",
                    {"action": {"skill": "CHECK", "point": {"frame_id": frame1, "x": can_xy[0], "y": can_xy[1]}}},
                )
            )
            assert act_check["feedback"]["code"] == "EXECUTED" and act_check["planning_step"] == 2
            assert len(imgs_check) == 2, f"Expected scene PNG + CHECK PNG, got {len(imgs_check)}"
            assert imgs_check[1]["width"] == 720 and imgs_check[1]["height"] == 450
            check_meta = act_check["observation"]["inspection_view"]
            assert check_meta is not None and check_meta["point_selectable"] is False
            (img_dir / "F251_step2_check_card.png").write_bytes(imgs_check[1]["bytes"])
            frame2 = act_check["observation"]["frame_id"]
            report["checks"].append(
                {
                    "check": "act_CHECK_multiview_ImageContent",
                    "planning_step": act_check["planning_step"],
                    "scene_frame_id": frame2,
                    "check_frame_id": check_meta["frame_id"],
                    "check_png_sha256": imgs_check[1]["sha256"],
                    "check_size": [imgs_check[1]["width"], imgs_check[1]["height"]],
                }
            )

            # 7. Negative control: point action using CHECK's frame_id (or stale frame_id) -> INVALID_ACTION, charges 1 step
            act_stale, _ = _parse_mcp_response(
                await session.call_tool(
                    "act",
                    {
                        "action": {
                            "skill": "GRASP",
                            "point": {"frame_id": check_meta["frame_id"], "x": can_xy[0], "y": can_xy[1]},
                        }
                    },
                )
            )
            assert act_stale["feedback"]["code"] == "INVALID_ACTION"
            assert act_stale["planning_step"] == 3 and act_stale["episode_status"] == "RUNNING"
            frame3 = act_stale["observation"]["frame_id"]
            report["checks"].append(
                {
                    "check": "negative_control_check_or_stale_frame_rejected_and_charged",
                    "planning_step": act_stale["planning_step"],
                    "feedback_code": act_stale["feedback"]["code"],
                }
            )

            # 8. Positive manipulation: GRASP on current scene frame3 -> EXECUTED
            act_grasp, imgs_grasp = _parse_mcp_response(
                await session.call_tool(
                    "act",
                    {"action": {"skill": "GRASP", "point": {"frame_id": frame3, "x": can_xy[0], "y": can_xy[1]}}},
                )
            )
            assert act_grasp["feedback"]["code"] == "EXECUTED" and act_grasp["planning_step"] == 4
            (img_dir / "F251_step4_grasped.png").write_bytes(imgs_grasp[0]["bytes"])
            frame4 = act_grasp["observation"]["frame_id"]
            report["checks"].append(
                {
                    "check": "act_GRASP_native_physx",
                    "planning_step": act_grasp["planning_step"],
                    "png_sha256": imgs_grasp[0]["sha256"],
                }
            )

            # 9. Positive manipulation: PLACE on current scene frame4 -> EXECUTED
            act_place, imgs_place = _parse_mcp_response(
                await session.call_tool(
                    "act",
                    {"action": {"skill": "PLACE", "point": {"frame_id": frame4, "x": place_xy[0], "y": place_xy[1]}}},
                )
            )
            assert act_place["feedback"]["code"] == "EXECUTED" and act_place["planning_step"] == 5
            (img_dir / "F251_step5_placed.png").write_bytes(imgs_place[0]["bytes"])
            report["checks"].append(
                {
                    "check": "act_PLACE_native_physx",
                    "planning_step": act_place["planning_step"],
                    "png_sha256": imgs_place[0]["sha256"],
                }
            )

            # 10. REPORT_DONE (false completion since Coke was moved to tray, not Honey Maid to side table)
            act_done, _ = _parse_mcp_response(
                await session.call_tool("act", {"action": {"skill": "REPORT_DONE"}})
            )
            assert act_done["terminal"] is True
            assert act_done["episode_status"] == "FAIL_FALSE_COMPLETION"
            assert act_done["planning_step"] == 6
            assert "success" not in act_done and "restoration" not in act_done
            st_done, _ = _parse_mcp_response(await session.call_tool("episode_status", {}))
            assert st_done["status"] == "FAIL_FALSE_COMPLETION" and st_done["terminal"] is True
            report["checks"].append(
                {
                    "check": "act_REPORT_DONE_terminal_and_private_fields_hidden",
                    "planning_step": act_done["planning_step"],
                    "episode_status": act_done["episode_status"],
                }
            )

            # 11. Post-termination action rejected without charging step
            act_post, _ = _parse_mcp_response(
                await session.call_tool(
                    "act",
                    {"action": {"skill": "VIEW", "yaw": 0.0, "pitch": -12.0, "zoom": 1.0}},
                )
            )
            assert act_post["feedback"]["code"] == "INVALID_ACTION"
            assert act_post["planning_step"] == 6 and act_post["terminal"] is True
            report["checks"].append(
                {
                    "check": "post_termination_action_rejected_without_step_charge",
                    "planning_step": act_post["planning_step"],
                }
            )

            # 12. Same-ID reset_episode(F251) -> fresh episode_id & planning_step=0
            res_same, imgs_same = _parse_mcp_response(
                await session.call_tool("reset_episode", {"scenario_id": args.scenario})
            )
            assert res_same["episode_id"] != ep1_id
            assert res_same["reset"]["planning_step"] == 0
            import numpy as np
            arr1 = np.asarray(Image.open(io.BytesIO(imgs1[0]["bytes"])), dtype=np.float32)
            arr2 = np.asarray(Image.open(io.BytesIO(imgs_same[0]["bytes"])), dtype=np.float32)
            mae = float(np.abs(arr1 - arr2).mean())
            assert mae < 5.0, f"Same-ID reset initial frame diverged: MAE={mae}"
            report["checks"].append(
                {
                    "check": "same_id_reset_clean_state_and_consistent_initial_png",
                    "new_episode_id": res_same["episode_id"],
                    "initial_png_mae_255": round(mae, 4),
                    "exact_sha256_match": imgs_same[0]["sha256"] == imgs1[0]["sha256"],
                }
            )

            # 13. Cross-ID reset_episode(F262) -> refrigerator OPEN & CLOSE
            if args.include_fridge:
                res_f262, imgs_f262 = _parse_mcp_response(
                    await session.call_tool("reset_episode", {"scenario_id": "F262"})
                )
                assert res_f262["task_id"] == "rab_F262"
                v262, _ = _parse_mcp_response(
                    await session.call_tool(
                        "act",
                        {"action": {"skill": "VIEW", "yaw": -164.0, "pitch": -20.0, "zoom": 1.0}},
                    )
                )
                f_fid = v262["observation"]["frame_id"]
                o262, imgs_o262 = _parse_mcp_response(
                    await session.call_tool(
                        "act",
                        {"action": {"skill": "OPEN", "point": {"frame_id": f_fid, "x": 0.5, "y": 0.5}}},
                    )
                )
                assert o262["feedback"]["code"] == "EXECUTED"
                (img_dir / "F262_fridge_opened.png").write_bytes(imgs_o262[0]["bytes"])
                close_x = 0.25 if is_unitree else 0.5
                c262, imgs_c262 = _parse_mcp_response(
                    await session.call_tool(
                        "act",
                        {
                            "action": {
                                "skill": "CLOSE",
                                "point": {"frame_id": o262["observation"]["frame_id"], "x": close_x, "y": 0.5},
                            }
                        },
                    )
                )
                assert c262["feedback"]["code"] == "EXECUTED"
                (img_dir / "F262_fridge_closed.png").write_bytes(imgs_c262[0]["bytes"])
                report["checks"].append(
                    {
                        "check": "cross_id_reset_F262_and_fridge_OPEN_CLOSE",
                        "episode_id": res_f262["episode_id"],
                        "opened_png_sha256": imgs_o262[0]["sha256"],
                        "closed_png_sha256": imgs_c262[0]["sha256"],
                    }
                )

    report["ok"] = True
    report["total_wall_s"] = round(time.monotonic() - t_start, 3)
    report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="Output JSON report path")
    parser.add_argument("--scenario", default="F251", help="Primary RAB task ID to test")
    parser.add_argument(
        "--include-fridge",
        action="store_true",
        default=True,
        help="Also verify cross-ID reset and refrigerator OPEN/CLOSE on F262",
    )
    raise SystemExit(asyncio.run(run(parser.parse_args())))
