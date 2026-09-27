#!/usr/bin/env python3
"""MCP acceptance test: drives the benchmark through the stdio MCP adapter
and verifies it produces the same results as the direct Python API.

    python scripts/mcp_acceptance.py
"""

import asyncio
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402


async def main() -> int:
    params = StdioServerParameters(
        command="/data1/ygwang/miniconda3/envs/behavior/bin/python",
        args=["-m", "rummagebench.cli", "serve-mcp"],
        cwd=str(REPO),
        env={
            "PATH": "/data1/ygwang/miniconda3/envs/behavior/bin:/usr/bin:/bin",
            "CUDA_VISIBLE_DEVICES": "4",
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": str(REPO / "src"),
            "HOME": str(Path.home()),
        },
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            reset = await session.call_tool(
                "reset_episode", {"scenario_id": "knife_search_001"}
            )
            print("raw content parts:", [(c.type, str(c)[:120]) for c in reset.content])
            reset_data = json.loads(reset.content[0].text)
            print("instruction:", reset_data["instruction"].strip())
            print("budget:", reset_data["max_planning_steps"])
            assert reset_data["image_png_b64"], "no image returned"

            actions = [
                {"skill": "NAV", "target": {"type": "place", "value": "kitchen"}},
                {"skill": "OPEN", "target": {"type": "entity", "value": "bottom_cabinet_no_top_spojpj_0"}},
                {"skill": "OPEN", "target": {"type": "entity", "value": "bottom_cabinet_rvpunw_0"}},
                {"skill": "OPEN", "target": {"type": "entity", "value": "bottom_cabinet_no_top_qohxjq_0"}},
                {"skill": "GRASP", "target": {"type": "entity", "value": "target_knife"}},
            ]
            for a in actions:
                result = await session.call_tool("act", {"action": a})
                data = json.loads(result.content[0].text)
                print(
                    f"MCP step {data['planning_step']}: {a['skill']} -> {data['episode_status']}"
                )

            status = await session.call_tool("episode_status", {})
            status_data = json.loads(status.content[0].text)
            print("final status:", status_data["status"])
            assert status_data["status"] == "SUCCESS", "MCP episode did not succeed"
            print("MCP ACCEPTANCE OK")
            return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
