"""Real stdio MCP public protocol smoke acceptance (not task completion)."""
import argparse, asyncio, json, os, sys
from pathlib import Path
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def run(args):
    root=Path(__file__).resolve().parents[1]
    gpu=os.environ["RUMMAGEBENCH_MCP_GPU"]
    assert os.environ.get("CUDA_VISIBLE_DEVICES")==gpu, "GPU settings must match"
    env=dict(os.environ, PYTHONPATH=str(root/"src"),PYTHONUNBUFFERED="1")
    report={"ok":False,"checks":[],"kind":"AGENT public MCP smoke; not goal success"}
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    async with stdio_client(StdioServerParameters(command=sys.executable,args=["-m","rummagebench.cli","serve-mcp"],cwd=str(root),env=env)) as (read,write):
        async with ClientSession(read,write) as session:
            await session.initialize()
            async def call(name,arguments):
                response=await session.call_tool(name,arguments)
                assert not response.isError, f"{name} returned MCP error"
                value=json.loads(next(c.text for c in response.content if c.type=="text"))
                assert value.get("ok",True), f"{name} infrastructure failure"
                return value
            data=await call("reset_episode",{"scenario_id":args.scenario})
            obs=data["reset"]
            assert set(obs)=={"instruction","frame_id","image_png_b64","planning_step","max_planning_steps","skill_library","feedback","observe_views"}
            assert obs["image_png_b64"] and obs["planning_step"]==0
            assert set(obs["skill_library"])=={"MOVE","TURN","OPEN","CLOSE","GRASP","PLACE","OBSERVE","REPORT_DONE"}
            report["checks"].append("reset schema and actual RGB")
            again=(await call("observe",{}))["observation"]
            assert again==obs
            report["checks"].append("observe does not advance snapshot")
            result=await call("act",{"action":{"skill":"OPEN","point":{"frame_id":"deliberately-stale","x":.5,"y":.5}}})
            assert result["feedback"]["code"]=="INVALID_ACTION" and result["planning_step"]==1
            assert result["episode_status"]=="RUNNING"
            report["checks"].append("stale frame rejected and budget charged")
            result=await call("act",{"action":{"skill":"REPORT_DONE"}})
            assert result["episode_status"]=="FAIL_FALSE_COMPLETION" and result["planning_step"]==2
            status=await call("episode_status",{})
            assert status["status"]=="FAIL_FALSE_COMPLETION"
            report["checks"].append("false REPORT_DONE terminal status")
            report["ok"]=True
            out.write_text(json.dumps(report,indent=2))
            print(json.dumps(report),flush=True)
    return 0

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",required=True)
    parser.add_argument("--scenario",default="knife_search_001")
    raise SystemExit(asyncio.run(run(parser.parse_args())))
