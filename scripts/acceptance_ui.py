"""Run the actual localhost UI with separate evaluator-only evidence."""
import base64,json,sys
from pathlib import Path
from agent_acceptance_driver import snapshot,compare
from rummagebench.environment.env import InteractiveSearchEnv
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
sys.argv=[sys.argv[0],*sys.argv[2:]]
original_init=InteractiveSearchEnv.__init__
original_step=InteractiveSearchEnv.step
def init(self,*args,**kwargs):
 kwargs["trace_path"]=out/"private_trace.jsonl"
 original_init(self,*args,**kwargs)
InteractiveSearchEnv.__init__=init
def step(self,action):
 before=snapshot(self._backend) if action.get("skill")=="OBSERVE" else None
 physical_before=bool(any(getattr(self._backend._robot,"_ag_obj_in_hand",{}).values()))
 semantic_before=self._session.world_state.held_object is not None
 result=original_step(self,action)
 index=result["planning_step"]
 (out/f"public_{index}.json").write_text(json.dumps(result))
 (out/f"rgb_{index}.png").write_bytes(base64.b64decode(result["observation"]["image_png_b64"]))
 audit={"step":index,"skill":action.get("skill"),"physical_held_before":physical_before,"semantic_held_before":semantic_before}
 if before is not None:
  audit["restore_errors"]=compare(before,snapshot(self._backend))
  audit["physical_held_after"]=bool(any(getattr(self._backend._robot,"_ag_obj_in_hand",{}).values()))
  audit["semantic_held_after"]=self._session.world_state.held_object is not None
 with (out/"private_audit.jsonl").open("a") as f:f.write(json.dumps(audit)+"\n")
 return result
InteractiveSearchEnv.step=step
import serve_play
raise SystemExit(serve_play.main())
