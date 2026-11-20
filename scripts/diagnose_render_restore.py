"""Measure render freshness after OBSERVE without advancing physics."""
import json,time,traceback,base64
from pathlib import Path
import numpy as np
from PIL import Image
from agent_acceptance_driver import snapshot,compare
from rummagebench.adapters.python_api import create_session
out=Path("runs/server_acceptance/debug_20260930_takeover/render_restore_2")
out.mkdir(exist_ok=True);report={"ok":False,"frames":[]};session=None
def save(): (out/"report.json").write_text(json.dumps(report,indent=2))
try:
 session=create_session("scenarios/knife_search_001/scenario.yaml",run_dir=out/"private",mode="agent")
 session.reset();b=session._backend
 for i in range(8):base=b.capture_visual_frame()
 session._capture_current()
 public=session.observe()
 Image.fromarray(base.rgb).save(out/"baseline.png")
 (out/"ready.json").write_text(json.dumps({"frame_id":public.frame_id,"planning_step":public.planning_step}))
 from diagnose_sofa_pair import audit
 try:
  (out/"private_collision_pairs.json").write_text(json.dumps(audit(b),indent=2))
 except Exception as e:
  (out/"private_collision_pairs.json").write_text(json.dumps({"error":repr(e)}))
 state=snapshot(b)
 deadline=time.monotonic()+180
 while not (out/"action.json").exists():
  if time.monotonic()>deadline:raise TimeoutError("waiting for public RGB action")
  time.sleep(.2)
 result=session.step(json.loads((out/"action.json").read_text()))
 report["feedback"]=result.feedback.model_dump();report["restore_errors"]=compare(state,snapshot(b))
 returned=session._store.get(result.observation.frame_id)
 finite=np.isfinite(base.depth)&np.isfinite(returned.depth)
 report["returned_frame"]={"seg_same_fraction":float((base.instance_segmentation==returned.instance_segmentation).mean()),"depth_mae":float(np.abs(base.depth[finite]-returned.depth[finite]).mean()),"finite_mask_same":float((np.isfinite(base.depth)==np.isfinite(returned.depth)).mean()),"pose_delta_max":float(np.max(np.abs(base.camera_extrinsics-returned.camera_extrinsics)))}
 (out/"returned.png").write_bytes(base64.b64decode(result.observation.image_png_b64))
 save()
 for i in range(9):
  frame=b.capture_visual_frame()
  finite=np.isfinite(base.depth)&np.isfinite(frame.depth)
  row={"index":i,"rgb_mae":float(np.abs(base.rgb.astype(float)-frame.rgb.astype(float)).mean()),"seg_same_fraction":float((base.instance_segmentation==frame.instance_segmentation).mean()),"finite_mask_same":float((np.isfinite(base.depth)==np.isfinite(frame.depth)).mean()),"depth_mae":float(np.abs(base.depth[finite]-frame.depth[finite]).mean()),"pose_delta_max":float(np.max(np.abs(base.camera_extrinsics-frame.camera_extrinsics)))}
  report["frames"].append(row);Image.fromarray(frame.rgb).save(out/f"after_{i}.png");save();print("[render_restore]",row,flush=True)
 b.validate_physics_state();report["ok"]=True
except Exception as e:report["error"]=repr(e);traceback.print_exc()
finally:
 save()
 if session is not None:session._backend.close()
