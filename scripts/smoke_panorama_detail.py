"""Transport diagnostic, NOT a scored/model-controlled episode."""
import base64,json,os,sys
from pathlib import Path
from rummagebench.environment.env import InteractiveSearchEnv
from rummagebench.agents.humanclaw_vlm import HumanCLAWPolicy
from rummagebench.agents.local_vlm import request_json
out=Path('runs/panorama_understanding_v1/live_smoke');out.mkdir(parents=True,exist_ok=False)
try:
 env=InteractiveSearchEnv('build/vla_difficulty_v1/T01_knife_fetch__L1_S1/scenario.yaml')
 session=env._session;session.expose_embodiment=True
 initial=session.reset()
 pose=env._backend.robot_pose()
 current=session.step({'skill':'VIEW','yaw':90,'pitch':0}).observation
 assert current.image_png_b64==initial.image_png_b64
 assert current.frame_id==initial.frame_id
 assert current.detail_view is not None
 assert current.detail_view.frame_id!=current.frame_id
 assert env._backend.robot_pose()==pose
 (out/'panorama.png').write_bytes(base64.b64decode(current.image_png_b64))
 (out/'detail.png').write_bytes(base64.b64decode(current.detail_view.image_png_b64))
 health=request_json('http://127.0.0.1:8091/health')
 assert health.get('image_protocol')=='panorama-plus-detail-v4'
 policy=HumanCLAWPolicy('http://127.0.0.1:8091',health['model_id'],out/'model_calls.jsonl',max_new_tokens=2048)
 action=policy(current.model_dump(),[])
 (out/'result.json').write_text(json.dumps(dict(status='passed',purpose='interface smoke, not model performance',scripted_action={'skill':'VIEW','yaw':90},panorama_frame_id=current.frame_id,detail_frame_id=current.detail_view.frame_id,model_proposed_action_not_executed=action),indent=2))
 print('PASS: live panorama+detail delivered; proposed action not executed',flush=True)
except Exception as e:
 import traceback
 traceback.print_exc();(out/'error.json').write_text(json.dumps({'error':str(e)}));sys.stdout.flush();sys.stderr.flush();os._exit(1)
sys.stdout.flush();sys.stderr.flush();os._exit(0)
