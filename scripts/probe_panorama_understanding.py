"""Small frozen-image comprehension diagnostic; no robot actions or task hints."""
import argparse,base64,json,hashlib
from pathlib import Path
from rummagebench.agents.local_vlm import request_json

p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--endpoint',default='http://127.0.0.1:8091');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
if (a.out/'calls.jsonl').exists():raise SystemExit('choose a new output folder')
health=request_json(a.endpoint+'/health');root=Path('runs/harness_v3_pilot/qwen3-vl-4b/seed_0/T01_knife_fetch__L1_S1/episode')
question='''This is a visual comprehension test, not an action-planning task.
Return one JSON object with: projection (equirectangular or perspective),
left_right_edges_adjacent (boolean in 3D), top_direction (up/down/forward),
bottom_direction (up/down/forward),
landmarks: {double_wooden_door: {x: normalized horizontal center, direction: front/left/right/back},
refrigerator: {x: number, direction: front/left/right/back},
sofa: {x: number, direction: front/left/right/back}}.
Use null when uncertain. Directions refer to the camera reference heading.
Classify front within 45 degrees, left 45..135, right -45..-135, and back beyond 135.
Do not guess hidden objects. No movement or manipulation is requested.'''
calibration='''Camera calibration: the attached image is a full 360x180 degree equirectangular panorama.
Its center points forward, x=0.25 points left, x=0.75 points right, the seam is behind;
the top points up and bottom down. These describe the projection, not the scene contents.\n'''
rows=[]
for name,paths,prompt in [('unexplained',[root/'step_000.png'],question),('calibrated',[root/'step_000.png'],calibration+question),('paired',[root/'step_004.png',root/'step_005.png'],'''Image 1 is a full equirectangular panorama: center=forward, quarter-left=left, quarter-right=right, seam=back. Image 2 is a perspective detail from the SAME camera position at the SAME time. Return JSON with detail_direction (front/left/right/back), shared_landmark (describe one large object visible in both), robot_moved (boolean; does selecting this detail require moving the robot?). Do not plan actions.''')]:
 images=[dict(label=f'Image {i+1}',image_png_b64=base64.b64encode(path.read_bytes()).decode()) for i,path in enumerate(paths)]
 result=request_json(a.endpoint+'/generate',dict(model=health['model_id'],prompt=prompt,images=images,image_png_b64=images[0]['image_png_b64'],max_new_tokens=1024,seed=0),timeout=300)
 row=dict(condition=name,prompt=prompt,images=[dict(path=str(x),sha256=hashlib.sha256(x.read_bytes()).hexdigest()) for x in paths],**result)
 rows.append(row)
 with (a.out/'calls.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
 print(name,result['text'],flush=True)
(a.out/'metadata.json').write_text(json.dumps(dict(server=health,scope='one scene, two frozen panorama states; diagnostic not benchmark accuracy',calls=len(rows)),indent=2))
