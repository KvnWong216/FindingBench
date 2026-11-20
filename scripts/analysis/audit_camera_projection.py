"""Project disclosed executor footprint into a recorded ego image (audit only).

This is a ground-plane projection, not a visible contour or collision oracle.
No rendered-body visibility classification is inferred from RGB.
"""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/"runs/harness_v42_ondemand_pilot/qwen3-vl-4b/seed_0/T01_knife_fetch__L1_S1"
OUT=ROOT/"runs/harness_v42_review"
call=json.loads((SOURCE/"model_calls.jsonl").read_text().splitlines()[0])
obs=json.loads(call["prompt"].split("\nInput:\n")[-1])["observation"]
emb=obs["embodiment"]
forward=np.array(emb["body_forward_in_camera"])
up=np.array(emb["body_up_in_camera"])
left=np.cross(up,forward)
R=np.stack([forward,left,up],axis=1)
origin=np.array(emb["camera_origin_in_base_m"])
h=emb["executor_footprint_half_extent_m"]
boundary=[]
for a,b in [((-h,-h),(h,-h)),((h,-h),(h,h)),((h,h),(-h,h)),((-h,h),(-h,-h))]:
    for t in np.linspace(0,1,30):
        p=np.array([a[0]*(1-t)+b[0]*t,a[1]*(1-t)+b[1]*t,0.])
        c=R@(p-origin)
        yaw=np.arctan2(-c[0],c[2]);pitch=np.arcsin(-c[1]/np.linalg.norm(c))
        boundary.append([float(.5-yaw/(2*np.pi)),float(.5-pitch/np.pi)])
image=Image.open(SOURCE/"episode/step_000.png").convert("RGB")
d=ImageDraw.Draw(image)
for a,b in zip(boundary,boundary[1:]):
    if abs(a[0]-b[0])<.5:
        d.line((a[0]*image.width,a[1]*image.height,b[0]*image.width,b[1]*image.height),
               fill="#ffff00",width=5)
font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",25)
d.rectangle((0,0,image.width,90),fill="#172338")
d.text((16,12),"AUDIT ONLY: yellow = projected 0.8 m executor square on base-height plane",font=font,fill="white")
d.text((16,48),"Not an observed contour. Occlusion is NOT tested. Never supplied to the VLM.",font=font,fill="white")
image.save(OUT/"camera_footprint_projection.png")
result=dict(source=str(SOURCE),embodiment=emb,projected_boundary=boundary,
            vertical_range_normalized=[min(p[1] for p in boundary),max(p[1] for p in boundary)],
            orthogonality_max_error=float(np.abs(R.T@R-np.eye(3)).max()),
            supplied_to_model=False,visibility_tested=False,
            conclusion="Footprint projects into lower panorama. Whether RGB exposes it requires visual/segmentation evidence.")
(OUT/"camera_projection_audit.json").write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!="projected_boundary"},indent=2))
