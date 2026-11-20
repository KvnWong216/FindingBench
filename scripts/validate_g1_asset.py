"""Native G1 import, rendered preview and 50-pose FK cross-validation."""
import os,json
from pathlib import Path
os.environ['OMNIGIBSON_HEADLESS']='True';os.environ['OMNIGIBSON_NO_OMNI_LOGS']='True'
import pinocchio as pin
import omnigibson as og
from omnigibson.macros import gm
gm.ENABLE_TRANSITION_RULES=False
from rummagebench.sim.omnigibson.env_factory import load_robot_config,load_env_config
cfg=load_env_config();cfg['scene']={'type':'Scene'};cfg['robots']=[load_robot_config('g1_bench')]
cfg['robots'][0].update(name='robot_0',position=[0,0,.01],orientation=[0,0,0,1])
env=og.Environment(configs=cfg);r=env.scene.robots[0]
print('ROBOT_LOADED',list(r.joints), r.aabb,flush=True)
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image
camera=og.sim.viewer_camera
position=np.array([2.,-2.,1.6]);forward=np.array([0,0,.7])-position;forward/=np.linalg.norm(forward)
right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right);up=np.cross(right,forward)
import torch
camera.set_position_orientation(position=torch.tensor(position,dtype=torch.float32),orientation=torch.tensor(Rotation.from_matrix(np.column_stack([right,up,-forward])).as_quat(),dtype=torch.float32))
for _ in range(5):og.sim.render()
rgb=camera.get_obs()[0]['rgb']
if hasattr(rgb,'cpu'):rgb=rgb.cpu().numpy()
Image.fromarray(np.asarray(rgb)[...,:3].astype(np.uint8)).save('build/robots/g1_bench/preview.png')
from rummagebench.sim.omnigibson.robot_export import export_robot_urdf
out=Path('build/robots/g1_bench/g1_bench_validated.urdf')
report=export_robot_urdf(r,out,eef_link='right_eef_link',base_link='base_link',validate_samples=50)
Path('build/robots/g1_bench/smoke.json').write_text(json.dumps(report,indent=2,default=str))
print('G1_SMOKE_PASSED',flush=True)
os._exit(0)
