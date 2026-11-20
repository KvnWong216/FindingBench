"""Renderer-only following spectator camera. No mesh, collider, or agent input."""
import numpy as np
from scipy.spatial.transform import Rotation


class FollowExo:
    def __init__(self, backend, output):
        from pathlib import Path
        from omnigibson.sensors import VisionSensor
        self.backend, self.output = backend, Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.camera = VisionSensor(relative_prim_path="/diagnostic_follow_exo",
                                  name="diagnostic_follow_exo", modalities=["rgb"],
                                  image_width=960, image_height=640,
                                  focal_length=10, horizontal_aperture=24, clipping_range=(.01, 1000))
        self.camera.load(backend._env.scene)
        self.camera.initialize()
        self.records = []

    def capture(self, step):
        import json
        from PIL import Image
        backend = self.backend
        before = backend.robot_pose()
        base = np.asarray(before[0])
        # Follow translation; retain a stable world-relative azimuth. Camera
        # sits below the ceiling and above tables. It never controls the body.
        eye = base + [1.4, -.8, 1.95]
        target = base + [0, 0, .85]
        back = eye-target
        back /= np.linalg.norm(back)
        right = np.cross([0, 0, 1], back)
        right /= np.linalg.norm(right)
        up = np.cross(back, right)
        quat = Rotation.from_matrix(np.stack([right, up, back], axis=1)).as_quat()
        self.camera.set_position_orientation(position=eye, orientation=quat)
        for _ in range(6):
            backend._sim.render()
        self.camera.get_obs()
        backend._sim.render()
        data, _ = self.camera.get_obs()
        rgb = data["rgb"].cpu().numpy() if hasattr(data["rgb"], "cpu") else np.asarray(data["rgb"])
        Image.fromarray(rgb[..., :3].astype(np.uint8)).save(self.output / f"step_{step:03d}_exo.png")
        after = backend.robot_pose()
        if not np.allclose(before[0], after[0], atol=1e-7) or not np.allclose(before[1], after[1], atol=1e-7):
            raise RuntimeError("Spectator rendering changed the robot pose")
        self.records.append(dict(step=step, camera_position=eye.tolist(), camera_quaternion=quat.tolist(),
                                 robot_position=list(before[0]), no_physics_step=True, supplied_to_model=False))
        (self.output / "camera.json").write_text(json.dumps(self.records, indent=2))
