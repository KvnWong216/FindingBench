"""Independent diagnostic replay, never a scored run or model observation.

Uses the completed Qwen reference trace. Requires the GPU to be free of the suite.
An external RGB camera observes the replay; original ego images remain untouched.
"""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main():
    import numpy as np
    import torch
    from PIL import Image
    from scipy.spatial.transform import Rotation
    from rummagebench.environment.env import InteractiveSearchEnv
    from reason_feedback_session import ReasonFeedbackSession
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ROOT / "runs/harness_v42_review/exo_replay")
    p.add_argument("--steps", type=int, default=5)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    source = ROOT / "runs/harness_v42_ondemand_pilot/qwen3-vl-4b/seed_0/T01_knife_fetch__L1_S1"
    events = [json.loads(line) for line in (source / "episode/events.jsonl").read_text().splitlines()]
    events = [r for r in events if r.get("event_type") == "action"][:a.steps]
    manifest = dict(status="initializing", scored=False, source=str(source),
                    note="Separate seed-0 replay, not original exo recording. No VLM receives this camera.",
                    steps=[], camera="fixed external perspective RGB")
    def save():
        (a.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    save()
    try:
        env = InteractiveSearchEnv(ROOT / "build/vla_difficulty_v1/T01_knife_fetch__L1_S1/scenario.yaml", seed=0)
        session = env._session
        session.__class__ = ReasonFeedbackSession
        session.set_planning_budget(100)
        session.set_trace(a.out / "replay_events.jsonl")
        session.reset()
        backend = env._backend
        camera = backend._sim.viewer_camera
        if camera is None:
            # Headless evaluation deliberately disables the viewer. Add an
            # independent RGB-only diagnostic sensor in this replay process.
            from omnigibson.sensors import VisionSensor
            camera = VisionSensor(relative_prim_path="/diagnostic_exo",
                                  name="diagnostic_exo", modalities=["rgb"],
                                  image_width=1200, image_height=650,
                                  focal_length=12, horizontal_aperture=24,
                                  clipping_range=(.01, 1000))
            camera.load(backend._env.scene)
            camera.initialize()
        eye = np.array([2.5, -6.0, 2.65])
        target = np.array([.5, -8.0, .8])
        back = eye-target
        back /= np.linalg.norm(back)
        right = np.cross([0, 0, 1], back)
        right /= np.linalg.norm(right)
        up = np.cross(back, right)
        quat = Rotation.from_matrix(np.stack([right, up, back], axis=1)).as_quat()
        camera.set_position_orientation(position=torch.tensor(eye, dtype=torch.float32),
                                        orientation=torch.tensor(quat, dtype=torch.float32))
        manifest["camera_pose"] = dict(position=eye.tolist(), orientation_xyzw=quat.tolist())
        for e in events:
            step = e["planning_step"]
            pos = backend.robot_pose()[0]
            error = float(np.linalg.norm(np.asarray(pos)-np.asarray(e["robot_pose_before"])))
            if error > .02:
                raise RuntimeError(f"Replay pose diverged before step {step}: {error:.4f} m")
            for _ in range(3):
                backend._sim.render()
            camera.get_obs()
            backend._sim.render()
            obs, _ = camera.get_obs()
            rgb = obs["rgb"]
            if hasattr(rgb, "cpu"):
                rgb = rgb.cpu().numpy()
            image = Image.fromarray(np.asarray(rgb)[..., :3].astype(np.uint8))
            filename = f"step_{step-1:03d}_exo.png"
            image.save(a.out / filename)
            original = Image.open(source / "episode" / f"step_{step-1:03d}.png").convert("RGB")
            width = 1200
            original = original.resize((width, round(original.height*width/original.width)))
            image = image.resize((width, round(image.height*width/image.width)))
            from PIL import ImageDraw
            combined = Image.new("RGB", (width, original.height+image.height+100), "#172338")
            d = ImageDraw.Draw(combined)
            d.text((16, 14), f"STEP {step} BEFORE ACTION | TOP: original ego panorama", fill="white")
            combined.paste(original, (0, 40))
            d.text((16, original.height+54), "BOTTOM: independent diagnostic replay / external RGB (not model input)", fill="white")
            combined.paste(image, (0, original.height+90))
            combined.save(a.out / f"step_{step-1:03d}_ego_exo.jpg")
            result = session.step(e["raw_action"])
            row = dict(step=step, pose_error_m=error, image=filename,
                       expected_feedback=e["feedback"], replay_feedback=result.feedback.code.value,
                       natural_feedback=result.feedback.message)
            manifest["steps"].append(row)
            save()
            if row["expected_feedback"] != row["replay_feedback"]:
                raise RuntimeError(f"Execution differs at step {step}; replay is not aligned.")
        manifest["status"] = "complete"
        save()
    except Exception as error:
        import traceback
        traceback.print_exc()
        manifest.update(status="error", error=f"{type(error).__name__}: {error}")
        save()
        return 1
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)  # OmniGibson teardown is unreliable on this headless host.
