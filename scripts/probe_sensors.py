#!/usr/bin/env python3
"""Isolated sensor-modality probe for the AGENT visual protocol (acceptance).

Boots one OmniGibsonBackend with the requested robot.obs_modalities override,
warms up, captures --frames consecutive frames and verifies per-frame
synchronization invariants (same-camera shapes, depth validity, instance
labels, intrinsics and world pose finiteness). Exits 0 on success, 3 on
capture failure, with a JSON summary on stdout. Designed to run under an
external `timeout` so hangs become failures, never silently ignored.
"""

import argparse
import json
import sys
import subprocess
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--modalities", default="rgb",
                        help="comma list, e.g. rgb,depth_linear,seg_instance")
    parser.add_argument("--defer-modalities", action="store_true",
                        help="Diagnostic: attach requested annotators after scene setup")
    parser.add_argument("--setup-stage", choices=("full", "spawn_only", "placements"),
                        default="full", help="Diagnostic setup ablation; non-full runs are NOT acceptance")
    parser.add_argument("--spawn-stopped", action="store_true",
                        help="Diagnostic: import task objects with simulation stopped")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include", default=None,
                        help="comma list of include_sensor_names substrings")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    mods = [m.strip() for m in args.modalities.split(",") if m.strip()]

    from rummagebench.core.scenario import load_scenario
    from rummagebench.sim.omnigibson.backend import OmniGibsonBackend

    scenario = load_scenario(REPO / "scenarios" / "knife_search_001" / "scenario.yaml")
    original_placements = None
    if args.setup_stage != "full":
        original_placements = list(scenario.placements)
        scenario.initial_states = {}
        if args.setup_stage == "spawn_only":
            scenario.placements = []
        print(f"[probe] DIAGNOSTIC ONLY setup_stage={args.setup_stage}", flush=True)
    scenario.robot.obs_modalities = ["rgb"] if args.defer_modalities else list(mods)
    if args.include:
        scenario.robot.include_sensor_names = [
            i.strip() for i in args.include.split(",") if i.strip()]
    print(f"[probe] include_sensor_names={scenario.robot.include_sensor_names}", flush=True)
    backend = (OmniGibsonBackend(seed=args.seed, spawn_stopped=True) if args.spawn_stopped
               else OmniGibsonBackend(seed=args.seed))
    if args.setup_stage == "spawn_only":
        seed_pose = backend._seed_pose_for
        def original_seed_pose(spec):
            # Preserve the full run's pre-placement spawn positions.
            placements = backend._scenario.placements
            backend._scenario.placements = original_placements
            try:
                return seed_pose(spec)
            finally:
                backend._scenario.placements = placements
        backend._seed_pose_for = original_seed_pose
    try:
        t0 = time.time()
        backend.setup(scenario)
        if args.defer_modalities:
            wanted = list(mods)
            if any(m in wanted for m in ("seg_instance", "seg_instance_id")):
                wanted = ["seg_semantic"] + wanted
            print("[probe] attaching modalities after setup", flush=True)
            for sensor in backend._robot.sensors.values():
                if "rgb" not in sensor.modalities:
                    continue
                for modality in wanted:
                    sensor.add_modality(modality)
            for _ in range(4):
                backend._sim.render()
            print("[probe] deferred modalities rendered", flush=True)
        print(f"[probe] setup ok in {time.time()-t0:.1f}s mods={mods}", flush=True)

        frame = None
        for i in range(args.warmup):
            frame = backend.capture_visual_frame()
        print(f"[probe] warmup done ({args.warmup} frames)", flush=True)

        report = {"modalities": mods, "frames": [], "convention": None,
                  "setup_stage": args.setup_stage, "acceptance_eligible": args.setup_stage == "full"}
        failures = []
        t0 = time.time()
        for i in range(args.frames):
            t_frame = time.time()
            try:
                frame = backend.capture_visual_frame()
                backend.validate_physics_state()
            except Exception as e:
                failures.append({"frame": i, "error": f"{type(e).__name__}: {e}"})
                print(f"[probe] frame {i}: CAPTURE_FAIL {e}", flush=True)
                continue
            import numpy as np

            depth = np.asarray(frame.depth)
            seg = np.asarray(frame.instance_segmentation)
            K = np.asarray(frame.camera_intrinsics)
            T = np.asarray(frame.camera_extrinsics)
            finite_depth = int(np.isfinite(depth).sum())
            entry = {
                "frame": i,
                "size": [frame.image_width, frame.image_height],
                "depth_finite_px": finite_depth,
                "depth_total_px": int(depth.size),
                # A no-hit ray has +inf depth and background instance 0.
                # Preserve it verbatim; it cannot ground an agent point.
                "no_hit_background_px": int((np.isposinf(depth) & (seg == 0)).sum()),
                "invalid_depth_px": int(((~np.isfinite(depth) & ~(np.isposinf(depth) & (seg == 0)))
                                         | (np.isfinite(depth) & (depth <= 0))).sum()),
                "seg_nonzero_px": int((seg != 0).sum()),
                "labels": len(getattr(backend, "_instance_labels", {})),
                "K_finite": bool(np.isfinite(K).all() and K[0, 0] > 0),
                "pose_finite": bool(np.isfinite(T).all()),
                "convention": frame.depth_convention,
                "bridge": frame.meta.get("bridge"),
                "sensor_name": frame.meta.get("sensor_name"),
                "world_state_finite": True,
                "dt_ms": round((time.time() - t_frame) * 1000, 1),
            }
            report["frames"].append(entry)
            ok = (entry["invalid_depth_px"] == 0 and entry["depth_finite_px"] > 0
                  and entry["seg_nonzero_px"] > 0 and entry["labels"] > 0
                  and entry["K_finite"] and entry["pose_finite"])
            if not ok:
                failures.append({"frame": i, "entry": entry})
            if i % 10 == 0 or not ok:
                print(f"[probe] frame {i}: {'OK' if ok else 'BAD'} {entry}", flush=True)
        report["elapsed_s"] = round(time.time() - t0, 1)
        report["failures"] = failures
        report["ok"] = not failures
        out = json.dumps(report, indent=2)
        print(out, flush=True)
        if args.out:
            Path(args.out).write_text(out)
        return 0 if not failures else 3
    finally:
        print("[probe] closing backend", flush=True)
        backend.close()


def supervise(argv):
    """Keep acceptance status independent of native Kit shutdown's exit code."""
    with tempfile.TemporaryDirectory(prefix="findingbench-probe-") as directory:
        report_path = Path(directory) / "report.json"
        child_args = list(argv)
        requested_out = None
        for i, arg in enumerate(child_args):
            if arg == "--out":
                requested_out = Path(child_args[i + 1])
                del child_args[i:i + 2]
                break
            if arg.startswith("--out="):
                requested_out = Path(arg.split("=", 1)[1])
                del child_args[i]
                break
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker",
             *child_args, "--out", str(report_path)], check=False)
        if result.returncode != 0:
            return result.returncode if result.returncode > 0 else 128 - result.returncode
        if not report_path.exists():
            print("[probe] INVALID: worker exited without a completed report", file=sys.stderr)
            return 1
        report = json.loads(report_path.read_text())
        if requested_out is not None:
            requested_out.write_text(report_path.read_text())
        return 0 if report.get("ok") is True else 3


if __name__ == "__main__":
    raise SystemExit(main() if "--worker" in sys.argv else supervise(sys.argv[1:]))
