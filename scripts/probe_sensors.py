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
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modalities", default="rgb",
                        help="comma list, e.g. rgb,depth_linear,seg_instance")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    mods = [m.strip() for m in args.modalities.split(",") if m.strip()]

    from rummagebench.core.scenario import load_scenario
    from rummagebench.sim.omnigibson.backend import OmniGibsonBackend

    scenario = load_scenario(REPO / "scenarios" / "knife_search_001" / "scenario.yaml")
    scenario.robot.obs_modalities = list(mods)
    backend = OmniGibsonBackend(seed=args.seed)
    t0 = time.time()
    backend.setup(scenario)
    print(f"[probe] setup ok in {time.time()-t0:.1f}s mods={mods}", flush=True)

    frame = None
    for i in range(args.warmup):
        frame = backend.capture_visual_frame()
    print(f"[probe] warmup done ({args.warmup} frames)", flush=True)

    report = {"modalities": mods, "frames": [], "convention": None}
    failures = []
    t0 = time.time()
    for i in range(args.frames):
        t_frame = time.time()
        try:
            frame = backend.capture_visual_frame()
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
            "seg_nonzero_px": int((seg != 0).sum()),
            "labels": len(getattr(backend, "_instance_labels", {})),
            "K_finite": bool(np.isfinite(K).all() and K[0, 0] > 0),
            "pose_finite": bool(np.isfinite(T).all()),
            "convention": frame.depth_convention,
            "bridge": frame.meta.get("bridge"),
            "dt_ms": round((time.time() - t_frame) * 1000, 1),
        }
        report["frames"].append(entry)
        ok = (entry["depth_finite_px"] == entry["depth_total_px"]
              and entry["seg_nonzero_px"] > 0 and entry["labels"] > 0
              and entry["K_finite"] and entry["pose_finite"])
        if not ok:
            failures.append({"frame": i, "entry": entry})
        if i % 10 == 0 or not ok:
            print(f"[probe] frame {i}: {OK if ok else BAD} {entry}", flush=True)
    report["elapsed_s"] = round(time.time() - t0, 1)
    report["failures"] = failures
    report["ok"] = not failures
    out = json.dumps(report, indent=2)
    print(out, flush=True)
    if args.out:
        Path(args.out).write_text(out)
    return 0 if not failures else 3


if __name__ == "__main__":
    raise SystemExit(main())
