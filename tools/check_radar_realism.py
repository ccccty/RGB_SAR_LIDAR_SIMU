"""Inspect one frame using the measured RD detector without changing scene files.

Example: python tools/check_radar_realism.py --scene out/aviation_baseline_v2/airliner_parallel.json
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simkit import plan, radar, scene


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    cfg = scene.load_config(args.scene)
    frames = int(cfg.get("frames", {}).get("n", 1))
    if not (0 <= args.frame < frames):
        ap.error("frame outside configured run")
    cfg.setdefault("sensors", {}).setdefault("radar", {})["detector"] = "rd_threshold"
    state = next(plan.plan_frames(cfg, 1, cfg.get("seed", 0), cfg.get("frames", {}).get("dt", 0.2),
                                 frame_offset=args.frame, total_frames=frames))
    out = radar.simulate(scene.build_world(state["cfg"], jitter=state["jitter"]), state["cfg"])
    associated = [d for d in out["detections"] if d["obj"] >= 0]
    unassociated = [d for d in out["detections"] if d["obj"] < 0]
    report = dict(scene=args.scene, frame=args.frame,
                  rd_shape_range_doppler=list(out["rd_map"].shape),
                  target_detections=len(associated), unassociated_detections=len(unassociated),
                  processing=out["processing"], detections=out["detections"],
                  finite_power=bool(np.isfinite(out["rd_map"]).all()),
                  real_instrument_validation=False,
                  caveat="Unassociated peaks may be thermal noise or configured artificial clutter; "
                         "thermal Pfa does not calibrate artificial clutter")
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
