"""Inspect one frame using the measured RD detector without changing scene files.

The RD image is one display channel (one aperture on the sector centre), so the
noise statistics reported here are those of a single exponential field and the
detector gate is -ln(Pfa) on that channel -- not the maximum over the scanned
beams, which would put the gate ~12 dB high and make the map look flat.

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
    # rd_map is one display channel, so its cells are exponential draws and the
    # percentiles below are the noise floor's shape, not the maximum over the
    # scanned beams: a 10-90 spread near 13.4 dB is a correct single channel,
    # 1-2 dB means the map is still an extreme-value statistic and every
    # threshold quoted from it is ~12 dB off.
    rd = np.asarray(out["rd_map"], dtype=float)
    db = radar.to_db(rd)
    cell_noise = float(out["processing"].get("single_beam_fft_noise_power") or 0.0)
    noise = dict(p10_db=round(float(np.percentile(db, 10)), 2),
                 p50_db=round(float(np.percentile(db, 50)), 2),
                 p90_db=round(float(np.percentile(db, 90)), 2),
                 p99_db=round(float(np.percentile(db, 99)), 2),
                 p10_p90_spread_db=round(float(np.percentile(db, 90) - np.percentile(db, 10)), 2),
                 exponential_spread_db=13.45,
                 peak_snr_db=(round(float(10.0 * np.log10(rd.max() / cell_noise)), 2)
                              if cell_noise > 0.0 else None))
    report = dict(scene=args.scene, frame=args.frame,
                  rd_shape_range_doppler=list(out["rd_map"].shape),
                  rd_noise=noise,
                  target_detections=len(associated), unassociated_detections=len(unassociated),
                  processing=out["processing"], detections=out["detections"],
                  finite_power=bool(np.isfinite(out["rd_map"]).all()),
                  real_instrument_validation=False,
                  caveat="Unassociated peaks may be thermal noise or configured artificial clutter; "
                         "thermal Pfa does not calibrate artificial clutter. The RD map is a single "
                         "display channel, so an unassociated peak is a noise draw or clutter, not a "
                         "second beam's view of the same cell")
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
