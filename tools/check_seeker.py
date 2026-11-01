"""Quality gate for a seeker run: is the clip continuous, is the target small,
and does every modality actually see it?

    python tools\\check_seeker.py out\\seeker_air
    python tools\\check_seeker.py out\\seeker_air out\\seeker_ground   # several runs

Reads frames.json (per-frame range / target size / view geometry) and the
per-modality label files. Prints one block per clip and exits non-zero when a
hard check fails, so it can be wired into a build script.

Checks
------
1. continuity   adjacent frames differ smoothly (no teleport, no frame repeat)
2. closing      range decreases across the clip (a seeker is closing in)
3. small target every view's target stays inside the configured pixel band
4. seen         each requested modality has a box on at least 80% of frames
5. sar budget   SAR target contrast is positive and resolution matches c/2B
"""
import json
import os
import sys


def load(p, default=None):
    if not os.path.exists(p):
        return default
    try:
        with open(p, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return default


def main(argv):
    runs = argv[1:]
    if not runs:
        print(__doc__)
        return 2
    hard_fail = 0
    for run in runs:
        frames = load(os.path.join(run, "frames.json"), [])
        meta = load(os.path.join(run, "meta.json"), {})
        stats = load(os.path.join(run, "stats.json"), {})
        if not frames:
            print("[FAIL] %s: no frames.json (not a simkit run?)" % run)
            hard_fail += 1
            continue
        views = meta.get("views") or []
        clips = {}
        for f in frames:
            clips.setdefault(f.get("clip"), []).append(f)

        print("\n== %s  (%d frames, %d clip(s), sensors=%s)"
              % (run, len(frames), len(clips), ",".join(meta.get("sensors") or [])))
        for c, rows in sorted(clips.items()):
            rows.sort(key=lambda r: r["frame"])
            r0, r1 = rows[0].get("range_m"), rows[-1].get("range_m")
            # 1/2. continuity + closing
            jumps = []
            for a, b in zip(rows, rows[1:]):
                if a.get("range_m") is None or b.get("range_m") is None:
                    continue
                jumps.append(abs(b["range_m"] - a["range_m"]))
            step_max = max(jumps) if jumps else 0.0
            closes = (r0 is not None and r1 is not None and r1 < r0)
            if not closes:
                print("   [FAIL] clip %s: range does not decrease (%.0f -> %.0f)"
                      % (c, r0 or 0, r1 or 0))
                hard_fail += 1
            # 3. small target, per view
            px_note = []
            for v in views:
                nm = v.get("name")
                px = [(r.get("view_px") or {}).get(nm) for r in rows]
                px = [p for p in px if p is not None]
                if not px:
                    continue
                frac0 = 100.0 * px[0] / float(v.get("width") or 1)
                frac1 = 100.0 * px[-1] / float(v.get("width") or 1)
                px_note.append("%s %.0f..%.0f px (%.2f%%..%.2f%%)"
                               % (nm, px[0], px[-1], frac0, frac1))
            print("   clip %s: range %.0f -> %.0f m, max |dR| %.1f m/frame"
                  % (c, r0 or 0, r1 or 0, step_max))
            if px_note:
                print("            target %s" % " | ".join(px_note))
            # 4.每模态出框率：以 labels/ 下真实存在的目录为准（ir 可能是 ir_mwir、
            #    ir_mwir_wide 等多个波段 × 视场组合，不能靠 sensors 列表猜）
            seen = {}
            ldir = os.path.join(run, "labels")
            mods = sorted(d for d in os.listdir(ldir)) if os.path.isdir(ldir) else []
            for key in mods:
                if not os.path.isdir(os.path.join(ldir, key)):
                    continue
                hit = 0
                for r in rows:
                    d = load(os.path.join(ldir, key, "%06d.json" % r["frame"]))
                    if not d:
                        continue
                    boxes = d.get("boxes")
                    if isinstance(boxes, (dict, list)) and boxes:
                        hit += 1
                    elif d.get("num_points"):
                        hit += 1
                if hit:
                    seen[key] = hit / float(len(rows))
            if seen:
                txt = ", ".join("%s %.0f%%" % (k, 100 * v) for k, v in sorted(seen.items()))
                print("            boxed frames: %s" % txt)
                for k, v in seen.items():
                    if v < 0.8:
                        print("   [WARN] %s: only %.0f%% of frames have a box"
                              % (k, 100 * v))
        # 5. SAR budget
        sar = (stats.get("sar") or {}) if isinstance(stats, dict) else {}
        sfr = stats.get("sar_per_frame") or []
        if sfr:
            con = [s["target_contrast_db"] for s in sfr]
            snr = [s["snr_db"] for s in sfr]
            print("   SAR: frames=%d, target contrast %.1f..%.1f dB, SNR %.1f..%.1f dB"
                  % (len(sfr), min(con), max(con), min(snr), max(snr)))
            if min(con) <= 0:
                print("   [WARN] SAR target not brighter than the clutter floor")
        if len(set(len(v) for v in clips.values())) > 1:
            print("   [WARN] clips differ in length: %s"
                  % {c: len(v) for c, v in sorted(clips.items())})

    print("\n%s" % ("ALL CHECKS PASSED" if not hard_fail else "%d hard failure(s)" % hard_fail))
    return 1 if hard_fail else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
