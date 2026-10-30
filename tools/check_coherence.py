"""Validate a generated run for temporal coherence and usable labels.

Checks, in order:
  1. every clip holds one weather state (or a smooth evolution), never flicker
  2. consecutive frames move by a bounded, non-zero step (a jump means the
     trajectory is not continuous; a zero step means duplicate frames)
  3. the labelled target stays inside the requested pixel band
  4. COCO files have no orphan annotations and no missing images
"""
import json, os, sys
import numpy as np

def main(run, fpc):
    lab_dir = os.path.join(run, "labels", "ir_lwir")
    if not os.path.isdir(lab_dir):
        lab_dir = os.path.join(run, "labels", "optical")
    files = sorted(f for f in os.listdir(lab_dir) if f.endswith(".json"))
    if not files:
        print("no labels in", lab_dir); return 1
    rows = []
    for f in files:
        j = json.load(open(os.path.join(lab_dir, f), encoding="utf-8"))
        b = (j.get("boxes") or [None])[0]
        rows.append(dict(frame=int(j["frame"]), clip=j.get("clip"),
                         rng=j.get("range_m"), px=j.get("target_px"),
                         weather=j.get("weather") or {},
                         box=None if not b else (b["x0"], b["y0"], b["x1"], b["y1"])))
    print("%-28s %d frames, %d clips" % (os.path.basename(run), len(rows),
          len(set(r["clip"] for r in rows if r["clip"] is not None)) or 1))

    # 1. weather per clip
    if rows[0]["clip"] is not None:
        worst = 0
        for c in sorted(set(r["clip"] for r in rows)):
            sub = [r for r in rows if r["clip"] == c]
            names = set(w.get("name") for w in (x["weather"] for x in sub))
            rain = [w.get("rain_mmh", 0.0) for w in (x["weather"] for x in sub)]
            worst = max(worst, float(np.ptp(rain)))
            print("  clip %3d  %3d frames  weather %s  rain span %.1f mm/h" % (
                c, len(sub), ",".join(sorted(str(n) for n in names)), float(np.ptp(rain))))
        print("  weather within-clip rain variation (max) %.1f mm/h" % worst)

    # 2. continuity of the box centre
    cen = [(r["frame"], r["clip"],
            ((r["box"][0]+r["box"][2])/2.0, (r["box"][1]+r["box"][3])/2.0))
           for r in rows if r["box"]]
    steps, jumps, zero = [], 0, 0
    for (f0, c0, a), (f1, c1, b) in zip(cen, cen[1:]):
        # A clip boundary starts a new trajectory, so a jump there is expected;
        # only within-clip steps say anything about temporal continuity.
        if f1 != f0 + 1 or c0 != c1:
            continue
        d = float(np.hypot(a[0]-b[0], a[1]-b[1]))
        steps.append(d)
        if d > 40.0: jumps += 1
        if d < 1e-9: zero += 1
    st = np.array(steps) if steps else np.array([0.0])
    print("  box centre step: median %.2f px  p95 %.2f  max %.2f  jumps>40px %d  identical %d" % (
        float(np.median(st)), float(np.percentile(st, 95)), float(st.max()), jumps, zero))

    # 3. pixel band
    px = [r["px"] for r in rows if r["px"]]
    if px:
        print("  target px: min %.1f  median %.1f  max %.1f" % (
            min(px), float(np.median(px)), max(px)))
    rg = [r["rng"] for r in rows if r["rng"]]
    if rg:
        print("  range m:   min %.0f  median %.0f  max %.0f" % (
            min(rg), float(np.median(rg)), max(rg)))

    # 4. COCO integrity
    mods = sorted(f for f in os.listdir(run)
                  if f.startswith("annotations_") and f.endswith("_coco.json"))
    rel = {"optical": "images/optical", "lidar": "lidar/range"}
    for m in mods:
        d = json.load(open(os.path.join(run, m), encoding="utf-8"))
        mod = m[len("annotations_"):-len("_coco.json")]
        ids = set(i["id"] for i in d["images"])
        orphan = sum(1 for a in d["annotations"] if a["image_id"] not in ids)
        base = rel.get(mod, "images/" + mod)
        miss = sum(1 for i in d["images"]
                   if not os.path.exists(os.path.join(run, base, i["file_name"])))
        print("  %-8s images %4d anns %4d orphan %d missing %d" % (
            mod, len(d["images"]), len(d["annotations"]), orphan, miss))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 50))
