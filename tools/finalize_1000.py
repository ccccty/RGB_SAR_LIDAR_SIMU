"""Stitch the ds1000 chunk runs, then merge them into one COCO dataset."""
import json, os, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from simkit import dataset

RUNS = os.path.join(ROOT, "out", "ds1000")
SCENES = [("ds_air_uav", "scenes/ds_air_uav.json", 360),
          ("ds_ground_vehicle", "scenes/ds_ground_vehicle.json", 340),
          ("ds_sea_boat", "scenes/ds_sea_boat.json", 300)]

print("== merging chunks")
t0 = time.time()
for name, cfg, n in SCENES:
    d = os.path.join(RUNS, name)
    if not os.path.isdir(d):
        print("  missing", d); continue
    parts = [f for f in os.listdir(d) if ".part" in f and f.endswith(".json")]
    dataset.merge_parts(d, os.path.join(ROOT, cfg), n_frames=n,
                        sensors=("optical", "ir", "radar", "lidar"), quiet=False)
    for f in parts:
        os.remove(os.path.join(d, f))
    print("  %-20s parts=%d" % (name, len(parts)))

print("== merging runs into one dataset (%.1fs)" % (time.time() - t0))
out = os.path.join(ROOT, "out", "dataset_1000")
r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "merge_dataset.py"),
                    out] + [os.path.join(RUNS, s[0]) for s in SCENES],
                   capture_output=True, text=True, encoding="utf-8")
print(r.stdout.strip() or r.stderr[-1500:])
