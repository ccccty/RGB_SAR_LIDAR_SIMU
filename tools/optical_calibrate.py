"""Optical calibration: match simulated sky luma to real-photo sky luma."""
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from png_stats import read_png, luma  # noqa: E402
from simkit import dataset  # noqa: E402


def sky_luma(path):
    meta, pix, pal = read_png(path)
    vals, w, h = luma(meta, pix)
    band = max(1, h // 4) * w
    top = vals[:band]
    s = sorted(vals)
    return sum(top) / len(top), s[int(len(s) * 0.5)], s[int(len(s) * 0.95)]


def main():
    exp = float(sys.argv[1]) if len(sys.argv) > 1 else 0.35
    base = json.load(open(os.path.join(ROOT, "scenes", "demo_air_uav.json"), encoding="utf-8"))
    base["frames"] = {"n": 3, "dt": 0.3}
    base["weather"] = {"states": [{"preset": "clear"}, {"preset": "cloudy"}, {"preset": "rain"}]}
    base["objects"][0]["model"] = os.path.join(ROOT, "assets", "models", "uav_demo.obj")
    base.setdefault("optical", {})["exposure"] = exp
    base["optical"]["samples"] = 12
    tmp = os.path.join(ROOT, "out", "_cal")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)
    p = os.path.join(tmp, "scene.json")
    json.dump(base, open(p, "w", encoding="utf-8"), indent=1)
    out = os.path.join(tmp, "run")
    dataset.run(p, out, n_frames=3, sensors=["optical"], seed=1, quiet=True)
    print("exposure %+.2f" % exp)
    for i, name in enumerate(("clear", "cloudy", "rain")):
        sky, med, p95 = sky_luma(os.path.join(out, "images", "optical", "%06d.png" % i))
        print("  %-7s sky %5.1f  median %5.0f  p95 %5.0f" % (name, sky, med, p95))
    print("  reference photos: sky 135/150/161/191/226 (median 161)")


main()
