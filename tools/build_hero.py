"""从最新演示运行重建首屏大图（assets/hero/）。"""
import json, os
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERO = os.path.join(ROOT, "assets", "hero")
W, H = 1280, 960


def frame(run, modality, n=0):
    p = os.path.join(ROOT, "out", run, "images", modality, "%06d.png" % n)
    if not os.path.isfile(p):
        return None
    a = np.asarray(Image.open(p)).astype(float)
    if a.ndim == 3:
        a = a.mean(axis=2)
    lo, hi = np.percentile(a, [1, 99])
    a = np.clip((a - lo) / max(hi - lo, 1) * 255, 0, 255)
    return Image.fromarray(a.astype(np.uint8)).convert("L").resize((W, H))


JOBS = [
    ("air_optical", "demo_air_headon", "optical", "空中 · 可见光", "air", "opt"),
    ("air_lwir", "demo_air_headon", "ir_preview_lwir", "空中 · 长波红外", "air", "ir"),
    ("ground_optical", "demo_ground_plate", "optical", "地面 · 可见光", "ground", "opt"),
    ("ground_lwir", "demo_ground_plate", "ir_preview_lwir", "地面 · 长波红外", "ground", "ir"),
    ("sea_optical", "demo_sea_plate", "optical", "海面 · 可见光", "sea", "opt"),
    ("sea_lwir", "demo_sea_plate", "ir_preview_lwir", "海面 · 长波红外", "sea", "ir"),
    ("ground_wx_clear", "demo_ground_plate", "optical", "地面 · 晴", "ground", "wx"),
    ("ground_golden", "demo_ground_golden", "optical", "地面 · 多云", "ground", "wx"),
    ("ground_rain", "demo_ground_rain", "optical", "地面 · 降雨", "ground", "wx"),
]


def main():
    os.makedirs(HERO, exist_ok=True)
    index = []
    for tag, run, mod, label, scene, group in JOBS:
        im = frame(run, mod)
        if im is None:
            print("  缺 %s / %s" % (run, mod))
            continue
        out = os.path.join(HERO, tag + ".webp")
        im.save(out, "WEBP", quality=86, method=6)
        size = os.path.getsize(out)
        index.append(dict(file=tag + ".webp", label=label, scene=scene,
                          group=group, w=W, h=H, bytes=size))
        print("  %-20s %6.1f KB  %s" % (tag + ".webp", size / 1024.0, label))
    with open(os.path.join(HERO, "index.json"), "w", encoding="utf-8") as fh:
        json.dump(index, fh, ensure_ascii=False, indent=1)
    print("  共 %d 张" % len(index))


if __name__ == "__main__":
    main()