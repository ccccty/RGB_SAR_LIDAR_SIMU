"""Contrast check: is the target actually visible inside its label box?"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from png_stats import read_png, luma  # noqa: E402


def check(img, lab):
    meta, pix, pal = read_png(img)
    vals, w, h = luma(meta, pix)
    lb = json.load(open(lab, encoding="utf-8"))
    boxes = lb.get("boxes") or {}
    if isinstance(boxes, list):
        boxes = dict((b.get("name", "obj%d" % i), b) for i, b in enumerate(boxes))
    mean_all = sum(vals) / len(vals)
    rows = []
    for name, b in boxes.items():
        if not isinstance(b, dict) or "x0" not in b:
            continue
        x0, y0 = int(b["x0"]), int(b["y0"])
        x1, y1 = int(round(b["x1"])), int(round(b["y1"]))
        acc, n = 0.0, 0
        for y in range(max(0, y0), min(h, y1)):
            base = y * w
            for x in range(max(0, x0), min(w, x1)):
                acc += vals[base + x]
                n += 1
        if n:
            m = acc / n
            rows.append((name, m, mean_all, m - mean_all, n))
    return rows


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "out/air"
    for sub, labdir in (("images/optical", "labels/optical"),
                        ("images/ir_preview_lwir", "labels/ir_lwir"),
                        ("images/ir_preview_mwir", "labels/ir_mwir")):
        d_img, d_lab = os.path.join(root, sub), os.path.join(root, labdir)
        if not os.path.isdir(d_img):
            continue
        print("==", sub)
        for fn in sorted(os.listdir(d_img))[:4]:
            lab = os.path.join(d_lab, fn.replace(".png", ".json"))
            if not os.path.exists(lab):
                continue
            for name, inside, scene, delta, n in check(os.path.join(d_img, fn), lab):
                print("  %s %-7s inside %6.1f  scene %6.1f  delta %+6.1f  px=%d"
                      % (fn, name, inside, scene, delta, n))


main()
