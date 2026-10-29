"""Build a contact sheet from a generated dataset (one PNG you can eyeball)."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from simkit import png as pnglib
from png_stats import read_png, luma


def load_gray(path):
    meta, pix, _pal = read_png(path)
    vals, w, h = luma(meta, pix)
    return np.asarray(vals, np.float64).reshape(h, w)


def main():
    out_dir = sys.argv[1]
    out_png = sys.argv[2]
    cols = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    rows = []
    for sub in ("optical", "ir_preview_mwir", "ir_preview_lwir"):
        d = os.path.join(out_dir, "images", sub)
        if not os.path.isdir(d):
            continue
        files = sorted(f for f in os.listdir(d) if f.endswith(".png"))[:cols]
        if not files:
            continue
        rows.append((sub, [load_gray(os.path.join(d, f)) for f in files]))
    if not rows:
        print("no images found")
        return
    h = min(min(i.shape[0] for i in imgs) for _, imgs in rows)
    w = min(min(i.shape[1] for i in imgs) for _, imgs in rows)
    sheet = np.full((h * len(rows), w * cols), 24.0)
    for r, (_, imgs) in enumerate(rows):
        for c, im in enumerate(imgs):
            sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = im[:h, :w]
    pnglib.write_gray8(out_png, np.clip(sheet, 0, 255).astype(np.uint8))
    print("wrote %s  %dx%d  rows=%s" % (out_png, sheet.shape[1], sheet.shape[0],
                                        ",".join(n for n, _ in rows)))


main()
