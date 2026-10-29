"""Build a contact sheet from a generated dataset (one PNG you can eyeball).

    python tools\\contact_sheet.py out\\final_air out\\sheet_air.png [cols]

One row per modality that actually produced images, so adding a sensor does not
require touching this file. Images are letterboxed rather than cropped because
the modalities have wildly different aspect ratios (radar range/angle is tall).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from simkit import png as pnglib
from png_stats import read_png, luma

MODS = [
    ("可见光", "images/optical"),
    ("MWIR", "images/ir_preview_mwir"),
    ("LWIR", "images/ir_preview_lwir"),
    ("雷达 距离-角度", "images/radar_ra"),
    ("激光 距离", "lidar/range"),
    ("激光 强度", "lidar/intensity"),
]


def load_gray(path):
    meta, pix, _pal = read_png(path)
    vals, w, h = luma(meta, pix)
    return np.asarray(vals, np.float64).reshape(h, w)


def fit(im, w, h):
    """Nearest-neighbour letterbox into a w x h tile."""
    ih, iw = im.shape
    s = min(w / float(iw), h / float(ih))
    dw, dh = max(1, int(iw * s)), max(1, int(ih * s))
    ys = (np.arange(dh) / s).astype(int).clip(0, ih - 1)
    xs = (np.arange(dw) / s).astype(int).clip(0, iw - 1)
    small = im[np.ix_(ys, xs)]
    tile = np.full((h, w), 16.0)
    y0, x0 = (h - dh) // 2, (w - dw) // 2
    tile[y0:y0 + dh, x0:x0 + dw] = small
    return tile


def main():
    out_dir = sys.argv[1]
    out_png = sys.argv[2]
    cols = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    rows = []
    for label, rel in MODS:
        d = os.path.join(out_dir, rel)
        if not os.path.isdir(d):
            continue
        files = sorted(f for f in os.listdir(d) if f.endswith(".png"))[:cols]
        if not files:
            continue
        rows.append((label, [load_gray(os.path.join(d, f)) for f in files]))
    if not rows:
        print("no images found")
        return 1
    tile_w, tile_h = 320, 300
    sheet = np.full((tile_h * len(rows), tile_w * cols), 24.0)
    for r, (_label, imgs) in enumerate(rows):
        for c, im in enumerate(imgs):
            sheet[r * tile_h:(r + 1) * tile_h, c * tile_w:(c + 1) * tile_w] = fit(im, tile_w, tile_h)
        # a 1 px separator keeps rows from blending together
        sheet[r * tile_h, :] = 70.0
    pnglib.write_gray8(out_png, np.clip(sheet, 0, 255).astype(np.uint8))
    print("wrote %s  %dx%d  rows=%s" % (out_png, sheet.shape[1], sheet.shape[0],
                                        ",".join(n for n, _ in rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
