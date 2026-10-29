"""ASCII preview of one image region so a text-only agent can inspect a render."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from png_stats import read_png, luma

RAMP = ' .:-=+*#%@'


def load(path):
    meta, pix, _pal = read_png(path)
    vals, w, h = luma(meta, pix)
    return np.asarray(vals, np.float64).reshape(h, w)


def main():
    path = sys.argv[1]
    cols = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    box = sys.argv[3] if len(sys.argv) > 3 else None
    img = load(path)
    h, w = img.shape
    if box:
        x0, y0, x1, y1 = [int(v) for v in box.split(',')]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        img = img[y0:y1, x0:x1]
    h, w = img.shape
    rows = max(4, int(cols * h / w * 0.5))
    ys = np.linspace(0, h - 1, rows).astype(int)
    xs = np.linspace(0, w - 1, cols).astype(int)
    sub = img[np.ix_(ys, xs)]
    lo, hi = np.percentile(img, 2), np.percentile(img, 98)
    n = len(RAMP) - 1
    print('%s  %dx%d  luma %.1f..%.1f' % (os.path.basename(path), w, h, lo, hi))
    for r in sub:
        print(''.join(RAMP[min(n, max(0, int((v - lo) / max(hi - lo, 1e-6) * n)))] for v in r))


main()
