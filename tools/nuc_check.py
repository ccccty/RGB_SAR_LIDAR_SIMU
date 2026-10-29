"""Check the IR fixed pattern: the same pixel noise must repeat across frames."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from png_stats import read_png  # noqa: E402


def gray16(path):
    meta, pix, pal = read_png(path)
    w, h = meta["w"], meta["h"]
    vals = []
    for i in range(0, len(pix), 2):
        vals.append(pix[i] * 256 + pix[i + 1])
    return vals, w, h


def corr(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((x - mb) ** 2 for x in b)
    cov = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    return cov / max(va * vb, 1e-12) ** 0.5


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "out/air/images/ir_lwir"
    f0, w, h = gray16(os.path.join(d, "000000.png"))
    f1, _, _ = gray16(os.path.join(d, "000003.png"))
    # residual after a 3x3 box blur isolates pixel-level fixed pattern
    def resid(v):
        out = []
        for y in range(h):
            for x in range(w):
                acc, n = 0.0, 0
                for dy in (-1, 0, 1):
                    yy = y + dy
                    if 0 <= yy < h:
                        for dx in (-1, 0, 1):
                            xx = x + dx
                            if 0 <= xx < w:
                                acc += v[yy * w + xx]
                                n += 1
                out.append(v[y * w + x] - acc / n)
        return out
    r0, r1 = resid(f0), resid(f1)
    sd0 = (sum((x - sum(r0) / len(r0)) ** 2 for x in r0) / len(r0)) ** 0.5
    sd1 = (sum((x - sum(r1) / len(r1)) ** 2 for x in r1) / len(r1)) ** 0.5
    print("%s  %dx%d  16bit" % (d, w, h))
    print("  pixel-level residual sd: frame0 %.2f counts, frame3 %.2f counts" % (sd0, sd1))
    print("  residual correlation between frames: %.3f  (1.0 = identical fixed pattern)" % corr(r0, r1))


main()
