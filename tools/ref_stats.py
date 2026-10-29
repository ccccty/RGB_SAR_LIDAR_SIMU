"""Statistics of the reference photos, for radiometric sanity checks."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from png_stats import read_png, luma  # noqa: E402


def pct(vals, q):
    n = len(vals)
    return vals[min(n - 1, max(0, int(q * n / 100.0)))]


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.join("assets", "ref", "png")
    print("%-42s %5s %5s %5s %5s %5s %5s" % ("file", "mean", "p5", "p50", "p95", "top25", "bot25"))
    for kind in ("optical", "ir"):
        d = os.path.join(root, kind)
        if not os.path.isdir(d):
            continue
        print("==", kind)
        for fn in sorted(os.listdir(d)):
            if not fn.lower().endswith(".png"):
                continue
            meta, pix, pal = read_png(os.path.join(d, fn))
            vals, w, h = luma(meta, pix)
            s = sorted(vals)
            band = max(1, h // 4) * w
            top = vals[:band]
            bot = vals[-band:]
            print("%-42s %5.1f %5.0f %5.0f %5.0f %5.1f %5.1f"
                  % (fn[:42], sum(vals) / len(vals), pct(s, 5), pct(s, 50), pct(s, 95),
                     sum(top) / len(top), sum(bot) / len(bot)))


main()
