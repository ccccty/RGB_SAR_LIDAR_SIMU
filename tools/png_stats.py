r"""Minimal PNG reader + image statistics (no external deps).

    python tools\png_stats.py out\opt1\images\optical\000000.png [--ascii]
"""
import struct
import sys
import zlib


def read_png(path):
    with open(path, "rb") as f:
        data = f.read()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
    pos, idat, pal, meta = 8, b"", None, {}
    while pos < len(data):
        ln = struct.unpack(">I", data[pos:pos + 4])[0]
        typ = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + ln]
        if typ == b"IHDR":
            w, h, depth, color = struct.unpack(">IIBB", body[:10])
            meta = dict(w=w, h=h, depth=depth, color=color)
        elif typ == b"PLTE":
            pal = body
        elif typ == b"IDAT":
            idat += body
        elif typ == b"IEND":
            break
        pos += 12 + ln
    raw = zlib.decompress(idat)
    ch = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[meta["color"]]
    bpp = max(1, ch * meta["depth"] // 8)
    stride = meta["w"] * ch * meta["depth"] // 8
    out = bytearray()
    prev = bytearray(stride)
    p = 0
    for _ in range(meta["h"]):
        f = raw[p]
        line = bytearray(raw[p + 1:p + 1 + stride])
        p += 1 + stride
        if f == 1:
            for i in range(bpp, stride):
                line[i] = (line[i] + line[i - bpp]) & 0xFF
        elif f == 2:
            for i in range(stride):
                line[i] = (line[i] + prev[i]) & 0xFF
        elif f == 3:
            for i in range(stride):
                a = line[i - bpp] if i >= bpp else 0
                line[i] = (line[i] + ((a + prev[i]) >> 1)) & 0xFF
        elif f == 4:
            for i in range(stride):
                a = line[i - bpp] if i >= bpp else 0
                b = prev[i]
                c = prev[i - bpp] if i >= bpp else 0
                pa, pb, pc = abs(b - c), abs(a - c), abs(a + b - 2 * c)
                pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                line[i] = (line[i] + pr) & 0xFF
        out += line
        prev = line
    return meta, bytes(out), pal


def luma(meta, pix):
    w, h, ch, depth = meta["w"], meta["h"], {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[meta["color"]], meta["depth"]
    step = 1 if depth == 8 else 2
    vals = []
    for y in range(h):
        row = y * w * ch * step
        for x in range(w):
            o = row + x * ch * step
            if ch >= 3:
                r, g, b = pix[o], pix[o + step], pix[o + 2 * step]
                vals.append(0.299 * r + 0.587 * g + 0.114 * b)
            else:
                vals.append(pix[o] if step == 1 else (pix[o] * 256 + pix[o + 1]) / 256.0)
    return vals, w, h


def main(argv):
    path = argv[1]
    meta, pix, pal = read_png(path)
    vals, w, h = luma(meta, pix)
    vals_sorted = sorted(vals)
    n = len(vals)
    mean = sum(vals) / n
    pct = lambda q: vals_sorted[min(n - 1, int(q * n))]
    sat = sum(1 for v in vals if v > 250) / n
    dark = sum(1 for v in vals if v < 5) / n
    print("%s  %dx%d color=%d depth=%d" % (path, w, h, meta["color"], meta["depth"]))
    print("  mean %.1f  min %.0f  max %.0f  p5 %.0f  p95 %.0f" % (mean, vals_sorted[0], vals_sorted[-1], pct(0.05), pct(0.95)))
    std = (sum((v - mean) ** 2 for v in vals) / n) ** 0.5
    print("  std %.1f  saturated(>250) %.1f%%  black(<5) %.1f%%  levels %d"
          % (std, 100 * sat, 100 * dark, len(set(int(v) for v in vals))))
    if "--ascii" in argv:
        cols, rows, chars = 40, 16, " .:-=+*#%@"
        for r in range(rows):
            line = ""
            for c in range(cols):
                x = int((c + 0.5) * w / cols)
                y = int((r + 0.5) * h / rows)
                line += chars[min(9, int(vals[y * w + x] / 255.0 * 9.99))]
            print("  " + line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
