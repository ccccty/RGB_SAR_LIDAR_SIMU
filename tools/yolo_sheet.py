"""Draw an exported YOLO flavour back onto its images as one contact sheet.

Reading .txt files tells you they parse; this tells you the boxes are where the
targets are, which is the only check that catches a coordinate-space mistake
(normalised vs pixel, letterboxed vs native, swapped x/y).

    python tools/yolo_sheet.py datasets/yolo optical
    python tools/yolo_sheet.py datasets/yolo ir_mwir_wide --n 8 --out out/sheet_wide.png
"""
import argparse
import glob
import json
import os

from PIL import Image, ImageDraw

COLORS = [(255, 64, 64), (64, 200, 255), (255, 210, 64), (120, 255, 120)]


def draw_one(img, lab, names, scale=1.0):
    W, H = img.size
    dr = ImageDraw.Draw(img)
    n = 0
    if os.path.exists(lab):
        for line in open(lab, encoding="utf-8"):
            p = line.split()
            if len(p) != 5:
                continue
            c, cx, cy, w, h = int(p[0]), *map(float, p[1:])
            x0, y0 = (cx - w / 2) * W, (cy - h / 2) * H
            x1, y1 = (cx + w / 2) * W, (cy + h / 2) * H
            col = COLORS[c % len(COLORS)]
            dr.rectangle([x0, y0, max(x1, x0 + 1), max(y1, y0 + 1)], outline=col, width=2)
            # a 2-pixel target is invisible at contact-sheet scale: bracket it
            if (x1 - x0) < 12 or (y1 - y0) < 12:
                b = 9
                for sx, sy, dx, dy in ((x0, y0, 1, 1), (x1, y0, -1, 1),
                                       (x0, y1, 1, -1), (x1, y1, -1, -1)):
                    dr.line([sx, sy, sx + dx * b, sy], fill=col, width=2)
                    dr.line([sx, sy, sx, sy + dy * b], fill=col, width=2)
            txt = names[c] if c < len(names) else str(c)
            dr.text((x0, max(0, y0 - 11)), txt, fill=col)
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="datasets/yolo")
    ap.add_argument("flavour", nargs="?", default="optical")
    ap.add_argument("--n", type=int, default=12, help="frames on the sheet")
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--tile", type=int, default=320, help="tile width in px")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    names = [l.strip() for l in open(os.path.join(args.root, "classes.txt"),
                                     encoding="utf-8") if l.strip()]
    files = []
    for split in ("train", "val"):
        files += sorted(glob.glob(os.path.join(args.root, "images", args.flavour,
                                               split, "*")))
    if not files:
        print("no images for flavour %s" % args.flavour)
        return 2
    step = max(1, len(files) // args.n)
    pick = files[::step][:args.n]

    tiles = []
    total_boxes = 0
    for f in pick:
        lab = f.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep)
        lab = os.path.splitext(lab)[0] + ".txt"
        im = Image.open(f).convert("RGB")
        im = im.resize((args.tile, int(im.height * args.tile / im.width)),
                       Image.LANCZOS)
        total_boxes += draw_one(im, lab, names)
        tiles.append((os.path.basename(f), im))

    tw, th = args.tile, tiles[0][1].height
    cap = 16
    cols = min(args.cols, len(tiles))
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, rows * (th + cap)), (18, 20, 24))
    dr = ImageDraw.Draw(sheet)
    for i, (nm, im) in enumerate(tiles):
        x, y = (i % cols) * tw, (i // cols) * (th + cap)
        sheet.paste(im, (x, y))
        dr.text((x + 4, y + th + 3), nm[:38], fill=(190, 195, 205))

    out = args.out or os.path.join(args.root, "sheet_%s.png" % args.flavour)
    sheet.save(out)
    print("%d tiles, %d boxes drawn -> %s" % (len(tiles), total_boxes, out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
