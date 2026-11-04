"""Export the simulated runs as YOLO detection datasets.

The simulation already writes one label JSON per frame per modality, all derived
from the same 3D world, the same camera pose and the same projection. This tool
turns that into what a trainer actually consumes:

    datasets/yolo/
      images/<flavour>/{train,val}/<scene>__f000123.jpg
      labels/<flavour>/{train,val}/<scene>__f000123.txt
      data_<flavour>.yaml          one dataset file per sensor flavour
      classes.txt  pairs.csv  export.json

Why one dataset per flavour instead of one dataset with all images: YOLO trains
on a single image space. Mixing a 2.2 deg seeker frame with an 18 deg search
frame, or a thermal frame with an RGB frame, teaches nothing about any of them.
Each flavour keeps its own image/box space, and `pairs.csv` records which frames
of different flavours are the *same instant*, so a fused or multi-branch model
can still pair them exactly.

Split: by clip, not at random. Every scene holds two 120-frame engagements, and
randomly splitting frames of one engagement puts a frame from t=3.01 s in train
and t=3.06 s in val -- the two are visually identical, so the validation score
would be a memorisation score. Clip 0 is train, clip 1 is val.

Boxes whose object was not actually painted in that image (a target below the
detector's sampling limit in the search lens, or one hidden behind a building)
are dropped: a label pointing at empty sky teaches a false positive.

Usage:
    python tools/export_yolo.py                     # all runs with >=60 frames
    python tools/export_yolo.py --scenes seeker_air2ground
    python tools/export_yolo.py --png               # lossless, ~4x larger
"""
import argparse
import csv
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# name -> (image dir, label dir, kind). IR uses the 8-bit preview: the 16-bit
# counts are for radiometry, not for a network that expects 8-bit input.
FLAVOURS = [
    ("optical", "images/optical", "labels/optical", "rgb"),
    ("optical_wide", "images/optical_wide", "labels/optical_wide", "rgb"),
    ("ir_mwir", "images/ir_preview_mwir", "labels/ir_mwir", "ir"),
    ("ir_lwir", "images/ir_preview_lwir", "labels/ir_lwir", "ir"),
    ("ir_mwir_wide", "images/ir_preview_mwir_wide", "labels/ir_mwir_wide", "ir"),
    ("ir_lwir_wide", "images/ir_preview_lwir_wide", "labels/ir_lwir_wide", "ir"),
    ("sar", "images/sar", "labels/sar", "sar"),
    # range-Doppler map: a detection is one range/Doppler cell, so the boxes come
    # from a different key than the image-space modalities
    ("radar_rd", "images/radar", "labels/radar", "radar", "boxes_rd"),
]

# flavours that share a camera pose, i.e. whose boxes MUST be identical
POSE_GROUPS = [
    ("optical", "ir_mwir", "ir_lwir"),
    ("optical_wide", "ir_mwir_wide", "ir_lwir_wide"),
]


def load_json(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return default


def scene_runs(runs, only=None, min_frames=60):
    out = []
    for nm in sorted(os.listdir(runs)):
        d = os.path.join(runs, nm)
        if not os.path.isdir(d) or not os.path.isdir(os.path.join(d, "labels", "optical")):
            continue
        if only and nm not in only:
            continue
        n = len([f for f in os.listdir(os.path.join(d, "labels", "optical"))
                 if f.endswith(".json")])
        if n < min_frames:
            continue
        out.append((nm, d, n))
    return out


def class_map(runs):
    """name -> class id, built from the COCO category tables."""
    cls_of = {}
    names = {}
    categories = set()
    for _nm, d, _n in runs:
        for p in sorted(os.listdir(d)):
            if not (p.startswith("annotations_") and p.endswith("_coco.json")):
                continue
            coco = load_json(os.path.join(d, p), {}) or {}
            cats = {c["id"]: c["name"] for c in coco.get("categories") or []}
            categories.update(cats.values())
            for a in coco.get("annotations") or []:
                cn = cats.get(a.get("category_id"))
                if a.get("object_name") and cn:
                    cls_of[a["object_name"]] = cn
    for cn in sorted(categories | set(cls_of.values())):
        names[cn] = len(names)
    return cls_of, names


def read_boxes(lab_path, bkey="boxes"):
    """({name: box}, (W, H)) from one frame label file."""
    d = load_json(lab_path)
    if not isinstance(d, dict):
        return None, None
    b = d.get(bkey)
    W, H = int(d.get("width") or 0), int(d.get("height") or 0)
    if isinstance(b, list):                       # older single-camera layout
        return [], (W, H)
    if not isinstance(b, dict):
        return [], (W, H)
    return b, (W, H)


def to_yolo(box, W, H, min_px=0.0):
    x0 = max(0.0, min(float(box["x0"]), W))
    x1 = max(0.0, min(float(box["x1"]), W))
    y0 = max(0.0, min(float(box["y0"]), H))
    y1 = max(0.0, min(float(box["y1"]), H))
    w, h = x1 - x0, y1 - y0
    if w <= 1e-6 or h <= 1e-6:
        return None
    if min_px and (w < min_px or h < min_px):
        return None
    return ((x0 + x1) / 2.0 / W, (y0 + y1) / 2.0 / H, w / W, h / H)


def save_image(src, dst, kind, fmt, quality, size=None):
    from PIL import Image
    im = Image.open(src)
    if kind == "rgb":
        im = im.convert("RGB")
    else:
        im = im.convert("L")
    if size:
        im = im.resize(size, Image.BILINEAR)
    if fmt == "jpg":
        im.save(dst, "JPEG", quality=quality, optimize=True, subsampling=0)
    else:
        im.save(dst, "PNG", optimize=False)
    return im.size


STACKS = [
    ("stack_narrow", [("optical", "images/optical", "labels/optical", "rgb"),
                      ("ir_mwir", "images/ir_preview_mwir", "labels/ir_mwir", "ir"),
                      ("ir_lwir", "images/ir_preview_lwir", "labels/ir_lwir", "ir")]),
    ("stack_wide", [("optical_wide", "images/optical_wide", "labels/optical_wide", "rgb"),
                    ("ir_lwir_wide", "images/ir_preview_lwir_wide",
                     "labels/ir_lwir_wide", "ir")]),
    ("stack_sar", [("optical", "images/optical", "labels/optical", "rgb"),
                   ("sar", "images/sar", "labels/sar", "sar")]),
    ("stack_radar", [("optical", "images/optical", "labels/optical", "rgb"),
                     ("radar_rd", "images/radar", "labels/radar", "radar", "boxes_rd")]),
    # 三路并排：可见光 + 中波红外 + 距离-多普勒，一眼看完"同一目标三种传感器"
    ("stack_air", [("optical", "images/optical", "labels/optical", "rgb"),
                   ("ir_mwir", "images/ir_preview_mwir", "labels/ir_mwir", "ir"),
                   ("radar_rd", "images/radar", "labels/radar", "radar", "boxes_rd")]),
]

# composite panels: (flavour, image dir, label dir, kind, boxes key)
for _i, _p in enumerate(STACKS):
    STACKS[_i] = (_p[0], [tuple(x) + (("boxes",) if len(x) == 4 else ())
                          for x in _p[1]])

SEP = 4          # separator width between panels, px
CAP = 20         # caption strip at the bottom of the annotated copy
BG = (10, 11, 14)


def frames_per_clip(d, n_lab):
    """(frames per split, is_smoke) for one run."""
    meta = load_json(os.path.join(d, "meta.json"), {}) or {}
    fpc = 0
    cfgp = meta.get("config")
    if cfgp and os.path.exists(cfgp):
        cfg = load_json(cfgp, {}) or {}
        fpc = int((cfg.get("clip") or {}).get("frames_per_clip") or 0)
    if not fpc:
        fpc = n_lab // 2 if n_lab else 1
    smoke = n_lab <= fpc
    if smoke:
        fpc = max(1, int(round(n_lab * 0.8)))
    return fpc, smoke


def compose_stacks(runs, out, cls_of, id_of, args):
    """Stitch several modalities of one instant into a single wide image.

    Two copies come out of every frame, on purpose:

    * `composites/` -- captioned, boxes drawn. This is the one a human looks at
      to confirm the modalities really line up.
    * `images/stack_*/` + `labels/stack_*/` -- clean, no text, no drawn boxes,
      with every panel's boxes translated into the stitched coordinate frame.
      That makes "one image holding three registered views" a trainable input
      (the simplest form of early fusion) instead of just a picture.

    Panels are pasted at native resolution, never rescaled: resampling a 2 px
    target to fit a common width is how a label quietly stops matching its pixel.
    """
    from PIL import Image, ImageDraw
    try:
        from PIL import ImageFont
        font = ImageFont.load_default(size=13)
    except Exception:                 # Pillow < 10.1 has no sized default font
        font = None

    stats = {}
    for stack, panels in STACKS:
        src = [tuple(p) for p in panels]
        n_img = n_ann = n_box = 0
        for scene, d, n_lab in runs:
            fpc, _smoke = frames_per_clip(d, n_lab)
            for fid in sorted(int(os.path.splitext(f)[0]) for f in
                              os.listdir(os.path.join(d, src[0][1]))
                              if f.endswith(".png")):
                have = []
                for nm, idir, ldir, kind, bkey in src:
                    ip = os.path.join(d, idir, "%06d.png" % fid)
                    lp = os.path.join(d, ldir, "%06d.json" % fid)
                    if os.path.exists(ip) and os.path.exists(lp):
                        have.append((nm, ip, lp, kind, bkey))
                if len(have) < 2:
                    continue
                split = "train" if fid < fpc else "val"
                imgs, meta = [], []
                for nm, ip, lp, kind, bkey in have:
                    im = Image.open(ip)
                    im = im.convert("RGB") if kind == "rgb" else im.convert("L").convert("RGB")
                    boxes, (W, H) = read_boxes(lp, bkey)
                    imgs.append((nm, im))
                    meta.append((boxes or {}, W or im.width, H or im.height))
                cw = sum(i.width for _n, i in imgs) + SEP * (len(imgs) - 1)
                ch = max(i.height for _n, i in imgs)
                clean = Image.new("RGB", (cw, ch), BG)
                ann = Image.new("RGB", (cw, ch + CAP), BG)
                dr = ImageDraw.Draw(ann)
                lines, x = [], 0
                for (nm, im), (bx, W, H) in zip(imgs, meta):
                    # panels differ in height (a RD map is 256 rows, a camera 576):
                    # centre them so the shorter one is not glued to the top edge
                    yoff = max(0, (ch - im.height) // 2)
                    clean.paste(im, (x, yoff))
                    ann.paste(im, (x, yoff))
                    col = COLORS_PANEL[src.index(next(s for s in src if s[0] == nm)) % 4]
                    for onm, b in sorted(bx.items()):
                        if not b.get("visible", True):
                            continue
                        cn = cls_of.get(onm)
                        if cn is None or cn not in id_of:
                            continue
                        x0, y0 = b["x0"] + x, b["y0"] + yoff
                        x1, y1 = b["x1"] + x, b["y1"] + yoff
                        dr.rectangle([x0, y0, max(x1, x0 + 1), max(y1, y0 + 1)],
                                     outline=col, width=2)
                        if (x1 - x0) < 14 or (y1 - y0) < 14:
                            k = 11
                            for sx, sy, dx, dy in ((x0, y0, 1, 1), (x1, y0, -1, 1),
                                                   (x0, y1, 1, -1), (x1, y1, -1, -1)):
                                dr.line([sx, sy, sx + dx * k, sy], fill=col, width=2)
                                dr.line([sx, sy, sx, sy + dy * k], fill=col, width=2)
                        y = to_yolo(dict(x0=b["x0"] + x, y0=b["y0"] + yoff,
                                         x1=b["x1"] + x, y1=b["y1"] + yoff), cw, ch)
                        if y:
                            lines.append("%d %.6f %.6f %.6f %.6f" % (id_of[cn], *y))
                    dr.text((x + 5, ch + 3), "%s  %dx%d" % (nm, im.width, im.height),
                            fill=col, font=font)
                    x += im.width + SEP
                stem = "%s__f%06d" % (scene, fid)
                ext = ".jpg" if args.fmt == "jpg" else ".png"
                idst = os.path.join(out, "images", stack, split)
                ldst = os.path.join(out, "labels", stack, split)
                adst = os.path.join(out, "composites", stack, split)
                for p in (idst, ldst, adst):
                    os.makedirs(p, exist_ok=True)
                clean.save(os.path.join(idst, stem + ext), "JPEG",
                           quality=args.quality, optimize=True, subsampling=0)
                ann.save(os.path.join(adst, stem + ".jpg"), "JPEG",
                         quality=88, optimize=True)
                with open(os.path.join(ldst, stem + ".txt"), "w", encoding="utf-8") as fh:
                    fh.write("\n".join(lines) + ("\n" if lines else ""))
                n_img += 1
                n_ann += 1
                n_box += len(lines)
        if n_img:
            stats[stack] = dict(images=n_img, boxes=n_box,
                                panels=[p[0] for p in panels])
            with open(os.path.join(out, "data_%s.yaml" % stack), "w", encoding="utf-8") as fh:
                fh.write("# %s -- %s stitched at native resolution\n"
                         % (stack, " + ".join(p[0] for p in panels)))
                fh.write('path: "%s"\n' % out.replace("\\", "/"))
                fh.write("train: images/%s/train\n" % stack)
                fh.write("val: images/%s/val\n" % stack)
                fh.write("names:\n")
                for cn, ci in sorted(id_of.items(), key=lambda kv: kv[1]):
                    fh.write("  %d: %s\n" % (ci, cn))
    return stats


COLORS_PANEL = [(255, 70, 70), (90, 200, 255), (255, 210, 70), (130, 240, 130)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(ROOT, "out"))
    ap.add_argument("--out", default=os.path.join(ROOT, "datasets", "yolo"))
    ap.add_argument("--scenes", default="", help="comma separated run names")
    ap.add_argument("--flavours", default="", help="comma separated flavour names")
    ap.add_argument("--fmt", default="jpg", choices=["jpg", "png"])
    ap.add_argument("--quality", type=int, default=92)
    ap.add_argument("--min-frames", type=int, default=60)
    ap.add_argument("--min-box-px", type=float, default=0.0,
                    help="drop boxes thinner than this many source pixels")
    ap.add_argument("--clean", action="store_true", help="wipe the output tree first")
    ap.add_argument("--no-compose", action="store_true",
                    help="skip the multi-modality stitched images")
    ap.add_argument("--only-compose", action="store_true",
                    help="only rebuild the stitched images from existing sources")
    args = ap.parse_args()

    only = {s.strip() for s in args.scenes.split(",") if s.strip()}
    want = {s.strip() for s in args.flavours.split(",") if s.strip()}
    runs = scene_runs(args.runs, only, args.min_frames)
    if not runs:
        print("no runs found under %s" % args.runs)
        return 2
    print("scenes: %s" % ", ".join("%s(%d)" % (n, c) for n, _d, c in runs))

    cls_of, names = class_map(runs)
    if not names:
        print("no classes found: are the annotations_*_coco.json files present?")
        return 2
    id_of = {n: i for n, i in names.items()}
    print("classes: %s" % ", ".join("%d:%s" % (i, n) for n, i in sorted(names.items(), key=lambda kv: kv[1])))

    if args.only_compose:
        st = compose_stacks(runs, args.out, cls_of, id_of, args)
        for nm, v in sorted(st.items()):
            print("%-14s %5d images %5d boxes  %s"
                  % (nm, v["images"], v["boxes"], " + ".join(v["panels"])))
        print("composites -> %s" % os.path.join(args.out, "composites"))
        return 0

    if args.clean and os.path.isdir(args.out):
        shutil.rmtree(args.out)
    os.makedirs(args.out, exist_ok=True)

    stats = {}
    pairs = []            # one row per (scene, clip, frame)
    reg_dev = []          # cross-flavour box disagreement, for the self-check
    dropped_invisible = 0
    dropped_tiny = 0

    for scene, d, n_lab in runs:
        # clip length decides the split; a run shorter than one clip falls back to
        # a temporal cut so `yolo train` still has a val split (see frames_per_clip)
        fpc, smoke = frames_per_clip(d, n_lab)

        # real timestamps come from the frame index, not from a hard-coded 20 fps
        tmap = {}
        for fp in ("frames.json",):
            arr = load_json(os.path.join(d, fp), None)
            if isinstance(arr, list):
                for e in arr:
                    if isinstance(e, dict) and "frame" in e and "t" in e:
                        tmap[int(e["frame"])] = float(e["t"])
        frames = sorted(int(os.path.splitext(f)[0]) for f in
                        os.listdir(os.path.join(d, "labels", "optical"))
                        if f.endswith(".json"))
        missing = [f for f in range(frames[0], frames[-1] + 1) if f not in set(frames)]
        if missing:
            print("  WARN %s: %d frame gaps (first %s)" % (scene, len(missing), missing[:5]))

        per_frame = {}
        for fid in frames:
            split = "train" if fid < fpc else "val"
            row = dict(scene=scene, clip=fid // fpc, frame=fid,
                       t=round(tmap.get(fid, (fid % fpc) / 20.0), 4), split=split)
            ok_any = False
            for name, idir, ldir, kind, *rest in FLAVOURS:
                bkey = rest[0] if rest else "boxes"
                if want and name not in want:
                    continue
                src = os.path.join(d, idir, "%06d.png" % fid)
                lab = os.path.join(d, ldir, "%06d.json" % fid)
                if not (os.path.exists(src) and os.path.exists(lab)):
                    continue
                boxes, (W, H) = read_boxes(lab, bkey)
                if boxes is None:
                    continue
                if not W or not H:
                    from PIL import Image
                    with Image.open(src) as im:
                        W, H = im.size
                lines = []
                for onm, b in sorted(boxes.items()):
                    if not b.get("visible", True):
                        dropped_invisible += 1
                        continue
                    y = to_yolo(b, W, H, args.min_box_px)
                    if y is None:
                        dropped_tiny += 1
                        continue
                    cn = cls_of.get(onm)
                    if cn is None or cn not in id_of:
                        continue
                    lines.append("%d %.6f %.6f %.6f %.6f" % (id_of[cn], *y))
                stem = "%s__f%06d" % (scene, fid)
                ext = ".jpg" if args.fmt == "jpg" else ".png"
                idst = os.path.join(args.out, "images", name, split)
                ldst = os.path.join(args.out, "labels", name, split)
                os.makedirs(idst, exist_ok=True)
                os.makedirs(ldst, exist_ok=True)
                save_image(src, os.path.join(idst, stem + ext), kind, args.fmt, args.quality)
                with open(os.path.join(ldst, stem + ".txt"), "w", encoding="utf-8") as fh:
                    fh.write("\n".join(lines) + ("\n" if lines else ""))
                st = stats.setdefault(name, dict(images=0, boxes=0, empty=0, by_split={}))
                st["images"] += 1
                st["boxes"] += len(lines)
                st["empty"] += (0 if lines else 1)
                bs = st["by_split"].setdefault(split, dict(images=0, boxes=0))
                bs["images"] += 1
                bs["boxes"] += len(lines)
                row[name] = "images/%s/%s/%s%s" % (name, split, stem, ext)
                per_frame[name] = (boxes, W, H)
                ok_any = True
            if ok_any:
                pairs.append(row)

            # registration self-check: same instant, same pose -> same box
            for grp in POSE_GROUPS:
                have = [g for g in grp if g in per_frame]
                if len(have) < 2:
                    continue
                base = per_frame[have[0]]
                for other in have[1:]:
                    ob, oW, oH = per_frame[other]
                    for onm, b in base[0].items():
                        c = ob.get(onm)
                        if not c or not b.get("visible", True) or not c.get("visible", True):
                            continue
                        dev = max(abs(b["x0"] - c["x0"]), abs(b["y0"] - c["y0"]),
                                  abs(b["x1"] - c["x1"]), abs(b["y1"] - c["y1"]))
                        reg_dev.append(dict(scene=scene, frame=fid, a=have[0], b=other,
                                            obj=onm, dev_px=round(float(dev), 3)))
            per_frame.clear()

    # ---- dataset yaml per flavour
    for name in sorted(stats):
        ypath = os.path.join(args.out, "data_%s.yaml" % name)
        with open(ypath, "w", encoding="utf-8") as fh:
            fh.write("# %s -- generated by tools/export_yolo.py\n" % name)
            fh.write('path: "%s"\n' % args.out.replace("\\", "/"))
            fh.write("train: images/%s/train\n" % name)
            fh.write("val: images/%s/val\n" % name)
            fh.write("names:\n")
            for cn, ci in sorted(names.items(), key=lambda kv: kv[1]):
                fh.write("  %d: %s\n" % (ci, cn))

    with open(os.path.join(args.out, "classes.txt"), "w", encoding="utf-8") as fh:
        for cn, ci in sorted(names.items(), key=lambda kv: kv[1]):
            fh.write("%s\n" % cn)

    keys = ["scene", "clip", "frame", "t", "split"] + [f[0] for f in FLAVOURS
                                                       if f[0] in stats]
    with open(os.path.join(args.out, "pairs.csv"), "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(pairs)

    stack_stats = {} if args.no_compose else compose_stacks(runs, args.out, cls_of, id_of, args)
    dev_max = max((r["dev_px"] for r in reg_dev), default=0.0)
    summary = dict(
        generated_by="tools/export_yolo.py",
        runs=[n for n, _d, _c in runs], classes=names, flavours=stats,
        frames_paired=len(pairs),
        split_rule=("smoke: first 80% train / last 20% val (single clip)"
                    if smoke else "clip 0 train / clip 1+ val"),
        fmt=args.fmt, dropped_invisible=dropped_invisible, dropped_tiny=dropped_tiny,
        stacks=stack_stats,
        # NOTE: this compares *labels*, which are derived from one projection, so
        # 0.00 is a regression guard on the label path -- it does NOT prove the two
        # images are pixel-aligned. Image-level alignment is limited by what each
        # renderer can paint (a 30 px optical silhouette vs a 17 px thermal blob).
        label_consistency=dict(pairs_checked=len(reg_dev), max_dev_px=dev_max,
                               meaning="same instant, same pose -> identical labels",
                               samples=reg_dev[:8]))
    with open(os.path.join(args.out, "export.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1)

    print("\n%-14s %7s %8s %8s %9s" % ("flavour", "images", "boxes", "empty", "boxes/img"))
    for name in sorted(stats):
        st = stats[name]
        print("%-14s %7d %8d %8d %9.2f" % (name, st["images"], st["boxes"], st["empty"],
                                           st["boxes"] / max(st["images"], 1)))
    if stack_stats:
        print("\n%-14s %7s %8s  %s" % ("stacked", "images", "boxes", "panels"))
        for nm, st in sorted(stack_stats.items()):
            print("%-14s %7d %8d  %s" % (nm, st["images"], st["boxes"],
                                         " + ".join(st["panels"])))
    print("\npaired frames: %d | dropped: %d invisible, %d degenerate"
          % (len(pairs), dropped_invisible, dropped_tiny))
    print("label consistency: %d cross-flavour box pairs, max deviation %.2f px"
          % (len(reg_dev), dev_max))
    if dev_max > 0.51:
        print("WARN: boxes of the same instant disagree by %.2f px -- the modalities "
              "are not registered" % dev_max)
    print("written to %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
