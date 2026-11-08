#!/usr/bin/env python
"""Publish ONE simkit run into the static site, leaving every other run alone.

`tools/export_web.py` re-exports the whole `out/` tree, which is the right tool
for a full rebuild but a risky one when a single showcase scene is being added:
it rewrites the images, manifest and boot payload of every published scene.
This module is the narrow alternative -- it writes only `data/<scene>/`,
appends one entry to `data/manifest.json` and one to `data/curated.json`, then
hands the new PNGs to `tools/opt_images.py` for the WebP + thumbnail pass.

The site's scene switching reads `data/curated.json` first and falls back to
`manifest.json`. Both are updated here, so the new card works on HTTP and when
the JSON is read directly, and `tools/check_data.js` still passes.

Usage
    python tools/export_scene_web.py --scene demo_sea_surface \
        --title "开阔海面 · 中小型舰船" --cls ship \
        --caption "..." --note "..." \
        --frames 0,1,2,3 --hero-mod optical --hero-frame 1

Idempotent: re-running replaces the entries with the same id instead of
appending duplicates. Requires Pillow (already in requirements.txt).
"""
import argparse
import datetime
import json
import os
import sys

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, ROOT)

# (site key, image dir under out/<scene>, label dir or None, has boxes)
MODS = [
    ("optical",         "images/optical",            "labels/optical",   True),
    ("mwir",            "images/ir_preview_mwir",    "labels/ir_mwir",   True),
    ("lwir",            "images/ir_preview_lwir",    "labels/ir_lwir",   True),
    ("radar",           "images/radar",              None,               False),
    ("radar_ra",        "images/radar_ra",           None,               False),
    ("lidar_range",     "lidar/range",               "labels/lidar",     True),
    ("lidar_intensity", "lidar/intensity",           None,               False),
]
# Match the sizes already shipped for every published scene. The site sizes its
# <img> from this table, so a new scene must not introduce a new aspect/size.
SITE_SIZE = {
    "optical": (1280, 960), "mwir": (640, 480), "lwir": (640, 480),
    "radar": (1280, 720), "radar_ra": (1280, 800),
    "lidar_range": (256, 192), "lidar_intensity": (256, 192),
}
MOD_TITLES = {
    "optical": "可见光 RGB", "mwir": "中波红外 3–5 µm", "lwir": "长波红外 8–12 µm",
    "radar": "雷达 · 距离-多普勒", "radar_ra": "雷达 · 距离-角度",
    "lidar_range": "激光雷达 · 距离", "lidar_intensity": "激光雷达 · 强度",
}


def jload(path, dflt=None):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return dflt


def jwrite(path, obj, indent=1):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=indent)
        fh.write("\n")


def fkey(i):
    return "%06d" % int(i)


def boxes_of(path, sx, sy):
    """Read a label file and rescale its boxes into the shipped image."""
    d = jload(path)
    if not isinstance(d, dict):
        return []
    raw = d.get("boxes")
    items = raw.values() if isinstance(raw, dict) else (raw if isinstance(raw, list) else [])
    out = []
    for b in items:
        if not isinstance(b, dict) or b.get("visible") is False or "x0" not in b:
            continue
        out.append({
            "x0": round(float(b["x0"]) * sx, 1), "y0": round(float(b["y0"]) * sy, 1),
            "x1": round(float(b["x1"]) * sx, 1), "y1": round(float(b["y1"]) * sy, 1),
            "label": b.get("name") or "target",
        })
    return out


def copy_frame(src, dst, size):
    """Write a PNG at the site's canonical size; returns the source size."""
    im = Image.open(src)
    ow, oh = im.size
    if im.mode not in ("RGB", "L"):
        im = im.convert("RGB" if im.mode != "I;16" else "L")
    if (ow, oh) != size:
        im = im.resize(size, Image.LANCZOS)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    im.save(dst)
    return ow, oh


def run(args):
    sid = args.scene
    src = os.path.join(ROOT, "out", sid)
    data = os.path.join(ROOT, "data")
    dest = os.path.join(data, sid)
    if not os.path.isdir(src):
        raise SystemExit("missing run: %s" % src)
    meta = jload(os.path.join(src, "meta.json"), {}) or {}
    frames = [int(x) for x in args.frames.split(",") if x.strip() != ""]
    if not frames:
        frames = [0]
    rows = {}
    for r in jload(os.path.join(src, "frames.json"), []) or []:
        rows[int(r.get("frame", -1))] = r
    weather_list = meta.get("weather") or [{}]

    available = []
    for key, rel, _lab, has_boxes in MODS:
        if os.path.isdir(os.path.join(src, rel)):
            available.append((key, rel, _lab, has_boxes))
    if not available:
        raise SystemExit("no modality directories under %s" % src)

    # 1. stage PNGs at the site size; opt_images converts them to WebP below.
    staged = {}
    for key, rel, lab, has_boxes in available:
        size = SITE_SIZE[key]
        for i in frames:
            name = fkey(i) + ".png"
            sp = os.path.join(src, rel, name)
            if not os.path.isfile(sp):
                continue
            ow, oh = copy_frame(sp, os.path.join(dest, key, name), size)
            boxes = []
            if lab and has_boxes:
                boxes = boxes_of(os.path.join(src, lab, fkey(i) + ".json"),
                                 size[0] / float(ow), size[1] / float(oh))
            staged.setdefault(i, {})[key] = dict(
                img="%s/%s/%s.webp" % (sid, key, fkey(i)),
                w=size[0], h=size[1], boxes=boxes)

    # 2. manifest.json: one scene, one entry per curated frame.
    mpath = os.path.join(data, "manifest.json")
    man = jload(mpath, {}) or {}
    man.setdefault("scenes", [])
    band = args.band or meta.get("radar_band") or "—"
    scene = {
        "id": sid, "title": args.title, "cls": args.cls, "band": band,
        "sensors": meta.get("sensors") or ["ir", "lidar", "optical", "radar"],
        "categories": [c.get("name") for c in (meta.get("categories") or [])
                       if isinstance(c, dict) and c.get("name")] or [args.cls],
        "mods": [{"key": k, "title": MOD_TITLES.get(k, k), "boxes": bool(hb)}
                 for k, _r, _l, hb in available],
        "views": meta.get("views") or [],
        "sar_meta": None,
        "frames": [],
    }
    for i in frames:
        row = rows.get(i, {})
        rd = jload(os.path.join(src, "labels", "radar", fkey(i) + ".json"), {}) or {}
        tg = rd.get("targets") or []
        scene["frames"].append({
            "i": i,
            "weather": weather_list[i % len(weather_list)],
            "mods": staged.get(i, {}),
            "t": row.get("t", i * 0.1),
            "range_m": row.get("range_m"),
            "target_px": row.get("target_px"),
            "views": row.get("views") or {},
            "radar": {"n": len(tg), "clutter": int(rd.get("clutter_count") or 0),
                      "boxes": [], "pts": []},
        })
    man["scenes"] = [s for s in man["scenes"] if s.get("id") != sid] + [scene]
    jwrite(mpath, man)

    # 3. curated.json: the card the site actually renders.
    cpath = os.path.join(data, "curated.json")
    cur = jload(cpath, {}) or {}
    cur.setdefault("scenes", [])
    hero_frame = args.hero_frame if args.hero_frame in frames else frames[0]
    entry = {
        "id": sid, "title": args.title, "cls": args.cls, "band": band,
        "aspect": args.aspect,
        "mods": [k for k, _r, _l, _hb in available],
        "frames": frames,
        "hero": {"mod": args.hero_mod if args.hero_mod in
                 [k for k, _r, _l, _hb in available] else available[0][0],
                 "frame": hero_frame},
        "note": args.note, "caption": args.caption,
    }
    cur["scenes"] = [s for s in cur["scenes"] if s.get("id") != sid] + [entry]
    cur["generated"] = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    cur["domain"] = args.domain
    jwrite(cpath, cur)

    # 4. keep the producing config beside the images, as export_web.py does.
    cfg_json = os.path.join(ROOT, "scenes", sid + ".json")
    if os.path.isfile(cfg_json):
        with open(cfg_json, encoding="utf-8") as fh:
            jwrite(os.path.join(dest, "cfg.json"), json.load(fh))

    # 5. WebP + thumbnails + boot.js, via the project's existing idempotent pass.
    import opt_images
    opt_images.optimize(data, thumb_w=420, max_w=1280)
    print("published %s: %d frames x %d modalities" % (sid, len(frames), len(available)))
    return 0


def main():
    ap = argparse.ArgumentParser(description="Publish one simkit run to the static site")
    ap.add_argument("--scene", required=True, help="run id under out/ and data/")
    ap.add_argument("--title", required=True)
    ap.add_argument("--cls", default="target")
    ap.add_argument("--band", default="")
    ap.add_argument("--aspect", default="4:3")
    ap.add_argument("--domain", default="air + sea")
    ap.add_argument("--frames", default="0,1,2,3")
    ap.add_argument("--hero-mod", default="optical")
    ap.add_argument("--hero-frame", type=int, default=0)
    ap.add_argument("--caption", default="")
    ap.add_argument("--note", default="")
    return run(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
