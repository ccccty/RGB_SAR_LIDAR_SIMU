#!/usr/bin/env python
"""Pick the frames that are actually worth showing on the site.

Every (scene, frame, modality) is measured on five terms: target share,
sharpness, local contrast, radar consistency and motion-blur risk.  A frame
must first clear hard gates (complete target inside the crop, big enough,
optical + lwir + radar all labelled and aligned, radar detecting every
target); only survivors are ranked, and the composite score just orders them.

Read-only with respect to data/, index.html, app.js, site.js and simkit/.
Writes _tmp/curation.json plus review sheets under _tmp/curation/.

Usage
    python tools/curate_web.py --all
    python tools/curate_web.py --scene demo_air_headon
"""
import argparse
import datetime
import json
import math
import os
from collections import namedtuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
DATA = os.path.join(ROOT, "data")
TMP = os.path.join(ROOT, "_tmp")
PREVIEW = os.path.join(TMP, "curation")
CURATION = os.path.join(TMP, "curation.json")

# ---------------------------------------------------------------- modality map
# key -> (image dir under out/<scene>, label dir under out/<scene>, has boxes)
MODS = [
    ("optical",         "images/optical",              "labels/optical",        True),
    ("mwir",            "images/ir_preview_mwir",      "labels/ir_mwir",        True),
    ("lwir",            "images/ir_preview_lwir",      "labels/ir_lwir",        True),
    ("radar",           "images/radar",                "labels/radar",          False),
    ("radar_ra",        "images/radar_ra",             None,                    False),
    ("lidar_range",     "lidar/range",                 "labels/lidar",          True),
    ("lidar_intensity", "lidar/intensity",             None,                    False),
]
WIDE_MODS = [
    ("optical_wide", "images/optical_wide",          "labels/optical_wide",  True),
    ("mwir_wide",    "images/ir_preview_mwir_wide",  "labels/ir_mwir_wide",  True),
    ("lwir_wide",    "images/ir_preview_lwir_wide",  "labels/ir_lwir_wide",  True),
]
ALL_MODS = MODS + WIDE_MODS
SITE_ORDER = ["optical", "optical_wide", "mwir", "lwir", "mwir_wide", "lwir_wide",
              "radar", "radar_ra", "lidar_range", "lidar_intensity", "sar"]
SHAPE = {k: (d, l) for k, d, l, _b in ALL_MODS}

# ---------------------------------------------------------------------- scenes
SCENES = [
    dict(id="demo_air_headon", title="迎头接近 · 战机", hero_mod="optical", max_pick=4,
         caption="迎头姿态下机翼前缘和机身轮廓清楚，背景是真实云层。"),
    dict(id="demo_air_pursuit", title="尾追跟拍 · 战机", hero_mod="optical", max_pick=4,
         caption="从后方跟拍，机翼后缘与尾喷清楚，目标在缓慢拉开距离。"),
    dict(id="final_air", title="大型无人机 · 侧后视角", hero_mod="optical", max_pick=3,
         caption="大型无人机侧后方，翼展约占画幅五分之一，纯色天空把机体轮廓衬出来。"),
    dict(id="seeker_air2air_high", title="高空目标 · 双视场", hero_mod="lwir", max_pick=4,
         caption="窄视场里目标只有三十来像素，先靠宽视场找到，再用窄视场分辨。"),
    dict(id="seeker_surface2air", title="低空爬升接近 · 双视场", hero_mod="lwir", max_pick=4,
         caption="从低空抬头接近，窄视场里目标只有十几像素，红外上是一个亮点。"),
    dict(id="demo_three_ships_plate", title="三机俯视板面", hero_mod="optical", max_pick=3,
         caption="俯视云海上的三架飞机同框，间距均匀，便于看编队几何。"),
    dict(id="demo_tri_aircraft", title="三机斜视交会", hero_mod="optical", max_pick=3,
         caption="三机斜视交会，机体在云层背景上的轮廓很小。"),
]
SCENE_BY_ID = {s["id"]: s for s in SCENES}

# --------------------------------------------------------------------- weights
W_RATIO, W_SHARP, W_CONTRAST, W_RADAR, W_MOTION = 0.30, 0.22, 0.18, 0.15, 0.15

# hard gates: absolute criteria, they decide "may this frame ship at all?"
GATE_MARGIN = 0.015       # box keeps >= 1.5 % of width/height from the border
GATE_MIN_RATIO = 0.00015  # absolute floor: >= 0.015 % of the frame (~9x9 px)
GATE_MIN_TARGET_PX = 28.0 # widest target box in px; at 900 px display that is < 20 px
GATE_RATIO_SHARE = 0.40   # and no worse than 40 % of the scene's largest target
GATE_MIN_SHARP = 8.0      # laplacian variance inside the target roi
GATE_MIN_CONTRAST = 2.5   # |mean(target) - mean(ring)| in dB over a 1 DN floor
GATE_MIN_SNR = 12.0       # radar detection margin
GATE_MAX_MOTION = 0.08    # box centre travel to a neighbour, fraction of width
GATE_MIN_IOU = 0.45       # optical vs lwir box agreement in normalised coords

# scene-level exceptions: a defect accepted knowingly, recorded in the table
ALLOW_CLIP = {
    "demo_air_headon": "机翼末端出画",
    "demo_air_pursuit": "机翼末端出画",
}

# frames a human threw out after looking at the contact sheet
MANUAL_REJECT = {}


class Box(namedtuple("_Box", "name x0 y0 x1 y1")):
    __slots__ = ()

    @property
    def area(self):
        return max(0.0, self.x1 - self.x0) * max(0.0, self.y1 - self.y0)

    @property
    def cx(self):
        return 0.5 * (self.x0 + self.x1)

    @property
    def cy(self):
        return 0.5 * (self.y0 + self.y1)

    def xyxy(self):
        return (self.x0, self.y0, self.x1, self.y1)


# ------------------------------------------------------------------- utilities
def jload(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def fkey(i):
    return "%06d" % i


def clip_box(box, w, h):
    x0, y0, x1, y1 = box
    return (min(max(x0, 0.0), w - 1.0), min(max(y0, 0.0), h - 1.0),
            min(max(x1, 0.0), w - 1.0), min(max(y1, 0.0), h - 1.0))


def parse_boxes(payload):
    """Label json -> [Box] for every visible instance."""
    raw = payload.get("boxes")
    out = []
    if isinstance(raw, dict):
        items = [(k, v) for k, v in raw.items() if isinstance(v, dict)]
    elif isinstance(raw, list):
        items = [(v.get("name") or "obj%d" % i, v)
                 for i, v in enumerate(raw) if isinstance(v, dict) and "x0" in v]
    else:
        items = []
    for name, v in items:
        if v.get("visible") is False or "x0" not in v:
            continue
        out.append(Box(name, float(v["x0"]), float(v["y0"]),
                       float(v["x1"]), float(v["y1"])))
    return out


def iou(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    union = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1])
             - inter)
    return inter / union if union > 0 else 0.0


def to_norm(box, w, h):
    return (box[0] / w, box[1] / h, box[2] / w, box[3] / h)


def lapvar(gray):
    """Variance of the 4-neighbour laplacian; higher means crisper edges."""
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    lap = (-4.0 * gray[1:-1, 1:-1] + gray[:-2, 1:-1] + gray[2:, 1:-1]
           + gray[1:-1, :-2] + gray[1:-1, 2:])
    return float(lap.var())


def ring_contrast(gray, box):
    """Target mean vs a surrounding ring, in dB over a 1 DN floor."""
    h, w = gray.shape
    x0, y0, x1, y1 = clip_box(box, w, h)
    pad = max(4.0, 0.8 * max(x1 - x0, y1 - y0))
    ix0, iy0, ix1, iy1 = int(x0), int(y0), int(math.ceil(x1)), int(math.ceil(y1))
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0, 0.0, 0.0
    mu_in = float(gray[iy0:iy1, ix0:ix1].mean())
    ox0, oy0 = max(0, int(x0 - pad)), max(0, int(y0 - pad))
    ox1, oy1 = min(w, int(math.ceil(x1 + pad))), min(h, int(math.ceil(y1 + pad)))
    outer = gray[oy0:oy1, ox0:ox1]
    mask = np.ones(outer.shape, dtype=bool)
    mask[max(0, iy0 - oy0):max(0, iy1 - oy0), max(0, ix0 - ox0):max(0, ix1 - ox0)] = False
    vals = outer[mask]
    mu_out = float(vals.mean()) if vals.size else mu_in
    return 20.0 * math.log10(abs(mu_in - mu_out) + 1.0), mu_in, mu_out


def radar_stats(path):
    if not os.path.isfile(path):
        return None
    lab = jload(path)
    tg = lab.get("targets") or []
    if not tg:
        return dict(ok=False, snr=0.0, n=0, range_m=0.0, v_r=0.0,
                    clutter=int(lab.get("clutter_count") or 0))
    return dict(ok=all(bool(t.get("detected")) for t in tg),
                snr=max(float(t.get("snr_db") or 0.0) for t in tg),
                n=len(tg),
                range_m=float(tg[0].get("range") or 0.0),
                v_r=float(tg[0].get("v_r") or 0.0),
                clutter=int(lab.get("clutter_count") or 0))


def mod_metrics(scene_dir, mod, idx, cache):
    """Per-modality numbers for one frame, or None when inputs are missing."""
    img_dir, lab_dir = SHAPE[mod]
    ipath = os.path.join(scene_dir, img_dir, fkey(idx) + ".png")
    lpath = os.path.join(scene_dir, lab_dir, fkey(idx) + ".json") if lab_dir else None
    if not os.path.isfile(ipath) or not lpath or not os.path.isfile(lpath):
        return None
    lab = jload(lpath)
    gray = np.asarray(Image.open(ipath).convert("L"), dtype=np.float32)
    h, w = gray.shape[:2]
    boxes = parse_boxes(lab)
    out = dict(mod=mod, w=w, h=h, n=len(boxes), ratio=0.0, sharp=0.0,
               sharp_full=lapvar(gray), contrast_db=0.0, margin=0.0,
               complete=True, boxes=boxes,
               motion_blur_sim=bool(lab.get("motion_blur_simulated")),
               cx=0.0, cy=0.0, mu_in=0.0, mu_out=0.0)
    if boxes:
        out["ratio"] = sum(b.area for b in boxes) / float(w * h)
        big = max(boxes, key=lambda b: b.area)
        x0, y0, x1, y1 = clip_box(big.xyxy(), w, h)
        out["cx"], out["cy"] = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        pad = max(2, int(0.35 * max(x1 - x0, y1 - y0)))
        rx0, ry0 = max(0, int(x0) - pad), max(0, int(y0) - pad)
        rx1, ry1 = min(w, int(x1) + pad), min(h, int(y1) + pad)
        out["sharp"] = lapvar(gray[ry0:ry1, rx0:rx1])
        out["contrast_db"], out["mu_in"], out["mu_out"] = ring_contrast(
            gray, (x0, y0, x1, y1))
        out["margin"] = min(x0 / w, y0 / h, (w - 1 - x1) / w, (h - 1 - y1) / h)
        out["complete"] = bool(x0 > 0.5 and y0 > 0.5
                               and x1 < w - 1.5 and y1 < h - 1.5)
    cache[(mod, idx)] = out
    return out


def discover(scene_dir, idxs):
    """Modality keys that physically exist for this scene."""
    have = [k for k, img_dir, _l, _b in MODS
            if os.path.isdir(os.path.join(scene_dir, img_dir))]
    first = idxs[0] if idxs else None
    for key, img_dir, lab_dir, _boxes in WIDE_MODS:
        if first is None or not os.path.isdir(os.path.join(scene_dir, img_dir)):
            continue
        if os.path.isfile(os.path.join(scene_dir, lab_dir, fkey(first) + ".json")):
            have.append(key)
    return [k for k in SITE_ORDER if k in have]


def site_mods(scene_id, scene_dir, idxs):
    """Modality keys the site can serve, in site order.  -> (mods, exported)"""
    have = discover(scene_dir, idxs)
    if not os.path.isdir(os.path.join(DATA, scene_id)):
        return have, False
    keep = []
    for k in have:
        try:
            files = os.listdir(os.path.join(DATA, scene_id, k))
        except OSError:
            continue
        if any(f.endswith((".webp", ".png")) for f in files):
            keep.append(k)
    return keep, True


# --------------------------------------------------------------------- scoring
def scene_score(scene):
    sid = scene["id"]
    sdir = os.path.join(OUT, sid)
    if not os.path.isdir(sdir):
        return None
    lab_dir = os.path.join(sdir, "labels", "optical")
    if not os.path.isdir(lab_dir):
        return dict(scene=scene, sdir=sdir, rows=[], gated=[], mods=[], meta={},
                    idxs=[], no_labels=True)
    meta_p = os.path.join(sdir, "meta.json")
    meta = jload(meta_p) if os.path.isfile(meta_p) else {}
    info = {}
    frames_p = os.path.join(sdir, "frames.json")
    if os.path.isfile(frames_p):
        for row in jload(frames_p):
            info[int(row.get("frame", 0))] = row
    idxs = sorted(int(f[:6]) for f in os.listdir(lab_dir) if f.endswith(".json"))
    mods = discover(sdir, idxs)
    cache, rows = {}, []
    for i in idxs:
        per = {}
        for key in mods:
            m = mod_metrics(sdir, key, i, cache)
            if m:
                per[key] = m
        prim = per.get("optical") or per.get("lwir") or per.get("mwir")
        if not prim:
            continue
        rows.append(dict(i=i, per=per, prim=prim,
                         radar=radar_stats(os.path.join(sdir, "labels", "radar", fkey(i) + ".json")),
                         t=info.get(i, {}).get("t"), range_m=info.get(i, {}).get("range_m")))
    # motion risk: how far the primary box centre travels to its neighbours
    present = {r["i"] for r in rows}
    for r in rows:
        jumps = []
        for j in (r["i"] - 1, r["i"] + 1):
            if j in present:
                m = cache.get((r["prim"]["mod"], j))
                if m and m["n"]:
                    jumps.append(math.hypot(m["cx"] - r["prim"]["cx"],
                                            m["cy"] - r["prim"]["cy"]))
        r["motion_frac"] = (float(np.mean(jumps)) if jumps else 0.0) / float(r["prim"]["w"])
    # hard gates
    need = [k for k in ("optical", "lwir", "radar") if k in mods]
    ratio_floor = max(GATE_MIN_RATIO,
                      GATE_RATIO_SHARE * max((r["prim"]["ratio"] for r in rows),
                                             default=0.0))
    manual = MANUAL_REJECT.get(sid) or {}
    for r in rows:
        prim, fails = r["prim"], []
        if not prim["n"]:
            fails.append("无框")
        if not prim["complete"]:
            if sid in ALLOW_CLIP:
                r["warn"] = ALLOW_CLIP[sid]
            else:
                fails.append("贴边")
        elif prim["margin"] < GATE_MARGIN:
            fails.append("近边")
        if prim["ratio"] < ratio_floor:
            fails.append("目标过小")
        bw_px = max((b.x1 - b.x0 for b in prim["boxes"]), default=0.0)
        if bw_px < GATE_MIN_TARGET_PX:
            fails.append("仅%.0fpx" % bw_px)
        if prim["sharp"] < GATE_MIN_SHARP:
            fails.append("糊")
        if prim["contrast_db"] < GATE_MIN_CONTRAST:
            fails.append("对比弱")
        if prim.get("motion_blur_sim"):
            fails.append("已加模糊")
        missing = [k for k in need if k not in r["per"]]
        if missing:
            fails.append("缺" + "/".join(missing))
        lwir = r["per"].get("lwir")
        r["iou"] = 0.0
        if lwir is not None:
            if not lwir["n"]:
                fails.append("红外无框")
            else:
                a = to_norm(prim["boxes"][0].xyxy(), prim["w"], prim["h"])
                b = to_norm(lwir["boxes"][0].xyxy(), lwir["w"], lwir["h"])
                r["iou"] = iou(a, b)
                if r["iou"] < GATE_MIN_IOU:
                    fails.append("模态错位")
        rd = r["radar"]
        if rd is None:
            fails.append("无雷达")
        elif not rd["ok"]:
            fails.append("雷达漏检")
        elif rd["snr"] < GATE_MIN_SNR:
            fails.append("雷达弱")
        if r["motion_frac"] > GATE_MAX_MOTION:
            fails.append("帧间跳")
        if r["i"] in manual:
            fails.append("人工剔除(" + manual[r["i"]] + ")")
        r["fails"], r["gate"] = fails, not fails
    # composite score, normalised inside the scene (only rank order matters)
    pool = [r for r in rows if r["gate"]] or rows

    def top_of(get):
        vals = [get(r) for r in pool]
        return max(vals) if any(v > 0 for v in vals) else 1.0

    tr = top_of(lambda r: r["prim"]["ratio"])
    ts = top_of(lambda r: r["prim"]["sharp"])
    tc = top_of(lambda r: r["prim"]["contrast_db"])
    for r in rows:
        if not r["gate"]:
            r["score"] = 0.0
            continue
        rd = r["radar"]
        s_radar = (min(1.0, max(0.0, (rd["snr"] - GATE_MIN_SNR) / 30.0))
                   if rd and rd["ok"] else 0.0)
        r["score"] = (W_RATIO * min(1.0, r["prim"]["ratio"] / tr)
                      + W_SHARP * min(1.0, r["prim"]["sharp"] / ts)
                      + W_CONTRAST * min(1.0, r["prim"]["contrast_db"] / tc)
                      + W_RADAR * s_radar
                      + W_MOTION * min(1.0, max(0.0, 1.0 - r["motion_frac"] / GATE_MAX_MOTION)))
    return dict(scene=scene, sdir=sdir, meta=meta, mods=mods, rows=rows,
                gated=[r for r in rows if r["gate"]], info=info, idxs=idxs)


# ------------------------------------------------------------------- selection
def pick_frames(rows, wanted, min_gap=2):
    """Top-ranked frames with a spacing rule so a strip is not one pose."""
    ranked = sorted([r for r in rows if r["gate"]], key=lambda r: -r["score"])
    if not ranked:
        return []
    chosen, floor = [], 0.55 * ranked[0]["score"]

    def take(gap):
        for row in ranked:
            if len(chosen) >= wanted:
                break
            if row in chosen:
                continue
            if gap and any(abs(row["i"] - c["i"]) < gap for c in chosen):
                continue
            if len(chosen) >= 3 and row["score"] < floor:
                continue
            chosen.append(row)

    for gap in (min_gap, 1, 0):
        take(gap)
    return sorted(chosen, key=lambda r: r["i"])


def wanted_count(n_gate, scene_cap):
    """3-6 frames per scene, but never more than the scene deserves."""
    base = 3 if n_gate <= 4 else (4 if n_gate <= 8 else 5)
    return min(base, n_gate, scene_cap, 6)


def aspect_of(res):
    prim = next((r["prim"] for r in res["rows"] if r["prim"].get("w")), None)
    if not prim:
        return "4:3"
    w, h = int(prim["w"]), int(prim["h"])
    g = math.gcd(w, h)
    return "%d:%d" % (w // g, h // g)


# ---------------------------------------------------------------- presentation
CJK_FONTS = ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf",
             "C:/Windows/Fonts/simsun.ttc"]


def font(size=15):
    for path in CJK_FONTS:
        if os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def draw_boxes(draw, boxes, scale, offset=(0.0, 0.0), color=(255, 64, 64)):
    for b in boxes:
        draw.rectangle([offset[0] + b.x0 * scale, offset[1] + b.y0 * scale,
                        offset[0] + b.x1 * scale, offset[1] + b.y1 * scale],
                       outline=color, width=2)


def contact_sheet(res, picks, path, tile_w=400):
    """Full frame plus a target crop per candidate, so blur and clipping show."""
    rows = [r for r in res["rows"] if r["gate"]]
    if len(rows) < 2:
        rows = sorted(res["rows"], key=lambda r: -r["score"])[:4]
    if not rows:
        return None
    mod = res["scene"]["hero_mod"]
    if mod not in rows[0]["per"]:
        mod = "optical"
    picked = {r["i"] for r in picks}
    tiles = []
    for r in rows:
        prim = r["per"].get(mod) or r["prim"]
        src = Image.open(os.path.join(res["sdir"], SHAPE[prim["mod"]][0],
                                      fkey(r["i"]) + ".png")).convert("RGB")
        w, h = src.size
        th = int(round(tile_w * h / float(w)))
        full = src.resize((tile_w, th), Image.LANCZOS)
        draw_boxes(ImageDraw.Draw(full), prim["boxes"], tile_w / float(w))
        big = max(prim["boxes"], key=lambda b: b.area) if prim["boxes"] else None
        if big is not None:
            # crop window: magnify tiny targets, stay near 1:1 for big ones
            zw = min(float(w), max(min((big.x1 - big.x0) * 4.0, tile_w * 1.6),
                                   tile_w * 0.25))
            zh = min(float(h), zw * th / float(tile_w))
            x0 = int(max(0, min(w - zw, big.cx - zw / 2.0)))
            y0 = int(max(0, min(h - zh, big.cy - zh / 2.0)))
            zoom = src.crop((x0, y0, int(x0 + zw), int(y0 + zh))).resize(
                (tile_w, th), Image.LANCZOS)
            scale = tile_w / float(zw)
            draw_boxes(ImageDraw.Draw(zoom), prim["boxes"], scale,
                       offset=(-x0, -y0), color=(255, 220, 0))
        else:
            zoom = full
        rd = r["radar"]
        line1 = "f%02d  %s  sh %.0f  ct %.1f" % (
            r["i"], fmt_ratio(prim["ratio"]), prim["sharp"], prim["contrast_db"])
        line2 = "iou %.2f  rd %s  score %.2f%s" % (
            r.get("iou", 0.0),
            ("%.0f" % rd["snr"]) if rd and rd["ok"] else "MISS",
            r["score"], "  PICK" if r["i"] in picked else "")
        tiles.append((full, zoom, line1, line2, r["i"] in picked))
    cols = min(4, len(tiles))
    lines = (len(tiles) + cols - 1) // cols
    th, bar, pad = tiles[0][0].size[1], 38, 8
    sheet = Image.new("RGB", (cols * tile_w + (cols + 1) * pad,
                              lines * (2 * th + bar + pad) + pad + 28), (24, 26, 30))
    dr, small, mon = ImageDraw.Draw(sheet), font(13), font(15)
    dr.text((pad, 6), "%s | %s | 模态 %s | 帧 %d | 候选 %d"
            % (res["scene"]["id"], res["scene"]["title"], mod,
               len(res["rows"]), len(rows)), fill=(235, 235, 235), font=mon)
    for n, (full, zoom, l1, l2, is_pick) in enumerate(tiles):
        cx = pad + (n % cols) * (tile_w + pad)
        cy = pad + 28 + (n // cols) * (2 * th + bar + pad)
        sheet.paste(full, (cx, cy))
        sheet.paste(zoom, (cx, cy + th))
        dr.rectangle([cx - 2, cy - 2, cx + tile_w + 1, cy + 2 * th + bar + 1],
                     outline=(120, 200, 120) if is_pick else (60, 64, 70), width=2)
        col = (255, 235, 150) if is_pick else (175, 180, 190)
        dr.text((cx + 3, cy + 2 * th + 1), l1, fill=col, font=small)
        dr.text((cx + 3, cy + 2 * th + 17), l2, fill=col, font=small)
    sheet.save(path)
    return path


def fmt_ratio(v):
    return "%.2f%%" % (100.0 * v)


def scene_note(hero, n_pick):
    rd = hero.get("radar") or {}
    bits = ["%d 帧" % n_pick]
    rng = (rd.get("range_m") or hero.get("range_m") or 0.0) / 1000.0
    if rng:
        bits.append("雷达 %.1f km" % rng)
    v = rd.get("v_r") or 0.0
    if abs(v) >= 1.0:
        bits.append(("闭速 %d m/s" if v > 0 else "远离 %d m/s") % abs(round(v)))
    if rd.get("n", 0) > 1:
        bits.append("%d 个目标" % rd["n"])
    return " · ".join(bits)


# ------------------------------------------------------------------------ main
def missing_exports(scene_id, mods, frames):
    """Which (mod, frame) pairs the site would 404 on."""
    gone = []
    for k in mods:
        for i in frames:
            if not any(os.path.isfile(os.path.join(DATA, scene_id, k, fkey(i) + ext))
                       for ext in (".webp", ".png")):
                gone.append("%s/%s" % (k, fkey(i)))
    return gone


def print_table(res, picks):
    sc = res["scene"]
    print("\n== %s  %s  (%s)" % (sc["id"], sc["title"],
                                 os.path.relpath(res["sdir"], ROOT)))
    if not res["rows"]:
        print("   无可用帧")
        return
    print("   帧号   占比     目标px    清晰   对比dB  位移%  三模态  雷达SNR   得分  结论")
    picked = {r["i"] for r in picks}
    for r in res["rows"]:
        prim, rd = r["prim"], r["radar"]
        tri = all(k in r["per"] for k in ("optical", "lwir", "radar"))
        rs = "%.0fdB" % rd["snr"] if rd and rd["ok"] else ("漏检" if rd else "无")
        if r["i"] in picked:
            verdict = "入选" + ("（例外：" + r["warn"] + "）" if r.get("warn") else "")
        else:
            verdict = ("；".join(r["fails"]) or "备选") + ("（例外）" if r.get("warn") else "")
        bw_px = max((b.x1 - b.x0 for b in prim["boxes"]), default=0.0)
        print("   %4d  %6.2f%%  %6.1f  %7.1f  %6.1f  %5.2f  %-6s  %7s  %5.2f  %s"
              % (r["i"], 100.0 * prim["ratio"], bw_px, prim["sharp"],
                 prim["contrast_db"], 100.0 * r["motion_frac"],
                 "齐" if tri else "缺", rs, r["score"], verdict))


def run(args):
    os.makedirs(PREVIEW, exist_ok=True)
    ids = [args.scene] if args.scene else [s["id"] for s in SCENES]
    out_scenes, skipped = [], []
    for sid in ids:
        scene = SCENE_BY_ID.get(sid)
        if scene is None:
            print("未知场景: %s" % sid)
            continue
        sheet_path = os.path.join(PREVIEW, "%s_contact.png" % sid)
        res = scene_score(scene)
        if res is None:
            skipped.append((sid, "out/%s 不存在，可能仍在渲染" % sid))
            print("\n== %s  跳过：out/%s 不存在（可能仍在渲染）" % (sid, sid))
            continue
        if res.get("no_labels"):
            skipped.append((sid, "缺少 labels/optical，渲染未完成"))
            print("\n== %s  跳过：缺少 labels/optical，渲染未完成" % sid)
            continue
        picks = pick_frames(res["rows"], wanted_count(len(res["gated"]),
                                                     scene.get("max_pick", 6)))
        print_table(res, picks)
        if not picks:
            why = {}
            for r in res["rows"]:
                for f in r["fails"]:
                    why[f.split("(")[0]] = why.get(f.split("(")[0], 0) + 1
            top = "、".join("%s %d 帧" % kv for kv in
                           sorted(why.items(), key=lambda kv: -kv[1])[:3])
            skipped.append((sid, "无合格帧（%s）" % top))
            print("   -> 无合格帧，本轮不上线：%s" % top)
            if os.path.isfile(sheet_path):      # drop a sheet from an older run
                os.remove(sheet_path)
            continue
        if not args.no_sheet:
            try:
                contact_sheet(res, picks, sheet_path)
                print("   复核图: %s" % os.path.relpath(sheet_path, ROOT))
            except Exception as exc:      # a broken sheet must not lose the ranking
                print("   复核图生成失败: %s" % exc)
        mods, exported = site_mods(sid, res["sdir"], res["idxs"])
        gone = missing_exports(sid, mods, [r["i"] for r in picks])
        if gone:
            print("   导出缺 %d 个（站点会 404）：%s%s"
                  % (len(gone), "、".join(gone[:4]), " …" if len(gone) > 4 else ""))
        else:
            print("   导出齐: %d 模态 × %d 帧" % (len(mods), len(picks)))
        hero = max(picks, key=lambda r: r["score"])
        out_scenes.append(dict(
            id=sid, title=scene["title"],
            cls=(res["meta"].get("categories") or [{}])[0].get("name") or "",
            band=res["meta"].get("radar_band") or "",
            aspect=aspect_of(res), mods=mods, frames=[r["i"] for r in picks],
            hero=dict(mod=scene["hero_mod"] if scene["hero_mod"] in mods else "optical",
                      frame=hero["i"]),
            note=scene_note(hero, len(picks)), caption=scene["caption"]))
        if not exported:
            print("   注意: data/%s 尚未导出，mods 取自 out/" % sid)
        print("   入选: %s  主视觉 f%02d   %s"
              % ([r["i"] for r in picks], hero["i"], out_scenes[-1]["note"]))
    payload = dict(generated=datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                   scenes=out_scenes)
    if not args.dry:
        with open(CURATION, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=1)
            fh.write("\n")
        print("\n写出 %s（%d 个场景）" % (os.path.relpath(CURATION, ROOT), len(out_scenes)))
    if skipped:
        print("跳过: " + "；".join("%s（%s）" % s for s in skipped))
    return payload


def main():
    ap = argparse.ArgumentParser(description="空域场景选帧：打分、挑帧、出复核图")
    ap.add_argument("--all", action="store_true", help="处理全部空域场景（默认）")
    ap.add_argument("--scene", help="只处理一个场景 id")
    ap.add_argument("--no-sheet", action="store_true", help="不生成复核对比图")
    ap.add_argument("--dry", action="store_true", help="只打印，不写 curation.json")
    args = ap.parse_args()
    payload = run(args)
    print("\n最终清单")
    for s in payload["scenes"]:
        print("  %-24s %-14s %-5s %s  frames=%s  hero=%s/%d  %s"
              % (s["id"], s["title"], s["aspect"], "/".join(s["mods"]),
                 s["frames"], s["hero"]["mod"], s["hero"]["frame"], s["note"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
