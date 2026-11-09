"""Publish the 2026-11-09 v1 real-photo samples to the static demo site.

Why this exists
    The showcase still led with three obviously synthetic frames: a desert
    plate with grey box/sphere stand-ins (v3jam_land), a flat blue gradient
    sky (final_air) and a murky grey cloud plate (demo_three_ships_plate).
    The 海陆空1109v1 dataset and the 18 workbench presets were rendered on
    real photographed plates (road / clouds / sea) with the glare artefact
    removed. This script puts 5 frames x 3 co-registered modalities from that
    dataset on the site and retires the stale cards.

Inputs (override the root with --src or the SIMKIT_V1_DIR env var)
    <repo parent>/dataest/海陆空1109v1/<domain>/images/
        optical/00000N.png          768x576 RGB
        ir_preview_lwir/00000N.png  768x576 L (8-bit preview)
        radar/00000N.png            1280x720 RGB
    _tmp/preset_samples_trimodal.png (optional): the glare-free three-domain
        contact sheet produced from the 18 curated presets. When it is missing
        the sheet is rebuilt from the v1 optical first frames instead.

Outputs (all inside the repository, so Pages serves them from main)
    data/v1_<domain>/{optical,lwir,radar}/00000N.webp       5 frames each
    data/_t/v1_<domain>/{optical,lwir,radar}/00000N.webp    thumbnails
    assets/figures/fig19_trimodal_v1.webp                   README hero figure
    data/curated.json    3 new scenes first, 3 stale scenes retired
    index.html           static hero / og:image / RD figure point at v1_air

The site reads curated.json first and only falls back to manifest.json, so the
new cards carry their own mods/frames/note/caption and manifest.json is left
untouched (a 400 KB single-line file; rewriting it would create a huge diff).

Usage
    env\\Scripts\\python.exe tools\\export_site_v1.py
    env\\Scripts\\python.exe tools\\export_site_v1.py --check   # report only
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shutil

from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
THUMBS = os.path.join(DATA, "_t")
FIGURES = os.path.join(ROOT, "assets", "figures")
# Derived from the repository's parent so a clone with a sibling dataest/
# checkout works without editing the script; override for any other layout.
V1 = os.environ.get("SIMKIT_V1_DIR") or os.path.join(
    os.path.dirname(ROOT), "dataest", "海陆空1109v1")
FRAMES = 5
MODS = ("optical", "lwir", "radar")

DOMAINS = (
    dict(key="air", src="air_plane", title="空域 · 云层实拍底板",
         cls="fighter", band="Ku",
         note="5 帧 · 云层实拍底板 · 飞机 + 6 枚下抛红外诱饵 · 三模态同轴配准",
         caption="真实云层照片作背景，飞机保持 30 px 级弱小目标；6 枚诱饵沿航向轴线对称、逐帧向下抛射，不随机体平移。"),
    dict(key="land", src="ground_vehicle", title="陆域 · 城市道路航拍底板",
         cls="vehicle", band="Ku",
         note="5 帧 · 环岛/多车道实拍底板 · 浅色厢式货车 + 匀速车流 + 静止路障",
         caption="真实航拍道路照片带车道线、停放车辆与建筑；主目标是 6.75 m 浅色厢式货车，3 辆正常车辆各自匀速行驶，3 个路侧障碍严格静止。"),
    dict(key="sea", src="sea_ship", title="海域 · 开阔海面实拍底板",
         cls="ship", band="Ku",
         note="5 帧 · 海浪实拍底板 · 导弹艇 + 航行小艇 + 锚泊小艇 · 海杂波脊",
         caption="真实海浪照片作底板，主目标舰船与一艘航行小艇、一艘锚泊小艇同框；雷达 RD 图上可见零多普勒海杂波脊。"),
)

RETIRE = ("final_air", "demo_three_ships_plate", "v3jam_land")


def jload(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def jwrite(path, obj):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1)
        fh.write("\n")


def _src_path(dom, mod, i):
    sub = {"optical": "optical", "lwir": "ir_preview_lwir",
           "radar": "radar"}[mod]
    return os.path.join(V1, dom["src"], "images", sub, "%06d.png" % i)


def _as_rgb(path):
    im = Image.open(path)
    if im.mode == "RGB":
        return im.copy()
    if im.mode in ("L", "I;16", "I", "P"):
        # 16-bit LWIR must be windowed to 8-bit before WebP encoding.
        if im.mode == "I;16":
            import numpy as np
            a = np.asarray(im, dtype=np.float32)
            lo, hi = float(a.min()), float(a.max())
            a = (a - lo) / max(hi - lo, 1.0) * 255.0
            return Image.fromarray(a.astype("uint8"), "L").convert("RGB")
        return im.convert("L").convert("RGB")
    return im.convert("RGB")


def export_frames(check=False):
    made = 0
    for dom in DOMAINS:
        sid = "v1_" + dom["key"]
        for mod in MODS:
            full = os.path.join(DATA, sid, mod)
            thumb = os.path.join(THUMBS, sid, mod)
            if not check:
                os.makedirs(full, exist_ok=True)
                os.makedirs(thumb, exist_ok=True)
            for i in range(FRAMES):
                src = _src_path(dom, mod, i)
                if not os.path.isfile(src):
                    raise SystemExit("缺少源帧：" + src)
                name = "%06d.webp" % i
                dst = os.path.join(full, name)
                tdst = os.path.join(thumb, name)
                if check:
                    made += 1
                    continue
                im = _as_rgb(src)
                im.save(dst, "WEBP", quality=86, method=5)
                t = im.copy()
                t.thumbnail((320, 240), Image.LANCZOS)
                t.save(tdst, "WEBP", quality=76, method=5)
                made += 1
    return made


def export_hero_figure(check=False):
    dst = os.path.join(FIGURES, "fig19_trimodal_v1.webp")
    src = os.path.join(ROOT, "_tmp", "preset_samples_trimodal.png")
    if check:
        return dst, os.path.isfile(dst)
    os.makedirs(FIGURES, exist_ok=True)
    if os.path.isfile(src):
        im = Image.open(src).convert("RGB")
    else:
        # Rebuild from the shipped v1 first frames, so the export stays
        # reproducible after _tmp/ is cleaned.
        cells = []
        for dom in DOMAINS:
            cells.append(_as_rgb(_src_path(dom, "optical", 0))
                         .resize((768, 576), Image.LANCZOS))
        im = Image.new("RGB", (768 * 3, 576 + 26), (245, 245, 245))
        d = ImageDraw.Draw(im)
        for i, (dom, cell) in enumerate(zip(DOMAINS, cells)):
            im.paste(cell, (i * 768, 26))
            d.text((i * 768 + 8, 6), dom["title"], fill=(20, 20, 20))
    im.save(dst, "WEBP", quality=88, method=5)
    return dst, True


def update_curated(check=False):
    path = os.path.join(DATA, "curated.json")
    cur = jload(path)
    keep = [s for s in cur["scenes"] if s["id"] not in RETIRE
            and not s["id"].startswith("v1_")]
    added = []
    for dom in DOMAINS:
        added.append(dict(
            id="v1_" + dom["key"], title=dom["title"], cls=dom["cls"],
            band=dom["band"], aspect="4:3",
            mods=list(MODS), frames=list(range(FRAMES)),
            hero=dict(mod="optical", frame=0),
            note=dom["note"], caption=dom["caption"]))
    cur["scenes"] = added + keep
    cur["generated"] = _dt.datetime.now().astimezone().replace(
        microsecond=0).isoformat()
    cur["domain"] = "air + land + sea (v1 real-plate) + jam"
    if not check:
        # Preserve the original indent=1 formatting, which export_jam_web.py
        # documents as a byte-stable round trip.
        jwrite(path, cur)
    return [s["id"] for s in cur["scenes"]]


def update_index(check=False):
    path = os.path.join(ROOT, "index.html")
    with open(path, "r", encoding="utf-8") as fh:
        html = fh.read()
    mapping = {
        "data/demo_air_headon/optical/000004.webp":
            "data/v1_air/optical/000000.webp",
        "data/demo_air_headon/mwir/000004.webp":
            "data/v1_air/lwir/000000.webp",
        "data/demo_air_headon/lwir/000004.webp":
            "data/v1_air/lwir/000000.webp",
        "data/demo_air_headon/radar/000004.webp":
            "data/v1_air/radar/000000.webp",
        "中波红外 3–5 µm": "长波红外 8–12 µm",
        # Static fallback text shown before site.js renders the model.
        "demo_air_headon": "v1_air",
    }
    hits = {k: html.count(k) for k in mapping}
    for old, new in mapping.items():
        html = html.replace(old, new)
    if not check:
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write(html)
    return hits


def main():
    global V1
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--src", default=V1,
                    help="v1 数据集根目录（默认取仓库同级 dataest/海陆空1109v1，"
                         "也可用 SIMKIT_V1_DIR 环境变量指定）")
    args = ap.parse_args()
    V1 = os.path.abspath(args.src)
    if not os.path.isdir(V1):
        raise SystemExit("找不到 v1 数据集：" + V1 +
                         "\n用 --src 指向它，或先跑 "
                         "tools/build_sealandair_1109v1.py --force 生成。")
    n = export_frames(args.check)
    hero, ok = export_hero_figure(args.check)
    scenes = update_curated(args.check)
    hits = update_index(args.check)
    print("frames:", n, "(45 = 3 domains x 5 frames x 3 modalities)")
    print("hero figure:", hero, "present=", ok)
    print("curated scenes:", len(scenes), scenes)
    print("index.html replacements:", hits)
    if not args.check:
        print("DONE: 站点数据已更新（manifest.json 未改）")


if __name__ == "__main__":
    main()
