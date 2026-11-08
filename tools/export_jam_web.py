#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""W1 干扰数据导出 —— 一键生成 v3 干扰场景卡片 + 干扰实验室图片与数据。

只用 Pillow / numpy（零其它依赖），绝不调用 export_scene_web.py 或
opt_images.optimize()，不改 boot.js / cloud / 既有场景目录。

用法：
    python tools/export_jam_web.py
    python tools/export_jam_web.py --frames 0,9,19,29
    python tools/export_jam_web.py --frames 0 5 10 29 --src D:/work/.../dataest
"""
import argparse
import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DEFAULT_SRC = Path("D:/work/多模态数据集生成/dataest")
DEFAULT_PROBE = ROOT / "_tmp" / "v3_probe"
REPORT = ROOT / "tools" / "v3_probe_report.json"

# 固定 dB 窗口，禁止逐帧 percentile 自适应。
LO, HI = -85.0, 5.0
STOPS = [
    (0.00, (0, 0, 4)),
    (0.15, (22, 11, 57)),
    (0.35, (87, 16, 110)),
    (0.55, (160, 44, 84)),
    (0.75, (222, 93, 32)),
    (0.90, (250, 168, 25)),
    (1.00, (252, 255, 164)),
]

DEFAULT_FRAMES = [0, 9, 19, 29]
THUMB_W = 420
RD_SIZE = (1280, 720)
Q = {"optical": 80, "lwir": 82, "radar": 88}

# sid 必须避开 site.js 的 NON_AIR 正则 /ground|surface|sea|ship|boat|vehicle|terrain|city/i
DOMAINS = [
    {
        "dom": "air_plane", "sid": "v3jam_air", "jam_id": "air", "cls": "fighter",
        "title": "干扰 · 空域有源/无源", "label": "空域 · 有源 + 无源 ECM",
        "kind": "ecm", "tier": "mid",
        "note": "30 帧 · 有源压制(3 W ERP / 200 MHz) + 箔条云(RCS 140 m²) + 假目标 + 红外诱饵 · 目标 31 px",
        "caption": "宽带噪声把 RD 基底整体抬高约 17 dB，目标仍可检出；光学目标轮廓不变，干扰只发生在雷达检测层。",
        "entities": [
            {"name": "噪声压制 jammer", "group": "active", "spec": "3 W ERP / 200 MHz"},
            {"name": "箔条云 chaff", "group": "passive", "spec": "RCS 140 m²"},
        ],
        "clean_label": "无干扰基线",
        "jam_label": "有源压制 + 箔条 + 红外诱饵",
    },
    {
        "dom": "ground_vehicle", "sid": "v3jam_land", "jam_id": "land", "cls": "vehicle",
        "title": "干扰 · 陆域地物/地杂波", "label": "陆域 · 地物杂物 + 地杂波",
        "kind": "objects", "tier": "strong",
        "note": "30 帧 · label:false 地面杂物/建筑 + 地杂波(clutter_density 0.08) · 目标 31 px",
        "caption": "地面域不抬升全图噪声基底，而是在零多普勒中心竖起强脊(约 +13 dB)；光学目标不变，杂物均不进 COCO 标注。",
        "entities": [
            {"name": "地面静态杂物", "group": "objects", "spec": "label:false 建筑/杂物 · clutter_density 0.08"},
            {"name": "地杂波 ground clutter", "group": "clutter", "spec": "ground_clutter_db -25 dB / clutter_db -6 dB"},
        ],
        "clean_label": "无干扰基线",
        "jam_label": "地面杂物 + 地杂波",
    },
    {
        "dom": "sea_ship", "sid": "v3jam_naval", "jam_id": "naval", "cls": "ship",
        "title": "干扰 · 海域海杂波脊", "label": "海域 · 海杂波零多普勒脊",
        "kind": "clutter", "tier": "strong",
        "note": "30 帧 · 海杂波零多普勒脊(clutter_ridge=True, scale 6 dB) · 目标 31 px",
        "caption": "海面同样以中央竖脊为特征：Rayleigh + 常数 gamma 合成的零多普勒脊较基线约 +13 dB；光学舰船轮廓不变。",
        "entities": [
            {"name": "海杂波零多普勒脊", "group": "clutter", "spec": "clutter_ridge=True · Rayleigh + 常数 gamma"},
            {"name": "海面杂波电平", "group": "clutter", "spec": "scale 6 dB / sea_clutter_db -15 dB"},
        ],
        "clean_label": "无干扰基线",
        "jam_label": "海杂波零多普勒脊",
    },
]


# ---------------------------------------------------------------- RD 伪彩 ----
def build_lut():
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        for j in range(len(STOPS) - 1):
            x0, c0 = STOPS[j]
            x1, c1 = STOPS[j + 1]
            if t <= x1 or j == len(STOPS) - 2:
                f = 0.0 if x1 == x0 else (t - x0) / (x1 - x0)
                f = min(max(f, 0.0), 1.0)
                lut[i] = [round(c0[k] + f * (c1[k] - c0[k])) for k in range(3)]
                break
    return lut


LUT = build_lut()

# 2048(或 1024) 个距离单元被 LANCZOS 压到 720 显示行时，RD 散斑会混叠成
# 横向条纹，q88 下每张 ~95 KB。对 256 级色标索引沿距离轴做 5-tap 均值
# 抗混叠（固定 dB 窗口 / LUT / LANCZOS / q88 全不变），既保留 +17 dB 基底
# 抬升与 +13 dB 中央脊，又把 27 张 RD 压到 4 MB 预算内。
RD_DEALIAS_K = 5


def _box1d(a, k):
    if k <= 1:
        return a
    p = k // 2
    pad = np.pad(a.astype(np.float64), ((p, p), (0, 0)), mode="edge")
    c = np.vstack([np.zeros((1, a.shape[1])), np.cumsum(pad, axis=0)])
    return (c[k:] - c[:-k]) / k


def rd_rgb(npy_path, size):
    """由 raw/radar/*.npy 生成干净伪彩 RD（无坐标轴/表格）。"""
    rd = np.load(npy_path)
    db = 10.0 * np.log10(np.maximum(rd, 1e-12))
    g = (np.clip((db - LO) / (HI - LO), 0.0, 1.0) * 255.0).astype(np.uint8)
    if RD_DEALIAS_K > 1:
        g = np.clip(_box1d(g, RD_DEALIAS_K), 0, 255).astype(np.uint8)
    rgb = LUT[g]                                  # [range, doppler, 3]
    img = Image.fromarray(rgb, "RGB")
    return img.resize(size, Image.LANCZOS)


def median_floor_db(npy_path):
    rd = np.load(npy_path)
    db = 10.0 * np.log10(np.maximum(rd, 1e-12))
    return float(np.median(db))


# ---------------------------------------------------------------- 图片输出 ----
def save_webp(img, path, quality, method=6):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path), "WEBP", quality=int(quality), method=method)


def load_capped(png_path, cap_w):
    """1280 宽上限：源更宽才缩，绝不放大（thumbnail 语义）。"""
    img = Image.open(png_path).convert("RGB")
    if img.width > cap_w:
        h = max(1, round(img.height * cap_w / img.width))
        img = img.resize((cap_w, h), Image.LANCZOS)
    return img


def load_width(png_path, width):
    """固定缩到 width 宽（jamlab 光学对照用）。"""
    img = Image.open(png_path).convert("RGB")
    h = max(1, round(img.height * width / img.width))
    return img.resize((width, h), Image.LANCZOS)


# ---------------------------------------------------------------- 指标实读 ----
def read_detection(validation):
    for items in validation.get("groups", {}).values():
        for it in items:
            m = re.search(r"detected\s*[=:]?\s*(\d+)\s*/\s*(\d+)", str(it.get("detail", "")))
            if m:
                n, total = int(m.group(1)), int(m.group(2))
                return n, total
    return 0, 30


def final_floor_mean(src, dom):
    floors = []
    for i in range(30):
        floors.append(median_floor_db(src / dom / "raw" / "radar" / f"{i:06d}.npy"))
    return float(np.mean(floors))


def now_stamp():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


# ---------------------------------------------------------------- JSON 追加 ----
def append_scenes(path, new_scenes, new_ids, *, indent=1, compact=False, top=None):
    """去掉同 id + append，保证已有条目对象逐字不变。"""
    raw = Path(path).read_text(encoding="utf-8")
    doc = json.loads(raw)
    if top:
        doc.update(top)
    doc["scenes"] = [s for s in doc["scenes"] if s.get("id") not in new_ids] + new_scenes
    if compact:
        out = json.dumps(doc, ensure_ascii=False, separators=(",", ":"))
    else:
        out = json.dumps(doc, ensure_ascii=False, indent=indent) + "\n"
    Path(path).write_text(out, encoding="utf-8")


# ---------------------------------------------------------------- 主流程 ----
def main():
    ap = argparse.ArgumentParser(description="导出 v3 干扰 WebP 与数据（W1）")
    ap.add_argument("--frames", nargs="*", default=None,
                    help="卡片帧号，默认 0 9 19 29；可写 '0,9,19,29' 或空格分隔")
    ap.add_argument("--src", default=str(DEFAULT_SRC), help="v3 干扰三套数据源根目录")
    ap.add_argument("--probe", default=str(DEFAULT_PROBE), help="探针基线根目录")
    args = ap.parse_args()

    if args.frames:
        toks = []
        for tok in args.frames:
            toks.extend(t for t in re.split(r"[,\s]+", str(tok)) if t != "")
        frames = [int(t) for t in toks]
    else:
        frames = list(DEFAULT_FRAMES)
    if not frames:
        frames = list(DEFAULT_FRAMES)

    src = Path(args.src)
    probe = Path(args.probe)
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    hero_frame = frames[1] if len(frames) > 1 else frames[0]
    new_ids = {d["sid"] for d in DOMAINS}
    counts = {"webp": 0, "bytes": 0}

    def note_save(img, path, quality):
        save_webp(img, path, quality)
        counts["webp"] += 1
        counts["bytes"] += Path(path).stat().st_size

    # 1) 场景卡片：每域 frames × 3 模态（full + thumb）
    for d in DOMAINS:
        dom, sid = d["dom"], d["sid"]
        for i in frames:
            tag = f"{i:06d}"
            # optical
            o_full = load_capped(src / dom / "images" / "optical" / f"{tag}.png", 1280)
            o_path = DATA / sid / "optical" / f"{tag}.webp"
            note_save(o_full, o_path, Q["optical"])
            note_save(o_full.resize((THUMB_W, round(o_full.height * THUMB_W / o_full.width)), Image.LANCZOS),
                      DATA / "_t" / sid / "optical" / f"{tag}.webp", max(60, Q["optical"] - 18))
            # lwir
            l_full = load_capped(src / dom / "images" / "ir_preview_lwir" / f"{tag}.png", 1280)
            l_path = DATA / sid / "lwir" / f"{tag}.webp"
            note_save(l_full, l_path, Q["lwir"])
            note_save(l_full.resize((THUMB_W, round(l_full.height * THUMB_W / l_full.width)), Image.LANCZOS),
                      DATA / "_t" / sid / "lwir" / f"{tag}.webp", max(60, Q["lwir"] - 18))
            # radar（干净伪彩 RD，1280x720）
            note_save(rd_rgb(src / dom / "raw" / "radar" / f"{tag}.npy", RD_SIZE),
                      DATA / sid / "radar" / f"{tag}.webp", Q["radar"])
            note_save(rd_rgb(src / dom / "raw" / "radar" / f"{tag}.npy",
                             (THUMB_W, round(RD_SIZE[1] * THUMB_W / RD_SIZE[0]))),
                      DATA / "_t" / sid / "radar" / f"{tag}.webp", max(60, Q["radar"] - 18))
        print(f"[cards] {sid}: {len(frames)} frames x 3 mods")

    # 2) 干扰实验室图片 + jam_lab.json
    jam_lab = {
        "generated": now_stamp(),
        "resolution": list(report["resolution"]),
        "frames": report["frames"],
        "domains": [],
    }
    for d in DOMAINS:
        dom, jam_id = d["dom"], d["jam_id"]
        validation = json.loads((src / dom / "validation.json").read_text(encoding="utf-8"))
        vm = validation["jam"]["metrics"]["rd"]
        n_det, total_det = read_detection(validation)

        base_probe = probe / f"{dom}_base" / "raw" / "radar" / "000000.npy"
        final0 = src / dom / "raw" / "radar" / "000000.npy"

        # clean / jam / 三档 / 光学对照
        clean_img = rd_rgb(base_probe, RD_SIZE)
        jam_img = rd_rgb(final0, RD_SIZE)
        note_save(clean_img, DATA / "jamlab" / f"{jam_id}_clean.webp", 88)
        note_save(jam_img, DATA / "jamlab" / f"{jam_id}_jam.webp", 88)
        for tier in ("base", "mid", "strong"):
            p = probe / f"{dom}_{tier}" / "raw" / "radar" / "000000.npy"
            note_save(rd_rgb(p, RD_SIZE), DATA / "jamlab" / f"{jam_id}_tier_{tier}.webp", 88)
        opt_img = load_width(src / dom / "images" / "optical" / "000000.png", 1280)
        note_save(opt_img, DATA / "jamlab" / f"{jam_id}_optical.webp", 80)

        # 指标实读
        t = report["domains"][dom]["tiers"]
        mean_final_floor = final_floor_mean(src, dom)
        base_floor = t["base"]["noise_floor_db"]
        floor_rise = round(mean_final_floor - base_floor, 1)
        ridge = round(vm["clutter_db"]["mean"], 1)
        snr = round(vm["target_snr_db"]["mean"], 1)
        detected_pct = round(100.0 * n_det / total_det) if total_det else 0

        metrics = [
            {"key": "floor", "label": "噪声基底抬升", "value": floor_rise, "unit": "dB", "max": 40},
            {"key": "ridge", "label": "零多普勒中心带", "value": ridge, "unit": "dB", "max": 45},
            {"key": "snr", "label": "目标 SNR 均值", "value": snr, "unit": "dB", "max": 80},
            {"key": "detected", "label": "逐帧检出", "value": detected_pct, "unit": "%", "max": 100},
        ]
        tiers = []
        for key, label in (("base", "无干扰"), ("mid", "中档"), ("strong", "强档")):
            tiers.append({
                "key": key, "label": label,
                "rd": f"jamlab/{jam_id}_tier_{key}.webp",
                "floor_db": round(t[key]["noise_floor_db"], 1),
                "center_db": round(t[key]["center_rel_db"], 1),
                "detected": bool(t[key]["detected"]),
            })
        tiers.append({
            "key": "final", "label": "最终采用",
            "rd": f"jamlab/{jam_id}_jam.webp",
            "floor_db": round(mean_final_floor, 1),
            "center_db": ridge,
            "detected": bool(total_det and n_det == total_det),
        })

        jam_lab["domains"].append({
            "id": jam_id,
            "label": d["label"],
            "kind": d["kind"],
            "tier": report["domains"][dom]["recommended_tier"],
            "entities": d["entities"],
            "metrics": metrics,
            "compare": {
                "clean": f"jamlab/{jam_id}_clean.webp",
                "jam": f"jamlab/{jam_id}_jam.webp",
                "clean_label": d["clean_label"],
                "jam_label": d["jam_label"],
                "w": RD_SIZE[0], "h": RD_SIZE[1],
            },
            "optical": f"jamlab/{jam_id}_optical.webp",
            "tiers": tiers,
            "note": d["note"],
        })
        print(f"[jamlab] {jam_id}: floor+{floor_rise} dB ridge={ridge} dB snr={snr} dB "
              f"detected={n_det}/{total_det} final_floor={mean_final_floor:.2f} dB")

    (DATA / "jam_lab.json").write_text(
        json.dumps(jam_lab, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    # 3) curated.json 追加 3 条（前 6 条逐字不动：indent=1 精确 round-trip）
    curated_new = []
    for d in DOMAINS:
        curated_new.append({
            "id": d["sid"],
            "title": d["title"],
            "cls": d["cls"],
            "band": "Ku",
            "aspect": "4:3",
            "mods": ["optical", "lwir", "radar"],
            "frames": list(frames),
            "hero": {"mod": "radar", "frame": hero_frame},
            "note": d["note"],
            "caption": d["caption"],
        })
    append_scenes(DATA / "curated.json", curated_new, new_ids,
                  indent=1, top={"generated": now_stamp(), "domain": "air + sea + jam"})

    # 4) manifest.json 追加 3 条（前 6 条逐字不动：原文件是紧凑格式，保持紧凑）
    manifest_new = []
    for d in DOMAINS:
        dom, sid = d["dom"], d["sid"]
        frames_meta = json.loads((src / dom / "frames.json").read_text(encoding="utf-8"))
        by_frame = {int(f["frame"]): f for f in frames_meta}
        out_frames = []
        for i in frames:
            tag = f"{i:06d}"
            label = json.loads((src / dom / "labels" / "optical" / f"{tag}.json").read_text(encoding="utf-8"))
            fm = by_frame.get(i, {})
            weather = fm.get("weather", label.get("weather", {}))
            opt_w = load_capped(src / dom / "images" / "optical" / f"{tag}.png", 1280).width
            scale = opt_w / 768.0
            boxes = [
                {"x0": round(float(b["x0"]) * scale, 2), "y0": round(float(b["y0"]) * scale, 2),
                 "x1": round(float(b["x1"]) * scale, 2), "y1": round(float(b["y1"]) * scale, 2),
                 "label": "target"}
                for b in label.get("boxes", {}).values()
            ]
            mods = {
                "optical": {"img": f"{sid}/optical/{tag}.webp", "w": opt_w,
                            "h": round(576 * scale), "boxes": boxes},
                "lwir": {"img": f"{sid}/lwir/{tag}.webp", "w": opt_w,
                         "h": round(576 * scale), "boxes": boxes},
                "radar": {"img": f"{sid}/radar/{tag}.webp", "w": RD_SIZE[0],
                          "h": RD_SIZE[1], "boxes": []},
            }
            out_frames.append({
                "i": i,
                "weather": weather,
                "mods": mods,
                "t": float(fm.get("t", 0.0)),
                "range_m": float(fm.get("range_m", 0.0)),
                "target_px": float(fm.get("target_px", 31.0)),
                "views": [],
                "radar": {"n": 0, "clutter": [], "boxes": [], "pts": []},
            })
        manifest_new.append({
            "id": sid,
            "title": d["title"],
            "cls": d["cls"],
            "band": "Ku",
            "budget": {},
            "sensors": [],
            "categories": [],
            "mods": ["optical", "lwir", "radar"],
            "views": [],
            "sar_meta": {},
            "frames": out_frames,
        })
    append_scenes(DATA / "manifest.json", manifest_new, new_ids, compact=True)

    print(f"[done] new webp={counts['webp']} bytes={counts['bytes']} "
          f"({counts['bytes']/1024/1024:.2f} MiB)  frames={frames}")


if __name__ == "__main__":
    main()
