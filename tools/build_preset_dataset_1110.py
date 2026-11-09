"""Build the 2026-11-10 preset dataset (18 scenes x 30 frames x 3 modalities).

Deliverable: D:\\work\\多模态数据集生成\\dataest\\1110

What it builds
    Every preset in configs/recipes/p_*.json -- 3 domains (air / land / sea)
    x 3 photographed backdrops x 2 ECM groups (active / passive) -- is expanded
    by simkit.scenegen.build_scene into a full scene config and rendered for
    30 frames instead of the 5 frames the workbench previews use.

    Layout::

        1110/
          air/<preset>/images/{optical,ir_lwir,ir_preview_lwir,radar}
          land/<preset>/...
          sea/<preset>/...
          scenes/<preset>.json         reproducible scene configs
          contact_<preset>.png         first/mid/last frame sheet
          index.json                   machine-readable scene table
          generation_report.json       per-scene acceptance
          README.md

Why the frames were raised to 30
    A 5-frame clip is 0.5 s: long enough to show that flares descend and traffic
    moves, too short to train a tracker or to estimate a detection curve. At the
    preset dt of 0.1 s, 30 frames give a 3 s continuous segment while keeping
    every scene's closing geometry inside its authored range band.

Usage
    env\\Scripts\\python.exe tools\\build_preset_dataset_1110.py
    env\\Scripts\\python.exe tools\\build_preset_dataset_1110.py --check
    env\\Scripts\\python.exe tools\\build_preset_dataset_1110.py --only p_land_city_active --frames 3
    env\\Scripts\\python.exe tools\\build_preset_dataset_1110.py --skip-gen   # restamp + verify only
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import glob
import json
import math
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECIPES = os.path.join(ROOT, "configs", "recipes")
PY = os.path.join(ROOT, "env", "Scripts", "python.exe")
DEFAULT_DEST = os.path.normpath(os.path.join(
    os.path.dirname(ROOT), "dataest", "1110"))
SENSORS = "optical,ir,radar"
MODALITIES = ("optical", "ir_lwir", "radar_rd")
DOMAIN_DIR = {"air": "air", "land": "land", "sea": "sea"}

# ---------------------------------------------------------------------------
# Dataset framing profile
# ---------------------------------------------------------------------------
# The workbench presets are tuned for a 5-frame preview, not for a detection
# dataset, and two of the three domains miss their own stated target scale:
#
#   air   footprint 350 m -> fighter 35 px   (note says 28-34 px)  keep
#   land  footprint 400 m -> van 13 x 7 px   (note says 28-34 px)  too small
#   sea   footprint 245 m -> patrol boat 111 px                    close-up
#
# Only the framing is overridden here -- ground coverage, plate span and the
# photographic centre. Every ECM entity, sensor and seed stays exactly as the
# preset authored it, so a scene still reproduces the preset's engagement.
#
# land: 175 m across -> 6.75 m van = 29.6 px. Plate span 350 m is 2x the frame,
#       so the REPEAT seam sits outside the picture, and the photo is only
#       upscaled 1.3x (sharp). The UV centre is picked per plate: the preset
#       default (0.20, 0.45) lands on the marina basin of the roundabout photo
#       once the plate is no longer stretched over 1200 m.
FRAMING = {
    "land": dict(ground_span_m=175.0, plate_span=350.0),
    "sea": None,     # 111 px patrol boat is a normal close-surveillance scale
    "air": None,     # 37 px fighter already matches the preset note
}

# Per-plate photographic centre in PHOTO coordinates (u right, v down). The
# Blender plate mapping flips v (offset_v = 1 - photo_v), so these are converted
# when the recipe is written -- keep this table readable against the photos.
PLATE_UV = {
    # multi-lane asphalt with lane markings, west of the roundabout
    "06_Drone_view_of_roundabout_Unsplash.jpg": (0.30, 0.58),
    # farm track crossing the field grid (the road runs along the top)
    "06_2011-08-17_13-24-50_Serbia_Popinci_jpg.jpg": (0.34, 0.20),
    # warehouse roofs + service road (the dataset substitutes this plate for
    # the authored Mechelen photo, which is an oblique town view with a visible
    # horizon and cannot fill a nadir frame -- see PLATE_OVERRIDE)
    "04_Starr-141025-2603-Casuarina_equisetifolia-aerial_view_li.jpg": (0.45, 0.58),
}

# The authored Mechelen plate is a long-range oblique of a town: mapped onto a
# nadir ground plane its upper half is horizon haze, which showed up as a pale
# band across the frame. Swap in the top-down warehouse aerial so the scene
# matches its own title ("工业区停车场"). Only the plate changes.
PLATE_OVERRIDE = {
    "03_Mechelen_aerial_jpg.jpg":
        "04_Starr-141025-2603-Casuarina_equisetifolia-aerial_view_li.jpg",
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def jload(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def jwrite(path, obj):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)


def preset_paths():
    return sorted(glob.glob(os.path.join(RECIPES, "p_*.json")))


def preset_domain(recipe):
    tags = recipe.get("tags") or []
    for key in ("air", "land", "sea"):
        if key in tags:
            return key
    name = str(recipe.get("name") or "")
    for key in ("air", "land", "sea"):
        if name.startswith("p_%s_" % key):
            return key
    raise SystemExit("无法判定预设域：%r" % (recipe.get("name"),))


def abs_asset(path, base=os.path.join(ROOT, "scenes")):
    """Blender resolves model paths against its own cwd; keep them absolute so
    the scene file is position-independent."""
    if not isinstance(path, str) or not path or os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base, path))


def _ecm_group(recipe, cfg):
    """active/passive from the preset name; ecm.mode is often absent in cfg."""
    name = str(recipe.get("name") or "")
    if name.endswith("_active"):
        return "active"
    if name.endswith("_passive"):
        return "passive"
    return "active" if (cfg.get("ecm") or {}).get("mode") == "active" else "passive"


def _target_px(cfg, asset, scale):
    """Longest projected dimension of the primary target, in pixels."""
    import math as _math
    try:
        from simkit.scenegen import _object_size
        sz = _object_size(asset, scale or 1.0)
        size = max(sz) if isinstance(sz, (list, tuple)) else float(sz)
    except Exception:
        return 0.0
    o = (cfg.get("sensors") or {}).get("optical") or {}
    h = abs(float((o.get("pos") or [0, 0, 0])[2] or 0.0))
    fov = float(o.get("fov_deg") or 0.0)
    if h <= 0 or fov <= 0:
        return 0.0
    foot = 2.0 * h * _math.tan(_math.radians(fov) / 2.0)
    return size * float(o.get("width") or 768) / max(foot, 1e-6)


def scene_from_recipe(recipe, frames, domain):
    from simkit import scenegen

    rec = json.loads(json.dumps(recipe))          # never mutate the preset file
    frame = FRAMING.get(domain)
    applied = {}
    if frame:
        bg = (rec.get("environment") or {}).get("background") or {}
        plate = str(bg.get("file") or "")
        if plate in PLATE_OVERRIDE:
            bg["file"] = PLATE_OVERRIDE[plate]
            applied["plate_swap"] = [plate, bg["file"]]
            plate = bg["file"]
        if frame.get("ground_span_m"):
            rec["view"]["ground_span_m"] = float(frame["ground_span_m"])
            applied["ground_span_m"] = float(frame["ground_span_m"])
        if frame.get("plate_span"):
            bg["span_m"] = float(frame["plate_span"])
            applied["plate_span_m"] = float(frame["plate_span"])
        uv = PLATE_UV.get(plate)
        if uv:
            # photo v points down, the plate mapping points up
            bg["offset"] = [float(uv[0]), round(1.0 - float(uv[1]), 6)]
            applied["plate_uv_photo"] = [float(uv[0]), float(uv[1])]
        rec["environment"]["background"] = bg

    cfg = scenegen.build_scene(rec)
    cfg["frames"]["n"] = frames
    cfg["clip"]["frames_per_clip"] = frames
    cfg["capture"]["frames"] = frames
    contract = cfg.setdefault("__trimodal_contract", {})
    contract["frames"] = frames
    contract["modalities"] = list(MODALITIES)
    cfg["realism"] = dict(enabled=True, jpeg=True)
    # The photographed plate already carries sun and lamp highlights; running
    # the synthetic glare stage on top turns white paint and flare pellets into
    # white blobs, which is fatal for a detection sample.
    cfg.setdefault("optical", {}).setdefault("camera", {}).setdefault(
        "flare", {})["enabled"] = False
    for o in cfg.get("objects") or []:
        if o.get("model"):
            o["model"] = abs_asset(o["model"])
    for holder, key in ((cfg.get("optical") or {}, "sky_image"),
                        (cfg.get("ir") or {}, "background_image"),
                        (cfg.get("ground") or {}, "backdrop")):
        if holder.get(key):
            holder[key] = abs_asset(holder[key])
    return cfg, applied


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------
def run_gen(scene_path, out_dir, frames, threads):
    cmd = [PY, "-m", "simkit", "gen", "-c", scene_path, "-o", out_dir,
           "-n", str(frames), "-s", SENSORS, "-q"]
    env = dict(os.environ)
    env["PYTHONPATH"] = ROOT
    if threads:
        env["SIMKIT_BLENDER_THREADS"] = str(threads)
    t0 = time.time()
    p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    dt = time.time() - t0
    tail = ((p.stdout or "") + "\n" + (p.stderr or ""))[-6000:]
    if p.returncode != 0:
        raise RuntimeError("simkit gen failed (%s) on %s:\n%s"
                           % (p.returncode, os.path.basename(scene_path), tail))
    return dt, tail


def stamp_optical_meta(run_dir, cfg, frames_n):
    """Deterministic pHYs + tEXt on every delivered optical PNG."""
    import numpy as np
    from PIL import Image
    sys.path.insert(0, ROOT)
    from simkit import png as png_writer

    folder = os.path.join(run_dir, "images", "optical")
    if not os.path.isdir(folder):
        return 0
    view = ((cfg.get("__trimodal_contract") or {}).get("view") or {})
    fov = float(view.get("fov_deg")
                or ((cfg.get("sensors") or {}).get("optical") or {}).get("fov_deg")
                or ((cfg.get("clip") or {}).get("seeker") or {}).get("fov_deg")
                or 10.0)
    fr_path = os.path.join(run_dir, "frames.json")
    frames = jload(fr_path) if os.path.isfile(fr_path) else []
    stamped = 0
    names = sorted(n for n in os.listdir(folder) if n.lower().endswith(".png"))
    for i, fn in enumerate(names):
        path = os.path.join(folder, fn)
        arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        meta = dict(modality="optical", fov_deg=fov,
                    radiometric_calibration="relative-linear; not absolute",
                    dataset="preset1110", scene=str(cfg.get("name") or ""))
        if i < len(frames):
            rng = (frames[i] or {}).get("range_m")
            if rng:
                meta["range_m"] = round(float(rng), 3)
                meta["gsd_m_per_px"] = round(
                    2.0 * float(rng) * math.tan(math.radians(fov) / 2.0)
                    / max(arr.shape[1], 1), 6)
        png_writer.write_rgb8(path, arr, meta=meta)
        stamped += 1
    return stamped


def count_files(path, suffix):
    if not os.path.isdir(path):
        return 0
    return len([n for n in os.listdir(path) if n.lower().endswith(suffix)])


def verify_scene(run_dir, cfg, frames_n):
    d = dict(
        optical=count_files(os.path.join(run_dir, "images", "optical"), ".png"),
        optical_jpg=count_files(os.path.join(run_dir, "images", "optical"), ".jpg"),
        ir_lwir=count_files(os.path.join(run_dir, "images", "ir_lwir"), ".png"),
        ir_preview=count_files(os.path.join(run_dir, "images", "ir_preview_lwir"), ".png"),
        radar_png=count_files(os.path.join(run_dir, "images", "radar"), ".png"),
        radar_npy=count_files(os.path.join(run_dir, "raw", "radar"), ".npy"),
    )
    fp = os.path.join(run_dir, "frames.json")
    d["frames"] = len(jload(fp)) if os.path.isfile(fp) else 0
    coco = os.path.join(run_dir, "annotations_ir_lwir_coco.json")
    d["coco_anns"] = (len((jload(coco) or {}).get("annotations") or [])
                      if os.path.isfile(coco) else 0)
    checks = {}
    for k in ("optical", "ir_lwir", "radar_png", "radar_npy", "frames"):
        checks[k + "_full"] = d[k] == frames_n
    checks["coco_complete"] = d["coco_anns"] == frames_n
    d["checks"] = checks
    d["passed"] = all(checks.values())
    return d


def contact_sheet(run_dir, name, frames_n):
    from PIL import Image, ImageDraw

    picks = sorted({0, frames_n // 4, frames_n // 2,
                    (3 * frames_n) // 4, frames_n - 1})
    cell_w, cell_h = 256, 192
    sheet = Image.new("RGB", (cell_w * len(picks), cell_h * 3 + 22),
                      (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    rows = [("optical", "images/optical"),
            ("ir_lwir", "images/ir_preview_lwir"),
            ("radar_rd", "images/radar")]
    for r, (label, sub) in enumerate(rows):
        draw.text((4, r * cell_h + 2), label, fill=(20, 20, 20))
        for c, idx in enumerate(picks):
            p = os.path.join(run_dir, sub, "%06d.png" % idx)
            if not os.path.isfile(p):
                continue
            im = Image.open(p).convert("RGB")
            im.thumbnail((cell_w, cell_h - 16))
            sheet.paste(im, (c * cell_w, r * cell_h + 16))
            draw.text((c * cell_w + 4, r * cell_h + 2), "f%d" % idx,
                      fill=(120, 120, 120))
    out = os.path.join(os.path.dirname(os.path.dirname(run_dir)),
                       "contact_" + name + ".png")
    sheet.save(out)
    return out


# ---------------------------------------------------------------------------
# docs
# ---------------------------------------------------------------------------
def write_readme(dest, scenes, frames_n, ok):
    lines = [
        "# 海 / 陆 / 空 预设全场景数据集（2026-11-10）",
        "",
        "本目录由 `tools/build_preset_dataset_1110.py` 从工作台的 18 个预设配方",
        "（`configs/recipes/p_*.json`，3 域 × 3 实拍底板 × 有源/无源干扰）批量生成。",
        "",
        "- 每个场景 **%d 帧**连续片段，三模态同源：可见光 / 长波红外 LWIR / 毫米波雷达距离-多普勒 RD。" % frames_n,
        "- 分辨率 768×576，dt = 0.1 s，共 %d 个场景、%d 帧、%d 张模态图。"
        % (len(scenes), len(scenes) * frames_n, len(scenes) * frames_n * 3),
        "- 底板为实拍照片（云层 / 城市道路 / 乡村 / 工业区 / 海面 / 港口 / 岛礁），目标为 3D 模型。",
        "",
        "## 目录结构",
        "",
        "```",
        "1110/",
        "  air|land|sea/<preset>/",
        "    images/optical/*.png          # 可见光（附同名 .jpg）",
        "    images/ir_lwir/*.png          # 16 位长波红外",
        "    images/ir_preview_lwir/*.png  # 8 位红外预览",
        "    images/radar/*.png            # 距离-多普勒",
        "    raw/radar/*.npy               # float32 功率张量",
        "    labels/                       # YOLO 标注",
        "    annotations_*.json            # COCO 标注",
        "    frames.json                   # 逐帧真值（距离 / 目标像素 / 天气）",
        "  scenes/<preset>.json            # 可复现场景配置（资产绝对路径）",
        "  contact_<preset>.png            # 首/中/末帧对照拼图",
        "  index.json                      # 场景索引（域 / 底板 / 干扰组 / 路径）",
        "  generation_report.json          # 逐场景验收",
        "```",
        "",
        "## 场景清单",
        "",
        "| 域 | 场景 | 底板 | 干扰组 | 帧数 |",
        "|---|---|---|---|---|",
    ]
    for s in scenes:
        lines.append("| %s | `%s` | %s | %s | %d |"
                     % (s["domain"], s["name"], s["backdrop"],
                        s["ecm_group"], s["frames"]))
    lines += [
        "",
        "## 复现",
        "",
        "```powershell",
        "set PYTHONPATH=D:\\work\\多模态数据集生成\\RGB_SAR_LIDAR_SIMU",
        "env\\Scripts\\python.exe tools\\build_preset_dataset_1110.py --force",
        "```",
        "",
        "`--skip-gen` 只重做元数据与验收（不重渲）；`--only <preset>` 只跑单个场景；",
        "`--jobs N` 控制场景级并发。",
        "",
        "## 取景档（与工作台预览的差别）",
        "",
        "配方本身按 5 帧预览调过，直接放大到 30 帧会暴露两处尺度问题，因此本数据集",
        "只覆盖**取景**参数，干扰实体、传感器与种子完全沿用配方：",
        "",
        "| 域 | 改动 | 原因 |",
        "|---|---|---|",
        "| 空 | 无 | 350 m 覆盖下战斗机约 37 px，符合配方自述 |",
        "| 陆 | 地面覆盖 400 → 175 m；底板跨度 1200 → 350 m；按底板重选取景中心 | 原设置下 6.75 m 货车仅 13×7 px，与配方自述的 28–34 px 不符 |",
        "| 海 | 无 | 245 m 覆盖下导弹艇约 111 px，属正常近距监视尺度 |",
        "| 陆·工业 | 底板换为厂区航拍 `04_Starr-…-Casuarina…` | 原底板是带地平线的斜视城镇照，铺到俯视地面会出现苍白雾带 |",
        "",
        "底板 UV 按照片坐标记录，写入场景时按 Blender 的 v 轴方向做 `1 - v` 换算。",
        "",
        "## 验收",
        "",
        "- 全部场景 %d 帧 × 3 模态齐备：**%s**" % (frames_n, "PASS" if ok else "FAIL"),
        "- 每张光学 PNG 内嵌 `tEXt`（模态 / FOV / 距离 / GSD）与 `pHYs` 物理分辨率。",
        "",
        "## 已知局限",
        "",
        "- 目标是 Cycles 渲染的 3D 模型，与实拍底板的照明/阴影不可能完全一致。",
        "- LWIR 背景由可见光底板做温度映射，只提供结构，不是标定过的辐射量。",
        "- 雷达 RD 为几何 PO 估计，未用实测 RCS / 海杂波 K 分布标定。",
        "- %d 帧 = %.1f s 片段，适合短时序训练与检测评估，不能替代长时跟踪集。"
        % (frames_n, frames_n * 0.1),
        "",
    ]
    with open(os.path.join(dest, "README.md"), "w", encoding="utf-8",
              newline="") as fh:
        fh.write("\n".join(lines))


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=DEFAULT_DEST)
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--threads", type=int, default=6,
                    help="Blender threads per scene")
    ap.add_argument("--jobs", type=int, default=3,
                    help="scenes rendered concurrently")
    ap.add_argument("--only", default=None, help="single preset name")
    ap.add_argument("--limit", type=int, default=0, help="first N presets")
    ap.add_argument("--skip-gen", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--no-contact", action="store_true")
    args = ap.parse_args()

    dest = os.path.abspath(args.dest)
    scene_dir = os.path.join(dest, "scenes")
    os.makedirs(scene_dir, exist_ok=True)

    recipes = []
    for path in preset_paths():
        rec = jload(path)
        if args.only and rec.get("name") != args.only:
            continue
        recipes.append(rec)
    if args.limit:
        recipes = recipes[:args.limit]
    if not recipes:
        raise SystemExit("没有匹配的预设")

    plan = []
    for rec in recipes:
        dom = preset_domain(rec)
        cfg, applied = scene_from_recipe(rec, args.frames, dom)
        cfg["name"] = rec["name"]
        scene_path = os.path.join(scene_dir, rec["name"] + ".json")
        jwrite(scene_path, cfg)
        run_dir = os.path.join(dest, DOMAIN_DIR[dom], rec["name"])
        primary = (cfg.get("objects") or [{}])[0]
        plan.append(dict(name=rec["name"], domain=dom, recipe=rec, cfg=cfg,
                         scene=scene_path, run=run_dir, framing=applied,
                         backdrop=os.path.basename(str(
                             (cfg.get("ground") or {}).get("backdrop")
                             or (cfg.get("optical") or {}).get("sky_image") or "-")),
                         ecm_group=_ecm_group(rec, cfg)))
        print("CFG %-30s %-4s %-34s px~%-5.1f %s" % (
            rec["name"], dom, plan[-1]["backdrop"],
            _target_px(cfg, (rec.get("objects") or [{}])[0].get("asset"),
                       (rec.get("objects") or [{}])[0].get("scale")),
            applied or "preset-as-authored"))

    if args.check:
        print("\nCHECK ok: %d scenes, %d frames each -> %s"
              % (len(plan), args.frames, dest))
        return 0

    if args.force and not args.skip_gen:
        for p in plan:
            if os.path.isdir(p["run"]):
                shutil.rmtree(p["run"])

    if not args.skip_gen:
        t0 = time.time()
        errors = []

        def work(p):
            os.makedirs(os.path.dirname(p["run"]), exist_ok=True)
            print("GEN  start %-30s %s" % (p["name"], p["domain"]), flush=True)
            dt, _ = run_gen(p["scene"], p["run"], args.frames, args.threads)
            print("GEN  done  %-30s %5.1f s (%.2f s/frame)"
                  % (p["name"], dt, dt / max(args.frames, 1)), flush=True)
            return dt

        with futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
            futs = {pool.submit(work, p): p for p in plan}
            for fut in futures.as_completed(futs):
                p = futs[fut]
                try:
                    p["gen_s"] = round(fut.result(), 1)
                except Exception as exc:            # keep going, report at end
                    p["gen_error"] = str(exc)[-1500:]
                    errors.append((p["name"], str(exc)[-400:]))
                    print("GEN  FAIL  %s" % p["name"], flush=True)
        print("GEN total %.1f s" % (time.time() - t0))
        for name, err in errors:
            print("!! %s: %s" % (name, err))

    scenes, ok = [], True
    for p in plan:
        stamped = stamp_optical_meta(p["run"], p["cfg"], args.frames)
        rep = verify_scene(p["run"], p["cfg"], args.frames)
        rep.update(name=p["name"], domain=p["domain"],
                   backdrop=p["backdrop"], ecm_group=p["ecm_group"],
                   frames_target=args.frames, stamped=stamped,
                   gen_seconds=p.get("gen_s"),
                   gen_error=p.get("gen_error"))
        if not args.no_contact:
            try:
                rep["contact"] = os.path.basename(
                    contact_sheet(p["run"], p["name"], args.frames))
            except Exception as exc:
                rep["contact_error"] = str(exc)[:200]
        ok = ok and rep["passed"]
        scenes.append(dict(
            name=p["name"], domain=p["domain"],
            title=(p["recipe"].get("title") or "").replace(
                "5帧", "%d帧" % args.frames),
            note=(p["recipe"].get("note") or ""),
            backdrop=p["backdrop"], ecm_group=p["ecm_group"],
            entities=[e["id"] for e in (p["cfg"].get("ecm") or {}).get("entities", [])],
            framing=p.get("framing") or {},
            target_px=round(_target_px(
                p["cfg"], (p["recipe"].get("objects") or [{}])[0].get("asset"),
                (p["recipe"].get("objects") or [{}])[0].get("scale")), 1),
            frames=args.frames,
            run=os.path.relpath(p["run"], dest).replace("\\", "/"),
            scene=os.path.relpath(p["scene"], dest).replace("\\", "/"),
            passed=rep["passed"]))
        print("[%s] %-30s optical=%d ir=%d rd=%d npy=%d coco=%d"
              % ("PASS" if rep["passed"] else "FAIL", p["name"],
                 rep["optical"], rep["ir_lwir"], rep["radar_png"],
                 rep["radar_npy"], rep["coco_anns"]))
        if not rep["passed"]:
            print("      failed:", [k for k, v in rep["checks"].items() if not v])

    jwrite(os.path.join(dest, "generation_report.json"), dict(
        dest=dest, frames_per_scene=args.frames, scenes=len(scenes),
        modalities=list(MODALITIES), dt_s=0.1,
        resolution=[768, 576],
        per_scene=scenes, passed=bool(ok)))
    jwrite(os.path.join(dest, "index.json"), dict(
        dataset="preset1110", frames=args.frames, dt_s=0.1,
        domains={k: [s["name"] for s in scenes if s["domain"] == k]
                 for k in ("air", "land", "sea")},
        scenes=scenes))
    write_readme(dest, scenes, args.frames, ok)
    print("\nOVERALL", "PASS" if ok else "FAIL", "->", dest)
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
