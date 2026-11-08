#!/usr/bin/env python
"""Independent acceptance gate for the 20-frame trimodal seeker datasets.

Checks five groups and writes validation.json + dataset.json into every run:
  config      - the generator must honour the authored config (no silent fallback)
  visibility  - REAL projected box 18..22 px, crisp but never dominant
  alignment   - optical / LWIR / radar share frame ids, pose, one target
  trainability- images/labels one-to-one, clean training sources declared
  stability   - seed propagated to the Blender side-car, deterministic run

Exit code 0 = all hard gates pass. Warnings do not fail the gate but are
reported, so low-information frames are visible instead of being hidden.

Usage
    python tools/validate_trimodal.py --dest "D:/work/多模态数据集生成/dataest"
    python tools/validate_trimodal.py --dest ... --emit-yolo
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PX_MIN, PX_MAX = 18.0, 22.0
W, H = 768, 576
SCENES = ["air_plane", "ground_vehicle", "sea_ship"]
CFG_PREFIX = "ds20_v2_"
CLASSES = {"air_plane": "fighter", "ground_vehicle": "vehicle", "sea_ship": "ship"}


def jload(path, dflt=None):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return dflt


def jwrite(path, obj):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1)
        fh.write("\n")


def ids(pattern):
    return sorted(int(os.path.basename(p)[:6]) for p in glob.glob(pattern)
                  if os.path.basename(p)[:6].isdigit())


def boxes(path):
    d = jload(path)
    if not d or not isinstance(d.get("boxes"), dict):
        return []
    return [b for b in d["boxes"].values()
            if isinstance(b, dict) and b.get("visible") is not False
            and all(k in b for k in ("x0", "y0", "x1", "y1"))]


def check_config(run, cfg_path):
    c = jload(cfg_path, {}) or {}
    out = []
    v = (c.get("clip") or {}).get("views") or []
    sk = (c.get("clip") or {}).get("seeker") or {}
    rd = (c.get("sensors") or {}).get("radar") or {}
    ir = c.get("ir") or {}

    def add(ok, name, detail):
        out.append(dict(ok=bool(ok), check=name, detail=detail))

    add(len(v) == 1 and v[0].get("width") == W and v[0].get("height") == H,
        "clip.views 单视图 768x576", str(v))
    for key in ("optical", "ir_camera"):
        s = (c.get("sensors") or {}).get(key) or {}
        add(s.get("width") == W and s.get("height") == H
            and s.get("fov_deg") == v[0].get("fov_deg"),
            "sensors.%s 与 views 一致" % key,
            "wxh=%sx%s fov=%s" % (s.get("width"), s.get("height"), s.get("fov_deg")))
    add(all(x not in (0, None) for x in (v[0].get("width"), v[0].get("height"))),
        "无 0x0 静默回退", "width=%s height=%s" % (v[0].get("width"), v[0].get("height")))
    add((c.get("frames") or {}).get("n") == 20
        and (c.get("clip") or {}).get("frames_per_clip") == 20,
        "20 帧 / 单 clip", "n=%s fpc=%s" % ((c.get("frames") or {}).get("n"),
                                            (c.get("clip") or {}).get("frames_per_clip")))
    add(sk.get("target_px") == [18, 22] and sk.get("target_m") is not None,
        "target_px 门禁与显式 target_m", "target_px=%s target_m=%s"
        % (sk.get("target_px"), sk.get("target_m")))
    fov = float(v[0].get("fov_deg") or 1)
    size = float(sk.get("target_m") or 1)
    band = [size * W / (2 * p * math.tan(math.radians(fov / 2)))
            for p in (PX_MAX, PX_MIN)]
    rng = sk.get("range_m") or [0, 0]
    # The authored [lo,hi] is a starting interval; plan._plan_seeker clamps it
    # into the target_px band at run time.  The authoritative gate is therefore
    # the ACTUAL per-frame slant range, not the authored endpoints.
    frames = jload(os.path.join(run, "frames.json"), []) if run else []
    actual = [float(r.get("range_m") or 0) for r in frames if r.get("range_m")]
    if actual:
        add(min(actual) >= band[0] * 0.95 and max(actual) <= band[1] * 1.05,
            "逐帧斜距落在 target_px 反解带内",
            "actual=%.0f..%.0f m  band=%.0f..%.0f m"
            % (min(actual), max(actual), band[0], band[1]))
    # The planner clamps authored [lo,hi] into the band, so an interval that
    # only grazes it is silently truncated. Require the authored interval to
    # COVER the whole band -- otherwise the config claims a distance range the
    # generator can never produce.
    add(min(rng) <= band[0] and max(rng) >= band[1],
        "配置 range_m 覆盖 target_px 反解带",
        "authored=%s band=%.0f..%.0f m"
        % ([round(x) for x in rng], band[0], band[1]))
    total_d = float(sk.get("wander_deg") or 0) + float(sk.get("los_jitter_deg") or 0)
    limit = 2 * fov / W
    add(total_d <= limit * 1.5, "抖动按 fov 缩放",
        "delta=%.4f deg  limit=%.4f deg" % (total_d, limit))
    add(float(((c.get("clip") or {}).get("rig") or {}).get("vibration_deg") or 0) == 0,
        "rig.vibration_deg 归零", "rig=%s" % ((c.get("clip") or {}).get("rig") or {}))
    n = int(rd.get("n_samples") or 0)
    bw = float(rd.get("bandwidth_mhz") or 1)
    r_max = 3e8 * n / (4 * bw * 1e6)
    add(r_max >= 1.4 * max(rng), "雷达距离窗覆盖斜距",
        "r_max=%.0f m  1.4R=%.0f m" % (r_max, 1.4 * max(rng)))
    add("pfa" in rd and "det_snr_db" not in rd,
        "门限单一来源 pfa", "pfa=%s det_snr_db=%s" % (rd.get("pfa"), rd.get("det_snr_db")))
    add(ir.get("bands") == ["lwir"] and ir.get("psf_px", 9) <= 1.0,
        "单波段 LWIR + 小 PSF", "bands=%s psf=%s" % (ir.get("bands"), ir.get("psf_px")))
    add(not (c.get("output") or {}).get("yolo"),
        "不写入仓库 datasets/yolo", "output=%s" % (c.get("output") or {}))

    # v2: the background MUST be a real photograph from the asset library and
    # must be shared by the optical ground plate, the optical sky card and the
    # IR plate, otherwise the three modalities show different scenes.
    bg = (c.get("ground") or {}).get("backdrop")
    sky = (c.get("optical") or {}).get("sky_image")
    irbg = (c.get("ir") or {}).get("background_image")
    has_ground = bool((c.get("ground") or {}).get("enabled"))
    if has_ground:
        # look_down: ground.backdrop is the plate that actually enters the
        # optical frame; optical sky card and IR plate must be the same photo.
        plate, keys_ok = bg, (bool(bg) and bg == sky == irbg)
    else:
        # Sky-only scenes have NO ground plane. A ground.backdrop would make
        # ir._composite_background project the plate onto z=0: the upper half
        # of the LWIR frame then comes back pure black (measured 51%). Two
        # presentation keys, and explicitly NO ground plate.
        plate, keys_ok = sky, (bool(sky) and sky == irbg and not bg)
    bg_ok = (keys_ok and "assets/backdrops/" in str(plate)
             and os.path.isfile(os.path.normpath(os.path.join(ROOT, "scenes", plate))))
    add(bg_ok, "实拍背景同源且几何分支正确",
        "ground_enabled=%s ground.backdrop=%s sky=%s ir=%s"
        % (has_ground, bg, sky, irbg))

    # v2 realism switches are opt-in per scene; never change kernel defaults.
    cam = (c.get("optical") or {}).get("camera") or {}
    stages = [k for k in ("vignetting", "chromatic_aberration", "flare", "jpeg")
              if isinstance(cam.get(k), dict) and cam[k].get("enabled")]
    real = ((c.get("atmosphere") or {}).get("profile") == "exponential"
            and len(stages) >= 3
            and (c.get("optical") or {}).get("auto_exposure") is False
            and (c.get("ir") or {}).get("preview_span_c") is not None
            and all(o.get("temp_c") is None for o in c.get("objects", [])))
    add(real, "真实性开关已启用",
        "atm=%s stages=%s auto_exposure=%s preview_span=%s temp_c=%s"
        % ((c.get("atmosphere") or {}).get("profile"), stages,
           (c.get("optical") or {}).get("auto_exposure"),
           (c.get("ir") or {}).get("preview_span_c"),
           [o.get("temp_c") for o in c.get("objects", [])]))
    return out


def check_visibility(run):
    rows = []
    for lab in sorted(glob.glob(os.path.join(run, "labels", "optical", "*.json"))):
        i = int(os.path.basename(lab)[:6])
        for b in boxes(lab):
            w = float(b["x1"]) - float(b["x0"])
            h = float(b["y1"]) - float(b["y0"])
            cx = (float(b["x0"]) + float(b["x1"])) / 2
            cy = (float(b["y0"]) + float(b["y1"])) / 2
            rows.append(dict(i=i, w=w, h=h,
                             width_pct=100.0 * w / W,
                             area_pct=100.0 * w * h / (W * H),
                             margin=min(cx / W, cy / H, (W - cx) / W, (H - cy) / H)))
    fr = jload(os.path.join(run, "frames.json"), []) or []
    nom = {int(r.get("frame", -1)): r.get("target_px") for r in fr}
    out = []

    def add(ok, name, detail):
        out.append(dict(ok=bool(ok), check=name, detail=detail))

    add(len(rows) == 20, "每帧恰好 1 个目标框", "boxes=%d" % len(rows))
    if not rows:
        return out, {}
    ws = [r["w"] for r in rows]
    add(min(ws) >= PX_MIN and max(ws) <= PX_MAX, "真实框宽 18–22 px",
        "min=%.2f max=%.2f mean=%.2f" % (min(ws), max(ws), float(np.mean(ws))))
    add(max(r["width_pct"] for r in rows) <= 3.0
        and max(r["area_pct"] for r in rows) <= 0.2,
        "小目标占比上限",
        "max width=%.3f%% max area=%.4f%%" % (max(r["width_pct"] for r in rows),
                                               max(r["area_pct"] for r in rows)))
    add(min(r["margin"] for r in rows) >= 0.02, "目标不贴边",
        "min margin=%.3f" % min(r["margin"] for r in rows))
    drift = [abs(r["w"] - nom[r["i"]]) for r in rows
             if nom.get(r["i"]) is not None and isinstance(nom[r["i"]], (int, float))]
    add(bool(drift) and max(drift) <= 0.5,
        "frames.target_px 与真实框一致（标注脱节）",
        "max |nominal-real|=%.3f px over %d frames" % (max(drift) if drift else -1, len(drift)))
    stat = dict(box_w_min=min(ws), box_w_max=max(ws), box_w_mean=float(np.mean(ws)),
                width_pct_max=max(r["width_pct"] for r in rows),
                area_pct_max=max(r["area_pct"] for r in rows),
                nominal_max_dev=max(drift) if drift else None)
    return out, stat


def check_alignment(run):
    out = []

    def add(ok, name, detail):
        out.append(dict(ok=bool(ok), check=name, detail=detail))

    groups = {
        "optical": os.path.join(run, "images", "optical", "*.png"),
        "ir_lwir": os.path.join(run, "images", "ir_lwir", "*.png"),
        "ir_preview": os.path.join(run, "images", "ir_preview_lwir", "*.png"),
        "radar": os.path.join(run, "images", "radar", "*.png"),
        "radar_clean": os.path.join(run, "images", "radar_clean", "*.png"),
        "radar_tensor": os.path.join(run, "raw", "radar", "*.npy"),
        "labels_optical": os.path.join(run, "labels", "optical", "*.json"),
        "labels_ir": os.path.join(run, "labels", "ir_lwir", "*.json"),
        "labels_radar": os.path.join(run, "labels", "radar", "*.json"),
    }
    sets = {k: ids(v) for k, v in groups.items()}
    base = list(range(20))
    for k, v in sets.items():
        add(v == base, "%s 帧号 0–19 齐全" % k,
            "count=%d first=%s last=%s" % (len(v), v[:1], v[-1:]))

    def deviation(opt, ir):
        """Normalised centre / size difference; 0 means byte-identical boxes."""
        return max(abs((opt["x0"] + opt["x1"]) / 2 - (ir["x0"] + ir["x1"]) / 2) / W,
                   abs((opt["y0"] + opt["y1"]) / 2 - (ir["y0"] + ir["y1"]) / 2) / H,
                   abs((opt["x1"] - opt["x0"]) - (ir["x1"] - ir["x0"])) / W)

    devs, rdevs, dets, dev_frames = [], [], [], []
    for i in base:
        a = boxes(os.path.join(run, "labels", "optical", "%06d.json" % i))
        b = boxes(os.path.join(run, "labels", "ir_lwir", "%06d.json" % i))
        if a and b:
            devs.append(deviation(a[0], b[0]))
            dev_frames.append(i)
        rlab = jload(os.path.join(run, "labels", "radar", "%06d.json" % i), {}) or {}
        tg = rlab.get("targets") or []
        if tg:
            dets.append(bool(tg[0].get("detected")))
            rng = float(tg[0].get("range") or 0)
            fr = (jload(os.path.join(run, "frames.json"), []) or [{}])
            truth = next((float(r.get("range_m") or 0) for r in fr
                          if int(r.get("frame", -1)) == i), 0)
            if truth:
                rdevs.append(abs(rng - truth))
    worst_i = dev_frames[int(np.argmax(devs))] if devs else -1
    add(bool(devs) and max(devs) <= 0.002, "光学↔LWIR 归一化框偏差 ≤0.002",
        "max=%.6f over %d frames (worst f%02d)" %
        (max(devs) if devs else -1, len(devs), worst_i))
    add(bool(dets) and all(dets), "雷达逐帧检出", "detected %d/%d" % (sum(dets), len(dets)))
    add(bool(rdevs) and max(rdevs) <= 30.0, "雷达距离与运动真值自洽",
        "max |R_radar - R_truth|=%.2f m" % (max(rdevs) if rdevs else -1))
    return out


def check_trainability(run, key):
    out = []

    def add(ok, name, detail, warn=False):
        out.append(dict(ok=bool(ok), check=name, detail=detail, warn=warn))

    coco = glob.glob(os.path.join(run, "annotations_*_coco.json"))
    add(len(coco) >= 1, "COCO 标注存在", ",".join(os.path.basename(p) for p in coco))
    # raw/radar ships both .npy (power tensor) and .npz (QA buffers); count only
    # the training product, otherwise a correct run reports 40 files.
    for name, rel, ext in (("optical", "images/optical", "*.png"),
                           ("ir_preview_lwir", "images/ir_preview_lwir", "*.png"),
                           ("radar_tensor", "raw/radar", "*.npy")):
        fs = sorted(glob.glob(os.path.join(run, rel, ext)))
        add(len(fs) == 20, "训练源 %s 20 个文件" % name, "count=%d" % len(fs))
    # 8-bit IR must not be a flat field (information, not just alignment)
    stds = []
    for p in sorted(glob.glob(os.path.join(run, "images", "ir_preview_lwir", "*.png"))):
        stds.append(float(np.asarray(Image.open(p).convert("L"), dtype=float).std()))
    add(bool(stds) and min(stds) >= 1.0, "红外预览非平场",
        "std min=%.2f mean=%.2f" % (min(stds) if stds else -1,
                                     float(np.mean(stds)) if stds else -1), warn=True)
    # A grounded plate on a sky-only scene once produced 51% pure-black LWIR
    # frames; std stayed high (textured lower half), so the flat-field check
    # above did NOT catch it. Gate the dead-black fraction explicitly.
    black = []
    for p in sorted(glob.glob(os.path.join(run, "images", "ir_preview_lwir", "*.png"))):
        a = np.asarray(Image.open(p).convert("L"), dtype=float)
        black.append(100.0 * float((a == 0).mean()))
    add(bool(black) and max(black) <= 2.0, "红外无大面积死黑",
        "max black=%.2f%%" % (max(black) if black else -1))
    # optical must carry real scene texture
    ost = []
    for p in sorted(glob.glob(os.path.join(run, "images", "optical", "*.png"))):
        ost.append(float(np.asarray(Image.open(p).convert("L"), dtype=float).std()))
    add(bool(ost) and min(ost) >= 1.0, "可见光非平场",
        "std min=%.2f mean=%.2f" % (min(ost) if ost else -1,
                                     float(np.mean(ost)) if ost else -1), warn=True)
    tensors = sorted(glob.glob(os.path.join(run, "raw", "radar", "*.npy")))
    shapes = set()
    for p in tensors:
        shapes.add(np.load(p, mmap_mode="r").shape)
    add(len(shapes) == 1 and all(s > 0 for s in next(iter(shapes), (0,))),
        "RD 张量形状一致且非空", "shapes=%s" % [str(s) for s in shapes])

    # The blue team found stats.json could stay stale after label repair, and
    # that COCO bbox/label agreement was never checked. Make both hard gates.
    for mod, coco_name in (("optical", "annotations_optical_coco.json"),
                           ("ir_lwir", "annotations_ir_lwir_coco.json")):
        d = jload(os.path.join(run, coco_name), {}) or {}
        by_img = {int(a["image_id"]): a for a in d.get("annotations", [])}
        bad = []
        for i in range(20):
            lab = jload(os.path.join(run, "labels", mod, "%06d.json" % i), {}) or {}
            bs = [b for b in (lab.get("boxes") or {}).values() if isinstance(b, dict)]
            a = by_img.get(i)
            if bool(bs) != bool(a):
                bad.append(i)
                continue
            if a and bs:
                b, bb = bs[0], a["bbox"]
                if (abs(bb[0] - b["x0"]) > 0.05 or abs(bb[1] - b["y0"]) > 0.05
                        or abs(bb[2] - (b["x1"] - b["x0"])) > 0.05):
                    bad.append(i)
        add(not bad, "COCO↔标签逐帧一致 (%s)" % mod,
            "mismatch frames=%s" % bad[:5])
    st = jload(os.path.join(run, "stats.json"), {}) or {}
    add(st.get("ir_boxes") == 20, "stats.ir_boxes 与补齐后标注同步",
        "ir_boxes=%s optical_boxes=%s" % (st.get("ir_boxes"), st.get("optical_boxes")))
    zero = []
    for i in range(20):
        lab = jload(os.path.join(run, "labels", "ir_lwir", "%06d.json" % i), {}) or {}
        bs = [b for b in (lab.get("boxes") or {}).values() if isinstance(b, dict)]
        if bs and int(bs[0].get("pixels") or 0) == 0:
            zero.append(i)
    add(not zero, "红外实例像素非零", "zero-response frames=%s" % zero, warn=True)
    return out


def check_stability(run, cfg_path):
    c = jload(cfg_path, {}) or {}
    meta = jload(os.path.join(run, "meta.json"), {}) or {}
    side = jload(os.path.join(run, "_render_cfg.json"), {}) or {}
    out = []

    def add(ok, name, detail):
        out.append(dict(ok=bool(ok), check=name, detail=detail))

    add(meta.get("seed") == c.get("seed"), "meta.seed 与配置一致",
        "meta=%s cfg=%s" % (meta.get("seed"), c.get("seed")))
    add(side.get("seed") == c.get("seed"),
        "Blender side-car seed 已透传（--seed 修复）",
        "sidecar=%s cfg=%s" % (side.get("seed"), c.get("seed")))
    add(meta.get("frames") == 20, "meta.frames=20", "frames=%s" % meta.get("frames"))
    return out


def emit_yolo(run, key):
    cls = CLASSES[key]
    written = 0
    for mod in ("optical", "ir_lwir"):
        dst = os.path.join(run, "labels", "yolo", mod)
        os.makedirs(dst, exist_ok=True)
        for lab in sorted(glob.glob(os.path.join(run, "labels", mod, "*.json"))):
            d = jload(lab, {}) or {}
            w, h = float(d.get("width") or W), float(d.get("height") or H)
            lines = []
            for b in boxes(lab):
                x0, y0, x1, y1 = (float(b[k]) for k in ("x0", "y0", "x1", "y1"))
                lines.append("0 %.6f %.6f %.6f %.6f" %
                             ((x0 + x1) / 2 / w, (y0 + y1) / 2 / h,
                              (x1 - x0) / w, (y1 - y0) / h))
            with open(os.path.join(dst, os.path.basename(lab).replace(".json", ".txt")),
                      "w", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + ("\n" if lines else ""))
            written += 1
    jwrite(os.path.join(run, "labels", "yolo", "classes.txt"),
           ["0:" + cls])
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=r"D:\work\多模态数据集生成\dataest")
    ap.add_argument("--emit-yolo", action="store_true")
    a = ap.parse_args()
    dest = os.path.abspath(a.dest)
    overall, summary = True, []
    for key in SCENES:
        run = os.path.join(dest, key)
        cfg = os.path.join(ROOT, "scenes", CFG_PREFIX + key + ".json")
        if not os.path.isdir(run):
            print("MISSING", run)
            overall = False
            continue
        groups = {
            "config": check_config(run, cfg),
        }
        vis, stats = check_visibility(run)
        groups["visibility"] = vis
        groups["alignment"] = check_alignment(run)
        groups["trainability"] = check_trainability(run, key)
        groups["stability"] = check_stability(run, cfg)
        hard = [c for g in groups.values() for c in g
                if not c["ok"] and not c.get("warn")]
        warns = [c for g in groups.values() for c in g
                 if not c["ok"] and c.get("warn")]
        ok = not hard
        overall &= ok
        report = dict(scene=key, passed=ok, stats=stats, groups=groups,
                      hard_failures=[c["check"] for c in hard],
                      warnings=[c["check"] for c in warns])
        jwrite(os.path.join(run, "validation.json"), report)
        if a.emit_yolo:
            report["yolo_txt_files"] = emit_yolo(run, key)
            jwrite(os.path.join(run, "validation.json"), report)
        jwrite(os.path.join(run, "dataset.json"), dict(
            name=key, classes=[CLASSES[key]], frames=20,
            modalities=["optical", "ir_lwir", "radar_rd"],
            train_sources=dict(optical="images/optical/*.png",
                               infrared="images/ir_preview_lwir/*.png (8-bit display product; "
                                        "images/ir_lwir/*.png is the 16-bit radiometric product)",
                               radar="raw/radar/*.npy (float32 range x doppler power tensor)"),
            notes="images/radar* are annotated display figures with axes/tables; "
                  "do not use them as training pixels. labels/radar holds the "
                  "range/velocity truth and the analytic detection gate.",
            optical_note="optical.sensor.enabled is OFF by design. An A/B probe "
                         "(tools/v2_probe_report.json) showed the HDR detector "
                         "chain compressed contrast (e.g. 176 -> 67 DN), so the "
                         "8-bit ISP path was kept. Consequence: the visible band "
                         "has NO atmospheric path radiance and NO photon/shot "
                         "noise; atmosphere=exponential still governs IR/SAR/"
                         "LiDAR. Optical pixels are NOT radiometrically "
                         "calibrated -- do not use them for quantitative "
                         "radiometry. camera vignetting / chromatic aberration / "
                         "flare / diffraction / sensor are applied (5 stages).",
            backdrop_note="Backgrounds are real photographs from "
                          "assets/backdrops tiled onto the ground plane. "
                          "KNOWN LIMITATION: look-down projection can expose a "
                          "low-frequency diagonal tiling seam (measured anti-"
                          "diagonal edge projection max/med = 1.99 in the "
                          "ground-vehicle run). Enlarging backdrop_span_m only "
                          "moves/enlarges the seam, it does not remove it; the "
                          "renderer's plate projection needs a separate fix. "
                          "It affects display only, not the projected boxes, "
                          "radar truth or IR geometry.",
            split_note="One 20-frame continuous clip: splits.json (train 0-16 / "
                       "val 17-19) has adjacent-frame temporal correlation. "
                       "For an unbiased evaluation, generate a second clip with "
                       "a different seed instead of treating val as independent.",
            detector_note="labels/radar.detected is an analytic SNR gate, not an "
                          "observed RD peak; train RD detectors against "
                          "raw/radar/*.npy with the supplied truth rows.",
            validation="validation.json"))
        print("\n== %-16s %s ==" % (key, "PASS" if ok else "FAIL"))
        for gname, checks in groups.items():
            bad = [c for c in checks if not c["ok"]]
            print("  %-14s %2d/%2d ok%s" % (gname, len(checks) - len(bad), len(checks),
                                             "" if not bad else "  -> " +
                                             "; ".join("%s [%s]" % (c["check"], c["detail"])
                                                       for c in bad[:3])))
        if stats:
            print("  visibility     box %.2f–%.2f px, max width %.3f%%, max area %.4f%%"
                  % (stats["box_w_min"], stats["box_w_max"],
                     stats["width_pct_max"], stats["area_pct_max"]))
        summary.append(dict(scene=key, passed=ok, **stats))
    jwrite(os.path.join(dest, "validation_summary.json"),
           dict(passed=overall, scenes=summary))
    print("\nOVERALL:", "PASS" if overall else "FAIL")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
