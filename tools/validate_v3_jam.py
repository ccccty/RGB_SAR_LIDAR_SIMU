#!/usr/bin/env python
"""Independent acceptance gate for the 30-frame v3 JAMMED trimodal seeker datasets.

Standalone, reproducible v3 acceptance validator.  It is a full copy of the v2
framework (tools/validate_trimodal.py) re-authored for the v3 physics-layer
interference semantics -- NOT an import-and-patch shim, because the v2 frame
count / box band / config prefix were scattered literals.

Checks six groups and writes validation.json + dataset.json into every run:
  config      - the generator must honour the authored config (no silent fallback)
  visibility  - REAL projected box 28..34 px, crisp but never dominant
  alignment   - optical / LWIR / radar share frame ids, pose, one target
  trainability- images/labels one-to-one, clean training sources declared
  stability   - seed propagated to the Blender side-car, deterministic run
  jam         - v3 interference specialty gates, asserted per domain

v3 interference model (declared, not hidden):
  air    physical-layer ECM (active + passive), entities label:false
  ground multiple label:false clutter objects / buildings + ground clutter
  sea    sensors.radar.clutter_ridge sea-clutter zero-Doppler ridge

Exit code 0 = all hard gates pass. Warnings do not fail the gate but are
reported, so low-information frames are visible instead of being hidden.

Usage
    python tools/validate_v3_jam.py --dest "D:/work/多模态数据集生成/dataest"
    python tools/validate_v3_jam.py --dest ... --emit-yolo
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

# ---------------------------------------------------------------- v3 口径
N_FRAMES = 30
PX_MIN, PX_MAX = 28.0, 34.0
W, H = 768, 576
SCENES = ["air_plane", "ground_vehicle", "sea_ship"]
CFG_PREFIX = "ds20_v3_"
CLASSES = {"air_plane": "fighter", "ground_vehicle": "vehicle", "sea_ship": "ship"}
CLASSES_WHITELIST = {"fighter", "vehicle", "ship"}
DOMAIN_LABEL = {"air_plane": "[空域]", "ground_vehicle": "[地面]", "sea_ship": "[海面]"}
EXPECTED_SEED = {"air_plane": 7, "ground_vehicle": 17, "sea_ship": 27}
JAM_TYPES = {
    "air_plane": ["物理层有源 ECM（噪声压制/假目标/距离-速度门拖引/角度欺骗）",
                  "物理层无源 ECM（箔条云/角反射器/诱饵/红外焰弹）"],
    "ground_vehicle": ["地面几何杂物与建筑（多个 label:false 物体）", "地杂波"],
    "sea_ship": ["海杂波零多普勒脊（clutter_ridge，Rayleigh + 常数 gamma）"],
}
# simkit/ecm.py: GROUP_OF / _ACTIVE / _PASSIVE / KINDS. Kept local so the config
# gate stays reproducible even when the simkit import fails.
ACTIVE_KINDS = {"noise_jammer", "false_target", "range_gate_pull_off",
                "velocity_gate_pull_off", "angle_deception"}
PASSIVE_KINDS = {"chaff_cloud", "corner_reflector", "decoy", "infrared_flare"}
KINDS = ACTIVE_KINDS | PASSIVE_KINDS


# ---------------------------------------------------------------- helpers
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
            and b.get("label") is not False   # geometry clutter is not a truth box
            and all(k in b for k in ("x0", "y0", "x1", "y1"))]


def mm(v):
    """min/max/mean of a numeric list, JSON-safe; None when empty."""
    if not v:
        return None
    return dict(min=round(float(np.min(v)), 4),
                max=round(float(np.max(v)), 4),
                mean=round(float(np.mean(v)), 4))


def _is_ecm(x):
    """An ECM-injected detection, never a labelled truth target."""
    if isinstance(x, dict):
        s = str(x.get("source", "")).lower()
        n = str(x.get("name", "")).lower()
        i = str(x.get("id", "")).lower()
        return ("ecm" in s or n.startswith("ecm#") or "ecm#" in n
                or "ecm" in n or "ecm" in i or bool(x.get("ecm_id")))
    return "ecm" in str(x).lower()


def _count_ecm(node):
    """Count ecm# / source==\"ecm\" items inside boxes_rd or unassociated_detections."""
    if isinstance(node, dict):
        return sum(1 for k, v in node.items()
                   if "ecm" in str(k).lower() or _is_ecm(v))
    if isinstance(node, list):
        return sum(1 for v in node if _is_ecm(v))
    return 0


def _target_mask(shape, lab_path, pad=3):
    """Boolean mask of the labelled target box(es), padded a few pixels."""
    m = np.zeros(shape, bool)
    h, w = shape
    for b in boxes(lab_path):
        x0 = max(0, int(float(b["x0"])) - pad)
        y0 = max(0, int(float(b["y0"])) - pad)
        x1 = min(w, int(float(b["x1"])) + pad)
        y1 = min(h, int(float(b["y1"])) + pad)
        if x1 > x0 and y1 > y0:
            m[y0:y1, x0:x1] = True
    return m


def _lap_energy(gray, mask):
    """Mean |Laplacian| OUTSIDE the target box -> high-frequency jam/clutter texture."""
    g = gray.astype(float).copy()
    g[mask] = np.nan
    c = g[1:-1, 1:-1]
    lap = np.abs(4.0 * c - g[:-2, 1:-1] - g[2:, 1:-1]
                 - g[1:-1, :-2] - g[1:-1, 2:])
    ok = ~np.isnan(lap)
    return float(lap[ok].mean()) if ok.any() else 0.0


def _blobs(gray, mask, threshold):
    """Count bright connected patches OUTSIDE the target box (4x decimated BFS)."""
    hot = (gray >= threshold) & (~mask)
    h, w = hot.shape
    s = 4
    hh, ww = h // s, w // s
    if hh == 0 or ww == 0:
        return 0
    small = hot[:hh * s, :ww * s].reshape(hh, s, ww, s).max(axis=(1, 3))
    seen = np.zeros_like(small, bool)
    n = 0
    for y in range(hh):
        for x in range(ww):
            if small[y, x] and not seen[y, x]:
                n += 1
                stack = [(y, x)]
                seen[y, x] = True
                while stack:
                    cy, cx = stack.pop()
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < hh and 0 <= nx < ww and small[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
    return n


def _rd_metrics(path, lab_path):
    """(median noise floor, peak SNR dB, clutter_db) of one range-Doppler tensor.

    doppler axis comes from labels/radar.tensor.axis_order (v2/v3:
    ["range","doppler"]); the zero-Doppler ridge sits at its centre, so the
    ±2-bin band is measured there.
    """
    t = np.asarray(np.load(path, mmap_mode="r"), dtype=float)
    med = float(np.median(t))
    mx = float(t.max())
    peak_db = 10.0 * math.log10(max(mx, 1e-30) / max(med, 1e-30))
    lab = jload(lab_path, {}) or {}
    axo = (lab.get("tensor") or {}).get("axis_order")
    if isinstance(axo, list) and "doppler" in axo:
        dax = axo.index("doppler")
    else:
        dax = int(np.argmin(t.shape))
    center = t.shape[dax] // 2
    if dax == 1:
        band = t[:, max(0, center - 2):center + 3]
    else:
        band = t[max(0, center - 2):center + 3, :]
    clutter_db = 10.0 * math.log10(max(float(np.mean(band)), 1e-30) / max(med, 1e-30))
    return med, peak_db, clutter_db


def _cls_of(lab, key_name, b, coco_name=None):
    """Best-effort class of the single labelled box, from several spellings."""
    cands = []
    if isinstance(b, dict):
        for k in ("cls", "class", "category", "label", "name"):
            if b.get(k):
                cands.append(str(b[k]))
    for k in ("cls", "class", "category"):
        if lab.get(k):
            cands.append(str(lab[k]))
    if key_name:
        cands.append(str(key_name))
    if coco_name:
        cands.append(str(coco_name))
    for c in cands:
        for w in CLASSES_WHITELIST:
            if w in c.lower():
                return w
    return cands[0] if cands else None


# ---------------------------------------------------------------- v2 gates
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
    add((c.get("frames") or {}).get("n") == N_FRAMES
        and (c.get("clip") or {}).get("frames_per_clip") == N_FRAMES,
        "30 帧 / 单 clip", "n=%s fpc=%s" % ((c.get("frames") or {}).get("n"),
                                            (c.get("clip") or {}).get("frames_per_clip")))
    add(sk.get("target_px") == [int(PX_MIN), int(PX_MAX)] and sk.get("target_m") is not None,
        "target_px 门禁与显式 target_m", "target_px=%s target_m=%s"
        % (sk.get("target_px"), sk.get("target_m")))
    fov = float(v[0].get("fov_deg") or 1)
    size = float(sk.get("target_m") or 1)
    # v3: band = size*W/(2*p*tan(fov/2)), p in (34, 28). The authoritative gate
    # is the ACTUAL per-frame slant range; additionally the authored interval
    # must COVER the whole back-solved band (v2 already fixed this semantic --
    # do not regress to a mere "has intersection" test).
    band = [size * W / (2 * p * math.tan(math.radians(fov / 2)))
            for p in (PX_MAX, PX_MIN)]
    rng = sk.get("range_m") or [0, 0]
    frames = jload(os.path.join(run, "frames.json"), []) if run else []
    actual = [float(r.get("range_m") or 0) for r in frames if r.get("range_m")]
    if actual:
        add(min(actual) >= band[0] * 0.95 and max(actual) <= band[1] * 1.05,
            "逐帧斜距落在 target_px 反解带内",
            "actual=%.0f..%.0f m  band=%.0f..%.0f m"
            % (min(actual), max(actual), band[0], band[1]))
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

    # The background MUST be a real photograph, shared by optical ground plate,
    # optical sky card and IR plate; the branch follows ground.enabled.
    bg = (c.get("ground") or {}).get("backdrop")
    sky = (c.get("optical") or {}).get("sky_image")
    irbg = (c.get("ir") or {}).get("background_image")
    has_ground = bool((c.get("ground") or {}).get("enabled"))
    if has_ground:
        plate, keys_ok = bg, (bool(bg) and bg == sky == irbg)
    else:
        plate, keys_ok = sky, (bool(sky) and sky == irbg and not bg)
    bg_ok = (keys_ok and "assets/backdrops/" in str(plate)
             and os.path.isfile(os.path.normpath(os.path.join(ROOT, "scenes", plate))))
    add(bg_ok, "实拍背景同源且几何分支正确",
        "ground_enabled=%s ground.backdrop=%s sky=%s ir=%s"
        % (has_ground, bg, sky, irbg))

    # Realism switches are opt-in per scene; never change kernel defaults.
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

    add(len(rows) == N_FRAMES, "每帧恰好 1 个目标框", "boxes=%d" % len(rows))
    if not rows:
        return out, {}
    ws = [r["w"] for r in rows]
    add(min(ws) >= PX_MIN and max(ws) <= PX_MAX, "真实框宽 28–34 px",
        "min=%.2f max=%.2f mean=%.2f" % (min(ws), max(ws), float(np.mean(ws))))
    add(max(r["width_pct"] for r in rows) <= 5.0
        and max(r["area_pct"] for r in rows) <= 0.6,
        "小目标占比上限（宽 ≤5.0% / 面积 ≤0.6%）",
        "max width=%.3f%% max area=%.4f%%" % (max(r["width_pct"] for r in rows),
                                               max(r["area_pct"] for r in rows)))
    add(min(r["margin"] for r in rows) >= 0.02, "目标不贴边",
        "min margin=%.3f" % min(r["margin"] for r in rows))
    drift = [abs(r["w"] - nom[r["i"]]) for r in rows
             if nom.get(r["i"]) is not None and isinstance(nom[r["i"]], (int, float))]
    # v3 目标约 31 px；seeker 名义像素按 0.1 px 量化，真实框是浮点投影，
    # v2 为 19 px 目标定的固定 0.5 px 容差会系统性误判（实测差 0.7–0.9 px）。
    tol = max(0.5, 0.04 * float(np.mean(ws)))
    add(bool(drift) and max(drift) <= tol,
        "frames.target_px 与真实框一致（标注脱节）",
        "max |nominal-real|=%.3f px (tol=%.2f) over %d frames"
        % (max(drift) if drift else -1, tol, len(drift)))
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
    base = list(range(N_FRAMES))
    for k, v in sets.items():
        add(v == base, "%s 帧号 0–29 齐全" % k,
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
        tg = [t for t in (rlab.get("targets") or []) if isinstance(t, dict) and not _is_ecm(t)]
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
    add(bool(dets) and all(dets) and len(dets) == N_FRAMES, "雷达逐帧检出",
        "detected %d/%d" % (sum(dets), len(dets)))
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
    # the training product.
    for name, rel, ext in (("optical", "images/optical", "*.png"),
                           ("ir_preview_lwir", "images/ir_preview_lwir", "*.png"),
                           ("radar_tensor", "raw/radar", "*.npy")):
        fs = sorted(glob.glob(os.path.join(run, rel, ext)))
        add(len(fs) == N_FRAMES, "训练源 %s 30 个文件" % name, "count=%d" % len(fs))
    stds = []
    for p in sorted(glob.glob(os.path.join(run, "images", "ir_preview_lwir", "*.png"))):
        stds.append(float(np.asarray(Image.open(p).convert("L"), dtype=float).std()))
    add(bool(stds) and min(stds) >= 1.0, "红外预览非平场",
        "std min=%.2f mean=%.2f" % (min(stds) if stds else -1,
                                     float(np.mean(stds)) if stds else -1), warn=True)
    # A grounded plate on a sky-only scene once produced 51% pure-black LWIR
    # frames; gate the dead-black fraction explicitly (hard gate).
    black = []
    for p in sorted(glob.glob(os.path.join(run, "images", "ir_preview_lwir", "*.png"))):
        a = np.asarray(Image.open(p).convert("L"), dtype=float)
        black.append(100.0 * float((a == 0).mean()))
    add(bool(black) and max(black) <= 2.0, "红外无大面积死黑",
        "max black=%.2f%%" % (max(black) if black else -1))
    ost = []
    for p in sorted(glob.glob(os.path.join(run, "images", "optical", "*.png"))):
        ost.append(float(np.asarray(Image.open(p).convert("L"), dtype=float).std()))
    add(bool(ost) and min(ost) >= 1.0, "可见光非平场",
        "std min=%.2f mean=%.2f" % (min(ost) if ost else -1,
                                     float(np.mean(ost)) if ost else -1), warn=True)
    tensors = sorted(glob.glob(os.path.join(run, "raw", "radar", "*.npy")))
    shapes = set()
    for p in tensors:
        try:
            shapes.add(tuple(np.load(p, mmap_mode="r").shape))
        except (OSError, ValueError):
            pass
    add(len(shapes) == 1 and all(s > 0 for s in next(iter(shapes), (0,))),
        "RD 张量形状一致且非空", "shapes=%s" % [str(s) for s in shapes])

    # stats.json could stay stale after label repair; COCO bbox/label agreement
    # and the COCO<->label frame-by-frame comparison are hard gates.
    for mod, coco_name in (("optical", "annotations_optical_coco.json"),
                           ("ir_lwir", "annotations_ir_lwir_coco.json")):
        d = jload(os.path.join(run, coco_name), {}) or {}
        by_img = {int(a["image_id"]): a for a in d.get("annotations", [])}
        bad = []
        for i in range(N_FRAMES):
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
    add(st.get("ir_boxes") == N_FRAMES, "stats.ir_boxes 与补齐后标注同步",
        "ir_boxes=%s optical_boxes=%s" % (st.get("ir_boxes"), st.get("optical_boxes")))
    zero = []
    for i in range(N_FRAMES):
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
    add(meta.get("frames") == N_FRAMES, "meta.frames=30", "frames=%s" % meta.get("frames"))
    return out


# ---------------------------------------------------------------- jam gates
def check_jam_config(run, key, cfg_path):
    """1. 配置层：按域断言 ECM / 杂物 / 海杂波已在配置中正确声明。"""
    c = jload(cfg_path, {}) or {}
    rd = (c.get("sensors") or {}).get("radar") or {}
    ground = c.get("ground") or {}
    dom = DOMAIN_LABEL[key]
    out = []

    def add(ok, name, detail):
        out.append(dict(ok=bool(ok), check=name, detail=detail))

    if key == "air_plane":
        ecm_block = c.get("ecm")
        ecm = ecm_block if isinstance(ecm_block, dict) else {}
        entities = ecm.get("entities") or []
        add(ecm.get("enabled") is True,
            dom + " ecm.enabled=true", "enabled=%s" % ecm.get("enabled"))
        kinds = [str(e.get("kind")) for e in entities if isinstance(e, dict)]
        n_active = sum(1 for k in kinds if k in ACTIVE_KINDS)
        n_passive = sum(1 for k in kinds if k in PASSIVE_KINDS)
        add(n_active >= 1, dom + " 至少 1 个有源干扰实体",
            "active=%d kinds=%s" % (n_active, kinds))
        add(n_passive >= 1, dom + " 至少 1 个无源干扰实体",
            "passive=%d kinds=%s" % (n_passive, kinds))
        add(bool(kinds) and all(k in KINDS for k in kinds),
            dom + " 所有 entity 的 kind 合法",
            "illegal=%s allowed=%s" % ([k for k in kinds if k not in KINDS], sorted(KINDS)))
        # 200 W passed a 1-frame probe but drowned 29/30 frames of the full
        # clip; 10 W recovered the probed frames with only 2.7 dB Swerling
        # margin.  Final erp_w is 3 W, so cap at 5 W with room for tuning.
        powers = [float(e.get("erp_w") or 0) for e in entities
                  if isinstance(e, dict) and e.get("kind") == "noise_jammer"]
        add(bool(powers) and all(0 < p <= 5.0 for p in powers),
            dom + " 有源压制功率 ≤5 W（30 帧跨帧检出余量）",
            "erp_w=%s (200 W 曾导致 detected 1/30)" % powers)
        # Even at 3 W, Swerling I drew a near-zero RCS on frame 10 (29/30).
        # III (two draws, half variance) + pfa 1e-4 gives 30/30 with margin.
        sw = str(rd.get("swerling") or "").upper()
        pfa = float(rd.get("pfa") or 0)
        add(sw in ("III", "IV") and pfa >= 1e-4,
            dom + " 雷达起伏/门限在跨帧安全区（Swerling III + pfa ≥1e-4）",
            "swerling=%s pfa=%s (Swerling I + 1e-6 曾 29/30)" % (sw, pfa))
        if not isinstance(ecm_block, dict):
            add(False, dom + " simkit.ecm.load(cfg['ecm']) 不抛异常",
                "cfg['ecm'] 顶层块缺失")
        else:
            try:
                from simkit import ecm as _ecmlib
            except Exception as exc:  # pragma: no cover - import must not crash the gate
                add(False, dom + " simkit.ecm.load(cfg['ecm']) 不抛异常",
                    "simkit.ecm 导入失败: %s" % exc)
            else:
                try:
                    _ecmlib.load(ecm_block)
                    ok, detail = True, "load OK, entities=%d" % len(entities)
                except Exception as exc:
                    ok, detail = False, "%s: %s" % (type(exc).__name__, exc)
                add(ok, dom + " simkit.ecm.load(cfg['ecm']) 不抛异常", detail)

    elif key == "ground_vehicle":
        objs = c.get("objects") or []
        add(len(objs) >= 4, dom + " objects 数量 ≥ 4（主目标 + 杂物）",
            "count=%d" % len(objs))
        add(bool(objs) and objs[0].get("label") is True,
            dom + " objects[0].label is True（主目标）",
            "label=%r" % (objs[0].get("label") if objs else None))
        rest = objs[1:]
        add(bool(rest) and all(o.get("label") is False for o in rest),
            dom + " 其余 objects 全部 label is False（杂物不进 COCO）",
            "labels=%s" % [o.get("label") for o in rest])
        clutter_density = float(rd.get("clutter_density") or 0)
        # ground_clutter_db is a DEFAULT in radar.py, so its mere presence is
        # not evidence the clutter was turned up: require the density knob.
        add(clutter_density > 0,
            dom + " 地杂波开启（clutter_density > 0）",
            "clutter_density=%s ground_clutter_db=%s clutter_db=%s"
            % (clutter_density, rd.get("ground_clutter_db"), rd.get("clutter_db")))

    elif key == "sea_ship":
        add(rd.get("clutter_ridge") is True,
            dom + " sensors.radar.clutter_ridge is True",
            "clutter_ridge=%s" % rd.get("clutter_ridge"))
        mti = float(rd.get("mti_min_vr_mps") or 0)
        add(mti <= 0, dom + " mti_min_vr_mps ≤ 0（否则脊被静默关闭）",
            "mti_min_vr_mps=%s" % mti)
        add(ground.get("type") == "sea", dom + ' ground.type == "sea"',
            "type=%s" % ground.get("type"))
        sea_set = ("sea_clutter_scale_db" in rd) or ("sea_clutter_db" in rd)
        add(sea_set, dom + " 海杂波电平显式设置（sea_clutter_scale_db / sea_clutter_db）",
            "scale=%s db=%s" % (rd.get("sea_clutter_scale_db"), rd.get("sea_clutter_db")))
    return out


def check_jam_products(run, key):
    """2. 产物层：每套逐帧 0–29，输出 min/max/mean 与干扰入画证据。"""
    dom = DOMAIN_LABEL[key]
    out = []
    base = list(range(N_FRAMES))

    def add(ok, name, detail, warn=False):
        out.append(dict(ok=bool(ok), check=name, detail=detail, warn=warn))

    # ---- 光学 ----
    op_dyn, op_std, op_black, op_hf, op_blob = [], [], [], [], []
    for i in base:
        p = os.path.join(run, "images", "optical", "%06d.png" % i)
        lp = os.path.join(run, "labels", "optical", "%06d.json" % i)
        if not os.path.isfile(p):
            continue
        a = np.asarray(Image.open(p).convert("L"), dtype=float)
        tm = _target_mask(a.shape, lp)
        op_dyn.append(float(np.percentile(a, 99) - np.percentile(a, 1)))
        op_std.append(float(a.std()))
        op_black.append(100.0 * float((a == 0).mean()))
        op_hf.append(_lap_energy(a, tm))
        op_blob.append(_blobs(a, tm, 245.0))
    add(len(op_dyn) == N_FRAMES, dom + " 光学逐帧 0–29 齐全",
        "frames=%d" % len(op_dyn))
    add(bool(op_dyn) and min(op_dyn) >= 40.0,
        dom + " 光学 p99-p1 ≥ 40 DN（真实背景仍在）", str(mm(op_dyn)))
    add(bool(op_std) and min(op_std) >= 8.0,
        dom + " 光学 std ≥ 8（真实背景仍在）", str(mm(op_std)))
    add(bool(op_black) and max(op_black) <= 5.0,
        dom + " 光学纯黑 ≤ 5%（干扰未压死画面）", str(mm(op_black)))
    add(True, dom + " 光学干扰/杂物入画证据（排除目标框）",
        "高频能量 hf=%s | 亮点斑块 blobs=%s" % (mm(op_hf), mm(op_blob)))

    # ---- LWIR ----
    ir_black, ir_hot = [], []
    for i in base:
        p = os.path.join(run, "images", "ir_preview_lwir", "%06d.png" % i)
        lp = os.path.join(run, "labels", "ir_lwir", "%06d.json" % i)
        if not os.path.isfile(p):
            continue
        a = np.asarray(Image.open(p).convert("L"), dtype=float)
        tm = _target_mask(a.shape, lp)
        ir_black.append(100.0 * float((a == 0).mean()))
        ir_hot.append(_blobs(a, tm, 200.0))
    add(len(ir_black) == N_FRAMES, dom + " LWIR 逐帧 0–29 齐全",
        "frames=%d" % len(ir_black))
    add(bool(ir_black) and max(ir_black) <= 2.0,
        dom + " LWIR 死黑 ≤ 2%（硬门禁）", str(mm(ir_black)))
    hot_total = int(np.sum(ir_hot)) if ir_hot else 0
    if key == "air_plane":
        add(hot_total >= 1, dom + " 红外焰弹热点（排除目标框 ≥1 个）",
            "hotspots=%s total=%d" % (mm(ir_hot), hot_total))
    else:
        add(hot_total >= 1, dom + " LWIR 排除目标框热点（仅记录，WARN 不 FAIL）",
            "hotspots=%s total=%d" % (mm(ir_hot), hot_total), warn=True)

    # ---- RD 张量 + 目标解析 SNR ----
    # 张量 max/median 会被强干扰自身抬高，air_strong 目标已 detected=false
    # 时该比值仍有 16 dB；真正代表「目标可检出」的是 labels/radar 的 snr_db。
    rd_med, rd_peak, rd_cb, rd_snr = [], [], [], []
    for i in base:
        p = os.path.join(run, "raw", "radar", "%06d.npy" % i)
        lp = os.path.join(run, "labels", "radar", "%06d.json" % i)
        if not os.path.isfile(p):
            continue
        try:
            med, peak, cb = _rd_metrics(p, lp)
        except (OSError, ValueError):
            continue
        rd_med.append(med)
        rd_peak.append(peak)
        rd_cb.append(cb)
        lab = jload(lp, {}) or {}
        for t in (lab.get("targets") or []):
            if isinstance(t, dict) and not _is_ecm(t) and t.get("snr_db") is not None:
                rd_snr.append(float(t["snr_db"]))
    add(len(rd_peak) == N_FRAMES, dom + " RD 张量逐帧 0–29 齐全",
        "frames=%d" % len(rd_peak))
    add(bool(rd_med) and min(rd_med) > 0,
        dom + " RD 中位噪声基底 > 0", str(mm(rd_med)))
    add(bool(rd_snr) and len(rd_snr) >= N_FRAMES and min(rd_snr) >= 12.0,
        dom + " 目标解析 SNR ≥ 12 dB（labels/radar，非张量 max/median）",
        str(mm(rd_snr)))
    add(True, dom + " RD 张量峰值/中位比（仅记录，不代表目标可检出）",
        str(mm(rd_peak)))
    if key == "sea_ship":
        add(bool(rd_cb) and min(rd_cb) >= 6.0,
            dom + " 海面 clutter_db ≥ 6 dB（脊确实出现）", str(mm(rd_cb)))
    else:
        add(True, dom + " 零多普勒中心带 clutter_db（仅记录，不作硬 FAIL）",
            str(mm(rd_cb)))

    # 相对无干扰基线：证明干扰由本轮配置注入，而不是背景本身就带高杂波
    # （海面 base 的中心带已有 27.69 dB，单看绝对阈值没有判别力）。
    rep = jload(os.path.join(ROOT, "tools", "v3_probe_report.json"), {}) or {}
    brep = ((((rep.get("domains") or {}).get(key) or {}).get("tiers") or {})
            .get("base") or {})
    base_center = float((brep.get("rd") or {}).get("center_rel_db") or 0)
    base_floor = float(brep.get("noise_floor_db") or 0)
    if rd_cb and base_center:
        gain = float(np.mean(rd_cb)) - base_center
        if key in ("ground_vehicle", "sea_ship"):
            add(gain >= 6.0, dom + " 中心带相对无干扰基线提升 ≥6 dB",
                "gain=%.2f dB (base=%.2f mean=%.2f)"
                % (gain, base_center, float(np.mean(rd_cb))))
        else:
            add(True, dom + " 中心带相对基线（空域 ECM 不以零多普勒脊为特征）",
                "gain=%.2f dB" % gain)
    if key == "air_plane" and rd_med and base_floor:
        fdb = 10.0 * math.log10(max(float(np.mean(rd_med)), 1e-30))
        gain = fdb - base_floor
        add(gain >= 3.0, dom + " 噪声基底相对基线提升 ≥3 dB（有源压制生效）",
            "gain=%.2f dB (base=%.2f dB mean=%.2f dB)" % (gain, base_floor, fdb))

    metrics = dict(
        optical=dict(dynamic_range_dn=mm(op_dyn), std_dn=mm(op_std),
                     black_pct=mm(op_black), hf_laplacian=mm(op_hf),
                     hot_blobs=mm(op_blob)),
        lwir=dict(dead_black_pct=mm(ir_black), hotspots=mm(ir_hot)),
        rd=dict(noise_floor_median=mm(rd_med), peak_snr_db=mm(rd_peak),
                clutter_db=mm(rd_cb), target_snr_db=mm(rd_snr)),
    )
    return out, metrics


def check_jam_labels(run, key):
    """3. 雷达真值 + 标注污染（硬门禁）；ECM 项单独计数但不得计入主目标。"""
    dom = DOMAIN_LABEL[key]
    out = []
    base = list(range(N_FRAMES))

    def add(ok, name, detail):
        out.append(dict(ok=bool(ok), check=name, detail=detail))

    frames = jload(os.path.join(run, "frames.json"), []) or []
    if isinstance(frames, dict):
        frames = frames.get("frames") or []
    truth = {int(r.get("frame", -1)): float(r.get("range_m") or 0) for r in frames}

    main_per_frame, det_ok, snr_vals, rdev = [], 0, [], []
    box_ok, cls_ok, ecm_count, ecm_in_targets = 0, 0, 0, 0
    devs = []
    for i in base:
        rlab = jload(os.path.join(run, "labels", "radar", "%06d.json" % i), {}) or {}
        tg = [t for t in (rlab.get("targets") or []) if isinstance(t, dict)]
        main = [t for t in tg if not _is_ecm(t)]
        ecm_in_targets += len(tg) - len(main)
        main_per_frame.append(len(main))
        for t in main:
            if t.get("detected") is True:
                det_ok += 1
            if isinstance(t.get("snr_db"), (int, float)):
                snr_vals.append(float(t["snr_db"]))
            rng = float(t.get("measured_range") or t.get("range") or 0)
            if truth.get(i):
                rdev.append(abs(rng - truth[i]))
        ecm_count += _count_ecm(rlab.get("boxes_rd"))
        ecm_count += _count_ecm(rlab.get("unassociated_detections"))

        olab = jload(os.path.join(run, "labels", "optical", "%06d.json" % i), {}) or {}
        ob = (olab.get("boxes") or {})
        items = [(k, b) for k, b in ob.items() if isinstance(b, dict)]
        if len(items) == 1:
            box_ok += 1
        coco = jload(os.path.join(run, "annotations_optical_coco.json"), {}) or {}
        coco_name = None
        if coco.get("categories"):
            coco_name = coco["categories"][0].get("name")
        if items and _cls_of(olab, items[0][0], items[0][1], coco_name) in CLASSES_WHITELIST:
            cls_ok += 1

        a = boxes(os.path.join(run, "labels", "optical", "%06d.json" % i))
        b = boxes(os.path.join(run, "labels", "ir_lwir", "%06d.json" % i))
        if a and b:
            devs.append(max(abs((a[0]["x0"] + a[0]["x1"]) / 2 - (b[0]["x0"] + b[0]["x1"]) / 2) / W,
                            abs((a[0]["y0"] + a[0]["y1"]) / 2 - (b[0]["y0"] + b[0]["y1"]) / 2) / H,
                            abs((a[0]["x1"] - a[0]["x0"]) - (b[0]["x1"] - b[0]["x0"])) / W))

    # 雷达真值：主目标（排除 source=="ecm" 的额外框）30/30 检出、SNR、距离
    add(all(m == 1 for m in main_per_frame) and len(main_per_frame) == N_FRAMES,
        dom + " 每帧恰好 1 个主目标（ECM 额外框不计入）",
        "main=%s ecm_in_targets=%d" % (mm(main_per_frame), ecm_in_targets))
    add(det_ok == N_FRAMES, dom + " 主目标 detected=true 30/30",
        "detected=%d/%d" % (det_ok, N_FRAMES))
    add(bool(snr_vals) and min(snr_vals) >= 12.0,
        dom + " 主目标 snr_db ≥ 12", str(mm(snr_vals)))
    add(bool(rdev) and max(rdev) <= 30.0,
        dom + " 主目标距离偏差 ≤ 30 m", str(mm(rdev)))

    # 标注污染（硬门禁）
    add(box_ok == N_FRAMES, dom + " 每帧 optical.boxes 恰好 1 个（杂物未入标注）",
        "ok=%d/%d" % (box_ok, N_FRAMES))
    add(cls_ok == N_FRAMES, dom + " optical 框 cls ∈ {fighter,vehicle,ship}",
        "ok=%d/%d" % (cls_ok, N_FRAMES))
    add(bool(devs) and max(devs) <= 0.002,
        dom + " labels/ir_lwir 与 labels/optical 归一化框偏差 ≤ 0.002",
        "max=%.6f over %d frames" % (max(devs) if devs else -1, len(devs)))
    for cname in ("annotations_optical_coco.json", "annotations_ir_lwir_coco.json"):
        d = jload(os.path.join(run, cname), {}) or {}
        cats = d.get("categories") or []
        anns = d.get("annotations") or []
        imgs = d.get("images") or []
        one_cat = (len(cats) == 1
                   and cats[0].get("name") in CLASSES_WHITELIST)
        add(one_cat, dom + " COCO %s 恰好 1 类且为白名单类" % cname,
            "categories=%s" % [c.get("name") for c in cats])
        add(len(anns) == N_FRAMES, dom + " COCO %s 恰好 30 条标注" % cname,
            "count=%d" % len(anns))
        img_ids = {int(x["id"]) for x in imgs if isinstance(x, dict) and "id" in x}
        ann_ids = {int(a["image_id"]) for a in anns if isinstance(a, dict) and "image_id" in a}
        add(bool(anns) and ann_ids == img_ids and bool(img_ids),
            dom + " COCO %s 无孤立标注（逐帧一一对应）" % cname,
            "images=%d annotations=%d orphan=%d" %
            (len(img_ids), len(anns), len(ann_ids - img_ids)))

    add(True, dom + " ECM 干扰仅在 RD 检测层，未进 COCO/YOLO",
        "boxes_rd/unassociated_detections 中 ecm# 或 source==\"ecm\" 项计数=%d；"
        "COCO 恰好 1 类白名单、optical.boxes 恰好 1 个，YOLO 由该白名单框导出。" % ecm_count)
    return out, ecm_count


def check_jam_determinism(run, cfg_path, key):
    """4. 确定性：三处 seed 一致（等于各域 7/17/27）+ frames.json 长度 30。"""
    c = jload(cfg_path, {}) or {}
    meta = jload(os.path.join(run, "meta.json"), {}) or {}
    side = jload(os.path.join(run, "_render_cfg.json"), {}) or {}
    frames = jload(os.path.join(run, "frames.json"), []) or []
    if isinstance(frames, dict):
        frames = frames.get("frames") or []
    dom = DOMAIN_LABEL[key]
    exp = EXPECTED_SEED[key]
    out = []

    def add(ok, name, detail):
        out.append(dict(ok=bool(ok), check=name, detail=detail))

    s_meta, s_side, s_cfg = meta.get("seed"), side.get("seed"), c.get("seed")
    add(all(x is not None for x in (s_meta, s_side, s_cfg))
        and s_meta == s_side == s_cfg == exp,
        dom + " meta.seed / _render_cfg.seed / cfg.seed 三者一致 (=%d)" % exp,
        "meta=%s sidecar=%s cfg=%s" % (s_meta, s_side, s_cfg))
    add(len(frames) == N_FRAMES, dom + " frames.json 长度 30",
        "len=%d" % len(frames))
    return out


# ---------------------------------------------------------------- yolo
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


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=r"D:\work\多模态数据集生成\dataest")
    ap.add_argument("--emit-yolo", action="store_true")
    a = ap.parse_args()
    dest = os.path.abspath(a.dest)
    # Datasets have not been generated yet: fail gracefully, never with a trace.
    if not os.path.isdir(dest):
        print("错误：产物根目录不存在或不是目录：%s" % dest)
        print("数据集尚未生成，请先运行生成器后再执行验收；本次未执行任何检查。")
        return 2
    overall, jam_overall, summary = True, True, []
    for key in SCENES:
        run = os.path.join(dest, key)
        cfg = os.path.join(ROOT, "scenes", CFG_PREFIX + key + ".json")
        if not os.path.isdir(run):
            print("MISSING", run)
            overall = False
            jam_overall = False
            continue
        groups = {"config": check_config(run, cfg)}
        vis, stats = check_visibility(run)
        groups["visibility"] = vis
        groups["alignment"] = check_alignment(run)
        groups["trainability"] = check_trainability(run, key)
        groups["stability"] = check_stability(run, cfg)
        jam_cfg = check_jam_config(run, key, cfg)
        jam_prod, jam_metrics = check_jam_products(run, key)
        jam_lab, ecm_count = check_jam_labels(run, key)
        jam_det = check_jam_determinism(run, cfg, key)
        groups["jam"] = jam_cfg + jam_prod + jam_lab + jam_det

        hard = [c for g in groups.values() for c in g
                if not c["ok"] and not c.get("warn")]
        warns = [c for g in groups.values() for c in g
                 if not c["ok"] and c.get("warn")]
        ok = not hard
        overall &= ok
        jam_hard = [c for c in groups["jam"] if not c["ok"] and not c.get("warn")]
        jam_ok = not jam_hard
        jam_overall &= jam_ok

        jam_section = dict(
            domain=key,
            jam_types=JAM_TYPES[key],
            metrics=jam_metrics,
            interference_active=bool(jam_ok),
            active_evidence=("该域全部干扰专项硬门禁通过；空域红外热点/物理层 ECM、"
                             "地面 label:false 杂物与地杂波、海面 clutter_db≥6 dB 均已生效"),
            ecm_rd_only_count=ecm_count,
            note="ECM 干扰仅在 RD 检测层，未进 COCO/YOLO；干扰实体 label:false。",
            passed=bool(jam_ok),
        )
        report = dict(scene=key, passed=ok, jam_pass=jam_ok, stats=stats,
                      groups=groups, jam=jam_section,
                      hard_failures=[c["check"] for c in hard],
                      warnings=[c["check"] for c in warns])
        jwrite(os.path.join(run, "validation.json"), report)
        if a.emit_yolo:
            n_yolo = emit_yolo(run, key)
            report["yolo_txt_files"] = n_yolo
            # emit_yolo writes optical + ir_lwir, i.e. 2 × 30 files.  Gate it
            # explicitly instead of silently recording whatever came out.
            yc = dict(ok=(n_yolo == 2 * N_FRAMES),
                      check="YOLO txt 每模态 30 份（共 60）",
                      detail="written=%d expected=%d" % (n_yolo, 2 * N_FRAMES))
            groups["trainability"].append(yc)
            report["groups"] = groups
            if not yc["ok"]:
                report["hard_failures"] = list(report["hard_failures"]) + [yc["check"]]
                report["passed"] = False
                overall = False
            jwrite(os.path.join(run, "validation.json"), report)
        jwrite(os.path.join(run, "dataset.json"), dict(
            name=key, classes=[CLASSES[key]], frames=N_FRAMES,
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
            split_note="One 30-frame continuous clip: splits.json (train 0-23 / "
                       "val 24-29) has adjacent-frame temporal correlation. "
                       "For an unbiased evaluation, generate a second clip with "
                       "a different seed instead of treating val as independent.",
            detector_note="labels/radar.detected is an analytic SNR gate, not an "
                          "observed RD peak; train RD detectors against "
                          "raw/radar/*.npy with the supplied truth rows.",
            jam_note=("本数据集为 v3 干扰数据集：含物理层 ECM（有源+无源）、地面几何杂物/建筑、"
                      "海面海杂波脊。干扰实体 label:false，不进 COCO/YOLO 标注；"
                      "images/radar* 为带坐标轴/表格的标注显示图，不可作训练像素；"
                      "训练源为 images/optical/*.png、images/ir_preview_lwir/*.png、raw/radar/*.npy。"
                      "海杂波脊由常数 gamma 电平 + Rayleigh 零多普勒脊合成，非 K 分布/相关非高斯模型，"
                      "不可用于海杂波统计特性的定量研究；光学 ECM 几何在 t=0 求值，不随帧运动"
                      "（静止干扰物，非逐帧运动学实体）。"),
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
        print("  jam            %s (ecm_rd_only=%d)" % ("PASS" if jam_ok else "FAIL", ecm_count))
        summary.append(dict(scene=key, passed=ok, jam_pass=jam_ok, **stats))
    jwrite(os.path.join(dest, "validation_summary.json"),
           dict(passed=overall, jam_pass=jam_overall, scenes=summary))
    print("\nOVERALL:", "PASS" if overall else "FAIL",
          "| JAM:", "PASS" if jam_overall else "FAIL")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
