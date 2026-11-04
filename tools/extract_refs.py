#!/usr/bin/env python3
r"""Extract render-driving parameters from the real-photo reference gallery.

For every accepted photo in assets/refs/catalog.json this script measures

  * target region          - no deep model: the frame border plus the sky is the
                             background colour model, and the object is the box
                             around the largest non-background component minus
                             sky; a hand-drawn box in REVIEW replaces it
  * target colour          - median sRGB, median linear RGB, and an illumination-
                             free linear albedo estimate
  * texture                - high-frequency power fraction, Laplacian variance
  * sky / background       - mean, std, radial power-spectrum slope
  * camera noise           - robust local high-pass sigma inside the sky patch

and aggregates per render category into configs/refs_params.json, together with
a comparison against simkit/materials.py (read-only: the library is imported,
never edited) and a suggested Base Color per category.

Albedo model
------------
The photos have unknown exposure, so absolute radiance is unavailable. What is
available is the *ratio* between the target and the sky in the same frame, which
is exposure invariant:

    albedo_linear = median_linear(target) / (k * median_linear(sky))

k folds the direct/diffuse illumination ratio into one number. k = 4.0 is the
clear-sky, sunlit case in which the median visible face receives about 60 % of
the horizontal global irradiance; k = 2.0 for hazy/high-cloud and k = 1.2 for
flat overcast (a uniformly bright dome makes sky radiance itself the
illuminance). The resulting value is an *effective* albedo - the Base Color a
Lambertian surface needs in the same lighting to reproduce the photo - which is
exactly the quantity Blender's Principled BSDF Base Color is compared against.
Uncertainty is roughly +-35 %, dominated by k and by the unknown tone curve.

Run with:
    .\env\Scripts\python.exe tools\extract_refs.py            # measure + aggregate
    .\env\Scripts\python.exe tools\extract_refs.py --sheet    # + review contact sheets
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import sys
from collections import deque

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFS = os.path.join(ROOT, "assets", "refs")
CATALOG = os.path.join(REFS, "catalog.json")
OUT_JSON = os.path.join(ROOT, "configs", "refs_params.json")
REVIEW_DIR = os.path.join(ROOT, "out", "refs_review")

CATEGORIES = ["军用飞机", "民航飞机", "直升机", "固定翼无人机", "旋翼无人机",
              "军舰", "民用船舶", "地面车辆", "军用地面车辆", "杂项"]

# Category -> the material the current scenes actually put on these objects.
# scene.py falls back to painted_metal for every object and none of the shipped
# scenes override it, so this is painted_metal for nine of the ten categories.
CATEGORY_MATERIAL = {
    "军用飞机": ("painted_metal", ["metal_bare"], "整机多为哑光灰/深灰涂装，少量裸金属蒙皮"),
    "民航飞机": ("painted_metal", ["plastic"], "白色/浅灰聚氨酯蒙皮 + 深色舷窗"),
    "直升机": ("painted_metal", ["metal_bare"], "军绿/灰涂装，旋翼为深色复合材料"),
    "固定翼无人机": ("painted_metal", ["metal_bare"], "浅灰/白色蒙皮，复合材料"),
    "旋翼无人机": ("painted_metal", ["plastic"], "白色/浅灰工程塑料外壳"),
    "军舰": ("painted_metal", ["metal_bare"], "海军灰（haze grey）低反照率涂装"),
    "民用船舶": ("painted_metal", ["plastic"], "白色上层建筑 + 深色船体/甲板"),
    "地面车辆": ("painted_metal", ["glass", "rubber"], "车漆颜色分散，含玻璃与轮胎"),
    "军用地面车辆": ("painted_metal", ["metal_bare"], "橄榄绿/沙色低反照率涂装"),
    "杂项": ("painted_metal", ["concrete", "metal_bare", "foliage"], "混凝土隔离墩/金属围栏/植被/机器人混合"),
}

# Human review. Keyed by catalog file name. accepted=False removes the photo
# from every statistic (the file stays on disk for audit); a box is a normalised
# (x0, y0, x1, y1) crop that replaces the automatic mask for that photo.
REVIEW = {
    # rejected after looking at the contact sheets in out/refs_review
    "01_Oregon_ANG_pilot_preparing_F-15_flight_at_Nellis_-_Weapons_Instr.jpg":
        dict(accepted=False, reason="驾驶舱剪影，目标不可分割"),
    "01_An_OH-58D_Kiowa_helicopter_departs_from_a_communications_site_in.jpg":
        dict(accepted=False, reason="沙尘/逆光，目标只占几像素"),
    "02_Bell_OH-58A_Kiowa_3-view_line_drawing.png":
        dict(accepted=False, reason="三视图线稿，不是照片"),
    "04_Bell_OH-58C_Kiowa_Helicopter_Ashburn.JPG":
        dict(accepted=False, reason="地面静态展示，人群占画面"),
    "05_UH-60_Black_Hawk_Flight_8298582_.jpg":
        dict(accepted=False, reason="座舱内视角"),
    "06_UH-60_Black_Hawk_Flight_8298572_.jpg":
        dict(accepted=False, reason="座舱内视角"),
    "01_Gear_Actuation-Boeing-737-800_EL-AL_approaching_VIE-DSC_3259w.jpg":
        dict(accepted=False, reason="起落架特写，不是整机"),
    "06_D-AIZR_Airbus_A320_of_Eurowings_at_Berlin-Tegel_Airport_2017_IMG.jpg":
        dict(accepted=False, reason="机翼局部特写"),
    "04_MQ-9_Reaper_on_the_361_Expeditionary_Attack_Squadron_flight_line.jpg":
        dict(accepted=False, reason="停机坪地面照"),
    "05_147th_Attack_Wing_Maintainers_Conduct_Post-Flight_Checks_on_MQ-9.jpg":
        dict(accepted=False, reason="地勤人员占画面"),
    "08_230430-F-WE773-438_-_AFSOC_Total_Force_landed_MC-130J_MQ-9_A-10s.jpg":
        dict(accepted=False, reason="落地滑行，目标被遮挡"),
    "09_New_York_and_Indiana_National_Guard_26932620565_.jpg":
        dict(accepted=False, reason="地面静态/构图混乱"),
    "07_DJI_Phantom_Unsplash_.jpg":
        dict(accepted=False, reason="极暗逆光，无有效天空参考"),
    "09_White_drone_mid-flight_Unsplash_.jpg":
        dict(accepted=False, reason="深色背景，目标区被天空模型污染"),
    "01_USS_Arleigh_Burke_is_underway._11177376336_.jpg":
        dict(accepted=False, reason="浪花遮挡舰体"),
    "04_USS_Arleigh_Burke_prepares_to_get_underway._13697732964_.jpg":
        dict(accepted=False, reason="系泊作业，人物占画面"),
    "05_US_Navy_111003-N-FI736-034_Logistics_Specialist_3rd_Class_Hyeon.jpg":
        dict(accepted=False, reason="甲板人物特写"),
    "06_US_Navy_060407-N-1045B-008_Sailors_aboard_the_Ticonderoga-class.jpg":
        dict(accepted=False, reason="甲板人物特写"),
    "02_The_Sea_Land_Container_ship_SS_Raphael_Semmes_arrives_at_Newport.jpg":
        dict(accepted=False, reason="前景小船与人物占画面"),
    "08_Sounds_of_Sea_Metal_tube_in_front_of_container_ship_at_Viaduct_H.jpg":
        dict(accepted=False, reason="前景金属管遮挡，逆光"),
    "06_Fuel_gauge_Toyota_Corolla_.jpg":
        dict(accepted=False, reason="仪表盘细节，不是整车"),
    "01_Completed_M1_Abrams_Tank_Chrysler_Lima_1979.jpg":
        dict(accepted=False, reason="厂房内人工照明"),
    "02_M1_Abrams_noBG.jpg":
        dict(accepted=False, reason="去背景剪贴图，无真实光照"),
    "04_M1_Abrams_tank_operators_map.png":
        dict(accepted=False, reason="操作手册地图，不是照片"),
    "08_M2-A2_Bradley_Postcard_52629519704_.jpg":
        dict(accepted=False, reason="明信片翻拍，带边框"),
    "09_M2_Bradley_interior.jpg":
        dict(accepted=False, reason="车内视角"),
    "02_Artistic_Jersey_Barrier_2024_02.jpg":
        dict(accepted=False, reason="彩绘艺术隔离墩，非典型混凝土"),
    "03_Artistic_Jersey_Barrier_05.jpg":
        dict(accepted=False, reason="彩绘艺术隔离墩，非典型混凝土"),
    "04_Artistic_Jersey_Barrier_06.jpg":
        dict(accepted=False, reason="彩绘艺术隔离墩，非典型混凝土"),
    "06_Chain_Link_Fence.jpg":
        dict(accepted=False, reason="极暗剪影，无有效天空参考"),
    "08_20200408_newspaper_chain-link_fence.jpg":
        dict(accepted=False, reason="报纸背景，非典型场景"),
    "09_Chain_link_fence_Unsplash_.jpg":
        dict(accepted=False, reason="深色逆光，目标区被天空模型污染"),
}

# Tag -> material, used by 杂项 where one category mixes several materials.
MISC_TAGS = [
    ("barrier", "concrete", "隔离墩/护栏"),
    ("fence", "metal_bare", "金属围栏"),
    ("shrub", "foliage", "灌木植被"),
    ("rover", "painted_metal", "巡视器/机器人"),
    ("ingenuity", "painted_metal", "小型旋翼机"),
    ("ship", "painted_metal", "船体"),
    ("container", "painted_metal", "集装箱船"),
]

TOL_SEED = 58.0        # colour distance to the sky seed colour
TOL_LOCAL = 11.0       # colour step between neighbouring sky pixels
FACTOR = 4             # segmentation runs on a 1/4 resolution copy


def review_for(name):
    """Prefix match so the table does not depend on a duplicated extension."""
    for key, val in REVIEW.items():
        k = key
        for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
            while k.lower().endswith(ext.lower()):
                k = k[:-len(ext)]
        if name.startswith(k) or k in name:
            return val
    return {}


def log(msg):
    print(str(msg), flush=True)


def now_iso():
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return default


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=1)
        fh.write("\n")
    os.replace(tmp, path)


# --------------------------------------------------------------------------
# small numeric helpers (no scipy in this environment)
# --------------------------------------------------------------------------

def srgb_to_linear(x):
    x = np.asarray(x, np.float64)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    x = np.clip(np.asarray(x, np.float64), 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


def luma_srgb(rgb):
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def gauss_blur(a, sigma):
    """Separable Gaussian, edge padded (scipy is not installed here)."""
    a = np.asarray(a, np.float64)
    if sigma <= 0.3:
        return a.copy()
    r = max(1, int(round(sigma * 2.6)))
    x = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-(x ** 2) / (2.0 * sigma * sigma))
    k /= k.sum()
    p = np.pad(a, ((r, r), (0, 0)), mode="edge")
    b = np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), 0, p)
    p = np.pad(b, ((0, 0), (r, r)), mode="edge")
    c = np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), 1, p)
    return c


def otsu(values, nbins=64):
    v = np.asarray(values, np.float64).ravel()
    lo, hi = float(np.min(v)), float(np.max(v))
    if hi - lo < 1e-9:
        return hi
    hist, edges = np.histogram(v, bins=nbins, range=(lo, hi))
    centres = 0.5 * (edges[:-1] + edges[1:])
    w0 = np.cumsum(hist)
    w1 = hist.sum() - w0
    m0 = np.cumsum(hist * centres)
    total = m0[-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        mu0 = m0 / np.maximum(w0, 1)
        mu1 = (total - m0) / np.maximum(w1, 1)
        between = w0 * w1 * (mu0 - mu1) ** 2
    between[~np.isfinite(between)] = -1
    return float(centres[int(np.argmax(between))])


def robust_sigma(values):
    """1.4826 * MAD, the usual outlier-resistant sigma."""
    v = np.asarray(values, np.float64).ravel()
    if v.size < 16:
        return 0.0
    med = np.median(v)
    mad = np.median(np.abs(v - med))
    return float(1.4826 * mad)


def highpass_sigma(values):
    """Noise sigma of a zero-mean high-pass residual.

    MAD first; a flat 8-bit sky can make more than half the residuals exactly
    equal, which collapses MAD to zero, so fall back to the mean absolute value
    of the well-behaved 95 % of the residual (E|n| = 0.7979 sigma for Gaussian).
    """
    v = np.asarray(values, np.float64).ravel()
    if v.size < 16:
        return 0.0
    sig = robust_sigma(v)
    if sig > 0.05:
        return sig
    a = np.abs(v - np.median(v))
    keep = a <= np.percentile(a, 95)
    if not keep.any():
        return 0.0
    return float(np.mean(a[keep]) / 0.7979)


def resize_to(arr, size, resample):
    img = Image.fromarray(arr)
    return np.asarray(img.resize(size, resample))


def resize_rgb(rgb, size):
    """Resize a float RGB image through 8 bit (PIL has no float RGB resize)."""
    u8 = np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return np.asarray(Image.fromarray(u8).resize(size, Image.BOX), np.float32) / 255.0


def erode_mask(mask, radius):
    if radius <= 0:
        return mask.copy()
    img = Image.fromarray((mask.astype(np.uint8) * 255))
    out = img.filter(ImageFilter.MinFilter(int(radius) * 2 + 1))
    return np.asarray(out) > 127


def dilate_mask(mask, radius):
    if radius <= 0:
        return mask.copy()
    img = Image.fromarray((mask.astype(np.uint8) * 255))
    out = img.filter(ImageFilter.MaxFilter(int(radius) * 2 + 1))
    return np.asarray(out) > 127


def label_components(mask):
    """4-connected components on a small boolean mask; returns list of dicts."""
    h, w = mask.shape
    labels = np.zeros((h, w), np.int32)
    comps = []
    cur = 0
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if labels[y0, x0]:
            continue
        cur += 1
        labels[y0, x0] = cur
        q = deque([(y0, x0)])
        area = 0
        miny = maxy = y0
        minx = maxx = x0
        pixels = []
        while q:
            y, x = q.popleft()
            area += 1
            pixels.append((y, x))
            if y < miny:
                miny = y
            if y > maxy:
                maxy = y
            if x < minx:
                minx = x
            if x > maxx:
                maxx = x
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not labels[ny, nx]:
                    labels[ny, nx] = cur
                    q.append((ny, nx))
        comps.append(dict(label=cur, area=area, bbox=(minx, miny, maxx, maxy),
                          pixels=pixels))
    return comps


def fill_holes(mask):
    """Fill interior holes: anything not reachable from the border."""
    h, w = mask.shape
    outside = np.zeros((h, w), bool)
    q = deque()
    for x in range(w):
        for y in (0, h - 1):
            if not mask[y, x] and not outside[y, x]:
                outside[y, x] = True
                q.append((y, x))
    for y in range(h):
        for x in (0, w - 1):
            if not mask[y, x] and not outside[y, x]:
                outside[y, x] = True
                q.append((y, x))
    while q:
        y, x = q.popleft()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < h and 0 <= nx < w and not mask[ny, nx] and not outside[ny, nx]:
                outside[ny, nx] = True
                q.append((ny, nx))
    return ~outside


# --------------------------------------------------------------------------
# sky and target segmentation
# --------------------------------------------------------------------------

def detect_sky(small_rgb):
    """Flood the sky from the top border of the (downsampled) frame.

    Input is 0..255 float RGB; the tolerances below are in 8-bit DN.

    Region growing uses the *local* colour step, so a zenith-to-horizon
    gradient is followed without the tolerance being blown by the total
    spread; a chromaticity guard keeps a blue-grey sea from being swallowed.
    """
    h, w = small_rgb.shape[:2]
    top = small_rgb[:2].reshape(-1, 3)
    med = np.median(top, axis=0)
    mad = np.median(np.abs(top - med), axis=0).max()
    tol = max(14.0, 3.0 * float(mad))
    sky = np.zeros((h, w), bool)
    q = deque()
    for y in range(min(2, h)):
        for x in range(w):
            d = np.abs(small_rgb[y, x] - med).max()
            if d <= tol:
                sky[y, x] = True
                q.append((y, x))
    if not q:
        return np.zeros((h, w), bool), 0.0
    max_row = int(h * 0.78)
    while q:
        y, x = q.popleft()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if ny < 0 or ny >= max_row or nx < 0 or nx >= w or sky[ny, nx]:
                continue
            c = small_rgb[ny, nx]
            d_seed = float(np.abs(c - med).max())
            d_local = float(np.abs(c - small_rgb[y, x]).max())
            sat = float(c.max() - c.min()) / max(float(c.max()), 1.0)
            # sky is blue-dominant or neutral; foliage is green-dominant and must
            # stop the flood, otherwise a tree canopy becomes the "sky" sample
            chromatic_ok = ((c[2] >= c[0] - 18.0) and (c[2] >= c[1] - 14.0)
                            and sat < 0.55)
            if (d_seed <= TOL_SEED and d_local <= TOL_LOCAL + 6.0) or \
               (d_local <= TOL_LOCAL and chromatic_ok and d_seed <= TOL_SEED * 1.6):
                sky[ny, nx] = True
                q.append((ny, nx))
    return sky, float(sky.mean())


def kmeans_colors(pix, k=4, iters=12):
    """Tiny k-means over background colours (no scipy/sklearn here)."""
    pix = np.asarray(pix, np.float64)
    order = np.argsort(pix.sum(axis=1))
    ps = pix[order]
    centers = ps[np.linspace(0, len(ps) - 1, k).astype(int)].copy()
    for _ in range(iters):
        d = ((pix[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        lab = d.argmin(1)
        for j in range(k):
            sel = lab == j
            if sel.any():
                centers[j] = pix[sel].mean(0)
    return centers


def morph(mask, r, grow):
    img = Image.fromarray((mask.astype(np.uint8) * 255))
    flt = ImageFilter.MaxFilter(2 * r + 1) if grow else ImageFilter.MinFilter(2 * r + 1)
    return np.asarray(img.filter(flt)) > 127


def gradient_mag(small_rgb):
    g = luma_srgb(small_rgb)
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[:, 1:-1] = g[:, 2:] - g[:, :-2]
    gy[1:-1, :] = g[2:, :] - g[:-2, :]
    return np.hypot(gx, gy)


def refine_target(small_rgb, mask, pad=0.65):
    """Expand a fragment to the whole object with a local background model.

    The frame-border model often locks onto the brightest panel of a low
    contrast airframe. Inside a window around that fragment the background is
    local and homogeneous, so re-running the same test recovers the rest of the
    object without leaking into the sky or the ground.
    """
    h, w = mask.shape[:2]
    ys, xs = np.nonzero(mask)
    if ys.size < 32:
        return mask, False
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    bh, bw = y1 - y0, x1 - x0
    ey0 = max(0, int(y0 - bh * pad))
    ey1 = min(h, int(y1 + bh * pad))
    ex0 = max(0, int(x0 - bw * pad))
    ex1 = min(w, int(x1 + bw * pad))
    if ey1 - ey0 < 24 or ex1 - ex0 < 24:
        return mask, False
    roi = small_rgb[ey0:ey1, ex0:ex1]
    rh, rw = roi.shape[:2]
    band = max(2, int(round(min(rh, rw) * 0.10)))
    ring = np.ones((rh, rw), bool)
    ring[band:rh - band, band:rw - band] = False
    pix = roi[ring].reshape(-1, 3)
    if len(pix) < 64:
        return mask, False
    if len(pix) > 3000:
        pix = pix[::max(1, len(pix) // 3000)]
    centers = kmeans_colors(pix, k=4)
    dist = np.min(np.linalg.norm(roi[:, :, None, :] - centers[None, None, :, :], axis=3),
                  axis=2)
    bgd = np.min(np.linalg.norm(pix[:, None, :] - centers[None, :, :], axis=2), axis=1)
    thr = max(30.0, float(np.percentile(bgd, 92)) + 4.0)
    binary = morph(morph(dist > thr, 1, False), 1, True)
    comps = label_components(binary)
    core = mask[ey0:ey1, ex0:ex1]
    best = None
    for comp in comps:
        m = np.zeros((rh, rw), bool)
        ys2 = np.fromiter((p[0] for p in comp["pixels"]), np.int32, len(comp["pixels"]))
        xs2 = np.fromiter((p[1] for p in comp["pixels"]), np.int32, len(comp["pixels"]))
        m[ys2, xs2] = True
        overlap = float((m & core).sum()) / max(1.0, float(core.sum()))
        if overlap < 0.35:
            continue
        m = fill_holes(m)
        area = int(m.sum())
        if area < 0.5 * core.sum() or area > 6.0 * core.sum():
            continue
        if best is None or area > best[0]:
            best = (area, m)
    if best is None:
        return mask, False
    out = np.zeros((h, w), bool)
    out[ey0:ey1, ex0:ex1] = best[1]
    if out.sum() < 64:
        return mask, False
    return out, True


def _mask_stats(mask, small_rgb, sky_median, grad, g_mean):
    """Geometry/colour score for a candidate target mask (0..1-ish)."""
    h, w = mask.shape[:2]
    area = float(mask.sum()) / float(h * w)
    if not (0.002 <= area <= 0.55):
        return None
    ys, xs = np.nonzero(mask)
    if ys.size < 64:
        return None
    boundary = mask & ~morph(mask, 1, False)
    edge = float(grad[boundary].mean() / g_mean) if boundary.any() else 0.0
    compact = float(mask.sum()) / max(1.0, float((ys.max() - ys.min() + 1) *
                                                 (xs.max() - xs.min() + 1)))
    sky_pen = 0.0
    if sky_median is not None:
        d = np.abs(small_rgb[mask] - sky_median).max(axis=1)
        sky_pen = float(np.mean(d < 28.0))
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    # a full-width band is the horizon/sea/ground, never a single object
    if (xs.min() <= 1 and xs.max() >= w - 2 and compact > 0.55
            and (xs.max() - xs.min()) > 0.7 * w):
        return None
    score = (area ** 0.22) * (0.35 + 0.65 * min(edge, 3.0)) *             (1.0 - 0.9 * sky_pen) * (0.4 + 0.6 * compact)
    return dict(score=float(score), area=area, edge=float(edge),
                compact=float(compact), sky_pen=float(sky_pen), bbox=bbox)


def _close_mask(m, close_r):
    m = fill_holes(m)
    m = morph(m, close_r, True)
    m = fill_holes(m)
    m = morph(m, max(1, close_r - 1), False)
    return m


def _border_bg_candidates(small_rgb, sky_small):
    """Foreground = colour far from the frame-border / sky colour clusters."""
    h, w = small_rgb.shape[:2]
    band = max(3, int(round(min(h, w) * 0.055)))
    border = np.zeros((h, w), bool)
    border[:band, :] = True
    border[-band:, :] = True
    border[:, :band] = True
    border[:, -band:] = True
    if sky_small is not None and sky_small.any():
        border |= sky_small
    pix = small_rgb[border].reshape(-1, 3)
    if len(pix) < 64:
        return [], None
    if len(pix) > 4000:
        pix = pix[::max(1, len(pix) // 4000)]
    centers = kmeans_colors(pix, k=8)
    dist = np.min(np.linalg.norm(small_rgb[:, :, None, :] - centers[None, None, :, :],
                                axis=3), axis=2)
    bgd = np.min(np.linalg.norm(pix[:, None, :] - centers[None, :, :], axis=2), axis=1)
    thr = max(28.0, float(np.percentile(bgd, 92)) + 4.0)
    binary = morph(morph(dist > thr, 1, False), 1, True)
    close_r = max(1, int(round(min(h, w) * 0.012)))
    out = []
    for comp in label_components(binary):
        x0, y0, x1, y1 = comp["bbox"]
        area_frac = comp["area"] / float(h * w)
        if not (0.002 <= area_frac <= 0.45):
            continue
        if (x1 - x0 + 1) > 0.97 * w or (y1 - y0 + 1) > 0.95 * h:
            continue
        n_contact = int(x0 <= 0) + int(y0 <= 0) + int(x1 >= w - 1) + int(y1 >= h - 1)
        if n_contact >= 3:
            continue
        m = np.zeros((h, w), bool)
        ys = np.fromiter((p[0] for p in comp["pixels"]), np.int32, len(comp["pixels"]))
        xs = np.fromiter((p[1] for p in comp["pixels"]), np.int32, len(comp["pixels"]))
        m[ys, xs] = True
        m = _close_mask(m, close_r)
        if int(m.sum()) >= 64:
            out.append(m)
    return out, thr


def _sky_complement_candidates(small_rgb, sky_small):
    """Foreground = everything that is not sky (works when the object fills the
    frame and the border is therefore all object)."""
    if sky_small is None or not sky_small.any():
        return []
    h, w = small_rgb.shape[:2]
    m = ~dilate_mask(sky_small.astype(bool), 1)
    m = morph(m, 1, False)
    close_r = max(1, int(round(min(h, w) * 0.012)))
    out = []
    for comp in label_components(m):
        x0, y0, x1, y1 = comp["bbox"]
        area_frac = comp["area"] / float(h * w)
        if not (0.01 <= area_frac <= 0.6):
            continue
        mm = np.zeros((h, w), bool)
        ys = np.fromiter((p[0] for p in comp["pixels"]), np.int32, len(comp["pixels"]))
        xs = np.fromiter((p[1] for p in comp["pixels"]), np.int32, len(comp["pixels"]))
        mm[ys, xs] = True
        mm = _close_mask(mm, close_r)
        if int(mm.sum()) >= 64:
            out.append(mm)
    return out


def detect_target(small_rgb, sky_small):
    """Segment the single salient object with two complementary hypotheses.

    A. border background: the frame edge (plus the sky) describes the
       background, so foreground is colour far from it. Correct when the object
       sits on a distinctive background.
    B. sky complement: everything that is not sky. Correct when the object
       fills the frame and hypothesis A inverts (the object becomes the
       "background" because it also covers the border).

    Both are scored on edge alignment, compactness and how much of the mask is
    sky-coloured; the better one wins. That removes the failure mode where a
    close-up airliner was masked as a ring of shadowed edges.
    """
    h, w = small_rgb.shape[:2]
    grad = gradient_mag(small_rgb)
    g_mean = float(grad.mean()) + 1e-6
    sky_median = (np.median(small_rgb[sky_small], axis=0)
                  if sky_small is not None and sky_small.any() else None)
    cands = []
    border_masks, thr = _border_bg_candidates(small_rgb, sky_small)
    for m in border_masks:
        cands.append(("border_bg", m))
    if sky_median is not None and float(sky_small.mean()) >= 0.08:
        for m in _sky_complement_candidates(small_rgb, sky_small):
            cands.append(("sky_complement", m))
    best = None
    for name, m in cands:
        st = _mask_stats(m, small_rgb, sky_median, grad, g_mean)
        if st is None:
            continue
        if best is None or st["score"] > best["stats"]["score"]:
            best = dict(name=name, mask=m, stats=st)
    if best is None:
        return None, dict(method="none", threshold=round(float(thr or 0), 1))
    mask, refined = refine_target(small_rgb, best["mask"])
    st = _mask_stats(mask, small_rgb, sky_median, grad, g_mean) or best["stats"]
    return mask, dict(method=best["name"] + ("+local" if refined else ""),
                      threshold=round(float(thr or 0), 1),
                      edge_align=round(st["edge"], 2),
                      compactness=round(st["compact"], 3),
                      sky_penalty=round(st["sky_pen"], 3),
                      refined=bool(refined))


def auto_mask(rgb, work_size):
    small = resize_rgb(rgb, work_size).astype(np.float64) * 255.0
    sky_small, sky_frac = detect_sky(small)
    tgt_small, info = detect_target(small, sky_small)
    info["sky_frac"] = sky_frac
    return small, sky_small, tgt_small, info


# --------------------------------------------------------------------------
# measurements
# --------------------------------------------------------------------------

def radial_psd(patch, mask):
    """Radial PSD of a masked patch: (frequency, power) with DC removed."""
    if patch.size < 1024 or mask.sum() < 64:
        return None
    med = float(np.median(patch[mask]))
    p = np.where(mask, patch, med)
    p = p - p.mean()
    h, w = p.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    spec = np.fft.rfft2(p * win)
    psd = (spec.real ** 2 + spec.imag ** 2) / float(h * w)
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.rfftfreq(w)[None, :]
    fr = np.sqrt(fy ** 2 + fx ** 2)
    nb = 25
    edges = np.linspace(0.0, 0.5, nb + 1)
    idx = np.clip(np.digitize(fr, edges) - 1, 0, nb - 1)
    freqs, powers = [], []
    for b in range(nb):
        sel = idx == b
        if int(sel.sum()) < 8:
            continue
        freqs.append(float(fr[sel].mean()))
        powers.append(float(psd[sel].mean()))
    return np.asarray(freqs), np.asarray(powers)


def fit_slope(freqs, powers, fmin=0.02, fmax=0.42):
    if freqs is None:
        return None
    sel = (freqs >= fmin) & (freqs <= fmax) & (powers > 0)
    if int(sel.sum()) < 4:
        return None
    x = np.log10(freqs[sel])
    y = np.log10(powers[sel])
    a = np.polyfit(x, y, 1)
    return float(a[0])


def largest_window(mask, max_side=512):
    """A rectangular window inside the mask, centred on its centroid.

    Starts from the mask bounding box (not the frame: a small target in a big
    photo would otherwise yield a window that is mostly background and the
    spectrum would measure the background) and shrinks until the window is at
    least 60 % covered by the mask.
    """
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    if ys.size < 64:
        return None
    y0b, y1b = int(ys.min()), int(ys.max()) + 1
    x0b, x1b = int(xs.min()), int(xs.max()) + 1
    cy, cx = int(np.median(ys)), int(np.median(xs))
    bh, bw = y1b - y0b, x1b - x0b
    best, best_fill = None, 0.0
    for scale in (1.0, 0.85, 0.7, 0.55, 0.45, 0.35):
        hh = int(min(max_side, max(48, bh * scale)))
        ww = int(min(max_side, max(48, bw * scale)))
        y0 = max(0, min(h - hh, cy - hh // 2))
        x0 = max(0, min(w - ww, cx - ww // 2))
        y1, x1 = min(h, y0 + hh), min(w, x0 + ww)
        if y1 - y0 < 48 or x1 - x0 < 48:
            continue
        fill = float(mask[y0:y1, x0:x1].mean())
        if fill > best_fill:
            best, best_fill = (y0, y1, x0, x1), fill
        if fill >= 0.6:
            return (y0, y1, x0, x1)
    return best if best_fill >= 0.25 else None


def measure_image(path, entry):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    if img.width > 1024:
        img = img.resize((1024, max(1, round(img.height * 1024 / img.width))),
                         Image.LANCZOS)
    rgb = np.asarray(img, np.float32) / 255.0
    h, w = rgb.shape[:2]
    lin = srgb_to_linear(rgb)
    lin_luma = luma_srgb(lin)
    gray_dn = luma_srgb(rgb) * 255.0

    work_size = (max(64, w // FACTOR), max(48, h // FACTOR))
    small, sky_small, tgt_small, info = auto_mask(rgb, work_size)
    sh, sw = small.shape[:2]
    sky_full = resize_to((sky_small.astype(np.uint8) * 255), (w, h), Image.NEAREST) > 127

    box = entry.get("manual_box")
    if box:
        x0 = int(round(float(box[0]) * sw))
        y0 = int(round(float(box[1]) * sh))
        x1 = int(round(float(box[2]) * sw))
        y1 = int(round(float(box[3]) * sh))
        region = np.zeros((sh, sw), bool)
        region[max(0, y0):min(sh, y1), max(0, x0):min(sw, x1)] = True
        source = "manual_box"
    elif tgt_small is not None and tgt_small.any():
        ys, xs = np.nonzero(tgt_small)
        bh = ys.max() - ys.min() + 1
        bw = xs.max() - xs.min() + 1
        my = max(2, int(round(bh * 0.07)))
        mx = max(2, int(round(bw * 0.07)))
        region = np.zeros((sh, sw), bool)
        region[max(0, int(ys.min()) - my):min(sh, int(ys.max()) + 1 + my),
               max(0, int(xs.min()) - mx):min(sw, int(xs.max()) + 1 + mx)] = True
        source = "auto_bbox"
    else:
        region = np.zeros((sh, sw), bool)
        region[int(0.25 * sh):int(0.75 * sh), int(0.25 * sw):int(0.75 * sw)] = True
        source = "fallback_center"

    # The object region is the box around the detected object minus the sky.
    # Distant white airliners against a hazy sky segment as a thin ring of
    # shadowed edges, so the mask is used for its bounding box only; colour is
    # then taken from every non-sky pixel inside that box ("difference from the
    # sky model", which is the simple segmentation this task allows).
    if sky_small.any() and int(sky_small.sum()) > 32:
        # Remove only the sky the flood fill actually found. A colour test
        # against the sky median is *not* used here: a white airliner under a
        # bright hazy sky sits within a few DN of the sky colour and would be
        # erased, leaving the dark windows and shadowed edges as the "paint".
        sky_free = region & ~dilate_mask(sky_small.astype(bool), 1)
        if int(sky_free.sum()) >= 0.30 * int(region.sum()) and int(sky_free.sum()) >= 64:
            region = sky_free
    # A box that covers more than a third of the frame is mostly context around
    # the object (a car parked in a street), so colour is taken from the middle
    # of it, which is where the object sits.
    if region.mean() > 0.32:
        ys, xs = np.nonzero(region)
        cy0, cy1 = int(ys.min()), int(ys.max()) + 1
        cx0, cx1 = int(xs.min()), int(xs.max()) + 1
        hh, ww = cy1 - cy0, cx1 - cx0
        iy, ix = int(hh * 0.15), int(ww * 0.15)
        region = np.zeros_like(region)
        region[cy0 + iy:cy1 - iy, cx0 + ix:cx1 - ix] = True
    target = resize_to((region.astype(np.uint8) * 255), (w, h), Image.NEAREST) > 127
    target = fill_holes(target) if source != "auto_bbox" else target
    area_frac = float(target.mean())
    if area_frac < 0.002 or area_frac > 0.60:
        if source != "manual_box":
            target = np.zeros((h, w), bool)
            target[int(0.28 * h):int(0.72 * h), int(0.28 * w):int(0.72 * w)] = True
            source = "fallback_center"

    er = max(2, int(round(min(h, w) / 260.0)))
    core = erode_mask(target, er)
    if int(core.sum()) < 400:
        core = erode_mask(target, max(1, er // 2))
    if int(core.sum()) < 200:
        core = target
    if int(core.sum()) < 100:
        raise RuntimeError("target mask too small (%d px)" % int(core.sum()))

    target_median_srgb = np.median(rgb[core], axis=0) * 255.0
    target_median_lin = np.median(lin[core], axis=0)
    target_lin_luma = float(np.median(lin_luma[core]))

    # sky / background reference
    sky_frac = float(sky_full.mean())
    if sky_frac >= 0.05:
        bg_mask = erode_mask(sky_full, max(2, er // 2))
        bg_kind = "sky"
    else:
        bg_mask = erode_mask(~dilate_mask(target, max(4, 2 * er)), max(2, er // 2))
        bg_kind = "scene"
    if int(bg_mask.sum()) < 2000:
        bg_mask = ~dilate_mask(target, max(4, 2 * er))
        bg_kind = "scene"
    sky_mean = float(np.mean(gray_dn[bg_mask]))
    sky_std = float(np.std(gray_dn[bg_mask]))
    sky_mean_rgb = np.mean(rgb[bg_mask], axis=0) * 255.0
    sky_luma_lin = float(np.median(lin_luma[bg_mask]))
    sky_sat = float(np.mean((rgb[bg_mask].max(axis=1) - rgb[bg_mask].min(axis=1)) /
                            np.maximum(rgb[bg_mask].max(axis=1), 1e-6)))

    # A sky sample is only a usable illuminant reference when it really is sky:
    # bright, not a dark object the flood fill ate, and not so cloudy that its
    # mean radiance stops tracking the solar irradiance.
    # A saturated blue region whose median IS the sky colour means the boxed
    # area is still sky (sky pockets between wing and tail are not reachable by
    # the flood fill). That sample cannot describe the paint.
    region_sky_like = bool(
        sky_frac >= 0.05 and sky_sat > 0.25 and
        float(np.max(np.abs(target_median_srgb - sky_mean_rgb))) < 25.0)
    sr, sg, sb = [float(v) for v in sky_mean_rgb]
    sky_greenish = bool(sg > sb + 8.0)
    sky_neutral = bool(sky_sat < 0.10)
    sky_blueish = bool(sb >= sr + 3.0)
    sky_ok = bool(bg_kind == "sky" and 80.0 <= sky_mean <= 252.0 and sky_std <= 65.0
                  and not sky_greenish and (sky_blueish or sky_neutral))
    sky_clear = bool(sky_ok and sky_sat >= 0.10 and sky_mean <= 235.0)

    # illumination model -> exposure-free albedo
    if sky_ok:
        if sky_sat < 0.09 and sky_mean > 205:
            k = 1.2
        elif sky_sat < 0.13 or sky_mean > 225:
            k = 2.0
        else:
            k = 4.0
    else:
        k = 1.6   # non-sky background: only a rough ambient proxy, flagged low
    ratio = target_lin_luma / max(sky_luma_lin, 1e-5)
    albedo = np.clip(target_median_lin / max(k * sky_luma_lin, 1e-5), 0.01, 0.95)
    target_p90 = float(np.percentile(lin_luma[core], 90))
    backlit = bool(sky_ok and target_p90 < 0.30 * sky_luma_lin)
    albedo_valid = bool(sky_ok and not backlit and not region_sky_like
                        and np.max(albedo) < 0.90)

    # texture: high-frequency power ratio on the target
    win = largest_window(core)
    hf_ratio = None
    if win:
        y0, y1, x0, x1 = win
        patch = gray_dn[y0:y1, x0:x1]
        pmask = core[y0:y1, x0:x1]
        if pmask.mean() > 0.5:
            psd = radial_psd(patch, pmask)
            if psd is not None:
                f, p = psd
                lo = (f >= 0.02) & (f <= 0.5)
                hi = (f >= 0.25) & (f <= 0.5)
                if p[lo].sum() > 0:
                    hf_ratio = float(p[hi].sum() / p[lo].sum())
    # Laplacian variance, DN^2
    lap = (4.0 * gray_dn[1:-1, 1:-1] - gray_dn[:-2, 1:-1] - gray_dn[2:, 1:-1]
           - gray_dn[1:-1, :-2] - gray_dn[1:-1, 2:])
    lap_mask = core[1:-1, 1:-1]
    lap_var = float(np.var(lap[lap_mask])) if int(lap_mask.sum()) > 100 else None

    # sky: PSD slope and camera noise
    psd_slope = None
    bwin = largest_window(bg_mask, max_side=512)
    if bwin:
        y0, y1, x0, x1 = bwin
        psd = radial_psd(gray_dn[y0:y1, x0:x1], bg_mask[y0:y1, x0:x1])
        psd_slope = fit_slope(*psd) if psd is not None else None
    d = (4.0 * gray_dn[1:-1, 1:-1] - gray_dn[:-2, 1:-1] - gray_dn[2:, 1:-1]
         - gray_dn[1:-1, :-2] - gray_dn[1:-1, 2:])
    nmask = bg_mask[1:-1, 1:-1]
    if int(nmask.sum()) > 500:
        cam_noise = highpass_sigma(d[nmask]) / math.sqrt(1.25)
    else:
        cam_noise = None

    # Two separate uses (see docs/参考图库.md): close-range frames carry
    # material/texture information, long-range frames carry the degradation /
    # target-pixel-count statistics. Keep them labelled instead of mixing them.
    if area_frac >= 0.05:
        usage = "A_material"
    elif area_frac <= 0.015:
        usage = "B_degradation"
    else:
        usage = "intermediate"

    edge_align = info.get("edge_align")
    if source == "manual_box":
        quality = "manual"
    elif source == "fallback_center":
        quality = "fallback"
    elif edge_align is not None and float(edge_align) < 1.5:
        quality = "marginal"     # mask does not follow a real contour
    elif area_frac < 0.008 or area_frac > 0.42:
        quality = "marginal"
    else:
        quality = "ok"

    return {
        "file": entry.get("file"),
        "title": entry.get("title"),
        "width": w,
        "height": h,
        "mask_source": source,
        "mask_area_frac": round(area_frac, 4),
        "mask_quality": quality,
        "bg_kind": bg_kind,
        "sky_frac": round(sky_frac, 4),
        "sky_sat": round(sky_sat, 3),
        "albedo_k": k,
        "target_median_srgb": [round(float(v), 1) for v in target_median_srgb],
        "target_median_linear": [round(float(v), 4) for v in target_median_lin],
        "target_lin_luma": round(target_lin_luma, 4),
        "sky_luma_lin": round(sky_luma_lin, 4),
        "target_over_sky": round(float(ratio), 3),
        "albedo_linear": [round(float(v), 4) for v in albedo],
        "albedo_valid": albedo_valid,
        "region_sky_like": region_sky_like,
        "backlit": backlit,
        "sky_ok": sky_ok,
        "sky_clear": sky_clear,
        "sky_greenish": sky_greenish,
        "target_p90_lin": round(target_p90, 4),
        "confidence": ("high" if (sky_clear and not backlit and not region_sky_like
                                    and area_frac >= 0.03)
                       else ("medium" if (sky_ok and not backlit and not region_sky_like)
                             else "low")),
        "texture_hf_ratio": None if hf_ratio is None else round(hf_ratio, 5),
        "target_lap_var": None if lap_var is None else round(lap_var, 2),
        "sky_mean": round(sky_mean, 2),
        "sky_std": round(sky_std, 2),
        "sky_mean_rgb": [round(float(v), 1) for v in sky_mean_rgb],
        "psd_slope": None if psd_slope is None else round(psd_slope, 3),
        "cam_noise_dn": None if cam_noise is None else round(cam_noise, 2),
        "auto_threshold": round(float(info.get("threshold") or 0), 4),
        "edge_align": info.get("edge_align"),
        "compactness": info.get("compactness"),
        "usage": usage,
    }


# --------------------------------------------------------------------------
# aggregation, comparison with simkit/materials.py
# --------------------------------------------------------------------------

def med(values):
    vals = [v for v in values if v is not None]
    return float(np.median(vals)) if vals else None


def load_materials():
    try:
        sys.path.insert(0, ROOT)
        from simkit import materials as matlib
        return {name: dict(spec) for name, spec in matlib.MATERIALS.items()}
    except Exception as exc:  # noqa: BLE001
        log("! could not import simkit.materials (%s); using the parsed table" % exc)
        return {}


def pct_delta(cur, new):
    return [round(float(n - c), 4) for c, n in zip(cur, new)]


def ratio(cur, new):
    cur_l = float(np.mean(cur))
    new_l = float(np.mean(new))
    return round(new_l / max(cur_l, 1e-6), 3)


SUGGEST_WEIGHT = {"high": 0.85, "medium": 0.70, "low": 0.45, "none": 0.25}


def suggest_albedo(measured, current, confidence="medium"):
    """Suggested Base Color: mostly the measured effective albedo, shrunk toward
    the current library value in proportion to how weak the measurement is, so a
    single backlit photo cannot flip a whole category to near-black."""
    if measured is None:
        return list(current)
    w = SUGGEST_WEIGHT.get(confidence, 0.7)
    out = []
    for m, c in zip(measured, current):
        v = w * float(m) + (1.0 - w) * float(c)
        out.append(round(float(np.clip(v, 0.03, 0.9)), 3))
    return out


def category_report(category, images, measured, materials):
    """measured: per-image stats dicts (already filtered to accepted, non-None)."""
    accepted = measured
    sky_ref = [im for im in accepted if im.get("bg_kind") == "sky"]
    usable = [im for im in accepted if im.get("albedo_valid")]
    high = [im for im in usable if im.get("confidence") == "high"]
    medium = [im for im in usable if im.get("confidence") in ("high", "medium")]
    if len(high) >= 2:
        albedo_src, albedo_conf = high, "high"
    elif len(medium) >= 2:
        albedo_src, albedo_conf = medium, "medium"
    elif len(usable) >= 1:
        albedo_src, albedo_conf = usable, "low"
    else:
        albedo_src, albedo_conf = [], "none"
    albedos = [im["albedo_linear"] for im in albedo_src if im.get("albedo_linear")]
    albedo = [round(float(np.median([a[i] for a in albedos])), 4)
              for i in range(3)] if albedos else None

    def agg(key):
        return med([im.get(key) for im in accepted])

    params = {
        "albedo_linear": albedo,
        "albedo_p25": [round(float(np.percentile([a[i] for a in albedos], 25)), 4)
                       for i in range(3)] if albedos else None,
        "albedo_p75": [round(float(np.percentile([a[i] for a in albedos], 75)), 4)
                       for i in range(3)] if albedos else None,
        "albedo_k": med([im["albedo_k"] for im in albedo_src]) if albedo_src else None,
        "albedo_confidence": albedo_conf,
        "albedo_uncertainty": "+-35% (k model and camera tone curve)",
        "albedo_n_high": len(high),
        "albedo_n_usable": len(usable),
        "albedo_excluded": {
            "backlit": sum(1 for im in accepted if im.get("backlit")),
            "invalid_sky_or_clip": sum(1 for im in accepted if not im.get("albedo_valid")),
        },
        "target_over_sky": med([im["target_over_sky"] for im in accepted]),
        "target_median_srgb": [round(float(v), 1) for v in
                               np.median([im["target_median_srgb"] for im in accepted], axis=0)],
        "texture_hf_ratio": agg("texture_hf_ratio"),
        "target_lap_var": agg("target_lap_var"),
        "sky_mean": agg("sky_mean"),
        "sky_std": agg("sky_std"),
        "psd_slope": agg("psd_slope"),
        "cam_noise_dn": agg("cam_noise_dn"),
        "samples": [im["file"].split("/")[-1] for im in accepted],
        "n_images": len(accepted),
        "n_albedo_samples": len(albedos),
        "n_sky_background": len(sky_ref),
        "image_titles": [im.get("title") for im in accepted],
    }

    n_a = sum(1 for im in accepted if im.get("usage") == "A_material")
    n_b = sum(1 for im in accepted if im.get("usage") == "B_degradation")
    n_mid = len(accepted) - n_a - n_b
    params["usage"] = {
        "dominant": ("A_material" if n_a >= n_b else "B_degradation"),
        "A_material": n_a,
        "B_degradation": n_b,
        "intermediate": n_mid,
        "note": ("近距离大目标照片（目标占画面 >= 5%%）用于反推涂装颜色/蒙皮纹理；"
                 "远距离小目标照片（<= 1.5%%）只用于目标占比/退化链的定性对照，"
                 "其 albedo 计入但置信度低。本类 %d 张偏 A，%d 张偏 B，%d 张居中。"
                 % (n_a, n_b, n_mid)),
    }

    material, alternatives, note = CATEGORY_MATERIAL[category]
    mat = materials.get(material, {})
    current = list(mat.get("albedo") or [0.22, 0.24, 0.28])
    suggested = suggest_albedo(albedo, current, albedo_conf)
    params["suggested_basis"] = (
        "%.0f%% 实测有效反照率 + %.0f%% 现有 MATERIALS 值（低置信时向现值收缩）"
        % (100 * SUGGEST_WEIGHT.get(albedo_conf, 0.7),
           100 * (1 - SUGGEST_WEIGHT.get(albedo_conf, 0.7))))
    params["simkit_material"] = material
    params["simkit_albedo_current"] = [round(float(v), 3) for v in current]
    params["simkit_albedo_delta"] = pct_delta(current, suggested)
    params["simkit_albedo_ratio"] = ratio(current, suggested)
    params["suggested_simkit_albedo"] = suggested
    if albedo:
        params["simkit_albedo_ratio"] = round(
            float(np.mean(suggested)) / max(float(np.mean(current)), 1e-6), 3)
    params["simkit_alternatives"] = {
        name: dict(albedo=[round(float(v), 3) for v in
                           (materials.get(name, {}).get("albedo") or [0, 0, 0])],
                   note=alternative_note(name))
        for name in alternatives if name in materials}
    params["notes"] = (
        "%s；%d 张样本中 %d 张以天空为参考（k=%s），%d 张以场景背景为参考（低置信）；"
        "albedo 由 %d 张通过质量门的样本取中位数（置信度 %s）。"
        % (note, len(accepted), len(sky_ref), params["albedo_k"],
           len(accepted) - len(sky_ref), len(albedo_src), albedo_conf))
    params["per_image"] = [
        {k: im.get(k) for k in ("file", "bg_kind", "mask_source", "mask_quality",
                                "usage", "confidence", "sky_ok", "sky_clear",
                                "backlit", "region_sky_like", "albedo_valid",
                                "albedo_linear", "albedo_k",
                                "target_over_sky", "texture_hf_ratio", "target_lap_var",
                                "sky_mean", "sky_std", "psd_slope", "cam_noise_dn",
                                "mask_area_frac", "target_median_srgb",
                                "target_median_linear", "sky_luma_lin", "sky_mean_rgb",
                                "edge_align", "compactness")}
        for im in accepted]
    return params


def alternative_note(name):
    return {
        "metal_bare": "裸金属/复合材料蒙皮参考",
        "plastic": "白/浅色工程塑料外壳参考",
        "concrete": "混凝土结构参考",
        "foliage": "植被参考",
        "glass": "车窗玻璃参考",
        "rubber": "轮胎/橡胶参考",
    }.get(name, "")


# --------------------------------------------------------------------------
# review sheets
# --------------------------------------------------------------------------

def write_sheet(category, rows, out_path, thumb_w=330):
    if not rows:
        return None
    cols = 3
    n = len(rows)
    rows_n = (n + cols - 1) // cols
    caption_h = 34
    pad = 8
    cell_w = thumb_w
    cell_h = int(round(thumb_w * 0.62)) + caption_h
    sheet = Image.new("RGB", (cols * cell_w + pad * (cols + 1),
                              rows_n * cell_h + pad * (rows_n + 1)), (32, 32, 34))
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(rows):
        im = Image.open(row["path"]).convert("RGB")
        im.thumbnail((thumb_w - 2 * pad, cell_h - caption_h - 2 * pad))
        tile = Image.new("RGB", (cell_w - 2 * pad, cell_h - caption_h - 2 * pad),
                         (0, 0, 0))
        tile.paste(im, ((tile.width - im.width) // 2, (tile.height - im.height) // 2))
        if row.get("mask") is not None:
            mask = row["mask"]
            rgba = np.asarray(tile).copy()
            small = np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).resize(
                (tile.width, tile.height), Image.NEAREST)) > 127
            edge = small & ~erode_mask(small, 1)
            rgba[edge] = (255, 40, 40)
            tile = Image.fromarray(rgba)
        cx = pad + (i % cols) * (cell_w + pad)
        cy = pad + (i // cols) * (cell_h + pad)
        sheet.paste(tile, (cx, cy))
        label = row["label"][:58]
        draw.text((cx + 4, cy + tile.height + 6), label, fill=(235, 235, 235))
        draw.rectangle([cx, cy, cx + tile.width - 1, cy + tile.height - 1],
                       outline=(90, 90, 96))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sheet.save(out_path, quality=86)
    return out_path


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def run(args):
    catalog = load_json(CATALOG, None)
    if not catalog:
        raise SystemExit("no catalogue at %s: run tools/fetch_refs.py first"
                         % os.path.relpath(CATALOG, ROOT))
    materials = load_materials()
    cats = args.categories.split(",") if args.categories else CATEGORIES
    cats = [c.strip() for c in cats if c.strip() in catalog.get("categories", {})]
    results = {}
    if args.sheet:
        os.makedirs(REVIEW_DIR, exist_ok=True)

    for category in cats:
        node = catalog["categories"].get(category) or {}
        images = node.get("images") or []
        log("[%s] %d image(s)" % (category, len(images)))
        rows = []
        for entry in images:
            name = str(entry.get("file", "")).split("/")[-1]
            path = os.path.join(REFS, entry.get("file", ""))
            review = review_for(name)
            if review.get("accepted") is False:
                entry["accepted"] = False
                entry["review"] = review.get("reason", "rejected in review")
                entry.pop("stats", None)
                continue
            if review.get("box"):
                entry["manual_box"] = list(review["box"])
            entry["accepted"] = True
            entry["review"] = review.get("reason", "")
            if not os.path.isfile(path):
                entry["stats"] = None
                entry["review"] = "file missing on disk"
                catalog.setdefault("errors", []).append(
                    {"category": category, "file": entry.get("file"),
                     "stage": "extract", "error": "file missing"})
                continue
            try:
                stats = measure_image(path, entry)
            except Exception as exc:  # noqa: BLE001
                entry["stats"] = None
                entry["review"] = "extract failed: %s" % exc
                catalog.setdefault("errors", []).append(
                    {"category": category, "file": entry.get("file"),
                     "stage": "extract", "error": str(exc)})
                log("  ! %s: %s" % (name, exc))
                continue
            entry["stats"] = stats
            entry["mask_source"] = stats["mask_source"]
            log("  %-50s %-11s %-6s a=%.3f hf=%.4f lap=%6.0f sky=%3.0f/%.0f "
                "sl=%.2f nz=%.1f edge=%s area=%.3f"
                % (name[:50], stats["mask_quality"] + "/" + stats["bg_kind"],
                   stats["confidence"], float(np.mean(stats["albedo_linear"])),
                   stats["texture_hf_ratio"] or 0.0, stats["target_lap_var"] or 0.0,
                   stats["sky_mean"], stats["sky_std"], stats["psd_slope"] or 0.0,
                   stats["cam_noise_dn"] or 0.0, stats["edge_align"],
                   stats["mask_area_frac"]) +
                ("  BACKLIT" if stats.get("backlit") else "") +
                ("" if stats.get("sky_ok") else "  NOSKY"))
            if args.sheet:
                us = stats.get("usage", "?")
                lab = "%02d %s %s a=%.2f sky=%.0f/%.0f ar=%.2f" % (
                    len(rows) + 1, name.split("_")[1][:22] if "_" in name else name[:22],
                    stats["confidence"], float(np.mean(stats["albedo_linear"])),
                    stats["sky_mean"], stats["sky_std"], stats["mask_area_frac"])
                if not stats.get("sky_ok"):
                    lab += " NOSKY"
                if stats.get("backlit"):
                    lab += " BACKLIT"
                rows.append(dict(path=path, label=lab, mask=None))
        if args.sheet and rows:
            for row in rows:
                im = Image.open(row["path"]).convert("RGB")
                if im.width > 1024:
                    im = im.resize((1024, round(im.height * 1024 / im.width)))
                small = resize_rgb(np.asarray(im, np.float32) / 255.0,
                                   (max(64, im.width // FACTOR),
                                    max(48, im.height // FACTOR))) * 255.0
                sky_small, _ = detect_sky(small)
                tgt, inf = detect_target(small, sky_small)
                row["mask"] = tgt
            for i in range(0, len(rows), 9):
                out = os.path.join(REVIEW_DIR, "%s_%02d.jpg" % (category, i // 9 + 1))
                write_sheet(category, rows[i:i + 6], out)
                log("  sheet: %s" % os.path.relpath(out, ROOT))

        measured = [e["stats"] for e in images
                    if e.get("accepted", True) and e.get("stats")]
        results[category] = category_report(category, images, measured, materials)
        if category == "杂项":
            results[category]["per_material"] = misc_per_material(measured, materials)
        node["count"] = len(measured)
        save_json(CATALOG, catalog)

    # ---- global comparison against the current library -------------------
    meta = {
        "generated": now_iso(),
        "source": "assets/refs (Wikimedia Commons photographs)",
        "catalogue": "assets/refs/catalog.json",
        "tool": "tools/extract_refs.py",
        "method": {
            "target_mask": "border-background model: k-means on border+sky colours, "
                           "largest component farther than 98th percentile of the "
                           "background spread; 1/4 resolution. Hand-drawn normalised "
                           "box replaces the auto mask where it is wrong.",
            "albedo": "median linear target / (k * median linear sky); k=4.0 clear, "
                      "2.0 hazy, 1.2 overcast, 1.6 scene-background fallback",
            "texture_hf_ratio": "sum PSD(f>=0.25 cyc/px) / sum PSD(0.02..0.5 cyc/px), "
                                "Hann-windowed, DC removed",
            "target_lap_var": "4-neighbour Laplacian variance in DN^2 inside the target",
            "sky_mean_std": "sRGB luma mean/std (0..255) inside the sky/background mask",
            "psd_slope": "log-log slope of the radial PSD of the sky patch, 0.02..0.42 cyc/px",
            "cam_noise_dn": "robust sigma of the local 4-neighbour Laplacian residual in "
                            "the sky patch, /sqrt(1.25), in DN",
        },
        "uncertainty": "albedo +-35%; texture metrics are JPEG-limited (JPEG removes "
                       "high-frequency energy, so hf_ratio is a lower bound); "
                       "cam_noise excludes JPEG quantisation",
        "usage_guidance": {
            "A_material": "近距大目标（mask_area_frac >= 5% 画面）：涂装颜色、蒙皮纹理、"
                          "天空/背景统计只应从这里取；本图库主体属于这一类。",
            "B_degradation": "远距小目标（mask_area_frac <= 1.5%）：只能用于目标像素占比、"
                             "PSF/噪声/大气退化的定性对照，不能用来定色。",
            "rule": "两类不要混用：在色/纹理上对标的样本，不能同时用来对齐退化链。",
            "not_in_gallery": "本图库不含 AOT / Anti-UAV410 那类远距退化数据（AOT 还是灰度）；"
                              "20 km 机载工况没有可直接对标的真实测试集。",
            "source_corrections": "VisDrone 10 类里没有飞机；MSTAR/SenSiAC 是地面车辆 "
                                  "SAR/IR；公开真实 SAR 飞机切片仅约 4k（SAR-AIRcraft-1.0 / "
                                  "SARDet-100K）。本图库只承担光学外观参考，不承担 SAR/IR 对标。",
        },
        "reference_materials": {
            name: {"albedo": [round(float(v), 3) for v in materials[name]["albedo"]],
                   "emissivity": materials[name].get("emissivity")}
            for name in sorted(materials)},
    }
    params = dict(meta)
    params.update(results)
    save_json(OUT_JSON, params)
    log("params: %s (%d category/ies)" % (os.path.relpath(OUT_JSON, ROOT), len(results)))
    return 0


def misc_per_material(measured, materials):
    """杂项 mixes concrete, metal, foliage and vehicles: split by file tag."""
    out = {}
    for tag, matname, label in MISC_TAGS:
        sel = [im for im in measured
               if tag in str(im.get("file", "")).lower()
               and im.get("albedo_linear")]
        if not sel:
            continue
        alb = [im["albedo_linear"] for im in sel]
        alb_med = [round(float(np.median([a[i] for a in alb])), 4) for i in range(3)]
        cur = [round(float(v), 3) for v in materials.get(matname, {}).get("albedo", [0, 0, 0])]
        out[tag] = {"material": matname, "label": label,
                    "albedo_linear": alb_med, "simkit_albedo_current": cur,
                    "suggested_simkit_albedo": suggest_albedo(alb_med, cur, "low"),
                    "samples": [im["file"].split("/")[-1] for im in sel]}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--categories", default="", help="comma separated subset")
    ap.add_argument("--sheet", action="store_true", help="write review contact sheets")
    args = ap.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
