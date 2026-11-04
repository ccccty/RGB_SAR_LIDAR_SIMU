"""Quantitative comparison of rendered images against real-photo references.

The tool exists to answer one question repeatedly and reproducibly: *where and
how much does a render differ from a real photograph of the same class?*  It
does that without a neural network, using only numpy image statistics, so it
runs offline on CPU and gives the same numbers for the same inputs.

Pipeline
--------
1.  Load every image (Pillow is used at the I/O boundary only), normalise it to
    sRGB in [0, 1], honour EXIF orientation, composite alpha over white, expand
    grayscale, and downscale so the long side is at most ANALYSIS_MAX_SIDE.
    8-bit sources decode exactly at 1/255; 16-bit/float sources keep their
    precision instead of being re-quantised.  Corrupt or undersized files are
    reported and skipped -- never fatal.
2.  Split each image into a *target* region and a *background* region.
    Synth frames have exact annotations (per-frame labels/*.json or COCO json)
    so their target mask is the union of the labelled boxes.  Reference photos
    usually have no annotation, so their target mask comes from a documented
    heuristic (central prior x difference from the sky colour model).  When the
    reference catalogue carries a `manual_box` for a file, that box is used
    instead.  Every report states which source produced each side's mask.
3.  Compute per-region statistics (global tone, 16-bin histogram, saturation,
    Laplacian variance, Sobel gradient distribution, radial power-spectrum
    slope and band fractions, median-residual noise sigma, hue/saturation
    distributions and white-balance opponent means).
4.  Aggregate across images with a median (robust to one odd frame), compare
    each feature against the reference aggregate, and attach a verdict from a
    small fixed vocabulary (偏干净/偏噪, 偏软/偏锐, 频谱偏平/偏陡, 色偏 ...).
    Gated features decide the exit status; content-dependent features are
    reported as informational only.

Why the numbers are what they are
---------------------------------
*   All metrics are computed on a common analysis grid (long side <= 384 px,
    bilinear downscale).  This makes a 1280 px photo and a 480 px render
    comparable at a common display scale, but it also means photo sensor noise
    is partly averaged away.  The report records native and analysis sizes and
    says so in `caveats`.
*   Luma is the sRGB-encoded Rec.709 luma (0.2126 R + 0.7152 G + 0.0722 B),
    normalised to [0, 1].  One 8-bit code value is therefore 1/255 = 0.0039.
*   The noise estimator is a 3x3 median residual with a MAD->sigma calibration
    (NOISE_RESIDUAL_MEDIAN_SCALE).  It is robust to the minority of edge pixels
    and returns ~0 for a synthetic smooth image.
*   The power-spectrum slope alpha is the exponent in P(f) ~ f^-alpha, fitted
    on log-spaced radial bins between PSD_F_LO and PSD_F_HI (cycles/pixel).
    White noise gives alpha ~ 0; a smooth ramp gives alpha > 5; photographs
    with a natural 1/f spectrum typically land around 2-3.
*   Every threshold is a module constant below with a comment on its basis.
    They are engineering starting points derived from 8-bit quantisation,
    visible-noise ratios and the observed feature spread; treat them as tunable
    policy, not physical constants.

Configuration files (all optional)
-----------------------------------
*   configs/refs_params.json: consumed tolerantly.  A class entry may carry
    "synth"/"render_dir" and "refs"/"ref_dir" paths plus a "heuristic" dict
    with center_frac/sky_frac/percentile/min_score.  The reference-measurement
    schema the extraction tool also writes (albedo, sky_mean, psd_slope, ...)
    is recognised as per-class data and attached to the report as ref_params
    instead of being mistaken for a class->directory mapping.
*   assets/refs/catalog.json: optional; supplies accepted flags, original
    image dimensions and an optional manual_box for reference target regions.
*   out/<dataset>/meta.json, labels/<modality>/*.json and
    annotations_<modality>_coco.json: synth annotations used as target masks.

Public API
----------
    load_image(path, max_side=...)            -> decoded image record
    global_stats(rgb, mask=None)              -> dict
    edge_stats(gray, mask=None)               -> dict
    radial_psd(gray)                          -> dict
    noise_sigma(gray, mask=None)              -> float
    color_stats(rgb, mask=None)               -> dict
    heuristic_target_mask(rgb, **params)      -> (mask, info)
    compare(synth_dir, ref_dir, out_json=None) -> report dict
    compare_all(...)                          -> multi-class report dict
    format_report(report) / format_all_report(multi)
    load_ref_params(path) / load_ref_catalog(path)

CLI: python tools/compare_refs.py --help
"""
from __future__ import annotations

import datetime as _dt
import json
import math
import os
import unicodedata

import numpy as np

try:  # Pillow is the only third-party dependency and is already in requirements.txt
    from PIL import Image, ImageOps
except ImportError:  # pragma: no cover - exercised only on a broken install
    Image = None
    ImageOps = None

__all__ = [
    "ImageLoadError",
    "ANALYSIS_MAX_SIDE",
    "compare",
    "compare_all",
    "load_ref_params",
    "load_ref_catalog",
    "load_image",
    "global_stats",
    "edge_stats",
    "radial_psd",
    "noise_sigma",
    "color_stats",
    "laplacian_variance",
    "sobel_gradient_distribution",
    "radial_psd_slope",
    "radial_band_fractions",
    "luma_histogram",
    "hue_saturation_distributions",
    "heuristic_target_mask",
    "analyze_image",
    "format_report",
    "format_all_report",
    "FEATURE_SPECS",
]

MODULE_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Analysis geometry and fixed algorithm parameters
# ---------------------------------------------------------------------------

#: Long side of the common analysis grid.  384 px is the smallest grid on which
#: the 0.02-0.35 cycles/px PSD fit still gets ~30 usable radial bins, while
#: keeping a 24+24 image comparison under a second or two on CPU.
ANALYSIS_MAX_SIDE = 384

#: Anything smaller than this cannot supply a 3x3 neighbourhood, a histogram or
#: a meaningful radial spectrum; skip it and say so in the report.
MIN_IMAGE_SIDE = 16

#: Below this side the radial spectrum has too few bins for a stable fit, so the
#: slope is reported as n/a instead of a number that looks authoritative.
MIN_PSD_SIDE = 48

#: A region needs enough pixels for a median and a variance to mean anything.
MIN_REGION_PIXELS = 64

#: Bound on images analysed per side.  Medians over 24 frames are already stable
#: and this keeps a full --all run comfortably interactive.
MAX_IMAGES_PER_SIDE = 24

#: Radial PSD fit: 24 log-spaced annuli between 0.02 and 0.35 cycles/pixel.
#: Below 0.02 the Hann window and scene framing dominate; above 0.35 JPEG
#: blocking and the sensor's noise floor flatten the curve for reasons that are
#: about encoding, not about the scene.
PSD_RADIAL_BINS = 24
PSD_F_LO = 0.02
PSD_F_HI = 0.35

#: Energy bands as cycles/pixel: structure / mid detail / fine detail / sensor
#: and JPEG floor.  The fractions always sum to 1 (DC excluded).
PSD_BANDS = ((0.0, 0.05), (0.05, 0.15), (0.15, 0.30), (0.30, 0.50))
PSD_BAND_NAMES = ("band_low", "band_mid", "band_high", "band_ultra")

#: 3x3 median residual r = x - median3(x).  For i.i.d. N(0, sigma) pixels a
#: Monte-Carlo run of 4M samples gives median(|r|) = 0.5789 sigma, so
#: sigma ~= 1.7272 * median(|r|).  This single scale factor is what makes the
#: estimator unbiased for white noise while the median keeps it robust to edges.
NOISE_RESIDUAL_MEDIAN_SCALE = 1.7272

#: A Sobel magnitude above this counts as an "edge pixel" for edge_frac.  0.10
#: on a [0,1] luma image is a 10 % per-pixel contrast step, well above JPEG
#: ringing and below real object boundaries.
GRAD_EDGE_THRESH = 0.10

HIST_BINS = 16
HUE_BINS = 12
SAT_BINS = 5
GRAD_BINS = 8

# ---------------------------------------------------------------------------
# Thresholds
#
# Everything below is a policy decision, and the comments say on what basis.
# Pixel-space units are [0, 1], i.e. 1/255 = 0.0039 per 8-bit code value.
# ---------------------------------------------------------------------------

#: A mean-luma shift below 0.05 (13 code values) is exposure/gain noise, not a
#: visible brightness disagreement between the two images.
THRESH_LUMA_MEAN_DELTA = 0.05

#: Luma std (contrast) is judged as a ratio; 1.5x is roughly the point where a
#: reviewer starts calling one image "flat" and the other "punchy".
THRESH_LUMA_STD_RATIO = 1.5

#: Fraction of pixels with any channel >= 250/255.  A 2-percentage-point
#: difference is already a visible clipping regime change on real photos.
THRESH_SAT_FRAC_DELTA = 0.02

#: Dark pixels (luma <= 0.02).  Content dependent, reported but not gated.
THRESH_DARK_FRAC_DELTA = 0.05

#: Noise sigma.  Below one 8-bit code value (0.004) the difference is invisible
#: and dominated by quantisation/JPEG, so that is the absolute floor; outside a
#: 2x ratio the grain is visibly different at 100 % view.
THRESH_NOISE_ABS_FLOOR = 0.004
THRESH_NOISE_RATIO = 2.0

#: Laplacian variance.  2x is a clearly different micro-detail level; the
#: absolute floor keeps two already-smooth images from failing on a tiny 2x
#: ratio of near-zero numbers.
THRESH_LAP_RATIO = 2.0
THRESH_LAP_ABS_FLOOR = 0.0005

#: Sobel gradient p90.  1.6x plus a 0.005 floor (about 1.3 code values).
THRESH_GRAD_RATIO = 1.6
THRESH_GRAD_ABS_FLOOR = 0.005

#: Power-spectrum slope alpha.  White noise is ~0, a linear ramp >5, real
#: photographs ~2-3.  0.35 slope units is a visible change in texture crispness
#: and sits well below the noise-vs-photo separation (~2.5), so it does not fire
#: on normal frame-to-frame content variation alone.
THRESH_PSD_ALPHA_DELTA = 0.35

#: High-band (0.15-0.5 cycles/px) energy fraction.  5 percentage points of the
#: total AC energy is a visible grain/aliasing difference at this scale.
THRESH_HIGH_BAND_DELTA = 0.05

#: Mean HSV saturation.  0.08 (8 points of 1.0) is a visible chroma shift.
THRESH_SAT_MEAN_DELTA = 0.08

#: Hasler-Suesstrunk colourfulness: 1.6x plus a 0.02 floor.
THRESH_COLORFULNESS_RATIO = 1.6
THRESH_COLORFULNESS_ABS_FLOOR = 0.02

#: Opponent white-balance means: R-G and B-(R+G)/2.  0.03 is ~8 code values,
#: about where a neutral grey starts reading as warm/cool.
THRESH_CAST_DELTA = 0.03

#: Distribution distances (reported, informative only).  L1 distance 0.35 on a
#: normalised histogram means a third of the mass has moved.
THRESH_HIST_L1 = 0.35
THRESH_HUE_L1 = 0.35

#: Target-region gates are looser than full-frame gates because the reference
#: mask is a heuristic, not a pixel-accurate annotation.
THRESH_TARGET_LUMA_DELTA = 0.08
THRESH_TARGET_STD_RATIO = 2.0
THRESH_TARGET_SAT_DELTA = 0.10
THRESH_TARGET_NOISE_RATIO = 3.0
THRESH_TARGET_LAP_RATIO = 2.5

#: Area ratio outside this range raises a warning (not a gate): target-region
#: statistics are then computed over very different scene compositions.
THRESH_TARGET_AREA_RATIO = 4.0

# ---------------------------------------------------------------------------
# Target-region heuristic defaults
# ---------------------------------------------------------------------------

#: Central candidate window half-width as a fraction of each image dimension.
DEFAULT_CENTER_FRAC = 0.42

#: Top band used to estimate the sky/background colour model.
DEFAULT_SKY_FRAC = 0.12

#: Percentile of the weighted colour-distance score kept as target pixels.
DEFAULT_SCORE_PERCENTILE = 70.0

#: Absolute floor for the score threshold, so a flat image cannot select noise.
DEFAULT_MIN_SCORE = 0.025

#: Weights on the three opponent-ish channels when measuring distance from the
#: sky model: luma, red-green, yellow-blue.  Chosen so a grey aircraft against a
#: blue sky is separated by chroma and value together.
SKY_DIST_WEIGHTS = (1.2, 1.5, 1.5)

#: Temporary files are never opened as images and image modules are never
#: treated as directories.
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")

#: Region labels used by the report renderer and by JSON consumers.
REGION_CN = {"full": "全图", "target": "目标区", "background": "背景区",
             "meta": "目标区"}

#: Scalars carried in every per-region statistics dict.
REGION_SCALARS = (
    "mean", "std", "p05", "p50", "p95", "p99",
    "r_mean", "g_mean", "b_mean", "sat_frac", "dark_frac",
    "lap_var", "grad_mean", "grad_p50", "grad_p90", "grad_p99", "edge_frac",
    "alpha", "psd_r2",
    "band_low", "band_mid", "band_high", "band_ultra",
    "noise_sigma",
    "sat_mean", "sat_std", "value_mean", "colorfulness",
    "cast_rg", "cast_by", "hue_mean", "hue_concentration",
)

#: Fixed-size arrays carried in every per-region statistics dict.
REGION_ARRAYS = ("hist", "hue_hist", "grad_hist", "sat_hist")

#: The feature table.  One dict per row printed in the report.
#: kind: "abs"    -> |synth-ref| <= tol
#:       "ratio"  -> ratio inside [1/tol, tol], with an absolute floor
#:       "noise"  -> ratio inside [1/tol, tol] OR |delta| <= floor
#: gate: whether this row contributes to the pass/fail exit status.
FEATURE_SPECS = (
    dict(key="full.luma_mean", region="full", metric="mean", label="亮度均值",
         kind="abs", tol=THRESH_LUMA_MEAN_DELTA, fmt="%.3f",
         neg="偏暗", pos="偏亮", gate=True),
    dict(key="full.luma_std", region="full", metric="std", label="对比度(亮度σ)",
         kind="ratio", tol=THRESH_LUMA_STD_RATIO, fmt="%.3f",
         neg="对比偏弱", pos="对比偏强", gate=True),
    dict(key="full.sat_frac", region="full", metric="sat_frac", label="饱和像素占比",
         kind="abs", tol=THRESH_SAT_FRAC_DELTA, fmt="%.2f%%", scale=100.0,
         neg="过曝像素偏少", pos="过曝像素偏多", gate=True),
    dict(key="full.dark_frac", region="full", metric="dark_frac", label="暗部像素占比",
         kind="abs", tol=THRESH_DARK_FRAC_DELTA, fmt="%.2f%%", scale=100.0,
         neg="暗部偏少", pos="暗部偏多", gate=False),
    dict(key="full.noise_sigma", region="full", metric="noise_sigma", label="噪声σ",
         kind="noise", tol=THRESH_NOISE_RATIO, floor=THRESH_NOISE_ABS_FLOOR,
         fmt="%.4f", neg="偏干净", pos="偏噪", gate=True),
    dict(key="full.lap_var", region="full", metric="lap_var", label="拉普拉斯方差",
         kind="ratio", tol=THRESH_LAP_RATIO, floor=THRESH_LAP_ABS_FLOOR,
         fmt="%.5f", neg="偏软", pos="偏锐", gate=True),
    dict(key="full.grad_p90", region="full", metric="grad_p90", label="Sobel梯度p90",
         kind="ratio", tol=THRESH_GRAD_RATIO, floor=THRESH_GRAD_ABS_FLOOR,
         fmt="%.4f", neg="边缘偏弱", pos="边缘偏强", gate=True),
    dict(key="full.psd_alpha", region="full", metric="alpha", label="功率谱斜率α",
         kind="abs", tol=THRESH_PSD_ALPHA_DELTA, fmt="%.2f",
         neg="频谱偏平", pos="频谱偏陡", gate=True),
    dict(key="full.band_high", region="full", metric="band_high", label="高频能量占比",
         kind="abs", tol=THRESH_HIGH_BAND_DELTA, fmt="%.1f%%", scale=100.0,
         neg="高频偏少", pos="高频偏多", gate=True),
    dict(key="full.band_low", region="full", metric="band_low", label="低频能量占比",
         kind="abs", tol=THRESH_HIGH_BAND_DELTA, fmt="%.1f%%", scale=100.0,
         neg="低频偏少", pos="低频偏多", gate=False),
    dict(key="full.sat_mean", region="full", metric="sat_mean", label="平均饱和度",
         kind="abs", tol=THRESH_SAT_MEAN_DELTA, fmt="%.3f",
         neg="饱和度偏低", pos="饱和度偏高", gate=True),
    dict(key="full.colorfulness", region="full", metric="colorfulness", label="色彩丰富度",
         kind="ratio", tol=THRESH_COLORFULNESS_RATIO, floor=THRESH_COLORFULNESS_ABS_FLOOR,
         fmt="%.3f", neg="色彩偏淡", pos="色彩偏艳", gate=True),
    dict(key="full.cast_rg", region="full", metric="cast_rg", label="R-G色偏",
         kind="abs", tol=THRESH_CAST_DELTA, fmt="%+.3f",
         neg="偏青绿", pos="偏红", gate=True),
    dict(key="full.cast_by", region="full", metric="cast_by", label="B-(R+G)/2色偏",
         kind="abs", tol=THRESH_CAST_DELTA, fmt="%+.3f",
         neg="偏黄", pos="偏蓝", gate=True),
    dict(key="target.area_frac", region=None, metric="target_area_frac",
         label="目标区面积占比", kind="ratio", tol=THRESH_TARGET_AREA_RATIO,
         fmt="%.2f%%", scale=100.0, neg="占比偏小", pos="占比偏大", gate=False),
    dict(key="target.luma_mean", region="target", metric="mean", label="目标亮度均值",
         kind="abs", tol=THRESH_TARGET_LUMA_DELTA, fmt="%.3f",
         neg="偏暗", pos="偏亮", gate=True),
    dict(key="target.luma_std", region="target", metric="std", label="目标对比度",
         kind="ratio", tol=THRESH_TARGET_STD_RATIO, fmt="%.3f",
         neg="目标细节偏弱", pos="目标细节偏强", gate=True),
    dict(key="target.noise_sigma", region="target", metric="noise_sigma", label="目标区噪声σ",
         kind="noise", tol=THRESH_TARGET_NOISE_RATIO, floor=THRESH_NOISE_ABS_FLOOR,
         fmt="%.4f", neg="目标偏干净", pos="目标偏噪", gate=True),
    dict(key="target.lap_var", region="target", metric="lap_var", label="目标区锐度",
         kind="ratio", tol=THRESH_TARGET_LAP_RATIO, floor=THRESH_LAP_ABS_FLOOR,
         fmt="%.5f", neg="目标偏软", pos="目标偏锐", gate=True),
    dict(key="target.sat_mean", region="target", metric="sat_mean", label="目标区饱和度",
         kind="abs", tol=THRESH_TARGET_SAT_DELTA, fmt="%.3f",
         neg="目标饱和度偏低", pos="目标饱和度偏高", gate=True),
    dict(key="background.luma_mean", region="background", metric="mean",
         label="背景亮度均值", kind="abs", tol=THRESH_LUMA_MEAN_DELTA, fmt="%.3f",
         neg="偏暗", pos="偏亮", gate=False),
    dict(key="background.noise_sigma", region="background", metric="noise_sigma",
         label="背景噪声σ", kind="noise", tol=THRESH_NOISE_RATIO,
         floor=THRESH_NOISE_ABS_FLOOR, fmt="%.4f",
         neg="背景偏干净", pos="背景偏噪", gate=True),
)

#: Exposed in every JSON report so a reader can audit what policy produced the
#: pass/fail bit without reading this file.
THRESHOLDS = {
    "analysis_max_side": ANALYSIS_MAX_SIDE,
    "luma_mean_delta": THRESH_LUMA_MEAN_DELTA,
    "luma_std_ratio": THRESH_LUMA_STD_RATIO,
    "sat_frac_delta": THRESH_SAT_FRAC_DELTA,
    "dark_frac_delta": THRESH_DARK_FRAC_DELTA,
    "noise_abs_floor": THRESH_NOISE_ABS_FLOOR,
    "noise_ratio": THRESH_NOISE_RATIO,
    "lap_ratio": THRESH_LAP_RATIO,
    "lap_abs_floor": THRESH_LAP_ABS_FLOOR,
    "grad_ratio": THRESH_GRAD_RATIO,
    "grad_abs_floor": THRESH_GRAD_ABS_FLOOR,
    "psd_alpha_delta": THRESH_PSD_ALPHA_DELTA,
    "high_band_delta": THRESH_HIGH_BAND_DELTA,
    "sat_mean_delta": THRESH_SAT_MEAN_DELTA,
    "colorfulness_ratio": THRESH_COLORFULNESS_RATIO,
    "colorfulness_abs_floor": THRESH_COLORFULNESS_ABS_FLOOR,
    "cast_delta": THRESH_CAST_DELTA,
    "hist_l1": THRESH_HIST_L1,
    "hue_l1": THRESH_HUE_L1,
    "target_luma_delta": THRESH_TARGET_LUMA_DELTA,
    "target_std_ratio": THRESH_TARGET_STD_RATIO,
    "target_sat_delta": THRESH_TARGET_SAT_DELTA,
    "target_noise_ratio": THRESH_TARGET_NOISE_RATIO,
    "target_lap_ratio": THRESH_TARGET_LAP_RATIO,
    "target_area_ratio": THRESH_TARGET_AREA_RATIO,
}

#: Static caveats included in every report.  These are not warnings about this
#: particular pair of directories; they are properties of the method.
METHOD_CAVEATS = (
    "所有指标在统一分析画布上计算：长边缩放至 %d px（bilinear），小于该尺寸的图不放大。"
    "参考照片通常像素更多，下采样会削弱其传感器噪声与细纹理，"
    "因此“偏干净/偏软”应结合原生分辨率人工复核。" % ANALYSIS_MAX_SIDE,
    "参考图目标区没有标注，采用“中心先验 x 天空色模型差异”启发式分割；"
    "目标区指标只作趋势参考，全图指标更可信。",
    "阈值是工程起点（见 simkit/refcompare.py 模块常量注释），"
    "不是物理真值；通过阈值不等于视觉可接受，仍需目视验收。",
    "16-bit/float 输入保留精度，8-bit 图像存在量化带纹：8-bit 平滑渐变的 "
    "PSD 斜率会接近照片，此时以噪声σ、锐度和高频能量为准。",
)


class ImageLoadError(RuntimeError):
    """An image file could not be decoded or is too small; callers skip it."""


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def _is_image_file(name):
    return name.lower().endswith(IMAGE_EXTENSIONS)


def _is_hidden(name):
    return name.startswith(".") or name.startswith("_")


def find_images(root, recursive=True):
    """Return image paths under *root* in deterministic sorted order."""
    if not root or not os.path.isdir(root):
        return []
    found = []
    if recursive:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if not _is_hidden(d))
            for name in sorted(filenames):
                if _is_image_file(name) and not _is_hidden(name):
                    found.append(os.path.join(dirpath, name))
    else:
        for name in sorted(os.listdir(root)):
            path = os.path.join(root, name)
            if os.path.isfile(path) and _is_image_file(name) and not _is_hidden(name):
                found.append(path)
    return sorted(found, key=lambda p: os.path.normcase(p))


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def _scaled_size(width, height, max_side, allow_upscale=False):
    longest = max(width, height)
    if longest <= max_side and not allow_upscale:
        return width, height
    scale = float(max_side) / float(longest)
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def _pil_rgb(im):
    """Return an RGB-mode PIL image, compositing alpha over white."""
    has_alpha = im.mode in ("RGBA", "LA", "PA") or (
        im.mode == "P" and "transparency" in im.info)
    if has_alpha:
        base = Image.new("RGBA", im.size, (255, 255, 255, 255))
        return Image.alpha_composite(base, im.convert("RGBA")).convert("RGB")
    return im.convert("RGB")


def load_image(path, max_side=ANALYSIS_MAX_SIDE, allow_upscale=False):
    """Decode one image into a normalised float32 sRGB array.

    Returns a dict with keys: path, rgb (h, w, 3 float32 in [0, 1]),
    native_size (w, h), analysis_size (w, h), scale, grayscale, bit_depth,
    notes.  8-bit sources are decoded exactly at 1/255; higher-bit-depth
    (16-bit / float) sources keep their precision in [0, 1] instead of being
    re-quantised, so a synthetic gradient is not mistaken for banding noise.

    Raises ImageLoadError for missing/undecodable/undersized files so a batch
    comparison can record the file and continue.
    """
    if Image is None:  # pragma: no cover
        raise ImageLoadError("Pillow is not installed")
    try:
        with Image.open(path) as im:
            im.load()
            native = tuple(int(v) for v in im.size)
            if min(native) < MIN_IMAGE_SIDE:
                raise ImageLoadError(
                    "image %dx%d is smaller than the %d px minimum"
                    % (native[0], native[1], MIN_IMAGE_SIDE))
            notes = []
            grayscale = im.mode in ("L", "LA", "I", "I;16", "I;16B", "I;16L",
                                    "I;16N", "F")
            try:
                transposed = ImageOps.exif_transpose(im)
            except Exception:
                transposed = im
            if transposed is not im and transposed.size != im.size:
                notes.append("EXIF orientation applied")
                native = tuple(int(v) for v in transposed.size)
            high_depth = transposed.mode in ("I", "I;16", "I;16B", "I;16L",
                                            "I;16N", "F")
            if high_depth:
                # Keep the extra precision instead of re-quantising to 8 bits:
                # a smooth 16-bit ramp must stay smooth or its power spectrum
                # would read as 8-bit quantisation banding (see the caveat in
                # the report).  Values are scaled by the p1-p99 range so a
                # 16-bit sensor image maps onto the same [0, 1] domain as the
                # 8-bit reference photos.
                arr = np.asarray(transposed, dtype=np.float64)
                lo, hi = np.percentile(arr, (1.0, 99.0))
                if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                    lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
                if hi <= lo:
                    arr = np.zeros_like(arr)
                else:
                    arr = (arr - lo) / (hi - lo)
                arr = np.clip(arr, 0.0, 1.0)
                w, h = transposed.size
                new_w, new_h = _scaled_size(w, h, int(max_side), allow_upscale)
                if (new_w, new_h) != (w, h):
                    float_im = Image.fromarray(arr.astype(np.float32), mode="F")
                    float_im = float_im.resize((new_w, new_h),
                                               Image.Resampling.BILINEAR)
                    arr = np.asarray(float_im, dtype=np.float32)
                    notes.append("resized %dx%d -> %dx%d (bilinear)"
                                 % (w, h, new_w, new_h))
                arr = np.repeat(arr.astype(np.float32)[..., None], 3, axis=2)
                notes.append("high-bit-depth source preserved as float [0,1] "
                             "(p1-p99 scaled)")
            else:
                rgb_im = _pil_rgb(transposed)
                w, h = rgb_im.size
                new_w, new_h = _scaled_size(w, h, int(max_side), allow_upscale)
                if (new_w, new_h) != (w, h):
                    rgb_im = rgb_im.resize((new_w, new_h), Image.Resampling.BILINEAR)
                    notes.append("resized %dx%d -> %dx%d (bilinear)"
                                 % (w, h, new_w, new_h))
                arr = np.asarray(rgb_im, dtype=np.float32) / 255.0
    except ImageLoadError:
        raise
    except Exception as exc:  # corrupt/truncated files raise a zoo of types
        raise ImageLoadError("%s: %s" % (os.path.basename(path), exc)) from exc
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ImageLoadError("%s: unexpected channel layout %r" % (path, arr.shape))
    if grayscale:
        notes.append("grayscale source expanded to RGB")
    return {
        "path": path,
        "rgb": arr.astype(np.float32, copy=False),
        "native_size": native,
        "analysis_size": (int(arr.shape[1]), int(arr.shape[0])),
        "scale": float(arr.shape[1]) / max(1.0, float(native[0])),
        "grayscale": bool(grayscale),
        "bit_depth": 16 if high_depth else 8,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# Low-level image operators (pure numpy)
# ---------------------------------------------------------------------------

def to_gray(rgb):
    """sRGB Rec.709 luma on an (h, w, 3) array in [0, 1]."""
    arr = np.asarray(rgb, dtype=np.float32)
    return (0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]).astype(np.float32)


def _median3(gray):
    """3x3 median filter (edge-clamped), pure numpy."""
    h, w = gray.shape
    padded = np.pad(gray, 1, mode="edge")
    stack = np.stack(
        [padded[dy:dy + h, dx:dx + w] for dy in range(3) for dx in range(3)],
        axis=0)
    return np.median(stack, axis=0)


def _laplacian(gray):
    """4-neighbour Laplacian, same shape as the input."""
    p = np.pad(gray, 1, mode="reflect")
    return p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] - 4.0 * gray


def _sobel(gray):
    """Sobel gradient magnitude on a [0, 1] image."""
    p = np.pad(gray, 1, mode="edge")
    gx = ((p[:-2, 2:] + 2.0 * p[1:-1, 2:] + p[2:, 2:])
          - (p[:-2, :-2] + 2.0 * p[1:-1, :-2] + p[2:, :-2])) / 8.0
    gy = ((p[2:, :-2] + 2.0 * p[2:, 1:-1] + p[2:, 2:])
          - (p[:-2, :-2] + 2.0 * p[:-2, 1:-1] + p[:-2, 2:])) / 8.0
    return np.sqrt(gx * gx + gy * gy)


def _erode(mask, iterations=1):
    """Binary erosion with a 3x3 square; used before masked edge statistics."""
    out = np.asarray(mask, dtype=bool)
    for _ in range(max(0, int(iterations))):
        p = np.pad(out, 1, mode="constant", constant_values=False)
        out = (p[:-2, 1:-1] & p[2:, 1:-1]
               & p[1:-1, :-2] & p[1:-1, 2:]
               & p[1:-1, 1:-1])
        if not out.any():
            break
    return out


def _majority3(mask, min_neighbours=5):
    """3x3 majority filter; removes isolated speckle from heuristic masks."""
    p = np.pad(np.asarray(mask, dtype=np.uint8), 1, mode="constant", constant_values=0)
    count = np.zeros_like(mask, dtype=np.int16)
    for dy in range(3):
        for dx in range(3):
            count += p[dy:dy + mask.shape[0], dx:dx + mask.shape[1]]
    out = count >= int(min_neighbours)
    return out if out.any() else mask


def _safe_median(values):
    arr = np.asarray(values, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.median(arr))


def _normalise_hist(hist):
    hist = np.asarray(hist, dtype=np.float64)
    total = hist.sum()
    if not np.isfinite(total) or total <= 0:
        return np.zeros_like(hist, dtype=np.float32)
    return (hist / total).astype(np.float32)


# ---------------------------------------------------------------------------
# Feature functions (each one answers one question; all return plain dicts)
# ---------------------------------------------------------------------------

def global_stats(rgb, mask=None):
    """Tone: mean/std/percentiles, 16-bin luma histogram, clipping fractions."""
    arr = np.asarray(rgb, dtype=np.float32)
    px = arr.reshape(-1, 3) if mask is None else arr[mask]
    if px.size == 0:
        return None
    luma = (0.2126 * px[:, 0] + 0.7152 * px[:, 1] + 0.0722 * px[:, 2])
    hist, _ = np.histogram(luma, bins=HIST_BINS, range=(0.0, 1.0))
    return {
        "mean": float(luma.mean()),
        "std": float(luma.std()),
        "p05": float(np.percentile(luma, 5)),
        "p50": float(np.percentile(luma, 50)),
        "p95": float(np.percentile(luma, 95)),
        "p99": float(np.percentile(luma, 99)),
        "r_mean": float(px[:, 0].mean()),
        "g_mean": float(px[:, 1].mean()),
        "b_mean": float(px[:, 2].mean()),
        "sat_frac": float(np.mean(np.any(px >= 250.0 / 255.0, axis=1))),
        "dark_frac": float(np.mean(luma <= 0.02)),
        "hist": _normalise_hist(hist),
    }


def edge_stats(gray, mask=None):
    """Edges: Laplacian variance plus the Sobel magnitude distribution."""
    gray = np.asarray(gray, dtype=np.float32)
    lap = _laplacian(gray)
    mag = _sobel(gray)
    if mask is None:
        lap_vals = lap.ravel()
        mag_vals = mag.ravel()
    else:
        m = np.asarray(mask, dtype=bool)
        eroded = _erode(m, 1)
        if eroded.sum() >= MIN_REGION_PIXELS:
            m = eroded
        if m.sum() < MIN_REGION_PIXELS:
            return None
        lap_vals = lap[m]
        mag_vals = mag[m]
    if lap_vals.size < MIN_REGION_PIXELS:
        return None
    grad_hist, _ = np.histogram(mag_vals, bins=GRAD_BINS, range=(0.0, 0.5))
    return {
        "lap_var": float(np.var(lap_vals)),
        "grad_mean": float(mag_vals.mean()),
        "grad_p50": float(np.percentile(mag_vals, 50)),
        "grad_p90": float(np.percentile(mag_vals, 90)),
        "grad_p99": float(np.percentile(mag_vals, 99)),
        "edge_frac": float(np.mean(mag_vals > GRAD_EDGE_THRESH)),
        "grad_hist": _normalise_hist(grad_hist),
    }


def radial_psd(gray):
    """Radially averaged power spectrum: 1/f^alpha slope and band fractions."""
    gray = np.asarray(gray, dtype=np.float64)
    h, w = gray.shape
    if min(h, w) < MIN_PSD_SIDE:
        return {
            "alpha": float("nan"),
            "psd_r2": float("nan"),
            "band_low": float("nan"),
            "band_mid": float("nan"),
            "band_high": float("nan"),
            "band_ultra": float("nan"),
        }
    centred = gray - gray.mean()
    window = np.outer(np.hanning(h), np.hanning(w))
    spectrum = np.fft.fftshift(np.fft.fft2(centred * window))
    power = spectrum.real ** 2 + spectrum.imag ** 2
    fy = np.fft.fftshift(np.fft.fftfreq(h))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(w))[None, :]
    radius = np.sqrt(fx * fx + fy * fy)
    total = float(power[radius > 0.0].sum())
    bands = []
    for lo, hi in PSD_BANDS:
        sel = (radius >= lo) & (radius < hi)
        bands.append(float(power[sel].sum()) / total if total > 0 else float("nan"))
    finite = (radius >= PSD_F_LO) & (radius <= PSD_F_HI)
    if finite.sum() < PSD_RADIAL_BINS:
        alpha = r2 = float("nan")
    else:
        edges = np.geomspace(PSD_F_LO, PSD_F_HI, PSD_RADIAL_BINS + 1)
        rr = radius[finite]
        pp = power[finite]
        idx = np.clip(np.digitize(rr, edges) - 1, 0, PSD_RADIAL_BINS - 1)
        averages = np.array([pp[idx == i].mean() if np.any(idx == i) else np.nan
                             for i in range(PSD_RADIAL_BINS)], dtype=np.float64)
        centres = np.sqrt(edges[:-1] * edges[1:])
        good = np.isfinite(averages) & (averages > 0)
        if good.sum() < 8:
            alpha = r2 = float("nan")
        else:
            x = np.log10(centres[good])
            y = np.log10(averages[good])
            slope, intercept = np.polyfit(x, y, 1)
            alpha = float(-slope)
            pred = slope * x + intercept
            ss_res = float(np.sum((y - pred) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "alpha": alpha,
        "psd_r2": r2,
        "band_low": bands[0],
        "band_mid": bands[1],
        "band_high": bands[2],
        "band_ultra": bands[3],
    }


def noise_sigma(gray, mask=None):
    """Robust sensor-noise sigma from the 3x3 median residual.

    The MAD of (x - median3(x)) is scaled by a Monte-Carlo calibration constant
    so i.i.d. Gaussian noise of sigma s is recovered as ~s.  Edges affect only a
    minority of pixels, and the median ignores them.
    """
    gray = np.asarray(gray, dtype=np.float32)
    residual = np.abs(gray - _median3(gray))
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.sum() < MIN_REGION_PIXELS:
            return float("nan")
        vals = residual[m]
    else:
        vals = residual.ravel()
    if vals.size < MIN_REGION_PIXELS:
        return float("nan")
    return float(NOISE_RESIDUAL_MEDIAN_SCALE * np.median(vals))


def _rgb_to_hsv_arrays(rgb):
    """Vectorised sRGB -> HSV for an (..., 3) float array in [0, 1]."""
    arr = np.asarray(rgb, dtype=np.float32)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    maxc = np.max(arr, axis=-1)
    minc = np.min(arr, axis=-1)
    delta = maxc - minc
    sat = np.where(maxc > 1e-8, delta / np.maximum(maxc, 1e-8), 0.0)
    hue = np.zeros_like(maxc)
    nonzero = delta > 1e-8
    with np.errstate(invalid="ignore", divide="ignore"):
        red = nonzero & (maxc == r)
        green = nonzero & (maxc == g) & ~red
        blue = nonzero & (maxc == b) & ~red & ~green
        hue[red] = np.mod((g[red] - b[red]) / delta[red], 6.0)
        hue[green] = (b[green] - r[green]) / delta[green] + 2.0
        hue[blue] = (r[blue] - g[blue]) / delta[blue] + 4.0
    hue = np.mod(hue * 60.0, 360.0)
    return hue, sat, maxc


def color_stats(rgb, mask=None):
    """Colour: HSV hue/saturation distributions plus white-balance opponents."""
    arr = np.asarray(rgb, dtype=np.float32)
    px = arr.reshape(-1, 3) if mask is None else arr[mask]
    if px.size == 0:
        return None
    hue, sat, val = _rgb_to_hsv_arrays(px)
    hue = hue.ravel()
    sat = sat.ravel()
    weights = sat
    hue_hist, _ = np.histogram(hue, bins=HUE_BINS, range=(0.0, 360.0), weights=weights)
    hue_hist = _normalise_hist(hue_hist)
    sat_hist, _ = np.histogram(sat, bins=SAT_BINS, range=(0.0, 1.0))
    sat_hist = _normalise_hist(sat_hist)
    if weights.sum() > 1e-9:
        angle = np.deg2rad(hue)
        cx = float(np.sum(weights * np.cos(angle)) / weights.sum())
        cy = float(np.sum(weights * np.sin(angle)) / weights.sum())
        hue_mean = float(np.mod(np.rad2deg(math.atan2(cy, cx)), 360.0))
        concentration = float(math.hypot(cx, cy))
    else:
        hue_mean = float("nan")
        concentration = 0.0
    r, g, b = px[:, 0], px[:, 1], px[:, 2]
    rg = r - g
    yb = 0.5 * (r + g) - b
    colorfulness = float(
        math.hypot(float(rg.std()), float(yb.std()))
        + 0.3 * math.hypot(float(rg.mean()), float(yb.mean())))
    return {
        "sat_mean": float(sat.mean()),
        "sat_std": float(sat.std()),
        "value_mean": float(val.mean()),
        "colorfulness": colorfulness,
        "cast_rg": float(rg.mean()),
        "cast_by": float(yb.mean()),
        "hue_mean": hue_mean,
        "hue_concentration": concentration,
        "hue_hist": hue_hist,
        "sat_hist": sat_hist,
    }


# Individual metric entry points.  The grouped functions above are what the
# report uses (one pass over each region), but each measurable feature is also
# available on its own so callers and tests can consume a single number/array.

def laplacian_variance(gray, mask=None):
    """4-neighbour Laplacian variance (sharpness) as a float."""
    stats = edge_stats(gray, mask)
    return float(stats["lap_var"]) if stats is not None else float("nan")


def sobel_gradient_distribution(gray, mask=None):
    """Normalised 8-bin Sobel magnitude distribution (0..0.5)."""
    stats = edge_stats(gray, mask)
    if stats is None:
        return np.full(GRAD_BINS, np.nan, dtype=np.float32)
    return stats["grad_hist"]


def radial_psd_slope(gray):
    """1/f^alpha radial power-spectrum slope alpha as a float."""
    return float(radial_psd(gray)["alpha"])


def radial_band_fractions(gray):
    """Four low/mid/high/ultra band energy fractions (sums to ~1)."""
    psd = radial_psd(gray)
    return np.array([psd["band_low"], psd["band_mid"], psd["band_high"],
                     psd["band_ultra"]], dtype=np.float32)


def luma_histogram(rgb, mask=None):
    """Normalised 16-bin luma histogram."""
    stats = global_stats(rgb, mask)
    return stats["hist"] if stats is not None else np.full(HIST_BINS, np.nan,
                                                           dtype=np.float32)


def hue_saturation_distributions(rgb, mask=None):
    """sRGB hue/saturation distributions and their first moments."""
    stats = color_stats(rgb, mask)
    if stats is None:
        return {"hue_hist": np.full(HUE_BINS, np.nan, dtype=np.float32),
                "sat_hist": np.full(SAT_BINS, np.nan, dtype=np.float32)}
    return {
        "hue_hist": stats["hue_hist"],
        "sat_hist": stats["sat_hist"],
        "hue_mean_deg": stats["hue_mean"],
        "hue_concentration": stats["hue_concentration"],
        "sat_mean": stats["sat_mean"],
        "sat_std": stats["sat_std"],
    }


def _empty_region():
    out = {key: float("nan") for key in REGION_SCALARS}
    out["hist"] = np.full(HIST_BINS, np.nan, dtype=np.float32)
    out["hue_hist"] = np.full(HUE_BINS, np.nan, dtype=np.float32)
    out["grad_hist"] = np.full(GRAD_BINS, np.nan, dtype=np.float32)
    out["sat_hist"] = np.full(SAT_BINS, np.nan, dtype=np.float32)
    return out


def region_stats(rgb, gray, mask=None, with_psd=False):
    """All statistics for one region (None mask means the whole frame)."""
    if mask is not None and int(np.asarray(mask).sum()) < MIN_REGION_PIXELS:
        return _empty_region()
    global_part = global_stats(rgb, mask)
    edge_part = edge_stats(gray, mask)
    color_part = color_stats(rgb, mask)
    if global_part is None or edge_part is None or color_part is None:
        return _empty_region()
    out = _empty_region()
    out.update(global_part)
    out.update(edge_part)
    out.update(color_part)
    out["noise_sigma"] = noise_sigma(gray, mask)
    if with_psd:
        out.update(radial_psd(gray))
    return out


def analyze_image(rgb, target_mask=None, target_source="none", path=None):
    """Compute full/target/background statistics for one already-loaded image."""
    gray = to_gray(rgb)
    full = region_stats(rgb, gray, None, with_psd=True)
    if target_mask is not None and int(np.asarray(target_mask).sum()) >= MIN_REGION_PIXELS:
        target = region_stats(rgb, gray, target_mask, with_psd=False)
        background = region_stats(rgb, gray, ~np.asarray(target_mask, dtype=bool),
                                  with_psd=False)
        area_frac = float(np.asarray(target_mask).mean())
    else:
        target = _empty_region()
        background = region_stats(rgb, gray, None, with_psd=False)
        area_frac = 0.0
        target_source = target_source or "none"
    return {
        "path": path,
        "full": full,
        "target": target,
        "background": background,
        "target_area_frac": area_frac,
        "target_source": target_source,
    }


# ---------------------------------------------------------------------------
# Target-region extraction
# ---------------------------------------------------------------------------

def _validate_heuristic_params(center_frac, sky_frac, percentile, min_score):
    try:
        center_frac = float(center_frac)
        sky_frac = float(sky_frac)
        percentile = float(percentile)
        min_score = float(min_score)
    except (TypeError, ValueError):
        center_frac, sky_frac, percentile, min_score = (
            DEFAULT_CENTER_FRAC, DEFAULT_SKY_FRAC,
            DEFAULT_SCORE_PERCENTILE, DEFAULT_MIN_SCORE)
    center_frac = min(0.6, max(0.10, center_frac))
    sky_frac = min(0.5, max(0.02, sky_frac))
    percentile = min(95.0, max(0.0, percentile))
    min_score = max(0.0, min_score)
    return center_frac, sky_frac, percentile, min_score


def heuristic_target_mask(rgb, center_frac=DEFAULT_CENTER_FRAC,
                          sky_frac=DEFAULT_SKY_FRAC,
                          percentile=DEFAULT_SCORE_PERCENTILE,
                          min_score=DEFAULT_MIN_SCORE):
    """Heuristic target region for images without annotations.

    The reference photos have no boxes, so this combines two cheap cues:

    * a central prior (subjects are normally near the frame centre), and
    * a colour-distance term from the sky/background model estimated on the top
      band of the image.

    Pixels in the central window whose weighted opponent-colour distance from
    the sky model exceeds a percentile threshold become the target.  A 3x3
    majority filter removes speckle.  The method and its parameters are returned
    in the info dict and printed in the report -- this is explicitly heuristic.
    """
    arr = np.asarray(rgb, dtype=np.float32)
    h, w = arr.shape[:2]
    center_frac, sky_frac, percentile, min_score = _validate_heuristic_params(
        center_frac, sky_frac, percentile, min_score)
    top_rows = max(4, int(h * sky_frac))
    top = arr[:top_rows].reshape(-1, 3)
    sky = np.median(top, axis=0)
    gray = to_gray(arr)
    a = arr[..., 0] - arr[..., 1]
    b = 0.5 * (arr[..., 0] + arr[..., 1]) - arr[..., 2]
    sky_l = float(0.2126 * sky[0] + 0.7152 * sky[1] + 0.0722 * sky[2])
    sky_a = float(sky[0] - sky[1])
    sky_b = float(0.5 * (sky[0] + sky[1]) - sky[2])
    wy, wa, wb = SKY_DIST_WEIGHTS
    dist = np.sqrt(wy * (gray - sky_l) ** 2
                   + wa * (a - sky_a) ** 2
                   + wb * (b - sky_b) ** 2)
    yy, xx = np.mgrid[0:h, 0:w]
    nx = (xx + 0.5) / float(w)
    ny = (yy + 0.5) / float(h)
    sigma = max(0.12, center_frac * 0.72)
    prior = np.exp(-(((nx - 0.5) / sigma) ** 2 + ((ny - 0.5) / sigma) ** 2))
    score = dist * prior
    candidate = (np.abs(nx - 0.5) <= center_frac) & (np.abs(ny - 0.5) <= center_frac)
    fallback = ""
    if candidate.sum() < MIN_REGION_PIXELS:
        candidate = np.ones((h, w), dtype=bool)
    threshold = max(min_score, float(np.percentile(score[candidate], percentile)))
    mask = (score > threshold) & candidate
    mask = _majority3(mask)
    area = float(mask.mean())
    if area < 0.002 or area > 0.65:
        # The heuristic decided the frame is mostly subject or mostly sky;
        # fall back to a plain central box so the report still has two regions.
        fallback = "central_box"
        mask = np.zeros((h, w), dtype=bool)
        y0, y1 = int(h * (0.5 - center_frac / 2.0)), int(h * (0.5 + center_frac / 2.0))
        x0, x1 = int(w * (0.5 - center_frac / 2.0)), int(w * (0.5 + center_frac / 2.0))
        mask[max(0, y0):max(1, y1), max(0, x0):max(1, x1)] = True
        area = float(mask.mean())
    info = {
        "method": "heuristic",
        "params": {
            "center_frac": center_frac,
            "sky_frac": sky_frac,
            "percentile": percentile,
            "min_score": min_score,
        },
        "sky_color": [float(v) for v in sky],
        "score_threshold": threshold,
        "area_frac": area,
        "fallback": fallback,
    }
    return mask, info


def _parse_box(box, default_native=None):
    """Normalise several box spellings to (x0, y0, x1, y1) in source pixels.

    Accepts dicts with x0/y0/x1/y1, dicts with bbox=[x, y, w, h], and 4-item
    sequences interpreted as xyxy when ordered, else xywh.  Returns None for
    unusable input.
    """
    native = default_native
    if isinstance(box, dict):
        if box.get("visible") is False:
            return None
        if all(k in box for k in ("x0", "y0", "x1", "y1")):
            vals = [box["x0"], box["y0"], box["x1"], box["y1"]]
        elif "bbox" in box and isinstance(box["bbox"], (list, tuple)):
            vals = list(box["bbox"])
        else:
            return None
        native = (box.get("width"), box.get("height")) if "width" in box else native
    else:
        try:
            vals = list(box)
        except TypeError:
            return None
    if len(vals) != 4:
        return None
    try:
        vals = [float(v) for v in vals]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in vals):
        return None
    x0, y0, x1, y1 = vals
    if x1 <= x0 or y1 <= y0:
        # xywh spelling
        x0, y0, w, h = vals
        if w <= 0 or h <= 0:
            return None
        x1, y1 = x0 + w, y0 + h
    if native and native[0] and native[1] and max(abs(v) for v in (x0, y0, x1, y1)) <= 1.5:
        x0, y0, x1, y1 = x0 * native[0], y0 * native[1], x1 * native[0], y1 * native[1]
    return x0, y0, x1, y1


def _boxes_mask(boxes, source_native, canvas_shape):
    """Union mask of boxes given in source pixel coordinates."""
    h, w = canvas_shape[:2]
    try:
        sw, sh = ((float(source_native[0]), float(source_native[1]))
                  if source_native else (float(w), float(h)))
    except (TypeError, ValueError, IndexError):
        sw, sh = float(w), float(h)
    if sw <= 0 or sh <= 0:
        sw, sh = float(w), float(h)
    mask = np.zeros((h, w), dtype=bool)
    for box in boxes:
        parsed = _parse_box(box, source_native)
        if parsed is None:
            continue
        x0, y0, x1, y1 = parsed
        # Normalised boxes were already expanded to pixels by _parse_box using
        # the native size; scale native pixels onto the analysis canvas.
        px0 = int(math.floor(x0 * w / sw))
        px1 = int(math.ceil(x1 * w / sw))
        py0 = int(math.floor(y0 * h / sh))
        py1 = int(math.ceil(y1 * h / sh))
        px0, px1 = max(0, min(w, px0)), max(0, min(w, px1))
        py0, py1 = max(0, min(h, py0)), max(0, min(h, py1))
        if px1 > px0 and py1 > py0:
            mask[py0:py1, px0:px1] = True
    return mask


def _manual_box_mask(box, source_native, canvas_shape):
    return _boxes_mask([box], source_native, canvas_shape)


# ---------------------------------------------------------------------------
# Annotation / catalogue ingestion
# ---------------------------------------------------------------------------

def _read_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _extract_boxes_from_label_doc(doc, native):
    raw = doc.get("boxes")
    items = []
    if isinstance(raw, dict):
        items = list(raw.values())
    elif isinstance(raw, list):
        items = raw
    else:
        return []
    boxes = []
    for item in items:
        parsed = _parse_box(item, native)
        if parsed is not None:
            boxes.append(parsed)
    return boxes


def _extract_coco_entries(data):
    """Return {basename: {boxes, native, source}} for a COCO document."""
    if not isinstance(data, dict) or "images" not in data or "annotations" not in data:
        return {}
    by_image = {}
    for ann in data.get("annotations", []):
        image_id = ann.get("image_id")
        bbox = ann.get("bbox")
        if image_id is None or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        try:
            x, y, w, h = [float(v) for v in bbox]
        except (TypeError, ValueError):
            continue
        if w <= 0 or h <= 0 or not all(math.isfinite(v) for v in (x, y, w, h)):
            continue
        by_image.setdefault(image_id, []).append((x, y, x + w, y + h))
    out = {}
    for image in data.get("images", []):
        image_id = image.get("id")
        name = os.path.basename(str(image.get("file_name", "")))
        if not name or image_id not in by_image:
            continue
        native = (image.get("width"), image.get("height"))
        out[name] = {
            "boxes": by_image[image_id],
            "native": native if native and native[0] and native[1] else None,
            "source": "coco",
        }
    return out


def build_synth_label_index(synth_dir, modality, image_paths, resolved_subdir=None):
    """Map image basename -> labelled boxes, from labels/<mod>/ or COCO json.

    Returns (index, info) where index[basename] = {boxes, native, source}.
    Per-frame label json wins over COCO because it also carries visibility.
    """
    index = {}
    sources = []
    warnings = []
    labels_root = os.path.join(synth_dir, "labels")
    label_dirs = []
    for sub in (modality, resolved_subdir):
        if sub:
            d = os.path.join(labels_root, sub)
            if os.path.isdir(d) and d not in label_dirs:
                label_dirs.append(d)
    if os.path.isdir(labels_root) and labels_root not in label_dirs:
        label_dirs.append(labels_root)
    # COCO first (lower priority for overlapping images)
    coco_names = ["annotations_%s_coco.json" % modality, "annotations_coco.json"]
    for name in coco_names:
        path = os.path.join(synth_dir, name)
        if not os.path.isfile(path):
            continue
        try:
            entries = _extract_coco_entries(_read_json(path))
        except (OSError, ValueError) as exc:
            warnings.append("无法读取 %s: %s" % (name, exc))
            continue
        if entries:
            for key, entry in entries.items():
                entry = dict(entry)
                entry["source"] = name
                index.setdefault(key, entry)
            sources.append(name)
    for directory in label_dirs:
        for name in sorted(os.listdir(directory)):
            if not name.lower().endswith(".json"):
                continue
            path = os.path.join(directory, name)
            try:
                doc = _read_json(path)
            except (OSError, ValueError) as exc:
                warnings.append("无法读取标注 %s: %s" % (os.path.relpath(path, synth_dir), exc))
                continue
            if not isinstance(doc, dict):
                continue
            image_name = doc.get("image")
            stem = os.path.splitext(name)[0]
            key = os.path.basename(str(image_name)) if image_name else stem + ".png"
            native = (doc.get("width"), doc.get("height"))
            if not native or not native[0] or not native[1]:
                native = None
            boxes = _extract_boxes_from_label_doc(doc, native)
            rel = os.path.relpath(path, synth_dir).replace(os.sep, "/")
            index[key] = {"boxes": boxes, "native": native, "source": rel}
            # also allow matching by the label stem if image extensions differ
            if not image_name:
                index.setdefault(stem, {"boxes": boxes, "native": native,
                                        "source": rel})
            sources.append(rel)
    info = {
        "sources": sorted(set(sources)),
        "entries": len(index),
        "warnings": warnings,
    }
    return index, info


def _lookup_label_entry(index, image_path):
    if not index:
        return None
    base = os.path.basename(image_path)
    if base in index:
        return index[base]
    stem = os.path.splitext(base)[0]
    if stem in index:
        return index[stem]
    for key, entry in index.items():
        if os.path.splitext(os.path.basename(key))[0] == stem:
            return entry
    return None


def load_ref_catalog(path):
    """Tolerantly read assets/refs/catalog.json (optional provenance catalogue).

    Only three things matter to this tool: whether an image was accepted, an
    optional manual target box, and the original pixel dimensions that box is
    expressed in.  Unknown keys are ignored; a missing or malformed catalogue is
    not an error.
    """
    result = {"path": path, "exists": os.path.isfile(path) if path else False,
              "entries": {}, "warnings": []}
    if not result["exists"]:
        return result
    try:
        data = _read_json(path)
    except (OSError, ValueError) as exc:
        result["warnings"].append("参考图目录 catalog.json 无法解析，忽略手工框: %s" % exc)
        return result
    categories = data.get("categories") if isinstance(data, dict) else None
    if not isinstance(categories, dict):
        categories = data if isinstance(data, dict) else {}
    for category, node in categories.items():
        if not isinstance(node, dict):
            continue
        for image in node.get("images", []) or []:
            if not isinstance(image, dict):
                continue
            rel = str(image.get("file") or "").replace("\\", "/")
            if not rel:
                continue
            entry = {
                "accepted": image.get("accepted", True),
                "manual_box": image.get("manual_box"),
                "mask_source": image.get("mask_source") or "",
                "width": image.get("width"),
                "height": image.get("height"),
                "category": category,
            }
            result["entries"][rel] = entry
            result["entries"].setdefault(os.path.basename(rel), entry)
    return result


def _catalog_entry(catalog, ref_dir, image_path):
    if not catalog or not catalog.get("entries"):
        return None
    ref_parent = os.path.dirname(os.path.abspath(ref_dir.rstrip("\\/")))
    abs_path = os.path.abspath(image_path)
    try:
        rel = os.path.relpath(abs_path, ref_parent).replace(os.sep, "/")
    except ValueError:
        rel = ""
    entries = catalog["entries"]
    for key in (rel, os.path.basename(abs_path)):
        if key in entries:
            return entries[key]
    return None


def _entry_box(entry):
    if not entry:
        return None
    box = entry.get("manual_box")
    if not box:
        return None
    if isinstance(box, dict):
        box = box.get("bbox") or box.get("box")
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        vals = [float(v) for v in box]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in vals):
        return None
    return vals


# ---------------------------------------------------------------------------
# Reference-parameter file ingestion
# ---------------------------------------------------------------------------

_PARAM_CONTAINER_KEYS = ("classes", "categories", "ref_classes", "datasets", "refs")
_SYNTH_KEYS = ("synth", "synth_dir", "render", "render_dir", "out", "output",
               "dataset", "dataset_dir", "synth_dataset")
_REFS_KEYS = ("refs", "refs_dir", "ref_dir", "reference", "reference_dir",
              "images", "ref_images")


def _as_path(value):
    return value if isinstance(value, str) and value.strip() else None


def _entry_heuristic(entry):
    for key in ("heuristic", "target_region", "target"):
        value = entry.get(key)
        if isinstance(value, dict):
            return dict(value)
        if value is True:
            return {}
    return None


#: Keys that only a class entry carries in the reference-parameter schema
#: (they hold numbers/arrays, unlike the string-only metadata dictionaries the
#: extraction tool also writes at the top level).
_MEASUREMENT_KEYS = (
    "albedo", "albedo_linear", "albedo_p25", "albedo_p75", "albedo_k",
    "albedo_uncertainty", "target_over_sky", "target_median_srgb",
    "texture_hf_ratio", "target_lap_var", "sky_mean", "sky_std", "psd_slope",
    "cam_noise_dn",
)
def _looks_like_class_entry(entry):
    """True when a mapping value is a class entry, not tool metadata.

    configs/refs_params.json currently mixes top-level metadata (strings and
    string maps such as "method" and "reference_materials") with per-class
    measurement blocks.  A class block is identified by a path/override key or
    by a measurement key whose value is numeric or a list -- metadata maps use
    strings only, so they are not mistaken for classes.
    """
    if not isinstance(entry, dict):
        return False
    for key in _SYNTH_KEYS + _REFS_KEYS + ("heuristic", "target_region"):
        if key in entry:
            return True
    for key in _MEASUREMENT_KEYS:
        if isinstance(entry.get(key), (int, float, list, tuple)):
            return True
    return False


def load_ref_params(path):
    """Tolerantly read configs/refs_params.json.

    The reference-library tooling owns this file and its schema may evolve, so
    several spellings are accepted: a mapping under "classes"/"categories"/
    "datasets", a flat mapping of class -> entry, or a list of entries with a
    "name"/"category" field.  Entry values may use synth/refs path aliases, and
    an entry may carry only reference measurements (no paths).  Metadata blocks
    and unknown keys are ignored rather than crashing the comparison.
    """
    result = {"path": path, "exists": os.path.isfile(path) if path else False,
              "classes": {}, "warnings": []}
    if not result["exists"]:
        return result
    try:
        data = _read_json(path)
    except (OSError, ValueError) as exc:
        result["warnings"].append("refs_params.json 无法解析，忽略配对提示: %s" % exc)
        return result
    entries = None
    if isinstance(data, dict):
        for key in _PARAM_CONTAINER_KEYS:
            value = data.get(key)
            if not isinstance(value, dict):
                continue
            explicit = key in ("classes", "categories", "ref_classes", "datasets")
            if explicit or any(_looks_like_class_entry(v) for v in value.values()):
                entries = value
                break
        if entries is None:
            # Flat mapping: keep only values that really look like class
            # entries.  Metadata contains top-level strings and string maps
            # ("source", "method", "uncertainty", "reference_materials").
            entries = {k: v for k, v in data.items()
                       if _looks_like_class_entry(v)}
    elif isinstance(data, list):
        entries = {}
        for item in data:
            if _looks_like_class_entry(item):
                name = item.get("name") or item.get("category") or item.get("class")
                if name:
                    entries[str(name)] = item
    if not isinstance(entries, dict):
        result["warnings"].append("refs_params.json 结构无法识别，忽略配对提示")
        return result
    for name, entry in entries.items():
        if isinstance(entry, str):  # class -> synth directory shorthand
            result["classes"][str(name)] = {"synth": entry, "refs": None,
                                            "heuristic": None, "raw": None}
            continue
        if not isinstance(entry, dict):
            continue
        synth = next((_as_path(entry[k]) for k in _SYNTH_KEYS
                      if _as_path(entry.get(k))), None)
        refs = next((_as_path(entry[k]) for k in _REFS_KEYS
                     if _as_path(entry.get(k))), None)
        result["classes"][str(name)] = {
            "synth": synth,
            "refs": refs,
            "heuristic": _entry_heuristic(entry),
            "raw": entry,
        }
    return result


def _resolve_path(value, config_path=None):
    if not value:
        return None
    if os.path.isabs(value):
        return os.path.normpath(value)
    candidates = [os.path.abspath(value)]
    if config_path:
        base = os.path.dirname(os.path.abspath(config_path))
        candidates.append(os.path.normpath(os.path.join(base, value)))
        candidates.append(os.path.normpath(os.path.join(os.path.dirname(base), value)))
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


# ---------------------------------------------------------------------------
# Side discovery / pairing
# ---------------------------------------------------------------------------

MODALITY_ALIASES = {
    "optical": ("optical", "rgb", "vis", "color"),
    "ir": ("ir_lwir", "ir", "ir_mwir"),
    "sar": ("sar",),
    "lidar": ("lidar",),
}


def synth_image_dir(synth_dir, modality="optical"):
    """Find the image folder for a modality inside a synth output directory."""
    if not synth_dir or not os.path.isdir(synth_dir):
        return None
    images_root = os.path.join(synth_dir, "images")
    aliases = MODALITY_ALIASES.get(modality, (modality,))
    for alias in aliases:
        candidate = os.path.join(images_root, alias)
        if os.path.isdir(candidate) and find_images(candidate, recursive=False):
            return candidate, alias
    if os.path.isdir(images_root):
        direct = find_images(images_root, recursive=False)
        if direct:
            return images_root, "images"
    return None, None


def discover_ref_classes(refs_root):
    """Class directories under assets/refs/ that actually contain images."""
    out = []
    if not refs_root or not os.path.isdir(refs_root):
        return out
    for name in sorted(os.listdir(refs_root)):
        if _is_hidden(name):
            continue
        directory = os.path.join(refs_root, name)
        if not os.path.isdir(directory):
            continue
        if find_images(directory, recursive=True):
            out.append({"name": name, "dir": directory})
    return out


def discover_synth_datasets(synth_root, modality="optical", max_depth=2):
    """Output directories under the synth root that contain a modality folder."""
    out = []
    if not synth_root or not os.path.isdir(synth_root):
        return out
    base_depth = os.path.abspath(synth_root).rstrip("\\/").count(os.sep)
    for dirpath, dirnames, filenames in os.walk(synth_root):
        rel_depth = os.path.abspath(dirpath).count(os.sep) - base_depth
        dirnames[:] = sorted(d for d in dirnames if not _is_hidden(d))
        if dirpath != synth_root:
            image_dir, resolved = synth_image_dir(dirpath, modality)
            if image_dir:
                out.append({
                    "name": os.path.basename(dirpath),
                    "dir": dirpath,
                    "image_dir": image_dir,
                    "resolved": resolved,
                })
                dirnames[:] = []  # do not descend into a dataset
                continue
        if rel_depth >= max_depth:
            dirnames[:] = []
    return sorted(out, key=lambda entry: os.path.normcase(entry["dir"]))


def _norm_name(text):
    text = unicodedata.normalize("NFKC", str(text)).lower()
    return "".join(ch for ch in text if ch.isalnum())


def _abs_key(path):
    return os.path.normcase(os.path.abspath(path)) if path else None


def _match_synth(name, synth_datasets, used):
    normalised = _norm_name(name)
    for dataset in synth_datasets:
        if _abs_key(dataset["dir"]) in used:
            continue
        if _norm_name(dataset["name"]) == normalised:
            return dataset
    for dataset in synth_datasets:
        if _abs_key(dataset["dir"]) in used:
            continue
        other = _norm_name(dataset["name"])
        if normalised and other and (normalised in other or other in normalised):
            return dataset
    return None


def pair_classes(ref_classes, synth_datasets, params, overrides=None,
                 refs_root=None, include_synth_leftovers=True):
    """Build (class, synth dir, ref dir) pairs from params, names or CLI overrides.

    Explicit --pair overrides win, then refs_params.json, then normalized name
    matching, then single-candidate elimination.  Anything left over is returned
    with a reason instead of being silently dropped; synth leftovers can be
    suppressed because the reference library, not the output tree, defines the
    classes worth comparing.
    """
    pairs = []
    unmatched = []
    warnings = []
    used_synth = set()
    used_ref = set()
    paired_names = set()
    params = params or {"classes": {}, "warnings": []}
    params_entries = params.get("classes", {})

    def add_pair(class_name, synth_dir, ref_dir, pairing, heuristic=None):
        pairs.append({
            "class": class_name,
            "synth_dir": synth_dir,
            "ref_dir": ref_dir,
            "pairing": pairing,
            "heuristic": heuristic,
        })
        used_synth.add(_abs_key(synth_dir))
        used_ref.add(_abs_key(ref_dir))
        paired_names.add(_norm_name(class_name))

    # 1) explicit CLI overrides
    for class_name, synth_dir in (overrides or {}).items():
        ref_dir = os.path.join(refs_root, class_name) if refs_root else None
        if (not ref_dir or not os.path.isdir(ref_dir)) and class_name in params_entries:
            ref_dir = _resolve_path(params_entries[class_name].get("refs"),
                                    params.get("path"))
        if not synth_dir or not os.path.isdir(synth_dir):
            unmatched.append({"class": class_name, "side": "synth",
                              "reason": "指定的合成目录不存在: %s" % synth_dir})
            continue
        if not ref_dir or not os.path.isdir(ref_dir):
            unmatched.append({"class": class_name, "side": "ref",
                              "reason": "参考类别目录不存在: %s" % ref_dir})
            continue
        add_pair(class_name, synth_dir, ref_dir, "override")

    # 2) params file pairs (entries may also be reference measurements only)
    for class_name, entry in sorted(params_entries.items()):
        if _norm_name(class_name) in paired_names:
            continue
        synth = _resolve_path(entry.get("synth"), params.get("path"))
        refs = _resolve_path(entry.get("refs"), params.get("path"))
        if refs is None and refs_root:
            refs = os.path.join(refs_root, class_name)
        if not synth:
            dataset = _match_synth(class_name, synth_datasets, used_synth)
            if dataset:
                synth = dataset["dir"]
        if not synth:
            # A params file may hold only reference measurements for a class;
            # the ref-class pass below reports it if it also has no synth pair.
            continue
        if not os.path.isdir(synth):
            unmatched.append({"class": class_name, "side": "synth",
                              "reason": "合成目录不存在: %s" % synth})
            continue
        if not refs or not os.path.isdir(refs):
            unmatched.append({"class": class_name, "side": "ref",
                              "reason": "参考图类别目录不存在: %s" % refs})
            continue
        add_pair(class_name, synth, refs, "params", entry.get("heuristic"))

    # 3) name matching for reference classes
    for ref_class in ref_classes:
        if _norm_name(ref_class["name"]) in paired_names:
            continue
        dataset = _match_synth(ref_class["name"], synth_datasets, used_synth)
        if dataset:
            add_pair(ref_class["name"], dataset["dir"], ref_class["dir"], "name")

    # 4) single-candidate elimination, only when nothing was paired explicitly
    remaining_refs = [c for c in ref_classes if _abs_key(c["dir"]) not in used_ref]
    remaining_synth = [d for d in synth_datasets
                       if _abs_key(d["dir"]) not in used_synth]
    if not pairs and len(remaining_refs) == 1 and len(remaining_synth) == 1:
        add_pair(remaining_refs[0]["name"], remaining_synth[0]["dir"],
                 remaining_refs[0]["dir"], "single")
        warnings.append("按唯一候选配对: %s <-> %s"
                        % (os.path.basename(remaining_synth[0]["dir"]),
                           remaining_refs[0]["name"]))

    for ref_class in ref_classes:
        if _abs_key(ref_class["dir"]) not in used_ref:
            unmatched.append({
                "class": ref_class["name"], "side": "ref",
                "reason": "没有可配对的合成数据集（目录名不匹配，refs_params.json 未指定）",
            })
    if include_synth_leftovers:
        for dataset in synth_datasets:
            if _abs_key(dataset["dir"]) not in used_synth:
                unmatched.append({
                    "class": dataset["name"], "side": "synth",
                    "reason": "没有可配对的参考类别（目录名不匹配，refs_params.json 未指定）",
                })
    return pairs, unmatched, warnings


# ---------------------------------------------------------------------------
# Aggregation and feature evaluation
# ---------------------------------------------------------------------------

def _aggregate_side(image_stats):
    """Median-aggregate per-image statistics for one side."""
    out = {
        "full": {}, "target": {}, "background": {},
        "target_area_frac": float("nan"),
        "sources": [],
        "n": len(image_stats),
    }
    if not image_stats:
        return out
    areas = [s.get("target_area_frac") for s in image_stats]
    out["target_area_frac"] = _safe_median(areas)
    out["sources"] = sorted({s.get("target_source") for s in image_stats
                             if s.get("target_source")})
    for region in ("full", "target", "background"):
        for key in REGION_SCALARS:
            values = [s[region].get(key) for s in image_stats if region in s]
            out[region][key] = _safe_median(values)
        for key in REGION_ARRAYS:
            arrays = []
            for s in image_stats:
                arr = s.get(region, {}).get(key)
                if arr is None:
                    continue
                arr = np.asarray(arr, dtype=np.float64)
                if arr.size and np.isfinite(arr).any():
                    arrays.append(arr)
            if arrays:
                stacked = np.stack(arrays, axis=0)
                med = np.nanmedian(stacked, axis=0)
                if key in ("hist", "hue_hist", "grad_hist", "sat_hist"):
                    med = _normalise_hist(med)
                out[region][key] = med.astype(np.float32)
    return out


def _get_value(agg, spec):
    region = spec.get("region")
    metric = spec["metric"]
    if region is None:
        value = agg.get(metric)
    else:
        value = agg.get(region, {}).get(metric)
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _ratio_within(synth, ref, tol, floor=0.0):
    if synth is None or ref is None:
        return None
    delta = abs(synth - ref)
    if delta <= floor:
        return True
    eps = 1e-9
    if synth <= eps and ref <= eps:
        return True
    if synth <= eps or ref <= eps:
        return False
    ratio = synth / ref
    return (1.0 / tol) <= ratio <= tol


def _evaluate(spec, synth, ref):
    if synth is None or ref is None:
        return None, "n/a"
    kind = spec["kind"]
    delta = synth - ref
    if kind == "abs":
        within = abs(delta) <= spec["tol"]
    elif kind == "ratio":
        within = _ratio_within(synth, ref, spec["tol"], spec.get("floor", 0.0))
    elif kind == "noise":
        floor = spec.get("floor", THRESH_NOISE_ABS_FLOOR)
        if abs(delta) <= floor:
            within = True
        elif synth <= 1e-9 and ref <= 1e-9:
            within = True
        elif synth <= 1e-9 or ref <= 1e-9:
            within = False
        else:
            ratio = synth / ref
            within = (1.0 / spec["tol"]) <= ratio <= spec["tol"]
    else:  # pragma: no cover - registry typo guard
        raise ValueError("unknown feature kind %r" % (kind,))
    if within:
        return within, "一致"
    return within, (spec["neg"] if delta < 0 else spec["pos"])


def _format_number(spec, value):
    if value is None:
        return "n/a"
    scale = float(spec.get("scale", 1.0))
    try:
        text = spec.get("fmt", "%.3f") % (value * scale)
    except (TypeError, ValueError):
        return "n/a"
    return text


def _threshold_text(spec):
    scale = float(spec.get("scale", 1.0))
    if spec.get("gate") is False:
        return "参考"
    if spec["kind"] == "abs":
        fmt = spec.get("fmt", "%.3f").replace("+", "")
        try:
            tol_text = fmt % (spec["tol"] * scale)
        except (TypeError, ValueError):
            tol_text = "%.3f" % (spec["tol"] * scale)
        return "±" + tol_text
    if spec["kind"] in ("ratio", "noise"):
        return "×[%.2f,%.2f]" % (1.0 / spec["tol"], spec["tol"])
    return ""  # pragma: no cover


def _hist_l1(a, b):
    if a is None or b is None:
        return None
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        return None
    return float(0.5 * np.abs(a - b).sum())


def _hue_distance(a, b):
    """Circular L1 distance between two normalised hue histograms."""
    if a is None or b is None:
        return None
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        return None
    best = None
    for shift in range(a.size):
        distance = float(0.5 * np.abs(a - np.roll(b, shift)).sum())
        best = distance if best is None else min(best, distance)
    return best


def build_features(synth_agg, ref_agg, synth_images=(), ref_images=()):
    """Compare aggregated statistics row by row; returns report rows."""
    rows = []
    for spec in FEATURE_SPECS:
        synth_value = _get_value(synth_agg, spec)
        ref_value = _get_value(ref_agg, spec)
        within, verdict = _evaluate(spec, synth_value, ref_value)
        delta = None
        ratio = None
        if synth_value is not None and ref_value is not None:
            delta = synth_value - ref_value
            if abs(ref_value) > 1e-9:
                ratio = synth_value / ref_value
        samples = None
        if spec.get("region") in ("full", "target", "background") and spec["metric"] != "area_frac":
            samples = {
                "synth": [_round_float(image.get(spec["region"], {}).get(spec["metric"]))
                          for image in synth_images],
                "ref": [_round_float(image.get(spec["region"], {}).get(spec["metric"]))
                        for image in ref_images],
            }
        rows.append({
            "key": spec["key"],
            "label": spec["label"],
            "region": spec.get("region") or "meta",
            "metric": spec["metric"],
            "kind": spec["kind"],
            "gate": bool(spec.get("gate", True)),
            "scale": float(spec.get("scale", 1.0)),
            "fmt": spec.get("fmt", "%.3f"),
            "threshold": {
                "kind": spec["kind"],
                "tol": float(spec["tol"]),
                "floor": float(spec.get("floor", 0.0)),
                "text": _threshold_text(spec),
            },
            "synth": _round_float(synth_value),
            "ref": _round_float(ref_value),
            "delta": _round_float(delta),
            "ratio": _round_float(ratio),
            "within_tolerance": within,
            "verdict": verdict,
            "samples": samples,
        })
    return rows


def _round_float(value, digits=6):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return round(value, digits)


def _distribution_block(synth_agg, ref_agg):
    hist_s = synth_agg.get("full", {}).get("hist")
    hist_r = ref_agg.get("full", {}).get("hist")
    hue_s = synth_agg.get("full", {}).get("hue_hist")
    hue_r = ref_agg.get("full", {}).get("hue_hist")
    return {
        "luma_hist_l1": _round_float(_hist_l1(hist_s, hist_r)),
        "luma_hist_l1_tolerance": THRESH_HIST_L1,
        "hue_hist_circular_l1": _round_float(_hue_distance(hue_s, hue_r)),
        "hue_hist_l1_tolerance": THRESH_HUE_L1,
        "note": "分布距离仅作参考，不参与通过/失败判定",
    }


# ---------------------------------------------------------------------------
# Report rendering (CJK-aware column alignment)
# ---------------------------------------------------------------------------

def _display_width(text):
    width = 0
    for ch in str(text):
        width += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
    return width


def _cell(text, width, align="left"):
    text = str(text)
    pad = max(0, width - _display_width(text))
    if align == "right":
        return " " * pad + text
    return text + " " * pad


def _row(cells, widths, aligns):
    return "  ".join(_cell(value, width, align)
                     for value, width, align in zip(cells, widths, aligns))


def _summary_line(report):
    summary = report.get("summary", {})
    if report.get("status") != "compared":
        return "结论: [未比较] " + "；".join(summary.get("reasons") or ["数据不足"])
    fail_count = int(summary.get("failed", 0))
    total = int(summary.get("gated", 0))
    if summary.get("pass"):
        return "结论: [通过] %d/%d 项门控指标在阈值内" % (total, total)
    reasons = "、".join(summary.get("reasons") or ["指标超阈值"])
    return "结论: [未通过] %d/%d 项超阈值 -> %s" % (fail_count, total, reasons)


def format_report(report, show_caveats=True):
    """Render one comparison report as a human-readable table."""
    lines = []
    title = "渲染 vs 实拍参考 对比报告"
    if report.get("class"):
        title += " · " + str(report["class"])
    lines.append("=" * 96)
    lines.append(title)
    lines.append("-" * 96)
    lines.append("合成: %s" % report.get("synth_dir", "?"))
    lines.append("参考: %s" % report.get("ref_dir", "?"))
    synth_info = report.get("images", {}).get("synth", {})
    ref_info = report.get("images", {}).get("ref", {})
    lines.append("图像: 合成 %d 张%s | 参考 %d 张%s | 跳过 %d 个文件"
                 % (synth_info.get("count", 0), _size_text(synth_info),
                    ref_info.get("count", 0), _size_text(ref_info),
                    len(report.get("skipped", []))))
    target = report.get("target_regions", {})
    lines.append("目标区: 合成=%s | 参考=%s"
                 % (target.get("synth", {}).get("text", "无"),
                    target.get("ref", {}).get("text", "无")))
    lines.append("分析画布: 长边 %s px（bilinear 下采样，小图不放大）"
                 % report.get("canvas", {}).get("max_side", ANALYSIS_MAX_SIDE))
    lines.append("")

    features = report.get("features") or []
    if features:
        header = ["特征", "区域", "合成", "参考", "差值", "阈值", "判读"]
        widths = [24, 6, 10, 10, 10, 14, 16]
        aligns = ["left", "left", "right", "right", "right", "left", "left"]
        lines.append(_row(header, widths, aligns))
        lines.append("-" * 96)
        for row in features:
            if row["synth"] is not None and row["ref"] is not None:
                delta = _format_number(row, row["delta"])
            else:
                delta = "n/a"
            verdict = row["verdict"]
            if row["gate"] and row["within_tolerance"] is False:
                verdict += " *"
            lines.append(_row([
                row["label"],
                REGION_CN.get(row["region"], row["region"]),
                _format_number(row, row["synth"]),
                _format_number(row, row["ref"]),
                delta,
                row["threshold"]["text"],
                verdict,
            ], widths, aligns))
        lines.append("-" * 96)
        lines.append("* = 超出阈值（参与通过/失败判定）；其余行仅参考")
    else:
        lines.append("（没有可对比的特征：一侧没有可用图像）")
    distributions = report.get("distributions") or {}
    if distributions.get("luma_hist_l1") is not None:
        hue_distance = distributions.get("hue_hist_circular_l1")
        hue_text = ("%.3f" % hue_distance
                    if hue_distance is not None else "n/a")
        lines.append("分布参考: 亮度直方图 L1=%.3f（阈值 %.2f）| 色相分布 L1=%s（阈值 %.2f）"
                     % (distributions["luma_hist_l1"],
                        distributions.get("luma_hist_l1_tolerance", THRESH_HIST_L1),
                        hue_text,
                        distributions.get("hue_hist_l1_tolerance", THRESH_HUE_L1)))
    for item in (report.get("skipped") or [])[:3]:
        lines.append("跳过: %s (%s)"
                     % (os.path.basename(item.get("path", "?")), item.get("reason", "")))
    if len(report.get("skipped") or []) > 3:
        lines.append("跳过: ... 另有 %d 个文件见 JSON" % (len(report["skipped"]) - 3))
    lines.append("")
    lines.append(_summary_line(report))
    warnings = report.get("warnings") or []
    for warning in warnings[:8]:
        lines.append("!! 警告: %s" % warning)
    if len(warnings) > 8:
        lines.append("!! 另有 %d 条警告见 JSON" % (len(warnings) - 8))
    if show_caveats:
        for caveat in report.get("caveats") or METHOD_CAVEATS:
            lines.append("· " + caveat)
    return "\n".join(lines)


def _size_text(side_info):
    sizes = side_info.get("native_sizes") or []
    bit_depths = side_info.get("bit_depths") or []
    bit_text = ""
    if bit_depths:
        bit_text = ", %s-bit" % "/".join(str(int(v)) for v in bit_depths)
    if not sizes:
        return ""
    unique = sorted({tuple(size) for size in sizes})
    if len(unique) == 1:
        return " (%dx%d%s)" % (unique[0][0], unique[0][1], bit_text)
    return " (%d 种尺寸%s)" % (len(unique), bit_text)


def format_all_report(multi):
    """Render a multi-class comparison summary."""
    lines = []
    lines.append("=" * 96)
    lines.append("批量对比: %d 类 · 通过 %d / 未通过 %d"
                 % (len(multi.get("classes", [])),
                    multi.get("summary", {}).get("passed", 0),
                    multi.get("summary", {}).get("failed", 0)))
    lines.append("参考根目录: %s | 合成根目录: %s"
                 % (multi.get("refs_root", "?"), multi.get("synth_root", "?")))
    params = multi.get("params", {})
    if params.get("path"):
        lines.append("参数文件: %s（%s）"
                     % (params.get("path"),
                        "已读取 %d 类" % params.get("classes", 0)
                        if params.get("exists") else "不存在，使用回退配对"))
    lines.append("")
    for report in multi.get("classes", []):
        lines.append(format_report(report, show_caveats=False))
        lines.append("")
    unmatched = multi.get("unmatched") or []
    if unmatched:
        ref_left = [item for item in unmatched if item.get("side") == "ref"]
        synth_left = [item for item in unmatched if item.get("side") != "ref"]
        lines.append("未配对:")
        for item in ref_left:
            lines.append("  - [参考] %s: %s"
                         % (item.get("class", "?"), item.get("reason", "")))
        if len(synth_left) <= 5:
            for item in synth_left:
                lines.append("  - [合成] %s: %s"
                             % (item.get("class", "?"), item.get("reason", "")))
        elif synth_left:
            lines.append("  - [合成] 另有 %d 个输出目录没有参考类别（如 %s；详见 JSON）"
                         % (len(synth_left),
                            "、".join(item.get("class", "?") for item in synth_left[:3])))
        lines.append("")
    for warning in multi.get("warnings") or []:
        lines.append("!! 警告: %s" % warning)
    summary = multi.get("summary", {})
    lines.append("总计: %d 类，通过 %d，未通过 %d%s"
                 % (len(multi.get("classes", [])), summary.get("passed", 0),
                    summary.get("failed", 0),
                    "" if summary.get("pass") else " -> 退出码 1"))
    for caveat in multi.get("caveats") or METHOD_CAVEATS:
        lines.append("· " + caveat)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _json_safe(value):
    """Convert numpy/NaN values into strict JSON (null for non-finite floats)."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is None or isinstance(value, (str, int, float)):
        return value
    return str(value)


def write_report_json(path, payload):
    """Write a report without NaN/Infinity, creating parent directories."""
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_json_safe(payload), fh, ensure_ascii=False, indent=1,
                  allow_nan=False)


def _now():
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Comparison drivers
# ---------------------------------------------------------------------------

def _load_side(files, detect_target, max_side, max_images, extra):
    """Load and analyse a list of files, collecting skips and notes."""
    stats = []
    skipped = []
    notes = []
    selected = files[:int(max_images)] if max_images else files
    if len(selected) < len(files):
        notes.append("只分析了前 %d 张（共 %d 张），可用 max_images 调整"
                     % (len(selected), len(files)))
    for path in selected:
        try:
            record = load_image(path, max_side=max_side)
        except ImageLoadError as exc:
            skipped.append({"path": path, "reason": str(exc)})
            continue
        try:
            mask, source, warnings = detect_target(path, record, extra)
            image_stats = analyze_image(record["rgb"], mask, source, path=path)
        except Exception as exc:  # one bad annotation must not kill the batch
            skipped.append({"path": path, "reason": "分析失败: %s" % exc})
            continue
        image_stats.update({
            "native_size": record["native_size"],
            "analysis_size": record["analysis_size"],
            "bit_depth": record.get("bit_depth", 8),
            "notes": list(record["notes"]) + list(warnings),
        })
        notes.extend(record["notes"])
        notes.extend(warnings)
        stats.append(image_stats)
    return {
        "stats": stats,
        "skipped": skipped,
        "notes": _dedupe(notes),
        "count": len(stats),
        "native_sizes": [s["native_size"] for s in stats],
        "analysis_sizes": [s["analysis_size"] for s in stats],
    }


def _dedupe(items):
    seen = set()
    out = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _synth_detector(synth_dir, modality, image_dir, resolved):
    index, info = build_synth_label_index(synth_dir, modality, [], resolved)
    state = {"used_labels": 0, "fallback": 0, "warnings": list(info["warnings"])}
    method = "annotation" if index else "heuristic"
    text = None
    if info["sources"]:
        display = sorted({os.path.dirname(source) for source in info["sources"]
                          if os.path.dirname(source)})
        if any(not os.path.dirname(source) for source in info["sources"]):
            display.append("COCO")
        text = "标注（%s）" % "、".join(display)

    def detect(path, record, extra):
        entry = _lookup_label_entry(index, path)
        native = entry.get("native") if entry else None
        boxes = entry.get("boxes") if entry else None
        if boxes:
            mask = _boxes_mask(boxes, native or record["native_size"],
                               record["rgb"].shape)
            if mask.sum() >= MIN_REGION_PIXELS:
                state["used_labels"] += 1
                return mask, entry.get("source", "labels"), []
        state["fallback"] += 1
        params = extra or {}
        mask, heuristic_info = heuristic_target_mask(record["rgb"], **params)
        # One aggregate warning is added by compare(); do not emit one per frame
        # (a whole unlabelled run would otherwise flood the report).
        return mask, "heuristic_fallback", []

    return detect, state, method, text, info


def _ref_detector(ref_dir, catalog, heuristic_params):
    state = {"heuristic": 0, "manual": 0, "fallbacks": 0,
             "warnings": [], "catalog_entries": 0}
    entries = (catalog or {}).get("entries", {})
    if entries:
        state["catalog_entries"] = len(entries)

    def detect(path, record, extra):
        entry = _catalog_entry(catalog, ref_dir, path)
        if entry and entry.get("accepted") is False:
            raise ImageLoadError("catalog 标记为 rejected: %s" % os.path.basename(path))
        box = _entry_box(entry)
        if box is not None:
            native = None
            if entry.get("width") and entry.get("height"):
                native = (entry["width"], entry["height"])
            mask = _manual_box_mask(box, native or record["native_size"],
                                    record["rgb"].shape)
            if mask.sum() >= MIN_REGION_PIXELS:
                state["manual"] += 1
                return mask, "catalog_manual_box", []
            state["warnings"].append(
                "参考图 %s 的 manual_box 无效或过小，改用启发式"
                % os.path.basename(path))
        params = extra or heuristic_params or {}
        mask, info = heuristic_target_mask(record["rgb"], **params)
        state["heuristic"] += 1
        if info.get("fallback"):
            state["fallbacks"] += 1
        return mask, "heuristic", []

    return detect, state


def compare(synth_dir, ref_dir, out_json=None, modality="optical",
            max_side=ANALYSIS_MAX_SIDE, max_images=MAX_IMAGES_PER_SIDE,
            params=None, ref_catalog=None, class_name=None, print_table=True,
            heuristic_params=None, catalog_path=None):
    """Compare rendered frames in *synth_dir* against photos in *ref_dir*.

    Parameters
    ----------
    synth_dir : output directory with images/<modality>/ and annotations.
    ref_dir : directory of reference photos (assets/refs/<class>).
    out_json : optional path; the machine-readable report is written here.
    params : optional dict from load_ref_params()['classes'][class] with an
        optional "heuristic" override for reference target extraction.
    ref_catalog : optional dict from load_ref_catalog(); defaults to
        <ref_dir>/../catalog.json when present.
    class_name : label used in the report and pairing.
    print_table : print the human-readable table (default True, as the CLI
        contract expects).

    Missing directories, missing labels, absent catalogues and corrupt images
    degrade to warnings/status values; this function does not raise for them.
    """
    synth_dir = os.path.normpath(str(synth_dir or ""))
    ref_dir = os.path.normpath(str(ref_dir or ""))
    if class_name is None:
        class_name = os.path.basename(ref_dir.rstrip("\\/")) or None
    warnings = []
    if str(modality).lower() != "optical":
        warnings.append("模态 %s 与彩色实拍参考图的可比性有限："
                        "色彩/亮度差异包含模态本身的影响" % modality)
    params = params or {}
    heuristic = dict(heuristic_params or {})
    if isinstance(params.get("heuristic"), dict):
        heuristic.update(params["heuristic"])
    heuristic = {k: v for k, v in heuristic.items()
                 if k in ("center_frac", "sky_frac", "percentile", "min_score")}

    if ref_catalog is None:
        candidate = os.path.join(os.path.dirname(ref_dir.rstrip("\\/")), "catalog.json")
        if catalog_path:
            candidate = catalog_path
        ref_catalog = load_ref_catalog(candidate)
    warnings.extend(ref_catalog.get("warnings", []))

    report = {
        "tool": "refcompare",
        "version": MODULE_VERSION,
        "mode": "single",
        "generated_at": _now(),
        "class": class_name,
        "synth_dir": synth_dir,
        "ref_dir": ref_dir,
        "modality": modality,
        "canvas": {
            "max_side": int(max_side),
            "resampler": "bilinear",
            "upscale": False,
        },
        "thresholds": dict(THRESHOLDS),
        "status": "empty",
        "images": {"synth": {}, "ref": {}},
        "target_regions": {},
        "features": [],
        "distributions": {},
        "summary": {},
        "warnings": warnings,
        "skipped": [],
        "caveats": list(METHOD_CAVEATS),
    }

    # -- collect image files -------------------------------------------------
    synth_image_path = None
    resolved = None
    synth_files = []
    if not os.path.isdir(synth_dir):
        warnings.append("合成目录不存在: %s" % synth_dir)
    else:
        synth_image_path, resolved = synth_image_dir(synth_dir, modality)
        if synth_image_path:
            synth_files = find_images(synth_image_path, recursive=False)
        else:
            warnings.append("合成目录中没有 images/%s 图像: %s" % (modality, synth_dir))
    ref_files = find_images(ref_dir, recursive=True) if os.path.isdir(ref_dir) else []
    if not os.path.isdir(ref_dir):
        warnings.append("参考目录不存在: %s" % ref_dir)
    if not synth_files:
        report["status"] = "no_synth"
        report["summary"] = {"pass": False, "failed": 0, "gated": 0,
                             "reasons": ["没有可用的合成图像"]}
        return _finish_report(report, out_json, print_table)
    if not ref_files:
        report["status"] = "no_reference"
        report["summary"] = {"pass": False, "failed": 0, "gated": 0,
                             "reasons": ["没有可用的参考图像"]}
        _fill_synth_only(report, synth_dir, modality, synth_image_path, resolved,
                         synth_files, max_side, max_images, heuristic)
        return _finish_report(report, out_json, print_table)

    # -- analyse both sides --------------------------------------------------
    detect_synth, synth_state, _synth_method, synth_text, label_info = _synth_detector(
        synth_dir, modality, synth_image_path, resolved)
    warnings.extend(synth_state.get("warnings", []))
    warnings.extend(label_info.get("warnings", []))
    synth_side = _load_side(synth_files, detect_synth, max_side, max_images, heuristic)

    detect_ref, ref_state = _ref_detector(ref_dir, ref_catalog, heuristic)
    ref_side = _load_side(ref_files, detect_ref, max_side, max_images, heuristic)
    warnings.extend(ref_state.get("warnings", []))

    report["images"] = {
        "synth": _side_summary(synth_side),
        "ref": _side_summary(ref_side),
    }
    report["skipped"] = synth_side["skipped"] + ref_side["skipped"]
    if synth_state["fallback"]:
        warnings.append("%d 张合成图缺少可用标注，目标区使用启发式回退"
                        % synth_state["fallback"])
    if ref_state["manual"]:
        ref_method_text = "启发式 + %d 张手工框" % ref_state["manual"]
    else:
        ref_method_text = "启发式（无标注）"
    synth_method_text = synth_text or "启发式回退（未找到标注）"
    if synth_state["used_labels"] == 0 and synth_files:
        synth_method_text = "启发式回退（标注未覆盖所有帧）"
    if synth_state["used_labels"] and synth_state["fallback"]:
        synth_method = "annotation+fallback"
    elif synth_state["used_labels"]:
        synth_method = "annotation"
    else:
        synth_method = "heuristic_fallback"
    ref_method = "heuristic+manual_box" if ref_state["manual"] else "heuristic"
    hparams = _validate_heuristic_params(
        heuristic.get("center_frac", DEFAULT_CENTER_FRAC),
        heuristic.get("sky_frac", DEFAULT_SKY_FRAC),
        heuristic.get("percentile", DEFAULT_SCORE_PERCENTILE),
        heuristic.get("min_score", DEFAULT_MIN_SCORE))
    report["target_regions"] = {
        "synth": {
            "method": synth_method,
            "text": synth_method_text,
            "images_with_annotation": synth_state["used_labels"],
            "images_with_fallback": synth_state["fallback"],
            "sources": label_info.get("sources", []),
        },
        "ref": {
            "method": ref_method,
            "text": ref_method_text,
            "images_with_heuristic": ref_state["heuristic"],
            "images_with_manual_box": ref_state["manual"],
            "images_with_heuristic_fallback": ref_state["fallbacks"],
            "catalog_entries": ref_state["catalog_entries"],
            "heuristic_params": {
                "center_frac": hparams[0],
                "sky_frac": hparams[1],
                "percentile": hparams[2],
                "min_score": hparams[3],
            },
            "note": "参考图没有标注，目标区由中心先验 + 天空色模型差异启发式分割",
        },
    }
    if ref_state["manual"]:
        warnings.append("参考图 %d 张使用了 catalog.json 的 manual_box，其余为启发式"
                        % ref_state["manual"])
    if ref_state["fallbacks"]:
        warnings.append("参考图 %d 张的启发式分割退化为中心框，目标区指标可信度下降"
                        % ref_state["fallbacks"])

    if not synth_side["stats"] or not ref_side["stats"]:
        report["status"] = "no_synth" if not synth_side["stats"] else "no_reference"
        report["summary"] = {"pass": False, "failed": 0, "gated": 0,
                             "reasons": ["一侧没有可分析的图像（全部损坏或跳过）"]}
        return _finish_report(report, out_json, print_table)

    synth_agg = _aggregate_side(synth_side["stats"])
    ref_agg = _aggregate_side(ref_side["stats"])
    report["aggregates"] = {
        "synth": _aggregate_json(synth_agg),
        "ref": _aggregate_json(ref_agg),
    }
    report["features"] = build_features(synth_agg, ref_agg,
                                        synth_side["stats"], ref_side["stats"])
    report["distributions"] = _distribution_block(synth_agg, ref_agg)
    report["status"] = "compared"
    if isinstance(params, dict) and params.get("raw"):
        report["ref_params"] = _json_safe(params["raw"])
        report["ref_params_note"] = (
            "configs/refs_params.json 中该类的参考量测；"
            "分辨率和单位与本站统一画布指标不同，仅作来源追溯")

    # -- target-area sanity warning -----------------------------------------
    if synth_agg["target_area_frac"] and ref_agg["target_area_frac"]:
        ratio = synth_agg["target_area_frac"] / max(ref_agg["target_area_frac"], 1e-9)
        if ratio > THRESH_TARGET_AREA_RATIO or ratio < 1.0 / THRESH_TARGET_AREA_RATIO:
            warnings.append(
                "目标区面积占比差异较大（合成 %.3f vs 参考 %.3f）：目标区指标需谨慎解读"
                % (synth_agg["target_area_frac"], ref_agg["target_area_frac"]))

    summary = _summarize(report["features"])
    report["summary"] = summary
    if synth_side["notes"] or ref_side["notes"]:
        report["notes"] = _dedupe(synth_side["notes"] + ref_side["notes"])
    report["warnings"] = _dedupe(warnings)
    return _finish_report(report, out_json, print_table)


def _summarize(features):
    failed = [row for row in features if row["gate"] and row["within_tolerance"] is False]
    unknown = [row for row in features if row["gate"] and row["within_tolerance"] is None]
    reasons = []
    for row in failed:
        if row["verdict"] not in reasons:
            reasons.append(row["verdict"])
    if unknown:
        reasons.append("%d 项门控指标无法计算" % len(unknown))
    gated = [row for row in features if row["gate"]]
    return {
        "pass": not failed and not unknown,
        "failed": len(failed),
        "gated": len(gated),
        "unknown": len(unknown),
        "reasons": reasons,
        "failed_keys": [row["key"] for row in failed],
    }


def _side_summary(side):
    return {
        "count": side["count"],
        "native_sizes": [list(size) for size in side["native_sizes"]],
        "analysis_sizes": [list(size) for size in side["analysis_sizes"]],
        "bit_depths": sorted({int(s.get("bit_depth", 8)) for s in side["stats"]}),
        "files": [os.path.basename(s["path"]) for s in side["stats"]],
        "skipped": side["skipped"],
    }


def _aggregate_json(agg):
    payload = {
        "n": agg.get("n", 0),
        "target_area_frac": _round_float(agg.get("target_area_frac")),
        "target_sources": agg.get("sources", []),
        "full": {key: _json_safe(agg["full"].get(key)) for key in REGION_SCALARS},
        "target": {key: _json_safe(agg["target"].get(key)) for key in REGION_SCALARS},
        "background": {key: _json_safe(agg["background"].get(key))
                       for key in REGION_SCALARS},
    }
    return payload


def _fill_synth_only(report, synth_dir, modality, image_path, resolved,
                     synth_files, max_side, max_images, heuristic):
    """When there is no reference side, still record what the render looks like."""
    detect_synth, state, method, text, info = _synth_detector(
        synth_dir, modality, image_path, resolved)
    side = _load_side(synth_files, detect_synth, max_side, max_images, heuristic)
    report["images"]["synth"] = _side_summary(side)
    report["skipped"] = side["skipped"]
    report["target_regions"] = {"synth": {"method": method, "text": text or "启发式回退"}}
    agg = _aggregate_side(side["stats"])
    report["synth_only"] = _aggregate_json(agg)
    report["warnings"] = _dedupe(report["warnings"] + state.get("warnings", []))


def _finish_report(report, out_json, print_table):
    clean = _json_safe(report)
    if out_json:
        write_report_json(out_json, clean)
    if print_table:
        print(format_report(clean))
    return clean


def compare_all(refs_root=None, synth_root=None, params_path=None, out_json=None,
                modality="optical", max_side=ANALYSIS_MAX_SIDE,
                max_images=MAX_IMAGES_PER_SIDE, print_table=True,
                overrides=None, catalog_path=None):
    """Compare every pair of reference class and rendered dataset.

    Pairing order: explicit overrides, refs_params.json, normalized name match,
    then single-candidate elimination.  Classes that cannot be paired are listed
    with a reason; a missing reference root or params file is reported, never
    raised.
    """
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    refs_root = refs_root or os.path.join(here, "assets", "refs")
    synth_root = synth_root or os.path.join(here, "out")
    params_path = params_path or os.path.join(here, "configs", "refs_params.json")

    ref_classes = discover_ref_classes(refs_root)
    synth_datasets = discover_synth_datasets(synth_root, modality)
    params = load_ref_params(params_path)
    warnings = list(params.get("warnings", []))
    if not os.path.isdir(refs_root):
        warnings.append("参考图根目录不存在: %s" % refs_root)
    if not os.path.isdir(synth_root):
        warnings.append("合成输出根目录不存在: %s" % synth_root)
    if not params.get("exists"):
        warnings.append("未找到 refs_params.json（%s），按目录名/单候选自动配对"
                        % params_path)

    pairs, unmatched, pair_warnings = pair_classes(
        ref_classes, synth_datasets, params, overrides=overrides,
        refs_root=refs_root, include_synth_leftovers=not overrides)
    warnings.extend(pair_warnings)
    if not pairs and ref_classes:
        warnings.append(
            "没有自动配对的类别；请用 --pair 类别=合成目录，或在 "
            "refs_params.json 的 classes 中给出 synth 映射")
    reports = []
    for pair in pairs:
        entry = params.get("classes", {}).get(pair["class"])
        report = compare(
            pair["synth_dir"], pair["ref_dir"], out_json=None,
            modality=modality, max_side=max_side, max_images=max_images,
            params=entry, catalog_path=catalog_path, class_name=pair["class"],
            print_table=False)
        report["pairing"] = pair["pairing"]
        reports.append(report)

    passed = sum(1 for report in reports if report.get("summary", {}).get("pass"))
    failed = len(reports) - passed
    multi = {
        "tool": "refcompare",
        "version": MODULE_VERSION,
        "generated_at": _now(),
        "mode": "all",
        "refs_root": refs_root,
        "synth_root": synth_root,
        "params": {
            "path": params_path,
            "exists": bool(params.get("exists")),
            "classes": len(params.get("classes", {})),
            "class_names": sorted(params.get("classes", {})),
        },
        "classes": reports,
        "unmatched": unmatched,
        "warnings": _dedupe(warnings),
        "summary": {
            "pass": bool(reports) and failed == 0,
            "passed": passed,
            "failed": failed,
            "classes": len(reports),
            "reasons": ([] if reports and failed == 0 else
                        ["没有全部通过的对比"] if reports else
                        ["没有可配对的类别"]),
        },
        "thresholds": dict(THRESHOLDS),
        "caveats": list(METHOD_CAVEATS),
    }
    if out_json:
        write_report_json(out_json, multi)
    if print_table:
        print(format_all_report(multi))
    return multi


# ---------------------------------------------------------------------------
# Small self-check used by the CLI for a fast sanity read
# ---------------------------------------------------------------------------

def describe_feature_set():
    """Return the feature keys and gated flags (used by --list-features)."""
    return [{"key": spec["key"], "label": spec["label"],
             "gate": bool(spec.get("gate", True))} for spec in FEATURE_SPECS]
