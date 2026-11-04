"""Dataset post-processing: triptych previews, a contact sheet, a README.

A run that can only be reviewed by opening twenty folders is a run nobody
reviews.  This is the last mile of simkit.dataset.run: it joins the sensor
previews of one frame side by side, lays the frames out on a contact sheet and
writes the parameter sheet a reader needs before touching the data -- what was
simulated, with which sensors, under which jamming, and what the limits are.

Only numpy + Pillow + the standard library are used: exporting must not add an
install requirement, and it has to work on a machine that only received the
dataset directory.

Robustness contract:

* a missing or empty modality directory is skipped and named on the returned
  object, never an exception for the whole batch;
* 16-bit IR frames are percentile-stretched for display, preferring the 8-bit
  ir_preview_* render when the dataset has one;
* every image is scaled by height only, so no modality is stretched.

The README is written for a human reader: it is a finished document, not a
skeleton, so every section falls back to a measured statement when an optional
input file is absent.
"""

from __future__ import annotations

import datetime
import glob
import json
import math
import os
import re
from collections import Counter

import numpy as np
from PIL import Image, ImageDraw, ImageFont

__all__ = ["triptych", "contact_sheet", "write_readme", "Generated", "ContactSheet"]

# Markdown inline code and fences, built without a literal backtick so this
# module stays easy to embed in generated code and docs.
_TICK = chr(96)


def _code(text):
    return "%s%s%s" % (_TICK, text, _TICK)


_FENCE = _TICK * 3

# Separator between two modality cells, in pixels and in colour.
SEP = 6
_SEP_RGB = (0x20, 0x20, 0x20)
# Cell background: dark enough that a letterboxed radar strip still reads as
# data, not as a hole in the image.
_CELL_RGB = (0x10, 0x10, 0x10)
# Label bar height as a fraction of the cell height (22 px on a 480 px cell).
_LABEL_RATIO = 22.0 / 480.0
_RESAMPLE = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.BICUBIC)

_FRAME_RE = re.compile(r"(\d+)$")

# Chinese display names.  Keys are the directory/modality names simkit writes,
# including the per-view tags (optical_wide, ir_mwir_wide, ...).
_CN = {
    "optical": "可见光",
    "ir": "红外",
    "ir_mwir": "中波红外",
    "ir_lwir": "长波红外",
    "ir_swir": "短波红外",
    "sar": "SAR 强度",
    "radar": "雷达 距离-多普勒",
    "radar_rd": "雷达 距离-多普勒",
    "radar_ra": "雷达 距离-角度",
    "lidar": "激光雷达距离",
    "lidar_range": "激光雷达距离",
    "lidar_intensity": "激光雷达强度",
}

# Directory layouts simkit has shipped: images/<mod>, lidar/<sub>, and the
# legacy "lidar_range" name dataset._preview_modalities still hands us.
_ALIAS = {
    "radar_rd": "images/radar",
    "lidar": "lidar/range",
    "lidar_range": "lidar/range",
    "lidar_intensity": "lidar/intensity",
}

_ORDER = ("optical", "ir_mwir", "ir_lwir", "ir_swir", "ir", "radar", "radar_ra",
          "sar", "lidar_range", "lidar_intensity", "lidar")

# A run that asked for optical + ir + radar writes five image directories
# (ir_mwir, ir_lwir, radar, radar_ra, ...).  Those are products of three
# sensors, and the README must not tell the reader they picked five modalities.
_FAMILY_CN = {"optical": "可见光", "ir": "红外", "radar": "雷达", "sar": "SAR",
              "lidar": "激光雷达"}


def _family(name):
    """Sensor family of one image directory."""
    name = str(name)
    if name.startswith("optical"):
        return "optical"
    if name.startswith("ir"):
        return "ir"
    if name.startswith("radar"):
        return "radar"
    if name == "sar":
        return "sar"
    if name.startswith("lidar"):
        return "lidar"
    return name

# A CJK-capable face is required for the Chinese labels; Pillow's bundled
# default font has no CJK glyphs.  First hit wins, and if none exists the label
# degrades to ASCII instead of drawing tofu boxes.
_FONT_CANDIDATES = (
    os.environ.get("SIMKIT_FONT", ""),
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/msyhbd.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/simsun.ttc",
    "C:/Windows/Fonts/Deng.ttf",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
)


# --------------------------------------------------------------------- results

class Generated(list):
    """list[str] of written previews plus what was skipped.

    dataset.run only needs len() and iteration, so the return value stays a
    real list.  skipped and frames are how a caller (or a human reading a log)
    can tell that, say, SAR was simply absent from this run instead of
    guessing from a frame count.
    """

    def __init__(self, paths=(), skipped=(), frames=0):
        super().__init__(paths)
        self.skipped = list(skipped)
        self.frames = int(frames)

    @property
    def note(self):
        """One-line summary of the skipped modalities (empty when none)."""
        if not self.skipped:
            return ""
        return "已跳过：" + "、".join(self.skipped)


class ContactSheet(str):
    """Path of the contact sheet, plus the grid it actually used."""

    def __new__(cls, path, *, cells=0, cols=1, rows=1, files=()):
        obj = str.__new__(cls, path)
        obj.cells = int(cells)
        obj.cols = int(cols)
        obj.rows = int(rows)
        obj.files = list(files)
        return obj


# ------------------------------------------------------------------- utilities

def _read_json(path):
    """Load a JSON file, returning None instead of raising."""
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _frames_in(directory):
    """Every image in directory as (frame_id, path), sorted by frame.

    Frame ids come from the trailing digits of the file name, which is what
    simkit writes (000123.png).  A name without digits falls back to its
    position, so a hand-assembled folder still produces a preview.
    """
    if not os.path.isdir(directory):
        return []
    found = []
    for name in os.listdir(directory):
        base, ext = os.path.splitext(name)
        if ext.lower() not in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
            continue
        m = _FRAME_RE.search(base)
        found.append((int(m.group(1)) if m else None, name, os.path.join(directory, name)))
    found.sort(key=lambda t: (t[0] is None, t[0] if t[0] is not None else 0, t[1]))
    return [(fid if fid is not None else i, path)
            for i, (fid, _name, path) in enumerate(found)]


def _first_dir(candidates):
    for path in candidates:
        if path and os.path.isdir(path):
            return path
    return None


def _glob_dirs(out_dir, *parts):
    pattern = os.path.join(out_dir, *parts)
    return sorted(p for p in glob.glob(pattern) if os.path.isdir(p))


def _resolve_modality(out_dir, modality):
    """Locate one modality's image directory, or return None.

    Handles the three layouts in the wild: images/optical, lidar/range
    (dataset._preview_modalities calls it lidar_range) and the IR pair where
    images/ir_mwir holds 16-bit counts and images/ir_preview_mwir the 8-bit
    render.
    """
    m = str(modality).strip().strip("/\\").replace("\\", "/")
    candidates = []
    if "/" in m:
        candidates.append(os.path.join(out_dir, m))
    else:
        if m.startswith("ir") and m not in ("ir",) and not m.startswith("ir_preview"):
            # ir_mwir -> images/ir_preview_mwir (8-bit is what a preview wants)
            candidates.append(os.path.join(out_dir, "images", "ir_preview" + m[2:]))
        if m.startswith("lidar_"):
            candidates.append(os.path.join(out_dir, "lidar", m.split("_", 1)[1]))
        alias = _ALIAS.get(m)
        if alias:
            candidates.append(os.path.join(out_dir, alias))
        candidates.append(os.path.join(out_dir, "images", m))
        candidates.append(os.path.join(out_dir, m))
        if m == "ir":
            candidates.extend(_glob_dirs(out_dir, "images", "ir_preview_*"))
        candidates.extend(_glob_dirs(out_dir, "images", "ir_preview_" + m))
    seen = set()
    ordered = []
    for path in candidates:
        key = os.path.normcase(os.path.abspath(path))
        if key in seen:
            continue
        seen.add(key)
        ordered.append(path)
    return _first_dir(ordered)


def _modality_label(modality):
    """ir_preview_mwir_wide -> 中波红外 · 宽视场."""
    m = str(modality).replace("\\", "/").strip("/")
    wide = m.endswith("_wide")
    if wide:
        m = m[:-len("_wide")]
    if m.startswith("ir_preview_"):
        m = "ir" + m[len("ir_preview"):]
    elif m.startswith("ir_preview"):
        m = "ir"
    label = _CN.get(m)
    if label is None:
        label = m.replace("_", " ")
    return label + (" · 宽视场" if wide else "")


def _modality_ascii(modality):
    """ASCII fallback used only when no CJK font is available."""
    return str(modality).replace("\\", "/").strip("/")


def _stretch_to_u8(array):
    """Percentile-stretch a 16-bit/f32 frame into 8-bit display values.

    A raw IR count image has a long tail of hot pixels; a plain min-max
    stretch renders the aircraft as a dot.  0.5/99.5 percentiles keep the
    target's internal structure visible.
    """
    a = np.asarray(array, dtype=np.float32)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return np.zeros(a.shape, np.uint8)
    lo, hi = np.percentile(finite, (0.5, 99.5))
    if not (math.isfinite(float(lo)) and math.isfinite(float(hi))) or hi <= lo:
        lo = float(finite.min())
        hi = float(finite.max())
    if hi <= lo:
        return np.zeros(a.shape, np.uint8)
    out = (a - float(lo)) * (255.0 / (float(hi) - float(lo)))
    return np.clip(out, 0, 255).astype(np.uint8)


def _open_rgb(path):
    """Open one frame as (RGB image, is_grayscale), stretching 16-bit data."""
    with Image.open(path) as im:
        im.load()
        gray = im.mode not in ("RGB", "RGBA")
        if im.mode in ("RGB", "RGBA"):
            return im.convert("RGB"), False
        if im.mode.startswith("I;16") or im.mode in ("I", "F"):
            return Image.fromarray(_stretch_to_u8(np.asarray(im)), "L").convert("RGB"), True
        return im.convert("RGB"), gray


def _to_rgb(path):
    """Open one frame as RGB, stretching 16-bit data for display."""
    return _open_rgb(path)[0]


def _font(size):
    """Return (font, has_cjk); (None, False) when nothing loads.

    Fonts are the one thing that may genuinely be absent on a build machine,
    so the caller treats None as "write no text" rather than failing.
    """
    for candidate in _FONT_CANDIDATES:
        if not candidate or not os.path.isfile(candidate):
            continue
        try:
            return ImageFont.truetype(candidate, int(size)), True
        except (OSError, ValueError):
            continue
    try:
        return ImageFont.load_default(size=int(size)), False
    except TypeError:
        pass
    except Exception:
        return None, False
    try:
        return ImageFont.load_default(), False
    except Exception:
        return None, False


def _draw_text(draw, xy, text, font, fill=(255, 255, 255)):
    """Draw with a black outline so the label survives a bright sky."""
    try:
        draw.text(xy, text, font=font, fill=fill, stroke_width=2,
                  stroke_fill=(0, 0, 0))
    except Exception:
        pass


# --------------------------------------------------------------------- triptych

def triptych(out_dir, modalities=("optical", "ir_mwir", "radar"), *,
             dst="images/preview", label_text=True,
             cell_size=(640, 480), max_width=2500):
    """Render one side-by-side preview per frame across modalities.

    Each modality owns a fixed cell (640x480 by default, background #101010)
    in which the frame is scaled to fit entirely and then centred, so a 16:1
    range-Doppler strip stays a 640x40 strip instead of forcing an 8192 px
    canvas.  Cells are separated by a 6 px #202020 bar, and a 22 px
    semi-transparent label bar above each cell carries the modality name and
    the frame number.  Three modalities therefore render 1932x502;
    label_text=False produces clean cells, and a machine without a usable font
    silently drops the text instead of failing.

    Files land in <out_dir>/<dst>/%06d.png, one per frame id, so a frame's
    preview and its images share a number.  A modality whose directory is
    missing (or empty) is skipped and named on the returned Generated list,
    which is still a plain list[str] of written paths.

    Parameters
    ----------
    out_dir : path
        Dataset output directory (the one holding images/).
    modalities : sequence of str
        Directory/modality names, e.g. ("optical", "ir_mwir", "radar").
    dst : str
        Destination directory relative to out_dir.
    label_text : bool
        Burn modality + frame labels into the tiles.
    cell_size : (int, int)
        Fixed cell reserved for one modality.  The image is scaled to fit
        entirely inside it and centred, so a 16:1 range-Doppler strip becomes
        a 640x40 strip in a 640x480 cell instead of forcing an 8192 px canvas.
    max_width : int
        Hard cap on the output width; a five-modality row is scaled down
        proportionally (cell width and height together) so a preview never
        becomes a panorama.
    """
    out_dir = os.fspath(out_dir)
    cell_w, cell_h = int(cell_size[0]), int(cell_size[1])
    if cell_w <= 0 or cell_h <= 0:
        raise ValueError("cell_size 必须是正整数，收到 %r" % (cell_size,))

    sources = []
    skipped = []
    for modality in modalities:
        directory = _resolve_modality(out_dir, modality)
        if directory is None:
            skipped.append("%s（目录不存在）" % modality)
            continue
        frames = _frames_in(directory)
        if not frames:
            skipped.append("%s（目录为空）" % modality)
            continue
        sources.append(dict(
            name=str(modality),
            label=_modality_label(modality),
            ascii=_modality_ascii(modality),
            dir=directory,
            by_frame={fid: path for fid, path in frames},
            order=[fid for fid, _path in frames],
        ))
    if not sources:
        raise FileNotFoundError(
            "没有找到可拼接的模态目录：%s（%s）"
            % (out_dir, "、".join(str(m) for m in modalities)))

    # Five modalities at 640 px would be a 3 224 px panorama; shrink the whole
    # cell (width and height together) so the tiles keep their proportions.
    count = len(sources)
    natural = count * cell_w + (count - 1) * SEP
    if max_width and natural > int(max_width):
        factor = float(int(max_width) - (count - 1) * SEP) / float(count * cell_w)
        factor = max(0.15, min(1.0, factor))
        cell_w = max(48, int(round(cell_w * factor)))
        cell_h = max(36, int(round(cell_h * factor)))

    label_h = max(16, int(round(cell_h * _LABEL_RATIO)))
    total_w = count * cell_w + (count - 1) * SEP
    total_h = cell_h + label_h
    font_size = max(11, int(round(18.0 * cell_h / 480.0)))
    font, has_cjk = _font(font_size) if label_text else (None, False)

    frame_ids = sorted({fid for src in sources for fid in src["order"]})
    destination = dst if os.path.isabs(dst) else os.path.normpath(os.path.join(out_dir, dst))
    os.makedirs(destination, exist_ok=True)

    written = []
    for fid in frame_ids:
        canvas = Image.new("RGB", (total_w, total_h), _CELL_RGB)
        for index, src in enumerate(sources):
            x0 = index * (cell_w + SEP)
            if index:
                canvas.paste(_SEP_RGB, (x0 - SEP, 0, x0, total_h))
            path = src["by_frame"].get(fid)
            if path is None:
                continue
            rgb, is_gray = _open_rgb(path)
            # Radar RD maps are point targets on a dark floor: a mean filter
            # would erase them, so they (and any single-channel frame shrunk
            # by 3x or more) are box-max pooled.
            image = _fit(rgb, cell_w, cell_h,
                         max_pool=("radar" in src["name"].lower()) or is_gray)
            canvas.paste(image, (x0 + (cell_w - image.width) // 2,
                                 label_h + (cell_h - image.height) // 2))
            if font is not None:
                # Semi-transparent bar above the picture so a bright sky cannot
                # swallow the modality name.  The text is drawn on the bar
                # layer, not on the canvas, so a glyph stroke can never bleed
                # into the first rows of the image below it.
                bar = Image.new("RGBA", (cell_w, label_h), (0, 0, 0, 150))
                _draw_text(ImageDraw.Draw(bar),
                           (8, max(1, (label_h - font_size) // 2)),
                           "%s %06d" % (src["label"] if has_cjk else src["ascii"], fid),
                           font)
                canvas.paste(bar, (x0, 0), bar)
        out_path = os.path.join(destination, "%06d.png" % fid)
        canvas.save(out_path)
        written.append(out_path)

    if not written:
        raise FileNotFoundError("没有可写的预览帧：%s" % out_dir)
    return Generated(written, skipped=skipped, frames=len(frame_ids))


# ----------------------------------------------------------------- contact sheet

def _sample(frames, limit):
    """Evenly spaced sample of frames, never more than limit."""
    if limit <= 0 or not frames:
        return []
    if len(frames) <= limit:
        return list(frames)
    if limit == 1:
        return [frames[len(frames) // 2]]
    picks = sorted({int(round(i * (len(frames) - 1) / (limit - 1)))
                    for i in range(limit)})
    return [frames[i] for i in picks]


def _max_pool_fit(image, width, height):
    """Box-filter a grayscale frame with MAX instead of a mean.

    A pulse-Doppler range-Doppler map is mostly floor plus a handful of 1-2 px
    returns.  A 6.4x area average divides a 255 return by 6.4 (~40, i.e. the
    target disappears); taking the block maximum keeps the point while still
    shrinking the strip to the cell.  Only used for radar / single-channel
    frames that are downsampled by 3x or more -- continuous-tone optical and
    thermal frames keep the resampling filter, or their noise would be
    amplified into mush.
    """
    gray = np.asarray(image.convert("L"))
    src_h, src_w = gray.shape
    if src_h < height or src_w < width:
        return image.resize((width, height), _RESAMPLE)
    ys = (np.arange(height) * src_h // height).astype(np.intp)
    xs = (np.arange(width) * src_w // width).astype(np.intp)
    pooled = np.maximum.reduceat(gray, ys, axis=0)
    pooled = np.maximum.reduceat(pooled, xs, axis=1)
    return Image.fromarray(np.ascontiguousarray(pooled), "L").convert("RGB")


def _fit(image, box_w, box_h, *, max_pool=False):
    """Scale image to fit inside the box, keeping its aspect ratio.

    max_pool switches a 3x+ downscale to block maximum, which is what keeps a
    two-pixel radar return visible instead of averaging it into the noise
    floor.
    """
    if image.width <= 0 or image.height <= 0:
        return image
    factor = min(float(box_w) / image.width, float(box_h) / image.height)
    width = max(1, int(round(image.width * factor)))
    height = max(1, int(round(image.height * factor)))
    if (width, height) == image.size:
        return image
    if max_pool and image.width >= width * 3 and image.height >= height:
        return _max_pool_fit(image, width, height)
    return image.resize((width, height), _RESAMPLE)


def contact_sheet(out_dir, *, dst="preview_sheet.png", cols=4, rows=4):
    """Lay up to cols * rows sampled preview frames onto one overview.

    Frames come from images/preview and are sampled evenly across the run, so
    a 2000-frame dataset still shows its beginning, middle and end.  The grid
    is sized to the frames that exist (cols = min(cols, frames)), so a two
    frame run produces a two-column sheet instead of a half-empty four-column
    canvas; the total sheet stays under roughly 2000x1400 for the default
    grid.

    Returns a ContactSheet (a str path with cells/cols/rows attributes)
    pointing at <out_dir>/<dst>.
    """
    out_dir = os.fspath(out_dir)
    cols_max = max(1, int(cols))
    rows_max = max(1, int(rows))
    preview_dir = os.path.join(out_dir, "images", "preview")
    frames = _frames_in(preview_dir)
    if not frames:
        raise FileNotFoundError("预览目录没有帧：%s" % preview_dir)

    picked = _sample(frames, cols_max * rows_max)
    thumbs = []
    for fid, path in picked:
        try:
            thumbs.append((fid, _fit(_to_rgb(path), 480, 260)))
        except OSError:
            continue
    if not thumbs:
        raise FileNotFoundError("预览帧无法读取：%s" % preview_dir)

    # A two-frame run must not reserve a four-column canvas: the grid follows
    # the frames that exist, so the sheet has no empty right half.
    used_cols = min(cols_max, len(thumbs))
    used_rows = (len(thumbs) + used_cols - 1) // used_cols
    cell_w = 480
    cell_h = max(thumb.height for _fid, thumb in thumbs) + 8
    gap, margin, caption_h = 10, 16, 20
    sheet_w = margin * 2 + used_cols * cell_w + (used_cols - 1) * gap
    sheet_h = margin * 2 + used_rows * (cell_h + caption_h) + (used_rows - 1) * gap
    sheet = Image.new("RGB", (sheet_w, sheet_h), (18, 18, 18))
    font, has_cjk = _font(13)
    draw = ImageDraw.Draw(sheet)

    for index, (fid, thumb) in enumerate(thumbs):
        col, row = index % used_cols, index // used_cols
        x0 = margin + col * (cell_w + gap)
        y0 = margin + row * (cell_h + caption_h + gap)
        draw.rectangle([x0, y0, x0 + cell_w - 1, y0 + cell_h + caption_h - 1],
                       fill=(26, 26, 26))
        sheet.paste(thumb, (x0 + (cell_w - thumb.width) // 2,
                            y0 + (cell_h - thumb.height) // 2))
        if font is not None:
            text = "帧 %06d" % fid if has_cjk else "frame %06d" % fid
            _draw_text(draw, (x0 + 6, y0 + cell_h + 2), text, font,
                       fill=(200, 200, 200))

    path = dst if os.path.isabs(dst) else os.path.join(out_dir, dst)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    sheet.save(path)
    return ContactSheet(path, cells=len(thumbs), cols=used_cols, rows=used_rows,
                        files=[p for _fid, p in picked])


# ---------------------------------------------------------------------- README

_SECTIONS = ("数据集概览", "场景配方", "传感器参数", "干扰实体",
             "轨迹事实", "标注统计", "复现命令", "已知局限")

_KIND_CN = {
    "noise_jammer": "噪声干扰机",
    "false_target": "假目标转发器",
    "chaff_cloud": "箔条云",
    "range_gate_puller": "距离波门拖引",
    "drfm": "数字射频存储转发",
    "spot_jammer": "瞄准式干扰机",
}

_MODE_CN = {"barrage": "阻塞式", "spot": "瞄准式", "swept": "扫频式"}


def _fmt(value, nd=1, unit=""):
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "—"
    return (("%%.%df" % nd) % number) + unit


def _fmt_int(value):
    try:
        return "%d" % int(value)
    except (TypeError, ValueError):
        return "—"


def _human_seconds(seconds):
    try:
        total = float(seconds)
    except (TypeError, ValueError):
        return "未记录"
    if total <= 0:
        return "未记录"
    minutes = int(total // 60)
    if minutes:
        return "%.2f s（%d 分 %d 秒）" % (total, minutes, int(round(total - minutes * 60)))
    return "%.2f s" % total


def _relpath(path):
    try:
        rel = os.path.relpath(path, os.getcwd())
    except (OSError, ValueError):
        return path
    return path if rel.startswith("..") else rel


def _scene_config(out_dir, meta):
    """Best available scene config: original path, a nearby copy, or the
    optical side-car render config (which also carries sensors and objects)."""
    raw = meta.get("config") if isinstance(meta, dict) else None
    candidates = []
    if raw:
        given = os.fspath(raw)
        candidates.append(given)
        candidates.append(os.path.join(out_dir, os.path.basename(given)))
        base = out_dir
        for _depth in (1, 2):
            base = os.path.dirname(base)
            candidates.append(os.path.join(base, "scenes", os.path.basename(given)))
    config = None
    for path in candidates:
        if path and os.path.isfile(path):
            config = path
            break
    if config is None:
        sidecar = os.path.join(out_dir, "_render_cfg.json")
        if os.path.isfile(sidecar):
            config = sidecar
    data = _read_json(config) if config else None
    return (data if isinstance(data, dict) else {}), (
        os.path.basename(config) if config else "")


def _dataset_name(out_dir, meta, cfg):
    for value in (meta.get("name"), cfg.get("name")):
        if value:
            return str(value)
    return os.path.basename(os.path.normpath(out_dir)) or "dataset"


def _generated_at(out_dir):
    try:
        stamp = os.path.getmtime(os.path.join(out_dir, "meta.json"))
    except OSError:
        stamp = None
    when = datetime.datetime.fromtimestamp(stamp) if stamp else datetime.datetime.now()
    return when.strftime("%Y-%m-%d %H:%M")


def _environment(cfg):
    ground = cfg.get("ground") or {}
    if not isinstance(ground, dict) or not ground.get("enabled"):
        return "空中（无地面或海面回波）"
    kind = str(ground.get("type", "")).lower()
    if kind in ("sea", "water", "ocean"):
        return "海面（海杂波 + 海面反射，海况与风速进入红外/雷达背景）"
    return "地面（地杂波 + 地形背景）"


def _weather_text(cfg, meta):
    weather = None
    if isinstance(meta.get("weather"), list) and meta["weather"]:
        weather = meta["weather"][0]
    if not isinstance(weather, dict):
        block = cfg.get("weather") or {}
        if isinstance(block, dict) and isinstance(block.get("states"), list) and block["states"]:
            weather = block["states"][0]
        elif isinstance(block, dict):
            weather = block
    if not isinstance(weather, dict):
        return "未记录"
    name = weather.get("name") or weather.get("preset") or "custom"
    parts = ["%s" % name]
    if weather.get("visibility_m") is not None:
        parts.append("能见度 %s km" % _fmt(float(weather["visibility_m"]) / 1000.0, 1))
    if weather.get("rain_mmh") is not None:
        parts.append("雨强 %s mm/h" % _fmt(weather["rain_mmh"], 1))
    if weather.get("wind_ms") is not None:
        parts.append("风速 %s m/s" % _fmt(weather["wind_ms"], 1))
    if weather.get("air_temp_c") is not None:
        parts.append("气温 %s ℃" % _fmt(weather["air_temp_c"], 1))
    if weather.get("cloud_cover") is not None:
        parts.append("云量 %s%%" % _fmt(float(weather["cloud_cover"]) * 100.0, 0))
    if weather.get("rh") is not None:
        parts.append("相对湿度 %s%%" % _fmt(float(weather["rh"]) * 100.0, 0))
    if weather.get("sea_state") is not None and float(weather["sea_state"] or 0) > 0:
        parts.append("海况 %s 级" % _fmt(weather["sea_state"], 1))
    if len(parts) > 1:
        return parts[0] + "：" + "、".join(parts[1:])
    return parts[0]


def _object_sizes(out_dir):
    """Object dimensions measured by the run, from the radar 3-D labels."""
    sizes = {}
    directory = os.path.join(out_dir, "labels", "radar")
    for path in sorted(glob.glob(os.path.join(directory, "*.json")))[:1]:
        data = _read_json(path)
        data = data if isinstance(data, dict) else {}
        boxes = data.get("boxes_3d") or []
        for box in boxes if isinstance(boxes, list) else []:
            name = str(box.get("name") or box.get("cls") or "")
            size = box.get("size")
            if name and isinstance(size, (list, tuple)) and len(size) == 3:
                sizes[name] = "%.1f × %.1f × %.1f m" % tuple(float(v) for v in size)
    return sizes


def _inventory(out_dir):
    """Every image directory the run wrote, with resolution and frame count."""
    entries = []
    previews = set()

    def scan(root, prefix=""):
        for name in sorted(os.listdir(root)):
            directory = os.path.join(root, name)
            if not os.path.isdir(directory):
                continue
            frames = _frames_in(directory)
            if not frames or name.startswith("preview"):
                continue
            if name.startswith("ir_preview"):
                previews.add("ir" + name[len("ir_preview"):])
                continue
            size = None
            try:
                with Image.open(frames[0][1]) as im:
                    size = im.size
            except OSError:
                size = None
            entries.append(dict(name=prefix + name, dir=directory,
                                count=len(frames), size=size, has_preview=False))

    images = os.path.join(out_dir, "images")
    if os.path.isdir(images):
        scan(images)
    lidar = os.path.join(out_dir, "lidar")
    if os.path.isdir(lidar):
        scan(lidar, prefix="lidar_")
    for entry in entries:
        entry["has_preview"] = entry["name"] in previews
    entries.sort(key=lambda e: (_ORDER.index(e["name"]) if e["name"] in _ORDER
                                else len(_ORDER), e["name"]))
    return entries


def _ir_band_of(name, bands):
    """MWIR/LWIR/SWIR of one IR product, from its directory name."""
    low = str(name).lower()
    for band in ("mwir", "lwir", "swir"):
        if band in low:
            return band.upper()
    return str(bands[0]).upper() if bands else "MWIR"


def _radar_range_window(out_dir, width, bandwidth_mhz):
    """Range window of the RD image: label axes when present, else width * dR.

    A run with n_samples=8192 and oversampling 2 writes 4096 range cells at
    c/(2B) = 3 m each, i.e. a 12.3 km window; the README has to say that
    rather than let the reader assume the old 3 km setting.
    """
    directory = os.path.join(out_dir, "labels", "radar")
    for path in sorted(glob.glob(os.path.join(directory, "*.json")))[:1]:
        data = _read_json(path)
        if not isinstance(data, dict):
            continue
        axes = data.get("axes") or {}
        ranges = axes.get("range_m") or []
        try:
            if len(ranges) >= 2:
                step = float(ranges[1]) - float(ranges[0])
                return dict(bins=len(ranges), r_max=float(ranges[-1]), step=step)
        except (TypeError, ValueError):
            pass
    if width and bandwidth_mhz:
        step = 149.896229 / float(bandwidth_mhz)
        return dict(bins=int(width), r_max=step * int(width), step=step)
    return None


def _sensor_row(entry, cfg, meta, out_dir):
    """One row of the sensor parameter table, from config + measured size."""
    sensors = cfg.get("sensors") or {}
    name = entry["name"]
    width, height = entry["size"] if entry["size"] else (None, None)
    res = "%s×%s" % (_fmt_int(width), _fmt_int(height)) if width else "未记录"
    previews = "（16-bit 原始 + 8-bit 预览）" if entry.get("has_preview") else ""
    views = meta.get("views") or []
    view_fov = views[0].get("fov_deg") if views and isinstance(views[0], dict) else None
    ir_bands = [str(b).upper() for b in (meta.get("ir_bands") or [])] or ["MWIR"]

    if name.startswith("optical"):
        camera = sensors.get("optical") or {}
        render = cfg.get("optical") or {}
        fov = camera.get("fov_deg", view_fov)
        params = []
        if render.get("samples") is not None:
            params.append("%s spp" % _fmt_int(render["samples"]))
        if render.get("exposure") is not None:
            params.append("曝光 %+.2f EV" % float(render["exposure"]))
        params.append("渲染后端 %s" % (render.get("backend") or camera.get("backend")
                                     or "Blender"))
        return ("可见光", "RGB 相机", "400–700 nm（可见光）", res + previews,
                _fmt(fov, 1, "°") if fov is not None else "—", "、".join(params))

    if name.startswith("ir"):
        block = sensors.get("ir_camera") or {}
        ir_cfg = cfg.get("ir") or {}
        fov = block.get("fov_deg", view_fov)
        params = []
        if ir_cfg.get("netd_mk") is not None:
            params.append("NETD %s mK" % _fmt_int(ir_cfg["netd_mk"]))
        if ir_cfg.get("psf_px") is not None:
            params.append("PSF %s px" % _fmt(ir_cfg["psf_px"], 1))
        nuc = ir_cfg.get("nuc") or {}
        if nuc.get("enabled"):
            params.append("NUC 开启（PRNU %s）" % _fmt(nuc.get("prnu"), 3))
        row_band = _ir_band_of(name, ir_bands)
        params.append("波段 %s" % row_band)
        label = _CN.get(name, "红外")
        band = {"MWIR": "3–5 µm（中波）", "LWIR": "8–14 µm（长波）",
                "SWIR": "1–3 µm（短波）"}.get(row_band, "红外波段")
        return (label, "红外热像", band, res + previews,
                _fmt(fov, 1, "°") if fov is not None else "—", "、".join(params))

    if name == "radar_ra":
        block = sensors.get("radar") or {}
        az, el = block.get("az_span_deg"), block.get("el_span_deg")
        fov = ("方位 %s° × 俯仰 %s°" % (_fmt(az, 1), _fmt(el, 1))
               if az is not None and el is not None else "—")
        return ("雷达 距离-角度", "脉冲多普勒 RA 图",
                "%s 波段" % (meta.get("radar_band") or block.get("band") or "—"),
                res, fov,
                "由同一 CPI 的检测点生成，%s×%s 方位/俯仰栅格"
                % (_fmt_int(block.get("n_az")), _fmt_int(block.get("n_el"))))

    if name == "radar":
        block = sensors.get("radar") or {}
        band = meta.get("radar_band") or block.get("band") or "—"
        az, el = block.get("az_span_deg"), block.get("el_span_deg")
        fov = ("方位 %s° × 俯仰 %s°" % (_fmt(az, 1), _fmt(el, 1))
               if az is not None and el is not None else "—")
        params = []
        if block.get("bandwidth_mhz") is not None:
            bw = float(block["bandwidth_mhz"])
            params.append("带宽 %s MHz（ΔR≈%s m）" % (_fmt(bw, 0), _fmt(149.896229 / bw, 2)))
        if block.get("n_samples") is not None:
            params.append("%s 采样 × %s chirp → RD 图 %s 距离格 × %s 多普勒格"
                          % (_fmt_int(block["n_samples"]),
                             _fmt_int(block.get("n_chirps")),
                             _fmt_int(width), _fmt_int(height)))
        span = _radar_range_window(out_dir, width, block.get("bandwidth_mhz"))
        if span:
            params.append("距离窗 %s km（%s m/格 × %s 格）"
                          % (_fmt(span["r_max"] / 1000.0, 2),
                             _fmt(span["step"], 2), _fmt_int(span["bins"])))
        if block.get("pri_us") is not None:
            params.append("PRI %s µs" % _fmt(block["pri_us"], 1))
        # The unambiguous Doppler window is the number this radar is actually
        # designed around: chirp, PRI, samples and bandwidth are all derived to
        # fit the closing speed inside it. Stating it lets a reader check the
        # RD axis instead of trusting it.
        if block.get("v_nyq_mps") is not None:
            params.append("多普勒不模糊 ±%s m/s%s"
                          % (_fmt(block["v_nyq_mps"], 1),
                             "" if block.get("unambiguous", True)
                             else "（本场景接近率超出，已发生模糊）"))
        if block.get("tx_power_dbm") is not None:
            params.append("发射 %s dBm" % _fmt(block["tx_power_dbm"], 1))
        if block.get("noise_figure_db") is not None:
            params.append("噪声系数 %s dB" % _fmt(block["noise_figure_db"], 1))
        return ("雷达 距离-多普勒", "脉冲多普勒 RD 图", "%s 波段" % band, res, fov,
                "、".join(params) or "—")

    if name == "sar":
        block = sensors.get("sar") or {}
        band = meta.get("sar_band") or block.get("band") or "—"
        params = []
        if block.get("fc_ghz") is not None:
            params.append("%s GHz" % _fmt(block["fc_ghz"], 2))
        if block.get("bandwidth_mhz") is not None:
            params.append("带宽 %s MHz（距离分辨率≈%s m）"
                          % (_fmt(block["bandwidth_mhz"], 0),
                             _fmt(149.896229 / float(block["bandwidth_mhz"]), 2)))
        if block.get("n_pulses") is not None:
            params.append("%s 脉冲、PRF %s Hz"
                          % (_fmt_int(block["n_pulses"]), _fmt_int(block.get("prf_hz"))))
        if block.get("polarization"):
            params.append("%s 极化" % block["polarization"])
        if block.get("tx_power_w") is not None:
            params.append("发射 %s W" % _fmt(block["tx_power_w"], 0))
        if block.get("swath_width_m") is not None:
            params.append("条带 %s m" % _fmt(block["swath_width_m"], 0))
        return ("SAR 强度", "条带式 SAR 快照", "%s 波段" % band, res,
                "正侧视，航高 %s m" % _fmt(block.get("altitude_m"), 0),
                "、".join(params) or "—")

    if name.startswith("lidar"):
        block = sensors.get("lidar") or {}
        fov = block.get("fov_deg", view_fov)
        params = []
        if block.get("max_range_m") is not None:
            params.append("测距 %s m" % _fmt(block["max_range_m"], 0))
        if block.get("range_spec_m") is not None:
            params.append("10%% 反射率作用距离 %s m" % _fmt(block["range_spec_m"], 0))
        if block.get("pulse_energy_nj") is not None:
            params.append("脉宽能量 %s nJ" % _fmt(block["pulse_energy_nj"], 0))
        if block.get("aperture_mm") is not None:
            params.append("口径 %s mm" % _fmt(block["aperture_mm"], 0))
        if block.get("n_az") is not None and block.get("n_el") is not None:
            params.append("%s×%s 扫描" % (_fmt_int(block["n_az"]), _fmt_int(block["n_el"])))
        intensity = name.endswith("intensity")
        return ("激光雷达强度" if intensity else "激光雷达距离",
                "激光强度图" if intensity else "激光距离图",
                "905 nm（近红外）", res,
                _fmt(fov, 1, "°") if fov is not None else "—", "、".join(params) or "—")

    return (_modality_label(name), "图像", "—", res, "—", "—")


def _range_series(frames):
    series = []
    for frame in frames or []:
        if not isinstance(frame, dict):
            continue
        value = frame.get("range_m")
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            series.append((float(frame.get("t") or 0.0), float(value)))
    series.sort(key=lambda item: item[0])
    return series


def _range_series_from_labels(out_dir):
    """Fallback trajectory: per-frame label JSONs carry range_m."""
    labels = os.path.join(out_dir, "labels")
    if not os.path.isdir(labels):
        return []
    for name in sorted(os.listdir(labels)):
        directory = os.path.join(labels, name)
        if not os.path.isdir(directory):
            continue
        series = []
        for path in sorted(glob.glob(os.path.join(directory, "*.json"))):
            data = _read_json(path)
            if not isinstance(data, dict):
                continue
            value = data.get("range_m")
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                series.append((float(data.get("frame") or len(series)), float(value)))
        if len(series) >= 2:
            return series
    return []


def _aspect_from_observations(out_dir):
    """Line-of-sight geometry at frame 0, if the run kept observations."""
    data = _read_json(os.path.join(out_dir, "observations", "000000.json"))
    if not isinstance(data, dict):
        return None
    observer = data.get("observer") or {}
    objects = data.get("objects") or []
    if not objects or not isinstance(objects[0], dict):
        return None
    target = objects[0]
    try:
        obs = np.asarray(observer.get("position_m"), dtype=float)
        tgt = np.asarray(target.get("center_world_m"), dtype=float)
        v_t = np.asarray(target.get("velocity_mps"), dtype=float)
        v_s = np.asarray(observer.get("velocity_mps"), dtype=float)
    except (TypeError, ValueError):
        return None
    los = tgt - obs
    norm = float(np.linalg.norm(los))
    if norm <= 1e-6 or not np.all(np.isfinite(los)):
        return None
    unit = los / norm
    speed_t = float(np.linalg.norm(v_t))
    cos = float(np.dot(v_t, unit) / speed_t) if speed_t > 1e-6 else 0.0
    closing = float(-np.dot(v_t - v_s, unit))
    return dict(cos=cos, closing=closing, speed_t=speed_t,
                speed_s=float(np.linalg.norm(v_s)))


def _aspect_label(cos):
    if cos < -0.5:
        return "迎头（目标朝载机方向飞）"
    if cos > 0.5:
        return "尾追（目标背向载机飞）"
    return "侧向/斜交（目标横穿视线）"


def _ecm_entities(meta, cfg):
    """Scene-level jamming entities, joined with the config's ERP values."""
    ecm = meta.get("ecm") if isinstance(meta, dict) else None
    if not isinstance(ecm, dict):
        return None
    entities = ecm.get("entities") or []
    if not entities:
        return None
    original = {}
    for item in (cfg.get("ecm") or {}).get("entities") or []:
        if isinstance(item, dict) and item.get("id"):
            original[str(item["id"])] = item
    groups = {}
    for frame in ecm.get("per_frame") or []:
        if not isinstance(frame, dict):
            continue
        for item in frame.get("entities") or []:
            if isinstance(item, dict) and item.get("id"):
                groups.setdefault(str(item["id"]), item)
    rows = []
    for item in entities:
        if not isinstance(item, dict):
            continue
        ident = str(item.get("id") or "—")
        source = original.get(ident, {})
        live = groups.get(ident, {})
        kind = str(item.get("kind") or source.get("kind") or "—")
        group = str(live.get("group") or source.get("group") or "")
        if not group:
            group = "passive" if kind == "chaff_cloud" else "active"
        power = []
        erp = source.get("erp_w", item.get("erp_w"))
        if erp:
            try:
                power.append("ERP %s W（%s dBW）" % (
                    _fmt(erp, 0), _fmt(10.0 * math.log10(float(erp)), 1)))
            except (TypeError, ValueError):
                power.append("ERP %s W" % erp)
        rcs = source.get("rcs_m2", item.get("rcs_m2"))
        if rcs:
            power.append("RCS %s m²" % _fmt(rcs, 1))
        if source.get("jam_bandwidth_mhz"):
            power.append("干扰带宽 %s MHz" % _fmt(source["jam_bandwidth_mhz"], 0))
        pos = item.get("pos") if item.get("pos") is not None else source.get("pos")
        vel = item.get("vel") if item.get("vel") is not None else source.get("vel")
        mode = item.get("mode") or source.get("mode")
        rows.append(dict(
            id=ident,
            group="有源" if group == "active" else "无源",
            kind=_KIND_CN.get(kind, kind) + (
                "（%s）" % _MODE_CN[mode] if mode in _MODE_CN else ""),
            power="、".join(power) or "—",
            pos="(%s, %s, %s)" % tuple(_fmt(v, 0) for v in pos)
            if isinstance(pos, (list, tuple)) and len(pos) == 3 else "—",
            vel="(%s, %s, %s)" % tuple(_fmt(v, 0) for v in vel)
            if isinstance(vel, (list, tuple)) and len(vel) == 3 else "—",
            note=str(source.get("note") or "—"),
        ))
    return ecm, rows


def _annotation_rows(out_dir, meta):
    """Per-modality image/box counts from COCO files and frame labels."""
    rows = {}
    seen = set()
    for base in (out_dir, os.path.join(out_dir, "labels")):
        if not os.path.isdir(base):
            continue
        for name in sorted(os.listdir(base)):
            if not (name.startswith("annotations_") and name.endswith(".json")):
                continue
            if ".part" in name or name in seen:
                continue
            seen.add(name)
            data = _read_json(os.path.join(base, name))
            if not isinstance(data, dict):
                continue
            modality = name[len("annotations_"):-len(".json")]
            if modality.endswith("_coco"):
                modality = modality[:-len("_coco")]
            row = rows.setdefault(modality, dict(images=0, boxes=0,
                                                 cats=Counter(), files=[]))
            row["images"] += len(data.get("images") or [])
            row["boxes"] += len(data.get("annotations") or [])
            categories = {c.get("id"): c.get("name")
                          for c in data.get("categories") or [] if isinstance(c, dict)}
            for ann in data.get("annotations") or []:
                if not isinstance(ann, dict):
                    continue
                key = categories.get(ann.get("category_id"))
                row["cats"][str(key or ann.get("category_id") or "未分类")] += 1
            row["files"].append(name)

    labels = os.path.join(out_dir, "labels")
    if os.path.isdir(labels):
        for name in sorted(os.listdir(labels)):
            directory = os.path.join(labels, name)
            if not os.path.isdir(directory):
                continue
            files = sorted(glob.glob(os.path.join(directory, "*.json")))
            if not files:
                continue
            row = rows.setdefault(name, dict(images=0, boxes=0,
                                             cats=Counter(), files=[]))
            if not row["images"]:
                row["images"] = len(files)
            if not row["boxes"]:
                total = 0
                cats = Counter()
                for path in files:
                    data = _read_json(path)
                    if not isinstance(data, dict):
                        continue
                    boxes = (data.get("boxes") or data.get("boxes_3d")
                             or data.get("boxes_rd") or {})
                    if isinstance(boxes, dict):
                        total += len(boxes)
                        for key in boxes:
                            cats[str(key)] += 1
                    elif isinstance(boxes, list):
                        total += len(boxes)
                        for item in boxes:
                            if isinstance(item, dict):
                                cats[str(item.get("cls") or item.get("name") or "目标")] += 1
                row["boxes"] = total
                if cats:
                    row["cats"] = cats
    order = {entry["name"]: index for index, entry in enumerate(_inventory(out_dir))}
    ranked = sorted(rows, key=lambda m: (order.get(m, len(order)),
                                         _ORDER.index(m) if m in _ORDER else len(_ORDER), m))
    return [(name, rows[name]) for name in ranked]


# ---------------------------------------------------------------------- public

def write_readme(out_dir, meta=None):
    """Write 数据集说明.md next to the data and return its path.

    Sections, in order: dataset overview, scene recipe, sensor parameters,
    jamming entities, trajectory facts, annotation statistics, reproduction
    command, known limitations.  meta may be the dict dataset.run already
    holds; otherwise meta.json is read from out_dir.

    The document is a finished deliverable: when an optional input is absent
    (no meta.json, no scene config) the section states what was measured
    instead of leaving a blank or a note to fill in later.
    """
    out_dir = os.fspath(out_dir)
    if meta is None:
        meta = _read_json(os.path.join(out_dir, "meta.json")) or {}
    if not isinstance(meta, dict):
        meta = {}
    cfg, cfg_name = _scene_config(out_dir, meta)
    frames = _read_json(os.path.join(out_dir, "frames.json")) or []
    if not isinstance(frames, list):
        frames = []
    stats = _read_json(os.path.join(out_dir, "stats.json")) or {}
    if not isinstance(stats, dict):
        stats = {}
    splits = _read_json(os.path.join(out_dir, "splits.json")) or {}
    if not isinstance(splits, dict):
        splits = {}
    inventory = _inventory(out_dir)
    families = []
    for entry in inventory:
        family = _family(entry["name"])
        if family not in families:
            families.append(family)
    annotations = _annotation_rows(out_dir, meta)
    name = _dataset_name(out_dir, meta, cfg)
    generated_at = _generated_at(out_dir)
    n_frames = meta.get("frames")
    if not isinstance(n_frames, int) or n_frames <= 0:
        n_frames = len(frames) or max((e["count"] for e in inventory), default=0)

    lines = []

    def add(text=""):
        lines.append(text)

    add("# 数据集说明：%s" % name)
    add()
    add("> 由 simkit 自动生成 · %s · %d 帧 · %d 个模态、%d 类图像产品。"
        "本文件描述的是当前目录里这份数据本身：先说明它是什么，再给出传感器与"
        "干扰参数，最后是复现命令与使用边界。"
        % (generated_at, int(n_frames), len(families), len(inventory)))
    add()

    # ------------------------------------------------------------- 1. 概览
    add("## 1. 数据集概览")
    add()
    add("| 项目 | 内容 |")
    add("|---|---|")
    add("| 数据集名 | %s |" % name)
    add("| 输出目录 | %s |" % _relpath(os.path.abspath(out_dir)).replace("\\", "/"))
    add("| 生成时间 | %s |" % generated_at)
    add("| 帧数 | %d |" % int(n_frames))
    if inventory:
        listed = "、".join(
            "%s %s×%s ×%d%s" % (
                _modality_label(entry["name"]),
                _fmt_int(entry["size"][0]) if entry["size"] else "?",
                _fmt_int(entry["size"][1]) if entry["size"] else "?",
                entry["count"],
                "（16-bit + 预览）" if entry.get("has_preview") else "")
            for entry in inventory)
    else:
        listed = "目录中没有找到图像"
    if inventory:
        add("| 模态清单 | %d 个模态（%s）、%d 类图像产品：%s |" % (
            len(families), "、".join(_FAMILY_CN.get(f, f) for f in families),
            len(inventory), listed))
    else:
        add("| 模态清单 | %s |" % listed)
    preview_frames = _frames_in(os.path.join(out_dir, "images", "preview"))
    if preview_frames:
        add("| 预览 | images/preview/ %d 张等高拼接图，preview_sheet.png 为抽样总览 |"
            % len(preview_frames))
    add("| 总耗时 | %s |" % _human_seconds(meta.get("seconds")))
    if meta.get("generated_by"):
        add("| 生成器 | %s |" % meta["generated_by"])
    if meta.get("seed") is not None:
        add("| 随机种子 | %s |" % _fmt_int(meta["seed"]))
    if cfg_name:
        config_cell = _code("scenes/" + cfg_name) if (cfg_name.endswith(".json")
                                                     and not cfg_name.startswith("_")) \
            else _code(cfg_name)
    else:
        config_cell = "未随数据集保存"
    add("| 场景配置 | %s |" % config_cell)
    views = meta.get("views") or []
    if views and isinstance(views[0], dict):
        add("| 传感器视角 | %s（%s° 视场，%s×%s） |" % (
            views[0].get("name") or "—", _fmt(views[0].get("fov_deg"), 1),
            _fmt_int(views[0].get("width")), _fmt_int(views[0].get("height"))))
    if splits:
        add("| 数据划分 | %s |" % "、".join(
            "%s %d 帧" % (key, len(value)) for key, value in splits.items()
            if isinstance(value, list)))
    add()
    add("图像按帧号（%s）在模态目录下对齐，标注为 COCO 或逐帧 JSON；"
        "images/preview/ 是按帧的等高拼接图（各模态同高、留 6 px 分隔条），"
        "preview_sheet.png 是抽样总览。" % _code("%06d.png"))
    add()

    # ------------------------------------------------------------- 2. 场景配方
    add("## 2. 场景配方")
    add()
    add("| 项目 | 取值 |")
    add("|---|---|")
    add("| 环境类型 | %s |" % _environment(cfg))
    add("| 天气 | %s |" % _weather_text(cfg, meta))
    weather_states = (cfg.get("weather") or {}).get("states") \
        if isinstance(cfg.get("weather"), dict) else None
    if isinstance(weather_states, list) and len(weather_states) > 1:
        add("| 天气序列 | %s |" % " → ".join(
            str(item.get("preset") or item.get("name") or "custom")
            for item in weather_states if isinstance(item, dict)))
    dt = None
    frame_block = cfg.get("frames") or {}
    if isinstance(frame_block, dict) and frame_block.get("dt") is not None:
        dt = float(frame_block["dt"])
    elif len(frames) >= 2 and isinstance(frames[0], dict) and isinstance(frames[1], dict):
        try:
            dt = float(frames[1]["t"]) - float(frames[0]["t"])
        except (KeyError, TypeError, ValueError):
            dt = None
    if dt:
        add("| 帧率 / 采样 | %s s/帧（%s Hz），共 %s |" % (
            _fmt(dt, 3), _fmt(1.0 / dt if dt else 0, 1),
            _fmt(dt * max(int(n_frames), 1), 2, " s")))
    objects = [o for o in (cfg.get("objects") or []) if isinstance(o, dict)]
    add("| 目标数量 | %d |" % (len(objects) or len(meta.get("categories") or [])))
    add()
    sizes = _object_sizes(out_dir)
    if objects:
        add("| 目标 | 类别 | 三维模型 | 尺寸 / 缩放 | 雷达 RCS 覆盖 | 温度 | 初始速度 |")
        add("|---|---|---|---|---|---|---|")
        for obj in objects:
            model = os.path.basename(str(obj.get("model") or "")) or "—"
            scaling = []
            if obj.get("scale") is not None:
                scaling.append("scale %s" % _fmt(obj["scale"], 3))
            if obj.get("height") is not None:
                scaling.append("按高度 %s m 归一" % _fmt(obj["height"], 1))
            if obj.get("decimate"):
                scaling.append("抽稀 %s 面" % _fmt_int(obj["decimate"]))
            if not scaling:
                scaling.append("按资产原始尺寸")
            size = sizes.get(str(obj.get("name") or ""))
            if size:
                scaling.append(size)
            add("| %s | %s | %s | %s | %s dBsm | %s | %s |" % (
                obj.get("name") or "—", obj.get("cls") or "—", _code(model),
                "，".join(scaling),
                _fmt(obj.get("rcs_db"), 1),
                _fmt(obj.get("temp_c"), 1, " ℃"),
                "(%s, %s, %s) m/s" % tuple(_fmt(v, 1) for v in (obj.get("vel") or [0, 0, 0]))))
        add()
    else:
        add("场景配置没有列出目标（objects 为空），本目录的标注统计见第 6 节。")
        add()

    # ------------------------------------------------------------- 3. 传感器
    add("## 3. 传感器参数")
    add()
    add("同一帧内所有模态共用一条时间轴；下表的分辨率取自目录中实际写出的图像，"
        "其余参数取自场景配置的 sensors 段（标称值）。")
    add()
    if inventory:
        add("| 模态 | 类型 | 波段 / 频率 | 分辨率 | 视场 | 关键参数 |")
        add("|---|---|---|---|---|---|")
        for entry in inventory:
            add("| %s |" % " | ".join(_sensor_row(entry, cfg, meta, out_dir)))
        add()
    else:
        add("目录中没有图像，无法列出传感器参数。")
        add()
    add("雷达的距离-多普勒图与距离-角度图来自同一次 CPI；红外同时保存 16-bit "
        "原始计数与 8-bit 预览，拼接图使用预览，16-bit 数据用于定量分析。")
    add()

    # ------------------------------------------------------------- 4. 干扰
    add("## 4. 干扰实体")
    add()
    ecm = _ecm_entities(meta, cfg)
    if ecm:
        block, rows = ecm
        add("场景级射频干扰共有 %d 个实体，随帧时间演化（meta.json 的 ecm 段）。" % len(rows))
        add()
        add("| 编号 | 有源 / 无源 | 类型 | 功率 / RCS | 初始位置 (m) | 速度 (m/s) | 作用 |")
        add("|---|---|---|---|---|---|---|")
        for row in rows:
            add("| %s | %s | %s | %s | %s | %s | %s |" % (
                row["id"], row["group"], row["kind"], row["power"],
                row["pos"], row["vel"], row["note"]))
        add()
        facts = []
        if block.get("frames") is not None:
            facts.append("干扰覆盖 %d 帧" % int(block["frames"]))
        js = block.get("js_db") or {}
        if isinstance(js, dict) and js.get("min") is not None:
            facts.append("干信比 J/S %s–%s dB" % (_fmt(js.get("min"), 1), _fmt(js.get("max"), 1)))
        rise = block.get("noise_rise_db") or {}
        if isinstance(rise, dict) and rise.get("min") is not None:
            facts.append("噪底抬升 %s–%s dB" % (_fmt(rise.get("min"), 1),
                                               _fmt(rise.get("max"), 1)))
        if block.get("injected_echoes") is not None:
            facts.append("注入假回波 %s 条" % _fmt_int(block["injected_echoes"]))
        if facts:
            add("实测效果：%s。" % "；".join(facts))
            add()
        add("有源干扰改变雷达的检测与测距，箔条云是无源散射体；红外与光学图像"
            "不包含射频干扰，但受同一场景的能见度与光照影响。")
        add()
    else:
        add("本数据集没有场景级干扰：meta.json 中不存在 ecm 段，各模态图像均未"
            "叠加人为有源或无源干扰。")
        add()
    interference = meta.get("interference")
    if isinstance(interference, dict):
        add("另有图像域干扰增强副本 interfered/（干扰域 %s，变体 %s），"
            "用于鲁棒性测试，与上表的射频干扰相互独立。"
            % (interference.get("domain") or "—", interference.get("variant") or "—"))
        add()

    # ------------------------------------------------------------- 5. 轨迹
    add("## 5. 轨迹事实")
    add()
    series = _range_series(frames) or _range_series_from_labels(out_dir)
    aspect = _aspect_from_observations(out_dir)
    if len(series) >= 2:
        first_t, first_r = series[0]
        last_t, last_r = series[-1]
        ranges = [r for _t, r in series]
        span = (last_t - first_t) or ((len(series) - 1) * (dt or 0.0))
        rate = (last_r - first_r) / span if span > 0 else 0.0
        add("| 项目 | 数值 |")
        add("|---|---|")
        add("| 起始距离 | %s m |" % _fmt(first_r, 2))
        add("| 结束距离 | %s m |" % _fmt(last_r, 2))
        add("| 距离区间 | %s – %s m |" % (_fmt(min(ranges), 2), _fmt(max(ranges), 2)))
        add("| 平均距离变化率 | %s %s |" % (
            ("%+.2f" % rate) if abs(rate) >= 0.005 else "±0.00",
            "m/s（距离拉开）" if rate > 0.05 else
            ("m/s（距离接近）" if rate < -0.05 else "m/s（基本不变）")))
        if len(series) >= 3:
            rates = []
            for (t_prev, r_prev), (t_next, r_next) in zip(series, series[1:]):
                if t_next > t_prev:
                    rates.append((r_next - r_prev) / (t_next - t_prev))
            if rates:
                add("| 距离变化率区间 | %+.2f ～ %+.2f m/s（正值为拉开） |"
                    % (min(rates), max(rates)))
        if aspect:
            add("| 交会几何 | %s；目标速度与视线夹角 %s° |" % (
                _aspect_label(aspect["cos"]),
                _fmt(math.degrees(math.acos(max(-1.0, min(1.0, aspect["cos"])))), 1)))
            add("| 速度 | 目标 %s m/s、载机 %s m/s；首帧视线距离变化 %s m/s |" % (
                _fmt(aspect["speed_t"], 1), _fmt(aspect["speed_s"], 1),
                _fmt(-aspect["closing"], 2)))
        else:
            sensor_vel = frames[0].get("sensor_velocity_mps") if frames else None
            velocities = [o.get("vel") for o in objects if o.get("vel")]
            label = "未记录观测文件，按距离趋势判定"
            if sensor_vel and velocities:
                try:
                    dot = float(np.dot(np.asarray(velocities[0], float),
                                       np.asarray(sensor_vel, float)))
                    label = ("同向飞行（速度矢量夹角 <90°）" if dot > 0
                             else "相向飞行（速度矢量夹角 >90°）")
                except (TypeError, ValueError):
                    pass
            add("| 交会几何 | %s |" % label)
        target_px = frames[0].get("target_px") if frames and isinstance(frames[0], dict) else None
        if target_px:
            add("| 首帧目标尺寸 | %s px（主视场） |" % _fmt(target_px, 1))
        add()
        add("距离为逐帧记录的仿真值，变化率由首末帧差分得到；%d 帧的采样区间为 %s s。"
            % (len(series), _fmt(span, 2)))
        add()
    else:
        add("本目录没有可用的距离记录（既无 frames.json 也无逐帧 range_m），"
            "轨迹参数无法从数据中复原；时间轴与帧数见第 1 节。")
        add()

    # ------------------------------------------------------------- 6. 标注
    add("## 6. 标注统计")
    add()
    if annotations:
        add("| 模态 | 图片数 | 标注数 | 类别分布 |")
        add("|---|---|---|---|")
        for modality, row in annotations:
            cats = "、".join("%s %d" % (key, count) for key, count in
                            sorted(row["cats"].items(), key=lambda kv: (-kv[1], kv[0])))
            add("| %s | %d | %d | %s |" % (_modality_label(modality),
                                           row["images"], row["boxes"], cats or "—"))
        add()
        total_boxes = sum(row["boxes"] for _m, row in annotations)
        files = sorted({name for _m, row in annotations for name in row["files"]})
        add("标注共 %d 个目标框；COCO 文件：%s。每个模态的图片与标注一一对应，"
            "image_id 在同一模态内连续。" % (
                total_boxes,
                "、".join(_code(name) for name in files) if files else "逐帧 JSON"))
        add()
    else:
        add("本目录没有找到 annotations_*_coco.json 或 labels/<模态>/ 逐帧标注。")
        add()
    if stats.get("radar_points") is not None:
        add("雷达点云/检测点共 %s 个，平均每帧 %s 个。" % (
            _fmt_int(stats["radar_points"]),
            _fmt(float(stats["radar_points"]) / max(int(n_frames), 1), 1)))
        add()
    if splits:
        add("数据划分：%s。划分按帧号给出，可直接用于 train/val 复制。"
            % "、".join("%s %d 帧" % (key, len(value))
                        for key, value in splits.items() if isinstance(value, list)))
        add()

    # ------------------------------------------------------------- 7. 复现
    add("## 7. 复现命令")
    add()
    sensors = [item for item in ("optical", "ir", "radar", "lidar", "sar")
               if item in (meta.get("sensors") or [])]
    if not sensors and inventory:
        found = set()
        for entry in inventory:
            base = entry["name"].split("_")[0]
            if base in ("optical", "ir", "radar", "lidar", "sar"):
                found.add(base)
        sensors = [item for item in ("optical", "ir", "radar", "lidar", "sar")
                   if item in found]
    sensor_arg = ",".join(sensors) or "optical,ir,radar"
    scene_arg = ("scenes/%s" % cfg_name) if (cfg_name and cfg_name.endswith(".json")
                                             and not cfg_name.startswith("_")) else "<scene.json>"
    seed_arg = " --seed %s" % _fmt_int(meta["seed"]) if meta.get("seed") is not None else ""
    out_arg = _relpath(os.path.abspath(out_dir)).replace("\\", "/")
    add(_FENCE + "bash")
    add("# 1) 重新生成数据集：同一场景 + 同一随机种子，逐帧可复现")
    add("python -m simkit gen -c %s -o %s -n %d -s %s%s"
        % (scene_arg, out_arg, int(n_frames), sensor_arg, seed_arg))
    add("")
    add("# 2) 只重建预览图、总览图与本说明（不重新渲染，几秒完成）")
    add("python -c \"from simkit import sheet; "
        "sheet.triptych('%s'); sheet.contact_sheet('%s'); sheet.write_readme('%s')\""
        % (out_arg, out_arg, out_arg))
    add(_FENCE)
    add()
    add("依赖仅为 %s 与 %s（见仓库 requirements.txt）；预览拼接不调用 Blender，"
        "也不修改任何图像与标注。" % (_code("numpy"), _code("Pillow")))
    add()

    # ------------------------------------------------------------- 8. 局限
    add("## 8. 已知局限")
    add()
    add("1. **合成数据**：全部图像与标注由物理仿真生成，不是外场采集。雷达 RCS 为"
        "几何 PO（物理光学）面元估计加 rcs_db 覆盖值，非实测；换一个 RCS "
        "假设，作用距离与检测概率都会随之移动。")
    add("2. **样本量不足**：本次 %d 帧只能验证管线、几何、标注与可见性，"
        "**不足以单独训练**检测器或分类器；需要更大批量时才具备训练价值。" % int(n_frames))
    add("3. **散射模型**：单次反射、无多次反射与腔体谐振，无极化统计与阴影遮挡，"
        "复杂目标（进气道、座舱、二面角）的 RCS 会偏低。")
    add("4. **传感器为标称模型**：NETD、噪声系数、发射功率、光学曝光取自配置文件，"
        "不是实测标定值；红外 8-bit 预览为百分位拉伸显示，不是辐亮度。")
    add("5. **运动学为协调转弯/直线**：没有气动积分与真实飞控，机动只有场景给定的"
        "过载与坡度上限。")
    add("6. **预览图不是训练数据**：拼接图按高度缩放、可能被拉伸对比度，只用于目视"
        "核查，不能用于标注、测量或训练。")
    add()
    add("本数据集适合做管线验证、回归测试、算法联调与演示；用于发表结论前，应说明"
        "上述合成与标定边界。")

    text = "\n".join(lines).rstrip() + "\n"
    path = os.path.join(out_dir, "数据集说明.md")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return path
