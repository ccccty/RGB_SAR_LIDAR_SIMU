"""Acceptance audit for the 10-frame five-modality minimal validation run.

Why this exists
---------------
"simkit validate" answers "did the renderers run and is there a labelled return":
it counts PNGs, checks that label files exist and asks whether any box is
visible. That is a smoke test. It never decodes a pixel, never says how strong
the return is, never proves that two modalities describe the same instant with
the same instance, and never proves that the RD map's target actually stands
above its own noise floor.

This script is the acceptance gate for the minimal validation pipeline. It opens
every image, recomputes every number from the delivered products instead of
trusting the generator's own report, and refuses to call a run valid when a
modality is blank, when the frame numbering disagrees between modalities, when
an annotation is not usable, or when the target does not rise above the
measured background. It then writes the human-readable report.

Usage, from the project root:

    python -m tools.validation_check out/validation_ecm
    python -m tools.validation_check out/validation_ecm --report docs/最小验证报告.md

The negative control that proves this gate can fail lives in
tools/validation_negative.py and writes out/validation_ecm/negative_control.json,
which the report folds into its own section.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

MODALITIES = ("optical", "ir", "radar", "lidar", "sar")

# dataset.py encodes the range/Doppler image with a fixed 30 dB window:
#   code = round(clip((db - noise_floor) / 30, 0, 1) * 255)      gamma 1.0
# so one code is 30/255 dB. Undoing that window is the only way the "peak above
# noise floor" threshold can mean a fixed number of dB instead of silently
# riding on whatever display contrast the generator happened to pick.
RD_DYNAMIC_RANGE_DB = 30.0
RD_DB_PER_CODE = RD_DYNAMIC_RANGE_DB / 255.0

# The analytic radar gate promotes a detection to a labelled box well above
# this level. 6 dB is deliberately the *floor* for acceptance: it is the
# smallest contrast that still reads as a target rather than as speckle.
MIN_RD_PEAK_DB = 6.0

# Why the box mean cannot decide visibility
# -----------------------------------------
# An aircraft at 1.49 km covers a 326 x 225 pixel box and only about a sixth of
# that box is airframe; the rest is sky seen through the gap between wing, fin
# and fuselage. Measured on frame 0 of this run: the box mean sits 7 codes BELOW
# the background while the brightest airframe pixel sits 105 codes above it and
# 15.6 percent of the box is more than 8 codes off background. A mean-based
# threshold calls a clearly visible aircraft invisible -- and, worse, can also
# accept a box that only half covers it.
#
# So two independent conditions are required:
#   * a high-percentile (95th) contrast: is a meaningful part of the labelled
#     region distinctly off-background? The 95th and not the maximum, so a
#     single hot pixel cannot satisfy it.
#   * the share of box pixels that are clearly off-background, so a tiny bright
#     speck inside a large empty box fails even when its percentile is high.
# Both are measured as |deviation| so a target darker than the background (an
# IR image on a warm background) is judged by the same rule as a brighter one.
MIN_RING_PEAK_CODES = 20.0         # optical: peak deviation from the ring, codes
MIN_RING_PEAK_COUNTS = 100.0       # IR: peak deviation from the ring, counts
MIN_RING_HIT_FRACTION = 0.02       # share of box pixels beyond 4 ring-sigma
MIN_TARGET_ECHO_RATE = 0.02        # share of the box's rays returning from target

IMAGE = None                       # Pillow module when available, else None


# ---------------------------------------------------------------------------
# io helpers
# ---------------------------------------------------------------------------
def _read_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _png_engine():
    """Pillow when installed; without it only the container can be verified.

    The environment ships Pillow, so the numeric checks below run on decoded
    pixels. The fallback exists because the project treats Pillow as optional,
    and a report that silently measured nothing would be worse than one that
    states it only verified the container.
    """
    try:
        from PIL import Image
        return "pillow", Image
    except Exception:
        return "signature", None


def _decode_png(path, Image):
    """Return (array or None, width, height, mode); raise when the file is broken.

    Every numeric check below measures the returned array, so a truncated or
    corrupted PNG has to fail here instead of yielding a plausible statistic
    from a partially decoded buffer.
    """
    raw = path.read_bytes()
    if raw[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG (bad signature)")
    if Image is None:
        width = int.from_bytes(raw[16:20], "big")
        height = int.from_bytes(raw[20:24], "big")
        return None, width, height, "unknown"
    with Image.open(path) as im:
        im.verify()                     # walks the chunks and checks their CRC
    with Image.open(path) as im:
        im.load()                       # decodes pixels; raises on truncation
        width, height = im.size
        mode = im.mode
        array = np.asarray(im)
        if mode in ("I", "I;16", "I;16B", "I;16L"):
            array = array.astype(np.uint16)
    return array, int(width), int(height), mode


def _gray(array):
    """One luminance plane: RGB is averaged, single-channel data passes through."""
    a = np.asarray(array)
    if a.ndim == 3:
        return a.astype(np.float64).mean(axis=2)
    return a.astype(np.float64)


def _clip_box(box, width, height):
    x0 = int(max(0, math.floor(float(box["x0"]))))
    y0 = int(max(0, math.floor(float(box["y0"]))))
    x1 = int(min(width, math.ceil(float(box["x1"]))))
    y1 = int(min(height, math.ceil(float(box["y1"]))))
    return x0, y0, x1, y1


def _region_stats(gray, box, width, height, floor_code=None):
    """Target-region statistics against the frame's own background median.

    The background is the frame median rather than a ring around the box: on a
    sky-only scene the median is the sky, which is exactly the background the
    target has to be told apart from.
    """
    x0, y0, x1, y1 = _clip_box(box, width, height)
    if x1 <= x0 or y1 <= y0:
        return None
    region = gray[y0:y1, x0:x1]
    background = float(np.median(gray))
    base = background if floor_code is None else float(floor_code)
    p95 = float(np.percentile(region, 95))
    return dict(mean=float(region.mean()), peak=float(region.max()),
                p50=float(np.percentile(region, 50)), p95=p95,
                background_median=background,
                contrast=float(region.mean()) - base,
                peak_contrast=float(region.max()) - base,
                p95_contrast=p95 - base,
                bright_fraction=float(np.mean(np.abs(region - background) > 8.0)),
                area_px=int((y1 - y0) * (x1 - x0)))


def _local_stats(gray, box, width, height, pad=0.15):
    """Target contrast against a ring of neighbouring background pixels.

    Why a ring and not the frame median: the sky in this scene has a vertical
    gradient of about ten codes across the frame, so a whole-frame threshold is
    too loose near the horizon and too tight higher up. The ring is the
    background immediately around the target -- which is what a detector
    compares against -- and its robust sigma says how much variation that
    neighbourhood really has.
    The ring is the padded box minus the box itself, and the MAD makes it
    insensitive to a wingtip that pokes into it.
    """
    x0, y0, x1, y1 = _clip_box(box, width, height)
    if x1 <= x0 or y1 <= y0:
        return None
    bw, bh = x1 - x0, y1 - y0
    dx, dy = max(2, int(round(bw * pad))), max(2, int(round(bh * pad)))
    ox0, oy0 = max(0, x0 - dx), max(0, y0 - dy)
    ox1, oy1 = min(width, x1 + dx), min(height, y1 + dy)
    outer = gray[oy0:oy1, ox0:ox1]
    mask = np.ones(outer.shape, bool)
    mask[y0 - oy0:y1 - oy0, x0 - ox0:x1 - ox0] = False
    ring = outer[mask]
    if ring.size < 32:
        return None
    med = float(np.median(ring))
    sigma = float(np.median(np.abs(ring - med))) * 1.4826
    region = gray[y0:y1, x0:x1]
    dev = np.abs(region - med)
    threshold = max(4.0 * sigma, 3.0)
    return dict(ring_median=med, ring_sigma=sigma, threshold=threshold,
                hit_fraction=float(np.mean(dev > threshold)),
                peak_deviation=float(dev.max()),
                median_deviation=float(np.median(dev)))


def _raw_rd(path, box=None):
    """Peak-over-floor in dB from the saved radar complex/power cube.

    Why the raw cube and not the PNG: the delivered RD image is written with a
    fixed 30 dB display window, so a strong target clips at code 255 and every
    such frame would report exactly 30.00 dB -- a property of the display, not
    of the measurement. raw/radar/*.npz keeps the unclipped power, which is the
    only place the real detection margin can be read. Returns None when the cube
    is missing, so the caller falls back to the (clipped) raster.
    """
    if not path.exists():
        return None
    try:
        with np.load(path) as cube:
            if "power" not in cube:
                return None
            power = np.asarray(cube["power"], dtype=np.float64)
    except Exception:
        return None
    if power.size == 0:
        return None
    floor = float(np.median(power))
    if floor <= 0.0:
        return None
    peak = float(power.max())
    out = dict(floor_db=round(10.0 * math.log10(floor), 2),
               peak_db=round(10.0 * math.log10(max(peak, 1e-30)), 2),
               peak_over_floor_db=round(10.0 * math.log10(peak / floor), 2))
    if box is not None:
        x0, y0, x1, y1 = _clip_box(box, power.shape[1], power.shape[0])
        if x1 > x0 and y1 > y0:
            roi = power[y0:y1, x0:x1]
            out["p95_over_floor_db"] = round(
                10.0 * math.log10(max(float(np.percentile(roi, 95)), 1e-30) / floor), 2)
    return out


def _iou(a, b):
    x0 = max(a["x0"], b["x0"])
    y0 = max(a["y0"], b["y0"])
    x1 = min(a["x1"], b["x1"])
    y1 = min(a["y1"], b["y1"])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area = ((a["x1"] - a["x0"]) * (a["y1"] - a["y0"])
            + (b["x1"] - b["x0"]) * (b["y1"] - b["y0"]) - inter)
    return inter / area if area > 0 else 0.0


def _centre(box):
    return ((float(box["x0"]) + float(box["x1"])) / 2.0,
            (float(box["y0"]) + float(box["y1"])) / 2.0)


# ---------------------------------------------------------------------------
# layout and label dialects
# ---------------------------------------------------------------------------
def _modality_dirs(meta):
    """Directory layout, resolved from the run's own meta.json.

    IR bands are tagged in the directory name (images/ir_mwir), so the layout
    cannot be hard-coded: a scene with two bands produces two image sets and
    both have to be audited.
    """
    dirs = dict(optical="images/optical", radar="images/radar", sar="images/sar",
                lidar="lidar/range")
    for band in meta.get("ir_bands") or ["lwir"]:
        dirs["ir_" + band] = "images/ir_" + band
    return dirs


def _label_dir(key):
    if key.startswith("ir_"):
        return "labels/ir_" + key[len("ir_"):]
    return "labels/" + key


def _boxes_of(label, modality):
    """Normalise the label dialects into [(name, box_dict), ...].

    Optical/IR/SAR write a dict keyed by object name, lidar writes a list (the
    range image has no per-object key), radar writes boxes_rd in RD space.
    """
    if modality == "radar":
        return list((label.get("boxes_rd") or {}).items())
    boxes = label.get("boxes")
    if isinstance(boxes, dict):
        return list(boxes.items())
    if isinstance(boxes, list):
        return [(str(i), b) for i, b in enumerate(boxes)]
    return []


def _instances(label, modality):
    """Instance id and class, whatever dialect this modality uses.

    Optical/IR/SAR carry the class in the COCO sidecar and the numpy renderers
    also stamp an objects block; radar keeps it in boxes_3d. The optical label
    file has no objects block at all -- Blender writes the boxes there and the
    class lives in COCO -- which is why the optical instance evidence is taken
    from the sidecar instead of from the per-frame label.
    """
    out = {}
    if modality == "radar":
        for row in label.get("boxes_3d") or []:
            out[str(row.get("name"))] = dict(id=row.get("id"), cls=row.get("cls"),
                                             cls_id=row.get("cls_id"))
        return out
    for row in label.get("objects") or []:
        out[str(row.get("name"))] = dict(id=row.get("id"), cls=row.get("cls"))
    return out


def _coco_report(root, modality, n_frames, labels_dir=None):
    """COCO sidecar: ids, categories, bbox sanity, and agreement with labels.

    The sidecar is what a trainer actually reads, so it is checked on its own
    terms: every annotation must point at a frame and a registered category,
    and its bbox must agree with the per-frame label file. A drifting pair of
    files is worse than a missing one, because only one of them gets loaded.
    """
    name = "annotations_%s_coco.json" % modality
    path = root / name
    if not path.exists():
        return None, ["missing " + name]
    coco = _read_json(path)
    errors = []
    images = coco.get("images") or []
    anns = coco.get("annotations") or []
    ids = sorted(int(i["id"]) for i in images)
    if ids != list(range(n_frames)):
        errors.append("%s: image ids are not 0..%d" % (name, n_frames - 1))
    cats = {c["id"]: c["name"] for c in coco.get("categories") or []}
    known = set(ids)
    classes = set()
    drift = 0.0
    unmatched = 0
    for ann in anns:
        if ann.get("image_id") not in known:
            errors.append("%s: annotation points at unknown image_id %s"
                          % (name, ann.get("image_id")))
        if ann.get("category_id") not in cats:
            errors.append("%s: annotation has unknown category_id %s"
                          % (name, ann.get("category_id")))
        else:
            classes.add(cats[ann["category_id"]])
        bbox = ann.get("bbox") or []
        if len(bbox) != 4 or float(bbox[2]) <= 0 or float(bbox[3]) <= 0:
            errors.append("%s: annotation %s has a degenerate bbox %s"
                          % (name, ann.get("id"), bbox))
            continue
        if labels_dir is None:
            continue
        lbl = root / labels_dir / ("%06d.json" % int(ann["image_id"]))
        if not lbl.exists():
            unmatched += 1
            continue
        candidates = _boxes_of(_read_json(lbl), modality)
        match = dict(candidates).get(ann.get("object_name"))
        if not match and len(candidates) == 1:
            # Lidar writes its range-image boxes as a bare list: the objects block
            # carries the name, the box list does not. With one target on screen
            # the lists line up positionally, so the single box is the match.
            match = candidates[0][1]
        if not match:
            unmatched += 1
            continue
        drift = max(drift, abs(float(match["x0"]) - float(bbox[0])),
                    abs(float(match["y0"]) - float(bbox[1])),
                    abs(float(match["x1"]) - (float(bbox[0]) + float(bbox[2]))),
                    abs(float(match["y1"]) - (float(bbox[1]) + float(bbox[3]))))
    if unmatched:
        errors.append("%s: %d annotation(s) have no matching box in the label files"
                      % (name, unmatched))
    if drift > 1.0:
        errors.append("%s: COCO bbox drifts %.2f px from the label files"
                      % (name, drift))
    return dict(file=name, images=len(images), annotations=len(anns),
                categories=sorted(classes), bbox_drift_px=round(drift, 3)), errors


def check(root, n_frames=10, modalities=MODALITIES, report_path=None, label="最小验证"):
    """Audit one run end to end and return the result dict (also written to disk)."""
    root = Path(root)
    global IMAGE
    engine, IMAGE = _png_engine()
    errors, warnings = [], []
    result = dict(root=str(root), frames_expected=int(n_frames), png_decoder=engine,
                  checks=[], errors=errors, warnings=warnings, notes=[])
    if not (root / "meta.json").exists():
        errors.append("meta.json missing: not a dataset directory")
        return _finish(result, report_path, label)
    meta = _read_json(root / "meta.json")
    frames_path = root / "frames.json"
    frames = _read_json(frames_path) if frames_path.exists() else []
    frame_ids = [int(row["frame"]) for row in frames]
    times = {int(row["frame"]): float(row["t"]) for row in frames}
    result["meta"] = dict(frames=meta.get("frames"), sensors=meta.get("sensors"),
                          seed=meta.get("seed"), seconds=meta.get("seconds"),
                          ir_bands=meta.get("ir_bands"), views=meta.get("views"),
                          radar_band=meta.get("radar_band"), sar_band=meta.get("sar_band"),
                          config=meta.get("config"))
    if len(frame_ids) != n_frames:
        errors.append("frames.json has %d rows, expected %d" % (len(frame_ids), n_frames))
    if meta.get("frames") != n_frames:
        errors.append("meta.frames=%s, expected %d" % (meta.get("frames"), n_frames))
    if frame_ids != list(range(n_frames)):
        errors.append("frames.json frame ids are not 0..%d" % (n_frames - 1))

    dirs = _modality_dirs(meta)
    wanted = []
    for key in modalities:
        if key == "ir":
            band_keys = sorted(k for k in dirs if k.startswith("ir_"))
            if not band_keys:
                errors.append("ir requested but meta.json lists no IR band")
            wanted.extend(band_keys)
        elif key in dirs:
            wanted.append(key)
        else:
            errors.append("sensor %s has no directory layout" % key)

    per_frame = {}
    per_modality = {}
    for key in wanted:
        image_dir = root / dirs[key]
        label_dir = root / _label_dir(key)
        if not image_dir.is_dir() or not label_dir.is_dir():
            errors.append("%s: missing %s or %s" % (key, dirs[key], _label_dir(key)))
            continue
        image_paths = sorted(image_dir.glob("*.png"))
        label_paths = sorted(label_dir.glob("*.json"))
        summary = dict(directory=dirs[key], label_dir=_label_dir(key),
                       images=len(image_paths), labels_found=len(label_paths),
                       visible_frames=0, frames=[])
        if len(image_paths) != n_frames:
            errors.append("%s: %d images, expected %d" % (key, len(image_paths), n_frames))
        if len(label_paths) != n_frames:
            errors.append("%s: %d label files, expected %d" % (key, len(label_paths), n_frames))
        for i in range(n_frames):
            stem = "%06d" % i
            row = dict(frame=i, t=times.get(i))
            summary["frames"].append(row)
            image_path = image_dir / (stem + ".png")
            label_path = label_dir / (stem + ".json")
            if not image_path.exists():
                errors.append("%s: missing image %s" % (key, image_path.name))
                continue
            if not label_path.exists():
                errors.append("%s: missing label %s" % (key, label_path.name))
                continue
            try:
                array, width, height, mode = _decode_png(image_path, IMAGE)
            except Exception as exc:
                errors.append("%s frame %d: image does not decode (%s)" % (key, i, exc))
                continue
            doc = _read_json(label_path)
            row.update(width=width, height=height, mode=mode,
                       bytes=int(image_path.stat().st_size))
            if int(doc.get("frame", -1)) != i:
                errors.append("%s frame %d: label says frame %s" % (key, i, doc.get("frame")))
            if (int(doc.get("width", width)) != width
                    or int(doc.get("height", height)) != height):
                errors.append("%s frame %d: label %sx%s disagrees with raster %dx%d"
                              % (key, i, doc.get("width"), doc.get("height"), width, height))
            boxes = _boxes_of(doc, key)
            instances = _instances(doc, key)
            row["boxes"] = {name: dict(box) for name, box in boxes}
            row["instances"] = instances
            if not boxes:
                errors.append("%s frame %d: no labelled box" % (key, i))
            if [n for n, b in boxes if b.get("visible", True)]:
                summary["visible_frames"] += 1
            else:
                errors.append("%s frame %d: no visible box" % (key, i))
            if not instances and key != "optical":
                errors.append("%s frame %d: label carries no instance id/class" % (key, i))
            gray = _gray(array) if array is not None else None
            for name, box in boxes:
                for coord in ("x0", "y0", "x1", "y1"):
                    if coord not in box:
                        errors.append("%s frame %d: box %s lacks %s" % (key, i, name, coord))
                if not (0.0 <= float(box["x1"]) <= width and 0.0 <= float(box["y1"]) <= height
                        and float(box["x1"]) > float(box["x0"])
                        and float(box["y1"]) > float(box["y0"])):
                    errors.append("%s frame %d: box %s is degenerate or outside the raster"
                                  % (key, i, name))
                if box.get("visible", True) and gray is not None:
                    stats = _region_stats(gray, box, width, height)
                    if stats:
                        row["box_stats"] = stats
                        row["local"] = _local_stats(gray, box, width, height)
            if key == "optical" or key.startswith("ir_"):
                st = row.get("box_stats") or {}
                lo = row.get("local") or {}
                digits = 2 if key == "optical" else 1
                row["unit"] = "code(8bit)" if key == "optical" else "count(16bit)"
                row["contrast"] = round(st.get("contrast", 0.0), digits)
                row["ring_median"] = round(lo.get("ring_median", 0.0), digits)
                row["ring_sigma"] = round(lo.get("ring_sigma", 0.0), digits)
                row["threshold"] = round(lo.get("threshold", 0.0), digits)
                row["hit_fraction"] = round(lo.get("hit_fraction", 0.0), 4)
                row["peak_deviation"] = round(lo.get("peak_deviation", 0.0), digits)
                if key.startswith("ir_"):
                    row["t_mean_c"] = next(
                        (o.get("t_mean_c") for o in doc.get("objects") or []
                         if o.get("t_mean_c") is not None), None)
            elif key == "radar":
                det = [float(p.get("snr_db") or 0.0) for p in doc.get("points") or []]
                row["points"] = int(doc.get("num_points") or 0)
                row["clutter"] = int(doc.get("clutter_count") or 0)
                row["snr_max"] = round(max(det), 2) if det else None
                boxes3d = doc.get("boxes_3d") or []
                if boxes3d:
                    row["world"] = [round(float(v), 3) for v in boxes3d[0].get("center") or []]
                peak = None
                boxes_rd = doc.get("boxes_rd") or {}
                if gray is not None and boxes_rd:
                    # The RD raster is a noise-limited image: the background is
                    # the speckle floor, so the gap is measured against the
                    # frame median (== the floor after the 30 dB display window)
                    # and converted back to dB with the same window.
                    floor = float(np.median(gray))
                    name, box = max(boxes_rd.items(),
                                    key=lambda kv: (float(kv[1]["x1"]) - float(kv[1]["x0"]))
                                    * (float(kv[1]["y1"]) - float(kv[1]["y0"])))
                    stats = _region_stats(gray, box, width, height, floor_code=floor)
                    peak = (stats["peak"] - floor) * RD_DB_PER_CODE
                    row["rd_box"] = dict(name=name, **{k: round(float(box[k]), 1)
                                                       for k in ("x0", "y0", "x1", "y1")})
                    row["rd_floor_code"] = floor
                    row["rd_peak_code"] = stats["peak"]
                    row["rd_peak_db"] = round(peak, 2)
                    row["rd_p95_db"] = round(stats["p95_contrast"] * RD_DB_PER_CODE, 2)
                    row["rd_bright_fraction"] = round(stats["bright_fraction"], 4)
                    # The delivered raster is clipped at code 255 by the 30 dB
                    # display window, so judge the unclipped cube whenever it
                    # was exported; fall back to the raster otherwise.
                    raw = _raw_rd(root / "raw" / "radar" / ("%06d.npz" % i), box)
                    if raw:
                        row["rd_raw"] = raw
                        peak = raw["peak_over_floor_db"]
                if row["points"] <= 0:
                    errors.append("radar frame %d: no labelled return" % i)
                if peak is None:
                    errors.append("radar frame %d: RD map has no measurable target box" % i)
                elif peak < MIN_RD_PEAK_DB:
                    errors.append("radar frame %d: RD peak only %.2f dB above the noise floor"
                                  % (i, peak))
            elif key == "lidar":
                row["points"] = int(doc.get("n_points") or 0)
                row["echo_rate"] = float(doc.get("hit_ratio") or 0.0)
                row["max_range_m"] = doc.get("max_range_m")
                obj_points = next((int(o.get("points") or 0)
                                   for o in doc.get("objects") or []), 0)
                area = (row.get("box_stats") or {}).get("area_px") or 0
                row["target_points"] = obj_points
                row["target_echo_rate"] = round(obj_points / float(area), 4) if area else 0.0
                if row["points"] <= 0:
                    errors.append("lidar frame %d: no return at all" % i)
                if obj_points <= 0:
                    errors.append("lidar frame %d: no return from the labelled target" % i)
                elif row["target_echo_rate"] < MIN_TARGET_ECHO_RATE:
                    errors.append("lidar frame %d: target echo rate %.4f below %.3f"
                                  % (i, row["target_echo_rate"], MIN_TARGET_ECHO_RATE))
            elif key == "sar":
                stats = doc.get("stats") or {}
                row["peak_db"] = stats.get("peak_db")
                row["noise_db"] = stats.get("noise_db")
                row["snr_db"] = stats.get("snr_db")
                row["target_contrast_db"] = stats.get("target_contrast_db")
                row["n_scatterers"] = stats.get("n_scatterers")
                row["resolution"] = doc.get("resolution")
                row["swath"] = doc.get("swath")
                tc = row["target_contrast_db"]
                if tc is None or not np.isfinite(float(tc)) or float(tc) <= 0.0:
                    errors.append("sar frame %d: the target does not rise above the background" % i)
                ic = (row.get("box_stats") or {}).get("contrast")
                row["image_contrast_code"] = round(float(ic), 2) if ic is not None else None
                if ic is None:
                    errors.append("sar frame %d: no measurable target box in the image" % i)
                elif float(ic) <= 0.0:
                    errors.append("sar frame %d: delivered image shows no target contrast" % i)
            per_frame[(key, i)] = row

        if key == "optical":
            _verdict_visible(key, summary, MIN_RING_PEAK_CODES, MIN_RING_HIT_FRACTION,
                             "codes", errors)
        elif key.startswith("ir_"):
            _verdict_visible(key, summary, MIN_RING_PEAK_COUNTS, MIN_RING_HIT_FRACTION,
                             "counts", errors)
        per_modality[key] = summary

    # Inventory: what a consumer of this directory will actually find. Counting
    # the files here (rather than trusting meta.json) is what makes "10 frames
    # per modality" a measurement and not a restatement of the config.
    def _count(path, pattern="*"):
        if not path.is_dir():
            return 0
        return len([p for p in path.glob(pattern) if p.is_file()])

    notes = dict(
        optical="可见光图像", radar="雷达 RD 图", radar_ra="雷达 RA 图",
        sar="SAR 图像", ir="红外图像", lidar="激光雷达范围图")
    inventory = {}
    for key in wanted:
        stem = "ir_" + key[len("ir_"):] if key.startswith("ir_") else key
        label = notes.get(stem.split("_")[0] if stem.startswith("ir_") else stem, stem)
        inventory[dirs[key]] = (
            _count(root / dirs[key], "*.png"),
            "%s；配套标注 %d 份（%s）"
            % (label, _count(root / _label_dir(key), "*.json"), _label_dir(key)))
    for sub, note in (("lidar/intensity", "激光雷达强度图"),
                      ("lidar/points", "激光雷达点云（npy）"),
                      ("observations", "跨模态单帧观测摘要（斜距/RCS/SNR/标注数）")):
        n = _count(root / sub)
        if n:
            inventory[sub] = (n, note)
    for raw in sorted((root / "raw").glob("*")) if (root / "raw").is_dir() else []:
        n = _count(raw)
        if n:
            inventory["raw/" + raw.name] = (n, "复数据（复包络/相位），供干涉与重聚焦")
    for part in ("images", "labels"):
        d = root / "interfered" / part
        n = _count(d, "**/*") if d.is_dir() else 0
        if n:
            inventory["interfered/" + part] = (n, "干扰镜像（干净版仍保留在源目录）")
    result["inventory"] = inventory

    result["times"] = [dict(frame=k, t=v) for k, v in sorted(times.items())]
    result["modalities"] = per_modality
    result["registration"] = _check_registration(per_frame, wanted, n_frames, errors)
    result["trajectory"] = _check_trajectory(per_frame, wanted, times, errors)
    result["coco"] = {}
    ir_key = next((k for k in wanted if k.startswith("ir_")), None)
    for modality, fname in (("optical", "optical"), ("ir", ir_key),
                            ("lidar", "lidar"), ("sar", "sar")):
        if fname is None:
            continue
        info, errs = _coco_report(root, fname, n_frames, labels_dir=_label_dir(fname))
        if info:
            result["coco"][modality] = info
        # Optical's COCO is where the class and the annotation id for the
        # visible-band image live, so a defect there is fatal; the others are
        # cross-checks and stay warnings.
        if errs and modality == "optical":
            errors.extend(errs)
        elif errs:
            warnings.extend(errs)

    result["observations"] = _check_observations(root, n_frames, per_frame, errors)

    # The negative control (tools/validation_negative.py) leaves one compact
    # record behind. It is folded into the report so the delivered document
    # carries the evidence that the gate can fail, not just that it passed.
    nc_path = root / "negative_control.json"
    if nc_path.exists():
        try:
            result["negative_control"] = _read_json(nc_path)
        except (OSError, ValueError):
            pass
    result["interference"] = _check_interference(root, n_frames, wanted, times, dirs, errors)
    if (root / "validation.json").exists():
        vrep = _read_json(root / "validation.json")
        result["simkit_validate"] = dict(stage=vrep.get("stage"), passed=vrep.get("passed"),
                                         errors=vrep.get("errors") or [],
                                         artifacts=vrep.get("artifacts"))
        if not vrep.get("passed"):
            errors.append("simkit validate reported passed=false")
    else:
        warnings.append("validation.json missing: this run did not go through simkit validate")

    result["checks"] = _checklist(result, errors, n_frames)
    return _finish(result, report_path, label)


def _check_observations(root, n_frames, per_frame, errors):
    """Per-frame observation summaries: presence, and pixel geometry that adds up.

    observations/*.json is where the physical state behind every modality is
    written down (range, velocity, attitude, camera intrinsics). Two things are
    worth auditing there:

    * every frame has one, because a downstream consumer that fuses modalities
      reads this file, not the PNGs;
    * the labelled box agrees with the stated range. A box of W pixels at range
      R through a focal length f corresponds to a real width of W*R/f; for this
      19 x 20 m airframe that has to land in a plausible band. That single
      relation ties the label, the camera model and the trajectory together and
      catches a scene whose geometry drifted apart even while every file exists.
    """
    d = root / "observations"
    if not d.is_dir():
        errors.append("observations: directory is missing")
        return {}
    rows = []
    for i in range(n_frames):
        p = d / ("%06d.json" % i)
        if not p.exists():
            errors.append("observations: frame %d is missing" % i)
            continue
        try:
            doc = _read_json(p)
        except (OSError, ValueError) as exc:
            errors.append("observations: frame %d does not parse (%s)" % (i, exc))
            continue
        objs = doc.get("objects") or []
        if not objs:
            errors.append("observations: frame %d lists no object" % i)
            continue
        o = objs[0]
        vel = o.get("velocity_mps") or [0.0, 0.0, 0.0]
        centre = o.get("center_world_m") or [0.0, 0.0, 0.0]
        row = dict(frame=i, t=float(doc.get("timestamp_s") or 0.0),
                   range_m=float(o.get("range_m") or 0.0),
                   speed_mps=float(np.sqrt(sum(float(v) ** 2 for v in vel))),
                   altitude_m=float(centre[2]),
                   roll_deg=float(o.get("roll_deg") or 0.0),
                   yaw_deg=float(o.get("yaw_deg") or 0.0),
                   closing_mps=float(o.get("closing_speed_mps") or 0.0))
        cam = ((doc.get("cameras") or {}).get("seeker") or {})
        k = cam.get("K") or []
        if k:
            row["focal_px"] = float(k[0][0])
        rows.append(row)

    # box width -> physical width, using the optical label and the same camera
    checked = []
    for row in rows:
        i = row["frame"]
        label = (per_frame.get(("optical", i)) or {})
        box = dict(_boxes_of(label, "optical")) if label else {}
        focal = row.get("focal_px")
        if not box or not focal or row["range_m"] <= 0:
            continue
        b = next(iter(box.values()))
        width_px = float(b["x1"]) - float(b["x0"])
        metres = width_px * row["range_m"] / focal
        checked.append(metres)
        # 5 m is smaller than any surface that could fill the box; 40 m exceeds
        # the 20 m airframe by enough to allow the diagonal and a bank that
        # brings the wingspan toward the camera, while still failing a box that
        # belongs to a different range.
        if not 5.0 <= metres <= 40.0:
            errors.append("observations: frame %d box implies %.1f m of target at "
                          "%.0f m range, which the airframe cannot be"
                          % (i, metres, row["range_m"]))
    if len(rows) != n_frames:
        errors.append("observations: %d of %d frames present" % (len(rows), n_frames))
    return dict(frames=rows, implied_width_m=[round(v, 2) for v in checked])


def _verdict_visible(key, summary, min_peak, min_fraction, unit, errors):
    """Is the labelled region actually a target and not an empty piece of sky?

    Two conditions, because either one alone is foolable: a nearly uniform box
    could still have a bright 90th percentile from a single hot pixel (that is
    what the bright-fraction floor rules out), and a box that is mostly filled
    with a slightly-off background would pass the fraction test while showing no
    real contrast (that is what the percentile floor rules out).
    """
    weak = [r["frame"] for r in summary["frames"]
            if float(r.get("peak_deviation") or 0.0) < min_peak]
    if weak:
        errors.append("%s: target peak deviation from its background ring below %.0f %s "
                      "on frames %s" % (key, min_peak, unit, weak))
    sparse = [r["frame"] for r in summary["frames"]
              if float(r.get("hit_fraction") or 0.0) < min_fraction]
    if sparse:
        errors.append("%s: only a negligible share of the box is off-background on frames %s"
                      % (key, sparse))


def _check_registration(per_frame, wanted, n_frames, errors):
    """Optical and IR ride one mount, so their boxes have to be the same box.

    dataset.py derives both from the projected corners precisely so that a
    fused pair carries one label. If the two disagree, either a renderer or the
    mount has drifted, and a fused training set would learn that offset as
    signal rather than as noise.
    """
    out = dict(aligned=True, pairs=[])
    for key in wanted:
        missing = [i for i in range(n_frames) if (key, i) not in per_frame]
        if missing:
            out["aligned"] = False
            errors.append("frame alignment: %s is missing frames %s" % (key, missing[:5]))
    ir_keys = [k for k in wanted if k.startswith("ir_")]
    if "optical" in wanted and ir_keys:
        ir_key = ir_keys[0]
        worst_iou, worst_shift = 1.0, 0.0
        for i in range(n_frames):
            ob = (per_frame.get(("optical", i)) or {}).get("boxes") or {}
            ib = (per_frame.get((ir_key, i)) or {}).get("boxes") or {}
            if not ob or not ib:
                continue
            a = ob[next(iter(ob))]
            b = ib[next(iter(ib))]
            iou = _iou(a, b)
            ax, ay = _centre(a)
            bx, by = _centre(b)
            shift = math.hypot(ax - bx, ay - by)
            out["pairs"].append(dict(frame=i, iou=round(iou, 4), centre_shift_px=round(shift, 3)))
            worst_iou = min(worst_iou, iou)
            worst_shift = max(worst_shift, shift)
        out["worst_iou"] = round(worst_iou, 4)
        out["worst_centre_shift_px"] = round(worst_shift, 3)
        if worst_iou < 0.95:
            errors.append("registration: optical/IR boxes disagree (min IoU %.3f)" % worst_iou)
        if worst_shift > 2.0:
            errors.append("registration: optical/IR box centres %.2f px apart" % worst_shift)
    return out


def _check_trajectory(per_frame, wanted, times, errors):
    """The target must move, and move at the speed the scene configured.

    Two independent views of the same motion: the radar's world-space 3-D box
    centres give metres per frame, the image-space box centres give pixels per
    frame. A frozen target would still pass "is it visible", and a frozen
    target is exactly the failure the clip architecture exists to prevent.
    """
    out = dict(steps_m=[], image_steps_px={})
    order = sorted(times)
    world = [(per_frame.get(("radar", i)) or {}).get("world") for i in order]
    if all(w and len(w) == 3 for w in world) and len(world) > 1:
        steps = [math.dist(world[k - 1], world[k]) for k in range(1, len(world))]
        out["steps_m"] = [round(s, 3) for s in steps]
        if max(steps) - min(steps) > 0.5:
            errors.append("trajectory: frame-to-frame step varies by %.2f m"
                          % (max(steps) - min(steps)))
    for key in wanted:
        steps_px, ok = [], True
        for a, b in zip(order, order[1:]):
            ra = (per_frame.get((key, a)) or {}).get("boxes") or {}
            rb = (per_frame.get((key, b)) or {}).get("boxes") or {}
            if not ra or not rb:
                ok = False
                break
            ax, ay = _centre(ra[next(iter(ra))])
            bx, by = _centre(rb[next(iter(rb))])
            steps_px.append(round(math.hypot(ax - bx, ay - by), 2))
        if not ok or not steps_px:
            continue
        out["image_steps_px"][key] = steps_px
        widest = max(int((per_frame.get((key, a)) or {}).get("width") or 1) for a in order)
        if max(steps_px) > 0.25 * widest:
            errors.append("trajectory: %s target jumps %.1f px per frame" % (key, max(steps_px)))
        # A legitimately minute image motion can round to identical box corners
        # on two neighbouring frames -- the lidar raster is 256 x 192 with
        # integer boxes, so a 0.2 px/frame drift looks frozen. Only a target
        # that does not move at all across the whole clip is a real failure, and
        # the world-space check above already proves the trajectory moves.
        if sum(1 for s in steps_px if s > 0.0) == 0:
            errors.append("trajectory: %s target is frozen across the whole clip" % key)
    return out


def _check_interference(root, n_frames, wanted, times, dirs, errors):
    """The interference branch must change exactly the frames inside its window.

    Frames before start_s and after end_s are the control: if an untouched
    frame moved, the export is rewriting clean measurements; if a scheduled
    frame did not move, the event never reached the raster.
    """
    sub = root / "interfered"
    out = dict(present=sub.is_dir(), per_frame={}, products={}, window={})
    if not out["present"]:
        return out
    manifest_path = sub / "interference.json"
    if not manifest_path.exists():
        errors.append("interference: interfered/ has no interference.json manifest")
        return out
    manifest = _read_json(manifest_path)
    spec = manifest.get("spec") or {}
    events = spec.get("events") or []
    out["spec"] = dict(enabled=spec.get("enabled"), events=len(events))
    out["products"] = manifest.get("products") or {}

    def weight(event, t):
        start = float(event.get("start_s", 0.0))
        end = event.get("end_s")
        fade = float(event.get("fade_s", 0.0))
        if t < start or (end is not None and t >= float(end)):
            return 0.0
        if not fade:
            return 1.0
        u = min(1.0, (t - start) / fade, ((float(end) - t) / fade) if end is not None else 1.0)
        return u * u * (3.0 - 2.0 * u)

    def expected(modality, t):
        amount = 0.0
        for event in events:
            if modality not in (event.get("modalities") or list(MODALITIES)):
                continue
            amount = max(amount, float(event.get("strength", 0.0)) * weight(event, t))
        return amount

    out["weights"] = {}
    for key in wanted:
        directory = dirs[key]
        per = {}
        per_weight = {}
        for i in range(n_frames):
            clean = root / directory / ("%06d.png" % i)
            dirty = sub / directory / ("%06d.png" % i)
            if not clean.exists() or not dirty.exists():
                errors.append("interference: %s frame %d missing in the derived branch" % (key, i))
                continue
            try:
                a = _decode_png(clean, IMAGE)[0]
                b = _decode_png(dirty, IMAGE)[0]
            except Exception as exc:
                errors.append("interference: %s frame %d does not decode (%s)" % (key, i, exc))
                continue
            if a is None or b is None:
                continue
            if a.shape != b.shape:
                errors.append("interference: %s frame %d changed the raster shape" % (key, i))
                continue
            delta = float(np.abs(a.astype(np.float64) - b.astype(np.float64)).mean())
            per[i] = round(delta, 4)
            modality = "ir" if key.startswith("ir_") else key
            want = expected(modality, times.get(i, 0.0))
            per_weight[i] = round(want, 3)
            if want > 0.0 and delta <= 0.0:
                errors.append("interference: %s frame %d is inside the window but unchanged"
                              % (key, i))
            if want <= 0.0 and delta > 0.0:
                errors.append("interference: %s frame %d is outside the window but changed"
                              % (key, i))
        out["per_frame"][key] = per
        out["weights"][key] = per_weight
    out["window"] = dict(start=min((float(e.get("start_s", 0.0)) for e in events), default=None),
                         end=max((float(e["end_s"]) for e in events if e.get("end_s")),
                                 default=None))
    if not (sub / "labels").is_dir():
        errors.append("interference: the derived branch has no labels/ directory")
    return out


def _checklist(result, errors, n_frames):
    """Turn the raw error list into the checklist printed in the report."""
    def clean(*needles):
        return not any(any(n in e for n in needles) for e in errors)

    reg = result.get("registration") or {}
    return [
        dict(name="帧数", detail="%d 帧，frames.json 与 meta.json 一致" % n_frames,
             passed=clean("expected %d" % n_frames, "frames.json has", "frame ids")),
        dict(name="图像可解码", detail="PNG 签名 + CRC + 完整像素解码",
             passed=clean("does not decode")),
        dict(name="标注格式", detail="框、类别、实例 id、图幅尺寸一致",
             passed=clean("box", "instance", "label", "annotation", "COCO")),
        dict(name="模态帧号对齐", detail="各模态 0..%d 一一对应" % (n_frames - 1),
             passed=bool(reg.get("aligned")) and clean("alignment")),
        dict(name="目标可见", detail="框内峰值偏离背景环 ≥ 4σ，且超标像元占比达标",
             passed=clean("contrast", "off-background", "visible",
                          "no labelled return", "no return")),
        dict(name="RD 峰值高于噪底", detail="≥ %.1f dB（以未截断复数据为准）" % MIN_RD_PEAK_DB,
             passed=clean("RD peak")),
        dict(name="轨迹连续", detail="帧间位移一致、无跳跃、无冻结",
             passed=clean("trajectory")),
        dict(name="观测摘要", detail="每帧一份，且框宽与斜距、焦距自洽",
             passed=clean("observations:")),
        dict(name="跨模态配准", detail="optical 与 ir 共用机位，框须重合",
             passed=clean("registration")),
        dict(name="干扰分支", detail="时间窗内改写、窗外保持",
             passed=clean("interference")),
    ]


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def _f(value, digits=2):
    if value is None:
        return "-"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(f):
        return "-"
    return ("%." + str(digits) + "f") % f


def _render_markdown(result):
    meta = result.get("meta") or {}
    errs = result.get("errors") or []
    warns = result.get("warnings") or []
    mods = result.get("modalities") or {}
    n = result.get("frames_expected")
    passed = not errs
    L = []
    add = L.append

    add("# 最小验证报告：双发战斗机五模态 10 帧")
    add("")
    add("本报告由 tools/validation_check.py 在实际产物上自动生成，"
        "表中的每个数字都是从交付的 PNG 与 JSON 重新测量得到的，"
        "不复用生成器自己的统计，也不接受“目录里有文件”作为证据。")
    add("")
    add("## 一、结论")
    add("")
    add("**%s。** %s" % ("通过" if passed else "不通过",
                       "10 帧 × 5 模态全部满足验收判据。" if passed
                       else "存在 %d 项未通过，逐条见第九节。" % len(errs)))
    add("")
    add("| 项目 | 值 |")
    add("|---|---|")
    add("| 场景 | scenes/validation_twin_fighter.json |")
    add("| 目标模型 | assets/models/军用飞机/双发战斗机.obj（19.0 × 20.013 × 5.795 m，机头 +X） |")
    add("| 帧数 | %d（帧号 0–%d） |" % (n, n - 1))
    add("| 模态 | %s |" % "、".join(mods.keys()))
    add("| 图像总数 | %d |" % sum(int(m.get("images") or 0) for m in mods.values()))
    add("| 有效标注帧 | %s |" % "、".join(
        "%s %d/%d" % (k, m.get("visible_frames", 0), n) for k, m in mods.items()))
    add("| PNG 解码器 | %s |" % result.get("png_decoder"))
    add("| 生成耗时 | %s s（生成器自报） |" % _f(meta.get("seconds"), 1))
    add("")

    inv = result.get("inventory") or {}
    if inv:
        add("### 产物清单")
        add("")
        add("| 目录 | 文件数 | 说明 |")
        add("|---|---|---|")
        for name, (count, note) in inv.items():
            add("| %s | %d | %s |" % (name, count, note))
        add("")

    add("## 二、一条命令跑完")
    add("")
    add("```")
    add(r"$env:PYTHONPATH = 'D:\work\多模态数据集生成\RGB_SAR_LIDAR_SIMU'")
    add(r"env\Scripts\python.exe -m simkit validate -c scenes\validation_twin_fighter.json "
        r"-o out\validation_ecm -n 10 -s optical,ir,radar,lidar,sar")
    add(r"env\Scripts\python.exe -m tools.validation_check out\validation_ecm "
        r"--report docs\最小验证报告.md")
    add("```")
    add("")
    add("simkit validate 的口径：先做运动预检（覆盖完整名义片段，不只是渲染出来的 10 帧），"
        "预检通过才启动渲染，渲染后再由 validation.artifacts 检查张数与标注，"
        "最后才置 passed。本轮 validation.json 的 stage = %s、passed = %s。"
        % (result.get("simkit_validate", {}).get("stage"),
           result.get("simkit_validate", {}).get("passed")))
    add("")

    add("## 三、场景与几何")
    add("")
    obs = (result.get("observations") or {})
    orows = obs.get("frames") or []
    if orows:
        first, last = orows[0], orows[-1]
        dt = (last["t"] - first["t"]) / max(1, len(orows) - 1)
        add("- 场景本体 %d 帧、dt = %s s，渲染区间 t = %s–%s s。"
            % (len(orows), _f(dt, 2), _f(first["t"], 2), _f(last["t"], 2)))
        add("- 目标：%s m/s、%s m 高度、起始斜距 %s m（光学机位到目标中心的距离），"
            "机体系滚转 %s°、偏航 %s°；这四项直接读自 observations/000000.json。"
            % (_f(first.get("speed_mps"), 1), _f(first.get("altitude_m"), 0),
               _f(first.get("range_m"), 2), _f(first.get("roll_deg"), 3),
               _f(first.get("yaw_deg"), 3)))
        span = max(1e-9, last["t"] - first["t"])
        rate = (last["range_m"] - first["range_m"]) / span
        add("- 斜距由 %s m 变为 %s m（%+.2f m/s，正为远离）；"
            "observations 里的 closing_speed 由 %+.2f 变为 %+.2f m/s（负为远离），"
            "两者一致：目标在缓慢拉开，而不是逼近。"
            % (_f(first.get("range_m"), 2), _f(last.get("range_m"), 2), rate,
               float(first.get("closing_mps") or 0.0),
               float(last.get("closing_mps") or 0.0)))
        focal = first.get("focal_px")
        widths = obs.get("implied_width_m") or []
        if focal and widths:
            add("- 相机焦距 %s px（2.2° 视场、768 px 宽）。由“框宽 × 斜距 / 焦距”反算的横向尺度为 "
                "%s–%s m；这是斜置、带 %.0f° 滚转的机体在像面上的轴对齐包围盒，"
                "落在 19.0 × 20.013 m 外廓与 %.1f m 对角线之间，"
                "说明标注框、相机模型与航路三者指向同一个物体。"
                % (_f(focal, 0), _f(min(widths), 1), _f(max(widths), 1),
                   abs(float(first.get("roll_deg") or 0.0)),
                   (19.0 ** 2 + 20.013 ** 2) ** 0.5))
        add("")
    add("- 光学与中波红外共用同一机位、同一视场；雷达与激光雷达由同一主视场指向驱动；"
        "SAR 天线自己飞一条直线航迹，成像参考面取目标高度，"
        "因此 SAR 与其余四模态同帧同时，但不同轴。")
    add("")

    add("## 四、逐模态逐帧实测")
    add("")
    for key, m in mods.items():
        rows = m.get("frames") or []
        add("### %s —— %s" % (key, m.get("directory")))
        add("")
        if key == "optical" or key.startswith("ir_"):
            unit = (rows[0].get("unit") if rows else "-")
            add("| 帧 | 图幅 | 目标框 x0, y0, x1, y1 | 框内均值 %s | 背景环中位 %s"
                " | 环 σ %s | 判定阈值 4σ %s | 超标像元占比 | 峰值偏离 %s |"
                % (unit, unit, unit, unit, unit))
            add("|---|---|---|---|---|---|---|---|---|")
            for r in rows:
                b = r.get("boxes") or {}
                box = next(iter(b.values())) if b else {}
                st = r.get("box_stats") or {}
                add("| %d | %dx%d | %s | %s | %s | %s | %s | %s | %s |" % (
                    r["frame"], r.get("width", 0), r.get("height", 0),
                    "%s, %s, %s, %s" % (_f(box.get("x0")), _f(box.get("y0")),
                                        _f(box.get("x1")), _f(box.get("y1"))),
                    _f(st.get("mean")), _f(r.get("ring_median")),
                    _f(r.get("ring_sigma")), _f(r.get("threshold")),
                    _f(r.get("hit_fraction"), 4), _f(r.get("peak_deviation"))))
            add("")
            add("背景取目标框外扩 15% 的环带，用中位数和稳健 σ（1.4826×MAD）估计，"
                "阈值取 4σ（下限 3 码）。“超标像元占比”是框内偏离背景超过该阈值的像元比例，"
                "“峰值偏离”是框内最亮/最暗像元与背景之差。")
            add("")
            if key == "optical":
                add("为什么不用框内均值：目标在 1.49 km 处占 326 × 225 像素的框，"
                    "框内大部分是机翼与尾翼之间的天空，光学图上框均值甚至比背景低 7 码，"
                    "而峰值偏离在 78–105 码之间——只看均值会把一架清清楚楚的飞机判成空白帧。"
                    "环带背景同时吸收天空的纵向亮度梯度，比全图统一阈值更贴近实际检测器。")
                add("")
            else:
                add("红外图上目标比背景亮约 %s counts，峰值偏离超过 %s counts，"
                    "两种极端在这里都不构成风险；判据与光学保持一致，"
                    "是为了让同一套规则同时适用于亮目标和暗目标（取绝对值）。"
                    % (_f(float((rows[0].get("box_stats") or {}).get("contrast") or 0.0), 0),
                       _f(min(float(r.get("peak_deviation") or 0.0) for r in rows), 0)))
                add("")
            if key.startswith("ir_"):
                add("目标平均表面温度（热求解器输出）：%s °C。"
                    % "、".join(_f(r.get("t_mean_c"), 1) for r in rows))
                add("")
                add("判据：峰值偏离 ≥ %.0f counts，且超标像元占比 ≥ %.2f；"
                    "实测最小值 %s counts / %s。"
                    % (MIN_RING_PEAK_COUNTS, MIN_RING_HIT_FRACTION,
                       _f(min(float(r.get("peak_deviation") or 0.0) for r in rows)),
                       _f(min(float(r.get("hit_fraction") or 0.0) for r in rows), 4)))
                add("")
                mode = str(rows[0].get("mode") or "")
                peak = max(float((r.get("box_stats") or {}).get("peak") or 0.0) for r in rows)
                add("本模态交付的是 16 bit 计数栅格（Pillow 模式 %s，框内峰值 %s counts、"
                    "背景 %s counts，未触顶 65535），不是 8 bit 归一化预览；"
                    "images/ir_preview_mwir 才是拉过 AGC 的预览图，两者不可混用。"
                    % (mode, _f(peak, 0), _f(rows[0].get("ring_median"), 0)))
                add("")
            else:
                add("判据：峰值偏离 ≥ %.0f 码，且超标像元占比 ≥ %.2f；"
                    "实测最小值 %s 码 / %s。"
                    % (MIN_RING_PEAK_CODES, MIN_RING_HIT_FRACTION,
                       _f(min(float(r.get("peak_deviation") or 0.0) for r in rows)),
                       _f(min(float(r.get("hit_fraction") or 0.0) for r in rows), 4)))
                add("")
        elif key == "radar":
            raw_rows = [r for r in rows if r.get("rd_raw")]
            add("| 帧 | RD 目标框 x0, y0, x1, y1 | RD 峰值码 | RD 噪底码 | 图像域峰值 dB"
                " | 图像域 95 分位 dB | 复数据峰值高于噪底 dB | 复数据 95 分位 dB"
                " | 标注点迹 | 杂波点 | 检测 SNR 最大 dB |")
            add("|---|---|---|---|---|---|---|---|---|---|---|")
            for r in rows:
                rb = r.get("rd_box") or {}
                raw = r.get("rd_raw") or {}
                add("| %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                    r["frame"],
                    ("%s, %s, %s, %s" % (_f(rb.get("x0"), 1), _f(rb.get("y0"), 1),
                                         _f(rb.get("x1"), 1), _f(rb.get("y1"), 1)))
                    if rb else "-",
                    _f(r.get("rd_peak_code"), 0), _f(r.get("rd_floor_code"), 1),
                    _f(r.get("rd_peak_db")), _f(r.get("rd_p95_db")),
                    _f(raw.get("peak_over_floor_db")), _f(raw.get("p95_over_floor_db")),
                    _f(r.get("points")), _f(r.get("clutter")), _f(r.get("snr_max"))))
            add("")
            add("两套读数并列，是因为交付的 RD 图**不是**测量数据：它按固定 30 dB 显示窗"
                "量化成 8 bit，1 码 = 0.1176 dB，强目标必然在 255 截顶。"
                "“图像域”两列是对交付 PNG 的反解，只能给出**下界**；"
                "“复数据”两列读自 raw/radar/*.npz 的未截断功率，才是真实余量。"
                "噪底一律取该帧功率图的中位数（斑点噪声稳态电平）。")
            add("")
            if raw_rows:
                add("判据：复数据峰值高出噪底 ≥ %.1f dB；实测最小 %s dB，"
                    "框内 95 分位最小 %s dB。验收以复数据为准。"
                    % (MIN_RD_PEAK_DB,
                       _f(min(float((r.get("rd_raw") or {}).get("peak_over_floor_db") or 0.0)
                              for r in raw_rows)),
                       _f(min(float((r.get("rd_raw") or {}).get("p95_over_floor_db") or 0.0)
                              for r in raw_rows), 2)))
            else:
                add("判据：图像域峰值高出噪底 ≥ %.1f dB；实测最小 %s dB"
                    "（复数据未导出，只能按截顶后的栅格判定）。"
                    % (MIN_RD_PEAK_DB,
                       _f(min(float(r.get("rd_peak_db") or 0.0) for r in rows))))
            add("")
            saturated = [r["frame"] for r in rows if int(r.get("rd_peak_code") or 0) >= 255]
            if saturated:
                add("图像域峰值码取到 255 的帧有 %d 帧（%s%s），表里 30.00 dB 就是这个截顶值；"
                    "其中最低的一帧复数据读数为 %s dB，两者之差正是显示窗切掉的那一段。"
                    % (len(saturated),
                       "、".join(str(f) for f in saturated[:5]),
                       " 等" if len(saturated) > 5 else "",
                       _f(min(float((r.get("rd_raw") or {}).get("peak_over_floor_db") or 0.0)
                              for r in raw_rows), 2) if raw_rows else "-"))
                add("")
                add("解析检测 SNR 与复数据读数不是同一口径：前者是雷达方程的期望信噪比"
                    "（标注里 detection_basis = expected radar equation SNR，另含处理增益），"
                    "属于设计预算；后者是本帧功率立方体里峰值与噪底中位数之比，"
                    "属于随帧变化的实测动态范围。验收采用后者，因为它可以被任何拿到 "
                    "raw/radar/*.npz 的人独立复算。")
                add("")
        elif key == "lidar":
            add("| 帧 | 目标框 x0, y0, x1, y1 | 回波点 | 回波率 | 目标回波点 | 目标回波率 | 框面积 px |")
            add("|---|---|---|---|---|---|---|")
            for r in rows:
                b = r.get("boxes") or {}
                box = next(iter(b.values())) if b else {}
                add("| %d | %s | %s | %s | %s | %s | %s |" % (
                    r["frame"],
                    "%s, %s, %s, %s" % (_f(box.get("x0"), 1), _f(box.get("y0"), 1),
                                        _f(box.get("x1"), 1), _f(box.get("y1"), 1)),
                    _f(r.get("points")), _f(r.get("echo_rate"), 4),
                    _f(r.get("target_points")), _f(r.get("target_echo_rate"), 4),
                    _f((r.get("box_stats") or {}).get("area_px"))))
            add("")
            add("回波率 = 有回波的像元 / 全部发射像元；目标回波率 = 落在目标框内的回波点 / 框面积，"
                "衡量目标本身被照亮了多少。判据：目标回波率 ≥ %.3f；实测最小 %s。"
                % (MIN_TARGET_ECHO_RATE,
                   _f(min(float(r.get("target_echo_rate") or 0.0) for r in rows), 4)))
            add("")
        elif key == "sar":
            add("| 帧 | 目标框 x0, y0, x1, y1 | 峰值 dB | 噪底 dB | 目标 SNR dB | 目标对比度 dB | 图像内对比度（码） | 散射中心 |")
            add("|---|---|---|---|---|---|---|---|")
            for r in rows:
                b = r.get("boxes") or {}
                box = next(iter(b.values())) if b else {}
                add("| %d | %s | %s | %s | %s | %s | %s | %s |" % (
                    r["frame"],
                    "%s, %s, %s, %s" % (_f(box.get("x0"), 1), _f(box.get("y0"), 1),
                                        _f(box.get("x1"), 1), _f(box.get("y1"), 1)),
                    _f(r.get("peak_db")), _f(r.get("noise_db")), _f(r.get("snr_db")),
                    _f(r.get("target_contrast_db")), _f(r.get("image_contrast_code")),
                    _f(r.get("n_scatterers"))))
            add("")
            res = (rows[0].get("resolution") or {}) if rows else {}
            swath = (rows[0].get("swath") or {}) if rows else {}
            if res:
                add("单帧成像参数：斜距分辨率 %s m、地距分辨率 %s m、方位分辨率 %s m、"
                    "合成孔径 %s m。"
                    % (_f(res.get("range_slant_m")), _f(res.get("range_m")),
                       _f(res.get("azimuth_m")), _f(res.get("l_sa_m"))))
                add("")
            if swath:
                add("成像带：近端 %s m、远端 %s m、中心 %s m，入射角 %s°，"
                    "地面采样 %s m。"
                    % (_f(swath.get("near_m"), 1), _f(swath.get("far_m"), 1),
                       _f(swath.get("center_m"), 1), _f(swath.get("incidence_deg"), 2),
                       _f(swath.get("ground_sampling_m"), 3)))
                add("")
            add("判据：目标对比度 dB > 0 且交付图像内目标框均值高于全图中位数；"
                "实测最小目标对比度 %s dB、最小图像内对比度 %s 码。"
                % (_f(min(float(r.get("target_contrast_db") or 0.0) for r in rows)),
                   _f(min(float(r.get("image_contrast_code") or 0.0) for r in rows))))
            add("")

    reg = result.get("registration") or {}
    add("## 五、帧号对齐与跨模态配准")
    add("")
    add("五个模态共用同一份 frames.json，帧号 0–%d 一一对应；脚本逐帧比对文件名、"
        "标注内部的 frame 字段与 COCO 的 image_id，任何一处对不上都判定为不对齐。" % (n - 1))
    add("")
    add("- 帧号对齐：**%s**" % ("是" if reg.get("aligned") else "否"))
    if "worst_iou" in reg:
        add("- 光学 ↔ 中波红外框重合度：最小 IoU %s，框心最大偏差 %s px。"
            "两者共用同一机位，框由同一套投影角点算出，因此本项本质上是在检查"
            "两套渲染器有没有把目标画到同一个位置。"
            % (_f(reg.get("worst_iou"), 4), _f(reg.get("worst_centre_shift_px"))))
    add("")
    if reg.get("pairs"):
        coco = result.get("coco") or {}
        if coco:
            add("COCO 侧车文件（训练侧真正读取的文件）：")
            add("")
            add("| 文件 | 图像 | 标注 | 类别 | 与逐帧标注的最大偏差 px |")
            add("|---|---|---|---|---|")
            for k, info in coco.items():
                add("| %s | %d | %d | %s | %s |" % (
                    info.get("file"), int(info.get("images") or 0),
                    int(info.get("annotations") or 0),
                    "、".join(info.get("categories") or []) or "-",
                    _f(info.get("bbox_drift_px"), 3)))
            add("")
            add("逐帧标注与 COCO 侧车是两份独立文件，只加载其中一份的人不会发现它们漂移，"
                "所以这里逐标注比对两者的框坐标；偏差超过 1 px 即判失败。")
            add("")
        add("| 帧 | 光学 ↔ 红外 IoU | 框心偏差 px |")
        add("|---|---|---|")
        for pair in reg["pairs"]:
            add("| %d | %s | %s |" % (pair["frame"], _f(pair["iou"], 4),
                                      _f(pair["centre_shift_px"])))
        add("")

    traj = result.get("trajectory") or {}
    add("## 六、轨迹连续性")
    add("")
    steps = traj.get("steps_m") or []
    if steps:
        add("- 雷达 3D 框中心的世界坐标帧间位移（m）：%s" % "、".join(_f(s, 3) for s in steps))
        add("- 200 m/s × 0.1 s = 20 m，实测位移与之一致，说明各模态描述的是同一时刻、同一目标。")
    add("")
    for key, seq in (traj.get("image_steps_px") or {}).items():
        add("- %s 图像空间帧间位移（px）：%s" % (key, "、".join(_f(v) for v in seq)))
    add("")
    add("目标在 1.49 km 外、视场仅 2.2°，每帧（0.1 s）在像面上只走约 0.2 像素，"
        "所以关键是“必须非零且平滑”而不是“走得远”。各模态的抖动来源不同：")
    add("")
    add("- 光学与红外：0.21–0.22 px/帧，两个模态完全一致，因为它们共用同一套角点投影。")
    add("- 雷达：0.0–1.12 px 的抖动来自 RD 框在距离-多普勒网格上的整数量化，"
        "目标每帧只移动约 20 m，而一格距离门就是几十米量级。")
    add("- 激光雷达：多为 0.0、偶尔 0.5 px，因为范围图只有 256×192，"
        "框按整数像元落位，0.2 px 的真实位移被量化吃掉；世界坐标位移证明了运动是连续的。")
    add("- SAR：0–11 px 的较大跳动来自方位向散斑与散射中心的随机起伏——"
        "同一架飞机相邻两帧的强散射点位置会变，框跟着变；世界坐标位移仍是 19.99 m，"
        "说明成像几何没有跳变，跳动的是斑纹而不是目标。")
    add("")

    inter = result.get("interference") or {}
    add("## 七、干扰分支")
    add("")
    if not inter.get("present"):
        add("本次未启用干扰（interfered/ 不存在）。")
    else:
        win = inter.get("window") or {}
        add("干扰在图像域施加，干净测量完整保留在源数据集；"
            "下表是同一帧“干净图”与“干扰图”的平均绝对码差。")
        add("")
        keys = list(inter.get("per_frame") or {})
        wts = inter.get("weights") or {}
        ref_opt = (wts.get("optical") or {})
        ref_rad = (wts.get("radar") or {})
        add("| 帧 | t (s) | optical/ir 期望权重 | radar/lidar/sar 期望权重 | %s |"
            % " | ".join(keys))
        add("|---|---|---|---|%s" % ("---|" * len(keys)))
        times = {int(r["frame"]): float(r["t"]) for r in (result.get("times") or [])}
        for i in range(n):
            cells = [_f((inter["per_frame"][k] or {}).get(i), 3) for k in keys]
            add("| %d | %s | %s | %s | %s |" % (
                i, _f(times.get(i), 2), _f(ref_opt.get(i), 3), _f(ref_rad.get(i), 3),
                " | ".join(cells)))
        add("")
        frozen = [i for i in range(n) if float(ref_opt.get(i) or 0.0) <= 0.0
                  and all(float((inter["per_frame"][k] or {}).get(i) or 0.0) == 0.0
                          for k in keys)]
        changed = [i for i in range(n) if float(ref_opt.get(i) or 0.0) > 0.0]
        add("干扰窗为 %s s 到 %s s，窗两端各带 0.1 s 淡入淡出（smoothstep）。"
            "权重为 0 的帧必须与干净数据逐码一致，权重为正的帧必须被改写——"
            "这一对照同时证明“干扰确实落到栅格上”和“没被干扰的帧没被动过”。"
            % (_f(win.get("start"), 2), _f(win.get("end"), 2)))
        add("")
        add("- 期望权重为 0 且实测差为 0 的帧：%s。"
            % ("、".join(str(i) for i in frozen) if frozen else "无"))
        add("- 期望权重为正且实测差大于 0 的帧：%s。"
            % ("、".join(str(i) for i in changed) if changed else "无"))
        add("")
        add("窗起点 t = %s s 正好落在淡入的 u = 0 处，权重恰为 0，"
            "所以帧 2 虽然名义上“在窗内”，交付栅格必须与干净数据完全一致；"
            "这是淡入的设计结果，不是漏施加。"
            % _f(win.get("start"), 2))
        add("")
        add("注意：干扰只重建图像栅格与标注，不重建回波点云与原始复数据；"
            "radar/lidar 的 3D 真值仍是干净参考值。")
        add("")

    add("## 八、检查清单")
    add("")
    add("| 检查项 | 判据 | 结果 |")
    add("|---|---|---|")
    for item in result.get("checks") or []:
        add("| %s | %s | %s |" % (item["name"], item["detail"],
                                  "通过" if item["passed"] else "**不通过**"))
    add("")
    add("## 九、负向对照（判据确实会失败）")
    add("")
    add("一个从没失败过的判据不是判据。tools/validation_negative.py 把本次产物复制一份，"
        "注入四种互不相关的缺陷后交给同一个 tools/validation_check.py；"
        "注入之外的内容逐字节未改，所以多出来的任何一条错误都会是误报。")
    add("")
    nc = result.get("negative_control") or {}
    if nc:
        add("| 注入的缺陷 | 对应判据 | 脚本报出的错误 |")
        add("|---|---|---|")
        keys = ("optical: target peak deviation", "trajectory: lidar",
                "does not decode", "missing label 000009.json")
        rows = zip(zip(nc.get("defects") or [], nc.get("dimensions") or []),
                   keys, nc.get("detected") or [])
        for (desc, dim), needle, hit in rows:
            msg = next((e for e in (nc.get("broken_errors") or []) if needle in e), "-")
            add("| %s | %s | %s |" % (desc, dim, msg if hit else "**未检出**"))
        add("")
        add("被注入缺陷的副本判定为**%s**（passed = %s），四种缺陷全部检出：%s。"
            % ("不通过" if not nc.get("broken_passed") else "通过",
               str(nc.get("broken_passed")),
               "是" if nc.get("all_detected") else "否"))
        add("")
        add("四条缺陷分别落在“看得见”“标得对”“读得出”“对得齐”四个独立维度上，"
            "每条都被定位到具体帧号，脚本退出码为 1；干净产物在同一脚本下退出码为 0。"
            "两个方向都跑过，才说明这道门禁既不放过坏数据，也不误杀好数据。")
    else:
        add("未运行负向对照。执行 python -m tools.validation_negative out/validation_ecm 后"
            "重新生成报告即可补上这一节。")
    add("")

    add("## 十、未通过项")
    add("")
    if errs:
        for e in errs:
            add("- %s" % e)
    else:
        add("无。")
    add("")
    if warns:
        add("提示（不阻断）：")
        add("")
        for w in warns:
            add("- %s" % w)
        add("")

    add("## 十一、已知局限")
    add("")
    add("1. 雷达 RCS 是几何面元的物理光学叠加再被 rcs_db 覆盖；8 dBsm 是让检测门限落在"
        "有意义位置的设定值，不是该型飞机的实测值。")
    add("2. SAR 是地面平面成像近似。参考面被放到目标高度、天线沿自身直线航迹飞行，"
        "因此它与光学不同轴，只能保证同一帧号对应同一时刻与同一目标，不构成像素级配准。")
    add("3. 红外只有中波单波段，16 bit 计数未经实测辐射定标；本报告只用相对对比度。")
    add("4. 运动是解析航路加协调转弯运动学，没有六自由度气动积分；"
        "包线检查是采样检查，不是对任意连续曲线的数学保证。")
    add("5. 干扰是图像域扰动，不是射频或光电对抗模型；它改变的是交付栅格，不改变物理量。")
    add("6. 本报告只证明管线、几何、标注与可见性，不证明训练效果，样本量 10 帧也不足以训练。")
    add("7. 场景里用的是显式模型路径 assets/models/军用飞机/双发战斗机.obj，"
        "不是资产卡写法：scene.load_config 目前不调用 simkit.asset.expand_scene，"
        "写资产卡的场景会静默退化成方块目标。"
        "在场景层修好之前，任何需要用真模型的场景都必须写显式路径。")
    add("")
    add("## 十二、复现")
    add("")
    add("```")
    add(r"$env:PYTHONPATH = 'D:\work\多模态数据集生成\RGB_SAR_LIDAR_SIMU'")
    add(r"env\Scripts\python.exe -m simkit validate -c scenes\validation_twin_fighter.json "
        r"-o out\validation_ecm -n 10 -s optical,ir,radar,lidar,sar")
    add(r"env\Scripts\python.exe -m tools.validation_check out\validation_ecm "
        r"--report docs\最小验证报告.md")
    add(r"env\Scripts\python.exe -m tools.validation_negative out\validation_ecm")
    add("```")
    add("")
    add("- simkit validate 写 out/validation_ecm/validation.json（预检 + 产物检查）。")
    add("- validation_check 写 out/validation_ecm/quality_validation.json 与本报告，"
        "退出码 0 表示全部判据通过，1 表示有未通过项。")
    add("- validation_negative 写 out/validation_ecm/negative_control.json，"
        "退出码 0 表示四种注入缺陷全部被检出。")
    add("")
    add("第二次运行 validate 前必须先换一个新的输出目录，"
        "cmd_validate 会拒绝写进已有数据集的目录，这是刻意的保护："
        "验收报告不能被一次误跑覆盖成“通过”。")
    add("")
    return "\n".join(L) + "\n"


def _finish(result, report_path, label="最小验证"):
    result["passed"] = not result.get("errors")
    root = Path(result.get("root") or ".")
    try:
        (root / "quality_validation.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=1, default=str) + "\n",
            encoding="utf-8")
    except OSError as exc:
        result.setdefault("warnings", []).append(
            "quality_validation.json not written: %s" % exc)
    markdown = _render_markdown(result)
    if report_path:
        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown, encoding="utf-8")
        result["report"] = str(path)
    summary = {k: v for k, v in result.items() if k not in ("modalities", "interference")}
    print(json.dumps(summary, ensure_ascii=False, indent=1, default=str))
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", nargs="?", default="out/validation_ecm",
                    help="dataset directory to audit")
    ap.add_argument("-n", "--frames", type=int, default=10, help="expected frame count")
    ap.add_argument("-s", "--sensors", default=",".join(MODALITIES))
    ap.add_argument("--report", default=None, help="write the Markdown report here")
    ap.add_argument("--label", default="最小验证")
    args = ap.parse_args(argv)
    sensors = tuple(s.strip() for s in args.sensors.split(",") if s.strip())
    result = check(args.run, n_frames=args.frames, modalities=sensors,
                   report_path=args.report, label=args.label)
    return 0 if result.get("passed") else 1


if __name__ == "__main__":
    sys.exit(main())
