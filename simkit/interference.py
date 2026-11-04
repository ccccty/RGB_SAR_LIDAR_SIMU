"""Deterministic image-domain disturbances for generic dataset robustness tests.

Active/passive are augmentation groups, not RF or countermeasure models.
Clean measurements stay in the source dataset; the derived branch contains
rasters and geometric annotations, never invented raw echoes or point clouds.
"""
import copy
import json
from pathlib import Path
import shutil
import zlib

import numpy as np

MODALITIES = ("optical", "ir", "radar", "lidar", "sar")
GROUPS = dict(additive_noise="active", glare="active", occlusion="passive", contrast_loss="passive")
FIELDS = {"id", "kind", "modalities", "start_s", "end_s", "fade_s", "strength",
          "center_uv", "size_uv", "drift_uv_s", "color"}


def preset(name):
    """Small, explicit presets; strengths refer to the output image code range."""
    active = [dict(id="noise", kind="additive_noise", strength=0.035),
              dict(id="glare", kind="glare", modalities=["optical", "ir"], strength=0.5,
                   center_uv=[0.6, 0.45], size_uv=[0.14, 0.14], drift_uv_s=[0.015, 0])]
    passive = [dict(id="contrast", kind="contrast_loss", strength=0.2),
               dict(id="screen", kind="occlusion", modalities=["optical", "ir", "lidar"],
                    strength=1.0, center_uv=[0.51, 0.5], size_uv=[0.12, 0.45], drift_uv_s=[0.02, 0])]
    choices = dict(off=[], active=active, passive=passive, mixed=active+passive)
    if name not in choices:
        raise ValueError("interference preset must be off, active, passive or mixed")
    return dict(enabled=name != "off", seed=7, events=copy.deepcopy(choices[name]))


def _number(value, name, low=None, high=None):
    if isinstance(value, bool):
        raise ValueError(name+" must be numeric")
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(name+" must be numeric") from None
    if not np.isfinite(value) or (low is not None and value < low) or (high is not None and value > high):
        raise ValueError(name+" is outside its finite range")
    return value


def validate(spec):
    if spec is None:
        return dict(enabled=False, seed=7, events=[])
    if not isinstance(spec, dict) or set(spec)-{"enabled", "seed", "events"}:
        raise ValueError("interference expects enabled, seed and events")
    if not isinstance(spec.get("enabled", True), bool):
        raise ValueError("interference.enabled must be boolean")
    seed = _number(spec.get("seed", 7), "interference.seed", 0, 2**32-1)
    if seed != int(seed):
        raise ValueError("interference.seed must be an integer")
    items = spec.get("events", [])
    if not isinstance(items, list) or len(items) > 64:
        raise ValueError("interference.events must be a list of at most 64 events")
    events, names = [], set()
    for index, item in enumerate(items):
        if not isinstance(item, dict) or set(item)-FIELDS:
            raise ValueError("unknown interference event fields")
        event = dict(item)
        kind = event.get("kind")
        if kind not in GROUPS:
            raise ValueError("unsupported interference kind: "+str(kind))
        name = event.get("id", "event_%d" % index)
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError("interference event IDs must be unique non-empty strings")
        names.add(name)
        defaults = ["optical", "ir", "lidar"] if kind in ("glare", "occlusion") else list(MODALITIES)
        modalities = event.get("modalities", defaults)
        if not isinstance(modalities, list) or not modalities or any(m not in MODALITIES for m in modalities):
            raise ValueError("invalid interference modalities")
        # A glare/occlusion mask has camera coordinates, not RD/RA coordinates.
        if kind in ("glare", "occlusion") and any(m in ("radar", "sar") for m in modalities):
            raise ValueError("glare/occlusion require explicit camera or lidar modalities")
        start = _number(event.get("start_s", 0), "start_s", 0)
        end = event.get("end_s")
        end = None if end is None else _number(end, "end_s", start)
        if end is not None and end <= start:
            raise ValueError("end_s must exceed start_s")
        event.update(id=name, kind=kind, modalities=list(dict.fromkeys(modalities)), start_s=start, end_s=end,
                     fade_s=_number(event.get("fade_s", 0), "fade_s", 0),
                     strength=_number(event.get("strength", 0.2), "strength", 0, 1))
        for key, default, low, high in (("center_uv", [0.5, 0.5], None, None),
                                        ("size_uv", [0.2, 0.2], 1e-6, 2),
                                        ("drift_uv_s", [0, 0], None, None),
                                        ("color", [0.5, 0.5, 0.5], 0, 1)):
            value = event.get(key, default)
            if not isinstance(value, (list, tuple)) or len(value) != len(default):
                raise ValueError(key+" has the wrong shape")
            event[key] = [_number(v, key, low, high) for v in value]
        events.append(event)
    return dict(enabled=spec.get("enabled", True), seed=int(seed), events=events)


def _weight(event, t):
    start, end, fade = event["start_s"], event["end_s"], event["fade_s"]
    if t < start or (end is not None and t >= end):
        return 0.0
    if not fade:
        return 1.0
    u = min(1.0, (t-start)/fade, (end-t)/fade if end is not None else 1.0)
    return u*u*(3-2*u)


def _mask(event, shape, t):
    H, W = shape
    yy, xx = np.mgrid[:H, :W]
    x, y = (xx+0.5)/W, (yy+0.5)/H
    center = np.array(event["center_uv"])+np.array(event["drift_uv_s"])*(t-event["start_s"])
    sx, sy = event["size_uv"]
    if event["kind"] == "glare":
        return np.exp(-0.5*(((x-center[0])/sx)**2+((y-center[1])/sy)**2))
    return ((np.abs(x-center[0]) <= sx/2) & (np.abs(y-center[1]) <= sy/2)).astype(float)


def _transmission(spec, modality, shape, t):
    result = np.ones(shape)
    if spec["enabled"]:
        for event in spec["events"]:
            if event["kind"] == "occlusion" and modality in event["modalities"]:
                result *= 1-event["strength"]*_weight(event, t)*_mask(event, shape, t)
    return result


def apply(image, spec, modality, frame, t, stream=""):
    """Return measured-code raster, synthetic transmission mask and event records."""
    if modality not in MODALITIES:
        raise ValueError("unknown modality "+modality)
    image = np.asarray(image)
    if (image.dtype not in (np.uint8, np.uint16) or image.ndim not in (2, 3)
            or min(image.shape[:2]) < 1 or (image.ndim == 3 and image.shape[2] != 3)):
        raise ValueError("interference requires a uint8/uint16 raster")
    if _number(frame, "frame", 0) != int(frame):
        raise ValueError("frame must be a non-negative integer")
    t = _number(t, "time")
    spec = validate(spec)
    limit = np.iinfo(image.dtype).max
    result = image.astype(float)/limit
    H, W = image.shape[:2]
    transmission = np.ones((H, W))
    records = []
    if not spec["enabled"]:
        return image.copy(), transmission, records
    for event in spec["events"]:
        amount = event["strength"]*_weight(event, float(t))
        if modality not in event["modalities"] or amount <= 0:
            continue
        kind = event["kind"]
        color = np.asarray(event["color"] if image.ndim == 3 else np.mean(event["color"]))
        mask = np.ones((H, W))
        if kind == "additive_noise":
            salt = zlib.crc32((event["id"]+"/"+modality+"/"+stream).encode("utf-8"))
            rng = np.random.default_rng(np.random.SeedSequence([spec["seed"], int(frame), salt]))
            result = result+rng.normal(0, amount, result.shape)
        elif kind == "glare":
            mask = _mask(event, (H, W), t)
            result = result+amount*(mask[..., None] if image.ndim == 3 else mask)
        elif kind == "occlusion":
            mask = _mask(event, (H, W), t)
            alpha = amount*mask
            transmission *= 1-alpha
            alpha = alpha[..., None] if image.ndim == 3 else alpha
            result = result*(1-alpha)+color*alpha
        else:  # Abstract image contrast loss, including radar RD/RA rasters.
            result = result*(1-amount)+color*amount
        records.append(dict(id=event["id"], kind=kind, group=GROUPS[kind], amount=amount,
                            affected_fraction=float(np.mean(mask > 1e-3)), domain="image_codes"))
    return np.rint(np.clip(result, 0, 1)*limit).astype(image.dtype), transmission, records


def _visibility(box, transmission):
    H, W = transmission.shape
    x0, y0 = max(0, int(np.floor(box["x0"]))), max(0, int(np.floor(box["y0"])))
    x1, y1 = min(W, int(np.ceil(box["x1"]))), min(H, int(np.ceil(box["y1"])))
    return float(transmission[y0:y1, x0:x1].mean()) if x1 > x0 and y1 > y0 else 0.0


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=1)+"\n", encoding="utf-8")


def export(source, target, spec):
    """Create a standalone, exportable image-only variant; never modify source."""
    from PIL import Image
    source, target = Path(source).resolve(), Path(target).resolve()
    if source == target or target in source.parents:
        raise ValueError("interference output must differ from and not contain the source")
    spec = validate(spec)
    read = lambda p: json.loads(p.read_text(encoding="utf-8"))
    meta, frames = read(source/"meta.json"), read(source/"frames.json")
    times = {int(row["frame"]): float(row["t"]) for row in frames}
    if not times:
        raise ValueError("source dataset has no frames")
    marker = target/"interference.json"
    if target.exists() and any(target.iterdir()):
        if not marker.exists() or read(marker).get("source") != str(source):
            raise ValueError("refusing to overwrite a directory not owned by this interference export")
        # Delete only files explicitly owned by the previous export.
        for rel in read(marker).get("files", [])+["quality_interference.json"]:
            previous = (target/rel).resolve()
            if target not in previous.parents:
                raise ValueError("invalid previous interference file manifest")
            previous.unlink(missing_ok=True)
    target.mkdir(parents=True, exist_ok=True)
    manifest = dict(schema_version=1, source=str(source), domain="image_codes", status="writing",
                    spec=spec, files=[], products={},
                    raw_measurements="clean reference only; raw echoes and point clouds are not augmented")
    _write(marker, manifest)
    grids, label_data = {}, {}
    products = [*source.glob("images/*/*.png"), *source.glob("lidar/*/*.png")]
    if not products:
        raise ValueError("source dataset has no images")
    files, annotation_counts = set(), {}
    for path in sorted(products):
        try:
            fid = int(path.stem)
        except ValueError:
            continue
        if fid not in times:
            continue
        product = path.parent.name
        modality = ("lidar" if path.parent.parent.name == "lidar" else
                    next((m for m in MODALITIES if product.startswith(m)), None))
        if modality is None:
            raise ValueError("unknown image product "+product)
        with Image.open(path) as im:
            array = np.asarray(im)
            if im.mode in ("I", "I;16"):
                array = array.astype(np.uint16)
            changed, transmission, events = apply(array, spec, modality, fid, times[fid], product)
        rel = path.relative_to(source)
        dst = target/rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(changed).save(dst)
        files.add(str(rel))
        key = "lidar_"+product if modality == "lidar" else product
        grids[(key, fid)] = (modality, array.shape[:2])
        entry = manifest["products"].setdefault(key, dict(images=0, affected_frames=0, scheduled_frames=0))
        entry["images"] += 1
        entry["scheduled_frames"] += bool(events)
        entry["affected_frames"] += bool(np.any(changed != array))
        label_name = ("lidar" if modality == "lidar" else "radar" if modality == "radar" else
                      product.replace("ir_preview", "ir", 1))
        lp = Path("labels")/label_name/(path.stem+".json")
        if not (source/lp).exists():
            raise ValueError("missing source annotation "+str(lp))
        label = label_data.setdefault(str(lp), read(source/lp))
        label.setdefault("interference", {})[key] = dict(events=events, source_image=str(rel),
                                                          domain="image_codes", clean_source=str(source/rel))
        if modality == "radar":
            label["detection_scope"] = "clean reference detections; not recomputed from augmented RD/RA images"
        # Only spatial raster boxes have image coordinates. Radar 3D/point
        # truth remains the clean reference; RD and RA products are not cameras.
        if modality != "radar" and not product.startswith("ir_preview") and product != "intensity":
            boxes = label.get("boxes", {})
            for box in boxes.values() if isinstance(boxes, dict) else boxes:
                box["source_visible"] = box.get("visible", True)
                box["synthetic_visibility"] = _visibility(box, transmission)
                box["visible"] = box["source_visible"] and box["synthetic_visibility"] > 1e-6
    for name, product in manifest["products"].items():
        if product["images"] != len(times):
            raise ValueError("source image product has missing frames: "+name)
    for rel, data in label_data.items():
        _write(target/rel, data)
        files.add(rel)
    for path in source.glob("annotations_*_coco.json"):
        coco = read(path)
        product = path.name[len("annotations_"):-len("_coco.json")]
        product = "lidar_range" if product == "lidar" else product
        image_ids = {im["id"]: int(Path(im["file_name"]).stem) for im in coco["images"]}
        if any((product, fid) not in grids for fid in image_ids.values()):
            raise ValueError("COCO image has no matching raster in "+path.name)
        annotations = []
        previous_key, trans = None, None
        for ann in coco["annotations"]:
            key = (product, image_ids[ann["image_id"]])
            grid = grids.get(key)
            if grid is None:
                raise ValueError("COCO image has no matching raster in "+path.name)
            if key != previous_key:
                trans = _transmission(spec, grid[0], grid[1], times[key[1]])
                previous_key = key
            x, y, w, h = ann["bbox"]
            fraction = _visibility(dict(x0=x, y0=y, x1=x+w, y1=y+h), trans)
            ann["synthetic_visibility"] = fraction
            if fraction > 1e-6:
                annotations.append(ann)
        coco["annotations"] = annotations
        annotation_counts[path.name] = len(annotations)
        _write(target/path.name, coco)
        files.add(path.name)
    for path in source.glob("splits*.json"):
        shutil.copyfile(path, target/path.name)
        files.add(path.name)
    for row in frames:
        rel = Path("observations")/("%06d.json" % row["frame"])
        if (source/rel).exists():
            observation = read(source/rel)
            observation["interference"] = dict(domain="image_codes", clean_source=str(source/rel),
                                                measurement_scope="clean geometric reference")
            _write(target/rel, observation)
            files.add(str(rel))
    _write(target/"frames.json", frames)
    meta.update(interference=dict(source=str(source), domain="image_codes", manifest="interference.json"),
                generated_by="simkit interference augmentation", raw_measurements_available=False)
    _write(target/"meta.json", meta)
    products = manifest["products"]
    stats = dict(frames=len(frames), image_domain_only=True, annotation_counts=annotation_counts,
                 optical_images=products.get("optical", {}).get("images", 0),
                 ir_images=sum(v["images"] for k, v in products.items() if k.startswith("ir") and not k.startswith("ir_preview")),
                 radar_frames=products.get("radar", {}).get("images", 0),
                 lidar_images=products.get("lidar_range", {}).get("images", 0),
                 sar_images=products.get("sar", {}).get("images", 0))
    _write(target/"stats.json", stats)
    files.update(("frames.json", "meta.json", "stats.json"))
    manifest.update(status="complete", files=sorted(files))
    _write(marker, manifest)
    return manifest
