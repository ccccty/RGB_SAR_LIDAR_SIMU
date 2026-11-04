"""Inspect real clean/augmented products without rerunning the augmenter.

Usage: python -m tools.check_interference out/run/interfered
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def check(root):
    root = Path(root).resolve()
    read = lambda path: json.loads(path.read_text(encoding="utf-8"))
    manifest = read(root/"interference.json")
    source = Path(manifest["source"])
    frames = read(root/"frames.json")
    errors, products = [], {}
    if manifest.get("status") != "complete" or manifest.get("domain") != "image_codes":
        errors.append("variant is incomplete or declares the wrong domain")
    if (root/"raw").exists() or (root/"lidar/points").exists():
        errors.append("image augmentation must not claim corrupted raw echoes or point clouds")
    for rel in manifest["files"]:
        path = (root/rel).resolve()
        if root not in path.parents or not path.is_file():
            errors.append("missing/invalid output path "+rel)
            continue
        if path.suffix != ".png":
            continue
        with Image.open(path) as im, Image.open(source/rel) as clean:
            a, b = np.asarray(im), np.asarray(clean)
        if a.shape != b.shape or a.dtype != b.dtype:
            errors.append("shape or bit depth changed: "+rel)
            continue
        product = ("lidar_"+path.parent.name if path.parent.parent.name == "lidar" else path.parent.name)
        row = products.setdefault(product, dict(images=0, changed_frames=0, mean_absolute_change=[], noise_std=[]))
        difference = a.astype(float)-b.astype(float)
        row["images"] += 1
        row["changed_frames"] += bool(np.any(difference))
        row["mean_absolute_change"].append(float(np.abs(difference).mean()))
        row["noise_std"].append(float(difference.std()))
    for product, row in products.items():
        if row["images"] != len(frames):
            errors.append("frame count mismatch for "+product)
        expected = manifest["products"][product]["affected_frames"]
        if row["changed_frames"] != expected:
            errors.append("actual pixel changes disagree with manifest for "+product)
        row["mean_absolute_change"] = float(np.mean(row["mean_absolute_change"]))
        row["noise_std"] = float(np.mean(row["noise_std"]))
    for path in root.glob("annotations_*_coco.json"):
        data = read(path)
        images = {im["id"]: im for im in data["images"]}
        for ann in data["annotations"]:
            if ann["image_id"] not in images or not 0 < ann.get("synthetic_visibility", 0) <= 1:
                errors.append("invalid COCO visibility/image link in "+path.name)
    for path in root.glob("labels/radar/*.json"):
        data = read(path)
        if "clean reference" not in data.get("detection_scope", ""):
            errors.append("radar detection reference scope missing")
    report = dict(passed=not errors, errors=errors, frames=len(frames), products=products,
                  scope="decoded image changes, source correspondence, COCO and reference metadata; image-domain only")
    (root/"quality_interference.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report["passed"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("variant", type=Path)
    raise SystemExit(0 if check(ap.parse_args().variant) else 1)
