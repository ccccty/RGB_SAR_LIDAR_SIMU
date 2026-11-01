"""Repair observed-visibility labels from saved instance passes, without rendering.

Used once for the v3 integration ID/name regression. Originals and exact repair
source are retained. Safe to re-run; raw sensor measurements are never changed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repair(run):
    cfg = read(run/"_render_cfg.json")
    objects = {obj["name"]: i for i, obj in enumerate(cfg["objects"])}
    changes = {}
    coco_data = {}
    for mod in ("optical", "ir_mwir", "ir_lwir"):
        labels = sorted((run/"labels"/mod).glob("*.json"))
        observed = {}
        for path in labels:
            before = digest(path)
            label = read(path)
            raw_path = run/"raw"/mod/(path.stem+".npz")
            with np.load(raw_path) as raw:
                ids = raw["instance_id"]
                if ids.shape != (label["height"], label["width"]):
                    raise ValueError("instance/label shape mismatch: "+str(path))
                for name, box in label.get("boxes", {}).items():
                    pixels = int(np.count_nonzero(ids == objects[name]))
                    box.update(pixels=pixels, visible=pixels>0, visibility_source="rendered_instance_pass")
                    key = "coverage_%d" % objects[name]
                    if key in raw:
                        box["effective_coverage_pixels"] = float(raw[key].sum())
            observed[int(label["frame"])] = label["boxes"]
            backup = run/"_before_visibility_fix"/path.relative_to(run)
            backup.parent.mkdir(parents=True, exist_ok=True)
            if not backup.exists():
                shutil.copyfile(path, backup)
            path.write_text(json.dumps(label, ensure_ascii=False, indent=1), encoding="utf-8")
            changes[str(path.relative_to(run))] = dict(before=before, after=digest(path))
        coco_path = run/("annotations_%s_coco.json" % mod)
        if coco_path.exists():
            before = digest(coco_path)
            backup = run/"_before_visibility_fix"/coco_path.name
            if not backup.exists():
                shutil.copyfile(coco_path, backup)
            coco = read(coco_path)
            annotations = []
            for annotation in coco["annotations"]:
                box = observed[annotation["image_id"]].get(annotation["object_name"], {})
                if not box.get("visible", False):
                    continue
                annotation.update(visible=True, raster_pixels=box["pixels"])
                annotations.append(annotation)
            coco["annotations"] = annotations
            coco_data[mod] = coco
            coco_path.write_text(json.dumps(coco, ensure_ascii=False, indent=1), encoding="utf-8")
            changes[coco_path.name] = dict(before=before, after=digest(coco_path))
    stats_path = run/"stats.json"
    if stats_path.exists():
        backup = run/"_before_visibility_fix"/stats_path.name
        if not backup.exists():
            shutil.copyfile(stats_path, backup)
        before = digest(stats_path)
        stats = read(stats_path)
        stats["optical_images"] = len(coco_data["optical"]["images"])
        stats["optical_boxes"] = len(coco_data["optical"]["annotations"])
        for band in ("mwir", "lwir"):
            stats["ir"][band]["images"] = len(coco_data["ir_"+band]["images"])
            stats["ir"][band]["boxes"] = len(coco_data["ir_"+band]["annotations"])
        stats["ir_images"] = sum(stats["ir"][band]["images"] for band in ("mwir", "lwir"))
        stats["ir_boxes"] = sum(stats["ir"][band]["boxes"] for band in ("mwir", "lwir"))
        counts = {}
        for coco in coco_data.values():
            names = {c["id"]: c["name"] for c in coco["categories"]}
            for annotation in coco["annotations"]:
                name = names[annotation["category_id"]]
                counts[name] = counts.get(name, 0)+1
        stats["boxes_per_class"] = counts
        points = []
        for path in sorted((run/"labels/radar").glob("*.json")):
            radar = read(path)
            points.append(dict(frame=radar["frame"], n=radar["num_points"],
                               clutter=len(radar.get("unassociated_detections", []))))
        stats["radar_points_per_frame"] = points
        stats_path.write_text(json.dumps(stats, indent=1), encoding="utf-8")
        changes[stats_path.name] = dict(before=before, after=digest(stats_path))
    report = dict(operation="instance-derived visibility repair", changed_files=changes,
                  source_sha256=digest(Path(__file__)), images_and_raw_changed=False)
    (run/"visibility_finalization.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", type=Path)
    args = ap.parse_args()
    runs = sorted(p.parent for p in args.path.glob("*/frames.json"))
    if not runs:
        ap.error("no completed runs")
    previous = args.path/"visibility_finalization.json"
    if previous.exists():
        shutil.copyfile(previous, args.path/("visibility_finalization_"+digest(previous)[:12]+".json"))
    reports = {run.name: repair(run) for run in runs}
    snapshot = args.path/"_finalization_source"/"finalize_realism.py"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    if snapshot.exists():
        shutil.copyfile(snapshot, snapshot.with_name("finalize_realism_"+digest(snapshot)[:12]+".py"))
    shutil.copyfile(__file__, snapshot)
    report = dict(source_sha256=digest(Path(__file__)), runs=reports)
    (args.path/"visibility_finalization.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Rebuilt visibility and COCO for", len(runs), "runs; originals and repair source retained")


if __name__ == "__main__":
    main()
