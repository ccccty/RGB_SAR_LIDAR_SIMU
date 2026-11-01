"""Reconcile IR labels with the optical occlusion test, and rebuild IR boxes.

    python tools\\relabel_wide.py out\\seeker_air2air_occluded [more runs...]

Two defects this repairs, both found by checking the urban-occlusion scene:

1. **Wide-lens boxes were dropped by size floors.** The search lens puts a 2-3 px
   target on the detector, and both box paths had a floor (a mask needs 8 pixels,
   a projected box needs 30 px^2). The search view shipped images with almost no
   labels -- 1-7% of frames, measured on the air-to-air scene.
2. **Projected boxes ignored occlusion.** When a building hides the target the
   mask is empty, and the projection fallback then added a box for a target that
   is not in the picture. The narrow IR labels had boxes on all 30 blocked frames.

Both are fixed in `simkit/dataset.py` for future runs. This tool repairs runs
whose pixels are already rendered, using data the run already recorded:

* the per-frame lens poses and object poses live in `optical_frames.partNNN.json`;
* Blender already ran an authoritative occlusion test for each optical lens and
  recorded the verdict as `boxes[name].visible`.

So every `ir_*` label is rebuilt from geometry, and an object only gets a box
when the optical pass for the *same lens* could see it. Images are never touched.
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from simkit import labels, scene

_BOX_CACHE = {}


def load(p, default=None):
    try:
        with open(p, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return default


def rot_z(deg):
    a = np.radians(float(deg))
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def local_box(spec, base_dir):
    """Object bbox in its own frame; prepare_mesh decimates, so cache it once."""
    key = spec.get("name") or id(spec)
    if key not in _BOX_CACHE:
        _m, lo, hi = scene.prepare_mesh(spec, base_dir)
        _BOX_CACHE[key] = np.array([[x, y, z] for x in (lo[0], hi[0])
                                    for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    return _BOX_CACHE[key]


def project_box(spec, base_dir, pose, cam, width, height):
    pr = labels.project_points(local_box(spec, base_dir) @ rot_z(pose.get("yaw", 0.0)).T
                               + np.asarray(pose["pos"], float), cam, width, height)
    if (pr[:, 2] <= 0.1).all():
        return None
    x0, x1 = float(pr[:, 0].min()), float(pr[:, 0].max())
    y0, y1 = float(pr[:, 1].min()), float(pr[:, 1].max())
    x0, x1 = max(x0, 0.0), min(x1, float(width))
    y0, y1 = max(y0, 0.0), min(y1, float(height))
    # no area floor: a 2 px target is still a target
    if x1 - x0 < 1.0:
        cx = 0.5 * (x0 + x1)
        x0, x1 = max(0.0, cx - 0.5), min(float(width), cx + 0.5)
    if y1 - y0 < 1.0:
        cy = 0.5 * (y0 + y1)
        y0, y1 = max(0.0, cy - 0.5), min(float(height), cy + 0.5)
    return dict(x0=round(x0, 1), y0=round(y0, 1), x1=round(x1, 1), y1=round(y1, 1),
                visible=True, source="corners")


def rebuild_coco(run, rows, specs, cls_id, band):
    acc = dict(info=dict(description="simkit synthetic dataset (%s)" % band, version="0.1"),
               images=[], annotations=[],
               categories=[dict(id=cls_id[c], name=c, supercategory="target")
                           for c in sorted(cls_id)])
    for fid, W, H, boxes in rows:
        acc["images"].append(dict(id=fid, file_name="%06d.png" % fid, width=W, height=H))
        for aid, (nm, b) in enumerate(sorted(boxes.items())):
            spec = specs.get(nm)
            if spec is None:
                continue
            acc["annotations"].append(dict(
                id=fid * 1000 + aid, image_id=fid,
                category_id=cls_id.get(spec.get("cls", "target"), 1),
                bbox=[b["x0"], b["y0"], round(b["x1"] - b["x0"], 2),
                      round(b["y1"] - b["y0"], 2)],
                area=round((b["x1"] - b["x0"]) * (b["y1"] - b["y0"]), 2),
                iscrowd=0, object_name=nm, source=b.get("source", "mask")))
    labels.write_json(os.path.join(run, "annotations_%s_coco.json" % band), acc)
    return acc


def main(argv):
    runs = argv[1:]
    if not runs:
        print(__doc__)
        return 2
    for run in runs:
        if not os.path.isdir(run):
            continue
        meta = load(os.path.join(run, "meta.json"), {})
        cfg_path = meta.get("config")
        if not cfg_path or not os.path.exists(cfg_path):
            print("skip %s: no meta.config" % run)
            continue
        cfg = scene.load_config(cfg_path)
        base = cfg.get("_base_dir", ".")
        specs = {o["name"]: o for o in cfg.get("objects", [])
                 if o.get("label", True) and o.get("name")}
        cls_id = {c: i + 1 for i, c in enumerate(sorted({o.get("cls", "target")
                                                         for o in specs.values()}))}
        frames = {}
        for p in sorted(glob.glob(os.path.join(run, "optical_frames.part*.json"))) + \
                sorted(glob.glob(os.path.join(run, "optical_frames.json"))):
            for f in load(p, {}).get("frames") or []:
                frames[int(f["frame"])] = f
        if not frames:
            print("skip %s: no optical manifest" % run)
            continue

        # Optical labels from the first builds listed every mesh in the world,
        # so unlabelled scenery appeared as extra "target" boxes on the site.
        # Nothing needs re-rendering to fix that: keep only labelled objects.
        for lab_dir in sorted(glob.glob(os.path.join(run, "labels", "optical*"))):
            pruned = 0
            for p in sorted(glob.glob(os.path.join(lab_dir, "*.json"))):
                d = load(p)
                if not d or not isinstance(d.get("boxes"), dict):
                    continue
                keep = {k: v for k, v in d["boxes"].items() if k in specs}
                if len(keep) != len(d["boxes"]):
                    d["boxes"] = keep
                    d["pruned_by"] = "tools/relabel_wide.py"
                    labels.write_json(p, d)
                    pruned += 1
            if pruned:
                print("%-26s %-16s pruned scenery boxes in %d frames"
                      % (os.path.basename(run), os.path.basename(lab_dir), pruned))

        for lab_dir in sorted(glob.glob(os.path.join(run, "labels", "ir*"))):
            if not os.path.isdir(lab_dir):
                continue
            band = os.path.basename(lab_dir)                     # ir_mwir / ir_mwir_wide
            tag = "_wide" if band.endswith("_wide") else ""
            rows, changed, dropped, added = [], 0, 0, 0
            for fid, fr in sorted(frames.items()):
                d = load(os.path.join(lab_dir, "%06d.json" % fid))
                if d is None:
                    continue
                view = ((fr.get("views") or {}).get("wide" if tag else "seeker")
                        or (fr.get("views") or {}).get("seeker"))
                if not view:
                    continue
                cam = dict(pos=view["pos"], look_at=view["look_at"],
                           fov_deg=view["fov_deg"])
                W, H = int(view["width"]), int(view["height"])
                opt = load(os.path.join(run, "labels", "optical" + tag,
                                        "%06d.json" % fid), {}) or {}
                vis = {nm: b.get("visible", True)
                       for nm, b in (opt.get("boxes") or {}).items()}
                before = d.get("boxes")
                before = dict(before) if isinstance(before, dict) else \
                    {b.get("object_name", "?"): b for b in (before or [])}
                boxes = {}
                for nm, spec in specs.items():
                    pose = (fr.get("objects") or {}).get(nm)
                    if pose is None or vis.get(nm) is not True:
                        if nm in before:
                            dropped += 1          # occluded: no box, by construction
                        continue
                    b = project_box(spec, base, pose, cam, W, H)
                    if b is None:
                        continue
                    boxes[nm] = b
                    if nm not in before:
                        added += 1
                if boxes != before:
                    changed += 1
                    d["boxes"] = boxes
                    d["relabelled_by"] = "tools/relabel_wide.py"
                    labels.write_json(os.path.join(lab_dir, "%06d.json" % fid), d)
                rows.append((fid, W, H, boxes))
            if not rows:
                continue
            acc = rebuild_coco(run, rows, specs, cls_id, band)
            print("%-26s %-16s %4d frames  %4d boxes  changed %3d (+%d/-%d)"
                  % (os.path.basename(run), band, len(rows), len(acc["annotations"]),
                     changed, added, dropped))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
