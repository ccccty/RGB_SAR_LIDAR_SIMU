"""Annotation export: 2D boxes from masks/projection, COCO and radar 3D labels."""
import json
import os

import numpy as np


def project_points(pts, cam, width, height):
    from .scene import sensor_basis
    p, r, u, f = sensor_basis(cam["pos"], cam["look_at"], cam.get("up", (0, 0, 1)))
    fx = (width / 2.0) / np.tan(np.radians(cam.get("fov_deg", 30.0)) / 2.0)
    rel = np.asarray(pts, float) - p
    z = rel @ f
    zs = np.where(z > 1e-6, z, 1e-6)
    sx = width / 2.0 + fx * (rel @ r) / zs
    sy = height / 2.0 - fx * (rel @ u) / zs
    return np.stack([sx, sy, z], axis=1)


# Small objects are the point of a detection dataset: a UAV a few pixels across
# is a legitimate (hard) example, not noise. Below ~8 px the mask is a stray
# speckle, so that is where the cut sits.
def boxes_from_mask(obj_id, world, min_px=8):
    """2D boxes from the object-id buffer; fall back to projected corners."""
    out = {}
    for o in world.objs:
        if not o.get("label", True):
            continue
        m = obj_id == o["id"]
        n = int(m.sum())
        if n < min_px:
            continue
        ys, xs = np.where(m)
        out[o["id"]] = dict(x0=float(xs.min()), y0=float(ys.min()),
                            x1=float(xs.max() + 1), y1=float(ys.max() + 1),
                            pixels=n, source="mask")
    return out


def boxes_from_corners(world, cam, width, height, min_frac=1e-4):
    out = {}
    for o in world.objs:
        if not o.get("label", True):
            continue
        pr = project_points(o["corners"], cam, width, height)
        if (pr[:, 2] <= 0.1).all():
            continue
        x0, y0 = pr[:, 0].min(), pr[:, 1].min()
        x1, y1 = pr[:, 0].max(), pr[:, 1].max()
        x0, x1 = max(x0, 0.0), min(x1, float(width))
        y0, y1 = max(y0, 0.0), min(y1, float(height))
        if (x1 - x0) * (y1 - y0) < min_frac * width * height:
            continue
        out[o["id"]] = dict(x0=x0, y0=y0, x1=x1, y1=y1, pixels=0, source="corners")
    return out


def to_coco(boxes, world, image_id, file_name, width, height):
    anns = []
    for aid, (oid, b) in enumerate(sorted(boxes.items())):
        o = next((x for x in world.objs if x["id"] == oid), None)
        if o is None or not o.get("label", True):
            continue
        anns.append(dict(
            id=image_id * 1000 + aid, image_id=image_id, category_id=o["cls_id"],
            bbox=[round(b["x0"], 2), round(b["y0"], 2),
                  round(b["x1"] - b["x0"], 2), round(b["y1"] - b["y0"], 2)],
            area=round((b["x1"] - b["x0"]) * (b["y1"] - b["y0"]), 2),
            iscrowd=0, source=b.get("source", "mask"),
            object_name=o["name"],
        ))
    return dict(id=image_id, file_name=file_name, width=width, height=height), anns


def radar_labels(world, detections, frame_id):
    boxes = []
    for o in world.objs:
        if not o.get("label", True):
            continue
        c = np.asarray(o["corners"])
        boxes.append(dict(id=o["id"], name=o["name"], cls=o["cls"], cls_id=o["cls_id"],
                          center=[round(float(v), 3) for v in c.mean(axis=0)],
                          size=[round(float(v), 3) for v in o["size"]],
                          yaw=round(float(o["yaw"]), 2)))
    pts = [d for d in detections if d["obj"] >= 0]
    clutter = [d for d in detections if d["obj"] < 0]
    return dict(frame=frame_id, boxes_3d=boxes,
                points=[dict(xyz=[round(d["x"], 3), round(d["y"], 3), round(d["z"], 3)],
                             range=round(d["range"], 3), az=round(d["az"], 3),
                             el=round(d["el"], 3), v_r=round(d["v_r"], 3),
                             snr_db=round(d["snr_db"], 2), obj=d["obj"]) for d in pts],
                clutter_count=len(clutter),
                num_points=len(pts))


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1)
