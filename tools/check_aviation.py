"""Audit saved aviation observations and emit a paired sequence manifest.

Reports geometric calibration consistency separately from image registration
and detectability. Missing observations are legitimate, missing files are not.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _check_core(run):
    frames = read(run / "frames.json")
    errors, rows, ranges, detected = [], [], [], []
    max_projection_error = 0.0
    for frame in frames:
        stem = "%06d" % frame["frame"]
        opath = run / "observations" / (stem+".json")
        if not opath.exists():
            errors.append("missing " + str(opath))
            continue
        obs = read(opath)
        obj = obs["objects"][0]
        ranges.append(obj["range_m"])
        p = np.array(obs["observer"]["position_m"])
        offset = np.array(obj["center_world_m"])-p
        relative_v = np.array(obj["velocity_mps"])-np.array(obs["observer"]["velocity_mps"])
        closing = -float(offset @ relative_v)/np.linalg.norm(offset)
        if abs(closing-obj["closing_speed_mps"]) > 1e-6:
            errors.append(stem+": inconsistent radial velocity")
        for name, cam in obs["cameras"].items():
            center = np.r_[obj["center_world_m"], 1.0]
            xyz = (np.array(cam["world_to_camera"]) @ center)[:3]
            uv = np.array(cam["K"]) @ xyz
            dev = float(np.max(np.abs(uv[:2]/uv[2]-obj["projections"][name]["center_px"])))
            max_projection_error = max(max_projection_error, dev)
        row = dict(frame=frame["frame"], clip=frame["clip"],
                   timestamp_s=obs["sequence_timestamp_s"], track_id=obj["track_id"],
                   observations=str(opath.relative_to(run)),
                   range_m=obj["range_m"], closing_speed_mps=closing)
        for mod, folder in [("rgb", "images/optical"), ("mwir", "images/ir_mwir"),
                            ("lwir", "images/ir_lwir"), ("rd", "images/radar"),
                            ("rd_raw", "raw/radar")]:
            rel = folder+"/"+stem+(".npz" if mod == "rd_raw" else ".png")
            row[mod] = rel
            if not (run / rel).exists():
                errors.append("missing " + rel)
        lp = run / "labels/radar" / (stem+".json")
        if lp.exists():
            radar = read(lp)
            detected.append(bool(radar["points"]))
            row["radar_detected"] = bool(radar["points"])
            for point in radar["points"]:
                # Returns lie on the surface, truth is the box centre; allow
                # aircraft extent, but not a wrong sign or platform velocity.
                if abs(point["v_r_true"]-closing) > 2.0:
                    errors.append(stem+": radar closing speed disagrees with kinematics")
            raw_path = run / row["rd_raw"]
            if raw_path.exists():
                with np.load(raw_path) as raw:
                    r, v = raw["range_m"], raw["closing_speed_mps"]
                    if raw["power"].shape != (len(v), len(r)) or not np.isfinite(raw["power"]).all():
                        errors.append(stem+": invalid RD shape/power")
                    if not (np.diff(r)>0).all() or not (np.diff(v)>0).all():
                        errors.append(stem+": non-monotonic RD axes")
        else:
            errors.append("missing " + str(lp))
        # Keep signal visibility separate from a geometrical annotation.
        for mod in ("optical", "ir_mwir", "ir_lwir"):
            lp = run / "labels" / mod / (stem+".json")
            if not lp.exists():
                errors.append("missing " + str(lp))
                continue
            boxes = read(lp).get("boxes", {})
            box = boxes.get(obj["track_id"], {})
            row[mod+"_raster_visible"] = bool(box.get("visible", False))
            row[mod+"_raster_pixels"] = box.get("pixels", 0)
        rows.append(row)
    if max_projection_error > 1e-5:
        errors.append("camera calibration projection error exceeds 1e-5 pixels")
    with (run / "pairs.csv").open("w", newline="", encoding="utf-8") as fh:
        if rows:
            writer = csv.DictWriter(fh, fieldnames=sorted(set().union(*(r.keys() for r in rows))))
            writer.writeheader()
            writer.writerows(rows)
    report = dict(frames=len(frames), range_m=[min(ranges), max(ranges)] if ranges else None,
                  radar_detected_frames=sum(detected),
                  max_calibration_projection_error_px=max_projection_error,
                  image_content_registration="not yet independently validated",
                  real_sensor_validation="none",
                  classification_training_ready=False,
                  classification_training_reason="two model instances only; uncalibrated sensors/materials; QA sequences",
                  errors=errors, passed=not errors)
    (run / "quality.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return not errors


DEFAULT_REGISTRATION = dict(min_pixels=8, max_centroid_delta_px=3.0, min_bbox_iou=0.1,
                            max_median_depth_delta_m=5.0, relative_depth_tolerance=0.002)


def _bbox(mask):
    yy, xx = np.where(mask)
    if not len(xx):
        return None
    h, w = mask.shape
    return [xx.min()/w, yy.min()/h, (xx.max()+1)/w, (yy.max()+1)/h]


def _bbox_iou(a, b):
    if a is None or b is None:
        return None
    area = max(0.0, min(a[2], b[2])-max(a[0], b[0]))*max(0.0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-area
    return float(area/union) if union > 0 else None


def _resample_nearest(array, shape):
    h, w = shape
    ih, iw = array.shape
    yy = np.minimum(((np.arange(h)+0.5)*ih/h).astype(int), ih-1)
    xx = np.minimum(((np.arange(w)+0.5)*iw/w).astype(int), iw-1)
    return array[np.ix_(yy, xx)]


def compare_geometry(rgb, other, object_id, limits=None):
    """Measure independent passes, not common annotation boxes.

    Assumes the declared co-located primary field of view. Different grids are
    compared in normalized FOV coordinates and are descriptive only. A sparse
    mask cannot produce a reliable estimate, especially if absent in one pass.
    """
    limits = {**DEFAULT_REGISTRATION, **(limits or {})}
    a, b = rgb["instance_id"] == object_id, other["instance_id"] == object_id
    na, nb = int(a.sum()), int(b.sum())
    sa, sb = a.shape, b.shape
    result = dict(status="insufficient_support", rgb_pixels=na, other_pixels=nb,
                  rgb_shape=list(sa), other_shape=list(sb), centroid_delta_px=None,
                  bbox_iou=None, mask_iou=None, overlap_over_smaller_mask=None,
                  common_depth_pixels=0, median_depth_delta_m=None, p95_depth_delta_m=None,
                  errors=[], interpretation="instance/depth geometry; not radiometric detectability")
    if na == 0 or nb == 0:
        result["reason"] = "target absent from at least one sampled instance pass"
        return result
    ya, xa = np.where(a)
    yb, xb = np.where(b)
    ca = np.array([(xa+0.5).mean()/sa[1], (ya+0.5).mean()/sa[0]])
    cb = np.array([(xb+0.5).mean()/sb[1], (yb+0.5).mean()/sb[0]])
    result["centroid_delta_px"] = float(np.linalg.norm((ca-cb)*[sa[1], sa[0]]))
    result["bbox_iou"] = _bbox_iou(_bbox(a), _bbox(b))
    sampled_b = b if sa == sb else _resample_nearest(b, sa)
    intersection, union = a & sampled_b, a | sampled_b
    result["mask_iou"] = float(intersection.sum()/union.sum())
    denominator = min(na, int(sampled_b.sum()))
    result["overlap_over_smaller_mask"] = float(intersection.sum()/denominator) if denominator else None
    da = rgb["depth_z_m"]
    db = other["depth_z_m"] if sa == sb else _resample_nearest(other["depth_z_m"], sa)
    valid = intersection & np.isfinite(da) & np.isfinite(db) & (da > 0) & (db > 0)
    result["common_depth_pixels"] = int(valid.sum())
    if valid.any():
        delta = np.abs(da[valid]-db[valid])
        result["median_depth_delta_m"] = float(np.median(delta))
        result["p95_depth_delta_m"] = float(np.percentile(delta, 95))
    edge = any(m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any() for m in (a, b))
    result["edge_clipped"] = bool(edge)
    if min(na, nb) < int(limits["min_pixels"]):
        result["reason"] = "too few original mask pixels for a stable estimate"
        return result
    if sa != sb:
        result.update(status="different_sampling_report_only", reason="nearest-resampled normalized-FOV comparison only")
        return result
    if edge:
        result.update(status="edge_clipped_report_only", reason="truncated silhouette; descriptive metrics only")
        return result
    if result["centroid_delta_px"] > limits["max_centroid_delta_px"]:
        result["errors"].append("instance centroid displacement exceeds pixel tolerance")
    if result["bbox_iou"] < limits["min_bbox_iou"]:
        result["errors"].append("instance bounding-box overlap below tolerance")
    if valid.any():
        depth_scale = float(np.median(0.5*(da[valid]+db[valid])))
        tolerance = max(limits["max_median_depth_delta_m"], limits["relative_depth_tolerance"]*depth_scale)
        result["depth_tolerance_m"] = tolerance
        if result["median_depth_delta_m"] > tolerance:
            result["errors"].append("shared-pixel camera-Z depth disagreement exceeds tolerance")
    result["status"] = "failed" if result["errors"] else "measured_within_tolerance"
    return result


def _load_geometry(path, errors):
    try:
        with np.load(path, allow_pickle=False) as raw:
            result = {k: raw[k].copy() for k in ("instance_id", "depth_z_m")}
            for key in ("radiance", "linear_scene"):
                if key in raw and not np.isfinite(raw[key]).all():
                    raise ValueError("non-finite "+key)
        ids, depth = result["instance_id"], result["depth_z_m"]
        if ids.ndim != 2 or depth.shape != ids.shape or min(ids.shape) < 1:
            raise ValueError("instance/depth must be equal nonempty 2D grids")
        if not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("instance_id must be integer")
        if ((ids >= 0) & (~np.isfinite(depth) | (depth <= 0))).any():
            raise ValueError("instance pixels have missing/nonpositive camera-Z depth")
        return result
    except (OSError, ValueError, KeyError) as exc:
        errors.append(str(path)+": invalid geometry buffer: "+str(exc))
        return None


def _check_measured_rd(label, path, errors, stem):
    if not path.exists():
        return
    try:
        with np.load(path, allow_pickle=False) as raw:
            r, v, power = raw["range_m"], raw["closing_speed_mps"], raw["power"]
            if len(r) < 2 or len(v) < 2 or power.shape != (len(v), len(r)):
                raise ValueError("RD needs at least two bins on each axis")
            dr, dv = float(np.median(np.diff(r))), float(np.median(np.diff(v)))
            if not (dr > 0 and dv > 0 and np.allclose(np.diff(r), dr) and np.allclose(np.diff(v), dv)):
                raise ValueError("uniform monotonic axes required for alias validation")
            if not np.isfinite(power).all() or (power < 0).any():
                raise ValueError("non-finite/negative RD power")
            measured = (label.get("processing") or {}).get("detector") == "rd_threshold"
            period = dv*len(v)
            for point in label.get("points", [])+label.get("unassociated_detections", []):
                ir = int(point.get("range_bin", np.argmin(abs(r-point["range"]))))
                iv = int(point.get("doppler_bin", np.argmin(abs(v-point["v_r"]))))
                if not (0 <= ir < len(r) and 0 <= iv < len(v)):
                    errors.append(stem+": detection bin outside raw RD")
                    continue
                if measured:
                    if abs(point["range"]-r[ir]) > 0.002 or abs(point["v_r"]-v[iv]) > 0.002:
                        errors.append(stem+": measured RD detection not on declared physical bin")
                    if any(power[iv, ir] < power[(iv+dy) % len(v), ix]
                           for dy in (-1, 0, 1) for ix in range(max(0, ir-1), min(len(r), ir+2))):
                        errors.append(stem+": measured RD detection is not a local peak")
                    threshold = (label.get("processing") or {}).get("threshold_power")
                    if threshold is not None and power[iv, ir] <= threshold:
                        errors.append(stem+": detection below saved RD threshold")
                    if "power" in point and not np.isclose(point["power"], power[iv, ir], rtol=1e-5, atol=1e-20):
                        errors.append(stem+": detection power differs from saved RD cell")
                if "v_r_true" in point:
                    alias = (float(point["v_r_true"])-v[0]) % period+v[0]
                    error = abs((float(point["v_r"])-alias+period/2) % period-period/2)
                    if error > (dv/2+0.002 if measured else 0.002):
                        errors.append(stem+": measured velocity inconsistent with single-PRF truth alias")
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        errors.append(stem+": invalid saved RD: "+str(exc))


def check(run):
    """Core completeness checks plus independent geometry and RD audits."""
    run = Path(run)
    _check_core(run)
    report = read(run / "quality.json")
    errors, registration = report["errors"], []
    cfg_path = run / "_render_cfg.json"
    cfg = read(cfg_path) if cfg_path.exists() else {}
    limits = {**DEFAULT_REGISTRATION, **cfg.get("geometry_registration", {})}
    geometry_required = bool(cfg.get("optical", {}).get("export_geometry") or limits.get("require_geometry") or
                             any(p.is_file() for mod in ("optical", "ir_mwir", "ir_lwir")
                                 for p in (run / "raw" / mod).glob("*.npz")))
    with (run / "pairs.csv").open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    unassociated_total = 0
    for row in rows:
        stem, track_id = "%06d" % int(row["frame"]), row["track_id"]
        rp = run / "labels/radar" / (stem+".json")
        radar = read(rp) if rp.exists() else {}
        _check_measured_rd(radar, run / row["rd_raw"], errors, stem)
        n_unknown = len(radar.get("unassociated_detections", [])) if "unassociated_detections" in radar else radar.get("clutter_count", 0)
        unassociated_total += n_unknown
        row.update(radar_target_detections=len(radar.get("points", [])), radar_unassociated_detections=n_unknown)
        instance_id = next((o["id"] for o in radar.get("boxes_3d", []) if o.get("name") == track_id), None)
        buffers = {}
        for mod, alias in (("optical", "rgb"), ("ir_mwir", "mwir"), ("ir_lwir", "lwir")):
            label_path = run / "labels" / mod / (stem+".json")
            label = read(label_path) if label_path.exists() else {}
            if instance_id is None:
                instance_id = next((o["id"] for o in label.get("objects", []) if o.get("name") == track_id), None)
            rel = "raw/"+mod+"/"+stem+".npz"
            row[alias+"_raw"] = rel if (run / rel).exists() else ""
            if (run / rel).exists():
                buffers[mod] = _load_geometry(run / rel, errors)
                if buffers[mod] is not None and "height" in label and "width" in label:
                    if buffers[mod]["instance_id"].shape != (label["height"], label["width"]):
                        errors.append(stem+": "+mod+" raw shape disagrees with image/label grid")
            elif geometry_required:
                errors.append("missing "+rel)
        for mod in ("ir_mwir", "ir_lwir"):
            if buffers.get("optical") is not None and buffers.get(mod) is not None and instance_id is not None:
                metric = compare_geometry(buffers["optical"], buffers[mod], instance_id, limits)
            else:
                metric = dict(status="unavailable", errors=[], reason="independent geometry or instance mapping absent")
            metric.update(frame=int(row["frame"]), track_id=track_id, pair="optical/"+mod)
            registration.append(metric)
            row[mod+"_registration_status"] = metric["status"]
            errors.extend(stem+" "+mod+": "+error for error in metric["errors"])
    if not rows:
        errors.append("no complete frame observations")
    statuses = {s: sum(m["status"] == s for m in registration) for s in sorted({m["status"] for m in registration})}
    measured = statuses.get("measured_within_tolerance", 0)
    content = ("failed" if statuses.get("failed") else
               "measured_geometry_consistency" if measured == len(registration) and measured else
               "partially_measured_geometry_consistency" if measured else
               "insufficient_support" if statuses.get("insufficient_support") else "not independently validated")
    report.update(image_content_registration=content, radar_unassociated_detections=unassociated_total,
                  registration=dict(thresholds=limits, geometry_required=geometry_required, status_counts=statuses,
                                    pairs=registration, scope="independent instance/depth passes; not radiometric detectability or real-sensor validation"),
                  errors=errors, passed=not errors)
    with (run / "pairs.csv").open("w", newline="", encoding="utf-8") as fh:
        if rows:
            writer = csv.DictWriter(fh, fieldnames=sorted(set().union(*(r.keys() for r in rows))))
            writer.writeheader()
            writer.writerows(rows)
    (run / "quality.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(run.name, json.dumps({k: v for k, v in report.items() if k != "registration"}, ensure_ascii=False))
    return report["passed"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", type=Path)
    args = ap.parse_args()
    runs = [args.path] if (args.path / "frames.json").exists() else sorted(
        p.parent for p in args.path.glob("*/frames.json"))
    if not runs:
        ap.error("no generated sequences")
    results = [check(run) for run in runs]
    raise SystemExit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
