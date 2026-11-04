"""Cheap, deterministic motion preflight before invoking any renderer."""
import numpy as np

from . import interference, plan


def inspect(cfg, n_frames=10, sensors=()):
    errors, warnings, tracks = [], [], []
    disturbances = interference.validate(cfg.get("interference"))
    dt = float((cfg.get("frames") or {}).get("dt", 0.2))
    if not np.isfinite(dt) or dt <= 0 or n_frames < 1:
        raise ValueError("motion validation needs positive dt and frame count")
    configured = cfg.get("sensors") or {}
    for sensor in sensors:
        key = "ir_camera" if sensor == "ir" else sensor
        if key not in configured:
            errors.append("requested sensor is not configured: " + sensor)
    fpc = plan.frames_per_clip(cfg, n_frames)
    for clip in range((n_frames+fpc-1)//fpc):
        cp = plan.ClipPlan(cfg, clip, fpc, dt, cfg.get("seed", 0))
        # Inspect the full nominal clip even when rendering only a short prefix.
        times = np.linspace(0, (fpc-1)*dt, max(161, min(fpc*4, 2001)))
        for obj in cfg["objects"] + [dict(name="observer", kinematics=cp.spec.get("observer_kinematics", {}))]:
            name = obj["name"]
            observer = name == "observer" and obj not in cfg["objects"]
            state = (lambda t: (np.asarray(cp.camera(t)[0]), cp.camera_velocity(t))) if observer else (
                lambda t: cp.object_state(obj, t)[:2])
            h = 1e-3
            p = np.array([state(t)[0] for t in times])
            v = np.array([state(t)[1] for t in times])
            derivative = np.array([(state(t+h)[0]-state(t-h)[0])/(2*h) for t in times])
            a = np.array([(state(t+h)[1]-state(t-h)[1])/(2*h) for t in times])
            speed = np.linalg.norm(v, axis=1)
            acceleration = np.linalg.norm(a, axis=1)
            mismatch = float(np.max(np.linalg.norm(v-derivative, axis=1)))
            prefix = "clip %d %s: " % (clip, name)
            if not all(np.isfinite(x).all() for x in (p, v, a)):
                errors.append(prefix+"non-finite motion state")
            if mismatch > 0.05:
                errors.append(prefix+"velocity disagrees with position derivative by %.3f m/s" % mismatch)
            bounds = obj.get("kinematics") or {}
            metrics = dict(max_speed_mps=float(speed.max()), min_speed_mps=float(speed.min()),
                           max_accel_mps2=float(acceleration.max()),
                           max_climb_mps=float(np.abs(v[:, 2]).max()), min_altitude_m=float(p[:, 2].min()))
            if not observer:
                metrics["max_bank_deg"] = max(abs(cp.object_attitude(obj, t)[1]) for t in times)
                if obj.get("motion_type") == "ground" and cp.ground_z is not None:
                    if np.max(np.abs(p[:, 2]-cp.ground_z)) > 1e-6:
                        errors.append(prefix+"ground target leaves the surface")
            for key, limit in bounds.items():
                if key not in metrics:
                    errors.append(prefix+"unknown kinematic limit " + key)
                elif (metrics[key] < float(limit)-1e-6 if key.startswith("min_") else metrics[key] > float(limit)+1e-6):
                    errors.append(prefix+"%s = %.3f violates %.3f" % (key, metrics[key], limit))
            tracks.append(dict(clip=clip, name=name, **metrics, derivative_error_mps=mismatch))
        if cp.mode == "platform":
            wanted, actual = cp.path.speed_cfg, cp.path.speed_track
            if abs(actual-wanted) > max(1, 0.2*wanted):
                warnings.append("clip %d: standoff solve changes platform speed %.2f -> %.2f m/s" % (clip, wanted, actual))
            if cp.path.standoff_cfg is not None and abs(cp.path.standoff_actual-cp.path.standoff_cfg) > 1:
                warnings.append("clip %d: end standoff constraint is not met" % clip)
    if n_frames > fpc:
        warnings.append("clip boundaries start independent sequences; do not track across them")
    # A motion preflight is not an acceptance verdict. Reporting passed=True here
    # let a run that died during generation leave a validation.json that still
    # read passed=True (actually observed in out/validation: stage="preflight",
    # passed=True, and images/ and labels/ both empty). Preflight reports
    # motion_ok; only artifacts() decides "passed".
    return dict(motion_ok=not errors, passed=False, errors=errors, warnings=warnings,
                interference=dict(enabled=disturbances["enabled"], events=len(disturbances["events"]), domain="image_codes"),
                frames_to_render=n_frames, nominal_frames_per_clip=fpc,
                checks_cover="full nominal clip, including analytic derivative consistency",
                tracks=tracks)


def artifacts(out_dir, n_frames, sensors, allow_empty_targets=False):
    """Check produced images, labels and actual sensor returns, not directory presence."""
    import json
    from pathlib import Path
    root = Path(out_dir)
    errors, counts, signal_frames = [], {}, {}
    cfg = json.loads((root/"meta.json").read_text(encoding="utf-8"))
    dirs = dict(optical="images/optical", radar="images/radar", lidar="lidar/range", sar="images/sar")
    if "ir" in sensors:
        for band in cfg.get("ir_bands", ["lwir"]):
            # Legacy untagged IR runs use images/ir.
            tagged = "images/ir_"+band
            dirs["ir_"+band] = tagged if (root/tagged).exists() else "images/ir"
    wanted = {k: v for k, v in dirs.items() if k in sensors or (k.startswith("ir_") and "ir" in sensors)}
    for modality, directory in wanted.items():
        found = sorted((root/directory).glob("*.png"))
        counts[modality] = len(found)
        if len(found) != n_frames:
            errors.append("%s: expected %d images, found %d" % (modality, n_frames, len(found)))
        label_dir = "labels/"+("lidar" if modality == "lidar" else Path(directory).name)
        positive = 0
        for i in range(n_frames):
            image = root/directory/("%06d.png" % i)
            if not image.exists():
                errors.append("missing "+str(image.relative_to(root)))
            elif image.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
                errors.append("invalid PNG "+str(image.relative_to(root)))
            path = root/label_dir/("%06d.json" % i)
            if not path.exists():
                errors.append("missing "+str(path.relative_to(root)))
                continue
            label = json.loads(path.read_text(encoding="utf-8"))
            boxes = label.get("boxes") or {}
            values = boxes.values() if isinstance(boxes, dict) else boxes
            positive += bool(label.get("points") if modality == "radar" else
                             any(b.get("visible", True) for b in values))
        signal_frames[modality] = positive
        if not positive and not allow_empty_targets:
            errors.append(modality+": no labelled sensor return in the validation sample")
    return dict(passed=not errors, errors=errors, image_counts=counts, labelled_frames=signal_frames)