"""Dataset generator: frames -> images + annotations, with domain randomisation."""
import json
import os
import subprocess
import sys
import time

import numpy as np

from . import (engine, interference, ir, labels, lidar, png, radar, radar_scope,
               scene, thermal, weather)


def _read_json(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def _rng(seed):
    return np.random.default_rng(int(seed))


def _segment_blocked(p0, p1, blockers):
    """True when the camera->target sight line passes through scenery.

    Scenery is approximated by a sphere around each non-labelled object. A
    sphere is yaw-independent, which matters because pose jitter rotates those
    objects every frame, and it errs towards rejecting a placement rather than
    shipping a frame whose target is hidden behind a barrier.
    """
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    v = p1 - p0
    l2 = float(v.dot(v))
    for c, r in blockers:
        if l2 <= 1e-12:
            d = float(np.linalg.norm(c - p0))
        else:
            u = float(np.clip((c - p0).dot(v) / l2, 0.0, 1.0))
            d = float(np.linalg.norm(p0 + u * v - c))
        if d < r:
            return True
    return False


def object_poses(cfg, t, jitter):
    out = {}
    for spec in cfg.get("objects", []):
        name = spec.get("name")
        pos = np.array(spec.get("pos", [0, 0, 0]), float) + np.array(spec.get("vel", [0, 0, 0]), float) * t
        yaw = float(spec.get("rot", 0.0))
        if jitter and name in jitter:
            j = jitter[name]
            pos = pos + np.array(j.get("dpos", [0, 0, 0]), float)
            yaw += float(j.get("dyaw", 0.0))
        out[name] = dict(pos=[round(float(v), 4) for v in pos], yaw=round(yaw, 3))
    return out


def plan_frames(cfg, n_frames, base_seed, dt, frame_offset=0, total_frames=None):
    """Delegate to simkit.plan: a run is a sequence of continuous clips.

    Kept as a module-level name because merge_parts and the self-test import it,
    but the per-frame randomisation that used to live here is gone on purpose:
    independent draws are what made the previous dataset incoherent.
    """
    from . import plan
    for s in plan.plan_frames(cfg, n_frames, base_seed, dt,
                              frame_offset=frame_offset, total_frames=total_frames):
        yield s


#: Renderers cannot allocate a 0-sized or unbounded frame; reject those here
#: instead of letting them fail with an opaque numpy/Blender traceback.
_MAX_DIM = 4096


def _check_dims(w, h, where):
    """Raise ValueError for a width/height that is invalid or out of range."""
    for label, val in (("width", w), ("height", h)):
        if val is None:            # not declared: _view_pose falls back to 480/360
            continue
        try:
            n = int(val)
        except (TypeError, ValueError):
            raise ValueError("%s 的 %s 不是有效整数：%r" % (where, label, val))
        if n <= 0 or n > _MAX_DIM:
            raise ValueError("%s 的 %s=%s 非法（须为 1..%d）；请修正场景里的 width/height。"
                             % (where, label, n, _MAX_DIM))


#: Sensor blocks that carry a camera image size. Deliberately NOT sar/radar/
#: lidar: those blocks describe physics grids, not a requested W/H.
_SIZE_CAMERAS = ("optical", "eo_camera", "ir_camera", "camera")


def _size_override_present(cfg, width, height):
    """True when the scene asks for a W/H that non-optical modes cannot honour.

    The workbench size feature writes the recipe dims into the scene JSON
    (``clip.views[]`` for dual/seeker views and the camera blocks on the legacy
    path); neither CLI nor studio passes width/height to run(), so the cfg
    checks below are the path that actually fires in production. The explicit
    kwargs are kept only as a compatible fallback.
    """
    if width is not None or height is not None:
        return True
    if cfg.get("width") is not None or cfg.get("height") is not None:
        return True
    for v in ((cfg.get("clip") or {}).get("views") or []):
        if isinstance(v, dict) and (v.get("width") is not None
                                    or v.get("height") is not None):
            return True
    sensors = cfg.get("sensors") or {}
    for name in _SIZE_CAMERAS:
        cam = sensors.get(name)
        if isinstance(cam, dict) and (cam.get("width") is not None
                                      or cam.get("height") is not None):
            return True
    rv = ((cfg.get("recipe") or {}).get("view") or {})
    if isinstance(rv, dict) and (rv.get("width") is not None
                                 or rv.get("height") is not None):
        return True
    return False


def _gsd_m_per_px(range_m, fov_deg, width):
    """Ground sample distance at the target: 2*range*tan(fov/2)/W."""
    if range_m is None or fov_deg is None or not width:
        return None
    try:
        r, f, w = float(range_m), float(fov_deg), float(width)
    except (TypeError, ValueError):
        return None
    if r <= 0 or f <= 0 or f >= 180 or w <= 0:
        return None
    return round(2.0 * r * np.tan(np.deg2rad(f) / 2.0) / w, 6)


def _views_of(states, cfg):
    """Union of the field-of-view definitions used anywhere in the run.

    A clip may render a narrow seeker lens and a wide search lens at the same
    instant; the directory layout has to know every tag before the first frame
    is written. Runs without "views" fall back to the legacy single camera so
    every older scene keeps producing exactly the layout it used to.
    """
    out, seen = [], set()
    for s in states:
        for name, v in (s.get("views") or {}).items():
            if name in seen:
                continue
            seen.add(name)
            out.append(dict(name=name, tag=str(v.get("tag", "") or ""),
                            fov_deg=v.get("fov_deg"), width=v.get("width"),
                            height=v.get("height")))
    if not out:
        cam = ((cfg.get("sensors") or {}).get("optical")
               or (cfg.get("sensors") or {}).get("ir_camera") or {})
        out = [dict(name="optical", tag="", fov_deg=cam.get("fov_deg"),
                    width=cam.get("width"), height=cam.get("height"))]
    # One strict gate for every declared dimension, on both the clip.views and
    # the legacy path, before any renderer can see a 0 or a huge allocation.
    for v in out:
        _check_dims(v.get("width"), v.get("height"), "view %r" % v["name"])
    out[0]["tag"] = ""        # the primary view owns the legacy paths
    return out


def _view_pose(state, view, cfg):
    """Per-frame camera dict for one view (falls back to the sensor config)."""
    v = (state.get("views") or {}).get(view["name"])
    if v:
        # Only a missing/None value falls back; an explicit 0 must reach
        # _check_dims and be rejected instead of silently becoming 480/360.
        w, h = v.get("width"), v.get("height")
        w = view["width"] if w is None else w
        h = view["height"] if h is None else h
        w = 480 if w is None else w
        h = 360 if h is None else h
        width, height = int(w), int(h)
        _check_dims(width, height, "view %r (frame)" % view["name"])
        return dict(pos=[float(x) for x in v["pos"]],
                    look_at=[float(x) for x in v.get("look_at", [0, 0, 0])],
                    fov_deg=float(v.get("fov_deg") or view["fov_deg"]),
                    width=width, height=height,
                    roll_deg=float(v.get("roll_deg", 0.0) or 0.0))
    cam = ((state.get("cfg") or {}).get("sensors") or {}).get("ir_camera", {})
    _cw, _ch = cam.get("width"), cam.get("height")
    width = int(480 if _cw is None else _cw)
    height = int(360 if _ch is None else _ch)
    _check_dims(width, height, "view %r (ir_camera)" % view["name"])
    return dict(pos=cam.get("pos", [0, 0, 0]), look_at=cam.get("look_at", [0, 0, 1]),
                fov_deg=float(cam.get("fov_deg", 30.0)),
                width=width, height=height,
                roll_deg=0.0)


def _visible_from(world, cam, obj):
    """True when nothing stands between the camera and this object's centre."""
    try:
        c = np.asarray(cam["pos"], float)
        corners = np.asarray(obj["corners"], float)
        tgt = corners.mean(axis=0)
        d = tgt - c
        L = float(np.linalg.norm(d))
        if L < 1e-6:
            return True
        hit = world.bvh.ray_cast(c, (d / L)[None, :])
        if not hit["hit"][0]:
            return True
        t = float(hit["t"][0])
        return t >= L - 1.0
    except (KeyError, AttributeError, TypeError, ValueError):
        # No bvh, no corners, malformed geometry: treat as visible, but only for
        # the shapes of failure that mean "this world cannot answer", not for
        # every exception. A blanket except Exception also hid numerical faults
        # and silently scored occluded targets as positive samples with no log.
        if os.environ.get("SIMKIT_STRICT_VISIBILITY"):
            raise
        return True


def _boxes_for(world, obj_id, cam, width, height, min_px=1, seen=None):
    """Registered 2D boxes: geometry decides the box, the mask decides visibility.

    Two renderers see the same 17-pixel aircraft differently: Blender's id pass
    covers its full silhouette (30 px wide) while the thermal rasteriser, shading
    face by face, drops the thin triangles and boxes only the fuselage (17 px).
    Using each renderer's own mask therefore puts the optical and the thermal
    label a few pixels apart on the same object -- a shift that does not exist in
    the world and that a fused training set would learn as signal.

    So the box always comes from projecting the object's 3D extent, and the id
    mask is only asked "is any of it visible". Every camera that shares a pose
    then produces byte-identical labels, which is what "registered" means.
    """
    if cam is None:
        return (labels.boxes_from_mask(obj_id, world, min_px=min_px)
                if obj_id is not None else {})
    proj = labels.boxes_from_corners(world, cam, width, height, min_frac=0.0)
    # obj_id is the id buffer of the *numpy* renderers; the Blender pass hands its
    # verdict over as `seen` (names it actually drew) instead.
    mask = (labels.boxes_from_mask(obj_id, world, min_px=1)
            if obj_id is not None else {})
    seen = set(seen or ())
    out = {}
    for o in world.objs:
        if not o.get("label", True) or o["id"] not in proj:
            continue
        mb = mask.get(o["id"])
        drawn = o["name"] in seen
        if mb is None and not drawn and not _visible_from(world, cam, o):
            continue                      # genuinely behind something: no label
        b = dict(proj[o["id"]])
        b["pixels"] = int(mb.get("pixels", 0)) if mb else 0
        b["visible"] = (mb is not None) or drawn
        b["source"] = "corners"
        out[o["id"]] = b
    return out


#: Metres of airframe one tile of the plate covers.
_SKIN_SPAN_M = 20.0


def _skin_plate(cfg, out_dir):
    """Bake albedo x modulation into a colour plate; return its path or None.

    The plate carries the FINAL surface colour, so the renderer only has to
    connect an image texture straight into Base Color.  A minimal Blender repro
    proved that path renders (a checkerboard cube showed the pattern) while the
    multiply-node chain used before did not, so the simplest verified wiring is
    the one that ships.

    Off unless the scene sets optical.skin.
    """
    opt = cfg.get("optical") or {}
    # A scene may point at its own plate (a checkerboard while wiring, a baked
    # texture later); baking over it silently replaced whatever was supplied.
    given = opt.get("skin_patch")
    if given and os.path.isfile(str(given)):
        return os.path.abspath(str(given))
    if not opt.get("skin"):
        return None
    mat = None
    for spec in cfg.get("objects", []):
        name = str(spec.get("mat") or "")
        if name == "painted_metal" or name.startswith("paint_"):
            mat = name
            break
    if mat is None:
        return None
    try:
        from . import materials as matlib
        from . import skin
        albedo = np.asarray(matlib.get(mat)["albedo"], float)
        px_m = float(opt.get("skin_px_m", 0.04))
        field = skin.SkinField(mat, seed=int(cfg.get("seed", 0)),
                               pixel_scale_m=px_m,
                               contrast=float(opt.get("skin_contrast", 1.0)))
        mod = np.asarray(field.patch(_SKIN_SPAN_M * 0.6, _SKIN_SPAN_M,
                                     shape=(512, 512), pixel_scale_m=px_m), float)
        rgb = np.clip(mod[..., None] * albedo[None, None, :], 0.0, 1.0)
        path = os.path.abspath(os.path.join(out_dir, "_skin_plate.png"))
        from PIL import Image as _Image
        _Image.fromarray((rgb * 255.0 + 0.5).astype(np.uint8), mode="RGB").save(path)
        return path
    except Exception:
        return None


def _render_config(cfg, views, out_dir, part=None, base_seed=None):
    """Side-car scene config for the Blender pass.

    Blender re-reads the config from disk, so two things have to be true there:
    the per-clip view list is present, and every model path is absolute (the
    side-car does not live next to the scene JSON, so relative OBJ paths would
    silently resolve to the wrong place).

    base_seed is the effective CLI/config seed used by plan_frames for the numpy
    modalities.  Blender derives its photon/camera noise from cfg['seed'] with
    the SAME formula (seed * 100003 + frame), so pinning the side-car to the
    scene file's literal seed split the random streams: --seed changed IR/radar
    but left the optical frames bit-identical.  Propagate it explicitly; when
    the caller omits it the value equals cfg['seed'] and nothing changes.
    """
    rc = json.loads(json.dumps(cfg))
    if base_seed is not None:
        rc["seed"] = int(base_seed)
    base = rc.get("_base_dir", ".")
    for spec in rc.get("objects", []):
        m = spec.get("model")
        if m and not os.path.isabs(m):
            spec["model"] = os.path.abspath(os.path.join(base, m))
        material_source = spec.get("optical_material_source")
        if material_source and not os.path.isabs(material_source):
            spec["optical_material_source"] = os.path.abspath(os.path.join(base, material_source))
    # Decoys and jammers are physical bodies for radar/IR/lidar, but this side-car
    # is what the Blender pass builds from, so they have to be listed here too.
    from . import ecm as _ecmlib
    for spec in _ecmlib.render_objects(cfg):
        rc.setdefault("objects", []).append(spec)
    plate = _skin_plate(cfg, out_dir)
    if plate:
        rc.setdefault("optical", {})["skin_patch"] = plate
    rc["_base_dir"] = os.path.abspath(base)
    rc.setdefault("optical", {})["views"] = [
        dict(name=v["name"], tag=v["tag"], fov_deg=v.get("fov_deg"),
             width=v.get("width"), height=v.get("height")) for v in views]
    name = "_render_cfg.json" if part is None else "_render_cfg.part%03d.json" % int(part)
    path = os.path.join(out_dir, name)
    labels.write_json(path, rc)
    return rc, os.path.abspath(path)


def _preserve_renderer_visibility(world, boxes_by_id, renderer_boxes):
    """Transfer measured visibility by object name, never integer-ID lookup."""
    names = {obj["id"]: obj["name"] for obj in world.objs}
    for oid, box in boxes_by_id.items():
        source = renderer_boxes.get(names[oid], {})
        if source.get("pixels") is not None:
            box["pixels"] = int(source["pixels"])
        if "visibility_source" in source:
            box["visibility_source"] = source["visibility_source"]
        box["visible"] = bool(source.get("visible", box.get("visible", True)))
    return boxes_by_id


def _sensors_on_disk(out_dir):
    """Which modalities actually have images under `out_dir`."""
    img = os.path.join(out_dir, "images")
    if not os.path.isdir(img):
        return []
    have = set()
    for d in os.listdir(img):
        if not os.path.isdir(os.path.join(img, d)):
            continue
        try:
            if not any(n.endswith(".png") for n in os.listdir(os.path.join(img, d))):
                continue
        except OSError:
            continue
        # "ir_preview_lwir" also starts with "ir", so a preview-only run would
        # claim a sensor it cannot actually produce 16-bit data for. The preview
        # tree is a derived view of images/ir_*, never a product in its own right.
        if d.startswith("ir_preview"):
            continue
        if d.startswith("optical"):
            have.add("optical")
        elif d.startswith("ir"):
            have.add("ir")
        elif d.startswith("radar"):
            have.add("radar")
        elif d.startswith("sar"):
            have.add("sar")
    # lidar is the odd one out: rangemaps live in lidar/range, not images/
    for sub in ("lidar/range", "lidar/intensity"):
        d = os.path.join(out_dir, sub)
        if os.path.isdir(d) and any(n.endswith(".png") for n in os.listdir(d)):
            have.add("lidar")
            break
    return sorted(have)


# --------------------------------------------------------------------------
# Optional JPEG product
# --------------------------------------------------------------------------
#: Quality used when output.jpeg is switched on without an explicit number.
JPEG_DEFAULT_QUALITY = 90


def _jpeg_settings(cfg):
    """Parse the optional output.jpeg block; None when switched off.

    The switch deliberately lives under output and not under
    optical.camera.jpeg: the latter is the sensor model evaluated inside
    Blender, whose bundled Python has no Pillow.  This one runs in dataset.py
    after the render, where the project interpreter (and Pillow) is present.
    Absent/False keeps the historical PNG-only output byte for byte.
    """
    raw = (cfg.get("output") or {}).get("jpeg")
    if raw is None or raw is False:
        return None
    if isinstance(raw, dict):
        if not raw.get("enabled", True):
            return None
        quality = raw.get("quality", JPEG_DEFAULT_QUALITY)
        span = raw.get("quality_range")
    elif isinstance(raw, bool):                    # True: just switch it on
        quality, span = JPEG_DEFAULT_QUALITY, None
    elif isinstance(raw, int):                     # "jpeg": 90
        quality, span = raw, None
    else:
        raise ValueError("output.jpeg must be a boolean, an integer quality, or an object")
    try:
        quality = int(quality)
    except (TypeError, ValueError):
        raise ValueError("output.jpeg.quality must be an integer in 1..100")
    if not 1 <= quality <= 100:
        raise ValueError("output.jpeg.quality must be an integer in 1..100")
    if span is not None:
        try:
            low, high = (int(v) for v in span)
        except (TypeError, ValueError):
            raise ValueError("output.jpeg.quality_range must be [low, high]")
        if not 1 <= low <= high <= 100:
            raise ValueError("output.jpeg.quality_range must satisfy "
                             "1 <= low <= high <= 100")
        span = (low, high)
    return dict(quality=quality, quality_range=span)


def _jpeg_quality(settings, base_seed, frame):
    """Quality for one frame, drawn from quality_range when it is set.

    Seeding on (run seed, frame) makes a quality_range run reproducible: the
    same dataset config always produces the same quantisation tables, which is
    what lets someone re-render one frame and compare it.
    """
    span = settings.get("quality_range")
    if span is None:
        return int(settings["quality"])
    low, high = span
    rng = np.random.default_rng([int(base_seed), int(frame)])
    return int(rng.integers(low, high + 1))


def _mark_label_jpeg(out_dir, tag, frame, filename, quality):
    """Record the JPEG copy in the frame label (geometry labels are untouched)."""
    path = os.path.join(out_dir, "labels", "optical" + tag, "%06d.json" % int(frame))
    try:
        with open(path, "r", encoding="utf-8") as stream:
            label = json.load(stream)
    except (OSError, ValueError):
        return
    label["jpeg"] = dict(image=filename, quality=int(quality))
    labels.write_json(path, label)


def _write_jpeg_copies(out_dir, settings, views, frame_ids, base_seed, quiet=False):
    """Write a JPEG copy beside every optical PNG the renderer produced.

    The PNG remains the canonical product.  Every downstream consumer here
    (COCO file_name, YOLO export, contact sheets, consistency checks) resolves
    the '.png' and deleting it would silently break them, so the JPEG is an
    additional product and the COCO references keep pointing at the PNG that
    still exists.  Callers that want JPEG-only storage can prune the PNGs
    afterwards, once they have verified their own consumers.

    Returns a small accounting dict for meta.json, or None when switched off.
    """
    if settings is None:
        return None
    from PIL import Image
    from . import camera
    written = missing = 0
    png_bytes = jpeg_bytes = 0
    qualities = {}
    for view in views:
        tag = view.get("tag", "") or ""
        for frame in frame_ids:
            stem = "%06d" % int(frame)
            source = os.path.join(out_dir, "images", "optical" + tag, stem + ".png")
            if not os.path.exists(source):
                missing += 1
                continue
            quality = _jpeg_quality(settings, base_seed, frame)
            target = os.path.join(out_dir, "images", "optical" + tag, stem + ".jpg")
            with Image.open(source) as image:
                rgb = np.asarray(image.convert("RGB"), np.uint8)
            camera.write_jpeg(target, rgb, quality)
            written += 1
            png_bytes += os.path.getsize(source)
            jpeg_bytes += os.path.getsize(target)
            key = str(quality)
            qualities[key] = qualities.get(key, 0) + 1
            _mark_label_jpeg(out_dir, tag, frame, stem + ".jpg", quality)
    report = dict(enabled=True, quality=int(settings["quality"]),
                  quality_range=(None if settings.get("quality_range") is None
                                 else list(settings["quality_range"])),
                  files=written, missing=missing,
                  png_bytes=png_bytes, jpeg_bytes=jpeg_bytes, qualities=qualities)
    if png_bytes:
        report["ratio"] = round(jpeg_bytes / float(png_bytes), 4)
    if missing and not quiet:
        print("WARN jpeg: %d optical PNG(s) missing, skipped" % missing, flush=True)
    return report


def _jpeg_inventory(out_dir, settings):
    """Summarise the JPEG copies already on disk.

    A chunked run writes its own frames and then returns before meta.json
    exists; merge_parts is the only place that can describe the whole run.
    """
    images = os.path.join(out_dir, "images")
    if not os.path.isdir(images):
        return None
    files = png_bytes = jpeg_bytes = 0
    for name in sorted(os.listdir(images)):
        folder = os.path.join(images, name)
        if not name.startswith("optical") or not os.path.isdir(folder):
            continue
        for fn in sorted(os.listdir(folder)):
            if not fn.endswith(".jpg"):
                continue
            target = os.path.join(folder, fn)
            files += 1
            jpeg_bytes += os.path.getsize(target)
            source = os.path.join(folder, fn[:-4] + ".png")
            if os.path.exists(source):
                png_bytes += os.path.getsize(source)
    if not files:
        return None
    report = dict(enabled=True, quality=int(settings["quality"]),
                  quality_range=(None if settings.get("quality_range") is None
                                 else list(settings["quality_range"])),
                  files=files, png_bytes=png_bytes, jpeg_bytes=jpeg_bytes)
    if png_bytes:
        report["ratio"] = round(jpeg_bytes / float(png_bytes), 4)
    return report


def _ecm_summary(cfg, frames):
    """Condense per-frame ECM reports into what a consumer needs to know.

    A dataset that carries jamming must say so, and must say it in numbers a
    reader can check: which entities were present, how far the noise floor was
    pushed, and how many detections the jammers themselves contributed.  The
    per-frame detail is kept next to the summary so nothing is rounded away.
    """
    spec = cfg.get("ecm") or {}
    rises = [float((f.get("noise") or {}).get("snr_drop_db") or 0.0) for f in frames]
    js = [float((f.get("noise") or {}).get("js_db") or -999.0) for f in frames]
    injected = [int(f.get("injected_echoes") or 0) for f in frames]
    return dict(
        domain="rf",
        model="simkit.ecm",
        entities=[dict(id=e.get("id"), kind=e.get("kind"), mode=e.get("mode"),
                       pos=e.get("pos"), vel=e.get("vel"))
                  for e in (spec.get("entities") or [])],
        frames=len(frames),
        noise_rise_db=dict(min=round(min(rises), 3), max=round(max(rises), 3)),
        js_db=dict(min=round(min(js), 3), max=round(max(js), 3)),
        injected_echoes=sum(injected),
        per_frame=frames,
    )


def _po_rcs_db(world, obj_id, u, lam):
    """Physical-optics RCS of one object seen from the radar, in dBsm.

    radar.coherent_po_sigma sums the facet fields before squaring, so the answer
    does not depend on how finely the airframe was triangulated. The Lambertian
    diffuse term is not included: this is the specular number the range-Doppler
    echo is dominated by, and quoting a composite would hide the aspect
    dependence that makes a fighter's RCS swing 20 dB in a few degrees.
    """
    try:
        from . import materials as matlib
        faces = np.asarray(world.face_obj) == int(obj_id)
        if not faces.any():
            return None
        tri = world.verts[world.faces[faces]]
        centres = tri.mean(axis=1)
        normals = radar._face_normals(world)[faces]
        areas = radar._face_areas(world)[faces]
        names = sorted(matlib.MATERIALS)
        rho_tab = np.array([matlib.MATERIALS[n]["rf_rho"] for n in names])
        rho = rho_tab[world.face_mat[faces]]
        sigma = radar.coherent_po_sigma(centres, normals, areas, rho,
                                        -np.asarray(u, float), np.asarray(u, float), lam)
        if not np.isfinite(sigma) or sigma <= 0.0:
            return None
        return 10.0 * np.log10(sigma)
    except (AttributeError, KeyError, TypeError, ValueError, ZeroDivisionError):
        return None


def radar_target_rows(world, radar_cfg, detections):
    """Truth range/closing speed per labelled object, plus its measured peak.

    The RD/RA figures print this list, not the detection list. A target the
    detector missed still has a true range and closing speed, and printing only
    detections made a clutter peak look like the target it was standing next to;
    conversely a target whose echo merged with another one shows up as two rows
    at the same range, which is exactly the information a reader needs to see
    that a single bright cell is two aircraft.
    """
    pos = np.asarray(radar_cfg.get("pos", [0.0, 0.0, 0.0]), float)
    v_pl = np.asarray(radar_cfg.get("vel", [0.0, 0.0, 0.0]), float)
    _fc = float(radar_cfg.get("fc_ghz") or 0.0)
    if _fc <= 0.0:
        _fc = 15.0                      # Ku default, the band most scenes pick
    lam = radar.C_LIGHT / (_fc * 1e9)
    try:
        _p, right, up, forward = scene.sensor_basis(
            radar_cfg.get("pos"), radar_cfg.get("look_at"),
            radar_cfg.get("up", (0.0, 0.0, 1.0)), radar_cfg.get("roll_deg", 0.0))
    except (KeyError, TypeError, ValueError):
        right, up, forward = np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])
    best = {}
    for det in detections or []:
        oid = int(det.get("obj", -1))
        if oid < 0:
            continue
        cur = best.get(oid)
        if cur is None or float(det.get("snr_db", 0.0)) > float(cur.get("snr_db", 0.0)):
            best[oid] = det
    rows = []
    for o in getattr(world, "objs", None) or []:
        if not o.get("label", True):
            continue
        c = np.asarray(o["corners"], float).mean(axis=0)
        v = c - pos
        R = float(np.linalg.norm(v))
        if R < 1e-6:
            continue
        u = v / R
        front = float(u @ forward)
        az = float(np.degrees(np.arctan2(float(u @ right), front))) if front != 0.0 else 0.0
        el = float(np.degrees(np.arctan2(float(u @ up), front))) if front != 0.0 else 0.0
        vr = float(-(u @ (np.asarray(o.get("vel", [0.0, 0.0, 0.0]), float) - v_pl)))
        # RCS of this airframe at this aspect, from the same physical-optics
        # integral the echo is built from. A table that lists range and closing
        # speed but not RCS cannot answer "how bright is this thing supposed to
        # be", which is the first question a radar dataset has to answer.
        rcs_db = o.get("rcs_db")
        if rcs_db is None:
            rcs_db = _po_rcs_db(world, o["id"], u, lam)
        row = dict(name=o["name"], cls=o.get("cls"), range=round(R, 1),
                   v_r=round(vr, 1), az=round(az, 2), el=round(el, 2),
                   rcs_dbsm=(round(float(rcs_db), 1) if rcs_db is not None else None))
        det = best.get(int(o["id"]))
        if det is not None:
            row.update(measured_range=round(float(det["range"]), 1),
                       measured_v_r=round(float(det["v_r"]), 2),
                       snr_db=round(float(det["snr_db"]), 1), detected=True)
        else:
            row["detected"] = False
        rows.append(row)
    return rows


def run(cfg_path, out_dir, n_frames=None, sensors=("ir", "radar"), seed=None, quiet=False,
        frame_offset=0, part=None, total_frames=None, width=None, height=None):
    cfg = scene.load_config(cfg_path)
    jpeg_settings = _jpeg_settings(cfg)
    disturbances = interference.validate(cfg.get("interference"))
    if disturbances["enabled"]:
        from PIL import Image  # Fail before rendering if the optional dependency is missing.
    sensors = tuple(sensors)
    if "all" in sensors:
        sensors = ("optical", "ir", "radar", "lidar", "sar")
    # SAR has no engine-default fallback: without sensors.sar the old gate
    # (have_sar) skipped the whole branch and the run exited 0 with an empty
    # product. Fail loudly instead. ir/radar/lidar keep their intentional
    # "undeclared -> engine defaults" behaviour; do not touch that.
    if "sar" in sensors and not (cfg.get("sensors") or {}).get("sar"):
        raise ValueError(
            "请求了 sar，但场景缺少 sensors.sar 配置，SAR 不会产出任何图像。"
            "请在场景 JSON 的 sensors 下补一个 sar 块，最小可用配置："
            '"sensors": {"sar": {"band": "X", "n_cross": 256, "n_samples": 512, '
            '"swath_center_m": 2500, "swath_width_m": 280}}。'
            "SAR 图幅由 n_cross × n_samples 决定，不能用 width/height 覆盖。")
    if jpeg_settings is not None and "optical" in sensors:
        # Fail before a multi-hour render: the JPEG pass runs here, in the
        # project interpreter, and needs the one dependency Blender lacks.
        from PIL import Image  # noqa: F401
    r = cfg.get("randomize", {})
    base_seed = int(seed if seed is not None else cfg.get("seed", 0))
    n_frames = int(n_frames if n_frames is not None else cfg.get("frames", {}).get("n", 1))
    dt = float(cfg.get("frames", {}).get("dt", 0.2))

    cls_names = sorted({o.get("cls", "target") for o in cfg.get("objects", [])
                        if o.get("label", True)})
    cls_id = {c: i + 1 for i, c in enumerate(cls_names)}

    ir_bands = (cfg.get("ir", {}) or {}).get("bands") or [None]

    # A run may be generated in chunks: frame_offset shifts both the random
    # stream and the COCO image ids, so the chunks concatenate into one valid
    # dataset. Image ids therefore start at frame_offset, not at 0.
    img_off = int(frame_offset)
    states = list(plan_frames(cfg, n_frames, base_seed, dt, frame_offset=frame_offset,
                              total_frames=total_frames))
    views = _views_of(states, cfg)
    # Legacy scenes declare sizes in sensors.* only. _views_of keeps its
    # optical-first priority, so an ir_camera with different dimensions is
    # silently ignored; say so instead of shipping a size nobody asked for.
    _sm = cfg.get("sensors") or {}
    _opt, _irc = _sm.get("optical"), _sm.get("ir_camera")
    if _opt and _irc and ((_opt.get("width"), _opt.get("height"))
                          != (_irc.get("width"), _irc.get("height"))):
        print("  [警告] 场景同时在 sensors.optical 与 sensors.ir_camera 声明了尺寸，"
              "但两者不一致（optical=%sx%s，ir_camera=%sx%s）；legacy 路径按既有优先级"
              "取 optical 并忽略 ir_camera 的尺寸。请统一两者或改用 clip.views 明确指定。"
              % (_opt.get("width"), _opt.get("height"),
                 _irc.get("width"), _irc.get("height")), file=sys.stderr)

    # Only the sensors this run actually produces get directories: an empty
    # images/sar in an IR-only run makes the exporter and the site think the
    # modality exists.
    dirs = {}
    if "radar" in sensors:
        # Two images per map: the annotated figure (axes + target table) and the
        # clean picture (same map and axes, no table, no markers). The table used
        # to be painted into the only image, so a training loader saw text and a
        # paper had no plain sensor picture to show.
        dirs.update(radar="images/radar", radar_ra="images/radar_ra",
                    radar_clean="images/radar_clean",
                    radar_ra_clean="images/radar_ra_clean",
                    label_radar="labels/radar")
    if "lidar" in sensors:
        dirs.update(lidar_range="lidar/range", lidar_intensity="lidar/intensity",
                    lidar_points="lidar/points", label_lidar="labels/lidar")
    have_sar = "sar" in sensors and bool((cfg.get("sensors") or {}).get("sar"))
    if have_sar:
        dirs.update(sar="images/sar", label_sar="labels/sar")
    for v in views:
        t = v["tag"]
        if "optical" in sensors:
            dirs["optical" + t] = "images/optical" + t
            dirs["label_optical" + t] = "labels/optical" + t
        if "ir" in sensors:
            for bnm in ir_bands:
                bt = "" if bnm is None else "_" + bnm
                dirs["ir" + bt + t] = "images/ir" + bt + t
                dirs["ir_preview" + bt + t] = "images/ir_preview" + bt + t
                dirs["label_ir" + bt + t] = "labels/ir" + bt + t
    for d in dirs.values():
        os.makedirs(os.path.join(out_dir, d), exist_ok=True)

    cats = [dict(id=i, name=c, supercategory="target") for c, i in cls_id.items()]

    def coco_shell(desc):
        return dict(info=dict(description=desc, version="0.1"),
                    images=[], annotations=[], categories=json.loads(json.dumps(cats)))

    # One COCO per (band, view): a narrow seeker lens and a wide search lens
    # are separate image sets with separate boxes, and merging them would make
    # one "image_id" mean two pictures.
    coco_irs = {}
    for bnm in ir_bands:
        for v in views:
            bdesc = "IR" if bnm is None else "IR " + bnm
            coco_irs[(bnm, v["tag"])] = coco_shell(
                "simkit synthetic dataset (%s%s)" % (bdesc, " " + v["name"] if v["tag"] else ""))
    coco_opt = {v["tag"]: coco_shell("simkit synthetic dataset (optical%s)"
                                     % (" " + v["name"] if v["tag"] else ""))
                for v in views}
    coco_lidar = coco_shell("simkit synthetic dataset (lidar range image)")
    coco_sar = coco_shell("simkit synthetic dataset (SAR stripmap)")
    radar_frames = []
    sar_frames = []
    # Per-frame ECM effect, kept so a dataset says whether it was generated
    # with scene-level jamming and exactly what that jamming measured.
    ecm_frames = []
    jpeg_report = None

    t_start = time.time()

    # A missing sensor falls back to engine defaults for everything except
    # optical, which is a hard error. That asymmetry is fine, but it used to be
    # silent: demo_optical declares only an optical camera and still produced
    # IR and radar frames from parameters the scene never chose. Say so once.
    _declared = set((cfg.get("sensors") or {}).keys())
    _alias = {"ir": ("ir_camera", "ir"), "optical": ("optical",),
              "radar": ("radar",), "sar": ("sar",), "lidar": ("lidar",)}
    for _s in sensors:
        if not _declared:
            break
        if not any(a in _declared for a in _alias.get(_s, (_s,))):
            print("  [提示] 场景未声明 %s 传感器，使用引擎默认参数生成。"
                  "如需精确复现，请在场景里补 sensors.%s。"
                  % (_s, "ir_camera" if _s == "ir" else _s), file=sys.stderr)

    # Non-optical image sizes come from physics, not from a W/H request. When a
    # caller supplies a size override, say which parameters actually set the
    # grid instead of silently ignoring the request. stdout only, quiet skips it.
    _phys = [m for m in ("sar", "radar", "lidar") if m in sensors]
    if _phys and _size_override_present(cfg, width, height) and not quiet:
        print("  [提示] %s 的图幅由物理参数决定，不受 width/height 覆盖影响。"
              "要达到目标 W×H，请改这些配置：" % "/".join(_phys))
        if "sar" in sensors:
            print("    SAR    = sensors.sar.n_cross × sensors.sar.n_samples")
        if "radar" in sensors:
            print("    雷达RD = (sensors.radar.n_samples // 2) × sensors.radar.n_chirps")
        if "lidar" in sensors:
            print("    LiDAR  = sensors.lidar.n_el × sensors.lidar.n_az")

    if "optical" in sensors:
        # A scene may legitimately have no optical camera at all (demo_ir only
        # declares ir_camera). Asking for one anyway used to die deep inside the
        # Blender script with a bare KeyError: 'optical'; say what is actually
        # available instead, because the fix is to change -s, not to debug.
        _views = ((cfg.get("clip") or {}).get("views") or cfg.get("views") or [])
        _sm = cfg.get("sensors") or {}
        if not _views and not _sm.get("optical"):
            _have = sorted(_sm.keys()) or ["(无)"]
            _hint = {"ir_camera": "ir", "radar": "radar", "sar": "sar",
                     "lidar": "lidar"}.get(_have[0], "ir")
            raise RuntimeError(
                "该场景没有可见光相机，无法渲染 optical。场景声明的传感器：%s。"
                "请改用 -s %s。" % ("、".join(_have), _hint))
        from . import external
        backend = (cfg.get("optical") or {}).get("backend", "blender")
        manifest = external.write_manifest(cfg, states, out_dir, part=part)
        if backend == "blender":
            from . import optical
            if not optical.available(cfg):
                raise RuntimeError("optical sensor requested but Blender not found "
                                   "(set SIMKIT_BLENDER or optical.blender)")
            script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  "blender", "render_optical.py")
            # The renderer reads the config from disk, so the per-clip views
            # (and absolute model paths, because the temp config lives
            # elsewhere) are baked into a side-car file instead of mutating the
            # scene JSON the user owns.
            rcfg, rcfg_path = _render_config(cfg, views, out_dir, part,
                                             base_seed=base_seed)
            optical.render_frames(rcfg, rcfg_path, out_dir, manifest, script, quiet=quiet,
                                  view_tags=[v["tag"] for v in views])
        else:
            external.run(cfg, out_dir, manifest, backend, quiet=quiet)
        # The renderer wrote PNGs (and, in Blender, cannot see Pillow).  The
        # optional lossy product is written here, after the render, beside the
        # lossless one.  COCO keeps naming the .png that still exists.
        jpeg_report = _write_jpeg_copies(
            out_dir, jpeg_settings, views, [int(s["frame"]) for s in states],
            base_seed, quiet=quiet)

    for si, s in enumerate(states):
        # plan_frames already applied frame_offset, so f is the global frame id
        # and the file numbers below stay unique across chunks.
        f = s["frame"]
        fid = f
        t, jit = s["t"], s["jitter"]
        fr_cfg = s["cfg"]
        if set(sensors) & {"ir", "radar", "lidar", "sar", "optical"}:
            # Scene-level ECM is time-parameterised, so the frame clock has to be
            # on the config BEFORE the world is built: build_world attaches the
            # entity triangles at their t-th pose, and a chaff cloud that is
            # read at t=0 for every frame would never bloom or drift. The radar
            # branch below re-assigns the same key for its own clarity; setting
            # it here is what makes the geometry, the optics and the radar agree
            # on which instant this frame is.
            fr_cfg["_ecm_t"] = float(t)
            w = scene.build_world(fr_cfg, t=0.0, jitter=jit)
            for o in w.objs:
                # scenery objects carry no dataset class (they are geometry only)
                o["cls_id"] = cls_id.get(o["cls"], 0) if o.get("label", True) else 0

        # Radar and lidar are body-fixed to the same gimbal as the camera, so all
        # four modalities share one line of sight. Two rules fought here:
        #   * the scenes author radar/lidar pointing at the world origin, which in
        #     a seeker geometry is nowhere near the target -- an aircraft 9 deg off
        #     that boresight falls outside a 1.2 deg beam and a 4 deg lidar raster
        #     and both returned pure noise at ANY transmit power (measured: 0
        #     detections from 10 W to 100 kW);
        #   * pointing the radar at the target while the camera keeps looking at
        #     the authored aim point broke co-boresight instead: on a nadir plate
        #     the same aircraft sat 0.68 deg apart in elevation between optical and
        #     radar (about 220 px in a 2.2 deg frame), so the two modalities could
        #     not be fused at all.
        # The gimbal answer is: follow the camera. Re-aim only when the primary
        # target would otherwise fall outside THIS sensor's field of regard, which
        # is exactly the case the old rule was written for.
        if ("radar" in sensors or "lidar" in sensors) and views:
            vp0 = _view_pose(s, views[0], cfg)
            _cam_pos = np.asarray(vp0["pos"], float)
            _aim = [float(x) for x in vp0["look_at"]]
            _tgt = None
            for _o in getattr(w, "objs", None) or []:
                if _o.get("label", True):
                    try:
                        _tgt = np.asarray(_o["corners"], float).mean(axis=0)
                    except (KeyError, TypeError, ValueError):
                        _tgt = None
                    break
            for _sen in ("radar", "lidar"):
                if _sen not in sensors:
                    continue
                blk = fr_cfg.setdefault("sensors", {}).setdefault(_sen, {})
                blk["pos"] = [float(x) for x in vp0["pos"]]
                # Same gimbal means the same roll: take the imaging view's up
                # vector. Left at a scene default of (0,0,1) in a look-down view,
                # sensor_basis' degenerate-up fallback could trigger on one
                # aperture and not the other, rolling the two 51 deg apart.
                _cam_up = ((fr_cfg.get("sensors") or {}).get("optical") or {}).get("up") \
                    or ((fr_cfg.get("sensors") or {}).get("ir_camera") or {}).get("up")
                if _cam_up:
                    blk["up"] = [float(v) for v in _cam_up]
                _use = list(_aim)
                if _tgt is not None:
                    try:
                        _p, _r, _u, _f = scene.sensor_basis(
                            blk["pos"], _aim, blk.get("up", (0.0, 0.0, 1.0)),
                            blk.get("roll_deg", 0.0))
                        _d = _tgt - _cam_pos
                        _n = float(np.linalg.norm(_d))
                        if _n > 1e-6:
                            _d = _d / _n
                            _az = float(np.degrees(np.arctan2(float(_d @ _r), float(_d @ _f))))
                            _el = float(np.degrees(np.arctan2(float(_d @ _u), float(_d @ _f))))
                            if _sen == "radar":
                                _lim_az = 0.5 * float(blk.get("az_span_deg") or 2.0)
                                _lim_el = 0.5 * float(blk.get("el_span_deg") or 1.6)
                            else:
                                _lim_az = 0.45 * float(blk.get("fov_deg") or 4.0)
                                _lim_el = 0.45 * float(blk.get("vfov_deg") or blk.get("fov_deg") or 4.0)
                            if abs(_az) > _lim_az or abs(_el) > _lim_el:
                                # Outside this sensor's own field of regard: the
                                # authored aim cannot see it at all, so point at
                                # the target and say so.
                                _use = [float(x) for x in _tgt]
                                print("  [提示] %s 视野未覆盖目标（方位 %.2f° / 俯仰 %.2f°，"
                                      "半视场 %.2f°×%.2f°），该帧改瞄目标。"
                                      % (_sen, _az, _el, _lim_az, _lim_el), file=sys.stderr)
                    except (KeyError, TypeError, ValueError):
                        _use = list(_aim)
                blk["look_at"] = _use
                blk["vel"] = list(s["sensor_velocity_mps"])

        if cfg.get("research_baseline"):
            from .observations import describe
            labels.write_json(os.path.join(out_dir, "observations", "%06d.json" % f),
                              describe(s, w))

        if "ir" in sensors:
            T = thermal.solve(w, fr_cfg, shadow=cfg.get("ir", {}).get("shadow", True))
            # Engine heat: the nozzle ring is a temperature override on the
            # airframe; the plume is extra geometry with a prescribed field.
            # Both are computed from the ORIGINAL world so the exhaust never
            # shadows the aircraft or answers a radar pulse.
            _hot_i, _hot_t = engine.nozzle_faces(w, fr_cfg)
            if len(_hot_i):
                T = np.array(T, float)
                T[_hot_i] = _hot_t
            _ir_world, _plume_t = engine.attach(w, fr_cfg)
            # T stays sized to the original world: the label block indexes it with
            # w.face_obj, and the plume faces are not objects.
            T_render = (np.concatenate([np.asarray(T, float), _plume_t])
                        if len(_plume_t) else np.asarray(T, float))
            for bnm in ir_bands:
                bt = "" if bnm is None else "_" + bnm
                fc = fr_cfg
                if bnm is not None:
                    fc = json.loads(json.dumps(fr_cfg))
                    fc.setdefault("ir", {})["band"] = bnm
                for v in views:
                    vp = _view_pose(s, v, cfg)
                    out = ir.render(_ir_world, fc, T_render, cam=vp)
                    H, W = out["counts_u16"].shape
                    name = "%06d.png" % f
                    vt = v["tag"]
                    if cfg.get("research_baseline"):
                        raw_dir = os.path.join(out_dir, "raw", "ir"+bt+vt)
                        os.makedirs(raw_dir, exist_ok=True)
                        np.savez_compressed(os.path.join(raw_dir, "%06d.npz" % f),
                                            depth_z_m=out["depth"].astype(np.float32),
                                            instance_id=out["obj_id"].astype(np.int32),
                                            radiance=out["radiance"].astype(np.float32),
                                            **{"coverage_%d" % oid: coverage.astype(np.float32)
                                               for oid, coverage in (out.get("instance_coverage") or {}).items()})
                    # Built only from this run's deterministic data: the view
                    # pose, the declared range and the rendered shape.
                    _ir_meta = dict(
                        modality="ir",
                        fov_deg=round(float(vp["fov_deg"]), 4),
                        radiometric_calibration=(
                            "16-bit counts: linear encoding of band radiance; "
                            "8-bit: display preview only, not calibrated radiance"))
                    _gsd = _gsd_m_per_px(s.get("range_m"), vp["fov_deg"], W)
                    if _gsd is not None:
                        _ir_meta["gsd_m_per_px"] = _gsd
                    png.write_gray16(os.path.join(out_dir, dirs["ir" + bt + vt], name),
                                     out["counts_u16"], meta=_ir_meta)
                    png.write_gray8(os.path.join(out_dir, dirs["ir_preview" + bt + vt], name),
                                    out["preview"], meta=_ir_meta)
                    boxes = _boxes_for(w, out["obj_id"], vp, W, H)
                    img, anns = labels.to_coco(boxes, w, fid, name, W, H)
                    coco_irs[(bnm, vt)]["images"].append(img)
                    coco_irs[(bnm, vt)]["annotations"].extend(anns)
                    labels.write_json(os.path.join(out_dir, dirs["label_ir" + bt + vt],
                                                   "%06d.json" % f),
                                      dict(frame=f, sensor="ir", band=bnm or "lwir",
                                           view=v["name"], fov_deg=vp["fov_deg"],
                                           clip=s.get("clip"),
                                           range_m=s.get("range_m"),
                                           target_px=(s.get("view_px") or {}).get(v["name"],
                                                                                  s.get("target_px")),
                                           weather=dict(fr_cfg.get("weather", {})),
                                           sampling=out.get("sampling"),
                                           image=name, width=W, height=H,
                                           boxes={o["name"]: boxes[o["id"]] for o in w.objs
                                                  if o["id"] in boxes},
                                           objects=[dict(id=o["id"], name=o["name"], cls=o["cls"],
                                                         t_mean_c=float(T[w.face_obj == o["id"]].mean() - 273.15))
                                                    for o in w.objs]))

        if "radar" in sensors:
            # Scene-level ECM needs the frame clock: a chaff cloud blooms and a
            # gate pull-off walks away over time, so "which instant is this"
            # has to reach the radar. ecm.frame_time reads _ecm_t first; _frame_t
            # is kept because the SAR branch below zeroes it for its own anchor.
            fr_cfg["_ecm_t"] = float(t)
            fr_cfg["_frame_t"] = float(t)
            fr_cfg["_frame_index"] = int(f)
            out = radar.simulate(w, fr_cfg)
            # The forward model already returns what the jammers did to this
            # frame.  Dropping it here made a dataset that was generated WITH
            # scene-level ECM indistinguishable from one generated without, so
            # the numbers a consumer needs to know which is which are kept.
            if out.get("ecm"):
                ecm_frames.append(dict(frame=int(f), t=float(t), **out["ecm"]))
            # Every radar frame also ships as a tensor, not only as a figure: a
            # fusion model reads the range-Doppler power map with its axes, and
            # the frame index is the sibling key to optical/IR of the same frame.
            # float32 (n_range, n_doppler), the same orientation the figure draws.
            raw_dir = os.path.join(out_dir, "raw", "radar")
            os.makedirs(raw_dir, exist_ok=True)
            np.save(os.path.join(raw_dir, "%06d.npy" % f),
                    np.asarray(out["rd_map"], np.float32))
            if cfg.get("research_baseline"):
                np.savez_compressed(os.path.join(raw_dir, "%06d.npz" % f),
                                    power=out["rd_map"].T,
                                    range_m=out["range_axis"],
                                    closing_speed_mps=out["doppler_axis"])
            name = "%06d.png" % f
            r_ax = np.asarray(out["range_axis"], float)
            d_ax = np.asarray(out["doppler_axis"], float)
            _rc = out.get("radar_cfg") or {}
            # Ground truth for the figure's own target table: every labelled
            # target with its true range and closing speed, plus the peak the
            # detector measured for it. Two rows at the same range mean two
            # airframes inside one resolution cell -- the honest reading of a
            # single bright cell, and the number a reader needs to see why.
            rows = radar_target_rows(w, _rc, out["detections"])
            _meta = dict(band=str(_rc.get("band") or "—"),
                         fc_ghz=float(_rc.get("fc_ghz") or 0.0) or None,
                         range_res_m=float(r_ax[1] - r_ax[0]) if r_ax.size > 1 else None,
                         vel_res_mps=float(d_ax[1] - d_ax[0]) if d_ax.size > 1 else None,
                         frame=int(f), scene=str(cfg.get("name") or ""))
            # Range/Doppler figure. The map is one receiver channel's power, so
            # the window is anchored on the noise median and the speckle spread
            # fills it; the axes and the target table are drawn into the PNG so
            # the picture carries its own numbers.
            # 40 dB scope window: the noise floor is exponential (10-90 spread
            # 13.4 dB), so 30 dB clipped the bottom of the speckle and made the
            # field read as a flat blue wash; 40 dB keeps the grain while the
            # peak still saturates, which is what a real STC curve gives.
            rd_fig = radar_scope.render_rd(out["rd_map"], r_ax, d_ax,
                                           targets=rows, meta=_meta, dyn_db=40.0)
            # The figure is a display product; the physical grid is the RD map
            # behind it: (n_samples//2) range bins x n_chirps doppler bins.
            _rad_cal = ("dB colour scale; display figure (axes/target table), "
                        "not a calibrated radiance product")
            _ns, _nc = int(_rc.get("n_samples") or 0), int(_rc.get("n_chirps") or 0)
            _rd_grid = "%dx%d" % (_ns // 2 if _ns else int(r_ax.size),
                                  _nc if _nc else int(d_ax.size))
            _rd_meta = dict(modality="radar_rd", grid=_rd_grid,
                            radiometric_calibration=_rad_cal)
            png.write_rgb8(os.path.join(out_dir, dirs["radar"], name),
                           rd_fig["rgb"], meta=_rd_meta)
            rd_clean = radar_scope.render_rd(out["rd_map"], r_ax, d_ax,
                                             targets=rows, meta=_meta, dyn_db=40.0,
                                             annotate=False)
            png.write_rgb8(os.path.join(out_dir, dirs["radar_clean"], name),
                           rd_clean["rgb"], meta=_rd_meta)
            rd_h, rd_w = int(rd_fig["rgb"].shape[0]), int(rd_fig["rgb"].shape[1])
            # A detection is one range/Doppler cell, i.e. a fraction of a pixel
            # in the figure, so its box is padded to the echo's few-cell extent
            # with a 5 px floor: a consumer has to be able to see the label.
            _px0, _py0, _px1, _py1 = (float(v) for v in rd_fig["plot"])
            _cw = max(1.0, (_px1 - _px0) / max(r_ax.size, 1))
            _ch = max(1.0, (_py1 - _py0) / max(d_ax.size, 1))
            _padx, _pady = max(2.5, 1.5 * _cw), max(2.5, 1.5 * _ch)
            # labelled objects only: a fence is scenery, and putting it in the
            # label file makes the YOLO export either skip it (class unknown) or,
            # worse, teach the detector that fences are targets
            name_of = {o["id"]: o["name"] for o in w.objs if o.get("label", True)}
            # A detection with obj == -1 is not world geometry. Two very
            # different things land there, and they must not be conflated:
            #   * a real ECM echo, which ecm.radar_apply marks with ecm_id;
            #   * clutter / a false alarm, which matches nothing in the scene.
            # Naming every obj == -1 "ecm#NN" put 2741 phantom jamming boxes into
            # scenes with no jamming at all (97.9% of all radar boxes across the
            # 32-scene regression), so a clutter frame looked like a deception set.
            boxes_rd = {}
            ecm_hits = {}
            for det in out["detections"]:
                oid = int(det["obj"])
                ecm_id = det.get("ecm_id")
                if oid < 0:
                    if not ecm_id:
                        # clutter: reported under unassociated_detections, never
                        # as a box, or it teaches a detector that grass is a target
                        continue
                    hit = ecm_hits.get(ecm_id, 0)
                    ecm_hits[ecm_id] = hit + 1
                    nm = "ecm#%s:%02d" % (ecm_id, hit)
                else:
                    nm = name_of.get(oid)
                if nm is None:
                    continue
                px = float(np.clip(float(rd_fig["range_px"](float(det["range"]))),
                                   _px0, _px1))
                py = float(np.clip(float(rd_fig["doppler_px"](float(det["v_r"]))),
                                   _py0, _py1))
                b = dict(x0=max(0.0, px - _padx), y0=max(0.0, py - _pady),
                         x1=min(float(rd_w), px + _padx), y1=min(float(rd_h), py + _pady),
                         pixels=1, visible=True,
                         source=("ecm" if oid < 0 else "detection"))
                prev = boxes_rd.get(nm)
                if prev:                      # several cells can hit one object
                    b["x0"] = min(prev["x0"], b["x0"]); b["y0"] = min(prev["y0"], b["y0"])
                    b["x1"] = max(prev["x1"], b["x1"]); b["y1"] = max(prev["y1"], b["y1"])
                    b["pixels"] = prev["pixels"] + 1
                boxes_rd[nm] = b
            # Range/azimuth map: one row per azimuth look, one column per range
            # bin. The beam raster is (n_el x n_az), so elevation is collapsed by
            # a maximum instead of a sum -- one beam carries the target, and
            # averaging over elevation would bury a single-beam echo in clutter.
            ra = np.asarray(out["range_angle"], float)      # (n_beams, n_s)
            # The per-beam spectrum is not sliced to the positive-frequency band
            # the way the RD map is; keep only the bins the range axis describes
            # or the RA picture would show a mirrored negative-frequency half.
            ra = ra[:, :r_ax.size]
            _az = np.asarray(out["beams"], float)[:, 0]
            _uniq_az, _inv = np.unique(np.round(_az, 6), return_inverse=True)
            ra_az = np.zeros((_uniq_az.size, ra.shape[1]))
            np.maximum.at(ra_az, _inv, ra)
            ra_fig = radar_scope.render_ra(ra_az, r_ax, _uniq_az, targets=rows,
                                           meta=dict(_meta, ra_layout="beams_range"))
            _ra_meta = dict(modality="radar_ra",
                            grid="%dx%d" % (int(ra_az.shape[1]), int(ra_az.shape[0])),
                            radiometric_calibration=_rad_cal)
            png.write_rgb8(os.path.join(out_dir, dirs["radar_ra"], name),
                           ra_fig["rgb"], meta=_ra_meta)
            ra_clean = radar_scope.render_ra(ra_az, r_ax, _uniq_az, targets=rows,
                                             meta=dict(_meta, ra_layout="beams_range"),
                                             annotate=False)
            png.write_rgb8(os.path.join(out_dir, dirs["radar_ra_clean"], name),
                           ra_clean["rgb"], meta=_ra_meta)
            rlab = labels.radar_labels(w, out["detections"], fid)
            # Where the brightest echo sits in that picture, in figure pixels and
            # in physical units: the map is mostly noise floor and one 1-cell
            # return is otherwise impossible to find by eye.
            _ay, _ax = np.unravel_index(int(np.argmax(ra_az)), ra_az.shape)
            rlab["ra_peak"] = dict(px=float(ra_fig["range_px"](float(r_ax[_ax]))),
                                   py=float(ra_fig["doppler_px"](float(_uniq_az[_ay]))),
                                   db=round(float(radar.to_db(ra_az.max())), 1),
                                   w=int(ra_fig["rgb"].shape[1]),
                                   h=int(ra_fig["rgb"].shape[0]),
                                   range_m=round(float(r_ax[_ax]), 1),
                                   az_deg=round(float(_uniq_az[_ay]), 3))
            if out.get("targets_beyond_range"):
                # Silence is the worst outcome here: the dataset looks fine, the
                # labels are empty, and nothing on screen says why.
                _far = out["targets_beyond_range"]
                print("  [警告] 雷达距离窗 %.0f m 小于目标斜距 %s m，目标不会被检测到。"
                      "请调大 sensors.radar.n_samples 或减小带宽。"
                      % (float(out.get("range_window_m") or 0),
                         "/".join("%.0f" % r for r in _far[:4])), file=sys.stderr)
            rlab["boxes_rd"] = boxes_rd
            # What a consumer needs to map a detection back onto the picture
            # without re-deriving the renderer's layout: the plot rectangle and
            # the pixel position of the first/last axis value.
            rlab["figure"] = dict(
                rd=dict(plot=[float(v) for v in rd_fig["plot"]],
                        db_window=[float(v) for v in rd_fig["db_window"]],
                        range_m=[float(r_ax[0]), float(r_ax[-1])],
                        range_px=[float(rd_fig["range_px"](float(r_ax[0]))),
                                  float(rd_fig["range_px"](float(r_ax[-1])))],
                        v_r=[float(d_ax[0]), float(d_ax[-1])],
                        v_px=[float(rd_fig["doppler_px"](float(d_ax[0]))),
                              float(rd_fig["doppler_px"](float(d_ax[-1])))]),
                ra=dict(plot=[float(v) for v in ra_fig["plot"]],
                        db_window=[float(v) for v in ra_fig["db_window"]],
                        az_deg=[float(_uniq_az[0]), float(_uniq_az[-1])],
                        az_px=[float(ra_fig["doppler_px"](float(_uniq_az[0]))),
                               float(ra_fig["doppler_px"](float(_uniq_az[-1])))]))
            # The clean copy has no table, so its map keeps the full width and
            # its own mapping. Boxes stay in the annotated figure's pixels; a
            # consumer that trains on the clean map converts with this rect.
            rlab["figure"]["rd"]["clean"] = dict(
                path="images/radar_clean/%06d.png" % f,
                plot=[float(v) for v in rd_clean["plot"]],
                range_px=[float(rd_clean["range_px"](float(r_ax[0]))),
                          float(rd_clean["range_px"](float(r_ax[-1])))],
                v_px=[float(rd_clean["doppler_px"](float(d_ax[0]))),
                      float(rd_clean["doppler_px"](float(d_ax[-1])))])
            rlab["figure"]["ra"]["clean"] = dict(
                path="images/radar_ra_clean/%06d.png" % f,
                plot=[float(v) for v in ra_clean["plot"]],
                az_px=[float(ra_clean["doppler_px"](float(_uniq_az[0]))),
                       float(ra_clean["doppler_px"](float(_uniq_az[-1])))])
            # Same world the radar saw, and the radar own position: a target
            # beyond the window returns nothing and the frame exits 0, so a
            # silent zero here would look exactly like a clean miss.
            _rwin = float(out.get("range_window_m") or 0.0)
            _labeled = [o for o in (getattr(w, "objs", None) or [])
                        if o.get("label", True)]
            rlab["range_window_m"] = _rwin or None
            # Only obj >= 0 counts as a real target detection: clutter peaks hide
            # a completely unseen target behind "there are detections".
            if _labeled and not any(int(d["obj"]) >= 0 for d in out["detections"]):
                print("  [警告] 该帧雷达零检测，但场景有 %d 个标注目标；"
                      "距离窗 %.0f m，请检查 sensors.radar.n_samples 是否足够。"
                      % (len(_labeled), _rwin), file=sys.stderr)
            rlab["processing"] = out.get("processing")
            rlab["targets"] = rows
            rlab["tensor"] = dict(path="raw/radar/%06d.npy" % f, dtype="float32",
                                  shape=[int(np.shape(out["rd_map"])[0]),
                                         int(np.shape(out["rd_map"])[1])],
                                  axis_order=["range", "doppler"],
                                  unit="W (received power, one-way normalised)")
            # The pieces a fused loader needs to rebuild this frame without the
            # scene file: the instant, where the antenna was, and the waveform
            # the numbers were measured with.
            rlab["t"] = round(float(t), 4)
            rlab["sensor_pose"] = dict(pos=[float(v) for v in (_rc.get("pos") or [])],
                                       look_at=[float(v) for v in (_rc.get("look_at") or [])],
                                       up=[float(v) for v in (_rc.get("up") or [])])
            rlab["waveform"] = dict(
                fc_ghz=_rc.get("fc_ghz"), bandwidth_mhz=_rc.get("bandwidth_mhz"),
                n_samples=_rc.get("n_samples"), n_chirps=_rc.get("n_chirps"),
                chirp_us=_rc.get("chirp_us"), pri_us=_rc.get("pri_us"),
                tx_power_dbm=_rc.get("tx_power_dbm"),
                noise_figure_db=_rc.get("noise_figure_db"),
                range_res_m=float(r_ax[1] - r_ax[0]) if r_ax.size > 1 else None,
                vel_res_mps=float(d_ax[1] - d_ax[0]) if d_ax.size > 1 else None,
                vel_nyquist_mps=float(np.max(np.abs(d_ax))),
                range_window_m=out.get("range_window_m"))
            rlab["unassociated_detections"] = [d for d in out["detections"] if d["obj"] < 0]
            rlab["axes"] = dict(range_m=out["range_axis"].tolist(),
                                 closing_speed_mps=out["doppler_axis"].tolist(),
                                 velocity_convention="positive closing; single-PRF aliased")
            # How the RA picture was built from the beam raster. Without this a
            # consumer cannot tell a 40x20 scanned raster from a 40-beam fan and
            # would read the y axis as if it were degrees of azimuth on a fan.
            rlab["beam_raster"] = dict(
                n_az=int(_rc.get("n_az") or 0), n_el=int(_rc.get("n_el") or 0),
                az_span_deg=_rc.get("az_span_deg"), el_span_deg=_rc.get("el_span_deg"),
                beamwidth_az_deg=_rc.get("beamwidth_az_deg"),
                beamwidth_el_deg=_rc.get("beamwidth_el_deg"),
                ra_layout="range x azimuth",
                elevation_reduction="max over n_el (one beam carries the echo)")
            rlab["width"], rlab["height"] = int(rd_w), int(rd_h)
            labels.write_json(os.path.join(out_dir, dirs["label_radar"], "%06d.json" % f), rlab)
            radar_frames.append(dict(
                frame=int(f), image=name,
                width=int(rlab.get("width") or 0), height=int(rlab.get("height") or 0),
                n=int(rlab.get("num_points", 0)),
                clutter=int(rlab.get("clutter_count", 0)),
                boxes=rlab.get("boxes_3d") or [],
                targets=rlab.get("targets") or [],
                boxes_rd=boxes_rd,
                tensor=rlab.get("tensor"),
                snr_max=round(float(max([d["snr_db"] for d in out["detections"]], default=0.0)), 2)))

        if "lidar" in sensors:
            out = lidar.simulate(w, fr_cfg)
            name = "%06d.png" % f
            H, W = out["range_m"].shape
            # range image: nearest return is brightest, empty sky is black
            rg = out["range_m"].copy()
            far = rg <= 0.0
            rg[far] = out["max_range_m"]
            vis = np.clip(1.0 - rg / out["max_range_m"], 0.0, 1.0) ** 0.55
            vis[far] = 0.0
            # Both products share the (n_el, n_az) beam raster: H = n_el, W = n_az.
            _lid_meta = dict(modality="lidar_range",
                             grid="%dx%d" % (int(H), int(W)),
                             range_m=round(float(out["max_range_m"]), 4))
            png.write_gray8(os.path.join(out_dir, dirs["lidar_range"], name),
                            np.round(vis * 255).astype(np.uint8), meta=_lid_meta)
            png.write_gray8(os.path.join(out_dir, dirs["lidar_intensity"], name),
                            np.round(np.clip(out["intensity"], 0, 1) * 255).astype(np.uint8),
                            meta=dict(_lid_meta, modality="lidar_intensity"))
            lidar.write_ply(os.path.join(out_dir, dirs["lidar_points"], "%06d.ply" % f), out["cloud"], out["origin"])
            boxes = _boxes_for(w, out["obj_id"], None, W, H)
            img, anns = labels.to_coco(boxes, w, fid, name, W, H)
            coco_lidar["images"].append(img)
            coco_lidar["annotations"].extend(anns)
            labels.write_json(os.path.join(out_dir, dirs["label_lidar"], "%06d.json" % f),
                              dict(frame=f, sensor="lidar", image=name, width=W, height=H,
                                   clip=s.get("clip"), range_m=s.get("range_m"),
                                   weather=dict(fr_cfg.get("weather", {})),
                                   hit_ratio=round(out["hit_ratio"], 4),
                                   n_points=out["n_points"],
                                   max_range_m=out["max_range_m"],
                                   boxes=list(boxes.values()),
                                   objects=[dict(id=o["id"], name=o["name"], cls=o["cls"],
                                                 points=int((out["obj_id"] == o["id"]).sum()))
                                            for o in w.objs if o.get("label", True)]))

        if have_sar:
            from . import sar as sar_mod
            # Per-frame snapshot anchored on the target. A 6-second clip at
            # 180 m/s carries the airframe ~1 km along track, which slides the
            # target out of an azimuth window a few hundred metres wide after
            # about 40 frames (the other scenes' SAR checks measure exactly
            # this). Anchoring each frame on the target and zeroing the
            # along-track time keeps every SAR image centred on the object while
            # weather and geometry still vary frame to frame.
            sar_cfg = fr_cfg.setdefault("sensors", {}).setdefault("sar", {})
            tgt = next((o for o in cfg.get("objects", []) if o.get("label", True)), None)
            if tgt is not None and tgt.get("name") in (s.get("poses") or {}):
                tp = s["poses"][tgt["name"]]["pos"]
                anc = list(sar_cfg.get("anchor") or [tp[0], tp[1], tp[2]])
                sar_cfg["anchor"] = [float(tp[0]), float(tp[1]), float(anc[2])]
                fr_cfg["_frame_t"] = 0.0
            sout = sar_mod.simulate(w, fr_cfg)
            name = "%06d.png" % f
            img8 = sout["image_u8"]
            H, W = img8.shape
            _sc = sout.get("cfg") or {}
            # SAR is a calibrated gamma0 product; the 8-bit PNG is a display
            # stretch of it. grid = n_cross x n_samples (the focused image).
            _sar_meta = dict(
                modality="sar",
                grid="%dx%d" % (int(_sc.get("n_cross") or H),
                                int(_sc.get("n_samples") or W)),
                radiometric_calibration=("calibrated gamma0 product; "
                                         "8-bit display stretch, not raw DN"))
            _sar_swath = sout.get("swath") or {}
            if _sar_swath.get("ground_sampling_m"):
                _sar_meta["gsd_m_per_px"] = round(
                    float(_sar_swath["ground_sampling_m"]), 6)
            png.write_gray8(os.path.join(out_dir, dirs["sar"], name), img8,
                            meta=_sar_meta)
            boxes = {}
            if sout.get("obj_map") is not None:
                boxes = labels.boxes_from_mask(sout["obj_map"], w, min_px=2)
            # A ship or a rover can be a handful of resolution cells; when the
            # mask is too thin to box, fall back to the geometry the kernel
            # already knows, so a hard-but-real target never loses its label.
            geo = sout.get("target_boxes") or {}
            for o in w.objs:
                if o.get("label", True) and o["id"] not in boxes and o["name"] in geo:
                    b = geo[o["name"]]
                    boxes[o["id"]] = dict(x0=float(b[0]), y0=float(b[1]),
                                          x1=float(b[2]), y1=float(b[3]),
                                          pixels=0, source="geometry")
            img, anns = labels.to_coco(boxes, w, fid, name, W, H)
            coco_sar["images"].append(img)
            coco_sar["annotations"].extend(anns)
            st = dict(sout.get("stats") or {})
            labels.write_json(os.path.join(out_dir, dirs["label_sar"], "%06d.json" % f),
                              dict(frame=f, sensor="sar", image=name, width=W, height=H,
                                   clip=s.get("clip"), range_m=s.get("range_m"),
                                   resolution=sout.get("resolution"),
                                   swath=sout.get("swath"),
                                   platform=sout.get("platform"),
                                   stats=st,
                                   boxes={o["name"]: boxes[o["id"]] for o in w.objs
                                          if o["id"] in boxes},
                                   objects=[dict(id=o["id"], name=o["name"], cls=o["cls"])
                                            for o in w.objs if o.get("label", True)]))
            sar_frames.append(dict(
                frame=f, boxes=len(boxes),
                peak_db=round(float(st.get("peak_db") or 0.0), 2),
                snr_db=round(float(st.get("snr_db") or 0.0), 2),
                target_contrast_db=round(float(st.get("target_contrast_db") or 0.0), 2),
                n_scatterers=int(st.get("n_scatterers") or 0)))

        if "optical" in sensors:
            for vidx, v in enumerate(views):
                vt = v["tag"]
                lp = os.path.join(out_dir, dirs["label_optical" + vt], "%06d.json" % f)
                if not os.path.exists(lp):
                    continue
                with open(lp, "r", encoding="utf-8") as fh:
                    lb = json.load(fh)
                # Replace the renderer's silhouette boxes with the projected
                # extent every other modality uses, so a fused pair carries one
                # label, not two conventions (see _boxes_for).
                vp = _view_pose(s, v, cfg)
                prev = lb.get("boxes") or {}
                uid = _boxes_for(w, None, vp, int(lb["width"]), int(lb["height"]),
                                 seen=[name for name, box in prev.items() if box.get("visible", True)])
                uid = _preserve_renderer_visibility(w, uid, prev)
                # _boxes_for keys by object id; the label file keys by name, which
                # is what every consumer (COCO, YOLO, the site) needs to know the
                # class. Losing it here silently turned into "images with no boxes".
                id2name = {o["id"]: o["name"] for o in w.objs}
                lb["boxes"] = {id2name.get(k, str(k)): v for k, v in uid.items()}
                lb["box_source"] = "corners (unified)"
                labels.write_json(lp, lb)
                anns = []
                for aid, (nm, b) in enumerate(lb["boxes"].items()):
                    if not b.get("visible", True):
                        continue  # geometry truth remains in JSON; not a detector positive
                    spec = next((o for o in cfg["objects"] if o.get("name") == nm), None)
                    if spec is None or not spec.get("label", True):
                        continue
                    # image ids stay global per view: two views of one instant
                    # are two images, not one image with double the boxes
                    # Same allocator every other modality uses. The arithmetic
                    # is identical to the fid*10000 + vidx*1000 + aid it replaces
                    # (annotation_id multiplies by stride*10 and adds view*stride),
                    # but it now raises instead of silently aliasing ids when one
                    # frame carries more than ANNOTATION_ID_STRIDE boxes.
                    anns.append(dict(id=labels.annotation_id(fid, aid, vidx), image_id=fid,
                                     category_id=cls_id[spec.get("cls", "target")],
                                     bbox=[b["x0"], b["y0"], round(b["x1"] - b["x0"], 2),
                                           round(b["y1"] - b["y0"], 2)],
                                     area=round((b["x1"] - b["x0"]) * (b["y1"] - b["y0"]), 2),
                                     iscrowd=0, object_name=nm, visible=b.get("visible", True)))
                coco_opt[vt]["images"].append(dict(id=fid, file_name="%06d.png" % f,
                                                   width=lb["width"], height=lb["height"]))
                coco_opt[vt]["annotations"].extend(anns)

        if not quiet:
            print("frame %d/%d done" % (f + 1, n_frames), flush=True)

    # COCO is written per chunk when the run is split (part is not None) and
    # concatenated afterwards, so parallel workers never fight over one file.
    def anno_name(mod):
        if part is None:
            return "annotations_%s_coco.json" % mod
        return "annotations_%s_coco.part%03d.json" % (mod, int(part))

    if "ir" in sensors:
        for bnm in ir_bands:
            bt = "" if bnm is None else "_" + bnm
            for v in views:
                labels.write_json(os.path.join(out_dir, anno_name("ir" + bt + v["tag"])),
                                  coco_irs[(bnm, v["tag"])])
    if "optical" in sensors:
        for v in views:
            labels.write_json(os.path.join(out_dir, anno_name("optical" + v["tag"])),
                              coco_opt[v["tag"]])
    # Radar is the one modality whose pictures are figures, not pixels: the
    # RD map is a tensor with its own axes, and a fused loader needs the frame
    # index plus the truth range/closing speed that the figure prints. One file
    # ties the tensor, the figure box and the truth row together for every
    # frame, using the same "frame" key as the optical/IR label directories.
    if "radar" in sensors and radar_frames:
        # COCO-shaped on purpose: the README, the site and any fused loader
        # already read annotations_*_coco.json, so a bespoke schema would show
        # up as "radar 0 images / 0 boxes". Each image record carries the tensor
        # path and the truth table on top of the COCO fields.
        _rimg, _rann = [], []
        for _r in radar_frames:
            _rows = _r.get("targets") or []
            _rimg.append(dict(id=int(_r["frame"]), file_name=_r.get("image"),
                              width=_r.get("width"), height=_r.get("height"),
                              tensor=_r.get("tensor"), targets=_rows,
                              n_points=_r.get("n"), snr_max=_r.get("snr_max")))
            _cls_of = {row["name"]: row.get("cls") for row in _rows}
            for _nm, _b in (_r.get("boxes_rd") or {}).items():
                _w = float(_b["x1"] - _b["x0"])
                _h = float(_b["y1"] - _b["y0"])
                _rann.append(dict(id=len(_rann) + 1, image_id=int(_r["frame"]),
                                  category_id=int(cls_id.get(_cls_of.get(_nm), 0)),
                                  bbox=[round(float(_b["x0"]), 2), round(float(_b["y0"]), 2),
                                        round(_w, 2), round(_h, 2)],
                                  area=round(_w * _h, 2), iscrowd=0,
                                  object_name=_nm, source=_b.get("source", "detection")))
        labels.write_json(os.path.join(out_dir, "annotations_radar.json"),
                          dict(info=dict(description="simkit radar range-Doppler frames "
                                                     "(per-frame truth + figure boxes)",
                                         version="0.1",
                                         coordinate_system="figure pixels, top-left origin",
                                         tensor_layout="float32 (range, doppler)"),
                               categories=cats, images=_rimg, annotations=_rann))
    if "lidar" in sensors:
        labels.write_json(os.path.join(out_dir, anno_name("lidar")), coco_lidar)
    if have_sar:
        labels.write_json(os.path.join(out_dir, anno_name("sar")), coco_sar)

    # Per-frame index: range, target size per view, clip id, weather. The site
    # plots range/size curves from this instead of re-deriving them from every
    # modality's label files, and reviewers get one file that says what the run
    # actually contains frame by frame.
    idx_rows = [dict(frame=int(s["frame"]), t=round(float(s["t"]), 4), clip=s.get("clip"),
                     sequence_start=s.get("sequence_start", False),
                     sequence_timestamp_s=s.get("sequence_timestamp_s"),
                     range_m=s.get("range_m"), target_px=s.get("target_px"),
                     view_px=s.get("view_px"),
                     sensor_velocity_mps=s.get("sensor_velocity_mps"),
                     view_roll={k: round(float(v.get("roll_deg", 0.0) or 0.0), 3)
                                for k, v in (s.get("views") or {}).items()},
                     weather=dict(s.get("weather") or {}))
                for s in states]
    labels.write_json(os.path.join(
        out_dir, "frames.json" if part is None else "frames.part%03d.json" % int(part)),
        idx_rows)
    if part is not None:
        # A chunk does not own the run: no splits, no stats, no meta. Whoever
        # merged the chunks writes those, once, from the whole dataset.
        labels.write_json(os.path.join(out_dir, "radar_frames.part%03d.json" % int(part)),
                          radar_frames)
        if sar_frames:
            labels.write_json(os.path.join(out_dir, "sar_frames.part%03d.json" % int(part)),
                              sar_frames)
        return dict(chunk=int(part), frames=n_frames, offset=int(frame_offset),
                    radar_frames=radar_frames, sar_frames=sar_frames)
    # train/val split + quick QC stats
    idx = list(range(n_frames))
    n_val = max(1, int(round(n_frames * 0.15))) if n_frames >= 8 else 0
    val = sorted(idx[-n_val:]) if n_val else []
    train = [i for i in idx if i not in set(val)]
    labels.write_json(os.path.join(out_dir, "splits.json"),
                      dict(train=["%06d" % i for i in train], val=["%06d" % i for i in val]))
    # Stats are rebuilt from what is actually on disk, not from this invocation.
    # Regenerating one sensor into an existing run (a common way to iterate on,
    # say, the radar) must not blank the optical/IR/LiDAR counts.
    stats_path = os.path.join(out_dir, "stats.json")
    stats = {}
    if os.path.exists(stats_path):
        try:
            stats = _read_json(stats_path)
        except (OSError, ValueError):
            stats = {}
    stats["frames"] = n_frames

    def count_coco(path):
        if not os.path.exists(path):
            return 0, 0
        try:
            d = _read_json(path)
        except (OSError, ValueError):
            return 0, 0
        return len(d.get("images", [])), len(d.get("annotations", []))

    n_i, n_a = count_coco(os.path.join(out_dir, "annotations_optical_coco.json"))
    stats["optical_images"], stats["optical_boxes"] = n_i, n_a
    # Every extra field of view is its own image set, reported separately so the
    # site can show "narrow: 120 frames / wide: 120 frames" instead of one
    # ambiguous total.
    stats["optical_views"] = {}
    for v in views:
        if not v["tag"]:
            continue
        fn = "annotations_optical%s_coco.json" % v["tag"]
        n_i, n_a = count_coco(os.path.join(out_dir, fn))
        stats["optical_views"][v["name"]] = dict(images=n_i, boxes=n_a, coco=fn,
                                                 fov_deg=v.get("fov_deg"))
    n_i, n_a = count_coco(os.path.join(out_dir, "annotations_lidar_coco.json"))
    stats["lidar_images"], stats["lidar_boxes"] = n_i, n_a
    n_i, n_a = count_coco(os.path.join(out_dir, "annotations_sar_coco.json"))
    stats["sar_images"], stats["sar_boxes"] = n_i, n_a
    # IR band keys come from the config; counts come from disk, so a radar-only
    # rerun still reports the IR frames an earlier run left in place.
    ir_stats = stats.get("ir") or {}
    for bnm in ir_bands:
        bt = "" if bnm is None else "_" + bnm
        for v in views:
            fn = "annotations_ir%s%s_coco.json" % (bt, v["tag"])
            n_i, n_a = count_coco(os.path.join(out_dir, fn))
            key = (bnm or "lwir") if not v["tag"] else "%s%s" % (bnm or "lwir", v["tag"])
            ir_stats[key] = dict(images=n_i, boxes=n_a, coco=fn, fov_deg=v.get("fov_deg"))
    for bnm in ("mwir", "lwir"):
        if bnm not in ir_stats:
            fn = "annotations_ir_%s_coco.json" % bnm
            n_i, n_a = count_coco(os.path.join(out_dir, fn))
            if n_i or n_a:
                ir_stats[bnm] = dict(images=n_i, boxes=n_a, coco=fn)
    stats["ir"] = ir_stats
    stats["ir_images"] = sum(v["images"] for v in ir_stats.values())
    stats["ir_boxes"] = sum(v["boxes"] for v in ir_stats.values())
    if "radar" in sensors and radar_frames:
        stats["radar_frames"] = len(radar_frames)
        stats["radar_points"] = int(sum(r["n"] for r in radar_frames))
        stats["radar_boxes"] = int(sum(len(r["boxes"]) for r in radar_frames))
        stats["radar_points_per_frame"] = [
            dict(frame=r["n"], clutter=r["clutter"], boxes=len(r["boxes"]),
                 snr_max=r["snr_max"]) for r in radar_frames]
    if have_sar and sar_frames:
        stats["sar_frames"] = len(sar_frames)
        labels.write_json(os.path.join(out_dir, "sar_frames.json"), sar_frames)
        stats["sar_per_frame"] = sar_frames

    # cross-modality class tally, recomputed from every COCO present
    stats["boxes_per_class"] = {}
    for fn in sorted(os.listdir(out_dir)):
        if not (fn.startswith("annotations_") and fn.endswith("_coco.json")):
            continue
        try:
            d = _read_json(os.path.join(out_dir, fn))
        except (OSError, ValueError):
            continue
        names = {c["id"]: c["name"] for c in d.get("categories", [])}
        for a in d.get("annotations", []):
            cn = names.get(a.get("category_id"), "?")
            stats["boxes_per_class"][cn] = stats["boxes_per_class"].get(cn, 0) + 1
    labels.write_json(stats_path, stats)

    # `sensors` is what THIS run rendered. A later partial pass (`-s optical`,
    # used to retouch exposure without redoing radar) then rewrites meta.json and
    # the dataset starts claiming it only has optics -- or, after a manual fix,
    # claiming a SAR channel on an air-to-air scene that cannot produce one.
    # The directories on disk are the only honest answer, so trust them and fall
    # back to the argument only when nothing was written.
    meta = dict(config=os.path.abspath(cfg_path), frames=n_frames,
                sensors=_sensors_on_disk(out_dir) or list(sensors),
                seed=base_seed,
                seconds=round(time.time() - t_start, 2) if t_start else 0.0,
                categories=cats, generated_by="simkit 0.2",
                ir_bands=[b or "lwir" for b in ir_bands],
                views=[dict(name=v["name"], tag=v["tag"], fov_deg=v.get("fov_deg"),
                            width=v.get("width"), height=v.get("height"))
                       for v in _views_of(states, cfg)],
                radar_band=(cfg.get("sensors", {}).get("radar", {}) or {}).get("band"),
                sar_band=(cfg.get("sensors", {}).get("sar", {}) or {}).get("band"),
                weather=[dict(s["weather"]) for s in states])
    if ecm_frames:
        meta["ecm"] = _ecm_summary(cfg, ecm_frames)
    if jpeg_report is not None:
        meta["jpeg"] = jpeg_report
    labels.write_json(os.path.join(out_dir, "meta.json"), meta)
    if disturbances["enabled"]:
        interference.export(out_dir, os.path.join(out_dir, "interfered"), disturbances)
        meta["interference"] = dict(variant="interfered", domain="image_codes")
        labels.write_json(os.path.join(out_dir, "meta.json"), meta)
    # ---- post-processing: composite preview, dataset README, YOLO export.
    # None of this may lose the dataset that just took an hour to render, so a
    # failure is recorded in meta["post"] instead of raised. Preview and README
    # default to on: a dataset you can only review by opening twenty folders is
    # one nobody reviews.
    out_opts = cfg.get("output") or {}
    post = {}
    want_prev = bool(out_opts.get("preview", True))
    want_doc = bool(out_opts.get("readme", True))
    if want_prev or want_doc:
        try:
            from . import sheet as s_sheet
            if want_prev:
                made = s_sheet.triptych(out_dir, modalities=_preview_modalities(sensors))
                post["preview_frames"] = len(made)
                post["contact_sheet"] = os.path.basename(s_sheet.contact_sheet(out_dir))
            if want_doc:
                post["readme"] = os.path.basename(s_sheet.write_readme(out_dir, meta))
        except ImportError:
            post["note"] = "simkit.sheet 不可用，跳过拼接图与说明文档"
        except Exception as exc:
            post["error"] = "%s: %s" % (type(exc).__name__, exc)
    if out_opts.get("yolo"):
        try:
            script = os.path.join(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))), "tools", "export_yolo.py")
            run_name = os.path.basename(os.path.normpath(out_dir))
            # --min-frames 1 defeats the tool's 60-frame default: a 20-frame
            # validation set is a legitimate thing to export, and silently
            # skipping it is how you end up with an empty datasets/ folder.
            subprocess.run([sys.executable, script,
                            "--runs", os.path.dirname(os.path.abspath(out_dir)),
                            "--scenes", run_name, "--min-frames", "1"],
                           cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           capture_output=True, timeout=900)
            post["yolo"] = "datasets/yolo"
        except Exception as exc:
            post["yolo_error"] = "%s: %s" % (type(exc).__name__, exc)
    if post:
        meta["post"] = post
        labels.write_json(os.path.join(out_dir, "meta.json"), meta)
    if not quiet:
        print("done: %d frames -> %s (%.1fs)" % (n_frames, out_dir, time.time() - t_start))
    return meta


def _preview_modalities(sensors):
    """Map requested sensors onto the directory names sheet.triptych expects."""
    names = {"optical": "optical", "ir": "ir_mwir", "radar": "radar",
             "sar": "sar", "lidar": "lidar_range"}
    return [names[s] for s in sensors if s in names]


MOD_SUFFIX = ("optical", "lidar", "ir", "ir_mwir", "ir_lwir", "ir_swir", "ir_lwir_only")


def _part_files(out_dir):
    """Group every annotations_*_coco.partNNN.json by modality."""
    groups = {}
    for fn in sorted(os.listdir(out_dir)):
        if not (fn.startswith("annotations_") and fn.endswith(".json")):
            continue
        if ".part" not in fn:
            continue
        head, tail = fn.split(".part", 1)
        mod = head[len("annotations_"):]
        if mod.endswith("_coco"):
            mod = mod[:-len("_coco")]
        groups.setdefault(mod, []).append(fn)
    return groups


def merge_parts(out_dir, cfg_path, n_frames=None, sensors=None, seed=None, quiet=False,
                t_start=None):
    """Concatenate chunk outputs into one COCO per modality plus stats/meta.

    Every chunk wrote the same modality with disjoint image ids, so merging is a
    plain concatenation in part order. Image ids are already global (they start
    at the chunk's frame_offset), which keeps the merged file valid COCO.
    """
    cfg = scene.load_config(cfg_path)
    jpeg_settings = _jpeg_settings(cfg)
    # Interference is a dataset-level post-process: validate it from the merged
    # config here and apply it once at the end. A per-chunk export would write N
    # partial mirrors and the chunks would disagree about the event window.
    disturbances = interference.validate(cfg.get("interference"))
    sensors = tuple(sensors or ("optical", "ir", "radar", "lidar", "sar"))
    if "all" in sensors:
        sensors = ("optical", "ir", "radar", "lidar", "sar")
    base_seed = int(seed if seed is not None else cfg.get("seed", 0))
    n_frames = int(n_frames if n_frames is not None else cfg.get("frames", {}).get("n", 1))
    ir_bands = (cfg.get("ir", {}) or {}).get("bands") or [None]

    merged = {}
    for mod, files in sorted(_part_files(out_dir).items()):
        acc = None
        for fn in files:
            try:
                d = _read_json(os.path.join(out_dir, fn))
            except (OSError, ValueError):
                continue
            if acc is None:
                acc = dict(d)
            else:
                acc["images"].extend(d.get("images", []))
                acc["annotations"].extend(d.get("annotations", []))
        if acc is None:
            continue
        acc["images"].sort(key=lambda i: i["id"])
        acc["annotations"].sort(key=lambda a: a["id"])
        name = "annotations_%s_coco.json" % mod
        labels.write_json(os.path.join(out_dir, name), acc)
        merged[mod] = acc

    # radar frame records live in their own per-chunk files
    rf = []
    for fn in sorted(os.listdir(out_dir)):
        if fn.startswith("radar_frames.part") and fn.endswith(".json"):
            try:
                rf.extend(_read_json(os.path.join(out_dir, fn)))
            except (OSError, ValueError):
                pass
    rf.sort(key=lambda r: r.get("frame", 0))
    if rf:
        labels.write_json(os.path.join(out_dir, "radar_frames.json"), rf)

    # SAR per-frame records, same per-chunk arrangement as the radar ones
    sf = []
    for fn in sorted(os.listdir(out_dir)):
        if fn.startswith("sar_frames.part") and fn.endswith(".json"):
            try:
                sf.extend(_read_json(os.path.join(out_dir, fn)))
            except (OSError, ValueError):
                pass
    sf.sort(key=lambda r: r.get("frame", 0))
    if sf:
        labels.write_json(os.path.join(out_dir, "sar_frames.json"), sf)

    # per-frame index, concatenated from the chunks
    fr_rows = []
    for fn in sorted(os.listdir(out_dir)):
        if fn.startswith("frames.part") and fn.endswith(".json"):
            try:
                fr_rows.extend(_read_json(os.path.join(out_dir, fn)))
            except (OSError, ValueError):
                pass
    fr_rows.sort(key=lambda r: r.get("frame", 0))
    if fr_rows:
        labels.write_json(os.path.join(out_dir, "frames.json"), fr_rows)

    cls_names = sorted({o.get("cls", "target") for o in cfg.get("objects", [])
                        if o.get("label", True)})
    cats = [dict(id=i + 1, name=c, supercategory="target")
            for i, c in enumerate(cls_names)]

    # splits over the whole run
    idx = list(range(n_frames))
    n_val = max(1, int(round(n_frames * 0.15))) if n_frames >= 8 else 0
    val = sorted(idx[-n_val:]) if n_val else []
    train = [i for i in idx if i not in set(val)]
    labels.write_json(os.path.join(out_dir, "splits.json"),
                      dict(train=["%06d" % i for i in train], val=["%06d" % i for i in val]))

    # stats rebuilt from what is on disk
    stats_path = os.path.join(out_dir, "stats.json")
    stats = {}
    if os.path.exists(stats_path):
        try:
            stats = _read_json(stats_path)
        except (OSError, ValueError):
            stats = {}
    stats["frames"] = n_frames

    def count(mod):
        d = merged.get(mod)
        if d is None:
            p = os.path.join(out_dir, "annotations_%s_coco.json" % mod)
            if not os.path.exists(p):
                return 0, 0
            try:
                d = _read_json(p)
            except (OSError, ValueError):
                return 0, 0
        return len(d.get("images", [])), len(d.get("annotations", []))

    n_i, n_a = count("optical")
    stats["optical_images"], stats["optical_boxes"] = n_i, n_a
    # extra views arrive as their own modality files (optical_wide, ...), so the
    # merged set is discovered rather than assumed
    stats["optical_views"] = {}
    for mod in sorted(merged):
        if not mod.startswith("optical") or mod == "optical":
            continue
        n_i, n_a = count(mod)
        stats["optical_views"][mod[len("optical"):].lstrip("_")] = dict(
            images=n_i, boxes=n_a, coco="annotations_%s_coco.json" % mod)
    n_i, n_a = count("lidar")
    stats["lidar_images"], stats["lidar_boxes"] = n_i, n_a
    n_i, n_a = count("sar")
    stats["sar_images"], stats["sar_boxes"] = n_i, n_a
    ir_stats = {}
    for mod in sorted(merged):
        if not mod.startswith("ir"):
            continue
        rest = mod[2:]                       # "", "_mwir", "_mwir_wide"
        band, vtag = "lwir", ""
        for b in ("swir", "mwir", "lwir"):
            if rest.startswith("_" + b):
                band, vtag = b, rest[len(b) + 1:].lstrip("_")
                break
        n_i, n_a = count(mod)
        ir_stats[band if not vtag else band + "_" + vtag] = dict(
            images=n_i, boxes=n_a, coco="annotations_%s_coco.json" % mod)
    for bnm in ir_bands:
        key = bnm or "lwir"
        if key not in ir_stats:
            n_i, n_a = count("ir" + ("" if bnm is None else "_" + bnm))
            if n_i or n_a:
                ir_stats[key] = dict(images=n_i, boxes=n_a,
                                     coco="annotations_ir%s_coco.json"
                                          % ("" if bnm is None else "_" + bnm))
    stats["ir"] = ir_stats
    stats["ir_images"] = sum(v["images"] for v in ir_stats.values())
    stats["ir_boxes"] = sum(v["boxes"] for v in ir_stats.values())
    if rf:
        stats["radar_frames"] = len(rf)
        stats["radar_points"] = int(sum(r["n"] for r in rf))
        stats["radar_boxes"] = int(sum(len(r["boxes"]) for r in rf))
        stats["radar_points_per_frame"] = [
            dict(frame=r.get("frame", r["n"]), clutter=r["clutter"], boxes=len(r["boxes"]),
                 snr_max=r["snr_max"]) for r in rf]
    if sf:
        stats["sar_frames"] = len(sf)
        stats["sar_per_frame"] = sf

    stats["boxes_per_class"] = {}
    for mod in sorted(merged):
        d = merged[mod]
        names = {c["id"]: c["name"] for c in d.get("categories", [])}
        for a in d.get("annotations", []):
            cn = names.get(a.get("category_id"), "?")
            stats["boxes_per_class"][cn] = stats["boxes_per_class"].get(cn, 0) + 1
    labels.write_json(stats_path, stats)

    states = list(plan_frames(cfg, n_frames, base_seed,
                              float(cfg.get("frames", {}).get("dt", 0.2))))
    # `sensors` is what THIS run rendered. A later partial pass (`-s optical`,
    # used to retouch exposure without redoing radar) then rewrites meta.json and
    # the dataset starts claiming it only has optics -- or, after a manual fix,
    # claiming a SAR channel on an air-to-air scene that cannot produce one.
    # The directories on disk are the only honest answer, so trust them and fall
    # back to the argument only when nothing was written.
    meta = dict(config=os.path.abspath(cfg_path), frames=n_frames,
                sensors=_sensors_on_disk(out_dir) or list(sensors),
                seed=base_seed,
                seconds=round(time.time() - t_start, 2) if t_start else 0.0,
                categories=cats, generated_by="simkit 0.2",
                ir_bands=[b or "lwir" for b in ir_bands],
                views=[dict(name=v["name"], tag=v["tag"], fov_deg=v.get("fov_deg"),
                            width=v.get("width"), height=v.get("height"))
                       for v in _views_of(states, cfg)],
                radar_band=(cfg.get("sensors", {}).get("radar", {}) or {}).get("band"),
                sar_band=(cfg.get("sensors", {}).get("sar", {}) or {}).get("band"),
                weather=[dict(s["weather"]) for s in states])
    if jpeg_settings is not None:
        jpeg_report = _jpeg_inventory(out_dir, jpeg_settings)
        if jpeg_report is not None:
            meta["jpeg"] = jpeg_report
    labels.write_json(os.path.join(out_dir, "meta.json"), meta)
    # The interference mirror has to be built from the merged dataset, after
    # meta.json exists: interference.export reads meta.json and frames.json to
    # rebuild the COCO files, so exporting any earlier fails on a missing file.
    # A chunked run never reaches run()'s own export (the part branch returns
    # early), so this is the only place it can happen for a merged run.
    if disturbances["enabled"]:
        interference.export(out_dir, os.path.join(out_dir, "interfered"), disturbances)
        meta["interference"] = dict(variant="interfered", domain="image_codes")
        labels.write_json(os.path.join(out_dir, "meta.json"), meta)
    if not quiet:
        print("merged %d modalities, %d frames -> %s" % (len(merged), n_frames, out_dir))
    return meta
