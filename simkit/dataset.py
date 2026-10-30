"""Dataset generator: frames -> images + annotations, with domain randomisation."""
import json
import os
import time

import numpy as np

from . import ir, labels, lidar, png, radar, scene, thermal, weather


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
    """Same randomisation for every sensor: decide the world state once.

    frame_offset shifts the whole plan, which is what lets a large run be
    generated in several independent chunks and stitched back together: chunk k
    starts its random stream where chunk k-1 stopped, so the result is identical
    to one long run.
    """
    r = cfg.get("randomize", {})
    mot = cfg.get("motion", {})
    turntable = mot.get("mode") == "turntable"
    center = np.array(mot.get("center", [0.0, 0.0, 0.0]), float)
    wstates = weather.states_of(cfg)
    tot = int(total_frames or n_frames)
    # Static scenery footprints, computed once: placement must not drop the
    # labelled target behind (or inside) one of them.
    blockers = []
    for o in cfg.get("objects", []):
        if o.get("label", True):
            continue
        try:
            _m, lo, hi = scene.prepare_mesh(o, cfg.get("_base_dir", "."))
        except Exception:
            continue
        c = (np.asarray(lo, float) + np.asarray(hi, float)) / 2.0 \
            + np.asarray(o.get("pos", [0, 0, 0]), float)
        rad = 0.5 * float(np.linalg.norm(np.asarray(hi, float) - np.asarray(lo, float)))
        if rad > 0:
            blockers.append((c, rad))
    states = []
    for f0 in range(n_frames):
        f = f0 + int(frame_offset)
        t = f * dt
        rng = _rng(base_seed * 7919 + f)
        jit = {}
        if r.get("enabled", True):
            for o in cfg.get("objects", []):
                jxy = float(r.get("pose_jitter_xy_m", 0.0))
                jyaw = float(r.get("pose_jitter_yaw_deg", 0.0))
                jz = float(r.get("pose_jitter_z_m", 0.0))
                if jxy > 0 or jyaw > 0 or jz > 0:
                    jit[o["name"]] = dict(
                        dpos=[rng.normal(0, jxy), rng.normal(0, jxy), rng.normal(0, jz)],
                        dyaw=rng.normal(0, jyaw),
                    )
        # Place the labelled target somewhere inside the sensor's field of view
        # at a sampled range. Without this a 1000-frame set is 1000 pictures of
        # the same silhouette at the same distance, which trains nothing.
        pl = r.get("placement") if r.get("enabled", True) else None
        if pl:
            cam = (cfg.get("sensors", {}) or {}).get(pl.get("camera", "ir_camera"))
            if cam is None:
                cam = (cfg.get("sensors", {}) or {}).get("optical")
            if cam is not None:
                half_x = 0.5 * float(cam.get("fov_deg", 30.0)) * float(pl.get("fov_frac", 0.8))
                asp = float(cam.get("height", 3)) / max(float(cam.get("width", 4)), 1e-9)
                half_y = half_x * asp
                p0, right, up, fwd = scene.sensor_basis(
                    cam["pos"], cam["look_at"], cam.get("up", (0, 0, 1)))
                rng_lo, rng_hi = pl.get("range_m", [10.0, 50.0])
                targets = pl.get("targets") or [o["name"] for o in cfg.get("objects", [])
                                                if o.get("label", True)]
                for nm in targets:
                    spec = next((o for o in cfg.get("objects", []) if o["name"] == nm), None)
                    if spec is None:
                        continue
                    # build_world puts the target at base + vel*t before the
                    # jitter offset is added, so the offset has to cancel the
                    # motion too or the sampled range only holds on frame 0.
                    vel = np.array(spec.get("vel", [0, 0, 0]), float)
                    base = np.array(spec.get("pos", [0, 0, 0]), float) + vel * t
                    for _try in range(24):
                        rang = float(rng.uniform(float(rng_lo), float(rng_hi)))
                        # Angles are half-spans already scaled by fov_frac;
                        # scaling again here made the fan narrower than
                        # configured while the distance still came out right,
                        # so ranges drifted past the requested maximum.
                        az = np.radians(rng.uniform(-half_x, half_x))
                        el = np.radians(rng.uniform(-half_y, half_y))
                        d = fwd + np.tan(az) * right + np.tan(el) * up
                        d = d / (np.linalg.norm(d) + 1e-12)
                        if pl.get("keep_z"):
                        # Ground/sea objects live at a fixed height, so the flat
                        # ray has to be dropped onto that plane: keep the
                        # horizontal aim, then re-derive the range. Forcing z
                        # after the fact left the sampled elevation pointing
                        # above or below the object, and the target fell out of
                        # the frustum on most frames.
                            dz = base[2] - float(p0[2])
                            hz = np.hypot(d[0], d[1])
                            if hz < 1e-6 or abs(fwd[2]) > 0.99:
                                pos = np.asarray(p0, float) + d * rang
                            else:
                                horiz = float(np.clip(rang * hz, 1e-3, 1e5))
                                # keep the sign/heading of the horizontal ray
                                pos = np.array([p0[0] + d[0] / hz * horiz,
                                                p0[1] + d[1] / hz * horiz,
                                                base[2]])
                            rang = float(np.linalg.norm(pos - np.asarray(p0, float)))
                        else:
                            pos = np.asarray(p0, float) + d * rang
                        # Only rejects a candidate that sits behind scenery;
                        # on the last attempt the loop keeps whatever it has so
                        # a crowded scene still yields a frame.
                        if not blockers or not _segment_blocked(p0, pos, blockers):
                            break
                    e = jit.setdefault(nm, dict(dpos=[0.0, 0.0, 0.0], dyaw=0.0))
                    e["dpos"] = [float(pos[0] - base[0]), float(pos[1] - base[1]),
                                 float(pos[2] - base[2])]
        if turntable:
            ang = 360.0 * f / max(tot, 1)
            ca, sa = np.cos(np.radians(ang)), np.sin(np.radians(ang))
            for o in cfg.get("objects", []):
                p0 = np.array(o.get("pos", [0, 0, 0]), float) - center
                pr = np.array([ca * p0[0] - sa * p0[1], sa * p0[0] + ca * p0[1], p0[2]]) + center
                e = jit.setdefault(o["name"], dict(dpos=[0.0, 0.0, 0.0], dyaw=0.0))
                e["dpos"] = [float(pr[0] - o.get("pos", [0, 0, 0])[0]),
                             float(pr[1] - o.get("pos", [0, 0, 0])[1]), 0.0]
                e["dyaw"] = float(e["dyaw"] + ang)
        fr_cfg = json.loads(json.dumps(cfg))
        if r.get("enabled", True):
            if "temp_jitter_c" in r:
                fr_cfg["air"]["temp_c"] = float(cfg["air"]["temp_c"]) + float(
                    rng.uniform(-r["temp_jitter_c"], r["temp_jitter_c"]))
            if "sun_az_jitter_deg" in r:
                fr_cfg["sun"]["az_deg"] = float(cfg["sun"]["az_deg"]) + float(
                    rng.uniform(-r["sun_az_jitter_deg"], r["sun_az_jitter_deg"]))
        fr_cfg["_frame_seed"] = int(base_seed + f)
        if r.get("enabled", True):
            # sensor jitter: a dataset where every image is shot from the exact
            # same point teaches the network that pose, not the object
            js = float(r.get("sensor_jitter_m", 0.0))
            jyaw_s = float(r.get("sensor_jitter_yaw_deg", 0.0))
            if js > 0 or jyaw_s > 0:
                for skey in ("ir_camera", "optical", "radar", "lidar", "sar"):
                    sc = (fr_cfg.get("sensors", {}) or {}).get(skey)
                    if not isinstance(sc, dict) or "pos" not in sc:
                        continue
                    dpos = rng.normal(0, js, 3) if js > 0 else np.zeros(3)
                    sc["pos"] = [float(sc["pos"][0] + dpos[0]),
                                 float(sc["pos"][1] + dpos[1]),
                                 float(sc["pos"][2] + dpos[2] * 0.25)]
                    la = np.array(sc.get("look_at", [0, 0, 0]), float)
                    if jyaw_s > 0:
                        v = la - np.asarray(sc["pos"], float)
                        ang = np.radians(rng.normal(0, jyaw_s))
                        ca, sa = np.cos(ang), np.sin(ang)
                        la = np.asarray(sc["pos"], float) + np.array(
                            [ca * v[0] - sa * v[1], sa * v[0] + ca * v[1], v[2]])
                    sc["look_at"] = [float(la[0]), float(la[1]), float(la[2])]
            # sensor-level parameter jitter, e.g. which radar band this frame uses
            for skey, table in (r.get("sensor_choices") or {}).items():
                sc = (fr_cfg.get("sensors", {}) or {}).get(skey)
                if not isinstance(sc, dict):
                    continue
                for pkey, opts in table.items():
                    if opts:
                        sc[pkey] = opts[int(rng.integers(0, len(opts)))]
        wst = wstates[f % len(wstates)]
        fr_cfg["weather"] = dict(wst)
        fr_cfg["_weather"] = weather.derive(wst)
        if "temp_c" in wst:
            fr_cfg.setdefault("air", {})["temp_c"] = float(wst["temp_c"])
        yield dict(frame=f, t=t, jitter=jit, cfg=fr_cfg, weather=wst,
                   poses=object_poses(cfg, t, jit))


def run(cfg_path, out_dir, n_frames=None, sensors=("ir", "radar"), seed=None, quiet=False,
        frame_offset=0, part=None, total_frames=None):
    cfg = scene.load_config(cfg_path)
    sensors = tuple(sensors)
    if "all" in sensors:
        sensors = ("optical", "ir", "radar", "lidar")
    r = cfg.get("randomize", {})
    base_seed = int(seed if seed is not None else cfg.get("seed", 0))
    n_frames = int(n_frames if n_frames is not None else cfg.get("frames", {}).get("n", 1))
    dt = float(cfg.get("frames", {}).get("dt", 0.2))

    cls_names = sorted({o.get("cls", "target") for o in cfg.get("objects", [])
                        if o.get("label", True)})
    cls_id = {c: i + 1 for i, c in enumerate(cls_names)}

    ir_bands = (cfg.get("ir", {}) or {}).get("bands") or [None]
    dirs = dict(radar="images/radar", radar_ra="images/radar_ra",
                optical="images/optical",
                label_radar="labels/radar", label_optical="labels/optical",
                lidar_range="lidar/range", lidar_intensity="lidar/intensity",
                lidar_points="lidar/points", label_lidar="labels/lidar")
    for bnm in ir_bands:
        tag = "" if bnm is None else "_" + bnm
        dirs["ir" + tag] = "images/ir" + tag
        dirs["ir_preview" + tag] = "images/ir_preview" + tag
        dirs["label_ir" + tag] = "labels/ir" + tag
    for d in dirs.values():
        os.makedirs(os.path.join(out_dir, d), exist_ok=True)

    cats = [dict(id=i, name=c, supercategory="target") for c, i in cls_id.items()]

    def coco_shell(desc):
        return dict(info=dict(description=desc, version="0.1"),
                    images=[], annotations=[], categories=json.loads(json.dumps(cats)))

    coco_irs = {}
    for bnm in ir_bands:
        bdesc = "IR" if bnm is None else "IR " + bnm
        coco_irs[bnm] = coco_shell("simkit synthetic dataset (%s)" % bdesc)
    coco_opt = coco_shell("simkit synthetic dataset (optical)")
    coco_lidar = coco_shell("simkit synthetic dataset (lidar range image)")
    radar_frames = []

    # A run may be generated in chunks: frame_offset shifts both the random
    # stream and the COCO image ids, so the chunks concatenate into one valid
    # dataset. Image ids therefore start at frame_offset, not at 0.
    img_off = int(frame_offset)
    states = list(plan_frames(cfg, n_frames, base_seed, dt, frame_offset=frame_offset,
                              total_frames=total_frames))
    t_start = time.time()

    if "optical" in sensors:
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
            optical.render_frames(cfg, cfg_path, out_dir, manifest, script, quiet=quiet)
        else:
            external.run(cfg, out_dir, manifest, backend, quiet=quiet)

    for s in states:
        # plan_frames already applied frame_offset, so f is the global frame id
        # and the file numbers below stay unique across chunks.
        f = s["frame"]
        fid = f
        t, jit = s["t"], s["jitter"]
        fr_cfg = s["cfg"]
        if set(sensors) & {"ir", "radar", "lidar", "sar"}:
            w = scene.build_world(cfg, t=t, jitter=jit)
            for o in w.objs:
                # scenery objects carry no dataset class (they are geometry only)
                o["cls_id"] = cls_id.get(o["cls"], 0) if o.get("label", True) else 0

        if "ir" in sensors:
            T = thermal.solve(w, fr_cfg, shadow=cfg.get("ir", {}).get("shadow", True))
            for bnm in ir_bands:
                tag = "" if bnm is None else "_" + bnm
                fc = fr_cfg
                if bnm is not None:
                    fc = json.loads(json.dumps(fr_cfg))
                    fc.setdefault("ir", {})["band"] = bnm
                out = ir.render(w, fc, T)
                H, W = out["counts_u16"].shape
                name = "%06d.png" % f
                png.write_gray16(os.path.join(out_dir, dirs["ir" + tag], name), out["counts_u16"])
                png.write_gray8(os.path.join(out_dir, dirs["ir_preview" + tag], name), out["preview"])
                boxes = labels.boxes_from_mask(out["obj_id"], w)
                img, anns = labels.to_coco(boxes, w, fid, name, W, H)
                coco_irs[bnm]["images"].append(img)
                coco_irs[bnm]["annotations"].extend(anns)
                labels.write_json(os.path.join(out_dir, dirs["label_ir" + tag], "%06d.json" % f),
                                  dict(frame=f, sensor="ir", band=bnm or "lwir",
                                       weather=dict(fr_cfg.get("weather", {})),
                                       image=name, width=W, height=H,
                                       boxes=list(boxes.values()),
                                       objects=[dict(id=o["id"], name=o["name"], cls=o["cls"],
                                                     t_mean_c=float(T[w.face_obj == o["id"]].mean() - 273.15))
                                                for o in w.objs]))

        if "radar" in sensors:
            out = radar.simulate(w, fr_cfg)
            name = "%06d.png" % f
            # Both maps are noise-dominated: the median IS the noise floor. The
            # display window is anchored on that median with a fixed 45 dB
            # contrast range, which is roughly what a scope's STC curve gives
            # you. Anchoring on a high percentile instead lets one very bright
            # target (the sea scene) swallow the whole scale.
            def scope(db, dyn_db=45.0, gamma=0.85):
                lo = float(np.percentile(db, 50.0))      # the noise floor
                hi = lo + dyn_db                        # fixed contrast window
                v = np.clip((db - lo) / dyn_db, 0.0, 1.0) ** gamma
                return np.round(v * 255.0)

            # range/Doppler map, range on the horizontal axis (waterfall layout)
            rd_db = radar.to_db(out["rd_map"])
            png.write_gray8(os.path.join(out_dir, dirs["radar"], name),
                            np.ascontiguousarray(scope(rd_db.T)).astype(np.uint8))
            # range/angle map: one row per beam, one column per range bin.
            # This is the picture that answers "did it see the target".
            ra_db = radar.to_db(out["range_angle"].T)      # (n_beams, n_s)
            ra_img = np.ascontiguousarray(scope(ra_db.T))
            png.write_gray8(os.path.join(out_dir, dirs["radar_ra"], name),
                            ra_img.astype(np.uint8))
            rlab = labels.radar_labels(w, out["detections"], fid)
            # Where the brightest echo sits in that picture. A range/angle map is
            # mostly noise floor; without this the viewer cannot tell a target
            # from speckle without hunting for one white pixel.
            iy, ix = np.unravel_index(int(np.argmax(ra_img)), ra_img.shape)
            rlab["ra_peak"] = dict(px=int(ix), py=int(iy),
                                   db=round(float(ra_db.max()), 1),
                                   w=int(ra_img.shape[1]), h=int(ra_img.shape[0]))
            labels.write_json(os.path.join(out_dir, dirs["label_radar"], "%06d.json" % f), rlab)
            radar_frames.append(dict(
                n=int(rlab.get("num_points", 0)),
                clutter=int(rlab.get("clutter_count", 0)),
                boxes=rlab.get("boxes_3d") or [],
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
            png.write_gray8(os.path.join(out_dir, dirs["lidar_range"], name),
                            np.round(vis * 255).astype(np.uint8))
            png.write_gray8(os.path.join(out_dir, dirs["lidar_intensity"], name),
                            np.round(np.clip(out["intensity"], 0, 1) * 255).astype(np.uint8))
            lidar.write_ply(os.path.join(out_dir, dirs["lidar_points"], "%06d.ply" % f), out["cloud"], out["origin"])
            boxes = labels.boxes_from_mask(out["obj_id"], w)
            img, anns = labels.to_coco(boxes, w, fid, name, W, H)
            coco_lidar["images"].append(img)
            coco_lidar["annotations"].extend(anns)
            labels.write_json(os.path.join(out_dir, dirs["label_lidar"], "%06d.json" % f),
                              dict(frame=f, sensor="lidar", image=name, width=W, height=H,
                                   weather=dict(fr_cfg.get("weather", {})),
                                   hit_ratio=round(out["hit_ratio"], 4),
                                   n_points=out["n_points"],
                                   max_range_m=out["max_range_m"],
                                   boxes=list(boxes.values()),
                                   objects=[dict(id=o["id"], name=o["name"], cls=o["cls"],
                                                 points=int((out["obj_id"] == o["id"]).sum()))
                                            for o in w.objs if o.get("label", True)]))

        if "optical" in sensors:
            lp = os.path.join(out_dir, dirs["label_optical"], "%06d.json" % f)
            if os.path.exists(lp):
                with open(lp, "r", encoding="utf-8") as fh:
                    lb = json.load(fh)
                anns = []
                for aid, (nm, b) in enumerate(lb["boxes"].items()):
                    spec = next((o for o in cfg["objects"] if o.get("name") == nm), None)
                    if spec is None or not spec.get("label", True):
                        continue
                    anns.append(dict(id=fid * 1000 + aid, image_id=fid,
                                     category_id=cls_id[spec.get("cls", "target")],
                                     bbox=[b["x0"], b["y0"], round(b["x1"] - b["x0"], 2),
                                           round(b["y1"] - b["y0"], 2)],
                                     area=round((b["x1"] - b["x0"]) * (b["y1"] - b["y0"]), 2),
                                     iscrowd=0, object_name=nm, visible=b.get("visible", True)))
                coco_opt["images"].append(dict(id=fid, file_name="%06d.png" % f,
                                               width=lb["width"], height=lb["height"]))
                coco_opt["annotations"].extend(anns)

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
            tag = "" if bnm is None else "_" + bnm
            labels.write_json(os.path.join(out_dir, anno_name("ir" + tag)), coco_irs[bnm])
    if "optical" in sensors:
        labels.write_json(os.path.join(out_dir, anno_name("optical")), coco_opt)
    if "lidar" in sensors:
        labels.write_json(os.path.join(out_dir, anno_name("lidar")), coco_lidar)
    if part is not None:
        # A chunk does not own the run: no splits, no stats, no meta. Whoever
        # merged the chunks writes those, once, from the whole dataset.
        labels.write_json(os.path.join(out_dir, "radar_frames.part%03d.json" % int(part)),
                          radar_frames)
        return dict(chunk=int(part), frames=n_frames, offset=int(frame_offset),
                    radar_frames=radar_frames)
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
            stats = json.load(open(stats_path, encoding="utf-8"))
        except (OSError, ValueError):
            stats = {}
    stats["frames"] = n_frames

    def count_coco(path):
        if not os.path.exists(path):
            return 0, 0
        try:
            d = json.load(open(path, encoding="utf-8"))
        except (OSError, ValueError):
            return 0, 0
        return len(d.get("images", [])), len(d.get("annotations", []))

    n_i, n_a = count_coco(os.path.join(out_dir, "annotations_optical_coco.json"))
    stats["optical_images"], stats["optical_boxes"] = n_i, n_a
    n_i, n_a = count_coco(os.path.join(out_dir, "annotations_lidar_coco.json"))
    stats["lidar_images"], stats["lidar_boxes"] = n_i, n_a
    # IR band keys come from the config; counts come from disk, so a radar-only
    # rerun still reports the IR frames an earlier run left in place.
    ir_stats = stats.get("ir") or {}
    for bnm in ir_bands:
        tag = "" if bnm is None else "_" + bnm
        fn = "annotations_ir%s_coco.json" % tag
        n_i, n_a = count_coco(os.path.join(out_dir, fn))
        ir_stats[bnm or "lwir"] = dict(images=n_i, boxes=n_a, coco=fn)
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

    # cross-modality class tally, recomputed from every COCO present
    stats["boxes_per_class"] = {}
    for fn in sorted(os.listdir(out_dir)):
        if not (fn.startswith("annotations_") and fn.endswith("_coco.json")):
            continue
        try:
            d = json.load(open(os.path.join(out_dir, fn), encoding="utf-8"))
        except (OSError, ValueError):
            continue
        names = {c["id"]: c["name"] for c in d.get("categories", [])}
        for a in d.get("annotations", []):
            cn = names.get(a.get("category_id"), "?")
            stats["boxes_per_class"][cn] = stats["boxes_per_class"].get(cn, 0) + 1
    labels.write_json(stats_path, stats)

    meta = dict(config=os.path.abspath(cfg_path), frames=n_frames, sensors=list(sensors),
                seed=base_seed, seconds=round(time.time() - t_start, 2),
                categories=cats, generated_by="simkit 0.2",
                ir_bands=[b or "lwir" for b in ir_bands],
                radar_band=(cfg.get("sensors", {}).get("radar", {}) or {}).get("band"),
                weather=[dict(s["weather"]) for s in states])
    labels.write_json(os.path.join(out_dir, "meta.json"), meta)
    if not quiet:
        print("done: %d frames -> %s (%.1fs)" % (n_frames, out_dir, time.time() - t_start))
    return meta

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
    sensors = tuple(sensors or ("optical", "ir", "radar", "lidar"))
    if "all" in sensors:
        sensors = ("optical", "ir", "radar", "lidar")
    base_seed = int(seed if seed is not None else cfg.get("seed", 0))
    n_frames = int(n_frames if n_frames is not None else cfg.get("frames", {}).get("n", 1))
    ir_bands = (cfg.get("ir", {}) or {}).get("bands") or [None]

    merged = {}
    for mod, files in sorted(_part_files(out_dir).items()):
        acc = None
        for fn in files:
            try:
                d = json.load(open(os.path.join(out_dir, fn), encoding="utf-8"))
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
                rf.extend(json.load(open(os.path.join(out_dir, fn), encoding="utf-8")))
            except (OSError, ValueError):
                pass
    rf.sort(key=lambda r: r.get("frame", 0))
    if rf:
        labels.write_json(os.path.join(out_dir, "radar_frames.json"), rf)

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
            stats = json.load(open(stats_path, encoding="utf-8"))
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
                d = json.load(open(p, encoding="utf-8"))
            except (OSError, ValueError):
                return 0, 0
        return len(d.get("images", [])), len(d.get("annotations", []))

    n_i, n_a = count("optical")
    stats["optical_images"], stats["optical_boxes"] = n_i, n_a
    n_i, n_a = count("lidar")
    stats["lidar_images"], stats["lidar_boxes"] = n_i, n_a
    ir_stats = {}
    for bnm in ir_bands:
        tag = "" if bnm is None else "_" + bnm
        key = bnm or "lwir"
        n_i, n_a = count("ir" + tag)
        ir_stats[key] = dict(images=n_i, boxes=n_a, coco="annotations_ir%s_coco.json" % tag)
    for bnm in ("mwir", "lwir"):
        if bnm not in ir_stats:
            n_i, n_a = count("ir_" + bnm)
            if n_i or n_a:
                ir_stats[bnm] = dict(images=n_i, boxes=n_a,
                                     coco="annotations_ir_%s_coco.json" % bnm)
    stats["ir"] = ir_stats
    stats["ir_images"] = sum(v["images"] for v in ir_stats.values())
    stats["ir_boxes"] = sum(v["boxes"] for v in ir_stats.values())
    if rf:
        stats["radar_frames"] = len(rf)
        stats["radar_points"] = int(sum(r["n"] for r in rf))
        stats["radar_boxes"] = int(sum(len(r["boxes"]) for r in rf))
        stats["radar_points_per_frame"] = [
            dict(frame=r["n"], clutter=r["clutter"], boxes=len(r["boxes"]),
                 snr_max=r["snr_max"]) for r in rf]

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
    meta = dict(config=os.path.abspath(cfg_path), frames=n_frames, sensors=list(sensors),
                seed=base_seed,
                seconds=round(time.time() - t_start, 2) if t_start else 0.0,
                categories=cats, generated_by="simkit 0.2",
                ir_bands=[b or "lwir" for b in ir_bands],
                radar_band=(cfg.get("sensors", {}).get("radar", {}) or {}).get("band"),
                weather=[dict(s["weather"]) for s in states])
    labels.write_json(os.path.join(out_dir, "meta.json"), meta)
    if not quiet:
        print("merged %d modalities, %d frames -> %s" % (len(merged), n_frames, out_dir))
    return meta