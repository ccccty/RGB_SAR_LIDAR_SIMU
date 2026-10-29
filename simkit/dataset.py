"""Dataset generator: frames -> images + annotations, with domain randomisation."""
import json
import os
import time

import numpy as np

from . import ir, labels, lidar, png, radar, scene, thermal, weather


def _rng(seed):
    return np.random.default_rng(int(seed))


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


def plan_frames(cfg, n_frames, base_seed, dt):
    """Same randomisation for every sensor: decide the world state once."""
    r = cfg.get("randomize", {})
    mot = cfg.get("motion", {})
    turntable = mot.get("mode") == "turntable"
    center = np.array(mot.get("center", [0.0, 0.0, 0.0]), float)
    wstates = weather.states_of(cfg)
    states = []
    for f in range(n_frames):
        rng = _rng(base_seed * 7919 + f)
        jit = {}
        if r.get("enabled", True):
            for o in cfg.get("objects", []):
                jxy = float(r.get("pose_jitter_xy_m", 0.0))
                jyaw = float(r.get("pose_jitter_yaw_deg", 0.0))
                if jxy > 0 or jyaw > 0:
                    jit[o["name"]] = dict(
                        dpos=[rng.normal(0, jxy), rng.normal(0, jxy), 0.0],
                        dyaw=rng.normal(0, jyaw),
                    )
        if turntable:
            ang = 360.0 * f / max(n_frames, 1)
            ca, sa = np.cos(np.radians(ang)), np.sin(np.radians(ang))
            for o in cfg.get("objects", []):
                p0 = np.array(o.get("pos", [0, 0, 0]), float) - center
                pr = np.array([ca * p0[0] - sa * p0[1], sa * p0[0] + ca * p0[1], p0[2]]) + center
                e = jit.setdefault(o["name"], dict(dpos=[0.0, 0.0, 0.0], dyaw=0.0))
                e["dpos"] = [float(pr[0] - o.get("pos", [0, 0, 0])[0]),
                             float(pr[1] - o.get("pos", [0, 0, 0])[1]), 0.0]
                e["dyaw"] = float(e["dyaw"] + ang)
        t = f * dt
        fr_cfg = json.loads(json.dumps(cfg))
        if r.get("enabled", True):
            if "temp_jitter_c" in r:
                fr_cfg["air"]["temp_c"] = float(cfg["air"]["temp_c"]) + float(
                    rng.uniform(-r["temp_jitter_c"], r["temp_jitter_c"]))
            if "sun_az_jitter_deg" in r:
                fr_cfg["sun"]["az_deg"] = float(cfg["sun"]["az_deg"]) + float(
                    rng.uniform(-r["sun_az_jitter_deg"], r["sun_az_jitter_deg"]))
        fr_cfg["_frame_seed"] = int(base_seed + f)
        wst = wstates[f % len(wstates)]
        fr_cfg["weather"] = dict(wst)
        fr_cfg["_weather"] = weather.derive(wst)
        if "temp_c" in wst:
            fr_cfg.setdefault("air", {})["temp_c"] = float(wst["temp_c"])
        states.append(dict(frame=f, t=t, jitter=jit, cfg=fr_cfg, weather=wst,
                           poses=object_poses(cfg, t, jit)))
    return states


def run(cfg_path, out_dir, n_frames=None, sensors=("ir", "radar"), seed=None, quiet=False):
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
    coco_irs = {}
    for bnm in ir_bands:
        bdesc = "IR" if bnm is None else "IR " + bnm
        coco_irs[bnm] = dict(info=dict(description="simkit synthetic dataset (%s)" % bdesc,
                                       version="0.1"),
                             images=[], annotations=[], categories=json.loads(json.dumps(cats)))
    coco_opt = dict(info=dict(description="simkit synthetic dataset (optical)", version="0.1"),
                    images=[], annotations=[], categories=[
                        dict(id=i, name=c, supercategory="target") for c, i in cls_id.items()])
    coco_lidar = dict(info=dict(description="simkit synthetic dataset (lidar range image)", version="0.1"),
                      images=[], annotations=[], categories=[
                          dict(id=i, name=c, supercategory="target") for c, i in cls_id.items()])
    radar_frames = []

    states = plan_frames(cfg, n_frames, base_seed, dt)
    t_start = time.time()

    if "optical" in sensors:
        from . import external
        backend = (cfg.get("optical") or {}).get("backend", "blender")
        manifest = external.write_manifest(cfg, states, out_dir)
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
        f, t, jit = s["frame"], s["t"], s["jitter"]
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
                img, anns = labels.to_coco(boxes, w, f, name, W, H)
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
            ra_db = radar.to_db(out["range_angle"].T)      # (n_s, n_beams)
            png.write_gray8(os.path.join(out_dir, dirs["radar_ra"], name),
                            np.ascontiguousarray(scope(ra_db.T)).astype(np.uint8))
            rlab = labels.radar_labels(w, out["detections"], f)
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
            img, anns = labels.to_coco(boxes, w, f, name, W, H)
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
                    anns.append(dict(id=f * 1000 + aid, image_id=f,
                                     category_id=cls_id[spec.get("cls", "target")],
                                     bbox=[b["x0"], b["y0"], round(b["x1"] - b["x0"], 2),
                                           round(b["y1"] - b["y0"], 2)],
                                     area=round((b["x1"] - b["x0"]) * (b["y1"] - b["y0"]), 2),
                                     iscrowd=0, object_name=nm, visible=b.get("visible", True)))
                coco_opt["images"].append(dict(id=f, file_name="%06d.png" % f,
                                               width=lb["width"], height=lb["height"]))
                coco_opt["annotations"].extend(anns)

        if not quiet:
            print("frame %d/%d done" % (f + 1, n_frames), flush=True)

    if "ir" in sensors:
        for bnm in ir_bands:
            tag = "" if bnm is None else "_" + bnm
            labels.write_json(os.path.join(out_dir, "annotations_ir%s_coco.json" % tag),
                              coco_irs[bnm])
    if "optical" in sensors:
        labels.write_json(os.path.join(out_dir, "annotations_optical_coco.json"), coco_opt)
    if "lidar" in sensors:
        labels.write_json(os.path.join(out_dir, "annotations_lidar_coco.json"), coco_lidar)
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

    if "optical" in sensors:
        n_i, n_a = count_coco(os.path.join(out_dir, "annotations_optical_coco.json"))
        stats["optical_images"], stats["optical_boxes"] = n_i, n_a
    if "lidar" in sensors:
        n_i, n_a = count_coco(os.path.join(out_dir, "annotations_lidar_coco.json"))
        stats["lidar_images"], stats["lidar_boxes"] = n_i, n_a
    if "ir" in sensors:
        ir = {}
        for bnm in ir_bands:
            tag = "" if bnm is None else "_" + bnm
            fn = "annotations_ir%s_coco.json" % tag
            n_i, n_a = count_coco(os.path.join(out_dir, fn))
            ir[bnm or "lwir"] = dict(images=n_i, boxes=n_a, coco=fn)
        stats["ir"] = ir
        stats["ir_images"] = sum(v["images"] for v in ir.values())
        stats["ir_boxes"] = sum(v["boxes"] for v in ir.values())
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
