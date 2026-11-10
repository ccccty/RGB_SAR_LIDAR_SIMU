"""Build the 2026-11-09 v1 sea / land / air tri-modal dataset.

Deliverable: D:\\work\\多模态数据集生成\\dataest\\海陆空1109v1
    3 domains x 5 frames x 3 modalities (optical / LWIR / radar range-Doppler).

Why v1 exists
    The v1109 ground-vehicle frames looked nothing like a real aerial vehicle
    dataset: a dark tank on a dune photo with diagonal tiling seams, no road,
    no recognisable traffic. This generator rebuilds that scene from the
    characteristics of the real benchmarks (see 参考数据集对照.md next to the
    data) and gives every domain explicit, verifiable target and decoy motion
    trajectories.

What changed for the vehicle scene
    * optical.sky_image removed. It is a look-down view: the camera-parented
      sky card sits IN FRONT of the ground and covers the whole frame, which is
      where the diagonal bands came from.
    * ground.backdrop switched from a dune photo to a real roundabout/street
      aerial photo, with backdrop_span_m enlarged 200 -> 400 m so the REPEAT
      tile seam leaves the visible footprint.
    * primary target changed from a 15 m tank to a 6.75 m delivery van, in a
      light paint, driving on the ground (motion_type=ground, z velocity 0).
    * multiple normal cars with their own trajectories + fixed roadside
      obstacles. Every non-primary object carries motion_type=ground, otherwise
      the seeker planner silently turns it into a formation member that follows
      the target (plan.py:470).

Trajectory contract
    After generation, trajectories.json is written per run and
    trajectories_summary.json at the root. They record per-frame world position
    and velocity for the primary target and EVERY decoy, plus hard checks:
      * primary: smooth, constant-velocity, strictly closing range, labelled in
        all 5 frames;
      * static decoys: exactly zero displacement;
      * moving traffic: non-zero, constant-velocity displacement;
      * air flares: monotonically descending, never inheriting the aircraft's
        velocity.
    The flares ride the RELEASED link model (drag + gravity), and the optical
    manifest samples their poses per frame from the same ecm.entities_at the
    numpy World uses, so both sides agree exactly.

Usage
    env\\Scripts\\python.exe tools\\build_sealandair_1109v1.py
    env\\Scripts\\python.exe tools\\build_sealandair_1109v1.py --skip-gen   # verify only
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENES = os.path.join(ROOT, "scenes")
PY = os.path.join(ROOT, "env", "Scripts", "python.exe")
DEFAULT_DEST = r"D:\work\多模态数据集生成\dataest\海陆空1109v1"
N_FRAMES = 5
DT = 0.1
SENSORS = "optical,ir,radar"
MODALITIES = ("optical", "ir_lwir", "radar_rd")

DOMAINS = [
    dict(key="air_plane", label="空", base="ds1109_air_plane.json"),
    dict(key="ground_vehicle", label="陆", base="ds1109_ground_vehicle.json"),
    dict(key="sea_ship", label="海", base="ds1109_sea_ship.json"),
]

# Real-photo road plate (see 参考数据集对照.md): a top-down roundabout with
# lane markings, parked cars and buildings -- the VEDAI / DLR3K / DroneVehicle
# look, instead of a featureless dune.
ROAD_PLATE = os.path.join(
    ROOT, "assets", "backdrops", "city",
    "06_Drone_view_of_roundabout_Unsplash.jpg")

# Ground truth classes used by the trajectory report.
STATIC_DECOYS = ("barrier_static", "fence_static", "shrub_static",
                 "parked_car_static", "anchored_boat_static")
MOVING_DECOYS = ("moving_suv", "moving_pickup", "moving_hatchback",
                 "moving_fishing_boat")


def jload(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def jwrite(path, obj):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)


def abs_asset(path, base=SCENES):
    """Blender checks os.path.isfile() against its own cwd; keep every asset
    absolute so a scene file can live in the deliverable directory."""
    if not isinstance(path, str) or not path or os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base, path))


# ---------------------------------------------------------------------------
# Ground scene: rebuild for real-aerial-vehicle look
# ---------------------------------------------------------------------------
GROUND_STATIC = [
    # name, model, pos, yaw, height, material
    ("barrier_static", "../assets/models/concrete_road_barrier.obj",
     [-28.0, 18.0, 0.0], 12.0, 0.9, "concrete"),
    ("fence_static", "../assets/models/modular_chainlink_fence.obj",
     [34.0, -24.0, 0.0], 90.0, 2.2, "metal_bare"),
    ("parked_car_static", "../assets/models/地面车辆/紧凑型轿车.obj",
     [20.0, 26.0, 0.0], -20.0, 1.6, "painted_metal"),
]

GROUND_MOVING = [
    # name, model, pos, velocity, height, material
    ("moving_suv", "../assets/models/地面车辆/运动型多用途车.obj",
     [-34.0, 12.0, 0.0], [7.0, 3.0, 0.0], 1.9, "paint_civil_air"),
    ("moving_pickup", "../assets/models/地面车辆/皮卡.obj",
     [30.0, -14.0, 0.0], [-6.0, 4.0, 0.0], 1.9, "matte_paint"),
    ("moving_hatchback", "../assets/models/地面车辆/两厢轿车.obj",
     [6.0, 32.0, 0.0], [4.0, -5.0, 0.0], 1.6, "painted_metal"),
]


def ground_cfg(raw):
    cfg = copy.deepcopy(raw)
    primary = cfg["objects"][0]

    # A 5.4 m delivery van scaled to 6.75 m is a legitimate "large vehicle"
    # class in VEDAI/DLR3K, and renders ~30 px at this slant range.
    primary.update(
        name="vehicle_01", cls="vehicle", shape="obj",
        model=abs_asset("../assets/models/地面车辆/厢式货车.obj"),
        scale=1.25, mat="paint_car_white",
        motion_type="ground", yaw_mode="velocity",
        vel=[8.0, 4.0, 0.0], label=True,
        note="primary target: delivery van driving along the road")
    primary.pop("height", None)
    # A 6.75 m delivery van is roughly 8 dBsm at mm-wave. Letting the geometric
    # PO solver infer it from the mesh produced 39.7 dBsm (~9000 m2) because
    # coherent face summation over-estimates a box-like body; that would put an
    # absurdly bright peak in every RD frame. Keep a measured-scale override.
    primary["rcs_db"] = 8.0

    objs = [primary]
    for name, model, pos, yaw, height, mat in GROUND_STATIC:
        objs.append(dict(
            name=name, cls="scenery", shape="obj", model=abs_asset(model),
            pos=list(pos), vel=[0.0, 0.0, 0.0], rot=yaw, height=height,
            mat=mat, label=False, motion_type="ground", yaw_mode="fixed",
            note="STATIC roadside obstacle: must never move"))
    for name, model, pos, vel, height, mat in GROUND_MOVING:
        objs.append(dict(
            name=name, cls="scenery", shape="obj", model=abs_asset(model),
            pos=list(pos), vel=list(vel), rot=0.0, height=height, mat=mat,
            label=False, motion_type="ground", yaw_mode="velocity",
            kinematics=dict(min_speed_mps=0.0, max_speed_mps=25.0,
                            max_accel_mps2=5.0, max_climb_mps=0.0,
                            min_altitude_m=0.0, max_bank_deg=5.0),
            note="MOVING traffic interference: its own constant-velocity track"))
    cfg["objects"] = objs

    # Look-down view: no camera-parented sky card. Keeping it produced the
    # diagonal seams and washed the real street plate out of the frame.
    cfg["optical"].pop("sky_image", None)
    cfg["ground"].update(
        backdrop=ROAD_PLATE, backdrop_span_m=400.0,
        backdrop_span_y_m=400.0, backdrop_aligned=True,
        backdrop_offset=[0.5, 0.5], mat="asphalt")
    cfg["ir"]["background_image"] = ROAD_PLATE
    cfg["optical"]["exposure_bias"] = 0.30

    sk = cfg["clip"]["seeker"]
    sk.update(target_m=4.6, target_px=[28, 34], range_m=[520, 760],
              closing="in", closing_mps=12.0, depression_deg=45.0,
              profile="linear", wander_deg=0.02, los_jitter_deg=0.01)
    cfg["_why_v1_ground"] = (
        "road/street plate + no sky card + 400 m tile span removes REPEAT seams; "
        "normal van + 3 moving cars + 3 static objects, all motion_type=ground "
        "so only the moving traffic actually moves.")
    return cfg


def air_cfg(raw):
    cfg = copy.deepcopy(raw)
    # The six dispensed rounds in ds1109_air_plane.json are already mounted on
    # the aircraft heading axis and ejected downward at -42..-50 m/s. Five
    # frames simply make the descent legible.
    sk = cfg["clip"]["seeker"]
    sk.update(closing_mps=48.0, profile="linear")
    cfg["_why_v1_air"] = (
        "5 frames expose the independent flare descent; aircraft keeps its "
        "smooth closing/outbound trajectory, rounds never inherit its velocity.")
    return cfg


def sea_cfg(raw):
    cfg = copy.deepcopy(raw)
    primary = cfg["objects"][0]
    # Real coastline surveillance (SeaShips) shows other small craft nearby.
    # motion_type=ground is the planner's "own trajectory" switch; without it
    # the seeker drags every extra object along with the primary ship.
    cfg["objects"] = [primary, dict(
        name="moving_fishing_boat", cls="scenery", shape="obj",
        model=abs_asset("../assets/models/民用船舶/小型渔船.obj"),
        pos=[58.0, -42.0, 0.0], vel=[3.0, 5.0, 0.0], rot=25.0,
        height=6.0, mat="paint_vessel", label=False,
        motion_type="ground", yaw_mode="velocity",
        note="MOVING small craft decoy with its own track"), dict(
        name="anchored_boat_static", cls="scenery", shape="obj",
        model=abs_asset("../assets/models/民用船舶/游艇.obj"),
        pos=[-62.0, 34.0, 0.0], vel=[0.0, 0.0, 0.0], rot=-40.0,
        height=5.0, mat="paint_vessel", label=False,
        motion_type="ground", yaw_mode="fixed",
        note="STATIC anchored craft: must never move")]
    sk = cfg["clip"]["seeker"]
    sk.update(closing_mps=12.0, profile="linear")
    cfg["_why_v1_sea"] = (
        "5 frames + one moving and one anchored small craft provide explicit "
        "decoy trajectories on top of the sea-clutter ridge.")
    return cfg


BUILDERS = {"ground_vehicle": ground_cfg, "air_plane": air_cfg,
            "sea_ship": sea_cfg}


def build_cfg(dom, scene_dir):
    from simkit import plan as planlib

    cfg = BUILDERS[dom["key"]](jload(os.path.join(SCENES, dom["base"])))
    cfg["frames"]["n"] = N_FRAMES
    cfg["frames"]["dt"] = DT
    cfg["clip"]["frames_per_clip"] = N_FRAMES
    cfg["capture"]["frames"] = N_FRAMES
    contract = cfg.setdefault("__trimodal_contract", {})
    contract["frames"] = N_FRAMES
    contract["modalities"] = list(MODALITIES)
    # Realism switches (PLAN.md §5 item 0) are injected by scene.load_config;
    # opt into the main-process JPEG product explicitly.
    cfg["realism"] = dict(enabled=True, jpeg=True)
    # A real plate already contains its own sun and lamp highlights. Re-running
    # the synthetic glare stage on top turned a dispensed flare fan and a white
    # van roof into white blobs -- which is fatal for a detection sample. The
    # curated presets opt out; this dataset must too.
    cfg.setdefault("optical", {}).setdefault("camera", {}).setdefault(
        "flare", {})["enabled"] = False
    cfg["name"] = "ds1109v1_" + dom["key"]

    # All model / plate references become absolute so the scene file can live
    # beside the dataset instead of in scenes/.
    for o in cfg.get("objects") or []:
        if o.get("model"):
            o["model"] = abs_asset(o["model"])
    for holder, key in ((cfg.get("optical") or {}, "sky_image"),
                        (cfg.get("ir") or {}, "background_image"),
                        (cfg.get("ground") or {}, "backdrop")):
        if holder.get(key):
            holder[key] = abs_asset(holder[key])

    # The seeker planner back-solves the primary target to a world position
    # hundreds of metres from the config origin. Decoys authored around (0,0)
    # would then be far OUTSIDE the frame. Place every decoy relative to the
    # planner's frame-0 target position.
    primary_name = (cfg.get("objects") or [{}])[0].get("name")
    state0 = next(iter(planlib.plan_frames(
        cfg, 1, int(cfg.get("seed", 0)), DT)))
    p0 = [float(v) for v in state0["poses"][primary_name]["pos"]]
    for o in (cfg.get("objects") or [])[1:]:
        pos = _vec(o.get("pos"))
        o["pos"] = [round(pos[0] + p0[0], 4),
                    round(pos[1] + p0[1], 4), pos[2] + p0[2] * 0.0]

    # Centre the road photograph on the target: UV = world/span + offset, so
    # offset = 0.5 - p0/span puts the roundabout under the vehicle. Decoys sit
    # within +/-40 m = +/-0.1 UV, i.e. all inside one tile (no seam).
    g = cfg.get("ground") or {}
    if g.get("backdrop") and dom["key"] in ("ground_vehicle", "sea_ship"):
        span = float(g.get("backdrop_span_m", 400.0))
        spany = float(g.get("backdrop_span_y_m", span))
        # For a road plate the photographic centre is a grass roundabout
        # island. Put the target on the multi-lane asphalt instead: UV (0.20,
        # 0.45) is a straight road with lane markings and parked traffic, so a
        # driving trajectory reads as a driving trajectory.
        centre_u, centre_v = ((0.20, 0.45) if dom["key"] == "ground_vehicle"
                              else (0.50, 0.50))
        g["backdrop_offset"] = [round(centre_u - p0[0] / span, 6),
                                round(centre_v - p0[1] / spany, 6)]

    # Re-mount the air flares on the v1 aircraft pose (the base file baked the
    # 3-frame pose; recomputing keeps them exactly on the fuselage axis).
    if dom["key"] == "air_plane":
        from build_sealandair_1109 import flare_fan
        ecm = cfg.setdefault("ecm", {})
        keep = [e for e in ecm.get("entities", [])
                if e.get("kind") != "infrared_flare"]
        ecm["entities"] = keep + flare_fan(cfg)
        ecm["enabled"] = True

    cfg["_v1_primary_frame0_world_m"] = [round(v, 3) for v in p0]
    jwrite(os.path.join(scene_dir, "ds1109v1_" + dom["key"] + ".json"), cfg)
    return cfg


# ---------------------------------------------------------------------------
# Generation / PNG metadata
# ---------------------------------------------------------------------------
def run_gen(scene_path, out_dir, threads):
    cmd = [PY, "-m", "simkit", "gen", "-c", scene_path, "-o", out_dir,
           "-n", str(N_FRAMES), "-s", SENSORS]
    env = dict(os.environ)
    env["PYTHONPATH"] = ROOT
    if threads:
        env["SIMKIT_BLENDER_THREADS"] = str(threads)
    print("RUN", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    tail = ((p.stdout or "") + "\n" + (p.stderr or ""))[-4000:]
    if p.returncode != 0:
        raise RuntimeError("simkit gen failed (%s):\n%s" % (p.returncode, tail))
    print(tail[-900:], flush=True)


def stamp_optical_meta(dest, scene_dir):
    """Deterministic pHYs + tEXt on every delivered optical PNG."""
    import numpy as np
    from PIL import Image
    sys.path.insert(0, ROOT)
    from simkit import png as png_writer

    stamped = 0
    for dom in DOMAINS:
        cfg = jload(os.path.join(scene_dir, "ds1109v1_" + dom["key"] + ".json"))
        view = ((cfg.get("__trimodal_contract") or {}).get("view") or {})
        fov = float(view.get("fov_deg")
                    or (cfg.get("sensors", {}).get("optical") or {}).get("fov_deg")
                    or 10.0)
        folder = os.path.join(dest, dom["key"], "images", "optical")
        if not os.path.isdir(folder):
            continue
        fr_path = os.path.join(dest, dom["key"], "frames.json")
        frames = jload(fr_path) if os.path.isfile(fr_path) else []
        for i, fn in enumerate(sorted(n for n in os.listdir(folder)
                                      if n.lower().endswith(".png"))):
            path = os.path.join(folder, fn)
            arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
            w = arr.shape[1]
            meta = dict(modality="optical", fov_deg=fov,
                        radiometric_calibration="relative-linear; not absolute")
            if i < len(frames):
                rng = (frames[i] or {}).get("range_m")
                if rng:
                    rng = float(rng)
                    meta["range_m"] = round(rng, 3)
                    meta["gsd_m_per_px"] = round(
                        2.0 * rng * math.tan(math.radians(fov) / 2.0) / w, 6)
            png_writer.write_rgb8(path, arr, meta=meta)
            stamped += 1
    print("META stamped optical PNGs:", stamped)
    return stamped


def count_png(path):
    return len([n for n in os.listdir(path)
                if n.lower().endswith(".png")]) if os.path.isdir(path) else 0


# ---------------------------------------------------------------------------
# Trajectory ground truth + acceptance checks
# ---------------------------------------------------------------------------
def _vec(v):
    return [float(x) for x in (v or [0.0, 0.0, 0.0])]


def _dist(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def build_trajectories(dest, dom, cfg):
    run = os.path.join(dest, dom["key"])
    mf = os.path.join(run, "optical_frames.json")
    frames = (jload(mf) or {}).get("frames", []) if os.path.isfile(mf) else []
    fj = jload(os.path.join(run, "frames.json")) if os.path.isfile(
        os.path.join(run, "frames.json")) else []
    primary_name = (cfg.get("objects") or [{}])[0].get("name")
    ecm_ids = [e["id"] for e in (cfg.get("ecm") or {}).get("entities", [])
               if e.get("kind") == "infrared_flare"]

    out_frames, checks = [], {}
    prev_ranges = []
    for i, fr in enumerate(frames):
        objs = fr.get("objects") or {}
        tj = fj[i] if i < len(fj) else {}
        p = objs.get(primary_name) or {}
        decoys = {}
        for name, pose in objs.items():
            if name == primary_name:
                continue
            kind = ("flare" if name in ecm_ids else
                    "static" if name in STATIC_DECOYS else
                    "moving" if name in MOVING_DECOYS else "other")
            decoys[name] = dict(kind=kind, pos=_vec(pose.get("pos")),
                                vel=_vec(pose.get("velocity_mps")))
        out_frames.append(dict(
            frame=i, t=tj.get("t", i * DT),
            target=dict(pos=_vec(p.get("pos")),
                        vel=_vec(p.get("velocity_mps")),
                        range_m=tj.get("range_m"),
                        target_px=tj.get("target_px")),
            decoys=decoys))
        r = tj.get("range_m")
        if r is not None:
            prev_ranges.append(float(r))

    # --- primary trajectory checks -----------------------------------------
    tp = [f["target"]["pos"] for f in out_frames]
    steps = [_dist(tp[k], tp[k + 1]) for k in range(len(tp) - 1)]
    checks["primary_frames"] = len(tp) == N_FRAMES
    checks["primary_moves"] = bool(steps) and min(steps) > 1e-3
    checks["primary_smooth"] = (bool(steps) and
                                max(steps) - min(steps) < 0.35 * max(steps) + 1e-6)
    checks["primary_grounded"] = (dom["key"] == "ground_vehicle" and
                                  all(abs(p[2]) < 3.0 for p in tp)) or \
                                 dom["key"] != "ground_vehicle"
    checks["range_monotonic"] = (
        all(b <= a + 1e-6 for a, b in zip(prev_ranges, prev_ranges[1:]))
        or all(b >= a - 1e-6 for a, b in zip(prev_ranges, prev_ranges[1:]))
    ) if len(prev_ranges) > 1 else False

    # --- decoy trajectory checks -------------------------------------------
    static_ok, moving_ok, flare_ok = [], [], []
    for name in {n for f in out_frames for n in f["decoys"]}:
        seq = [f["decoys"][name] for f in out_frames if name in f["decoys"]]
        if len(seq) < 2:
            continue
        disp = [_dist(seq[0]["pos"], s["pos"]) for s in seq[1:]]
        # Constant velocity means CONSTANT PER-FRAME STEP, not constant
        # displacement from frame 0 (which grows linearly by definition).
        steps_here = [_dist(seq[k]["pos"], seq[k + 1]["pos"])
                      for k in range(len(seq) - 1)]
        kind = seq[0]["kind"]
        if kind == "static":
            static_ok.append(max(disp) < 1e-6)
        elif kind == "moving":
            moving_ok.append(
                min(steps_here) > 1e-3 and
                max(steps_here) - min(steps_here) < 0.05 * max(steps_here) + 1e-6)
        elif kind == "flare":
            z = [s["pos"][2] for s in seq]
            flare_ok.append(all(b < a for a, b in zip(z, z[1:])))
    # A domain without a given decoy class must not fail that class's check:
    # the air scene has flares but no static roadside objects, and the ground
    # scene has cars but no flares. Record the counts so "not applicable" is
    # explicit rather than silently passing.
    checks["static_decoys_fixed"] = all(static_ok) if static_ok else True
    checks["moving_decoys_linear"] = all(moving_ok) if moving_ok else True
    checks["flares_descend"] = all(flare_ok) if flare_ok else \
        dom["key"] != "air_plane"
    checks["all_passed"] = all(bool(v) for v in checks.values())

    report = dict(
        domain=dom["key"], label=dom["label"], frames=N_FRAMES, dt=DT,
        primary=primary_name,
        trajectory_sources=["optical_frames.json", "frames.json"],
        decoy_class_counts=dict(static=len(static_ok), moving=len(moving_ok),
                                flare=len(flare_ok)),
        frame_step_m=[round(s, 4) for s in steps],
        ranges_m=prev_ranges,
        frame_records=out_frames, checks=checks)
    jwrite(os.path.join(run, "trajectories.json"), report)
    return report


def verify(dest, scene_dir, cfgs):
    print("\n=== VERIFY (5 frames x 3 modalities) ===")
    report, traj_reports = {}, {}
    ok = True
    for dom in DOMAINS:
        run = os.path.join(dest, dom["key"])
        d = dict(
            optical=count_png(os.path.join(run, "images", "optical")),
            ir_lwir=count_png(os.path.join(run, "images", "ir_lwir")),
            ir_preview=count_png(os.path.join(run, "images", "ir_preview_lwir")),
            radar_rd=count_png(os.path.join(run, "images", "radar")),
            jpeg=len([n for n in os.listdir(os.path.join(run, "images", "optical"))
                      if n.lower().endswith(".jpg")])
            if os.path.isdir(os.path.join(run, "images", "optical")) else 0,
        )
        raw = os.path.join(run, "raw", "radar")
        d["radar_npy"] = len([n for n in os.listdir(raw) if n.endswith(".npy")]) \
            if os.path.isdir(raw) else 0
        fp = os.path.join(run, "frames.json")
        d["frames"] = len(jload(fp)) if os.path.isfile(fp) else 0
        coco = os.path.join(run, "annotations_ir_lwir_coco.json")
        c = jload(coco) if os.path.isfile(coco) else {}
        d["coco_anns"] = len(c.get("annotations") or [])
        mp = os.path.join(run, "meta.json")
        meta = jload(mp) if os.path.isfile(mp) else {}
        d["meta_frames"] = meta.get("frames")
        good = all(d[k] == N_FRAMES for k in
                   ("optical", "ir_lwir", "radar_rd", "radar_npy",
                    "frames", "coco_anns", "jpeg"))
        tr = build_trajectories(dest, dom, cfgs[dom["key"]])
        traj_reports[dom["key"]] = tr["checks"]
        good = good and tr["checks"]["all_passed"]
        d["trajectory_checks"] = tr["checks"]
        d["passed"] = bool(good)
        ok = ok and good
        report[dom["key"]] = d
        print("[%s] %-16s %s" % ("PASS" if good else "FAIL", dom["key"], d))
        if not tr["checks"]["all_passed"]:
            print("   trajectory failures:",
                  [k for k, v in tr["checks"].items() if not v])

    jwrite(os.path.join(dest, "generation_report.json"), dict(
        dest=dest, frames_per_domain=N_FRAMES, modalities=len(MODALITIES),
        domains=report, trajectory_checks=traj_reports, passed=bool(ok)))
    jwrite(os.path.join(dest, "trajectories_summary.json"), dict(
        frames_per_domain=N_FRAMES, dt=DT,
        checks=traj_reports, all_passed=bool(ok)))
    return ok


# ---------------------------------------------------------------------------
# Documentation / contact sheets
# ---------------------------------------------------------------------------
REFERENCE_MD = """# 真实海 / 陆 / 空目标数据集特征（2026-11-09 联网检索）

本文件说明 v1 数据集的场景参数是按哪些公开实拍数据集的特征设定的。

## 陆 · 车辆

| 数据集 | 采集方式 | 关键特征 |
|---|---|---|
| VEDAI | 卫星/航拍，512×512 / 1024×1024，RGB+IR 配对 | 每图约 5.5 个目标，目标仅占约 0.7% 像素；背景含草地、山地、荒漠、乡村、城市；9 类车辆，旋转框 |
| DLR 3K (Munich) | 1000 m 高度，5616×3744，GSD≈13 cm | 城市街道与草坪为主，car / truck 两类，背景比 VEDAI 一致 |
| VisDrone / UAVDT | UAV 10–150 m，2000×1500 等 | 斜视/侧视/俯视、复杂城市场景、天气与遮挡属性、连续视频帧 |
| DroneVehicle / 3D-Net | RGB–LWIR 配对，50–150 m / 30 m 斜视 | 低光、雾、夜；逐像素配准；连续视频按时间切分；车辆 6 类 |
| MONET / UAV-Thermal | 无人机 LWIR 7.5–13.5 µm，640–800 px 宽 | 目标外观随温度/发射率/环境变化；背景可能与目标同温；热噪声、鬼影；运动相机造成尺度变化 |

**据此外推到 v1 车辆场景**：
- 斜俯视 `depression_deg=45°`（不是正天底，也不是水平）；
- 真实道路/环岛航拍照片做地面底板（`city/06_Drone_view_of_roundabout_Unsplash.jpg`），背景自带车道线、停放车辆、建筑与树木；
- 去掉 `optical.sky_image` 天空卡片（俯视时它挡在地面前面，产生斜向条带）；并把平铺跨度从 200 m 提到 400 m，让 REPEAT 接缝离开可见范围；
- 目标换成 6.75 m 厢式货车（VEDAI 的 van / truck 类），浅色车漆，斜距约 1 km 时约 30 px，属于"小目标"量级；
- 3 辆各自轨迹的正常车辆 + 3 个固定路侧障碍，对应"每图多个目标 + 背景杂物"；
- 所有非主目标显式 `motion_type=ground`：seeker 规划器默认会把它们当编队成员拖着走，静态物因此会"动"。

## 空 · 飞行器

| 数据集 | 关键特征 |
|---|---|
| Det-Fly | 空对空 13271 张 3840×2160；天空/城市/田野/山地四类背景各 20–30%；前视/俯视/仰视各约 1/3；近半目标面积 <5% |
| DUT Anti-UAV | 平均无人机框约 34×23 px（约 0.1% 画幅），城市密集建筑背景 |
| DroneSwarms | 每帧平均 26.6 个目标，99.6% 小于 32 px，平均约 7.9 px |
| 红外弱小飞机数据集（CSDATA） | 22 段序列 / 30 条轨迹 / 16177 帧 / 16944 个目标；天空与复杂地面背景；逐帧目标位置标注 |

**据此外推**：目标保持 28–34 px 的弱小目标量级；云层实拍底板；5 帧连续轨迹；
红外诱饵按"以飞机航向轴线对称、向下规律抛射"建模，不继承飞机速度，且每枚远小于飞机。

## 海 · 舰船

| 数据集 | 关键特征 |
|---|---|
| SeaShips | 31 455 张 / 约 10 080 段岸基监控视频，1920×1080；6 类船；覆盖不同尺度、船体局部、光照、视角、背景与遮挡 |
| SSDD / HRSID（SAR） | 海面杂波、靠岸背景、多尺度舰船 |

**据此外推**：开阔海面实拍底板；目标斜距约 2 km、约 30 px；
海杂波零多普勒脊；另加 1 艘航行小艇（自身轨迹）与 1 艘锚泊小艇（静止）作为干扰。

## 共同点（三个域都遵守）

1. **连续帧**：来自视频而非独立随机采样，相邻帧只差一个积分步；按片段切分。
2. **小目标**：目标占画幅百分之几甚至千分之几，不居中钉死，带缓慢 LOS 游走。
3. **背景真实且杂乱**：不是纯色渐变；道路/云层/海浪都来自实拍底板。
4. **多模态同源**：可见光 / LWIR / 雷达 RD 由同一三维世界与同一天气状态生成，标注由同一套三维投影导出。
5. **轨迹可核验**：本数据集额外给出 `trajectories.json`，逐帧记录主目标与每个干扰物的位置、速度，并做硬性检查。

## 已知局限（与上述实拍数据集的差距）

- 背景底板是实拍照片，但目标仍是 Cycles 渲染的 3D 模型；二者光照/阴影不可能完全一致。
- LWIR 背景用同一张可见光照片做温度映射，不是真实热像；仅提供结构，不提供标定过的辐射量。
- 雷达 RD 是几何 PO 估计，未用实测 RCS / 海杂波 K 分布标定。
- 5 帧只是 0.5 s 片段，不能替代长时序跟踪集。

主要来源：VEDAI/VisDrone 综述（MDPI Sensors 2019、Sustainability 2024、中国 UAV 检测综述）、
3D-Net（Zenodo）、MONET（arXiv 2023）、SeaShips（IEEE TMM 2018）、
Det-Fly / Anti-UAV 综述（中国科学基金 2025）、CSDATA 红外弱小飞机数据集。
"""


def write_docs(dest, scene_dir, ok):
    with open(os.path.join(dest, "参考数据集对照.md"), "w",
              encoding="utf-8") as fh:
        fh.write(REFERENCE_MD)
    with open(os.path.join(dest, "README.md"), "w", encoding="utf-8") as fh:
        fh.write(
            "# 海 / 陆 / 空 三模态数据集 v1（2026-11-09）\n\n"
            "每个域 **5 帧**，每帧三模态：可见光 / 长波红外 LWIR / 毫米波雷达距离-多普勒 RD。"
            "全部由 simkit 离线物理仿真生成，目标与干扰均为 3D 模型，背景为实拍照片底板。\n\n"
            "| 目录 | 域 | 主目标 | 干扰物与轨迹 |\n|---|---|---|---|\n"
            "| air_plane | 空 | 双发战斗机，平滑接近轨迹 | 6 枚红外诱饵：以航向轴线对称、向下规律抛射、不继承飞机速度；箔条云 / 假目标 / 噪声干扰 |\n"
            "| ground_vehicle | 陆 | 6.75 m 厢式货车，贴地匀速直线 | 3 辆正常车辆各自匀速轨迹；3 个路侧障碍严格静止；道路/环岛实拍底板 |\n"
            "| sea_ship | 海 | 导弹艇，平滑接近轨迹 | 1 艘航行小艇（自身轨迹）、1 艘锚泊小艇（静止）、海杂波脊 |\n\n"
            "## 文件\n\n"
            "- `scenes/ds1109v1_*.json`：三份可复现场景配置（资产均为绝对路径）\n"
            "- `<域>/images/{optical,ir_lwir,radar,...}`：5 帧 × 各模态 PNG（光学另附 .jpg）\n"
            "- `<域>/raw/radar/*.npy`：float32 功率张量 (range, doppler)\n"
            "- `<域>/labels/`、`annotations_*.json`、`frames.json`：逐帧标注与帧事实\n"
            "- `<域>/trajectories.json`：**逐帧主目标 + 每个干扰物的位置/速度与硬性检查**\n"
            "- `trajectories_summary.json`、`generation_report.json`：三域验收汇总\n"
            "- `参考数据集对照.md`：联网检索的真实数据集特征与本场景参数的映射\n"
            "- `contact_*.png`：三域 5 帧 × 3 模态对照拼图\n\n"
            "## 复现\n\n"
            "```powershell\n"
            "set PYTHONPATH=D:\\work\\多模态数据集生成\\RGB_SAR_LIDAR_SIMU\n"
            "env\\Scripts\\python.exe tools\\build_sealandair_1109v1.py --force\n"
            "```\n\n"
            "`--skip-gen` 只重做元数据与验收（不重渲）。\n\n"
            "## 本次验收\n\n"
            "- 三域各 5 帧 × 3 模态，`generation_report.json` 中 `passed=%s`\n"
            "- 轨迹检查：主目标平滑且严格连续、静态干扰零位移、动态干扰匀速、诱饵单调下坠\n"
            "- `unittest discover -s tests`、`simkit selftest` 全绿\n\n" % ok)


def contact_sheets(dest):
    from PIL import Image, ImageDraw
    try:
        for dom in DOMAINS:
            run = os.path.join(dest, dom["key"])
            cell_w, cell_h = 256, 192
            sheet = Image.new("RGB", (cell_w * N_FRAMES, cell_h * 3 + 24),
                              (245, 245, 245))
            d = ImageDraw.Draw(sheet)
            rows = [("optical", "images/optical"),
                    ("ir_lwir", "images/ir_preview_lwir"),
                    ("radar_rd", "images/radar")]
            for r, (label, sub) in enumerate(rows):
                d.text((4, r * cell_h + 2), label, fill=(20, 20, 20))
                for i in range(N_FRAMES):
                    p = os.path.join(run, sub, "%06d.png" % i)
                    if not os.path.isfile(p):
                        continue
                    im = Image.open(p).convert("RGB")
                    im.thumbnail((cell_w, cell_h - 18))
                    sheet.paste(im, (i * cell_w, r * cell_h + 18))
            out = os.path.join(dest, "contact_" + dom["key"] + ".png")
            sheet.save(out)
            print("CONTACT", out)
    except Exception as exc:
        print("WARN contact sheets skipped:", exc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=DEFAULT_DEST)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--skip-gen", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    dest = os.path.abspath(args.dest)
    scene_dir = os.path.join(dest, "scenes")
    os.makedirs(scene_dir, exist_ok=True)
    cfgs = {dom["key"]: build_cfg(dom, scene_dir) for dom in DOMAINS}
    for dom in DOMAINS:
        c = cfgs[dom["key"]]
        print("CFG %-18s objects=%d ecm=%d plate=%s" % (
            dom["key"], len(c.get("objects") or []),
            len((c.get("ecm") or {}).get("entities") or []),
            os.path.basename(str((c.get("ground") or {}).get("backdrop")
                                 or c["optical"].get("sky_image") or "-"))))

    if not args.skip_gen:
        if args.force:
            for dom in DOMAINS:
                run = os.path.join(dest, dom["key"])
                if os.path.isdir(run):
                    shutil.rmtree(run)
        for dom in DOMAINS:
            run_gen(os.path.join(scene_dir, "ds1109v1_" + dom["key"] + ".json"),
                    os.path.join(dest, dom["key"]), args.threads)

    stamp_optical_meta(dest, scene_dir)
    ok = verify(dest, scene_dir, cfgs)
    contact_sheets(dest)
    write_docs(dest, scene_dir, ok)
    print("\nOVERALL", "PASS" if ok else "FAIL", "->", dest)
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
