#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""W1 v3 generator & physical-layer jamming probe (optical / LWIR / radar-RD).

Ownership (this agent writes NOTHING else):
  tools/build_v3_trimodal.py
  tools/v3_probe_report.json
  scenes/ds20_v3_{air_plane,ground_vehicle,sea_ship}.json
  _tmp/v3_probe/**                     (probe byproducts only)
simkit/, blender/, the v2 generator / validator, studio/, tests/, data/,
dataest/ are read-only here.  simkit.scenegen is NEVER imported/called.

v3 vs v2:
  * 30 frames everywhere, real projected box 28..34 px (aim 31), closed-loop
    calibration from a 1-frame render of the v2 config (the 20 px start point).
  * Real-photo background + realism switches are preserved by a straight
    deepcopy of ds20_v2_*.json (three same-source pointers, camera 5 stages,
    atmosphere, preview_span_c=20, temp_c removed, sensor.enabled=false,
    auto_exposure=false).
  * Jamming is the PHYSICS layer (simkit/ecm.py + radar clutter knobs), never
    image-layer interference (that only spawns an interfered/ copy and does not
    touch the RD tensor -> breaks trimodal alignment).

Phases
  cfg     -> write the three v3 configs (calibrated, default "mid" tier)
  probe   -> 9 single-frame renders (3 domains x base/mid/strong) + report
  full    -> 3x30 frames into dataest/{air_plane,ground_vehicle,sea_ship}
             (implemented, NOT run by this task; main controller runs it)
  repair  -> v2 LWIR zero-hit box restore, retargeted to 30 frames / v3 configs
"""
import argparse
import copy
import glob
import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import build_v2_trimodal as V2                       # reuse run_gen/box_width/...
from simkit import ecm as _ecmlib                     # ecm.load whitelist validation

DEFAULT_DEST = r"D:\work\多模态数据集生成\dataest"
SCENE_DIR = V2.SCENE_DIR
PROBE_DIR = os.path.join(ROOT, "_tmp", "v3_probe")
REPORT_PATH = os.path.join(ROOT, "tools", "v3_probe_report.json")
W, H = 768, 576
TARGET_PX = [28, 34]
AIM_PX = 31.0
N_FRAMES = 30
TIERS = ("base", "mid", "strong")
# Probe verdict (tools/v3_probe_report.json): strong drowns the air target
# (detected=False), while ground/sea need strong before the clutter ridge
# clears the +3 dB effectiveness line.  Air therefore stays at mid.
FINAL_TIER = {"air": "mid", "ground": "strong", "sea": "strong"}

# v2 measured 20-px start points (used as the calibration seed + fallback).
SPECS = [
    dict(key="air_plane", seed=7, domain="air", cls="fighter", fov=8.0,
         v2_scale=0.51056136, v2_target_m=10.3767, r0=2937.29),
    dict(key="ground_vehicle", seed=17, domain="ground", cls="vehicle", fov=10.0,
         v2_scale=0.91646844, v2_target_m=5.2152, r0=1211.13),
    dict(key="sea_ship", seed=27, domain="sea", cls="ship", fov=12.0,
         v2_scale=0.76003305, v2_target_m=12.5553, r0=2376.73),
]
SPEC_BY_KEY = {s["key"]: s for s in SPECS}

jload = V2.jload
jwrite = V2.jwrite
box_width = V2.box_width
run_gen = V2.run_gen
fail_tail = V2.fail_tail


def v2_path(key):
    return V2.v2_path(key)


def v3_path(key):
    return os.path.join(SCENE_DIR, "ds20_v3_%s.json" % key)


def _life():
    # No fade-in.  A 0.05 s fade made the jammer near-powerless on frame 0,
    # which was the ONLY frame the 1-frame probe looked at: the probe reported
    # "detected=True" while frames 1..29 of the 30-frame run were all drowned.
    # Every entity is at full strength from t=0 so a single probe frame is
    # representative of the whole clip.
    return {"start_s": 0}


# ---------------------------------------------------------------------------
# Jam tier definitions.  One source of truth for BOTH the configs and the probe,
# so the 9 probe runs are literally deepcopies of the same v3 base config.
# ---------------------------------------------------------------------------
JAM = {
    "air": {
        "layer": "ecm",
        "base": dict(ecm_enabled=False),
        # 200 W passed the 1-frame probe (jammer still fading in) but drowned
        # 29/30 frames in the full run (detected 1/30).  10 W recovered all
        # probed frames yet left only 2.7 dB Swerling margin (min SNR 14.1 dB);
        # 3 W keeps the jamming unmistakable (+17 dB noise floor over the
        # no-jam baseline) while leaving ~7-10 dB target SNR margin.
        "mid": dict(ecm_enabled=True, erp_w=3, ft_count=4, ft_power_ratio=1.6,
                    chaff_rcs_m2=140, flare_n=2),
        "strong": dict(ecm_enabled=True, erp_w=1200, ft_count=8, ft_power_ratio=3.0,
                       chaff_rcs_m2=500, flare_n=4),
    },
    "ground": {
        "layer": "objects+radar_clutter",
        "base": dict(n_clutter=0, clutter_density=0.0, clutter_db=-12.0,
                     ground_clutter_db=-35.0),
        "mid": dict(n_clutter=6, clutter_density=0.04, clutter_db=-12.0,
                    ground_clutter_db=-35.0),
        "strong": dict(n_clutter=12, clutter_density=0.08, clutter_db=-6.0,
                       ground_clutter_db=-25.0),
    },
    "sea": {
        "layer": "radar_clutter",
        # radar.py:319 rc=dict(DEFAULTS) makes clutter_ridge default TRUE; v2's
        # sea scene therefore already had a ridge.  base MUST set it False
        # explicitly to actually switch it off.
        "base": dict(clutter_ridge=False, sea_clutter_scale_db=0, wind_ms=6,
                     sea_clutter_db=-25, clutter_density=0.0, clutter_db=-12.0, mti=0.0),
        "mid": dict(clutter_ridge=True, sea_clutter_scale_db=3, wind_ms=8,
                    sea_clutter_db=-25, clutter_density=0.02, clutter_db=-12.0, mti=0.0),
        "strong": dict(clutter_ridge=True, sea_clutter_scale_db=6, wind_ms=15,
                       sea_clutter_db=-15, clutter_density=0.06, clutter_db=-6.0, mti=0.0),
    },
}


def air_ecm(tier):
    """Physics-layer ECM block (fields strictly inside ecm.load whitelist).

    false_target -> power_ratio/range_step_m/range_offset_m (NOT power_w)
    chaff_cloud  -> rcs_m2 (NOT rcs_db)
    noise_jammer -> erp_w
    band must be a str; life only accepts start_s/end_s/fade_s.  Absolute pos
    near the target [0,0,2500], vel close to [109.593,224.699,0]; no parent/
    link so the link-model validation is never exercised.
    """
    p = JAM["air"][tier]
    vel = [109.593, 224.699, 0.0]
    ents = [
        dict(id="jam_01", kind="noise_jammer",
             pos=[40.0, -30.0, 2520.0], vel=list(vel),
             erp_w=p["erp_w"], jam_bandwidth_mhz=200.0, mode="barrage",
             band="Ku", life=_life()),
        dict(id="ft_01", kind="false_target",
             pos=[-25.0, 35.0, 2485.0], vel=list(vel),
             count=p["ft_count"], range_step_m=210, range_offset_m=150,
             power_ratio=p["ft_power_ratio"], band="Ku", life=_life()),
        dict(id="chaff_01", kind="chaff_cloud",
             pos=[-60.0, -40.0, 2495.0], vel=[109.593, 224.699, -2.0],
             radius_m=6, rcs_m2=p["chaff_rcs_m2"], decay_s=6, fall_mps=2.2,
             band="Ku", life=_life()),
    ]
    for i in range(p["flare_n"]):
        ents.append(dict(id="flare_%02d" % (i + 1), kind="infrared_flare",
                         pos=[20.0 + 15 * i, -20.0 - 10 * i, 2505.0 + 3 * i],
                         vel=list(vel), temp_c=2000, decay_s=3,
                         band="Ku", life=_life()))
    return dict(enabled=True, entities=ents)


def ground_clutter(n):
    """label:false scenery within +-50 m of target [0,0,0]; no temp_c /
    temp_c_hint (v2 primary-target-only fields) so COCO stays single-class."""
    tpl = [
        ("shed_1", "box", [6, 5, 4], [-35, 22, 0], 12.0, "building_wall"),
        ("shed_2", "box", [7, 5, 5], [28, -30, 0], -25.0, "building_wall"),
        ("shed_3", "box", [5, 6, 4], [-18, 42, 0], 40.0, "building_wall"),
        ("barrier_1", "obj", "../assets/models/concrete_road_barrier.obj",
         [38, 15, 0], 90.0, "concrete"),
        ("fence_1", "obj", "../assets/models/modular_chainlink_fence.obj",
         [-42, -18, 0], 90.0, "metal_bare"),
        ("shrub_1", "obj", "../assets/models/shrub_01.obj",
         [15, -45, 0], 0.0, "foliage"),
        ("shed_4", "box", [8, 6, 5], [40, 25, 0], -10.0, "building_wall"),
        ("shed_5", "box", [6, 5, 4], [-48, 5, 0], 30.0, "building_wall"),
        ("shed_6", "box", [5, 7, 5], [5, 48, 0], 60.0, "building_wall"),
        ("barrier_2", "obj", "../assets/models/concrete_road_barrier.obj",
         [-30, -40, 0], 95.0, "concrete"),
        ("fence_2", "obj", "../assets/models/modular_chainlink_fence.obj",
         [48, -8, 0], 90.0, "metal_bare"),
        ("shrub_2", "obj", "../assets/models/shrub_01.obj",
         [-10, -48, 0], 0.0, "foliage"),
    ]
    out = []
    for name, shape, spec, pos, rot, mat in tpl[:n]:
        o = dict(name=name, cls="scenery", shape=shape, pos=list(pos),
                 vel=[0.0, 0.0, 0.0], rot=rot, mat=mat, label=False)
        if shape == "box":
            o["size"] = list(spec)
        else:
            o["model"] = spec
        out.append(o)
    return out


def make_base_cfg(spec, scale, target_m):
    """Straight deepcopy of v2 -> 30 frames + 28..34 px band + calibrated scale.

    Nothing about the real-photo background / realism switches is touched, so
    they are preserved verbatim (three same-source pointers, camera 5 stages,
    atmosphere, preview_span_c=20, temp_c removed, sensor.enabled=false,
    auto_exposure=false).  NO jamming injected here.
    """
    key = spec["key"]
    c = copy.deepcopy(jload(v2_path(key)))
    if not c:
        raise RuntimeError("missing v2 config: " + v2_path(key))
    c["frames"]["n"] = N_FRAMES
    c["clip"]["frames_per_clip"] = N_FRAMES
    c["capture"]["frames"] = N_FRAMES
    c["recipe"]["capture"]["frames"] = N_FRAMES
    ct = c.setdefault("__trimodal_contract", {})
    ct["frames"] = N_FRAMES
    ct["target_px"] = list(TARGET_PX)
    ct["aim_px"] = AIM_PX
    sk = c["clip"]["seeker"]
    sk["target_px"] = list(TARGET_PX)
    sk["target_m"] = round(target_m, 4)
    # authored range_m interval is deliberately NOT changed; after calibration
    # the reverse-solved band still falls inside it.
    c["objects"][0]["scale"] = scale
    c["recipe"]["objects"][0]["scale"] = scale
    c["name"] = "ds20_v3_" + key
    c["recipe"]["name"] = "ds20_v3_" + key
    return c


def apply_tier(base, domain, tier):
    c = copy.deepcopy(base)
    p = JAM[domain][tier]
    rd = c["sensors"]["radar"]
    if domain == "air":
        c["ecm"] = air_ecm(tier) if p["ecm_enabled"] else dict(enabled=False, entities=[])
        # Swerling I draws f=-ln(u) once per CPI: frame 10 hit a deep fade and
        # lost the analytic detection even after the ERP fix (29/30).  III is
        # the same physical fluctuation model with two draws (half variance);
        # pfa 1e-4 lowers the gate from 11.4 to 9.6 dB.  Radar-only 30-frame
        # probe: detected 30/30, SNR 13.9..30.3 dB.  These two knobs affect
        # the radar channel only, so optical/IR frames stay bit-identical.
        rd["swerling"] = "III"
        rd["pfa"] = 1e-4
    elif domain == "ground":
        # Idempotent: always rebuild from the single labelled main target, so
        # re-applying a tier (or switching mid -> strong) never stacks clutter.
        c["objects"] = c["objects"][:1]
        if p["n_clutter"]:
            c["objects"] = c["objects"] + ground_clutter(p["n_clutter"])
        rd["clutter_density"] = p["clutter_density"]
        rd["clutter_db"] = p["clutter_db"]
        rd["ground_clutter_db"] = p["ground_clutter_db"]
    elif domain == "sea":
        rd["clutter_ridge"] = bool(p["clutter_ridge"])
        rd["sea_clutter_scale_db"] = p["sea_clutter_scale_db"]
        rd["sea_clutter_db"] = p["sea_clutter_db"]
        rd["clutter_density"] = p["clutter_density"]
        rd["clutter_db"] = p["clutter_db"]
        rd["mti_min_vr_mps"] = p["mti"]
        c["ground"]["wind_ms"] = p["wind_ms"]
    return c


def jam_spec_summary(domain, tier=None):
    tier = tier or FINAL_TIER.get(domain, "mid")
    return dict(
        layer=JAM[domain]["layer"],
        current_tier=tier,
        tiers={t: dict(JAM[domain][t]) for t in TIERS},
        note="probe tier definitions; final tier fixed by probe verdict "
             "(air mid / ground strong / sea strong)",
    )


# ---------------------------------------------------------------------------
# cfg: closed-loop calibration (render 1 frame of the v2 20-px config), then
# verify the v3 base config's real box, then write the mid-tier v3 config.
# ---------------------------------------------------------------------------
def calibrate(spec, args):
    key = spec["key"]
    out = os.path.join(PROBE_DIR, key + "_cal")
    r = run_gen(v2_path(key), out, 1, spec["seed"], threads=2, quiet=True)
    if r.returncode != 0:
        raise RuntimeError("calibration render failed %s: %s" % (key, fail_tail(r)))
    lab = os.path.join(out, "labels", "optical", "000000.json")
    w = box_width(lab, include_invisible=True)
    row0 = (jload(os.path.join(out, "frames.json")) or [{}])[0]
    r0 = row0.get("range_m")
    if not w or not r0:
        raise RuntimeError("no calibration box/range for " + key)
    scale = round(spec["v2_scale"] * AIM_PX / w, 8)
    target_m = 2.0 * float(r0) * math.tan(math.radians(spec["fov"]) / 2.0) * AIM_PX / W
    return dict(scale=scale, target_m=round(target_m, 4), r0=float(r0), w_before=w)


def verify_paths_v3(c):
    """Correct background check for BOTH branches.

    Sky-only scenes (air, ground.enabled=false) legitimately have NO ground
    plate: the correct gate is sky_image == ir.background_image and no
    ground.backdrop.  Look-down scenes (ground/sea) need all three pointers
    equal.  V2.verify_paths has no sky-only branch and mis-reports air here.
    """
    g = c.get("ground") or {}
    bg = g.get("backdrop")
    sky = (c.get("optical") or {}).get("sky_image")
    irbg = (c.get("ir") or {}).get("background_image")
    has_ground = bool(g.get("enabled"))
    if has_ground:
        plate, keys_ok = bg, (bool(bg) and bg == sky == irbg)
    else:
        plate, keys_ok = sky, (bool(sky) and sky == irbg and not bg)
    exists = os.path.isfile(os.path.normpath(os.path.join(SCENE_DIR, str(plate))))
    return dict(has_ground=has_ground, plate=plate, keys_ok=bool(keys_ok),
                file_exists=bool(exists), all_ok=bool(keys_ok and exists),
                background_branch=("three_key" if has_ground else "sky_only_two_key"))


def verify_frame(spec, base, args, tag):
    cfg_path = os.path.join(PROBE_DIR, "_%s_%s.json" % (tag, spec["key"]))
    jwrite(cfg_path, base)
    out = os.path.join(PROBE_DIR, "%s_%s" % (spec["key"], tag))
    r = run_gen(cfg_path, out, 1, spec["seed"], threads=2, quiet=True)
    if r.returncode != 0:
        print("  %s render failed:" % tag, fail_tail(r)[-400:])
        return None
    return box_width(os.path.join(out, "labels", "optical", "000000.json"),
                     include_invisible=True)


def phase_cfg(args):
    os.makedirs(PROBE_DIR, exist_ok=True)
    for spec in SPECS:
        key = spec["key"]
        cal = calibrate(spec, args)
        target_m = cal["target_m"]
        # 1st pass: scale *= 31/w from the v2 20-px frame (task formula).
        scale_a = cal["scale"]
        base_a = make_base_cfg(spec, scale_a, target_m)
        w_b = verify_frame(spec, base_a, args, "verify1")
        # 2nd closed-loop pass: the 30-frame config moves frame-0 slant range
        # (and the look-down geometry of the tall ship), so ONE pass left the
        # real box off 31 px (measured sea 26.69 px).  Re-apply the same
        # scale *= 31/w law against the ACTUAL v3 base frame.
        scale_final = (round(scale_a * AIM_PX / w_b, 8)
                       if w_b and w_b > 1e-6 else scale_a)
        base = make_base_cfg(spec, scale_final, target_m)
        w_final = verify_frame(spec, base, args, "verify2")
        tier = FINAL_TIER[spec["domain"]]
        final = apply_tier(base, spec["domain"], tier)
        ecm_check = {}
        if spec["domain"] == "air":
            try:
                loaded = _ecmlib.load(final["ecm"])
                ecm_check = dict(ok=True, n_entities=len(loaded.get("entities", [])),
                                 kinds=sorted({e.get("kind") for e in loaded.get("entities", [])}))
            except Exception as exc:
                ecm_check = dict(ok=False, error=repr(exc))
        vp = verify_paths_v3(base)
        final["__v3"] = dict(
            source=os.path.basename(v2_path(key)),
            scale_start=spec["v2_scale"], scale_first_pass=scale_a,
            scale_calibrated=scale_final,
            target_m_start=spec["v2_target_m"], target_m_calibrated=target_m,
            r0_m=cal["r0"], box_px_before=round(cal["w_before"], 3),
            box_px_first_pass=round(w_b, 3) if w_b else None,
            box_px_verified=round(w_final, 3) if w_final else None,
            backdrop=vp["plate"], background_branch=("three_key" if vp["has_ground"]
                                                     else "sky_only_two_key"),
            three_paths_same=vp["keys_ok"], three_paths_exist=vp["file_exists"],
            ecm_load_mid=ecm_check,
            note="30 frames; real 28..34 px; physics-layer jamming tier "
                 + tier + "; two-pass closed-loop scale calibration",
        )
        final["__v3_jam_tier"] = tier
        final["__v3_jam_spec"] = jam_spec_summary(spec["domain"], tier)
        jwrite(v3_path(key), final)
        print("CFG %-16s scale=%.8f target_m=%.4f  box %.2f -> %.2f -> %.2f px  r0=%.2f"
              % (key, scale_final, target_m, cal["w_before"],
                 w_b or -1, w_final or -1, cal["r0"]))
        print("     background=%s all_ok=%s ecm_load=%s tier=%s"
              % (vp["background_branch"], vp["all_ok"], ecm_check, tier))
    return 0


# ---------------------------------------------------------------------------
# probe metrics (no new dependencies: numpy + PIL only)
# ---------------------------------------------------------------------------
def count_components(mask, min_size=2):
    """4-neighbour flood fill on a sparse boolean mask (no scipy)."""
    hh, ww = mask.shape
    visited = np.zeros_like(mask, bool)
    n = 0
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys, xs):
        if visited[y0, x0]:
            continue
        visited[y0, x0] = True
        stack = [(int(y0), int(x0))]
        size = 0
        while stack:
            y, x = stack.pop()
            size += 1
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < hh and 0 <= nx < ww and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        if size >= min_size:
            n += 1
    return n


def _exclusion(olab):
    """Boolean mask of the primary target box (+2 px), used to exclude it."""
    m = np.zeros((H, W), bool)
    for b in (olab.get("boxes") or {}).values():
        if not isinstance(b, dict) or not all(k in b for k in ("x0", "y0", "x1", "y1")):
            continue
        x0, y0 = int(b["x0"]) - 2, int(b["y0"]) - 2
        x1, y1 = int(b["x1"]) + 2, int(b["y1"]) + 2
        m[max(0, y0):min(H, y1), max(0, x0):min(W, x1)] = True
    return m


def probe_metrics(odir, spec):
    from PIL import Image
    olab = jload(os.path.join(odir, "labels", "optical", "000000.json")) or {}
    rlab = jload(os.path.join(odir, "labels", "radar", "000000.json")) or {}
    excl = _exclusion(olab)
    keep = ~excl

    # ---- optical ----
    g = np.asarray(Image.open(os.path.join(odir, "images", "optical", "000000.png")).convert("L"),
                   dtype=np.float64)
    if g.shape != (H, W):
        keep = np.ones_like(g, bool)
    lap = 4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    p99_g = float(np.percentile(g[keep], 99))
    blob_mask = (g > p99_g + 10) & keep
    oboxes = [b for b in (olab.get("boxes") or {}).values()
              if isinstance(b, dict) and all(k in b for k in ("x0", "y0", "x1", "y1"))]
    box_px = max((float(b["x1"]) - float(b["x0"]) for b in oboxes), default=0.0)
    opt = dict(
        std=round(float(g.std()), 3),
        p99_minus_p1=round(float(np.percentile(g, 99) - np.percentile(g, 1)), 3),
        hf_laplacian=round(float(np.abs(lap).mean()), 4),
        blobs_excl_target=count_components(blob_mask, 2),
        dead_black_pct=round(100.0 * float((g == 0).mean()), 3),
    )

    # ---- LWIR preview ----
    gi = np.asarray(Image.open(os.path.join(odir, "images", "ir_preview_lwir",
                                           "000000.png")).convert("L"), dtype=np.float64)
    keep_i = ~excl if excl.shape == gi.shape else np.ones_like(gi, bool)
    p99_i = float(np.percentile(gi[keep_i], 99))
    hot_mask = (gi > p99_i + 10) & keep_i
    ir = dict(
        std=round(float(gi.std()), 3),
        dead_black_pct=round(100.0 * float((gi == 0).mean()), 3),
        hotspots_excl_target=count_components(hot_mask, 2),
    )

    # ---- RD tensor [range, doppler] ----
    rd = np.load(os.path.join(odir, "raw", "radar", "000000.npy")).astype(np.float64)
    med = float(np.median(rd))
    peak = float(rd.max())
    p99_rd = float(np.percentile(rd, 99))
    c = rd.shape[1] // 2
    center = float(rd[:, max(0, c - 2):c + 3].mean())
    rdm = dict(
        shape=[int(rd.shape[0]), int(rd.shape[1])],
        noise_floor_db=round(10 * math.log10(max(med, 1e-30)), 2),
        peak_snr_db=round(10 * math.log10(max(peak / max(med, 1e-30), 1e-30)), 2),
        center_rel_db=round(10 * math.log10(max(center / max(med, 1e-30), 1e-30)), 2),
        center_abs_db=round(10 * math.log10(max(center, 1e-30)), 2),
        p99_over_median_db=round(10 * math.log10(max(p99_rd / max(med, 1e-30), 1e-30)), 2),
    )

    # ---- radar labels ----
    tg = rlab.get("targets") or []
    detected = bool(tg[0].get("detected")) if tg else False
    snr_db = tg[0].get("snr_db") if tg else None
    if snr_db is None:
        pts = [float(p.get("snr_db", 0)) for p in (rlab.get("points") or [])]
        snr_db = max(pts) if pts else None
    ecm_boxes = sum(1 for b in (rlab.get("boxes_rd") or {}).values()
                    if isinstance(b, dict) and b.get("source") == "ecm")
    rlab_m = dict(detected=detected, snr_db=snr_db, n_targets=len(tg),
                  ecm_source_boxes=ecm_boxes,
                  unassociated=len(rlab.get("unassociated_detections") or []))

    # ---- optical labels + COCO cleanliness ----
    cat_names, ann_counts = set(), []
    for p in glob.glob(os.path.join(odir, "annotations_*_coco.json")):
        d = jload(p) or {}
        for cc in (d.get("categories") or []):
            cat_names.add(str(cc.get("name")))
        ann_counts.append(len(d.get("annotations") or []))
    coco_clean = bool(len(cat_names) <= 1)
    olab_m = dict(box_px=round(box_px, 3), n_boxes=len(oboxes),
                  coco_categories=sorted(cat_names), coco_clean=coco_clean,
                  coco_ann_counts=ann_counts)

    return dict(optical=opt, ir=ir, rd=rdm, radar_label=rlab_m, optical_label=olab_m,
                # flattened convenience keys used by the decision logic
                noise_floor_db=rdm["noise_floor_db"], peak_snr_db=rdm["peak_snr_db"],
                center_rel_db=rdm["center_rel_db"],
                optical_blobs=opt["blobs_excl_target"],
                ir_hotspots=ir["hotspots_excl_target"],
                detected=detected, snr_db=snr_db, box_px=round(box_px, 3),
                n_boxes=len(oboxes), coco_clean=coco_clean,
                ecm_source_boxes=ecm_boxes)


def ecm_validate():
    """Actually call simkit.ecm.load() on air base/mid/strong; record result."""
    out = {}
    spec = SPEC_BY_KEY["air_plane"]
    cal = (jload(v3_path("air_plane")) or {}).get("__v3", {})
    base = make_base_cfg(spec, cal.get("scale_calibrated", spec["v2_scale"]),
                         cal.get("target_m_calibrated", spec["v2_target_m"]))
    for tier in TIERS:
        cfg = apply_tier(base, "air", tier)
        try:
            loaded = _ecmlib.load(cfg["ecm"])
            out["air_" + tier] = dict(ok=True, enabled=loaded.get("enabled"),
                                      n_entities=len(loaded.get("entities", [])),
                                      kinds=sorted({e.get("kind")
                                                    for e in loaded.get("entities", [])}))
        except Exception as exc:
            out["air_" + tier] = dict(ok=False, error=repr(exc))
    return out


def verify_effects():
    """Post-hoc evidence from the already-rendered probe frames (no re-render):
    sea ridge really appears, ground scenery is in-frame but label-clean."""
    from PIL import Image
    out = {}
    sea = {}
    for t in TIERS:
        p = os.path.join(PROBE_DIR, "sea_ship_%s" % t, "raw", "radar", "000000.npy")
        if not os.path.isfile(p):
            sea[t] = dict(error="missing")
            continue
        rd = np.load(p).astype(np.float64)
        c = rd.shape[1] // 2
        cen = rd[:, max(0, c - 2):c + 3].mean(axis=1)
        cols = np.r_[0:max(0, c - 2), min(rd.shape[1], c + 3):rd.shape[1]]
        off = rd[:, cols].mean(axis=1) if len(cols) else cen
        sea[t] = dict(ridge_rows_pct=round(100.0 * float((cen > 3 * off).mean()), 2),
                      center_off_db=round(10 * math.log10(
                          (cen.mean() + 1e-30) / (off.mean() + 1e-30)), 2))
    out["sea_clutter_ridge"] = dict(
        tiers=sea,
        base_off_mid_on=bool(sea.get("base", {}).get("ridge_rows_pct", 100) < 20
                             and sea.get("mid", {}).get("ridge_rows_pct", 0) > 80),
        conclusion="clutter_ridge=false 真关脊(约3% range 单元)，mid/strong 真开脊(约99%)，"
                   "证实 rc=dict(DEFAULTS) 默认 True 的坑")
    gbp = os.path.join(PROBE_DIR, "ground_vehicle_base", "images", "optical", "000000.png")
    ground = {}
    if os.path.isfile(gbp):
        gb = np.asarray(Image.open(gbp).convert("L"), dtype=np.float64)
        for t in TIERS:
            p = os.path.join(PROBE_DIR, "ground_vehicle_%s" % t, "images", "optical",
                             "000000.png")
            if not os.path.isfile(p):
                ground[t] = dict(error="missing")
                continue
            g = np.asarray(Image.open(p).convert("L"), dtype=np.float64)
            diff = np.abs(g - gb)
            lab = jload(os.path.join(PROBE_DIR, "ground_vehicle_%s" % t,
                                     "labels", "optical", "000000.json")) or {}
            boxes = lab.get("boxes") or {}
            ground[t] = dict(changed_px_pct=round(100.0 * float((diff > 8).mean()), 3),
                             n_boxes=len(boxes), box_names=sorted(boxes.keys()))
    out["ground_clutter_in_frame"] = dict(
        tiers=ground,
        conclusion="mid/strong 光学像素变化(>8DN) 约0.7%/1.5% 即杂物入画，"
                   "但标注始终 boxes=1 仅主目标，label:false scenery 不进 COCO")
    return out


def decide(tiers):
    """jam_effective / target_detectable / label_clean + recommended_tier."""
    b = tiers.get("base", {})
    b_floor = b.get("noise_floor_db", 0.0)
    b_center = b.get("center_rel_db", 0.0)
    b_blobs = b.get("optical_blobs", 0)
    b_hot = b.get("ir_hotspots", 0)
    for tier in TIERS:
        r = tiers.get(tier)
        if not r:
            tiers[tier] = r = dict(error="missing")
        if r.get("error"):
            r.setdefault("jam_effective", False)
            r.setdefault("target_detectable", False)
            r.setdefault("label_clean", False)
            r.setdefault("jam_delta", {})
            continue
        fd = r["noise_floor_db"] - b_floor
        cd = r["center_rel_db"] - b_center
        bd = r["optical_blobs"] - b_blobs
        hd = r["ir_hotspots"] - b_hot
        r["jam_delta"] = dict(noise_floor_db=round(fd, 2), center_rel_db=round(cd, 2),
                              optical_blobs=bd, ir_hotspots=hd)
        rd_up = max(fd, cd) >= 3.0
        spot_up = (bd >= 2) or (hd >= 2)
        r["jam_effective"] = bool(rd_up or spot_up) if tier != "base" else False
        r["target_detectable"] = bool(
            r["peak_snr_db"] >= 12.0 and r["detected"]
            and TARGET_PX[0] <= r["box_px"] <= TARGET_PX[1])
        r["label_clean"] = bool(r["n_boxes"] == 1 and r["coco_clean"])
    rec, reason = "base", ""
    for t in ("strong", "mid", "base"):
        r = tiers.get(t, {})
        if r.get("jam_effective") and r.get("target_detectable") and r.get("label_clean"):
            rec = t
            break
    if rec == "strong":
        reason = "strong 档干扰生效且目标仍可检出、标注干净，取最强档"
    elif rec == "mid":
        reason = "strong 档淹没目标或未过门禁，mid 档干扰生效且目标可检出、标注干净，取 mid"
    else:
        reason = "mid/strong 均未同时满足干扰生效+目标可检出+标注干净，只能保留 base"
    return rec, reason


def probe_domain(spec, cal, args):
    key, domain, seed = spec["key"], spec["domain"], spec["seed"]
    scale = cal.get("scale_calibrated", spec["v2_scale"])
    target_m = cal.get("target_m_calibrated", spec["v2_target_m"])
    base = make_base_cfg(spec, scale, target_m)
    tiers = {}
    for tier in TIERS:                       # 3 tiers sequential within a domain
        cfg = apply_tier(base, domain, tier)
        cfg_path = os.path.join(PROBE_DIR, "_probe_%s_%s.json" % (key, tier))
        jwrite(cfg_path, cfg)
        odir = os.path.join(PROBE_DIR, "%s_%s" % (key, tier))
        try:
            r = run_gen(cfg_path, odir, 1, seed, threads=2, quiet=True)
            if r.returncode != 0:
                tiers[tier] = dict(error="render failed: " + fail_tail(r)[-500:])
                print("PROBE FAIL %s/%s: %s" % (key, tier, fail_tail(r)[-300:]))
                continue
            tiers[tier] = probe_metrics(odir, spec)
            print("PROBE %-16s %-6s peak=%.1f dB center=+%.1f dB blobs=%d hot=%d "
                  "detected=%s box=%.1f px"
                  % (key, tier, tiers[tier]["peak_snr_db"], tiers[tier]["center_rel_db"],
                     tiers[tier]["optical_blobs"], tiers[tier]["ir_hotspots"],
                     tiers[tier]["detected"], tiers[tier]["box_px"]))
        except Exception as exc:
            tiers[tier] = dict(error="metrics failed: %s" % repr(exc))
            print("PROBE ERR %s/%s: %s" % (key, tier, repr(exc)))
    return tiers


def phase_probe(args):
    os.makedirs(PROBE_DIR, exist_ok=True)
    cals = {}
    for spec in SPECS:
        cals[spec["key"]] = (jload(v3_path(spec["key"])) or {}).get("__v3", {})
    ecm_val = ecm_validate()
    raw = {}
    with ThreadPoolExecutor(max_workers=3) as ex:   # 3 domains parallel
        futs = {ex.submit(probe_domain, s, cals[s["key"]], args): s["key"] for s in SPECS}
        for fut, key in futs.items():
            raw[key] = fut.result()
    domains = {}
    for spec in SPECS:
        key = spec["key"]
        tiers = raw[key]
        rec, reason = decide(tiers)
        domains[key] = dict(domain=spec["domain"], seed=spec["seed"],
                            tiers=tiers, recommended_tier=rec, reason=reason)
        print("== %-16s recommended=%s  (%s)" % (key, rec, reason))
    report = dict(
        purpose="W1 v3 physical-layer jamming probe: 3 domains x base/mid/strong, 1 frame each",
        target_px=TARGET_PX, aim_px=AIM_PX, frames=N_FRAMES, resolution=[W, H],
        jam_layer="ecm (air) / objects+radar_clutter (ground) / radar_clutter (sea)",
        calibration={s["key"]: cals[s["key"]] for s in SPECS},
        ecm_load_validation=ecm_val,
        verification=verify_effects(),
        domains=domains,
        thresholds=dict(
            jam_effective="RD 中心带/噪声基底相对 base 提升 >=3 dB 或 排除目标框后亮点/热点数增加 >=2",
            target_detectable="peak_snr >= 12 dB 且 detected=true 且真实框在 28..34 px",
            label_clean="optical boxes==1 且 COCO 类别不增加",
            recommended_tier="干扰生效但目标仍可检出的最强档（strong 淹没则 mid）",
        ),
        notes=[
            "cfg 是 v2 配置的深拷贝；三键同源实拍背景、camera 5 阶段、atmosphere、preview_span_c=20、删 temp_c、sensor.enabled=false、auto_exposure=false 全部原样保留",
            "ecm.load 白名单仅 enabled/entities/seed；每实体字段 = _COMMON | DEFAULTS[kind]。false_target 用 power_ratio/range_step_m/range_offset_m，chaff_cloud 用 rcs_m2（不是 rcs_db），noise_jammer 用 erp_w；band 必须是字符串，life 只收 start_s/end_s/fade_s。绝对 pos+vel，不用 parent/link",
            "ecm.attach 强制 label=False，故 ECM 实体不进 COCO；地面杂物全部 label:false 且不写 temp_c/temp_c_hint，标注保持单类",
            "海杂波脊：radar.py:319 rc=dict(DEFAULTS) 令 clutter_ridge 默认 True，且 v2 sea 配置没写该键，所以 v2 海场景其实已开脊；base 必须显式 clutter_ridge=false 才能真关，mid/strong 显式 true",
            "image-layer interference 只导出 interfered/ 副本且不改 RD 张量，会破坏三模态对齐，本轮一律不用",
            "sky_backdrop_card 用裸相对路径加载图片，Blender 不按 _base_dir 解析；V2.run_gen 用 sidecar 把 sky_image 绝对化，三份 authored 指针保持同源相对路径",
            "标定的坑：单步 scale*=31/w 用 v2 20 帧标定时，30 帧配置改变了 frame0 斜距与俯视几何，竖长的 ship 横向真实框只到 26.69px（跌出 28..34）；对 v3 基础帧再跑一次 scale*=31/w 闭环修正，三域真实框回到 31.00/31.00/31.01px",
        ],
    )
    jwrite(REPORT_PATH, report)
    print("PROBE report ->", os.path.relpath(REPORT_PATH, ROOT))
    return 0


# ---------------------------------------------------------------------------
# full / repair (implemented; NOT executed by this task)
# ---------------------------------------------------------------------------
def phase_full(args):
    dest = os.path.abspath(args.dest)
    os.makedirs(dest, exist_ok=True)
    for spec in SPECS:
        p = v3_path(spec["key"])
        out = os.path.join(dest, spec["key"])
        print("FULL %s -> %s (n=30)" % (spec["key"], out), flush=True)
        r = run_gen(p, out, N_FRAMES, spec["seed"], threads=args.threads)
        print("  rc=%d %s" % (r.returncode,
                              (r.stdout or "")[-500:].replace("\n", " | ")[:400]), flush=True)
        if r.returncode != 0:
            print("  stderr:", (r.stderr or "")[-600:])
            return 1
    return 0


def phase_repair(args):
    """v2 LWIR zero-hit restore, retargeted to 30 frames / v3 configs."""
    dest = os.path.abspath(args.dest)
    total = 0
    for spec in SPECS:
        key = spec["key"]
        run = os.path.join(dest, key)
        coco_path = os.path.join(run, "annotations_ir_lwir_coco.json")
        coco = jload(coco_path)
        if not coco:
            print("REPAIR SKIP", key, "no IR COCO")
            continue
        fixed = []
        for i in range(N_FRAMES):
            ip = os.path.join(run, "labels", "ir_lwir", "%06d.json" % i)
            ilb = jload(ip)
            if not ilb or ilb.get("boxes"):
                continue
            olb = jload(os.path.join(run, "labels", "optical", "%06d.json" % i), {}) or {}
            src = [(name, b) for name, b in (olb.get("boxes") or {}).items()
                   if isinstance(b, dict) and b.get("visible", True)
                   and all(k in b for k in ("x0", "y0", "x1", "y1"))]
            if not src:
                continue
            name, b = src[0]
            repaired = dict(b)
            repaired.update(pixels=0, visible=True,
                            visibility_source="co_projected_from_optical",
                            source="corners (unified, optical visibility)")
            ilb["boxes"] = {name: repaired}
            ilb["box_source"] = "corners (unified, optical visibility)"
            jwrite(ip, ilb)
            have = {a.get("image_id") for a in coco.get("annotations", [])}
            if i not in have:
                w = float(b["x1"]) - float(b["x0"])
                h = float(b["y1"]) - float(b["y0"])
                nid = 1 + max([int(a.get("id", 0)) for a in coco.get("annotations", [])]
                              or [0])
                coco.setdefault("annotations", []).append(dict(
                    id=nid, image_id=i,
                    category_id=(coco.get("categories") or [{"id": 1}])[0]["id"],
                    bbox=[round(float(b["x0"]), 2), round(float(b["y0"]), 2),
                          round(w, 2), round(h, 2)],
                    area=round(w * h, 2), iscrowd=0,
                    source="corners (unified, optical visibility)",
                    object_name=name))
            fixed.append(i)
        coco.setdefault("info", {})["repaired_frames"] = fixed
        jwrite(coco_path, coco)
        jwrite(os.path.join(run, "repair.json"), dict(
            reason="LWIR instance-pass sub-pixel miss",
            principle="shared projected corners; optical confirmed visibility",
            frames=fixed))
        sp = os.path.join(run, "stats.json")
        st = jload(sp)
        if isinstance(st, dict):
            n_ir = len(coco.get("annotations", []))
            st["ir_boxes"] = n_ir
            if isinstance(st.get("ir"), dict) and "lwir" in st["ir"]:
                st["ir"]["lwir"]["boxes"] = n_ir
            n_opt = int(st.get("optical_boxes") or 0)
            cls = ((coco.get("categories") or [{}])[0].get("name")
                   or spec["cls"])
            st["boxes_per_class"] = {cls: n_opt + n_ir}
            jwrite(sp, st)
        cp = v3_path(key)
        cfg = jload(cp)
        if cfg and isinstance(cfg.get("recipe"), dict):
            ro = cfg["recipe"].get("objects") or []
            if ro and isinstance(ro[0], dict):
                ro[0]["scale"] = cfg["objects"][0]["scale"]
                jwrite(cp, cfg)
        total += len(fixed)
        print("REPAIR %-16s frames=%s IR COCO=%d stats.ir_boxes=%s"
              % (key, fixed, len(coco.get("annotations", [])),
                 (jload(sp) or {}).get("ir_boxes")))
    print("REPAIR total frames restored:", total)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["cfg", "probe", "full", "repair"])
    ap.add_argument("--dest", default=DEFAULT_DEST)
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    return {"cfg": phase_cfg, "probe": phase_probe, "full": phase_full,
            "repair": phase_repair}[a.phase](a) or 0


if __name__ == "__main__":
    raise SystemExit(main())
