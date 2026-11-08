#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build v2 20-frame trimodal (optical / LWIR / radar-RD) seeker datasets.

W1 "realism & scene" line.  Ownership: this file, the three scenes/ds20_v2_*.json
configs and tools/v2_probe_report.json.  NOTHING else is written by --phase
cfg/probe; --phase full/repair only write into dataest/{air_plane,
ground_vehicle,sea_ship} and the three v2 configs.

Differences vs the v1 build (tools/build_trimodal_20.py):

  * real projected box 18..22 px (v1 was 6..14 px), px ~ scale, so the scale
    and target_m are re-calibrated by a closed loop probe.
  * background is a REAL photograph from assets/backdrops, one fixed plate per
    domain, wired into ground.backdrop + optical.sky_image +
    ir.background_image (all three point at the SAME ../assets/... path,
    resolved relative to scenes/ via _base_dir).
  * realism switches are opt-in, only in the v2 configs (no kernel defaults
    are touched): optical.sensor (HDR electron chain), atmosphere.exponential
    (4 scale heights), optical.camera vignetting/CA/flare/jpeg, fixed
    auto_exposure=false + exposure, ir.preview_span_c, and objects[].temp_c is
    REMOVED so thermal.solve actually runs.

Engineering constraints honoured:

  * simkit/scenegen.py is NEVER imported or called (W3 owns it).  The v2
    configs are a straight deepcopy of the three v1 scenes/ds20_*.json with
    fields overwritten, so the two lines stay decoupled.
  * dataset._render_config absolutises model paths but KEEPS _base_dir;
    render_optical.add_ground_backdrop / environment_sky and
    ir._background_image_path all resolve ../assets/... against _base_dir.
    sky_backdrop_card() is the one exception: it calls bpy.data.images.load()
    with the plain relative path and Blender does NOT resolve that against
    _base_dir or the process CWD (measured "No such file or directory"), so
    air-scene optical frames would fall back to a flat gradient.  run_gen()
    hands the renderer a sidecar config (inside out_dir) with sky_image
    absolutised from scenes/, while the three AUTHORED pointers stay relative.
  * optical.camera.jpeg needs Pillow, which Blender's bundled Python lacks;
    simkit/camera.py catches that RuntimeError and keeps the lossless frame
    with a warning, so enabling it is safe (recorded in the probe report).

Usage
    python tools/build_v2_trimodal.py --phase cfg
    python tools/build_v2_trimodal.py --phase probe
    python tools/build_v2_trimodal.py --phase full      # main controller runs this
    python tools/build_v2_trimodal.py --phase repair
"""
import argparse
import copy
import glob
import json
import math
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DEFAULT_DEST = r"D:\work\多模态数据集生成\dataest"
SCENE_DIR = os.path.join(ROOT, "scenes")
PROBE_DIR = os.path.join(ROOT, "_tmp", "v2_probe")
REPORT_PATH = os.path.join(ROOT, "tools", "v2_probe_report.json")
W, H = 768, 576
TARGET_PX = [18, 22]
AIM_PX = 20.0

# v1 facts, measured in this workspace (simkit.scene.prepare_mesh, scale=1 bbox
# [19.0,20.013,5.795] / [9.8,5.595,2.561] / [25.0,9.455,5.942]).  real v1 box
# widths 9.3/9.4/9.5 px are the px ~ scale starting point for 20 px.
SPECS = [
    dict(key="air_plane", seed=7, v1_scale=0.24682024, v1_box_px=9.3,
         rcs_db=5.0, n_samples=4096),
    dict(key="ground_vehicle", seed=17, v1_scale=0.44130741, v1_box_px=9.4,
         rcs_db=2.0, n_samples=2048),
    dict(key="sea_ship", seed=27, v1_scale=0.36935305, v1_box_px=9.5,
         rcs_db=20.0, n_samples=4096),
]
SPEC_BY_KEY = {s["key"]: s for s in SPECS}
CLASSES_HINT = {"air_plane": "fighter", "ground_vehicle": "vehicle",
                "sea_ship": "ship"}

BACKDROPS = {
    "air_plane": "../assets/backdrops/sky/17_Cloudscapes_from_6E-6325_DEL-IXR_1_jpg.jpg",
    "sea_ship": "../assets/backdrops/open/03_Sea_wave_abstract_texture_Unsplash.jpg",
}


def jload(path, dflt=None):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return dflt


def jwrite(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1)
        fh.write("\n")


def v1_path(key):
    return os.path.join(SCENE_DIR, "ds20_%s.json" % key)


def v2_path(key):
    return os.path.join(SCENE_DIR, "ds20_v2_%s.json" % key)


def backdrop_rel(key):
    """Fixed real plate per domain; rural/11_* is discovered by glob."""
    if key == "ground_vehicle":
        hits = sorted(glob.glob(os.path.join(ROOT, "assets", "backdrops",
                                            "rural", "11_*.jpg")))
        if not hits:
            raise RuntimeError("assets/backdrops/rural/11_*.jpg not found")
        return os.path.relpath(hits[0], SCENE_DIR).replace(os.sep, "/")
    return BACKDROPS[key]


def resolve_rel(rel, base=SCENE_DIR):
    return os.path.normpath(os.path.join(base, rel))


# ---------------------------------------------------------------------------
# cfg: deepcopy v1 -> v2, overwrite fields only
# ---------------------------------------------------------------------------
def make_v2_cfg(spec):
    key = spec["key"]
    c = jload(v1_path(key))
    if not c:
        raise RuntimeError("missing v1 config: " + v1_path(key))
    c = copy.deepcopy(c)

    fov = float(c["clip"]["views"][0]["fov_deg"])
    rng = [float(x) for x in c["clip"]["seeker"]["range_m"]]
    r_nom = 0.5 * (rng[0] + rng[1])
    # Full-frame ground span at the nominal range; target_m is the physical
    # width of AIM_PX pixels of that span.  Gives 10.380 / 5.217 / 12.536 m.
    span = 2.0 * r_nom * math.tan(math.radians(fov) / 2.0)
    target_m = span * AIM_PX / W
    rel = backdrop_rel(key)

    # --- one real plate, three pointers -----------------------------------
    g = c["ground"]
    g["backdrop"] = rel
    g["backdrop_span_m"] = round(span, 4)
    g["backdrop_span_y_m"] = round(span * 0.75, 4)
    g["backdrop_aligned"] = True
    g["backdrop_offset"] = [0.5, 0.5]
    opt = c["optical"]
    opt["sky_image"] = rel
    c["ir"]["background_image"] = rel

    # --- seeker band / apparent size --------------------------------------
    sk = c["clip"]["seeker"]
    sk["target_px"] = list(TARGET_PX)
    sk["target_m"] = round(target_m, 4)

    # --- realism opt-in ----------------------------------------------------
    opt["sensor"] = dict(
        enabled=True,
        airlight_linear_rgb=[0.35, 0.45, 0.60],
        psf_sigma_px=0.6,
        reference_exposure_s=0.001,
        full_well_e=30000.0,
        white_level_linear=1.0,
        read_noise_e=3.0,
        prnu=0.001,
        bits=12,
        seed=spec["seed"],
        shot_noise=True,
    )
    c["acquisition"] = {"rgb_exposure_s": 0.001}
    opt["auto_exposure"] = False
    opt["exposure"] = 0.0            # fixed stop; auto_exposure would iterate it
    # CAMERA_SKIP drops "sensor" and "diffraction" when optical.sensor.enabled,
    # so keep them listed but they are not double-applied.
    opt["camera"] = {
        "diffraction": {"enabled": True, "f_number": 2.8},
        "sensor": {"enabled": True},
        "vignetting": {"enabled": True, "fov_deg": fov,
                       "strength": 0.55, "f_number": 2.8},
        "chromatic_aberration": {"enabled": True, "lateral_px": 0.6},
        "flare": {"enabled": True, "threshold": 0.94,
                  "strength": 0.22, "count": 3},
        "jpeg": {"enabled": True, "quality": 90},
    }
    c["atmosphere"] = {
        "profile": "exponential",
        "aerosol_scale_height_m": 1200.0,
        "molecular_scale_height_m": 8000.0,
        "water_scale_height_m": 2000.0,
        "rain_scale_height_m": 2000.0,
    }
    c["ir"]["preview_span_c"] = 20

    obj = c["objects"][0]
    obj.pop("temp_c", None)          # let thermal.solve compute the skin temp
    # temp_c_hint (v1) is kept as metadata; region_overrides would be the
    # fallback if a probe showed no IR contrast.
    scale0 = round(spec["v1_scale"] * AIM_PX / spec["v1_box_px"], 8)
    obj["scale"] = scale0
    c["recipe"]["objects"][0]["scale"] = scale0
    c["recipe"]["name"] = "ds20_v2_" + key
    c["name"] = "ds20_v2_" + key

    ct = c.setdefault("__trimodal_contract", {})
    ct["target_px"] = list(TARGET_PX)
    ct["aim_px"] = AIM_PX
    c["__v2"] = dict(
        source=os.path.basename(v1_path(key)),
        backdrop=rel,
        backdrop_span_m=round(span, 4),
        scale_start=scale0,
        target_m_start=round(target_m, 4),
        r_nom_m=r_nom,
        note="realism switches opt-in; temp_c removed for thermal.solve",
    )
    return c


def verify_paths(c):
    """Three pointers -> one existing file, resolved relative to scenes/."""
    rels = {
        "ground.backdrop": c["ground"].get("backdrop"),
        "optical.sky_image": c["optical"].get("sky_image"),
        "ir.background_image": c["ir"].get("background_image"),
    }
    out = {}
    for name, rel in rels.items():
        p = resolve_rel(str(rel))
        out[name] = dict(path=rel, resolved=os.path.abspath(p),
                         exists=os.path.isfile(p))
    out["all_same"] = len({str(v) for v in rels.values()}) == 1
    out["all_exist"] = all(v["exists"] for v in out.values()
                           if isinstance(v, dict))
    return out


def phase_cfg(args):
    for spec in SPECS:
        c = make_v2_cfg(spec)
        vp = verify_paths(c)
        jwrite(v2_path(spec["key"]), c)
        print("CFG %-16s scale0=%s target_m0=%s span=%.3f m"
              % (spec["key"], c["objects"][0]["scale"],
                 c["clip"]["seeker"]["target_m"], c["__v2"]["backdrop_span_m"]))
        print("     backdrop=%s  three_paths_same=%s all_exist=%s"
              % (c["__v2"]["backdrop"], vp["all_same"], vp["all_exist"]))
        if not (vp["all_same"] and vp["all_exist"]):
            print("     !! background path verification FAILED", vp)
            return 1
    return 0


# ---------------------------------------------------------------------------
# rendering helpers
# ---------------------------------------------------------------------------
def gen_env(threads=None):
    env = dict(os.environ, PYTHONPATH=ROOT, PYTHONDONTWRITEBYTECODE="1",
               PYTHONIOENCODING="utf-8")
    if threads:
        env["SIMKIT_BLENDER_THREADS"] = str(threads)
    return env


def run_gen(cfg_path, out_dir, n, seed, threads=None, quiet=False):
    """Render one run of a v2 config.

    Measured pitfall (Blender 4.5.9, --python-expr): sky_backdrop_card() calls
    bpy.data.images.load() with the *plain relative* optical.sky_image.  Blender
    resolves that against its own .blend base, NOT against _base_dir or the
    process CWD, so ../assets/backdrops/... fails with "No such file or
    directory" and the optical frame silently falls back to a flat gradient
    (measured: grey std 4.7 vs 40.1 on the real plate).  add_ground_backdrop /
    environment_sky / ir._background_image_path all join _base_dir first, so
    only sky_image needs help.

    Fix WITHOUT touching the kernel or the authored relative paths: hand the
    renderer a sidecar config (inside out_dir) whose optical.sky_image has been
    absolutised from scenes/, and pin _base_dir to scenes/.  The three AUTHORED
    pointers in scenes/ds20_v2_*.json stay relative and identical.
    """
    c = jload(cfg_path)
    if not c:
        raise RuntimeError("missing render config: " + cfg_path)
    rc = copy.deepcopy(c)
    sky = (rc.get("optical") or {}).get("sky_image")
    if sky and not os.path.isabs(str(sky)):
        rc["optical"]["sky_image"] = os.path.abspath(resolve_rel(str(sky)))
    rc["_base_dir"] = os.path.abspath(SCENE_DIR)
    render_cfg = os.path.join(out_dir, "_v2_render_cfg.json")
    jwrite(render_cfg, rc)
    cmd = [sys.executable, "-m", "simkit", "gen", "-c", os.path.abspath(render_cfg),
           "-o", os.path.abspath(out_dir), "-n", str(n),
           "-s", "optical,ir,radar", "--seed", str(seed)]
    if quiet:
        cmd.append("-q")
    return subprocess.run(cmd, cwd=SCENE_DIR, env=gen_env(threads),
                          capture_output=True, text=True,
                          encoding="utf-8", errors="replace")


def box_width(path, include_invisible=False):
    d = jload(path)
    if not d or not isinstance(d.get("boxes"), dict):
        return None
    bs = [b for b in d["boxes"].values() if isinstance(b, dict)
          and all(k in b for k in ("x0", "y0", "x1", "y1"))]
    if not include_invisible:
        bs = [b for b in bs if b.get("visible") is not False]
    if not bs:
        return None
    return max(float(b["x1"]) - float(b["x0"]) for b in bs)


def fail_tail(r):
    return ((r.stdout or "")[-1400:] + " | " + (r.stderr or "")[-700:])


def ab_metrics(png_path):
    """Realism A/B numbers: p99-p1, unique grey levels, R-G corr, flattest tile std."""
    from PIL import Image
    rgb = np.asarray(Image.open(png_path).convert("RGB"), dtype=np.float64)
    gray = np.asarray(Image.open(png_path).convert("L"), dtype=np.float64)
    p99_p1 = float(np.percentile(gray, 99.0) - np.percentile(gray, 1.0))
    uniq = int(len(np.unique(gray.astype(np.uint8))))
    r = rgb[:, :, 0].ravel()
    g = rgb[:, :, 1].ravel()
    if float(r.std()) < 1e-9 or float(g.std()) < 1e-9:
        corr = None
    else:
        corr = float(np.corrcoef(r, g)[0, 1])
        if not math.isfinite(corr):
            corr = None
    gh, gw = gray.shape
    bh, bw = max(1, gh // 6), max(1, gw // 8)
    tiles = []
    for y in range(0, gh - bh + 1, bh):
        for x in range(0, gw - bw + 1, bw):
            tiles.append(float(gray[y:y + bh, x:x + bw].std()))
    return dict(
        p99_minus_p1=round(p99_p1, 3),
        unique_grey_levels=uniq,
        rg_corr=(round(corr, 6) if corr is not None else None),
        flattest_tile_std=round(min(tiles), 4),
        mean_luma=round(float(gray.mean()), 3),
    )


def ab_decision(a, b):
    """Which chain is more realistic?  3 bigger-is-better metrics, best of 3."""
    wins = [
        b["p99_minus_p1"] > a["p99_minus_p1"],
        b["unique_grey_levels"] > a["unique_grey_levels"],
        b["flattest_tile_std"] > a["flattest_tile_std"],
    ]
    return dict(hdr_wins=int(sum(wins)), hdr_better=bool(sum(wins) >= 2),
                metrics=dict(contrast=wins[0], levels=wins[1], texture=wins[2]))


def ir_preview_std(out_dir):
    p = os.path.join(out_dir, "images", "ir_preview_lwir", "000000.png")
    if not os.path.isfile(p):
        return None
    from PIL import Image
    return round(float(np.asarray(Image.open(p).convert("L"),
                                  dtype=np.float64).std()), 3)


# ---------------------------------------------------------------------------
# probe: 1-frame scale/target_m calibration + sensor A/B
# ---------------------------------------------------------------------------
def phase_probe(args):
    os.makedirs(PROBE_DIR, exist_ok=True)
    report = dict(
        purpose="W1 v2 realism & scale probe (A=no-HDR / B=HDR electron chain)",
        target_px=TARGET_PX, aim_px=AIM_PX, resolution=[W, H],
        scenes={}, ab_method="same calibrated config, only optical.sensor.enabled differs",
        notes=[],
    )
    for spec in SPECS:
        key = spec["key"]
        p = v2_path(key)
        c = jload(p)
        if not c:
            print("PROBE SKIP", key, "no v2 config; run --phase cfg first")
            continue
        fov = float(c["clip"]["views"][0]["fov_deg"])
        # Frame 1 (calibration) uses the HDR chain ON; it is also the B frame.
        c["optical"]["sensor"]["enabled"] = True
        jwrite(p, c)

        out0 = os.path.join(PROBE_DIR, key + "_cal")
        r = run_gen(p, out0, 1, spec["seed"], threads=args.threads)
        if r.returncode != 0:
            print("PROBE FAIL", key, "calibration", fail_tail(r))
            report["scenes"][key] = dict(error="calibration render failed")
            continue
        lab0 = os.path.join(out0, "labels", "optical", "000000.json")
        w_before = box_width(lab0, include_invisible=True)
        row0 = (jload(os.path.join(out0, "frames.json")) or [{}])[0]
        nom = row0.get("target_px")
        r0 = row0.get("range_m")
        if not w_before or not isinstance(nom, (int, float)) or not r0:
            print("PROBE FAIL", key, "no box/nominal/range", w_before, nom, r0)
            report["scenes"][key] = dict(error="no calibration signal")
            continue
        aim = float(nom)
        old_scale = float(c["objects"][0]["scale"])
        new_scale = round(old_scale * aim / w_before, 8)
        # target_m MUST equal the calibrated real projected width.  Derive it
        # from the MEASURED frame-0 range and nominal px (probe ground truth),
        # not from the theoretical midpoint.
        target_m_measured = 2.0 * float(r0) * math.tan(math.radians(fov) / 2.0) * aim / W

        # Commit the calibration to the v2 config (HDR stays on for now).
        c["objects"][0]["scale"] = new_scale
        c["recipe"]["objects"][0]["scale"] = new_scale
        c["clip"]["seeker"]["target_m"] = round(target_m_measured, 4)
        c["__v2"]["scale_calibrated"] = new_scale
        c["__v2"]["target_m_calibrated"] = round(target_m_measured, 4)
        jwrite(p, c)

        # Frame 2 = B (HDR on, calibrated config); also verifies the real box.
        out_b = os.path.join(PROBE_DIR, key + "_B")
        rb = run_gen(p, out_b, 1, spec["seed"], threads=args.threads)
        if rb.returncode != 0:
            print("PROBE FAIL", key, "B frame", fail_tail(rb))
            report["scenes"][key] = dict(error="B render failed")
            continue
        w_after = box_width(os.path.join(out_b, "labels", "optical", "000000.json"))
        w_after_inv = box_width(os.path.join(out_b, "labels", "optical", "000000.json"),
                                include_invisible=True)
        b_img = os.path.join(out_b, "images", "optical", "000000.png")

        # Frame 3 = A: identical config, only sensor.enabled=false.  Stored in
        # _tmp, so pin _base_dir explicitly to keep ../assets/... resolvable.
        ca = copy.deepcopy(c)
        ca["optical"]["sensor"]["enabled"] = False
        ca["_base_dir"] = os.path.abspath(SCENE_DIR)
        a_cfg = os.path.join(PROBE_DIR, "ab_%s_A.json" % key)
        jwrite(a_cfg, ca)
        out_a = os.path.join(PROBE_DIR, key + "_A")
        ra = run_gen(a_cfg, out_a, 1, spec["seed"], threads=args.threads)
        if ra.returncode != 0:
            print("PROBE FAIL", key, "A frame", fail_tail(ra))
            report["scenes"][key] = dict(error="A render failed")
            continue
        a_img = os.path.join(out_a, "images", "optical", "000000.png")

        A = ab_metrics(a_img)
        B = ab_metrics(b_img)
        dec = ab_decision(A, B)
        final_hdr = bool(dec["hdr_better"])
        c["optical"]["sensor"]["enabled"] = final_hdr
        jwrite(p, c)

        print("PROBE %-16s box %.3f -> %.3f px (nom %.1f, visible %.3f)"
              % (key, w_before, w_after_inv or -1, aim, w_after or -1))
        print("       scale %s -> %s  target_m=%.4f m  R0=%.1f m"
              % (old_scale, new_scale, target_m_measured, float(r0)))
        print("       A(no-HDR) p99-p1=%s uniq=%s rg=%s flat=%s"
              % (A["p99_minus_p1"], A["unique_grey_levels"], A["rg_corr"],
                 A["flattest_tile_std"]))
        print("       B(HDR)   p99-p1=%s uniq=%s rg=%s flat=%s  -> keep_hdr=%s"
              % (B["p99_minus_p1"], B["unique_grey_levels"], B["rg_corr"],
                 B["flattest_tile_std"], final_hdr))
        report["scenes"][key] = dict(
            seed=spec["seed"], fov_deg=fov,
            backdrop=c["__v2"]["backdrop"],
            backdrop_span_m=c["__v2"]["backdrop_span_m"],
            range_m=c["clip"]["seeker"]["range_m"],
            frame0_range_m=float(r0), nominal_target_px=aim,
            box_px_before=round(w_before, 3),
            box_px_after=round(w_after_inv, 3) if w_after_inv else None,
            box_px_after_visible=round(w_after, 3) if w_after else None,
            in_band=bool(w_after_inv and TARGET_PX[0] <= w_after_inv <= TARGET_PX[1]),
            scale_start=old_scale, scale_calibrated=new_scale,
            target_m_start=c["__v2"]["target_m_start"],
            target_m_calibrated=round(target_m_measured, 4),
            A_no_hdr=A, B_hdr=B, decision=dec,
            final_sensor_enabled=final_hdr,
            ir_preview_std=ir_preview_std(out_b),
            paths=verify_paths(c),
        )
    report["notes"] = [
        "calibration: scale *= nominal_target_px / measured geometry box",
        "target_m derived from measured frame-0 range + nominal px (real projected width)",
        "A/B: only optical.sensor.enabled differs; CAMERA_SKIP drops sensor+diffraction in the camera post chain when the HDR chain is on",
        "if HDR chain scores worse, final config keeps sensor.enabled=false (do not force it on)",
        "optical.camera.jpeg needs Pillow in Blender; simkit/camera.py catches the missing codec and keeps the lossless frame",
        "sky_backdrop_card() calls bpy.data.images.load() with the plain relative path; Blender does NOT resolve ../assets/... against _base_dir or CWD (measured No such file or directory, air optical frame fell to a flat gradient, grey std 4.7). run_gen() hands the renderer a sidecar config with sky_image absolutised from scenes/, so the authored three relative pointers stay untouched and the real plate shows (std 39.3 vs 40.1 on source)",
        "IR preview: background-plate compositing emits RuntimeWarning invalid value in cast (ir.py:1039-1040) on dark plate pixels; kernel-owned, target still visible (ir_preview_std 38-79)",
    ]
    jwrite(REPORT_PATH, report)
    print("PROBE report ->", os.path.relpath(REPORT_PATH, ROOT))
    return 0


# ---------------------------------------------------------------------------
# full / repair
# ---------------------------------------------------------------------------
def phase_full(args):
    dest = os.path.abspath(args.dest)
    os.makedirs(dest, exist_ok=True)
    for spec in SPECS:
        p = v2_path(spec["key"])
        out = os.path.join(dest, spec["key"])
        print("FULL %s -> %s" % (spec["key"], out), flush=True)
        r = run_gen(p, out, 20, spec["seed"], threads=args.threads)
        print("  rc=%d %s" % (r.returncode,
                              (r.stdout or "")[-500:].replace("\n", " | ")[:400]),
              flush=True)
        if r.returncode != 0:
            print("  stderr:", (r.stderr or "")[-600:])
            return 1
    return 0


def phase_repair(args):
    """v1 build_trimodal_20.phase_repair, retargeted at v2 configs / v2 runs.

    Restores LWIR boxes lost to the thermal rasteriser's sub-pixel miss, then
    re-syncs COCO, stats.json and the embedded recipe scale.  Idempotent.
    """
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
        for i in range(20):
            ip = os.path.join(run, "labels", "ir_lwir", "%06d.json" % i)
            ilb = jload(ip)
            if not ilb or ilb.get("boxes"):
                continue
            olb = jload(os.path.join(run, "labels", "optical", "%06d.json" % i), {})
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
            reason="LWIR instance-pass sub-pixel miss (kernel dataset.py:891)",
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
                   or CLASSES_HINT.get(key, "target"))
            st["boxes_per_class"] = {cls: n_opt + n_ir}
            jwrite(sp, st)

        cp = v2_path(key)
        cfg = jload(cp)
        if cfg and isinstance(cfg.get("recipe"), dict):
            ro = cfg["recipe"].get("objects") or []
            if ro and isinstance(ro[0], dict):
                ro[0]["scale"] = cfg["objects"][0]["scale"]
                jwrite(cp, cfg)

        total += len(fixed)
        print("REPAIR %-16s frames=%s  IR COCO annotations=%d  stats ir_boxes=%s"
              % (key, fixed, len(coco.get("annotations", [])),
                 (jload(sp) or {}).get("ir_boxes")))
    print("REPAIR total frames restored:", total)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                    choices=["cfg", "probe", "full", "repair"])
    ap.add_argument("--dest", default=DEFAULT_DEST)
    ap.add_argument("--threads", type=int, default=6,
                    help="SIMKIT_BLENDER_THREADS per worker (default 6)")
    a = ap.parse_args()
    return {"cfg": phase_cfg, "probe": phase_probe, "full": phase_full,
            "repair": phase_repair}[a.phase](a) or 0


if __name__ == "__main__":
    raise SystemExit(main())
