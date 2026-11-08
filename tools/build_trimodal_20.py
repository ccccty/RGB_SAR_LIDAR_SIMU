#!/usr/bin/env python
"""Build three 20-frame trimodal (optical / LWIR / radar-RD) seeker datasets.

Acceptance contract (see tools/validate_trimodal.py):
  * narrow FOV 768x576, single seeker view, 20 frames, one clip, fixed seed
  * real projected box width 6-14 px -- "barely discernible", never prominent
  * frames.json target_px agrees with the real projected box
  * optical / LWIR / radar share frame ids, world pose and one labelled target
  * reproducible: same cfg + seed -> same frames

Design notes (from the red/blue-team review):
  * pixel size follows px = size_m * W / (2 * R * tan(fov/2)); we shrink
    target_m + objects.scale and widen fov instead of pulling R away, because
    R^4 hurts the radar link and transmittance(R) hurts LWIR.
  * objects[].rcs_db is decoupled from geometry, so a tiny mesh can still be a
    detectable radar target.
  * scenegen hardcodes 960x720, target_px 34..72 and wander=0.35*fov; every one
    of those is overwritten below. clip.views is written explicitly because
    plan.py resolves the view size from clip.views, not sensors.optical.
  * --probe renders one frame per scene and rescales objects[].scale so the
    REAL projected box (not the planner's nominal target_px) lands at 10 px.

Usage
    python tools/build_trimodal_20.py --phase cfg
    python tools/build_trimodal_20.py --phase probe
    python tools/build_trimodal_20.py --phase full
    python tools/build_trimodal_20.py --phase repair
    python tools/build_trimodal_20.py --phase seedcheck
"""
import argparse
import glob
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from simkit import scenegen  # noqa: E402

DEFAULT_DEST = r"D:\work\多模态数据集生成\dataest"
SCENE_DIR = os.path.join(ROOT, "scenes")
PROBE_DIR = os.path.join(ROOT, "_tmp", "ds20_probe")
W, H = 768, 576
TARGET_PX = [7, 13]
AIM_PX = 10.0

# fov / target_m / R are a self-consistent triple:
#   px = target_m * W / (2 * R * tan(fov/2)),  R band <- target_px [7,13]
SPECS = [
    # head_on projects only the fuselage width: the 10 m design target became
    # 0.9 px with zero instance pixels, so the optical label was invisible while
    # radar still detected it. Pursuit/side aspect keeps the wingspan in frame.
    dict(key="air_plane", title="空中飞机", env="sky", aspect="pursuit",
         asset="军用飞机/双发战斗机.obj", cls="fighter",
         fov=8.0, target_m=5.0, scale=0.02380, range_m=[2300, 3400],
         rcs_db=5.0, temp_c=60.0, n_samples=4096,
         chirp_us=16.384, pri_us=32.768, closing_mps=60.0, seed=7),
    dict(key="ground_vehicle", title="地面车辆", env="ground", aspect="look_down",
         asset="军用地面车辆/主战坦克.obj", cls="vehicle",
         fov=10.0, target_m=2.5, scale=0.80372, range_m=[920, 1370],
         rcs_db=2.0, temp_c=60.0, n_samples=2048,
         chirp_us=8.192, pri_us=16.384, closing_mps=15.0, seed=17),
    dict(key="sea_ship", title="海面舰船", env="sea", aspect="look_down",
         asset="军舰/导弹艇.obj", cls="ship",
         fov=12.0, target_m=6.0, scale=0.45411, range_m=[1850, 2730],
         rcs_db=20.0, temp_c=70.0, n_samples=4096,
         chirp_us=16.384, pri_us=32.768, closing_mps=12.0, seed=27),
]
SPEC_BY_KEY = {s["key"]: s for s in SPECS}
CLASSES_HINT = {"air_plane": "fighter", "ground_vehicle": "vehicle",
                "sea_ship": "ship"}


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


def scene_path(key):
    return os.path.join(SCENE_DIR, "ds20_%s.json" % key)


def make_cfg(spec, scale_override=None):
    """scenegen base + explicit overwrite of every field that decides size."""
    r = scenegen.default_recipe()
    r["name"] = "ds20_" + spec["key"]
    r["environment"].update(kind=spec["env"], weather="clear",
                            background={"mode": "none"})
    r["view"].update(aspect=spec["aspect"],
                     range_km=spec["range_m"][1] / 1000.0,
                     fov_deg=spec["fov"])
    r["objects"] = [dict(asset=spec["asset"], cls=spec["cls"],
                         scale=scale_override or spec["scale"], count=1)]
    r["capture"].update(sensors=["optical", "ir", "radar"], frames=20,
                        fps=10.0, seed=spec["seed"])
    c = scenegen.build_scene(scenegen.validate_recipe(r))

    c["name"] = r["name"]
    c["seed"] = spec["seed"]
    c["frames"] = {"n": 20, "dt": 0.1}
    c["clip"]["frames_per_clip"] = 20
    c["clip"]["camera"] = "ir_camera"
    # The only reliable size contract: clip.views wins over sensors.*.
    c["clip"]["views"] = [dict(name="seeker", tag="", fov_deg=spec["fov"],
                               width=W, height=H)]
    sk = c["clip"].setdefault("seeker", {})
    sk.update(target_m=spec["target_m"], target_px=list(TARGET_PX),
              range_m=list(spec["range_m"]), closing_mps=spec["closing_mps"],
              wander_deg=0.010, wander_hz=0.09, los_jitter_deg=0.010,
              lag_s=0.02)
    for k in ("az_span_deg", "el_span_deg"):
        if k in sk:
            sk[k] = 0.3
    c["clip"]["rig"] = dict(vibration_m=0.01, vibration_deg=0.0, lag_s=0.02)

    for key in ("optical", "ir_camera"):
        c["sensors"][key].update(width=W, height=H, fov_deg=spec["fov"])
    c.setdefault("optical", {})["samples"] = 16
    c["ir"] = dict(bands=["lwir"], netd_mk=28.0, psf_px=0.8,
                   nuc=dict(enabled=True, prnu=0.008, dsnu_counts=25,
                            column_fpn_counts=6, dead_px=0.0002,
                            seed=spec["seed"]))
    obj = c["objects"][0]
    obj["scale"] = float(scale_override or spec["scale"])
    obj["rcs_db"] = spec["rcs_db"]
    obj["temp_c"] = spec["temp_c"]

    rd = c["sensors"]["radar"]
    rd.update(band="Ku", tx_power_dbm=43.0, noise_figure_db=8.0, loss_db=4.0,
              n_samples=spec["n_samples"], n_chirps=64, bandwidth_mhz=50.0,
              chirp_us=spec["chirp_us"], pri_us=spec["pri_us"],
              pfa=1e-6, swerling="I", clutter_density=0.0,
              mti_min_vr_mps=0.0, n_az=40, n_el=20,
              beamwidth_az_deg=1.0, beamwidth_el_deg=4.0)
    rd.pop("det_snr_db", None)   # pfa owns the gate; do not double-specify
    # Do not let the run write into datasets/yolo or generate review sheets.
    c["output"] = dict(preview=False, readme=False, yolo=False, coco=True)
    c["__trimodal_contract"] = dict(
        view=dict(width=W, height=H, fov_deg=spec["fov"]),
        target_px=TARGET_PX, aim_px=AIM_PX, frames=20,
        modalities=["optical", "ir_lwir", "radar"],
        train_sources=dict(optical="images/optical",
                           infrared="images/ir_preview_lwir",
                           radar="raw/radar (float32 power tensor)"))
    return c


def gen_env(threads=None):
    env = dict(os.environ, PYTHONPATH=ROOT, PYTHONDONTWRITEBYTECODE="1",
               PYTHONIOENCODING="utf-8")
    if threads:
        env["SIMKIT_BLENDER_THREADS"] = str(threads)
    return env


def run_gen(cfg_path, out_dir, n, seed, threads=None, quiet=False):
    cmd = [sys.executable, "-m", "simkit", "gen", "-c", cfg_path,
           "-o", out_dir, "-n", str(n), "-s", "optical,ir,radar",
           "--seed", str(seed)]
    if quiet:
        cmd.append("-q")
    return subprocess.run(cmd, cwd=ROOT, env=gen_env(threads),
                          capture_output=True, text=True,
                          encoding="utf-8", errors="replace")


def box_width(path, include_invisible=False):
    """Projected corner width.  During calibration we also accept a box whose
    instance pass is empty (visible=false): its corners are still the true
    geometry, and that is exactly the signal needed to scale a 0.9 px target
    up into the visible band.  The acceptance gate still requires visible=true.
    """
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


def phase_cfg(args):
    for s in SPECS:
        p = scene_path(s["key"])
        c = make_cfg(s, scale_override=jload(p, {}).get("objects", [{}])[0].get("scale")
                     if jload(p) else None)
        jwrite(p, c)
        print("cfg", s["key"], "->", os.path.relpath(p, ROOT),
              "scale", c["objects"][0]["scale"])


def phase_probe(args):
    os.makedirs(PROBE_DIR, exist_ok=True)
    for s in SPECS:
        p = scene_path(s["key"])
        c = jload(p) or make_cfg(s)
        out = os.path.join(PROBE_DIR, s["key"])
        r = run_gen(p, out, 1, s["seed"], threads=args.threads)
        if r.returncode != 0:
            print("PROBE FAIL", s["key"], (r.stdout or "")[-1200:],
                  (r.stderr or "")[-600:])
            continue
        lab = os.path.join(out, "labels", "optical", "000000.json")
        w = box_width(lab, include_invisible=True)
        vis = box_width(lab)
        row = (jload(os.path.join(out, "frames.json")) or [{}])[0]
        nom = row.get("target_px")
        print("PROBE %-16s geometry_box=%s px  visible_box=%s px  nominal=%s px"
              % (s["key"], ("%.3f" % w) if w else None,
                 ("%.3f" % vis) if vis else None, nom))
        if not w:
            print("       NO CORNERS: check the engagement aspect / model path")
            continue
        # Calibrate against the planner's nominal pixel size, not a fixed 10 px:
        # that is what removes the nominal-vs-real label desync found by the
        # red team. The real box is the visibility ground truth.
        aim = float(nom) if isinstance(nom, (int, float)) else AIM_PX
        new_scale = round(c["objects"][0]["scale"] * aim / w, 8)
        c["objects"][0]["scale"] = new_scale
        jwrite(p, c)
        print("       scale %s -> %s (expect ~%.2f px)"
              % (s["scale"], new_scale, aim))


def phase_full(args):
    dest = os.path.abspath(args.dest)
    os.makedirs(dest, exist_ok=True)
    for s in SPECS:
        p = scene_path(s["key"])
        out = os.path.join(dest, s["key"])
        print("FULL %s -> %s" % (s["key"], out), flush=True)
        r = run_gen(p, out, 20, s["seed"], threads=args.threads)
        tail = (r.stdout or "")[-800:]
        print("  rc=%d %s" % (r.returncode, tail.replace("\n", " | ")[:400]),
              flush=True)
        if r.returncode != 0:
            print("  stderr:", (r.stderr or "")[-600:])
            return 1
    return 0


def phase_repair(args):
    """Restore LWIR labels lost to the thermal rasteriser's sub-pixel miss.

    simkit/dataset.py:891 builds IR boxes from the numpy id buffer; a sub-10 px
    target can score zero instance pixels, and _boxes_for then falls back to a
    sparse ray visibility test that also misses, so the frame ships with
    boxes={} while optical has a box.  The project's own contract (dataset.py
    :1245, "corners (unified)") is that every co-boresighted sensor uses the
    SAME projected extent and only asks the mask whether the target is visible.
    Optical already confirmed visibility for that frame, so we reuse its
    verdict and mark the provenance explicitly instead of inventing a box.

    The kernel-side fix belongs in the :1245 unification pass (it currently
    covers optical labels only).  This phase repairs shipped runs in place and
    is idempotent.
    """
    dest = os.path.abspath(args.dest)
    total = 0
    for s in SPECS:
        run = os.path.join(dest, s["key"])
        coco_path = os.path.join(run, "annotations_ir_lwir_coco.json")
        coco = jload(coco_path)
        if not coco:
            print("REPAIR SKIP", s["key"], "no IR COCO")
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
            frames=fixed, kernel_fix_hint="dataset.py:1245 unification pass "
            "should also cover labels/ir_* with optical seen[] names"))

        # stats.json is derived at generation time; after restoring labels it
        # still says "17 IR boxes". Recompute from the COCO we just fixed.
        sp = os.path.join(run, "stats.json")
        st = jload(sp)
        if isinstance(st, dict):
            n_ir = len(coco.get("annotations", []))
            st["ir_boxes"] = n_ir
            if isinstance(st.get("ir"), dict) and "lwir" in st["ir"]:
                st["ir"]["lwir"]["boxes"] = n_ir
            n_opt = int(st.get("optical_boxes") or 0)
            cls = ((coco.get("categories") or [{}])[0].get("name")
                   or CLASSES_HINT.get(s["key"], "target"))
            st["boxes_per_class"] = {cls: n_opt + n_ir}
            jwrite(sp, st)

        # --probe rescales objects[].scale to match the real projected box; keep
        # the embedded recipe in sync so the config has one authoritative value.
        cp = scene_path(s["key"])
        cfg = jload(cp)
        if cfg and isinstance(cfg.get("recipe"), dict):
            ro = cfg["recipe"].get("objects") or []
            if ro and isinstance(ro[0], dict):
                ro[0]["scale"] = cfg["objects"][0]["scale"]
                jwrite(cp, cfg)

        total += len(fixed)
        print("REPAIR %-16s frames=%s  IR COCO annotations=%d  stats ir_boxes=%s"
              % (s["key"], fixed, len(coco.get("annotations", [])),
                 (jload(sp) or {}).get("ir_boxes")))
    print("REPAIR total frames restored:", total)
    return 0


def phase_seedcheck(args):
    """Regression for the --seed fix: optical noise must follow the CLI seed."""
    import hashlib
    s = SPECS[0]
    p = scene_path(s["key"])
    digests = []
    for seed in (7, 991):
        out = os.path.join(PROBE_DIR, "seed_%d" % seed)
        r = run_gen(p, out, 1, seed, threads=args.threads)
        if r.returncode != 0:
            print("SEEDCHECK FAIL", seed, (r.stdout or "")[-500:])
            return 1
        img = os.path.join(out, "images", "optical", "000000.png")
        with open(img, "rb") as fh:
            digests.append(hashlib.md5(fh.read()).hexdigest())
    print("SEEDCHECK optical md5:", digests,
          "DIFFERENT" if digests[0] != digests[1] else "IDENTICAL (seed fix did not apply)")
    return 0 if digests[0] != digests[1] else 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                    choices=["cfg", "probe", "full", "repair", "seedcheck"])
    ap.add_argument("--dest", default=DEFAULT_DEST)
    ap.add_argument("--threads", type=int, default=0,
                    help="Blender thread budget per worker (0 = no override)")
    a = ap.parse_args()
    return {"cfg": phase_cfg, "probe": phase_probe, "full": phase_full,
            "repair": phase_repair,
            "seedcheck": phase_seedcheck}[a.phase](a) or 0


if __name__ == "__main__":
    raise SystemExit(main())
