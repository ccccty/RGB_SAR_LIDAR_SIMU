"""Standalone SAR check: one scene -> PNG + label JSON + a printed budget.

    PYTHONPATH=. ./env/Scripts/python.exe tools/sar_check.py -c scenes/sar_air_uav.json -o out/sar_check

Writes  <out>/images/sar/000000.png, <out>/labels/sar/000000.json
        <out>/sar_debug/000000_db16.png  (16-bit calibrated intensity)
and prints resolution / swath / statistics / label box / timing / memory.

This is the independent cross-check for simkit/sar.py (the same numbers the
dataset pipeline will report), not a second implementation.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simkit import labels, png, sar, scene, weather  # noqa: E402


def _rss_mb():
    """Peak resident memory of this process, portably (no psutil)."""
    try:                                                # POSIX
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        pass
    try:                                                # Windows
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = ctypes.c_void_p   # pseudo-handle is 64-bit
        fn = k32.K32GetProcessMemoryInfo
        fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), wintypes.DWORD]
        fn.restype = wintypes.BOOL
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        if fn(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
            return pmc.PeakWorkingSetSize / (1024.0 * 1024.0)
    except Exception:
        pass
    return float("nan")


def _boxes(obj_map, world, target_boxes, min_px=4):
    """Label boxes: obj_map mask where it is thick enough, geometric otherwise.

    A 0.4 m target in 1 m resolution cells owns 0-2 pixels, so the mask box is
    skipped by the min_px gate and the projected box from simulate()["target_boxes"]
    is used instead -- same fallback the integration layer asked for.
    """
    out = []
    for o in world.objs:
        if not o.get("label", True):
            continue
        ys, xs = np.where(obj_map == o["id"])
        px = int(len(ys))
        gb = target_boxes.get(o["name"])
        if px >= min_px:
            x0, y0, x1, y1, src = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1, "mask"
        elif gb is not None:
            x0, y0, x1, y1, src = int(gb[0]), int(gb[1]), int(gb[2]), int(gb[3]), "geometry"
        else:
            continue
        out.append(dict(id=int(o["id"]), name=o["name"], cls=o["cls"],
                        label=o["cls"], box=[x0, y0, x1, y1],
                        x0=float(x0), y0=float(y0), x1=float(x1), y1=float(y1),
                        pixels=px, source=src))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("-o", "--out", default="out/sar_check")
    ap.add_argument("-f", "--frame", type=int, default=0)
    ap.add_argument("--sensors", default="sar")
    a = ap.parse_args(argv)

    cfg = scene.load_config(a.config)
    st = weather.resolve(cfg.get("weather"))
    fr_cfg = json.loads(json.dumps(cfg))
    fr_cfg["weather"] = dict(st)
    fr_cfg["_weather"] = weather.derive(st)
    fr_cfg["_frame_seed"] = int(a.frame)
    t = float(a.frame) * float((cfg.get("frames") or {}).get("dt", 0.2))
    fr_cfg["_frame_t"] = t

    t0, c0 = time.time(), time.process_time()
    world = scene.build_world(fr_cfg, t=0.0)
    t_world, c_world = time.time() - t0, time.process_time() - c0
    t0, c0 = time.time(), time.process_time()
    out = sar.simulate(world, fr_cfg)
    t_sim, c_sim = time.time() - t0, time.process_time() - c0

    name = "%06d.png" % a.frame
    d_img = os.path.join(a.out, "images", "sar")
    d_lab = os.path.join(a.out, "labels", "sar")
    d_dbg = os.path.join(a.out, "sar_debug")
    for d in (d_img, d_lab, d_dbg):
        os.makedirs(d, exist_ok=True)
    png.write_gray8(os.path.join(d_img, name), out["image_u8"])
    # calibrated intensity, 16-bit so the dB values survive the round trip
    db = out["image_db"]
    lo = float(np.percentile(db, 1.0))
    q = np.clip((db - lo) / max(float(db.max()) - lo, 1e-9), 0.0, 1.0)
    png.write_gray16(os.path.join(d_dbg, "%06d_db16.png" % a.frame),
                     np.round(q * 65535.0).astype(np.uint16))

    boxes = _boxes(out["obj_map"], world, out["target_boxes"])
    lab = dict(frame=int(a.frame), sensor="sar", image=name,
               width=int(out["image_u8"].shape[1]), height=int(out["image_u8"].shape[0]),
               axes=dict(cols_uniform_in="slant_range",
                         range_m=[float(out["range_axis_m"][0]), float(out["range_axis_m"][-1])],
                         slant_m=[float(out["range_slant_axis_m"][0]),
                                  float(out["range_slant_axis_m"][-1])],
                         cross_m=[float(out["cross_axis_m"][0]), float(out["cross_axis_m"][-1])]),
               resolution=out["resolution"], swath=out["swath"],
               platform=out["platform"], stats=out["stats"],
               polarization=out["cfg"]["polarization"], band=out["cfg"]["band"],
               boxes=boxes, target_boxes=out["target_boxes"],
               objects=[dict(id=o["id"], name=o["name"], cls=o["cls"],
                             pos=[float(v) for v in o["pos"]])
                        for o in world.objs])
    labels.write_json(os.path.join(d_lab, "%06d.json" % a.frame), lab)

    # ---- report ---------------------------------------------------------
    r, s, p = out["resolution"], out["stats"], out["platform"]
    print("scene     %s  frame %d  t=%.2f s" % (a.config, a.frame, t))
    print("image     %dx%d  (%d rows azimuth x %d cols range)"
          % (out["image_u8"].shape[0], out["image_u8"].shape[1],
             out["image_u8"].shape[0], out["image_u8"].shape[1]))
    print("platform  pos=(%.1f, %.1f, %.1f)  v=%.1f m/s  H=%.0f m  heading %.0f deg  "
          "squint %.0f deg  look %s  pose %s"
          % (p["pos"][0], p["pos"][1], p["pos"][2], float(np.linalg.norm(p["vel"])),
             p["altitude_m"], p["heading_deg"], p["squint_deg"], p["side"],
             p["pose_source"]))
    print("resolution  range %.2f m ground (%.2f m slant = c/2B)   azimuth %.2f m (aperture %.1f m)"
          % (r["range_m"], r["range_slant_m"], r["azimuth_m"], r["l_sa_m"]))
    sw = out["swath"]
    print("swath     %.0f..%.0f m (width %.0f m, centre %.0f m), incidence %.1f deg, %.2f m/sample (%.1fx)"
          % (sw["near_m"], sw["far_m"], sw["width_m"], sw["center_m"],
             sw["incidence_deg"], sw["range_sampling_m"], sw["sampling_ratio"]))
    print("stats     peak %.1f dB  mean %.1f dB  NESZ/noise %.1f dB  SNR %.1f dB  "
          "dyn %.1f dB  n_scatterers %d"
          % (s["peak_db"], s["mean_db"], s["noise_db"], s["snr_db"],
             s["dynamic_range_db"], s["n_scatterers"]))
    print("target    peak %.1f dB  contrast %.1f dB  gain %.1f dBi  looks %d  cell %.2f m^2"
          % (s["target_peak_db"], s["target_contrast_db"], s["gain_db"], s["looks"],
             s["cell_area_m2"]))
    # geometric cross-check for every box
    hd, lh, side, sq = sar._platform_basis(out["cfg"])
    rs = out["range_slant_axis_m"]
    dR = float(rs[1] - rs[0])
    du = float(out["cross_axis_m"][1] - out["cross_axis_m"][0])
    for b in boxes:
        o = next(x for x in world.objs if x["id"] == b["id"])
        T = np.asarray(o["pos"], float)
        rel = T - np.asarray(p["pos"], float)
        u = float(rel @ hd)
        R = float(np.linalg.norm(rel))          # true slant range to the target
        row = int(np.argmin(np.abs(out["cross_axis_m"] - u)))
        col = int(np.argmin(np.abs(rs - R)))
        cx = 0.5 * (b["x0"] + b["x1"])
        cy = 0.5 * (b["y0"] + b["y1"])
        print("box %-12s x%d..%d y%d..%d  %d px (%s)  centre (row %.1f, col %.1f)  "
              "geom (row %d, col %d)  R=%.1f m"
              % (b["name"], b["x0"], b["x1"], b["y0"], b["y1"], b["pixels"], b["source"],
                 cy, cx, row, col, R))
        print("            offset %.1f px azimuth / %.1f px range (%.2f m / %.2f m)"
              % (cy - row, cx - col, (cy - row) * du, (cx - col) * dR))
    if not boxes:
        print("box       none (target not imaged)")
    # what the label box looked like *not* counting the fallback: if this is
    # empty while a box above exists, obj_map was too thin and the geometric
    # projection is what saved the annotation
    print("geom boxes (x0,y0,x1,y1) %s" % (out["target_boxes"] or "{}"))
    print("mask px   %s" % {o["name"]: int((out["obj_map"] == o["id"]).sum())
                            for o in world.objs if o.get("label", True)})
    print("timing    world %.2f s + sar %.2f s = %.2f s wall   (%.2f s CPU)   peak RSS %.0f MB"
          % (t_world, t_sim, t_world + t_sim, c_world + c_sim, _rss_mb()))
    print("files     %s  %s" % (os.path.join(d_img, name), os.path.join(d_lab, "%06d.json" % a.frame)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
