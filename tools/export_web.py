"""Export a run's outputs into the static web app's data folder.

Reads the per-run directories produced by `python -m simkit gen` and emits

    web/data/manifest.json      one index for every scene/frame/modality
    web/data/<scene>/...        downscaled copies of every image
    web/data/<scene>/cfg.json   the scene config that produced it

Nothing here needs a server: the site is pure static files, so the same folder
works on GitHub Pages and from a local `file://` open.
"""
import json
import os
import math
import shutil
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SCENES = [
    ("final_air", "空中 · Global Hawk", "uav", "Ka"),
    ("final_ground", "地面 · 巡视器", "vehicle", "W"),
    ("final_sea", "海面 · 帆船", "boat", "Ka"),
]

# Scene title/class/band lookup for runs exported by name. Anything not listed
# still exports; it just gets a generic title.
SCENE_META = {
    "final_air": ("空中 · Global Hawk", "uav", "Ka"),
    "ds_air_uav": ("空中 · 无人机", "uav", "Ka"),
    "ds_ground_vehicle": ("地面 · 巡视器", "vehicle", "W"),
    "ds_sea_boat": ("海面 · 船只", "boat", "Ka"),
    "ds1000_air": ("空中 · 无人机", "uav", "Ka"),
    "ds1000_ground": ("地面 · 巡视器", "vehicle", "W"),
    "ds1000_sea": ("海面 · 船只", "boat", "Ka"),
}
WEB_FRAMES = int(os.environ.get("SIMKIT_WEB_FRAMES", "0")) or None

MOD_IMG = [
    ("optical", "images/optical", "可见光 RGB"),
    ("mwir", "images/ir_preview_mwir", "中波红外 3–5 µm"),
    ("lwir", "images/ir_preview_lwir", "长波红外 8–12 µm"),
    ("radar_ra", "images/radar_ra", "雷达 · 距离-角度"),
    ("lidar_range", "lidar/range", "激光雷达 · 距离"),
    ("lidar_intensity", "lidar/intensity", "激光雷达 · 强度"),
]
MOD_LABEL = {
    "optical": "labels/optical",
    "mwir": "labels/ir_mwir",
    "lwir": "labels/ir_lwir",
    "radar_ra": None,
    "lidar_range": "labels/lidar",
}

# The range/Doppler map (images/radar) is a 1600x64 strip where the target is a
# couple of pixels; it is a legitimate dataset product but useless as a gallery
# picture, so it stays out of the site. The range/angle map is what reads.
WEB_MAX_W = 900


def read_ply(path, max_pts=1500):
    with open(path, "rb") as fh:
        raw = fh.read()
    end = raw.find(b"end_header")
    if end < 0:
        return []
    hdr = raw[:end].decode("ascii", "replace").splitlines()
    n = 0
    for line in hdr:
        if line.startswith("element vertex"):
            n = int(line.split()[-1])
    body = raw[end + len("end_header"):]
    body = body.lstrip(b"\r\n")
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "<f4"),
                   ("i", "<f4"), ("s", "<f4"), ("o", "<i4")])
    if len(body) < n * dt.itemsize:
        return []
    a = np.frombuffer(body[: n * dt.itemsize], dt)
    if n > max_pts:
        step = int(np.ceil(n / float(max_pts)))
        a = a[::step]
    q = lambda v, s: [int(round(float(x) * s)) for x in v]
    return dict(
        n=int(n), shown=int(len(a)),
        x=q(a["x"], 100), y=q(a["y"], 100), z=q(a["z"], 100),   # centimetres
        i=[int(round(float(v) * 255)) for v in a["i"]],
        o=[int(v) for v in a["o"]],
    )


def _shrink(src, dst, max_w=WEB_MAX_W):
    """Copy a gray PNG, box-averaging it down to max_w pixels wide.

    Radar range/angle maps are 1600x1920; shipped at full size they would be
    ~1.5 MB each for pictures that only ever get shown 600 px wide. Box
    averaging (rather than striding) keeps faint point targets from being
    dropped by the sampler.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from png_stats import read_png

    from simkit import png as pngw

    try:
        meta, pix, _pal = read_png(src)
    except Exception:
        shutil.copy(src, dst)
        return f_png_size(src)
    w, h = meta["w"], meta["h"]
    if meta["color"] != 0 or meta["depth"] != 8 or w <= max_w:
        shutil.copy(src, dst)
        return (w, h)
    a = np.frombuffer(pix, dtype=np.uint8).reshape(h, w).astype(np.float64)
    f = max(1, int(math.ceil(w / float(max_w))))
    ha, wa = (h // f) * f, (w // f) * f
    a = a[:ha, :wa].reshape(ha // f, f, wa // f, f).mean(axis=(1, 3))
    out = np.round(a).astype(np.uint8)
    pngw.write_gray8(dst, out)
    return (out.shape[1], out.shape[0])


def f_png_size(path):
    try:
        with open(path, "rb") as fh:
            head = fh.read(33)
        if head[:8] == b"\x89PNG\r\n\x1a\n":
            return struct.unpack(">II", head[16:24])
    except OSError:
        pass
    return (640, 480)


def radar_band_table():
    """Per-band link-budget constants, taken straight from the kernel.

    The site's live radar calculator reads this table instead of re-deriving
    the numbers, so the web numbers can never drift from simkit/radar.py.
    """
    import numpy as np

    from simkit import radar as rmod
    from simkit import weather as wmod

    rc = dict(rmod.DEFAULTS)
    k_b, t0, c = 1.380649e-23, 290.0, 2.99792458e8
    loss = 10.0 ** (float(rc["loss_db"]) / 10.0)
    nf = 10.0 ** (float(rc["noise_figure_db"]) / 10.0)
    bn = float(rc["n_samples"]) / (float(rc["chirp_us"]) * 1e-6)
    pn_dbw = 10.0 * np.log10(k_b * t0 * nf * bn * loss)
    pg_db = 10.0 * np.log10(max(int(rc["n_chirps"]), 1))
    out = {}
    for name, b in wmod.RADAR_BANDS.items():
        bw_az = float(b["beamwidth_az_deg"])
        bw_el = float(b["beamwidth_el_deg"])
        gain_db = 10.0 * np.log10(
            4.0 * np.pi * float(rc["antenna_eff"])
            / (np.radians(bw_az) * np.radians(bw_el)))
        out[name] = dict(
            fc_ghz=b["fc_ghz"], lam_m=c / (b["fc_ghz"] * 1e9),
            bw_az_deg=bw_az, bw_el_deg=bw_el,
            gain_db=round(float(gain_db), 3), pn_dbw=round(float(pn_dbw), 3),
            loss_db=float(rc["loss_db"]), pg_db=round(float(pg_db), 3),
            gas_db_km=b["gas_db_km"], rain_k=b["rain_k"], rain_a=b["rain_a"],
            tx_power_dbm=float(rc["tx_power_dbm"]),
            det_snr_db=float(rc["det_snr_db"]),
        )
    return out


def scene_budget(cfg_path, bands):
    """Target range vs max detection range for one scene, in dB of margin.

    This is the number to check a frame against: if margin < 0 the scene is
    outside the link budget and 'points 0' is physics, not a bug.
    """
    import numpy as np

    from simkit import radar as rmod
    from simkit import weather as wmod

    if not cfg_path or not os.path.exists(cfg_path):
        return None
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    user = (cfg.get("sensors") or {}).get("radar") or {}
    rc = dict(rmod.DEFAULTS)
    band = user.get("band")
    if band in wmod.RADAR_BANDS:
        for k in ("fc_ghz", "beamwidth_az_deg", "beamwidth_el_deg"):
            if k not in user:
                rc[k] = wmod.RADAR_BANDS[band][k]
    rc.update(user)
    bb = bands.get(band) if band else None
    if bb is None:
        bb = bands.get(wmod.radar_band_of(float(rc["fc_ghz"])))
    lam = bb["lam_m"]
    bw_az = np.radians(max(float(rc["beamwidth_az_deg"]),
                           float(rc["az_span_deg"]) / max(int(rc["n_az"]), 1)))
    bw_el = np.radians(max(float(rc["beamwidth_el_deg"]),
                           float(rc["el_span_deg"]) / max(int(rc["n_el"]), 1)))
    gain = 4.0 * np.pi * float(rc["antenna_eff"]) / (bw_az * bw_el)
    loss = 10.0 ** (float(rc["loss_db"]) / 10.0)
    nf = 10.0 ** (float(rc["noise_figure_db"]) / 10.0)
    bn = float(rc["n_samples"]) / (float(rc["chirp_us"]) * 1e-6)
    pn = 1.380649e-23 * 290.0 * nf * bn * loss
    pg = 10.0 * np.log10(max(int(rc["n_chirps"]), 1))

    tx_w = 10.0 ** ((float(rc["tx_power_dbm"]) - 30.0) / 10.0)

    def snr(rng, sigma, att):
        rng = max(float(rng), 1.0)
        pr = (tx_w * gain * gain * lam * lam * sigma
              / ((4.0 * np.pi) ** 3 * rng ** 4 * loss))
        pr *= 10.0 ** (-(att * 2.0 * rng / 1000.0) / 10.0)
        return 10.0 * np.log10(max(pr, 1e-300) / pn) + pg

    # the scene's own target: first labelled object
    tgt = None
    for o in cfg.get("objects") or []:
        if o.get("label", True) and o.get("name"):
            tgt = o
            break
    if tgt is None:
        return None
    pos = np.asarray(tgt.get("pos", [0, 0, 0]), float)
    rng = float(np.linalg.norm(pos - np.asarray(rc["pos"], float)))
    rcs_db = float(tgt.get("rcs_db", 0.0))
    sigma = 10.0 ** (rcs_db / 10.0)
    det = float(rc["det_snr_db"])

    out = dict(band=band, fc_ghz=bb["fc_ghz"], target=tgt["name"],
               range_m=round(rng, 1), rcs_db=rcs_db, det_snr_db=det,
               tx_power_dbm=float(rc["tx_power_dbm"]), n_chirps=int(rc["n_chirps"]),
               gain_dbi=round(float(10.0 * np.log10(gain)), 1), by_weather=[])
    for name, wcfg in (("clear", {"preset": "clear"}), ("cloudy", {"preset": "cloudy"}),
                       ("rain", {"preset": "rain"})):
        wd = wmod.derive(wmod.resolve(wcfg))
        att = float(wd["radar_att_db_km"].get(band or "", 0.0))
        # bisect for the max range at this weather
        lo, hi = 1.0, 500000.0
        if snr(lo, sigma, att) < det:
            rmax = 0.0
        elif snr(hi, sigma, att) >= det:
            rmax = hi
        else:
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if snr(mid, sigma, att) >= det:
                    lo = mid
                else:
                    hi = mid
            rmax = lo
        out["by_weather"].append(dict(
            preset=name, att_db_km=round(att, 4),
            max_range_m=round(rmax, 1),
            margin_db=round(float(snr(rng, sigma, att) - det), 2),
        ))
    return out


def main(dest, runs):
    data = os.path.join(dest, "data")
    if os.path.isdir(data):
        shutil.rmtree(data)
    os.makedirs(data, exist_ok=True)

    bands = radar_band_table()
    scenes = []
    wanted = [(s[0], s[1], s[2], s[3]) for s in SCENES if os.path.isdir(os.path.join(runs, s[0]))]
    if not wanted:
        # No canonical runs: export whatever the runs/ directory holds, which is
        # how the 1000-frame dataset gets published. A merged dataset package has
        # one directory per scene but a single scenes.json at its root instead of
        # a per-scene meta.json, so accept both.
        root_scenes = {}
        sp = os.path.join(runs, "scenes.json")
        if os.path.exists(sp):
            try:
                for s in json.load(open(sp, encoding="utf-8")):
                    root_scenes[s.get("name")] = s
            except (OSError, ValueError):
                root_scenes = {}
        for nm in sorted(os.listdir(runs)):
            p = os.path.join(runs, nm)
            if not os.path.isdir(p):
                continue
            if not (os.path.exists(os.path.join(p, "meta.json")) or nm in root_scenes):
                continue
            title, cls, band = SCENE_META.get(nm, (nm, nm, "Ka"))
            wanted.append((nm, title, cls, band))
    for scene_id, title, cls, band in wanted:
        src = os.path.join(runs, scene_id)
        if not os.path.isdir(src):
            print("skip (missing):", src)
            continue
        out = os.path.join(data, scene_id)
        os.makedirs(out, exist_ok=True)
        shutil.rmtree(out, ignore_errors=True)
        os.makedirs(out, exist_ok=True)

        cfg_src = None
        meta = {}
        mp = os.path.join(src, "meta.json")
        if not os.path.exists(mp) and scene_id in (locals().get("root_scenes") or {}):
            rec = root_scenes[scene_id]
            meta = dict(weather=rec.get("weather") or [],
                        config=rec.get("config"),
                        frames=[{"frame": i} for i in range(int(rec.get("stats", {}).get("frames", 0)) or 0)])
            mp = None
        if mp and os.path.exists(mp):
            meta = json.load(open(mp, encoding="utf-8"))
            cfg_src = meta.get("config")
        budget = None
        if cfg_src and os.path.exists(cfg_src):
            if not os.path.isabs(cfg_src):
                cfg_src = os.path.join(os.path.dirname(os.path.dirname(
                    os.path.abspath(__file__))), cfg_src)
            if os.path.exists(cfg_src):
                shutil.copy(cfg_src, os.path.join(out, "cfg.json"))

        n_frames = len(meta.get("weather") or []) or 6
        # a merged package carries every scene's frame count, not one weather
        # record per frame; when the request asks for more frames than there are
        # weather states the states simply repeat, which is what generation did.
        have = int((meta.get("stats") or {}).get("frames") or 0) if isinstance(meta.get("stats"), dict) else 0
        n_frames = max(n_frames, have)
        if WEB_FRAMES:
            n_frames = min(n_frames, int(WEB_FRAMES))
        frames = []
        for f in range(n_frames):
            name = "%06d.png" % f
            wl = meta.get("weather") or [{}]
            fr = dict(i=f, weather=wl[f % len(wl)], mods={})
            for mod, rel, label in MOD_IMG:
                sp = os.path.join(src, rel, name)
                if not os.path.exists(sp):
                    continue
                os.makedirs(os.path.join(out, mod), exist_ok=True)
                w, h = _shrink(sp, os.path.join(out, mod, name))
                entry = dict(img="%s/%s/%s" % (scene_id, mod, name))
                entry["w"], entry["h"] = w, h
                lb = MOD_LABEL.get(mod)
                if lb:
                    lp = os.path.join(src, lb, "%06d.json" % f)
                    if os.path.exists(lp):
                        d = json.load(open(lp, encoding="utf-8"))
                        entry["boxes"] = [
                            dict(x0=round(b["x0"], 1), y0=round(b["y0"], 1),
                                 x1=round(b["x1"], 1), y1=round(b["y1"], 1),
                                 label=next((o["name"] for o in d.get("objects", [])
                                             if o["id"] == oid), "target"))
                            for oid, b in sorted(d.get("boxes", {}).items())
                        ] if isinstance(d.get("boxes"), dict) else [
                            dict(x0=round(b["x0"], 1), y0=round(b["y0"], 1),
                                 x1=round(b["x1"], 1), y1=round(b["y1"], 1), label="target")
                            for b in d.get("boxes", [])
                        ]
                fr["mods"][mod] = entry

            rp = os.path.join(src, "labels/radar", "%06d.json" % f)
            if os.path.exists(rp):
                d = json.load(open(rp, encoding="utf-8"))
                # The shipped image may have been box-averaged down, so the peak
                # pixel from the kernel is rescaled into the shipped image's
                # coordinate space here rather than in the browser.
                pk = d.get("ra_peak")
                if pk and pk.get("w"):
                    ra = fr["mods"].get("radar_ra") or {}
                    sx = (ra.get("w") or pk["w"]) / float(pk["w"])
                    sy = (ra.get("h") or pk["h"]) / float(pk["h"])
                    pk = dict(px=pk["px"] * sx, py=pk["py"] * sy, db=pk["db"],
                              w=ra.get("w"), h=ra.get("h"))
                fr["radar"] = dict(
                    n=d.get("num_points", 0), clutter=d.get("clutter_count", 0),
                    boxes=d.get("boxes_3d", []), peak=pk,
                    pts=[[round(p["range"], 2), round(p["az"], 3), round(p["el"], 3),
                          round(p["snr_db"], 1), p["obj"]] for p in d.get("points", [])],
                )
            lp = os.path.join(src, "lidar/points", "%06d.ply" % f)
            if os.path.exists(lp):
                fr["cloud"] = read_ply(lp)
                lj = os.path.join(src, "labels/lidar", "%06d.json" % f)
                if os.path.exists(lj):
                    d = json.load(open(lj, encoding="utf-8"))
                    fr["cloud"]["hit"] = d.get("hit_ratio", 0)
                    fr["cloud"]["maxr"] = d.get("max_range_m", 0)
            frames.append(fr)

        budget = scene_budget(cfg_src if cfg_src and os.path.exists(cfg_src) else None, bands)
        scenes.append(dict(id=scene_id, title=title, cls=cls, band=band, budget=budget,
                           sensors=list(meta.get("sensors", [])),
                           categories=[c["name"] for c in meta.get("categories", [])],
                           frames=frames))

    with open(os.path.join(data, "radar_bands.json"), "w", encoding="utf-8") as fh:
        json.dump(bands, fh, ensure_ascii=False, indent=1, sort_keys=True)
    print("radar bands: " + ", ".join(
        "%s(fc=%.1f,G=%.1f)" % (k, v["fc_ghz"], v["gain_db"]) for k, v in sorted(bands.items())))

    man = dict(
        generated="simkit",
        scenes=scenes,
        radar_pts_max=800,
        radar_bands=list(bands.keys()),
    )
    with open(os.path.join(data, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(man, fh, ensure_ascii=False, separators=(",", ":"))

    # Double-click-to-run copy of the same payload. A browser refuses fetch() on
    # file:// URLs, but it will happily load a classic <script>; so the same data
    # is also shipped as one assignment that index.html includes before app.js.
    # That is what makes the site usable in an air-gapped machine with no server.
    with open(os.path.join(data, "boot.js"), "w", encoding="utf-8") as fh:
        fh.write("/* generated by tools/export_web.py - do not edit */\n")
        fh.write("window.SIMKIT_DATA = ")
        json.dump(dict(manifest=man, bands=bands), fh,
                  ensure_ascii=False, separators=(",", ":"))
        fh.write(";\n")
    total = 0
    for root, _, files in os.walk(data):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    print("web data: %d scenes, %.1f MB" % (len(scenes), total / 1048576.0))


if __name__ == "__main__":
    dest = sys.argv[1] if len(sys.argv) > 1 else "."
    runs = sys.argv[2] if len(sys.argv) > 2 else "out"
    main(dest, runs)