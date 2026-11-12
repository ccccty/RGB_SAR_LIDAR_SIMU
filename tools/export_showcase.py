# -*- coding: utf-8 -*-
"""Export the 2026-11-11 sea run + the 18 recipe wall into data/ for the site.

Source dataset:
  D:/work/多模态数据集生成/dataest/1110/seav2/p_sea_port_passive   (main run)
  D:/work/多模态数据集生成/dataest/1110/seav2/p_sea_port_active    (jammed twin)
  D:/work/多模态数据集生成/dataest/1110/{air,land,sea}/p_*         (18 recipes)

The radar RD panels are re-rendered here from raw/radar/*.npy instead of
shipping the 1280x720 matplotlib-free figures: the page draws its own axes,
hover readout and truth marker on top, which needs a bare power map with a
known pixel-to-metre mapping.
"""
import json
import os
import sys

import numpy as np
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from simkit.radar_scope import cmap_lut  # noqa: E402

SRC = r"D:/work/多模态数据集生成/dataest/1110"
RUN_ID = "p_sea_port_passive"
JAM_ID = "p_sea_port_active"
OUT = os.path.join(REPO, "data")

RD_W, RD_H = 896, 448
DB_WINDOW = 40.0
WEBP_Q = 78


def load(path):
    with open(path, "r", encoding="utf-8-sig") as fh:
        return json.load(fh)


def save_webp(img, rel, width, quality=WEBP_Q):
    dst = os.path.join(OUT, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    im = img.convert("RGB") if img.mode not in ("RGB", "RGBA") else img
    if width and im.width != width:
        h = int(round(im.height * width / float(im.width)))
        im = im.resize((width, h), Image.LANCZOS)
    im.save(dst, "WEBP", quality=quality, method=6)
    return dst


def rd_panel(npy_path, r_max, v_nyq):
    """Render one power map to a bare turbo panel; return (img, hi_db)."""
    arr = np.load(npy_path).astype(np.float64)
    db = 10.0 * np.log10(np.maximum(arr, 1e-30))
    # Same anchor as simkit.radar_scope: the display window starts at the noise
    # median, so half the speckle stays visible and the peak clips at the top.
    lo = float(np.median(db))
    hi = lo + DB_WINDOW
    norm = np.clip((db - lo) / DB_WINDOW, 0.0, 1.0)
    rgb = cmap_lut("turbo")[(norm * 255.0).astype(np.uint8)]
    # rows = doppler with +v at the top (row 0 is the image's top edge),
    # cols = range; the scope convention is closing speed pointing up.
    rgb = np.transpose(rgb, (1, 0, 2))[::-1]
    img = Image.fromarray(rgb, "RGB").resize((RD_W, RD_H), Image.BILINEAR)
    return img, hi, arr.shape


def peak_of(arr):
    k = np.unravel_index(int(np.argmax(arr)), arr.shape)
    return int(k[0]), int(k[1])


def collect(run_dir, tag, do_images=True):
    frames = load(os.path.join(run_dir, "frames.json"))
    ann = load(os.path.join(run_dir, "annotations_radar.json"))
    scene = load(os.path.join(SRC, "seav2", "scenes", tag + ".json"))
    boxes = {}
    coco_path = os.path.join(run_dir, "annotations_optical_coco.json")
    if os.path.exists(coco_path):
        for a in (load(coco_path).get("annotations") or []):
            bb = a.get("bbox") or []
            if len(bb) == 4:
                boxes[int(a.get("image_id"))] = [round(float(v), 1) for v in bb]
    rz = scene["sensors"]["radar"]
    band = rz.get("band", "Ku")
    fc = {"Ku": 15.0, "X": 10.0, "K": 24.0, "Ka": 35.0, "W": 77.0, "W94": 94.0}.get(band, 15.0)
    v_res = 2.0 * rz["v_nyq_mps"] / float(rz["n_chirps"])
    radar = {
        "band": band, "fc_ghz": fc,
        "r_max_m": rz["r_max_m"], "v_nyq_mps": rz["v_nyq_mps"],
        "range_res_m": rz["range_res_m"], "v_res_mps": round(v_res, 3),
        "tx_dbm": rz.get("tx_power_dbm"), "n_chirps": rz["n_chirps"],
        "bw_mhz": rz.get("bandwidth_mhz"), "n_samples": rz["n_samples"],
    }
    out_frames = []
    shapes = set()
    for i, fr in enumerate(frames):
        im = ann["images"][i]
        npy = os.path.join(run_dir, "raw", "radar", "%06d.npy" % i)
        arr = np.load(npy) if os.path.exists(npy) else None
        row = {
            "i": i,
            "t": round(float(fr.get("t", 0.0)), 2),
            "range_m": round(float(fr.get("range_m", 0.0)), 1),
            "px": round(float(fr.get("target_px") or 0.0), 1),
            "snr": round(float(im.get("snr_max") or 0.0), 2),
            "npts": int(im.get("n_points") or 0),
            "targets": [
                {"name": t.get("name"), "cls": t.get("cls"),
                 "r": round(float(t.get("range") or 0.0), 1),
                 "v": round(float(t.get("v_r") or 0.0), 2),
                 "az": round(float(t.get("az") or 0.0), 2),
                 "rcs": round(float(t.get("rcs_dbsm") or 0.0), 1),
                 "det": bool(t.get("detected"))}
                for t in (im.get("targets") or [])
            ],
        }
        if i in boxes:
            row["box"] = boxes[i]
        if arr is not None:
            bi, bj = peak_of(arr)
            shapes.add(arr.shape)
            nbins = arr.shape[0]
            dbv = 10.0 * np.log10(np.maximum(arr, 1e-30))
            hi = float(np.median(dbv)) + DB_WINDOW
            row["peak"] = {
                "r": round((bi + 0.5) / nbins * rz["r_max_m"], 1),
                "v": round((bj + 0.5) / arr.shape[1] * 2.0 * rz["v_nyq_mps"] - rz["v_nyq_mps"], 2),
                "db": round(hi, 1),
            }
            row["db_hi"] = round(hi, 1)
            row["db_lo"] = round(hi - DB_WINDOW, 1)
        out_frames.append(row)

    if do_images:
        for i in range(len(frames)):
            pad = "%06d" % i
            save_webp(Image.open(os.path.join(run_dir, "images", "optical", pad + ".png")),
                      "run/%s/optical/%s.webp" % (tag, pad), 768)
            save_webp(Image.open(os.path.join(run_dir, "images", "ir_preview_lwir", pad + ".png")),
                      "run/%s/lwir/%s.webp" % (tag, pad), 768)
            save_webp(Image.open(os.path.join(run_dir, "images", "ir_preview_mwir", pad + ".png")),
                      "run/%s/mwir/%s.webp" % (tag, pad), 768)
            if os.path.exists(os.path.join(run_dir, "raw", "radar", pad + ".npy")):
                img, _hi, _sh = rd_panel(os.path.join(run_dir, "raw", "radar", pad + ".npy"),
                                         rz["r_max_m"], rz["v_nyq_mps"])
                save_webp(img, "run/%s/rd/%s.webp" % (tag, pad), RD_W, quality=70)
        save_webp(Image.open(os.path.join(run_dir, "images", "optical", "000012.png")),
                  "run/%s/strip/lead.webp" % tag, 1024)
        for i in range(len(frames)):
            pad = "%06d" % i
            save_webp(Image.open(os.path.join(run_dir, "images", "optical", pad + ".png")),
                      "run/%s/strip/%s.webp" % (tag, pad), 176, quality=66)

    # Numeric parameters of each ECM entity live in the recipe, not in meta.json;
    # the note strings in that file are mojibake, so only numbers are copied.
    cfg_ecm = {}
    for e in ((scene.get("recipe") or {}).get("ecm") or {}).get("entities") or []:
        if isinstance(e, dict) and e.get("id"):
            cfg_ecm[e["id"]] = {k: v for k, v in e.items()
                                if k != "note" and isinstance(v, (int, float, str, bool))}
    ecm = load(os.path.join(run_dir, "meta.json")).get("ecm") or {}
    per_frame = ecm.get("per_frame") or []
    mid = per_frame[len(per_frame) // 2] if per_frame else {}
    at_frame = {}
    for ent in (mid.get("entities") or []):
        if isinstance(ent, dict) and ent.get("id"):
            at_frame[ent["id"]] = ent
    for jm in (mid.get("jammers") or []):
        if isinstance(jm, dict) and jm.get("id"):
            at_frame.setdefault(jm["id"], {}).update({"range_m": jm.get("range_m")})
    entities = []
    for e in (ecm.get("entities") or []):
        entities.append({
            "id": e.get("id"), "kind": e.get("kind"), "group": e.get("group"),
            "count": e.get("count"), "band": e.get("band"),
            "erp_w": e.get("erp_w"), "mode": e.get("mode"),
            "rcs_m2": e.get("rcs_m2"),
            "cfg": cfg_ecm.get(e.get("id")) or {},
            "at": at_frame.get(e.get("id")),
        })
    for e in entities:
        a = e.get("at") or {}
        e["at"] = {
            "r": round(float(a.get("range_m") or 0.0), 1) if a.get("range_m") is not None else None,
            "v": round(float(a.get("v_r_mps") or 0.0), 1) if a.get("v_r_mps") is not None else None,
            "rcs": round(float(a.get("rcs_m2") or 0.0), 1) if a.get("rcs_m2") is not None else None,
            "det": bool(a.get("detected")),
        } if a else None
    return {
        "id": tag, "frames": len(frames), "radar": radar,
        "rd": {"w": RD_W, "h": RD_H, "db_window": DB_WINDOW,
               "r_max_m": rz["r_max_m"], "v_nyq_mps": rz["v_nyq_mps"]},
        "series": out_frames,
        "ecm": {
            "model": ecm.get("model"), "domain": ecm.get("domain"),
            "js_db": span(ecm.get("js_db")),
            "noise_rise_db": span(ecm.get("noise_rise_db")),
            "drop_db": drop_series(ecm.get("per_frame")),
            "injected_echoes": ecm.get("injected_echoes"),
            "entities": entities,
        },
        "tensor": sorted(shapes),
    }


def weather_of(run_dir):
    fr = load(os.path.join(run_dir, "frames.json"))
    w = fr[0].get("weather") or {}
    return {k: w.get(k) for k in ("cloud_cover", "visibility_m", "rain_mmh", "wind_ms",
                                  "air_temp_c", "rh", "sun_elev_deg", "sea_state", "name")}


def span(v):
    """ecm fields arrive as a scalar or as {"min","max"}; -999 means absent."""
    if isinstance(v, dict):
        lo, hi = v.get("min"), v.get("max")
    else:
        lo = hi = v
    try:
        lo, hi = float(lo), float(hi)
    except (TypeError, ValueError):
        return None
    if lo <= -900:
        return None
    return [round(lo, 1), round(hi, 1)]


def drop_series(per_frame):
    """Per-frame measured ECM effect: SNR drop (dB) and injected echo count."""
    out = []
    for f in (per_frame or []):
        if not isinstance(f, dict):
            continue
        noise = f.get("noise") or {}
        out.append({
            "i": f.get("frame"), "t": round(float(f.get("t") or 0.0), 2),
            "snr_drop_db": round(float(noise.get("snr_drop_db") or 0.0), 1),
            "js_db": round(float(noise.get("js_db") or 0.0), 1),
            "echoes": f.get("injected_echoes") or 0,
            "false": f.get("false_detections") or 0,
        })
    return out


def recipes():
    out = []
    for dom in ("air", "land", "sea"):
        root = os.path.join(SRC, dom)
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            d = os.path.join(root, name)
            if not name.startswith("p_") or not os.path.isdir(d):
                continue
            cfg = os.path.join(SRC, "scenes", name + ".json")
            env, ecm, view = {}, {}, {}
            if os.path.exists(cfg):
                sc = load(cfg)
                rec = sc.get("recipe") or {}
                env = rec.get("environment") or {}
                ecm = rec.get("ecm") or {}
                view = rec.get("view") or {}
            frames = load(os.path.join(d, "frames.json"))
            meta = load(os.path.join(d, "meta.json"))
            band = meta.get("radar_band") or "Ku"
            pad = "000002"
            src = os.path.join(d, "images", "optical", pad + ".png")
            thumb = "recipes/%s.webp" % name
            if os.path.exists(src):
                save_webp(Image.open(src), thumb, 620, quality=74)
            out.append({
                "id": name, "domain": dom,
                "backdrop": (env.get("background") or {}).get("file") or "",
                "category": (env.get("background") or {}).get("category") or "",
                "weather": env.get("weather") or "",
                "ecm": ecm.get("mode") or ("active" if name.endswith("_active") else "passive"),
                "band": band,
                "range_m": round(float(frames[0].get("range_m") or 0.0)),
                "px": round(float(frames[0].get("target_px") or 0.0), 1),
                "frames": len(frames),
                "thumb": thumb,
                "fov_deg": view.get("fov_deg"),
                "aspect": view.get("aspect"),
            })
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    run = collect(os.path.join(SRC, "seav2", RUN_ID), RUN_ID)
    jam = collect(os.path.join(SRC, "seav2", JAM_ID), JAM_ID)
    run["weather"] = weather_of(os.path.join(SRC, "seav2", RUN_ID))
    jam["weather"] = run["weather"]
    rec = recipes()
    doc = {
        "generated": "2026-11-12",
        "source": "dataest/1110 seav2 (2026-11-11) + 18 recipe presets",
        "run": run, "jam": jam, "recipes": rec,
        "counts": {"recipes": len(rec), "frames": run["frames"]},
    }
    dst = os.path.join(OUT, "showcase.json")
    body = json.dumps(doc, ensure_ascii=False, separators=(",", ":"))
    with open(dst, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(body)
    # A script copy so the page also works from file:// where fetch() is blocked.
    with open(os.path.join(OUT, "showcase.js"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("window.SIMKIT_SHOWCASE=" + body + ";\n")
    print("wrote", dst, os.path.getsize(dst), "bytes;", len(rec), "recipes;",
          run["frames"], "frames; tensor", run["tensor"])
    print("run snr", run["series"][0]["snr"], "->", run["series"][-1]["snr"])
    print("jam snr sample", [jam["series"][i]["snr"] for i in (0, 5, 12, 20)])


if __name__ == "__main__":
    main()
