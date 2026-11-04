
"""Run the ECM demo scene and record what the jamming actually did.

Writes, under out/ecm_demo_fighter/report/:
  frames.json      per-frame numbers for the clean and jammed passes
  rd_clean.png     range-Doppler map with no ECM
  rd_ecm.png       the same frame with the jammer, repeater and chaff on
  rd_diff.png      the difference, so the injected cells are visible on their own
Everything here is measured by running the forward models; nothing is asserted
about a picture.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from PIL import Image
from simkit import radar, scene, lidar, labels

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "out", "ecm_demo_fighter")
REP = os.path.join(OUT, "report")
os.makedirs(REP, exist_ok=True)


def frame_cfg(t, on):
    cfg = scene.load_config(os.path.join(BASE, "scenes", "ecm_demo_fighter.json"))
    cfg["_ecm_t"] = float(t)
    if not on:
        cfg["ecm"] = dict(enabled=False, entities=[])
    return cfg


def png(path, array, lo=None, hi=None):
    """Save an RD map as a picture you can read numbers off.

    The cube is linear power and spans five decades, so a linear grey ramp shows
    one white dot and a black field. It is drawn in dB relative to its own peak,
    which is how a range-Doppler display is actually read, and transposed so
    range runs left to right.
    """
    a = np.asarray(array, float).T
    a = 10.0 * np.log10(np.maximum(a, 1e-30))
    hi = float(a.max()) if hi is None else hi
    lo = hi - 60.0 if lo is None else lo
    g = np.clip((a - lo) / (hi - lo) if hi > lo else a * 0.0, 0.0, 1.0)
    Image.fromarray((g * 255.0).astype(np.uint8)).resize(
        (a.shape[1] * 2, a.shape[0] * 8), Image.NEAREST).save(path)


rows, first = [], None
for f in range(6):
    t = f * 0.25
    row = dict(frame=f, t=t)
    maps = {}
    for tag, on in (("clean", False), ("ecm", True)):
        cfg = frame_cfg(t, on)
        w = scene.build_world(cfg, t=0.0)
        out = radar.simulate(w, cfg)
        rd = np.asarray(out["rd_map"], float)
        det = out["detections"]
        maps[tag] = rd
        lo = lidar.simulate(w, cfg)
        row[tag] = dict(
            noise_floor_w=float(out["pn"]),
            noise_floor_peak_w=float(np.max(out["pn_beam"])),
            detections=len(det),
            detections_on_target=sum(1 for d in det if d["obj"] >= 0),
            detections_injected=sum(1 for d in det if d.get("source")),
            max_rd_power=float(rd.max()),
            peak_snr_db=round(float(max([d["snr_db"] for d in det], default=0.0)), 3),
            lidar_echo_rate=round(float(lo["hit_ratio"]), 5),
            lidar_points=int(lo["n_points"]),
            lidar_ecm_points=int(np.sum(np.isin(lo["cloud"]["obj"],
                                                list(getattr(w, "ecm_ids", ()))))),
            world_objects=[o["name"] for o in w.objs],
            triangles=int(len(w.faces)))
        if on and out["ecm"]:
            e = out["ecm"]
            row["ecm_report"] = dict(
                noise_rise_db=round(float(e["noise"]["snr_drop_db"]), 3),
                jammer_to_signal_db=round(float(e["noise"]["js_db"]), 3),
                echoes_injected=int(e["injected_echoes"]),
                deception_copies=int(e["deception_copies"]),
                injected_detections=len(e["entities"][1].get("injected") or []),
                entities=[dict(id=x["id"], kind=x["kind"],
                               range_m=x.get("range_m"), rcs_m2=x.get("rcs_m2"),
                               extent_m=x.get("extent_m"),
                               copies=len(x.get("injected") or []))
                          for x in e["entities"]])
        if tag == "ecm" and f == 0:
            first = (rd, maps["clean"])
        if tag == "clean" and f == 0:
            clean0 = rd
    rows.append(row)

diff = first[0] - clean0
labels.write_json(os.path.join(REP, "frames.json"), rows)
labels.write_json(os.path.join(REP, "summary.json"), dict(
    scene="scenes/ecm_demo_fighter.json",
    frames=len(rows),
    note="clean and jammed passes run through the same forward models; only the ecm block differs",
    clean=dict(detections=rows[0]["clean"]["detections"],
               noise_floor_w=rows[0]["clean"]["noise_floor_w"],
               peak_snr_db=rows[0]["clean"]["peak_snr_db"]),
    jammed=dict(detections=rows[0]["ecm"]["detections"],
                noise_floor_peak_w=rows[0]["ecm"]["noise_floor_peak_w"],
                noise_rise_db=rows[0]["ecm_report"]["noise_rise_db"],
                injected_detections=rows[0]["ecm_report"]["injected_detections"],
                chaff_rcs_t0=rows[0]["ecm_report"]["entities"][2]["rcs_m2"],
                chaff_rcs_t_end=rows[-1]["ecm_report"]["entities"][2]["rcs_m2"],
                chaff_range_growth_m=round(
                    rows[-1]["ecm_report"]["entities"][2]["range_m"]
                    - rows[0]["ecm_report"]["entities"][2]["range_m"], 1))))

# every panel shares one dB window so the three are directly comparable
peak = float(max(np.max(clean0), np.max(first[0])))
top = 10.0 * np.log10(max(peak, 1e-30))
png(os.path.join(REP, "rd_clean.png"), clean0, top - 60.0, top)
png(os.path.join(REP, "rd_ecm.png"), first[0], top - 60.0, top)
png(os.path.join(REP, "rd_diff.png"), np.maximum(diff, 0.0), top - 60.0, top)
print("wrote", REP)
