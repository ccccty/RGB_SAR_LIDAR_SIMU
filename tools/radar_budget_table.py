"""Radar range budget table: max detection range vs rain rate and band.

Uses the same radar equation and weather state machine as simkit.radar, but
solves for range instead of tracing beams -- this is the number you check a
simulated frame against.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from simkit import radar as radar_mod  # noqa: E402
from simkit import weather  # noqa: E402

C = 299792458.0
K_BOLTZ = 1.380649e-23
T0 = 290.0


def setup(cfg, band_name, rain_mmh):
    rc = dict(radar_mod.DEFAULTS)
    rc.update((cfg.get("sensors") or {}).get("radar") or {})
    w = weather.resolve({"preset": "rain" if rain_mmh > 0 else "clear",
                         "rain_mmh": max(rain_mmh, 0.0)})
    wd = weather.derive(w)
    band = weather.RADAR_BANDS[band_name]
    lam = C / (band["fc_ghz"] * 1e9)
    # Per-band antenna gain from that band's own beamwidths -- the kernel does
    # the same in radar.simulate(), so the table matches what frames show.
    gain = (4.0 * np.pi * float(rc["antenna_eff"])
            / (np.radians(band["beamwidth_az_deg"])
               * np.radians(band["beamwidth_el_deg"])))
    loss = 10.0 ** (float(rc["loss_db"]) / 10.0)
    nf = 10.0 ** (float(rc["noise_figure_db"]) / 10.0)
    bn = float(rc["n_samples"]) / (float(rc["chirp_us"]) * 1e-6)
    pn = K_BOLTZ * T0 * nf * bn * loss
    return dict(rc=rc, lam=lam, gain=gain, loss=loss, pn=pn, wd=wd, band=band)


def snr_at(s, r_m, sigma_m2, att_db_km):
    pt = 10.0 ** ((float(s["rc"]["tx_power_dbm"]) - 30.0) / 10.0)
    pr = (pt * s["gain"] ** 2 * s["lam"] ** 2 * sigma_m2
          / ((4.0 * np.pi) ** 3 * r_m ** 4 * s["loss"]))
    pr *= 10.0 ** (-(att_db_km * 2.0 * r_m / 1000.0) / 10.0)
    pg = 10.0 * np.log10(max(int(s["rc"]["n_chirps"]), 1))
    return 10.0 * np.log10(pr / s["pn"]) + pg


def main():
    scene_path = sys.argv[1] if len(sys.argv) > 1 else "scenes/demo_radar_2km.json"
    cfg = json.load(open(os.path.join(ROOT, scene_path), encoding="utf-8"))
    sigma = 10.0 ** (-10.0 / 10.0)          # the 1.6 m vehicle box
    det = float(cfg["sensors"]["radar"].get("det_snr_db", 12.0))
    print("detection threshold %.0f dB, target sigma = -10 dBsm (%.2f m^2)"
          % (det, sigma))
    print("max detection range (km), margin over threshold in brackets")
    bands = ("X", "Ku", "K", "Ka", "W", "W94")
    print("| rain mm/h | " + " | ".join("%s %.0f GHz" % (b, weather.RADAR_BANDS[b]["fc_ghz"])
                                        for b in bands) + " |")
    print("|---|" + "---|" * len(bands))
    for rain in (0.0, 1.0, 4.0, 16.0, 64.0):
        row = []
        for b in bands:
            s = setup(cfg, b, rain)
            att = s["wd"]["radar_att_db_km"].get(b, 0.0)
            # bisection on range
            lo, hi = 10.0, 60000.0
            if snr_at(s, lo, sigma, att) < det:
                row.append("--")
                continue
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if snr_at(s, mid, sigma, att) >= det:
                    lo = mid
                else:
                    hi = mid
            row.append("%.1f" % (lo / 1000.0))
        print("| %.0f | %s |" % (rain, " | ".join(row)))
    print("")
    print("per-band one-way attenuation dB/km (gas + rain):")
    for rain in (0.0, 4.0, 16.0, 64.0):
        wd = weather.derive(weather.resolve({"preset": "rain" if rain else "clear",
                                             "rain_mmh": rain}))
        print("  rain %5.0f mm/h: " % rain + "  ".join(
            "%s %.3f" % (b, wd["radar_att_db_km"][b]) for b in bands))


main()
