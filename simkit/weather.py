"""Weather state machine: one state vector drives all three sensor modalities.

State = 8 scalars: cloud_cover, visibility_m, rain_mmh, wind_ms, air_temp_c,
rh, sun_elev_deg, sea_state.  derive() turns it into the physical quantities
each modality consumes, so that "rain" means the same thing in optics, IR and
radar at the same frame.  Formulas are documented approximations (Koschmieder,
Swinbank, ITU-R P.676/P.838, Marshall-Palmer); tunable by design.
"""
import math

FIELDS = ("cloud_cover", "visibility_m", "rain_mmh", "wind_ms",
          "air_temp_c", "rh", "sun_elev_deg", "sea_state")

DEFAULTS = dict(cloud_cover=0.1, visibility_m=20000.0, rain_mmh=0.0,
                wind_ms=5.0, air_temp_c=25.0, rh=0.5, sun_elev_deg=55.0,
                sea_state=0.0)

PRESETS = {
    "clear":  dict(cloud_cover=0.05, visibility_m=20000.0, rain_mmh=0.0),
    "cloudy": dict(cloud_cover=0.85, visibility_m=10000.0, rain_mmh=0.0),
    "rain":   dict(cloud_cover=0.95, visibility_m=4000.0,  rain_mmh=8.0),
    "fog":    dict(cloud_cover=0.6,  visibility_m=800.0,   rain_mmh=0.0),
    "storm":  dict(cloud_cover=1.0,  visibility_m=1500.0,  rain_mmh=32.0),
}

IR_BANDS = dict(mwir=[3.0, 5.0], lwir=[8.0, 12.0], swir=[1.0, 3.0])
IR_TAU_CLEAR = dict(mwir=0.10, lwir=0.06, swir=0.15)   # per km, mid-latitude approx

# fc, default beamwidths, gas attenuation (ITU-R P.676 approx, dB/km),
# rain attenuation gamma = k * R^alpha dB/km (ITU-R P.838-3, h-pol approx).
RADAR_BANDS = {
    "X":   dict(fc_ghz=9.5,  beamwidth_az_deg=1.5, beamwidth_el_deg=5.0,  gas_db_km=0.008, rain_k=0.0101, rain_a=1.276),
    "Ku":  dict(fc_ghz=15.0, beamwidth_az_deg=1.0, beamwidth_el_deg=4.0,  gas_db_km=0.015, rain_k=0.0367, rain_a=1.154),
    "K":   dict(fc_ghz=24.0, beamwidth_az_deg=0.8, beamwidth_el_deg=3.0,  gas_db_km=0.020, rain_k=0.090,  rain_a=1.065),
    "Ka":  dict(fc_ghz=35.0, beamwidth_az_deg=0.5, beamwidth_el_deg=2.0,  gas_db_km=0.075, rain_k=0.187,  rain_a=1.021),
    "W":   dict(fc_ghz=77.0, beamwidth_az_deg=2.0, beamwidth_el_deg=12.0, gas_db_km=0.40,  rain_k=0.90,   rain_a=0.86),
    "W94": dict(fc_ghz=94.0, beamwidth_az_deg=1.0, beamwidth_el_deg=8.0,  gas_db_km=0.45,  rain_k=1.40,   rain_a=0.81),
}


def resolve(wcfg):
    """Merge a weather config: preset name, partial dict, or full dict."""
    if wcfg is None:
        wcfg = {}
    if isinstance(wcfg, str):
        wcfg = dict(preset=wcfg)
    base = dict(DEFAULTS)
    if wcfg.get("preset") in PRESETS:
        base.update(PRESETS[wcfg["preset"]])
    base.update({k: float(v) for k, v in wcfg.items() if k in FIELDS})
    if wcfg.get("name"):
        base["name"] = wcfg["name"]
    elif wcfg.get("preset"):
        base["name"] = wcfg["preset"]
    return base


def states_of(cfg):
    """Per-frame weather list: single state, or {'states': [...]} cycled."""
    w = cfg.get("weather")
    if isinstance(w, dict) and isinstance(w.get("states"), list):
        return [resolve(s) for s in w["states"]]
    return [resolve(w)]


def derive(w):
    """Weather state -> physical quantities consumed by every modality."""
    cloud = min(max(float(w["cloud_cover"]), 0.0), 1.0)
    rain = max(float(w["rain_mmh"]), 0.0)
    vis_km = max(float(w["visibility_m"]) / 1000.0, 0.05)
    rh = min(max(float(w["rh"]), 0.0), 1.0)
    wind = max(float(w["wind_ms"]), 0.0)
    T_air = float(w["air_temp_c"]) + 273.15
    sin_el = max(math.sin(math.radians(float(w["sun_elev_deg"]))), 0.0)

    # solar: documented clear-sky approximation with cloud/diffuse split
    if sin_el <= 0.01:
        G_dir = G_dif = 0.0
    else:
        m = 1.0 / sin_el
        G_dir = 1361.0 * 0.75 ** (m ** 0.678) * sin_el
        G_dif = 0.15 * 1361.0 * sin_el * (0.3 + 0.7 * cloud)
    solar = G_dir * (1.0 - 0.85 * cloud ** 2.5) + G_dif

    # sky long-wave temperature: Swinbank clear sky blended toward cloud base
    t_clear = 0.0552 * T_air ** 1.5 * (1.0 - 0.25 * rh)
    t_sky = t_clear + (T_air - 2.0 - t_clear) * cloud

    wetness = min(1.0, rain / 4.0)

    beta_opt = 3.912 / vis_km  # Koschmieder extinction, per km

    # 905 nm extinction: aerosol scattering scales as lambda^-1.3 from the
    # Koschmieder value at 550 nm (so a little weaker), rain drops are close to
    # neutral but much stronger per mm/h.  Natural units (1/km), like beta_opt.
    beta_lidar = 0.8 * beta_opt + 0.072 * rain ** 0.72

    tau_ir = {}
    vis_scale = (20.0 / vis_km) ** 0.3
    for b, tau0 in IR_TAU_CLEAR.items():
        tau_ir[b] = tau0 * (1.0 + 0.4 * rh) * vis_scale + 0.02 * rain ** 0.8

    att = {}
    for name, p in RADAR_BANDS.items():
        att[name] = p["gas_db_km"] + p["rain_k"] * rain ** p["rain_a"]

    return dict(
        cloud_cover=cloud, rain_mmh=rain, visibility_km=vis_km,
        solar_w_m2=solar, solar_factor=solar / 700.0,
        t_sky_k=t_sky, wetness=wetness, t_air_k=T_air,
        beta_opt_per_km=beta_opt, beta_lidar_per_km=beta_lidar, tau_ir_per_km=tau_ir,
        radar_att_db_km=att,
        # optical backend helpers
        sun_factor_opt=max(0.02, (1.0 - 0.9 * cloud) * (0.15 + 0.85 * sin_el)),
        turbidity=2.0 + 7.0 * cloud + 2.0 * min(1.0, rain / 16.0),
    )


def band_name_of(band_um):
    for k, v in IR_BANDS.items():
        if abs(band_um[0] - v[0]) < 1e-9 and abs(band_um[1] - v[1]) < 1e-9:
            return k
    return None


def radar_band_of(fc_ghz):
    best, bd = None, 1e9
    for k, p in RADAR_BANDS.items():
        d = abs(p["fc_ghz"] - fc_ghz)
        if d < bd:
            best, bd = k, d
    return best
