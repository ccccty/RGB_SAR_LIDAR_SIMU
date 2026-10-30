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



# ---------------------------------------------------------------------------
# Clip timeline
# ---------------------------------------------------------------------------
def _lerp_state(a, b, u):
    out = dict(a)
    for k in FIELDS:
        out[k] = float(a[k]) + (float(b[k]) - float(a[k])) * u
    if a.get("name"):
        out["name"] = a["name"]
    return out


def clip_states(cfg, n_frames, dt, rng):
    """Weather for one clip, as a continuous timeline of length n_frames.

    Modes
    -----
    fixed     one resolved state for the whole run (a per-weather dataset)
    per_clip  every clip draws its own preset from "pool" -- one dataset,
              many skies, and each clip internally consistent
    evolve    start at one preset and drift smoothly to another inside the clip
    cycle     legacy: cycle an explicit "states" list frame by frame
    """
    spec = cfg.get("weather")
    if not isinstance(spec, dict):
        spec = {"preset": spec} if spec else {}
    mode = str(spec.get("mode", "fixed")).lower()
    # Backwards compatibility: older scenes carry a bare "states" list and no
    # mode. Dropping them to the default clear sky would silently change what
    # those configs render, so they keep the legacy per-frame cycle.
    if not spec.get("mode") and isinstance(spec.get("states"), list):
        mode = "cycle"
    pool = spec.get("pool") or ([spec.get("preset")] if spec.get("preset") else ["clear"])
    pool = [p for p in pool if p]

    if mode == "cycle" and isinstance(spec.get("states"), list):
        st = [resolve(s) for s in spec["states"]]
        return [dict(st[i % len(st)]) for i in range(n_frames)]

    if mode == "evolve" and pool:
        a = resolve({"preset": pool[int(rng.integers(0, len(pool)))]})
        b = resolve({"preset": pool[int(rng.integers(0, len(pool)))]})
        if int(rng.integers(0, 3)) == 0:
            b = a                      # sometimes hold one sky for the clip
        out = []
        for i in range(n_frames):
            u = 0.0 if n_frames <= 1 else i / float(n_frames - 1)
            u = u * u * (3.0 - 2.0 * u)          # smoothstep, C1 continuous
            out.append(_lerp_state(a, b, u))
        return out

    if mode == "per_clip":
        base = resolve({"preset": pool[int(rng.integers(0, len(pool)))]})
        drift = dict(base)
        for k, amp in (("sun_elev_deg", 2.0), ("wind_ms", 0.8), ("air_temp_c", 0.6)):
            drift[k] = float(base[k]) + float(rng.uniform(-amp, amp))
        out = []
        for i in range(n_frames):
            u = 0.0 if n_frames <= 1 else i / float(n_frames - 1)
            out.append(_lerp_state(base, drift, u))
        return out

    base = resolve(spec)
    return [dict(base) for _ in range(n_frames)]


def _kim_q(vis_km):
    """Aerosol size exponent, Kim et al. 2001 piecewise fit for 550 nm."""
    v = max(float(vis_km), 1e-3)
    if v > 50.0:
        return 1.6
    if v > 6.0:
        return 1.3
    if v > 1.0:
        return 0.16 * v + 0.34
    if v > 0.5:
        return v - 0.5
    return 0.0


def _water_vapour(w, t_c, rh):
    """Precipitable water [cm] and the per-km vapour density [cm/km]."""
    e = max(rh, 0.0) * 6.112 * math.exp(17.62 * t_c / (243.12 + t_c))   # hPa
    w_km = 21.67 * e / max(t_c + 273.15, 1.0)                           # cm/km
    return w_km * 2.0, w_km                                             # ~2 km scale height


def _sky_emissivity(t_c, rh, cloud):
    """Clark-Allen clear-sky emissivity with the Walton/EnergyPlus cloud term."""
    td = t_c - (100.0 - min(max(rh, 1e-3), 1.0) * 100.0) / 5.0          # dew point [C]
    eps = 0.787 + 0.764 * math.log(max(td + 273.15, 1.0) / 273.0)
    n = 10.0 * min(max(cloud, 0.0), 1.0)
    eps *= (1.0 + 0.0224 * n - 0.0035 * n ** 2 + 0.00028 * n ** 3)
    return min(max(eps, 0.3), 0.99)


# Band coupling of the sky: the sky is NOT a blackbody at its broadband
# temperature. In 3-5 um it is almost transparent (kappa 0.04-0.12), in 8-12 um
# it is close to opaque (kappa 0.40-0.55). Feeding a broadband 280 K sky into
# the band Planck integral overstates MWIR sky radiance by roughly 5x and makes
# sky and ground read the same, which destroys any contrast cue.
SKY_KAPPA = dict(mwir=(0.18, 0.55, 0.04, 0.12), lwir=(0.42, 0.55, 0.40, 0.55))


def _sky_kappa(w_cm, cloud):
    out = {}
    for b, (a, k, lo, hi) in SKY_KAPPA.items():
        kap = 1.0 - math.exp(-a * max(w_cm, 1e-3) ** k)
        kap = min(max(kap, lo), hi)
        # cloud fills the gap towards an opaque (radiating) sky
        kap = kap + (1.0 - kap) * 0.55 * min(max(cloud, 0.0), 1.0)
        out[b] = min(max(kap, 0.0), 0.99)
    return out


def derive(w):
    """Weather state -> physical quantities consumed by every modality."""
    cloud = min(max(float(w["cloud_cover"]), 0.0), 1.0)
    rain = max(float(w["rain_mmh"]), 0.0)
    vis_km = max(float(w["visibility_m"]) / 1000.0, 0.05)
    rh = min(max(float(w["rh"]), 0.0), 1.0)
    wind = max(float(w["wind_ms"]), 0.0)
    t_c = float(w["air_temp_c"])
    T_air = t_c + 273.15
    sin_el = max(math.sin(math.radians(float(w["sun_elev_deg"]))), 0.0)

    # solar: direct normal + diffuse horizontal. The direct term is DNI, NOT the
    # horizontal irradiance: the caller already multiplies by cos(incidence), so
    # folding sin(elevation) in here counted it twice and starved tilted
    # surfaces at low sun.
    if sin_el <= 0.01:
        dni = dif = 0.0
        airmass = 38.0
    else:
        # Kasten-Young air mass, finite at the horizon (1/sin diverges)
        el_deg = max(float(w["sun_elev_deg"]), 0.0)
        airmass = 1.0 / (sin_el + 0.50572 * (el_deg + 6.07995) ** -1.6364)
        dni = 1361.0 * 0.7 ** (airmass ** 0.678)
        # cloud extinction is exponential in optical depth, not a magic power
        tau_c = 0.1 + 18.0 * cloud ** 2
        dni *= math.exp(-tau_c)
        dif = 0.30 * 1361.0 * sin_el * (0.2 + 0.8 * cloud) * math.exp(-0.6 * tau_c)
    solar = dni * sin_el + dif

    # sky: Clark-Allen emissivity -> grey equivalent temperature, plus the
    # per-band coupling that decides how much of it a given camera actually sees
    w_cm, w_km = _water_vapour(w, t_c, rh)
    eps_sky = _sky_emissivity(t_c, rh, cloud)
    t_sky = eps_sky ** 0.25 * T_air
    kappa = _sky_kappa(w_cm, cloud)

    wetness = min(1.0, rain / 4.0)
    beta_opt = 3.912 / vis_km  # Koschmieder extinction at 550 nm, per km

    # 905 nm lidar: aerosol scattering scales as lambda^-q from the 550 nm
    # Koschmieder value; raindrops are near-neutral but much stronger per mm/h
    q = _kim_q(vis_km)
    beta_905 = beta_opt * (0.905 / 0.55) ** (-q)
    beta_lidar = beta_905 + 0.072 * rain ** 0.72

    # Thermal IR: aerosol (Kruse/Kim) + molecular absorption driven by water
    # vapour. 8-12 um is dominated by water vapour (aerosol is nearly
    # negligible there); 3-5 um keeps a fixed CO2/N2O background.
    beta_aer_mwir = beta_opt * (4.0 / 0.55) ** (-q)
    beta_aer_lwir = beta_opt * (10.0 / 0.55) ** (-q)
    mol = dict(mwir=0.28 + 0.06 * w_km, lwir=0.02 + 0.13 * w_km,
               swir=0.20 + 0.10 * w_km)
    aer = dict(mwir=beta_aer_mwir, lwir=beta_aer_lwir, swir=beta_opt)
    tau_ir = {}
    for b in ("mwir", "lwir", "swir"):
        tau_ir[b] = (aer[b] + mol[b]) * (1.0 + 0.5 * wetness) + 0.02 * rain ** 0.8

    att = {}
    for name, p in RADAR_BANDS.items():
        att[name] = p["gas_db_km"] + p["rain_k"] * rain ** p["rain_a"]

    return dict(
        cloud_cover=cloud, rain_mmh=rain, visibility_km=vis_km,
        solar_w_m2=solar, solar_factor=solar / 700.0,
        dni_w_m2=dni, diffuse_w_m2=dif, airmass=airmass,
        t_sky_k=t_sky, eps_sky=eps_sky, sky_kappa=kappa,
        water_vapour_cm=w_cm, wetness=wetness, t_air_k=T_air,
        beta_opt_per_km=beta_opt, beta_lidar_per_km=beta_lidar,
        beta_mol_per_km=mol, beta_aer_per_km=aer,
        tau_ir_per_km=tau_ir,
        radar_att_db_km=att,
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