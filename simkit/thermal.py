"""Steady-state per-face thermal balance (simplified lumped radiometry).

Physics kept: solar absorption, sky/ground long-wave exchange, convective
cooling, internal heat sources, per-object isothermal override.
Deliberately omitted: conduction between faces, shadowing of multiple bounces,
non-grey spectral properties. Good enough for detection-style IR imagery;
documented in README as an approximation.
"""
import numpy as np

from . import materials as matlib
from . import weather as weatherlib

SIGMA = 5.670374419e-8


def sun_vector(elev_deg, az_deg):
    el = np.radians(elev_deg)
    az = np.radians(az_deg)
    return np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])


def sky_temp(T_air, rh=0.5):
    """Grey-equivalent sky temperature for the no-derived-weather fallback.

    The old implementation was Swinbank * (1 - 0.25*RH), which has the humidity
    sign backwards and reads 25-35 K too cold. The main loop uses the derived
    ``_weather.t_sky_k``; this helper only runs for bare API calls and unit
    tests, and must still return the same Clark-Allen/EnergyPlus physics.
    """
    return weatherlib.sky_temperature_k(float(T_air) - 273.15, rh)


def face_normals(world):
    t = world.verts[world.faces]
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    ln[ln == 0] = 1.0
    return n / ln


def face_centers(world):
    return world.verts[world.faces].mean(axis=1)


def region_overrides(world):
    """Explicit object-local regions; later matching regions win.

    Selectors are intersected. Temperatures are absolute overrides, fluxes
    replace the material's internal heat flux before solving the balance.
    Geometry/material metadata is supplied by scene.build_world.
    """
    count = len(world.faces)
    temperature = np.full(count, np.nan)
    flux = np.full(count, np.nan)
    local = getattr(world, "face_local_centers", None)
    source = getattr(world, "face_source_material", None)
    group = getattr(world, "face_group", None)
    for obj in world.objs:
        for region in obj.get("thermal_regions", []):
            mask = world.face_obj == obj["id"]
            selected = False
            for key, values in (("material_names", source), ("group_names", group)):
                if key in region:
                    if values is None:
                        raise ValueError("thermal region requires World " + key)
                    choices = region[key]
                    if not isinstance(choices, list) or not choices:
                        raise ValueError("thermal region " + key + " must be a non-empty list")
                    mask &= np.isin(values, choices)
                    selected = True
            if "bounds_local_m" in region:
                bounds = np.asarray(region["bounds_local_m"], float)
                if local is None or bounds.shape != (2, 3) or not np.isfinite(bounds).all() or (bounds[1]<bounds[0]).any():
                    raise ValueError("thermal region needs valid local bounds [[min],[max]]")
                mask &= ((local >= bounds[0]) & (local <= bounds[1])).all(axis=1)
                selected = True
            if not selected:
                raise ValueError("thermal region requires an explicit material, group or local-bounds selector")
            if not any(key in region for key in ("temp_c", "heat_flux_w_m2")):
                raise ValueError("thermal region requires temp_c or heat_flux_w_m2")
            if "temp_c" in region:
                value = float(region["temp_c"])+273.15
                if not np.isfinite(value) or not 150 <= value <= 2000:
                    raise ValueError("thermal region temperature outside 150..2000 K")
                temperature[mask] = value
            if "heat_flux_w_m2" in region:
                value = float(region["heat_flux_w_m2"])
                if not np.isfinite(value) or value < 0:
                    raise ValueError("thermal region heat flux must be finite and non-negative")
                flux[mask] = value
    return temperature, flux


#: Classes that have a cockpit the thermal model should treat as glass. A
#: canopy reflects the cold sky and conducts badly, so it reads far cooler than
#: the skin around it -- one of the few shapes that identifies an aircraft in a
#: low-resolution thermal frame.
COCKPIT_CLS = frozenset(("fighter", "airliner", "helicopter", "uav_fixed"))


def aero_heating(world, cfg, nrm):
    """Extra convective conductance and recovery temperature per face.

    A target moving at 250 m/s is not in natural convection. The boundary layer
    is turbulent and forced, so the surface is pinned near the recovery
    temperature instead of the air temperature, and the stagnation side runs
    warmer than the lee side. That difference is the only reason a fast aircraft
    shows any structure at all: without it the whole airframe solves to one
    value and renders as a flat cut-out.

    Returns (h_extra, t_ref) where t_ref is the effective air temperature the
    face convects against.
    """
    count = len(nrm)
    h_extra = np.zeros(count)
    t_ref = np.full(count, np.nan)
    air = cfg.get("air", {})
    if not cfg.get("ir", {}).get("aero_heat", True):
        return h_extra, t_ref
    T_air = float(air.get("temp_c", 20.0)) + 273.15
    rho = float(air.get("density_kg_m3", 1.112))
    cp = 1005.0
    gamma = 1.4
    r_cov = 0.89
    for obj in world.objs:
        if not obj.get("label", True):
            continue
        vel = np.asarray(obj.get("vel", [0.0, 0.0, 0.0]), float)
        speed = float(np.linalg.norm(vel))
        if speed < 15.0:
            continue
        mask = world.face_obj == obj["id"]
        if not mask.any():
            continue
        vhat = vel / speed
        # turbulent flat plate: h ~ 0.0025 rho v cp is the right order for the
        # whole airframe; the exact correlation is not worth its parameters here
        h_extra[mask] += 0.0025 * rho * speed * cp
        mach2 = (speed / 340.0) ** 2
        # recovery factor on the windward side, falling off to the freestream on
        # the lee side; cos(incidence) is the natural weight
        cosv = np.clip(-(nrm[mask] @ vhat), 0.0, 1.0)
        rise = T_air * r_cov * 0.5 * (gamma - 1.0) * mach2
        t_ref[mask] = T_air + rise * cosv
    return h_extra, t_ref


def canopy_cooling(world, cfg, T, nrm, cent):
    """Drop canopy faces toward the sky temperature.

    Glass is opaque out of band and reflective in band, so a canopy does not
    reach radiative equilibrium with the sun the way painted metal does. The
    patch is approximated from geometry only: the upper front of a fuselage,
    which is where every one of these models puts its cockpit.
    """
    if not cfg.get("ir", {}).get("canopy_cool", True):
        return T
    w = cfg.get("_weather")
    T_sky = float(w["t_sky_k"]) if w else 0.0552 * (float(cfg.get("air", {}).get("temp_c", 20.0)) + 273.15) ** 1.5
    out = np.array(T, float, copy=True)
    for obj in world.objs:
        if not obj.get("label", True) or obj.get("cls") not in COCKPIT_CLS:
            continue
        mask = world.face_obj == obj["id"]
        if not mask.any():
            continue
        pos = np.asarray(obj.get("pos", [0.0, 0.0, 0.0]), float)
        vel = np.asarray(obj.get("vel", [0.0, 0.0, 0.0]), float)
        speed = float(np.linalg.norm(vel))
        fwd = vel / speed if speed > 1e-6 else np.array([1.0, 0.0, 0.0])
        rel = cent[mask] - pos
        along = rel @ fwd
        lo, hi = float(along.min()), float(along.max())
        length = max(hi - lo, 1e-6)
        # front 45 % of the fuselage, upper 40 % of its height, and either
        # facing up or into the wind -- a canopy, not the whole nose
        forward = (along - lo) > 0.55 * length
        up = nrm[mask][:, 2] > 0.15
        top = rel[:, 2] > np.percentile(rel[:, 2], 55.0)
        sel = forward & (up | top)
        if not sel.any():
            continue
        idx = np.flatnonzero(mask)[sel]
        cold = 0.55 * out[idx] + 0.45 * T_sky
        out[idx] = np.clip(cold, 200.0, out[idx])
    return out


def solve(world, cfg, shadow=True):
    """Return per-face temperature in Kelvin."""
    mat_names = sorted(matlib.MATERIALS)
    props = np.array([[matlib.MATERIALS[n]["emissivity"],
                       matlib.MATERIALS[n]["alpha_solar"],
                       matlib.MATERIALS[n]["h_int"]] for n in mat_names])

    air = cfg.get("air", {})
    T_air = float(air.get("temp_c", 20.0)) + 273.15
    rh = float(air.get("rh", 0.5))
    wind = float(air.get("wind_mps", 2.0))
    T_ground = float(cfg.get("ground", {}).get("temp_c", air.get("temp_c", 20.0) + 5.0)) + 273.15

    sun = cfg.get("sun", {})
    G = float(sun.get("irradiance", 700.0)) if sun.get("on", True) else 0.0
    sdir = sun_vector(float(sun.get("elev_deg", 35.0)), float(sun.get("az_deg", 120.0)))

    w = cfg.get("_weather")
    dni = dif = None
    if w:
        # weather state machine wins over the static sun/air settings.
        # dni is the DIRECT NORMAL irradiance: the cos(incidence) factor is
        # applied below, so using the horizontal irradiance here counted
        # sin(elevation) twice and starved tilted surfaces at low sun.
        G = float(w["solar_w_m2"]) if sun.get("on", True) else 0.0
        dni = float(w.get("dni_w_m2", w["solar_w_m2"])) if sun.get("on", True) else 0.0
        dif = float(w.get("diffuse_w_m2", 0.0)) if sun.get("on", True) else 0.0
        wind = float(cfg.get("weather", {}).get("wind_ms", wind))

    eps = props[world.face_mat, 0]
    alpha = props[world.face_mat, 1]
    q_int = props[world.face_mat, 2]

    region_temperature, region_flux = region_overrides(world)
    q_int = np.where(np.isfinite(region_flux), region_flux, q_int)

    nrm = face_normals(world)
    cent = face_centers(world)
    from . import atmosphere
    if atmosphere.enabled(cfg):
        T_air = atmosphere.temperature_k(cent[:, 2], cfg)

    cos_inc = np.clip(nrm @ sdir, 0.0, None)
    shade = np.ones(len(cent))
    if shadow and G > 0:
        orig = cent + nrm * 2e-3
        lit = cos_inc > 0.0
        shade[~lit] = 0.0
        if lit.any():
            # one batched cast for every lit face; the old per-face loop was the
            # single most expensive thing in the whole generator
            dirs = np.broadcast_to(sdir, (int(lit.sum()), 3))
            t = world.bvh.ray_cast_batch(orig[lit], dirs)["t"]
            shade[lit] = np.where(np.isfinite(t), 0.0, 1.0)

    nz = np.clip(nrm[:, 2], -1.0, 1.0)
    vf_sky = 0.5 * (1.0 + nz)
    vf_gnd = 1.0 - vf_sky

    T_sky = float(w["t_sky_k"]) if w else sky_temp(T_air, rh)
    if atmosphere.enabled(cfg):
        # Retain the declared grey-sky emissivity, but use local air temperature
        # rather than applying a sea-level temperature to a high aircraft.
        eps_sky = float(w["eps_sky"]) if w else 0.8
        T_sky = eps_sky**0.25*T_air
    wet = float(w["wetness"]) if w else 0.0
    h = 5.7 + 3.8 * wind
    h = np.full(len(cent), h)
    # Forced convection from the target's own airspeed. A 250 m/s airframe is
    # pinned near its recovery temperature, not the ambient one; without this
    # every face solved to a single value and the frame had no structure.
    _h_extra, _t_ref = aero_heating(world, cfg, nrm)
    h = h + _h_extra
    if wet > 0.0:
        # rain: wet ground evaporates (extra cooling) and darkens (albedo drop)
        gnd = world.face_obj == -1
        h = np.where(gnd, h + wet * (8.0 + 15.0 * wind), h)
        alpha = np.where(gnd, alpha * (1.0 - 0.25 * wet), alpha)

    if dni is None:
        direct = alpha * G * cos_inc * shade
        diffuse = 0.0
    else:
        direct = alpha * dni * cos_inc * shade
        # diffuse sky light: without it every shaded face is pure black by day,
        # while a real one is lit by the whole sky dome
        diffuse = alpha * dif * vf_sky
    q_in = (direct + diffuse
            + eps * SIGMA * (vf_sky * T_sky ** 4 + vf_gnd * T_ground ** 4)
            + q_int)

    # The air a face convects against: ambient, or the recovery temperature on
    # the windward side of a fast target.
    T_conv = np.where(np.isfinite(_t_ref), _t_ref, T_air)
    T = np.array(T_conv, float)
    for _ in range(60):
        fval = eps * SIGMA * T ** 4 + h * (T - T_conv) - q_in
        dfun = 4.0 * eps * SIGMA * T ** 3 + h
        T = T - fval / dfun
        T = np.clip(T, 180.0, 1200.0)

    if "temp_c" in cfg.get("ground", {}):
        T[world.face_obj == -1] = T_ground
    for o in world.objs:
        if o.get("temp_c") is not None:
            T[world.face_obj == o["id"]] = float(o["temp_c"]) + 273.15
    mask = np.isfinite(region_temperature)
    T[mask] = region_temperature[mask]
    T = canopy_cooling(world, cfg, T, nrm, cent)
    return T
