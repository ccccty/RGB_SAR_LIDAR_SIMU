"""Steady-state per-face thermal balance (simplified lumped radiometry).

Physics kept: solar absorption, sky/ground long-wave exchange, convective
cooling, internal heat sources, per-object isothermal override.
Deliberately omitted: conduction between faces, shadowing of multiple bounces,
non-grey spectral properties. Good enough for detection-style IR imagery;
documented in README as an approximation.
"""
import numpy as np

from . import materials as matlib

SIGMA = 5.670374419e-8


def sun_vector(elev_deg, az_deg):
    el = np.radians(elev_deg)
    az = np.radians(az_deg)
    return np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])


def sky_temp(T_air, rh=0.5):
    """Swinbank clear-sky temperature with a humidity correction."""
    return 0.0552 * T_air ** 1.5 * (1.0 - 0.25 * np.clip(rh, 0.0, 1.0))


def face_normals(world):
    t = world.verts[world.faces]
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    ln[ln == 0] = 1.0
    return n / ln


def face_centers(world):
    return world.verts[world.faces].mean(axis=1)


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
    if w:
        # weather state machine wins over the static sun/air settings
        G = float(w["solar_w_m2"]) if sun.get("on", True) else 0.0
        wind = float(cfg.get("weather", {}).get("wind_ms", wind))

    eps = props[world.face_mat, 0]
    alpha = props[world.face_mat, 1]
    q_int = props[world.face_mat, 2]

    nrm = face_normals(world)
    cent = face_centers(world)

    cos_inc = np.clip(nrm @ sdir, 0.0, None)
    shade = np.ones(len(cent))
    if shadow and G > 0:
        orig = cent + nrm * 2e-3
        bvh = world.bvh
        for i in range(len(cent)):
            if cos_inc[i] <= 0.0:
                shade[i] = 0.0
                continue
            t = bvh.ray_cast(orig[i], sdir[None, :])["t"][0]
            shade[i] = 0.0 if np.isfinite(t) else 1.0

    nz = np.clip(nrm[:, 2], -1.0, 1.0)
    vf_sky = 0.5 * (1.0 + nz)
    vf_gnd = 1.0 - vf_sky

    T_sky = float(w["t_sky_k"]) if w else sky_temp(T_air, rh)
    wet = float(w["wetness"]) if w else 0.0
    h = 5.7 + 3.8 * wind
    h = np.full(len(cent), h)
    if wet > 0.0:
        # rain: wet ground evaporates (extra cooling) and darkens (albedo drop)
        gnd = world.face_obj == -1
        h = np.where(gnd, h + wet * (8.0 + 15.0 * wind), h)
        alpha = np.where(gnd, alpha * (1.0 - 0.25 * wet), alpha)

    q_in = (alpha * G * cos_inc * shade
            + eps * SIGMA * (vf_sky * T_sky ** 4 + vf_gnd * T_ground ** 4)
            + q_int)

    T = np.full(len(cent), T_air, float)
    for _ in range(60):
        fval = eps * SIGMA * T ** 4 + h * (T - T_air) - q_in
        dfun = 4.0 * eps * SIGMA * T ** 3 + h
        T = T - fval / dfun
        T = np.clip(T, 180.0, 1200.0)

    if "temp_c" in cfg.get("ground", {}):
        T[world.face_obj == -1] = T_ground
    for o in world.objs:
        if o.get("temp_c") is not None:
            T[world.face_obj == o["id"]] = float(o["temp_c"]) + 273.15
    return T
