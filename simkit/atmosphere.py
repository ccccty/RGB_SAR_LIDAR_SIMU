"""Lightweight altitude-aware Beer--Lambert atmosphere for research rendering.

Opt in with ``atmosphere.profile = 'exponential'``. Aerosol, dry-air and
water-vapour extinction have independent scale heights. These are configurable
engineering assumptions, not a line-by-line spectral atmosphere or a calibrated
prediction of a real sensor's detection range. Geometry is flat-Earth.
"""
import numpy as np

from . import weather


def enabled(cfg):
    profile = (cfg.get("atmosphere") or {}).get("profile", "homogeneous")
    if profile not in ("homogeneous", "exponential"):
        raise ValueError("atmosphere.profile must be homogeneous or exponential")
    return profile == "exponential"


def _state(cfg):
    return cfg.get("_weather") or weather.derive(weather.resolve(cfg.get("weather")))


def _number(spec, key, default, positive=False):
    value = float(spec.get(key, default))
    if not np.isfinite(value) or (positive and value <= 0):
        raise ValueError("invalid atmosphere." + key)
    return value


def _components(cfg, modality):
    """Reference-level extinction [1/km], height scales [m]."""
    spec = cfg.get("atmosphere") or {}
    w = _state(cfg)
    if modality in ("visible", "optical", "rgb"):
        total = float(w["beta_opt_per_km"])
        dry = min(total, _number(spec, "visible_rayleigh_per_km", 0.011))
        aer, vapour, rain = total-dry, 0.0, 0.0
    elif modality in ("mwir", "lwir", "swir"):
        dry = dict(mwir=0.28, lwir=0.02, swir=0.20)[modality]
        vapour = max(float(w["beta_mol_per_km"][modality])-dry, 0.0)
        aer = float(w["beta_aer_per_km"][modality])
        wet = 1.0 + 0.5*float(w.get("wetness", 0.0))
        dry, vapour, aer = dry*wet, vapour*wet, aer*wet
        rain = max(float(w["tau_ir_per_km"][modality])-dry-vapour-aer, 0.0)
    else:
        raise ValueError("unsupported atmosphere modality: " + str(modality))
    overrides = spec.get("extinction_at_reference_per_km") or {}
    if modality in overrides:
        # Explicit broadband total replaces the model with a single aerosol
        # profile rather than silently preserving unknown component ratios.
        aer, dry, vapour, rain = float(overrides[modality]), 0.0, 0.0, 0.0
    coefficients = np.array([aer, dry, vapour, rain], float)
    if not np.isfinite(coefficients).all() or (coefficients < 0).any():
        raise ValueError("atmosphere extinction must be finite and non-negative")
    scales = np.array([
        _number(spec, "aerosol_scale_height_m", 1200.0, True),
        _number(spec, "molecular_scale_height_m", 8000.0, True),
        _number(spec, "water_scale_height_m", 2000.0, True),
        _number(spec, "rain_scale_height_m", 2000.0, True)], float)
    return coefficients, scales


def optical_depth(origin_m, endpoints_m, cfg, modality="visible"):
    """Integrate extinction from one origin to N endpoints, returning shape (...).

    Integrates an exponential exactly above reference altitude; portions below
    reference use constant reference density. This also handles reversed paths,
    nearly horizontal paths and zero length without unstable subtraction.
    """
    p = np.asarray(origin_m, float)
    q = np.asarray(endpoints_m, float)
    if p.shape != (3,) or q.shape[-1:] != (3,) or not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError("atmosphere paths require finite xyz coordinates")
    distance = np.linalg.norm(q-p, axis=-1)/1000.0
    coefficients, scales = _components(cfg, modality)
    if not enabled(cfg):
        return distance*float(coefficients.sum())
    reference = _number(cfg.get("atmosphere") or {}, "reference_altitude_m", 0.0)
    za = np.minimum(p[2], q[..., 2])-reference
    zb = np.maximum(p[2], q[..., 2])-reference
    dz = zb-za
    below = np.maximum(np.minimum(zb, 0.0)-za, 0.0)
    zlo, zhi = np.maximum(za, 0.0), np.maximum(zb, 0.0)
    tau = np.zeros_like(distance)
    for beta, height in zip(coefficients, scales):
        if beta == 0:
            continue
        integral_z = below + height*np.exp(-zlo/height)*(-np.expm1(-(zhi-zlo)/height))
        # For horizontal rays use the limiting density instead of 0/0.
        average = np.exp(-np.maximum(za, 0.0)/height)
        average = np.divide(integral_z, dz, out=np.asarray(average).copy(), where=dz > 1e-8)
        tau += beta*distance*average
    return tau


def path_transmittance(origin_m, endpoints_m, cfg, modality="visible"):
    return np.exp(-optical_depth(origin_m, endpoints_m, cfg, modality))


def temperature_k(altitude_m, cfg):
    """Declared reference temperature and lapse, capped at a declared minimum."""
    spec = cfg.get("atmosphere") or {}
    reference = _number(spec, "reference_altitude_m", 0.0)
    reference_c = _number(spec, "temperature_at_reference_c",
                          cfg.get("air", {}).get("temp_c", 20.0))
    lapse = _number(spec, "temperature_lapse_k_per_km", 6.5)
    floor = _number(spec, "minimum_temperature_k", 216.65, True)
    if not enabled(cfg):
        return np.full_like(np.asarray(altitude_m, float), reference_c+273.15)
    return np.maximum(reference_c+273.15-lapse*(np.asarray(altitude_m)-reference)/1000.0, floor)


def sky_endpoints(origin_m, directions, cfg):
    """Finite background proxy: atmospheric top upward, reference plane downward.

    Nearly horizontal paths are capped by background_distance_m (200 km by
    default). This is explicitly not Earth-curvature or terrain modelling.
    """
    spec = cfg.get("atmosphere") or {}
    p = np.asarray(origin_m, float)
    d = np.asarray(directions, float)
    d = d/np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-12)
    cap = _number(spec, "background_distance_m", 200000.0, True)
    top = _number(spec, "top_altitude_m", 80000.0)
    bottom = _number(spec, "reference_altitude_m", 0.0)
    dz = d[..., 2]
    target_z = np.where(dz >= 0.0, top, bottom)
    length = np.divide(target_z-p[2], dz, out=np.full_like(dz, cap), where=np.abs(dz) > 1e-9)
    length = np.clip(length, 0.0, cap)
    return p+d*length[..., None]


def sky_transmittance(origin_m, directions, cfg, modality="visible"):
    return path_transmittance(origin_m, sky_endpoints(origin_m, directions, cfg), cfg, modality)


def path_emission(origin_m, endpoints_m, cfg, band_um, modality="lwir"):
    """Thermal path radiance, using LTE emission and piecewise temperatures.

    Uses a small Planck LUT and exact segment extinction, avoiding a full
    wavelength integral per pixel per segment. No multiple scattering, spectral
    lines, clouds-as-volumes or refraction are implied.
    """
    from .ir import planck_band
    p, q = np.asarray(origin_m, float), np.asarray(endpoints_m, float)
    spec = cfg.get("atmosphere") or {}
    steps = int(spec.get("emission_steps", 12))
    if not 1 <= steps <= 64:
        raise ValueError("atmosphere.emission_steps must be 1..64")
    shape = q.shape[:-1]
    accum = np.zeros(shape, float)
    foreground_tau = np.ones(shape, float)
    temperatures = np.linspace(150.0, 400.0, 512)
    radiance = planck_band(temperatures, *band_um)
    coefficients, scales = _components(cfg, modality)
    exponential = enabled(cfg)
    for index in range(steps):
        a = p+(q-p)*(index/steps)
        b = p+(q-p)*((index+1)/steps)
        # optical_depth accepts a single origin. Translate each short segment
        # to altitude-only equivalent because extinction depends solely on z.
        # The common displacement and segment length are known analytically.
        dz = b[..., 2]-a[..., 2]
        length = np.linalg.norm(b-a, axis=-1)
        depth = np.zeros(shape, float)
        reference = float(spec.get("reference_altitude_m", 0.0))
        for beta, height in zip(coefficients, scales):
            za = np.minimum(a[..., 2], b[..., 2])-reference
            zb = np.maximum(a[..., 2], b[..., 2])-reference
            zlo, zhi = np.maximum(za, 0), np.maximum(zb, 0)
            below = np.maximum(np.minimum(zb, 0)-za, 0)
            integ = below+height*np.exp(-zlo/height)*(-np.expm1(-(zhi-zlo)/height))
            average = np.exp(-np.maximum(za, 0)/height)
            average = np.divide(integ, np.abs(dz), out=np.asarray(average).copy(), where=np.abs(dz)>1e-8)
            if not exponential:
                average = np.ones_like(average)
            depth += beta*length/1000.0*average
        transmission = np.exp(-depth)
        temp = temperature_k((a[..., 2]+b[..., 2])/2, cfg)
        local = np.interp(temp, temperatures, radiance)
        accum += foreground_tau*(1.0-transmission)*local
        foreground_tau *= transmission
    return accum
