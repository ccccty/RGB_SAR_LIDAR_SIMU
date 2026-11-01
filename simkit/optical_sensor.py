"""Relative scene-linear RGB -> atmosphere -> PSF -> electron/count model.

Radiance/electron conversion is a declared research response, not absolute
radiometric calibration. Geometry is kept separate from noisy measurements.
"""
import numpy as np

from . import atmosphere, scene


def gaussian(image, sigma):
    if sigma <= 0:
        return np.asarray(image, float).copy()
    radius = max(1, int(np.ceil(3*sigma)))
    x = np.arange(-radius, radius+1)
    kernel = np.exp(-0.5*(x/sigma)**2)
    kernel /= kernel.sum()
    result = np.asarray(image, float)
    for axis in (0, 1):
        pad = [(0, 0)]*result.ndim
        pad[axis] = (radius, radius)
        padded = np.pad(result, pad, mode="edge")
        output = np.zeros_like(result)
        for i, weight in enumerate(kernel):
            take = [slice(None)]*result.ndim
            take[axis] = slice(i, i+result.shape[axis])
            output += weight*padded[tuple(take)]
        result = output
    return result


def srgb(linear):
    value = np.clip(linear, 0, 1)
    return np.where(value <= 0.0031308, 12.92*value, 1.055*value**(1/2.4)-0.055)


def forward(linear_rgb, depth_z_m, cam, cfg, frame_seed, exposure_stops=0.0):
    linear = np.asarray(linear_rgb, float)
    depth = np.asarray(depth_z_m, float)
    if linear.ndim != 3 or linear.shape[2] != 3 or depth.shape != linear.shape[:2]:
        raise ValueError("RGB/depth shapes disagree")
    if not np.isfinite(linear).all():
        raise ValueError("non-finite scene-linear RGB")
    settings = (cfg.get("optical") or {}).get("sensor") or {}
    W, H = linear.shape[1], linear.shape[0]
    valid = np.isfinite(depth) & (depth > 0) & (depth < 1e9)
    tau = np.ones((H, W), float)
    if atmosphere.enabled(cfg) and valid.any():
        p, right, up, fwd = scene.sensor_basis(cam["pos"], cam["look_at"],
                                               roll_deg=cam.get("roll_deg", 0.0))
        fx = W/(2*np.tan(np.radians(cam["fov_deg"])/2))
        ys, xs = np.nonzero(valid)
        rays = (fwd + ((xs+0.5-W/2)/fx)[:, None]*right -
                ((ys+0.5-H/2)/fx)[:, None]*up)
        endpoints = p+rays*depth[valid, None]
        tau[valid] = atmosphere.path_transmittance(p, endpoints, cfg, "visible")
    airlight = np.asarray(settings.get("airlight_linear_rgb", [0.35, 0.45, 0.60]), float)
    if airlight.shape != (3,) or not np.isfinite(airlight).all() or (airlight < 0).any():
        raise ValueError("airlight_linear_rgb must contain three non-negative values")
    received = tau[:, :, None]*linear+(1-tau[:, :, None])*airlight
    sigma = float(settings.get("psf_sigma_px", 0.6))
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("PSF sigma must be finite and non-negative")
    received = gaussian(received, sigma)
    exposure = float(cfg.get("acquisition", {}).get("rgb_exposure_s", 0.001))
    reference = float(settings.get("reference_exposure_s", 0.001))
    well = float(settings.get("full_well_e", 30000.0))
    white = float(settings.get("white_level_linear", 1.0))
    read_noise = float(settings.get("read_noise_e", 3.0))
    prnu_sigma = float(settings.get("prnu", 0.001))
    bits = int(settings.get("bits", 12))
    if (not np.isfinite([exposure, reference, well, white, read_noise, prnu_sigma, exposure_stops]).all()
            or min(exposure, reference, well, white) <= 0 or min(read_noise, prnu_sigma) < 0
            or not 8 <= bits <= 16):
        raise ValueError("invalid RGB sensor exposure/electron/bit-depth setting")
    expectation = np.maximum(received, 0)*(2**float(exposure_stops))*(exposure/reference)*well/white
    fixed = np.random.default_rng(int(settings.get("seed", 12345)))
    prnu = np.maximum(1+fixed.normal(0, prnu_sigma, linear.shape), 0)
    expectation = np.minimum(expectation*prnu, well)
    random = np.random.default_rng(int(frame_seed))
    electrons = (random.poisson(expectation) if settings.get("shot_noise", True) else expectation.copy())
    electrons = electrons+random.normal(0, read_noise, linear.shape)
    max_count = 2**bits-1
    counts = np.rint(np.clip(electrons/well, 0, 1)*max_count).astype(np.uint16)
    measured = counts.astype(float)/max_count
    return dict(image_u8=np.rint(srgb(measured)*255).astype(np.uint8),
                counts_u16=counts, linear_received=received.astype(np.float32),
                transmittance=tau.astype(np.float32),
                metadata=dict(model="relative-linear electron response", bits=bits,
                              exposure_s=exposure, exposure_stops=float(exposure_stops),
                              reference_exposure_s=reference, full_well_e=well,
                              white_level_linear=white, read_noise_e=read_noise, prnu=prnu_sigma,
                              psf_sigma_px=sigma, display_transform="sRGB OETF",
                              airlight_linear_rgb=airlight.tolist(),
                              absolute_radiometric_calibration=False))
