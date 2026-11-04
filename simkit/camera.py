# -*- coding: utf-8 -*-
"""Camera defects for the visible-band render: the flaws a real lens+sensor
leaves on an otherwise clean Cycles frame.

Why this module exists
----------------------
A Cycles frame with 16-64 samples and the denoiser is *cleaner* than any
photograph: measured on an air-to-air frame the sky's high-pass residual has a
standard deviation of 0.37 DN, while a real ISO-100 frame sits at 1-2.5 DN and
its shadows at 1.5-3 DN.  A detector trained on the clean version learns that
sky is a flat field, that every edge is exactly sharp and that the three
channels are independent, none of which is true of the camera it will face.

The chain, in the physical order light meets it:

    vignetting            cos^4 lens falloff towards the corners
    chromatic_aberration  lateral R/B magnification difference (sub-pixel)
    diffraction           aperture + pixel-aperture MTF (Airy OTF, not Gaussian)
    flare                 glare halo + ghost spots around highlights
    sensor                CFA mosaic, shot/read/row/FPN noise, demosaic
    jpeg                  optional lossy round trip

The sensor stage is the important one.  Adding independent noise to the three
planes -- what the first version of this module did -- gives grey grain.  A real
camera measures *one* colour per photosite through a Bayer filter and
reconstructs the other two by interpolation, so the noise of a red sample is
copied into the four green sites around it.  That shared sample is what makes
ISO grain look like coloured confetti instead of film grain, and it cannot be
produced by any per-channel model.  The mosaic and the bilinear demosaic below
are therefore not decoration: they are the mechanism that generates chroma
noise, and tests/test_camera.py measures the resulting channel correlation.

Tonal domain
------------
Lens stages run in linear light (a vignette attenuates photons, not code
values).  The sensor stage quantises in the display domain, because the
acceptance numbers -- "shadow sigma 1.5-3 DN", "sky sigma 1-2.5 DN" -- are
measured on the 8-bit output.  Shot noise is physical (Poisson on electrons,
full well / ISO); read and fixed-pattern terms are specified in output DN, the
way they are measured on a real 8-bit stream, and mapped back through the local
slope of the sRGB curve so that a shadow pixel and a highlight pixel end up
with the same readout noise in DN.  'ir'-style physical bookkeeping is shown in
the metadata (electrons_per_dn) when the model is real.

Every stage has its own switch.  With all switches off 'apply' returns the
caller's array object untouched -- not a copy that happens to compare equal --
so the renderer skips the read/modify/write cycle and the file stays bit for
bit what Blender wrote (tests/test_camera.py pins this).

Configuration lives under 'optical.camera' in the scene JSON; see DEFAULTS for
the accepted keys and the module docstring of each stage for their meaning.
"""
import math

import numpy as np

# Physical order: lens first, then detector, then compression.
STAGES = ("vignetting", "chromatic_aberration", "diffraction", "flare",
          "sensor", "jpeg")

DEFAULTS = {
    # Detector noise.  The numbers are calibrated against real 8-bit output:
    # a 12000 e- well, a 2 e- amplifier and a 1.8 DN output noise floor (ADC +
    # row readout) put a flat patch at ~1.5-1.7 DN from black to sky, which is
    # the range a real ISO-100 camera produces (measured, see tests).
    "sensor": dict(enabled=True, iso=100.0, full_well_e=12000.0,
                   read_noise_e=2.0, read_noise_dn=1.8,
                   dsnu_dn=0.35, row_fpn_dn=0.25, column_fpn_dn=0.15,
                   row_noise_dn=0.35, prnu=0.003, black_level_dn=5.0,
                   shot_noise=True, demosaic=True, cfa_pattern="RGGB",
                   bits=8, seed=12345),
    "vignetting": dict(enabled=False, fov_deg=None, f_number=2.8,
                       strength=1.0, mechanical=0.0,
                       focal_mm=None, sensor_width_mm=None),
    "chromatic_aberration": dict(enabled=False, lateral_px=0.6,
                                 red_px=None, blue_px=None),
    # Diffraction default ON.  A real edge measures MTF50 0.33 cyc/px on the
    # Cycles frame; adding the aperture OTF of an f/5.6 lens on 10-16 um
    # pixels and then reconstructing through the CFA lands at 0.20-0.21, the
    # range a real visible-band camera shows.  The 16 um pitch is what makes
    # the pixel aperture, rather than diffraction, the dominant term -- 960 x
    # 720 on 16 um pixels is a 15 x 11.5 mm seeker format.
    "diffraction": dict(enabled=True, f_number=5.6, wavelength_um=0.55,
                        pixel_pitch_um=16.0, pixel_aperture=True),
    "flare": dict(enabled=False, threshold=0.94, strength=0.35, count=3,
                  radius_px=22.0, ghost_scale=0.6, ghost_count=2,
                  ghost_radius_scale=0.45, seed=11),
    "jpeg": dict(enabled=False, quality=85, quality_range=None),
}


def settings(camera_block):
    """Merge a user 'optical.camera' block over the defaults."""
    block = dict(camera_block or {})
    out = {}
    for stage, defaults in DEFAULTS.items():
        merged = dict(defaults)
        given = block.get(stage)
        if given is not None:
            if not isinstance(given, dict):
                raise ValueError("optical.camera.%s must be an object" % stage)
            merged.update(given)
        out[stage] = merged
    return out


def scene_settings(cfg):
    """Settings from a full scene config ('cfg["optical"]["camera"]')."""
    optical = (cfg or {}).get("optical") or {}
    return settings(optical.get("camera"))


def _finite(name, value, low=None, high=None, positive=False):
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError("%s must be a number" % name)
    if not np.isfinite(value):
        raise ValueError("%s must be finite" % name)
    if positive and value <= 0.0:
        raise ValueError("%s must be positive" % name)
    if low is not None and value < low:
        raise ValueError("%s must be >= %s" % (name, low))
    if high is not None and value > high:
        raise ValueError("%s must be <= %s" % (name, high))
    return value


def _validate(stage, values):
    if stage == "sensor":
        _finite("optical.camera.sensor.iso", values["iso"], positive=True)
        _finite("optical.camera.sensor.full_well_e", values["full_well_e"], positive=True)
        for key in ("read_noise_e", "read_noise_dn", "dsnu_dn", "row_fpn_dn",
                    "column_fpn_dn", "row_noise_dn", "prnu", "black_level_dn"):
            _finite("optical.camera.sensor." + key, values[key], low=0.0)
        if float(values["prnu"]) >= 1.0:
            raise ValueError("optical.camera.sensor.prnu must be < 1")
        bits = values["bits"]
        if int(bits) != bits or not 8 <= int(bits) <= 16:
            raise ValueError("optical.camera.sensor.bits must be an integer in 8..16")
        pattern = str(values["cfa_pattern"]).upper()
        if sorted(pattern) != ["B", "G", "G", "R"]:
            raise ValueError("optical.camera.sensor.cfa_pattern must be one of "
                             "RGGB, BGGR, GRBG, GBRG")
    elif stage == "vignetting":
        _finite("optical.camera.vignetting.strength", values["strength"], low=0.0)
        _finite("optical.camera.vignetting.f_number", values["f_number"], positive=True)
        _finite("optical.camera.vignetting.mechanical", values["mechanical"],
                low=0.0, high=0.95)
        if values.get("fov_deg") is not None:
            _finite("optical.camera.vignetting.fov_deg", values["fov_deg"],
                    positive=True, high=179.0)
        if (values.get("focal_mm") is None) != (values.get("sensor_width_mm") is None):
            raise ValueError("optical.camera.vignetting.focal_mm and sensor_width_mm "
                             "must be given together")
        for key in ("focal_mm", "sensor_width_mm"):
            if values.get(key) is not None:
                _finite("optical.camera.vignetting." + key, values[key], positive=True)
    elif stage == "chromatic_aberration":
        _finite("optical.camera.chromatic_aberration.lateral_px",
                values["lateral_px"], low=0.0)
        for key in ("red_px", "blue_px"):
            if values.get(key) is not None:
                _finite("optical.camera.chromatic_aberration." + key, values[key])
    elif stage == "diffraction":
        _finite("optical.camera.diffraction.f_number", values["f_number"], positive=True)
        _finite("optical.camera.diffraction.wavelength_um",
                values["wavelength_um"], positive=True)
        _finite("optical.camera.diffraction.pixel_pitch_um",
                values["pixel_pitch_um"], positive=True)
    elif stage == "flare":
        _finite("optical.camera.flare.threshold", values["threshold"], low=0.0, high=1.0)
        _finite("optical.camera.flare.strength", values["strength"], low=0.0)
        _finite("optical.camera.flare.radius_px", values["radius_px"], positive=True)
        _finite("optical.camera.flare.ghost_scale", values["ghost_scale"], low=0.0)
        _finite("optical.camera.flare.ghost_radius_scale",
                values["ghost_radius_scale"], low=0.0)
        for key in ("count", "ghost_count"):
            ivalue = values[key]
            if int(ivalue) != ivalue or int(ivalue) < 0:
                raise ValueError("optical.camera.flare.%s must be a non-negative "
                                 "integer" % key)
        if int(values["count"]) < 1:
            raise ValueError("optical.camera.flare.count must be >= 1")
    elif stage == "jpeg":
        quality = values["quality"]
        if int(quality) != quality or not 1 <= int(quality) <= 100:
            raise ValueError("optical.camera.jpeg.quality must be an integer in 1..100")
        span = values.get("quality_range")
        if span is not None:
            if len(span) != 2:
                raise ValueError("optical.camera.jpeg.quality_range must be [low, high]")
            low, high = (int(v) for v in span)
            if not 1 <= low <= high <= 100:
                raise ValueError("optical.camera.jpeg.quality_range must satisfy "
                                 "1 <= low <= high <= 100")


def active_stages(cfg_settings, skip=()):
    """Stage names that are switched on, in physical order."""
    skip = set(skip or ())
    out = []
    for stage in STAGES:
        if stage not in skip and cfg_settings[stage].get("enabled", False):
            _validate(stage, cfg_settings[stage])
            out.append(stage)
    return out


def has_effect(cfg, skip=()):
    """True when apply() would touch the frame at all."""
    return bool(active_stages(scene_settings(cfg), skip=skip))


# --------------------------------------------------------------------------
# tone curve helpers
# --------------------------------------------------------------------------

def srgb_to_linear(value):
    """Inverse sRGB OETF (value in 0..1, returns 0..1 linear)."""
    value = np.asarray(value, np.float64)
    return np.where(value <= 0.04045, value / 12.92,
                    ((value + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(value):
    """sRGB OETF (linear 0..1 -> encoded 0..1)."""
    value = np.clip(np.asarray(value, np.float64), 0.0, 1.0)
    return np.where(value <= 0.0031308, 12.92 * value,
                    1.055 * value ** (1.0 / 2.4) - 0.055)


def tone_gradient(linear):
    """d(output DN) / d(linear DN) at this light level (the sRGB slope).

    A readout noise quoted in output DN is 1/g as large when it is injected
    before the tone curve; that factor is what keeps a shadow pixel and a
    highlight pixel at the same *measured* read noise.
    """
    linear = np.clip(np.asarray(linear, np.float64), 0.0, 1.0)
    out = np.empty_like(linear)
    low = linear <= 0.0031308
    out[low] = 12.92
    out[~low] = 1.055 * (1.0 / 2.4) * np.power(linear[~low], 1.0 / 2.4 - 1.0)
    return out


# --------------------------------------------------------------------------
# CFA mosaic / bilinear demosaic
# --------------------------------------------------------------------------

def _cfa_masks(height, width, pattern):
    pattern = str(pattern).upper()
    if sorted(pattern) != ["B", "G", "G", "R"]:
        raise ValueError("cfa_pattern must be one of RGGB, BGGR, GRBG, GBRG")
    masks = {name: np.zeros((height, width), bool) for name in "RGB"}
    for dy in range(2):
        for dx in range(2):
            masks[pattern[dy * 2 + dx]][dy::2, dx::2] = True
    return masks


def mosaic(linear_rgb, pattern="RGGB"):
    """Keep one colour per photosite, as a Bayer-filtered sensor does.

    The returned (H, W) plane is the sensor's own signal: at an R site only
    red exists, at a G site only green.  Everything downstream must therefore
    be reconstructed, which is exactly the step that colours the noise.
    """
    linear = np.asarray(linear_rgb, np.float64)
    if linear.ndim != 3 or linear.shape[2] != 3:
        raise ValueError("mosaic expects an (H, W, 3) linear frame")
    height, width = linear.shape[:2]
    masks = _cfa_masks(height, width, pattern)
    cfa = np.zeros((height, width), np.float64)
    for index, name in enumerate("RGB"):
        mask = masks[name]
        cfa[mask] = linear[:, :, index][mask]
    return cfa


def _convolve3(plane, kernel):
    """3x3 correlation with zero padding (the frame edge is not a mirror)."""
    height, width = plane.shape
    padded = np.pad(plane, 1, mode="constant")
    out = np.zeros_like(plane)
    for dy in range(3):
        for dx in range(3):
            weight = kernel[dy][dx]
            if weight:
                out += weight * padded[dy:dy + height, dx:dx + width]
    return out


_GREEN_KERNEL = np.array([[0.0, 1.0, 0.0],
                          [1.0, 4.0, 1.0],
                          [0.0, 1.0, 0.0]]) / 4.0
_RB_KERNEL = np.array([[1.0, 2.0, 1.0],
                       [2.0, 4.0, 2.0],
                       [1.0, 2.0, 1.0]]) / 4.0


def demosaic(cfa, pattern="RGGB"):
    """Bilinear CFA reconstruction back to (H, W, 3) linear RGB.

    Bilinear is the point: the two missing colours of a site are averages of
    their own neighbours, so one noisy sample is spread over up to four
    output pixels.  That is where the correlated, blocky chroma noise of a
    real OETF/IPEG pipeline comes from.  A smarter demosaic would hide more of
    it and make the synthetic frame look cleaner than the camera it models.
    """
    cfa = np.asarray(cfa, np.float64)
    if cfa.ndim != 2:
        raise ValueError("demosaic expects an (H, W) CFA plane")
    height, width = cfa.shape
    masks = _cfa_masks(height, width, pattern)
    planes = []
    for name, kernel in (("R", _RB_KERNEL), ("G", _GREEN_KERNEL), ("B", _RB_KERNEL)):
        masked = np.where(masks[name], cfa, 0.0)
        # Normalise by the weight that actually landed on a sample of this
        # colour: at a native site that is the kernel centre (1.0), on an
        # interpolated site it is the sum of the contributing neighbours.  A
        # fixed /4 kernel is right only in the interior; at the frame border
        # it darkens or brightens whole edges.
        weight = _convolve3(masks[name].astype(np.float64), kernel)
        planes.append(_convolve3(masked, kernel) / np.maximum(weight, 1e-12))
    return np.stack(planes, axis=2)


# --------------------------------------------------------------------------
# sensor: CFA mosaic -> shot/read/FPN noise -> demosaic
# --------------------------------------------------------------------------

def electrons_per_dn(sensor):
    """Conversion gain, electrons per output DN, at the configured ISO.

    The full well is defined at ISO 100; analogue/digital gain above it scales
    the well that maps to full scale by 100/ISO.  At ISO 100 with a 12000 e-
    well one of 256 codes carries 47 e-, so a sky code of 111 holds 5200 e-
    and carries 72 e- of shot noise -- 1.5 DN.  At ISO 800 the same code holds
    650 e- and carries 25 e- -- 5.5 DN.
    """
    bits = int(sensor["bits"])
    full_scale = float(2 ** bits - 1)
    iso = float(sensor["iso"])
    return float(sensor["full_well_e"]) / full_scale / max(iso / 100.0, 1e-9)


def sensor_noise(linear_rgb, sensor, frame_seed=0):
    """Linear scene -> linear scene with detector noise, through a Bayer CFA.

    Three terms, in the order a real readout adds them:

    * shot noise -- Poisson on the photo-electrons.  It is the term that keeps
      the sky from being flat, and it is the only term that scales with light.
    * readout noise -- a Gaussian floor in output DN plus the physical
      amplifier noise in electrons.  A modern 2 e- amplifier is 0.04 DN, which
      is why the electron-only model produced 0.3 DN shadows; what an 8-bit
      stream actually shows at ISO 100 is 1.5-3 DN once row noise and the ADC
      floor are included, so those are quoted where they are measured.
    * fixed pattern -- per-photosite gain (PRNU) and offset (DSNU), plus row
      and column FPN, all drawn once from 'sensor.seed' so the same camera
      keeps the same pattern across frames.  Row readout noise is temporal and
      is drawn per frame: on a rolling shutter every row is amplified by a
      slightly different offset, which is visible as horizontal banding in the
      shadows.

    The sample is taken on the CFA before demosaicing; the caller gets back a
    full RGB plane because a frame is not deliverable until it has been
    reconstructed.  With 'demosaic' false the noise is applied per channel
    instead, which is the old, wrong model and is kept only as an A/B switch.
    """
    values = dict(DEFAULTS["sensor"])
    values.update(sensor or {})
    _validate("sensor", values)
    linear = np.clip(np.asarray(linear_rgb, np.float64), 0.0, 1.0)
    if linear.ndim != 3 or linear.shape[2] != 3:
        raise ValueError("sensor_noise expects an (H, W, 3) linear frame")
    pattern = str(values["cfa_pattern"]).upper()
    use_cfa = bool(values.get("demosaic", True))
    signal = mosaic(linear, pattern) if use_cfa else linear.copy()
    gain = electrons_per_dn(values)
    dn = signal * 255.0
    rng = np.random.default_rng(int(frame_seed))
    if values.get("shot_noise", True):
        dn = rng.poisson(np.maximum(dn, 0.0) * gain).astype(np.float64) / gain
    slope = np.maximum(tone_gradient(np.clip(signal, 0.0, 1.0)), 1e-3)
    height, width = signal.shape[:2]
    row_tail = (1,) * (signal.ndim - 1)
    column_tail = (1,) * (signal.ndim - 2)
    fixed = np.random.default_rng(int(values["seed"]))
    prnu = fixed.normal(0.0, float(values["prnu"]), signal.shape)
    dsnu = fixed.normal(0.0, float(values["dsnu_dn"]), signal.shape)
    row_fpn = fixed.normal(0.0, float(values["row_fpn_dn"]), (height,) + row_tail)
    column_fpn = fixed.normal(0.0, float(values["column_fpn_dn"]),
                              (1, width) + column_tail)
    row_temporal = rng.normal(0.0, float(values["row_noise_dn"]), (height,) + row_tail)
    read_dn = rng.normal(0.0, float(values["read_noise_dn"]), signal.shape)
    read_e = rng.normal(0.0, float(values["read_noise_e"]), signal.shape) / gain
    dn = dn * (1.0 + prnu)
    dn = dn + (dsnu + row_fpn + column_fpn + row_temporal + read_dn) / slope
    dn = dn + read_e
    # Output black level.  A camera does not hand back zero for zero light: the
    # ADC pedestal (4/255 here) keeps the shadow above the clamp, so the
    # readout noise is not rectified into a half-normal and a true-black patch
    # still shows the 1.5-3 DN a real ISO-100 frame shows.  The white point is
    # preserved, so the frame is not simply brightened.
    black = float(values["black_level_dn"])
    if black > 0.0:
        black_linear = float(srgb_to_linear(np.array(black / 255.0))) * 255.0
        dn = black_linear + dn * (255.0 - black_linear) / 255.0
    linear_out = np.clip(dn / 255.0, 0.0, 1.0)
    if use_cfa:
        linear_out = demosaic(linear_out, pattern)
    return linear_out


# --------------------------------------------------------------------------
# vignetting: the cos^4 natural falloff of a rectilinear lens
# --------------------------------------------------------------------------

def vignette_map(height, width, fov_deg=None, focal_mm=None, sensor_width_mm=None,
                 f_number=2.8, strength=1.0, mechanical=0.0):
    """Relative illumination of every pixel, 0..1, for a rectilinear lens.

    A rectilinear projection maps field angle theta to image radius
    r = f tan(theta).  The exit pupil seen from the image point shrinks by
    cos(theta) in one direction and by cos(theta) more in the other, so the
    illuminance falls as cos^4(theta) -- the classic law, not a fitted curve.
    'strength' multiplies the exponent (1.0 = textbook cos^4), and 'mechanical'
    adds the wide-aperture loss real glass shows beyond cos^4:
    1 - mechanical * (2.8 / N) * (1 - cos theta); with the default 0.0 the
    model is pure cos^4.

    The field of view comes from 'fov_deg' (what the renderer knows) or from
    'focal_mm'/'sensor_width_mm' when a real lens prescription is given.
    """
    height, width = int(height), int(width)
    if height < 1 or width < 1:
        raise ValueError("vignetting needs a non-empty frame")
    if focal_mm is not None and sensor_width_mm is not None:
        f_px = _finite("optical.camera.vignetting.focal_mm", focal_mm, positive=True)
        f_px = f_px / _finite("optical.camera.vignetting.sensor_width_mm",
                              sensor_width_mm, positive=True) * width
    elif fov_deg is not None:
        fov = _finite("optical.camera.vignetting.fov_deg", fov_deg,
                      positive=True, high=179.0)
        f_px = 0.5 * width / math.tan(math.radians(fov) / 2.0)
    else:
        raise ValueError("vignetting needs fov_deg or focal_mm+sensor_width_mm")
    yy = np.arange(height, dtype=np.float64) + 0.5 - height / 2.0
    xx = np.arange(width, dtype=np.float64) + 0.5 - width / 2.0
    radius2 = yy[:, None] ** 2 + xx[None, :] ** 2
    cos_theta = f_px / np.sqrt(f_px * f_px + radius2)
    falloff = np.power(cos_theta, 4.0 * float(strength))
    if mechanical > 0.0:
        aperture = 1.0 - float(mechanical) * (2.8 / float(f_number)) * (1.0 - cos_theta)
        falloff = falloff * np.clip(aperture, 0.05, 1.0)
    return np.clip(falloff, 0.0, 1.0)


def apply_vignetting(linear_rgb, values, fov_deg=None):
    """Attenuate the linear frame by the lens falloff (no blur, no shift)."""
    image = np.asarray(linear_rgb, np.float64)
    height, width = image.shape[:2]
    fov = values.get("fov_deg")
    if fov is None:
        fov = fov_deg
    factor = vignette_map(height, width, fov_deg=fov,
                          focal_mm=values.get("focal_mm"),
                          sensor_width_mm=values.get("sensor_width_mm"),
                          f_number=float(values["f_number"]),
                          strength=float(values["strength"]),
                          mechanical=float(values["mechanical"]))
    return np.clip(image * factor[:, :, None], 0.0, None)


# --------------------------------------------------------------------------
# lateral chromatic aberration: R/B radial magnification difference
# --------------------------------------------------------------------------

def _bilinear(channel, x, y):
    """Bilinear sample of a 2-D array at (x, y), edge-clamped."""
    height, width = channel.shape
    x0 = np.floor(x)
    y0 = np.floor(y)
    fx = x - x0
    fy = y - y0
    x0i = x0.astype(np.int64)
    y0i = y0.astype(np.int64)
    x0c = np.clip(x0i, 0, width - 1)
    x1c = np.clip(x0i + 1, 0, width - 1)
    y0c = np.clip(y0i, 0, height - 1)
    y1c = np.clip(y0i + 1, 0, height - 1)
    top = channel[y0c, x0c] * (1.0 - fx) + channel[y0c, x1c] * fx
    bottom = channel[y1c, x0c] * (1.0 - fx) + channel[y1c, x1c] * fx
    return top * (1.0 - fy) + bottom * fy


def _radial_scale(channel, scale):
    height, width = channel.shape
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    return _bilinear(channel, cx + (xx - cx) / scale, cy + (yy - cy) / scale)


def chromatic_aberration(linear_rgb, values):
    """Lateral CA: the red image plane scaled out, the blue one scaled in.

    A singlet has one focal length per wavelength.  The red focus is longer,
    so the red image is magnified by a fraction of a pixel at the corner and
    the blue one shrunk; green is the reference.  The displacement is linear
    in image radius, applied here as a bilinear resampling of the R and B
    planes, so the shift stays sub-pixel without re-rasterising the frame.
    """
    image = np.asarray(linear_rgb, np.float64)
    height, width = image.shape[:2]
    corner = math.hypot((width - 1) / 2.0, (height - 1) / 2.0)
    lateral = float(values["lateral_px"])
    red_px = values.get("red_px")
    blue_px = values.get("blue_px")
    red = float(lateral if red_px is None else red_px)
    blue = float(-lateral if blue_px is None else blue_px)
    if red == 0.0 and blue == 0.0:
        return image
    out = image.copy()
    out[:, :, 0] = _radial_scale(image[:, :, 0], 1.0 + red / max(corner, 1e-9))
    out[:, :, 2] = _radial_scale(image[:, :, 2], 1.0 + blue / max(corner, 1e-9))
    return np.clip(out, 0.0, None)


# --------------------------------------------------------------------------
# diffraction: aperture OTF * pixel-aperture MTF
# --------------------------------------------------------------------------

def diffraction_otf(shape, f_number=8.0, wavelength_um=0.55, pixel_pitch_um=3.45,
                    pixel_aperture=True):
    """Incoherent diffraction OTF of a circular aperture, in cycles/pixel.

    A stopped-down lens is not a Gaussian blur: its optical transfer function
    is the autocorrelation of the circular pupil,

        OTF(nu) = (2/pi) * (arccos(nu) - nu*sqrt(1 - nu^2)),  nu = f / f_c,

    which reaches exactly zero at the cutoff f_c = pitch / (lambda * N) -- the
    ringing Airy rings live in the PSF, and this is their spectrum.  The
    square pixel adds its own sinc(f) (a one-pixel box integrates the light
    that lands across it), which is why the usable MTF stops well before
    Nyquist.  A Gaussian kernel -- what the optical sensor model used -- has
    no cutoff and no rings, so an edge blurred by it does not look like glass.
    """
    height, width = int(shape[0]), int(shape[1])
    if height < 2 or width < 2:
        raise ValueError("diffraction needs a frame of at least 2x2")
    f_number = _finite("optical.camera.diffraction.f_number", f_number, positive=True)
    wavelength = _finite("optical.camera.diffraction.wavelength_um",
                         wavelength_um, positive=True)
    pitch = _finite("optical.camera.diffraction.pixel_pitch_um",
                    pixel_pitch_um, positive=True)
    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.rfftfreq(width)[None, :]
    frequency = np.sqrt(fx * fx + fy * fy)
    cutoff = pitch / (wavelength * f_number)
    nu = np.clip(frequency / cutoff, 0.0, 1.0)
    otf = (2.0 / np.pi) * (np.arccos(nu) - nu * np.sqrt(np.maximum(1.0 - nu * nu, 0.0)))
    if pixel_aperture:
        otf = otf * np.sinc(frequency)
    return otf


def apply_diffraction(linear_rgb, values):
    """Convolve the linear frame with the lens + pixel-aperture OTF."""
    image = np.asarray(linear_rgb, np.float64)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("apply_diffraction expects an (H, W, 3) linear frame")
    otf = diffraction_otf(image.shape[:2],
                          f_number=float(values["f_number"]),
                          wavelength_um=float(values["wavelength_um"]),
                          pixel_pitch_um=float(values["pixel_pitch_um"]),
                          pixel_aperture=bool(values.get("pixel_aperture", True)))
    out = np.empty_like(image)
    for channel in range(3):
        plane = image[:, :, channel]
        out[:, :, channel] = np.fft.irfft2(np.fft.rfft2(plane) * otf, s=plane.shape)
    return np.clip(out, 0.0, None)


# --------------------------------------------------------------------------
# lens flare: veiling glare around highlights + mirrored ghost spots
# --------------------------------------------------------------------------

def find_highlights(image_u8, threshold=0.94, count=3, min_sep=8.0):
    """Brightest pixels above 'threshold' (0..1 of full scale), kept apart.

    Only the brightest few thousand candidates are scanned: half a frame can
    be sunlit concrete above the threshold, and a full sort of those would
    cost more than the render.
    """
    image = np.asarray(image_u8)
    if image.ndim == 3:
        lum = image.max(axis=2).astype(np.float64)
    else:
        lum = image.astype(np.float64)
    ys, xs = np.nonzero(lum >= float(threshold) * 255.0)
    if not len(ys):
        return []
    if len(ys) > 4096:
        keep = np.argsort(-lum[ys, xs], kind="stable")[:4096]
        ys, xs = ys[keep], xs[keep]
    order = np.argsort(-lum[ys, xs], kind="stable")
    picked = []
    sep2 = float(min_sep) ** 2
    for i in order:
        y = int(ys[i])
        x = int(xs[i])
        if all((y - py) ** 2 + (x - px) ** 2 >= sep2 for py, px in picked):
            picked.append((y, x))
            if len(picked) >= int(count):
                break
    return picked


def apply_flare(linear_rgb, values):
    """Add veiling glare around highlights and their mirrored ghosts.

    A real lens scatters a few percent of a bright source across the image
    plane, and every air/glass surface reflects part of it back as a dim,
    inverted copy of the source.  The model is the two visible consequences: a
    Gaussian halo at the source and up to 'ghost_count' spots mirrored through
    the frame centre, each further away and dimmer than the last.  Highlights
    are found on the display-referred value, because the threshold an operator
    sets is the one they see; the light itself is added linearly.  With no
    source above the threshold the frame is returned untouched, so enabling
    the switch stays inert on a scene without highlights.
    """
    image = np.clip(np.asarray(linear_rgb, np.float64), 0.0, None)
    display = linear_to_srgb(np.clip(image, 0.0, 1.0)) * 255.0
    sources = find_highlights(display, threshold=float(values["threshold"]),
                              count=int(values["count"]), min_sep=6.0)
    if not sources:
        return image
    height, width = image.shape[:2]
    strength = float(values["strength"])
    sigma = float(values["radius_px"])
    ghost_scale = float(values["ghost_scale"])
    ghost_radius = sigma * float(values["ghost_radius_scale"])
    ghost_count = int(values["ghost_count"])
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0
    tint = np.array([1.0, 0.96, 0.84])          # warm, like most coatings
    add = np.zeros((height, width, 3), np.float64)
    for (y, x) in sources:
        peak = float(image[y, x].max())
        halo = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * sigma ** 2))
        add += (strength * peak * halo)[:, :, None] * tint
        for k in range(ghost_count):
            step = ghost_scale * (k + 1)
            gy = cy + step * (cy - y)
            gx = cx + step * (cx - x)
            if not (0.0 <= gy < height and 0.0 <= gx < width):
                continue
            amp = strength * peak * 0.6 / (k + 1)
            ghost = np.exp(-((xx - gx) ** 2 + (yy - gy) ** 2) / (2.0 * ghost_radius ** 2))
            add += (amp * ghost)[:, :, None] * tint
    return np.clip(image + add, 0.0, None)


# --------------------------------------------------------------------------
# JPEG
# --------------------------------------------------------------------------

def jpeg_roundtrip(image_u8, quality=85):
    """Encode + decode a frame at 'optical.camera.jpeg.quality'.

    A dataset that ships JPEG files does not show the sensor's own noise; it
    shows noise *after* 8x8 DCT quantisation, which smears it into blocks and
    rings around every edge.  Reproducing that needs the real codec, so this
    calls libjpeg through an in-memory Pillow buffer.  Pillow is imported
    here, not at module scope: Blender's bundled Python does not ship it, and
    a top-level import would break every render.
    """
    quality = int(quality)
    if not 1 <= quality <= 100:
        raise ValueError("JPEG quality must be an integer in 1..100")
    try:
        from PIL import Image
    except ImportError as exc:                  # pragma: no cover - Blender only
        raise RuntimeError("JPEG 后处理需要 Pillow（Blender 自带 Python 未安装）") from exc
    import io
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(image_u8, np.uint8), mode="RGB").save(
        buffer, format="JPEG", quality=quality, subsampling=2, optimize=False)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return np.asarray(decoded.convert("RGB"), np.uint8)


def write_jpeg(path, image_u8, quality=85):
    """Write a real '.jpg' file (PNG stays the lossless default product)."""
    quality = int(quality)
    if not 1 <= quality <= 100:
        raise ValueError("JPEG quality must be an integer in 1..100")
    try:
        from PIL import Image
    except ImportError as exc:                  # pragma: no cover - Blender only
        raise RuntimeError("JPEG 后处理需要 Pillow（Blender 自带 Python 未安装）") from exc
    Image.fromarray(np.asarray(image_u8, np.uint8), mode="RGB").save(
        path, format="JPEG", quality=quality, subsampling=2)


# --------------------------------------------------------------------------
# whole chain
# --------------------------------------------------------------------------

def apply(image_u8, cfg, frame_seed=0, fov_deg=None, skip=()):
    """Run the enabled stages over one 8-bit frame.

    Returns dict(image_u8=..., metadata=..., changed=bool).  When nothing is
    switched on the input array object is returned unchanged and 'changed' is
    False, which lets the Blender renderer skip the decode/edit/re-encode
    cycle and leave the PNG byte-identical.

    Lens stages run in linear light and are encoded back once; the sensor
    stage quantises at the end.  That keeps a single quantisation step in the
    chain instead of one per stage.
    """
    image = np.asarray(image_u8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("camera postprocess expects an (H, W, 3) frame")
    if image.dtype != np.uint8:
        raise ValueError("camera postprocess expects an 8-bit frame")
    values = scene_settings(cfg)
    stages = active_stages(values, skip=skip)
    metadata = dict(model="camera defect chain (Bayer sensor + lens)", stages=stages,
                    applied=[], warnings=[])
    if not stages:
        return dict(image_u8=image_u8, metadata=metadata, changed=False)
    linear_stages = [stage for stage in stages if stage != "jpeg"]
    out = image
    if linear_stages:
        linear = srgb_to_linear(out.astype(np.float64) / 255.0)
        for stage in linear_stages:
            if stage == "vignetting":
                linear = apply_vignetting(linear, values["vignetting"], fov_deg=fov_deg)
            elif stage == "chromatic_aberration":
                linear = chromatic_aberration(linear, values["chromatic_aberration"])
            elif stage == "diffraction":
                linear = apply_diffraction(linear, values["diffraction"])
                metadata.update(
                    diffraction_f_number=float(values["diffraction"]["f_number"]),
                    diffraction_wavelength_um=float(
                        values["diffraction"]["wavelength_um"]),
                    pixel_pitch_um=float(values["diffraction"]["pixel_pitch_um"]))
            elif stage == "flare":
                linear = apply_flare(linear, values["flare"])
            elif stage == "sensor":
                if bool(values["sensor"].get("demosaic", True)):
                    metadata["cfa_pattern"] = str(values["sensor"]["cfa_pattern"]).upper()
                linear = sensor_noise(linear, values["sensor"], frame_seed=frame_seed)
                metadata.update(iso=float(values["sensor"]["iso"]),
                                electrons_per_dn=electrons_per_dn(values["sensor"]),
                                read_noise_e=float(values["sensor"]["read_noise_e"]),
                                read_noise_dn=float(values["sensor"]["read_noise_dn"]),
                                dsnu_dn=float(values["sensor"]["dsnu_dn"]),
                                row_fpn_dn=float(values["sensor"]["row_fpn_dn"]),
                                column_fpn_dn=float(values["sensor"]["column_fpn_dn"]),
                                prnu=float(values["sensor"]["prnu"]))
        out = np.rint(np.clip(linear_to_srgb(linear), 0.0, 1.0) * 255.0).astype(np.uint8)
    if "jpeg" in stages:
        try:
            quality = int(values["jpeg"]["quality"])
            span = values["jpeg"].get("quality_range")
            if span is not None:
                # A dataset scraped from real phones does not sit at one
                # quality: one frame is 72, the next is 90.  Drawing per frame
                # stops a detector from learning a single quantisation table.
                low, high = (int(v) for v in span)
                quality = int(np.random.default_rng(int(frame_seed)).integers(low, high + 1))
            out = jpeg_roundtrip(out, quality)
            metadata["jpeg_quality"] = quality
        except RuntimeError as exc:
            # A missing codec must not fail a multi-hour render; the frame
            # stays lossless and the label records why.
            metadata["warnings"].append(str(exc))
    metadata["applied"] = [stage for stage in stages
                           if not (stage == "jpeg" and "jpeg_quality" not in metadata)]
    return dict(image_u8=out, metadata=metadata, changed=True)

