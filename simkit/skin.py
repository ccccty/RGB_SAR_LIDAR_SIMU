"""Procedural skin texture: albedo modulation as a function of object-space position.

STATUS: wired into the optical pass. The plate is baked as albedo x modulation
and connected through GENERATED coordinates into Base Color. Two things had to
be fixed first, and both are worth remembering:

  1. add_mesh_object builds meshes with from_pydata, which creates NO UV layer.
     An image texture without an explicit vector input samples a single texel,
     so the plate collapsed to one flat colour and looked like a no-op.
  2. The plate stores LINEAR albedo. Read as sRGB, Blender gamma-decodes it and
     the airframe came out at 72 DN instead of 95.

Measured after both fixes: target-region detail 472 -> 546 (0.38x -> 0.44x of a
reference photograph), contrast unchanged, brightness restored. Sweeps over
pattern amplitude (1/2/3/5) and texel scale (0.02/0.04/0.08 m) both put the
current defaults at the optimum.

Why this exists
---------------
A rendered airframe with one flat material has no surface detail at all.  Measured
against the reference photographs in assets/refs: our Laplacian variance 35.4 vs
4660 for a real F-15, contrast 0.046 vs 0.18.  The real surface carries skin
joints, rivet rows, grime running downstream of every joint, and panels painted
at slightly different times.  Those features are what the statistics are made of.

This module synthesises them as a *multiplicative albedo factor*, 1.0 = unchanged.
It is pure numpy: no Blender, no external texture files, no new dependencies, and
it is evaluated from the 3-D position of the surface point -- never from a screen
coordinate -- so the pattern is welded to the object and cannot crawl when the
camera moves.

    field = SkinField("aircraft", size=(19.4, 13.0, 5.6), seed=0,
                      pixel_scale_m=0.04)
    albedo = base_albedo * field(points)      # points: (N, 3) object-local metres

Coordinate convention (matches the scene builder)
-------------------------------------------------
-x is forward: the airflow, and for a hull the bow.  +z is up.  y is the
transverse axis.  Points are in the object's own frame, which is why the texture
stays put when the object moves.  modulation_world applies an inverse pose for
callers that only have world coordinates.

Layers
------
1. seams    Panel/frame joints.  Circumferential rings every panel_pitch_m in x,
            longitudinal lines every stringer_spacing_m of arc around the x axis.
            10-20% darker, with per-joint depth variation and long-wavelength
            fade so the grid does not read as a printed decal.
2. rivets   Rows of dots flanking every joint, offset by rivet_offset_m, pitch
            rivet_spacing_m, position jittered per dot, 5-10% darker.
3. grime    Airflow streaks stretched along -x (2-6 m long, 0.25-0.4 m wide),
            locked to the downstream side of every circumferential joint, plus
            gravity run-off stretched around the body either side of a
            longitudinal line (where water leaves a stringer or a weld).  The
            two are mixed by grime_gravity.  5-15% darker.
4. panels   One random brightness offset per (dx, dtheta) cell, 1-3%, piecewise
            constant -- the "repainted at a different time" look.
5. mottle   Large-scale paint film variation, symmetric +-2%, mean-neutral.

Category presets (PRESETS) change joint density, joint depth, rivet pitch, streak
length and how much of the grime is gravity-driven: a warship is welded, streaked
and long; an airliner is clean and lightly panelled; a ground vehicle is
short-panelled and dirty.

Numbers that matter
-------------------
*   Deterministic: the value depends only on (position, seed, resolved params).
    Re-evaluating the same coordinate returns the same float, bit for bit.
*   Mean preserving: the analytic expected darkening of the enabled layers is
    compensated by one multiplicative gain derived from the parameters, so
    mean(modulation) ~ 1.0 and colour/solar absorptivity do not shift.
*   Vectorised: 100k points in ~50 ms, 1,000 x 1,000 points in ~0.5-0.6 s
    (evaluated in cache-sized blocks above 100k, so peak memory stays flat).

Pixel-scale filtering
---------------------
A 2 cm joint is sub-pixel at 4 cm/px and is erased by the camera PSF.  Pass
pixel_scale_m (metres of surface per rendered pixel, e.g. size_m/bbox_px) and the
narrow layers are widened to a floor of min_seam_px pixels (default 1.8) so they
survive the optics.  Rivets are widened the same way but their amplitude is
reduced to roughly preserve the total ink, so a far target grows panel lines and
grime rather than rows of black dots.

The pattern is defined on a fuselage-like parameterisation (x, theta around x).
Wings and flat panels away from the axis still get joints, spaced by the local
radius clamped to the body, so they read as radiating rib lines; that is an
approximation and it is noted at _RADIUS_CLAMP.

Metric scales: airframe skin panels 0.3-1.0 m, joint gap 0.01-0.02 m, rivet pitch
0.05-0.10 m.  Targets for the resulting modulation are the texture_hf_ratio band
0.012-0.043 in configs/refs_params.json and a modulation std of 3-8% at a
0.03-0.05 m pixel footprint.

Measured on this machine, 24 m unrolled patch sampled at 0.04 m/px (the
1.2 km / 367 px framing): aircraft mean 1.0008 std 0.0729, ship mean 1.0013
std 0.0716, vehicle mean 0.9934 std 0.0713; min 0.46, p1 0.74, p99 1.08.
Applying that field to the acceptance frame multiplies its Laplacian variance
by 2.7x (13.7 -> 36.9) while leaving the mean colour untouched.
"""

import math

import numpy as np

__all__ = [
    "SkinField", "modulation", "preset", "PRESETS", "CATEGORY_ALIASES",
    "hf_ratio", "preview_patch", "write_preview", "DEFAULT_MIN_SEAM_PX",
]


# ---------------------------------------------------------------------------
# Presets: metric skin parameters per platform family
# ---------------------------------------------------------------------------
# Every length is in metres and describes the *real* aircraft/ship/vehicle, not
# the render.  body_radius_m is the nominal distance from the x axis to the skin:
# it sets how many longitudinal lines the skin ring is divided into and how arc
# length is measured for widths.  default_length_m is the length the preset was
# written for; another object length scales the radius by L/default.

_BASE_AIRCRAFT = dict(
    default_length_m=12.0, body_radius_m=0.95,
    panel_pitch_m=0.85, stringer_spacing_m=0.60,
    seam_width_m=0.020, seam_depth=0.17,
    rivet_spacing_m=0.070, rivet_radius_m=0.009, rivet_offset_m=0.022,
    rivet_darken=0.075,
    grime_strength=0.13, grime_streak_len_m=2.6, grime_width_m=0.26,
    grime_run_m=0.90, grime_gravity=0.25,
    panel_var=0.022, mottle=0.020,
)

_BASE_SHIP = dict(
    default_length_m=154.0, body_radius_m=8.0,
    panel_pitch_m=1.60, stringer_spacing_m=1.20,
    seam_width_m=0.030, seam_depth=0.20,
    rivet_spacing_m=0.120, rivet_radius_m=0.012, rivet_offset_m=0.035,
    rivet_darken=0.050,
    grime_strength=0.18, grime_streak_len_m=5.5, grime_width_m=0.38,
    grime_run_m=1.60, grime_gravity=0.70,
    panel_var=0.028, mottle=0.024,
)

_BASE_VEHICLE = dict(
    default_length_m=4.6, body_radius_m=0.90,
    panel_pitch_m=1.05, stringer_spacing_m=0.80,
    seam_width_m=0.018, seam_depth=0.14,
    rivet_spacing_m=0.090, rivet_radius_m=0.010, rivet_offset_m=0.028,
    rivet_darken=0.060,
    grime_strength=0.17, grime_streak_len_m=1.1, grime_width_m=0.30,
    grime_run_m=0.50, grime_gravity=0.80,
    panel_var=0.026, mottle=0.028,
)


def _variant(base, **changes):
    out = dict(base)
    out.update(changes)
    return out


PRESETS = {
    # --- the three families the caller asks for ----------------------------
    "aircraft": dict(_BASE_AIRCRAFT),
    "ship": dict(_BASE_SHIP),
    "vehicle": dict(_BASE_VEHICLE),
    # --- airframes ---------------------------------------------------------
    # Civil transports: fewer, larger panels, clean white paint, weak joints.
    "airliner": _variant(_BASE_AIRCRAFT,
                         default_length_m=38.0, body_radius_m=2.10,
                         panel_pitch_m=1.05, stringer_spacing_m=0.75,
                         seam_depth=0.13, rivet_darken=0.060,
                         grime_strength=0.08, panel_var=0.015, mottle=0.014),
    # Helicopters: small, dense, oil-stained, usually dark paint.
    "helicopter": _variant(_BASE_AIRCRAFT,
                           default_length_m=13.0, body_radius_m=1.10,
                           panel_pitch_m=0.62, stringer_spacing_m=0.45,
                           seam_depth=0.16, grime_strength=0.15,
                           grime_streak_len_m=1.4, grime_gravity=0.55,
                           panel_var=0.026, mottle=0.022),
    # Composite UAV: bonded, mostly joint-free; what is left is mould seams.
    "uav_fixed": _variant(_BASE_AIRCRAFT,
                          default_length_m=11.0, body_radius_m=0.55,
                          panel_pitch_m=0.70, stringer_spacing_m=0.50,
                          seam_depth=0.10, rivet_darken=0.030,
                          rivet_spacing_m=0.100,
                          grime_strength=0.10, panel_var=0.018, mottle=0.022),
    # Small rotary UAV: injection-moulded shell, no rivets at all.
    "uav_rotor": _variant(_BASE_AIRCRAFT,
                          default_length_m=0.30, body_radius_m=0.05,
                          panel_pitch_m=0.10, stringer_spacing_m=0.09,
                          seam_width_m=0.004, seam_depth=0.10,
                          rivet_darken=0.0, rivet_spacing_m=0.05,
                          grime_strength=0.10, grime_streak_len_m=0.5,
                          grime_width_m=0.06, grime_run_m=0.15,
                          grime_gravity=0.20, panel_var=0.020, mottle=0.025),
    # --- hulls -------------------------------------------------------------
    # Warship: welded plate, strong rust run-off, long streaks.
    "warship": _variant(_BASE_SHIP,
                        default_length_m=154.0, body_radius_m=8.0,
                        panel_pitch_m=1.60, seam_depth=0.21,
                        grime_strength=0.18, panel_var=0.026),
    # Merchant ship: fewer, bigger plates; rust still dominates.
    "vessel": _variant(_BASE_SHIP,
                       default_length_m=200.0, body_radius_m=9.0,
                       panel_pitch_m=2.20, seam_depth=0.15,
                       rivet_darken=0.030, grime_strength=0.15,
                       panel_var=0.024),
    # --- ground ------------------------------------------------------------
    "mil_vehicle": _variant(_BASE_VEHICLE,
                            default_length_m=7.5, body_radius_m=1.10,
                            panel_pitch_m=1.30, seam_depth=0.15,
                            grime_strength=0.20, panel_var=0.030, mottle=0.030),
}

# Chinese category names from configs/refs_params.json and simkit material names
# both route to a preset, so the renderer can pass whatever it already has.
CATEGORY_ALIASES = {
    "aircraft": "aircraft", "airplane": "aircraft", "plane": "aircraft",
    "fixed_wing": "aircraft", "military_aircraft": "aircraft", "jet": "aircraft",
    "军用飞机": "aircraft", "飞机": "aircraft",
    "airliner": "airliner", "civil_air": "airliner",
    "民航飞机": "airliner",
    "helicopter": "helicopter", "rotorcraft": "helicopter",
    "直升机": "helicopter",
    "uav_fixed": "uav_fixed", "uav": "uav_fixed", "drone": "uav_fixed",
    "固定翼无人机": "uav_fixed",
    "uav_rotor": "uav_rotor", "quadcopter": "uav_rotor",
    "旋翼无人机": "uav_rotor",
    "ship": "ship", "boat": "ship", "舰船": "ship",
    "船舶": "ship", "船只": "ship",
    "warship": "warship", "navy": "warship", "军舰": "warship",
    "vessel": "vessel", "merchant": "vessel",
    "民用船舶": "vessel",
    "vehicle": "vehicle", "car": "vehicle", "truck": "vehicle",
    "地面车辆": "vehicle", "车辆": "vehicle",
    "mil_vehicle": "mil_vehicle", "tank": "mil_vehicle",
    "军用地面车辆": "mil_vehicle",
    # simkit material names (simkit/materials.py) -- lets the renderer key off
    # the material it already assigned instead of re-deriving the category.
    "painted_metal": "aircraft", "paint_civil_air": "airliner",
    "paint_uav_fixed": "uav_fixed", "paint_uav_rotor": "uav_rotor",
    "paint_warship": "warship", "paint_vessel": "vessel",
    "paint_scenery": "vehicle", "metal_bare": "aircraft", "glass": "aircraft",
}

DEFAULT_MIN_SEAM_PX = 1.8

# Points per evaluation block.  Above this the layer stack is walked in chunks;
# the arithmetic is element-wise so the result does not change.
_CHUNK = 100000


# ---------------------------------------------------------------------------
# Deterministic hashing and value noise
# ---------------------------------------------------------------------------

# The hash runs in 32 bits: value noise needs ~21 mantissa bits and half the
# memory traffic of a 64-bit mix, which is what the inner loop is bounded by.
_U32 = np.uint32
_M1 = _U32(0x9E3779B1)
_M2 = _U32(0x85EBCA77)
_M3 = _U32(0xC2B2AE3D)
_K1 = _U32(0x7FEB352D)
_K2 = _U32(0x846CA68B)
_S16 = _U32(16)
_S15 = _U32(15)
_SHIFT11 = _U32(11)
# 2**-21: the hash keeps 21 mantissa bits, which is far more than value noise
# needs and lets the whole interpolation run in float32.
_INV21 = 1.0 / 2097152.0
_ONE = _U32(1)


def _seed32(seed):
    """Fold an arbitrary integer seed into 32 bits (negative values included)."""
    return _U32(int(seed) & 0xFFFFFFFF)


def _hash_u32(ux, uy, uz, seed):
    """Avalanche-mixed 32-bit hash of integer lattice coords and a seed.

    Three odd-constant products xor'd together are already well spread; two
    multiply-xorshift rounds (the lowbias32 finaliser) finish the avalanche.
    """
    h = (ux * _M1) ^ (uy * _M2) ^ (uz * _M3) ^ seed
    h = h ^ (h >> _S16)
    h = h * _K1
    h = h ^ (h >> _S15)
    h = h * _K2
    h = h ^ (h >> _S16)
    return h


def _hash3(ix, iy, iz, seed):
    """Same hash for callers holding signed integer coordinates."""
    return _hash_u32(ix.astype(_U32), iy.astype(_U32), iz.astype(_U32), seed)


def _unit(h):
    """Map a hash to a float64 in [0, 1)."""
    return (h >> _SHIFT11).astype(np.float64) * _INV21


def _unit32(h):
    """Map a hash to a float32 in [0, 1); used inside the noise interpolation."""
    return (h >> _SHIFT11).astype(np.float32) * np.float32(_INV21)


def _rand(ix, iy, iz, seed):
    return _unit(_hash3(ix, iy, iz, seed))


def _value_noise3(x, y, z, seed):
    """Trilinearly interpolated 3-D value noise on the integer lattice, in [0, 1).

    The eight corner hashes are the dominant cost of the whole module, so the
    lattice coordinates are cast to uint32 once and the hash is inlined rather
    than routed through a helper that would re-cast on every corner.
    """
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    z = np.asarray(z, dtype=np.float32)
    x0 = np.floor(x)
    y0 = np.floor(y)
    z0 = np.floor(z)
    fx = x - x0
    fy = y - y0
    fz = z - z0
    # smoothstep weights kill the lattice creases that plain lerp leaves behind
    fx = fx * fx * (3.0 - 2.0 * fx)
    fy = fy * fy * (3.0 - 2.0 * fy)
    fz = fz * fz * (3.0 - 2.0 * fz)
    ux = x0.astype(_U32)
    uy = y0.astype(_U32)
    uz = z0.astype(_U32)
    ux1 = ux + _ONE
    uy1 = uy + _ONE
    uz1 = uz + _ONE

    def at(i, j, k):
        return _unit32(_hash_u32(i, j, k, seed))

    a00 = at(ux, uy, uz)
    b00 = at(ux1, uy, uz)
    a10 = at(ux, uy1, uz)
    b10 = at(ux1, uy1, uz)
    a01 = at(ux, uy, uz1)
    b01 = at(ux1, uy, uz1)
    a11 = at(ux, uy1, uz1)
    b11 = at(ux1, uy1, uz1)
    c00 = a00 + (b00 - a00) * fx
    c10 = a10 + (b10 - a10) * fx
    c01 = a01 + (b01 - a01) * fx
    c11 = a11 + (b11 - a11) * fx
    c0 = c00 + (c10 - c00) * fy
    c1 = c01 + (c11 - c01) * fy
    return c0 + (c1 - c0) * fz


def _fbm3(x, y, z, seed, octaves=2, gain=0.5):
    """Fractional Brownian motion of value noise, roughly in [0, 1)."""
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    z = np.asarray(z, dtype=np.float32)
    total = None
    amp = 1.0
    norm = 0.0
    scale = 1.0
    for octave in range(max(1, int(octaves))):
        n = _value_noise3(x * np.float32(scale) + np.float32(17.31 * octave),
                          y * np.float32(scale) + np.float32(9.73 * octave),
                          z * np.float32(scale) + np.float32(5.17 * octave),
                          seed + 7919 * octave)
        if total is None:
            total = np.float32(amp) * n
        else:
            total += np.float32(amp) * n
        norm += amp
        amp *= gain
        scale *= 2.0
    return total * np.float32(1.0 / norm)


def _smoothstep(lo, hi, x):
    t = np.clip((x - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def preset(name):
    """Resolve a category/material name to a copy of its parameter dict."""
    if isinstance(name, dict):
        return dict(name)
    key = str(name).strip()
    resolved = CATEGORY_ALIASES.get(key)
    if resolved is None:
        resolved = CATEGORY_ALIASES.get(key.lower())
    if resolved is None:
        resolved = key.lower()
    if resolved not in PRESETS:
        raise KeyError("unknown skin category %r (known: %s)"
                       % (name, ", ".join(sorted(PRESETS))))
    return dict(PRESETS[resolved])


def _as_points(points):
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim == 1:
        if arr.size != 3:
            raise ValueError("points must be (N, 3) or a single (3,) point")
        arr = arr.reshape(1, 3)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("points must be (N, 3); got shape %r" % (arr.shape,))
    return arr


def hf_ratio(image):
    """High-frequency power fraction, the metric used by configs/refs_params.json.

    sum PSD(f >= 0.25 cyc/px) / sum PSD(0.02 .. 0.5 cyc/px), Hann-windowed, DC
    removed.  Same definition as the extraction tool behind refs_params.json;
    duplicated here (a dozen lines of FFT) so the module stays importable without
    the analysis stack.
    """
    a = np.asarray(image, dtype=np.float64)
    if a.ndim != 2 or min(a.shape) < 16:
        return float("nan")
    a = a - a.mean()
    h, w = a.shape
    window = np.outer(np.hanning(h), np.hanning(w))
    spec = np.fft.fftshift(np.fft.fft2(a * window))
    power = spec.real ** 2 + spec.imag ** 2
    fy = np.fft.fftshift(np.fft.fftfreq(h))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(w))[None, :]
    radius = np.sqrt(fx * fx + fy * fy)
    base = power[(radius >= 0.02) & (radius <= 0.5)].sum()
    if base <= 0:
        return float("nan")
    return float(power[radius >= 0.25].sum() / base)


# ---------------------------------------------------------------------------
# The field
# ---------------------------------------------------------------------------

# Local radius used to convert an angular offset into metres.  Wings and flat
# panels sit far from the x axis, where the angle changes much faster than the
# surface does; clamping the radius keeps joint spacing finite out there.  The
# joints then radiate from the axis like rib lines, which is an acceptable
# approximation for a first pass and is stated in the module docstring.
_RADIUS_CLAMP = (0.30, 2.5)

# Mean of the two streak masks over a uniform surface sample.  The masks are
# smoothstep() of a 2-octave value-noise fBm, whose marginal distribution does
# not depend on the sampling scale, so these are constants of the layer
# definition and not of the caller's sampling.  Calibrated on a 20 m x 20 m
# unrolled patch (tests/test_skin.py::test_mean_is_preserved).
_GRIME_FLOW_MEAN = 0.31
_GRIME_GRAV_MEAN = 0.25

# Threshold bands on the streak noise.  A narrow band turns the smooth fBm into
# distinct stains instead of a uniform wash; the widths were chosen so the mean
# darkening stays in the 1-4% range while the darkest streak cores reach the
# layer strength.
_GRIME_FLOW_BAND = (0.51, 0.63)
_GRIME_GRAV_BAND = (0.55, 0.72)

# Mean of a joint's peak darkening across the line: edge falloff plus the
# per-joint strength distribution (0.62..1.17, mean ~0.90).
_SEAM_MEAN_FRACTION = 0.88


class SkinField:
    """A deterministic procedural skin, evaluated per surface point.

    Parameters
    ----------
    category : str or dict
        Preset name ("aircraft", "ship", "vehicle", "warship", a Chinese
        category such as the one in refs_params.json, or a simkit material name
        such as "paint_warship"; see CATEGORY_ALIASES) or a full parameter dict.
    size : float or (3,) sequence, optional
        Object length in metres, or its (length, width, height) bounding box.
        Only the length and the body radius derived from it matter.
    seed : int
        Any integer.  The same seed reproduces the same field exactly.
    enabled : bool
        False short-circuits to an array of exactly 1.0.
    seams, rivets, grime, panels, mottle : bool
        Per-layer switches.  A disabled layer contributes exactly 0.0, so the
        result is bit-reproducible with or without the switch.
    contrast : float
        One knob over every layer amplitude (1.0 = preset, 1.4 = 40% stronger).
        Use it to trade realism for the contrast a detector's input needs.
    normalize_mean : bool
        Compensate the analytic expected darkening so the mean stays at 1.0.
    pixel_scale_m : float, optional
        Metres of surface per rendered pixel.  Widens sub-pixel joints to
        min_seam_px pixels so the camera PSF does not erase them.
    min_seam_px : float
        Minimum visual width of a joint, in pixels, when pixel_scale_m is set.
    params : dict, optional
        Extra parameter overrides (same keys as PRESETS).
    **overrides
        Shorthand for params.
    """

    def __init__(self, category="aircraft", size=None, seed=0, enabled=True,
                 seams=True, rivets=True, grime=True, panels=True, mottle=True,
                 contrast=1.0, normalize_mean=True, pixel_scale_m=None,
                 min_seam_px=DEFAULT_MIN_SEAM_PX, params=None, **overrides):
        self.category = category if isinstance(category, str) else "custom"
        self.seed = int(seed)
        self.enabled = bool(enabled)
        self.layers = dict(seams=bool(seams), rivets=bool(rivets),
                           grime=bool(grime), panels=bool(panels),
                           mottle=bool(mottle))
        self.normalize_mean = bool(normalize_mean)
        self.min_seam_px = float(min_seam_px)
        p = preset(category)
        if params:
            p.update(params)
        if overrides:
            p.update(overrides)
        self._check_params(p)
        p = self._apply_contrast(p, float(contrast))
        self.params = p
        self.contrast = float(contrast)

        length, radius = self._geometry(size, p)
        self.length = length
        self.radius = radius
        pitch = min(p["panel_pitch_m"], max(0.06, length / 5.0))
        self.pitch = float(max(pitch, 1e-3))
        spacing = min(p["stringer_spacing_m"], max(0.04, length / 6.0))
        count = int(round(2.0 * np.pi * radius / max(spacing, 1e-4)))
        self.ring_count = int(np.clip(count, 4, 40))
        self.dtheta = 2.0 * np.pi / self.ring_count
        self.arc = 2.0 * np.pi * radius / self.ring_count
        self.pixel_scale_m = None if pixel_scale_m is None else float(pixel_scale_m)
        # Layer salts: changing one layer's pattern must not shift the others.
        self._salts = dict(seam=0x51F3, rivet=0x9A17, grime=0x2C6D,
                           panel=0x71B5, mottle=0xD4E9)

    # -- construction helpers ------------------------------------------------

    @staticmethod
    def _check_params(p):
        known = set(_BASE_AIRCRAFT)
        if not known.issuperset(p):
            raise KeyError("unknown skin parameters: %s"
                           % ", ".join(sorted(set(p) - known)))
        missing = known - set(p)
        if missing:
            raise KeyError("missing skin parameters: %s" % ", ".join(sorted(missing)))

    @staticmethod
    def _geometry(size, p):
        if size is None:
            return float(p["default_length_m"]), float(p["body_radius_m"])
        if np.isscalar(size):
            length = float(size)
            radius = 0.065 * length
        else:
            dims = [float(v) for v in size]
            if len(dims) == 1:
                length = dims[0]
                radius = 0.065 * length
            elif len(dims) >= 3:
                length = dims[0]
                radius = 0.32 * min(dims[1], dims[2])
            else:
                raise ValueError("size must be a length or a (length, width, height)")
        if not np.isfinite(length) or length <= 0.0:
            raise ValueError("size must be positive and finite")
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError("size must be positive and finite")
        if np.isscalar(size) or (not np.isscalar(size) and len(size) < 3):
            # A bare length says nothing about the cross-section, so scale the
            # preset's body radius with the object's length: a 40 m aircraft and
            # a 12 m one must not share the same longitudinal count.
            radius *= float(np.clip(length / p["default_length_m"], 0.35, 3.0))
        return length, float(np.clip(radius, 0.01, 60.0))

    @staticmethod
    def _apply_contrast(p, contrast):
        p = dict(p)
        c = float(np.clip(contrast, 0.0, 4.0))
        p["seam_depth"] = float(np.clip(p["seam_depth"] * c, 0.0, 0.34))
        p["rivet_darken"] = float(np.clip(p["rivet_darken"] * c, 0.0, 0.18))
        p["grime_strength"] = float(np.clip(p["grime_strength"] * c, 0.0, 0.34))
        p["panel_var"] = float(np.clip(p["panel_var"] * c, 0.0, 0.07))
        p["mottle"] = float(np.clip(p["mottle"] * c, 0.0, 0.09))
        return p

    def with_contrast(self, contrast):
        """A copy of this field with every layer amplitude scaled."""
        return SkinField(self.category, size=self.length, seed=self.seed,
                         enabled=self.enabled, contrast=contrast,
                         normalize_mean=self.normalize_mean,
                         pixel_scale_m=self.pixel_scale_m,
                         min_seam_px=self.min_seam_px,
                         params=dict(self.params), **self.layers)

    @classmethod
    def for_framing(cls, category, size, pixels_across, **kwargs):
        """Build a field for a target covering pixels_across pixels.

        Convenience wrapper: the pixel footprint is length / pixels_across, which
        is what the joint-width floor needs to know.
        """
        length = float(size) if np.isscalar(size) else float(size[0])
        return cls(category, size=size,
                   pixel_scale_m=length / max(float(pixels_across), 1.0), **kwargs)

    # -- resolved widths -----------------------------------------------------

    def _widths(self, pixel_scale_m):
        p = self.params
        px = self.pixel_scale_m if pixel_scale_m is None else float(pixel_scale_m)
        width = p["seam_width_m"]
        if px and px > 0.0:
            width = max(width, self.min_seam_px * px)
        width = min(width, 0.45 * self.pitch)
        radius = p["rivet_radius_m"]
        rivet_scale = 1.0
        if px and px > 0.0:
            grown = min(max(radius, 0.55 * px), 0.35 * p["rivet_spacing_m"])
            if grown > radius:
                # A dot that grows to fill a pixel must lose most of its density,
                # otherwise a distant aircraft turns into a black dotted mass.
                rivet_scale = max(min(1.0, 1.3 * (radius / grown) ** 2), 0.25)
                radius = grown
        aa_seam = 0.5 * (px if (px and px > 0) else width)
        aa_seam = max(aa_seam, 0.25 * width)
        aa_rivet = max(0.5 * (px if (px and px > 0) else 0.0), 0.35 * radius)
        return dict(px=px, seam_width=width, rivet_radius=radius,
                    rivet_scale=rivet_scale, aa_seam=aa_seam,
                    aa_rivet=aa_rivet)

    # -- layer pieces --------------------------------------------------------

    def _stripe(self, dist, width, aa):
        # Full coverage inside half the width, linear ramp over the filter width.
        return np.clip(0.5 + (0.5 * width - np.abs(dist)) / aa, 0.0, 1.0)

    def _line_strength(self, index, along, salt):
        # Per-joint depth varies, and every ~1.8 m of seam it fades in and out.
        # Without this the joint grid reads as a printed decal.
        cell = np.floor(along / 1.8).astype(np.int64)
        r = _rand(index.astype(np.int64), cell,
                  np.zeros_like(cell), _seed32(self.seed + salt))
        return 0.62 + 0.55 * r

    def _seams(self, x, theta, rc, w, aa, out):
        p = self.params
        u = x / self.pitch
        ring = np.rint(u)
        dist = (u - ring) * self.pitch
        strength = self._line_strength(ring, theta * rc, self._salts["seam"])
        out += p["seam_depth"] * np.clip(self._stripe(dist, w, aa) * strength, 0.0, 1.0)

        q = theta / self.dtheta
        line = np.rint(q)
        arc_dist = (q - line) * self.dtheta * rc
        strength = self._line_strength(line, x, self._salts["seam"] + 1)
        out += p["seam_depth"] * np.clip(self._stripe(arc_dist, w, aa) * strength, 0.0, 1.0)
        return ring, q

    def _rivets(self, x, theta, rc, ring, q, radius, aa, scale, out):
        p = self.params
        if p["rivet_darken"] <= 0.0 or scale <= 0.0:
            return
        amp = p["rivet_darken"] * scale
        pitch = max(p["rivet_spacing_m"], 1e-3)
        offset = p["rivet_offset_m"]
        seed = _seed32(self.seed + self._salts["rivet"])

        # Rows flanking each circumferential ring: dots march along the arc.
        arc_units = theta * rc / pitch
        cell = np.floor(arc_units)
        jitter = (_rand(cell.astype(np.int64), np.rint(x / self.pitch).astype(np.int64),
                        np.zeros_like(cell), seed) - 0.5) * 0.45 * pitch
        d_arc = theta * rc - (cell + 0.5) * pitch - jitter
        base_x = np.rint(x / self.pitch) * self.pitch
        a = x - (base_x - offset)
        b = x - (base_x + offset)
        near = np.minimum(np.sqrt(a * a + d_arc * d_arc),
                          np.sqrt(b * b + d_arc * d_arc))
        cov = np.clip((radius - near) / max(aa, 1e-6), 0.0, 1.0)
        out += amp * cov * cov * (3.0 - 2.0 * cov)

        # Rows flanking each longitudinal line: dots march along x.
        ux = x / pitch
        cell = np.floor(ux)
        jitter = (_rand(cell.astype(np.int64), np.rint(q).astype(np.int64),
                        np.full(cell.shape, 7, dtype=np.int64), seed + 1)
                  - 0.5) * 0.45 * pitch
        d_x = x - (cell + 0.5) * pitch - jitter
        lat = (q - np.rint(q)) * self.dtheta * rc
        a = lat - offset
        b = lat + offset
        near = np.minimum(np.sqrt(a * a + d_x * d_x), np.sqrt(b * b + d_x * d_x))
        cov = np.clip((radius - near) / max(aa, 1e-6), 0.0, 1.0)
        out += amp * cov * cov * (3.0 - 2.0 * cov)

    def _grime(self, x, y, z, theta, rc, q, out):
        p = self.params
        inv_len = 1.0 / max(p["grime_streak_len_m"], 1e-3)
        inv_wid = 1.0 / max(p["grime_width_m"], 1e-3)
        seed = _seed32(self.seed + self._salts["grime"])

        # Airflow streaks: noise sampled 10x slower along x than across it, so a
        # feature is a streak and not a cloud.
        n_flow = _fbm3(x * inv_len, y * inv_wid, z * inv_wid, seed, octaves=2)
        s_flow = _smoothstep(_GRIME_FLOW_BAND[0], _GRIME_FLOW_BAND[1], n_flow)
        # Stains collect downstream of a joint.  -x is downstream, so the joint
        # feeding this point is the next one at larger x.
        u = x / self.pitch
        upstream = np.ceil(u) * self.pitch
        run = np.exp(-np.clip(upstream - x, 0.0, None) / max(p["grime_run_m"], 1e-3))
        leak = 0.35 + 0.65 * _rand(np.ceil(u).astype(np.int64),
                                   np.zeros_like(u, dtype=np.int64),
                                   np.zeros_like(u, dtype=np.int64), seed + 11)
        gate_flow = run * leak

        # Gravity run-off.  Water leaves the airframe downwards, and on a mostly
        # vertical surface "down" is the circumferential direction, so this noise
        # is stretched along the arc (theta*R) and not along world z: sampling z
        # directly would make the streaks vanish on the fuselage sides where z is
        # nearly constant.
        n_grav = _fbm3(theta * rc * inv_len * 2.2, x * inv_wid, y * inv_wid,
                       seed + 101, octaves=1)
        s_grav = _smoothstep(_GRIME_GRAV_BAND[0], _GRIME_GRAV_BAND[1], n_grav)
        lat = (q - np.rint(q)) * self.dtheta * rc
        gate_grav = 0.45 + 0.55 * np.exp(-(lat / (0.7 * p["grime_width_m"])) ** 2)

        gravity = float(np.clip(p["grime_gravity"], 0.0, 1.0))
        flow_w = 1.0 - gravity
        out += p["grime_strength"] * (flow_w * s_flow * gate_flow
                                      + gravity * s_grav * gate_grav)

    def _panels(self, x, theta, out):
        p = self.params
        if p["panel_var"] <= 0.0:
            return
        cell_x = np.floor(x / self.pitch)
        cell_t = np.floor(theta / self.dtheta)
        h = _rand(cell_x.astype(np.int64), cell_t.astype(np.int64),
                  np.zeros_like(cell_x, dtype=np.int64),
                  _seed32(self.seed + self._salts["panel"]))
        out += (h - 0.5) * 2.0 * p["panel_var"]

    def _mottle(self, x, y, z, out):
        p = self.params
        if p["mottle"] <= 0.0:
            return
        n = _fbm3(x * (1.0 / 2.2), y * (1.0 / 2.2), z * (1.0 / 2.2),
                  _seed32(self.seed + self._salts["mottle"]), octaves=1)
        out += (n - 0.5) * 2.0 * p["mottle"]

    # -- evaluation ----------------------------------------------------------

    def _expected_darkening(self, w, rivet_radius, rivet_scale):
        p = self.params
        dark = 0.0
        if self.layers["seams"]:
            cov_t = min(1.0, w / self.pitch)
            cov_l = min(1.0, w / max(self.arc, 1e-6))
            dark += p["seam_depth"] * (cov_t + cov_l) * _SEAM_MEAN_FRACTION
        if self.layers["rivets"]:
            pitch = max(p["rivet_spacing_m"], 1e-3)
            area = (2.0 / max(self.pitch * pitch, 1e-9)
                    + 2.0 / max(self.arc * pitch, 1e-9))
            dark += p["rivet_darken"] * rivet_scale * min(
                0.35, area * np.pi * rivet_radius ** 2)
        if self.layers["grime"]:
            dark += self._grime_mean()
        return float(dark)

    def _grime_mean(self):
        """Analytic mean of the grime layer.

        The two masks are products of independent-ish factors, so their means
        multiply: a streak mask mean times the mean of its gate.  The downstream
        gate is exponential in a uniformly distributed distance (mean
        run/pitch*(1-exp(-pitch/run))); the gravity gate is a Gaussian of an arc
        offset distributed uniformly over one longitudinal bay.  Neither depends
        on how the caller samples the surface, which is what keeps the mean
        compensation stable at any resolution.
        """
        p = self.params
        strength = p["grime_strength"]
        if strength <= 0.0:
            return 0.0
        gravity = float(np.clip(p["grime_gravity"], 0.0, 1.0))
        pitch = max(self.pitch, 1e-6)
        run = max(p["grime_run_m"], 1e-6)
        e_run = (run / pitch) * (1.0 - math.exp(-pitch / run))
        sigma = max(0.7 * p["grime_width_m"], 1e-6)
        half_bay = max(0.5 * self.arc, 1e-6)
        e_gate = (sigma / (2.0 * half_bay)) * math.sqrt(math.pi) \
            * math.erf(half_bay / sigma)
        gate_grav = 0.45 + 0.55 * e_gate
        flow = (1.0 - gravity) * _GRIME_FLOW_MEAN * e_run * 0.675
        grav = gravity * _GRIME_GRAV_MEAN * gate_grav
        return float(strength * (flow + grav))

    def modulation_world(self, points, inv_rotation=None, origin=None,
                         pixel_scale_m=None):
        """Evaluate on world coordinates for an object with the given pose.

        inv_rotation is the 3x3 world-to-local rotation and origin the object's
        world translation.  The texture is a function of the local coordinate
        only, so moving the object moves its skin with it.
        """
        pts = _as_points(points)
        local = pts if origin is None else pts - np.asarray(origin, dtype=np.float64)
        if inv_rotation is not None:
            local = local @ np.asarray(inv_rotation, dtype=np.float64).T
        return self.modulation(local, pixel_scale_m=pixel_scale_m)

    def modulation(self, points, pixel_scale_m=None):
        """Albedo multiplier for each point of an (N, 3) object-local array."""
        pts = _as_points(points)
        n = pts.shape[0]
        if n == 0:
            return np.ones(0, dtype=np.float64)
        if not self.enabled:
            return np.ones(n, dtype=np.float64)
        if n > _CHUNK:
            # The layer stack is element-wise, so splitting a million points into
            # cache-sized blocks halves its wall time and caps peak memory.  The
            # result is bit-identical to the single call.
            return np.concatenate([self._modulation(pts[i:i + _CHUNK],
                                                    pixel_scale_m)
                                   for i in range(0, n, _CHUNK)])
        return self._modulation(pts, pixel_scale_m)

    def _modulation(self, pts, pixel_scale_m=None):
        n = pts.shape[0]
        finite = np.isfinite(pts).all(axis=1)
        if not finite.all():
            pts = pts.copy()
            pts[~finite] = 0.0
        x = pts[:, 0]
        y = pts[:, 1]
        z = pts[:, 2]

        rc = np.hypot(y, z)
        np.clip(rc, _RADIUS_CLAMP[0] * self.radius,
                _RADIUS_CLAMP[1] * self.radius, out=rc)
        theta = np.arctan2(z, y)

        widths = self._widths(pixel_scale_m)
        w = widths["seam_width"]
        aa = widths["aa_seam"]

        dark = np.zeros(n, dtype=np.float64)
        ring = q = None
        if self.layers["seams"]:
            ring, q = self._seams(x, theta, rc, w, aa, dark)
        if self.layers["rivets"]:
            if ring is None:
                ring = np.rint(x / self.pitch)
                q = theta / self.dtheta
            self._rivets(x, theta, rc, ring, q, widths["rivet_radius"],
                         widths["aa_rivet"], widths["rivet_scale"], dark)
        if self.layers["grime"]:
            if q is None:
                q = theta / self.dtheta
            self._grime(x, y, z, theta, rc, q, dark)

        mod = 1.0 - dark
        if self.layers["panels"]:
            self._panels(x, theta, mod)
        if self.layers["mottle"]:
            self._mottle(x, y, z, mod)
        if self.normalize_mean:
            expected = self._expected_darkening(w, widths["rivet_radius"],
                                                widths["rivet_scale"])
            if expected > 1e-6:
                mod *= 1.0 / max(1.0 - expected, 0.35)
        np.clip(mod, 0.45, 1.35, out=mod)
        if not finite.all():
            mod[~finite] = 1.0
        return mod

    __call__ = modulation

    # -- surfaces and statistics --------------------------------------------

    def patch(self, width_m, height_m, shape=(512, 512), pixel_scale_m=None):
        """Modulation over an unrolled strip of the body, as a 2-D array.

        Rows run along x (length), columns around the axis over width_m of arc
        (clipped to the full circumference).  Useful both for previewing and for
        baking a UV texture.
        """
        if np.isscalar(shape):
            rows = cols = int(shape)
        else:
            rows, cols = int(shape[0]), int(shape[1])
        arc = min(float(width_m), 2.0 * np.pi * self.radius)
        span = arc / self.radius
        th = np.linspace(-0.5 * span, 0.5 * span, cols)
        xs = np.linspace(-0.5 * float(height_m), 0.5 * float(height_m), rows)
        grid_t, grid_x = np.meshgrid(th, xs)
        r = self.radius
        pts = np.stack([grid_x.ravel(),
                        r * np.sin(grid_t.ravel()),
                        r * np.cos(grid_t.ravel())], axis=1)
        return self.modulation(pts, pixel_scale_m=pixel_scale_m).reshape(rows, cols)

    def stats(self, points, pixel_scale_m=None):
        m = self.modulation(points, pixel_scale_m=pixel_scale_m)
        if m.size == 0:
            return dict(n=0)
        return dict(n=int(m.size), mean=float(m.mean()), std=float(m.std()),
                    min=float(m.min()), max=float(m.max()),
                    p01=float(np.percentile(m, 1)),
                    p99=float(np.percentile(m, 99)))

    def describe(self, pixel_scale_m=None):
        widths = self._widths(pixel_scale_m)
        p = self.params
        px = widths["px"]
        return dict(category=self.category, seed=self.seed,
                    length_m=round(self.length, 3),
                    body_radius_m=round(self.radius, 3),
                    panel_pitch_m=round(self.pitch, 3),
                    ring_count=self.ring_count,
                    arc_spacing_m=round(self.arc, 3),
                    seam_width_m=round(widths["seam_width"], 4),
                    seam_width_px=(None if not px
                                   else round(widths["seam_width"] / px, 2)),
                    rivet_radius_m=round(widths["rivet_radius"], 4),
                    rivet_scale=round(widths["rivet_scale"], 3),
                    seam_depth=p["seam_depth"], rivet_darken=p["rivet_darken"],
                    grime_strength=p["grime_strength"],
                    panel_var=p["panel_var"], mottle=p["mottle"],
                    layers=dict(self.layers))

    def __repr__(self):
        return ("SkinField(%r, seed=%d, length=%.2fm, pitch=%.2fm, rings=%d)"
                % (self.category, self.seed, self.length, self.pitch,
                   self.ring_count))


def modulation(points, category="aircraft", size=None, seed=0, **kwargs):
    """One-shot convenience wrapper around SkinField."""
    return SkinField(category, size=size, seed=seed, **kwargs).modulation(points)


# ---------------------------------------------------------------------------
# Preview / diagnostics (used by tests and by the integration report)
# ---------------------------------------------------------------------------

def _image_stats(image):
    flat = np.asarray(image, dtype=np.float64).ravel()
    return dict(n=int(flat.size), mean=float(flat.mean()), std=float(flat.std()),
                min=float(flat.min()), max=float(flat.max()))


def preview_patch(category="aircraft", size=None, width_m=3.0, height_m=3.0,
                  shape=(512, 512), seed=0, pixel_scale_m=None, **kwargs):
    """Unrolled 2-D modulation strip for one category."""
    field = SkinField(category, size=size, seed=seed,
                      pixel_scale_m=pixel_scale_m, **kwargs)
    return field.patch(width_m, height_m, shape=shape, pixel_scale_m=pixel_scale_m)


def write_preview(path, categories=("aircraft", "ship", "vehicle"), seed=0,
                  width_m=3.0, height_m=3.0, shape=(420, 640), size=None,
                  pixel_scale_m=None, lo=0.72, hi=1.06, **kwargs):
    """Write a multi-row PNG of the modulation, one row per category.

    The grey mapping is fixed (lo -> 0, hi -> 255) so rows can be compared against
    each other and against 1.0; a per-row metric stretch would hide the absolute
    darkening that this module is supposed to keep small.
    """
    from simkit import png as _png

    rows = []
    report = {}
    for category in categories:
        field = SkinField(category, size=size, seed=seed,
                          pixel_scale_m=pixel_scale_m, **kwargs)
        m = field.patch(width_m, height_m, shape=shape, pixel_scale_m=pixel_scale_m)
        v = np.clip((m - lo) / max(hi - lo, 1e-9), 0.0, 1.0) * 255.0
        rows.append(v.astype(np.uint8))
        entry = _image_stats(m)
        entry["hf_ratio"] = hf_ratio(m)
        entry["describe"] = field.describe(pixel_scale_m)
        report[category] = entry
    width = rows[0].shape[1]
    blocks = []
    for i, row in enumerate(rows):
        if i:
            blocks.append(np.full((4, width), 40, dtype=np.uint8))
        blocks.append(row)
    image = np.concatenate(blocks, axis=0)
    _png.write_gray8(str(path), image)
    return report