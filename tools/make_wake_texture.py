"""Procedural Kelvin wake texture generator for the seav2 sea datasets.

Physics (deep water, sub-critical speed -- the regime every surface combatant
cruises in):

* the wake lives inside a wedge whose half angle is arcsin(1/3) = 19.47 deg,
  independent of speed (Kelvin 1887);
* the longest waves are the transverse ones along the track, wavelength
  Lambda0 = 2 pi V^2 / g -- 68 m at 20 kn, 106 m at 25 kn;
* each transverse crest is an arc of a circle centred on the sailing line at
  TWICE the arc's own distance from the source (Wikipedia, Kelvin wake pattern),
  which gives a closed-form crest phase field: for a point (d, a) astern,
  3c^2 - 4dc + (d^2 + a^2) = 0 with c the crest radius;
* the divergent waves ride near the two arms at about 35 deg to the course;
* a turbulent centreline streak (propeller wash + hull drag) persists for
  minutes and widens like x^0.25 (Tunaley & Buller);
* everything fades with distance -- viscous decay goes as 1/V^5.

Texture layout: ship heading along +u (array columns), the stern sits at
u = 0.84 * width, the wake trails toward u = 0, the bow V just ahead of the
stern.  Alpha carries the foam; RGB is near-white spray.  The seav2 builder
composites this onto the backdrop photograph in the ship frame, so the optical
(Cycles) and IR (plate sampler) bands show the same wake.
"""
import math
import os

import numpy as np
from PIL import Image

G = 9.81
KELVIN_DEG = 19.471220          # arcsin(1/3)
KELVIN_TAN = math.tan(math.radians(KELVIN_DEG))


def wake_params(speed_mps, ship_len_m):
    """Derived wavelength / Froude number from the two free parameters."""
    lambda0 = 2.0 * math.pi * speed_mps ** 2 / G      # transverse wavelength
    froude = speed_mps / math.sqrt(G * ship_len_m)
    return dict(lambda0=lambda0, froude=froude,
                cusp_lambda=2.0 * lambda0 / 3.0)


def wake_alpha(wake_len_m, speed_mps, ship_len_m,
               px_per_m=1.0, seed=7, aspect=1.35):
    """RGBA uint8 array.  Ground footprint:

    u (columns): wake_len_m astern + 2 ship lengths ahead (bow V + margin)
    v (rows):    +/- wake_len_m * tan(Kelvin) * aspect margin
    """
    p = wake_params(speed_mps, ship_len_m)
    lam = max(p["lambda0"], 4.0)
    half_w = wake_len_m * KELVIN_TAN * aspect
    w_px = int(round((wake_len_m + 3.0 * ship_len_m) * px_per_m))
    h_px = int(round(2.0 * half_w * px_per_m))
    rng = np.random.default_rng(seed)

    u_stern = wake_len_m + ship_len_m          # stern position, METRES
    # columns: u=0 at the stern, wake trails toward column 0, bow V ahead.
    # NOTE u_stern is metres; converting to pixels happens below (a metre value
    # used directly as a column index once clipped the wake to 1/px_per_m of
    # its length and left 300 m of empty canvas ahead of the ship).
    u = np.arange(w_px)[None, :] / px_per_m - u_stern      # m, + = ahead
    v = (np.arange(h_px)[:, None] - h_px / 2.0) / px_per_m  # m, cross track

    d = -u                                     # distance astern (>=0 behind)
    ahead = np.clip(-d, 0.0, None)
    astern = np.clip(d, 0.0, None)

    alpha = np.zeros((h_px, w_px), np.float32)

    # ---- turbulent centreline streak (propeller wash) -----------------------
    # Widens like x^0.25 (Tunaley & Buller); at 300 m astern a 28 m hull's
    # streak is ~20 m across, not the 6 m a pure hull-width gaussian gives.
    w_half = 0.18 * ship_len_m + 1.00 * np.power(
        astern + 1.0, 0.25) * math.sqrt(ship_len_m)
    core = np.exp(-(v / np.maximum(w_half, 0.4)) ** 2)
    turb = core * np.exp(-astern / (6.0 * ship_len_m)) * (astern > 0.3)
    turb += core * np.exp(-astern / (3.0 * ship_len_m)) * 0.5 * (astern > 0.3)
    alpha += 0.55 * turb

    # ---- divergent waves riding the two arms --------------------------------
    # The arms are the DOMINANT feature of an aerial wake photo: two CONTINUOUS
    # bright rays at +/-19.5 deg whose brightness is modulated by the wave
    # train.  Gating the ray by the ripple (max(sin,0)) broke it into beads;
    # modulating keeps the line while the crests pulse.
    arm_v = d * KELVIN_TAN
    for s in (-1.0, 1.0):
        arm = np.exp(-((v - s * arm_v) / (0.12 * lam + 2.0)) ** 2)
        rip = np.maximum(
            np.sin(2.0 * math.pi * d / max(lam * math.cos(math.radians(35.0)),
                                           1e-6) + 1.1 * s * v / lam), 0.0)
        alpha += 0.85 * arm * (0.45 + 0.55 * rip) \
            * np.exp(-d / (9.0 * lam)) * (d > 1.0)

    # ---- Kelvin wedge: transverse crest arcs --------------------------------
    disc = np.clip(d * d - 3.0 * v * v, 0.0, None)
    c = (2.0 * d - np.sqrt(disc)) / 3.0            # crest radius (inner root)
    inside = (np.abs(v) <= d * KELVIN_TAN) & (d > 0.5)
    phase = 2.0 * math.pi * c / lam
    # Sharpen the positive half-cycle (**4) so each crest reads as a thin bright
    # arc with dark water between, not a scalloped slab filling the wedge.
    crest = np.maximum(np.sin(phase), 0.0) ** 4
    trans = np.where(inside, crest, 0.0)
    trans *= np.exp(-d / (7.0 * lam))              # radial fade of the pattern
    alpha += 0.55 * trans

    # ---- bow wave: a short bright V ahead of the stern ----------------------
    bow_d = ahead - 0.45 * ship_len_m
    bow_arm = np.abs(v) - np.clip(bow_d, 0.0, None) * KELVIN_TAN
    bow = np.exp(-(bow_arm / (0.16 * ship_len_m + 0.4)) ** 2) \
        * np.exp(-np.clip(bow_d, 0.0, None) / (1.6 * ship_len_m)) \
        * ((bow_d > -0.2 * ship_len_m) & (np.abs(v) < 1.1 * ship_len_m))
    alpha += 0.60 * bow

    # ---- spray granularity + overall fade -----------------------------------
    # NOTE: no alpha floor here. A `max(fade, 0.05)` floor gave every sprite
    # pixel a constant 5 % white veil, which the seav2 baker composited over
    # the whole 400 m bounding box -- a huge uniform slab across the photograph
    # instead of a wake.  Real foam dies with distance; so must the alpha.
    alpha *= (0.80 + 0.40 * rng.random(alpha.shape).astype(np.float32))
    alpha *= np.exp(-astern / (15.0 * lam))
    alpha = np.clip(alpha, 0.0, 0.90) * 255.0

    rgb = np.full((h_px, w_px, 3), 236, np.uint8)
    return np.dstack([rgb, alpha.astype(np.uint8)]), dict(
        u_stern_px=u_stern * px_per_m, w_px=w_px, h_px=h_px, lam=lam,
        froude=p["froude"], half_w_m=half_w)


def save_wake(path, wake_len_m, speed_mps, ship_len_m,
              px_per_m=1.0, seed=7, aspect=1.35):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    rgba, info = wake_alpha(wake_len_m, speed_mps, ship_len_m,
                            px_per_m=px_per_m, seed=seed, aspect=aspect)
    Image.fromarray(rgba, "RGBA").save(path)
    return path, info


if __name__ == "__main__":
    out, info = save_wake("assets/wake/kelvin_wake_preview.png",
                          wake_len_m=420.0, speed_mps=10.3, ship_len_m=32.0,
                          px_per_m=2.5)
    print("saved", out, info)
