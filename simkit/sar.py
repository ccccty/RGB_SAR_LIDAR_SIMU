"""Stripmap SAR forward model: ground-plane ray casting -> PO reflectivity ->
range-Doppler focusing -> speckle -> calibrated gamma0 image.

Why not one ray per pulse
-------------------------
A real system transmits n_pulses x n_samples samples and focuses them.  That is
O(1e8) geometric operations for a 1k x 1k image and takes minutes per frame in
pure numpy.  This module uses the range-Doppler equivalence instead: for a
broadside stripmap geometry the azimuth phase history of a scatterer at
azimuth offset u is

    h(u) = exp(-j 4 pi sqrt(R0^2 + u^2) / lambda)

which is *shift invariant* in u for a fixed ground range.  A matched filter over
h therefore collapses to a convolution of the scene reflectivity with the
compressed point-spread function k (x) k*, i.e. a sinc of width

    rho_a = 0.886 * lambda * R / (2 * L_sa)

so the whole focusing stage is one separable convolution with an analytic PSF
whose width is derived from the same physics as a pulse-by-pulse processor.
The deterministic propagation phase exp(-j 4 pi R / lambda) is *not* convolved
back in: a matched filter cancels it by construction, so the convolution input
is the real amplitude sqrt(gamma0) and the surviving per-pixel phase is the
random one of the speckle field (see the note above the focusing block).

Approximations are deliberate and listed in docs/SAR成像.md:
  * no range cell migration / no azimuth ambiguity (PRF assumed adequate),
  * ground-plane backprojection (one ray per pixel from the aperture centre),
  * single bounce, flat-facet physical optics, no shadowing by the atmosphere,
  * noise added in the image domain rather than per pulse,
  * terrain height enters through the slant-range binning, so elevated targets
    lay over towards near range, but there is no per-pulse migration track.
"""
import numpy as np

from . import materials as matlib
from .radar import C_LIGHT, K_BOLTZ, T0, facet_sigma
from .weather import radar_band_of

# band -> default carrier, only used when fc_ghz is not given explicitly
BAND_FC = {"X": 9.6, "C": 5.4, "Ku": 15.0, "L": 1.3, "S": 3.0}

# 0.8859: full width at -3 dB of sin(pi x)/(pi x), i.e. rho == c/2B is a -3 dB
# width, which is the definition a pulse-compression datasheet quotes.
SINC_3DB = 0.8859

MAT_NAMES = sorted(matlib.MATERIALS)
RHO_TAB = np.array([matlib.MATERIALS[n]["rf_rho"] for n in MAT_NAMES])
ROUGH_TAB = np.array([matlib.MATERIALS[n]["rough_m"] for n in MAT_NAMES])

DEFAULTS = dict(
    band="X", fc_ghz=9.6, bandwidth_mhz=150.0, prf_hz=1200.0,
    n_pulses=512, n_samples=1024,
    altitude_m=2000.0, speed_mps=120.0, heading_deg=90.0, squint_deg=90.0,
    swath_center_m=2500.0, swath_width_m=560.0, polarization="VV",
    tx_power_w=40.0, noise_figure_db=4.0, loss_db=3.0,
    antenna_len_m=1.2, antenna_h_m=0.35, resolution_scale=1.0,
    # --- extensions beyond the contract keys (all optional in a scene) ---
    n_cross=512, az_extent_m=0.0, oversample=2.0, side="right", looks=1.0,
    duty=0.08,
    antenna_eff=0.6, specular_scale_db=-20.0, diffuse_gamma=0.06,
    ground_clutter_db=0.0, display_dyn_range_db=26.0, display_anchor_pct=10.0,
    display_gamma=0.8, seed=0, anchor=None,
    # per-frame pose (dataset.py rewrites these from the clip plan; a standalone
    # run without them falls back to the geometric construction below)
    pos=None, look_at=None,
    # deterministic mode: turn the stochastic terms off to inspect the impulse
    # response (calibration / PSF measurement). Images always run with both on.
    speckle=True, thermal_noise=True,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _merge(cfg):
    rc = dict(DEFAULTS)
    user = (cfg.get("sensors") or {}).get("sar") or {}
    band = user.get("band")
    if band in BAND_FC and "fc_ghz" not in user:
        rc["fc_ghz"] = BAND_FC[band]
    rc.update(user)
    rc["band"] = band or rc["band"]
    for k, v in DEFAULTS.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            rc[k] = float(rc[k])
    for k in ("n_pulses", "n_samples", "n_cross"):
        rc[k] = max(2, int(round(rc[k])))
    return rc


def _conv_axis(a, ker, axis):
    """Odd-length 1-D convolution along one axis, reflect-padded ends."""
    n = len(ker)
    c = n // 2
    pad = [(0, 0), (0, 0)]
    pad[axis] = (c, c)
    ap = np.pad(a, pad, mode="reflect")
    out = np.zeros_like(a)
    for k in range(n):
        sl = [slice(None), slice(None)]
        sl[axis] = slice(k, k + a.shape[axis])
        out += ker[k] * ap[tuple(sl)]
    return out


def _psf_kernel(width_m, dx, lobes=3.0):
    """Sinc PSF whose full -3 dB width is exactly width_m (unweighted RD)."""
    w = max(float(width_m), 0.25 * dx)
    half = max(1, int(np.ceil(lobes * w / dx)))
    x = np.arange(-half, half + 1, dtype=float) * dx
    k = np.sinc(SINC_3DB * x / w)
    s = k.sum()
    return k / (s if abs(s) > 1e-12 else 1.0)


def _face_normals(w):
    t = w.verts[w.faces]
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    ln[ln == 0] = 1.0
    return n / ln


def _face_areas(w):
    t = w.verts[w.faces]
    return 0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)


def _scene_center(world, cfg):
    """Ground point the swath is aimed at.

    Taken from the *config* object positions, not from the world's current
    pose: anchoring on the instantaneous target would make the platform chase
    the target and turn a stripmap into a spotlight. A stripmap flies a straight
    line and the ground patch (and any moving target) scrolls through the image,
    which is the geometry the frame-to-frame drift of the dataset should show.
    """
    pts = [np.asarray(o.get("pos", [0, 0, 0]), float) for o in cfg.get("objects", [])
           if o.get("label", True)]
    if not pts:
        pts = [np.asarray(o["corners"], float).mean(axis=0) for o in world.objs]
    if not pts:
        return np.zeros(3)
    c = np.mean(pts, axis=0)
    c[2] = 0.0
    return c


def _heading_basis(heading_deg, squint_deg, side="right"):
    """Along-track unit vector and horizontal look unit vector."""
    hd = np.array([np.cos(np.radians(heading_deg)), np.sin(np.radians(heading_deg)), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(hd, up)
    right /= np.linalg.norm(right) + 1e-12
    lat = right if str(side).lower() != "left" else -right
    d = np.radians(float(squint_deg) - 90.0)      # 0 => broadside
    lh = np.cos(d) * lat + np.sin(d) * hd
    lh[2] = 0.0
    return hd, lh / (np.linalg.norm(lh) + 1e-12)


def _platform_basis(rc):
    """Along-track and look unit vectors, plus the effective side and squint.

    A clip rewrites sensors.sar.pos / look_at with the instantaneous pose of the
    platform (dataset.py does this every frame so the swath follows the moving
    sensor). When both are present the look direction is taken straight from
    pos -> look_at and the side / squint angle are *derived* from it, so the
    imaged patch tracks the platform and cross-modal registration holds.
    heading_deg still owns the along-track axis. Without a pose the geometry is
    built from heading_deg / squint_deg / side, which is what a standalone tool
    run uses.
    """
    heading = float(rc["heading_deg"])
    hd = np.array([np.cos(np.radians(heading)), np.sin(np.radians(heading)), 0.0])
    right = np.cross(hd, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right) + 1e-12
    side = str(rc.get("side") or "right").lower()
    squint = float(rc["squint_deg"])
    p, la = rc.get("pos"), rc.get("look_at")
    if p is not None and la is not None:
        rel = np.asarray(la, float).reshape(3) - np.asarray(p, float).reshape(3)
        rel[2] = 0.0
        n = float(np.linalg.norm(rel))
        if n > 1e-9:
            lh = rel / n
            side = "right" if float(lh @ right) >= 0.0 else "left"
            lat = right if side == "right" else -right
            squint = 90.0 + np.degrees(np.arctan2(float(lh @ hd), float(lh @ lat)))
            return hd, lh, side, float(squint)
    hd, lh = _heading_basis(heading, squint, side)
    return hd, lh, side, float(squint)


def resolution_of(rc):
    """Theoretical (ground-projected) resolution for a merged config."""
    lam = C_LIGHT / (float(rc["fc_ghz"]) * 1e9)
    b = float(rc["bandwidth_mhz"]) * 1e6
    h = float(rc["altitude_m"])
    g = float(rc["swath_center_m"])
    r0 = np.hypot(h, g)
    sin_inc = g / r0                                   # sin of the incidence angle
    rho_r_slant = C_LIGHT / (2.0 * b) * float(rc["resolution_scale"])
    pri = 1.0 / float(rc["prf_hz"])
    t_int = float(rc["n_pulses"]) * pri
    l_flight = float(rc["speed_mps"]) * t_int
    l_beam = lam * r0 / max(float(rc["antenna_len_m"]), 1e-3)
    l_sa = min(l_flight, l_beam)
    # Azimuth resolution from the two-way phase history itself.  phi(u) =
    # -4 pi sqrt(R0^2+u^2)/lambda, so |d2phi/du2| = 4 pi/(lambda R0) rad/m^2, the
    # Doppler rate is K_a = k*v^2/(2 pi) Hz/s, the Doppler bandwidth over the
    # aperture is B_d = K_a*T_int = 2 v L_sa/(lambda R0), and a matched filter of
    # that bandwidth compresses to 0.886*v/B_d = 0.886*lambda*R0/(2*L_sa).
    v = max(float(rc["speed_mps"]), 1e-6)
    k_phi = 4.0 * np.pi / (lam * r0)
    k_a = k_phi * v * v / (2.0 * np.pi)
    t_eff = l_sa / v                      # aperture actually coherently integrated
    b_dop = max(k_a * t_eff, 1e-9)
    rho_a = SINC_3DB * v / b_dop * float(rc["resolution_scale"])
    return dict(range_m=rho_r_slant / max(sin_inc, 1e-6), range_slant_m=rho_r_slant,
                azimuth_m=rho_a, l_sa_m=l_sa, l_flight_m=l_flight, l_beam_m=l_beam,
                sin_inc=sin_inc, r0_m=r0, lam_m=lam, t_int_s=t_int, b_dop_hz=b_dop,
                doppler_rate_hz_s=k_a)


def _system(rc, lam):
    """Radar constants: gain, signal scale, noise and NESZ."""
    b = float(rc["bandwidth_mhz"]) * 1e6
    area = max(float(rc["antenna_len_m"]), 1e-3) * max(float(rc["antenna_h_m"]), 1e-3)
    gain = 4.0 * np.pi * area * float(rc["antenna_eff"]) / lam ** 2
    loss = 10.0 ** (float(rc["loss_db"]) / 10.0)
    nf = 10.0 ** (float(rc["noise_figure_db"]) / 10.0)
    pri = 1.0 / float(rc["prf_hz"])
    tp = max(1e-9, float(rc["duty"]) * pri)
    pt = float(rc["tx_power_w"])
    # Pr = ksys * sigma / R^4 ; single-pulse matched-filter SNR = tp * Pr / (k T0 F)
    ksys = pt * gain ** 2 * lam ** 2 / ((4.0 * np.pi) ** 3 * loss)
    ktf = K_BOLTZ * T0 * nf
    return dict(gain=gain, gain_db=10.0 * np.log10(max(gain, 1e-30)), ksys=ksys,
                ktf=ktf, pn=ktf * b, tp_s=tp, n_int=float(rc["n_pulses"]))


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------
def simulate(world, cfg, rng=None):
    rc = _merge(cfg)
    lam = C_LIGHT / (float(rc["fc_ghz"]) * 1e9)
    sysd = _system(rc, lam)
    if rng is None:
        seed = int(cfg.get("_frame_seed", rc["seed"]))
        rng = np.random.default_rng(seed * 7919 + 13)

    # ---- platform -------------------------------------------------------
    hd, lh, side, squint = _platform_basis(rc)
    g_c = float(rc["swath_center_m"])
    half_w = 0.5 * float(rc["swath_width_m"])
    near_g, far_g = g_c - half_w, g_c + half_w
    n_range = int(rc["n_samples"])
    dt = float((cfg.get("frames") or {}).get("dt", 0.2))
    t = float(cfg.get("_frame_t", float(cfg.get("_frame_seed", 0)) * dt))
    vel = hd * float(rc["speed_mps"])

    center = (_scene_center(world, cfg) if rc.get("anchor") is None
              else np.array(rc["anchor"], float))
    ground_z = float(center[2])
    p_fix = rc.get("pos")
    if p_fix is not None:
        # A pose was supplied: it IS the aperture centre, already at this frame.
        # Adding vel*t again would double-count the motion the clip plan applied.
        pos = np.asarray(p_fix, float).reshape(3).copy()
        alt = float(pos[2] - ground_z)
    else:
        alt = float(rc["altitude_m"])
        c_xy = np.array([center[0], center[1], 0.0])
        pos = c_xy - g_c * lh
        pos[2] = ground_z + alt
        pos = pos + vel * t
    rc["altitude_m"] = alt
    rc["side"] = side
    rc["squint_deg"] = squint
    res = resolution_of(rc)

    # Columns are uniform in SLANT range, which is what a radar actually
    # measures: the range PSF is then uniform and an elevated target lands at
    # its own slant range, i.e. it shows the standard layover towards near
    # range. A ground-range grid would instead smear every target towards far
    # range (and would need a range-dependent PSF).
    r_near, r_far = float(np.hypot(alt, near_g)), float(np.hypot(alt, far_g))
    ranges_slant = np.linspace(r_near, r_far, n_range)
    dR = (r_far - r_near) / max(n_range - 1, 1)
    ranges = np.sqrt(np.maximum(ranges_slant ** 2 - alt ** 2, 0.0))   # ground, for aiming
    dg = (far_g - near_g) / max(n_range - 1, 1)                       # mean ground step

    # ---- resolutions, swath, azimuth axis --------------------------------
    rho_r_slant = float(res["range_slant_m"])
    rho_a = float(res["azimuth_m"])
    amp_os = max(1.0, float(rc["oversample"]))
    az_ext = float(rc["az_extent_m"]) or (int(rc["n_cross"]) * rho_a / amp_os)
    n_cross = int(rc["n_cross"])
    cross = np.linspace(-az_ext / 2.0, az_ext / 2.0, n_cross)
    du_px = az_ext / max(n_cross - 1, 1)

    # ---- one ray per pixel, from the aperture centre ---------------------
    U, G = np.meshgrid(cross, ranges, indexing="ij")
    q = pos[None, :] + U.reshape(-1, 1) * hd[None, :] + G.reshape(-1, 1) * lh[None, :]
    q[:, 2] = ground_z
    d = q - pos[None, :]
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-12
    hit = world.bvh.ray_cast(pos, d)

    tri, ok = hit["tri"], hit["hit"]
    # ground area of one cell, per column (a slant-uniform grid is non-uniform
    # in ground range, and gamma0 is RCS per m^2 of GROUND)
    dg_col = np.gradient(ranges) * du_px
    cell_area_col = np.maximum(dg_col, 1e-6)
    cell_area = du_px * dg
    area_flat = np.tile(cell_area_col, n_cross)

    areas = _face_areas(world) if not hasattr(world, "face_areas") else world.face_areas()
    normals = _face_normals(world)
    spec_scale = 10.0 ** (float(rc["specular_scale_db"]) / 10.0)
    diff_gamma = float(rc["diffuse_gamma"])
    ground_scale = 10.0 ** (float(rc["ground_clutter_db"]) / 10.0)

    # per-cell reflectivity (gamma0, i.e. RCS per m^2 of ground) and the
    # two-way propagation phase that a pulse-by-pulse processor would carry.
    gam = np.zeros(len(d))
    obj_of = np.full(len(d), -1, np.int64)
    pt_hit = np.zeros((len(d), 3))
    gam_coarse_obj = np.zeros(len(d))
    id_coarse_obj = np.full(len(d), -1, np.int64)
    idx = np.where(ok)[0]
    if len(idx):
        t_i = tri[idx]
        n = normals[t_i]
        di = d[idx]
        cos_i = np.abs((n * (-di)).sum(axis=1))
        rho_m = RHO_TAB[world.face_mat[t_i]]
        rgh = ROUGH_TAB[world.face_mat[t_i]]
        # the illuminated patch is the ground cell seen at this incidence
        ca = area_flat[idx]
        foot = ca * np.clip(cos_i, 0.0, 1.0)
        sig = facet_sigma(areas[t_i], cos_i, rho_m, rgh, lam, foot,
                          spec_scale, diff_gamma)
        g_i = sig / ca
        gnd = world.face_obj[t_i] < 0
        if gnd.any():
            g_i[gnd] *= ground_scale
        gam[idx] = np.maximum(g_i, 0.0)
        obj_of[idx] = world.face_obj[t_i]
        pt_hit[idx] = hit["point"][idx]
        # Object pixels are re-integrated by the quadrature pass below (which
        # bins by the hit point, so an elevated face lands half a cell away from
        # the ray's own pixel). Park the coarse estimate as a hole filler: with
        # it dropped, every cell where the two binnings disagree stayed black.
        objm = ~gnd
        if objm.any():
            hp = pt_hit[idx[objm]] - pos[None, :]
            ri_o = np.clip(np.rint(((hp @ hd) - cross[0]) / du_px).astype(int),
                           0, n_cross - 1)
            ci_o = np.clip(np.rint((np.linalg.norm(hp, axis=1) - r_near) / dR).astype(int),
                           0, n_range - 1)
            obj_flat = ri_o * n_range + ci_o
            gam_coarse_obj[obj_flat] = g_i[objm]
            id_coarse_obj[obj_flat] = world.face_obj[t_i][objm]
            gam[idx[objm]] = 0.0          # the pixel the ray aimed at is shadowed
            obj_of[idx[objm]] = -1

    rr = np.where(ok, hit["t"], np.inf)
    rr[~ok] = np.hypot(alt, G.reshape(-1))[~ok]

    # ---- sub-pixel object quadrature ------------------------------------
    # One ray per pixel samples the scene on a lattice: a 0.4 m plate at 1 m
    # pixels is hit or missed depending on sub-metre grid phase, so a target's
    # brightness would flicker from frame to frame. Re-sample every object on a
    # 4x finer lattice and integrate its RCS into the pixel that owns each hit
    # (gamma0 = sum(sigma) / cell area, which is energy conserving and makes a
    # point target's image independent of grid phase).
    if len(world.objs) and n_cross > 1:
        qstep = max(min(du_px, dg) / 4.0, 0.02)
        for o in world.objs:
            c3 = np.asarray(o["corners"], float)
            cs = c3 - pos[None, :]
            zmax = float(np.max(c3[:, 2])) - ground_z
            cs[:, 2] = 0.0
            us, gs = cs @ hd, cs @ lh
            # rays that reach a target h metres up are aimed h/tan(depression)
            # further out, so the sampling window has to grow towards far range
            dep = np.arctan2(alt, max(g_c, 1e-3))
            shift = zmax / max(np.tan(dep), 1e-6)
            u_lo, u_hi = max(us.min() - du_px, cross[0]), min(us.max() + du_px, cross[-1])
            g_lo, g_hi = (max(gs.min() - dg, near_g),
                          min(gs.max() + shift + dg, far_g))
            if u_hi <= u_lo or g_hi <= g_lo:
                continue                       # object outside the imaged area
            uu = np.arange(u_lo, u_hi + qstep, qstep)
            gg = np.arange(g_lo, g_hi + qstep, qstep)
            budget = 60000
            while uu.size * gg.size > budget:  # keep the cost bounded
                uu, gg = uu[::2], gg[::2]
            UU, GG = np.meshgrid(uu, gg, indexing="ij")
            qq = pos[None, :] + UU.reshape(-1, 1) * hd[None, :] + GG.reshape(-1, 1) * lh[None, :]
            qq[:, 2] = ground_z
            dd = qq - pos[None, :]
            dd /= np.linalg.norm(dd, axis=1, keepdims=True) + 1e-12
            ho = world.bvh.ray_cast(pos, dd)
            m = ho["hit"] & (world.face_obj[ho["tri"]] == o["id"])
            if not m.any():
                continue
            tq = ho["tri"][m]
            nq = normals[tq]
            cq = np.abs((nq * (-dd[m])).sum(axis=1))
            # The coherent (specular) term goes as a^2, so a per-ray footprint of
            # qstep^2 would make a smooth facet dimmer as the quadrature gets
            # finer -- a sampling artefact, not physics. Instead take the facet's
            # own RCS (coherent patch clipped to one resolution cell) and hand
            # each ray the share of the facet it actually samples. The per-cell
            # sum is then independent of qstep.
            ca_q = float(np.mean(cell_area_col))
            sq_f = facet_sigma(areas[tq], cq, RHO_TAB[world.face_mat[tq]],
                               ROUGH_TAB[world.face_mat[tq]], lam,
                               ca_q * np.clip(cq, 0.0, 1.0), spec_scale, diff_gamma)
            frac = np.minimum(qstep * qstep * np.clip(cq, 0.0, 1.0)
                              / np.maximum(areas[tq], 1e-9), 1.0)
            sq = sq_f * frac
            # bin by the hit point's own (u, R): that is what puts an elevated
            # scatterer at its true slant range instead of h/tan(dep) too far
            hp = ho["point"][m] - pos[None, :]
            u_hit = hp @ hd
            r_hit = np.linalg.norm(hp, axis=1)
            ri = np.clip(np.rint((u_hit - cross[0]) / du_px).astype(int), 0, n_cross - 1)
            ci = np.clip(np.rint((r_hit - r_near) / dR).astype(int), 0, n_range - 1)
            flat = ri * n_range + ci
            gam += np.bincount(flat, weights=sq / area_flat[flat], minlength=len(d))[:len(d)]
            obj_of[flat] = o["id"]
            rr[flat] = r_hit
            pt_hit[flat] = ho["point"][m]

    hole = (gam <= 0.0) & (id_coarse_obj >= 0)
    if hole.any():
        gam[hole] = gam_coarse_obj[hole]
        obj_of[hole] = id_coarse_obj[hole]

    att_db = 0.0
    w = cfg.get("_weather")
    if w:
        att = w.get("radar_att_db_km", {}).get(rc.get("band") or radar_band_of(float(rc["fc_ghz"])))
        if att:
            att_db = -float(att) * 2.0 * (rr / 1000.0)
    att = 10.0 ** (att_db / 10.0)

    # ---- focusing ------------------------------------------------------
    # The matched filter acts on the *phase history*, and a perfect azimuth
    # compressor cancels the quadratic exp(-j 4 pi R(u)/lambda) term.  That
    # cancellation is exactly why a focused image is the scene reflectivity
    # convolved with k(x)k*, so the convolution input is the real amplitude
    # sqrt(gamma0); the surviving per-pixel phase is the random one carried by
    # the speckle field below (the deterministic ramp is gone, which is what
    # "focused" means -- it is still what sets rho_a, see resolution_of()).
    # Convolving gamma*exp(-j4piR/lam) instead re-injects the ramp and paints a
    # spurious 50 m interference lattice over the ground: the round-trip phase
    # changes by ~400 rad per metre of range, so neighbouring cells add with
    # essentially random phase and modulate the image with a moire pattern.
    amp = np.sqrt(np.maximum(gam, 0.0)).astype(np.float32)

    # ---- focusing: separable matched filter ------------------------------
    shape = (n_cross, n_range)
    amp2 = amp.reshape(shape)
    k_az = _psf_kernel(rho_a, du_px)
    k_r = _psf_kernel(rho_r_slant, dR)
    amp2 = _conv_axis(_conv_axis(amp2, k_az, 0), k_r, 1)

    amp_objs = []
    ids = [o["id"] for o in world.objs]
    for oid in ids:
        a = np.zeros_like(amp2)
        m = obj_of == oid
        if m.any():
            a.reshape(-1)[m] = amp[m]
            a = _conv_axis(_conv_axis(a, k_az, 0), k_r, 1)
        amp_objs.append(a)

    # ---- speckle + thermal noise ---------------------------------------
    # Speckle is the interference of the many sub-resolution scatterers inside
    # one resolution cell, so it is correlated over exactly the PSF footprint:
    # the complex field is blurred by the same kernel as the image and only then
    # modulates the reflectivity envelope. Drawing independent per-pixel
    # exponentials instead gives white salt-and-pepper, which looks like sensor
    # noise rather than SAR and destroys the spatial statistics a detector
    # could learn. The field is renormalised to unit mean power so the dB
    # calibration (gamma0) stays exact.
    looks = max(1, int(round(float(rc["looks"]))))
    if bool(rc["speckle"]):
        acc = np.zeros(shape)
        for _ in range(looks):
            c = _conv_axis(_conv_axis(_cn(rng, shape), k_az, 0), k_r, 1)
            acc += np.abs(c) ** 2
        acc /= float(looks)
        pw = amp2 ** 2 * (acc / max(float(acc.mean()), 1e-12))
    else:
        pw = amp2 ** 2

    # NESZ per cell: the gamma0 that would give SNR = 1 after compression
    nesz_col = (rr ** 4) * sysd["ktf"] / (sysd["n_int"] * sysd["tp_s"] * sysd["ksys"]
                                          * area_flat * np.maximum(att, 1e-12))
    nesz_col = nesz_col.reshape(shape).mean(axis=0)
    noise = (nesz_col[None, :] * np.abs(_cn(rng, shape)) ** 2
             if bool(rc["thermal_noise"]) else np.zeros(shape))

    power = pw + noise
    image_db = (10.0 * np.log10(np.maximum(power, 1e-12))).astype(np.float32)

    # ---- object map: brightest PSF response per pixel -------------------
    # Per-object maps go through the *same* PSF, so a target bled into its
    # neighbour resolution cells is still labelled with the object that owns
    # the energy. The -12 dB cut keeps the sinc sidelobes from growing boxes.
    obj_map = np.full(shape, -1, np.int32)
    best = np.zeros(shape)
    bg_energy = np.abs(amp2 - sum(amp_objs)) ** 2 if amp_objs else np.abs(amp2) ** 2
    for oid, a in zip(ids, amp_objs):
        e = np.abs(a) ** 2
        emax = float(e.max()) if e.size else 0.0
        if emax <= 0.0:
            continue
        m = (e > best) & (e > bg_energy) & (e > 0.06 * emax)
        obj_map[m] = oid
        best[m] = e[m]

    # ---- geometric boxes ------------------------------------------------
    # A target smaller than one resolution cell owns only a couple of obj_map
    # pixels, and a 2-pixel mask is a fragile label. These boxes are the pure
    # projection of each labelled object's corners through the same (u, |R|)
    # geometry the quadrature bins with, so a box exists even when the mask does
    # not; the mask box is unioned in so neither can lose the target. Pixel
    # coordinates are [x0, y0, x1, y1] with x = column (range) and y = row
    # (azimuth), and x1/y1 EXCLUSIVE -- the same half-open convention as
    # labels.boxes_from_mask(), so the integration layer can drop them straight
    # into a box dict. Only label:true objects get a box: scenery is imaged and
    # appears in obj_map but is not an annotation.
    boxes = {}
    for o in world.objs:
        if not (o.get("label", True) and o.get("visible", True)):
            continue
        rel = np.asarray(o["corners"], float) - pos[None, :]
        uu = rel @ hd
        r_c = np.linalg.norm(rel, axis=1)
        r0 = int(np.clip(np.floor((uu.min() - cross[0]) / du_px), 0, n_cross - 1))
        r1 = int(np.clip(np.ceil((uu.max() - cross[0]) / du_px), 0, n_cross - 1))
        c0 = int(np.clip(np.floor((r_c.min() - r_near) / dR), 0, n_range - 1))
        c1 = int(np.clip(np.ceil((r_c.max() - r_near) / dR), 0, n_range - 1))
        m = obj_map == o["id"]
        if m.any():
            ys, xs = np.where(m)
            r0, r1 = min(r0, int(ys.min())), max(r1, int(ys.max()))
            c0, c1 = min(c0, int(xs.min())), max(c1, int(xs.max()))
        name = str(o.get("name") or ("obj%d" % o["id"]))
        if name in boxes:                     # duplicate names would collide
            name = "%s#%d" % (name, o["id"])
        boxes[name] = [int(c0), int(r0), min(int(c1) + 1, n_range),
                       min(int(r1) + 1, n_cross)]

    # ---- display stretch (same idea as dataset.scope) -------------------
    # Anchor on a low percentile instead of the median: in a focused SAR image
    # the median is the ground return, not the noise floor, and anchoring there
    # throws half the terrain texture below black. The 10th percentile sits on
    # the dark tail of the speckle, so the ground lands mid-grey and the fixed
    # window keeps every frame of a clip photometrically comparable.
    lo = float(np.percentile(image_db, float(rc["display_anchor_pct"])))
    dyn = float(rc["display_dyn_range_db"])
    v = np.clip((image_db - lo) / dyn, 0.0, 1.0) ** float(rc["display_gamma"])
    image_u8 = np.round(v * 255.0).astype(np.uint8)

    # ---- statistics -----------------------------------------------------
    peak_db = float(image_db.max())
    noise_db = float(10.0 * np.log10(max(float(np.median(nesz_col)), 1e-12)))
    tgt = obj_map >= 0
    bg = ~tgt
    if tgt.any():
        target_peak_db = float(image_db[tgt].max())
        tgt_mean = float(image_db[tgt].mean())
    else:
        target_peak_db = float("nan")
        tgt_mean = float("nan")
    bg_med = float(np.median(image_db[bg])) if bg.any() else float("nan")
    stats = dict(
        peak_db=peak_db,
        mean_db=float(image_db.mean()),
        noise_db=noise_db,
        snr_db=target_peak_db - noise_db if tgt.any() else float("nan"),
        dynamic_range_db=float(np.percentile(image_db, 99.0) - np.percentile(image_db, 1.0)),
        n_scatterers=int(len(idx)),
        target_peak_db=target_peak_db,
        # signal-to-clutter ratio, the standard definition: target peak over the
        # median clutter level. Using the box mean would instead measure how
        # much radar shadow happens to fall inside the box.
        target_contrast_db=(target_peak_db - bg_med) if tgt.any() else float("nan"),
        target_mean_db=tgt_mean,
        nesz_db=noise_db,
        gain_db=sysd["gain_db"],
        cell_area_m2=float(cell_area),
        looks=looks,
    )

    # ---- scatterer list --------------------------------------------------
    order = np.argsort(-gam)[:2000]
    scatterers = [dict(x=float(pt_hit[i, 0]), y=float(pt_hit[i, 1]), z=float(pt_hit[i, 2]),
                       sigma=float(gam[i] * cell_area), obj=int(obj_of[i]))
                  for i in order if gam[i] > 0.0]

    out_cfg = dict(rc)
    return dict(
        image_u8=image_u8,
        image_db=image_db,
        obj_map=obj_map,
        range_axis_m=ranges.astype(float),
        range_slant_axis_m=ranges_slant.astype(float),
        cross_axis_m=cross.astype(float),
        resolution=dict(range_m=float(res["range_m"]), azimuth_m=float(rho_a),
                        range_slant_m=float(rho_r_slant), l_sa_m=float(res["l_sa_m"])),
        swath=dict(near_m=float(near_g), far_m=float(far_g), width_m=float(far_g - near_g),
                   center_m=float(g_c), incidence_deg=float(np.degrees(np.arcsin(res["sin_inc"]))),
                   range_sampling_m=float(dR), sampling_ratio=float(rho_r_slant / max(dR, 1e-9)),
                   near_slant_m=float(r_near), far_slant_m=float(r_far),
                   ground_sampling_m=float(dg)),
        platform=dict(pos=[float(x) for x in pos], vel=[float(x) for x in vel],
                      altitude_m=float(alt), heading_deg=float(rc["heading_deg"]),
                      squint_deg=float(rc["squint_deg"]), t_s=float(t),
                      side=str(side), pose_source=("config" if p_fix is not None else "derived")),
        stats=stats,
        target_boxes=boxes,
        scatterers=scatterers,
        cfg=out_cfg,
    )


def _cn(rng, shape):
    """Unit-power circular complex Gaussian speckle / noise."""
    r = rng.standard_normal(shape).astype(np.float32)
    i = rng.standard_normal(shape).astype(np.float32)
    return (r + 1j * i) / np.sqrt(2.0)


# ---------------------------------------------------------------------------
# self test -- six physical assertions, no fallbacks, no fudge factors
# ---------------------------------------------------------------------------
def _st_scene(sar_kw=None, objects=None, ground=None, **cfg_kw):
    """Small deterministic world for the self test (build_world, no files)."""
    from . import scene as sc
    sar_cfg = dict(band="X", bandwidth_mhz=150.0, n_cross=128, n_samples=256,
                   altitude_m=2000.0, speed_mps=120.0, swath_center_m=2500.0,
                   swath_width_m=280.0, az_extent_m=200.0, looks=4.0)
    sar_cfg.update(sar_kw or {})
    cfg = dict(
        ground=ground if ground is not None else
        dict(enabled=True, size=6000, mat="asphalt", z=0.0),
        objects=objects if objects is not None else [_st_plate(0.0, 0.0)],
        sensors=dict(sar=sar_cfg), frames=dict(n=1, dt=0.2), _base_dir=".", seed=1,
        _frame_seed=0,
    )
    cfg.update(cfg_kw)
    return sc.build_world(cfg), cfg


def _st_plate(x, y):
    """Sub-resolution flat plate: a range cut through it is the PSF itself."""
    return dict(name="tgt", cls="target", shape="box", size=[0.4, 0.4, 0.06],
                pos=[x, y, 0.0], rot=0.0, mat="metal_bare")


def _expected_px(out, target_pos):
    """Where the geometry says a ground target must land, in pixels."""
    rc = out["cfg"]
    hd, lh, _side, _sq = _platform_basis(rc)
    p = np.asarray(out["platform"]["pos"], float)
    rel = np.asarray(target_pos, float) - p
    rel[2] = 0.0
    u, g = float(rel @ hd), float(rel @ lh)
    row = int(np.argmin(np.abs(out["cross_axis_m"] - u)))
    col = int(np.argmin(np.abs(out["range_axis_m"] - g)))
    return row, col, u, g


def _fwhm(vals, xs, i0):
    """Full width at half maximum around index i0 (linear interpolation)."""
    vals = np.asarray(vals, float)
    xs = np.asarray(xs, float)
    half = vals[i0] * 0.5
    i0 = int(np.clip(i0, 1, len(vals) - 2))
    j = i0
    while j > 0 and vals[j] > half:
        j -= 1
    dv = vals[j + 1] - vals[j]
    a = xs[j] + (half - vals[j]) * (xs[j + 1] - xs[j]) / (dv if abs(dv) > 1e-30 else 1e-30)
    k = i0
    while k < len(vals) - 1 and vals[k] > half:
        k += 1
    dv = vals[k] - vals[k - 1]
    b = xs[k - 1] + (half - vals[k - 1]) * (xs[k] - xs[k - 1]) / (dv if abs(dv) > 1e-30 else 1e-30)
    return abs(b - a)


def selftest():
    fails = 0
    rep = []

    def check(name, ok, detail):
        nonlocal fails
        fails += 0 if ok else 1
        rep.append(("PASS" if ok else "FAIL", name, detail))

    # -- 1: range resolution == c / 2B (the -3 dB width of the compressed pulse)
    b_hz = 150.0e6
    rho_th = C_LIGHT / (2.0 * b_hz)
    # isolated target on a test range: no ground clutter, so the range cut is
    # the compressed pulse and nothing else
    # no speckle, no noise: the range cut is the deterministic impulse response
    o1 = simulate(*_st_scene(dict(n_cross=64, n_samples=1024, swath_width_m=256.0,
                                  az_extent_m=40.0, speckle=False,
                                  thermal_noise=False),
                             ground=dict(enabled=False)))
    db1 = o1["image_db"]
    iy, ix = np.unravel_index(int(np.argmax(db1)), db1.shape)
    prof = (10.0 ** (db1[max(0, iy - 2):iy + 3, :] / 10.0)).mean(axis=0)
    g_ax = o1["range_axis_m"]
    h, g_pk = float(o1["cfg"]["altitude_m"]), float(g_ax[ix])
    sin_inc = g_pk / np.hypot(h, g_pk)
    width_slant = _fwhm(prof, g_ax, ix) * sin_inc
    err1 = abs(width_slant - rho_th) / rho_th
    check("range resolution = c/2B", err1 <= 0.05,
          "measured -3dB %.3f m (slant) vs theory %.3f m, err %+.1f%%"
          % (width_slant, rho_th, 100.0 * (width_slant - rho_th) / rho_th))

    # -- 2: azimuth resolution follows the stripmap aperture (D/2) law ------
    # Two separate facts, deliberately named apart so neither reads as a
    # contradiction of the other:
    #   * antenna length: rho_a = 0.886*L_ant/2 once the real beam limits the
    #     coherent aperture, so a LONGER antenna is COARSER (D/2 law);
    #   * integration time: more pulses (or a slower platform) lengthen the
    #     synthetic aperture and SHARPEN azimuth -- the knob you actually turn.
    o_s = simulate(*_st_scene(dict(antenna_len_m=1.0)))
    o_l = simulate(*_st_scene(dict(antenna_len_m=4.0)))
    a_s = o_s["resolution"]["azimuth_m"]
    a_l = o_l["resolution"]["azimuth_m"]
    check("azimuth coarsens with antenna length (D/2 law)", a_l > a_s * 1.05,
          "L=1.0 m -> %.2f m, L=4.0 m -> %.2f m (longer antenna = coarser azimuth)"
          % (a_s, a_l))
    o_p1 = simulate(*_st_scene(dict(n_pulses=256, antenna_len_m=0.5)))
    o_p2 = simulate(*_st_scene(dict(n_pulses=1024, antenna_len_m=0.5)))
    a1 = o_p1["resolution"]["azimuth_m"]
    a2 = o_p2["resolution"]["azimuth_m"]
    check("azimuth sharpens with integration time", a2 < 0.6 * a1,
          "n_pulses 256 -> %.2f m, 1024 -> %.2f m (aperture %.0f -> %.0f m)"
          % (a1, a2, o_p1["resolution"]["l_sa_m"], o_p2["resolution"]["l_sa_m"]))

    # -- 3: point target peak sits where the geometry says ----------------
    # plate offset from the swath centre along track, so the check cannot pass
    # by accident of a centred grid.
    tgt = _st_plate(0.0, 60.0)
    o3 = simulate(*_st_scene(dict(anchor=[0.0, 0.0, 0.0], n_cross=256,
                                  az_extent_m=300.0), objects=[tgt]))
    row_ex, col_ex, u_ex, g_ex = _expected_px(o3, tgt["pos"])
    pk = np.unravel_index(int(np.argmax(o3["image_db"])), o3["image_db"].shape)
    d_row, d_col = abs(int(pk[0]) - row_ex), abs(int(pk[1]) - col_ex)
    check("point target peak at geometric position", d_row <= 2 and d_col <= 2,
          "expected (row %d, col %d) for u=%.1f m g=%.1f m, peak (row %d, col %d), "
          "err %d/%d px" % (row_ex, col_ex, u_ex, g_ex, pk[0], pk[1], d_row, d_col))

    # -- 4: +3 dB transmit power -> +3 dB SNR ----------------------------
    s_a = simulate(*_st_scene(dict(tx_power_w=40.0)))["stats"]
    s_b = simulate(*_st_scene(dict(tx_power_w=80.0)))["stats"]
    d_snr = s_b["snr_db"] - s_a["snr_db"]
    check("+3 dB tx power -> +3 dB SNR", abs(d_snr - 3.0103) <= 0.5,
          "SNR %.2f -> %.2f dB (delta %+.2f dB), NESZ %.1f -> %.1f dB"
          % (s_a["snr_db"], s_b["snr_db"], d_snr, s_a["noise_db"], s_b["noise_db"]))

    # -- 5: ground inside the footprint returns, outside is noise only ----
    # The swath is always centred on the anchor, so aim the beam AT the 2400 m
    # plane's x=+1200 edge: the near half of the swath then hits ground and the
    # far half looks into empty space and may only see the receiver noise floor.
    o5 = simulate(*_st_scene(dict(anchor=[1200.0, 0.0, 0.0], n_cross=32, n_samples=512,
                                  swath_center_m=2500.0, swath_width_m=1000.0),
                             ground=dict(enabled=True, size=2400, mat="asphalt", z=0.0),
                             objects=[]))
    ax, db5 = o5["range_axis_m"], o5["image_db"]
    cov = db5[:, (ax > 2050.0) & (ax < 2450.0)]
    bare = db5[:, (ax > 2550.0) & (ax < 2950.0)]
    pw = lambda a: 10.0 * np.log10(max(float((10.0 ** (a / 10.0)).mean()), 1e-30))
    d_cov, d_bare = pw(cov), pw(bare)
    check("in-footprint ground echo > outside", d_cov > d_bare + 10.0,
          "covered %.1f dB vs bare %.1f dB (%.1f dB)" % (d_cov, d_bare, d_cov - d_bare))

    # -- 6: label box lands on the target, contrast positive --------------
    m = o3["obj_map"] >= 0
    if m.any():
        ys, xs = np.where(m)
        cy = 0.5 * (float(ys.min()) + float(ys.max()))
        cx = 0.5 * (float(xs.min()) + float(xs.max()))
        ddy, ddx = abs(cy - row_ex), abs(cx - col_ex)
        ok = ddy <= 2.0 and ddx <= 3.0 and o3["stats"]["target_contrast_db"] > 0.0
        check("target bbox on target, contrast > 0", ok,
              "bbox rows %d-%d cols %d-%d (centre %.1f,%.1f) vs true (%d,%d), "
              "offset %.1f/%.1f px, contrast %+.1f dB"
              % (ys.min(), ys.max(), xs.min(), xs.max(), cy, cx, row_ex, col_ex,
                 ddy, ddx, o3["stats"]["target_contrast_db"]))
    else:
        check("target bbox on target, contrast > 0", False, "obj_map empty")

    # -- 7: target_boxes exists even when the mask is only a pixel or two ---
    # 4 m range x 6.4 m azimuth pixels against a 0.4 m plate: obj_map can be
    # empty or a single pixel, which is exactly the case the integration layer
    # asked this field to rescue.
    o7 = simulate(*_st_scene(dict(anchor=[0.0, 0.0, 0.0], n_cross=32, n_samples=64,
                                  swath_width_m=256.0, az_extent_m=200.0),
                             objects=[_st_plate(0.0, 60.0)]))
    bx = o7["target_boxes"].get("tgt")
    row7, col7, _u7, _g7 = _expected_px(o7, [0.0, 60.0, 0.0])
    npix = int((o7["obj_map"] == 0).sum())
    ok7 = (bx is not None and len(o7["target_boxes"]) == 1
           and bx[0] <= col7 < bx[2] and bx[1] <= row7 < bx[3])
    check("target_boxes covers sub-resolution target", ok7,
          "mask %d px, box %s (x1/y1 exclusive), true pixel (row %d, col %d)"
          % (npix, bx, row7, col7))

    # -- 8: a supplied pose is the aperture centre (dataset.py rewrites it) --
    # pos must move the platform (and the swath) with it, and look_at must set
    # the look side/squint on its own: here the platform is nudged 40 m along
    # track and the target must still land where the geometry says, which only
    # holds if `pos` -- not a re-derived p0 -- is what the rays start from.
    base = simulate(*_st_scene(objects=[_st_plate(0.0, 0.0)]))
    hd_b, lh_b, _sb, _qb = _platform_basis(base["cfg"])
    p_base = np.asarray(base["platform"]["pos"], float)
    p_shift = p_base + hd_b * 40.0
    o8 = simulate(*_st_scene(dict(pos=list(p_shift),
                                  look_at=list(p_shift + lh_b * 2500.0)),
                             objects=[_st_plate(0.0, 0.0)]))
    d_pos = float(np.linalg.norm(np.asarray(o8["platform"]["pos"], float) - p_shift))
    same_alt = abs(o8["platform"]["altitude_m"] - base["platform"]["altitude_m"]) < 1e-6
    same_side = o8["platform"]["side"] == base["platform"]["side"]
    row8, col8, _u8, _g8 = _expected_px(o8, [0.0, 0.0, 0.0])
    pk8 = np.unravel_index(int(np.argmax(o8["image_db"])), o8["image_db"].shape)
    ok8 = (d_pos < 1e-6 and same_alt and same_side
           and abs(int(pk8[0]) - row8) <= 2 and abs(int(pk8[1]) - col8) <= 2)
    check("sensors.sar.pos is the aperture centre", ok8,
          "pos honoured (err %.2g m), alt %.1f m, side %s, peak (row %d, col %d) "
          "vs geometry (%d, %d)" % (d_pos, o8["platform"]["altitude_m"],
                                    o8["platform"]["side"], pk8[0], pk8[1],
                                    row8, col8))

    print("SAR selftest")
    for st, name, detail in rep:
        print("%s  %-38s %s" % (st, name, detail))
    print("%d/%d passed" % (len(rep) - fails, len(rep)))
    return fails
