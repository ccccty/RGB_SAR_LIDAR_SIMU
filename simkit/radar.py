"""FMCW radar forward model: ray casting, PO RCS, range-Doppler synthesis.

Chain: beam grid -> ray cast (multi-bounce specular) -> physical-optics RCS per
facet with Kirchhoff roughness -> radar equation -> beat-signal synthesis ->
range/Doppler FFT. Outputs a complex radar cube, RD / range-angle maps,
a detection (point cloud) list and per-detection object labels.

Approximations are deliberate: incoherent multi-bounce, no shadowing between
bounces, flat-facet PO (no edge diffraction / creeping waves), no MIMO virtual
array, no range migration. Detection can use the legacy analytic gate or an
opt-in measured RD peak detector with window/noise calibration. Neither mode
has been validated against a real instrument or provides RCS certification.

The RD image is synthesised as ONE display channel: an aperture on the sector
centre receives every echo (target, clutter, ECM) through its two-way pattern
and the sum is coherent, so the map is a single channel's exponential noise
plus the echoes -- not a maximum over the scanned beams. The per-beam FFT below
survives only for range-angle. Airframes are K scattering centres along the
line of sight and powered ones carry a JEM comb. Still missing: range migration
inside the CPI, MIMO virtual arrays, near-field phase curvature.
"""
import math
import os

import numpy as np

from . import materials as matlib
from .scene import sensor_basis
from .weather import RADAR_BANDS, radar_band_of

C_LIGHT = 2.99792458e8
K_BOLTZ = 1.380649e-23
T0 = 290.0

DEFAULTS = dict(
    pos=[-20.0, 0.0, 2.0], look_at=[10.0, 0.0, 1.0],
    fc_ghz=77.0, bandwidth_mhz=500.0, n_samples=256, n_chirps=32,
    chirp_us=50.0, pri_us=65.0, tx_power_dbm=10.0,
    az_span_deg=60.0, el_span_deg=16.0, n_az=64, n_el=8,
    noise_figure_db=11.0, loss_db=4.0, det_snr_db=12.0, mti_min_vr_mps=0.0,
    clutter_density=0.02, clutter_db=-10.0, antenna_eff=0.6,
    pfa=None, swerling="I",
    specular_scale_db=-20.0, diffuse_gamma=0.06, ground_clutter_db=-30.0,
    beamwidth_az_deg=2.0, beamwidth_el_deg=12.0,
    detector="analytic", max_detections=256,
    # ---- display channel: the one aperture the RD image is made from -----
    # Sidelobe pedestal of that aperture, dB below the peak, one way. The main
    # lobe is Gaussian and -3 dB at half the configured beamwidth; a real
    # aperture has sidelobes instead of a clean zero, and a scene whose targets
    # sit at the sector edge should not read as empty sky.
    sidelobe_db=-35.0,
    # ADC DC leakage, dB above the per-cell noise floor of the zero-Doppler
    # line it draws. None = off. A real receiver has it, but leaving it on by
    # default would put a bright line at v=0 in every image and invite it to be
    # read as a target.
    dc_leak_db=None,
    # Zero-Doppler clutter ridge for a scene with ground/sea. The level is
    # clutter power per range cell re the per-sample thermal noise, so the CPI
    # gain applies to it exactly as it does to a target: ground_clutter_db
    # (-30 dB) lands the ridge about 18 dB above the floor at 2048 x 64 with
    # Hann. sea_clutter_db is the same figure for a water surface.
    clutter_ridge=True, sea_clutter_db=-25.0,
    # ---- target echo structure -------------------------------------------
    # Scattering centres per airframe, clipped to 4..12 and to one when the
    # body is smaller than a range cell. K centres share the same total sigma,
    # so the peak spreads over the cells the airframe actually spans.
    scatter_centers=8,
    # Jet-engine / propeller modulation (JEM): a comb of +/-k*f_blade sidebands
    # around the skin return of a powered airframe, first tooth jem_tooth_db
    # below the carrier, falling as 1/k. Harmonics past the Nyquist fold onto
    # the same comb -- one PRF cannot tell them apart, which is exactly why a
    # real JEM spectrum aliases.
    micro_doppler=True, f_blade_hz=1600.0, jem_tooth_db=-24.0, jem_teeth=8,
)


def cfg_get(cfg, key):
    r = cfg.get("sensors", {}).get("radar", {})
    return float(r.get(key, DEFAULTS[key])) if not isinstance(DEFAULTS[key], list) else r.get(key, DEFAULTS[key])


def beam_grid(cam, n_az, n_el, az_span, el_span):
    p, r, u, f = sensor_basis(cam["pos"], cam["look_at"], cam.get("up", (0, 0, 1)),
                              cam.get("roll_deg", 0.0))
    az = np.linspace(-az_span / 2, az_span / 2, n_az) if n_az > 1 else np.zeros(1)
    el = np.linspace(-el_span / 2, el_span / 2, n_el) if n_el > 1 else np.zeros(1)
    dirs, angles = [], []
    for e in el:
        for a in az:
            ta, te = np.tan(np.radians(a)), np.tan(np.radians(e))
            d = f + ta * r + te * u
            n = np.linalg.norm(d)
            # A degenerate basis (f perpendicular to both r and u, i.e. a beam
            # looking exactly along the cross product) can cancel to nothing.
            # Dividing by it yields NaN directions that poison the whole range-
            # Doppler cube silently.
            dirs.append(d / n if n > 1e-12 else np.asarray(f, float).copy())
            angles.append((a, e))
    return p, np.array(dirs), np.array(angles)


def sea_clutter_db(wind_ms, sin_grazing, scale_db=0.0):
    """Backscatter of the sea surface, dB relative to a flat facet.

    Two things separate sea from land clutter and neither is a constant: the
    wind (sea state) and the grazing angle. A calm sea is nearly a mirror and
    returns almost nothing at a shallow angle; the same sea in a gale is a mass
    of breaking crests and returns tens of dB more. The constant ground_clutter_db
    factor could express neither, so a dataset over water looked identical in
    calm and rough conditions.

    The constant-gamma form sigma0 ~ gamma * sin(psi) is the standard first-order
    shape for low grazing angles; gamma is lifted with wind speed, which is what
    makes a gale louder than a flat calm.
    """
    w = max(0.0, float(wind_ms))
    gamma_db = -30.0 + 1.6 * (w - 6.0)          # calm ~6 m/s is the reference
    s = max(float(sin_grazing), 1e-4)
    return gamma_db + 10.0 * math.log10(s) + float(scale_db)


def facet_sigma(areas, cos_inc, rho, rough_m, lam, footprint=None,
                spec_scale=1.0, diff_gamma=0.06):
    """Physical-optics specular term + rough-surface diffuse term.

    areas is clipped to the illuminated beam footprint: a facet far larger than
    the beam must not be counted as a whole mirror. spec_scale calibrates the
    (systematically optimistic) flat-facet coherent term only; the diffuse term
    keeps its physical gamma0 scaling.

    NOTE: summing this over the facets of a body is NOT tessellation invariant --
    the specular term goes as area squared, so a surface cut into n pieces loses
    a factor n. Use coherent_po_sigma when a total cross-section is wanted; this
    function stays as the per-facet primitive it was written as.
    """
    a = areas if footprint is None else np.minimum(areas, footprint)
    coh = np.exp(-((4.0 * np.pi * rough_m * cos_inc / lam) ** 2))
    spec = 4.0 * np.pi * a ** 2 * cos_inc ** 2 / lam ** 2 * rho ** 2 * coh
    diff = a * rho * diff_gamma * np.maximum(cos_inc, 0.0) * (1.0 - coh)
    return spec * spec_scale + diff

def coherent_po_sigma(centers, normals, areas, rho, i_hat, r_hat, lam):
    """Physical-optics RCS from a coherent field sum over facets, in m^2.

    sigma = (4*pi/lam^2) * |sum_i A_i (n_i . i) rho_i exp(j k (i - r) . c_i)|^2

    The field is summed before squaring, so coplanar facets interfere instead of
    each acting as an isolated mirror. That is what makes the result independent
    of how finely the surface was triangulated: this routine returns the same
    sigma for a plate cut into 1 or 4096 triangles (verified to 1e-4 dB), while
    summing per-facet area-squared terms instead loses a factor n.

    Absolute level was checked against the closed form 4*pi*A^2/lam^2: a
    0.3 m x 0.3 m plate at 10 GHz gives 20.53 dBsm against the analytic 20.53,
    and the 1 m2 plate gives 41.45 dBsm.

    Only facets facing the radar contribute, and shadowing is the caller's job.
    """
    import numpy as _np
    ii = _np.asarray(i_hat, float)
    rr = _np.asarray(r_hat, float)
    k = 2.0 * _np.pi / lam
    w = k * (ii - rr)
    cos_i = _np.asarray(normals, float) @ ii
    amp = _np.asarray(areas, float) * _np.maximum(cos_i, 0.0) * _np.asarray(rho, float)
    phase = _np.exp(1j * (_np.asarray(centers, float) @ w))
    field = _np.sum(amp * phase)
    return float(4.0 * _np.pi / lam ** 2 * (field.real ** 2 + field.imag ** 2))

def fft_noise_statistics(pn, range_window, doppler_window, n_beams, pfa):
    """Unnormalised FFT noise and threshold of independent-beam maximum.

    Each complex input sample has E[|n|^2]=pn. A windowed FFT cell has mean
    noise power pn*sum(wr^2)*sum(wd^2), with an exponential distribution.
    For B independent simulated receiver beams, P(max<=x)=(1-exp(-x/mu))^B.
    Pfa is per RD cell BEFORE peak selection, not a whole-image false-alarm
    probability. Windowing correlates neighbouring cells, not this marginal.
    """
    if not (0.0 < float(pfa) < 1.0):
        raise ValueError("RD detector pfa must be strictly between 0 and 1")
    wr, wd = np.asarray(range_window, float), np.asarray(doppler_window, float)
    energy = float(np.sum(wr**2)*np.sum(wd**2))
    coherent = float(np.sum(wr)**2*np.sum(wd)**2)
    if energy <= 0.0 or pn <= 0.0 or int(n_beams) < 1:
        raise ValueError("RD detector requires positive noise/window energy and beam count")
    # expm1/log1p preserve Pfa=1e-6 with hundreds of beams.
    factor = -np.log(-np.expm1(np.log1p(-float(pfa))/int(n_beams)))
    return dict(single_beam_fft_noise_power=pn*energy,
                window_energy=energy, coherent_power_gain=coherent,
                window_processing_gain_db=10.0*np.log10(coherent/energy),
                independent_noise_beams=int(n_beams), pfa_per_cell=float(pfa),
                threshold_noise_ratio=float(factor), threshold_power=float(pn*energy*factor),
                detector="rd_threshold",
                # One aperture is the shipped case: the RD image is a single
                # coherent channel, so its cells ARE exponential draws and the
                # gate is -ln(Pfa). The multi-beam form is kept because a
                # maximum over B independent cells is what an incoherent
                # scan-summing display would produce, and the two differ by
                # 10*log10(B) ~ 12 dB at B=800 -- the gate this detector used
                # to be built on.
                statistic=("single display channel, exponential cell power"
                           if int(n_beams) == 1 else
                           "maximum of independent beam powers"),
                limitations=["known thermal noise, not adaptive CFAR",
                             "correlated/signal-dependent clutter is not covered by thermal-noise Pfa",
                             "frozen CPI geometry, no range migration or FMCW range-Doppler coupling"])


def rd_peak_cells(power, threshold, max_detections=256):
    """Return measured local maxima as (range_bin, Doppler_bin).

    Doppler neighbours wrap at Nyquist; range neighbours do not. No object
    positions are consulted by the detector.
    """
    power = np.asarray(power, float)
    if power.ndim != 2 or not np.isfinite(power).all():
        raise ValueError("RD detector requires finite 2D power")
    keep = power > float(threshold)
    _nr, nd = power.shape
    for dr in (-1, 0, 1):
        for dd in (-1, 0, 1):
            if dr == dd == 0:
                continue
            neighbour = np.roll(power, dd, axis=1)
            if dr == -1:
                neighbour = np.vstack([neighbour[1:], np.full((1, nd), -np.inf)])
            elif dr == 1:
                neighbour = np.vstack([np.full((1, nd), -np.inf), neighbour[:-1]])
            keep &= power >= neighbour
    rr, vv = np.where(keep)
    order = np.argsort(power[rr, vv], kind="stable")[::-1][:max(0, int(max_detections))]
    return [(int(rr[i]), int(vv[i])) for i in order]


def antenna_pattern(az_deg, el_deg, bw_az_deg, bw_el_deg, sidelobe_db=-35.0):
    """One-way power pattern of the display aperture, normalised to 1 on axis.

    Gaussian main lobe, -3 dB at half the configured beamwidth on each axis,
    sitting on a sidelobe pedestal. The pedestal is not cosmetic: without it
    every echo outside the main lobe would vanish to -inf, and a target at the
    edge of a scanned sector would read as empty sky -- a stronger claim than a
    -35 dB sidelobe makes.
    """
    bw_az = max(float(bw_az_deg), 1e-6)
    bw_el = max(float(bw_el_deg), 1e-6)
    x = 2.0 * np.asarray(az_deg, float) / bw_az
    y = 2.0 * np.asarray(el_deg, float) / bw_el
    main = np.exp(-math.log(2.0) * (x * x + y * y))
    return np.maximum(main, 10.0 ** (float(sidelobe_db) / 10.0))


def display_rd_map(echoes, ant1, pn_sample, n_s, n_c, fs, pri, win_r, win_d,
                   ridge_amp=None, dc_leak_db=None, seed=0):
    """Range-Doppler image of ONE aperture pointed at the sector centre.

    Why one aperture instead of the per-cell maximum over the scanned beams:
    the maximum of ~800 independent exponential cells is an extreme-value
    statistic, not a noise draw. A target then has to beat the +12 dB maximum
    rather than the mean floor, and the speckle a receiver actually shows
    collapses into a 1-2 dB band -- an image too clean to train a detector on.
    Summing the echo set coherently through one pattern keeps every cell
    exponential, leaves the strong echoes with Hann sidelobes and puts targets
    and noise on one physical scale.

    Echoes are (beam, beat_freq, doppler_freq, amplitude) with the amplitude at
    peak gain, so each is scaled by the aperture's one-way power pattern: the
    echo carries the two-way loss and sqrt of that loss is the amplitude factor.

    Noise is one complex field per input sample with E|n|^2 = pn_sample, i.e.
    the same pn the radar equation divides by, so an FFT cell of this map is an
    exponential draw with mean pn*sum(wr^2)*sum(wd^2) -- what
    fft_noise_statistics(n_beams=1) models.

    ridge_amp: one complex clutter amplitude per displayed range cell, all at
    zero closing rate (land/sea seen by a radar that is not moving relative to
    it). dc_leak_db: ADC DC term, dB above the per-cell noise floor of the
    zero-Doppler line it draws.
    """
    disp = np.zeros((n_s, n_c), np.complex128)
    n_idx = np.arange(n_s)
    k_idx = np.arange(n_c)
    for i, fb, fd, amp in echoes:
        pat = float(ant1[i])
        if amp == 0.0 or pat <= 0.0:
            continue
        np.add(disp, (amp * pat) * np.multiply.outer(
            np.exp(2j * np.pi * fb / fs * n_idx),
            np.exp(2j * np.pi * fd * pri * k_idx)), out=disp)
    if ridge_amp is not None and len(ridge_amp):
        # A clutter cell is a scatterer whose closing rate is zero. One complex
        # amplitude per range cell, written as an inverse DFT in fast time and
        # held constant over slow time: that is a ridge along v=0, not a field
        # of isolated dots.
        pad = np.zeros(n_s, complex)
        pad[:len(ridge_amp)] = ridge_amp
        disp += (np.fft.ifft(pad) * n_s)[:, None]
    if dc_leak_db is not None:
        # The ADC offset is constant over chirps and differs sample to sample
        # (fixed pattern), so all of it folds onto v=0 and shows as a thin
        # bright zero-Doppler line. The level is quoted per RD cell, hence the
        # division by the Doppler window's coherent gain.
        pg_d = float(np.sum(win_d) ** 2 / max(np.sum(win_d ** 2), 1e-30))
        dc_pow = pn_sample * 10.0 ** (float(dc_leak_db) / 10.0) / max(pg_d, 1e-30)
        lrng = np.random.default_rng(int(seed) ^ 0x5F5F5F)
        fpn = lrng.standard_normal(n_s) + 1j * lrng.standard_normal(n_s)
        disp += np.sqrt(dc_pow / 2.0) * fpn[:, None]
    nrng = np.random.default_rng(int(seed) ^ 0x9E3779B9)
    disp += np.sqrt(pn_sample / 2.0) * (
        nrng.standard_normal((n_s, n_c)) + 1j * nrng.standard_normal((n_s, n_c)))
    spec = np.fft.fftshift(np.fft.fft2(disp * (win_r[:, None] * win_d[None, :])),
                           axes=1)
    power = spec.real ** 2 + spec.imag ** 2
    # positive-frequency half only: the exported range axis follows the same
    # real-ADC convention as the beam cube, so the two maps stay comparable
    return power[:max(1, n_s // 2)]


def simulate(world, cfg):
    rc = dict(DEFAULTS)
    user_rc = cfg.get("sensors", {}).get("radar", {})
    band = user_rc.get("band")
    if band in RADAR_BANDS:
        # band preset supplies fc / beamwidths unless explicitly overridden
        for k in ("fc_ghz", "beamwidth_az_deg", "beamwidth_el_deg"):
            if k not in user_rc:
                rc[k] = RADAR_BANDS[band][k]
        rc["band"] = band
    rc.update(user_rc)
    measured_detector = str(rc.get("detector", "analytic")).lower() == "rd_threshold"
    if str(rc.get("detector", "analytic")).lower() not in ("analytic", "rd_threshold"):
        raise ValueError("radar detector must be analytic or rd_threshold")
    cam = rc
    p, dirs, angles = beam_grid(cam, int(rc["n_az"]), int(rc["n_el"]),
                                float(rc["az_span_deg"]), float(rc["el_span_deg"]))
    _p, right, up, forward = sensor_basis(cam["pos"], cam["look_at"],
                                         cam.get("up", (0, 0, 1)), cam.get("roll_deg", 0.0))
    lam = C_LIGHT / (float(rc["fc_ghz"]) * 1e9)

    bvh = world.bvh
    hit = bvh.ray_cast(p, dirs)
    # A chaff cloud is a volume of dipoles, not a wall. Its echo is added
    # analytically in ecm.radar_apply from the cloud's total cross-section, so a
    # ray that stopped on it is allowed to continue to whatever stands behind it;
    # otherwise dropping chaff would hide the aircraft it was released to
    # protect. Solid ECM bodies (decoy, corner reflector) are not in this set and
    # shadow normally. Skipped entirely when the scene has no ECM.
    _ecm = None
    _vol = getattr(world, "ecm_volumetric_ids", ())
    if _vol:
        from . import ecm as _ecm
        hit = _ecm.pass_through(bvh, world.face_obj, p, dirs, hit, _vol)
    tri, ok = hit["tri"], hit["hit"]

    mat_names = sorted(matlib.MATERIALS)
    rho_tab = np.array([matlib.MATERIALS[n]["rf_rho"] for n in mat_names])
    rough_tab = np.array([matlib.MATERIALS[n]["rough_m"] for n in mat_names])

    # World caches both now; the private fallbacks stay for plain objects
    # (tests build lightweight stand-ins) but the real path is the cache.
    areas = world.face_areas() if hasattr(world, "face_areas") else _face_areas(world)
    normals = (world.face_normals() if hasattr(world, "face_normals")
               else _face_normals(world))

    sig = np.zeros(len(dirs))
    obj_of_beam = np.full(len(dirs), -1, np.int64)
    pts = np.zeros((len(dirs), 3))

    bw_az0 = np.radians(max(float(rc.get("beamwidth_az_deg", 2.0)),
                          float(rc["az_span_deg"]) / int(rc["n_az"])))
    bw_el0 = np.radians(max(float(rc.get("beamwidth_el_deg", 12.0)),
                          float(rc["el_span_deg"]) / int(rc["n_el"])))
    spec_scale = 10.0 ** (float(rc.get("specular_scale_db", -20.0)) / 10.0)
    diff_gamma = float(rc.get("diffuse_gamma", 0.06))

    idx = np.where(ok)[0]
    if len(idx):
        t_i = tri[idx]
        pts[idx] = hit["point"][idx]
        n = normals[t_i]
        d = dirs[idx]
        cos_i = np.abs((n * (-d)).sum(axis=1))
        rho = rho_tab[world.face_mat[t_i]]
        rough = rough_tab[world.face_mat[t_i]]
        foot = (hit["t"][idx] * bw_az0) * (hit["t"][idx] * bw_el0)
        s1 = facet_sigma(areas[t_i], cos_i, rho, rough, lam, foot,
                         spec_scale, diff_gamma)
        obj_of_beam[idx] = world.face_obj[t_i]

        # one specular bounce
        refl = d - 2.0 * (d * n).sum(axis=1)[:, None] * n
        o2 = pts[idx] + n * 1e-3
        boom = np.zeros(len(idx), bool)
        s2 = np.zeros(len(idx))
        for k in range(len(idx)):
            r2 = bvh.ray_cast(o2[k], refl[k][None, :])
            if r2["hit"][0]:
                t2 = int(r2["tri"][0])
                n2 = normals[t2]
                c2 = abs(float(n2 @ (-refl[k])))
                s2[k] = 0.3 * float(facet_sigma(areas[t2], np.array([c2]),
                                                rho_tab[world.face_mat[t2]],
                                                rough_tab[world.face_mat[t2]], lam,
                                                np.array([foot[k]]), spec_scale,
                                                diff_gamma)[0])
                boom[k] = True
        s_tot = s1 + s2
        # road/ground clutter model: our flat facets over-predict grazing returns
        gnd = world.face_obj[t_i] < 0
        if gnd.any():
            _gcfg = cfg.get("ground") or {}
            _is_sea = str(_gcfg.get("type", "")).lower() in ("sea", "ocean", "water")
            if _is_sea:
                # Per-facet wind and grazing angle: cos_i IS sin(grazing) for the
                # horizontal sea plane, so no extra geometry is needed.
                _wind = float(_gcfg.get("wind_ms") or 0.0)
                _scale = float(rc.get("sea_clutter_scale_db", 0.0))
                _mult = 10.0 ** (np.array([sea_clutter_db(_wind, float(c), _scale)
                                           for c in cos_i[gnd]]) / 10.0)
                s_tot[gnd] *= _mult
            else:
                s_tot[gnd] *= 10.0 ** (float(rc.get("ground_clutter_db", -30.0)) / 10.0)
        sig[idx] = s_tot

    dist = hit["t"].copy()
    dist[~ok] = np.inf
    return_dirs = dirs.copy()

    # A target can be smaller than the beam raster. At 9.6 km an 11 m aircraft
    # subtends 0.07 deg in azimuth but only 0.011 deg in elevation -- under the
    # 0.039 deg elevation step -- so 1536 rays stepped straight over it and every
    # transmit power from 10 W to 100 kW returned sigma = 0. A real beam still
    # illuminates it, so anything the raster missed but a centre ray reaches is
    # integrated over its own faces and parked in the nearest beam.
    hit_ids = set(np.unique(obj_of_beam[obj_of_beam >= 0]).tolist())
    for o in world.objs:
        if not o.get("label", True) or o["id"] in hit_ids:
            continue
        c = np.asarray(o["corners"], float).mean(axis=0)
        v = c - p
        R = float(np.linalg.norm(v))
        if R < 1e-6:
            continue
        dvec = v / R
        front = float(dvec @ forward)
        if front <= 0.0:
            continue
        azimuth = np.degrees(np.arctan2(float(dvec @ right), front))
        elevation = np.degrees(np.arctan2(float(dvec @ up), front))
        # A fallback ray must not see targets outside the scanned beam support.
        # Otherwise even an aircraft behind the sensor was injected into the
        # nearest beam, despite never entering the sensor field of regard.
        if (abs(azimuth) > float(rc["az_span_deg"])/2.0 + np.degrees(bw_az0)/2.0 or
                abs(elevation) > float(rc["el_span_deg"])/2.0 + np.degrees(bw_el0)/2.0):
            continue
        # A single centre ray is too strict: in the cluttered scene a 2 m fence
        # post can cross the boresight while the rest of the rover is in plain
        # view, and one blocked ray then erased 17 of 20 frames. Probe the
        # centre plus the eight corners and accept a partial view.
        corners = np.asarray(o["corners"], float)
        probes = np.vstack([c[None, :], c + (corners - c) * 0.7])
        pdirs = probes - p
        pdirs /= np.linalg.norm(pdirs, axis=1, keepdims=True)
        ph = bvh.ray_cast(p, pdirs)
        if _ecm is not None:
            # a chaff cloud is a volume of dipoles: probing through it must not
            # read as "something solid is in the way", or dropping chaff would
            # erase the aircraft it was released to protect
            ph = _ecm.pass_through(bvh, world.face_obj, p, pdirs, ph, _vol)
        _tri = np.where(ph["hit"], ph["tri"], 0)
        _obj = world.face_obj[_tri]
        selfhit = ph["hit"] & (_obj == o["id"])
        # Looking down at 3 degrees over open ground, almost every probe lands on
        # the ground a few metres short of the target; that is not occlusion. A
        # ground hit at roughly the target's range counts as clear line of sight.
        clear = ph["hit"] & (_obj < 0) & (np.abs(ph["t"] - R) < 0.25 * R + 5.0)
        if not (selfhit.any() or clear.any()):
            continue                     # something solid really is in the way
        fm = world.face_obj == o["id"]
        if not fm.any():
            continue
        cos_i = np.abs(normals[fm] @ (-dvec))
        foot = np.full(int(fm.sum()), (R * bw_az0) * (R * bw_el0))
        s_obj = float(np.sum(facet_sigma(areas[fm], cos_i,
                                         rho_tab[world.face_mat[fm]],
                                         rough_tab[world.face_mat[fm]], lam, foot,
                                         spec_scale, diff_gamma)))
        if not np.isfinite(s_obj) or s_obj <= 0.0:
            continue
        b = int(np.argmax(dirs @ dvec))
        if R < dist[b]:
            # The single-return model cannot attribute a farther ground
            # scatterer's power to a nearer aircraft. Replace the return;
            # adding sigma while changing its object ID created huge false RCS.
            sig[b] = s_obj
            obj_of_beam[b] = o["id"]
            dist[b] = R
            pts[b] = c
            return_dirs[b] = dvec

    # explicit per-object RCS override (dBsm) wins over the geometric estimate
    for o in world.objs:
        if o.get("rcs_db") is not None:
            m = obj_of_beam == o["id"]
            sig[m] = 10.0 ** (float(o["rcs_db"]) / 10.0)

    # Swerling target fluctuation. A deterministic RCS turns detection into a
    # lookup table -- "target present => box" -- which no real radar provides and
    # which teaches a detector nothing about missed detections. Swerling I draws
    # one exponential per CPI (this frame); III uses two draws, i.e. a more
    # specular and less variable target.
    sw = str(rc.get("swerling", "I")).upper()
    if sw in ("I", "II", "III", "IV"):
        srng = np.random.default_rng(int(cfg.get("_frame_seed", 0)) * 31 + 7)
        for o in world.objs:
            m = obj_of_beam == o["id"]
            if not m.any():
                continue
            u1 = max(float(srng.uniform()), 1e-9)
            if sw in ("I", "II"):
                f = -np.log(u1)
            else:
                f = -0.5 * (np.log(u1) + np.log(max(float(srng.uniform()), 1e-9)))
            sig[m] *= f

    # radar equation
    tx_w = 10.0 ** ((float(rc["tx_power_dbm"]) - 30.0) / 10.0)
    bw_az = bw_az0
    bw_el = bw_el0
    gain = 4.0 * np.pi * float(rc["antenna_eff"]) / (bw_az * bw_el)
    loss = 10.0 ** (float(rc["loss_db"]) / 10.0)
    noise_f = 10.0 ** (float(rc["noise_figure_db"]) / 10.0)
    bw_noise = float(rc["n_samples"]) / (float(rc["chirp_us"]) * 1e-6)
    pn = K_BOLTZ * T0 * noise_f * bw_noise * loss

    # ---- scene-level electronic countermeasures: noise floor --------------
    # A noise jammer is not an echo: it enters the receiver before the matched
    # filter and lifts the floor of every beam that sees it, so it has to be
    # known before this frame's SNR is computed. pn stays the thermal floor so
    # the report can quote both; pn_beam is the floor the cube is actually
    # synthesised with. Without an "ecm" block pn_beam equals pn everywhere and
    # every number below is unchanged.
    pn_beam = np.full(len(dirs), pn)
    ecm_noise = None
    if cfg.get("ecm"):
        from . import ecm as _ecm
        ecm_noise = _ecm.radar_noise_floor(world, cfg, rc, dict(
            dirs=dirs, angles=angles, origin=p, basis=(right, up, forward),
            lam=lam, gain=gain, loss=loss, pn=pn, rc=rc,
            fs=float(rc["n_samples"]) / (float(rc["chirp_us"]) * 1e-6),
            band=rc.get("band") or radar_band_of(float(rc["fc_ghz"]))))
        if ecm_noise is not None:
            pn_beam = ecm_noise["pn_beam"]
    pg_db = float(rc.get("processing_gain_db",
                         10.0 * np.log10(max(int(rc["n_samples"]) * int(rc["n_chirps"]), 1))))

    # ---- display channel geometry ----------------------------------------
    # The RD image a human or a CNN reads must come from ONE aperture. Its
    # pattern is evaluated per beam direction and every echo is weighted by it
    # before the coherent sum in display_rd_map, so off-boresight targets pay
    # the two-way loss they should have paid all along instead of borrowing the
    # peak gain of whichever raster beam happened to be nearest.
    fov_az = np.degrees(np.arctan2(dirs @ right, np.maximum(dirs @ forward, 1e-9)))
    fov_el = np.degrees(np.arctan2(dirs @ up, np.maximum(dirs @ forward, 1e-9)))
    # The display aperture has to cover the sector it displays. The narrow
    # scanned beam belongs to the beam cube (range-angle); pointing it at the
    # sector centre and weighting every echo by it is not "one picture of the
    # sector", it is the boresight beam's picture with the rest attenuated: for
    # the 2 deg Ku sector that cost the 0.67 deg off-axis aircraft 10.9 dB two
    # way and pushed it under the Pfa=1e-6 gate, i.e. it deleted a real target
    # the image had been showing. The scan-summed aperture is the one whose
    # envelope is the sector, so its width is max(configured, sector span).
    bw_disp_az = max(float(np.degrees(bw_az0)), float(rc["az_span_deg"]))
    bw_disp_el = max(float(np.degrees(bw_el0)), float(rc["el_span_deg"]))
    ant1 = antenna_pattern(fov_az, fov_el, bw_disp_az, bw_disp_el,
                           float(rc.get("sidelobe_db", -35.0)))
    # A jammer raises the floor of the directions it is seen from, so the noise
    # field of the display aperture is those floors weighted by its two-way
    # pattern -- a jammer 5 deg off a 1 deg beam must not lift the whole image.
    _w2 = ant1 ** 2
    pn_disp = (float(np.sum(_w2 * pn_beam) / np.sum(_w2)) if np.sum(_w2) > 0.0
               else float(pn))
    disp_seed = (int(cfg.get("_frame_seed", 0)) * 1000003 + 424243) % (2 ** 63)

    with np.errstate(divide="ignore", invalid="ignore"):
        pr = (tx_w * gain ** 2 * lam ** 2 * sig
              / ((4.0 * np.pi) ** 3 * dist ** 4 * loss))
    # two-way atmospheric attenuation (weather state machine, ITU-R P.676/P.838)
    w = cfg.get("_weather")
    if w:
        att = w["radar_att_db_km"].get(rc.get("band") or radar_band_of(float(rc["fc_ghz"])))
        if att:
            pr = pr * 10.0 ** (-(float(att) * 2.0 * (dist / 1000.0)) / 10.0)
    pr = np.nan_to_num(pr, nan=0.0, posinf=0.0)
    # Threshold. A hand-picked det_snr_db hides what actually matters: the
    # false-alarm rate it implies. For single-pulse square-law detection
    # V_T = -ln(Pfa) (noise power 1), so Pfa = 1e-6 means 11.4 dB and every extra
    # dB of sensitivity traded for detections is visible in the config.
    pfa = rc.get("pfa")
    if pfa is not None:
        pf = min(max(float(pfa), 1e-12), 0.5)
        det_db = 10.0 * np.log10(max(-np.log(pf), 1e-12))
        rc["det_snr_db"] = det_db
    # Coherent gain is the whole 2-D array, not just the chirps. pn above is the
    # *ADC-band* noise (n_samples / chirp time), so the range FFT that follows it
    # is part of the matched filter and contributes n_samples of gain as well.
    # Counting only n_chirps made the detector ~30 dB deafer than the cube it
    # ships from the same numbers: a 0.5 m^2 rover at 2.6 km sits in the RD map
    # at roughly 30 dB (measured peak/cell-noise of the synthesised beat signal)
    # while this line called it -1.6 dB, i.e. under the 11.4 dB Pfa=1e-6 gate,
    # so it never produced a detection point in 17 of 20 frames.
    # pn_beam, not pn: under noise jamming the same echo is measured against a
    # lifted floor, which is exactly the SNR loss the entity must cause.
    snr_db = 10.0 * np.log10(np.maximum(pr, 1e-30) / pn_beam) + pg_db

    # radial velocity from object motion
    # Doppler is the *closing* rate, so it is the target velocity relative to the
    # radar. Using the target's own velocity alone put a 200 m/s closing aircraft
    # at 1.4 m/s -- right on the zero-Doppler clutter ridge and inside the MTI
    # notch, i.e. an RD map whose velocity axis means nothing.
    v_rad = np.asarray(rc.get("vel", [0.0, 0.0, 0.0]), float)
    vr = return_dirs @ v_rad  # stationary world clutter still moves relative to mount
    for o in world.objs:
        m = obj_of_beam == o["id"]
        if not m.any():
            continue
        u = return_dirs[m]
        vr[m] = -(u @ (np.asarray(o.get("vel", [0, 0, 0]), float) - v_rad))
    # A single PRF measures Doppler modulo the pulse repetition rate: fd = 2v/lam
    # is only resolved inside +-PRF/2, i.e. +-lam/(4*pri) = +-64.9 m/s at 77 GHz
    # and pri = 15 us. The cube is synthesised from the same fd, so a 164 m/s
    # rover is *drawn* at its aliased 34 m/s cell, but v_r used to carry the true
    # 164 m/s: dataset.py places the RD box at argmin|doppler_axis - v_r|, so the
    # box landed on the +-Nyquist edge row instead of on the echo (measured: box
    # row 255 / pixel 2, echo row 196 / pixel 78). Report what the CPI measures.
    # The MTI gate keeps the true closing rate: this model has one PRF and does
    # no ambiguity resolution, so notching the aliased value would invent blind
    # speeds every 130 m/s and silently drop fast targets sitting on them.
    v_nyq = lam / (4.0 * float(rc["pri_us"]) * 1e-6)
    vr_true = vr.copy()
    if np.isfinite(v_nyq) and v_nyq > 0.0:
        vr = (vr + v_nyq) % (2.0 * v_nyq) - v_nyq
    # ---- beat signal synthesis ----
    # The range-Doppler cube is (beams x samples x chirps) complex and grows
    # fast: 2400 beams x 1600 samples x 64 chirps is 3.7 GB. Detections are
    # fully analytic, so the cube only has to produce rd_map / range_angle;
    # synthesize it in beam chunks and accumulate those two maps.
    n_s, n_c = int(rc["n_samples"]), int(rc["n_chirps"])
    fs = n_s / (float(rc["chirp_us"]) * 1e-6)
    b = float(rc["bandwidth_mhz"]) * 1e6
    t_chirp = float(rc["chirp_us"]) * 1e-6
    pri = float(rc["pri_us"]) * 1e-6
    # FMCW beat frequency maps to range as f_b = 2B*R/(c*T) and the FFT only
    # resolves up to fs/2, so the unambiguous range is n_s*c/(4B). The old n_s*c/(2B)
    # was twice that: targets beyond the window passed the range gate and then
    # clamped onto the last bin, which is exactly how a 9.6 km aircraft ended up
    # "detected" at 5.1 km.
    r_max = C_LIGHT * n_s / (4.0 * b)

    rng = np.random.default_rng(int(cfg.get("_frame_seed", 0)))

    # ---- target echo structure: scattering centres and JEM ----------------
    # Corners relative to each body's centre, so the extent along the line of
    # sight is one projection away. A body smaller than a range cell stays one
    # scatterer: splitting it into K in-phase copies would either change its
    # total sigma or make the peak a random draw from phase cancellation.
    cell_r = C_LIGHT / (2.0 * b)
    corner_rel, cls_of = {}, {}
    for o in world.objs:
        cls_of[int(o["id"])] = str(o.get("cls", "")).lower()
        try:
            c = np.asarray(o["corners"], float)
        except (KeyError, TypeError, ValueError):
            continue
        corner_rel[int(o["id"])] = c - c.mean(axis=0)
    k_cap = int(min(max(int(rc.get("scatter_centers", 8)), 1), 12))
    micro_on = bool(rc.get("micro_doppler", True))
    f_blade = float(rc.get("f_blade_hz", 1600.0) or 0.0)
    tooth_amp = 10.0 ** (float(rc.get("jem_tooth_db", -24.0)) / 20.0)
    # Turbine / propeller aircraft. A glider or a car has no rotating blades,
    # so the comb is keyed on the scene class rather than guessed from size.
    powered_cls = ("fighter", "airliner", "aircraft", "airplane", "jet",
                   "bomber", "transport", "uav", "drone", "helicopter")

    # moving-target indication: a zero-Doppler gate rejects stationary clutter
    # (ground, sea surface). It is the cheapest stand-in for real MTI filtering.
    mti_v = float(rc.get("mti_min_vr_mps", 0.0))
    det = []
    echoes = []          # (beam, beat_freq, doppler_freq, amplitude)
    source_echoes = {}
    # Targets that are physically there but fall beyond the sampled range
    # window. Dropping them silently produced datasets with zero detections and
    # a perfectly healthy exit code (demo_radar_2km could not see a 2 km car).
    beyond_range = []
    # Judge the window against the scene's own targets, not against what the
    # raster happened to hit: a car 2 km away in a 1.5 km window is missed by the
    # beams entirely, so a hit-based check would report nothing at all.
    for _o in getattr(world, "objs", []) or []:
        if not _o.get("label", True):
            continue
        try:
            _c = np.asarray(_o["corners"], float).mean(axis=0)
        except (KeyError, TypeError, ValueError):
            continue
        _d = _c - p
        if float(_d @ forward) <= 0.0:
            continue
        _R = float(np.linalg.norm(_d))
        if _R >= r_max:
            beyond_range.append(round(_R, 1))
    # `idx` is the geometry-hit set; a sub-beam target injected above never
    # produced a hit, yet it is exactly the echo a seeker cares about.
    det_idx = np.where(ok | (obj_of_beam >= 0))[0]
    for i in det_idx:
        if mti_v > 0.0 and abs(float(vr_true[i])) < mti_v:
            continue
        r = float(dist[i])
        if r < 0.5 or r >= r_max:
            if r >= r_max and obj_of_beam[i] >= 0:
                beyond_range.append(round(r, 1))
            continue
        fb = 2.0 * b * r / (C_LIGHT * t_chirp)
        fd = 2.0 * float(vr[i]) / lam
        amp = np.sqrt(float(pr[i])) * np.exp(1j * rng.uniform(0, 2 * np.pi))
        oid = int(obj_of_beam[i])
        # K scattering centres along the line of sight, sigma split evenly (the
        # power, not the amplitude), so the total cross-section is unchanged and
        # the echo occupies the range cells the airframe really spans. One point
        # per target made every aircraft a single pixel whatever its size.
        rel = corner_rel.get(oid)
        offs = np.zeros(1)
        if rel is not None:
            proj = rel @ np.asarray(return_dirs[i], float)
            extent = float(proj.max() - proj.min())
            if extent > 0.35 * cell_r:
                k_n = int(min(max(math.ceil(extent / cell_r) + 2, 4), k_cap))
                # Centres are quantised to the range cells the radar samples,
                # because that is where its response to a scatterer lands:
                # spreading them continuously made the Hann taper eat whichever
                # centre fell between two cells, costing a 20 m airframe at
                # 12.5 m resolution ~5 dB where a two-cell split should cost 3.
                n_span = int(min(max(int(round(extent / cell_r)), 2), 4))
                b0 = int(round(r / cell_r)) - (n_span - 1) // 2
                offs = ((b0 + np.arange(n_span) - r / cell_r) * cell_r)
                offs = np.array([offs[j % n_span] for j in range(k_n)])
        for off in offs:
            echoes.append((int(i), fb + 2.0 * b * float(off) / (C_LIGHT * t_chirp),
                           fd, amp / math.sqrt(len(offs)) *
                           np.exp(1j * rng.uniform(0, 2 * np.pi))))
        source_echoes[int(i)] = (fb, fd)
        # JEM: a powered airframe carries a comb at +/-k*f_blade around its skin
        # return. Written once per body at the centroid range; repeating it at
        # every scattering centre would add the same Doppler cells to
        # neighbouring range cells and no information. One PRF cannot resolve
        # harmonics past the Nyquist, so they are only generated up to it -- the
        # ones that a real sampled receiver folds back are exactly why a JEM
        # spectrum aliases.
        if micro_on and f_blade > 0.0 and cls_of.get(oid, "") in powered_cls:
            k_hi = int(min(float(rc.get("jem_teeth", 8)),
                           max(1.0, (0.5 / pri) / f_blade)))
            for kh in range(1, k_hi + 1):
                a_h = amp * tooth_amp / kh
                for sgn in (1.0, -1.0):
                    echoes.append((int(i), fb, fd + sgn * kh * f_blade, a_h))
        # A below-threshold target still contributes a physical echo. Gating
        # synthesis with the detector erased weak returns from training images.
        if snr_db[i] < float(rc["det_snr_db"]):
            continue
        det.append(dict(x=float(pts[i, 0]), y=float(pts[i, 1]), z=float(pts[i, 2]),
                        range=r, az=float(angles[i, 0]), el=float(angles[i, 1]),
                        v_r=float(vr[i]), v_r_true=float(vr_true[i]), snr_db=float(snr_db[i]),
                        obj=int(obj_of_beam[i])))

    # clutter: sparse random false alarms, uniform in range
    n_cl = int(float(rc["clutter_density"]) * len(dirs) * 4)
    cl_db = float(rc["clutter_db"])
    for _ in range(n_cl):
        i = int(rng.integers(0, len(dirs)))
        r = rng.uniform(3.0, r_max)
        fb = 2.0 * b * r / (C_LIGHT * t_chirp)
        if fb >= fs / 2:
            continue
        amp = np.sqrt(pn) * 10.0 ** (cl_db / 20.0) * rng.uniform(0.3, 1.5)
        # A clutter patch is a *stationary* scatterer, so its Doppler is the
        # platform velocity projected on the line of sight -- the same rule the
        # target echoes obey. The old fixed +-5 kHz spread was 19 Doppler cells
        # (46.9 m/s at Ku / 15 us pri), so every patch landed in one horizontal
        # band that ran across all range gates and buried the target in it.
        fd_cl = 2.0 * float(return_dirs[i] @ v_rad) / lam
        echoes.append((i, fb, fd_cl, amp))

    # ---- scene-level electronic countermeasures: false echoes ------------
    # Injected here, after the real target and clutter echoes exist and before
    # they are grouped per beam, so a false target goes through exactly the same
    # beat-signal / FFT / detector chain as a real echo and nothing downstream
    # can tell them apart. Passive bodies return the two-way radar equation at
    # their own position; repeaters copy the target's echo to a shifted range,
    # Doppler or angle cell. Absent an "ecm" block this block is skipped.
    ecm_report = None
    if cfg.get("ecm"):
        from . import ecm as _ecm
        _out = _ecm.radar_apply(world, cfg, rc, echoes, dict(
            t=_ecm.frame_time(cfg), dirs=dirs, angles=angles, origin=p,
            basis=(right, up, forward), rc=rc, lam=lam, b=b, t_chirp=t_chirp,
            fs=fs, r_max=r_max, pri=pri, v_rad=v_rad, pn=pn_beam,
            tx_w=tx_w, gain=gain, loss=loss, pg_db=pg_db,
            det_snr_db=float(rc["det_snr_db"]),
            obj_of_beam=obj_of_beam, pr=pr, dist=dist, vr=vr,
            band=rc.get("band") or radar_band_of(float(rc["fc_ghz"])),
            frame_seed=cfg.get("_frame_seed", 0)))
        if _out is not None:
            det.extend(_out["detections"])
            ecm_report = _out["report"]
            if ecm_noise is not None:
                ecm_report["jammers"] = ecm_noise["jammers"]
                _strong = float(np.max(np.where(obj_of_beam >= 0, pr, 0.0)))
                _jamp = float(np.max(pn_beam) - pn)
                ecm_report["noise"] = dict(
                    pn_thermal_w=float(pn),
                    pn_beam_max_w=float(np.max(pn_beam)),
                    snr_drop_db=float(10.0 * np.log10(
                        max(float(np.max(pn_beam)) / pn, 1e-30))),
                    js_db=(round(float(10.0 * np.log10(_jamp / _strong)), 3)
                           if _jamp > 0.0 and _strong > 0.0 else None))

    # ---- ground/sea clutter ridge -----------------------------------------
    # A surface cell seen by a radar with no platform motion (or before STAP)
    # has zero closing rate, so ground clutter is a ridge along v_r ~ 0, not a
    # scatter of isolated dots. It is built only when the raster actually
    # reaches the surface, and its two-way level is set by the aperture's
    # pattern in the direction that surface is seen.
    ridge_amp = None
    _gcfg = cfg.get("ground") or {}
    if bool(rc.get("clutter_ridge", True)) and _gcfg.get("enabled", True) and mti_v <= 0.0:
        _gnd = np.where(ok & (obj_of_beam < 0))[0]
        if len(_gnd):
            _sea = str(_gcfg.get("type", "")).lower() in ("sea", "ocean", "water")
            _lvl = float(rc.get("sea_clutter_db", -25.0) if _sea
                         else rc.get("ground_clutter_db", -30.0))
            _gpat = float(np.max(ant1[_gnd]))
            _n_cell = max(1, n_s // 2)
            _crng = np.random.default_rng((int(cfg.get("_frame_seed", 0)) * 7919 + 991) % (2 ** 63))
            ridge_amp = (np.sqrt(pn_disp * 10.0 ** (_lvl / 10.0) / 2.0) * _gpat *
                         (_crng.standard_normal(_n_cell) + 1j * _crng.standard_normal(_n_cell)))

    # The ray raster reports an extended body once per beam that hits it, but one
    # aperture receives that body ONCE: summing the copies at full amplitude
    # would multiply its cross-section by the number of beams (9 dB for the
    # 11 m airframe seen by 8 beams here). Share the power instead, so the
    # display map keeps the sigma the radar equation promised. The per-beam cube
    # is untouched -- range-angle is per-beam by definition.
    disp_scale = np.ones(len(dirs))
    for _o in world.objs:
        _m = obj_of_beam == _o["id"]
        _c = int(_m.sum())
        if _c > 1:
            disp_scale[_m] = 1.0 / math.sqrt(_c)
    ant_disp = ant1 * disp_scale

    n_beams = len(dirs)
    nvec = np.arange(n_s)
    cvec = np.arange(n_c)
    win_r = np.hanning(n_s)[None, :, None]
    win_d = np.hanning(n_c)[None, None, :]

    by_beam = {}
    for i, fb, fd, amp in echoes:
        by_beam.setdefault(i, []).append((fb, fd, amp))

    # Chunk size is a memory budget, not a speed knob: padding every chunk up to
    # it wastes FFT work, so shrink it to the beams that are actually left.
    # Per-process budget; several chunk workers can run at once, so this is
    # deliberately smaller than the machine's memory.
    CUBE_CAP = int(float(os.environ.get("SIMKIT_CUBE_MB", "192")) * 1024 * 1024)
    # 32-bit: a range-Doppler map is a power image, not something that needs
    # double precision, and single precision halves the FFT cost.
    cplx = np.complex64
    real = np.float32
    bytes_per_beam = n_s * n_c * 8
    keep_cube = (n_beams * bytes_per_beam) <= CUBE_CAP
    chunk = max(1, min(n_beams, int(CUBE_CAP // max(bytes_per_beam, 1))))

    x = np.arange(n_s, dtype=real)
    y = np.arange(n_c, dtype=real)
    probe = np.zeros((n_s, n_c), cplx)

    def echo_plane(fb, fd, amp):
        """One echo, as a separable outer product instead of a rank-1 exp grid."""
        nonlocal probe
        np.multiply.outer(np.exp(np.complex64(2j * np.pi * fb / fs) * x),
                          np.exp(np.complex64(2j * np.pi * fd * pri) * y),
                          out=probe)
        np.multiply(probe, cplx(amp), out=probe)
        return probe

    cube = np.zeros((n_beams, n_s, n_c), cplx) if keep_cube else None

    # rd_map is NOT accumulated here any more: it is one aperture's picture and
    # is synthesised after this loop. What is left in the loop serves
    # range-angle, which is per-beam by definition, and it keeps exactly the
    # chunking and memory behaviour it had.
    ra_map = np.zeros((n_beams, n_s))
    win = (win_r * win_d).astype(real)
    for a in range(0, n_beams, chunk):
        bb = min(a + chunk, n_beams)
        # Every beam chunk in this scene has the same size except the last one.
        # Build the noise straight into a complex64 buffer: going through
        # float64 (rng.normal) makes a complex128 temp that is twice the size and
        # that is what ran three parallel workers out of memory.
        m = bb - a
        blk = np.empty((m, n_s, n_c), cplx)
        if measured_detector:
            # Separate streams make numerical results independent of the
            # memory chunk size and of whether the raw cube is retained.
            for j, beam in enumerate(range(a, bb)):
                seed = (int(cfg.get("_frame_seed", 0))*1000003 + beam*7907 + 20001) % (2**63)
                noise_rng = np.random.default_rng(seed)
                blk.real[j] = noise_rng.standard_normal((n_s, n_c), dtype=np.float32)
                blk.imag[j] = noise_rng.standard_normal((n_s, n_c), dtype=np.float32)
        else:
            blk.real = rng.standard_normal((m, n_s, n_c), dtype=np.float32)
            blk.imag = rng.standard_normal((m, n_s, n_c), dtype=np.float32)
        # per-beam noise power, so a jammer's antenna pattern shapes the floor
        # instead of dimming the entire picture by one global factor
        blk *= real(np.sqrt(pn_beam[a:bb, None, None] / 2.0))
        for i in range(a, bb):
            for fb, fd, amp in by_beam.get(i, ()):
                blk[i - a] += echo_plane(fb, fd, amp)
        if keep_cube:
            cube[a:bb] = blk
        blk *= win
        spec = np.fft.fftshift(np.fft.fft2(blk, axes=(1, 2)), axes=2)
        power = np.abs(spec)          # complex64 -> float32, no imaginary cast
        np.square(power, out=power)
        # Range-angle is per-beam by construction, so the maximum is taken along
        # Doppler only. The RD image no longer lives here: a maximum over ~800
        # beams is an extreme-value statistic, and taking it cell by cell was
        # what flattened the speckle into 1-2 dB of dead grey.
        ra_map[a:bb] = power.max(axis=2)

    axis_r = np.fft.fftfreq(n_s, 1.0 / fs) * C_LIGHT * t_chirp / (2.0 * b)
    # Export the positive-frequency band used by this real-ADC-range convention.
    # Internally the synthetic noise/tone buffer is complex; its negative bins
    # are not a literal conjugate mirror. No claim of a full IQ receiver is made.
    _half = max(1, n_s // 2)
    axis_r = axis_r[:_half]
    axis_d = np.fft.fftshift(np.fft.fftfreq(n_c, pri)) * lam / 2.0

    # ---- the one-aperture range-Doppler image ------------------------------
    # Every echo of this CPI -- target centres, JEM teeth, clutter, ECM copies
    # -- reaches this aperture through its own direction's two-way pattern and
    # is summed coherently, then one complex noise field is added. Cost is a
    # single (n_samples x n_chirps) transform, i.e. one beam's worth of work.
    rd_map = display_rd_map(echoes, ant_disp, pn_disp, n_s, n_c, fs, pri,
                            win_r.ravel(), win_d.ravel(), ridge_amp=ridge_amp,
                            dc_leak_db=rc.get("dc_leak_db"), seed=disp_seed)

    processing = dict(detector="analytic", detection_basis="expected radar equation SNR",
                      noise_power_input=pn,
                      limitations=["analytic gate, not an observed-RD detector",
                                   "frozen CPI geometry; no range migration or FMCW range-Doppler coupling"])
    if measured_detector:
        # One display channel means the RD cell is a single exponential draw, not
        # a maximum over 800 of them: keeping n_beams here would set the gate
        # 10*log10(800) = 29 dB too high and hide every target that is not
        # enormous.
        processing = fft_noise_statistics(
            pn, win_r.ravel(), win_d.ravel(), 1,
            float(rc.get("pfa") if rc.get("pfa") is not None else 1e-6))
        processing["noise_power_input"] = pn
        processing["expected_snr_basis"] = (
            "on-bin tone through the display aperture: Hann coherent/window-noise "
            "gains plus the two-way pattern at the echo's own angle")
        # the threshold is computed from thermal noise on purpose: the radar does
        # not know it is being jammed, which is exactly why a noise jammer turns
        # into false alarms in the RD image instead of being quietly compensated
        snr_db = (10.0*np.log10(np.maximum(pr, 1e-30)/pn_beam) +
                  processing["window_processing_gain_db"])
        processing["display_channel"] = ("one aperture on the sector centre, every echo "
                                         "weighted by its two-way pattern and summed coherently")
        processing["association_rule"] = (
            "post-detection: nearest echo within two range/Doppler bins, expected target "
            "power > half measured power")
        processing["max_detections"] = int(rc["max_detections"])
        cells = rd_peak_cells(rd_map, processing["threshold_power"], rc["max_detections"])
        # Association set = the echo list itself. A single display channel has no
        # winning beam per cell, so the beam a peak came from cannot be read off
        # the image; every echo already carries its own range/Doppler bin and the
        # object it belongs to.
        assoc = []
        for _i in source_echoes:
            if obj_of_beam[_i] < 0:
                continue
            assoc.append((int(np.argmin(np.abs(axis_r - dist[_i]))),
                          int(np.argmin(np.abs(axis_d - vr[_i]))), int(_i)))
        # Every echo that belongs to a body, so the expected power at a cell can
        # be the COHERENT sum of everything that body contributes there -- its
        # scattering centres and its JEM comb, with the phases the synthesis
        # used. One echo's own power would under-predict a body spread over
        # several range cells, and the association would then reject a real
        # target for being brighter than the single echo predicts.
        obj_echoes = {}
        for _b, _fb, _fd, _a in echoes:
            _o = int(obj_of_beam[_b])
            if _o >= 0:
                obj_echoes.setdefault(_o, []).append((_fb, _fd, _a, _b))
        det = []
        w_r, w_d = win_r.ravel(), win_d.ravel()
        n_f = np.arange(n_s)
        c_f = np.arange(n_c)
        for ir, iv in cells:
            measured_range, measured_v = float(axis_r[ir]), float(axis_d[iv])
            observed_power = float(rd_map[ir, iv])
            oid, beam, signal_power = -1, -1, 0.0
            # Detection above is entirely image-based. Associate afterwards; a
            # noise peak that happens to sit near a weak target must not
            # automatically acquire its object ID.
            cand = []
            for _ir, _iv, _i in assoc:
                dr = abs(ir - _ir)
                dj = abs(iv - _iv)
                dj = min(dj, n_c - dj)
                if dr <= 2 and dj <= 2:
                    cand.append((dr*dr + dj*dj, int(_i)))
            if cand:
                # The guard has to compare the peak with what EVERY body near
                # this cell explains, not with one echo's share: several bodies
                # can share a cell (the studio scene puts three aircraft in
                # one), so their coherent sum legitimately exceeds the largest
                # single expectation by up to the number of bodies. A noise
                # peak near a faint target has no such explanation and stays
                # unassociated -- that is what this bound is for.
                totals = []
                for _o in {int(obj_of_beam[_i]) for _, _i in cand}:
                    acc = 0j
                    for _fb, _fd, _a, _b in obj_echoes.get(_o, ()):
                        ar = np.sum(w_r*np.exp(2j*np.pi*(_fb/fs-ir/n_s)*n_f))
                        ad = np.sum(w_d*np.exp(2j*np.pi*(_fd*pri-(iv-n_c//2)/n_c)*c_f))
                        acc += _a*float(ant_disp[_b])*ar*ad
                    totals.append((float(acc.real**2 + acc.imag**2), int(_o)))
                expected = sum(_v for _v, _ in totals)
                if totals and expected > 0.25*observed_power:
                    signal_power, oid = max(totals)
                    beam = min((c_ for c_ in cand if int(obj_of_beam[c_[1]]) == oid),
                               key=lambda c_: c_[0])[1]
            associated = oid >= 0
            if associated:
                xyz = pts[beam]
                az, el = float(angles[beam, 0]), float(angles[beam, 1])
            else:
                # A single-channel map measures range and Doppler, not angle: an
                # unassociated peak is placed on the display boresight at its
                # measured range instead of inventing a beam.
                xyz = p + forward*measured_range
                az, el = 0.0, 0.0
            item = dict(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]),
                        range=measured_range, v_r=measured_v,
                        range_bin=ir, doppler_bin=iv, beam_index=beam,
                        az=az, el=el,
                        snr_db=float(10.0*np.log10(observed_power/processing["single_beam_fft_noise_power"])),
                        power=observed_power, expected_target_power=signal_power,
                        obj=oid, detection_basis="measured_rd_peak",
                        associated_to_truth=associated)
            if associated:
                item["range_true"] = float(dist[beam])
                item["v_r_true"] = float(vr_true[beam])
                item["xyz_basis"] = "associated_truth_surface"
                item["antenna_two_way_db"] = float(20.0*np.log10(max(float(ant1[beam]), 1e-30)))
            else:
                item["xyz_basis"] = "measured_range_and_display_boresight"
            det.append(item)
        processing["threshold_crossing_peak_count"] = len(cells)
        processing["unassociated_peak_count"] = sum(d["obj"] < 0 for d in det)
        processing["pfa_scope"] = "per RD cell before local-maximum selection, independent thermal noise only"

    return dict(
        cube=cube, rd_map=rd_map, range_angle=ra_map,
        range_axis=axis_r, doppler_axis=axis_d,
        # Ranges of real targets that the range window cut off; empty means the
        # window was wide enough for everything the scene put in the air.
        targets_beyond_range=sorted(set(beyond_range)),
        range_window_m=float(r_max),
        detections=det, sigma=sig, snr_db=snr_db, beams=angles,
        radar_cfg=rc, pn=pn, cube_chunked=(cube is None), processing=processing,
        # per-beam noise floor the cube was actually synthesised with: without a
        # scene jammer it is pn in every beam, so nothing downstream changes
        pn_beam=pn_beam,
        # None unless the scene carried an "ecm" block: callers that predate
        # countermeasures see exactly the dict they always saw, plus this key.
        ecm=ecm_report,
    )


def _face_normals(w):
    t = w.verts[w.faces]
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    ln[ln == 0] = 1.0
    return n / ln


def _face_areas(w):
    t = w.verts[w.faces]
    return 0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)


def to_db(x, floor=1e-12):
    return 10.0 * np.log10(np.maximum(x, floor))