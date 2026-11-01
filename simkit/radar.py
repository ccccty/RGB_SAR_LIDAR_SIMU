"""FMCW radar forward model: ray casting, PO RCS, range-Doppler synthesis.

Chain: beam grid -> ray cast (multi-bounce specular) -> physical-optics RCS per
facet with Kirchhoff roughness -> radar equation -> beat-signal synthesis ->
range/Doppler FFT. Outputs a complex radar cube, RD / range-angle maps,
a detection (point cloud) list and per-detection object labels.

Approximations are deliberate: incoherent multi-bounce, no shadowing between
bounces, flat-facet PO (no edge diffraction / creeping waves), no MIMO virtual
array, no range migration. Sufficient for detector training data; not for RCS
certification.
"""
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
)


def cfg_get(cfg, key):
    r = cfg.get("sensors", {}).get("radar", {})
    return float(r.get(key, DEFAULTS[key])) if not isinstance(DEFAULTS[key], list) else r.get(key, DEFAULTS[key])


def beam_grid(cam, n_az, n_el, az_span, el_span):
    p, r, u, f = sensor_basis(cam["pos"], cam["look_at"], cam.get("up", (0, 0, 1)),
                              cam.get("roll_deg", 0.0))
    az = np.linspace(-az_span / 2, az_span / 2, n_az)
    el = np.linspace(-el_span / 2, el_span / 2, n_el)
    dirs, angles = [], []
    for e in el:
        for a in az:
            ta, te = np.tan(np.radians(a)), np.tan(np.radians(e))
            d = f + ta * r + te * u
            dirs.append(d / np.linalg.norm(d))
            angles.append((a, e))
    return p, np.array(dirs), np.array(angles)


def facet_sigma(areas, cos_inc, rho, rough_m, lam, footprint=None,
                spec_scale=1.0, diff_gamma=0.06):
    """Physical-optics specular term + rough-surface diffuse term.

    areas is clipped to the illuminated beam footprint: a facet far larger than
    the beam must not be counted as a whole mirror. spec_scale calibrates the
    (systematically optimistic) flat-facet coherent term only; the diffuse term
    keeps its physical gamma0 scaling.
    """
    a = areas if footprint is None else np.minimum(areas, footprint)
    coh = np.exp(-((4.0 * np.pi * rough_m * cos_inc / lam) ** 2))
    spec = 4.0 * np.pi * a ** 2 * cos_inc ** 2 / lam ** 2 * rho ** 2 * coh
    diff = a * rho * diff_gamma * np.maximum(cos_inc, 0.0) * (1.0 - coh)
    return spec * spec_scale + diff


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
    cam = rc
    p, dirs, angles = beam_grid(cam, int(rc["n_az"]), int(rc["n_el"]),
                                float(rc["az_span_deg"]), float(rc["el_span_deg"]))
    lam = C_LIGHT / (float(rc["fc_ghz"]) * 1e9)

    bvh = world.bvh
    hit = bvh.ray_cast(p, dirs)
    tri, ok = hit["tri"], hit["hit"]

    mat_names = sorted(matlib.MATERIALS)
    rho_tab = np.array([matlib.MATERIALS[n]["rf_rho"] for n in mat_names])
    rough_tab = np.array([matlib.MATERIALS[n]["rough_m"] for n in mat_names])

    areas = world.face_areas() if hasattr(world, "face_areas") else _face_areas(world)
    normals = _face_normals(world)

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
            s_tot[gnd] *= 10.0 ** (float(rc.get("ground_clutter_db", -30.0)) / 10.0)
        sig[idx] = s_tot

    dist = hit["t"].copy()
    dist[~ok] = np.inf

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
        # A single centre ray is too strict: in the cluttered scene a 2 m fence
        # post can cross the boresight while the rest of the rover is in plain
        # view, and one blocked ray then erased 17 of 20 frames. Probe the
        # centre plus the eight corners and accept a partial view.
        corners = np.asarray(o["corners"], float)
        probes = np.vstack([c[None, :], c + (corners - c) * 0.7])
        pdirs = probes - p
        pdirs /= np.linalg.norm(pdirs, axis=1, keepdims=True)
        ph = bvh.ray_cast(p, pdirs)
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
        sig[b] += s_obj
        obj_of_beam[b] = o["id"]
        if R < dist[b]:
            dist[b] = R
            pts[b] = c

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
    pg_db = float(rc.get("processing_gain_db",
                         10.0 * np.log10(max(int(rc["n_samples"]) * int(rc["n_chirps"]), 1))))
    snr_db = 10.0 * np.log10(np.maximum(pr, 1e-30) / pn) + pg_db

    # radial velocity from object motion
    # Doppler is the *closing* rate, so it is the target velocity relative to the
    # radar. Using the target's own velocity alone put a 200 m/s closing aircraft
    # at 1.4 m/s -- right on the zero-Doppler clutter ridge and inside the MTI
    # notch, i.e. an RD map whose velocity axis means nothing.
    v_rad = np.asarray(rc.get("vel", [0.0, 0.0, 0.0]), float)
    vr = np.zeros(len(dirs))
    for o in world.objs:
        m = obj_of_beam == o["id"]
        if not m.any():
            continue
        u = dirs[m]
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

    # moving-target indication: a zero-Doppler gate rejects stationary clutter
    # (ground, sea surface). It is the cheapest stand-in for real MTI filtering.
    mti_v = float(rc.get("mti_min_vr_mps", 0.0))
    det = []
    echoes = []          # (beam, beat_freq, doppler_freq, amplitude)
    # `idx` is the geometry-hit set; a sub-beam target injected above never
    # produced a hit, yet it is exactly the echo a seeker cares about.
    det_idx = np.where(ok | (obj_of_beam >= 0))[0]
    for i in det_idx:
        if snr_db[i] < float(rc["det_snr_db"]):
            continue
        if mti_v > 0.0 and abs(float(vr_true[i])) < mti_v:
            continue
        r = float(dist[i])
        if r < 0.5 or r > r_max:
            continue
        fb = 2.0 * b * r / (C_LIGHT * t_chirp)
        fb = min(fb, fs / 2.0 * 0.99)
        fd = 2.0 * float(vr[i]) / lam
        amp = np.sqrt(float(pr[i])) * np.exp(1j * rng.uniform(0, 2 * np.pi))
        echoes.append((int(i), fb, fd, amp))
        det.append(dict(x=float(pts[i, 0]), y=float(pts[i, 1]), z=float(pts[i, 2]),
                        range=r, az=float(angles[i, 0]), el=float(angles[i, 1]),
                        v_r=float(vr[i]), snr_db=float(snr_db[i]),
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
        echoes.append((i, fb, rng.uniform(-1, 1) * 5e3, amp))

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

    rd_map = np.zeros((n_s, n_c))
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
        blk.real = rng.standard_normal((m, n_s, n_c), dtype=np.float32)
        blk.imag = rng.standard_normal((m, n_s, n_c), dtype=np.float32)
        blk *= real(np.sqrt(pn / 2.0))
        for i in range(a, bb):
            for fb, fd, amp in by_beam.get(i, ()):
                blk[i - a] += echo_plane(fb, fd, amp)
        if keep_cube:
            cube[a:bb] = blk
        blk *= win
        spec = np.fft.fftshift(np.fft.fft2(blk, axes=(1, 2)), axes=2)
        power = np.abs(spec)          # complex64 -> float32, no imaginary cast
        np.square(power, out=power)
        # An RD map is one beam's picture. Summing all beams incoherently adds
        # 10*log10(n_beams) = +31 dB to the noise floor while a point target is
        # in one or two beams, so a target the detector calls 30 dB strong (the
        # rover in the low air-to-ground scene) sat at 1/255 in the shipped image
        # and its RD box landed on a black cell, while the ground clutter that
        # fills every beam took the bright cells. Take the per-cell maximum over
        # beams instead -- the same statistic ra_map already uses along Doppler.
        np.maximum(rd_map, power.max(axis=0), out=rd_map)
        ra_map[a:bb] = power.max(axis=2)

    axis_r = np.fft.fftfreq(n_s, 1.0 / fs) * C_LIGHT * t_chirp / (2.0 * b)
    # A real beat signal has a conjugate-symmetric spectrum, so the second half of
    # the range axis is a mirror, not new information. Keeping it drew every echo
    # twice (once at +R, once at -R) and left the range axis non-monotonic.
    _half = max(1, n_s // 2)
    rd_map = rd_map[:_half]
    axis_r = axis_r[:_half]
    axis_d = np.fft.fftshift(np.fft.fftfreq(n_c, pri)) * lam / 2.0

    return dict(
        cube=cube, rd_map=rd_map, range_angle=ra_map,
        range_axis=axis_r, doppler_axis=axis_d,
        detections=det, sigma=sig, snr_db=snr_db, beams=angles,
        radar_cfg=rc, pn=pn, cube_chunked=(cube is None),
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