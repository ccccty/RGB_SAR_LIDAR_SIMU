"""Thermal infrared forward model: Planck radiance, atmosphere, sensor, raster."""
import numpy as np

from . import materials as matlib
from . import spectra as speclib
from .scene import sensor_basis
from .weather import IR_BANDS, band_name_of

H_PLANCK = 6.62607015e-34
C_LIGHT = 2.99792458e8
K_BOLTZ = 1.380649e-23


def planck_band(T, lam1_um, lam2_um, n=48):
    """In-band spectral radiance W/(m^2 sr) for temperature(s) T in Kelvin."""
    lam = np.linspace(lam1_um, lam2_um, n)[None, :] * 1e-6
    c1 = 2.0 * H_PLANCK * C_LIGHT ** 2
    c2 = H_PLANCK * C_LIGHT / K_BOLTZ
    T = np.asarray(T, float).reshape(-1, 1)
    with np.errstate(over="ignore", invalid="ignore"):
        B = c1 / lam ** 5 / (np.expm1(c2 / (lam * T)))
    B = np.nan_to_num(B, nan=0.0, posinf=0.0)
    lam_step = (lam2_um - lam1_um) / (n - 1) * 1e-6
    return B.sum(axis=1) * lam_step


def transmittance(range_m, tau_per_km):
    return np.exp(-tau_per_km * np.asarray(range_m, float) / 1000.0)


def rasterize(world, cam, face_value, background=0.0):
    """Z-buffer flat shading of per-face scalar values."""
    W, H = int(cam["width"]), int(cam["height"])
    p, r, u, f = sensor_basis(cam["pos"], cam["look_at"], cam.get("up", (0, 0, 1)))
    fx = (W / 2.0) / np.tan(np.radians(cam.get("fov_deg", 30.0)) / 2.0)
    fy = fx

    rel = world.verts - p
    xc = rel @ r
    yc = rel @ u
    zc = rel @ f
    zs = np.where(zc > 1e-6, zc, 1e-6)
    su = W / 2.0 + fx * xc / zs
    sv = H / 2.0 - fy * yc / zs

    img = np.full((H, W), np.nan, float)
    depth = np.full((H, W), np.inf, float)
    pid = np.full((H, W), -1, np.int32)
    tri = world.faces
    zt = zc[tri]
    ok = (zt > 0.05).all(axis=1)

    order = np.where(ok)[0]
    # draw far faces first is irrelevant (z-test), iterate all
    for i in order:
        uu = su[tri[i]]
        vv = sv[tri[i]]
        x0 = max(int(np.floor(uu.min())), 0)
        x1 = min(int(np.ceil(uu.max())) + 1, W)
        y0 = max(int(np.floor(vv.min())), 0)
        y1 = min(int(np.ceil(vv.max())) + 1, H)
        if x1 <= x0 or y1 <= y0:
            continue
        xs = np.arange(x0, x1) + 0.5
        ys = np.arange(y0, y1) + 0.5
        gx, gy = np.meshgrid(xs, ys)
        denom = ((vv[1] - vv[2]) * (uu[0] - uu[2]) + (uu[2] - uu[1]) * (vv[0] - vv[2]))
        if abs(denom) < 1e-9:
            continue
        l0 = ((vv[1] - vv[2]) * (gx - uu[2]) + (uu[2] - uu[1]) * (gy - vv[2])) / denom
        l1 = ((vv[2] - vv[0]) * (gx - uu[2]) + (uu[0] - uu[2]) * (gy - vv[2])) / denom
        l2 = 1.0 - l0 - l1
        inside = (l0 >= -1e-6) & (l1 >= -1e-6) & (l2 >= -1e-6)
        if not inside.any():
            continue
        invz = l0 / zt[i, 0] + l1 / zt[i, 1] + l2 / zt[i, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            zpix = np.where(np.abs(invz) > 1e-12, 1.0 / invz, np.inf)
        sub_d = depth[y0:y1, x0:x1]
        upd = inside & (zpix < sub_d)
        if not upd.any():
            continue
        sub_d[upd] = zpix[upd]
        img[y0:y1, x0:x1][upd] = face_value[i]
        pid[y0:y1, x0:x1][upd] = world.face_obj[i]

    img = np.where(np.isnan(img), background, img)
    return img, depth, pid


def gaussian_blur(img, sigma):
    if sigma <= 0:
        return img
    rad = max(1, int(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-rad, rad + 1) / sigma) ** 2)
    k /= k.sum()
    out = img
    pad = np.pad(out, ((0, 0), (rad, rad)), mode="edge")
    tmp = np.zeros_like(out)
    for i, w in enumerate(k):
        tmp += w * pad[:, i:i + out.shape[1]]
    pad = np.pad(tmp, ((rad, rad), (0, 0)), mode="edge")
    out = np.zeros_like(tmp)
    for i, w in enumerate(k):
        out += w * pad[i:i + tmp.shape[0], :]
    return out


def radiance_to_temp(img, band, t_lo=200.0, t_hi=450.0, n=512):
    """Invert the Planck band integral: radiance -> apparent temperature (K)."""
    tg = np.linspace(t_lo, t_hi, n)
    lg = planck_band(tg, band[0], band[1])
    order = np.argsort(lg)
    return np.interp(img, lg[order], tg[order])


def _equalize(img, levels=512):
    """Histogram equalisation: what a thermal camera's AGC does for display."""
    lo, hi = np.percentile(img, [0.2, 99.8])
    x = np.clip((img - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
    hist, edges = np.histogram(x, bins=levels, range=(0.0, 1.0))
    cdf = np.cumsum(hist).astype(np.float64)
    cdf /= max(cdf[-1], 1.0)
    return np.interp(x, edges[:-1], cdf)


def render(world, cfg, temps_k, cam=None):
    """Full IR frame: radiance -> sensor -> digital counts."""
    cam = cam or cfg["sensors"]["ir_camera"]
    ircfg = cfg.get("ir", {})
    band = ircfg.get("band_um") or IR_BANDS.get(ircfg.get("band"), [8.0, 12.0])
    bname = ircfg.get("band") or band_name_of(band)
    tau = ircfg.get("atm_tau_per_km")
    if tau is None:
        w = cfg.get("_weather")
        if w:
            tau = w["tau_ir_per_km"].get(bname or band_name_of(band) or "lwir", 0.06)
        else:
            tau = 0.06
    tau = float(tau)
    netd = ircfg.get("netd_mk", 30.0) / 1000.0
    blur = ircfg.get("psf_px", 1.2)
    gain = ircfg.get("gain", 1.0)

    mat_names = sorted(matlib.MATERIALS)
    # emissivity: a measured spectra curve wins over the tabulated band value
    def _eps(name):
        s = matlib.MATERIALS[name]
        if s.get("spectra"):
            v = speclib.emissivity_for(s, band)
            if v is not None:
                return v
        return matlib.eps(s, bname)

    eps_face = np.array([_eps(n) for n in mat_names])[world.face_mat]

    p, r, u, f = sensor_basis(cam["pos"], cam["look_at"], cam.get("up", (0, 0, 1)))
    cent = world.verts[world.faces].mean(axis=1)
    rng = np.linalg.norm(cent - p, axis=1)

    # per-face temperature texture: real surfaces are never perfectly uniform
    tex = float(ircfg.get("texture_sigma_c", 0.6))
    if tex > 0:
        rng_tex = np.random.default_rng(int(cfg.get("_frame_seed", 0)) * 131 + 7)
        temps_k = temps_k + rng_tex.normal(0.0, tex, temps_k.shape)
    L_self = eps_face * planck_band(temps_k, band[0], band[1])
    t_env = cfg.get("air", {}).get("temp_c", 20.0) + 273.15
    L_env = planck_band(np.array([t_env]), band[0], band[1])[0]
    L = transmittance(rng, tau) * (L_self + (1.0 - eps_face) * L_env)

    # sky background: cold Swinbank sky (weather-aware), not a fixed fraction
    w = cfg.get("_weather")
    t_sky = float(w["t_sky_k"]) if w else 0.0552 * t_env ** 1.5
    L_sky = planck_band(np.array([t_sky]), band[0], band[1])[0]
    img, depth, pid = rasterize(world, cam, L, background=L_sky)
    img = gaussian_blur(img, blur)

    # NETD: noise expressed in radiance via dL/dT
    dT = 3.0
    dLdT = (planck_band(temps_k + dT, band[0], band[1]) -
            planck_band(temps_k, band[0], band[1])).mean() / dT
    rng_noise = np.random.default_rng(int(cfg.get("_frame_seed", 0)))
    img = img + rng_noise.normal(0.0, dLdT * netd, img.shape)

    # 16-bit counts: one fixed physical window (200..450 K) so every frame and
    # every scene shares the same radiometric scale.
    t_img = radiance_to_temp(img, band)
    t0, t1 = 200.0, 450.0
    # quantise to detector counts BEFORE non-uniformity, so NUC acts on the
    # digital signal (as in a real readout) instead of on a float ideal image
    counts = np.round(np.clip((t_img - t0) / (t1 - t0), 0, 1) * 65535.0 * gain)

    # sensor non-uniformity (NUC residuals). Fixed per detector, not per frame:
    # the seed comes from the config, so the pattern stays constant across the
    # whole dataset -- that is what makes it "fixed pattern" noise.
    nuc = ircfg.get("nuc") or {}
    if nuc.get("enabled", False):
        nrng = np.random.default_rng(int(nuc.get("seed", 12345)))
        prnu = 1.0 + nrng.normal(0.0, float(nuc.get("prnu", 0.01)), counts.shape)
        dsnu = nrng.normal(0.0, float(nuc.get("dsnu_counts", 30.0)), counts.shape)
        counts = counts * prnu + dsnu
        t_img = t_img + dsnu * ((t1 - t0) / 65535.0)
        col = float(nuc.get("column_fpn_counts", 0.0))
        if col > 0.0:
            c = nrng.normal(0.0, col, (1, counts.shape[1]))
            counts = counts + c
            t_img = t_img + c * ((t1 - t0) / 65535.0)
        dead = float(nuc.get("dead_px", 0.0))
        if dead > 0.0:
            m = nrng.random(counts.shape) < dead
            counts[m] = 0.0
            t_img[m] = t0
        counts = np.clip(counts, 0.0, 65535.0)
    lo, hi = np.percentile(img, [0.5, 99.5])
    norm = np.clip((img - lo) / max(hi - lo, 1e-12), 0, 1) * gain
    mode = ircfg.get("preview_mode", "temp")
    if mode == "eq":
        prev = _equalize(img)
    elif mode == "linear":
        prev = norm
    else:
        t_img = radiance_to_temp(img, band)
        win = ircfg.get("preview_window_c")
        if win:
            tlo, thi = win[0] + 273.15, win[1] + 273.15
        else:
            med = float(np.median(t_img))
            half = min(20.0, max(8.0, (np.percentile(t_img, 98) - np.percentile(t_img, 2)) * 0.6))
            tlo, thi = med - half, med + half
        prev = np.clip((t_img - tlo) / max(thi - tlo, 1e-6), 0, 1)
    return dict(
        radiance=img,
        counts_u16=counts.astype(np.uint16),
        preview=(np.clip(prev, 0, 1) * 255).astype(np.uint8),
        apparent_temp_k=t_img,
        depth=depth,
        obj_id=pid,
        temperature_k=temps_k,
    )
