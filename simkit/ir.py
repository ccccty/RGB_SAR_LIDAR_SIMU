"""Thermal infrared forward model: Planck radiance, atmosphere, sensor, raster."""
import numpy as np

from . import materials as matlib
from . import spectra as speclib
from . import thermal, atmosphere
from .scene import sensor_basis
from .weather import IR_BANDS, band_name_of

H_PLANCK = 6.62607015e-34
C_LIGHT = 2.99792458e8
K_BOLTZ = 1.380649e-23


def planck_band(T, lam1_um, lam2_um, n=48):
    """In-band spectral radiance W/(m^2 sr) for temperature(s) T in Kelvin."""
    if n < 2 or not 0 < lam1_um < lam2_um:
        raise ValueError("Planck integration needs >=2 samples and 0 < lambda1 < lambda2")
    lam = np.linspace(lam1_um, lam2_um, n)[None, :] * 1e-6
    c1 = 2.0 * H_PLANCK * C_LIGHT ** 2
    c2 = H_PLANCK * C_LIGHT / K_BOLTZ
    T = np.asarray(T, float).reshape(-1, 1)
    with np.errstate(over="ignore", invalid="ignore"):
        B = c1 / lam ** 5 / (np.expm1(c2 / (lam * T)))
    B = np.nan_to_num(B, nan=0.0, posinf=0.0)
    # Endpoint rectangles over-counted the band radiance by about 2% at the
    # default 48 samples. A trapezoidal spectral integral has the right limits.
    return np.trapezoid(B, lam[0], axis=1)


def transmittance(range_m, tau_per_km):
    return np.exp(-tau_per_km * np.asarray(range_m, float) / 1000.0)


def _clip_near_triangle(vertices, near=0.05):
    """Clip a camera-space triangle to Z>=near, retaining its surface geometry."""
    polygon = []
    previous = vertices[-1]
    previous_inside = previous[2] >= near
    for current in vertices:
        current_inside = current[2] >= near
        if current_inside != previous_inside:
            fraction = (near-previous[2])/(current[2]-previous[2])
            polygon.append(previous+(current-previous)*fraction)
        if current_inside:
            polygon.append(current)
        previous, previous_inside = current, current_inside
    return [np.array([polygon[0], polygon[index], polygon[index+1]])
            for index in range(1, len(polygon)-1)]


def rasterize(world, cam, face_value, background=0.0, subpixel_mode="legacy"):
    """Z-buffer flat shading of per-face scalar values."""
    W, H = int(cam["width"]), int(cam["height"])
    p, r, u, f = sensor_basis(cam["pos"], cam["look_at"], cam.get("up", (0, 0, 1)),
                              cam.get("roll_deg", 0.0))
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

    def projected_triangles():
        for i in np.where((zt >= 0.05).any(axis=1))[0]:
            if (zt[i] >= 0.05).all():
                yield i, su[tri[i]], sv[tri[i]], zt[i]
            else:
                # Previously throwing away the whole face left holes in the
                # ground whenever a triangle straddled the camera plane.
                vertices = np.column_stack([xc[tri[i]], yc[tri[i]], zc[tri[i]]])
                for piece in _clip_near_triangle(vertices):
                    yield (i, W/2+fx*piece[:, 0]/piece[:, 2],
                           H/2-fy*piece[:, 1]/piece[:, 2], piece[:, 2])

    # draw far faces first is irrelevant (z-test), iterate all
    for i, uu, vv, zi in projected_triangles():
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
        invz = l0 / zi[0] + l1 / zi[1] + l2 / zi[2]
        with np.errstate(divide="ignore", invalid="ignore"):
            zpix = np.where(np.abs(invz) > 1e-12, 1.0 / invz, np.inf)
        sub_d = depth[y0:y1, x0:x1]
        upd = inside & (zpix < sub_d)
        if not upd.any():
            continue
        sub_d[upd] = zpix[upd]
        img[y0:y1, x0:x1][upd] = face_value[i]
        pid[y0:y1, x0:x1][upd] = world.face_obj[i]

    # An object smaller than a pixel can fall between pixel centres and leave no
    # trace at all: at 8 km the aircraft is 2.6 px across in the search lens and
    # the triangle test drew exactly zero pixels of it, so the frame carried a
    # label but no target. A real focal plane still returns one pixel of signal
    # there, so stamp the object's own mean radiance into the pixel its centre
    # projects to (a point-target response at the sampling limit).
    if subpixel_mode not in ("legacy", "none", "coverage"):
        raise ValueError("ir.subpixel_mode must be legacy, none or coverage")
    present = set(np.unique(pid).tolist())
    for o in getattr(world, "objs", []):
        if subpixel_mode != "legacy":
            break
        if not o.get("label", True) or o["id"] in present:
            continue
        rel = np.asarray(o["corners"], float).mean(axis=0) - p
        z = float(rel @ f)
        if z <= 0.05:
            continue
        xu = int(W / 2.0 + fx * float(rel @ r) / z)
        yv = int(H / 2.0 - fy * float(rel @ u) / z)
        if not (0 <= xu < W and 0 <= yv < H):
            continue
        if depth[yv, xu] < z:
            continue                     # something really is in front of it
        sel = world.face_obj == o["id"]
        val = float(np.mean(face_value[sel])) if sel.any() else background
        img[yv, xu] = val
        pid[yv, xu] = o["id"]
        depth[yv, xu] = z

    img = np.where(np.isnan(img), background, img)
    return img, depth, pid


def surface_temperature_texture(world, cfg):
    """Smooth object-local variation, invariant to frame time, pose and noise.

    Detector temporal noise belongs later in the sensor chain. Independent
    per-face/per-frame temperatures invented flickering heat sources and made
    the same triangle differ between MWIR and LWIR.
    """
    import zlib
    spec = cfg.get("ir") or {}
    amplitude = float(spec.get("texture_sigma_c", 0.6))
    if amplitude <= 0:
        return np.zeros(len(world.faces))
    if spec.get("texture_mode", "surface") == "legacy_per_frame":
        rng = np.random.default_rng(int(cfg.get("_frame_seed", 0))*131+7)
        return rng.normal(0.0, amplitude, len(world.faces))
    if spec.get("texture_mode", "surface") != "surface":
        raise ValueError("ir.texture_mode must be surface or legacy_per_frame")
    local = getattr(world, "face_local_centers", None)
    if local is None:
        local = thermal.face_centers(world).copy()
        for obj in world.objs:
            selected = world.face_obj == obj["id"]
            theta = np.radians(float(obj["yaw"]))
            rotation = np.array([[np.cos(theta), -np.sin(theta), 0],
                                 [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
            local[selected] = (local[selected]-obj["pos"]) @ rotation
    scale = float(spec.get("texture_scale_m", 1.0))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("ir.texture_scale_m must be positive")
    result = np.zeros(len(world.faces))
    names = {-1: "background", **{obj["id"]: obj["name"] for obj in world.objs}}
    for identifier in np.unique(world.face_obj):
        selected = world.face_obj == identifier
        seed = int(spec.get("texture_seed", cfg.get("seed", 0)))
        seed = (seed+zlib.crc32(names.get(int(identifier), str(identifier)).encode("utf-8"))) % (2**32)
        rng = np.random.default_rng(seed)
        directions = rng.normal(size=(8, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        phases = rng.uniform(0, 2*np.pi, 8)
        wavelength = float(spec.get("ground_texture_scale_m", 100.0)) if identifier < 0 else scale
        variation = np.sin(local[selected] @ directions.T * (2*np.pi/wavelength) + phases)
        result[selected] = amplitude*np.sqrt(2/8)*variation.sum(axis=1)
    return result


def _pixel_atmosphere(source, depth, cam, cfg, band, bname, legacy_tau, background):
    """Propagate radiance per pixel, preserving depth's forward-camera-Z units."""
    H, W = source.shape
    p, right, up, forward = sensor_basis(cam["pos"], cam["look_at"],
                                         cam.get("up", (0, 0, 1)), cam.get("roll_deg", 0))
    fx = W/(2*np.tan(np.radians(cam.get("fov_deg", 30))/2))
    x = (np.arange(W)+0.5-W/2)/fx
    y = -(np.arange(H)+0.5-H/2)/fx
    directions = forward + x[None, :, None]*right + y[:, None, None]*up
    hit = np.isfinite(depth)
    if not atmosphere.enabled(cfg):
        distance = np.where(hit, depth, 0)*np.linalg.norm(directions, axis=2)
        tau = transmittance(distance, legacy_tau)
        env = float(cfg.get("air", {}).get("temp_c", 20))+273.15
        path = (1-tau)*planck_band([env], *band)[0]
        return np.where(hit, source*tau+path, background)
    flat_d = directions.reshape(-1, 3)
    flat_depth = depth.ravel()
    flat_source = source.ravel()
    output = np.empty_like(flat_source)
    for start in range(0, len(flat_d), 32768):
        stop = min(start+32768, len(flat_d))
        d = flat_d[start:stop]
        z = flat_depth[start:stop]
        valid = np.isfinite(z)
        endpoints = atmosphere.sky_endpoints(p, d, cfg)
        endpoints[valid] = p+d[valid]*z[valid, None]
        tau = atmosphere.path_transmittance(p, endpoints, cfg, bname or "lwir")
        path = atmosphere.path_emission(p, endpoints, cfg, band, bname or "lwir")
        output[start:stop] = np.where(valid, flat_source[start:stop]*tau, 0.0)+path
    return output.reshape(H, W)


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

    p, r, u, f = sensor_basis(cam["pos"], cam["look_at"], cam.get("up", (0, 0, 1)),
                              cam.get("roll_deg", 0.0))
    cent = world.verts[world.faces].mean(axis=1)
    rng = np.linalg.norm(cent - p, axis=1)

    # Stable surface variation is shared by both IR bands and across frames.
    temps_k = temps_k + surface_temperature_texture(world, cfg)
    w = cfg.get("_weather")
    t_env = cfg.get("air", {}).get("temp_c", 20.0) + 273.15
    b_air = planck_band(np.array([t_env]), band[0], band[1])[0]
    b_self = planck_band(temps_k, band[0], band[1])
    L_self = eps_face * b_self

    # Sky radiance is NOT a blackbody at the broadband sky temperature. In 3-5 um
    # the sky is almost transparent (kappa 0.04-0.12) while in 8-12 um it is
    # nearly opaque (0.40-0.55); using one broadband temperature in the band
    # integral made MWIR sky about five times too bright and erased the contrast
    # a seeker actually detects.
    if w:
        t_sky = float(w["t_sky_k"])
        kap = float((w.get("sky_kappa") or {}).get(bname or "lwir", 0.5))
    else:
        t_sky = 0.0552 * t_env ** 1.5
        kap = 0.5
    L_sky = kap * b_air                       # what the camera sees looking up
    L_sky_grey = planck_band(np.array([t_sky]), band[0], band[1])[0]

    # What a surface reflects: sky above, air/ground below, weighted by the
    # face's own view factor. A bare-metal target is 90 % reflection, so a
    # hemispherical average would make every metal read as the air temperature.
    nrm = thermal.face_normals(world)
    vf_sky = 0.5 * (1.0 + np.clip(nrm[:, 2], -1.0, 1.0))
    L_env = vf_sky * L_sky_grey + (1.0 - vf_sky) * b_air

    # Isothermal path: the atmosphere both attenuates the target and adds its
    # own emission. Omitting the second term biases every scene cold by an
    # amount that grows with range, which teaches a detector "far = cold".
    L = L_self + (1.0-eps_face)*L_env
    samples = int(ircfg.get("samples_per_axis", 1))
    if samples not in (1, 2, 3, 4):
        raise ValueError("ir.samples_per_axis must be 1..4")
    subpixel_mode = ircfg.get("subpixel_mode", "legacy")
    if subpixel_mode == "coverage" and samples < 2:
        raise ValueError("ir coverage mode requires samples_per_axis >= 2")
    high_cam = dict(cam, width=int(cam["width"])*samples, height=int(cam["height"])*samples)
    img, depth, pid = rasterize(world, high_cam, L, background=0.0,
                                subpixel_mode=subpixel_mode if samples == 1 else "none")
    img = _pixel_atmosphere(img, depth, high_cam, cfg, band, bname, tau, L_sky)
    img = gaussian_blur(img, blur*samples)
    instance_coverage = {}
    if samples > 1:
        H, W = int(cam["height"]), int(cam["width"])
        shape = (H, samples, W, samples)
        img = img.reshape(shape).mean(axis=(1, 3))
        for obj in world.objs:
            if obj.get("label", True):
                instance_coverage[obj["id"]] = (pid == obj["id"]).reshape(shape).mean(axis=(1, 3))
        ds = depth.reshape(shape).transpose(0, 2, 1, 3).reshape(H, W, -1)
        ids = pid.reshape(shape).transpose(0, 2, 1, 3).reshape(H, W, -1)
        nearest = np.argmin(ds, axis=2)
        depth = np.take_along_axis(ds, nearest[..., None], axis=2)[..., 0]
        pid = np.take_along_axis(ids, nearest[..., None], axis=2)[..., 0]

    # NETD is a temperature noise, so the radiance noise is netd * dL/dT and
    # dL/dT must be taken per pixel: between 250 K and 350 K it changes by 22x in
    # MWIR, so a single mean value over-amplifies hot regions and flattens cold
    # ones.
    t_map = np.asarray(radiance_to_temp(img, band), float).reshape(img.shape)
    flat = t_map.ravel()
    dLdT = (planck_band(flat + 1.0, band[0], band[1]) -
            planck_band(flat, band[0], band[1])).reshape(img.shape)
    rng_noise = np.random.default_rng(int(cfg.get("_frame_seed", 0)))
    img = img + rng_noise.normal(0.0, 1.0, img.shape) * (dLdT * netd)

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
            # A fixed +-20 K window centred on the median is what a camera with no
            # AGC would use, and on a sky-filled frame the sky's own gradient fills
            # the window: a 1.5 K aircraft then landed on 9 of 255 grey levels and
            # the thermal panel looked empty in every composite (measured).
            # Stretch to the frame's own 1..99.5 percentile instead -- the same
            # thing a real AGC does -- so a hot target saturates to white.
            lo_k, hi_k = np.percentile(t_img, [1.0, 99.5])
            span = float(hi_k - lo_k)
            if span < 6.0:                      # near-isothermal frame: hold 6 K
                mid = 0.5 * (float(hi_k) + float(lo_k))
                lo_k, hi_k = mid - 3.0, mid + 3.0
            else:
                lo_k, hi_k = float(lo_k), float(hi_k)
            tlo, thi = lo_k, hi_k
        prev = np.clip((t_img - tlo) / max(thi - tlo, 1e-6), 0, 1)
    return dict(
        radiance=img,
        counts_u16=np.clip(counts, 0.0, 65535.0).astype(np.uint16),
        preview=(np.clip(prev, 0, 1) * 255).astype(np.uint8),
        apparent_temp_k=t_img,
        depth=depth,
        obj_id=pid,
        temperature_k=temps_k,
        instance_coverage=instance_coverage,
        sampling=dict(samples_per_axis=samples, subpixel_mode=subpixel_mode,
                      depth_rule="minimum forward-camera-Z among pixel subsamples" if samples > 1 else "pixel-center forward-camera-Z"),
    )
