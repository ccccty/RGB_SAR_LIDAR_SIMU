"""Thermal infrared forward model: Planck radiance, atmosphere, sensor, raster."""
import math
import os

import numpy as np

from . import materials as matlib
from . import spectra as speclib
from . import thermal, atmosphere
from .scene import sensor_basis
from .weather import IR_BANDS, band_name_of

H_PLANCK = 6.62607015e-34
C_LIGHT = 2.99792458e8
K_BOLTZ = 1.380649e-23


_C1 = 2.0 * H_PLANCK * C_LIGHT ** 2
_C2 = H_PLANCK * C_LIGHT / K_BOLTZ

# Everything that depends only on the spectral grid is a pure function of
# (lambda1, lambda2, n): one frame calls planck_band nine times with the same
# band but different temperature arrays, so rebuilding linspace/lam**5/diff on
# every call is pure overhead. The tables are immutable once built and keyed by
# the band, so a dataset that alternates MWIR and LWIR keeps one per band.
_PLANCK_TABLES = {}

# Row block for the big calls (the per-pixel dL/dT needs ~7e5 rows x 48 lambdas).
# The whole-array form materialises several 265 MB temporaries per call, once
# per frame; a block keeps the working set in L2/L3 instead. Each row is
# computed independently of the others, so blocking changes speed, not values.
_PLANCK_CHUNK = 384


def _planck_table(lam1_um, lam2_um, n):
    key = (float(lam1_um), float(lam2_um), int(n))
    table = _PLANCK_TABLES.get(key)
    if table is None:
        lam = np.linspace(lam1_um, lam2_um, n)[None, :] * 1e-6
        lam1d = lam[0]
        table = dict(lam=lam,                       # (1, n) metres
                     lam1d=lam1d,
                     inv_lam5=_C1 / lam ** 5,        # (1, n)
                     weight=np.diff(lam1d))          # (n-1,) trapezoid edges
        table["inv_lam5"].setflags(write=False)
        table["lam1d"].setflags(write=False)
        table["weight"].setflags(write=False)
        _PLANCK_TABLES[key] = table
    return table


def _trapezoid_uniform(B, weight):
    """Trapezoidal band integral on the table's own wavelength grid.

    numpy spells the same sum as (diff(x) * (B[:, 1:] + B[:, :-1]) / 2.0)
    .sum(axis=1); caching the edge widths turns it into two array ops plus one
    reduction. Multiplication is commutative in IEEE 754, so the result is the
    same double for every row.
    """
    return ((B[:, 1:] + B[:, :-1]) * weight / 2.0).sum(axis=1)


def _planck_rows(T, table):
    """Planck band integral of a (M, 1) block of temperatures."""
    with np.errstate(over="ignore", invalid="ignore"):
        B = table["inv_lam5"] / np.expm1(_C2 / (table["lam"] * T))
    # Identical to np.nan_to_num(B, nan=0.0, posinf=0.0) for every physically
    # reachable temperature: expm1 reaches +inf only for T -> 0, and the far
    # field of the spectrum is then 0 either way. -inf needs a negative
    # temperature, which is not a radiance: it also becomes 0 here.
    bad = ~np.isfinite(B)
    if bad.any():
        B[bad] = 0.0
    return _trapezoid_uniform(B, table["weight"])


def planck_band(T, lam1_um, lam2_um, n=48):
    """In-band spectral radiance W/(m^2 sr) for temperature(s) T in Kelvin."""
    if n < 2 or not 0 < lam1_um < lam2_um:
        raise ValueError("Planck integration needs >=2 samples and 0 < lambda1 < lambda2")
    table = _planck_table(lam1_um, lam2_um, n)
    T = np.asarray(T, float).reshape(-1, 1)
    rows = T.shape[0]
    if rows <= _PLANCK_CHUNK:
        return _planck_rows(T, table)
    # Endpoint rectangles over-counted the band radiance by about 2% at the
    # default 48 samples. A trapezoidal spectral integral has the right limits.
    out = np.empty(rows, float)
    for start in range(0, rows, _PLANCK_CHUNK):
        stop = min(start + _PLANCK_CHUNK, rows)
        out[start:stop] = _planck_rows(T[start:stop], table)
    return out


# ---- dL/dT: analytic derivative table ------------------------------------
# The detector-noise term needs dL/dT at every pixel (~7e5 apparent
# temperatures per 960x720 frame). The difference form evaluates the whole
# 48-sample band integral twice per pixel -- 0.7 s of a 1.9 s frame on this
# machine. dB/dT has a closed form, so one table per band (built once, ~10 ms)
# plus an interpolation covers a frame in ~40 ms.
_DLDT_TABLES = {}
# 150..650 K comfortably contains radiance_to_temp's 200..450 K working window,
# so np.interp never clamps inside a frame.
_DLDT_RANGE = (150.0, 650.0, 4001)


def dldt_table(lam1_um, lam2_um, n=48, t_lo=None, t_hi=None, points=None):
    """(T grid, dL/dT) of a band, analytic and cached per (band, grid).

    dB/dT for B = c1/lambda^5 / (exp(x) - 1) with x = c2/(lambda T) is

        dB/dT = B * (x / T) * (1 + 1/(exp(x) - 1))

    and the band value is the same trapezoid integral used by planck_band, so
    the table is the exact derivative of the radiance model, not a secant.
    """
    t_lo = float(_DLDT_RANGE[0] if t_lo is None else t_lo)
    t_hi = float(_DLDT_RANGE[1] if t_hi is None else t_hi)
    points = int(_DLDT_RANGE[2] if points is None else points)
    if not (points >= 2 and 0.0 < t_lo < t_hi):
        raise ValueError("dL/dT table needs >=2 points and 0 < T_lo < T_hi")
    key = (float(lam1_um), float(lam2_um), int(n), t_lo, t_hi, points)
    grid = _DLDT_TABLES.get(key)
    if grid is None:
        table = _planck_table(lam1_um, lam2_um, n)
        grid_t = np.linspace(t_lo, t_hi, points)
        deriv = np.empty((points, table["lam1d"].size), float)
        for start in range(0, points, _PLANCK_CHUNK):
            stop = min(start + _PLANCK_CHUNK, points)
            block = grid_t[start:stop, None]
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                x = _C2 / (table["lam"] * block)
                em1 = np.expm1(x)
                B = table["inv_lam5"] / em1
                deriv[start:stop] = B * (x / block) * (1.0 + 1.0 / em1)
        bad = ~np.isfinite(deriv)
        if bad.any():
            deriv[bad] = 0.0
        grid = (grid_t, _trapezoid_uniform(deriv, table["weight"]))
        grid[0].setflags(write=False)
        grid[1].setflags(write=False)
        _DLDT_TABLES[key] = grid
    return grid


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
    face_obj = world.face_obj
    # Pixel-centre coordinates for the whole frame, sliced per triangle: the
    # arithmetic below is unchanged, only the two np.meshgrid (ny, nx) copies
    # and the per-triangle np.arange are gone. A face-heavy scene calls this
    # loop tens of thousands of times per frame, so those allocations were
    # measurable; broadcast views produce the very same doubles.
    px = np.arange(W) + 0.5
    py = np.arange(H) + 0.5
    for i, uu, vv, zi in projected_triangles():
        x0 = max(int(np.floor(uu.min())), 0)
        x1 = min(int(np.ceil(uu.max())) + 1, W)
        y0 = max(int(np.floor(vv.min())), 0)
        y1 = min(int(np.ceil(vv.max())) + 1, H)
        if x1 <= x0 or y1 <= y0:
            continue
        denom = ((vv[1] - vv[2]) * (uu[0] - uu[2]) + (uu[2] - uu[1]) * (vv[0] - vv[2]))
        if abs(denom) < 1e-9:
            continue
        gx = (px[x0:x1])[None, :]
        gy = (py[y0:y1])[:, None]
        a0 = vv[1] - vv[2]
        a1 = uu[2] - uu[1]
        a2 = vv[2] - vv[0]
        a3 = uu[0] - uu[2]
        gx0 = gx - uu[2]
        gy0 = gy - vv[2]
        l0 = (a0 * gx0 + a1 * gy0) / denom
        l1 = (a2 * gx0 + a3 * gy0) / denom
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
        pid[y0:y1, x0:x1][upd] = face_obj[i]

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


def preview_window(t_img, ircfg):
    """Display window (kelvin) of the 8-bit thermal preview.

    Three policies, in priority order:

    * an explicit 'ir.preview_window_c' wins -- the operator pinned the window
      and the frame must not move it;
    * otherwise the frame's own 1..99.5 percentile span is used, which is what
      an AGC does and what makes a hot target saturate to white.  A
      near-isothermal frame holds 6 K so the preview is not all one code.
    * 'ir.preview_span_c' caps that automatic span at N kelvin around the frame
      median.  A real display runs a fixed-width window (+-10 K or so) instead
      of stretching whatever the frame happens to span: a 40 K sky/target
      range on 256 codes puts a 28 mK NETD on 0.18 of a grey level, so the
      noise sits in the 16-bit product and is invisible in the preview
      (measured).  Centring on the median -- not on the middle of the
      percentile range -- keeps the sky mid-grey instead of clipping it to
      black.  None keeps the historical percentile stretch bit for bit.

    Returns (t_lo, t_hi) in kelvin.
    """
    win = ircfg.get("preview_window_c")
    if win:
        return win[0] + 273.15, win[1] + 273.15
    lo_k, hi_k = np.percentile(t_img, [1.0, 99.5])
    span = float(hi_k - lo_k)
    if span < 6.0:                      # near-isothermal frame: hold 6 K
        mid = 0.5 * (float(hi_k) + float(lo_k))
        lo_k, hi_k = mid - 3.0, mid + 3.0
    else:
        lo_k, hi_k = float(lo_k), float(hi_k)
    cap_c = ircfg.get("preview_span_c")
    if cap_c is not None:
        cap_c = float(cap_c)
        if not np.isfinite(cap_c) or cap_c <= 0.0:
            raise ValueError("ir.preview_span_c must be a positive number of kelvin")
        if hi_k - lo_k > cap_c:
            mid = float(np.median(t_img))
            lo_k, hi_k = mid - 0.5*cap_c, mid + 0.5*cap_c
    return lo_k, hi_k


def _equalize(img, levels=512):
    """Histogram equalisation: what a thermal camera's AGC does for display."""
    lo, hi = np.percentile(img, [0.2, 99.8])
    x = np.clip((img - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
    hist, edges = np.histogram(x, bins=levels, range=(0.0, 1.0))
    cdf = np.cumsum(hist).astype(np.float64)
    cdf /= max(cdf[-1], 1.0)
    return np.interp(x, edges[:-1], cdf)


def _background_image_path(cfg):
    """Resolve ir.background_image (or top-level background_image) to a file.

    Relative paths are resolved against the scene file's directory, matching
    how object models are resolved.  Returns None when no plate is configured,
    which keeps the no-background render path bit-for-bit identical.
    """
    ircfg = cfg.get("ir") or {}
    raw = ircfg.get("background_image")
    if raw is None:
        raw = cfg.get("background_image")
    if raw is None or str(raw).strip() == "":
        return None
    path = os.path.expanduser(str(raw))
    if not os.path.isabs(path):
        path = os.path.join(str(cfg.get("_base_dir") or "."), path)
    return os.path.normpath(path)


def _background_temperature_k(cfg):
    """Kelvin anchor for the plate's radiometric scale (default 260 K).

    A photographic plate is not a calibrated radiometer, so its 8-bit value is
    interpreted as a fraction of the band radiance of a blackbody at this
    temperature.  That keeps the background in the same radiometric order as
    the scene instead of dumping raw 0..255 (or 0..65535) values into it.
    """
    ircfg = cfg.get("ir") or {}
    # Default to the scene's own ground temperature, not a fixed 260 K. A
    # 260 K anchor put the photograph at -13 C while the ground, the buildings
    # and the target were all solving near +20 C, so every piece of geometry
    # saturated against a plate that represented frozen terrain.
    _ground_c = (cfg.get("ground") or {}).get("temp_c")
    if _ground_c is None:
        _ground_c = float(cfg.get("air", {}).get("temp_c", 20.0)) + 8.0
    raw = ircfg.get("background_temp_k",
                    cfg.get("background_temp_k", float(_ground_c) + 273.15))
    value = float(raw)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("红外背景温度必须为正的开尔文温度 ir.background_temp_k，收到 %r" % (raw,))
    return value


_BACKGROUND_PLATES = {}


def _read_radiance_hdr(path):
    """Decode a Radiance .hdr (RGBE) into a float32 (H, W, 3) luminance array.

    Pillow cannot open Radiance HDR, yet that is the format every CC0 sky plate
    ships in and the only one Blender takes as an environment texture. Refusing
    the combination "HDRI sky + IR band" would make the background feature
    unusable for exactly the case it was added for, so the RGBE pixels are read
    here -- about forty lines, no new dependency.
    """
    with open(path, "rb") as fh:
        data = fh.read()
    if not data.startswith(b"#?"):
        raise ValueError("不是 Radiance HDR 文件：%s" % path)
    pos = 0
    width = height = 0
    while pos < len(data):
        end = data.find(b"\n", pos)
        if end < 0:
            raise ValueError("Radiance 头不完整：%s" % path)
        line = data[pos:end].strip()
        pos = end + 1
        if line:
            continue
        # the blank line ends the header; the resolution line follows
        end = data.find(b"\n", pos)
        if end < 0:
            raise ValueError("Radiance 缺少分辨率行：%s" % path)
        parts = data[pos:end].strip().split()
        pos = end + 1
        if len(parts) != 4 or not parts[1].isdigit() or not parts[3].isdigit():
            raise ValueError("Radiance 分辨率行无法解析：%r"
                             % data[pos:end].strip()[:40])
        height, width = int(parts[1]), int(parts[3])
        break
    if width <= 0 or height <= 0:
        raise ValueError("Radiance 分辨率无效：%s" % path)

    out = np.zeros((height, width, 3), np.float32)
    plane = np.zeros((4, width), np.uint8)
    for y in range(height):
        rle = (width >= 8 and width < 32768 and pos + 4 <= len(data)
               and data[pos] == 2 and data[pos + 1] == 2
               and ((data[pos + 2] << 8) | data[pos + 3]) == width)
        if rle:
            pos += 4
            for ch in range(4):
                x = 0
                while x < width:
                    if pos >= len(data):
                        raise ValueError("Radiance 数据被截断：%s" % path)
                    n = data[pos]
                    pos += 1
                    if n > 128:
                        n -= 128
                        if pos >= len(data):
                            raise ValueError("Radiance 数据被截断：%s" % path)
                        plane[ch, x:x + n] = data[pos]
                        pos += 1
                    else:
                        if pos + n > len(data):
                            raise ValueError("Radiance 数据被截断：%s" % path)
                        plane[ch, x:x + n] = np.frombuffer(data, np.uint8, n, pos)
                        pos += n
                    x += n
        else:
            need = width * 4
            if pos + need > len(data):
                raise ValueError("Radiance 数据被截断：%s" % path)
            plane[:] = np.frombuffer(data, np.uint8, need, pos).reshape(4, width)
            pos += need
        scale = np.where(plane[3] > 0,
                         np.exp2(plane[3].astype(np.float32) - 136.0),
                         0.0).astype(np.float32)
        for c in range(3):
            out[y, :, c] = plane[c].astype(np.float32) * scale
    return out


def _load_plate_gray(path):
    """Open a plate as an 8-bit luminance image, Radiance HDR included."""
    with open(path, "rb") as fh:
        magic = fh.read(2)
    if magic == b"#?":
        hdr = _read_radiance_hdr(path)
        # Reinhard tone map: unbounded radiance -> 0..1 without clipping the sky
        # to white (a raw / max() would collapse every cloud to 255).
        lum = (0.2126 * hdr[:, :, 0] + 0.7152 * hdr[:, :, 1]
               + 0.0722 * hdr[:, :, 2])
        mapped = lum / (1.0 + np.maximum(lum, 0.0))
        from PIL import Image
        return Image.fromarray(np.clip(mapped * 255.0, 0, 255).astype(np.uint8),
                               mode="L")
    from PIL import Image
    with Image.open(path) as handle:
        handle.load()
        return handle.convert("L")


def _plate_view(gray, cam, w, h):
    """Sample the equirectangular plate through the camera's own lens.

    The plate is a full 360x180 panorama. Pasting all of it into a 2.2 deg seeker
    frame is what made a cloudy IR image sit next to a clear optical one: the
    optical renderer projects the environment through the lens, so it shows a
    tiny patch of sky, while this code showed the whole sky at once. Cropping the
    window the lens actually covers puts the two bands back on the same weather.

    Returns an (h, w) float32 array in 0..1, or None when the camera pose is not
    usable (caller then falls back to a centre crop).
    """
    src = np.asarray(gray, dtype=np.float32)
    if src.ndim != 2 or src.size == 0:
        return None
    ih, iw = src.shape
    try:
        fov = float(cam.get("fov_deg") or 0.0)
        pos = np.asarray(cam.get("pos"), float)
        look = np.asarray(cam.get("look_at", [0.0, 0.0, 0.0]), float)
    except (TypeError, ValueError):
        return None
    if not (0.1 <= fov <= 170.0) or pos.shape != (3,) or look.shape != (3,):
        return None
    d = look - pos
    n = float(np.linalg.norm(d))
    if n < 1e-9:
        return None
    d = d / n
    az = math.degrees(math.atan2(d[1], d[0]))
    el = math.degrees(math.asin(float(np.clip(d[2], -1.0, 1.0))))
    vfov = fov * (h / float(w)) if h else fov
    # equirectangular convention: u runs with azimuth (0 at -180 deg), v runs
    # from the zenith at row 0 to the nadir at the last row
    u_c = (az + 180.0) / 360.0
    v_c = (90.0 - el) / 180.0
    half_u = 0.5 * fov / 360.0
    half_v = 0.5 * vfov / 180.0
    us = (u_c + np.linspace(-half_u, half_u, w, endpoint=False)
          + 0.5 * fov / 360.0 / max(w, 1))
    vs = (v_c + np.linspace(-half_v, half_v, h, endpoint=False)
          + 0.5 * vfov / 180.0 / max(h, 1))
    # Bilinear, not nearest: a 2.2 deg window covers barely a dozen texels of a
    # 2048-wide panorama, and nearest-neighbour blew those up into visible tiles.
    u_pix = np.mod(us, 1.0) * iw - 0.5
    v_pix = np.clip(vs, 0.0, 1.0) * ih - 0.5
    u0 = np.floor(u_pix).astype(np.int64)
    v0 = np.floor(v_pix).astype(np.int64)
    fu = (u_pix - u0)[None, :].astype(np.float32)
    fv = (v_pix - v0)[:, None].astype(np.float32)
    u1 = np.mod(u0 + 1, iw)
    u0 = np.mod(u0, iw)
    v1 = np.clip(v0 + 1, 0, ih - 1)
    v0 = np.clip(v0, 0, ih - 1)
    p00 = src[np.ix_(v0, u0)]
    p01 = src[np.ix_(v0, u1)]
    p10 = src[np.ix_(v1, u0)]
    p11 = src[np.ix_(v1, u1)]
    top = p00 * (1.0 - fu) + p01 * fu
    bottom = p10 * (1.0 - fu) + p11 * fu
    return (top * (1.0 - fv) + bottom * fv) / np.float32(255.0)


def _plate_view_ground(gray, cam, cfg, w, h):
    """Sample a ground plate by projecting each pixel onto the z=0 plane.

    The plate is a photograph of terrain, not a 360x180 panorama. Feeding it to
    the equirectangular path made a nadir camera sample the pole of the sphere:
    the whole frame collapsed onto a few texels and a city block rendered as a
    flat grey checkerboard instead of a city.

    The mapping matches the optical renderer exactly (Object coordinates, scale
    1/span, offset backdrop_offset), so the two bands show the same terrain under
    the same target.
    """
    g = cfg.get("ground") or {}
    # Only when a plate is actually attached to the ground. backdrop_span_m
    # survives in scenes whose backdrop is empty, and using the span without the
    # photograph sent a sky plate through the ground projection: the IR band then
    # showed a different crop of the file than the optical band did, which is
    # exactly the "why are the backgrounds different" report.
    if not g.get("backdrop"):
        return None
    span = float(g.get("backdrop_span_m") or 0.0)
    if span <= 0.0:
        return None
    span_y = float(g.get("backdrop_span_y_m") or span)
    try:
        off = g.get("backdrop_offset") or (0.5, 0.5)
        off_u, off_v = float(off[0]), float(off[1])
        pos = np.asarray(cam.get("pos"), float)
        look = np.asarray(cam.get("look_at", [0.0, 0.0, 0.0]), float)
        fov = float(cam.get("fov_deg") or 0.0)
    except (TypeError, ValueError, IndexError):
        return None
    if not (0.1 <= fov <= 170.0):
        return None
    src = np.asarray(gray, dtype=np.float32)
    if src.ndim != 2 or src.size == 0:
        return None
    ih, iw = src.shape
    p, right, up, fwd = sensor_basis(pos, look, cam.get("up", (0, 0, 1)),
                                     cam.get("roll_deg", 0.0))
    fx = (w / 2.0) / math.tan(math.radians(fov) / 2.0)
    px = (np.arange(w) + 0.5 - w / 2.0) / fx
    py = -(np.arange(h) + 0.5 - h / 2.0) / fx
    gx, gy = np.meshgrid(px, py)
    dirs = fwd[None, None, :] + right[None, None, :] * gx[:, :, None] \
        + up[None, None, :] * gy[:, :, None]
    dz = dirs[:, :, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (0.0 - float(p[2])) / dz
    ok = np.isfinite(t) & (t > 0.0)
    if not ok.any():
        return None
    hit = p[None, None, :] + dirs * t[:, :, None]
    uu = hit[:, :, 0] / span + off_u
    vv = hit[:, :, 1] / span_y + off_v
    # wrap by the tile, then bilinear: the plate repeats every span_m
    uu = np.mod(uu, 1.0)
    vv = np.mod(vv, 1.0)
    x0 = np.floor(uu * (iw - 1)).astype(np.int64)
    y0 = np.floor(vv * (ih - 1)).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, iw - 1)
    y1 = np.clip(y0 + 1, 0, ih - 1)
    ax = (uu * (iw - 1) - x0).astype(np.float32)
    ay = (vv * (ih - 1) - y0).astype(np.float32)
    out = ((src[y0, x0] * (1 - ax) + src[y0, x1] * ax) * (1 - ay)
           + (src[y1, x0] * (1 - ax) + src[y1, x1] * ax) * ay)
    out = np.where(ok, out, np.nan)
    # _load_plate_gray hands back 0..255; the composite expects 0..1. Clipping
    # the raw values to 0..1 turned the whole plate into solid white.
    scale = 255.0 if float(np.nanmax(out)) > 1.5 else 1.0
    return np.clip(out / scale, 0.0, 1.0).astype(np.float32)


def _background_plate(path, w, h, cam=None, cfg=None):
    """Load an 8-bit plate and fit it to the frame: centre crop + resize.

    This is an unregistered approximation, deliberately: the IR camera is
    assumed to share the optical axis and (configured) field of view with the
    plate, so there is no per-frame pose to solve.  The plate is centre-cropped
    to the frame's aspect ratio and resampled to (h, w); camera motion, parallax
    and lens distortion inside the plate are ignored.

    Returns a float32 (h, w) array in 0..1 -- the 8-bit luminance divided by
    255.  A missing or undecodable file is an explicit ValueError, never a
    silently black frame.
    """
    w, h = int(w), int(h)
    if w <= 0 or h <= 0:
        raise ValueError("红外背景图的目标分辨率必须为正数，收到 %dx%d" % (w, h))
    path = os.path.abspath(str(path))
    if not os.path.isfile(path):
        raise ValueError("红外背景图不存在或不是文件：%s" % path)
    view_key = None
    if cam:
        try:
            view_key = (round(float(cam.get("fov_deg") or 0.0), 4),
                        tuple(round(float(v), 2) for v in cam.get("pos")),
                        tuple(round(float(v), 2) for v in cam.get("look_at")))
        except (TypeError, ValueError):
            view_key = None
    try:
        stat = os.stat(path)
        cache_key = (path, w, h, stat.st_mtime_ns, stat.st_size, view_key)
    except OSError:
        cache_key = None
    if cache_key is not None:
        cached = _BACKGROUND_PLATES.get(cache_key)
        if cached is not None:
            return cached
    from PIL import Image
    try:
        gray = _load_plate_gray(path)
    except Exception as exc:
        raise ValueError("红外背景图无法解码：%s（%s）" % (path, exc)) from exc
    # Only an equirectangular panorama (2:1) is projected through the lens. A
    # 3:1 photo is still just a photo -- treating every wide image as a panorama
    # put the horizon in the wrong place for ordinary plates.
    if cam:
        _gw, _gh = gray.size
        _ratio = _gw / float(_gh) if _gh else 0.0
    # A ground plate is terrain seen through a lens, so project it through the
    # lens: pixels -> z=0 plane -> plate uv. Only fall through to the panorama
    # path when the scene really is a sky panorama behind the target.
    if cfg is not None:
        ground = _plate_view_ground(gray, cam, cfg, w, h) if cam else None
        if ground is not None:
            ground = np.ascontiguousarray(ground, dtype=np.float32)
            ground.setflags(write=False)
            if cache_key is not None:
                _BACKGROUND_PLATES[cache_key] = ground
            return ground
    if cam and 1.85 <= _ratio <= 2.15:
        projected = _plate_view(gray, cam, w, h)
        if projected is not None:
            projected = np.ascontiguousarray(projected, dtype=np.float32)
            projected.setflags(write=False)
            if cache_key is not None:
                _BACKGROUND_PLATES[cache_key] = projected
            return projected
    iw, ih = gray.size
    if iw <= 0 or ih <= 0:
        raise ValueError("红外背景图尺寸无效：%s（%dx%d）" % (path, iw, ih))
    target_aspect = w / float(h)
    source_aspect = iw / float(ih)
    if source_aspect > target_aspect:
        crop_w = max(1, int(round(ih * target_aspect)))
        crop_h = ih
        left = max(0, (iw - crop_w) // 2)
        box = (left, 0, left + crop_w, crop_h)
    else:
        crop_w = iw
        crop_h = max(1, int(round(iw / target_aspect)))
        top = max(0, (ih - crop_h) // 2)
        box = (0, top, crop_w, top + crop_h)
    plate = gray.crop(box)
    if plate.size != (w, h):
        resampling = getattr(Image, "Resampling", Image)
        plate = plate.resize((w, h), resampling.BILINEAR)
    result = np.asarray(plate, dtype=np.float32) / np.float32(255.0)
    result.setflags(write=False)
    if cache_key is not None:
        _BACKGROUND_PLATES[cache_key] = result
    return result


def _composite_background(img, counts, preview, temperature, depth, cfg, band,
                          gain, t0, t1, cam=None, obj_id=None):
    """Paste the background plate into pixels that saw no geometry.

    Only pixels whose depth is not finite (the rasteriser's miss marker) or
    exactly zero (the serialised no-return convention) are replaced.  Geometry
    pixels are copied through untouched, so a target can never be painted over
    by its background.

    The paste happens after the sensor chain: the plate defines an image plane,
    not a scattering surface, so blurring it into the target would produce a
    halo exactly where the label is.  The display image keeps the plate's own
    8-bit grey (that is what the caller uploaded to look at), while counts and
    apparent temperature carry the radiometric proxy described below.
    """
    path = _background_image_path(cfg)
    if path is None:
        return img, counts, preview, temperature
    if not os.path.isfile(path):
        raise ValueError("红外背景图不存在或不是文件：%s" % path)
    depth = np.asarray(depth, dtype=np.float64)
    void = ~np.isfinite(depth) | (depth == 0.0)
    # A terrain backdrop replaces the synthetic ground plane, not just the misses.
    # The plate is a photograph of real terrain; the plane underneath it is a
    # handful of giant triangles whose flat facets were what a top-down plate
    # scene actually rendered, as a blocky grey checkerboard, while the photo
    # only showed through the pixels where no ground hit at all.
    if (cfg.get("ground") or {}).get("backdrop") and obj_id is not None:
        void = void | (np.asarray(obj_id) == -1)
    if not void.any():
        return img, counts, preview, temperature
    H, W = depth.shape
    plate = _background_plate(path, W, H, cam=cam, cfg=cfg)
    bg_temp_k = _background_temperature_k(cfg)
    # Linear radiometric mapping: grey 0 -> no radiance, grey 255 -> the band
    # radiance of a blackbody at bg_temp_k.  Since Planck radiance is monotonic
    # in temperature, the plate can never outshine a target hotter than
    # bg_temp_k -- the order-of-magnitude blow-up this mapping avoids.
    blackbody = float(planck_band([bg_temp_k], band[0], band[1])[0])
    L_plate = plate.astype(np.float64) * blackbody
    T_plate = np.asarray(radiance_to_temp(L_plate.ravel(), band), float).reshape(H, W)
    counts_plate = np.round(np.clip((T_plate - t0) / (t1 - t0), 0.0, 1.0) * 65535.0 * gain)
    # The plate is already a display image.  A thermal AGC would map 260 K to a
    # dark grey, which would make an uploaded "sky" PNG look like a black frame
    # and defeat the point of supplying it; the visible panel therefore keeps
    # the plate's own grey level, while counts/temperature stay radiometric.
    preview_plate = plate.astype(np.float64)
    img = np.array(img, dtype=np.float64, copy=True)
    counts = np.array(counts, dtype=np.float64, copy=True)
    preview = np.array(preview, dtype=np.float64, copy=True)
    temperature = np.array(temperature, dtype=np.float64, copy=True)
    img[void] = L_plate[void]
    counts[void] = counts_plate[void]
    preview[void] = preview_plate[void]
    temperature[void] = T_plate[void]
    return img, counts, preview, temperature


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
        # No derived weather state (bare API call / unit probe). This is a
        # simplified grey-sky estimate; the production render loop always has
        # _weather and uses the Clark-Allen t_sky_k there. It deliberately has
        # NO (1 - 0.25*RH) factor -- the humidity sign of that correction was
        # backwards (see weather.sky_temperature_k and thermal.sky_temp).
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
    # "analytic" is the exact dL/dT from the per-band table: it removes the
    # half-step bias of a 1 K secant and costs ~10 ms per frame instead of
    # ~700 ms at 960x720. Its output change against the historical form is one
    # detector count on ~4 % of pixels (radiance differs by <= 1.5e-5
    # relative), because only the noise scale moves. "difference" reproduces
    # the old numbers bit for bit and is what an existing dataset needs if it
    # has to be regenerated exactly.
    dldt_mode = ircfg.get("dldt_mode", "analytic")
    if dldt_mode == "difference":
        dLdT = (planck_band(flat + 1.0, band[0], band[1]) -
                planck_band(flat, band[0], band[1])).reshape(img.shape)
    elif dldt_mode == "analytic":
        grid_t, grid_d = dldt_table(band[0], band[1])
        dLdT = np.interp(flat, grid_t, grid_d).reshape(img.shape)
        # The historical path mapped a non-finite temperature (it can only come
        # from a malformed radiance) to a zero slope through nan_to_num; keep
        # that behaviour instead of propagating NaN into the noise.
        dLdT = np.where(np.isfinite(dLdT), dLdT, 0.0)
    else:
        raise ValueError("ir.dldt_mode must be difference or analytic")
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
        # One place decides the display window: a fixed operator window, the
        # frame's own 1..99.5 percentile stretch (the historical AGC), or that
        # stretch capped by ir.preview_span_c.  Nothing here touches the
        # 16-bit product.
        tlo, thi = preview_window(t_img, ircfg)
        prev = np.clip((t_img - tlo) / max(thi - tlo, 1e-6), 0, 1)
    # The plate replaces the sky/void, not the target: only non-finite depth
    # pixels are touched, after blur/noise/AGC, so geometry pixels stay
    # bit-for-bit what they were without a background.
    img, counts, prev, t_img = _composite_background(
        img, counts, prev, t_img, depth, cfg, band, gain, t0, t1, cam=cam,
        obj_id=pid)
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
