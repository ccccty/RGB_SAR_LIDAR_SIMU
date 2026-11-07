"""Oscilloscope-style range-Doppler / range-azimuth figures.

The dataset's radar PNGs used to be bare 8-bit strips. A figure that has to
survive a paper or a dataset browser needs its own axes, dB scale and target
parameters. This module turns one power map plus axis vectors into such a
picture with numpy and Pillow only (matplotlib is not installed), so the LUT,
ticks, grid, colour bar and table are drawn here.

x is range in metres, y is radial velocity (RD) or beam azimuth (RA), and
positive closing speed / azimuth point up; returned rectangles are half-open.
A CJK face is used when present, else the figure degrades to ASCII, not tofu.
"""

import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_CJK_FONTS = ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/msyhbd.ttc",
              "C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/simsun.ttc")
_ASCII_FONTS = ("C:/Windows/Fonts/consola.ttf", "C:/Windows/Fonts/arial.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")

def _font_paths():
    """Candidate (path, is_cjk) pairs in priority order; patched by the test."""
    env = os.environ.get("SIMKIT_SCOPE_FONT")
    out = [(env, True)] if env else []
    out += [(p, True) for p in _CJK_FONTS]
    out += [(p, False) for p in _ASCII_FONTS]
    return out

def _load_font(size):
    """Return (font, has_cjk); never raises, so a missing font is not fatal."""
    for path, is_cjk in _font_paths():
        if path and os.path.isfile(path):
            try:
                return ImageFont.truetype(path, int(size)), bool(is_cjk)
            except (OSError, ValueError):
                continue
    try:
        return ImageFont.load_default(size=int(size)), False
    except (TypeError, ValueError):
        return ImageFont.load_default(), False

# 256-level LUTs written out here instead of imported: the figure must be
# byte-for-byte reproducible on a machine without matplotlib. Anchor stops are
# published turbo / inferno samples, linearly interpolated.
_CMAP_STOPS = {
    "turbo": ((0.000, (48, 18, 59)), (0.083, (65, 69, 171)), (0.167, (70, 117, 237)),
              (0.250, (62, 155, 254)), (0.333, (33, 185, 242)), (0.417, (17, 208, 208)),
              (0.500, (34, 225, 159)), (0.583, (82, 235, 105)), (0.667, (148, 237, 61)),
              (0.750, (214, 223, 42)), (0.833, (251, 182, 47)), (0.917, (248, 119, 31)),
              (1.000, (180, 4, 38))),
    "inferno": ((0.000, (0, 0, 4)), (0.125, (22, 11, 57)), (0.250, (66, 10, 104)),
                (0.375, (106, 23, 110)), (0.500, (147, 38, 103)), (0.625, (188, 55, 84)),
                (0.750, (221, 81, 58)), (0.875, (243, 120, 25)), (1.000, (252, 255, 164))),
}
_LUT_CACHE = {}
# A scope video curve: half of an exponential noise floor sits in the bottom 20%
# of a dB window, so a linear index prints black. Gamma lifts the floor, not edges.
_DISPLAY_GAMMA = 1.0

def cmap_lut(name="turbo"):
    """256x3 uint8 colour table for a named map (turbo, inferno)."""
    key = str(name).lower()
    if key in _LUT_CACHE:
        return _LUT_CACHE[key]
    stops = _CMAP_STOPS.get(key)
    if stops is None:
        raise ValueError("unknown colormap %r: use turbo or inferno" % (name,))
    pos = np.array([s[0] for s in stops], float) * 255.0
    rgb = np.array([s[1] for s in stops], float)
    grid = np.arange(256, dtype=float)
    lut = np.stack([np.interp(grid, pos, rgb[:, c]) for c in range(3)], axis=-1)
    lut = np.clip(np.round(lut), 0, 255).astype(np.uint8)
    lut.setflags(write=False)
    _LUT_CACHE[key] = lut
    return lut

def _text(cjk):
    """User-visible strings: Chinese when a CJK face loaded, ASCII otherwise."""
    if cjk:
        return dict(
            rd="雷达距离-多普勒图 (RD)", ra="雷达距离-角度图 (RA)",
            x="距离 / m", vel="径向速度 / (m/s)", az="方位 / °", beam="波束序号",
            zero_v="零速", zero_az="零方位", hdr_name="编号/类别", hdr_range="距离/m",
            hdr_vel="径向速度/(m/s)", hdr_az="方位/°", hdr_snr="SNR/dB",
            noise="噪声基底 %.1f dB", peak="峰值 SNR %.1f dB", band="频段",
            rres="距离分辨率", vres="速度分辨率", ares="角度分辨率",
            frame="第 %d 帧", scene="场景", more="另有 %d 个目标未列出",
            db="dB", sep=" · ")
    return dict(
        rd="Range-Doppler (RD)", ra="Range-Azimuth (RA)",
        x="Range / m", vel="Radial velocity / (m/s)", az="Azimuth / deg",
        beam="Beam index", zero_v="zero", zero_az="0 deg", hdr_name="ID/class",
        hdr_range="Range/m", hdr_vel="Rad.vel/(m/s)", hdr_az="Az/deg", hdr_snr="SNR/dB",
        noise="Noise floor %.1f dB", peak="peak SNR %.1f dB", band="band",
        rres="dR", vres="dv", ares="dAz", frame="frame %d", scene="scene",
        more="+%d more targets", db="dB", sep=" | ")

def _fmt(value, kind):
    """Human-readable cell text; '--' when the value is missing."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "--"
    if not np.isfinite(v):
        return "--"
    if kind == "range":
        return "%.0f" % v if abs(v) >= 1000.0 else "%.1f" % v
    if kind == "vel":
        return "%.1f" % v if abs(v) >= 100.0 else "%.2f" % v
    return "%.2f" % v if kind == "az" else "%.1f" % v

def _targets(targets):
    """Normalise caller targets to plain floats so formatting never fails."""
    out = []
    for i, t in enumerate(targets or ()):
        if isinstance(t, dict):
            name = t.get("name") or t.get("label") or t.get("cls") or ""
            out.append(dict(index=i, name=str(name),
                            range_m=t.get("range_m", t.get("range", t.get("r"))),
                            v_r_mps=t.get("v_r_mps", t.get("v_r", t.get("v"))),
                            az_deg=t.get("az_deg", t.get("az", t.get("angle"))),
                            snr_db=t.get("snr_db", t.get("snr"))))
    return out

def _to_db(power):
    p = np.asarray(power, dtype=float)
    if p.ndim != 2:
        raise ValueError("power must be 2-D, got shape %r" % (p.shape,))
    floor = max(float(np.nanmax(p)) * 1e-9, 1e-30)
    return 10.0 * np.log10(np.maximum(np.nan_to_num(p, nan=0.0, posinf=0.0), floor))

def _axis(values, name):
    a = np.asarray(values, dtype=float).ravel()
    if a.size < 2 or not np.all(np.isfinite(a)) or a[-1] == a[0]:
        raise ValueError("%s must hold two or more finite, distinct values" % name)
    return a

def _nice_ticks(lo, hi, target=7):
    """1/2/5 x 10^k ticks covering [lo, hi] without matplotlib."""
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return [float(lo)]
    raw = (hi - lo) / max(int(target) - 1, 1)
    mag = 10.0 ** np.floor(np.log10(raw))
    step = mag
    for m in (1.0, 2.0, 5.0, 10.0):
        if raw <= m * mag:
            step = m * mag
            break
    first = np.ceil(lo / step) * step
    return [float(v) for v in np.arange(first, hi + step * 0.5, step)
            if lo - 1e-9 <= v <= hi + 1e-9]

_fmt_tick = lambda v, span: ("%.0f" if abs(span) >= 200.0 else  # noqa: E731
                               "%.1f" if abs(span) >= 20.0 else
                               "%.2f" if abs(span) >= 2.0 else "%.3f") % v

def _layout(size, has_sub, panel_w=0.0):
    """Figure layout; a non-zero panel_w reserves a right-hand gutter."""
    W, H = int(size[0]), int(size[1])
    cbar_gap, cbar_w, cbar_lab = 12, 18, 62
    top = 78 if has_sub else 62
    x1 = W - 26 - cbar_w - cbar_gap - cbar_lab - (panel_w + 26.0 if panel_w else 0.0)
    plot = (96, top, int(x1), H - 64)
    return plot, (plot[2] + cbar_gap, plot[1], plot[2] + cbar_gap + cbar_w, plot[3])

def _tw(draw, s, font):
    try:
        return float(draw.textlength(s, font=font))
    except Exception:
        return float(len(s)) * 0.62 * getattr(font, "size", 14)

def _panel_model(rows, kind, tdict):
    """Headers, cell text and alignment for the parameter table."""
    if kind == "rd":
        keys = (("name", "hdr_name"), ("range_m", "hdr_range"), ("v_r_mps", "hdr_vel"),
                ("az_deg", "hdr_az"), ("snr_db", "hdr_snr"))
    else:
        keys = (("name", "hdr_name"), ("range_m", "hdr_range"), ("az_deg", "hdr_az"),
                ("v_r_mps", "hdr_vel"), ("snr_db", "hdr_snr"))
    headers = [tdict[k[1]] for k in keys]
    cells = []
    for r in rows:
        line = []
        for key, _hdr in keys:
            if key == "name":
                line.append(("%d %s" % (r["index"], r["name"])).strip())
            else:
                line.append(_fmt(r[key], key.split("_")[0]))
        cells.append(line)
    return headers, ["left"] + ["right"] * (len(keys) - 1), cells

def _panel_geometry(draw, font, headers, cells, pad=10, row_h=19):
    """Panel pixel size measured from its own content."""
    widths = [max([_tw(draw, headers[c], font)] +
                  [_tw(draw, line[c], font) for line in cells]) + 14.0
              for c in range(len(headers))]
    return int(np.ceil(sum(widths) + 2 * pad)), \
        int(np.ceil(2 * pad + 20 + 5 + row_h * len(cells) + 6 + 19)), widths

def _clear_of(panel, pts, halo):
    if pts is None or len(pts) == 0:
        return True
    p = np.asarray(pts, float)
    return not bool(np.any((p[:, 0] >= panel[0] - halo) & (p[:, 0] <= panel[2] + halo) &
                           (p[:, 1] >= panel[1] - halo) & (p[:, 1] <= panel[3] + halo)))

def _draw_title(draw, lines, left, max_w):
    """Title + optional subtitle, shrinking the title until it fits."""
    for size in (22, 20, 18, 16, 14):
        font, _ = _load_font(size)
        if _tw(draw, lines[0], font) <= max_w:
            break
    draw.text((left, 10), lines[0], font=font, fill=(238, 243, 250))
    if len(lines) > 1 and lines[1]:
        sub, _ = _load_font(14)
        draw.text((left, 38), lines[1], font=sub, fill=(150, 165, 182))

def _draw_axes(draw, plot, xt, yt, x_of, y_of, xlab, ylab, font, xspan, yspan,
               y_int=False, y_zero=None, zero_label=None):
    """Frame, ticks, labels, grid and the zero reference line."""
    x0, y0, x1, y1 = plot
    for v in xt:
        px = x_of(v)
        draw.line([(px, y0), (px, y1)], fill=(58, 66, 78), width=1)
        draw.line([(px, y1), (px, y1 + 6)], fill=(190, 198, 210), width=1)
        txt = _fmt_tick(v, xspan)
        draw.text((px - _tw(draw, txt, font) / 2.0, y1 + 9), txt, font=font,
                  fill=(196, 205, 216))
    for v in yt:
        py = y_of(v)
        draw.line([(x0, py), (x1, py)], fill=(58, 66, 78), width=1)
        draw.line([(x0 - 6, py), (x0, py)], fill=(190, 198, 210), width=1)
        txt = ("%d" % round(v)) if y_int else _fmt_tick(v, yspan)
        draw.text((x0 - 10 - _tw(draw, txt, font), py - 8), txt, font=font,
                  fill=(196, 205, 216))
    draw.rectangle([x0, y0, x1, y1], outline=(150, 162, 178), width=1)
    draw.text(((x0 + x1) / 2.0 - _tw(draw, xlab, font) / 2.0, y1 + 32), xlab,
              font=font, fill=(214, 222, 232))
    draw.text((x0, y0 - 20), ylab, font=font, fill=(214, 222, 232))
    if y_zero is not None and y0 <= y_of(y_zero) <= y1:
        py = y_of(y_zero)
        draw.line([(x0, py), (x1, py)], fill=(236, 178, 76), width=1)
        if zero_label:
            draw.text((x1 - _tw(draw, zero_label, font) - 6, py + 3), zero_label,
                      font=font, fill=(240, 196, 116))

def _draw_colorbar(base, draw, cbar, lut, lo, hi, db_label, font):
    """Vertical dB scale labelled at its floor, midpoint and ceiling."""
    x0, y0, x1, y1 = cbar
    w, h = max(x1 - x0 - 2, 1), max(y1 - y0 - 2, 1)
    grad = np.ascontiguousarray(np.repeat(lut[::-1][:, None, :], w, axis=1))
    base.paste(Image.fromarray(grad, "RGB").resize((w, h), Image.BILINEAR),
               (x0 + 1, y0 + 1))
    draw.rectangle([x0, y0, x1, y1], outline=(150, 162, 178), width=1)
    for frac, val in ((0.0, hi), (0.5, (lo + hi) / 2.0), (1.0, lo)):
        py = int(round(y0 + frac * (y1 - y0 - 1)))
        draw.line([(x1, py), (x1 + 5, py)], fill=(190, 198, 210), width=1)
        draw.text((x1 + 8, py - 8), "%.1f" % val, font=font, fill=(206, 214, 224))
    draw.text((x0 - 4, y0 - 18), db_label, font=font, fill=(206, 214, 224))

def _draw_panel(base, rect, headers, aligns, cells, widths, font, tdict,
                floor_db, peak_snr, truncated, alpha=214):
    """Semi-transparent parameter table with its noise-floor footer."""
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rounded_rectangle(list(rect), radius=7, fill=(6, 9, 14, alpha),
                           outline=(96, 190, 235, 255), width=1)
    pad, row_h = 10, 19
    tx, ty = rect[0] + pad, rect[1] + pad
    for c, hdr in enumerate(headers):
        dx = tx + widths[c] - 14 - _tw(draw, hdr, font) if aligns[c] == "right" else tx
        draw.text((dx, ty), hdr, font=font, fill=(146, 214, 246))
        tx += widths[c]
    ty += 20
    draw.line([(rect[0] + pad, ty), (rect[2] - pad, ty)], fill=(96, 190, 235), width=1)
    ty += 5
    for line in cells:
        tx = rect[0] + pad
        for c, txt in enumerate(line):
            dx = tx + widths[c] - 14 - _tw(draw, txt, font) if aligns[c] == "right" else tx
            draw.text((dx, ty), txt, font=font, fill=(242, 246, 251))
            tx += widths[c]
        ty += row_h
    ty += 3
    draw.line([(rect[0] + pad, ty), (rect[2] - pad, ty)], fill=(96, 190, 235), width=1)
    draw.text((rect[0] + pad, ty + 4),
              tdict["noise"] % floor_db + " / " + (tdict["peak"] % peak_snr),
              font=font, fill=(186, 200, 214))
    if truncated:
        draw.text((rect[0] + pad, ty + 21), tdict["more"] % truncated, font=font,
                  fill=(255, 176, 102))
    base.paste(overlay, (0, 0), overlay)

def _draw_targets(draw, marks, font):
    """Scope markers + index labels; labels fan out for shared cells."""
    for r, px, py in marks:
        draw.ellipse([px - 6, py - 6, px + 6, py + 6], outline=(255, 208, 84), width=1)
        k = r["index"] % 4
        dx, dy = ((8, -16), (8, 2), (-18, -16), (-18, 2))[k]
        draw.text((px + dx, py + dy), str(r["index"]), font=font,
                  fill=(255, 226, 150), stroke_width=2, stroke_fill=(10, 12, 16))

def _render_map(power, range_axis, y_axis, kind, targets, meta, size, dyn_db,
                cmap, y_deg=False, annotate=True):
    meta = dict(meta or {})
    db = _to_db(power)
    lo = float(np.median(db))
    hi = lo + float(dyn_db)
    t = np.clip((db - lo) / float(dyn_db), 0.0, 1.0)
    idx = np.rint(np.power(t, _DISPLAY_GAMMA) * 255.0).astype(np.uint8)
    lut = cmap_lut(cmap)
    all_rows = _targets(targets)
    # annotate=False is the "clean" product: same picture, no table and no
    # target markers, so a paper can show the sensor image next to the annotated
    # one and a training loader gets the map without painted overlay.
    rows = all_rows[:8] if annotate else []
    truncated = max(len(all_rows) - len(rows), 0)

    font, cjk = _load_font(14 if size[0] >= 1100 else 12)
    font_small, _ = _load_font(12 if size[0] >= 1100 else 11)
    tdict = _text(cjk)
    headers, aligns, cells = _panel_model(rows, kind, tdict)
    W, H = int(size[0]), int(size[1])
    pw, ph, widths = _panel_geometry(ImageDraw.Draw(Image.new("RGB", (8, 8))),
                                     font, headers, cells)
    ph += 17 if truncated else 0        # room for the "N more" line
    pw, ph = min(pw, int(W * 0.48)), min(ph, int(H * 0.55))

    rmin, rmax = float(range_axis[0]), float(range_axis[-1])
    ymin, ymax = float(y_axis[0]), float(y_axis[-1])
    n_r, n_y = db.shape

    def _maps(plot):
        x0, y0, x1, y1 = plot
        return (lambda r: x0 + (np.asarray(r, float) - rmin) * ((x1 - x0) / (rmax - rmin)),
                lambda v: y1 - (np.asarray(v, float) - ymin) * ((y1 - y0) / (ymax - ymin)))

    def _marks(plot):
        x0, y0, x1, y1 = plot
        out = []
        for r in rows:
            vy = r["v_r_mps"] if kind == "rd" else r["az_deg"]
            if r["range_m"] is None or vy is None or (kind == "ra" and not y_deg):
                continue
            rv, vv = float(r["range_m"]), float(vy)
            if rmin <= rv <= rmax and ymin <= vv <= ymax:
                out.append((r, float(x0 + (rv - rmin) / (rmax - rmin) * (x1 - x0)),
                            float(y1 - (vv - ymin) / (ymax - ymin) * (y1 - y0))))
        return out

    def _scene(plot):
        """Hot pixels of the map, target markers and the brightest cell."""
        x0, y0, x1, y1 = plot
        yy, xx = np.nonzero(t >= 0.62)
        hot = np.stack([x0 + (xx + 0.5) / n_r * (x1 - x0),
                        y1 - (yy + 0.5) / n_y * (y1 - y0)], axis=1) \
            if xx.size else np.empty((0, 2))
        peak = np.unravel_index(int(np.argmax(db)), db.shape)
        peak_px = (x0 + (peak[0] + 0.5) / n_r * (x1 - x0),
                   y1 - (peak[1] + 0.5) / n_y * (y1 - y0))
        return hot, _marks(plot), peak_px

    plot, cbar = _layout(size, bool(meta.get("scene")))
    x_of, y_of = _maps(plot)
    hot, marks, peak_px = _scene(plot)
    # No table means the map gets the full width back: a reserved gutter with
    # nothing in it reads as a rendering fault, not as "clean".
    panel, mode = None, "clean"
    if annotate:
        panel = (plot[2] - 12 - pw, plot[1] + 12, plot[2] - 12, plot[1] + 12 + ph)
        # Prefer a corner overlay (a real scope keeps the map full size), but
        # move the table into a reserved gutter the moment it would hide a
        # target or the brightest echo: a parameter table over the return it
        # describes is worse than a narrower map.
        inset_ok = (panel[0] >= plot[0] + 12 and panel[3] <= plot[3] - 12
                    and _clear_of(panel, hot, 2.0)
                    and _clear_of(panel, [(m[1], m[2]) for m in marks], 24.0))
        mode = "inset"
        if not inset_ok:
            plot, cbar = _layout(size, bool(meta.get("scene")), panel_w=pw)
            x_of, y_of = _maps(plot)
            hot, marks, peak_px = _scene(plot)
            panel = (W - 26 - pw, plot[1], W - 26, plot[1] + ph)
            mode = "gutter"

    base = Image.new("RGB", (W, H), (12, 15, 20))
    draw = ImageDraw.Draw(base)
    # Rows run from the highest y-value at the top, so the axis is flipped for
    # display while the mapping formula above stays the single source of truth.
    disp = np.ascontiguousarray(lut[idx.T[::-1, :]])
    x0, y0, x1, y1 = plot
    tw, th = max(x1 - x0, 1), max(y1 - y0, 1)
    # Decimate the range axis by AREA before the vertical stretch: bilinear
    # downsampling of a fine speckle field aliases into long vertical filaments
    # that read as structure in the map when they are only the display
    # resampling. A 1024-bin map in a 635-px gutter is exactly that case.
    src = Image.fromarray(disp, "RGB")
    if src.width > tw:
        src = src.resize((tw, src.height), Image.BOX)
    base.paste(src.resize((tw, th), Image.BILINEAR), (x0, y0))

    parts = []
    band, fc = meta.get("band"), meta.get("fc_ghz")
    if band and fc is not None:
        parts.append(str(band) + " " + tdict["band"] + " / %.1f GHz" % float(fc))
    elif band or fc is not None:
        parts.append((str(band) + " " + tdict["band"]) if band
                     else "%.1f GHz" % float(fc))
    if meta.get("range_res_m"):
        parts.append(tdict["rres"] + " %.2f m" % float(meta["range_res_m"]))
    res = meta.get("vel_res_mps") if kind == "rd" else meta.get("az_res_deg")
    if res:
        parts.append((tdict["vres"] + " %.3f m/s" % float(res)) if kind == "rd"
                     else (tdict["ares"] + " %.3f°" % float(res)))
    if meta.get("frame") is not None:
        parts.append(tdict["frame"] % int(meta["frame"]))
    title = [tdict["sep"].join([tdict["rd"] if kind == "rd" else tdict["ra"]] + parts)]
    if meta.get("scene"):
        title.append(tdict["scene"] + " " + str(meta["scene"]))
    _draw_title(draw, title, plot[0], W - 2 * plot[0])

    xt = _nice_ticks(rmin, rmax, 7)
    if y_deg:
        yt, ylab = _nice_ticks(ymin, ymax, 6), tdict["az"]
    else:
        yt = [float(v) for v in _nice_ticks(ymin, ymax, 5)]
        ylab = tdict["beam"] if kind == "ra" else tdict["vel"]
    zero, zlab = None, None
    if kind == "rd" and ymin <= 0.0 <= ymax:
        zero, zlab = 0.0, tdict["zero_v"]
    elif kind == "ra" and y_deg and ymin <= 0.0 <= ymax:
        zero, zlab = 0.0, tdict["zero_az"]
    _draw_axes(draw, plot, xt, yt, x_of, y_of, tdict["x"], ylab, font_small,
               rmax - rmin, ymax - ymin, y_int=not y_deg, y_zero=zero,
               zero_label=zlab)

    if annotate:
        _draw_targets(draw, marks, font)
    _draw_colorbar(base, draw, cbar, lut, lo, hi, tdict["db"], font_small)
    if annotate:
        _draw_panel(base, panel, headers, aligns, cells, widths, font, tdict, lo,
                    float(np.max(db) - lo), truncated)

    return dict(rgb=np.asarray(base, dtype=np.uint8),
                plot=tuple(int(v) for v in plot),
                panel=(tuple(int(v) for v in panel) if panel else None),
                panel_mode=mode,
                annotate=bool(annotate),
                range_px=lambda r: x_of(r),
                doppler_px=lambda v: y_of(v),
                beam_px=lambda v: y_of(v),      # RA parity with doppler_px
                db_window=(lo, hi),
                target_rows=rows,
                cjk=bool(cjk),
                title=title,
                y_label=ylab,
                ticks=dict(x=xt, y=yt),
                peak_px=tuple(float(v) for v in peak_px))

def render_rd(power, range_axis, doppler_axis, targets=None, meta=None,
              size=(1280, 720), dyn_db=30.0, cmap="turbo", annotate=True):
    """Render one range-Doppler frame as an oscilloscope-style RGB figure.

    power is (n_range, n_doppler) in watts and doppler_axis is positive
    closing speed in m/s, drawn upward. The window is anchored on the median
    (noise floor) and is dyn_db wide. Targets accept name/label/cls plus
    range(_m), v_r(_mps), az(_deg), snr(_db); meta accepts band, fc_ghz,
    range_res_m, vel_res_mps, frame and scene.
    """
    power = np.asarray(power, dtype=float)
    r_ax = _axis(range_axis, "range_axis")
    d_ax = _axis(doppler_axis, "doppler_axis")
    if power.shape != (r_ax.size, d_ax.size):
        raise ValueError("power %r does not match range/doppler axes (%d, %d)"
                         % (power.shape, r_ax.size, d_ax.size))
    return _render_map(power, r_ax, d_ax, "rd", targets, meta, size, dyn_db, cmap,
                       annotate=annotate)

def render_ra(range_angle, range_axis, beam_axis_deg, targets=None, meta=None,
              size=(1280, 800), dyn_db=45.0, cmap="turbo", annotate=True):
    """Render one range-azimuth frame; same return contract as render_rd.

    range_angle is (n_beams, n_range) unless meta['ra_layout'] is
    'range_beams'. A strictly increasing beam_axis_deg becomes the y axis in
    degrees; any other vector (or None) falls back to beam index, because a
    2-D beam raster has no single angle per row. beam_px mirrors doppler_px
    so callers can share code.
    """
    ra = np.asarray(range_angle, dtype=float)
    r_ax = _axis(range_axis, "range_axis")
    meta = dict(meta or {})
    layout = str(meta.get("ra_layout") or "beams_range").lower()
    if layout in ("range_beams", "range,beam"):
        if ra.shape[0] != r_ax.size:
            raise ValueError("range_angle[0] must equal the range axis length")
        ra = ra.T
    elif ra.ndim != 2 or ra.shape[1] != r_ax.size:
        if ra.ndim == 2 and ra.shape[0] == r_ax.size:
            ra = ra.T
        else:
            raise ValueError("range_angle shape %r does not end in %d range bins"
                             % (ra.shape, r_ax.size))
    beams = int(ra.shape[0])
    y_deg = False
    y_axis = np.arange(beams, dtype=float)
    if beam_axis_deg is not None:
        angles = np.asarray(beam_axis_deg, dtype=float).ravel()
        if angles.size == beams and np.all(np.isfinite(angles)) \
                and np.all(np.diff(angles) > 0):
            y_axis, y_deg = angles, True
    # _render_map keeps x on the first axis, so hand it (range, beam); the map
    # itself stays (beam, range) as the caller and every RA writer store it.
    return _render_map(ra.T, r_ax, y_axis, "ra", targets, meta, size, dyn_db, cmap,
                       y_deg=y_deg, annotate=annotate)
