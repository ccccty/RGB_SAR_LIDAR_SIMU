
"""Registered optical / MWIR / radar-RD contact sheet.

Strict grid: one row per frame, one column per modality, every cell the same
size. Each cell holds the target region magnified (primary, where a 0.35 m body
is actually legible) and the full frame as a thumbnail (context, proves where in
the field the target sits). Single source of truth for the boxes is the run's
label JSON, so the sheet cannot drift from the dataset.
"""
import json, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = os.path.join(ROOT, 'out', '1102')
OUT = os.path.join(ROOT, 'out', 'sheet_1102_registered.png')

FRAMES = [0, 6, 12, 19]
PANELS = [('optical',  'images/optical',        'labels/optical', 'boxes',    (255, 96, 96)),
          ('ir_mwir',  'images/ir_preview_mwir', 'labels/ir_mwir', 'boxes',    (96, 216, 255)),
          ('radar_rd', 'images/radar',           'labels/radar',   'boxes_rd', (255, 214, 74))]

BG      = (12, 13, 17)
GRID    = (46, 49, 58)
ZW, ZH  = 330, 248          # zoom tile
TW, TH = 132, 99          # full-frame thumbnail
LABEL_H = 20
CELL_W  = ZW + 10 + TW
CELL_H  = ZH + LABEL_H
GAP     = 10
HEAD_H  = 30


def enhance(im, gain=1.0, gamma=1.0):
    """Display-only tone curve: out = ((in * gain) ** gamma).

    Deliberately NOT a percentile stretch. An optical frame is ~99 % sky at
    grey 85-95; anchoring on percentiles maps that narrow band across the whole
    range and clips the airframe (grey 17-70) to pure black, which erases the
    very thing the sheet exists to show. A gain plus gamma is predictable and
    keeps the target/sky ratio intact.
    """
    a = np.asarray(im.convert('RGB'), dtype=np.float32) * gain
    b = np.clip(a / 255.0, 0.0, 1.0) ** gamma
    return Image.fromarray((b * 255.0 + 0.5).astype(np.uint8))


def font(sz):
    for p in (r'C:\Windows\Fonts\consola.ttf', r'C:\Windows\Fonts\arial.ttf'):
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, sz)
            except Exception:
                pass
    return ImageFont.load_default()


F_LBL = font(17)
F_HD  = font(20)
F_TAG = font(15)


def load(fid, name, idir, ldir, key, col):
    lab = json.load(open(os.path.join(RUN, ldir, '%06d.json' % fid), encoding='utf-8'))
    bx = (lab.get(key) or {})
    im = Image.open(os.path.join(RUN, idir, '%06d.png' % fid))
    im = im.convert('RGB') if name != 'radar_rd' else im.convert('L').convert('RGB')
    # radar's noise floor is a few counts out of 255: stretch hard or the whole
    # panel is black. Camera frames only need a mild lift off the sky pedestal.
    # radar: gamma < 1 lifts the noise floor into view (a scope's brightness knob).
    # cameras: linear gain only, so the dark airframe keeps its contrast to sky.
    im = enhance(im, gain=1.0, gamma=0.45) if name == 'radar_rd' else enhance(im, gain=1.45, gamma=1.0)
    return im, bx


def roi(bx, W, H, name):
    """Crop window around the target, sized so the target is a clear feature."""
    if not bx:
        return None
    b = max(bx.values(), key=lambda v: (v['x1'] - v['x0']) * (v['y1'] - v['y0']))
    cx, cy = (b['x0'] + b['x1']) / 2, (b['y0'] + b['y1']) / 2
    if name == 'radar_rd':
        hw, hh = 20, 16          # a detection is 1-3 cells; keep the crop tight
    else:
        s = max(b['x1'] - b['x0'], b['y1'] - b['y0'], 24) * 1.9
        hw, hh = s, s * 0.75
        hw, hh = min(hw, cx, W - cx), min(hh, cy, H - cy)
    x0, y0 = int(max(0, cx - hw)), int(max(0, cy - hh))
    x1, y1 = int(min(W, cx + hw)), int(min(H, cy + hh))
    return x0, y0, max(x0 + 8, x1), max(y0 + 8, y1)


def draw_box(dr, b, x0, y0, w, h, step, col, thick=2):
    """Rectangle for a real extent, crosshair for a point target."""
    bw, bh = (b['x1'] - b['x0']) * step, (b['y1'] - b['y0']) * step
    if bw < 8 or bh < 8:
        cx, cy = ((b['x0'] + b['x1']) / 2 - x0) * step, ((b['y0'] + b['y1']) / 2 - y0) * step
        k = 9
        dr.line([cx - k, cy, cx + k, cy], fill=col, width=thick)
        dr.line([cx, cy - k, cx, cy + k], fill=col, width=thick)
        dr.ellipse([cx - 2, cy - 2, cx + 2, cy + 2], outline=col, width=1)
    else:
        dr.rectangle([(b['x0'] - x0) * step, (b['y0'] - y0) * step,
                      (b['x1'] - x0) * step, (b['y1'] - y0) * step],
                     outline=col, width=thick)


cols = len(PANELS)
rows = len(FRAMES)
SW = GAP + cols * (CELL_W + GAP)
SH = GAP + HEAD_H + rows * (CELL_H + GAP)
sheet = Image.new('RGB', (SW, SH), BG)
dr = ImageDraw.Draw(sheet)

# header
for j, (name, _i, _l, _k, col) in enumerate(PANELS):
    x = GAP + j * (CELL_W + GAP)
    dr.text((x + 4, GAP + 6), name, fill=col, font=F_HD)

for i, fid in enumerate(FRAMES):
    cy0 = GAP + HEAD_H + i * (CELL_H + GAP)
    for j, (name, idir, ldir, key, col) in enumerate(PANELS):
        cx0 = GAP + j * (CELL_W + GAP)
        dr.rectangle([cx0, cy0, cx0 + CELL_W, cy0 + CELL_H], outline=GRID, width=1)
        im, bx = load(fid, name, idir, ldir, key, col)
        W, H = im.size

        # --- magnified target region ---
        rb = roi(bx, W, H, name)
        if rb is None:
            zoom = Image.new('RGB', (ZW, ZH), BG)
        else:
            crop = im.crop(rb)
            sc = min(ZW / crop.width, ZH / crop.height)
            zw, zh = max(1, int(crop.width * sc)), max(1, int(crop.height * sc))
            zoom = Image.new('RGB', (ZW, ZH), BG)
            zoom.paste(crop.resize((zw, zh), Image.LANCZOS),
                       ((ZW - zw) // 2, (ZH - zh) // 2))
            # boxes re-drawn in the zoom's own coordinate space
            dz = ImageDraw.Draw(zoom)
            ox, oy = (ZW - zw) // 2, (ZH - zh) // 2
            for b in sorted(bx.values(), key=lambda v: -(v['x1'] - v['x0'])):
                step = sc
                bw = (b['x1'] - b['x0']) * step
                bh = (b['y1'] - b['y0']) * step
                if bw < 8 or bh < 8:
                    px = ox + ((b['x0'] + b['x1']) / 2 - rb[0]) * step
                    py = oy + ((b['y0'] + b['y1']) / 2 - rb[1]) * step
                    k = 10
                    dz.line([px - k, py, px + k, py], fill=col, width=2)
                    dz.line([px, py - k, px, py + k], fill=col, width=2)
                else:
                    dz.rectangle([ox + (b['x0'] - rb[0]) * step, oy + (b['y0'] - rb[1]) * step,
                                  ox + (b['x1'] - rb[0]) * step, oy + (b['y1'] - rb[1]) * step],
                                 outline=col, width=2)
        sheet.paste(zoom, (cx0 + 1, cy0 + 1))

        # --- full frame thumbnail ---
        th = im.copy()
        dt = ImageDraw.Draw(th)
        if rb:
            dt.rectangle(rb, outline=(235, 235, 235), width=max(1, W // 400))
        for b in sorted(bx.values(), key=lambda v: -(v['x1'] - v['x0'])):
            xs = [b['x0'], b['y0'], b['x1'], b['y1']]
            if (xs[2] - xs[0]) < 6 or (xs[3] - xs[1]) < 6:
                px, py = (xs[0] + xs[2]) / 2, (xs[1] + xs[3]) / 2
                k = max(3, W // 90)
                dt.line([px - k, py, px + k, py], fill=col, width=2)
                dt.line([px, py - k, px, py + k], fill=col, width=2)
            else:
                dt.rectangle(xs, outline=col, width=2)
        ts = min(TW / th.width, TH / th.height)
        th2 = th.resize((max(1, int(th.width * ts)), max(1, int(th.height * ts))), Image.LANCZOS)
        tile = Image.new('RGB', (TW, TH), BG)
        tile.paste(th2, ((TW - th2.width) // 2, (TH - th2.height) // 2))
        sheet.paste(tile, (cx0 + ZW + 10, cy0 + 1 + (ZH - TH) // 2))
        dr.rectangle([cx0 + ZW + 10, cy0 + 1 + (ZH - TH) // 2,
                      cx0 + ZW + 10 + TW, cy0 + 1 + (ZH - TH) // 2 + TH],
                     outline=GRID, width=1)

        nb = len(bx)
        dr.text((cx0 + 4, cy0 + ZH + 2), 'f%02d   %d box   %dx%d' % (fid, nb, W, H),
                fill=(216, 220, 230), font=F_LBL)

sheet.save(OUT)
print('wrote %s  %dx%d' % (OUT, sheet.width, sheet.height))
