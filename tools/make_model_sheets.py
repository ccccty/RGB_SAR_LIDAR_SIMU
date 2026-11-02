# -*- coding: utf-8 -*-
"""Compose per-category contact sheets from the rendered previews.

  python tools/make_model_sheets.py

Reads assets/models/catalog.json and the transparent PNGs in
assets/models/previews/, then writes one labelled sheet per category to
assets/models/sheets/<category>.png plus a combined sheets/index.json.

Chinese labels need a CJK font; Windows ships one at msyh.ttc. If no font is
found the sheets are still written, just without text.
"""
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "assets", "models")
PREV = os.path.join(MODELS, "previews")
SHEETS = os.path.join(MODELS, "sheets")

CATEGORY_EN = {
    "旋翼无人机": "Rotary-wing UAV",
    "固定翼无人机": "Fixed-wing UAV",
    "直升机": "Helicopter",
    "民航飞机": "Civil aircraft",
    "军用飞机": "Military aircraft",
    "地面车辆": "Ground vehicle",
    "军用地面车辆": "Military ground vehicle",
    "军舰": "Naval vessel",
    "民用船舶": "Civil vessel",
}

CELL_W, CELL_H = 300, 250      # card size
PAD = 14
COLS = 4
LABEL_H = 46
BG = (250, 251, 252)
CARD = (255, 255, 255)
BORDER = (224, 228, 233)
TEXT = (32, 38, 46)
SUBTEXT = (120, 129, 140)

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
]


def load_font(size):
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def trim(img):
    """Crop the surround so every subject is scaled to the same card."""
    a = np.array(img)
    if a.shape[2] == 4:
        mask = a[:, :, 3] > 8
    else:
        # previews beside the OBJ are flattened onto white
        mask = (a[:, :, :3] < 246).any(axis=2)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return img
    return img.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))


def fit_on_card(subject, w, h):
    s = min(w / subject.width, h / subject.height)
    new = (max(1, int(subject.width * s)), max(1, int(subject.height * s)))
    return subject.resize(new, Image.LANCZOS)


def build_sheet(category, records, font_name, font_sub):
    cols = min(COLS, max(1, len(records)))
    rows = (len(records) + cols - 1) // cols
    width = PAD + cols * (CELL_W + PAD)
    header = 66
    height = header + PAD + rows * (CELL_H + PAD)

    sheet = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(sheet)
    draw.text((PAD + 4, 20), category, font=font_name, fill=TEXT)
    en = CATEGORY_EN.get(category, "")
    draw.text((PAD + 8 + draw.textlength(category, font=font_name), 30),
              "  %s · %d 个模型" % (en, len(records)), font=font_sub, fill=SUBTEXT)

    for i, rec in enumerate(records):
        r, c = divmod(i, cols)
        x = PAD + c * (CELL_W + PAD)
        y = header + PAD + r * (CELL_H + PAD)

        draw.rounded_rectangle([x, y, x + CELL_W, y + CELL_H], radius=8,
                               fill=CARD, outline=BORDER, width=1)

        # prefer the RGBA render (transparent surround trims cleanly), fall back
        # to the white-flattened copy that ships beside the OBJ
        cand = [os.path.join(PREV, rec["name_cn"] + ".png"),
                os.path.join(ROOT, rec.get("preview", ""))]
        png = next((p for p in cand if p and os.path.exists(p)), None)
        if png:
            subject = trim(Image.open(png).convert("RGBA"))
            subject = fit_on_card(subject, CELL_W - 28, CELL_H - LABEL_H - 20)
            sheet.paste(subject, (x + (CELL_W - subject.width) // 2,
                                  y + 12 + (CELL_H - LABEL_H - 20 - subject.height) // 2),
                        subject)

        d = rec["dims_m"]
        draw.text((x + 12, y + CELL_H - LABEL_H + 4), rec["name_cn"],
                  font=font_name, fill=TEXT)
        draw.text((x + 12, y + CELL_H - LABEL_H + 26),
                  "%.1f × %.1f × %.1f m" % (d[0], d[1], d[2]),
                  font=font_sub, fill=SUBTEXT)
    return sheet


def main():
    catalog = json.load(open(os.path.join(MODELS, "catalog.json"), encoding="utf-8"))
    os.makedirs(SHEETS, exist_ok=True)
    if not os.path.isdir(PREV):
        raise SystemExit("no previews; run tools/render_model_previews.py first")

    font_name = load_font(19)
    font_sub = load_font(13)

    by_cat = {}
    for rec in catalog:
        by_cat.setdefault(rec["category"], []).append(rec)

    index = []
    for cate, recs in by_cat.items():
        sheet = build_sheet(cate, recs, font_name, font_sub)
        out = os.path.join(SHEETS, cate + ".png")
        sheet.save(out, optimize=True)
        index.append(dict(category=cate, category_en=CATEGORY_EN.get(cate, ""),
                          count=len(recs),
                          file="assets/models/sheets/%s.png" % cate,
                          bytes=os.path.getsize(out)))
        print("[sheet] %-10s %2d models -> %6.1f KB"
              % (cate, len(recs), os.path.getsize(out) / 1024))

    with open(os.path.join(SHEETS, "index.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(index, fh, ensure_ascii=False, indent=1)
    print("[sheet] %d sheets" % len(index))


if __name__ == "__main__":
    main()