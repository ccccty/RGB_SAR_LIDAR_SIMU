# -*- coding: utf-8 -*-
"""Pull three 1110 preset runs into data/ as first-class site scenes.

The site serves frames from data/<id>/<mod>/<frame>.webp with 320 px thumbs in
data/_t/<id>/<mod>/. The 1110 runs keep their originals under
dataest/1110/{air,land,sea,seav2}/p_*, so this script only re-encodes: PNG ->
WebP, plus a 320 px thumb, no re-render.

Usage:  python tools/export_1110_cards.py
"""
import os

from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "data")
SRC = r"D:/work/多模态数据集生成/dataest/1110"
FRAMES = [0, 7, 15, 22, 29]

# dest id -> (source run dir, {mod: subdir})
SCENES = [
    ("s1110_air_skyclear", "air/p_air_skyclear_passive", {
        "optical": "images/optical", "mwir": "images/ir_preview_mwir",
        "lwir": "images/ir_preview_lwir", "radar": "images/radar",
    }),
    ("s1110_land_industrial", "land/p_land_industrial_passive", {
        "optical": "images/optical", "mwir": "images/ir_preview_mwir",
        "lwir": "images/ir_preview_lwir", "radar": "images/radar",
    }),
    ("s1110_sea_port_wake", "seav2/p_sea_port_passive", {
        "optical": "images/optical", "mwir": "images/ir_preview_mwir",
        "lwir": "images/ir_preview_lwir", "radar": "images/radar",
    }),
]


def save(img, path, width, quality):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    im = img.convert("RGB")
    if im.width != width:
        im = im.resize((width, int(round(im.height * width / float(im.width)))), Image.LANCZOS)
    im.save(path, "WEBP", quality=quality, method=6)
    return os.path.getsize(path)


def main():
    total = 0
    for sid, rel, mods in SCENES:
        run = os.path.join(SRC, rel)
        n = 0
        for mod, sub in mods.items():
            for f in FRAMES:
                pad = "%06d" % f
                src = os.path.join(run, sub, pad + ".png")
                if not os.path.exists(src):
                    print("  missing", src)
                    continue
                img = Image.open(src)
                total += save(img, os.path.join(OUT, sid, mod, pad + ".webp"), img.width, 82)
                total += save(img, os.path.join(OUT, "_t", sid, mod, pad + ".webp"), 320, 70)
                n += 1
        print("%-24s %d files" % (sid, n))
    print("total %.2f MB" % (total / 1024.0 / 1024.0))


if __name__ == "__main__":
    main()
