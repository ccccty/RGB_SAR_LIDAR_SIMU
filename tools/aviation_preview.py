"""Make a labelled QA montage; image crops are explicitly nearest-neighbour zoomed.

Optional preview dependency: Pillow. This does not modify generated sensor data.
"""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", type=Path)
    args = ap.parse_args()
    runs = sorted(p.parent for p in args.path.glob("*/quality.json"))
    if not runs:
        ap.error("run check_aviation.py first")
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 14)
    head_font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 18)
    sheet = Image.new("RGB", (1100, 90+len(runs)*310), "#151b24")
    draw = ImageDraw.Draw(sheet)
    draw.text((15, 12), "Civil aviation QA | 2 models x 3 flight geometries | frame 0", font=head_font, fill="white")
    draw.text((15, 40), "Research assumptions, uncalibrated. Crops: 64 x 64 original pixels, nearest zoom x4.", font=font, fill="#c0cad8")
    for i, run in enumerate(runs):
        meta = json.loads((run/"observations/000000.json").read_text(encoding="utf-8"))
        radar = json.loads((run/"labels/radar/000000.json").read_text(encoding="utf-8"))
        obj = meta["objects"][0]
        cx, cy = next(iter(obj["projections"].values()))["center_px"]
        x, y = int(round(cx)), int(round(cy))
        top = 90+i*310
        text = "%s | %.2f km | closing %+.1f m/s | RD detected: %s" % (
            run.name, obj["range_m"]/1000, obj["closing_speed_mps"], bool(radar["points"]))
        draw.text((15, top), text, font=font, fill="white")
        for col, (title, rel) in enumerate([
            ("RGB crop x4", "images/optical"),
            ("MWIR preview crop x4", "images/ir_preview_mwir"),
            ("LWIR preview crop x4", "images/ir_preview_lwir"),
            ("RD full field (display PNG)", "images/radar")]):
            image = Image.open(run/rel/"000000.png").convert("RGB")
            if col < 3:
                tile = image.crop((x-32, y-32, x+32, y+32)).resize((256, 256), Image.Resampling.NEAREST)
            else:
                tile = ImageOps.pad(image, (256, 256), method=Image.Resampling.NEAREST, color="#080b10")
            left = 15+col*272
            draw.text((left, top+23), title, font=font, fill="#c0cad8")
            sheet.paste(tile, (left, top+45))
    dst = args.path/"preview.jpg"
    sheet.save(dst, quality=92)
    print(dst)


if __name__ == "__main__":
    main()
