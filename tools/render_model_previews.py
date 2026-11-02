"""Render a preview PNG for every model in assets/models.

  python tools/render_model_previews.py           # all models
  python tools/render_model_previews.py --force   # re-render existing ones

Two stages, like the fetch pipeline: this script is the plain-Python driver,
the rendering itself happens inside Blender via tools/previews.blender.py
(Workbench engine, transparent film, three-quarter view). Each preview lands in
assets/models/previews/<name>.png and is then copied next to its OBJ as
<name>.png, flattened onto white and quantised, so browsing the folder or the
GitHub web UI shows the shape without opening a modelling tool.

Run tools/make_model_sheets.py afterwards to rebuild the per-category sheets.
"""
import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "assets", "models")
PREV = os.path.join(MODELS, "previews")
BLENDER = os.path.join(ROOT, "third_party", "blender", "blender.exe")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-render existing previews")
    args = ap.parse_args()

    if not os.path.exists(BLENDER):
        raise SystemExit("Blender not found at %s" % BLENDER)

    catalog = os.path.join(MODELS, "catalog.json")
    if not os.path.exists(catalog):
        raise SystemExit("missing %s" % catalog)
    records = json.load(open(catalog, encoding="utf-8"))
    os.makedirs(PREV, exist_ok=True)

    if args.force:
        todo = records
    else:
        todo = [r for r in records
                if not os.path.exists(os.path.join(PREV, r["name_cn"] + ".png"))]
    if not todo:
        print("[previews] all %d previews already rendered" % len(records))
    else:
        queue = os.path.join(PREV, "_queue.json")
        with open(queue, "w", encoding="utf-8") as fh:
            json.dump(todo, fh, ensure_ascii=False)
        cmd = [BLENDER, "-b", "--factory-startup",
               "-P", os.path.join(ROOT, "tools", "previews.blender.py"),
               "--", queue, PREV]
        print("[previews] rendering %d models ..." % len(todo))
        subprocess.run(cmd, check=True)

    # copy next to each OBJ, flattened to white so the folder view is legible
    from PIL import Image
    placed = 0
    for rec in records:
        src = os.path.join(PREV, rec["name_cn"] + ".png")
        if not os.path.exists(src):
            continue
        dst = os.path.join(ROOT, rec["file"].replace(".obj", ".png"))
        im = Image.open(src).convert("RGBA")
        flat = Image.new("RGB", im.size, (255, 255, 255))
        flat.paste(im, mask=im.split()[3])
        flat = flat.quantize(colors=200, method=Image.MEDIANCUT, dither=Image.NONE)
        tmp = dst + ".tmp"
        flat.save(tmp, format="PNG", optimize=True)
        os.replace(tmp, dst)
        rec["preview"] = rec["file"].replace(".obj", ".png")
        placed += 1

    with open(catalog, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(records, fh, ensure_ascii=False, indent=1)
    print("[previews] %d PNGs placed beside their OBJ" % placed)


if __name__ == "__main__":
    sys.exit(main())