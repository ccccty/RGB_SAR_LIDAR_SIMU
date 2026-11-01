#!/usr/bin/env python
"""Shrink the web payload: PNG -> WebP, thumbnails, and a lazy point-cloud split.

The site is a static folder that is committed to the repo, so every byte here is
paid for twice: once on the wire and once in git.  The raw export is 83 MB of
PNGs plus a 646 KB manifest whose point-cloud arrays are 69% of it, against 23 KB
of gzip for everything else.  This module does three things and nothing else:

 1. rewrites every PNG under data/ as WebP (quality per modality, grayscale
    kept grayscale so radar / SAR tones do not shift),
 2. writes a 420 px thumbnail per image into data/_t/ for the filmstrip,
 3. moves each frame's point cloud out of manifest.json into data/cloud/<id>.json
    (plus _index.json with per-scene counts) so the first paint only ships what
    the first tab needs.  boot.js -- the double-click / air-gapped copy -- keeps
    the full payload on purpose.

Idempotent: re-running after a conversion finds no PNGs, keeps the existing
cloud files and just re-reports sizes.  tools/export_web.py calls optimize()
at the end so a fresh export lands in the same shape.
"""
import argparse
import json
import os
import sys

from PIL import Image

# Lossy quality per modality.  Optical/IR are photographic.  Radar and SAR are
# single-channel statistical data where speckle is the signal, so they get a
# higher quality: inventing or smoothing speckle would change what a detector
# sees.
Q = {
    "optical": 80, "optical_wide": 80,
    "mwir": 82, "lwir": 82, "mwir_wide": 82, "lwir_wide": 82,
    "radar_ra": 88, "radar": 88, "sar": 88,
    "lidar_range": 90, "lidar_intensity": 90,
}
DEFAULT_Q = 84


def q_for(rel):
    parts = rel.replace("\\", "/").split("/")
    mod = parts[-2] if len(parts) >= 2 else ""
    return Q.get(mod, DEFAULT_Q)


def to_webp(src, dst, max_w, quality):
    im = Image.open(src)
    w, h = im.size
    if max_w and w > max_w:
        im = im.resize((max_w, max(1, int(round(h * max_w / float(w))))), Image.LANCZOS)
    if im.mode in ("I;16", "I", "F"):
        im = im.convert("L")
    elif im.mode not in ("RGB", "L"):
        im = im.convert("RGB")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    im.save(dst, "WEBP", quality=quality, method=6)
    return os.path.getsize(dst), im.size


def thumb(src, dst, width, quality):
    im = Image.open(src)
    w, h = im.size
    if w > width:
        im = im.resize((width, max(1, int(round(h * width / float(w))))), Image.LANCZOS)
    if im.mode not in ("RGB", "L"):
        im = im.convert("RGB")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    im.save(dst, "WEBP", quality=max(60, quality - 18), method=6)
    return os.path.getsize(dst)


def _load_existing_clouds(data):
    clouds = {}
    cdir = os.path.join(data, "cloud")
    if os.path.isdir(cdir):
        for f in sorted(os.listdir(cdir)):
            if not f.endswith(".json") or f.startswith("_"):
                continue
            with open(os.path.join(cdir, f), encoding="utf-8") as fh:
                clouds[os.path.splitext(f)[0]] = json.load(fh)
    return clouds


def optimize(data="data", thumb_w=420, max_w=1280, keep_png=False, dry_run=False,
             quiet=False):
    mpath = os.path.join(data, "manifest.json")
    bpath = os.path.join(data, "boot.js")
    with open(mpath, encoding="utf-8") as fh:
        man = json.load(fh)

    pngs = []
    for dirpath, _, files in os.walk(data):
        if os.path.basename(dirpath) in ("_t", "cloud"):
            continue
        for f in files:
            if f.lower().endswith(".png"):
                pngs.append(os.path.join(dirpath, f))
    pngs.sort()

    before = sum(os.path.getsize(p) for p in pngs)
    after = tsum = 0
    n = 0
    for p in pngs:
        rel = os.path.relpath(p, data).replace("\\", "/")
        dst = os.path.join(data, os.path.splitext(rel)[0] + ".webp")
        tdst = os.path.join(data, "_t", os.path.splitext(rel)[0] + ".webp")
        if dry_run:
            continue
        sz, _ = to_webp(p, dst, max_w, q_for(rel))
        after += sz
        tsum += thumb(p, tdst, thumb_w, q_for(rel))
        if not keep_png:
            os.remove(p)
        n += 1

    # ---- point clouds: out of the wire manifest, into one file per scene ----
    clouds = _load_existing_clouds(data)
    for s in man.get("scenes", []):
        c = clouds.get(s["id"], {})
        for fr in s.get("frames", []):
            if "cloud" in fr:
                c[str(fr.get("i", len(c)))] = fr.pop("cloud")
        if c:
            clouds[s["id"]] = c

    full = json.loads(json.dumps(man))          # boot.js keeps everything
    lean = json.loads(json.dumps(man))          # the wire copy drops the clouds

    if not dry_run:
        cdir = os.path.join(data, "cloud")
        os.makedirs(cdir, exist_ok=True)
        index = {}
        for sid, c in clouds.items():
            with open(os.path.join(cdir, sid + ".json"), "w", encoding="utf-8") as fh:
                json.dump(c, fh, ensure_ascii=False, separators=(",", ":"))
            index[sid] = dict(frames=len(c),
                              points=sum(len(v.get("x") or []) for v in c.values()))
        with open(os.path.join(cdir, "_index.json"), "w", encoding="utf-8") as fh:
            json.dump(index, fh, ensure_ascii=False, separators=(",", ":"), sort_keys=True)

        for s in full["scenes"]:
            c = clouds.get(s["id"], {})
            for fr in s["frames"]:
                fr["cloud"] = c.get(str(fr.get("i", "")),
                                    dict(n=0, shown=0, x=[], y=[], z=[], i=[], o=[],
                                         hit=0.0, maxr=0))

        with open(mpath, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(lean, ensure_ascii=False, separators=(",", ":")).replace('.png"', '.webp"'))

        bands = {}
        bpath_json = os.path.join(data, "radar_bands.json")
        if os.path.exists(bpath_json):
            with open(bpath_json, encoding="utf-8") as fh:
                bands = json.load(fh)
        with open(bpath, "w", encoding="utf-8") as fh:
            fh.write("/* generated by tools/export_web.py + tools/opt_images.py - do not edit */\n")
            fh.write("window.SIMKIT_DATA = ")
            fh.write(json.dumps(dict(manifest=full, bands=bands),
                                ensure_ascii=False, separators=(",", ":")).replace('.png"', '.webp"'))
            fh.write(";\n")

    if not quiet:
        mb = lambda v: v / 1048576.0
        print("images   : %d files  %.1f MB -> %.1f MB (webp)  + %.1f MB thumbs"
              % (n, mb(before), mb(after), mb(tsum)))
        print("manifest : cloud split into %d scene files" % len(clouds))
        if not dry_run:
            tot = 0
            for dirpath, _, files in os.walk(data):
                for f in files:
                    tot += os.path.getsize(os.path.join(dirpath, f))
            print("data/ now: %.1f MB" % mb(tot))
    return dict(pngs=n, before=before, after=after, thumbs=tsum)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--thumb", type=int, default=420)
    ap.add_argument("--max-w", type=int, default=1280)
    ap.add_argument("--keep-png", action="store_true",
                    help="keep the source PNGs next to the WebP (doubles repo size)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    optimize(a.data, a.thumb, a.max_w, a.keep_png, a.dry_run)


if __name__ == "__main__":
    sys.exit(main())
