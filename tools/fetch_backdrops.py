#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collect real, freely licensed *background* photos for the simulator.

    python tools\fetch_backdrops.py probe     # search Commons, save candidate pool
    python tools\fetch_backdrops.py stage    # download the top N per category
    python tools\fetch_backdrops.py sheet    # contact sheet of the staged images
    python tools\fetch_backdrops.py finalize # copy the picked ones + rebuild catalog

Sources are Wikimedia Commons files whose licence reads CC0 / public domain /
CC BY (CC BY-SA is only a fallback and stays flagged).  Only the 2048 px
rendition is pulled: enough for a background plate, light enough for the repo.

The staged review is deliberate.  A Commons search for "aerial forest" happily
returns a scanned postcard, a cloud deck shot from a plane, or a town called
Forest Hill, and only looking at the pixels tells them apart.  So: probe ->
stage -> eyeball a contact sheet -> record the keepers in picks.json -> finalize.
"""
import argparse
import hashlib
import io
import json
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_DIR = os.path.join(ROOT, "assets", "backdrops")
STAGE_DIR = os.path.join(OUT_DIR, "_stage")
SHEET_DIR = os.path.join(OUT_DIR, "_sheets")
CATALOG = os.path.join(OUT_DIR, "catalog.json")
CANDIDATES = os.path.join(OUT_DIR, "_candidates.json")
MANIFEST = os.path.join(STAGE_DIR, "manifest.json")
PICKS = os.path.join(OUT_DIR, "picks.json")

API = "https://commons.wikimedia.org/w/api.php"
UA = ("RGB_SAR_LIDAR_SIMU-backdrops/1.0 "
      "(simulation background plates; contact: simu@example.org)")
THUMB_WIDTH = 2048                # what we ask Commons for
MAX_WIDTH = 2048                  # what we actually store, whatever it hands back
MIN_W, MIN_H = 1024, 768          # hard floor from the task
# The CDN often answers a 2048 px request with its 3840 px bucket instead, so the
# byte cap has to tolerate a big source that we downscale on the way to disk.
MAX_BYTES = 14 * 1024 * 1024

# Ordered queries per category.  The leading ones are the precise CirrusSearch
# forms (category restricted, "from the air" phrasing, Unsplash CC0 pools);
# the broad trailing ones only pad a category that is still short.
CATEGORY_QUERIES = {
    "forest": [
        'aerial view forest canopy filetype:bitmap',
        'incategory:"Aerial photographs of forests" filetype:bitmap',
        'forest aerial Unsplash filetype:bitmap',
        'Luftbild Wald filetype:bitmap',
        'aerial view of pine forest filetype:bitmap',
        'aerial photograph woodland from above filetype:bitmap',
        'drone view forest filetype:bitmap',
        'aerial view forest filetype:bitmap',
    ],
    "city": [
        'incategory:"Aerial photographs of cities" filetype:bitmap',
        'aerial view city centre filetype:bitmap',
        'city aerial Unsplash filetype:bitmap',
        'Luftbild Stadt filetype:bitmap',
        'aerial photo city skyline filetype:bitmap',
        'drone view city from above filetype:bitmap',
        'aerial view downtown skyline filetype:bitmap',
        'aerial view city filetype:bitmap',
        # second round: rooftop / street-grid views for a nadir-ish plate
        'city rooftops from above filetype:bitmap',
        'aerial photograph city street grid filetype:bitmap',
        'aerial view town houses filetype:bitmap',
    ],
    "port": [
        'incategory:"Aerial photographs of ports and harbours" filetype:bitmap',
        'aerial view container terminal filetype:bitmap',
        'harbour aerial Unsplash filetype:bitmap',
        'Luftbild Hafen filetype:bitmap',
        'aerial view docks cranes filetype:bitmap',
        'container port from above filetype:bitmap',
        'aerial photo harbour breakwater filetype:bitmap',
        'aerial view port filetype:bitmap',
        # second round: the first pass was heavy on black-and-white archive
        # scans, so ask for the big modern container ports by name
        'Rotterdam port aerial filetype:bitmap',
        'Hamburg harbour aerial filetype:bitmap',
        'container terminal aerial photograph filetype:bitmap',
        'aerial view shipyard dry dock filetype:bitmap',
        'cargo terminal from above filetype:bitmap',
    ],
    "island": [
        'incategory:"Aerial photographs of islands" filetype:bitmap',
        'aerial view island coast filetype:bitmap',
        'island aerial Unsplash filetype:bitmap',
        'Luftbild Insel filetype:bitmap',
        'aerial view atoll filetype:bitmap',
        'drone view coastline aerial filetype:bitmap',
        'island from the air filetype:bitmap',
        'aerial view island filetype:bitmap',
    ],
    "open": [
        'ocean waves aerial Unsplash filetype:bitmap',
        'aerial view ocean swells filetype:bitmap',
        'open ocean from aircraft filetype:bitmap',
        'ocean surface waves from above filetype:bitmap',
        'aerial view ocean waves filetype:bitmap',
        'open sea waves filetype:bitmap',
        'Atlantic ocean waves aerial filetype:bitmap',
        'aerial view sea filetype:bitmap',
        # second round: the first pass mostly found coastline and haze,
        # these aim squarely at open water with visible swell texture
        'ocean waves Unsplash filetype:bitmap',
        'sea surface waves aerial photograph filetype:bitmap',
        'ocean swell from the air filetype:bitmap',
        'waves open water texture filetype:bitmap',
        'Pacific ocean aerial view filetype:bitmap',
        'North sea waves filetype:bitmap',
        'ocean water surface photograph filetype:bitmap',
        # third round: the Unsplash CC0 pool on Commons is the only place with
        # genuine open-water plates (no beach, no ship, no horizon haze)
        'Unsplash ocean filetype:bitmap',
        'Unsplash sea waves filetype:bitmap',
        'ocean waves Unsplash filetype:bitmap',
        'ocean surface from height filetype:bitmap',
    ],
}

# Titles that are never a usable background plate.
BLOCK_TITLE = re.compile(
    r"(diagram|line[- ]?drawing|\bdrawing\b|\bsketch\b|orthograph|blueprint|"
    r"schematic|3[- ]?view|\bmap\b|\bmaps\b|\bchart\b|\blogo\b|\bicon\b|"
    r"silhouette|\bposter\b|\bstamp\b|\bcoin\b|\bflag\b|painting|"
    r"engraving|lithograph|\bseal\b|postcard)", re.I)

# Words that mean the camera was in the air.
AERIAL_STRONG = re.compile(
    r"(aerial|airborne|from the air|from above|bird'?s[- ]eye|top[- ]view|"
    r"overhead view|drone|luftbild|luftaufnahme|vue a[eé]rienne|"
    r"vista a[eé]rea|satellite image|air photo)", re.I)
AERIAL_WEAK = re.compile(
    r"(panorama|skyline|looking down|view from|high angle)", re.I)

LICENSE_PAT = {
    "pd": re.compile(r"(public domain|pd[- ]?(us|self|old|art|gov)|no restrictions)", re.I),
    "cc0": re.compile(r"\bcc0\b|creative commons zero", re.I),
    "ccby": re.compile(r"\bcc[ -]?by(?![- ]?sa)\b", re.I),
    "ccbysa": re.compile(r"\bcc[ -]?by[- ]?sa\b", re.I),
}
LICENSE_RANK = {"pd": 3, "cc0": 3, "ccby": 2, "ccbysa": 1}


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def log(msg):
    print(msg, flush=True)


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def strip_html(raw):
    """Commons extmetadata is HTML; keep the readable text only."""
    if not raw:
        return ""
    text = re.sub(r"<[^>]+>", " ", str(raw))
    text = (text.replace("&amp;", "&").replace("&quot;", chr(34))
                .replace("&#039;", chr(39)).replace("&apos;", chr(39))
                .replace("&lt;", "<").replace("&gt;", ">").replace("&nbsp;", " "))
    return re.sub(r"\s+", " ", text).strip()


def slug(text, limit=64):
    text = text.replace("File:", "")
    text = re.sub(r"\.[A-Za-z]{2,5}$", "", text)
    text = text.replace(" ", "_")
    text = re.sub(r"[^\w\-.]+", "", text, flags=re.UNICODE)
    return text[:limit].strip("._-") or "image"


def load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return default


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def api_get(params, retries=4, timeout=45):
    """One Commons API call, retried on transport errors and rate limits."""
    url = API + "?" + urllib.parse.urlencode(params)
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": UA,
                "Accept": "application/json",
                "Accept-Language": "en-US,en;q=0.8",
            })
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = resp.read()
            return json.loads(payload.decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last = "HTTP %s for %s" % (exc.code, params.get("gsrsearch", ""))
            if exc.code in (429, 500, 502, 503, 504):
                time.sleep(3 * (attempt + 1))
                continue
            raise RuntimeError(last) from exc
        except Exception as exc:  # noqa: BLE001 - flaky network is expected
            last = "%s: %s" % (type(exc).__name__, exc)
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(last or "unknown API failure")


def extmeta(info, key):
    meta = (info.get("extmetadata") or {}).get(key) or {}
    return meta.get("value")


def license_kind(raw):
    text = strip_html(raw)
    if not text:
        return ""
    for kind in ("cc0", "ccbysa", "ccby", "pd"):
        if LICENSE_PAT[kind].search(text):
            return kind
    return ""


# --------------------------------------------------------------------------- #
# search
# --------------------------------------------------------------------------- #
def search_images(query, limit=40):
    data = api_get({
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrnamespace": "6",
        "gsrsearch": query,
        "gsrlimit": str(limit),
        "prop": "imageinfo|categories",
        "cllimit": "50",
        "clshow": "!hidden",
        "iiprop": "url|size|mime|extmetadata",
        "iiurlwidth": str(THUMB_WIDTH),
        "iiextmetadatafilter": ("LicenseShortName|Artist|Credit|ImageDescription|"
                                "LicenseUrl|ObjectName|DateTimeOriginal|AttributionRequired"),
    })
    pages = (data.get("query") or {}).get("pages") or {}
    return sorted(pages.values(), key=lambda p: p.get("index", 0))


def candidate_row(page, query):
    info = (page.get("imageinfo") or [{}])[0]
    title = str(page.get("title") or "")
    cats = [str(c.get("title", "")).replace("Category:", "")
            for c in (page.get("categories") or [])]
    lic_raw = strip_html(extmeta(info, "LicenseShortName"))
    desc = strip_html(extmeta(info, "ImageDescription"))
    blob = " ".join([title, desc, " ".join(cats)])
    return {
        "title": title,
        "pageid": page.get("pageid"),
        "query": query,
        "thumburl": info.get("thumburl") or "",
        "url": info.get("url") or "",
        "orig_width": int(info.get("width") or 0),
        "orig_height": int(info.get("height") or 0),
        "bytes": int(info.get("size") or 0),
        "mime": str(info.get("mime") or ""),
        "license": lic_raw,
        "license_kind": license_kind(lic_raw),
        "license_url": strip_html(extmeta(info, "LicenseUrl")),
        "author": (strip_html(extmeta(info, "Artist"))
                   or strip_html(extmeta(info, "Credit"))
                   or "Unknown"),
        "credit": strip_html(extmeta(info, "Credit")),
        "description": desc[:600],
        "source_url": info.get("descriptionurl") or "",
        "categories": cats,
        "aerial": bool(AERIAL_STRONG.search(blob)),
        "aerial_cat": any(re.search(r"aerial|satellite|drone|luftbild", c, re.I)
                          for c in cats),
        "weak": bool(AERIAL_WEAK.search(blob)),
    }


def score(row):
    """Rank candidates: licence first, then the aerial signal, then pixels."""
    s = LICENSE_RANK.get(row["license_kind"], 0) * 100
    if row["aerial"]:
        s += 40
    if row["aerial_cat"]:
        s += 30
    if row["weak"]:
        s += 10
    w, h = row["orig_width"], row["orig_height"]
    if w and h:
        if w > h:
            s += 12
        if w >= 3000 and h >= 2000:
            s += 8
    if row["mime"] != "image/jpeg":
        s -= 15
    if BLOCK_TITLE.search(row["title"]):
        s -= 80
    return s


def probe(categories, per_query=40):
    """Search every query and persist the raw candidate pool."""
    previous = (load_json(CANDIDATES, {}).get("candidates") or {})
    all_rows = {}
    for cat, queries in CATEGORY_QUERIES.items():
        if categories and cat not in categories:
            continue
        for q in queries:
            log("[search] %-6s %s" % (cat, q))
            try:
                pages = search_images(q, limit=per_query)
            except Exception as exc:  # noqa: BLE001
                log("   ! failed: %s" % exc)
                continue
            for page in pages:
                row = candidate_row(page, q)
                if row["title"] and row["thumburl"]:
                    all_rows.setdefault(row["title"], row)
            log("   -> %d candidates" % len(pages))
            time.sleep(0.4)
    by_cat = {}
    for cat, queries in CATEGORY_QUERIES.items():
        if categories and cat not in categories:
            # A partial probe must not wipe the categories it did not visit.
            by_cat[cat] = previous.get(cat, [])
            continue
        rows = [r for r in all_rows.values() if r["query"] in queries]
        rows.sort(key=score, reverse=True)
        by_cat[cat] = rows
        log("\n== %s (%d candidates) ==" % (cat, len(rows)))
        for r in rows[:20]:
            log("%5d %-50s %4dx%-4d %-13s %s%s" % (
                score(r), r["title"][:50], r["orig_width"], r["orig_height"],
                r["license"][:13], r["license_kind"],
                "  AERIAL" if r["aerial"] or r["aerial_cat"] else ""))
    save_json(CANDIDATES, {"generated": now_iso(),
                           "queries": {c: qs for c, qs in CATEGORY_QUERIES.items()},
                           "candidates": by_cat})
    log("\nwrote %s" % CANDIDATES)


# --------------------------------------------------------------------------- #
# download
# --------------------------------------------------------------------------- #
def fetch_thumb(url, retries=4, timeout=120):
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": UA,
                "Accept": "image/jpeg,image/png,*/*",
            })
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read(), resp.headers.get("Content-Type", "")
        except urllib.error.HTTPError as exc:
            last = "HTTP %s" % exc.code
            if exc.code in (429, 500, 502, 503, 504):
                time.sleep(4 * (attempt + 1))
                continue
            raise RuntimeError(last) from exc
        except Exception as exc:  # noqa: BLE001
            last = "%s: %s" % (type(exc).__name__, exc)
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(last or "download failed")


def save_plate(img, path):
    """Write a background plate at no more than MAX_WIDTH px wide."""
    from PIL import Image
    img = img.convert("RGB")
    if img.width > MAX_WIDTH:
        height = max(1, round(img.height * MAX_WIDTH / float(img.width)))
        img = img.resize((MAX_WIDTH, height), Image.LANCZOS)
    img.save(path, "JPEG", quality=92, optimize=True)
    return img.size


def stage_one(row, path):
    """Download + fully decode one candidate into path. Returns (w, h, sha) or raises."""
    from PIL import Image
    raw, ctype = fetch_thumb(row["thumburl"] or row["url"])
    if len(raw) > MAX_BYTES:
        raise RuntimeError("%.1f MB over cap" % (len(raw) / 1048576.0))
    if ctype and not ctype.lower().startswith("image/"):
        raise RuntimeError("content-type %s" % ctype)
    with Image.open(io.BytesIO(raw)) as img:
        img.load()
        w, h = img.size
        if w < MIN_W or h < MIN_H:
            raise RuntimeError("rendition only %dx%d" % (w, h))
        save_plate(img, path)
    with Image.open(path) as chk:
        chk.load()
        w, h = chk.size
    return w, h, hashlib.sha256(raw).hexdigest()


def stage_rows(cat, rows, want, manifest):
    """Download rows for one category, appending to its manifest slice."""
    cat_dir = os.path.join(STAGE_DIR, cat)
    os.makedirs(cat_dir, exist_ok=True)
    have = manifest["images"].get(cat, [])
    # Stored titles are stripped of the "File:" prefix, the search rows still
    # carry it; accept both forms or a second run re-downloads everything.
    seen = set()
    for e in have:
        seen.add(e["title"])
        seen.add("File:" + e["title"])
    saved = 0
    for row in rows:
        if len(have) >= want:
            break
        if row["title"] in seen or BLOCK_TITLE.search(row["title"]):
            continue
        if row["orig_width"] and row["orig_width"] < MIN_W:
            continue
        if row["mime"] not in ("image/jpeg", "image/png"):
            continue
        name = "%02d_%s.jpg" % (len(have) + 1, slug(row["title"], 48))
        path = os.path.join(cat_dir, name)
        try:
            w, h, sha = stage_one(row, path)
        except Exception as exc:  # noqa: BLE001
            log("[fail ] %-6s %s (%s)" % (cat, row["title"][:60], exc))
            if os.path.exists(path):
                os.remove(path)
            continue
        entry = {
            "file": name,
            "category": cat,
            "title": row["title"].replace("File:", ""),
            "author": row["author"],
            "license": row["license"],
            "license_kind": row["license_kind"],
            "license_url": row["license_url"],
            "source_url": row["source_url"],
            "width": w,
            "height": h,
            "orig_width": row["orig_width"],
            "orig_height": row["orig_height"],
            "bytes": os.path.getsize(path),
            "aerial": bool(row["aerial"] or row["aerial_cat"]),
            "query": row["query"],
            "credit": row["credit"],
            "description": row["description"][:400],
            "sha256": sha,
            "downloaded": now_iso(),
        }
        have.append(entry)
        seen.add(row["title"])
        saved += 1
        log("[ok   ] %-6s %s  %dx%d  %s" % (cat, name, w, h, row["license"]))
    manifest["images"][cat] = have
    log("-- %-6s staged %d (total %d)" % (cat, saved, len(have)))
    save_json(MANIFEST, manifest)
    return saved


def stage(categories, want):
    pool = load_json(CANDIDATES, {}).get("candidates") or {}
    if not pool:
        raise SystemExit("no candidate pool; run probe first")
    manifest = load_json(MANIFEST, {"generated": "", "images": {}})
    for cat in CATEGORY_QUERIES:
        if categories and cat not in categories:
            continue
        rows = [r for r in pool.get(cat, []) if r["license_kind"]]
        rows.sort(key=score, reverse=True)
        stage_rows(cat, rows, want, manifest)
    manifest["generated"] = now_iso()
    save_json(MANIFEST, manifest)


def stage_titles(path):
    """Stage an explicit title list, for plates the scorer ranks too low.

    Format: {"open": ["File:Blue ocean surface (Unsplash).jpg", ...]}
    Order in the file is the order downloaded.
    """
    pool = load_json(CANDIDATES, {}).get("candidates") or {}
    spec = load_json(path, {})
    if not spec:
        raise SystemExit("empty title list: %s" % path)
    manifest = load_json(MANIFEST, {"generated": "", "images": {}})
    for cat, titles in spec.items():
        if cat not in CATEGORY_QUERIES:
            log("! unknown category %s" % cat)
            continue
        by_title = {r["title"]: r for r in pool.get(cat, [])}
        rows, missing = [], []
        for title in titles:
            row = by_title.get(title)
            if row is None:
                missing.append(title)
            else:
                rows.append(row)
        if missing:
            log("! %d titles not in the %s pool: %s"
                % (len(missing), cat, ", ".join(m[:48] for m in missing)))
        if rows:
            stage_rows(cat, rows, 10 ** 6, manifest)
    manifest["generated"] = now_iso()
    save_json(MANIFEST, manifest)


def dedupe():
    """Drop repeated titles from the staging manifest and their files."""
    manifest = load_json(MANIFEST, {"images": {}})
    dropped = 0
    for cat, entries in manifest["images"].items():
        seen, keep = set(), []
        for entry in entries:
            if entry["title"] in seen:
                path = os.path.join(STAGE_DIR, cat, entry["file"])
                if os.path.isfile(path):
                    os.remove(path)
                dropped += 1
                continue
            seen.add(entry["title"])
            keep.append(entry)
        manifest["images"][cat] = keep
        log("%-6s %d kept" % (cat, len(keep)))
    save_json(MANIFEST, manifest)
    log("dropped %d duplicate rows" % dropped)


# --------------------------------------------------------------------------- #
# contact sheet
# --------------------------------------------------------------------------- #
def load_font(size):
    from PIL import ImageFont
    for name in ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:  # noqa: BLE001
            continue
    return ImageFont.load_default()


def sheet(categories, thumb_w=440, cols=4, start=0, count=16):
    from PIL import Image, ImageDraw
    manifest = load_json(MANIFEST, {"images": {}})
    os.makedirs(SHEET_DIR, exist_ok=True)
    font = load_font(16)
    for cat, entries in manifest["images"].items():
        if categories and cat not in categories:
            continue
        if not entries:
            continue
        entries = entries[start:start + count]
        if not entries:
            continue
        tile_h = int(thumb_w * 0.72)
        rows = (len(entries) + cols - 1) // cols
        img = Image.new("RGB", (cols * thumb_w, rows * (tile_h + 24)), (20, 20, 22))
        draw = ImageDraw.Draw(img)
        for i, entry in enumerate(entries):
            r, c = divmod(i, cols)
            i = i + start
            x0, y0 = c * thumb_w, r * (tile_h + 24)
            path = os.path.join(STAGE_DIR, cat, entry["file"])
            try:
                with Image.open(path) as im:
                    im = im.convert("RGB")
                    im.thumbnail((thumb_w - 8, tile_h - 8))
                    img.paste(im, (x0 + (thumb_w - im.width) // 2,
                                  y0 + 24 + (tile_h - im.height) // 2))
            except Exception as exc:  # noqa: BLE001
                draw.text((x0 + 8, y0 + 40), "BAD %s" % exc, fill=(255, 90, 90), font=font)
            draw.text((x0 + 6, y0 + 3), "%s  %d" % (cat, i + 1),
                      fill=(255, 225, 120), font=font)
        out = os.path.join(SHEET_DIR, "%s_%02d.png" % (cat, start + 1))
        img.save(out)
        log("wrote %s (%dx%d, %d tiles)" % (out, img.width, img.height, len(entries)))


# --------------------------------------------------------------------------- #
# finalize
# --------------------------------------------------------------------------- #
def finalize(categories):
    """Copy the picked staged images into category folders and rebuild catalog."""
    from PIL import Image
    manifest = load_json(MANIFEST, {"images": {}})
    picks = load_json(PICKS, {})
    if not picks:
        raise SystemExit("no picks.json; run sheet, review, then write picks")
    final_entries = []
    for cat, names in picks.items():
        if categories and cat not in categories:
            continue
        by_file = {e["file"]: e for e in manifest["images"].get(cat, [])}
        out_dir = os.path.join(OUT_DIR, cat)
        os.makedirs(out_dir, exist_ok=True)
        for old in os.listdir(out_dir):
            if old.lower().endswith((".jpg", ".jpeg", ".png")):
                os.remove(os.path.join(out_dir, old))
        idx = 0
        for name in names:
            entry = by_file.get(name)
            if not entry:
                log("  ! %s/%s not in manifest" % (cat, name));
                continue
            src = os.path.join(STAGE_DIR, cat, name)
            if not os.path.isfile(src):
                log("  ! missing staged file %s" % src)
                continue
            idx += 1
            dst_name = "%02d_%s" % (idx, re.sub(r"^\d+_", "", name))
            dst = os.path.join(out_dir, dst_name)
            with Image.open(src) as im:
                save_plate(im, dst)
            with Image.open(dst) as chk:
                chk.load()
                w, h = chk.size
            if w < MIN_W or h < MIN_H:
                log("  ! %s only %dx%d, dropped" % (dst, w, h));
                os.remove(dst)
                idx -= 1
                continue
            final = dict(entry)
            final.update({"file": "%s/%s" % (cat, dst_name), "width": w, "height": h,
                          "bytes": os.path.getsize(dst)})
            final_entries.append(final)
    final_entries.sort(key=lambda e: (e["category"], e["file"]))
    catalog = {
        "source": "Wikimedia Commons (https://commons.wikimedia.org)",
        "api": API,
        "license_policy": "CC0 / public domain / CC BY (CC BY-SA only as fallback)",
        "rendition": "%d px wide thumbnails, re-encoded to JPEG q92" % THUMB_WIDTH,
        "min_size": [MIN_W, MIN_H],
        "generated": now_iso(),
        "count": len(final_entries),
        "categories": {c: sum(1 for e in final_entries if e["category"] == c)
                       for c in CATEGORY_QUERIES},
        "images": final_entries,
    }
    save_json(CATALOG, catalog)
    log("\ncatalog: %s (%d images, %.1f MB)" % (
        CATALOG, len(final_entries),
        sum(e["bytes"] for e in final_entries) / 1048576.0))
    for cat, n in catalog["categories"].items():
        log("  %-6s %d" % (cat, n))


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("probe", "stage", "stage-titles", "dedupe",
                                    "sheet", "finalize"))
    ap.add_argument("--categories", default="")
    ap.add_argument("--want", type=int, default=16)
    ap.add_argument("--per-query", type=int, default=40)
    ap.add_argument("--titles", default="", help="JSON title list for stage-titles")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=16)
    args = ap.parse_args(argv)
    cats = ([c.strip() for c in args.categories.split(",") if c.strip()]
            if args.categories else [])
    unknown = [c for c in cats if c not in CATEGORY_QUERIES]
    if unknown:
        raise SystemExit("unknown categories: %s" % ",".join(unknown))
    os.makedirs(OUT_DIR, exist_ok=True)
    {"probe": lambda: probe(cats, args.per_query),
     "stage": lambda: stage(cats, args.want),
     "stage-titles": lambda: stage_titles(args.titles),
     "dedupe": dedupe,
     "sheet": lambda: sheet(cats, start=args.start, count=args.count),
     "finalize": lambda: finalize(cats)}[args.mode]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
