#!/usr/bin/env python3
r"""Download real-photo references from Wikimedia Commons, one folder per render category.

Every render object in assets/models belongs to one of ten categories. This tool
searches Commons with *model-specific* queries (F-15 Eagle, MQ-9 Reaper, Boeing
737, OH-58 Kiowa ... - never the bare word "aircraft"), keeps only uploads with
explicit Commons licence metadata, and writes the files into
assets/refs/<category>/ plus a provenance catalogue at assets/refs/catalog.json.

Design notes
------------
* Only Commons (no other host). Files without licence metadata are skipped, so
  the gallery cannot contain a photo of unknown provenance.
* Network is assumed flaky: every query and every download is retried and any
  failure is appended to catalogue["errors"] and never aborts the run.
* Thumbnails are requested at iiurlwidth=1024 and the on-disk file is left
  byte-for-byte as served (no re-encode) so licence checksums stay valid.
* The catalogue is rewritten after every category, so an interrupted run keeps
  what it already fetched and can be re-run idempotently.

Run with:
    .\env\Scripts\python.exe tools\fetch_refs.py            # all categories
    .\env\Scripts\python.exe tools\fetch_refs.py --list     # show plan only
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFS_DIR = os.path.join(ROOT, "assets", "refs")
CATALOG = os.path.join(REFS_DIR, "catalog.json")
MODELS_CATALOG = os.path.join(ROOT, "assets", "models", "catalog.json")
API = "https://commons.wikimedia.org/w/api.php"
UA = ("RGB_SAR_LIDAR_SIMU-refs/1.0 (multimodal simulation reference gallery; "
      "contact: simu@example.org)")
THUMB_WIDTH = 1024
MAX_BYTES = 500 * 1024          # hard cap per image, as required by the task
MIN_W, MIN_H = 900, 500         # below this the target is too few pixels to measure

# Titles that describe something other than an exterior view of the object:
# diagrams, cockpit interiors, maps, logos. Cheaper to reject on the title than
# to download and throw the bytes away.
BLOCK_TITLE = re.compile(
    r"(diagram|line[- ]?drawing|\bdrawing\b|orthograph|blueprint|schematic|"
    r"3[- ]?view|\bmap\b|\bchart\b|\blogo\b|\bicon\b|silhouette|no[- ]?bg|"
    r"cutout|transparent|cockpit|interior|instrument panel|dashboard|"
    r"\bstamp\b|\bcoin\b|\bposter\b|postcard|\bpatch\b)", re.I)

# Ordered, model-specific searches. The first query that yields usable photos
# wins the slots; later ones only run when a category is still short.
#
# CirrusSearch syntax earns its keep here: deepcategory:"Aircraft in flight"
# restricts the result set to the recursive category, which is what removes the
# cockpit/detail/static shots that a plain "in flight" phrase still returns;
# filetype:bitmap drops SVG diagrams before they are downloaded. Queries without
# those operators are kept as fallbacks.
CATEGORY_QUERIES = {
    "军用飞机": [
        'F-15 Eagle deepcategory:"Aircraft in flight" filetype:bitmap -cockpit',
        'F-16 Fighting Falcon deepcategory:"Aircraft in flight" filetype:bitmap -cockpit',
        'Su-27 deepcategory:"Aircraft in flight" filetype:bitmap -cockpit',
        'F-4 Phantom II deepcategory:"Aircraft in flight" filetype:bitmap',
        'Dassault Rafale deepcategory:"Aircraft in flight" filetype:bitmap',
        "Fokker Dr.I in flight filetype:bitmap",
        'B-52 Stratofortress deepcategory:"Aircraft in flight" filetype:bitmap',
        "F-15 Eagle in flight filetype:bitmap -cockpit",
    ],
    "民航飞机": [
        'Boeing 737 deepcategory:"Aircraft in flight" filetype:bitmap -cockpit',
        'Airbus A320 deepcategory:"Aircraft in flight" filetype:bitmap -cockpit',
        'Boeing 747 deepcategory:"Aircraft in flight" filetype:bitmap -cockpit',
        'Boeing 777 deepcategory:"Aircraft in flight" filetype:bitmap -cockpit',
        'ATR 72 deepcategory:"Aircraft in flight" filetype:bitmap',
        "Embraer E190 in flight filetype:bitmap",
        'Airbus A380 deepcategory:"Aircraft in flight" filetype:bitmap',
    ],
    "直升机": [
        'OH-58 Kiowa deepcategory:"Helicopters in flight" filetype:bitmap',
        'UH-60 Black Hawk deepcategory:"Helicopters in flight" filetype:bitmap',
        'AH-64 Apache deepcategory:"Helicopters in flight" filetype:bitmap',
        'Mi-8 deepcategory:"Helicopters in flight" filetype:bitmap',
        'Eurocopter AS350 deepcategory:"Helicopters in flight" filetype:bitmap',
        'CH-47 Chinook deepcategory:"Helicopters in flight" filetype:bitmap',
    ],
    "固定翼无人机": [
        'MQ-9 Reaper deepcategory:"Aircraft in flight" filetype:bitmap',
        "RQ-4 Global Hawk in flight filetype:bitmap",
        'MQ-1 Predator deepcategory:"Aircraft in flight" filetype:bitmap',
        "MQ-9 Reaper in flight filetype:bitmap",
        "Bayraktar TB2 in flight filetype:bitmap",
    ],
    "旋翼无人机": [
        "DJI Phantom in flight filetype:bitmap",
        "DJI Mavic in flight filetype:bitmap",
        "quadcopter in flight filetype:bitmap",
        "DJI Inspire in flight filetype:bitmap",
        "multirotor UAV in flight filetype:bitmap",
    ],
    "军舰": [
        "Arleigh Burke destroyer underway filetype:bitmap",
        "Ticonderoga cruiser underway filetype:bitmap",
        "Type 052D destroyer filetype:bitmap",
        "Iowa-class battleship underway filetype:bitmap",
        "Los Angeles-class submarine filetype:bitmap",
        "missile boat underway filetype:bitmap",
    ],
    "民用船舶": [
        "container ship at sea filetype:bitmap",
        "cruise ship at sea filetype:bitmap",
        "sailing yacht at sea filetype:bitmap",
        "fishing boat at sea filetype:bitmap",
        "oil tanker at sea filetype:bitmap",
        "ocean liner at sea filetype:bitmap",
        "bulk carrier at sea filetype:bitmap",
    ],
    "地面车辆": [
        "Volkswagen Golf filetype:bitmap -interior -dashboard",
        "Toyota Corolla filetype:bitmap -interior -dashboard",
        "Ford Explorer filetype:bitmap -interior",
        "city bus filetype:bitmap",
        "box truck filetype:bitmap",
        "Ford F-150 filetype:bitmap -interior",
        "Porsche 911 filetype:bitmap -interior",
    ],
    "军用地面车辆": [
        "M1 Abrams tank filetype:bitmap",
        "M2 Bradley filetype:bitmap",
        "M939 truck filetype:bitmap",
        "Humvee filetype:bitmap",
        "Leopard 2 tank filetype:bitmap",
        "T-72 tank filetype:bitmap",
    ],
    "杂项": [
        "Jersey barrier filetype:bitmap",
        "chain-link fence filetype:bitmap",
        "shrub filetype:bitmap",
        "Curiosity rover filetype:bitmap",
        "Ingenuity Mars helicopter filetype:bitmap",
        "Mars rover filetype:bitmap",
    ],
}

# Models that the category's reference gallery is meant to drive. Filled from
# assets/models/catalog.json; the root-level props are the "杂项" objects.
MISC_MODELS = [
    "concrete_road_barrier", "global_hawk", "modular_chainlink_fence", "rover",
    "ship_medium", "shrub_01", "uav_demo", "vehicle_demo",
]


def log(msg):
    print(msg, flush=True)


def now_iso():
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def norm_key(text):
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def strip_html(raw):
    """Commons extmetadata comes back as HTML fragments; keep the readable text."""
    if raw is None:
        return ""
    text = str(raw)
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    text = (text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
                .replace("&quot;", '"').replace("&#039;", "'").replace("&nbsp;", " "))
    text = re.sub(r"\s+", " ", text).strip()
    return text[:300]


def slug(text, limit=64):
    text = unicodedata.normalize("NFKD", str(text))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")
    text = re.sub(r"_+", "_", text)
    return (text or "ref")[:limit].strip("_")


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
        json.dump(data, fh, ensure_ascii=False, indent=1, sort_keys=False)
        fh.write("\n")
    os.replace(tmp, path)


def model_map():
    """category -> model names, from the shipped model catalogue."""
    out = {name: [] for name in CATEGORY_QUERIES}
    for row in load_json(MODELS_CATALOG, []) or []:
        cat = row.get("category")
        name = row.get("name_cn") or row.get("file")
        if cat in out and name:
            out[cat].append(name)
    out["杂项"] = list(MISC_MODELS)
    return out


def api_get(params, retries=3, timeout=45):
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
            if exc.code in (429, 503, 502, 500):
                time.sleep(3 * (attempt + 1))
                continue
            raise RuntimeError(last) from exc
        except Exception as exc:  # noqa: BLE001 - flaky network is expected
            last = "%s: %s" % (type(exc).__name__, exc)
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(last or "unknown API failure")


def search_images(query, limit=5):
    data = api_get({
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrnamespace": "6",
        "gsrsearch": query,
        "gsrlimit": str(limit),
        "prop": "imageinfo",
        "iiprop": "url|size|mime|extmetadata",
        "iiurlwidth": str(THUMB_WIDTH),
        "iiextmetadatafilter": ("LicenseShortName|Artist|Credit|ImageDescription|"
                                "LicenseUrl|ObjectName"),
    })
    pages = (data.get("query") or {}).get("pages") or {}
    rows = sorted(pages.values(), key=lambda p: p.get("index", 0))
    return rows


def fetch_thumb(url, retries=3, timeout=90):
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
            if exc.code in (429, 503, 502, 500):
                time.sleep(5 * (attempt + 1))
                continue
            raise RuntimeError(last) from exc
        except Exception as exc:  # noqa: BLE001
            last = "%s: %s" % (type(exc).__name__, exc)
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(last or "download failed")


def probe_image(raw):
    """(width, height) of the served bytes; raises when the file is not an image."""
    import io

    from PIL import Image
    img = Image.open(io.BytesIO(raw))
    return img.size


def pick_imageinfo(page):
    info = (page.get("imageinfo") or [{}])[0]
    return info


def extmeta(info, key):
    meta = (info.get("extmetadata") or {}).get(key) or {}
    return meta.get("value")


def accept_candidate(info, title=""):
    """Return (ok, reason). Only photographs with a stated licence are usable."""
    if BLOCK_TITLE.search(str(title)):
        return False, "title looks like a diagram/interior"
    mime = str(info.get("mime") or "")
    if mime not in ("image/jpeg", "image/png"):
        return False, "mime %s not a photograph" % mime
    w, h = int(info.get("width") or 0), int(info.get("height") or 0)
    if w < MIN_W or h < MIN_H:
        return False, "too small %dx%d" % (w, h)
    ar = w / float(h) if h else 0.0
    if ar < 0.7 or ar > 2.6:
        return False, "aspect %.2f out of range" % ar
    if info.get("size") and int(info["size"]) < 40 * 1024:
        return False, "original smaller than 40 KB"
    lic = strip_html(extmeta(info, "LicenseShortName"))
    if not lic:
        return False, "no licence metadata"
    return True, lic


def download_one(page, cat_dir, idx, category, query):
    """Fetch one candidate. Returns (entry, status) with status in ok/skip/error."""
    info = pick_imageinfo(page)
    title = str(page.get("title") or "")
    ok, reason = accept_candidate(info, title)
    if not ok:
        return None, ("skip", reason, title, info)
    thumb = info.get("thumburl") or info.get("url")
    if not thumb:
        return None, ("skip", "no image url", title, info)
    if int(info.get("thumbwidth") or 0) not in (0, THUMB_WIDTH):
        # Commons may return a bigger rendition; keep the contract by asserting it.
        pass
    raw, ctype = fetch_thumb(thumb)
    if len(raw) > MAX_BYTES:
        return None, ("skip", "%.0f KB > 500 KB at %d px"
                      % (len(raw) / 1024.0, THUMB_WIDTH), title, info)
    if ctype and not ctype.lower().startswith("image/"):
        return None, ("skip", "content-type %s" % ctype, title, info)
    try:
        w, h = probe_image(raw)
    except Exception as exc:  # noqa: BLE001
        return None, ("skip", "unreadable image (%s)" % exc, title, info)
    if w < 800 or h < 450:
        return None, ("skip", "thumbnail only %dx%d" % (w, h), title, info)
    ext = ".png" if str(info.get("mime")) == "image/png" else ".jpg"
    name = "%02d_%s%s" % (idx, slug(title.replace("File:", "")), ext)
    path = os.path.join(cat_dir, name)
    with open(path, "wb") as fh:
        fh.write(raw)
    entry = {
        "file": os.path.relpath(path, REFS_DIR).replace("\\", "/"),
        "title": title,
        "category": category,
        "query": query,
        "width": int(info.get("width") or w),
        "height": int(info.get("height") or h),
        "thumb_width": w,
        "thumb_height": h,
        "bytes": len(raw),
        "mime": str(info.get("mime") or ""),
        "license": strip_html(extmeta(info, "LicenseShortName")),
        "license_url": strip_html(extmeta(info, "LicenseUrl")),
        "artist": strip_html(extmeta(info, "Artist")),
        "credit": strip_html(extmeta(info, "Credit")),
        "description": strip_html(extmeta(info, "ImageDescription")),
        "source_page": info.get("descriptionurl") or "",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "downloaded": now_iso(),
        "accepted": True,
        "review": "",
        "manual_box": None,
        "mask_source": "",
    }
    return entry, ("ok", "", title, info)


def category_dir(category):
    path = os.path.join(REFS_DIR, category)
    os.makedirs(path, exist_ok=True)
    return path


def run(args):
    cats = list(CATEGORY_QUERIES)
    if args.categories:
        wanted = [c.strip() for c in args.categories.split(",") if c.strip()]
        unknown = [c for c in wanted if c not in CATEGORY_QUERIES]
        if unknown:
            raise SystemExit("unknown category: %s" % ", ".join(unknown))
        cats = wanted
    models = model_map()
    catalog = load_json(CATALOG, {}) or {}
    catalog.setdefault("source", "Wikimedia Commons")
    catalog.setdefault("api", API)
    catalog.setdefault("categories", {})
    catalog.setdefault("errors", [])
    catalog.setdefault("skipped", [])
    catalog["rules"] = {
        "host": "commons.wikimedia.org only",
        "license_required": True,
        "thumb_width": THUMB_WIDTH,
        "max_bytes": MAX_BYTES,
        "min_source_size": [MIN_W, MIN_H],
    }
    catalog["generated"] = now_iso()

    for category in cats:
        node = catalog["categories"].setdefault(category, {
            "models": models.get(category, []),
            "queries": CATEGORY_QUERIES[category],
            "images": [],
        })
        node["models"] = models.get(category, [])
        node["queries"] = CATEGORY_QUERIES[category]
        have = {norm_key(im.get("title")) for im in node.get("images", [])}
        # "Container ship X in harbour 04" and "... 05" are the same frame twice;
        # keep one by comparing the first letters of the slug.
        have_dupe = {slug(im.get("title", "").replace("File:", ""))[:26]
                     for im in node.get("images", [])}
        have_files = {im.get("file") for im in node.get("images", [])}
        accepted = [im for im in node.get("images", []) if im.get("accepted", True)]
        target = args.target
        if len(accepted) >= target:
            log("[%s] already has %d accepted reference(s), skipping" % (category, len(accepted)))
            continue
        log("[%s] %d/%d accepted, searching ..." % (category, len(accepted), target))
        cat_dir = category_dir(category)
        idx = len(node.get("images", []))
        for query in CATEGORY_QUERIES[category]:
            if len([im for im in node.get("images", []) if im.get("accepted", True)]) >= target:
                break
            try:
                pages = search_images(query, limit=args.search_limit)
            except Exception as exc:  # noqa: BLE001
                catalog["errors"].append({"category": category, "query": query,
                                          "stage": "search", "error": str(exc),
                                          "when": now_iso()})
                log("  ! search failed: %s (%s)" % (query, exc))
                continue
            log("  query: %s -> %d candidate(s)" % (query, len(pages)))
            for page in pages:
                if len([im for im in node.get("images", []) if im.get("accepted", True)]) >= target:
                    break
                title = str(page.get("title") or "")
                if norm_key(title) in have:
                    continue
                dupe = slug(title.replace("File:", ""))[:26]
                if dupe in have_dupe:
                    catalog["skipped"].append({"category": category, "title": title,
                                               "reason": "near-duplicate of a kept photo",
                                               "when": now_iso()})
                    have.add(norm_key(title))
                    continue
                info = pick_imageinfo(page)
                if info.get("thumburl") in (None, "") and info.get("url") in (None, ""):
                    catalog["skipped"].append({"category": category, "title": title,
                                               "reason": "no thumb url", "when": now_iso()})
                    have.add(norm_key(title))
                    continue
                try:
                    entry, status = download_one(page, cat_dir, idx + 1, category, query)
                except Exception as exc:  # noqa: BLE001
                    catalog["errors"].append({"category": category, "query": query,
                                              "title": title, "stage": "download",
                                              "error": str(exc), "when": now_iso()})
                    log("  ! download failed: %s (%s)" % (title, exc))
                    have.add(norm_key(title))
                    continue
                kind, reason, title, info = status
                have.add(norm_key(title))
                if kind == "ok":
                    # never overwrite an existing distinct file
                    if entry["file"] in have_files:
                        continue
                    node["images"].append(entry)
                    have_files.add(entry["file"])
                    have_dupe.add(dupe)
                    idx += 1
                    log("  + %s  (%.0f KB, %s)" % (
                        entry["file"], entry["bytes"] / 1024.0, entry["license"]))
                else:
                    catalog["skipped"].append({
                        "category": category, "title": title, "reason": reason,
                        "query": query, "when": now_iso()})
                    log("  - skip %s: %s" % (title, reason))
                time.sleep(args.sleep)
            save_json(CATALOG, catalog)
            time.sleep(args.sleep)
        save_json(CATALOG, catalog)
        n = len([im for im in node.get("images", []) if im.get("accepted", True)])
        log("[%s] done: %d accepted" % (category, n))

    for category, node in catalog["categories"].items():
        node["count"] = len([im for im in node.get("images", []) if im.get("accepted", True)])
    catalog["generated"] = now_iso()
    save_json(CATALOG, catalog)
    total = sum(node.get("count", 0) for node in catalog["categories"].values())
    log("catalogue: %s (%d image(s), %d error(s), %d skipped)"
        % (os.path.relpath(CATALOG, ROOT), total, len(catalog["errors"]),
           len(catalog["skipped"])))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--categories", default="", help="comma separated subset")
    ap.add_argument("--target", type=int, default=6, help="accepted images per category")
    ap.add_argument("--search-limit", type=int, default=5, help="per-query candidates")
    ap.add_argument("--sleep", type=float, default=0.4, help="pause between HTTP calls")
    ap.add_argument("--list", action="store_true", help="print the plan and exit")
    args = ap.parse_args(argv)
    if args.list:
        models = model_map()
        for cat, queries in CATEGORY_QUERIES.items():
            log("%-8s target=%d models=%d" % (cat, args.target, len(models.get(cat, []))))
            for q in queries:
                log("    %s" % q)
        return 0
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
