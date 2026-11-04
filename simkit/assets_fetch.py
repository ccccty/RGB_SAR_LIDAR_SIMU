"""CC0 asset fetcher for the simulation platform.

Sky HDRIs come from Poly Haven and ground materials from ambientCG; both
libraries publish their assets under CC0.  The module is deliberately offline
first:

* a file already in assets/hdri/ (or the material cache) is returned
  without touching the network, and weather downloads are tagged in the
  catalog so a later hdri_for_weather hit needs no HTTP request at all;
* every successful download is recorded in assets/catalog.json with the
  id, size, source, licence, URL, byte count, download time and SHA-256;
* network, HTTP or content-validation failures raise AssetsUnavailable
  so callers can fall back to a procedural gradient sky instead of receiving
  a broken path.

Downloads stream into a temporary file in the destination directory and are
moved into place with os.replace, therefore a half-written .hdr never appears
under its final name.  A configurable timeout (20 s by default) and byte cap
protect the caller from a hung or absurdly large response.

Only the standard library is used (urllib.request et al.); numpy is not
required by this module.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import warnings
import zipfile
from datetime import datetime, timezone

__all__ = [
    "AssetsUnavailable",
    "HDRI_SIZES",
    "MATERIAL_SIZES",
    "WEATHER_HDRI_CATEGORIES",
    "MATERIAL_QUERIES",
    "UrllibFetcher",
    "hdri_for_weather",
    "download_hdri",
    "list_cached",
    "material_for",
    "read_catalog",
    "write_catalog",
]

# --------------------------------------------------------------------------- paths

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(ROOT, "assets")
HDRI_DIR = os.path.join(ASSETS_DIR, "hdri")
MATERIAL_DIR = os.path.join(ASSETS_DIR, "materials")
CATALOG_PATH = os.path.join(ASSETS_DIR, "catalog.json")

# ------------------------------------------------------------------------- network

POLYHAVEN_API = "https://api.polyhaven.com"
AMBIENTCG_API = "https://ambientcg.com/api/v2/full_json"
USER_AGENT = "simkit-assets/1.0 (CC0 asset fetcher)"

DEFAULT_TIMEOUT = 20.0
MAX_DOWNLOAD_BYTES = 512 * 1024 * 1024
MAX_JSON_BYTES = 16 * 1024 * 1024

CATALOG_VERSION = 1
HDRI_SIZES = ("1k", "2k", "4k", "8k", "16k", "24k")
MATERIAL_SIZES = ("1k", "2k", "4k", "8k", "12k", "16k")

# Weather preset (simkit.studio.WEATHER_MODES) -> Poly Haven HDRI categories.
# The first category that yields a downloadable asset wins.
WEATHER_HDRI_CATEGORIES = {
    "clear": ("pure skies", "clear"),
    "cloudy": ("partly cloudy", "overcast"),
    "rain": ("overcast",),
    "fog": ("overcast",),
}

# Ground kind -> ordered ambientCG search terms.  Unknown kinds return None.
MATERIAL_QUERIES = {
    "asphalt": ("asphalt",),
    "road": ("asphalt", "road"),
    "concrete": ("concrete",),
    "grass": ("grass",),
    "foliage": ("grass",),
    "forest": ("grass",),
    "sand": ("sand",),
    "desert": ("sand", "dirt"),
    "soil": ("dirt", "ground"),
    "gravel": ("gravel",),
    "rock": ("rock",),
    "snow": ("snow",),
    # ambientCG has no dedicated water material; "sea" resolves to the CC0
    # foam/ocean scans, which tile as a sea-surface detail map.
    "seawater": ("sea", "ocean", "foam"),
    "water": ("sea", "ocean", "foam"),
}

_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_STEM_SIZE = re.compile(r"^(?P<id>.+)_(?P<size>1k|2k|4k|8k|12k|16k|24k)$", re.IGNORECASE)
_HDR_EXT = ".hdr"


class AssetsUnavailable(RuntimeError):
    """A CC0 asset could not be obtained from a local cache or the network.

    Callers should catch this and fall back to a procedural sky/material.  The
    message always names the URL or local cache file that failed.
    """


# --------------------------------------------------------------------------- helpers

def _utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_name(value, what="asset id"):
    text = str(value).strip()
    if not _SAFE_NAME.match(text) or ".." in text:
        raise ValueError("invalid %s: %r" % (what, value))
    return text


def _choice(value, allowed, what):
    text = str(value).strip().lower()
    if text not in allowed:
        raise ValueError("invalid %s %r (expected one of: %s)" % (what, value, ", ".join(allowed)))
    return text


def _network_options(timeout, max_bytes):
    try:
        timeout = float(timeout)
    except (TypeError, ValueError) as exc:
        raise ValueError("timeout must be a positive number") from exc
    try:
        max_bytes = int(max_bytes)
    except (TypeError, ValueError) as exc:
        raise ValueError("max_bytes must be a positive integer") from exc
    if timeout <= 0:
        raise ValueError("timeout must be a positive number")
    if max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    return timeout, max_bytes


def _assets_url(category):
    return POLYHAVEN_API + "/assets?" + urllib.parse.urlencode({"t": "hdris", "c": category})


def _files_url(asset_id):
    return POLYHAVEN_API + "/files/" + urllib.parse.quote(asset_id, safe="")


def _material_search_url(query):
    return AMBIENTCG_API + "?" + urllib.parse.urlencode(
        {"type": "Material", "limit": "10", "q": query, "include": "downloadData"})


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _get_json(fetcher, url):
    try:
        data = fetcher.get_json(url)
    except AssetsUnavailable:
        raise
    except Exception as exc:  # a custom fetcher may raise urllib/socket errors
        raise AssetsUnavailable("cannot fetch %s: %s" % (url, exc)) from exc
    if not isinstance(data, (dict, list)):
        raise AssetsUnavailable("unexpected JSON payload from %s" % url)
    return data


def _get_json_dict(fetcher, url):
    data = _get_json(fetcher, url)
    if not isinstance(data, dict):
        raise AssetsUnavailable("unexpected JSON payload from %s (expected an object)" % url)
    return data


class UrllibFetcher:
    """Small HTTP client seam so tests can inject a fake fetcher.

    get_json(url) and download(url, fileobj, *, max_bytes) are the only
    methods the module calls.  A fake only has to provide those two.
    """

    def __init__(self, timeout=DEFAULT_TIMEOUT, max_bytes=MAX_DOWNLOAD_BYTES):
        self.timeout, self.max_bytes = _network_options(timeout, max_bytes)

    def get_json(self, url):
        request = urllib.request.Request(
            url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read(MAX_JSON_BYTES + 1)
        except AssetsUnavailable:
            raise
        except (urllib.error.HTTPError, urllib.error.URLError,
                TimeoutError, OSError, ValueError) as exc:
            raise AssetsUnavailable("cannot fetch JSON %s: %s" % (url, exc)) from exc
        if len(raw) > MAX_JSON_BYTES:
            raise AssetsUnavailable("JSON response too large from %s (> %d bytes)"
                                    % (url, MAX_JSON_BYTES))
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AssetsUnavailable("invalid JSON from %s: %s" % (url, exc)) from exc

    def download(self, url, fileobj, *, max_bytes=None):
        limit = int(self.max_bytes if max_bytes is None else max_bytes)
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                declared = response.headers.get("Content-Length")
                if declared is not None:
                    try:
                        declared_bytes = int(declared)
                    except (TypeError, ValueError):
                        declared_bytes = None
                    if declared_bytes is not None and declared_bytes > limit:
                        raise AssetsUnavailable(
                            "refusing %s: Content-Length %d exceeds limit %d"
                            % (url, declared_bytes, limit))
                total = 0
                while True:
                    chunk = response.read(256 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > limit:
                        raise AssetsUnavailable(
                            "refusing %s: response exceeds limit %d" % (url, limit))
                    fileobj.write(chunk)
                return total
        except AssetsUnavailable:
            raise
        except (urllib.error.HTTPError, urllib.error.URLError,
                TimeoutError, OSError, ValueError) as exc:
            raise AssetsUnavailable("download failed %s: %s" % (url, exc)) from exc


class _SizeLimitedWriter:
    """Count bytes written through a fetcher and abort before the cap."""

    def __init__(self, stream, limit):
        self.stream = stream
        self.limit = int(limit)
        self.written = 0

    def write(self, data):
        self.written += len(data)
        if self.written > self.limit:
            raise AssetsUnavailable("download exceeds size limit of %d bytes" % self.limit)
        return self.stream.write(data)

    def flush(self):
        return self.stream.flush()


def _download_to(url, destination, fetcher, max_bytes, *, expected_size=None, validator=None):
    """Stream url into a temporary file and atomically rename it.

    Returns (bytes_written, sha256_hex).  On any failure the temporary file
    is removed and no file appears at destination.
    """
    destination = os.path.abspath(destination)
    directory = os.path.dirname(destination)
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=".part-" + os.path.basename(destination) + "-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "wb") as handle:
            limited = _SizeLimitedWriter(handle, max_bytes)
            fetcher.download(url, limited, max_bytes=int(max_bytes))
            handle.flush()
            os.fsync(handle.fileno())
        written = os.path.getsize(temporary)
        if written <= 0:
            raise AssetsUnavailable("empty download from %s" % url)
        if expected_size is not None:
            try:
                expected = int(expected_size)
            except (TypeError, ValueError):
                expected = None
            if expected is not None and expected != written:
                raise AssetsUnavailable(
                    "truncated download from %s (%d of %d bytes)" % (url, written, expected))
        if validator is not None:
            validator(temporary)
        digest = _sha256_file(temporary)
        os.replace(temporary, destination)
        temporary = None
        return written, digest
    except AssetsUnavailable:
        raise
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        raise AssetsUnavailable("download failed %s: %s" % (url, exc)) from exc
    finally:
        if temporary is not None and os.path.exists(temporary):
            try:
                os.unlink(temporary)
            except OSError:
                pass


def _validate_hdri(path):
    with open(path, "rb") as handle:
        head = handle.read(256)
    stripped = head.lstrip(b"\xef\xbb\xbf \t\r\n")
    if stripped.startswith(b"<"):
        raise AssetsUnavailable("server returned HTML/XML instead of a Radiance HDR")
    if not (head.startswith(b"#?") or b"RADIANCE" in head):
        raise AssetsUnavailable("downloaded file is not a Radiance .hdr image")


def _validate_zip(path):
    if not zipfile.is_zipfile(path):
        raise AssetsUnavailable("downloaded file is not a zip archive")


# --------------------------------------------------------------------------- catalog

def _empty_catalog():
    return {"version": CATALOG_VERSION, "hdri": {}, "materials": {}}


def _normalise_catalog(data):
    if not isinstance(data, dict):
        return _empty_catalog()
    catalog = dict(data)
    catalog.setdefault("version", CATALOG_VERSION)
    for section in ("hdri", "materials"):
        value = catalog.get(section)
        if isinstance(value, list):
            converted = {}
            for index, item in enumerate(value):
                if isinstance(item, dict):
                    key = item.get("file") or item.get("path") or item.get("id") or str(index)
                    converted[os.path.basename(str(key))] = item
            catalog[section] = converted
        elif not isinstance(value, dict):
            catalog[section] = {}
    return catalog


def read_catalog(path=None):
    """Read assets/catalog.json; missing or corrupt files yield an empty catalog."""
    path = os.path.abspath(path or CATALOG_PATH)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return _empty_catalog()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        warnings.warn("ignoring unreadable asset catalog %s: %s" % (path, exc), RuntimeWarning)
        return _empty_catalog()
    if not isinstance(data, dict):
        warnings.warn("ignoring malformed asset catalog %s: expected an object" % path,
                      RuntimeWarning)
        return _empty_catalog()
    return _normalise_catalog(data)


def write_catalog(catalog, path=None):
    """Atomically write the catalog, creating its parent directory if needed."""
    path = os.path.abspath(path or CATALOG_PATH)
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    payload = json.dumps(_normalise_catalog(catalog), indent=2, sort_keys=True,
                         ensure_ascii=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=".catalog-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None and os.path.exists(temporary):
            try:
                os.unlink(temporary)
            except OSError:
                pass
    return path


def _try_write_catalog(catalog):
    try:
        write_catalog(catalog)
    except (OSError, TypeError, ValueError) as exc:
        warnings.warn("could not update asset catalog: %s" % exc, RuntimeWarning)


def _record_hdri(asset_id, size, path, url, byte_count, sha256, weather_tags=()):
    catalog = read_catalog()
    section = catalog.setdefault("hdri", {})
    key = os.path.basename(path)
    existing = section.get(key) if isinstance(section.get(key), dict) else {}
    tags = sorted(set(existing.get("weather") or []) | set(weather_tags or []))
    entry = {
        "id": asset_id,
        "size": size,
        "file": key,
        "source": "polyhaven",
        "license": "CC0",
        "url": url,
        "bytes": int(byte_count),
        "sha256": sha256,
        "downloaded": existing.get("downloaded") or _utc_now(),
    }
    if tags:
        entry["weather"] = tags
    section[key] = entry
    _try_write_catalog(catalog)


def _tag_cached_hdri(path, asset_id, size, weather_tags):
    catalog = read_catalog()
    section = catalog.setdefault("hdri", {})
    key = os.path.basename(path)
    existing = dict(section.get(key) or {})
    tags = sorted(set(existing.get("weather") or []) | set(weather_tags or []))
    existing.setdefault("id", asset_id)
    existing.setdefault("size", size)
    existing.setdefault("file", key)
    existing.setdefault("source", "polyhaven")
    existing.setdefault("license", "CC0")
    existing.setdefault("bytes", os.path.getsize(path))
    existing.setdefault("downloaded", _utc_now())
    if tags:
        existing["weather"] = tags
    section[key] = existing
    _try_write_catalog(catalog)


# --------------------------------------------------------------------------- HDRI

def _hdri_cache_path(asset_id, size):
    return os.path.join(HDRI_DIR, "%s_%s%s" % (asset_id, size, _HDR_EXT))


def _hdri_cache_candidates(asset_id, size):
    """Size-suffixed cache plus the legacy <id>.hdr spelling (read-only support)."""
    return (
        _hdri_cache_path(asset_id, size),
        os.path.join(HDRI_DIR, "%s%s" % (asset_id, _HDR_EXT)),
    )


def _legacy_cache_matches(path, size):
    entry = (read_catalog().get("hdri") or {}).get(os.path.basename(path)) or {}
    recorded = str(entry.get("size") or "").lower()
    return not recorded or recorded == size


def _weather_key(weather):
    if isinstance(weather, dict):
        for key in ("preset", "weather", "mode", "key"):
            if weather.get(key):
                weather = weather[key]
                break
    if weather is None:
        return None
    text = str(weather).strip().lower().replace("_", " ").replace("-", " ")
    return text or None


def _num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _select_asset(assets, category):
    """Choose deterministically: category match, then downloads, then id."""
    if not isinstance(assets, dict) or not assets:
        return None
    wanted = str(category).strip().lower()

    def rank(item):
        asset_id, meta = item
        meta = meta if isinstance(meta, dict) else {}
        categories = {str(value).strip().lower() for value in (meta.get("categories") or [])}
        return (
            1 if wanted in categories else 0,
            _num(meta.get("download_count")),
            _num(meta.get("download_count_month")),
            str(asset_id),
        )

    return max(assets.items(), key=rank)


def _hdri_result(asset_id, size, path, cached):
    catalog = read_catalog()
    entry = (catalog.get("hdri") or {}).get(os.path.basename(path)) or {}
    try:
        byte_count = os.path.getsize(path)
    except OSError:
        byte_count = entry.get("bytes")
    return {
        "id": entry.get("id") or asset_id,
        "path": os.path.abspath(path),
        "source": entry.get("source") or "polyhaven",
        "license": entry.get("license") or "CC0",
        "size": entry.get("size") or size,
        "bytes": byte_count,
        "sha256": entry.get("sha256"),
        "url": entry.get("url"),
        "downloaded": entry.get("downloaded"),
        "cached": bool(cached),
    }


def _find_cached_for_weather(weather, size):
    catalog = read_catalog()
    for filename, entry in sorted((catalog.get("hdri") or {}).items()):
        if not isinstance(entry, dict):
            continue
        if str(entry.get("size") or "").lower() != size:
            continue
        if weather not in (entry.get("weather") or []):
            continue
        if os.path.basename(filename) != filename or filename in ("", ".", ".."):
            continue
        path = os.path.join(HDRI_DIR, filename)
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            return _hdri_result(entry.get("id") or _stem_id(filename), size, path, cached=True)
    return None


def _stem_id(filename):
    stem = os.path.splitext(os.path.basename(filename))[0]
    match = _STEM_SIZE.match(stem)
    return match.group("id") if match else stem


def _stem_size(filename):
    match = _STEM_SIZE.match(os.path.splitext(os.path.basename(filename))[0])
    return match.group("size").lower() if match else None


def _acquire_hdri(asset_id, size, *, weather_tags=(), fetcher=None,
                  timeout=DEFAULT_TIMEOUT, max_bytes=MAX_DOWNLOAD_BYTES, force=False):
    path = _hdri_cache_path(asset_id, size)
    if not force:
        for candidate in _hdri_cache_candidates(asset_id, size):
            if not (os.path.isfile(candidate) and os.path.getsize(candidate) > 0):
                continue
            if candidate != path and not _legacy_cache_matches(candidate, size):
                continue
            path = candidate
            if weather_tags:
                _tag_cached_hdri(path, asset_id, size, weather_tags)
            return path, True

    timeout, max_bytes = _network_options(timeout, max_bytes)
    fetcher = fetcher or UrllibFetcher(timeout=timeout, max_bytes=max_bytes)
    metadata = _get_json_dict(fetcher, _files_url(asset_id))
    hdri = metadata.get("hdri")
    if not isinstance(hdri, dict) or size not in hdri:
        available = ", ".join(sorted(hdri)) if isinstance(hdri, dict) else "none"
        raise AssetsUnavailable("HDRI %r has no %s variant (available: %s)"
                                % (asset_id, size, available))
    variant = hdri[size]
    hdr = variant.get("hdr") if isinstance(variant, dict) else None
    if not isinstance(hdr, dict) or not hdr.get("url"):
        raise AssetsUnavailable("HDRI %r has no downloadable .hdr at %s" % (asset_id, size))
    url = str(hdr["url"])
    if not url.lower().startswith(("http://", "https://")):
        raise AssetsUnavailable("refusing non-HTTP HDRI URL %r" % url)
    byte_count, digest = _download_to(
        url, path, fetcher, max_bytes,
        expected_size=hdr.get("size"), validator=_validate_hdri)
    _record_hdri(asset_id, size, path, url, byte_count, digest, weather_tags=weather_tags)
    return path, False


def download_hdri(asset_id, size="2k", *, fetcher=None, timeout=DEFAULT_TIMEOUT,
                  max_bytes=MAX_DOWNLOAD_BYTES, force=False):
    """Return the absolute path of asset_id at size, downloading if needed."""
    asset_id = _safe_name(asset_id, "HDRI asset id")
    size = _choice(size, HDRI_SIZES, "HDRI size")
    path, _ = _acquire_hdri(asset_id, size, fetcher=fetcher, timeout=timeout,
                            max_bytes=max_bytes, force=force)
    return os.path.abspath(path)


def hdri_for_weather(weather, size="2k", *, fetcher=None, timeout=DEFAULT_TIMEOUT,
                     max_bytes=MAX_DOWNLOAD_BYTES):
    """Pick a CC0 sky HDRI for a weather preset.

    clear/cloudy/rain/fog map to Poly Haven categories.  Unknown presets return
    None; a known preset that cannot be served (offline, HTTP error, no matching
    asset) raises AssetsUnavailable.
    """
    key = _weather_key(weather)
    if key not in WEATHER_HDRI_CATEGORIES:
        return None
    size = _choice(size, HDRI_SIZES, "HDRI size")

    cached = _find_cached_for_weather(key, size)
    if cached is not None:
        return cached

    timeout, max_bytes = _network_options(timeout, max_bytes)
    fetcher = fetcher or UrllibFetcher(timeout=timeout, max_bytes=max_bytes)
    failures = []
    for category in WEATHER_HDRI_CATEGORIES[key]:
        try:
            assets = _get_json_dict(fetcher, _assets_url(category))
            selected = _select_asset(assets, category)
            if selected is None:
                failures.append("category %r returned no assets" % category)
                continue
            asset_id, _ = selected
            # Rain and fog share the overcast category, so tag the download
            # with every preset the chosen category can satisfy.
            weather_tags = tuple(sorted(
                name for name, categories in WEATHER_HDRI_CATEGORIES.items()
                if category in categories))
            path, was_cached = _acquire_hdri(
                _safe_name(asset_id, "HDRI asset id"), size, weather_tags=weather_tags,
                fetcher=fetcher, timeout=timeout, max_bytes=max_bytes)
            return _hdri_result(asset_id, size, path, cached=was_cached)
        except AssetsUnavailable as exc:
            failures.append("%s: %s" % (category, exc))
    raise AssetsUnavailable("no HDRI available for weather %r (%s)"
                            % (key, "; ".join(failures) or "no candidates"))


def list_cached():
    """List locally present HDRIs and extracted materials with catalog metadata."""
    catalog = read_catalog()
    entries = []
    hdri_section = catalog.get("hdri") or {}
    if os.path.isdir(HDRI_DIR):
        for filename in sorted(os.listdir(HDRI_DIR)):
            if not filename.lower().endswith(_HDR_EXT):
                continue
            path = os.path.abspath(os.path.join(HDRI_DIR, filename))
            if not os.path.isfile(path):
                continue
            entry = hdri_section.get(filename) or {}
            if not isinstance(entry, dict):
                entry = {}
            try:
                byte_count = os.path.getsize(path)
            except OSError:
                continue
            entries.append({
                "kind": "hdri",
                "id": entry.get("id") or _stem_id(filename),
                "size": entry.get("size") or _stem_size(filename),
                "path": path,
                "source": entry.get("source") or "polyhaven",
                "license": entry.get("license") or "CC0",
                "bytes": byte_count,
                "sha256": entry.get("sha256"),
                "url": entry.get("url"),
                "downloaded": entry.get("downloaded"),
            })
    material_section = catalog.get("materials") or {}
    for _, entry in sorted(material_section.items()):
        if not isinstance(entry, dict):
            continue
        texture = _material_texture_path(entry)
        if not texture or not os.path.isfile(texture):
            continue
        entries.append({
            "kind": "material",
            "id": entry.get("id"),
            "size": entry.get("size"),
            "path": texture,
            "source": entry.get("source") or "ambientcg",
            "license": entry.get("license") or "CC0",
            "bytes": entry.get("bytes"),
            "sha256": entry.get("sha256"),
            "url": entry.get("url"),
            "downloaded": entry.get("downloaded"),
        })
    return entries


# ------------------------------------------------------------------------- materials

def _material_texture_path(entry):
    texture = entry.get("texture")
    if not texture:
        return None
    if os.path.isabs(str(texture)):
        return os.path.abspath(str(texture))
    return os.path.abspath(os.path.join(MATERIAL_DIR, str(texture)))


def _material_cache_result(entry, kind):
    texture = _material_texture_path(entry)
    return {
        "id": entry.get("id"),
        "kind": kind,
        "path": texture,
        "texture": texture,
        "size": entry.get("size"),
        "source": entry.get("source") or "ambientcg",
        "license": entry.get("license") or "CC0",
        "bytes": entry.get("bytes"),
        "sha256": entry.get("sha256"),
        "url": entry.get("url"),
        "downloaded": entry.get("downloaded"),
        "cached": True,
    }


def _cached_material(kind, size):
    catalog = read_catalog()
    for _, entry in sorted((catalog.get("materials") or {}).items()):
        if not isinstance(entry, dict):
            continue
        if str(entry.get("size") or "").lower() != size:
            continue
        kinds = entry.get("kinds")
        if not isinstance(kinds, list):
            kinds = [entry.get("kind")] if entry.get("kind") else []
        if kind not in [str(value).lower() for value in kinds]:
            continue
        texture = _material_texture_path(entry)
        if texture and os.path.isfile(texture):
            return _material_cache_result(entry, kind)
    return None


def _first_material(data):
    assets = data.get("foundAssets") if isinstance(data, dict) else None
    if not isinstance(assets, list):
        return None
    for asset in assets:
        if isinstance(asset, dict) and asset.get("assetId"):
            return asset
    return None


def _pick_zip_download(asset, size):
    folders = asset.get("downloadFolders")
    default = folders.get("default") if isinstance(folders, dict) else None
    categories = default.get("downloadFiletypeCategories") if isinstance(default, dict) else None
    zip_category = categories.get("zip") if isinstance(categories, dict) else None
    downloads = zip_category.get("downloads") if isinstance(zip_category, dict) else None
    if not isinstance(downloads, list):
        return None
    wanted = size.upper() + "-JPG"
    for item in downloads:
        if not isinstance(item, dict):
            continue
        attribute = str(item.get("attribute") or "").strip().upper()
        filename = str(item.get("fileName") or "").strip().upper()
        if attribute == wanted or filename.endswith("_%s.ZIP" % wanted):
            return item
    return None


def _texture_score(name):
    lower = name.lower()
    extension = os.path.splitext(lower)[1]
    scores = {".jpg": 40, ".jpeg": 40, ".png": 30, ".tif": 20, ".tiff": 20}
    score = scores.get(extension)
    if score is None:
        return -1
    if any(token in lower for token in ("_color", "albedo", "diffuse", "basecolor", "_base_color")):
        score += 100
    elif any(token in lower for token in ("normal", "rough", "metal", "ao", "height", "displacement")):
        score -= 20
    return score


def _find_color_texture(directory):
    candidates = []
    for current, _, files in os.walk(directory):
        for name in files:
            path = os.path.join(current, name)
            score = _texture_score(name)
            if score >= 0:
                candidates.append((score, path))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    return os.path.abspath(candidates[-1][1])


def _extract_zip(zip_path, destination):
    destination = os.path.abspath(destination)
    parent = os.path.dirname(destination)
    os.makedirs(parent, exist_ok=True)
    temporary = tempfile.mkdtemp(prefix=".extract-", dir=parent)
    try:
        with zipfile.ZipFile(zip_path) as archive:
            base = os.path.abspath(temporary)
            for info in archive.infolist():
                target = os.path.abspath(os.path.join(base, info.filename))
                if target != base and not target.startswith(base + os.sep):
                    raise AssetsUnavailable(
                        "unsafe path %r in %s" % (info.filename, zip_path))
            archive.extractall(temporary)
        if os.path.isdir(destination):
            shutil.rmtree(destination)
        os.replace(temporary, destination)
        temporary = None
        return destination
    except AssetsUnavailable:
        raise
    except (OSError, zipfile.BadZipFile) as exc:
        raise AssetsUnavailable("cannot extract %s: %s" % (zip_path, exc)) from exc
    finally:
        if temporary is not None and os.path.isdir(temporary):
            shutil.rmtree(temporary, ignore_errors=True)


def _record_material(asset_id, kind, size, zip_path, texture, url, byte_count, sha256):
    catalog = read_catalog()
    section = catalog.setdefault("materials", {})
    key = os.path.basename(zip_path)
    existing = section.get(key) if isinstance(section.get(key), dict) else {}
    kinds = existing.get("kinds")
    if not isinstance(kinds, list):
        kinds = [existing.get("kind")] if existing.get("kind") else []
    kinds = sorted(set(str(value).lower() for value in kinds) | {kind})
    entry = {
        "id": asset_id,
        "kinds": kinds,
        "size": size,
        "zip": key,
        "texture": os.path.relpath(texture, MATERIAL_DIR).replace(os.sep, "/"),
        "source": "ambientcg",
        "license": "CC0",
        "url": url,
        "bytes": int(byte_count),
        "sha256": sha256,
        "downloaded": existing.get("downloaded") or _utc_now(),
    }
    section[key] = entry
    _try_write_catalog(catalog)


def _acquire_material(asset, kind, size, fetcher, max_bytes):
    asset_id = _safe_name(asset.get("assetId"), "material asset id")
    download = _pick_zip_download(asset, size)
    if download is None:
        raise AssetsUnavailable("material %r has no %s JPG zip" % (asset_id, size))
    url = download.get("downloadLink") or download.get("fullDownloadPath")
    if not url or not str(url).lower().startswith(("http://", "https://")):
        raise AssetsUnavailable("material %r has no HTTP download URL" % asset_id)

    os.makedirs(MATERIAL_DIR, exist_ok=True)
    zip_path = os.path.join(MATERIAL_DIR, "%s_%s.zip" % (asset_id, size))
    if os.path.isfile(zip_path) and os.path.getsize(zip_path) > 0 \
            and zipfile.is_zipfile(zip_path):
        digest = _sha256_file(zip_path)
        byte_count = os.path.getsize(zip_path)
    else:
        byte_count, digest = _download_to(
            str(url), zip_path, fetcher, max_bytes,
            expected_size=download.get("size"), validator=_validate_zip)

    destination = os.path.join(MATERIAL_DIR, "%s_%s" % (asset_id, size))
    texture = None
    catalog = read_catalog()
    cached_entry = (catalog.get("materials") or {}).get(os.path.basename(zip_path))
    if isinstance(cached_entry, dict):
        candidate = _material_texture_path(cached_entry)
        if candidate and os.path.isfile(candidate):
            texture = candidate
    if texture is None:
        if not os.path.isdir(destination):
            _extract_zip(zip_path, destination)
        texture = _find_color_texture(destination)
    if texture is None:
        if os.path.isdir(destination):
            shutil.rmtree(destination, ignore_errors=True)
        raise AssetsUnavailable("material %r contained no usable color texture" % asset_id)

    _record_material(asset_id, kind, size, zip_path, texture, str(url), byte_count, digest)
    return {
        "id": asset_id,
        "kind": kind,
        "path": texture,
        "texture": texture,
        "size": size,
        "source": "ambientcg",
        "license": "CC0",
        "bytes": byte_count,
        "sha256": digest,
        "url": str(url),
        "downloaded": _utc_now(),
        "cached": False,
    }


def material_for(kind, size="1k", *, fetcher=None, timeout=DEFAULT_TIMEOUT,
                 max_bytes=MAX_DOWNLOAD_BYTES):
    """Return a CC0 ground texture path for kind (asphalt/grass/sand/...).

    Unknown kinds and searches with no result return None.  Transfer failures
    raise AssetsUnavailable.
    """
    key = str(kind).strip().lower()
    queries = MATERIAL_QUERIES.get(key)
    if not queries:
        return None
    size = _choice(size, MATERIAL_SIZES, "material size")

    cached = _cached_material(key, size)
    if cached is not None:
        return cached

    timeout, max_bytes = _network_options(timeout, max_bytes)
    fetcher = fetcher or UrllibFetcher(timeout=timeout, max_bytes=max_bytes)
    failures = []
    for query in queries:
        try:
            data = _get_json_dict(fetcher, _material_search_url(query))
            asset = _first_material(data)
            if asset is None:
                continue
            return _acquire_material(asset, key, size, fetcher, max_bytes)
        except AssetsUnavailable as exc:
            failures.append("%s: %s" % (query, exc))
    if failures:
        raise AssetsUnavailable("no material available for %r (%s)"
                                % (key, "; ".join(failures)))
    return None
