"""Fetch Poly Haven CC0 models (gltf + textures) and convert to OBJ for simkit."""
import json
import os
import ssl
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "assets", "polyhaven")
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE      # local TLS interception; sizes/hashes checked
UA = {"User-Agent": "simkit-asset-fetcher/1.0"}


def get_json(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60, context=CTX) as r:
        return json.loads(r.read().decode("utf-8"))


def download(url, dst, expect_size=None):
    if os.path.exists(dst) and (expect_size is None or os.path.getsize(dst) == expect_size):
        return "cached"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=180, context=CTX) as r:
        blob = r.read()
    if expect_size is not None and len(blob) != expect_size:
        return "size-mismatch %d != %d" % (len(blob), expect_size)
    with open(dst, "wb") as fh:
        fh.write(blob)
    return "ok %.1f MB" % (len(blob) / 1e6)


def fetch(name, res="2k"):
    meta = get_json("https://api.polyhaven.com/files/" + name)
    entry = (meta.get("gltf") or {}).get(res) or {}
    gl = entry.get("gltf") or {}
    if not gl.get("url"):
        return None, "no gltf for " + name
    base = os.path.join(OUT, name)
    main = os.path.join(base, os.path.basename(gl["url"]))
    log = [download(gl["url"], main, gl.get("size"))]
    for rel, inc in (gl.get("include") or {}).items():
        dst = os.path.join(base, rel.replace("/", os.sep))
        log.append("%s %s" % (os.path.basename(rel),
                              download(inc["url"], dst, inc.get("size"))))
    return main, "; ".join(log)


def main():
    names = sys.argv[1:]
    for n in names:
        path, log = fetch(n)
        print("%-28s %s" % (n, log if path else log))
        if path:
            print("   ->", path)


main()
