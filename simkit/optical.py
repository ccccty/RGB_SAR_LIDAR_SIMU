"""Bridge to Blender (headless) for the visible-band sensor."""
import json
import os
import shutil
import subprocess

import numpy as np


def find_blender(cfg=None):
    cands = []
    if cfg:
        b = cfg.get("optical", {}).get("blender") or cfg.get("sensors", {}).get("optical", {}).get("blender")
        if b:
            cands.append(b)
    cands.append(os.environ.get("SIMKIT_BLENDER", ""))
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for pat in ("third_party", "tools", "vendor"):
        d = os.path.join(here, pat)
        if os.path.isdir(d):
            for name in sorted(os.listdir(d), reverse=True):
                if name.lower().startswith("blender"):
                    p = os.path.join(d, name, "blender.exe")
                    if os.path.exists(p):
                        cands.append(p)
                    p2 = os.path.join(d, name, "blender")
                    if os.path.exists(p2):
                        cands.append(p2)
    for c in cands:
        if c and os.path.exists(c):
            return c
    found = shutil.which("blender")
    return found


def available(cfg=None):
    return find_blender(cfg) is not None


def render_frames(cfg, cfg_path, out_dir, manifest_path, script_path, quiet=False):
    exe = find_blender(cfg)
    if not exe:
        raise RuntimeError("Blender not found: set SIMKIT_BLENDER or optical.blender in the config")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cmd = [exe, "-b", "--factory-startup", "-noaudio", "-P", script_path, "--",
           cfg_path, out_dir, manifest_path, root]
    if not quiet:
        print("  blender:", " ".join(cmd[:6]), "...")
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        tail = (p.stdout or "")[-3000:] + "\n" + (p.stderr or "")[-3000:]
        raise RuntimeError("blender failed (rc=%d)\n%s" % (p.returncode, tail))
    if not quiet:
        for line in (p.stdout or "").splitlines():
            if line.startswith("[simkit]"):
                print(" ", line)
    # Blender returns 0 even when the inline Python script raised, so the only
    # trustworthy signal is the files themselves.
    missing = []
    for i in range(_count_frames(manifest_path)):
        img = os.path.join(out_dir, "images", "optical", "%06d.png" % i)
        lab = os.path.join(out_dir, "labels", "optical", "%06d.json" % i)
        if not os.path.exists(img):
            missing.append(img)
        if not os.path.exists(lab):
            missing.append(lab)
    if missing:
        out_lines = (p.stdout or "").splitlines()[-12:]
        err_lines = (p.stderr or "").splitlines()[-12:]
        tail = chr(10).join(out_lines + err_lines)
        raise RuntimeError("blender produced no output for %d file(s), e.g. %s | %s"
                           % (len(missing), missing[0], tail[-2200:]))
    return p.returncode


def _count_frames(manifest_path):
    try:
        with open(manifest_path, "r", encoding="utf-8") as fh:
            return len(json.load(fh).get("frames") or [])
    except Exception:
        return 0
