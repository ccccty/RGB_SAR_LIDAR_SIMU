"""Bridge to Blender (headless) for the visible-band sensor."""
import json
import os
import shutil
import subprocess
import time

import numpy as np


def camera_clip_end(camera_pos, world_corners):
    """Far plane from actual world geometry, including translation and motion."""
    corners = np.asarray(world_corners, float).reshape(-1, 3)
    if not len(corners):
        return 100.0
    reach = float(np.max(np.linalg.norm(corners-np.asarray(camera_pos, float), axis=1)))
    return max(100.0, reach*1.25+10.0)


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


def render_frames(cfg, cfg_path, out_dir, manifest_path, script_path, quiet=False,
                  view_tags=None):
    exe = find_blender(cfg)
    if not exe:
        raise RuntimeError("Blender not found: set SIMKIT_BLENDER or optical.blender in the config")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cmd = [exe, "-b", "--factory-startup", "-noaudio"]
    # Cycles grabs every core by default. Three chunk workers then oversubscribe
    # a 22-thread machine by 3x and the whole run is slower than one worker
    # (measured: 64 s/frame instead of 25). The build script divides the cores
    # among the workers and passes the budget down through this variable.
    thr = os.environ.get("SIMKIT_BLENDER_THREADS")
    if thr:
        cmd += ["-t", str(max(1, int(thr)))]
    cmd += ["-P", script_path, "--", cfg_path, out_dir, manifest_path, root]
    if not quiet:
        print("  blender:", " ".join(cmd[:6]), "...")
    started_at = time.time_ns()
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        tail = (p.stdout or "")[-3000:] + "\n" + (p.stderr or "")[-3000:]
        raise RuntimeError("blender failed (rc=%d)\n%s" % (p.returncode, tail))
    if "Traceback (most recent call last)" in (p.stdout or "")+(p.stderr or ""):
        raise RuntimeError("Blender script failed despite a zero process exit code:\n"+
                           ((p.stdout or "")+(p.stderr or ""))[-3000:])
    if not quiet:
        for line in (p.stdout or "").splitlines():
            if line.startswith("[simkit]"):
                print(" ", line)
    # Blender returns 0 even when the inline Python script raised, so the only
    # trustworthy signal is the files themselves.
    # Check the frame ids this manifest actually asked for. Chunks carry global
    # ids, so counting from 0 would happily accept a neighbour chunk's files.
    tags = [""] if view_tags is None else list(view_tags)
    missing = []
    for tag in tags:
        for i in _frame_ids(manifest_path):
            img = os.path.join(out_dir, "images", "optical" + tag, "%06d.png" % i)
            lab = os.path.join(out_dir, "labels", "optical" + tag, "%06d.json" % i)
            if not os.path.exists(img) or os.stat(img).st_mtime_ns < started_at:
                missing.append(img)
            if not os.path.exists(lab) or os.stat(lab).st_mtime_ns < started_at:
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


def _frame_ids(manifest_path):
    """Global frame ids of the manifest; falls back to 0..n-1 for old manifests."""
    try:
        with open(manifest_path, "r", encoding="utf-8") as fh:
            frames = json.load(fh).get("frames") or []
    except Exception:
        return []
    return [int(f.get("frame", i)) for i, f in enumerate(frames)]
