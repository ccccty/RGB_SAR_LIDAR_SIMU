"""External renderer contract.

Any program that writes frames + boxes in the contract layout can act as the
visible-band backend.  simkit hands it a manifest describing every frame and
world pose; the renderer only has to draw pixels and report 2D boxes.

Contract
--------
argv:  <exe> --config <cfg.json> --manifest <manifest.json> --out <dir>
input manifest:::
    {"frames": [{"t": 0.0,
                 "objects": {"car_01": {"pos": [x, y, z], "yaw": deg}, ...}}]}
output (created by the renderer):
    <out>/images/optical/NNNNNN.png          8-bit RGB
    <out>/labels/optical/NNNNNN.json          same schema as the Blender backend:
        {"frame": 0, "image": "000000.png", "width": W, "height": H,
         "boxes": {"car_01": {"x0":..,"y0":..,"x1":..,"y1":..,"visible":true}}}
"""
import json
import os
import shlex
import subprocess


def spec_for(cfg, name):
    return (cfg.get("backends", {}) or {}).get(name, {}) or {}


def resolve(cfg, name):
    """Return (exe, extra_args) for a backend name, or (None, None)."""
    s = spec_for(cfg, name)
    exe = s.get("exe") or os.environ.get("SIMKIT_BACKEND_" + name.upper())
    if not exe:
        return None, None
    exe = os.path.expandvars(os.path.expanduser(exe))
    return exe, s.get("args", [])


def run(cfg, out_dir, manifest_path, name, quiet=False):
    exe, extra = resolve(cfg, name)
    if not exe:
        raise RuntimeError("backend '%s' not configured: add backends.%s.exe to the scene JSON "
                           "or set SIMKIT_BACKEND_%s" % (name, name, name.upper()))
    if not os.path.exists(exe):
        raise RuntimeError("backend '%s' executable not found: %s" % (name, exe))
    cmd = [exe] + [str(a) for a in extra] + [
        "--config", os.path.abspath(cfg.get("_path", "scene.json")),
        "--manifest", os.path.abspath(manifest_path),
        "--out", os.path.abspath(out_dir),
    ]
    if not quiet:
        print("  backend %s: %s" % (name, " ".join(shlex.quote(c) for c in cmd[:3]) + " ..."))
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        raise RuntimeError("backend '%s' failed (rc=%d)\n%s\n%s"
                           % (name, p.returncode, (p.stdout or "")[-2000:], (p.stderr or "")[-2000:]))
    if not quiet:
        for line in (p.stdout or "").splitlines():
            if line.strip():
                print("  [%s] %s" % (name, line))
    return check_outputs(out_dir)


def check_outputs(out_dir):
    img_dir = os.path.join(out_dir, "images", "optical")
    lab_dir = os.path.join(out_dir, "labels", "optical")
    imgs = sorted(f for f in os.listdir(img_dir)) if os.path.isdir(img_dir) else []
    if not imgs:
        raise RuntimeError("backend wrote no images to %s" % img_dir)
    missing = []
    for f in imgs:
        stem = os.path.splitext(f)[0]
        if not os.path.exists(os.path.join(lab_dir, stem + ".json")):
            missing.append(stem)
    if missing:
        raise RuntimeError("backend wrote %d images but no labels for: %s"
                           % (len(imgs), ", ".join(missing[:5])))
    return dict(images=len(imgs))


def write_manifest(cfg, states, out_dir, part=None):
    from . import labels as lab
    # Parallel chunks share the output directory, so the manifest name must be
    # per-chunk or one worker renders another worker's frames.
    if part is None:
        path = os.path.join(out_dir, "optical_frames.json")
    else:
        path = os.path.join(out_dir, "optical_frames.part%03d.json" % int(part))
    # "frame" is the global frame id. Without it a chunked run renders every
    # chunk as 000000.png.. and the chunks silently overwrite each other.
    def _cams(s):
        out = {}
        for k, v in ((s.get("cfg") or {}).get("sensors") or {}).items():
            if isinstance(v, dict) and "pos" in v:
                out[k] = dict(pos=[float(x) for x in v["pos"]],
                              look_at=[float(x) for x in v.get("look_at", [0, 0, 0])],
                              fov_deg=v.get("fov_deg"))
        return out

    # "camera" is per frame because a seeker/gimbal mount moves: without it the
    # optical renderer would shoot every frame from the static config pose while
    # IR/radar/lidar tracked the target, and the boxes would disagree.
    lab.write_json(path, dict(frames=[dict(frame=int(s["frame"]), t=s["t"],
                                            objects=s["poses"],
                                            camera=_cams(s),
                                            range_m=s.get("range_m"),
                                            weather=dict(s.get("weather") or {}))
                                       for s in states]))
    return path
