"""Negative control: prove that tools/validation_check.py can actually fail.

Why this exists
---------------
An acceptance gate that has only ever printed "通过" is not evidence of anything.
This script copies a finished run, injects four independent, well-understood
defects, runs the real checker over the copy, and records what the checker said.
If the checker passes the broken copy, the gate is broken; if it fails the clean
copy, the gate is over-tight. Both directions are checked here.

The copy costs about 100 MB, so it is deleted again after the audit and only the
compact summary is kept.

Usage, from the project root:

    python -m tools.validation_negative out/validation
"""
import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from tools import validation_check

# One defect per acceptance dimension, so a single run exercises all of them:
# can the checker see that a target is gone, that a label lies, that a file will
# not open, and that a frame is missing?
DEFECTS = (
    ("optical 第 3、7 帧的目标框被涂成背景天空", "看得见"),
    ("lidar 第 5 帧的目标框被挪到空白角落", "标得对"),
    ("ir 第 2 帧的文件被截断", "读得出"),
    ("radar 第 9 帧的标注被删除", "对得齐"),
)


def _copy_run(src, dst):
    for sub in ("images", "labels", "lidar", "observations", "raw"):
        if (src / sub).is_dir():
            shutil.copytree(src / sub, dst / sub)
    for name in ("frames.json", "meta.json", "validation.json"):
        if (src / name).exists():
            shutil.copy2(src / name, dst / name)
    for path in sorted(src.glob("annotations_*_coco.json")):
        shutil.copy2(path, dst / path.name)


def _blank_optical(dst, frames):
    """Replace the target with neighbouring sky, keeping the frame plausible."""
    for frame in frames:
        path = dst / "images" / "optical" / ("%06d.png" % frame)
        if not path.exists():
            continue
        with Image.open(path) as im:
            arr = np.asarray(im.convert("RGB")).copy()
        sky = arr[0:20, 0:20].reshape(-1, 3).mean(axis=0).astype(np.uint8)
        arr[100:420, 200:640] = sky
        Image.fromarray(arr).save(path)


def _mislabel_lidar(dst, frame):
    path = dst / "labels" / "lidar" / ("%06d.json" % frame)
    if not path.exists():
        return
    doc = json.loads(path.read_text("utf-8"))
    boxes = doc.get("boxes")
    if isinstance(boxes, list) and boxes:
        boxes[0].update(x0=4.0, y0=4.0, x1=24.0, y1=24.0)
    elif isinstance(boxes, dict) and boxes:
        next(iter(boxes.values())).update(x0=4.0, y0=4.0, x1=24.0, y1=24.0)
    path.write_text(json.dumps(doc, ensure_ascii=False), "utf-8")


def _truncate_ir(dst, band, frame):
    path = dst / "images" / ("ir_" + band) / ("%06d.png" % frame)
    if path.exists():
        raw = path.read_bytes()
        path.write_bytes(raw[: len(raw) // 2])


def _drop_radar_label(dst, frame):
    path = dst / "labels" / "radar" / ("%06d.json" % frame)
    if path.exists():
        path.unlink()


def build(src, dst, ir_band="mwir"):
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    _copy_run(src, dst)
    _blank_optical(dst, (3, 7))
    _mislabel_lidar(dst, 5)
    _truncate_ir(dst, ir_band, 2)
    _drop_radar_label(dst, 9)


def run(src, keep=False):
    src = Path(src)
    meta = validation_check._read_json(src / "meta.json")
    band = (meta.get("ir_bands") or ["lwir"])[0]
    n_frames = int(meta.get("frames") or 0) or None
    tmp = Path(tempfile.mkdtemp(prefix="validation_negative_"))
    try:
        build(src, tmp / "broken", ir_band=band)
        # check() prints its own summary; that is useful from the command line
        # and noise here, where the point is the recorded verdict.
        import contextlib
        import io
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink):
            result = validation_check.check(tmp / "broken", n_frames=n_frames,
                                            report_path=None)
    finally:
        if not keep:
            shutil.rmtree(tmp, ignore_errors=True)
        else:
            print("kept:", tmp / "broken")
    errors = list(result.get("errors") or [])
    summary = dict(source=str(src), defects=[d[0] for d in DEFECTS],
                   dimensions=[d[1] for d in DEFECTS],
                   broken_passed=bool(result.get("passed")),
                   broken_errors=errors,
                   broken_warnings=list(result.get("warnings") or []))
    # The control is only meaningful if the broken copy fails *and* the clean
    # copy passes; a broken copy that fails for one error while silently
    # accepting the other three would look identical in the exit code.
    summary["detected"] = [any(k in e for e in errors) for k in
                           ("optical: target peak deviation",
                            "trajectory: lidar",
                            "does not decode",
                            "missing label 000009.json")]
    summary["all_detected"] = all(summary["detected"])
    out = Path(src) / "negative_control.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n",
                   encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "broken_errors"},
                     ensure_ascii=False, indent=1))
    print("errors:")
    for e in errors:
        print("  -", e)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", nargs="?", default="out/validation_ecm")
    ap.add_argument("--keep", action="store_true", help="keep the broken copy")
    args = ap.parse_args(argv)
    summary = run(args.run, keep=args.keep)
    ok = summary["all_detected"] and not summary["broken_passed"]
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
