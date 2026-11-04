"""Audit a 10-frame flight sample, including independent RGB/IR geometry.

Run from the repository root: python -m tools.check_minimal out/validation_fighter_10
"""
import argparse
import json
from pathlib import Path

import numpy as np

from simkit import validation
from .check_aviation import compare_geometry


def check(root):
    root = Path(root)
    meta = json.loads((root/"meta.json").read_text(encoding="utf-8"))
    frames = json.loads((root/"frames.json").read_text(encoding="utf-8"))
    report = validation.artifacts(root, len(frames), meta["sensors"])
    geometry = []
    for row in frames:
        stem = "%06d" % row["frame"]
        # Decode every delivered image using Pillow when available.
        from PIL import Image
        for path in [*root.glob("images/*/"+stem+".png"), *root.glob("lidar/*/"+stem+".png")]:
            try:
                with Image.open(path) as im:
                    im.verify()
            except Exception as exc:
                report["errors"].append("invalid image %s: %s" % (path, exc))
        for band in meta.get("ir_bands", []):
            a, b = root/"raw/optical"/(stem+".npz"), root/("raw/ir_"+band)/(stem+".npz")
            if not a.exists() or not b.exists():
                report["errors"].append("independent geometry missing for "+stem+" "+band)
                continue
            with np.load(a) as optical, np.load(b) as infrared:
                result = compare_geometry(optical, infrared, 0)
            geometry.append(dict(frame=row["frame"], band=band, **result))
            if result["status"] != "measured_within_tolerance":
                report["errors"].append("geometry check "+stem+": "+result["status"])
    report.update(geometry=geometry, passed=not report["errors"],
                  scope="image decoding and independent instance/depth agreement; not real-sensor calibration")
    (root/"quality_minimal.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "geometry"}, indent=2))
    return report["passed"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", type=Path)
    raise SystemExit(0 if check(ap.parse_args().run) else 1)
