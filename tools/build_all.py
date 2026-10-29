"""One command: scenes -> datasets -> merged COCO -> website data -> thumbnails.

    python tools\\build_all.py                 # everything, 6 frames per scene
    python tools\\build_all.py -n 3 -s ir,radar
    python tools\\build_all.py --skip-gen      # just re-export the site

This exists because the documented pipeline is four separate commands with
three different argument conventions, and getting them out of order produces a
website that silently shows stale images. It runs the same steps README
describes, in that order, and stops on the first failure.
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable

SCENES = [
    ("final_air", "scenes/final_air_uav.json"),
    ("final_ground", "scenes/final_ground_vehicle.json"),
    ("final_sea", "scenes/final_sea_boat.json"),
]


def run(args, label):
    t0 = time.time()
    print("\n== %s" % label)
    print("   " + " ".join(args[1:]))
    env = dict(os.environ)
    env["PYTHONPATH"] = ROOT
    rc = subprocess.call(args, cwd=ROOT, env=env)
    if rc != 0:
        print("FAILED (%s) after %.1fs" % (label, time.time() - t0))
        sys.exit(rc)
    print("   ok  %.1fs" % (time.time() - t0))


def main():
    ap = argparse.ArgumentParser(description="build the whole pipeline")
    ap.add_argument("-n", "--frames", type=int, default=6)
    ap.add_argument("-s", "--sensors", default="ir,radar,lidar,optical")
    ap.add_argument("--skip-gen", action="store_true", help="only merge/export/thumbnail")
    ap.add_argument("--skip-optical", action="store_true",
                    help="skip the Blender pass (slow; needs the portable Blender)")
    args = ap.parse_args()

    sensors = args.sensors
    if args.skip_optical:
        sensors = ",".join(s for s in sensors.split(",") if s.strip() and s.strip() != "optical")

    if not args.skip_gen:
        for name, cfg in SCENES:
            run([PY, "-m", "simkit", "gen", "-c", cfg, "-o", "out/" + name,
                 "-n", str(args.frames), "-s", sensors], "generate " + name)

    merged = ["out/dataset"] + ["out/" + n for n, _ in SCENES]
    run([PY, "tools/merge_dataset.py"] + merged, "merge into one COCO")
    run([PY, "tools/export_web.py", ".", "out"], "export website data")
    for n, _ in SCENES:
        run([PY, "tools/contact_sheet.py", "out/" + n, "out/sheet_" + n + ".png", "6"],
            "thumbnail " + n)

    print("\nall done.  open index.html, or: node tools/serve_web.js")


if __name__ == "__main__":
    main()
