"""One command: seeker clips -> datasets -> merged COCO -> website data.

    python tools\\build_seeker.py                      # 3 scenes x 240 frames, all sensors
    python tools\\build_seeker.py --no-optical          # numpy only, minutes not hours
    python tools\\build_seeker.py -n 24 --jobs 4        # 4 chunked workers per scene
    python tools\\build_seeker.py --skip-gen            # just re-merge / re-export

Why a separate entry point from build_all.py: the seeker set is a different
product. Each scene is two 120-frame clips at 20 fps, rendered from a diving,
weaving platform through two field-of-view lenses at once, with SAR on board.
That is 4-8x the pixels per scene and it wants chunked parallelism, so it gets
its own defaults instead of quietly changing the ones the old demo relies on.

Chunking is exact: `--part i/N` derives clip index and local time from the
GLOBAL frame id, so N workers reproduce the single-process run bit for bit.
"""
import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable

SCENES = [
    # four engagement geometries, two 120-frame clips each
    ("seeker_air2air_high", "scenes/sk3_high_air2air.json", 240),
    ("seeker_surface2air", "scenes/sk3_low_surface2air.json", 240),
    ("seeker_air2air_occluded", "scenes/sk3_low_air2air_occluded.json", 240),
    ("seeker_air2ground", "scenes/sk3_low_air2ground_clutter.json", 240),
]


def call(args, label, env):
    t0 = time.time()
    rc = subprocess.call(args, cwd=ROOT, env=env)
    return rc, label, time.time() - t0


def run(args, label):
    env = dict(os.environ)
    env["PYTHONPATH"] = ROOT
    print("\n== %s" % label)
    print("   " + " ".join(args[1:]))
    rc, _, dt = call(args, label, env)
    if rc != 0:
        print("FAILED (%s) after %.1fs" % (label, dt))
        sys.exit(rc)
    print("   ok  %.1fs" % dt)


def gen_scene(name, cfg, frames, sensors, jobs, quiet=False):
    """Generate one scene, optionally as `jobs` parallel chunks then merged."""
    env = dict(os.environ)
    env["PYTHONPATH"] = ROOT
    # Split the cores between chunks: each Cycles instance defaults to all of
    # them, and N instances then fight over the same cores.
    cores = os.cpu_count() or 4
    env["SIMKIT_BLENDER_THREADS"] = str(max(1, cores // max(jobs, 1)))
    out = "out/" + name
    if jobs <= 1:
        rc, _, dt = call([PY, "-m", "simkit", "gen", "-c", cfg, "-o", out,
                          "-n", str(frames), "-s", sensors] + (["-q"] if quiet else []),
                         "generate " + name, env)
        if rc != 0:
            sys.exit(rc)
        print("   ok  %-22s %.1fs" % (name, dt))
        return
    print("\n== generate %s (%d chunks x %d frames)" % (name, jobs, frames))
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = []
        for i in range(jobs):
            args = [PY, "-m", "simkit", "gen", "-c", cfg, "-o", out,
                    "-n", str(frames), "-s", sensors, "--part", "%d/%d" % (i, jobs)]
            if quiet:
                args.append("-q")
            futs.append(ex.submit(call, args, "%s part %d/%d" % (name, i, jobs), env))
        bad = []
        for f in futs:
            rc, label, dt = f.result()
            print("   %-28s %s %.1fs" % (label, "ok " if rc == 0 else "FAIL", dt))
            if rc != 0:
                bad.append(label)
    if bad:
        sys.exit("chunk(s) failed: " + ", ".join(bad))
    print("   merging chunks ...")
    run([PY, "-m", "simkit", "merge", "-c", cfg, "-o", out,
         "-n", str(frames), "-s", sensors], "merge chunks " + name)
    print("   ok  %-22s %.1fs" % (name, time.time() - t0))


def main():
    ap = argparse.ArgumentParser(description="build the seeker-view dataset")
    ap.add_argument("-n", "--frames", type=int, default=None,
                    help="frames per scene (default: each scene's own count)")
    ap.add_argument("-s", "--sensors", default="optical,ir,radar,lidar,sar")
    ap.add_argument("--jobs", type=int, default=1, help="parallel chunks per scene")
    ap.add_argument("--no-optical", action="store_true",
                    help="skip the Blender pass (numpy sensors only, much faster)")
    ap.add_argument("--skip-gen", action="store_true", help="only merge / export / thumbnail")
    ap.add_argument("--only", default="",
                    help="comma-separated scene names to (re)generate, e.g. seeker_air2ground")
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args()

    sensors = args.sensors
    if args.no_optical:
        sensors = ",".join(s for s in sensors.split(",") if s.strip() and s.strip() != "optical")

    only = {s.strip() for s in (args.only or "").split(",") if s.strip()}
    if not args.skip_gen:
        for name, cfg, frames in SCENES:
            if only and name not in only:
                print("skip (not in --only): " + name)
                continue
            if not os.path.exists(os.path.join(ROOT, cfg)):
                print("skip (no scene): " + cfg)
                continue
            gen_scene(name, cfg, args.frames or frames, sensors, args.jobs, args.quiet)

    runs = ["out/" + n for n, _, _ in SCENES if os.path.isdir(os.path.join(ROOT, "out", n))]
    if not runs:
        sys.exit("nothing generated: no out/ directories")

    run([PY, "tools/merge_dataset.py", "out/seeker_dataset"] + runs, "merge into one COCO")
    run([PY, "tools/export_web.py", ".", "out"], "export website data")
    for n, _, _ in SCENES:
        if os.path.isdir(os.path.join(ROOT, "out", n)):
            run([PY, "tools/contact_sheet.py", "out/" + n, "out/sheet_" + n + ".png", "4"],
                "thumbnail " + n)
    print("\nall done.  double-click index.html, or: node tools/serve_web.js")


if __name__ == "__main__":
    main()
