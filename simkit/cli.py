"""simkit command line."""
import argparse
import json
import os
import sys
import time

from . import dataset, scene


def cmd_info(args):
    cfg = scene.load_config(args.scene)
    print(json.dumps({k: v for k, v in cfg.items() if not k.startswith("_")},
                     ensure_ascii=False, indent=2)[:4000])


def _split(spec):
    """'3/12' -> (3, 12); used by --part for chunked generation."""
    if not spec:
        return None, None, None
    a, _, b = str(spec).partition("/")
    if not b:
        raise SystemExit("--part wants INDEX/TOTAL, e.g. --part 3/12")
    i, n = int(a), int(b)
    if n < 1 or not (0 <= i < n):
        raise SystemExit("--part out of range: %s" % spec)
    return i, n, None


def cmd_gen(args):
    sensors = tuple(s.strip() for s in args.sensors.split(",") if s.strip())
    part, nparts, _ = _split(args.part)
    if part is None:
        dataset.run(args.scene, args.out, n_frames=args.frames, sensors=sensors,
                    seed=args.seed, quiet=args.quiet)
        return 0
    # frame counts per chunk; the remainder goes to the first chunks
    total = int(args.frames if args.frames is not None else 0)
    if not total:
        raise SystemExit("--part needs -n <total frames>")
    per = [total // nparts + (1 if i < total % nparts else 0) for i in range(nparts)]
    offset = sum(per[:part])
    dataset.run(args.scene, args.out, n_frames=per[part], sensors=sensors,
                seed=args.seed, quiet=args.quiet, frame_offset=offset, part=part,
                total_frames=total)
    return 0


def cmd_merge(args):
    sensors = tuple(s.strip() for s in args.sensors.split(",") if s.strip())
    dataset.merge_parts(args.out, args.scene, n_frames=args.frames, sensors=sensors,
                        seed=args.seed, quiet=args.quiet, t_start=time.time())
    return 0


def cmd_selftest(args):
    from .selftest import run
    run()


def cmd_vega(args):
    from . import vega
    vega.describe()
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="simkit", description="Offline multi-sensor dataset generator")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen", help="generate a dataset from a scene config")
    g.add_argument("-c", "--scene", required=True)
    g.add_argument("-o", "--out", required=True)
    g.add_argument("-n", "--frames", type=int, default=None)
    g.add_argument("-s", "--sensors", default="ir,radar")
    g.add_argument("--seed", type=int, default=None)
    g.add_argument("-q", "--quiet", action="store_true")
    g.add_argument("--part", default=None,
                   help="generate only chunk INDEX/TOTAL of the run (parallel workers)")
    g.set_defaults(func=cmd_gen)

    m = sub.add_parser("merge", help="stitch chunked generation into one dataset")
    m.add_argument("-c", "--scene", required=True)
    m.add_argument("-o", "--out", required=True)
    m.add_argument("-n", "--frames", type=int, default=None, help="total frames of the run")
    m.add_argument("-s", "--sensors", default="ir,radar,lidar,optical")
    m.add_argument("--seed", type=int, default=None)
    m.add_argument("-q", "--quiet", action="store_true")
    m.set_defaults(func=cmd_merge)

    i = sub.add_parser("info", help="print a scene config")
    i.add_argument("-c", "--scene", required=True)
    i.set_defaults(func=cmd_info)

    s = sub.add_parser("selftest", help="run internal correctness checks")
    s.set_defaults(func=cmd_selftest)

    v = sub.add_parser("vega", help="check the Vega Prime integration status")
    v.set_defaults(func=cmd_vega)

    args = ap.parse_args(argv)
    rc = args.func(args)
    if isinstance(rc, int):
        sys.exit(rc)


if __name__ == "__main__":
    main()