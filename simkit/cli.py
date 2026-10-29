"""simkit command line."""
import argparse
import json
import os
import sys

from . import dataset, scene


def cmd_info(args):
    cfg = scene.load_config(args.scene)
    print(json.dumps({k: v for k, v in cfg.items() if not k.startswith("_")},
                     ensure_ascii=False, indent=2)[:4000])


def cmd_gen(args):
    sensors = tuple(s.strip() for s in args.sensors.split(",") if s.strip())
    dataset.run(args.scene, args.out, n_frames=args.frames, sensors=sensors,
                seed=args.seed, quiet=args.quiet)


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
    g.set_defaults(func=cmd_gen)

    i = sub.add_parser("info", help="print a scene config")
    i.add_argument("-c", "--scene", required=True)
    i.set_defaults(func=cmd_info)

    s = sub.add_parser("selftest", help="run internal correctness checks")
    s.set_defaults(func=cmd_selftest)

    v = sub.add_parser("vega", help="check the Vega Prime integration status")
    v.set_defaults(func=cmd_vega)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
