"""Cross-entity consistency checker for multi-target scenes.

Wraps simkit.consistency.inspect(): runs the 25 rules of
docs/多目标运动仿真调研.md section 4 over one scene or every scene, prints a
matrix of which scene fails which check, and exits non-zero when anything fails.

Usage
-----
    python tools/check_consistency.py                           # every scene
    python tools/check_consistency.py scenes/demo_all.json      # one scene
    python tools/check_consistency.py scenes/demo_all.json --verbose
    python tools/check_consistency.py --frames 30 --json out/consistency.json
    python tools/check_consistency.py --list-checks

Exit codes: 0 every check passed (skips count as pass), 1 at least one check
failed or a scene could not be inspected, 2 usage error.
"""
import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from simkit import consistency  # noqa: E402

STATUS = {True: "PASS", False: "FAIL"}


def _scene_label(path, res):
    """The report is keyed by file name, which is what a user typed."""
    s = res.setdefault("summary", {})
    if s.get("scene") and s.get("name") is None:
        s["name"] = s["scene"]
    s["scene"] = os.path.basename(path)
    s["path"] = os.path.abspath(path)
    return res


def _error_result(path, exc):
    import traceback
    detail = "%s: %s" % (type(exc).__name__, exc)
    return dict(
        ok=False,
        checks=[dict(id="ERR", group="?", ok=False, applicable=True,
                     detail=detail, value=None, limit=None)],
        summary=dict(scene=os.path.basename(path), path=os.path.abspath(path),
                     mode="error", frames=0, dt=None, failed=["ERR"],
                     skipped=[], checks=1, passed=0, notes=[],
                     error=detail, traceback=traceback.format_exc()),
    )


def _print_detail(results):
    for res in results:
        s = res.get("summary") or {}
        print("=" * 100)
        print("%s  ok=%s  mode=%s  frames=%s  dt=%s  objects=%s  ecm=%s"
              % (s.get("scene"), res.get("ok"), s.get("mode"), s.get("frames"),
                 s.get("dt"), len(s.get("objects") or []), s.get("ecm_entities", 0)))
        for c in res.get("checks", []):
            status = "skip" if not c.get("applicable", True) else STATUS[c["ok"]]
            value = "-" if c.get("value") is None else ("%.6g" % c["value"])
            limit = "-" if c.get("limit") is None else ("%.6g" % c["limit"])
            print("  %-3s %-4s value=%-12s limit=%-12s %s"
                  % (c["id"], status, value, limit, c["detail"]))
        for note in s.get("notes") or []:
            print("  note  %s" % note)


def _write_json(path, results):
    full = path if os.path.isabs(path) else os.path.join(ROOT, path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=1)
    return full


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Check multi-target scenes against docs/多目标运动仿真调研.md section 4")
    ap.add_argument("scenes", nargs="*",
                    help="scene or recipe JSON files (default: every scenes/*.json)")
    ap.add_argument("--frames", type=int, default=None,
                    help="sample at most N evenly spaced frames (default: every frame)")
    ap.add_argument("--json", default=None, help="write the full report as JSON")
    ap.add_argument("--verbose", action="store_true",
                    help="print every check for every scene, not only failures")
    ap.add_argument("--list-checks", action="store_true",
                    help="print the check catalogue (id, group, threshold) and exit")
    args = ap.parse_args(argv)

    if args.list_checks:
        for spec in consistency.CHECK_SPECS:
            print("%-3s %-2s %-46s %s" % (spec["id"], spec["group"],
                                          spec["title"], spec["threshold"]))
        return 0

    paths = list(args.scenes)
    if not paths:
        paths = sorted(glob.glob(os.path.join(ROOT, "scenes", "*.json")))
    if not paths:
        print("no scene files given and scenes/*.json is empty", file=sys.stderr)
        return 2
    if any(not os.path.exists(p) for p in paths):
        missing = [p for p in paths if not os.path.exists(p)]
        print("missing input: %s" % ", ".join(missing), file=sys.stderr)
        return 2

    results = []
    for path in paths:
        try:
            res = consistency.inspect(path, frames=args.frames)
        except Exception as exc:                       # one bad scene must not hide the rest
            res = _error_result(path, exc)
        results.append(_scene_label(path, res))

    if args.verbose:
        _print_detail(results)
        print("=" * 100)
    print(consistency.format_report(results))
    if args.json:
        print("json: %s" % _write_json(args.json, results))
    bad = sum(1 for r in results if not r.get("ok"))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
