#!/usr/bin/env python3
r"""Compare rendered frames against the real-photo reference library.

Single class:
    python tools/compare_refs.py --synth out\final_air ^
        --refs assets\refs\军用飞机 --out _tmp\cmp.json

Every reference class:
    python tools\compare_refs.py --all --out _tmp\cmp_all.json

Exit status is 0 only when every gated feature is inside its threshold and both
sides produced usable images; 1 otherwise (including "reference library not
built yet", which degrades to a report instead of a traceback).  Usage errors
(missing --synth/--refs, bad --pair syntax) exit 2, the argparse convention.

The tool consumes, but never edits:
    assets/refs/<class>/*.jpg          reference photos
    assets/refs/catalog.json           optional provenance / manual boxes
    configs/refs_params.json           optional class -> synth dir mapping
    out/<dataset>/images/<modality>/   renders
    out/<dataset>/labels/<modality>/   per-frame annotations
    out/<dataset>/annotations_*_coco.json
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from simkit import refcompare  # noqa: E402


EXAMPLES = """examples:
  # one class, human table + machine readable json
  python tools/compare_refs.py --synth out/final_air --refs assets/refs/军用飞机 --out _tmp/cmp.json

  # every class known to assets/refs/; pairing comes from a "synth" entry in
  # configs/refs_params.json, directory-name matching or --pair (fallback)
  python tools/compare_refs.py --all --out _tmp/cmp_all.json

  # when the params file does not exist yet, pair classes explicitly:
  python tools/compare_refs.py --all --pair 军用飞机=out/final_air --out _tmp/cmp_all.json
"""


def _configure_stdout():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):  # pragma: no cover - exotic streams
            pass


def _parse_pairs(values):
    overrides = {}
    for item in values or []:
        if "=" not in item:
            raise ValueError("--pair 需要 类别=合成目录 的形式，收到 %r" % item)
        class_name, synth_dir = item.split("=", 1)
        class_name = class_name.strip()
        synth_dir = synth_dir.strip()
        if not class_name or not synth_dir:
            raise ValueError("--pair 的类别和目录都不能为空: %r" % item)
        overrides[class_name] = synth_dir
    return overrides


def build_parser():
    parser = argparse.ArgumentParser(
        prog="compare_refs.py",
        description="把渲染图与实拍参考图库放在一起量化比较（纯 numpy 图像统计）。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EXAMPLES)
    parser.add_argument("--synth", metavar="DIR",
                        help="合成输出目录（含 images/<modality> 与 labels/<modality>）")
    parser.add_argument("--refs", metavar="DIR",
                        help="参考图类别目录（assets/refs/<类名>）")
    parser.add_argument("--all", action="store_true",
                        help="遍历参考图库所有类别；按 refs_params.json 映射、"
                             "目录名或唯一候选配对，未配对会列出")
    parser.add_argument("--out", metavar="JSON",
                        help="机器可读报告的输出路径；省略则只打印")
    parser.add_argument("--refs-root", metavar="DIR",
                        default=os.path.join(ROOT, "assets", "refs"),
                        help="参考图库根目录（默认 assets/refs）")
    parser.add_argument("--synth-root", metavar="DIR",
                        default=os.path.join(ROOT, "out"),
                        help="合成输出根目录（默认 out）")
    parser.add_argument("--params", metavar="JSON",
                        default=os.path.join(ROOT, "configs", "refs_params.json"),
                        help="可选：类别->目录映射与参考量测"
                             "（默认 configs/refs_params.json）")
    parser.add_argument("--catalog", metavar="JSON", default=None,
                        help="参考图目录 JSON（默认 <refs 目录>/../catalog.json）")
    parser.add_argument("--pair", action="append", metavar="类别=DIR", default=[],
                        help="--all 时显式指定类别到合成目录；可重复")
    parser.add_argument("--modality", default="optical",
                        help="图像子目录/标注前缀，默认 optical")
    parser.add_argument("--max-side", type=int,
                        default=refcompare.ANALYSIS_MAX_SIDE,
                        help="统一分析画布的长边像素（默认 %d）"
                             % refcompare.ANALYSIS_MAX_SIDE)
    parser.add_argument("--max-images", type=int,
                        default=refcompare.MAX_IMAGES_PER_SIDE,
                        help="每侧最多分析的图像数（默认 %d）"
                             % refcompare.MAX_IMAGES_PER_SIDE)
    parser.add_argument("--class-name", default=None,
                        help="报告中使用的类别名（默认取参考目录名）")
    parser.add_argument("--quiet", action="store_true",
                        help="不打印完整特征表，只打印结论摘要")
    parser.add_argument("--list-features", action="store_true",
                        help="列出参与判定的特征后退出")
    return parser


def _brief(report):
    summary = report.get("summary") or {}
    name = report.get("class") or os.path.basename(str(report.get("ref_dir", "?")))
    status = report.get("status")
    if status != "compared":
        reason = "；".join(summary.get("reasons") or ["数据不足"])
        return "[%s] 未比较: %s" % (name, reason)
    gated = summary.get("gated", 0)
    failed = summary.get("failed", 0)
    if summary.get("pass"):
        return "[%s] 通过: %d/%d 项门控指标在阈值内" % (name, gated, gated)
    reasons = "、".join(summary.get("reasons") or ["指标超阈值"])
    return "[%s] 未通过: %d/%d 项超阈值 -> %s" % (name, failed, gated, reasons)


def main(argv=None):
    _configure_stdout()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_features:
        print("%-24s %-28s %s" % ("key", "label", "gated"))
        for item in refcompare.describe_feature_set():
            print("%-24s %-28s %s" % (item["key"], item["label"],
                                      "yes" if item["gate"] else "no"))
        return 0

    if args.all:
        if args.synth or args.refs:
            parser.error("--all 不能与 --synth/--refs 同时使用；单类比较请分别给出两者")
        try:
            overrides = _parse_pairs(args.pair)
        except ValueError as exc:
            parser.error(str(exc))
        multi = refcompare.compare_all(
            refs_root=args.refs_root,
            synth_root=args.synth_root,
            params_path=args.params,
            out_json=args.out,
            modality=args.modality,
            max_side=args.max_side,
            max_images=args.max_images,
            print_table=not args.quiet,
            overrides=overrides,
            catalog_path=args.catalog,
        )
        summary = multi.get("summary") or {}
        if args.quiet:
            print("批量对比: %d 类，通过 %d，未通过 %d"
                  % (len(multi.get("classes") or []),
                     summary.get("passed", 0), summary.get("failed", 0)))
            for report in multi.get("classes") or []:
                print("  " + _brief(report))
            unmatched = multi.get("unmatched") or []
            ref_left = [item for item in unmatched if item.get("side") == "ref"]
            synth_left = [item for item in unmatched if item.get("side") != "ref"]
            for item in ref_left:
                print("  未配对 [参考] %s: %s"
                      % (item.get("class", "?"), item.get("reason", "")))
            if len(synth_left) <= 5:
                for item in synth_left:
                    print("  未配对 [合成] %s: %s"
                          % (item.get("class", "?"), item.get("reason", "")))
            elif synth_left:
                print("  未配对 [合成] 另有 %d 个输出目录没有参考类别（详见 JSON）"
                      % len(synth_left))
            for warning in multi.get("warnings") or []:
                print("!! " + warning)
        elif not (multi.get("classes") or []):
            print("没有可比较的类别（可用 --pair 类别=合成目录 指定配对）。")
        if args.out:
            print("JSON: %s" % os.path.abspath(args.out))
        return 0 if summary.get("pass") else 1

    if not args.synth or not args.refs:
        parser.error("需要同时给出 --synth 和 --refs，或使用 --all")

    class_name = args.class_name or os.path.basename(os.path.normpath(args.refs))
    params_entry = None
    params = refcompare.load_ref_params(args.params)
    for warning in params.get("warnings") or []:
        print("!! " + warning)
    params_entry = params.get("classes", {}).get(class_name)
    if params_entry is None:
        # tolerate a params file keyed by a normalized spelling
        for name, entry in params.get("classes", {}).items():
            if name.strip().lower() == class_name.strip().lower():
                params_entry = entry
                break

    report = refcompare.compare(
        args.synth,
        args.refs,
        out_json=args.out,
        modality=args.modality,
        max_side=args.max_side,
        max_images=args.max_images,
        params=params_entry,
        class_name=class_name,
        print_table=not args.quiet,
        catalog_path=args.catalog,
    )
    if args.quiet:
        print(_brief(report))
        for warning in report.get("warnings") or []:
            print("!! " + warning)
    if args.out:
        print("JSON: %s" % os.path.abspath(args.out))
    return 0 if (report.get("summary") or {}).get("pass") else 1


if __name__ == "__main__":
    sys.exit(main())
