"""End-to-end acceptance for the dataset platform.

Runs one recipe through the real pipeline and checks the things a user would
check by opening the folder: did every requested modality appear, are the COCO
boxes inside the frame, is there a composite preview, is there a README.

    python tools/accept_platform.py                       # 4 frames, all checks
    python tools/accept_platform.py --frames 20            # the full acceptance
    python tools/accept_platform.py --recipe configs/recipes/ecm_suppression_headon.json
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from simkit import scenegen, scene            # noqa: E402

MOD_DIR = {"optical": "images/optical", "ir": "images/ir_mwir",
           "radar": "images/radar", "sar": "images/sar", "lidar": "lidar/range"}


def _pngs(d):
    if not os.path.isdir(d):
        return []
    return sorted(f for f in os.listdir(d) if f.lower().endswith(".png"))


def check(run_dir, recipe, frames):
    """Return [(ok, label, detail)] -- one row per thing worth asserting."""
    rows = []
    want = [s for s in (recipe.get("capture") or {}).get("sensors") or []]

    for s in want:
        rel = MOD_DIR.get(s)
        if not rel:
            continue
        got = _pngs(os.path.join(run_dir, rel))
        rows.append((len(got) == frames, "%s 图像数 = 帧数" % s,
                     "%d / %d" % (len(got), frames)))

    # COCO: parseable, and every box actually lies inside its image.
    # Radar is the odd one out: a detection is one range/Doppler cell, so the
    # labels are per-frame point files under labels/radar/, not a COCO set. That
    # is a property of the sensor, not a gap in the export.
    rdir = os.path.join(run_dir, "labels", "radar")
    if "radar" in want:
        got = _pngs(rdir) + [f for f in (os.listdir(rdir) if os.path.isdir(rdir) else [])
                             if f.endswith(".json")]
        rows.append((len(got) == frames, "radar 逐帧点标注",
                     "%d / %d" % (len(got), frames)))
        # A radar image with no detection is a blank image. Counting files is
        # not enough: a target sitting beyond r_max = c*n/(4B) passes every
        # file-level check and still trains the detector on noise.
        hit = 0
        for f in range(frames):
            p = os.path.join(rdir, "%06d.json" % f)
            if not os.path.isfile(p):
                continue
            try:
                with open(p, "r", encoding="utf-8") as fh:
                    jl = json.load(fh)
            except ValueError:
                continue
            if jl.get("boxes_rd"):
                hit += 1
        need = max(1, int(0.8 * frames))
        rows.append((hit >= need, "radar 真的检测到目标",
                     "%d / %d 帧有检测（要求 >= %d）" % (hit, frames, need)))

    for s in want:
        name = {"optical": "annotations_optical_coco.json",
                "ir": "annotations_ir_mwir_coco.json",
                "sar": "annotations_sar_coco.json"}.get(s)
        if not name:
            continue
        # The COCO file sits at the run root next to meta.json; labels/ holds the
        # per-frame JSON. Look in both so a layout change cannot look like a
        # missing annotation file.
        p = None
        for cand in (os.path.join(run_dir, name), os.path.join(run_dir, "labels", name)):
            if os.path.isfile(cand):
                p = cand
                break
        if p is None:
            rows.append((False, "%s COCO 存在" % s, "缺少 " + name))
            continue
        try:
            with open(p, "r", encoding="utf-8") as fh:
                doc = json.load(fh)
        except ValueError as exc:
            rows.append((False, "%s COCO 可解析" % s, str(exc)))
            continue
        size = {im["id"]: (im.get("width"), im.get("height"))
                for im in doc.get("images") or []}
        bad = 0
        for a in doc.get("annotations") or []:
            w, h = size.get(a.get("image_id"), (None, None))
            if not w or not h:
                bad += 1
                continue
            x, y, bw, bh = (a.get("bbox") or [0, 0, 0, 0])[:4]
            if x < -0.5 or y < -0.5 or x + bw > w + 0.5 or y + bh > h + 0.5:
                bad += 1
        n_ann = len(doc.get("annotations") or [])
        # Zero annotations is not a pass: an empty label file means this
        # modality contributed nothing to training.
        rows.append((bad == 0 and n_ann > 0, "%s COCO 框在画面内" % s,
                     "%d 张图 / %d 标注 / %d 越界" % (len(size), n_ann, bad)))

    # Composite preview: one triptych per frame plus a contact sheet.
    prev = _pngs(os.path.join(run_dir, "images/preview"))
    rows.append((len(prev) == frames, "三模态拼接图", "%d / %d" % (len(prev), frames)))
    rows.append((os.path.isfile(os.path.join(run_dir, "preview_sheet.png")),
                 "总览接触表", "preview_sheet.png"))

    # The human-readable document is the deliverable the user asked for.
    doc = os.path.join(run_dir, "数据集说明.md")
    if os.path.isfile(doc):
        with open(doc, "r", encoding="utf-8") as fh:
            text = fh.read()
        need = ["场景", "传感器", "干扰", "标注", "复现", "局限"]
        miss = [k for k in need if k not in text]
        rows.append((not miss, "数据集说明.md 章节齐全",
                     "%d 字%s" % (len(text), "" if not miss else "，缺: " + ",".join(miss))))
    else:
        rows.append((False, "数据集说明.md 存在", "缺少"))

    # ECM must be recorded when the recipe asked for it.
    if (recipe.get("ecm") or {}).get("entities"):
        with open(os.path.join(run_dir, "meta.json"), "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        rows.append((bool(meta.get("ecm")), "meta.json 记录 ECM",
                     "entities = %d" % len((meta.get("ecm") or {}).get("entities") or [])))

    post = json.load(open(os.path.join(run_dir, "meta.json"), encoding="utf-8")).get("post") or {}
    rows.append(("error" not in post, "后处理无报错", str(post.get("error") or "ok")))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recipe", default=os.path.join(
        ROOT, "configs", "recipes", "trilateral_sky_headon.json"))
    ap.add_argument("--frames", type=int, default=4, help="small by default: this is a smoke test")
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "accept_platform"))
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args(argv)

    with open(args.recipe, "r", encoding="utf-8") as fh:
        recipe = scenegen.validate_recipe(json.load(fh))
    recipe.setdefault("capture", {})["frames"] = int(args.frames)

    cfgp = os.path.join(ROOT, "_tmp", "accept_scene.json")
    os.makedirs(os.path.dirname(cfgp), exist_ok=True)
    with open(cfgp, "w", encoding="utf-8") as fh:
        json.dump(scenegen.build_scene(recipe), fh, ensure_ascii=False, indent=1)

    if os.path.isdir(args.out) and not args.keep:
        shutil.rmtree(args.out)
    sensors = ",".join(recipe["capture"]["sensors"])
    cmd = [sys.executable, "-m", "simkit", "gen", "-c", cfgp,
           "-o", args.out, "-n", str(args.frames), "-s", sensors]
    print("运行: %s" % " ".join(cmd[1:]))
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        print(proc.stdout[-2000:])
        print(proc.stderr[-2000:])
        print("FAIL: 生成失败")
        return 1

    rows = check(args.out, recipe, args.frames)
    width = max(len(r[1]) for r in rows)
    bad = 0
    for ok, label, detail in rows:
        print("  %s  %-*s  %s" % ("PASS" if ok else "FAIL", width, label, detail))
        bad += 0 if ok else 1
    print()
    print("%d 项检查，%d 项通过，%d 项失败" % (len(rows), len(rows) - bad, bad))
    print("产物: " + args.out)
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
