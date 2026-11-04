#!/usr/bin/env python
"""One-page realism scorecard for the rendered airframe.

Renders one fixed close-up scene, crops the target with its own annotation box,
and compares the target region against the reference photographs in
assets/refs/<class>/ (segmented with simkit.refcompare's heuristic).

The generator is judged against real photographs, so every optimisation round
has to answer: did the target surface get measurably closer to a photographed
one?  Exit code 0 when every gated statistic is inside its band.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image
from numpy.lib.stride_tricks import sliding_window_view

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

# Bands: see docs/真实数据对比.md §4.14.  Contrast and mean colour are what the
# surface texture moves; a bare albedo multiply moves neither width nor detail.
GATES = {
    "contrast_ratio": 0.55,    # ours / reference, must reach at least this
    "colour_shift": 45.0,      # max |mean RGB difference| vs the reference
    # A regression guard, not an improvement target: the bbox Laplacian is
    # dominated by the sky inside the box, so it cannot express "how much
    # surface detail does the airframe have". The honest figure is
    # lap_ratio below, measured on the target's own pixels with both sides
    # normalised to the same target size.
    "lap_var_min": 13.0,
}


def _lap(gray):
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], float)
    return float((sliding_window_view(gray, (3, 3)) * k).sum(axis=(2, 3)).var())


def _measure(rgb255):
    return dict(mean=[float(rgb255[..., i].mean()) for i in range(3)],
                contrast=float(rgb255.std() / 255.0),
                lap=_lap(rgb255.mean(axis=2)))


def render(out_dir, skin=True, strength=1.0, px_m=0.04, skin_contrast=1.0,
           seed=1, f_number=None):
    from simkit import scenegen
    base = json.load(open(os.path.join(ROOT, "configs", "recipes",
                                       "trilateral_sky_headon.json"), encoding="utf-8"))
    rec = json.loads(json.dumps(base))
    # CLEAR, deliberately. The reference photographs are sunlit, and overcast is
    # by definition a flat dome: measured target-region std was 18.3 DN under
    # cloud against 30.1 DN under sun for the identical scene, so comparing a
    # cloudy render with a sunlit photo measures the weather, not the model.
    rec["environment"]["weather"] = "clear"
    rec["environment"]["background"] = {"mode": "hdri",
                                        "id": "kloofendal_43d_clear_puresky"}
    rec["objects"] = [{"asset": "军用飞机/双发战斗机.obj", "cls": "fighter",
                       "scale": 1.0, "count": 1}]
    rec["capture"].update(sensors=["optical"], frames=1, seed=int(seed))
    rec["view"].update(range_km=1.2, aspect="pursuit")
    cfg = scenegen.build_scene(scenegen.validate_recipe(rec))
    cfg.setdefault("optical", {})["skin"] = bool(skin)
    cfg["optical"]["skin_px_m"] = float(px_m)
    cfg["optical"]["skin_contrast"] = float(skin_contrast)
    if f_number:
        cfg["optical"].setdefault("camera", {}).setdefault("diffraction", {})[
            "f_number"] = float(f_number)
    cfg["optical"]["skin_strength"] = float(strength)
    os.makedirs(out_dir, exist_ok=True)
    # A recipe writes "../assets/models/...", which resolves against the DIRECTORY
    # the scene sits in -- one level below the root. A nested path adds a level
    # and the models are not found.
    scene_path = os.path.join(ROOT, "_tmp", "_score_scene.json")
    os.makedirs(os.path.dirname(scene_path), exist_ok=True)
    json.dump(cfg, open(scene_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    cmd = [sys.executable, "-m", "simkit", "gen", "-c", scene_path,
           "-o", out_dir, "-n", "1", "-s", "optical"]
    try:
        proc = subprocess.run(cmd, cwd=ROOT, env=dict(os.environ, PYTHONPATH=ROOT),
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            tail = (proc.stderr or b"").decode("utf-8", "replace").strip().splitlines()
            raise SystemExit("render failed:\n  " + "\n  ".join(tail[-6:]))
    finally:
        try:
            os.remove(scene_path)
        except OSError:
            pass
    return out_dir


def ours_stats(out_dir):
    """Target-region statistics using the SAME segmentation as the references.

    Cropping our render with the annotation box and comparing that against a
    pixel-masked reference measured apples against oranges: the box is mostly
    sky, so our std came out 0.25x when the same-basis figure is 0.39x. Running
    the reference heuristic on both sides is the only symmetric option, because
    the optical pass does not persist an instance mask.
    """
    from simkit import refcompare as rc
    img = np.asarray(Image.open(os.path.join(out_dir, "images", "optical",
                                             "000000.png")).convert("RGB"), float)
    mask, _info = rc.heuristic_target_mask((img / 255.0).astype(np.float32))
    if mask is None or int(mask.sum()) < 500:
        raise SystemExit("target segmentation found almost nothing in the render")
    sel = img[mask]
    # lap_var is a TARGET statistic.  Computing it on the whole frame measured
    # the sky gradient: the airframe covers 1.7% of the pixels, so a 7% albedo
    # modulation moved the frame-wide number not at all. Crop to the mask's own
    # bounding box first.
    ys, xs = np.where(mask)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    patch = img[y0:y1, x0:x1].mean(axis=2)
    return (dict(mean=[float(sel[:, i].mean()) for i in range(3)],
                 contrast=float(sel.std() / 255.0),
                 lap=_lap(patch)),
            (int(mask.sum()), (x1 - x0, y1 - y0)))


def masked_lap(rgb255, mask):
    """Laplacian energy over the target's own pixels, not its bounding box.

    A box crop mixes the airframe with the background: the reference photos have
    textured clouds inside the box and our sky is nearly flat, so a box statistic
    measures the background as much as the aircraft.
    """
    gray = rgb255.mean(axis=2)
    resp = (sliding_window_view(gray, (3, 3)) * np.array(
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]], float)).sum(axis=(2, 3))
    inner = mask[1:-1, 1:-1]
    if inner.shape != resp.shape:
        return None
    sel = resp[inner]
    return float(np.mean(sel ** 2)) if sel.size >= 100 else None


def normalised_lap(rgb255, mask, target_px=256):
    """Scale-normalise so the target spans target_px, then measure."""
    ys, xs = np.where(mask)
    span = max(int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
    scale = float(target_px) / max(span, 1)
    if abs(scale - 1.0) > 0.02:
        im = Image.fromarray(np.clip(rgb255, 0, 255).astype(np.uint8))
        im = im.resize((max(int(round(im.width * scale)), 8),
                        max(int(round(im.height * scale)), 8)), Image.LANCZOS)
        rgb255 = np.asarray(im, float)
        try:
            mask, _info = rc_module().heuristic_target_mask(
                (rgb255 / 255.0).astype(np.float32))
        except Exception:
            return None
        if mask is None or int(mask.sum()) < 200:
            return None
    return masked_lap(rgb255, mask)


def rc_module():
    from simkit import refcompare
    return refcompare


def reference_stats(class_name, target_px=None):
    from simkit import refcompare as rc
    ref_dir = os.path.join(ROOT, "assets", "refs", class_name)
    rows = []
    for name in sorted(os.listdir(ref_dir)):
        path = os.path.join(ref_dir, name)
        if not os.path.isfile(path):
            continue
        try:
            rgb01 = rc.load_image(path)["rgb"]
        except Exception:
            continue
        try:
            mask, _info = rc.heuristic_target_mask(rgb01)
        except Exception:
            mask = None
        # Scale-normalise before measuring. A reference photo shot at 100 m shows
        # the airframe across ~900 px, so its panel lines are metres wide in
        # pixels; our 1.2 km frame renders the same aircraft across ~380 px, where
        # a real 2 cm seam is sub-pixel and invisible even in the photograph.
        # Comparing the two without normalising the target size measures the
        # shooting distance, not the model.
        if target_px and mask is not None and int(mask.sum()) >= 2000:
            ys, xs = np.where(mask)
            span = max(int(ys.max() - ys.min()), int(xs.max() - xs.min()), 1)
            if span > target_px:
                scale = float(target_px) / float(span)
                h, w = rgb01.shape[:2]
                small = Image.fromarray((rgb01 * 255.0).astype(np.uint8)).resize(
                    (max(int(round(w * scale)), 8), max(int(round(h * scale)), 8)),
                    Image.LANCZOS)
                rgb01 = np.asarray(small, float) / 255.0
                try:
                    mask, _info = rc.heuristic_target_mask(rgb01)
                except Exception:
                    mask = None
        if mask is not None and int(mask.sum()) >= 500:
            sel = rgb01[mask] * 255.0
        else:                       # fall back to the central half
            h, w = rgb01.shape[:2]
            sel = rgb01[h // 4: 3 * h // 4, w // 4: 3 * w // 4].reshape(-1, 3) * 255.0
        rows.append([[float(sel[:, i].mean()) for i in range(3)],
                     float(sel.std() / 255.0)])
    if not rows:
        return None
    return dict(mean=[float(np.median([r[0][i] for r in rows])) for i in range(3)],
                contrast=float(np.median([r[1] for r in rows])),
                n=len(rows))


def _detail_ratio(out_dir):
    """Target detail as a fraction of the reference median, or NaN."""
    try:
        rgb = np.asarray(Image.open(os.path.join(out_dir, "images", "optical",
                                                  "000000.png")).convert("RGB"), float)
        msk, _ = rc_module().heuristic_target_mask(
            (rgb / 255.0).astype(np.float32))
        if msk is None or int(msk.sum()) < 300:
            return float("nan")
        ours = normalised_lap(rgb, msk)
        ref_dir = os.path.join(ROOT, "assets", "refs", "军用飞机")
        refs = []
        for name in sorted(os.listdir(ref_dir))[:10]:
            path = os.path.join(ref_dir, name)
            if not os.path.isfile(path):
                continue
            try:
                r2 = rc_module().load_image(path)["rgb"]
                m2, _ = rc_module().heuristic_target_mask(r2)
            except Exception:
                continue
            if m2 is None or int(m2.sum()) < 300:
                continue
            v = normalised_lap(r2 * 255.0, m2)
            if v:
                refs.append(v)
        if not ours or not refs:
            return float("nan")
        return ours / float(np.median(refs))
    except Exception:
        return float("nan")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="_tmp/score")
    ap.add_argument("--class", dest="cls", default="军用飞机")
    ap.add_argument("--reuse", action="store_true")
    ap.add_argument("--f-number", type=float, default=None,
                    help="aperture for the diffraction term (default 5.6)")
    ap.add_argument("--seeds", type=int, default=1,
                    help="render this many seeds and report the median; the detail "
                         "statistic swings by more than 2x between poses, so a "
                         "single frame cannot decide a change")
    ap.add_argument("--no-skin", action="store_true",
                    help="render without the procedural skin plate (A/B)")
    ap.add_argument("--skin-strength", type=float, default=1.0,
                    help="exaggerate the skin modulation away from its mean")
    ap.add_argument("--skin-contrast", type=float, default=1.0,
                    help="amplitude of the baked skin pattern")
    ap.add_argument("--skin-px", type=float, default=0.04,
                    help="metres per pixel used when baking the skin plate")
    args = ap.parse_args(argv)

    out_dir = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    if not (args.reuse and os.path.isfile(os.path.join(out_dir, "images", "optical",
                                                       "000000.png"))):
        render(out_dir, skin=not args.no_skin, strength=args.skin_strength,
               px_m=args.skin_px, skin_contrast=args.skin_contrast)
    if args.seeds and args.seeds > 1 and not args.reuse:
        samples = []
        for k in range(args.seeds):
            d = os.path.join(out_dir, "_seed%d" % k)
            render(d, skin=not args.no_skin, strength=args.skin_strength,
                   px_m=args.skin_px, skin_contrast=args.skin_contrast,
                   seed=1 + k, f_number=args.f_number)
            samples.append(_detail_ratio(d))
            print("    seed %d: detail ratio %.2fx" % (k, samples[-1]))
        vals = [s for s in samples if s == s]
        if vals:
            print("  detail ratio median over %d seeds: %.2fx"
                  % (len(vals), float(np.median(vals))))
        out_dir = os.path.join(out_dir, "_seed0")
    got, size = ours_stats(out_dir)
    ref = reference_stats(args.cls, target_px=max(size[1]))
    if ref is None:
        raise SystemExit("no usable reference photos for " + args.cls)

    ratio = got["contrast"] / max(ref["contrast"], 1e-9)
    shift = float(np.sqrt(sum((got["mean"][i] - ref["mean"][i]) ** 2 for i in range(3))))
    print("  参照类: %s（%d 张实拍）   机身 mask %d px，包围盒 %dx%d（两边同口径）"
          % (args.cls, ref["n"], size[0], size[1][0], size[1][1]))
    print("  %-16s %14s %14s %10s %s"
          % ("指标", "我们", "实拍", "比值", "判定"))
    ok = True
    row = "contrast"; ok = ok and ratio >= GATES["contrast_ratio"]
    print("  %-16s %14.4f %14.4f %9.2fx  %s"
          % (row, got["contrast"], ref["contrast"], ratio,
             "PASS" if ratio >= GATES["contrast_ratio"] else
             "FAIL (< %.2fx)" % GATES["contrast_ratio"]))
    ok = ok and shift <= GATES["colour_shift"]
    print("  %-16s %14s %14s %9.1f   %s"
          % ("mean_rgb", "%.0f,%.0f,%.0f" % tuple(got["mean"]),
             "%.0f,%.0f,%.0f" % tuple(ref["mean"]), shift,
             "PASS" if shift <= GATES["colour_shift"] else
             "FAIL (> %.0f)" % GATES["colour_shift"]))
    # Rigorous detail ratio: same mask, same target size on both sides.
    try:
        ours_lap = normalised_lap(
            np.asarray(Image.open(os.path.join(out_dir, "images", "optical",
                                               "000000.png")).convert("RGB"), float),
            rc_module().heuristic_target_mask(
                (np.asarray(Image.open(os.path.join(out_dir, "images", "optical",
                                                    "000000.png")).convert("RGB"),
                            float) / 255.0).astype(np.float32))[0])
        ref_laps = []
        ref_dir = os.path.join(ROOT, "assets", "refs", args.cls)
        for name in sorted(os.listdir(ref_dir))[:10]:
            path = os.path.join(ref_dir, name)
            if not os.path.isfile(path):
                continue
            try:
                rgb = rc_module().load_image(path)["rgb"]
                msk, _ = rc_module().heuristic_target_mask(rgb)
            except Exception:
                continue
            if msk is None or int(msk.sum()) < 300:
                continue
            v = normalised_lap(rgb * 255.0, msk)
            if v:
                ref_laps.append(v)
        if ours_lap and ref_laps:
            ratio = ours_lap / float(np.median(ref_laps))
            print("  %-16s %14.1f %14.1f %9.2fx  %s"
                  % ("detail (mask)", ours_lap, float(np.median(ref_laps)), ratio,
                     "PASS" if ratio >= 0.55 else "gap remains"))
    except Exception as exc:
        print("  detail (mask)   skipped:", exc)

    lap_ok = got["lap"] >= GATES["lap_var_min"]
    ok = ok and lap_ok
    print("  %-16s %14.1f %14s %9s   %s"
          % ("lap_var", got["lap"], "%.1f target" % GATES["lap_var_min"], "-",
             "PASS" if lap_ok else "FAIL (< %.0f)" % GATES["lap_var_min"]))
    print()
    print("  结果:", "达标" if ok else "未达标")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
