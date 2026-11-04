"""三模态监督：把可测量量和公开文献量级对比，逐轮修正模型。

判据区间来自物理常识与公开数据，不是从当前输出反推的——否则监督只会
证明自己一直是对的。任何一项失败都会打印实测值与期望区间。
"""

import argparse, json, os, subprocess, sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = []


def _run(scene, out, sensors, frames=1, seed=7, extra=()):
    env = dict(os.environ, PYTHONPATH=ROOT, PYTHONDONTWRITEBYTECODE="1",
               PYTHONIOENCODING="utf-8")
    argv = [sys.executable, "-m", "simkit", "gen", "-c", scene, "-o", out,
            "-n", str(frames), "-s", sensors, "--seed", str(seed)] + list(extra)
    p = subprocess.run(argv, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)
    return p.returncode, (p.stderr or b"").decode("utf-8", "replace")


def _png(p):
    from PIL import Image
    return np.asarray(Image.open(p)).astype(float)


def _lab(p):
    if not os.path.isfile(p):
        return {}
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def _crop(img, b, pad=1):
    h, w = img.shape[:2]
    return img[max(0, int(b["y0"]) - pad):min(h, int(b["y1"]) + 1 + pad),
               max(0, int(b["x0"]) - pad):min(w, int(b["x1"]) + 1 + pad)]


def _sharp(a):
    if a.size < 9:
        return 0.0
    k = (4 * a[1:-1, 1:-1] - a[:-2, 1:-1] - a[2:, 1:-1]
         - a[1:-1, :-2] - a[1:-1, 2:])
    return float(np.abs(k).mean())


def chk(name, val, lo, hi, unit="", why=""):
    ok = (lo is None or val >= lo) and (hi is None or val <= hi)
    RESULTS.append((name, val, lo, hi, unit, ok, why))
    return ok


def report():
    bad = [r for r in RESULTS if not r[5]]
    print()
    print("  %-30s %12s  %-20s %s" % ("指标", "实测", "期望", "判定"))
    print("  " + "-" * 86)
    for n, v, lo, hi, u, ok, why in RESULTS:
        print("  %-30s %12.3f  %-20s %s" % (
            n, v, "%s..%s %s" % (lo, hi, u),
            "OK" if ok else "FAIL  " + why))
    print()
    print("  %d 项，%d 通过，%d 失败" % (len(RESULTS), len(RESULTS) - len(bad), len(bad)))
    return len(bad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="_tmp/sup_air.json")
    ap.add_argument("--out", default="_tmp/sup")
    a = ap.parse_args()
    work = a.out
    os.makedirs(work, exist_ok=True)

    rc, err = _run(a.scene, os.path.join(work, "on"), "optical,ir,radar", 1)
    if rc != 0:
        print("  生成失败:", err.strip().splitlines()[-3:])
        return 2
    base = os.path.join(work, "on")

    # ---- 红外 ----------------------------------------------------------
    ir_p = os.path.join(base, "images", "ir_mwir", "000000.png")
    lb = _lab(os.path.join(base, "labels", "ir_mwir", "000000.json"))
    ir = _png(ir_p)
    boxes = list((lb.get("boxes") or {}).values())
    objs = lb.get("objects") or []
    if objs:
        t = float(objs[0].get("t_mean_c") or 0.0)
        chk("红外 目标平均温度", t, 18.0, 70.0, "C",
            "蒙皮应接近环境+日照，喷口只是局部")
    if boxes:
        sub = _crop(ir, boxes[0])
        bg = float(np.median(ir))
        chk("红外 目标/背景比", float(np.percentile(sub, 95) / max(bg, 1e-9)),
            0.80, 8.0, "", "顶视下车辆与路面可以接近同温") 
        chk("红外 目标框动态范围", float(sub.max() - sub.min()), 800.0, None,
            "count", "等温说明没有温度结构")
        chk("红外 目标框起伏比", float(sub.std() / max(sub.mean(), 1e-9)),
            0.008, 0.85, "", "过低说明目标是一块平斑") 
    chk("红外 全帧饱和", float((ir >= 65535).mean() * 100), 0.0, 2.0, "%")

    # 喷流：关掉发动机热再渲一帧，差异必须存在
    cfg = _lab(a.scene)
    cfg.setdefault("ir", {})["engine_heat"] = False
    cfg.pop("_base_dir", None)
    off_scene = os.path.join(work, "_no_engine.json")
    with open(off_scene, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, ensure_ascii=False)
    rc, _e = _run(off_scene, os.path.join(work, "off"), "ir", 1)
    if rc == 0:
        off = _png(os.path.join(work, "off", "images", "ir_mwir", "000000.png"))
        frac = float((np.abs(ir - off) > 1).mean() * 100)
        chk("红外 喷流像素占比", frac, 0.15, 6.0, "%",
            "关掉发动机热后画面应有可见差异")

    # ---- 光学 ----------------------------------------------------------
    op = os.path.join(base, "images", "optical", "000000.png")
    if os.path.isfile(op):
        oi = _png(op)
        chk("光学 饱和", float((oi >= 255).mean() * 100), 0.0, 3.0, "%")
        chk("光学 死黑", float((oi <= 2).mean() * 100), 0.0, 55.0, "%")
        obl = _lab(os.path.join(base, "labels", "optical", "000000.json"))
        ob = list((obl.get("boxes") or {}).values())
        if ob:
            osub = _crop(oi, ob[0])
            chk("光学 目标锐度", _sharp(osub), 1.0, None, "",
                "低于 1 说明糊了")
            # Percentiles, not the box mean: a side-on airframe fills
            # about a fifth of its own box, so a mean reports the sky
            # and read 0.016 on a target swinging 95..162 against 120.
            m = float(np.median(oi))
            hi_c = abs(float(np.percentile(osub, 90)) - m) / max(m, 1e-9)
            lo_c = abs(float(np.percentile(osub, 10)) - m) / max(m, 1e-9)
            chk("光学 目标对比度", max(hi_c, lo_c), 0.05, 0.90, "",
                "太低看不见，太高像贴图")

    # ---- 雷达 ----------------------------------------------------------
    rl = _lab(os.path.join(base, "labels", "radar", "000000.json"))
    # RCS comes from the scene, not the label: it is a scene input,
    # and comparing it with published cross sections is the only way
    # to catch a target that is orders of magnitude off.
    try:
        rcs = float((_lab(a.scene).get("objects") or [{}])[0].get("rcs_db"))
    except (TypeError, ValueError):
        rcs = None
    if rl:
        rd = rl.get("boxes_rd") or {}
        tgt = [k for k in rd if not k.startswith("ecm#")]
        chk("雷达 目标框数", float(len(tgt)), 1.0, None, "个")
        b3 = rl.get("boxes_3d") or []
        chk("雷达 真值3D框", float(len(b3)), 1.0, None, "个")
        pk = (rl.get("ra_peak") or {}).get("db")
        if pk is not None:
            chk("雷达 峰值电平", float(pk), -40.0, 20.0, "dB",
                "过低头部方向图或功率有问题")
        if rcs is not None:
            chk("雷达 RCS 量级", float(rcs), -5.0, 32.0, "dBsm",
                "应与公开量级同阶")
        # range_window_m is a scalar in some writers and a list in others
        rw = rl.get("range_window_m")
        if isinstance(rw, (list, tuple)):
            rw = rw[-1] if rw else None
        if rw:
            chk("雷达 量程窗口", float(rw), 500.0, None, "m",
                "距离窗必须覆盖目标斜距")
        npts = rl.get("num_points")
        if npts is not None:
            chk("雷达 点云点数", float(npts), 1.0, None, "个",
                "目标应产生至少一个回波点")

    # ---- 一致性 --------------------------------------------------------
    if boxes and ob:
        W = float(lb.get("width") or 960)
        H = float(lb.get("height") or 720)
        dx = abs((ob[0]["x0"] + ob[0]["x1"]) / 2 - (boxes[0]["x0"] + boxes[0]["x1"]) / 2) / W
        dy = abs((ob[0]["y0"] + ob[0]["y1"]) / 2 - (boxes[0]["y0"] + boxes[0]["y1"]) / 2) / H
        chk("跨模态 目标中心偏移", max(dx, dy) * 100, 0.0, 2.0, "%",
            "光学与红外必须看到同一个目标")

    return report()


if __name__ == "__main__":
    sys.exit(main())