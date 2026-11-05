"""重建在线演示用的场景批：真实背景照片 + 当前引擎。

每个场景都是一次完整的 simkit gen 运行，写进 out/<name>/，随后由
tools/export_web.py 汇总成 data/boot.js。

五条覆盖这次上线的能力：空域迎头、空域追击、陆域俯视板、海域俯视板、
双目标各自航线。背景一律取 assets/backdrops 里的实拍照片。
"""

import json, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from simkit import scenegen

DEMO_FRAMES = int(os.environ.get("DEMO_FRAMES", "8"))
DEMO_W, DEMO_H = 1280, 960

SPECS = {
    "demo_air_headon": dict(
        title="空中目标 · 迎头", kind="sky", cat="sky",
        asset="军用飞机/双发战斗机.obj", cls="fighter",
        aspect="head_on", range_km=1.2, fov_deg=2.2),
    "demo_air_pursuit": dict(
        title="空中目标 · 追击", kind="sky", cat="sky",
        asset="军用飞机/双发战斗机.obj", cls="fighter",
        aspect="pursuit", range_km=1.2, fov_deg=2.2),
    "demo_ground_plate": dict(
        title="地面目标 · 俯视板", kind="ground", cat="city",
        asset="军用地面车辆/主战坦克.obj", cls="vehicle",
        aspect="top_down", fov_deg=8.0, span=220.0),
    "demo_sea_plate": dict(
        title="海面目标 · 俯视板", kind="sea", cat="port",
        asset="军舰/导弹艇.obj", cls="ship",
        aspect="top_down", fov_deg=8.0, span=260.0),
    "demo_multi": dict(
        title="双目标 · 各自航线", kind="ground", cat="rural",
        asset="军用地面车辆/主战坦克.obj", cls="vehicle",
        aspect="top_down", fov_deg=8.0, span=260.0, multi=True),
    # 天气对照：故意不用实地背景板。板视图里照片会替换掉约 95% 的像素，
    # 天气只剩目标上那一点变化——三张图看着一模一样（实测差 2.3/255）。
    # 换成程序化地形后，太阳高度角、云量、雨强才作用到整幅画面。
    # 三档亮度/天气。实测：多云与雾只调天空和云量，俯视地面几乎不变
    # （差 1.4/255），只有太阳高度角和降雨真正改变画面。所以用
    # 正午 / 低太阳 / 降雨，三者分别是光照角度、光照角度、天气。
    "demo_wx_clear": dict(
        title="正午 · 晴", kind="ground", cat="city", wx=True, weather="clear", sun_elev=58.0,
        asset="军用地面车辆/主战坦克.obj", cls="vehicle",
        aspect="look_down", range_km=0.35, fov_deg=9.0),
    "demo_wx_cloudy": dict(
        title="低太阳 · 晴", kind="ground", cat="city", wx=True, weather="clear", sun_elev=7.0,
        asset="军用地面车辆/主战坦克.obj", cls="vehicle",
        aspect="look_down", range_km=0.35, fov_deg=9.0),
    "demo_wx_rain": dict(
        title="降雨", kind="ground", cat="city", wx=True, weather="rain", sun_elev=35.0,
        asset="军用地面车辆/主战坦克.obj", cls="vehicle",
        aspect="look_down", range_km=0.35, fov_deg=9.0),
}


def build(name, spec):
    r = scenegen.default_recipe()
    r["name"] = name
    _cat = spec.get("cat")
    r["environment"].update(
        kind=spec["kind"], weather=spec.get("weather", "clear"),
        background=({"mode": "library", "category": _cat} if _cat else {"mode": "none"}))
    if spec.get("span"):
        r["view"].update(aspect=spec["aspect"], ground_span_m=spec["span"],
                         fov_deg=spec["fov_deg"])
    else:
        r["view"].update(aspect=spec["aspect"], range_km=spec["range_km"],
                         fov_deg=spec["fov_deg"])
    base = dict(asset=spec["asset"], cls=spec["cls"], scale=1.0, count=1)
    if spec.get("span"):
        base["uv"] = [0.45, 0.52]
    if spec.get("multi"):
        r["objects"] = [
            dict(base, uv=[0.34, 0.58],
                 trajectory=dict(mode="waypoints", speed_mps=18,
                                 points=[[-70, 0, 0], [0, 26, 0], [70, 0, 0]])),
            dict(base, asset="军用地面车辆/高机动越野车.obj", uv=[0.66, 0.44],
                 trajectory=dict(mode="waypoints", speed_mps=18,
                                 points=[[0, -34, 0], [46, 0, 0], [0, 34, 0]])),
        ]
    else:
        r["objects"] = [base]
    r["capture"].update(sensors=["optical", "ir", "radar"],
                        frames=DEMO_FRAMES, fps=10.0, seed=7)
    cfg = scenegen.build_scene(scenegen.validate_recipe(r))
    if spec.get("wx"):
        # 天气对照关自动曝光：开着它会把太阳/云量/雨强带来的整体亮度差
        # 归一化掉，三张图只剩 1.8% 的差别，看不出「换了天气」。
        cfg.setdefault("optical", {})["auto_exposure"] = False
        if spec.get("sun_elev") is not None:
            cfg.setdefault("sun", {})["elev_deg"] = float(spec["sun_elev"])
        # 非俯视时地面照片默认带随机旋转，斜着铺上去很扎眼；天气对照要的是
        # 同一片地形在不同光照下，所以固定成无旋转、居中对齐。
        _g = cfg.setdefault("ground", {})
        _g["backdrop_aligned"] = True
        _g["backdrop_offset"] = [0.5, 0.5]
    for v in ((cfg.get("clip") or {}).get("views") or []):
        v["width"], v["height"] = DEMO_W, DEMO_H
    for key in ("optical", "ir_camera"):
        blk = (cfg.get("sensors") or {}).get(key)
        if isinstance(blk, dict):
            blk["width"], blk["height"] = DEMO_W, DEMO_H
    return cfg


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    os.makedirs(os.path.join(ROOT, "_tmp"), exist_ok=True)
    for name, spec in SPECS.items():
        if only and name != only:
            continue
        cfg = build(name, spec)
        scene_path = os.path.join(ROOT, "_tmp", "_demo_%s.json" % name)
        with open(scene_path, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, ensure_ascii=False, indent=1)
        out = os.path.join(ROOT, "out", name)
        env = dict(os.environ, PYTHONPATH=ROOT, PYTHONDONTWRITEBYTECODE="1",
                   PYTHONIOENCODING="utf-8")
        argv = [sys.executable, "-m", "simkit", "gen", "-c", scene_path,
                "-o", out, "-n", str(DEMO_FRAMES),
                "-s", "optical,ir,radar,lidar", "--seed", "7"]
        sys.stdout.write("  生成 %s ...\n" % name)
        sys.stdout.flush()
        p = subprocess.run(argv, cwd=ROOT, env=env)
        if p.returncode != 0:
            sys.stdout.write("  失败: %s\n" % name)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())