"""Local dataset studio: click parameters, get a dataset.

A single-process, dependency-free HTTP server that wraps the generator. It
exists because the command line needs seven flags and a JSON edit to answer a
question the user asks in one sentence ("give me rain, small targets, Ku band,
500 frames"), and because a dataset you cannot look at before committing an
hour of compute is a dataset you generate three times.

The UI is a form; this module turns the form into a *scene config* by overlaying
the choices on a preset file from ``scenes/``, then runs that config either
through ``simkit.plan.summary`` (instant, no rendering: what trajectory will I
get?) or through ``simkit gen`` in a subprocess (the real thing).

Endpoints
    GET  /                      the UI
    GET  /api/scenes            every scenes/*.json, with tags and defaults
    GET  /api/presets           option tables (weather, sensors, bands, limits)
    POST /api/plan              per-clip trajectory facts; no rendering (~0.1 s)
    GET  /api/plan?preset=..&frames=..   same, legacy query form
    POST /api/generate          start a full run  (JSON body)
    POST /api/preview           start an 8-frame run
    GET  /api/jobs              every run this server has started
    GET  /api/job/<id>          one run: state, progress, log tail, error tail
    POST /api/job/<id>/cancel   stop a run
    GET  /api/dataset/<id>      stats + annotated sample cards
    GET  /api/datasets          runs already on disk
    GET  /api/img/<id>/<path>   serve a generated image

Every user-supplied number is range-checked; a bad value answers 400 with a
Chinese message instead of raising inside a thread and killing the run. The
output directory is always a direct child of ``out/studio_runs``.

Nothing here touches the network, and the UI is plain ES5 so it works from a
USB stick on an air-gapped machine.
"""
import json
import os
import re
import subprocess
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENES = os.path.join(ROOT, "scenes")
RUNS = os.path.join(ROOT, "out", "studio_runs")
UI = os.path.join(ROOT, "studio")
TMP = os.path.join(ROOT, "_tmp")

# ---------------------------------------------------------------- option tables

WEATHER_MODES = {
    "clear": dict(mode="fixed", preset="clear"),
    "cloudy": dict(mode="fixed", preset="cloudy"),
    "rain": dict(mode="fixed", preset="rain"),
    "fog": dict(mode="fixed", preset="fog"),
    "mixed": dict(mode="per_clip", pool=["clear", "cloudy", "rain", "fog"]),
    "evolve": dict(mode="evolve", pool=["clear", "cloudy", "rain", "fog"]),
}
WEATHER_LABELS = {"clear": "晴", "cloudy": "多云", "rain": "雨", "fog": "雾",
                  "mixed": "混合", "evolve": "演变"}

SENSORS = [
    dict(key="optical", label="可见光", note="Blender 渲染，最慢"),
    dict(key="ir", label="红外", note="中波 + 长波，热辐射"),
    dict(key="radar", label="雷达", note="距离-角度 / 距离-多普勒"),
    dict(key="lidar", label="激光雷达", note="距离图 + 强度 + 点云"),
    dict(key="sar", label="SAR", note="合成孔径，需后端集成"),
]
SENSOR_KEYS = [s["key"] for s in SENSORS]

# fc is only a hint for the presets; the SAR model derives everything else
# from bandwidth and geometry.
SAR_BANDS = {
    "X": dict(fc_ghz=9.6, note="9.6 GHz，仰角窗口宽，工程上最常用"),
    "C": dict(fc_ghz=5.4, note="5.4 GHz，穿透好，分辨率换带宽"),
    "Ku": dict(fc_ghz=15.5, note="15.5 GHz，小天线也能窄波束，衰减偏大"),
}
SAR_DEFAULTS = dict(
    prf_hz=2000.0, n_pulses=256, n_samples=512, altitude_m=3000.0, speed_mps=150.0,
    heading_deg=90.0, squint_deg=0.0, swath_center_m=4000.0, swath_width_m=1200.0,
    polarization="VV", tx_power_w=100.0, noise_figure_db=4.0, loss_db=3.0,
    antenna_len_m=1.5, antenna_h_m=0.3, resolution_scale=1.0,
)

# Narrow = seeker (tracking) view, wide = search view. Tags match the contract:
# wide views carry "_wide" so the file names never collide.
VIEW_NARROW = dict(name="seeker", tag="", fov_deg=2.2, width=960, height=720)
VIEW_WIDE = dict(name="wide", tag="_wide", fov_deg=18.0, width=1280, height=960)

SIZE_PX = dict(small=[8, 20], medium=[18, 42], large=[35, 80])
CLOSING = {"in": "接近", "out": "远离", "hold": "保持"}
TRAJ_MODES = {
    "platform": "平台俯冲 + 蛇形（导引头）",
    "seeker": "相机静立 · 目标飞行（旧导引头）",
    "world": "世界轨迹（对象自带速度）",
}

# hard limits: anything outside answers 400, never a half-written run
LIM = dict(
    frames=(1, 20000), fps=(0.5, 120), fpc=(1, 5000), fov=(0.1, 120),
    dim=(64, 4096), px=(1, 4000), range_m=(1, 400000), seed=(0, 2 ** 32 - 1),
    closing_mps=(0.0, 3000.0), bw=(1.0, 2000.0), alt=(10.0, 30000.0),
    speed=(0.0, 3000.0), swath=(10.0, 200000.0), squint=(-75.0, 75.0),
    heading=(0.0, 360.0), dive=(-89.0, 89.0), port=(1024, 65535),
    weave=(0.0, 500.0), weave_hz=(0.0, 5.0), standoff=(10.0, 50000.0),
    back=(0.0, 200000.0),
)

_JOBS = {}
_LOCK = threading.Lock()
_RE_PROG = re.compile(r"frame\s+(\d+)/(\d+)")
_RE_COUNTER = re.compile(r"(\d{3,6})")
_CACHE = {}


class BadRequest(ValueError):
    """A user input the server refuses to guess at; answered with HTTP 400."""


def _now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


# ------------------------------------------------------------------- validation

def _fnum(body, key, label, lim, default=None):
    v = body.get(key)
    if v is None or v == "":
        if default is None:
            raise BadRequest("缺少参数：%s" % label)
        return float(default)
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise BadRequest("%s 不是数字：%r" % (label, v))
    if x != x or x in (float("inf"), float("-inf")):
        raise BadRequest("%s 不是有限数" % label)
    lo, hi = lim
    if x < lo or x > hi:
        raise BadRequest("%s 应在 %g ~ %g 之间，收到 %g" % (label, lo, hi, x))
    return x


def _inum(body, key, label, lim, default=None):
    x = _fnum(body, key, label, lim, default)
    if abs(x - round(x)) > 1e-9:
        raise BadRequest("%s 必须是整数，收到 %g" % (label, x))
    return int(round(x))


def _pair(body, key, label, lim, default=None):
    v = body.get(key)
    if v is None or v == "" or (isinstance(v, (list, tuple)) and not len(v)):
        if default is None:
            raise BadRequest("缺少参数：%s" % label)
        v = default
    if not isinstance(v, (list, tuple)) or len(v) != 2:
        raise BadRequest("%s 需要两个数，例如 [10, 45]" % label)
    a = _fnum(dict(v=v[0]), "v", label + " 下限", lim)
    b = _fnum(dict(v=v[1]), "v", label + " 上限", lim)
    if b < a:
        a, b = b, a
    if abs(b - a) < 1e-9:
        raise BadRequest("%s 上下限不能相同（%g）" % (label, a))
    return [a, b]


def _choice(body, key, label, table, default):
    v = body.get(key)
    if v is None or v == "":
        return default
    v = str(v)
    if v not in table:
        raise BadRequest("%s 只能取 %s，收到 %r" % (label, "/".join(sorted(table)), v))
    return v


def _safe_name(name, fallback=""):
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name or "")).strip("._-")
    return s[:40] or fallback


# ------------------------------------------------------------------- scenes

def _scene_paths():
    """stem -> absolute path for every scenes/*.json (never hard-coded)."""
    out = {}
    if not os.path.isdir(SCENES):
        return out
    for fn in sorted(os.listdir(SCENES)):
        if fn.lower().endswith(".json"):
            out[fn[:-5]] = os.path.join(SCENES, fn)
    return out


def _scene_file(name):
    """Accept 'sk_air_uav', 'sk_air_uav.json', the legacy 'air_uav', or ''."""
    paths = _scene_paths()
    if not paths:
        raise BadRequest("scenes/ 目录里没有场景文件")
    raw = str(name or "").strip()
    if not raw:
        return sorted(paths)[0]
    stem = raw[:-5] if raw.lower().endswith(".json") else raw
    if stem in paths:
        return paths[stem]
    for cand in (raw, "sk_" + stem, "sk2_" + stem, "ds_" + stem):
        if cand in paths:
            return paths[cand]
    raise BadRequest("未知场景：%r（可用：%s）" % (raw, ", ".join(sorted(paths))))


_CLS_CN = {"uav": "空中 · 无人机", "vehicle": "地面 · 车辆", "boat": "海面 · 船只",
           "ship": "海面 · 船只", "car": "地面 · 车辆", "aircraft": "空中 · 目标"}


def _scene_info(stem, path):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
    except (OSError, ValueError) as exc:
        return dict(key=stem, file=stem + ".json", title=stem, error=str(exc),
                    sensors=[], has_seeker=False, has_sar=False, mode=None)
    clip = cfg.get("clip") or {}
    sk = clip.get("seeker") or cfg.get("seeker") or {}
    sensors = cfg.get("sensors") or {}
    obj = next((o for o in cfg.get("objects") or [] if o.get("label", True)), {})
    cls = obj.get("cls") or "target"
    has_sar = "sar" in sensors
    views = [dict(name=v.get("name"), fov_deg=v.get("fov_deg"), width=v.get("width"))
             for v in (clip.get("views") or []) if isinstance(v, dict)]
    group = "导引头" if (sk.get("mode") == "platform" or stem.lower().startswith("sk2")) else \
            ("SAR" if has_sar else "基础")
    fov = None
    for nm in ("optical", "ir_camera", "eo_camera", "camera"):
        s = sensors.get(nm)
        if isinstance(s, dict) and s.get("fov_deg") is not None:
            fov = float(s["fov_deg"])
            break
    fr = cfg.get("frames") or {}
    return dict(
        key=stem, file=stem + ".json",
        title="%s · %s" % (group, _CLS_CN.get(cls, cls)),
        cls=cls, group=group, fov=fov,
        frames=fr.get("n"), dt=fr.get("dt"),
        # the form has a frame-rate control; a scene authored in dt must state
        # its own fps or the widget's 10 fps default contradicts the clip
        fps=(round(1.0 / float(fr["dt"]), 4) if fr.get("dt") else None),
        sensors=[k for k in ("optical", "ir_camera", "radar", "lidar", "sar") if k in sensors],
        has_seeker=bool(sk), has_sar=has_sar,
        mode=sk.get("mode") or ("seeker" if sk else "world"),
        views=views,
        target_px=sk.get("target_px"), range_m=sk.get("range_m"),
        closing=sk.get("closing"),
        note="%d 帧 · dt=%s · %s" % (fr.get("n") or 0, fr.get("dt"),
                                     "双视场" if len(views) > 1 else "单视场"),
    )


_PREFIX_ORDER = {"sk3": 0, "sk2": 0, "sk": 1, "sar": 1, "ds": 2, "final": 3,
                 "demo": 4, "template": 5}


def _scenes():
    rows = [(stem, _scene_info(stem, path)) for stem, path in _scene_paths().items()]
    order = {"导引头": 0, "基础": 1, "SAR": 2}
    rows.sort(key=lambda r: (order.get(r[1].get("group"), 3),
                             _PREFIX_ORDER.get(r[0].split("_")[0], 9), r[0]))
    return [r[1] for r in rows]


# --------------------------------------------------------------- body -> config

def _parse(body):
    """Validate the whole form and return a normalised dict. Raises BadRequest."""
    if not isinstance(body, dict):
        raise BadRequest("请求体必须是 JSON 对象")

    q = {}
    q["_present"] = set(body.keys())
    q["scene_path"] = _scene_file(body.get("scene") or body.get("preset"))
    q["scene"] = os.path.basename(q["scene_path"])[:-5]
    q["name"] = _safe_name(body.get("name"))
    q["frames"] = _inum(body, "frames", "帧数", LIM["frames"], 240)
    q["fps"] = _fnum(body, "fps", "帧率", LIM["fps"], 10.0)
    q["fpc"] = _inum(body, "frames_per_clip", "每段帧数", LIM["fpc"], 120)
    q["fpc"] = max(1, min(q["fpc"], q["frames"]))
    q["seed"] = _inum(body, "seed", "随机种子", LIM["seed"], 7)
    q["weather"] = _choice(body, "weather", "天气", WEATHER_MODES, "mixed")

    want = body.get("sensors")
    if want in (None, "", []):
        want = ["ir"]
    if not isinstance(want, (list, tuple)):
        raise BadRequest("传感器必须是一个数组，例如 [\"ir\",\"radar\"]")
    bad = [w for w in want if str(w) not in SENSOR_KEYS]
    if bad:
        raise BadRequest("未知传感器：%s（可用：%s）" % (", ".join(map(str, bad)),
                                                     ", ".join(SENSOR_KEYS)))
    q["sensors"] = []
    for w in want:                                   # dedupe, keep order
        w = str(w)
        if w not in q["sensors"]:
            q["sensors"].append(w)

    q["views_mode"] = _choice(body, "views_mode", "视场",
                              {"single": 1, "dual": 1}, "single")
    q["narrow"] = dict(
        fov_deg=_fnum(body, "narrow_fov", "窄视场角", LIM["fov"], VIEW_NARROW["fov_deg"]),
        width=_inum(body, "narrow_width", "窄视场宽", LIM["dim"], VIEW_NARROW["width"]),
        height=_inum(body, "narrow_height", "窄视场高", LIM["dim"], VIEW_NARROW["height"]))
    q["wide"] = dict(
        fov_deg=_fnum(body, "wide_fov", "宽视场角", LIM["fov"], VIEW_WIDE["fov_deg"]),
        width=_inum(body, "wide_width", "宽视场宽", LIM["dim"], VIEW_WIDE["width"]),
        height=_inum(body, "wide_height", "宽视场高", LIM["dim"], VIEW_WIDE["height"]))
    if q["views_mode"] == "dual" and q["wide"]["fov_deg"] <= q["narrow"]["fov_deg"]:
        raise BadRequest("双视场要求宽视场角 > 窄视场角（%g 对 %g）"
                         % (q["wide"]["fov_deg"], q["narrow"]["fov_deg"]))

    q["size"] = _choice(body, "size", "目标大小档", SIZE_PX, "medium")
    q["target_px"] = _pair(body, "target_px", "目标像素区间", LIM["px"], SIZE_PX[q["size"]])
    q["closing"] = _choice(body, "closing", "接近方式", CLOSING, "in")
    q["closing_mps"] = _fnum(body, "closing_mps", "接近速率", LIM["closing_mps"], 0.0) \
        if body.get("closing_mps") not in (None, "") else None
    q["range_m"] = _pair(body, "range_m", "视线距离区间", LIM["range_m"]) \
        if body.get("range_m") not in (None, "", []) else None

    q["traj_mode"] = _choice(body, "traj_mode", "轨迹模式", TRAJ_MODES, "platform")
    q["platform"] = dict(
        alt_m=_fnum(body, "platform_alt", "平台高度", LIM["alt"], 1200.0),
        speed_mps=_fnum(body, "platform_speed", "平台速度", LIM["speed"], 180.0),
        heading_deg=_fnum(body, "platform_heading", "平台航向", LIM["heading"], 90.0),
        dive_deg=_fnum(body, "platform_dive", "俯冲角", LIM["dive"], -14.0),
        weave_amp_m=_fnum(body, "platform_weave", "蛇形振幅", LIM["weave"], 18.0),
        weave_hz=_fnum(body, "platform_weave_hz", "蛇形频率", LIM["weave_hz"], 0.22),
        standoff_end_m=_fnum(body, "platform_standoff", "结束距离", LIM["standoff"], 260.0),
        back_m=_fnum(body, "platform_back", "起点后撤", LIM["back"], 3000.0))
    # Widget factory defaults, so _compose can tell "untouched" from "chosen".
    q["_defaults"] = dict(
        target_px=float(SIZE_PX["medium"][0]), closing="in", fps=10.0, weather="mixed",
        views_mode="single",
        narrow_fov=float(VIEW_NARROW["fov_deg"]), narrow_width=int(VIEW_NARROW["width"]),
        wide_fov=float(VIEW_WIDE["fov_deg"]), wide_width=int(VIEW_WIDE["width"]),
        alt_m=1200.0, speed_mps=180.0, heading_deg=90.0, dive_deg=-14.0,
        weave_amp_m=18.0, weave_hz=0.22, standoff_end_m=260.0, back_m=3000.0)
    q["platform_dirty"] = any(
        abs(float(q["platform"][k]) - float(q["_defaults"][k])) > 1e-9
        for k in ("alt_m", "speed_mps", "heading_deg", "dive_deg",
                  "weave_amp_m", "weave_hz", "standoff_end_m", "back_m"))

    q["sar"] = dict(
        enabled=bool(body.get("sar_enabled")),
        band=_choice(body, "sar_band", "SAR 波段", SAR_BANDS, "X"),
        bandwidth_mhz=_fnum(body, "sar_bandwidth_mhz", "SAR 带宽", LIM["bw"], 100.0),
        swath_center_m=_fnum(body, "sar_swath_center_m", "测绘带中心", LIM["swath"], 4000.0),
        swath_width_m=_fnum(body, "sar_swath_width_m", "测绘带宽度", LIM["swath"], 1200.0),
        altitude_m=_fnum(body, "sar_altitude_m", "SAR 平台高度", LIM["alt"], 3000.0),
        speed_mps=_fnum(body, "sar_speed_mps", "SAR 平台速度", LIM["speed"], 150.0),
        squint_deg=_fnum(body, "sar_squint_deg", "SAR 斜视", LIM["squint"], 0.0),
        polarization=_choice(body, "sar_polarization", "极化",
                             {"VV": 1, "HH": 1, "VH": 1, "HV": 1}, "VV"))
    if q["sar"]["enabled"] and "sar" not in q["sensors"]:
        q["sensors"].append("sar")
    if not q["sar"]["enabled"] and "sar" in q["sensors"]:
        q["sensors"].remove("sar")

    # A run that renders nothing is a bug report, not a dataset: keep one
    # image-producing modality alive and say so in the log.
    q["notes"] = []
    if q["sensors"] == ["sar"]:
        q["sensors"].insert(0, "ir")
        q["notes"].append("只开了 SAR：附加 ir 以生成可见样张（SAR 集成为主控任务，未集成时该传感器被忽略）")
    return q


def _abs_models(cfg):
    for o in cfg.get("objects", []) or []:
        m = o.get("model")
        if m and not os.path.isabs(m):
            o["model"] = os.path.normpath(os.path.join(SCENES, m))
    return cfg


def _load_scene(path):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
    except (OSError, ValueError) as exc:
        raise BadRequest("场景读取失败：%s" % exc)
    return _abs_models(cfg)


def _compose(q, cfg):
    """Overlay the validated form onto the scene config.

    Only fields the operator actually changed are applied. The form always sends
    every control (that is how HTML forms work), so a straight overlay means the
    form's factory defaults silently replace the curated engagement geometry --
    measured: picking `sk3_low_air2ground_clutter` and pressing preview produced
    a 313 m end range and a 247 px target instead of the scene's 2.1-3.6 km and
    17-30 px. Comparing against the widget defaults keeps "untouched" honest
    without needing the page to track dirtiness.
    """
    clip = cfg.setdefault("clip", {})
    dflt = q.get("_defaults") or {}

    def changed(key, value, tol=1e-9):
        """True when the operator moved this control away from its default."""
        ref = dflt.get(key)
        if ref is None:
            return True
        try:
            return abs(float(value) - float(ref)) > tol
        except (TypeError, ValueError):
            return value != ref

    # ---- trajectory
    if q["traj_mode"] == "world":
        clip.pop("seeker", None)
        cfg.pop("seeker", None)
    else:
        sk = clip.get("seeker") or cfg.get("seeker") or {}
        sk = json.loads(json.dumps(sk))
        if changed("target_px", q["target_px"][0]) or changed("target_px", q["target_px"][1]):
            sk["target_px"] = [float(q["target_px"][0]), float(q["target_px"][1])]
        if q["closing"] != dflt.get("closing"):
            sk["closing"] = q["closing"]
        if q["closing_mps"] is not None:
            sk["closing_mps"] = float(q["closing_mps"])
        if q["range_m"]:
            sk["range_m"] = [float(q["range_m"][0]), float(q["range_m"][1])]
        if q["traj_mode"] == "platform":
            sk["mode"] = "platform"
            p = q["platform"]
            base = (sk.get("platform") or {}) if not q["platform_dirty"] else {}
            keep = dict(base)

            def pick(field, incoming, scene_key=None):
                """Keep the scene's number unless the control was moved."""
                if q["platform_dirty"] or changed(field, incoming):
                    return float(incoming)
                v = keep.get(scene_key or field)
                return float(v) if v is not None else float(incoming)

            plat = dict(keep)
            plat.update(
                origin=[0.0, -pick("back_m", p["back_m"], "origin_y_back"), float(p["alt_m"])],
                speed_mps=pick("speed_mps", p["speed_mps"]),
                heading_deg=pick("heading_deg", p["heading_deg"]),
                dive_deg=pick("dive_deg", p["dive_deg"]),
                weave_amp_m=pick("weave_amp_m", p["weave_amp_m"]),
                weave_hz=pick("weave_hz", p["weave_hz"]),
                standoff_end_m=pick("standoff_end_m", p["standoff_end_m"]))
            # the scene's own origin is the truth when nothing was touched
            if not q["platform_dirty"] and not any(
                    changed(k, p[k]) for k in ("back_m", "alt_m", "heading_deg")):
                orig = (sk.get("platform") or {}).get("origin")
                if orig:
                    plat["origin"] = [float(v) for v in orig]
            if plat["origin"][2] != float(p["alt_m"]) and q["platform_dirty"]:
                plat["origin"][2] = float(p["alt_m"])
            sk["platform"] = plat
        else:
            sk.pop("mode", None)
        clip["seeker"] = sk
        cfg.pop("seeker", None)

    # ---- views. Single view keeps the old shape on purpose: clip.views absent
    # means dataset.py takes the legacy sensors[cam] path and nothing changes.
    # The reference is the SCENE's own lens set, not the widget default: a
    # curated dual-view scene must survive an untouched form even though the
    # form's own factory default is "single".
    scene_views = clip.get("views") or []
    scene_mode = "dual" if len(scene_views) > 1 else "single"
    ref_n = scene_views[0] if scene_views else VIEW_NARROW
    ref_w = scene_views[1] if len(scene_views) > 1 else VIEW_WIDE

    def _near(a, b):
        try:
            return abs(float(a) - float(b)) <= 1e-9
        except (TypeError, ValueError):
            return a == b

    # A lens control that never appeared in the request was never touched, so
    # the scene's own value stands even though the widget would default to a
    # different number (960 vs the seeker scenes' 768).
    lens_given = bool(q["_present"] & {"narrow_fov", "narrow_width", "narrow_height",
                                       "wide_fov", "wide_width", "wide_height"})
    same_narrow = (not lens_given) or (
        _near(q["narrow"]["fov_deg"], ref_n.get("fov_deg", q["narrow"]["fov_deg"]))
        and _near(q["narrow"]["width"], ref_n.get("width", q["narrow"]["width"])))
    same_wide = (not lens_given) or (
        _near(q["wide"]["fov_deg"], ref_w.get("fov_deg", q["wide"]["fov_deg"]))
        and _near(q["wide"]["width"], ref_w.get("width", q["wide"]["width"])))
    views_touched = (q["views_mode"] != scene_mode) or not same_narrow \
        or (q["views_mode"] == "dual" and not same_wide)
    if not views_touched and scene_views:
        pass                      # the scene's own lenses win
    else:
        views = [dict(VIEW_NARROW, fov_deg=q["narrow"]["fov_deg"],
                      width=q["narrow"]["width"], height=q["narrow"]["height"])]
        if q["views_mode"] == "dual":
            views.append(dict(VIEW_WIDE, fov_deg=q["wide"]["fov_deg"],
                              width=q["wide"]["width"], height=q["wide"]["height"]))
            clip["views"] = views
        else:
            clip.pop("views", None)

    # The camera dicts still drive the renderers that predate clip.views, so the
    # narrow view is written into every camera the scene owns -- but only when
    # the operator actually changed the lens.
    sensors = cfg.setdefault("sensors", {})
    if views_touched and not same_narrow:
        for name in ("ir_camera", "optical", "eo_camera", "camera"):
            cam = sensors.get(name)
            if isinstance(cam, dict):
                cam["fov_deg"] = float(q["narrow"]["fov_deg"])
                cam["width"] = int(q["narrow"]["width"])
                cam["height"] = int(q["narrow"]["height"])

    # ---- SAR
    if q["sar"]["enabled"]:
        s = q["sar"]
        block = dict(SAR_DEFAULTS)
        block.update(
            band=s["band"], fc_ghz=float(SAR_BANDS[s["band"]]["fc_ghz"]),
            bandwidth_mhz=float(s["bandwidth_mhz"]),
            swath_center_m=float(s["swath_center_m"]),
            swath_width_m=float(s["swath_width_m"]),
            altitude_m=float(s["altitude_m"]), speed_mps=float(s["speed_mps"]),
            squint_deg=float(s["squint_deg"]), polarization=s["polarization"])
        block["range_resolution_m"] = round(150.0 / max(block["bandwidth_mhz"], 1e-6), 3)
        sensors["sar"] = block
    else:
        sensors.pop("sar", None)

    # ---- frames / weather / seed
    # Frame rate is a free parameter of the scene, not a form constant: the
    # seeker clips are authored at 20 fps (dt 0.05) and letting the widget's
    # 10 fps default through doubles the clip length, which silently doubles the
    # closing distance and turned a 37 px target into a 170 px one (measured).
    if q["weather"] != dflt.get("weather") or not cfg.get("weather"):
        cfg["weather"] = dict(WEATHER_MODES[q["weather"]])
    fr = cfg.setdefault("frames", {})
    fr["n"] = int(q["frames"])
    if changed("fps", q["fps"]) or not fr.get("dt"):
        fr["dt"] = round(1.0 / max(q["fps"], 0.1), 4)
    clip["frames_per_clip"] = int(q["fpc"])
    # The seed selects WHICH engagement geometry inside the scene's band, so an
    # untouched seed must not reseed the scene (that alone moved the start range
    # by 230 m and changed the whole clip).
    if "seed" in q["_present"] or "seed" not in cfg:
        cfg["seed"] = int(q["seed"])
    return cfg


# --------------------------------------------------------------------- jobs

class Job:
    def __init__(self, jid, name, out_dir, cfg_path, args):
        self.id = jid
        self.name = name
        self.out = out_dir
        self.cfg = cfg_path
        self.args = args
        self.state = "queued"
        self.started = time.time()
        self.finished = None
        self.done = 0
        self.total = 0
        self.log = []
        self.error = None
        self.proc = None
        self.meta = {}

    def push(self, line):
        line = line.rstrip()
        if not line:
            return
        self.log.append(line)
        if len(self.log) > 600:
            self.log = self.log[-600:]
        m = _RE_PROG.search(line)
        if m:
            self.done, self.total = int(m.group(1)), int(m.group(2))

    def as_dict(self):
        d = dict(id=self.id, name=self.name, state=self.state, out=self.out,
                 done=self.done, total=self.total,
                 seconds=round((self.finished or time.time()) - self.started, 1),
                 error=self.error, log=self.log[-40:], config=self.cfg,
                 meta=self.meta)
        if self.state == "failed":
            # The generator's stderr is merged into stdout, so the tail of the
            # log *is* the traceback. Show 30 lines -- anchored on the traceback
            # header, because a child that flushes its stdout at exit pushes the
            # useful lines out of a naive window, and keeping the first 20 lines
            # plus the last 8 keeps both "where it broke" and "what it raised".
            lines = []
            for entry in self.log:
                lines.extend(entry.splitlines() or [""])
            start = None
            for i in range(len(lines) - 1, -1, -1):
                if lines[i].startswith("Traceback (most recent call last)"):
                    start = i
                    break
            block = lines[start:] if start is not None else lines
            for ln in reversed(block):
                if re.match(r"^[A-Za-z_][\w.]*(Error|Exception|Interrupt)\b", ln):
                    d["error_line"] = ln.strip()
                    break
            if len(block) > 30:
                block = (block[:20] + ["… 省略 %d 行 …" % (len(block) - 28)] + block[-8:])
            d["error_tail"] = block
        return d


def _decode(raw):
    """One line of child output, decoded for a Chinese Windows console.

    Our own prints are UTF-8, but a tool the pipeline shells out to (Blender,
    cmd.exe, a driver) may emit GBK. Decoding the merged stream as UTF-8
    turned those lines into U+FFFD soup, which is useless in an error report.
    """
    for enc in ("utf-8", "gbk", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", "replace")


def _spawn(job, argv):
    def run():
        job.state = "running"
        try:
            env = dict(os.environ)
            env["PYTHONPATH"] = ROOT
            env.setdefault("PYTHONIOENCODING", "utf-8")
            p = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, bufsize=0)
            job.proc = p
            for raw in p.stdout:
                job.push(_decode(raw))
            rc = p.wait()
            if rc == 0:
                job.state = "done"
            elif job.state != "cancelled":
                job.state = "failed"
                job.error = "生成进程退出码 %d" % rc
        except Exception as exc:                    # pragma: no cover
            job.state = "failed"
            job.error = "%s: %s" % (type(exc).__name__, exc)
            job.log.append(traceback.format_exc()[-800:])
        finally:
            job.finished = time.time()

    t = threading.Thread(target=run, daemon=True)
    t.start()


def start_job(body, preview=False):
    q = _parse(body)
    cfg = _compose(q, _load_scene(q["scene_path"]))
    stamp = time.strftime("%Y%m%d_%H%M%S")
    tag = "preview_" if preview else "ds_"
    jid = "%s%s_%s" % (tag, stamp, q["scene"])
    if q["name"]:
        jid += "_" + q["name"]
    jid = _safe_name(jid, "run")
    out_dir = os.path.normpath(os.path.join(RUNS, jid))
    if os.path.dirname(out_dir) != os.path.normpath(RUNS):
        raise BadRequest("输出目录越界：%s" % out_dir)
    os.makedirs(out_dir, exist_ok=True)
    cfg_path = os.path.join(out_dir, "scene.json")
    with open(cfg_path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, ensure_ascii=False, indent=1)

    frames = q["frames"]
    if preview:
        frames = max(2, min(frames, 8))
    sensors = [s for s in q["sensors"] if s in ("optical", "ir", "radar", "lidar", "sar")]
    argv = [sys.executable, "-m", "simkit", "gen", "-c", cfg_path, "-o", out_dir,
            "-n", str(frames), "-s", ",".join(sensors) or "ir"]
    argv += ["--seed", str(int(q["seed"]))]
    job = Job(jid, q["scene"] + ("" if not q["name"] else "/" + q["name"]),
              out_dir, cfg_path, argv)
    job.total = frames
    job.meta = dict(scene=q["scene"], frames=frames, sensors=sensors,
                    views=q["views_mode"], traj=q["traj_mode"],
                    weather=q["weather"], sar=q["sar"]["enabled"],
                    target_px=q["target_px"], fps=q["fps"], preview=preview)
    for note in q["notes"]:
        job.push("[studio] " + note)
    with _LOCK:
        _JOBS[jid] = job
    _spawn(job, argv)
    return job


def _run_dir(jid):
    if not re.match(r"^[A-Za-z0-9_.-]+$", jid or ""):
        return None
    j = _JOBS.get(jid)
    if j:
        return j.out
    cand = os.path.join(RUNS, jid)
    return cand if os.path.isdir(cand) else None


def _safe_join(base, rel):
    rel = unquote(rel).replace("\\", "/").lstrip("/")
    p = os.path.normpath(os.path.join(base, rel))
    base = os.path.normpath(base)
    if p != base and not p.startswith(base + os.sep):
        return None
    return p


# ------------------------------------------------------------------- planning

def _plan_payload(q, cfg):
    """Trajectory facts only: plan.summary, no world build, no rendering."""
    from simkit import plan as s_plan, scene as s_scene
    os.makedirs(TMP, exist_ok=True)
    tok = re.sub(r"[^A-Za-z0-9_.-]+", "_", "%s_%s_%s_%s" % (
        q["scene"], int(time.time() * 1000) % 10 ** 9, q["traj_mode"], q["frames"]))
    tmp = os.path.join(TMP, "plan_%s.json" % tok)
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, ensure_ascii=False)
    loaded = s_scene.load_config(tmp)
    dt = float(loaded.get("frames", {}).get("dt", 0.1))
    seed = int(loaded.get("seed", 0))
    rows = s_plan.summary(loaded, int(q["frames"]), seed, dt)

    # Object size used for the pixel arithmetic: the scene may state the
    # apparent size it reasons about (target_m), otherwise measure the mesh.
    sk = (loaded.get("clip") or {}).get("seeker") or {}
    size = float(sk.get("target_m") or 0.0)
    if size <= 0.0:
        tgt = next((o for o in loaded.get("objects") or [] if o.get("label", True)), None)
        try:
            size = s_plan.object_size_m(tgt, TMP) if tgt else 4.0
        except Exception:
            size = 4.0

    views = [dict(name="seeker", fov_deg=q["narrow"]["fov_deg"],
                  width=q["narrow"]["width"], height=q["narrow"]["height"])]
    if q["views_mode"] == "dual":
        views.append(dict(name="wide", fov_deg=q["wide"]["fov_deg"],
                          width=q["wide"]["width"], height=q["wide"]["height"]))

    for r in rows:
        r0, r1 = (r.get("range_m") or [0, 0])[:2]
        # plan.summary() already reports per-view pixel spans; when it does not
        # (older planner), solve them from the range endpoints so the table is
        # complete either way.
        got = r.get("view_px")
        if not isinstance(got, dict):
            got = {}
        for v in r.get("views") or []:
            if v.get("name") and v.get("target_px"):
                got.setdefault(v["name"], v["target_px"])
        vp = {}
        for v in views:
            nm = v["name"]
            x = got.get(nm)
            if isinstance(x, (int, float)):
                vp[nm] = [round(float(x), 1)] * 2
            elif isinstance(x, (list, tuple)) and len(x) >= 2:
                vp[nm] = [round(float(x[0]), 1), round(float(x[1]), 1)]
            else:
                vp[nm] = [round(s_plan.px_for_range(size, r0, v["fov_deg"], v["width"]), 1),
                          round(s_plan.px_for_range(size, r1, v["fov_deg"], v["width"]), 1)]
        r["view_px"] = vp
        r["ratio"] = {v["name"]: [round(100.0 * vp[v["name"]][0] / v["width"], 2),
                                  round(100.0 * vp[v["name"]][1] / v["width"], 2)]
                      for v in views}
        r["view_range_m"] = r.get("view_range_m", r0)
        if r.get("bank_deg") is None:
            roll = r.get("roll_deg")
            r["bank_deg"] = (roll[0] if isinstance(roll, (list, tuple)) and roll else roll)

    ratios = [x for r in rows for x in r["ratio"]["seeker"]]
    sar = dict(q["sar"])
    sar["range_res_m"] = round(150.0 / max(sar["bandwidth_mhz"], 1e-6), 3)
    sar["azimuth_res_m"] = None
    return dict(
        clips=rows, views=views,
        scene=dict(key=q["scene"], size_m=round(size, 2),
                   target_px=q["target_px"], range_m=q["range_m"],
                   traj=q["traj_mode"], weather=q["weather"],
                   sensors=q["sensors"], fov=q["narrow"]["fov_deg"]),
        sar=sar,
        summary=dict(
            clips=len(rows),
            frames_per_clip=(rows[0]["frames"] if rows else 0),
            frames=q["frames"], dt=round(dt, 4), fps=round(1.0 / dt, 3) if dt else None,
            seconds=round(q["frames"] * dt, 2),
            px_min=(min(ratios) if ratios else None),
            px_max=(max(ratios) if ratios else None),
            px_abs=[min(x for r in rows for x in r["view_px"]["seeker"]),
                    max(x for r in rows for x in r["view_px"]["seeker"])] if rows else None,
            sensors=q["sensors"], views=q["views_mode"],
            weather=q["weather"], sar=q["sar"]["enabled"], notes=q["notes"]),
    )


# ------------------------------------------------------------------- datasets

MOD_PREF = ["optical", "optical_wide", "mwir", "mwir_wide", "lwir", "lwir_wide",
            "sar", "radar_ra", "radar_rd", "lidar", "lidar_intensity"]
MOD_CN = {"optical": "可见光", "mwir": "中波红外", "lwir": "长波红外", "swir": "短波红外",
          "sar": "SAR 强度", "radar_ra": "雷达 距离-角度", "radar_rd": "雷达 距离-多普勒",
          "radar": "雷达 距离-多普勒", "lidar": "激光雷达距离",
          "lidar_intensity": "激光雷达强度"}


def _mod_label(mod):
    """images/ir_preview_lwir_wide -> 长波红外 · 宽视场"""
    m = str(mod)
    wide = m.endswith("_wide")
    if wide:
        m = m[:-5]
    m = m.replace("ir_preview_", "").replace("ir_", "")
    lab = MOD_CN.get(m, m)
    return lab + (" · 宽视场" if wide else "")


def _png_size(path):
    """Width/height from the IHDR chunk: no PIL, no third-party dependency."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(24)
        if len(head) < 24 or head[:8] != b"\x89PNG\r\n\x1a\n":
            return None, None
        w = int.from_bytes(head[16:20], "big")
        h = int.from_bytes(head[20:24], "big")
        return w, h
    except OSError:
        return None, None


def _frame_no(name):
    m = _RE_COUNTER.search(os.path.splitext(name)[0])
    return int(m.group(1)) if m else None


def _label_dirs(base):
    """Every labels/<sub>/ directory, so a new view tag needs no code change."""
    root = os.path.join(base, "labels")
    if not os.path.isdir(root):
        return {}
    return {nm: os.path.join(root, nm) for nm in os.listdir(root)
            if os.path.isdir(os.path.join(root, nm))}


def _label_dir_name(mod):
    """Which labels/<sub>/ holds this image's per-frame JSON.

    Deliberately explicit: a fuzzy "first directory that exists" fallback made
    the radar card report the infrared target box, which is worse than no
    metadata at all.
    """
    if mod.startswith("radar"):
        return "radar"
    if mod.startswith("lidar"):
        return "lidar"
    if mod == "sar":
        return "sar"
    if mod.startswith("ir_"):
        return mod.replace("ir_preview", "ir")       # ir_preview_mwir_wide -> ir_mwir_wide
    if mod.startswith("optical"):
        return mod                                   # optical / optical_wide
    return mod


def _label_meta(base, mod, fname, ldirs=None):
    """range_m / target_px / measured box for one frame, from its label JSON.

    The label file already carries ``target_px`` for the view it belongs to and
    the boxes measured from the object mask, which is what "target occupies
    n px" should mean to a reader of the gallery.
    """
    ldirs = ldirs if ldirs is not None else _label_dirs(base)
    sub = ldirs.get(_label_dir_name(mod))
    if sub is None:
        return {}
    fp = os.path.join(sub, os.path.splitext(fname)[0] + ".json")
    if not os.path.isfile(fp):
        return {}
    key = (fp, os.path.getmtime(fp))
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    try:
        with open(fp, "r", encoding="utf-8") as fh:
            d = json.load(fh)
    except (OSError, ValueError):
        return {}
    boxes = d.get("boxes") or []
    bw = bh = None
    if boxes:
        b = boxes[0]
        try:
            bw = round(float(b["x1"]) - float(b["x0"]), 1)
            bh = round(float(b["y1"]) - float(b["y0"]), 1)
        except (KeyError, TypeError, ValueError):
            bw = bh = None
    px = d.get("target_px")
    if isinstance(px, (list, tuple)):
        px = [round(float(x), 1) for x in px[:2]]
    elif isinstance(px, (int, float)):
        px = round(float(px), 1)
    out = dict(clip=d.get("clip"), range_m=d.get("range_m"), target_px=px,
               width=d.get("width"), height=d.get("height"), fov_deg=d.get("fov_deg"),
               view=d.get("view"), box=[bw, bh] if bw else None)
    ra = d.get("ra_peak") or {}
    if ra:
        out["peak_db"] = ra.get("db")
    res = d.get("resolution") or {}
    if res:
        out["res_range_m"] = res.get("range_m")
        out["res_az_m"] = res.get("azimuth_m")
    sw = d.get("swath") or {}
    if sw:
        out["swath_m"] = [sw.get("near_m"), sw.get("far_m")]
    st = d.get("stats") or {}
    if st:
        out["snr_db"] = st.get("snr_db")
        out["peak_db"] = out.get("peak_db", st.get("peak_db"))
    _CACHE[key] = out
    return out


def _pick(files, n=4):
    if not files:
        return []
    idx = sorted({0, len(files) // 4, len(files) // 2, len(files) - 1})[:n]
    return [files[i] for i in idx]


def _mod_rank(mod):
    """Narrow (seeker) view before its wide counterpart, then a fixed order."""
    base = mod[:-5] if mod.endswith("_wide") else mod
    base = base.replace("ir_preview_", "")
    try:
        return MOD_PREF.index(base) * 2 + (1 if mod.endswith("_wide") else 0)
    except ValueError:
        return 99


def _list_datasets():
    out = []
    if not os.path.isdir(RUNS):
        return out
    for nm in sorted(os.listdir(RUNS), reverse=True):
        d = os.path.join(RUNS, nm)
        if not os.path.isdir(d):
            continue
        st = {}
        st_p = os.path.join(d, "stats.json")
        if os.path.exists(st_p):
            try:
                with open(st_p, "r", encoding="utf-8") as fh:
                    st = json.load(fh)
            except (OSError, ValueError):
                st = {}
        mods = sorted(f for f in os.listdir(d)
                      if f.startswith("annotations_") and f.endswith("_coco.json"))
        size_mb = 0.0
        try:
            size_mb = round(sum(os.path.getsize(os.path.join(dp, f))
                                for dp, _s, fs in os.walk(d) for f in fs) / 1048576.0, 1)
        except OSError:
            pass
        out.append(dict(id=nm, dir=d, frames=st.get("frames"),
                        modalities=[m[len("annotations_"):-len("_coco.json")] for m in mods],
                        optical=st.get("optical_images"), ir=st.get("ir_images"),
                        lidar=st.get("lidar_images"), radar=st.get("radar_frames"),
                        sar=st.get("sar_images"), size_mb=size_mb,
                        # A run that died has a directory but no stats.json: say
                        # so instead of printing a row of misleading zeros.
                        ok=bool(st)))
    return out


def _dataset_info(jid):
    base = _run_dir(jid)
    if not base:
        return None
    info = dict(id=jid, dir=base, stats={}, meta={}, samples={}, cards=[], views=[])
    for nm, key in (("stats.json", "stats"), ("meta.json", "meta")):
        fp = os.path.join(base, nm)
        if os.path.exists(fp):
            try:
                with open(fp, "r", encoding="utf-8") as fh:
                    info[key] = json.load(fh)
            except (OSError, ValueError):
                pass
    seen = set()
    order = []
    ldirs = _label_dirs(base)
    img_root = os.path.join(base, "images")
    cand = []
    if os.path.isdir(img_root):
        for nm in os.listdir(img_root):
            d = os.path.join(img_root, nm)
            if not os.path.isdir(d):
                continue
            # images/ir_lwir is the raw 16-bit count image: unreadable in a
            # browser, and the preview copy next to it is the same frame.
            if nm.startswith("ir_") and not nm.startswith("ir_preview_"):
                continue
            cand.append((nm, "images/" + nm, d))
    for nm in ("range", "intensity"):
        d = os.path.join(base, "lidar", nm)
        if os.path.isdir(d):
            cand.append(("lidar" if nm == "range" else "lidar_intensity",
                         "lidar/" + nm, d))
    cand.sort(key=lambda c: (_mod_rank(c[0]), c[0]))
    for mod, rel, d in cand:
        if os.path.normpath(d) in seen:
            continue
        seen.add(os.path.normpath(d))
        order.append((mod, rel, _mod_label(mod), d))

    for mod, rel, label, d in order:
        files = sorted(f for f in os.listdir(d) if f.lower().endswith(".png"))
        if not files:
            continue
        info["samples"][mod] = files[:8]
        for fn in _pick(files, 4):
            w, h = _png_size(os.path.join(d, fn))
            meta = _label_meta(base, mod, fn, ldirs)
            px = meta.get("target_px")
            box = meta.get("box")
            ratio = None
            ref = box[0] if box else (px[0] if isinstance(px, list) else px)
            if ref and w:
                ratio = round(100.0 * float(ref) / w, 2)
            info["cards"].append(dict(
                mod=mod, label=label, file=fn, url="/api/img/%s/%s/%s" % (jid, rel, fn),
                w=w, h=h, frame=_frame_no(fn), target_px=px, box=box, ratio=ratio,
                range_m=meta.get("range_m"), clip=meta.get("clip"),
                fov_deg=meta.get("fov_deg"), peak_db=meta.get("peak_db"),
                snr_db=meta.get("snr_db"), res_range_m=meta.get("res_range_m"),
                res_az_m=meta.get("res_az_m"), swath_m=meta.get("swath_m")))
    info["views"] = ((info["meta"].get("views") if isinstance(info["meta"], dict) else None)
                     or (info["stats"].get("views") if isinstance(info["stats"], dict) else None)
                     or [])
    return info


# --------------------------------------------------------------------- server

def _jsonable(x, depth=0):
    """Make a payload strictly valid JSON.

    stats.json can legitimately contain NaN (an empty radar frame has no SNP
    ratio), and ``json.dumps`` writes a bare ``NaN`` that Python accepts and
    every browser rejects -- which turns the gallery into "unexpected token N".
    Non-finite floats become null here, so the UI degrades instead of breaking.
    """
    if depth > 16:
        return str(x)
    if isinstance(x, float):
        return x if x == x and x not in (float("inf"), float("-inf")) else None
    if x is None or isinstance(x, (str, bool, int)):
        return x
    if isinstance(x, dict):
        return {str(k): _jsonable(v, depth + 1) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_jsonable(v, depth + 1) for v in x]
    item = getattr(x, "item", None)          # numpy scalars / 0-d arrays
    if callable(item):
        try:
            return _jsonable(item(), depth + 1)
        except Exception:
            pass
    return str(x)


class Handler(BaseHTTPRequestHandler):
    server_version = "simkit-studio"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *a):        # keep the console for the run log
        pass

    def handle_one_request(self):
        # A browser that closes a keep-alive socket mid-request is normal on
        # Windows; without this the server prints a traceback per abandoned tab.
        try:
            BaseHTTPRequestHandler.handle_one_request(self)
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
            self.close_connection = True

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        if isinstance(body, (dict, list)):
            body = json.dumps(_jsonable(body), ensure_ascii=False,
                              allow_nan=False).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        if n > 1 << 20:
            raise BadRequest("请求体过大（%d 字节）" % n)
        raw = self.rfile.read(n)
        try:
            return json.loads(raw.decode("utf-8"))
        except ValueError:
            raise BadRequest("请求体不是合法 JSON")

    def _get_presets(self):
        return dict(
            scenes=_scenes(), presets=_scenes(),
            weather=[dict(key=k, label=WEATHER_LABELS.get(k, k)) for k in WEATHER_MODES],
            sensors=SENSORS, sensor_keys=SENSOR_KEYS,
            sar_bands=[dict(key=k, fc_ghz=v["fc_ghz"], note=v["note"])
                       for k, v in SAR_BANDS.items()],
            radar_bands=["X", "Ku", "K", "Ka", "W", "W94"],
            views=dict(narrow=VIEW_NARROW, wide=VIEW_WIDE),
            size_px=SIZE_PX, closing=CLOSING, traj_modes=TRAJ_MODES,
            limits=LIM, runs=RUNS)

    def do_GET(self):
        u = urlparse(self.path)
        p = u.path
        if p == "/favicon.ico":
            return self._send(204, b"", "image/x-icon")
        if p in ("/", "/index.html"):
            return self._static("index.html", "text/html; charset=utf-8")
        if p.startswith("/api/img/"):
            rest = p[len("/api/img/"):]
            jid, _, rel = rest.partition("/")
            base = _run_dir(jid)
            if not base:
                return self._send(404, dict(error="没有这个运行目录"))
            fp = _safe_join(base, rel)
            if not fp or not os.path.isfile(fp):
                return self._send(404, dict(error="没有这个文件"))
            ext = os.path.splitext(fp)[1].lower()
            ctype = {".png": "image/png", ".json": "application/json",
                     ".ply": "application/octet-stream"}.get(ext, "application/octet-stream")
            with open(fp, "rb") as fh:
                return self._send(200, fh.read(), ctype)
        if p == "/api/presets":
            return self._send(200, self._get_presets())
        if p == "/api/scenes":
            return self._send(200, dict(scenes=_scenes()))
        if p == "/api/jobs":
            with _LOCK:
                js = [j.as_dict() for j in _JOBS.values()]
            return self._send(200, dict(jobs=sorted(js, key=lambda j: j["id"], reverse=True)))
        if p.startswith("/api/job/"):
            jid = p[len("/api/job/"):].strip("/")
            j = _JOBS.get(jid)
            if not j:
                return self._send(404, dict(error="没有这个任务"))
            return self._send(200, j.as_dict())
        if p == "/api/datasets":
            return self._send(200, dict(datasets=_list_datasets()))
        if p == "/api/plan":
            q = parse_qs(u.query)
            body = dict(
                scene=(q.get("scene") or q.get("preset") or [""])[0],
                frames=(q.get("frames") or ["100"])[0],
                frames_per_clip=(q.get("frames_per_clip") or ["50"])[0],
                views_mode=(q.get("views_mode") or ["single"])[0],
                traj_mode=(q.get("traj_mode") or ["seeker"])[0])
            try:
                qq = _parse(body)
                return self._send(200, _plan_payload(qq, _compose(qq, _load_scene(qq["scene_path"]))))
            except BadRequest as exc:
                return self._send(400, dict(error=str(exc), clips=[]))
            except Exception as exc:                       # planning is best-effort
                return self._send(200, dict(clips=[], error="%s: %s" % (type(exc).__name__, exc)))
        if p.startswith("/api/dataset/"):
            jid = p[len("/api/dataset/"):].strip("/")
            info = _dataset_info(jid)
            if info is None:
                return self._send(404, dict(error="没有这个数据集"))
            return self._send(200, info)
        return self._static(p.lstrip("/"), None)

    def _static(self, rel, ctype):
        rel = rel or "index.html"
        fp = _safe_join(UI, rel)
        if not fp or not os.path.isfile(fp):
            return self._send(404, dict(error="找不到：" + rel))
        ext = os.path.splitext(fp)[1].lower()
        mime = {".html": "text/html; charset=utf-8", ".js": "application/javascript; charset=utf-8",
                ".css": "text/css; charset=utf-8", ".png": "image/png",
                ".json": "application/json; charset=utf-8"}.get(ext, "application/octet-stream")
        with open(fp, "rb") as fh:
            return self._send(200, fh.read(), mime)

    def do_POST(self):
        u = urlparse(self.path)
        try:
            if u.path == "/api/plan":
                q = _parse(self._body())
                return self._send(200, _plan_payload(
                    q, _compose(q, _load_scene(q["scene_path"]))))
            if u.path == "/api/generate":
                return self._send(200, start_job(self._body(), preview=False).as_dict())
            if u.path == "/api/preview":
                return self._send(200, start_job(self._body(), preview=True).as_dict())
        except BadRequest as exc:
            return self._send(400, dict(error=str(exc)))
        except Exception as exc:
            return self._send(500, dict(error="%s: %s" % (type(exc).__name__, exc),
                                        trace=traceback.format_exc()[-1200:]))
        if u.path.startswith("/api/job/") and u.path.endswith("/cancel"):
            jid = u.path[len("/api/job/"):-len("/cancel")].strip("/")
            j = _JOBS.get(jid)
            if not j:
                return self._send(404, dict(error="没有这个任务"))
            if j.proc and j.proc.poll() is None:
                j.proc.terminate()
                j.state = "cancelled"
            return self._send(200, j.as_dict())
        return self._send(404, dict(error="未知端点"))


def serve(port=8808, host="127.0.0.1"):
    try:
        port = int(port)
    except (TypeError, ValueError):
        raise SystemExit("端口必须是整数：%r" % (port,))
    if not (LIM["port"][0] <= port <= LIM["port"][1]):
        raise SystemExit("端口应在 %d ~ %d 之间，收到 %d" % (LIM["port"] + (port,)))
    if host not in ("127.0.0.1", "localhost", "::1"):
        print("warning: binding %s exposes the studio on the network" % host)
    os.makedirs(RUNS, exist_ok=True)
    os.makedirs(TMP, exist_ok=True)
    httpd = ThreadingHTTPServer((host, port), Handler)
    url = "http://%s:%d/" % (host, port)
    print("simkit studio -> %s" % url)
    print("scenes %d, runs land in %s" % (len(_scene_paths()), RUNS))
    print("press Ctrl+C to stop")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()
