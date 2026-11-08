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
    GET  /api/align/<id>        per-frame optical/IR/radar alignment table + radar image pairs
    GET  /api/datasets          runs already on disk
    GET  /api/img/<id>/<path>   serve a generated image

Every user-supplied number is range-checked; a bad value answers 400 with a
Chinese message instead of raising inside a thread and killing the run. The
output directory is always a direct child of ``out/studio_runs``.

Nothing here touches the network, and the UI is plain ES5 so it works from a
USB stick on an air-gapped machine.
"""
import json
import math
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
# Saved recipes: a form state you can name, reload and hand to someone else.
# Stored outside scenes/ on purpose -- a recipe is not a scene until scenegen
# turns it into one, and mixing the two makes "pick a template" ambiguous.
RECIPES = os.path.join(ROOT, "configs", "recipes")
# Output folders the operator picked outside out/studio_runs. Remembered on disk
# so the dataset browser still lists them after the server restarts.
_KNOWN_OUT = os.path.join(ROOT, "configs", "known_outputs.json")
_CUSTOM_OUT = set()

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
    "platform": "平台俯冲 + 蛇形（机载导引）",
    "seeker": "相机静立 · 目标飞行（旧机载导引）",
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

# Per-frame cost of each sensor, calibrated against the one full run we have
# measured end to end: out/validation_ecm, 10 frames x 5 sensors = 140.29 s and
# ~5.6 MB/frame once the raw complex cubes are excluded. These are estimates and
# the UI labels them as such; they are only here to stop someone from starting a
# 20000-frame all-sensor run on a laptop.
COST_SEC_PER_FRAME = dict(optical=9.0, ir=2.5, radar=0.8, lidar=0.6, sar=1.6)
COST_MB_PER_FRAME = dict(optical=2.4, ir=1.3, radar=0.15, lidar=0.05, sar=1.5)

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


def _bool(body, key, label, default=False):
    v = body.get(key)
    if v is None or v == "":
        return bool(default)
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    text = str(v).strip().lower()
    if text in ("1", "true", "yes", "on", "是"):
        return True
    if text in ("0", "false", "no", "off", "否"):
        return False
    raise BadRequest("%s 必须是布尔值，收到 %r" % (label, v))


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
        # paths is stem -> path; returning the stem made the caller try to open
        # a file named 'demo_air_uav' and fail with a bare ENOENT.
        return paths[sorted(paths)[0]]
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
    group = "机载导引" if (sk.get("mode") == "platform" or stem.lower().startswith("sk2")) else \
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
    order = {"机载导引": 0, "基础": 1, "SAR": 2}
    rows.sort(key=lambda r: (order.get(r[1].get("group"), 3),
                             _PREFIX_ORDER.get(r[0].split("_")[0], 9), r[0]))
    return [r[1] for r in rows]


# --------------------------------------------------------------- body -> config

# Chinese labels for the jamming entities. Kept next to the UI code because the
# physics layer names them in English and the operator should not have to.
_ECM_CN = {
    "noise_jammer": "压制干扰机",
    "false_target": "转发假目标",
    "range_gate_pull_off": "距离门拖引",
    "velocity_gate_pull_off": "速度门拖引",
    "angle_deception": "角度欺骗",
    "chaff_cloud": "箔条云",
    "corner_reflector": "角反射器",
    "decoy": "拖曳诱饵",
    "infrared_flare": "红外诱饵弹",
}


def _recipe_to_form(rec):
    """Flatten the recipe's nested blocks onto the flat form keys _parse reads."""
    cap = rec.get("capture") or {}
    view = rec.get("view") or {}
    env = rec.get("environment") or {}
    out = dict(
        frames=cap.get("frames"), fps=cap.get("fps"),
        sensors=cap.get("sensors"), seed=cap.get("seed"),
        aspect=view.get("aspect"), weather=env.get("weather"),
        narrow_fov=view.get("fov_deg"))
    # Image dimensions ride on the flat lens keys. Only keys the recipe carries
    # are offered; _parse fills absent body values, so an old recipe (no size
    # keys) keeps the factory 960x720 / 1280x960 defaults bit-for-bit.
    for form_key, view_key in (("narrow_width", "width"),
                               ("narrow_height", "height"),
                               ("wide_width", "wide_width"),
                               ("wide_height", "wide_height")):
        if view.get(view_key) is not None:
            out[form_key] = view[view_key]
    fx = rec.get("effects") or {}
    out["plume"] = bool(fx.get("plume", False))
    out["supersample"] = int(fx.get("supersample", 0) or 0)
    km = view.get("range_km")
    if km:
        lo = float(km) * 1000.0 * 0.85
        hi = float(km) * 1000.0 * 1.15
        out["range_m"] = [lo, hi]
    # Keep the form's own units for fov: the wide view has to stay wider than
    # the narrow one or _parse rejects the pair.
    if out.get("narrow_fov"):
        out["wide_fov"] = max(float(out["narrow_fov"]) * 6.0, 18.0)
    return out


def _trajectory_smooth(payload):
    """Turn a hand-drawn polyline into a flyable route.

    The operator clicks a few points on a top-down canvas; those points are not a
    flight path -- the corners are instantaneous heading changes, which no
    airframe survives. They are fitted with the same C1 cubic spline the planner
    uses for authored waypoints (motion.Waypoints), and the result is measured so
    the UI can say "this needs 1.2 g and 54 deg of bank" before anything renders.
    """
    from . import motion

    raw = payload.get("points")
    if not isinstance(raw, (list, tuple)):
        raise BadRequest("轨迹点必须是 [[x, y], ...] 这样的数组")
    pts = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            raise BadRequest("每个轨迹点必须是 [x, y] 两个坐标")
        try:
            x, y = float(item[0]), float(item[1])
        except (TypeError, ValueError):
            raise BadRequest("轨迹点坐标必须是数字") from None
        if not (abs(x) <= 50000.0 and abs(y) <= 50000.0):
            raise BadRequest("轨迹点超出范围（±50 km）")
        if not pts or abs(x - pts[-1][0]) > 1.0 or abs(y - pts[-1][1]) > 1.0:
            pts.append([x, y])
    if len(pts) < 2:
        raise BadRequest("至少需要两个不同的轨迹点")

    try:
        alt = float(payload.get("alt_m") or 2500.0)
        speed = float(payload.get("speed_mps") or 250.0)
    except (TypeError, ValueError):
        raise BadRequest("高度与速度必须是数字") from None
    if not (1.0 <= speed <= 1000.0):
        raise BadRequest("速度必须在 1–1000 m/s 之间")
    closed = bool(payload.get("closed"))
    P = [[p[0], p[1], alt] for p in pts]

    def _measure(pts3, spd):
        """Fit the points and report length, time, worst load and worst bank."""
        # ramp_in=False because this is the route the build will fly: a drawn
        # leg enters at speed.  Measuring it with an entry ramp reported a
        # duration the operator would never see and hid the fact that the whole
        # two-second clip was the ramp.
        path = motion.Waypoints(pts3, speed=spd, loop=closed, ramp_in=False)
        dur = max(float(path.duration), 1e-6)
        steps = 96
        sampled = []
        wa = wb = ws = 0.0
        for i in range(steps + 1):
            t = dur * i / steps
            pos, vel, acc = path.pos(t), path.vel(t), path.accel(t)
            sampled.append([round(float(pos[0]), 2), round(float(pos[1]), 2),
                            round(float(pos[2]), 2)])
            # Load factor is the LATERAL acceleration (lift), not the total. An
            # open route accelerates from rest along-track at the ends, and
            # counting that as g made a dead-straight line report 25 g and get
            # its speed cut to 50 m/s.
            vn = math.sqrt(float(vel[0]) ** 2 + float(vel[1]) ** 2
                           + float(vel[2]) ** 2)
            if vn > 1e-6:
                ah = [float(acc[i]) - (float(acc[0]) * float(vel[0])
                                       + float(acc[1]) * float(vel[1])
                                       + float(acc[2]) * float(vel[2])) / (vn * vn)
                      * float(vel[i]) for i in range(3)]
            else:
                ah = [0.0, 0.0, 0.0]
            wa = max(wa, math.sqrt(ah[0] ** 2 + ah[1] ** 2 + ah[2] ** 2))
            ws = max(ws, vn)
            try:
                wb = max(wb, abs(float(motion.coordinated_bank_deg(vel, acc))))
            except (ValueError, ZeroDivisionError):
                pass
        return path, sampled, wa, wb, ws

    # Hand-drawn corners are instantaneous heading changes, so a drawn route at
    # a fixed speed can need tens of g (a 7-point zigzag measured 37 g at
    # 250 m/s). Rounding the corners outward barely helps -- the demand is
    # v^2 * kappa, and it is the SPEED that sets it. So the optimiser solves for
    # the fastest speed this exact geometry can be flown at inside the load
    # limit, refits at that speed, and says what it did.
    LIMIT_G = 1.5
    MIN_SPEED = 25.0
    P_work = [list(p) for p in P]
    path, pts_out, worst_a, worst_bank, worst_speed = _measure(P_work, speed)
    g = worst_a / 9.80665
    g_at_request = g
    speed_used = float(speed)
    if g > LIMIT_G and g > 0.0:
        wanted = speed * math.sqrt(LIMIT_G / g)
        speed_used = max(MIN_SPEED, math.floor(wanted * 10.0) / 10.0)
        if speed_used < speed:
            try:
                path, pts_out, worst_a, worst_bank, worst_speed = _measure(
                    P_work, speed_used)
                g = worst_a / 9.80665
            except ValueError:
                speed_used = float(speed)
    tries = 1 if speed_used < float(speed) else 0
    warnings = []
    if g > 1.5:
        warnings.append("最大过载 %.2f g 超过固定翼无人机常规机动上限（约 1.5 g），"
                        "生成时会被包线判为不可飞；建议放大转弯半径" % g)
    if worst_bank > 60.0:
        warnings.append("最大坡度 %.0f° 偏大，真实机体通常不超过 60°" % worst_bank)
    if float(path.length) < 200.0:
        warnings.append("轨迹总长仅 %.0f m，帧数稍多就会飞出画面" % float(path.length))

    if tries:
        warnings.insert(0, "按 %.0f m/s 飞这条轨迹需要 %.1f g，已自动降到 %.1f m/s"
                           "（%.1f g）使其可飞；想要更快请把转弯画得更大"
                        % (float(speed), g_at_request, speed_used, g))
    return dict(ok=True, points=pts_out, length_m=round(float(path.length), 1),
                duration_s=round(float(path.duration), 2), max_g=round(g, 3),
                max_bank_deg=round(worst_bank, 1),
                max_speed_mps=round(worst_speed, 1),
                speed_requested=round(float(speed), 1),
                speed_used=round(speed_used, 1),
                relaxed=tries, closed=closed, warnings=warnings)


# ------------------------------------------------------- output destination

# Source and configuration folders a dataset must never be written into: a typo
# in the path box would scatter generated frames through the tree, where git and
# the asset scans would pick them up.
#
# "datasets" is deliberately NOT here. It is the YOLO export target and the
# obvious place an operator puts a dataset; refusing it was over-protection and
# made the natural choice ("...\datasets\1103") fail.
_PROTECTED_OUT = ("simkit", "scenes", "assets", "docs", "tools", "tests",
                  "blender", "studio", "configs", "env", "third_party",
                  "data", "rcs_refs")


def _safe_out_dir(raw):
    """Absolute, checked output folder, or None to use the default runs area."""
    if raw in (None, ""):
        return None
    p = os.path.expanduser(str(raw).strip().strip('"').strip("'"))
    if not p:
        return None
    if not os.path.isabs(p):
        p = os.path.join(ROOT, p)
    p = os.path.normpath(p)
    low, root_low = p.lower(), ROOT.lower()
    if low == root_low:
        raise BadRequest("输出目录不能是项目根目录本身")
    if os.path.dirname(p) == p:
        raise BadRequest("输出目录不能是磁盘根目录")
    for d in _PROTECTED_OUT:
        walk = os.path.join(root_low, d)
        if low == walk or low.startswith(walk + os.sep):
            raise BadRequest("输出目录不能放在 %s/ 里（那是源码或配置目录）：%s"
                             % (d, p))
    return p


def _unique_out_dir(base):
    """Do not mix two generations, and never write into someone else's folder."""
    if not os.path.exists(base):
        return base
    if os.path.isfile(os.path.join(base, "meta.json")):
        for i in range(2, 1000):
            cand = "%s_%d" % (base, i)
            if not os.path.exists(cand):
                return cand
        raise BadRequest("同名目录太多，请换一个输出目录：%s" % base)
    if os.path.isdir(base) and not os.listdir(base):
        return base
    raise BadRequest("目录已存在、非空，且不是本工具生成的数据集，拒绝写入：%s" % base)


def _writable_ancestor(path):
    """True when the nearest EXISTING ancestor is writable.

    A brand-new nested path such as D:/work/new/run01 has a parent that does not
    exist yet; testing the parent directly reported "not writable" and refused a
    perfectly legal destination. makedirs creates the chain, so the only thing
    that matters is whether the closest real folder accepts children.
    """
    p = os.path.dirname(os.path.normpath(path)) or "."
    while p and not os.path.exists(p):
        nxt = os.path.dirname(p)
        if nxt == p:
            return False
        p = nxt
    return bool(p) and os.path.isdir(p) and os.access(p, os.W_OK)


def _subdirs(path, cap=300):
    try:
        names = []
        with os.scandir(path) as it:
            for e in it:
                try:
                    if e.is_dir(follow_symlinks=False) and not e.name.startswith("."):
                        names.append(e.name)
                except OSError:
                    continue
        return sorted(names)[:cap]
    except OSError:
        return []


def _assets_payload():
    """Model library: every .obj under assets/models, grouped by its folder.

    The form needs a list it can show as a picker, with a relative path it can
    put straight into spec["model"]; absolute paths would leak this machine's
    layout into the recipe.
    """
    base = os.path.join(ROOT, "assets", "models")
    if not os.path.isdir(base):
        return dict(assets=[], error="资产库暂不可用：assets/models 不存在")
    # scenegen already resolves each model's class, size and triangle count. A
    # hand-rolled scan here dropped those three fields and the damage was not
    # cosmetic: the UI writes asset["cls"] into recipe.objects[0].cls, so without
    # it every tank and ship was annotated as "fighter".
    from . import scenegen
    try:
        rows = scenegen.list_assets()
    except Exception as exc:
        return dict(assets=[], error="资产库读取失败：%s: %s" % (type(exc).__name__, exc))
    out = []
    for row in rows:
        item = dict(row)
        try:
            item["size_kb"] = round(
                os.path.getsize(os.path.join(base, str(row["path"]))) / 1024.0, 1)
        except (OSError, KeyError, TypeError):
            item["size_kb"] = 0.0
        out.append(item)
    if not out:
        return dict(assets=[], error="资产库暂不可用：assets/models 下没有扫描到模型")
    return dict(assets=out)


BACKDROP_LABELS = {"sky": "云层天空", "forest": "森林", "city": "城市",
                   "desert": "荒漠", "rural": "乡村农田", "mountain": "山地",
                   "industrial": "工业区", "port": "港口", "island": "岛屿",
                   "open": "开阔海面"}
BACKDROP_DOMAIN_LABELS = {"air": "空域", "land": "陆域", "sea": "海域"}


def _backdrops_payload():
    """Real photograph plates collected under assets/backdrops.

    Served to the workbench so the operator picks a category (and the engine
    rotates through the files by seed) instead of memorising file paths.
    """
    from . import scenegen
    lib = scenegen.backdrop_library()
    out = []
    for key in scenegen.BACKDROP_CATEGORIES:
        files = lib.get(key) or []
        dom = scenegen.BACKDROP_DOMAIN.get(key, "land")
        out.append(dict(key=key, label=BACKDROP_LABELS.get(key, key),
                        domain=dom,
                        domain_label=BACKDROP_DOMAIN_LABELS.get(dom, dom),
                        count=len(files),
                        files=files[:60],
                        cover=("/api/backdrop/%s/%s" % (key, files[0])) if files else ""))
    total = sum(c["count"] for c in out)
    return dict(categories=out, total=total,
                error="" if total else "背景库为空：assets/backdrops 下没有图片")


def _browse_payload(raw):
    """List folders so the UI can offer a picker instead of a blind text box."""
    p = os.path.expanduser(str(raw or "").strip().strip('"').strip("'"))
    if not p:
        p = ROOT
    if not os.path.isabs(p):
        p = os.path.join(ROOT, p)
    p = os.path.normpath(p)
    if os.path.isdir(p):
        return dict(path=p, requested=p, exists=True, isdir=True,
                    parent=os.path.dirname(p), dirs=_subdirs(p),
                    writable=os.access(p, os.W_OK))
    parent = os.path.dirname(p)
    if os.path.isdir(parent):
        # A folder that does not exist yet is a legitimate choice: the platform
        # creates it. Report the parent so the UI can still navigate.
        return dict(path=parent, requested=p, exists=False, isdir=False,
                    parent=os.path.dirname(parent), dirs=_subdirs(parent),
                    writable=os.access(parent, os.W_OK))
    raise BadRequest("目录不存在：%s" % p)


# ------------------------------------------------------------ recipes & checks

def _recipes_dir():
    os.makedirs(RECIPES, exist_ok=True)
    return RECIPES


def _recipe_path(name):
    stem = _safe_name(name)
    if not stem:
        raise BadRequest("配方需要一个名字（字母、数字、下划线或短横线）")
    return os.path.join(_recipes_dir(), stem + ".json")


def _list_recipes():
    out = []
    d = _recipes_dir()
    for fn in sorted(os.listdir(d)):
        if not fn.lower().endswith(".json"):
            continue
        p = os.path.join(d, fn)
        try:
            with open(p, "r", encoding="utf-8") as fh:
                rec = json.load(fh)
        except (OSError, ValueError):
            continue
        out.append(dict(name=fn[:-5], path=p,
                        title=rec.get("title") or rec.get("name") or fn[:-5],
                        saved_at=rec.get("_saved_at")))
    return out


def _load_recipe(name):
    p = _recipe_path(name)
    if not os.path.isfile(p):
        raise BadRequest("没有这个配方：" + str(name))
    try:
        with open(p, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except ValueError as exc:
        raise BadRequest("配方文件不是合法 JSON：" + str(exc)) from exc


def _save_recipe(name, recipe):
    if not isinstance(recipe, dict):
        raise BadRequest("配方必须是 JSON 对象")
    p = _recipe_path(name)
    rec = json.loads(json.dumps(recipe))
    rec["_saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, ensure_ascii=False, indent=1)
    return p


def _write_tmp_scene(cfg, name):
    """Materialise a built scene so the existing plan/generate path can load it.

    The generator reads a config from disk, so a recipe has to become a file
    before it can run. The token keeps concurrent previews from colliding.
    """
    os.makedirs(TMP, exist_ok=True)
    tok = _safe_name(name) or "recipe"
    path = os.path.join(TMP, "scene_%s_%d.json" % (tok, int(time.time() * 1000) % 10 ** 9))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, ensure_ascii=False, indent=1)
    return path


def _ecm_kinds_payload():
    """Every jamming/infrared entity the physics layer knows, with its defaults."""
    from . import ecm as s_ecm
    kinds = []
    for k in s_ecm.KINDS:
        kinds.append(dict(key=k, group=s_ecm.GROUP_OF.get(k, "passive"),
                          label=_ECM_CN.get(k, k),
                          defaults=s_ecm.DEFAULTS.get(k, {})))
    return dict(kinds=kinds,
                groups=[dict(key="active", label="有源"),
                        dict(key="passive", label="无源")])


def _estimate_payload(q):
    """Rough time and disk, so nobody starts a 20000-frame run by accident."""
    sensors = [s for s in q["sensors"]]
    frames = int(q["frames"])
    sec = sum(COST_SEC_PER_FRAME.get(s, 1.0) for s in sensors) * frames
    mb = sum(COST_MB_PER_FRAME.get(s, 0.5) for s in sensors) * frames
    return dict(frames=frames, sensors=sensors,
                seconds=round(sec, 1), megabytes=round(mb, 1),
                note="按实测 140.29 s / 10 帧 / 5 模态标定，仅供参考")


def _preflight_payload(q):
    """Cheap geometric and physical sanity checks. Never renders."""
    errors, warnings = [], []
    sensors = list(q["sensors"])
    if not sensors:
        errors.append("至少选一个模态")
    if "optical" in sensors:
        try:
            from . import optical as s_opt
            if not s_opt.available(q.get("cfg") or {}):
                warnings.append("可见光需要 Blender；当前没找到，将只产出其它模态")
        except Exception:
            pass

    # Contradictory aspect and closing: head-on means the target comes at the
    # camera, so an "out" closing rate describes the opposite geometry.
    aspect = str(q.get("aspect") or "")
    closing = str(q.get("closing") or "")
    if aspect == "head_on" and closing == "out":
        warnings.append("迎头拍摄却设为远离：几何自相矛盾，建议把接近方式改成「接近」")
    if aspect == "pursuit" and closing == "in":
        warnings.append("追击拍摄却设为接近：目标会倒着飞，建议改成「远离」")
    if aspect in ("look_down", "top_down") and closing == "out":
        # A look-down engagement closes: the camera is above the target and the
        # slant range shrinks. "out" here would have the target recede downwards.
        warnings.append("俯视拍摄却设为远离：俯视是自上而下接近，建议改成「接近」")

    # Sub-pixel target: the pixel budget is the honest way to say "you asked for
    # a 500 m target at 40 km through a 2 deg lens and you will get three dots".
    try:
        fov = float(q["narrow"]["fov_deg"])
        width = int(q["narrow"]["width"])
        px_lo = float(q["target_px"][0])
        rng = q.get("range_m")
        if rng:
            f_px = width / (2.0 * math.tan(math.radians(fov) / 2.0))
            # Measure the actual target. Guessing 4 m made a 20 m aircraft at
            # 9 km read as 11 px and raised a warning that was simply wrong;
            # prepare_mesh is cached per process, so looking it up is cheap.
            size_m = 4.0
            try:
                from . import plan as s_plan, scene as s_scene
                sc = s_scene.load_config(q["scene_path"])
                tgt = next((o for o in sc.get("objects") or [] if o.get("label", True)), None)
                if tgt is not None:
                    size_m = float(s_plan.object_size_m(
                        tgt, os.path.dirname(q["scene_path"])))
            except Exception:
                pass
            px = size_m * f_px / max(float(rng[1]), 1e-6)
            if px < px_lo:
                warnings.append("最远 %.1f km 处目标仅约 %.1f px，低于你设的下限 %.0f px；"
                                "建议拉近距离或缩小视场角" % (float(rng[1]) / 1000.0, px, px_lo))
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        pass

    # A jammer that cannot beat the noise floor changes nothing, and a run that
    # silently contains no jamming is worse than a run that refuses.
    ecm_cfg = q.get("ecm")
    if isinstance(ecm_cfg, dict) and ecm_cfg.get("entities"):
        act = [e for e in ecm_cfg["entities"] if e.get("kind") == "noise_jammer"]
        total = sum(float(e.get("erp_w") or 0.0) for e in act)
        if act and total < 50.0:
            warnings.append("有源干扰总 ERP 仅 %.1f W；实测 Ku 波段 500 m 下 200 W 才到 "
                            "J/S +10 dB，低于 50 W 基本压不住" % total)

    # Doppler aliasing. A single-PRF radar folds |v| > lam/(4*PRI), so a fighter
    # closing at 250 m/s lands in the wrong Doppler cell and the RD image teaches
    # a velocity that is not the true one. The label already says "aliased"; the
    # operator still deserves to know before spending an hour on the run.
    try:
        from . import scene as s_scene
        sc = s_scene.load_config(q["scene_path"])
        rb = (sc.get("sensors") or {}).get("radar") or {}
        pri = float(rb.get("pri_us") or 0.0) * 1e-6
        fc_ghz = {"X": 9.6, "Ku": 15.5, "K": 24.0, "Ka": 35.0, "W": 94.0}.get(
            str(rb.get("band") or "Ku"), 15.5)
        sk = (sc.get("clip") or {}).get("seeker") or {}
        v = abs(float(sk.get("closing_mps") or 0.0))
        if pri > 0.0 and v > 0.0:
            lam = 2.998e8 / (fc_ghz * 1e9)
            v_nyq = lam / (4.0 * pri)
            if v > v_nyq:
                warnings.append(
                    "接近率 %.0f m/s 超过 %s 波段的多普勒不模糊速度 ±%.1f m/s"
                    "（PRI %s µs），雷达距离-多普勒图的多普勒轴会回绕；"
                    "标注已标记 single-PRF aliased，若要正确速度需改用更短 PRI 的波形"
                    % (v, str(rb.get("band") or "Ku"), v_nyq, rb.get("pri_us")))
    except (OSError, ValueError, TypeError, KeyError, ZeroDivisionError):
        pass

    # The destination is checked again at generate time; saying it now saves the
    # operator from configuring a whole run that will be refused at the end.
    od = q.get("out_dir")
    if od:
        if os.path.isdir(od) and os.path.isfile(os.path.join(od, "meta.json")):
            warnings.append("输出目录已有一份数据集，本次会自动改名到「%s_2」，不会覆盖"
                            % os.path.basename(os.path.normpath(od)))
        elif os.path.isdir(od) and os.listdir(od):
            errors.append("输出目录已存在且非空，也不是本工具生成的数据集，写入会被拒绝：%s"
                          % od)
        elif not _writable_ancestor(od):
            errors.append("输出目录所在位置不可写：%s" % od)

    est = _estimate_payload(q)
    if est["megabytes"] > 5000:
        errors.append("预计 %.0f GB，超出单次运行建议上限；请减少帧数或模态"
                      % (est["megabytes"] / 1024.0))
    elif est["megabytes"] > 1500:
        warnings.append("预计 %.0f GB，生成时间较长" % (est["megabytes"] / 1024.0))

    # ---- New contract: named, machine-readable checks. Older clients ignore
    # the extra key; the flat errors/warnings above keep their exact meaning.
    checks = {}

    def _check(key, ok, level, msg):
        checks[key] = dict(ok=bool(ok), level=level, message=msg)

    if not sensors:
        _check("sensors", False, "error", "至少选一个模态（sensors 为空或缺省）")
    else:
        _check("sensors", True, "ok", "已选 %d 个模态" % len(sensors))

    # size: 0x0 or out-of-range dimensions
    dim_bad = []
    lo, hi = LIM["dim"]
    for label, blk in (("窄", q.get("narrow")), ("宽", q.get("wide"))):
        if not isinstance(blk, dict):
            continue
        w, h = blk.get("width"), blk.get("height")
        if not w or not h or w <= 0 or h <= 0:
            dim_bad.append("%s视场为 0x0" % label)
        elif not (lo <= w <= hi and lo <= h <= hi):
            dim_bad.append("%s视场 %sx%s 超出 %s–%s" % (label, w, h, lo, hi))
    if dim_bad:
        msg = "；".join(dim_bad)
        errors.append("尺寸无效：" + msg)
        _check("size", False, "error", msg)
    else:
        _check("size", True, "ok", "尺寸在 %s–%s 范围内" % (lo, hi))

    # jitter: the stabilisation error must fit inside the narrow FOV
    fov_deg = float(q["narrow"]["fov_deg"])
    jit = q.get("jitter_deg")
    if jit is None:
        try:
            from . import scene as s_scene
            sc = s_scene.load_config(q["scene_path"])
            sk = (sc.get("clip") or {}).get("seeker") or {}
            jit = float(sk.get("los_jitter_deg", 0.0)) + float(sk.get("wander_deg", 0.0))
        except Exception:
            jit = 0.0
    jit = float(jit or 0.0)
    if jit > fov_deg:
        msg = "抖动 %.3f° 超过窄视场 %.3f°，目标会出画" % (jit, fov_deg)
        errors.append(msg)
        _check("jitter", False, "error", msg)
    elif jit > 0.5 * fov_deg:
        msg = "抖动 %.3f° 超过窄视场的一半 %.3f°" % (jit, fov_deg)
        warnings.append(msg)
        _check("jitter", True, "warn", msg)
    else:
        _check("jitter", True, "ok", "抖动在窄视场内")

    # radar window: the unambiguous range must cover 1.4 * slant range
    R = float((q.get("range_m") or [0.0, 8000.0])[1])
    r_max = None
    try:
        from . import scene as s_scene
        sc = s_scene.load_config(q["scene_path"])
        r_max = float(((sc.get("sensors") or {}).get("radar") or {}).get("r_max_m") or 0.0) or None
    except Exception:
        pass
    if r_max is not None and r_max < 1.4 * R:
        msg = ("雷达不模糊距离 %.0f m 小于 1.4×视线距离 %.0f m，远距目标会回绕"
               % (r_max, 1.4 * R))
        warnings.append(msg)
        _check("radar_window", False, "warn", msg)
    else:
        _check("radar_window", True, "ok", "雷达距离窗覆盖视线距离")

    return dict(errors=errors, warnings=warnings,
                ok=not errors, level=("error" if errors else
                                      "warn" if warnings else "ok"),
                estimate=est, checks=checks)


def _parse(body):
    """Validate the whole form and return a normalised dict. Raises BadRequest."""
    if not isinstance(body, dict):
        raise BadRequest("请求体必须是 JSON 对象")

    q = {}
    q["_present"] = set(body.keys())
    # Two ways in, on purpose: a saved recipe (build the scene from the form) and
    # a preset scene from scenes/ (overlay the form on a curated file). The
    # preset path stays because those 32 files encode engagement geometry that
    # took measurement to get right, and nobody should have to retype it.
    recipe = body.get("recipe")
    if isinstance(recipe, str) and recipe.strip():
        recipe = _load_recipe(recipe.strip())
    if isinstance(recipe, dict):
        from . import scenegen
        # Workbench switches arrive as flat form fields but belong in the recipe,
        # so fold them in before validation -- a recipe that is re-saved then
        # carries the same picture.
        rec = json.loads(json.dumps(recipe))
        fx = dict(rec.get("effects") or {})
        if body.get("plume") not in (None, ""):
            fx["plume"] = _bool(body, "plume", "生成尾焰")
        if body.get("supersample") not in (None, ""):
            fx["supersample"] = _inum(body, "supersample", "红外超采样", (0, 4), 0)
        if fx:
            rec["effects"] = fx
        try:
            q["recipe"] = scenegen.validate_recipe(rec)
            built = scenegen.build_scene(q["recipe"])
        except ValueError as exc:
            raise BadRequest("配方无效：" + str(exc)) from exc
        # A recipe is the source of truth for a recipe-originated run, but the
        # rest of _parse reads the flat form. Fold the recipe into the form
        # first, without overwriting anything the caller sent explicitly, or a
        # recipe asking for 20 frames would silently generate 240.
        for k, v in _recipe_to_form(q["recipe"]).items():
            if body.get(k) in (None, "", []) and v is not None:
                body[k] = v
        q["scene_path"] = _write_tmp_scene(built, q["recipe"].get("name") or "recipe")
        q["scene"] = _safe_name(q["recipe"].get("name")) or "recipe"
    else:
        q["recipe"] = None
        q["scene_path"] = _scene_file(body.get("scene") or body.get("preset"))
        q["scene"] = os.path.basename(q["scene_path"])[:-5]
    q["name"] = _safe_name(body.get("name"))
    # Optional switches: absent means "leave the scene alone".  The recipe path
    # already folded a sent value into recipe.effects, so these only matter for
    # the preset-scene path and for a preview taken before a recipe was saved.
    q["plume"] = (_bool(body, "plume", "生成尾焰")
                  if body.get("plume") not in (None, "") else None)
    q["supersample"] = (_inum(body, "supersample", "红外超采样", (0, 4), 0)
                        if body.get("supersample") not in (None, "") else None)
    q["frames"] = _inum(body, "frames", "帧数", LIM["frames"], 240)
    q["fps"] = _fnum(body, "fps", "帧率", LIM["fps"], 10.0)
    q["fpc"] = _inum(body, "frames_per_clip", "每段帧数", LIM["fpc"], 120)
    q["fpc"] = max(1, min(q["fpc"], q["frames"]))
    q["seed"] = _inum(body, "seed", "随机种子", LIM["seed"], 7)
    # Parallel frame ranges. 1 keeps the old single-process path; the UI fills
    # this from the benchmark's recommendation, and the operator can override it.
    q["workers"] = _inum(body, "workers", "并行度", (1, 16), 1)
    q["weather"] = _choice(body, "weather", "天气", WEATHER_MODES, "mixed")
    from . import interference
    mode = _choice(body, "interference_mode", "干扰增广",
                   {"inherit": 1, "off": 1, "active": 1, "passive": 1, "mixed": 1}, "inherit")
    try:
        q["interference"] = (interference.validate(body["interference"]) if "interference" in body else
                             interference.preset(mode) if mode != "inherit" else None)
    except ValueError as exc:
        raise BadRequest("干扰配置无效："+str(exc)) from exc

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
    # Aspect is the operator-facing word for what the geometry actually is:
    # whether the target comes at the camera or runs away from it. Closing rate
    # alone does not say this -- a receding target can still be filmed head-on.
    # Three values, matching the three buttons in the view selector. look_down is
    # the surface engagement; top_down is what the background-placement editor
    # sends, because clicking a pixel on a plate only means anything under a
    # straight-down camera.
    q["aspect"] = _choice(body, "aspect", "拍摄方位",
                          {"": 1, "head_on": 1, "pursuit": 1,
                           "look_down": 1, "top_down": 1}, "")
    # Where the dataset lands. Blank keeps the default out/studio_runs/<id>.
    q["out_dir"] = _safe_out_dir(body.get("output_dir"))
    raw_ecm = body.get("ecm")
    q["ecm"] = None
    if raw_ecm not in (None, "", [], {}):
        from . import ecm as s_ecm
        ents = raw_ecm.get("entities") if isinstance(raw_ecm, dict) else raw_ecm
        if ents:
            try:
                q["ecm"] = s_ecm.load(dict(enabled=True, entities=ents))
            except (ValueError, KeyError, TypeError) as exc:
                raise BadRequest("干扰实体无效：" + str(exc)) from exc

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

    # The IR exhaust switch is the one form control with no counterpart in a
    # curated scene file, so it is applied here rather than through `changed`.
    # None means "the form did not send it": a preset scene then keeps whatever
    # its author wrote, and a recipe run already carries it in ir.engine_heat.
    fx = (q.get("recipe") or {}).get("effects") or {}
    plume = q.get("plume")
    if plume is None:
        plume = fx.get("plume")
    if plume is not None:
        cfg.setdefault("ir", {})["engine_heat"] = bool(plume)
    supersample = q.get("supersample")
    if supersample is None:
        supersample = fx.get("supersample")
    if supersample and 2 <= int(supersample) <= 4:
        ir_block = cfg.setdefault("ir", {})
        ir_block["samples_per_axis"] = int(supersample)
        ir_block["subpixel_mode"] = "coverage"

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
    # A curated scene that already drives its own camera (observer_trajectory or
    # trajectory) must not get a seeker block injected from the form's default:
    # the planner would then have two motion sources and the run would stop
    # matching the scene it claims to be. Only an explicit traj_mode overrides.
    scene_drives_camera = bool(clip.get("observer_trajectory") or clip.get("trajectory"))
    # A plate view is one static camera over one photograph. Re-adding the form's
    # default seeker here put a moving platform behind the plate: the camera was
    # no longer where the placement maths put it, so a target clicked at
    # (0.32, 0.30) rendered at (0.49, 0.47).
    _plate_view = str(q.get("aspect") or "") == "top_down"
    if scene_drives_camera and "traj_mode" not in q["_present"]:
        clip.pop("seeker", None)
        cfg.pop("seeker", None)
    elif _plate_view or q["traj_mode"] == "world":
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
        # The view selector's choice has to reach the geometry, not just the
        # form: a hand-authored scene carries its own seeker block with no
        # aspect, so picking 迎头 in the workbench used to change nothing at all.
        if q.get("aspect"):
            sk["aspect"] = str(q["aspect"])
            sk["closing"] = "out" if str(q["aspect"]) == "pursuit" else "in"
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
        and _near(q["narrow"]["width"], ref_n.get("width", q["narrow"]["width"]))
        and _near(q["narrow"]["height"], ref_n.get("height", q["narrow"]["height"])))
    same_wide = (not lens_given) or (
        _near(q["wide"]["fov_deg"], ref_w.get("fov_deg", q["wide"]["fov_deg"]))
        and _near(q["wide"]["width"], ref_w.get("width", q["wide"]["width"]))
        and _near(q["wide"]["height"], ref_w.get("height", q["wide"]["height"])))
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
            # Single view drops clip.views, so an explicit lens (especially a
            # height-only edit, which same_narrow now detects) must also reach
            # the legacy camera blocks or the size would silently vanish.
            if lens_given:
                for name in ("ir_camera", "optical", "eo_camera", "camera"):
                    cam = cfg.setdefault("sensors", {}).get(name)
                    if isinstance(cam, dict):
                        cam["fov_deg"] = float(q["narrow"]["fov_deg"])
                        cam["width"] = int(q["narrow"]["width"])
                        cam["height"] = int(q["narrow"]["height"])

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
    # Interference augmentation: preset or fully specified events, validated in
    # _parse. Without this the form's choice never reached the scene.
    if q.get("interference") is not None:
        cfg["interference"] = q["interference"]
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
        # Parallel runs own several children; cancel has to reach all of them.
        self.procs = []
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




def _spawn_parts(job, base_argv, merge_argv, workers, sensors):
    """Render one dataset as N concurrent frame ranges, then merge them.

    The engine already speaks "gen --part i/K" plus "merge", so a dataset can be
    cut into independent frame ranges without touching the renderer. Each part
    writes its own pass files into the same output directory.

    Every part gets its own reader thread: a pipe nobody drains fills up and
    deadlocks the child, which is the classic way this kind of fan-out hangs
    halfway through a batch.
    """
    def run():
        job.state = "running"
        try:
            env = dict(os.environ)
            env["PYTHONPATH"] = ROOT
            env.setdefault("PYTHONIOENCODING", "utf-8")
            # cli.py splits the range the same way: the remainder goes to the
            # first chunks, so each part knows its global frame offset.
            total = max(1, int(job.total))
            per = [total // workers + (1 if i < total % workers else 0)
                   for i in range(workers)]
            offs = [sum(per[:i]) for i in range(workers)]
            procs = []
            for i in range(workers):
                cmd = list(base_argv) + ["--part", "%d/%d" % (i, workers)]
                p = subprocess.Popen(cmd, cwd=ROOT, env=env,
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, bufsize=0)
                procs.append(p)
            # Keep every child, not just the first: a cancel that kills only one
            # part leaves the rest rendering and the merge never runs.
            job.procs = procs
            job.push("[studio] 并行 %d 路分帧渲染" % workers)

            def pump(part_idx):
                """Relay a part's log with its frame numbers made global.

                Each child reports "frame 1/2" against its own range; pushing
                that verbatim overwrote job.total with the chunk size and the
                UI showed "4 / 2".
                """
                off = offs[part_idx]
                for raw in procs[part_idx].stdout:
                    line = _decode(raw)
                    m = _RE_PROG.search(line)
                    if m:
                        line = _RE_PROG.sub(
                            "frame %d/%d" % (off + int(m.group(1)), total), line, count=1)
                    job.push(line)

            readers = [threading.Thread(target=pump, args=(i,), daemon=True)
                       for i in range(len(procs))]
            for t in readers:
                t.start()
            job.proc = procs[0] if procs else None
            codes = [p.wait() for p in procs]
            for t in readers:
                t.join(timeout=5)
            bad = [(i, rc) for i, rc in enumerate(codes) if rc != 0]
            if job.state == "cancelled":
                return
            if bad:
                job.state = "failed"
                job.error = "分帧渲染失败：" + "，".join(
                    "第 %d 路退出码 %d" % (i, rc) for i, rc in bad)
                return
            job.push("[studio] 合并分块")
            m = subprocess.Popen(list(merge_argv), cwd=ROOT, env=env,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, bufsize=0)
            job.proc = m
            for raw in m.stdout:
                job.push(_decode(raw))
            rc = m.wait()
            if job.state == "cancelled":
                return
            if rc == 0:
                # merge_parts leaves the per-part JSON behind; on disk they look
                # like stale annotations and quietly accumulate.
                try:
                    for fn in os.listdir(job.out):
                        if ".part" in fn and fn.endswith(".json"):
                            os.remove(os.path.join(job.out, fn))
                except OSError:
                    pass
                job.done = job.total
                job.state = "done"
            else:
                job.state = "failed"
                job.error = "合并退出码 %d" % rc
        except Exception as exc:                      # pragma: no cover
            job.state = "failed"
            job.error = "%s: %s" % (type(exc).__name__, exc)
            job.log.append(traceback.format_exc()[-800:])
        finally:
            job.finished = time.time()

    t = threading.Thread(target=run, daemon=True)
    t.start()


def _remember_out_dir(path):
    """Remember a folder the operator actually generated into.

    The picker should offer the folders that are in use, not every directory on
    the disk; the list is small, JSON and lives next to the recipes.
    """
    try:
        p = os.path.normpath(os.path.abspath(path))
        known = _load_known_out_dirs()
        if p not in known:
            known.insert(0, p)
        known = known[:20]
        os.makedirs(os.path.dirname(_KNOWN_OUT), exist_ok=True)
        with open(_KNOWN_OUT, "w", encoding="utf-8") as fh:
            json.dump(known, fh, ensure_ascii=False, indent=1)
    except OSError:
        pass


def _load_known_out_dirs():
    try:
        with open(_KNOWN_OUT, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list):
            return [str(x) for x in data if isinstance(x, str)]
    except (OSError, ValueError):
        pass
    return []

def _place_preview(body):
    """Render one fast frame of the form exactly as it is now configured.

    This has to go through the same _parse/_compose path a real run uses. The
    first version built its own little recipe from a category and a single
    plate coordinate, so the preview genuinely was a different scene: it ignored
    every extra target, ignored the drawn routes, and mapped the sky category to
    a ground scene. The operator saw one thing and generation produced another.

    Only resolution and sample count are reduced; nothing else is approximated.
    """
    q = _parse(body)
    cfg = _compose(q, _load_scene(q["scene_path"]))
    nr = q["narrow"]
    ratio = (float(nr["height"]) / float(nr["width"])) if nr["width"] else 0.75
    # New contract: an explicit preview size wins. Without it the legacy
    # "width" key still drives the fast frame, but the height now follows the
    # narrow view's real aspect instead of a hard-coded 4:3.
    pw, ph = body.get("preview_width"), body.get("preview_height")
    if pw not in (None, ""):
        base_w = max(1, int(pw))
        base_h = (max(1, int(ph)) if ph not in (None, "")
                  else max(1, int(round(base_w * ratio))))
    elif ph not in (None, ""):
        base_h = max(1, int(ph))
        base_w = max(1, int(round(base_h / ratio))) if ratio else base_h
    else:
        base_w = max(160, min(960, int(body.get("width") or 320)))
        base_h = max(1, int(round(base_w * ratio)))
    views = ((cfg.get("clip") or {}).get("views") or [])
    if views:
        # Dual view keeps each view's own aspect and relative size; never
        # flatten every view onto the narrow 4:3 frame.
        scale = base_w / (float(nr["width"]) or 1.0)
        for v in views:
            vw = float(v.get("width") or nr["width"]) or 1.0
            vh = float(v.get("height") or nr["height"]) or 1.0
            v["width"] = max(1, int(round(vw * scale)))
            v["height"] = max(1, int(round(vh * scale)))
    else:
        # legacy path: dataset.py reads the size from the camera block
        for key in ("optical", "ir_camera"):
            blk = (cfg.get("sensors") or {}).get(key)
            if isinstance(blk, dict):
                blk["width"], blk["height"] = base_w, base_h
    cfg.setdefault("optical", {})["samples"] = int(body.get("samples") or 12)
    _abs_models(cfg)
    sensors = [s for s in q["sensors"] if s in ("optical", "ir", "radar", "sar", "lidar")]
    sensor = sensors[0] if sensors else "optical"
    out_dir = os.path.join(RUNS, "_place_preview")
    if os.path.isdir(out_dir):
        for nm in os.listdir(out_dir):
            fp = os.path.join(out_dir, nm)
            if os.path.isfile(fp):
                try:
                    os.remove(fp)
                except OSError:
                    pass
    os.makedirs(out_dir, exist_ok=True)
    cfg_path = os.path.join(out_dir, "scene.json")
    with open(cfg_path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, ensure_ascii=False, indent=1)
    argv = [sys.executable, "-m", "simkit", "gen", "-c", cfg_path,
            "-o", out_dir, "-n", "1", "-s", sensor,
            "--seed", str(int(q["seed"]))]
    env = dict(os.environ)
    env["PYTHONPATH"] = ROOT
    env.setdefault("PYTHONIOENCODING", "utf-8")
    t0 = time.time()
    proc = subprocess.run(argv, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)
    if proc.returncode != 0:
        tail = (proc.stderr or b"").decode("utf-8", "replace").strip().splitlines()
        raise BadRequest("预览渲染失败：" + " | ".join(tail[-3:]))
    sub = {"ir": "ir_mwir", "optical": "optical"}.get(sensor, sensor)
    rel = "images/%s/000000.png" % sub
    if not os.path.isfile(os.path.join(out_dir, rel)):
        boom = [nm for nm in os.listdir(os.path.join(out_dir, "images"))
                if os.path.isdir(os.path.join(out_dir, "images", nm))]
        if boom:
            rel = "images/%s/000000.png" % sorted(boom)[0]
    if not os.path.isfile(os.path.join(out_dir, rel)):
        raise BadRequest("预览没有产出图像：%s" % rel)
    return dict(ok=True, url="/api/img/_place_preview/" + rel,
                seconds=round(time.time() - t0, 2), sensor=sensor,
                size=[width, height],
                targets=len([o for o in cfg.get("objects") or []
                             if o.get("label", True)]))

def start_job(body, preview=False, still=False):
    q = _parse(body)
    cfg = _compose(q, _load_scene(q["scene_path"]))
    stamp = time.strftime("%Y%m%d_%H%M%S")
    tag = "still_" if still else ("preview_" if preview else "ds_")
    jid = "%s%s_%s" % (tag, stamp, q["scene"])
    if q["name"]:
        jid += "_" + q["name"]
    jid = _safe_name(jid, "run")
    if q.get("out_dir"):
        # The operator chose the folder. A preview must not overwrite the full
        # run they are about to make, so previews get their own suffix.
        wanted = q["out_dir"]
        if preview:
            wanted = wanted + "_preview"
        out_dir = _unique_out_dir(wanted)
    else:
        out_dir = os.path.normpath(os.path.join(RUNS, jid))
        if os.path.dirname(out_dir) != os.path.normpath(RUNS):
            raise BadRequest("输出目录越界：%s" % out_dir)
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError as exc:
        raise BadRequest("无法创建输出目录 %s：%s" % (out_dir, exc)) from exc
    if q.get("out_dir"):
        _remember_out_dir(out_dir)
    cfg_path = os.path.join(out_dir, "scene.json")
    with open(cfg_path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, ensure_ascii=False, indent=1)

    frames = q["frames"]
    if still:
        frames = 1
    elif preview:
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
    workers = 1 if (still or preview) else int(q.get("workers") or 1)
    workers = max(1, min(workers, max(1, frames)))
    if workers > 1:
        merge_argv = [sys.executable, "-m", "simkit", "merge", "-c", cfg_path,
                      "-o", out_dir, "-n", str(frames),
                      "-s", ",".join(sensors) or "ir",
                      "--seed", str(int(q["seed"]))]
        _spawn_parts(job, argv, merge_argv, workers, sensors)
    else:
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

    # Optics contract: per-view angular sampling and ground sample distance.
    # fov is radians inside both formulas so the units are physical (mrad and
    # metres per pixel). R is the far slant range, the honest worst case.
    rng = q.get("range_m")
    R = float(rng[1]) if rng else 8000.0
    tx_lo, tx_hi = float(q["target_px"][0]), float(q["target_px"][1])
    optics = []
    for v in views:
        fov_rad = math.radians(float(v["fov_deg"]))
        w = float(v["width"]) or 1.0
        optics.append(dict(
            name=v["name"], width=v["width"], height=v["height"],
            ifov_mrad=round(1000.0 * fov_rad / w, 6),
            gsd_m_per_px=round(2.0 * R * math.tan(fov_rad / 2.0) / w, 6),
            target_px_lo=tx_lo, target_px_hi=tx_hi))

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
        clips=rows, views=views, optics=optics,
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
          "radar": "雷达 距离-多普勒", "radar_clean": "雷达 距离-多普勒 · 纯净",
          "radar_ra_clean": "雷达 距离-角度 · 纯净",
          "lidar": "激光雷达距离", "lidar_intensity": "激光雷达强度"}


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
    if isinstance(boxes, dict):
        # The optical/IR labels key boxes by object name; only the radar labels
        # use a list. Indexing [0] on a dict raised KeyError and the whole
        # gallery request died with a closed connection.
        boxes = list(boxes.values())
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


def _variant_dirs(base):
    """Interference variants live in subdirectories that carry their own images.

    "clean" is the dataset itself; any child directory with an images/ folder is
    a variant an interference export produced.
    """
    out = []
    try:
        for nm in sorted(os.listdir(base)):
            d = os.path.join(base, nm)
            if os.path.isdir(d) and os.path.isdir(os.path.join(d, "images")):
                out.append(nm)
    except OSError:
        pass
    return out


def _dataset_info(jid, variant=None):
    base = _run_dir(jid)
    if not base:
        return None
    variants = ["clean"] + _variant_dirs(base)
    chosen = variant if variant in variants else "clean"
    prefix = "" if chosen == "clean" else chosen + "/"
    vroot = base if chosen == "clean" else os.path.join(base, chosen)
    info = dict(id=jid, dir=base, variant=chosen, variants=variants,
                stats={}, meta={}, samples={}, cards=[], views=[])
    for nm, key in (("stats.json", "stats"), ("meta.json", "meta")):
        fp = os.path.join(vroot, nm)
        if os.path.exists(fp):
            try:
                with open(fp, "r", encoding="utf-8") as fh:
                    info[key] = json.load(fh)
            except (OSError, ValueError):
                pass
    seen = set()
    order = []
    ldirs = _label_dirs(vroot)
    img_root = os.path.join(vroot, "images")
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
                mod=mod, label=label, file=fn,
                url="/api/img/%s/%s%s/%s" % (jid, prefix, rel, fn),
                w=w, h=h, frame=_frame_no(fn), target_px=px, box=box, ratio=ratio,
                range_m=meta.get("range_m"), clip=meta.get("clip"),
                fov_deg=meta.get("fov_deg"), peak_db=meta.get("peak_db"),
                snr_db=meta.get("snr_db"), res_range_m=meta.get("res_range_m"),
                res_az_m=meta.get("res_az_m"), swath_m=meta.get("swath_m")))
    info["views"] = ((info["meta"].get("views") if isinstance(info["meta"], dict) else None)
                     or (info["stats"].get("views") if isinstance(info["stats"], dict) else None)
                     or [])
    return info


# ------------------------------------------------------------------ alignment

# One page of the alignment table. A 20 000-frame run would otherwise make the
# browser read every per-frame label before showing anything; the panel asks for
# a window and reports the range it actually covered. The cap is the server's,
# not the UI's, because the query string is user input.
ALIGN_PAGE_MAX = 120


def _json_file(path):
    """JSON or None. A half-written label file must not break the whole table."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _box_rect(box):
    """One labels/*.json box -> pixel centre/size, or None if it is malformed.

    The panel compares three modalities on one row, so the box has to arrive in
    the same units the image is in: pixels, top-left origin. Nothing here
    re-projects or second-guesses the renderer.
    """
    if not isinstance(box, dict):
        return None
    try:
        x0, y0, x1, y1 = (float(box[k]) for k in ("x0", "y0", "x1", "y1"))
    except (KeyError, TypeError, ValueError):
        return None
    px = box.get("pixels")
    px = px if isinstance(px, (int, float)) else None
    visible = box.get("visible") is not False and (x1 > x0) and (y1 > y0)
    # A box with zero measured pixels is a projection that missed the body, not
    # a detection; counting it would make "three modalities agree" meaningless.
    if visible and px is not None and px <= 0:
        visible = False
    return dict(center=[round((x0 + x1) / 2.0, 1), round((y0 + y1) / 2.0, 1)],
                size=[round(x1 - x0, 1), round(y1 - y0, 1)],
                px=px, visible=visible)


def _num_or_none(value, digits=3):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return round(float(value), digits)


def _radar_target(target):
    """One radar truth/detection row from labels/radar or annotations_radar."""
    if not isinstance(target, dict):
        return None
    name = target.get("name")
    if not isinstance(name, str) or not name:
        return None
    return dict(name=name, cls=target.get("cls"),
                range_m=_num_or_none(target.get("range")),
                v_r=_num_or_none(target.get("v_r")),
                az_deg=_num_or_none(target.get("az")),
                el_deg=_num_or_none(target.get("el")),
                rcs_dbsm=_num_or_none(target.get("rcs_dbsm")),
                detected=target.get("detected") is True,
                measured_range_m=_num_or_none(target.get("measured_range")),
                measured_v_r=_num_or_none(target.get("measured_v_r")),
                snr_db=_num_or_none(target.get("snr_db")))


def _radar_pairs(jid, base, frames):
    """Frames that have both the annotated and the clean range-Doppler figure.

    Both trees are written by the radar renderer. Pairing them by frame number
    instead of by filename keeps the two captions on the same physical frame
    even if a future renderer changes the naming scheme.
    """
    rdir = os.path.join(base, "images", "radar")
    cdir = os.path.join(base, "images", "radar_clean")
    if not (os.path.isdir(rdir) and os.path.isdir(cdir)):
        return []
    labeled, clean = {}, {}
    for fn in sorted(os.listdir(rdir)):
        if fn.lower().endswith(".png"):
            n = _frame_no(fn)
            if n is not None:
                labeled.setdefault(n, fn)
    for fn in sorted(os.listdir(cdir)):
        if fn.lower().endswith(".png"):
            n = _frame_no(fn)
            if n is not None:
                clean.setdefault(n, fn)
    out = []
    for frame in frames:
        lf, cf = labeled.get(frame), clean.get(frame)
        if not lf or not cf:
            continue
        w, h = _png_size(os.path.join(rdir, lf))
        out.append(dict(frame=frame, file=lf,
                        labeled="/api/img/%s/images/radar/%s" % (jid, lf),
                        clean="/api/img/%s/images/radar_clean/%s" % (jid, cf),
                        w=w, h=h))
    return out


def _align_payload(jid, start=0, count=20):
    """Per-frame, per-target optical/IR/radar agreement, straight from labels.

    This is a read-only view of what the run already wrote: the COCO/JSON boxes
    the renderer produced, not a re-detection and not a re-render. A target is
    counted as detected per modality when its own label says it is visible (or,
    for radar, when the truth target is marked detected); "all three" therefore
    means the same frame actually carries all three observations.
    """
    base = _run_dir(jid)
    if not base:
        return None
    meta = _json_file(os.path.join(base, "meta.json")) or {}
    stats = _json_file(os.path.join(base, "stats.json")) or {}
    ldirs = _label_dirs(base)
    opt_dirs = sorted(d for d in ldirs
                      if d == "optical" or d.startswith("optical_"))
    ir_dirs = sorted(d for d in ldirs
                     if d.startswith("ir_") and not d.startswith("ir_preview"))
    radar_dir = ldirs.get("radar")

    ann = _json_file(os.path.join(base, "annotations_radar.json")) or {}
    # annotations_radar.json keys its images by COCO image id, while the label
    # files are named by frame number. They happen to coincide in today's
    # writer; reading the frame out of file_name keeps the table correct if a
    # future export starts the image ids at 1.
    ann_imgs, ann_names, ann_id_frame = {}, {}, {}
    for item in ann.get("images") or []:
        if not isinstance(item, dict):
            continue
        fid = item.get("id")
        frame = _frame_no(item.get("file_name") or "")
        if frame is None and isinstance(fid, int):
            frame = fid
        if frame is None:
            continue
        ann_imgs[frame] = item
        if isinstance(fid, int):
            ann_id_frame[fid] = frame
    for box in ann.get("annotations") or []:
        if not isinstance(box, dict):
            continue
        frame = ann_id_frame.get(box.get("image_id"))
        if frame is not None and isinstance(box.get("object_name"), str):
            ann_names.setdefault(frame, set()).add(box["object_name"])

    frames_total = int(meta.get("frames") or stats.get("frames") or 0)
    if frames_total <= 0:
        found = set()
        for sub in ([radar_dir] if radar_dir else []) + [ldirs[d] for d in opt_dirs]:
            try:
                names = os.listdir(sub)
            except OSError:
                continue
            for fn in names:
                n = _frame_no(fn)
                if n is not None:
                    found.add(n)
        frames_total = (max(found) + 1) if found else 0
    frames_total = max(0, frames_total)
    start = max(0, min(int(start), max(frames_total - 1, 0)))
    end = min(frames_total, start + max(1, int(count)))

    def label(ldir, frame):
        return _json_file(os.path.join(ldir, "%06d.json" % frame)) or {}

    models = {}
    frames_out = []
    totals = dict(rows=0, all3=0, optical=0, ir=0, radar=0)
    for frame in range(start, end):
        rad_by_name = {}
        t_val = None
        if radar_dir:
            rad = label(radar_dir, frame)
            t_val = _num_or_none(rad.get("t"), 6)
            for item in rad.get("targets") or []:
                row = _radar_target(item)
                if row:
                    rad_by_name[row["name"]] = row
        if not rad_by_name:
            # Older runs can lack a per-frame radar label but still carry the
            # radar COCO file; the truth targets there are the same structure.
            for item in (ann_imgs.get(frame) or {}).get("targets") or []:
                row = _radar_target(item)
                if row:
                    rad_by_name[row["name"]] = row

        opt_by_name, ir_by_name = {}, {}
        for sub in opt_dirs:
            d = label(ldirs[sub], frame)
            for name, box in (d.get("boxes") or {}).items():
                rect = _box_rect(box)
                if rect is None:
                    continue
                rect["view"] = d.get("view") or sub
                opt_by_name.setdefault(name, []).append(rect)
        for sub in ir_dirs:
            d = label(ldirs[sub], frame)
            for obj in d.get("objects") or []:
                if isinstance(obj, dict) and isinstance(obj.get("name"), str):
                    models.setdefault(obj["name"], obj.get("cls"))
            for name, box in (d.get("boxes") or {}).items():
                rect = _box_rect(box)
                if rect is None:
                    continue
                rect["band"] = d.get("band") or sub[len("ir_"):] or sub
                ir_by_name.setdefault(name, []).append(rect)

        rows = []
        names = set(opt_by_name) | set(ir_by_name) | set(rad_by_name)
        for name in sorted(names):
            opt = opt_by_name.get(name, [])
            ir = ir_by_name.get(name, [])
            rad = rad_by_name.get(name)
            det = dict(optical=any(b["visible"] for b in opt),
                       ir=any(b["visible"] for b in ir),
                       radar=bool(rad and rad["detected"]))
            row = dict(name=name, cls=models.get(name) or (rad or {}).get("cls"),
                       optical=opt, ir=ir, radar=rad, detected=det,
                       all3=det["optical"] and det["ir"] and det["radar"],
                       rd_annotation=name in ann_names.get(frame, set()))
            rows.append(row)
            totals["rows"] += 1
            totals["all3"] += 1 if row["all3"] else 0
            for key in ("optical", "ir", "radar"):
                totals[key] += 1 if det[key] else 0
        frames_out.append(dict(frame=frame, t=t_val, rows=rows))

    objects = sorted({row["name"] for fr in frames_out for row in fr["rows"]})
    notes = []
    if not opt_dirs:
        notes.append("这次运行没有 labels/optical：光学一列为空。")
    if not ir_dirs:
        notes.append("这次运行没有 labels/ir_*：红外一列为空。")
    if not radar_dir:
        notes.append("这次运行没有 labels/radar：雷达一列为空。")
    if not frames_out:
        notes.append("没有可读的逐帧标签：可能仍在生成，或这次运行没有写 labels/。")
    return dict(
        id=jid, dir=base, meta=dict(
            frames=frames_total, sensors=meta.get("sensors") or [],
            views=meta.get("views") or [], ir_bands=meta.get("ir_bands") or [],
            radar_band=meta.get("radar_band"),
            categories=meta.get("categories") or []),
        frames_total=frames_total, start=start, end=end,
        shown=len(frames_out),
        modalities=dict(optical=opt_dirs,
                        ir=[d[len("ir_"):] for d in ir_dirs],
                        radar=bool(radar_dir)),
        objects=[dict(name=nm, cls=models.get(nm)) for nm in objects],
        frames=frames_out, totals=totals,
        annotations=dict(frames=len(ann_imgs),
                         boxes=sum(len(v) for v in ann_names.values())),
        radar_pairs=_radar_pairs(jid, base, list(range(start, end))),
        notes=notes)


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

    def _reset_request_state(self):
        """Drop everything cached for the previous request on this connection."""
        self._cached_body = None
        self._cached_error = None

    def _body(self):
        """Read and cache the request body.

        Cached because every branch must consume it: a POST that returns without
        reading leaves the bytes in the socket, and on a keep-alive connection
        the next request line is parsed from them ("501 Unsupported method
        ('{}GET')" -- measured). Caching also lets drain() run at the top of
        do_POST without the handlers seeing an empty body.
        """
        err = getattr(self, "_cached_error", None)
        if err is not None:
            # A body that failed to parse must keep failing: _drain_body swallows
            # the first raise so the socket is still consumed, and the handler
            # that runs afterwards has to see the 400 rather than an empty dict.
            raise err
        if getattr(self, "_cached_body", None) is not None:
            return self._cached_body
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            self._cached_body = {}
            return {}
        if n > 1 << 20:
            self.rfile.read(n)          # drain before refusing
            raise BadRequest("请求体过大（%d 字节）" % n)
        raw = self.rfile.read(n)
        try:
            self._cached_body = json.loads(raw.decode("utf-8"))
        except ValueError:
            self._cached_body = {}
            self._cached_error = BadRequest("请求体不是合法 JSON")
            raise self._cached_error
        return self._cached_body

    def _drain_body(self):
        try:
            self._body()
        except BadRequest:
            pass

    def _get_presets(self):
        # The form's route check must quote the same airframe limits the motion
        # checker enforces, or the warning and the run log disagree. Read them
        # from motion.ENVELOPES instead of keeping a second table in JS.
        from . import motion as _motion
        return dict(
            scenes=_scenes(), presets=_scenes(),
            weather=[dict(key=k, label=WEATHER_LABELS.get(k, k)) for k in WEATHER_MODES],
            sensors=SENSORS, sensor_keys=SENSOR_KEYS,
            sar_bands=[dict(key=k, fc_ghz=v["fc_ghz"], note=v["note"])
                       for k, v in SAR_BANDS.items()],
            radar_bands=["X", "Ku", "K", "Ka", "W", "W94"],
            views=dict(narrow=VIEW_NARROW, wide=VIEW_WIDE),
            size_px=SIZE_PX, closing=CLOSING, traj_modes=TRAJ_MODES,
            limits=LIM, runs=RUNS,
            envelopes=_motion.ENVELOPES,
            envelope_by_class=_motion.ENVELOPE_BY_CLASS,
            # The form reads these from the presets payload. They were lost when
            # this block was restored from an older revision, and the visible
            # damage was severe: without ecm.kinds the interference panel
            # rendered "table not loaded" and no entity could be added at all.
            ecm=_ecm_kinds_payload(),
            env_kinds=[dict(key=k, label=v) for k, v in
                       __import__("simkit.scenegen", fromlist=["x"]).ENV_LABELS.items()],
            aspects=[dict(key="head_on", label="迎头", domain="air"),
                     dict(key="pursuit", label="追击", domain="air"),
                     dict(key="look_down", label="俯视", domain="surface")],
            cost=dict(sec=COST_SEC_PER_FRAME, mb=COST_MB_PER_FRAME),
            default_out=RUNS, project_root=ROOT, recipes=_list_recipes(),
            # Which target classes stand on the ground: the view selector greys
            # out 俯视 for an aircraft and 迎头/追击 for a tank.
            surface_cls=sorted(__import__("simkit.scenegen", fromlist=["x"]).SURFACE_CLS))

    def do_GET(self):
        # The Handler instance is reused across keep-alive requests, so any
        # per-request cache has to be cleared here. Leaving _cached_body set made
        # the second POST on one socket replay the first body while its own bytes
        # stayed in the buffer, and the request after it was parsed as garbage
        # (measured: 501 Unsupported method '{"frames":7...}GET').
        self._reset_request_state()
        # GET used to run without a catch: a BadRequest from a helper (a bad
        # output path, a malformed dataset id) escaped to socketserver, the
        # connection closed with NO response, and the browser's ajax.onerror
        # turned the whole page into "backend offline". A 400 is infinitely
        # more useful than a dropped socket.
        try:
            return self._get()
        except BadRequest as exc:
            return self._send(400, dict(error=str(exc)))
        except Exception as exc:
            return self._send(500, dict(error="%s: %s" % (type(exc).__name__, exc),
                                        trace=traceback.format_exc()[-1200:]))

    def _get(self):
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
        if p == "/api/recipes":
            return self._send(200, dict(recipes=_list_recipes()))
        if p == "/api/ecm-kinds":
            return self._send(200, _ecm_kinds_payload())
        if p == "/api/assets":
            return self._send(200, _assets_payload())
        if p == "/api/backdrops":
            return self._send(200, _backdrops_payload())
        if p.startswith("/api/backdrop/"):
            rest = p[len("/api/backdrop/"):]
            cat, _, fn = rest.partition("/")
            from . import scenegen as _sg
            if cat not in _sg.BACKDROP_CATEGORIES or not fn:
                return self._send(404, dict(error="没有这个背景"))
            fp = _safe_join(os.path.join(_sg.BACKDROP_DIR, cat), fn)
            if not fp or not os.path.isfile(fp):
                return self._send(404, dict(error="没有这个背景文件"))
            ext = os.path.splitext(fp)[1].lower()
            ctype = {".png": "image/png", ".jpg": "image/jpeg",
                     ".jpeg": "image/jpeg"}.get(ext, "application/octet-stream")
            with open(fp, "rb") as fh:
                return self._send(200, fh.read(), ctype)
        if p == "/api/browse":
            raw = (parse_qs(u.query).get("path") or [""])[0]
            return self._send(200, _browse_payload(raw))
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
        if p.startswith("/api/align/"):
            jid = p[len("/api/align/"):].strip("/")
            q = parse_qs(u.query)
            try:
                start = int((q.get("from") or ["0"])[0])
                count = int((q.get("frames") or ["20"])[0])
            except (TypeError, ValueError):
                return self._send(400, dict(error="from / frames 必须是整数"))
            data = _align_payload(jid, start, max(1, min(count, ALIGN_PAGE_MAX)))
            if data is None:
                return self._send(404, dict(error="没有这个数据集"))
            return self._send(200, data)
        if p.startswith("/api/dataset/"):
            jid = p[len("/api/dataset/"):].strip("/")
            info = _dataset_info(jid, (parse_qs(u.query).get("variant") or [None])[0])
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
        self._reset_request_state()
        u = urlparse(self.path)
        # Consume the body before any branch can return early, so a keep-alive
        # connection is never left with unread bytes in its buffer.
        self._drain_body()
        try:
            if u.path == "/api/plan":
                q = _parse(self._body())
                return self._send(200, _plan_payload(
                    q, _compose(q, _load_scene(q["scene_path"]))))
            if u.path == "/api/preflight":
                q = _parse(self._body())
                return self._send(200, _preflight_payload(q))
            if u.path == "/api/estimate":
                q = _parse(self._body())
                return self._send(200, _estimate_payload(q))
            if u.path == "/api/trajectory/smooth":
                return self._send(200, _trajectory_smooth(self._body()))
            if u.path == "/api/still":
                # One frame is the cheapest way to answer "does this look right",
                # and it still shows multi-object scenes and decoys.
                return self._send(200, start_job(self._body(), preview=True,
                                                 still=True).as_dict())
            if u.path == "/api/place":
                return self._send(200, _place_preview(self._body()))
            if u.path == "/api/bench":
                from . import bench as _bench
                b = self._body()
                sensor = (b.get("sensor") or "optical").strip()
                ladder = tuple(int(x) for x in (b.get("workers") or []) if int(x) > 0) or (1, 2, 4)
                return self._send(200, _bench.recommend(
                    bench=_bench.measure(workers=ladder, sensor=sensor), sensor=sensor))
            if u.path == "/api/recipe/save":
                b = self._body()
                name = b.get("name") or (b.get("recipe") or {}).get("name")
                p = _save_recipe(name, b.get("recipe") or b)
                return self._send(200, dict(saved=True, name=_safe_name(name), path=p))
            if u.path == "/api/recipe/load":
                b = self._body()
                return self._send(200, dict(recipe=_load_recipe(b.get("name"))))
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
            # A parallel job owns one child per frame range; killing only the
            # first leaves the rest rendering and the merge never runs.
            killed = False
            for p in (list(getattr(j, "procs", None) or []) +
                      ([j.proc] if j.proc else [])):
                if p is not None and p.poll() is None:
                    p.terminate()
                    killed = True
            if killed or j.state == "running":
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
