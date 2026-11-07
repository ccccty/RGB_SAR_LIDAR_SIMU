"""Recipe -> legal scene JSON: the generator behind "build a scene from zero".

The curated files in scenes/*.json are hand-authored. This module is the other
half of the authoring story: a small, fully specified recipe goes in and a
complete scene config comes out, ready for scene.load_config.

The recipe is a plain dict (JSON types only) with these blocks:

    name         clip name, no path separators
    environment  kind (sky/ground/sea), weather, background (none/hdri/upload)
    objects      [{asset, cls, scale, count}], asset relative to assets/models
    ecm          mode (none/active/passive/mixed) plus optional raw entities
    view         range_km, aspect (head_on/pursuit), fov_deg, views (single/dual)
    capture      sensors, frames, fps, seed
    output       preview/coco/yolo/readme switches, passed through untouched

Two contracts are load-bearing and are asserted by tests/test_scenegen.py:

1. build_scene output is consumed by scene.load_config + scene.build_world. The
   emitted "model" paths are ../assets/models/<asset> because a scene file
   lives in scenes/, one level below the repository root -- the same convention
   every curated scene uses. The environment block is not decoration: sky
   disables the ground entirely, ground builds a plane and sea builds a wave
   mesh, so the three produce different world.faces.

2. The ECM defaults are the measured calibration, not a placeholder. A Ku-band
   barrage jammer at 500 m standoff with 200 W ERP puts J/S at about +10 dB on
   this receiver. Scaling that down to a couple of watts would be
   indistinguishable from no jamming at all, so the generator never emits a
   token jammer: the constants below are the ones that were actually measured.

"aspect" drives the engagement geometry through clip.seeker:

    head_on  -> closing="in",  the airframe flies at the camera, nose first
    pursuit  -> closing="out", the airframe flies away, tail toward the camera

Both are realised by the seeker planner itself (the target velocity is the
line-of-sight rate), which is why each object also carries
yaw_mode="velocity": with the nose slaved to the velocity vector, "toward the
camera" and "tail toward the camera" follow from the closing sign instead of
being painted on with a hand-tuned yaw.

Closing speed is a property of the engagement, not of the clip: it comes from
the per-class table CLOSING_MPS (or from view.closing_mps when the recipe names
one), and the frame count only decides how much of the range band the target
crosses. Deriving the speed from frames/duration made a 2-frame preview of one
recipe reach Mach 3.5 while the 20-frame version flew at 300 m/s.

A top_down (nadir plate) view is geometrically degenerate for a multi-target
radar picture: the camera looks straight down, every placed target sits on the
same plane and flies level, so the line of sight is perpendicular to the track,
the radial velocity is ~0 for all of them, and only range can separate them. A
studio run with three fighters measured three ranges inside two range cells and
one bright RD peak. build_scene therefore staggers those targets in height
(along the line of sight) by TOP_DOWN_STAGGER_CELLS range cells, capped at
TOP_DOWN_STAGGER_FRAC of the camera height so the optical frame still contains
them, and records the ladder in cfg["view"] plus a cfg["notes"] line. The step
is solved in slant range, not height, because a target off the boresight turns
a metre of height into less than a metre of range. Explicit objects[].z always
wins. Velocity separation is impossible in this geometry: head_on / pursuit
give it for an airframe (line of sight along the track) and the look_down
aspect -- ASPECTS already carries it, with the depressed camera in _site() --
gives it for a surface target.

environment.background also has to reach the renderers, or a chosen plate is
decoration in the JSON only. It is written where they read it -- the top-level
"optical" render block (optical.sky_image) and the top-level "ir" radiometry
block (ir.background_image) -- and never on cfg["sensors"], where the same key is
silently ignored. "upload" is resolved to an existing absolute file (a missing
one is an error, never a silent gradient); "hdri" resolves a local CC0 HDRI
through simkit.assets_fetch, and when the asset is unavailable offline the build
still succeeds with the gradient sky and records why in cfg["notes"]. "none"
writes no background key at all, and radar/lidar/SAR never carry one.
"""
import json
import math
import random
import os
import re

from . import motion

ENV_KINDS = ('sky', 'ground', 'sea')
# "auto" defers to the environment (air -> head-on, surface -> look-down);
# "look_down" forces the depressed view: _site() puts the camera on the
# depression ray, so the line of sight is tilted and a surface target has a
# velocity component along it. That is the recipe entry for Doppler separation.
ASPECTS = ('auto', 'head_on', 'pursuit', 'look_down', 'top_down')

# ------------------------------------------------------------------ constants
# Mirrors simkit.studio.WEATHER_MODES / WEATHER_LABELS. Duplicated on purpose:
# studio.py is the web layer and importing it from the generator would drag an
# HTTP server into every build. The test suite compares the two tables when
# studio is importable, so drift is caught instead of silently tolerated.
WEATHER_MODES = {
    'clear': dict(mode='fixed', preset='clear'),
    'cloudy': dict(mode='fixed', preset='cloudy'),
    'rain': dict(mode='fixed', preset='rain'),
    'fog': dict(mode='fixed', preset='fog'),
    'mixed': dict(mode='per_clip', pool=['clear', 'cloudy', 'rain', 'fog']),
    'evolve': dict(mode='evolve', pool=['clear', 'cloudy', 'rain', 'fog']),
}
WEATHER_LABELS = {'clear': '晴', 'cloudy': '多云', 'rain': '雨', 'fog': '雾',
                  'mixed': '混合', 'evolve': '演变'}
ENV_LABELS = {'sky': '天空', 'ground': '地面', 'sea': '海面'}
BACKGROUND_MODES = ('none', 'hdri', 'upload', 'library')
BACKGROUND_LABELS = {'none': '纯色背景', 'hdri': 'HDRI', 'upload': '上传图',
                    'library': '图库'}

SENSOR_KEYS = ('optical', 'ir', 'radar', 'lidar', 'sar')
ECM_MODES = ('none', 'active', 'passive', 'mixed')
ECM_LABELS = {'none': '无', 'active': '有源', 'passive': '无源', 'mixed': '混合'}

# Hard limits, the same shape studio.py uses so a value the form accepted is
# accepted here, and a value it rejects gets a Chinese message not a traceback.
LIM = dict(range_km=(0.05, 100.0), fov_deg=(0.1, 120.0), frames=(1, 20000),
           fps=(0.5, 120.0), seed=(0, 2 ** 32 - 1), scale=(0.02, 50.0),
           count=(1, 64), objects=(1, 32), closing_mps=(0.1, 800.0))

# ------------------------------------------------------------------- assets
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, 'assets', 'models')
INDEX_PATH = os.path.join(ROOT, 'assets', 'asset_index.json')
# A scene file is written into scenes/, so ../assets/models resolves to the same
# tree asset_index.json names.
MODEL_PREFIX = '../assets/models'

# Directory name -> label class. This table is the contract: an unknown
# directory keeps its own sanitised name as the class instead of silently
# becoming "target", which would put a wrong class into a COCO file.
CATEGORY_CLS = {
    '军用飞机': 'fighter',
    '民航飞机': 'airliner',
    '固定翼无人机': 'uav_fixed',
    '旋翼无人机': 'uav_rotor',
    '直升机': 'helicopter',
    '军用地面车辆': 'vehicle',
    '地面车辆': 'vehicle',
    '军舰': 'ship',
    '民用船舶': 'ship',
}
CATEGORY_ORDER = tuple(CATEGORY_CLS) + ('sheets',)

# A handful of demo OBJs sit directly in assets/models (no category folder) and
# are not in asset_index.json. They still need a label class: "target" would be
# a wrong COCO class, and a filename-derived one would be an unreadable slug.
DEMO_CATEGORY = '演示模型'
FILE_CLS = {
    'global_hawk.obj': 'uav_fixed',
    'rover.obj': 'vehicle',
    'ship_medium.obj': 'ship',
    'uav_demo.obj': 'uav_rotor',
    'vehicle_demo.obj': 'vehicle',
    'concrete_road_barrier.obj': 'scenery',
    'modular_chainlink_fence.obj': 'scenery',
    'shrub_01.obj': 'scenery',
}

# classes that sit on the ground: the seeker planner pins their altitude to
# ground_z instead of letting the line-of-sight solve pick it
SURFACE_CLS = frozenset(('vehicle', 'tank', 'ship', 'building', 'person',
                         'artillery', 'launcher'))
AIR_CLS = frozenset(('fighter', 'airliner', 'uav_fixed', 'uav_rotor',
                     'helicopter'))

# Which domain a target class belongs to. A dataset may mix models freely inside
# one domain -- a fighter beside a UAV is a real formation -- but a ship in an
# air scene is not a scene, it is a mistake.
DOMAIN_OF_CLS = {
    "fighter": "sky", "airliner": "sky", "helicopter": "sky",
    "uav_fixed": "sky", "uav_rotor": "sky",
    "vehicle": "ground", "tank": "ground", "artillery": "ground",
    "launcher": "ground", "person": "ground", "building": "ground",
    "ship": "sea",
}
DOMAIN_LABELS = {"sky": "空域", "ground": "陆域", "sea": "海域"}

# Fallbacks used when the asset index has no profile for a model. RCS is a
# front-aspect order of magnitude, temperature is skin temperature in sunshine.
CLS_RCS_DB = {'fighter': 14.0, 'airliner': 18.0, 'uav_fixed': 1.0,
              'uav_rotor': -13.0, 'helicopter': 6.0, 'vehicle': 5.0,
              'ship': 25.0, 'target': 5.0}
CLS_TEMP_C = {'fighter': 35.0, 'airliner': 30.0, 'uav_fixed': 25.0,
              'uav_rotor': 26.0, 'helicopter': 28.0, 'vehicle': 32.0,
              'ship': 24.0, 'target': 30.0}
CLS_ENVELOPE = {
    'air': dict(min_speed_mps=60.0, max_speed_mps=700.0, max_accel_mps2=30.0,
                max_climb_mps=250.0, min_altitude_m=50.0, max_bank_deg=65.0),
    'ground': dict(min_speed_mps=0.0, max_speed_mps=25.0, max_accel_mps2=5.0,
                   max_climb_mps=1.0, min_altitude_m=0.0, max_bank_deg=5.0),
    'sea': dict(min_speed_mps=0.0, max_speed_mps=20.0, max_accel_mps2=2.0,
                max_climb_mps=1.0, min_altitude_m=0.0, max_bank_deg=8.0),
}

# Typical closing speed per class, in m/s. An engineering prior per class, NOT a
# measurement, and deliberately independent of the clip length: the frame count
# says how long the clip is, never how fast the aircraft flies. Deriving speed
# from frames/duration made a 2-frame preview of the same recipe reach Mach 3.5
# (1207.8 m/s) while the 20-frame version flew at 300 m/s, i.e. changing the
# frame count changed the physics. The recipe may override the table with
# view.closing_mps, but nothing in this module infers it from duration.
CLOSING_MPS = {
    'fighter': 250.0,      # head-on engagement, both sides closing
    'airliner': 220.0,
    'uav_fixed': 60.0,
    'uav_rotor': 25.0,
    'helicopter': 50.0,
    'vehicle': 15.0,
    'ship': 12.0,
    'boat': 12.0,
    'scenery': 6.0,        # not a target class; a slow drift beats a frozen clip
    'target': 50.0,
}

# Site geometry per environment: camera altitude, and the target altitude for an
# airframe / for a surface vehicle. The three states differ here and in the
# ground block, which is what makes build_world produce three different worlds.
SITE = {
    'sky': dict(cam_z=2500.0, air_z=3000.0, surface_z=0.0, air_temp_c=-22.0,
                rh=0.15),
    'ground': dict(cam_z=1200.0, air_z=600.0, surface_z=0.0, air_temp_c=15.0,
                   rh=0.45),
    'sea': dict(cam_z=260.0, air_z=300.0, surface_z=0.0, air_temp_c=12.0,
                rh=0.75),
}
# A ground or sea target has to be LOOKED DOWN ON. With the camera 6 m above a
# flat surface and the target a kilometre away the grazing angle is 0.3 deg, and
# at that angle the ground is always nearer than the vehicle standing on it: the
# single-return radar model keeps the nearer hit, so the target is permanently
# occluded by the terrain and every frame reports zero detections (measured on
# demo_radar_2km). Depression is derived from the environment, not guessed.
DEPRESSION_DEG = {'ground': 40.0, 'sea': 32.0}
# Air engagements are head-on or pursuit; surface ones are look-down. "auto"
# lets the environment decide, which is what the workbench default should be.
ASPECT_BY_ENV = {'sky': ('head_on', 'pursuit'),
                 'ground': ('look_down',),
                 'sea': ('look_down',)}
# Collected real photograph plates (assets/backdrops). A ground or sea dataset
# is the simulated target composited onto one of these, which is the only way a
# frame shows real forest or harbour texture instead of a procedural gradient.
BACKDROP_DIR = os.path.join(ROOT, 'assets', 'backdrops')
# Three domains, because a dataset is generated for one of them: an aircraft is
# shot against sky, a vehicle against terrain, a vessel against water. Each
# domain carries several scene types so a batch is not 20 copies of one forest.
BACKDROP_DOMAIN = {
    'sky': 'air',
    'forest': 'land', 'city': 'land', 'desert': 'land', 'rural': 'land',
    'mountain': 'land', 'industrial': 'land',
    'open': 'sea', 'port': 'sea', 'island': 'sea',
}
BACKDROP_CATEGORIES = tuple(BACKDROP_DOMAIN.keys())
WIND_MS = {'clear': 6.0, 'cloudy': 9.0, 'rain': 13.0, 'fog': 3.0,
           'mixed': 9.0, 'evolve': 9.0}

# Weather has to change the SKY ITSELF, not just its brightness: the gradient sky
# used to keep the clear-day blue in every preset, so picking "cloudy" only
# darkened the same blue and the frame still read as a sunny day. Overcast and
# rain scatter the direct beam, so the sky loses saturation, the zenith lifts and
# zenith and horizon converge; fog does the same and adds sky glow.
# Sun through the weather. A 2.2 deg seeker sees one patch of sky, so cloud
# structure cannot show up in the frame -- but the LIGHT can: a clear day is a
# hard sun with a bright edge on the airframe, overcast is a diffuse dome with
# almost no directional term. That is the difference the eye actually reads.
WEATHER_SUN = {
    'clear':  dict(irradiance=900.0, on=True),
    'cloudy': dict(irradiance=280.0, on=True),
    'rain':   dict(irradiance=140.0, on=True),
    'fog':    dict(irradiance=190.0, on=True),
    'mixed':  dict(irradiance=520.0, on=True),
    'evolve': dict(irradiance=520.0, on=True),
}

WEATHER_SKY = {
    'clear':  dict(zenith=[0.030, 0.075, 0.220], horizon=[0.52, 0.60, 0.72],
                   strength=1.00, note='直射蓝天，天顶与地平线对比强'),
    'cloudy': dict(zenith=[0.295, 0.315, 0.350], horizon=[0.560, 0.575, 0.600],
                   strength=0.88, note='阴天散射，天顶抬高、饱和度下降'),
    'rain':   dict(zenith=[0.150, 0.160, 0.180], horizon=[0.330, 0.340, 0.360],
                   strength=0.62, note='雨幕吸收，整体压暗且发灰'),
    'fog':    dict(zenith=[0.430, 0.440, 0.455], horizon=[0.700, 0.705, 0.710],
                   strength=0.80, note='雾天强前向散射，地平线发白、上下接近'),
    'mixed':  dict(zenith=[0.200, 0.220, 0.280], horizon=[0.520, 0.545, 0.590],
                   strength=0.85, note='混合天气，取晴与阴之间'),
    'evolve': dict(zenith=[0.200, 0.220, 0.280], horizon=[0.520, 0.545, 0.590],
                   strength=0.85, note='逐段演变，初值为混合天气'),
}
LOS_AZ_DEG = 64.0          # world azimuth of the line of sight

# Radar range window. radar.py gates every echo with
#     r_max = c * n_samples / (4 * B)
# so the sampling count decides how far the radar can see at all. A fixed 2048
# at 50 MHz reaches 3.07 km: an 8 km target never enters the window, every frame
# reports zero range-Doppler boxes, and every other check stays green -- an
# empty modality that looks exactly like a successful run.
C_LIGHT = 2.99792458e8
SAMPLE_MARGIN = 1.4        # keep the target off the last range cell
SAMPLE_MIN = 2048
SAMPLE_MAX = 32768
RADAR_BANDWIDTH_MHZ = 50.0

# Nadir plate views place every target on one plane, so the range-Doppler map
# has no velocity contrast to separate them by (the line of sight is vertical
# and the tracks are horizontal, i.e. perpendicular to it). Placed targets are
# therefore staggered in height -- along the line of sight -- by this many range
# cells, with the total ladder capped at this fraction of the camera height so a
# narrow optical frame still contains the formation.
#
# Eight cells is the number that makes two targets *resolvable* and it is not
# enough to make them *visible*: the displayed map spans the whole range window,
# so eight cells of a 6.4 km window is under ten pixels and the formation still
# reads as one streak (measured on the operator's three-aircraft plate). Thirty-
# two cells keeps each airframe a few cells wide and puts clear space between
# them on the figure; the cap below still compresses the step when a wide
# formation would otherwise be pushed out of the optical frame.
TOP_DOWN_STAGGER_CELLS = 32
# What the summary promises. "Resolvable" is a physics claim -- two targets in
# different range cells -- so it is measured against this smaller separation,
# not against the 32-cell step an operator's eye needs. A pinned altitude of,
# say, 120 m is 19 cells from the flat rung and is genuinely resolvable even
# though it is well inside the visual step, and the summary must not report that
# airframe as lost.
TOP_DOWN_STAGGER_PROMISE_CELLS = 8
TOP_DOWN_STAGGER_FRAC = 0.15
# 1-per-mille guard on the step: z is rounded to millimetres in the scene file,
# and 8 cells exactly would land a neighbour a millimetre inside the promise
# after that rounding.
TOP_DOWN_STAGGER_GUARD = 1.001
# Height is a shorter lever than slant range once a target sits off the
# boresight: dR/dz = cos(tilt). Reserve this fraction of the cap for that so a
# ladder sized in slant range never runs past the height budget.
TOP_DOWN_TILT_GUARD = 0.98
# Range resolution used when the waveform cannot be derived at all. A 60 m cell
# is coarser than anything build_scene emits; the fallback only keeps the ladder
# finite instead of dividing by a zero bandwidth.
RADAR_RES_FALLBACK_M = 60.0

# ECM: the measured calibration. 200 W ERP, Ku, 500 m, barrage over 200 MHz
# lands at J/S ~ +10 dB; anything near 2 W is a no-op on this receiver.
ECM_JAMMER_ERP_W = 200.0
ECM_JAMMER_RANGE_M = 500.0
ECM_JAMMER_BAND = 'Ku'
ECM_JAMMER_JS_DB = 10.0
ECM_JAMMER_NOTE = ('calibration: 200 W ERP, barrage over 200 MHz, Ku receiver, '
                   '500 m standoff -> J/S about +10 dB (measured). A few watts '
                   'would not raise the noise floor measurably, so the ERP is '
                   'never scaled down to a token value.')

_ASSETS = []
_ASSET_BY_PATH = {}
_OBJ_FACTS = {}
_INDEX = None


# ------------------------------------------------------------------ utilities
def _read_asset_index():
    global _INDEX
    if _INDEX is None:
        try:
            with open(INDEX_PATH, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            data = {}
        models = data.get('models') if isinstance(data, dict) else None
        _INDEX = [m for m in (models or []) if isinstance(m, dict)]
    return _INDEX


def _index_key(path):
    """'assets/models/军用飞机/x.obj' -> '军用飞机/x.obj' (forward slashes)."""
    p = str(path or '').replace(chr(92), '/')
    marker = 'assets/models/'
    if marker in p:
        p = p.split(marker, 1)[1]
    return p.strip('/')


def _obj_facts(full_path):
    """Face count and bbox extent straight from the file.

    Only the OBJ files without an asset_index entry need this, so the parse
    stays cheap: vertex lines for the bbox, face lines for the count.
    """
    if full_path in _OBJ_FACTS:
        return _OBJ_FACTS[full_path]
    lo = [math.inf] * 3
    hi = [-math.inf] * 3
    faces = 0
    try:
        with open(full_path, 'r', encoding='utf-8', errors='ignore') as fh:
            for line in fh:
                if line.startswith('v '):
                    parts = line.split()
                    if len(parts) < 4:
                        continue
                    for i in range(3):
                        try:
                            v = float(parts[i + 1])
                        except ValueError:
                            continue
                        lo[i] = min(lo[i], v)
                        hi[i] = max(hi[i], v)
                elif line.startswith('f '):
                    faces += max(len(line.split()) - 3, 1)   # n-gon -> n-2 tris
    except OSError:
        return (0, [1.0, 1.0, 1.0])
    if faces <= 0 or not math.isfinite(lo[0]):
        return (0, [1.0, 1.0, 1.0])
    size = [round(hi[i] - lo[i], 4) for i in range(3)]
    _OBJ_FACTS[full_path] = (int(faces), size)
    return _OBJ_FACTS[full_path]


def _cls_for(category):
    name = str(category or '')
    if name in CATEGORY_CLS:
        return CATEGORY_CLS[name]
    slug = re.sub('[^A-Za-z0-9_]+', '_', name).strip('_')
    return slug or 'target'


def _scan_assets():
    global _ASSETS
    if _ASSETS:
        return _ASSETS
    indexed = {}
    for entry in _read_asset_index():
        key = _index_key(entry.get('path'))
        if key:
            indexed[key] = entry
    rows = []
    if os.path.isdir(MODELS_DIR):
        for base, _dirs, files in os.walk(MODELS_DIR):
            for fname in files:
                if not fname.lower().endswith('.obj'):
                    continue
                full = os.path.join(base, fname)
                rel = os.path.relpath(full, MODELS_DIR).replace(chr(92), '/')
                entry = indexed.get(rel) or {}
                geometry = entry.get('geometry') or {}
                size = geometry.get('bbox_extent')
                if isinstance(size, (list, tuple)) and len(size) == 3:
                    tris = int(geometry.get('faces') or 0)
                    size = [round(float(v), 4) for v in size]
                else:
                    tris, size = _obj_facts(full)
                category = entry.get('category')
                if not category:
                    parts = rel.split('/')
                    category = parts[0] if len(parts) > 1 else DEMO_CATEGORY
                name = entry.get('name') or os.path.splitext(fname)[0]
                cls = FILE_CLS.get(fname.lower()) or _cls_for(category)
                rows.append(dict(category=str(category), name=str(name),
                                 path=rel, tris=int(tris),
                                 size_m=list(size), cls=cls))
                _ASSET_BY_PATH[rel] = rows[-1]
    rows.sort(key=lambda r: (
        CATEGORY_ORDER.index(r['category']) if r['category'] in CATEGORY_ORDER
        else len(CATEGORY_ORDER), r['name'], r['path']))
    _ASSETS = rows
    return _ASSETS


def list_assets():
    """Every OBJ under assets/models, with the geometry the index already knows.

    Returns [{category, name, path, tris, size_m: [x, y, z], cls}]. "path" is
    relative to assets/models, i.e. exactly the string a recipe's
    objects[].asset takes, and any file missing from asset_index.json is
    measured from the OBJ itself so the list never silently drops a model.
    """
    return [dict(row) for row in _scan_assets()]


def _asset_row(rel):
    _scan_assets()
    return _ASSET_BY_PATH.get(rel)


def _profile(rel):
    """First curated profile of an asset (rcs / thermal), or {}."""
    for entry in _read_asset_index():
        if _index_key(entry.get('path')) == rel:
            profiles = entry.get('profiles') or []
            if profiles and isinstance(profiles[0], dict):
                return profiles[0]
    return {}


# ------------------------------------------------------------------ validation
def _trim(v):
    if v is None:
        return '?'
    return ('%d' % v) if float(v) == int(v) else ('%g' % v)


def _number(value, field, low=None, high=None, unit=''):
    if isinstance(value, bool) or value is None:
        raise ValueError('%s 必须是数字（收到 %r）' % (field, value))
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError('%s 必须是数字（收到 %r）' % (field, value)) from None
    if not math.isfinite(v):
        raise ValueError('%s 必须是有限数字（收到 %r）' % (field, value))
    if (low is not None and v < low) or (high is not None and v > high):
        raise ValueError('%s 必须在 %s–%s%s 之间（收到 %r）'
                         % (field, _trim(low), _trim(high), unit, value))
    return v


def _integer(value, field, low, high):
    v = _number(value, field)
    if abs(v - round(v)) > 1e-9:
        raise ValueError('%s 必须是整数（收到 %r）' % (field, value))
    v = int(round(v))
    if v < low or v > high:
        raise ValueError('%s 必须在 %d–%d 之间（收到 %r）'
                         % (field, low, high, value))
    return v


def _choice(value, field, options, default=None):
    if value in (None, ''):
        if default is None:
            raise ValueError('%s 必填，可选值：%s' % (field, '、'.join(options)))
        return default
    if not isinstance(value, str) or value not in options:
        raise ValueError('%s 必须是 %s 之一（收到 %r）'
                         % (field, '、'.join(options), value))
    return value


def _text(value, field, default=None, pattern=None, pattern_msg=None):
    if value in (None, ''):
        if default is None:
            raise ValueError('%s 必填' % field)
        return default
    if not isinstance(value, str):
        raise ValueError('%s 必须是字符串（收到 %r）' % (field, value))
    v = value.strip()
    if not v:
        raise ValueError('%s 必填' % field)
    if pattern is not None and not re.match(pattern, v):
        raise ValueError('%s %s（收到 %r）'
                         % (field, pattern_msg or '格式不对', value))
    return v


def _bool(value, field, default=True):
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError('%s 必须是布尔值 true/false（收到 %r）' % (field, value))
    return value


def _asset_path(value, field):
    rel = _text(value, field).replace(chr(92), '/')
    while rel.startswith('./'):
        rel = rel[2:]
    parts = rel.split('/')
    if rel.startswith('/') or ':' in rel or '..' in parts:
        raise ValueError('%s 只能写 assets/models 内的相对路径（收到 %r）'
                         % (field, value))
    if not rel.lower().endswith('.obj'):
        raise ValueError('%s 必须是 .obj 文件（收到 %r）' % (field, value))
    full = os.path.join(MODELS_DIR, *parts)
    if not os.path.isfile(full):
        raise ValueError('%s 指向的模型不存在：%s（相对 assets/models）'
                         % (field, rel))
    return rel


TRAJECTORY_MODES = ('preset', 'drawn')


def _trajectory_spec(raw):
    """Normalise view.trajectory: the built-in engagement, or a drawn route.

    "preset" is the authored seeker geometry (distance + aspect). "drawn" is a
    route the operator clicked on the canvas; it is fitted by motion.Waypoints at
    build time, so a hand-drawn corner never reaches the airframe as an
    instantaneous heading change.
    """
    if raw is None:
        return dict(mode='preset')
    if not isinstance(raw, dict):
        raise ValueError('view.trajectory 必须是字典（收到 %r）' % (raw,))
    mode = _choice(raw.get('mode'), 'view.trajectory.mode', TRAJECTORY_MODES, 'preset')
    if mode == 'preset':
        return dict(mode='preset')
    pts_raw = raw.get('points')
    if not isinstance(pts_raw, (list, tuple)):
        raise ValueError('view.trajectory.points 必须是 [[x, y, z], ...] 数组')
    pts = []
    for item in pts_raw:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            raise ValueError('view.trajectory.points 的每个点必须是 [x, y, z]')
        try:
            xyz = [float(item[0]), float(item[1]), float(item[2])]
        except (TypeError, ValueError):
            raise ValueError('view.trajectory.points 的坐标必须是数字') from None
        if max(abs(v) for v in xyz) > 100000.0:
            raise ValueError('view.trajectory.points 超出范围（±100 km）')
        if not pts or max(abs(xyz[i] - pts[-1][i]) for i in range(3)) > 0.5:
            pts.append(xyz)
    if len(pts) < 2:
        raise ValueError('手绘轨迹至少需要两个不同的点')
    speed = _number(raw.get('speed_mps', 250.0), 'view.trajectory.speed_mps', 1.0, 1000.0)
    return dict(mode='drawn', points=pts, speed_mps=speed,
                closed=bool(raw.get('closed')))


def validate_recipe(recipe):
    """Normalise a recipe and reject anything the pipeline could not build.

    Returns a fresh dict: defaults filled in, numbers coerced, unknown keys
    dropped. Every error is a Chinese ValueError naming the offending field,
    because the caller is a form that shows the message to the user.
    """
    if not isinstance(recipe, dict):
        raise ValueError('配方必须是一个字典（收到 %s）' % type(recipe).__name__)

    # Chinese names are allowed on purpose: the operator names a dataset in the
    # language they think in, and Windows paths carry UTF-8 fine. Path separators
    # and wildcard characters stay banned because the name becomes a folder.
    name = _text(recipe.get('name'), 'name', 'demo',
                 r'^[A-Za-z0-9_.\-\u4e00-\u9fff]{1,64}$',
                 '只能包含中文、字母、数字、下划线、点和连字符（1–64 个字符）')

    env_raw = recipe.get('environment')
    if env_raw is None:
        env_raw = {}
    if not isinstance(env_raw, dict):
        raise ValueError('environment 必须是字典（收到 %r）' % (env_raw,))
    kind = _choice(env_raw.get('kind'), 'environment.kind', ENV_KINDS, 'sky')
    weather = _choice(env_raw.get('weather'), 'environment.weather',
                      tuple(WEATHER_MODES), 'clear')
    bg_raw = env_raw.get('background')
    if bg_raw is None:
        bg_raw = {'mode': 'none'}
    if not isinstance(bg_raw, dict):
        raise ValueError('environment.background 必须是字典（收到 %r）' % (bg_raw,))
    bg_mode = _choice(bg_raw.get('mode'), 'environment.background.mode',
                      BACKGROUND_MODES, 'none')
    background = {'mode': bg_mode}
    if bg_mode == 'hdri':
        hid = bg_raw.get('id')
        if not isinstance(hid, str) or not hid.strip():
            raise ValueError('environment.background 为 hdri 时必须给出 id')
        background['id'] = hid.strip()
    elif bg_mode == 'library':
        # Real photograph from assets/backdrops: the dataset is the simulated
        # target composited onto a collected plate, so the frame has honest
        # forest/city/port texture instead of a procedural gradient.
        cat = str(bg_raw.get('category') or '').strip()
        if cat not in BACKDROP_CATEGORIES:
            raise ValueError('environment.background 为 library 时 category 必须是 %s 之一'
                             '（收到 %r）' % ('/'.join(BACKDROP_CATEGORIES), cat))
        background['category'] = cat
        if bg_raw.get('file'):
            background['file'] = str(bg_raw['file']).strip()
    elif bg_mode == 'upload':
        hpath = bg_raw.get('path')
        if not isinstance(hpath, str) or not hpath.strip():
            raise ValueError('environment.background 为 upload 时必须给出 path')
        background['path'] = hpath.strip()
    environment = dict(kind=kind, weather=weather, background=background)

    objs_raw = recipe.get('objects')
    if objs_raw in (None, [], ()):
        raise ValueError('objects 不能为空：至少给一个目标模型'
                         '（如 {"asset": "军用飞机/双发战斗机.obj"}）')
    if not isinstance(objs_raw, (list, tuple)):
        raise ValueError('objects 必须是数组（收到 %r）' % (objs_raw,))
    if len(objs_raw) > LIM['objects'][1]:
        raise ValueError('objects 最多 %d 项（收到 %d）'
                         % (LIM['objects'][1], len(objs_raw)))
    objects = []
    for i, raw in enumerate(objs_raw):
        field = 'objects[%d]' % i
        if not isinstance(raw, dict):
            raise ValueError('%s 必须是字典（收到 %r）' % (field, raw))
        asset = _asset_path(raw.get('asset'), field + '.asset')
        row = _asset_row(asset) or {}
        default_cls = row.get('cls') or _cls_for(asset.split('/')[0])
        cls = _text(raw.get('cls') or default_cls, field + '.cls',
                    pattern=r'^[A-Za-z][A-Za-z0-9_]{0,31}$',
                    pattern_msg='只能包含字母、数字、下划线，且以字母开头')
        scale = _number(raw.get('scale', 1.0), field + '.scale',
                        LIM['scale'][0], LIM['scale'][1], ' 倍')
        count = _integer(raw.get('count', 1), field + '.count',
                         LIM['count'][0], LIM['count'][1])
        item = dict(asset=asset, cls=cls, scale=scale, count=count)
        # Placement on a backdrop plate: normalised image coordinates, origin at
        # the top-left. Optional per-object because a formation can be placed by
        # its leader alone.
        _uv = raw.get('uv')
        if _uv is not None:
            if not isinstance(_uv, (list, tuple)) or len(_uv) not in (2, 3):
                raise ValueError('%s.uv 必须是 [u, v] 或 [u, v, 间距]（收到 %r）'
                                 % (field, _uv))
            vals = [_number(_uv[k], '%s.uv[%d]' % (field, k), 0.0, 1.0) for k in (0, 1)]
            if len(_uv) == 3:
                vals.append(_number(_uv[2], field + '.uv[2]', 0.0, 1.0))
            item['uv'] = vals
        if raw.get('z') is not None:
            # The height ladder a top_down radar scene writes back can be a few
            # hundred metres to a couple of kilometres (0.15 x camera height, and
            # the camera sits at the slant range). The old 1000 m ceiling made a
            # re-saved scene unloadable: "objects[2].z 超出 -50~1000 m（收到
            # 1421.252）", so the operator's own recipe could not be regenerated.
            item['z'] = _number(raw.get('z'), field + '.z', -1000.0, 20000.0, ' m')
        # A per-target route. plan.py already lets obj["trajectory"] win over
        # formation and over the primary's line, so this only has to survive
        # validation; dropping it here was why "custom route per aircraft" did
        # nothing.
        _tj = raw.get('trajectory')
        if _tj is not None:
            if not isinstance(_tj, dict):
                raise ValueError('%s.trajectory 必须是字典（收到 %r）' % (field, _tj))
            item['trajectory'] = _norm_target_route(_tj, field + '.trajectory')
        objects.append(item)

    # Domain consistency: models may be mixed freely inside one domain (a
    # fighter beside a UAV), but not across them. A ship in an air scene is a
    # mistake that otherwise renders happily.
    _doms = {}
    for _i, _o in enumerate(objects):
        _d = DOMAIN_OF_CLS.get(_o["cls"])
        if _d is None:
            continue
        _doms.setdefault(_d, []).append("objects[%d](%s)" % (_i, _o["cls"]))
    if len(_doms) > 1:
        raise ValueError("目标类别跨域混搭：%s。同一数据集里的目标必须属于同一域"
                         "（空域只能混不同飞行器，陆域只能混不同车辆，"
                         "海域只能混不同船舶）"
                         % "；".join("%s: %s" % (DOMAIN_LABELS.get(k, k), "、".join(v))
                                    for k, v in sorted(_doms.items())))

    ecm_raw = recipe.get('ecm')
    if ecm_raw is None:
        ecm_raw = {'mode': 'none', 'entities': []}
    if not isinstance(ecm_raw, dict):
        raise ValueError('ecm 必须是字典（收到 %r）' % (ecm_raw,))
    ecm_mode = _choice(ecm_raw.get('mode'), 'ecm.mode', ECM_MODES, 'none')
    raw_ents = ecm_raw.get('entities')
    if raw_ents in (None, ''):
        raw_ents = []
    if not isinstance(raw_ents, (list, tuple)):
        raise ValueError('ecm.entities 必须是数组（收到 %r）' % (raw_ents,))
    ents = []
    for i, item in enumerate(raw_ents):
        if not isinstance(item, dict):
            raise ValueError('ecm.entities[%d] 必须是字典（收到 %r）' % (i, item))
        ents.append(dict(item))
    if ecm_mode == 'none' and ents:
        raise ValueError('ecm.mode 为 none 时不能带实体；'
                         '要加干扰请把 mode 设为 active/passive/mixed')
    ecm = dict(mode=ecm_mode, entities=ents)

    view_raw = recipe.get('view')
    if view_raw is None:
        view_raw = {}
    if not isinstance(view_raw, dict):
        raise ValueError('view 必须是字典（收到 %r）' % (view_raw,))
    # Optional: the user's own closing speed. When absent the class table
    # CLOSING_MPS decides -- never the frame count, which only says how long the
    # clip is.
    closing = view_raw.get('closing_mps')
    closing = (None if closing in (None, '') else
               _number(closing, 'view.closing_mps', LIM['closing_mps'][0],
                       LIM['closing_mps'][1], ' m/s'))
    view = dict(
        range_km=_number(view_raw.get('range_km', 8.0), 'view.range_km',
                         LIM['range_km'][0], LIM['range_km'][1], ' km'),
        aspect=_choice(view_raw.get('aspect'), 'view.aspect', ASPECTS, 'auto'),
        # Plate span in metres for the top_down (nadir) mode: it is the size of
        # the map the operator clicked on, so it has to be explicit.
        ground_span_m=_number(view_raw.get('ground_span_m'),
                              'view.ground_span_m', 50.0, 20000.0, ' m')
        if view_raw.get('ground_span_m') not in (None, '') else None,
        fov_deg=_number(view_raw.get('fov_deg', 2.2), 'view.fov_deg',
                        LIM['fov_deg'][0], LIM['fov_deg'][1], ' 度'),
        views=_choice(view_raw.get('views'), 'view.views',
                      ('single', 'dual'), 'single'),
        closing_mps=closing,
        trajectory=_trajectory_spec(view_raw.get('trajectory')))

    cap_raw = recipe.get('capture')
    if cap_raw is None:
        cap_raw = {}
    if not isinstance(cap_raw, dict):
        raise ValueError('capture 必须是字典（收到 %r）' % (cap_raw,))
    want = cap_raw.get('sensors')
    if want in (None, '', [], ()):
        raise ValueError('capture.sensors 不能为空：至少选一个传感器'
                         '（optical、ir、radar、lidar、sar）')
    if not isinstance(want, (list, tuple)):
        raise ValueError('capture.sensors 必须是数组（收到 %r）' % (want,))
    bad = [w for w in want if w not in SENSOR_KEYS]
    if bad:
        raise ValueError('capture.sensors 含未知传感器：%s（可用：%s）'
                         % ('、'.join(map(str, bad)), '、'.join(SENSOR_KEYS)))
    sensors = []
    for w in want:
        if w not in sensors:
            sensors.append(w)
    capture = dict(
        sensors=sensors,
        frames=_integer(cap_raw.get('frames', 20), 'capture.frames',
                        LIM['frames'][0], LIM['frames'][1]),
        fps=_number(cap_raw.get('fps', 10.0), 'capture.fps',
                    LIM['fps'][0], LIM['fps'][1], ' fps'),
        seed=_integer(cap_raw.get('seed', 7), 'capture.seed',
                      LIM['seed'][0], LIM['seed'][1]))

    out_raw = recipe.get('output')
    if out_raw is None:
        out_raw = {}
    if not isinstance(out_raw, dict):
        raise ValueError('output 必须是字典（收到 %r）' % (out_raw,))
    output = {k: _bool(out_raw.get(k), 'output.' + k, True)
              for k in ('preview', 'coco', 'yolo', 'readme')}

    # Sensor effects the operator can switch, kept in the recipe so a saved run
    # reproduces the same picture. `plume` is the exhaust.  The default is ON:
    # every scene built before this switch existed had engine heat, and a recipe
    # that says nothing must keep looking the way it looked.  The workbench always
    # sends an explicit value, so unticking its box is still how you turn it off.
    fx_raw = recipe.get('effects')
    if fx_raw is None:
        fx_raw = {}
    if not isinstance(fx_raw, dict):
        raise ValueError('effects 必须是字典（收到 %r）' % (fx_raw,))
    effects = dict(
        plume=_bool(fx_raw.get('plume'), 'effects.plume', True),
        # 0 = the renderer's own default (one sample per pixel).  2 or 3 trades
        # time for an antialiased plume: at engagement range the exhaust is a few
        # pixels wide and point sampling turns it into a jagged bar.
        supersample=_integer(fx_raw.get('supersample', 0),
                             'effects.supersample', 0, 4))

    normalised = dict(name=name, environment=environment, objects=objects,
                      ecm=ecm, view=view, capture=capture, output=output,
                      effects=effects)
    # entities are validated last: ecm.load owns their schema, and wrapping its
    # message keeps every failure the caller sees in Chinese.
    if ecm_mode != 'none':
        _check_entities(normalised)
    return normalised


# ---------------------------------------------------------------- environment
def depression_for(env, aspect='auto'):
    """Look-down angle in degrees, or None for a level engagement."""
    if aspect == 'look_down' or (aspect == 'auto' and env in DEPRESSION_DEG):
        return float(DEPRESSION_DEG.get(env, 35.0))
    return None


def _site(env, range_m, surface, depression_deg=None):
    """Camera and nominal target pose for one engagement."""
    spec = SITE[env]
    cam_z = spec['cam_z']
    tgt_z = spec['surface_z'] if surface else spec['air_z']
    if depression_deg is not None and surface:
        # Place the camera on a ray of the requested depression: the slant range
        # is what the pixel budget talks about, so altitude follows from it.
        dep = math.radians(max(1.0, min(89.0, float(depression_deg))))
        tgt_z = spec['surface_z']
        cam_z = tgt_z + range_m * math.sin(dep)
        horizontal = range_m * math.cos(dep)
        az = math.radians(LOS_AZ_DEG)
        back = (math.cos(az), math.sin(az))
        cam = (-back[0] * horizontal, -back[1] * horizontal, cam_z)
        tgt = (0.0, 0.0, tgt_z)
        los = (tgt[0] - cam[0], tgt[1] - cam[1], tgt[2] - cam[2])
        norm = math.sqrt(sum(v * v for v in los)) or 1.0
        return dict(cam=cam, tgt=tgt, unit=tuple(v / norm for v in los), az=az,
                    horizontal=horizontal, cam_z=cam_z, tgt_z=tgt_z,
                    perp=(-math.sin(az), math.cos(az), 0.0))
    if env == 'sky' and not surface:
        # A slanted line of sight makes the seeker solve drag the target down the
        # slant: 500 m of height over an 8 km engagement turned into a -270 m/s
        # descent and a nose pointing 132 deg away from the line of sight, i.e.
        # an aircraft apparently flying tail-first. An air-to-air engagement is
        # flown level, so the solver gets a level line to work with.
        tgt_z = cam_z
    dz = tgt_z - cam_z
    if abs(dz) > 0.8 * range_m:                     # keep the LOS near level
        dz = math.copysign(0.8 * range_m, dz)
        tgt_z = cam_z + dz
    horizontal = math.sqrt(max(range_m * range_m - dz * dz, 0.01))
    az = math.radians(LOS_AZ_DEG)
    back = (math.cos(az), math.sin(az))
    cam = (-back[0] * horizontal, -back[1] * horizontal, cam_z)
    tgt = (0.0, 0.0, tgt_z)
    los = (tgt[0] - cam[0], tgt[1] - cam[1], tgt[2] - cam[2])
    norm = math.sqrt(sum(v * v for v in los))
    return dict(cam=cam, tgt=tgt, unit=tuple(v / norm for v in los), az=az,
                horizontal=horizontal, cam_z=cam_z, tgt_z=tgt_z,
                perp=(-math.sin(az), math.cos(az), 0.0))


def _ground_block(env, weather, site):
    if env == 'sky':
        # an air scene carries no terrain at all: build_world adds no ground,
        # so the world is the airframe alone
        return {'enabled': False}
    size = round(max(200.0, 2.2 * site['horizontal']), 1)
    wind = float(WIND_MS.get(weather, 8.0))
    if env == 'sea':
        return {'enabled': True, 'type': 'sea', 'mat': 'seawater',
                'size': size, 'z': 0.0, 'wind_ms': wind,
                'nx': 48, 'ny': 48, 'seed': 7}
    return {'enabled': True, 'type': 'ground', 'mat': 'concrete',
            'size': size, 'z': 0.0, 'wind_ms': wind}


# --------------------------------------------------------------------- objects
def _norm_target_route(raw, field):
    """Validate one target's drawn route: waypoints the scene placed on a plate.

    Only waypoints are accepted here. A per-target line/arc would need the same
    handling as the primary target's and nobody has asked for it; refusing is
    better than silently dropping the points and flying a straight line.
    """
    mode = str(raw.get("mode") or "waypoints").lower()
    if mode not in ("waypoints", "drawn"):
        raise ValueError("%s.mode 目前只支持 waypoints（收到 %r）" % (field, mode))
    pts = raw.get("points")
    if not isinstance(pts, (list, tuple)) or len(pts) < 2:
        raise ValueError("%s.points 至少需要 2 个点（收到 %r）" % (field, type(pts).__name__))
    out = []
    for i, p in enumerate(pts):
        if not isinstance(p, (list, tuple)) or len(p) not in (2, 3):
            raise ValueError("%s.points[%d] 必须是 [x, y] 或 [x, y, z]" % (field, i))
        xyz = [_number(p[0], "%s.points[%d][0]" % (field, i), -50000.0, 50000.0, " m"),
               _number(p[1], "%s.points[%d][1]" % (field, i), -50000.0, 50000.0, " m")]
        xyz.append(_number(p[2], "%s.points[%d][2]" % (field, i), -50.0, 20000.0, " m")
                   if len(p) == 3 else 0.0)
        out.append([round(v, 3) for v in xyz])
    speed = _number(raw.get("speed_mps") or 0.0, field + ".speed_mps",
                    0.0, 1200.0, " m/s")
    return dict(mode="waypoints", points=out, speed_mps=float(speed),
                loop=bool(raw.get("loop", False)),
                # authored routes describe a leg already in flight; ramping in
                # from rest turned a 250 m/s jet into a catapult
                ramp_in=bool(raw.get("ramp_in", False)))


def _extend_cruise_route(route, duration_s, frame_half=None):
    """Append a straight leg so an open cruise route fills the whole clip.

    A Waypoints route brakes at its last point and holds there. A 20-frame clip
    at 10 fps is 2 s, which at 80 m/s is 160 m, but a stroke drawn on a 600 m
    plate is typically 20-40 m: the airframe finished its route in a quarter of
    a second and hovered for the rest of the clip. The dataset then showed
    aircraft that "fly arbitrarily" -- they had in fact stopped. Extending the
    final heading keeps the drawn manoeuvre and lets the airframe leave it at
    cruise, which is what a cruise leg means.

    Only open routes are extended, and never one that ramps in from rest: that
    route spends its first second accelerating, so its length is not the span
    the clip needs. Returns the metres added (0 when nothing changed).
    """
    pts = route.get("points") or []
    if len(pts) < 2 or route.get("loop") or route.get("ramp_in"):
        return 0.0
    speed = float(route.get("speed_mps") or 0.0)
    if not (speed > 0.0):
        return 0.0
    length = 0.0
    for a, b in zip(pts, pts[1:]):
        length += math.sqrt(sum((float(b[k]) - float(a[k])) ** 2 for k in range(3)))
    need = speed * max(float(duration_s), 0.0)
    if length <= 1e-6 or need <= length * 1.02:
        return 0.0
    direction = None
    for i in range(len(pts) - 1, 0, -1):
        d = [float(pts[i][k]) - float(pts[i - 1][k]) for k in range(3)]
        n = math.sqrt(sum(v * v for v in d))
        if n > 1e-6:
            direction = [v / n for v in d]
            break
    if direction is None:
        return 0.0
    # 5 % margin: the spline starts braking before the final knot, so a route
    # that exactly filled the clip would still hold for the last frames.
    extra = need - length + 0.05 * need
    last = [float(v) for v in pts[-1]]
    if frame_half:
        # Never fly the airframe out of the plate: a route extended past the
        # frame edge would leave the target out of the optical frame (and out of
        # its labels) for the last frames, which is worse than the short-route
        # freeze this extension is fixing. Scaled to 98 % of the frame.
        scale = 1.0
        for k in (0, 1):
            d = direction[k] * extra
            if abs(d) < 1e-9:
                continue
            limit = 0.98 * float(frame_half[k])
            room = (limit - last[k]) if d > 0 else (-limit - last[k])
            scale = min(scale, max(0.0, room / d))
        if scale < 0.05:
            return 0.0
        extra *= scale
    route["points"] = list(pts) + [
        [round(last[k] + direction[k] * extra, 3) for k in range(3)]]
    return extra


def _object_size(asset, scale):
    """Characteristic size, matching plan.object_size_m's convention."""
    row = _asset_row(asset) or {}
    ext = row.get('size_m') or [1.0, 1.0, 1.0]
    size = max(float(ext[0]), float(ext[1]), 0.6 * float(ext[2]), 0.2)
    return size * float(scale)


def _object_facts(asset, cls):
    """(rcs_db, temp_c): the curated profile when there is one, else the class."""
    prof = _profile(asset)
    rcs = (prof.get('rcs') or {}).get('value_m2')
    if isinstance(rcs, (int, float)) and not isinstance(rcs, bool) and rcs > 0:
        rcs_db = round(10.0 * math.log10(float(rcs)), 2)
    else:
        rcs_db = float(CLS_RCS_DB.get(cls, CLS_RCS_DB['target']))
    skin = (prof.get('thermal') or {}).get('skin_c')
    if isinstance(skin, (int, float)) and not isinstance(skin, bool):
        temp = float(skin)
    else:
        temp = float(CLS_TEMP_C.get(cls, CLS_TEMP_C['target']))
    return rcs_db, temp


def _envelope(cls, env=None):
    """The kinematic box the consistency checker will hold this object to.

    A sky scene has no ground plane, so z there is a plate coordinate rather
    than an altitude: the demo plate's bottom rung sits at z = 0 by
    construction, and applying the 50 m air floor reported it as "buried" on
    every one of the 20 frames (altitude_min, inside the B2 bucket).  Surface
    scenes keep the floor, because their z really is height above the ground.
    """
    if cls in AIR_CLS:
        out = dict(CLS_ENVELOPE['air'])
        if str(env) == 'sky':
            out['min_altitude_m'] = 0.0
        return out
    if cls == 'ship':
        return dict(CLS_ENVELOPE['sea'])
    return dict(CLS_ENVELOPE['ground'])


def _pixels(size_m, range_m, fov_deg, width_px):
    """Pixels an object of size_m spans at range_m (plan.px_for_range)."""
    half = math.tan(math.radians(fov_deg) / 2.0)
    return float(size_m) * float(width_px) / (2.0 * max(range_m, 1e-6) * half)


# ------------------------------------------------------------------------ ecm
# Bodies dispensed from the aircraft ride with it. A jammer is its own airframe,
# so it is deliberately absent: it flies its own line and should be able to leave.
_RIDES_WITH_TARGET = ('chaff_cloud', 'infrared_flare', 'corner_reflector',
                      'decoy', 'false_target')


def _complete_entity(ent, site, vel, life, index, parent=None):
    """Fill the placement fields a hand-edited entity does not carry.

    An operator who changes the jammer's ERP submits only the fields the panel
    shows. Without this the entity has no pos, lands at the world origin and
    illuminates nothing: the run succeeds and the jamming silently does not
    exist, which is worse than refusing it.
    """
    kind = str(ent.get('kind') or 'noise_jammer')
    tgt = [round(v, 3) for v in site['tgt']]
    cam, perp, unit = site['cam'], site['perp'], site['unit']
    if kind == 'noise_jammer':
        home = [round(cam[i] + unit[i] * ECM_JAMMER_RANGE_M, 3) for i in range(3)]
    elif kind == 'chaff_cloud':
        home = [round(tgt[i] - unit[i] * 60.0 + perp[i] * 20.0, 3) for i in range(3)]
    elif kind == 'corner_reflector':
        home = [round(tgt[i] + perp[i] * 30.0, 3) for i in range(3)]
    elif kind == 'infrared_flare':
        # A dispensed flare separates from the airframe; putting it on top of the
        # target made it invisible even once it had a visible radius.
        home = [round(tgt[i] - unit[i] * 45.0 + perp[i] * 18.0, 3) for i in range(3)]
        home[2] = round(home[2] - 25.0, 3)
    else:
        home = list(tgt)
    # Parent the dispensed bodies to the aircraft so they stay in its
    # neighbourhood. Absolute pos+vel made a chaff bloom fly off on its own line
    # while the seeker dragged the aircraft elsewhere, which looked like the decoy
    # moving and the target standing still.
    if parent is not None and kind in _RIDES_WITH_TARGET:
        def _rel(du, dp, dz=0.0):
            return [round(unit[i] * du + perp[i] * dp + (dz if i == 2 else 0.0), 3)
                    for i in range(3)]
        ent.setdefault('parent', parent)
        if kind == 'chaff_cloud':
            ent.setdefault('rel_pos', _rel(-60.0, 20.0))
            ent.setdefault('rel_vel', _rel(-6.0, 3.0, -1.5))
            # measured: a chaff cloud stops relative to the air in well under a
            # second (tau_q 0.03-0.2 s), so 2.5 s was an order too slow and the
            # cloud kept pace with the aircraft instead of falling behind.
            ent.setdefault('rel_decay_s', 0.3)
        elif kind == 'infrared_flare':
            ent.setdefault('rel_pos', _rel(-45.0, 18.0, -25.0))
            ent.setdefault('rel_vel', _rel(-10.0, 4.0, -2.0))
            # a burning flare is heavier and slower to decelerate (tau_q 0.25-0.6 s)
            ent.setdefault('rel_decay_s', 1.0)
        elif kind == 'corner_reflector':
            ent.setdefault('rel_pos', _rel(0.0, 30.0))
            ent.setdefault('rel_vel', [0.0, 0.0, 0.0])
            ent.setdefault('rel_decay_s', 4.0)
        elif kind == 'decoy':
            ent.setdefault('rel_pos',
                           _rel(-float(ent.get('tether_m') or 80.0), 0.0))
            ent.setdefault('rel_vel', [0.0, 0.0, 0.0])
        else:                                   # false_target rides inside
            ent.setdefault('rel_pos', [0.0, 0.0, 0.0])
            ent.setdefault('rel_vel', [0.0, 0.0, 0.0])

    ent.setdefault('id', 'ecm_%02d' % (index + 1))
    ent.setdefault('pos', home)
    ent.setdefault('vel', list(vel))
    ent.setdefault('life', dict(life))
    ent.setdefault('visible', True)
    if kind == 'noise_jammer':
        ent.setdefault('band', ECM_JAMMER_BAND)
        ent.setdefault('mode', 'barrage')
    return ent


# Reference-calibrated paint per platform family. The asset folder is checked
# first because it distinguishes 军用地面车辆 from 地面车辆, which the class table
# cannot; the class is the fallback for hand-written scenes.
# Only the families whose reference estimate cleared the 1.5x confidence bar --
# or showed a qualitative channel-order reversal -- get their own paint. The
# rest keep the generic value on purpose:
#   军用飞机 0.72x  -> below the bar, and a direct crop comparison said our
#                      render was too DARK, i.e. the two methods disagree;
#   直升机   0.34x  -> 3 of 4 references are backlit silhouettes (5-8x low);
#   地面车辆 / 军用地面车辆 -> 0.66x / 0.60x on a single usable photo each.
# Writing a number we cannot defend is worse than leaving the shared value.
_PAINT_BY_PATH = (
    ('民航飞机', 'paint_civil_air'),
    ('固定翼无人机', 'paint_uav_fixed'),
    ('旋翼无人机', 'paint_uav_rotor'),
    ('军舰', 'paint_warship'),
    ('民用船舶', 'paint_vessel'),
)
_PAINT_BY_CLS = {
    'airliner': 'paint_civil_air', 'uav_fixed': 'paint_uav_fixed',
    'uav_rotor': 'paint_uav_rotor', 'ship': 'paint_warship',
    'boat': 'paint_vessel', 'scenery': 'paint_scenery',
    # Ground vehicles are painted for IR camouflage: a matte, high-absorptance
    # topcoat, not the semi-gloss aircraft finish. Left on painted_metal a tank
    # in sunshine solved to within 1 K of the road and disappeared from the
    # thermal frame.
    'vehicle': 'matte_paint', 'tank': 'matte_paint',
    'artillery': 'matte_paint', 'launcher': 'matte_paint',
    'person': 'matte_paint',
}


def _clutter_block(env, category, span_m, seed, centre=(0.0, 0.0)):
    """Buildings and trees around the target, as real geometry.

    A ground vehicle is rarely standing in an empty field: in a town it is
    screened by buildings and in woodland by canopy. Those occluders have to be
    geometry, not a post-process, or the radar and lidar keep returning the
    vehicle through a wall the optical frame says is there.

    Scenery carries label=False, so it occludes without ever becoming an
    annotation.
    """
    rng = random.Random(int(seed) * 7919 + len(str(category)))
    out = []
    city = (category == 'city')
    n = rng.randint(6, 11) if city else rng.randint(3, 6)
    half = max(float(span_m) * 0.5, 60.0)
    for i in range(n):
        ang = rng.uniform(0.0, 2.0 * math.pi)
        rad = rng.uniform(0.22, 0.95) * half
        x = centre[0] + math.cos(ang) * rad
        y = centre[1] + math.sin(ang) * rad
        if city:
            w = rng.uniform(12.0, 34.0)
            d = rng.uniform(12.0, 34.0)
            h = rng.uniform(8.0, 38.0)
            mat = 'concrete'
            name = 'building_%02d' % (i + 1)
        else:
            w = rng.uniform(4.0, 9.0)
            d = rng.uniform(4.0, 9.0)
            h = rng.uniform(6.0, 16.0)
            mat = 'paint_scenery'
            name = 'tree_%02d' % (i + 1)
        out.append(dict(name=name, cls='scenery', shape='box',
                        size=[round(w, 2), round(d, 2), round(h, 2)],
                        # pos is the BASE of the mesh, not its centre (a box
                        # placed at h/2 floated 17 m up, measured).
                        pos=[round(x, 2), round(y, 2), 0.0],
                        rot=round(math.degrees(rng.uniform(0.0, math.pi)), 2),
                        mat=mat, temp_c=15.0, rcs_db=6.0,
                        visible=True, label=False, yaw_mode='fixed',
                        attitude_mode='none'))
    return out


def _paint_material(asset, cls):
    """The paint family this asset belongs to (see materials.MATERIALS)."""
    text = str(asset or '')
    for needle, mat in _PAINT_BY_PATH:
        if needle in text:
            return mat
    return _PAINT_BY_CLS.get(str(cls or ''), 'painted_metal')


def _entities_for(recipe, site, vel, surface, parent=None):
    """The ECM entities for a recipe: user entities win, else mode defaults."""
    ents = [dict(e) for e in recipe['ecm']['entities']]
    if ents:
        life = {'start_s': 0.0, 'fade_s': 0.05}
        return [_complete_entity(e, site, vel, life, k, parent=parent)
                for k, e in enumerate(ents)]
    mode = recipe['ecm']['mode']
    life = {'start_s': 0.0, 'fade_s': 0.05}
    tgt = [round(v, 3) for v in site['tgt']]
    cam = site['cam']
    perp = site['perp']
    out = []
    if mode in ('active', 'mixed'):
        # standoff jammer: the calibration distance is measured from the sensor,
        # so the pod sits at 500 m along the line of sight, not next to target
        jam = [round(cam[i] + site['unit'][i] * ECM_JAMMER_RANGE_M, 3)
               for i in range(3)]
        out.append(dict(
            id='escort_jammer', kind='noise_jammer', pos=jam,
            vel=[0.0, 0.0, 0.0], erp_w=ECM_JAMMER_ERP_W,
            jam_bandwidth_mhz=200.0, mode='barrage', band=ECM_JAMMER_BAND,
            look_at_sensor=True, sidelobe_db=-25.0, off_band_factor=0.05,
            rcs_m2=0.2, size=[0.35, 2.4, 0.35], mat='painted_metal',
            visible=True, life=dict(life), note=ECM_JAMMER_NOTE))
        out.append(dict(
            id='repeater_01', kind='false_target', pos=list(tgt),
            vel=list(vel), count=4, range_step_m=210.0, range_offset_m=150.0,
            power_ratio=1.6, band=ECM_JAMMER_BAND, visible=True,
            life=dict(life),
            note='DRFM repeater carried by the target; pulls four false ranges'))
    if mode in ('passive', 'mixed'):
        chaff = [round(site['tgt'][i] - site['unit'][i] * 60.0 + perp[i] * 20.0,
                       3) for i in range(3)]
        out.append(dict(
            id='chaff_01', kind='chaff_cloud', pos=chaff, vel=list(vel),
            radius_m=6.0, growth_mps=7.0, rcs_m2=140.0, decay_s=6.0,
            dilution_exponent=2.0, fall_mps=1.5, wind=[6.0, 0.0, 0.0],
            aspect=[1.0, 1.0, 1.0], band=ECM_JAMMER_BAND, mat='metal_bare',
            visible=True, life=dict(life),
            note='dispensed from the airframe, blooms and sinks behind it'))
        corner = [round(site['tgt'][i] + perp[i] * 30.0, 3) for i in range(3)]
        out.append(dict(
            id='corner_reflector', kind='corner_reflector', pos=corner,
            vel=list(vel), size_m=0.3, rcs_m2=None, band=ECM_JAMMER_BAND,
            mat='metal_bare', visible=True, life=dict(life),
            note='0.3 m trihedral: 28.8 dBsm at Ku, deliberately brighter than '
                 'the airframe it decoys'))
    return out


def _entity_block(recipe):
    """Entire ecm block a recipe would expand to, for early validation."""
    primary = recipe['objects'][0]
    surface = primary['cls'] in SURFACE_CLS
    site = _site(recipe['environment']['kind'],
                 recipe['view']['range_km'] * 1000.0, surface)
    return {'enabled': True,
            'entities': _entities_for(recipe, site, [0.0, 0.0, 0.0], surface)}


def _check_entities(recipe):
    from . import ecm as _ecm
    block = _entity_block(recipe)
    if not block['entities']:
        raise ValueError('ecm.mode 为 %s，但没有任何实体：请提供 ecm.entities'
                         % recipe['ecm']['mode'])
    try:
        _ecm.load(block)
    except ValueError as exc:
        raise ValueError('ecm.entities 不合法：%s' % exc) from None


# --- pulse-Doppler self-consistency -------------------------------------------
# Four numbers decide a PD radar's operating point and none of them is free:
#
#     r_max = c*n/(4B)        range window
#     n     = fs*T            samples per chirp
#     PRI   = T/duty          duty-cycle ceiling
#     v_nyq = lam/(4*PRI)     unambiguous closing speed
#
# Substituting gives v_nyq = lam*duty*fs/(16*B*r_max).  So for a fixed hardware
# sample rate the ONLY knob that buys velocity without giving up range is
# bandwidth -- and bandwidth is range resolution.  Hard-coding chirp/pri/samples
# is what produced a 3.07 km window for an 8.2 km target and a Doppler axis that
# folded 250 m/s into 50 m/s.  The schedule below is derived instead.
RADAR_FS_HZ = 250e6          # ADC ceiling of the modelled receiver
RADAR_DUTY = 0.5             # chirp / PRI ceiling (50 % LPI-style waveform)
# Worst range resolution the schedule will trade away for an unambiguous
# Doppler. Coarse range cells only make the target occupy fewer cells; a folded
# Doppler axis puts it in the wrong cell entirely, which is the more damaging of
# the two for a training set. 30 m is the hard stop -- beyond that the aircraft
# stops being resolvable in range at all.
RADAR_RES_FLOOR_M = 30.0
RADAR_FC_GHZ = {'X': 9.6, 'Ku': 15.5, 'K': 24.0, 'Ka': 35.0, 'W': 94.0}


def _radar_waveform(range_m, v_req, band='Ku', bandwidth_mhz=RADAR_BANDWIDTH_MHZ):
    """Derive (n_samples, bandwidth, chirp_us, pri_us, v_nyq) from physics.

    Starts at the bandwidth the scene asked for and gives it up -- down to
    RADAR_RES_FLOOR_M -- only as far as needed to fit the closing speed inside
    the unambiguous Doppler window.  When even the floor cannot carry the speed
    the result reports unambiguous=False instead of pretending.
    """
    r_need = SAMPLE_MARGIN * float(range_m)
    lam = C_LIGHT / (RADAR_FC_GHZ.get(str(band), 15.5) * 1e9)
    fs, duty = RADAR_FS_HZ, RADAR_DUTY
    b_floor = C_LIGHT / (2.0 * RADAR_RES_FLOOR_M)
    b = max(float(bandwidth_mhz) * 1e6, b_floor)
    v_req = float(v_req)

    best = None
    for _ in range(60):
        n_need = 4.0 * b * r_need / C_LIGHT
        power = int(math.ceil(math.log2(max(n_need, float(SAMPLE_MIN)))))
        count = min(SAMPLE_MAX, max(SAMPLE_MIN, 1 << power))
        t_chirp = count / fs
        pri = t_chirp / duty
        v_nyq = lam / (4.0 * pri)
        covers = C_LIGHT * count / (4.0 * b) >= r_need
        if covers and v_nyq >= v_req:
            best = (count, b, t_chirp, pri, v_nyq)
            break
        prev = best
        best = (count, b, t_chirp, pri, v_nyq)
        if b <= b_floor * 1.0001:
            # Floor reached and the speed still does not fit: keep the widest
            # window we can and let the caller say so.
            break
        b = max(b_floor, b * 0.9)

    count, b, t_chirp, pri, v_nyq = best
    # Never return a window that cannot see the target: widen the band rather
    # than hand back a schedule whose echo falls outside the samples.
    if C_LIGHT * count / (4.0 * b) < r_need:
        b = C_LIGHT * count / (4.0 * r_need)
    return dict(n_samples=int(count),
                bandwidth_mhz=round(b / 1e6, 3),
                chirp_us=round(t_chirp * 1e6, 3),
                pri_us=round(pri * 1e6, 3),
                v_nyq_mps=round(v_nyq, 2),
                unambiguous=bool(v_nyq >= v_req),
                range_res_m=round(C_LIGHT / (2.0 * b), 2),
                r_max_m=round(C_LIGHT * count / (4.0 * b), 1))


# --------------------------------------------------------------------- sensors
def _radar_window(range_m, bandwidth_mhz=RADAR_BANDWIDTH_MHZ):
    """(n_samples, bandwidth_mhz) whose unambiguous range covers the scene.

    Two knobs set the window, and both are honoured here instead of hard-coded:

      * samples -- n >= 4*B*1.4*range/c, rounded up to a power of two for the
        FFT (8 km at 50 MHz -> 8192, 16 km -> 16384);
      * bandwidth -- only when even SAMPLE_MAX cannot reach the range (a 50 MHz
        radar tops out at 49 km with 32768 samples) the band is narrowed so the
        window still covers the engagement, at the cost of range resolution.

    Leaving the count at a constant is what produced r_max = 3.07 km for an
    8.2 km target: the echo never landed inside the sampling window.
    """
    want = SAMPLE_MARGIN * float(range_m)
    band = float(bandwidth_mhz) * 1e6
    need = 4.0 * band * want / C_LIGHT
    power = int(math.ceil(math.log2(max(need, float(SAMPLE_MIN)))))
    count = min(SAMPLE_MAX, max(SAMPLE_MIN, 1 << power))
    if C_LIGHT * count / (4.0 * band) < want:
        band = C_LIGHT * count / (4.0 * want)
    return count, round(band / 1e6, 3)


def _radar_range_res(range_m, closing_mps):
    """Range resolution r_res = c/(2B) of the waveform _sensors() will emit.

    The nadir height ladder has to use the same cell height as the emitted
    sensors.radar block. Running the waveform derivation here -- instead of
    reading a hand-written constant -- keeps the two in sync, and avoids
    building the sensor block before the objects exist.

    Falls back to RADAR_RES_FALLBACK_M when no positive bandwidth comes out
    (a missing or nonsensical radar configuration), so a scene still builds
    instead of raising on a geometry guard.
    """
    wave = _radar_waveform(float(range_m), abs(float(closing_mps)))
    band = float(wave.get('bandwidth_mhz') or 0.0) * 1e6
    res = (C_LIGHT / (2.0 * band)) if band > 0.0 else float('nan')
    if not math.isfinite(res) or res <= 0.0:
        return float(RADAR_RES_FALLBACK_M)
    # Same two-decimal rounding the emitted sensors.radar.range_res_m carries,
    # so the ladder and the config the operator reads quote one number.
    return max(0.01, round(float(res), 2))


def _nadir_ladder_plan(specs, cam_z, range_m, speed):
    """Numbers that decide the top_down height ladder, or None when it is moot.

    Why a ladder at all: in a nadir plate view the optical and radar boresights
    are vertical, so a level track has no component along the line of sight.
    The closing rate is therefore ~0 for every target and the RD map has no
    Doppler axis to separate them by -- they can only be told apart in RANGE.
    A range cell is r_res = c/(2B) tall, so neighbours need
    TOP_DOWN_STAGGER_CELLS cells between them.

    The step is measured in SLANT RANGE. Height is a slightly shorter lever for
    a target off the boresight (dR/dz = cos(tilt)), so a ladder sized in height
    could land under the requirement; the caller solves z from the slant target
    instead. Room is capped in height because raising the whole formation
    toward the lens shrinks its angular margin in a narrow frame.

    specs are normalised recipe objects; an explicit z is excluded, and an
    object with a per-target route counts once because plan.py flies one
    airframe per route.
    """
    counts = [1 if s.get('trajectory') else int(s.get('count') or 1)
              for s in specs]
    stagger_n = sum(c for j, c in enumerate(counts)
                    if specs[j].get('z') is None)
    if stagger_n <= 1:
        return None
    r_res = _radar_range_res(range_m, speed)
    need = TOP_DOWN_STAGGER_CELLS * r_res * TOP_DOWN_STAGGER_GUARD
    room = TOP_DOWN_STAGGER_FRAC * float(cam_z) * TOP_DOWN_TILT_GUARD
    step = need
    capped = False
    if need * (stagger_n - 1) > room:
        step = room / float(stagger_n - 1)
        capped = True
    resolvable = stagger_n
    # Two different separations: what the eye needs (need, the step) and what
    # physics needs (cell, the promise behind "resolvable").
    cell = TOP_DOWN_STAGGER_PROMISE_CELLS * r_res * TOP_DOWN_STAGGER_GUARD
    if capped and cell > 0.0:
        # The height budget, not the cell height, now decides how many targets
        # still land in separate cells.
        resolvable = max(1, min(stagger_n, 1 + int(room // cell)))
    return dict(n=stagger_n, step=step, need=need, cell=cell, r_res=r_res,
                room=room, capped=capped, resolvable=resolvable, counts=counts)


def _nadir_separation_note(r):
    """'俯视错层·雷达可分辨 3 个目标' for the UI summary, or None.

    Mirrors the conditions build_scene uses: an explicit top_down aspect, a
    radar among the sensors, a plate to place on, and a ladder that actually
    applies. Cheap because validate_recipe already ran.
    """
    view = r['view']
    if view.get('aspect') != 'top_down':
        return None
    if 'radar' not in r['capture']['sensors']:
        return None
    bg = (r['environment'].get('background') or {}).get('mode')
    if bg not in ('library', 'upload'):
        return None                     # build_scene falls back to an engagement
    span = float(view.get('ground_span_m') or 0.0) or 600.0
    fov = float(view['fov_deg'])
    cam_z = (span / 2.0) / max(math.tan(math.radians(fov) / 2.0), 1e-6)
    primary = r['objects'][0]
    tj = view.get('trajectory') or {}
    if tj.get('mode') == 'drawn':
        speed = float(tj['speed_mps'])
    else:
        speed = float(view.get('closing_mps')
                      or CLOSING_MPS.get(primary['cls'], CLOSING_MPS['target']))
    plan = _nadir_ladder_plan(r['objects'], cam_z, cam_z, speed)
    heights = [s.get('z') for s in r['objects']]
    explicit = any(h is not None for h in heights)
    distinct = len({round(float(h), 3) for h in heights if h is not None}) > 1
    if not plan:
        # A re-saved scene carries the ladder as explicit z, so no new ladder
        # applies -- but those distinct altitudes are still what separates the
        # formation, and the summary should not go silent about it.
        return '俯视错层·雷达按高度分离' if explicit and distinct else None
    if explicit:
        # An explicit altitude breaks the regular chain, and this one-line
        # summary cannot solve the per-target placement; the scene's own
        # cfg["notes"] carries the measured count.
        return '俯视错层·雷达按高度分离'
    return '俯视错层·雷达可分辨 %d 个目标' % plan['resolvable']


def _sensors(cam, look, fov, range_m, site, v_req=0.0, up=None, env=None, plate=False):
    pos = [round(v, 3) for v in cam]
    aim = [round(v, 3) for v in look]
    camera = dict(pos=list(pos), look_at=list(aim), fov_deg=round(fov, 4),
                  width=960, height=720)
    if up is not None:
        # Looking straight down makes the default up (0,0,1) parallel to the view
        # direction, so the basis is degenerate and the frame comes out rolled:
        # a target placed at image v=0.72 landed at 0.79 (measured).
        camera["up"] = [float(v) for v in up]
    wave = _radar_waveform(range_m, v_req)
    radar = dict(band='Ku', az_span_deg=2.0, el_span_deg=1.6, n_az=40, n_el=20,
                 bandwidth_mhz=wave['bandwidth_mhz'], n_samples=wave['n_samples'],
                 chirp_us=wave['chirp_us'], pri_us=wave['pri_us'],
                 # Carried through so the README and the preflight can state the
                 # unambiguous window instead of leaving the operator to derive it.
                 v_nyq_mps=wave['v_nyq_mps'], unambiguous=wave['unambiguous'],
                 range_res_m=wave['range_res_m'], r_max_m=wave['r_max_m'],
                 beamwidth_el_deg=6.0, n_chirps=64, tx_power_dbm=43.0,
                 noise_figure_db=8.0, clutter_density=0.0,
                 # MTI is clutter rejection, not a target filter. A sky scene has
                 # no ground to notch, and a non-zero gate there only deletes the
                 # targets a plate geometry leaves near zero radial velocity
                 # (measured: three aircraft flying level over a plate dipped
                 # below 0.8 m/s in 3 of 8 frames and vanished from each of them,
                 # rd_map peak still 13 dB over the floor). Ground and sea keep
                 # the gate: there it is doing the job it exists for.
                 # MTI rejects stationary clutter; in a plate view the line of
                 # sight is near-vertical, so a vehicle driving toward the plate
                 # centre at 15 m/s shows 0.3 m/s of closing and the 0.8 m/s gate
                 # deletes it -- the ground plate came back with zero radar
                 # detections on every frame. A plate has no ground clutter at
                 # all (the backdrop is a photograph), so the gate buys nothing.
                 mti_min_vr_mps=(0.0 if (str(env) == 'sky' or plate) else 0.8),
                 pfa=1e-06, swerling='I', clutter_db=-12.0,
                 ground_clutter_db=-35.0, pos=list(pos), look_at=list(aim),
                 up=[0.0, 0.0, 1.0])
    # lidar's two range fields do different jobs and both scale with the scene:
    # max_range_m is the hard geometric cut, range_spec_m only anchors the noise
    # floor (the datasheet range for a 10 % target), so a target slightly beyond
    # the spec still returns a weaker echo instead of a second hard truncation.
    lidar = dict(pos=list(pos), look_at=list(aim), fov_deg=round(fov, 4),
                 width=960, height=720, n_az=256, n_el=192,
                 max_range_m=round(max(1500.0, 1.6 * range_m), 1),
                 range_spec_m=round(1.1 * range_m, 1),
                 pulse_energy_nj=4000.0, aperture_mm=50.0)
    sar = dict(band='X', fc_ghz=9.6, bandwidth_mhz=300.0, prf_hz=1200.0,
               n_pulses=512, n_samples=512, n_cross=512,
               altitude_m=round(max(600.0, site['cam_z']), 1), speed_mps=120.0,
               heading_deg=90.0, squint_deg=90.0, side='right',
               swath_center_m=round(0.3 * range_m, 1), swath_width_m=120.0,
               az_extent_m=120.0, polarization='VV', tx_power_w=40.0,
               noise_figure_db=4.0, loss_db=3.0, antenna_len_m=1.2,
               antenna_h_m=0.35, resolution_scale=1.0, looks=2, oversample=2,
               anchor=[round(v, 3) for v in aim])
    return {'optical': camera, 'ir_camera': dict(camera), 'radar': radar,
            'lidar': lidar, 'sar': sar}


# ---------------------------------------------------------------- background
# Weather presets that have no HDRI category of their own fall back to the one
# that looks closest; mixed/evolve are pool modes, not single skies.
HDRI_WEATHER = {'mixed': 'cloudy', 'evolve': 'cloudy'}


def backdrop_library():
    """Plates available per category, for the picker and the preflight."""
    out = {}
    for cat in BACKDROP_CATEGORIES:
        d = os.path.join(BACKDROP_DIR, cat)
        files = []
        if os.path.isdir(d):
            files = sorted(f for f in os.listdir(d)
                           if f.lower().endswith((".jpg", ".jpeg", ".png")))
        out[cat] = files
    return out


def backdrop_meta():
    """Per-file metadata from the collected catalog, keyed by relative path.

    The catalog records whether a plate is a true aerial (near-nadir) or an
    oblique view. That matters: an oblique plate has its own perspective baked
    in, so placing a target on it under a straight-down camera looks wrong.
    """
    fp = os.path.join(BACKDROP_DIR, "catalog.json")
    out = {}
    try:
        with open(fp, "r", encoding="utf-8") as fh:
            cat = json.load(fh)
    except (OSError, ValueError):
        return out
    for item in (cat.get("images") or []):
        rel = str(item.get("file") or "").replace("\\", "/")
        if rel:
            out[rel] = dict(aerial=bool(item.get("aerial")),
                            title=str(item.get("title") or ""),
                            license=str(item.get("license") or ""),
                            source_url=str(item.get("source_url") or ""))
    return out


def _pick_backdrop(category, name=None, seed=0, prefer_aerial=False):
    """Resolve a library plate. Explicit file wins, otherwise rotate by seed.

    Rotation instead of always the first file: a batch of ground datasets would
    otherwise all carry the identical forest photograph, and the background would
    stop being a source of variety at all.
    """
    d = os.path.join(BACKDROP_DIR, category)
    if name:
        cand = os.path.join(d, name)
        if os.path.isfile(cand):
            return cand
        raise ValueError("背景库中没有 %s/%s" % (category, name))
    files = sorted(f for f in os.listdir(d)
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))) if os.path.isdir(d) else []
    if not files:
        raise ValueError("背景库类别 %s 是空的（assets/backdrops/%s）" % (category, category))
    if prefer_aerial:
        # A straight-down camera over an oblique plate reads wrong: the photo's
        # own perspective fights the render.
        meta = backdrop_meta()
        aerial = [f for f in files if meta.get(category + "/" + f, {}).get("aerial")]
        if aerial:
            files = aerial
    return os.path.join(d, files[int(seed) % len(files)])


def _existing_file(value):
    """Absolute path of an existing file, preferring the path as given.

    A background plate may be absolute, relative to the cwd (an upload written
    by the web layer) or relative to the repository root. Whatever is found is
    returned absolute: the renderers resolve relative paths against the scene
    file's directory, which build_scene cannot know yet, so an absolute path is
    the only spelling that is safe to hand over.
    """
    raw = str(value).strip()
    candidates = [os.path.expanduser(raw)]
    if not os.path.isabs(raw):
        candidates.append(os.path.join(ROOT, raw))
    for cand in candidates:
        path = os.path.abspath(cand)
        if os.path.isfile(path):
            return path
    return None


# One plate per weather, so 晴 / 多云 / 雨 / 雾 give visibly different skies.
# The UI sends whatever id it last used, so the weather table is consulted FIRST:
# otherwise picking "cloudy" kept rendering the clear plate and the weather
# control had no visible effect at all.
WEATHER_HDRI = {
    'clear':  ('kloofendal_43d_clear_puresky',
               'kloofendal_48d_partly_cloudy_puresky'),
    'cloudy': ('kloofendal_48d_partly_cloudy_puresky', 'wasteland_clouds_puresky'),
    'rain':   ('kloofendal_overcast_puresky', 'overcast_soil_puresky'),
    'fog':    ('kloofendal_28d_misty_puresky', 'kloofendal_misty_morning_puresky'),
    'mixed':  ('kloofendal_48d_partly_cloudy_puresky',),
    'evolve': ('kloofendal_48d_partly_cloudy_puresky',),
}


def _hdri_path(asset_id, weather):
    """Local .hdr for this weather, else for the given id; None when unavailable."""
    from . import assets_fetch
    try:
        cached = {str(e.get('id')): e.get('path')
                  for e in assets_fetch.list_cached() if e.get('kind') == 'hdri'}
    except Exception:
        cached = {}
    for cand in WEATHER_HDRI.get(weather, ()):
        p = cached.get(cand)
        if p and os.path.isfile(p):
            return p
    if asset_id and cached.get(str(asset_id)):
        p = cached[str(asset_id)]
        if os.path.isfile(p):
            return p
    got = assets_fetch.hdri_for_weather(HDRI_WEATHER.get(weather, weather))
    if isinstance(got, dict):
        path = got.get('path')
        if path and os.path.isfile(path):
            return path
    return None


def _apply_background(cfg, environment, wanted, notes):
    """Translate environment.background into the keys the renderers actually read.

    Exactly two keys carry a background, and both live in RENDER blocks, never in
    the sensor geometry blocks:

        cfg['optical']['sky_image']        blender/render_optical.py: opt = cfg
                                           .get("optical"), then opt.get(...)
        cfg['ir']['background_image']      simkit/ir.py: ircfg = cfg.get("ir")

    Putting the path on sensors['optical'] instead looks right and is silently
    ignored -- a measured A/B render produced bit-identical pixels to the
    gradient sky -- so this function writes only the two keys above and nothing
    on cfg['sensors'] at all. Radar, lidar and SAR never get a background: a sky
    behind a range-Doppler map is not a thing.
    """
    background = environment['background']
    mode = background['mode']
    if mode == 'none':
        return
    path = None
    if mode == 'upload':
        # A missing plate is a hard error on purpose: falling back to the
        # gradient would ship a dataset whose background is not the chosen one.
        path = _existing_file(background['path'])
        if path is None:
            raise ValueError('背景图不存在：%s' % background['path'])
    elif mode == 'library':
        # top_down placement needs a near-nadir plate; the look-down engagement
        # can use anything.
        _prefer_aerial = bool((cfg.get('ground') or {}).get('backdrop_aligned')) or \
            bool(((cfg.get('clip') or {}).get('seeker') or {}).get('depression_deg') is None
                 and not (cfg.get('clip') or {}).get('seeker'))
        path = _pick_backdrop(background['category'],
                              background.get('file'),
                              int(cfg.get('seed') or 0),
                              prefer_aerial=_prefer_aerial)
        # A look-down frame shows the ground, so the plate has to become the
        # ground material; sky_image alone changes nothing when the camera looks
        # down at terrain.
        _g = cfg.setdefault('ground', {})
        if _g.get('enabled', True):
            _g['backdrop'] = path
            _span = float(background.get('span_m')
                          or (cfg.get('view') or {}).get('ground_span_m') or 300.0)
            _g['backdrop_span_m'] = _span
    elif mode == 'hdri':
        try:
            path = _hdri_path(background.get('id'), environment['weather'])
        except Exception as exc:          # offline, no cache, HTTP error, ...
            notes.append('内置天空不可用，已回退渐变天空（%s: %s）'
                         % (type(exc).__name__, exc))
        if path is None and not notes:
            notes.append('内置天空不可用，已回退渐变天空（本地没有可用的 HDRI）')
    if not path:
        return
    want = set(wanted or ())
    optical = cfg.get('optical')
    if 'optical' in want and isinstance(optical, dict):
        optical['sky_image'] = path
    ir_cfg = cfg.get('ir')
    if 'ir' in want and isinstance(ir_cfg, dict):
        ir_cfg['background_image'] = path


# ----------------------------------------------------------------- scene build
def build_scene(recipe):
    """Expand a recipe into a complete scene config.

    The result is plain JSON and is accepted by scene.load_config followed by
    scene.build_world; environment.kind genuinely changes the world (no ground /
    a plane / a wave mesh) and clip.seeker encodes the aspect.
    """
    r = validate_recipe(recipe)
    env = r['environment']['kind']
    weather = r['environment']['weather']
    view = r['view']
    cap = r['capture']
    fps = float(cap['fps'])
    frames = int(cap['frames'])
    dt = round(1.0 / fps, 6)
    range_m = view['range_km'] * 1000.0

    primary = r['objects'][0]
    primary_size = _object_size(primary['asset'], primary['scale'])
    surface = primary['cls'] in SURFACE_CLS
    # "auto" lets the environment pick the engagement: aircraft are met head-on
    # or chased, ground and sea targets are looked down on. An explicit aspect
    # still wins, so a curated scene can pin its own geometry.
    aspect_allowed = ASPECT_BY_ENV.get(env, ("head_on", "pursuit"))
    aspect = view["aspect"]
    top_down = (aspect == "top_down")
    _bg_mode = str(((r.get("environment") or {}).get("background") or {}).get("mode") or "")
    _has_plate = _bg_mode in ("library", "upload")
    if top_down and not _has_plate:
        # A plate view needs a plate. Without one there is nothing to click on,
        # so an air scene falls back to its own engagement instead of pointing a
        # camera at the ground under an aircraft.
        top_down = False
        aspect = aspect_allowed[0]
    elif top_down:
        # With a photograph under it the plate view is valid in any domain: the
        # camera frames a picture and the target stands on it. Refusing it for
        # the sky category silently discarded every click coordinate.
        aspect = "top_down"
    elif aspect == "auto" or aspect not in ("head_on", "pursuit", "look_down", "top_down"):
        aspect = aspect_allowed[0]
    elif aspect not in aspect_allowed:
        # Asking for head_on on a ship is not an error the operator can see:
        # a boat cannot be met head-on by a look-down camera, so fall back and
        # let the summary show what was actually used.
        aspect = aspect_allowed[0]
    view['aspect'] = aspect
    dep = depression_for(env, aspect) if not top_down else None
    span_m = float(view.get('ground_span_m') or 0.0) or 600.0
    if top_down:
        # Nadir plate view: the camera sits straight above the plate origin and
        # frames exactly ground_span_m across, so one image pixel maps to one
        # known patch of ground and an operator's click is a real position.
        _half = span_m / 2.0
        _h = _half / max(math.tan(math.radians(float(view['fov_deg'])) / 2.0), 1e-6)
        site = dict(cam=(0.0, 0.0, _h), tgt=(0.0, 0.0, 0.0),
                    unit=(0.0, 0.0, -1.0), az=0.0, horizontal=0.0,
                    cam_z=_h, tgt_z=0.0, perp=(1.0, 0.0, 0.0))
        range_m = _h
    else:
        site = _site(env, range_m, surface, dep)

    lo, hi = range_m * 0.85, range_m * 1.15
    # The closing speed is a property of the engagement, not of the clip: it
    # comes from the class table, or from view.closing_mps when the recipe says
    # so. The frame count decides how much of the range band the target crosses
    # (closing_mps * duration) and plan._plan_seeker clamps that inside
    # range_m, so a two-frame preview is a shorter traverse of the same
    # engagement instead of a Mach 3.5 fighter.
    speed = (float(view['closing_mps']) if view.get('closing_mps')
             else float(CLOSING_MPS.get(primary['cls'], CLOSING_MPS['target'])))
    drawn = (view.get('trajectory') or {}).get('mode') == 'drawn'
    if drawn:
        # A drawn route replaces the authored engagement: the airframe flies the
        # operator's path and the camera is parked where the requested range puts
        # it. Speed comes from the trajectory itself, not from the class table,
        # because the operator already chose how fast the route is flown.
        tj = view['trajectory']
        pts = tj['points']
        speed = float(tj['speed_mps'])
        cen = [sum(p[i] for p in pts) / float(len(pts)) for i in range(3)]
        back = (math.cos(math.radians(LOS_AZ_DEG)), math.sin(math.radians(LOS_AZ_DEG)))
        cam_d = [cen[0] - back[0] * range_m, cen[1] - back[1] * range_m, cen[2]]
        d0 = [pts[1][i] - pts[0][i] for i in range(3)]
        n0 = math.sqrt(sum(v * v for v in d0)) or 1.0
        site = dict(cam=tuple(cam_d), tgt=tuple(pts[0]),
                    unit=tuple(v / n0 for v in d0),
                    az=math.radians(LOS_AZ_DEG), horizontal=range_m,
                    cam_z=cam_d[2], tgt_z=pts[0][2],
                    perp=(-back[1], back[0], 0.0))
    sign = 1.0 if view['aspect'] == 'pursuit' else -1.0
    vel = [round(sign * site['unit'][i] * speed, 3) for i in range(3)]
    yaw = round(math.degrees(math.atan2(vel[1], vel[0])), 3)

    # ---- nadir multi-target height ladder --------------------------------
    # A top_down plate view is the one geometry where a multi-target radar
    # picture cannot work by itself: the camera looks straight down, the placed
    # targets share one plane and fly level, so the line of sight is
    # perpendicular to every track, the radial velocity is ~0 and Doppler
    # separates nothing. Range is the only axis left (a cell is c/(2B) tall),
    # hence the height ladder: height IS the line of sight here. The step is
    # solved per target in slant range, so a target off the boresight cannot
    # turn a metre of height into less than a metre of range.
    #
    # Velocity separation would need a line of sight that is not perpendicular
    # to the track: head_on / pursuit do that for an airframe, and look_down is
    # the recipe entry for a depressed view of a surface target (ASPECTS carries
    # it and _site() puts the camera on the requested depression ray). No nadir
    # plate view can produce it, which is why this ladder exists and no Doppler
    # trick is attempted.
    #
    # The budget is a fraction of the camera height: the optical frame is only
    # ground_span_m across, and stacking the formation toward the lens eats its
    # angular margin. An explicit objects[].z is never overwritten.
    ladder = None
    if top_down and 'radar' in cap['sensors']:
        _lplan = _nadir_ladder_plan(r['objects'], float(site['cam_z']),
                                    range_m, speed)
        if _lplan:
            ladder = dict(_lplan)
            ladder.update(prev=None, slants=[],
                          writeback=all(c == 1 for c in _lplan['counts']))

    objects = []
    # Messages about route extension, collected while the objects are placed
    # (the note block at the end reports them).
    _route_notes = []
    counter = 0
    cursor = 0.0
    for j, spec in enumerate(r['objects']):
        size = _object_size(spec['asset'], spec['scale'])
        rcs_db, temp_c = _object_facts(spec['asset'], spec['cls'])
        # Formation spacing has to fit the frame: a fixed 4x-wingspan gap put two
        # aircraft on the opposite edges of a 3 km / 2.2 deg frame. Cap it to a
        # fraction of the frame width at this range, and keep a floor so a close
        # pair does not overlap.
        frame_w = (2.0 * float(range_m)
                   * math.tan(math.radians(float(view['fov_deg'])) / 2.0))
        spacing = min(max(60.0, 4.0 * size), max(20.0, 0.30 * frame_w))
        count = int(spec['count'])
        # A placed object carries view-space coordinates from the operator's click
        # on the plate; that is a real position, not a formation offset.
        _uv = spec.get('uv')
        _route = spec.get('trajectory')
        # The envelope this object will be judged against, built exactly the way
        # the checker builds it (class defaults + this scene's kinematics block),
        # so the route is shaped for the rules it will actually meet.
        _kin = _envelope(spec['cls'], env)
        _env = motion.Envelope.from_spec(dict(cls=spec['cls'], kinematics=_kin))
        if _route:
            # An explicit per-target route means one airframe per entry: copying
            # it count times is how two aircraft ended up stacked at the same
            # pixel and the frame looked like it had one.
            count = 1
        for k in range(count):
            counter += 1
            lateral = cursor + (k - (count - 1) / 2.0) * spacing
            if _uv:
                _px, _py = float(_uv[0]), float(_uv[1])
                if len(_uv) > 2 and count > 1:
                    _px = _px + float(_uv[2]) * (k - (count - 1) / 2.0)
                # v is measured in FRAME heights, and the frame is 4:3, so the
                # plate covers span_m horizontally and span_m*3/4 vertically.
                _span_y = span_m * (720.0 / 960.0)
                _pos = [round((_px - 0.5) * span_m, 3),
                        round((0.5 - _py) * _span_y, 3),
                        round(float(spec.get('z') or 0.0), 3)]
            else:
                _pos = [round(site['tgt'][0] + site['perp'][0] * lateral, 3),
                        round(site['tgt'][1] + site['perp'][1] * lateral, 3),
                        round(site['tgt'][2], 3)]
            if ladder is not None:
                _cam_z = float(site['cam_z'])
                if spec.get('z') is not None:
                    # An explicit altitude wins in a plate view. The uv branch
                    # above already honours it; the formation branch would
                    # otherwise drop it on the plate.
                    _pos[2] = round(float(spec['z']), 3)
                elif ladder['prev'] is None:
                    # The first rung keeps the original placement height; the
                    # rest hang above it, toward the lens. Going below the plate
                    # would bury a ground target under the photograph.
                    _pos[2] = round(float(_pos[2]), 3)
                else:
                    # Solve z for an exact slant-range step:
                    #   R = sqrt(r_xy^2 + (cam_z - z)^2),  R = R_prev - step
                    _r2 = float(_pos[0]) ** 2 + float(_pos[1]) ** 2
                    _want = ladder['prev'] - ladder['step']
                    _dz_sq = _want * _want - _r2
                    _z = ((_cam_z - math.sqrt(_dz_sq)) if _dz_sq > 0.0 else 0.0)
                    _z = max(0.0, min(_z, ladder['room']))
                    _pos[2] = round(_z, 3)
                _r2 = float(_pos[0]) ** 2 + float(_pos[1]) ** 2
                ladder['prev'] = math.sqrt(_r2 + (_cam_z - float(_pos[2])) ** 2)
                # Every placed target joins the chain, explicit z included: the
                # resolvable count in the summary has to describe the picture,
                # not only the targets the ladder was allowed to move.
                ladder['slants'].append(ladder['prev'])
                if ladder['writeback'] and spec.get('z') is None:
                    # Reflect the ladder in the recipe block build_scene writes
                    # back, so a re-run reproduces these altitudes as explicit
                    # z instead of recomputing a different ladder. Only when
                    # every spec is a single airframe: one spec.z cannot express
                    # the different rungs of a count>1 formation.
                    spec['z'] = round(float(_pos[2]), 3)
            elif top_down and spec.get('z') is not None:
                # No ladder (radar absent): still honour an explicit altitude in
                # a plate view instead of dropping it on the photograph.
                _pos[2] = round(float(spec['z']), 3)
            # In a plate view the placement plane IS the world, so the "velocity
            # along the line of sight" the engagement model produces points
            # straight down: a parked ground vehicle came out as vel [0, 0, 15],
            # i.e. climbing at 15 m/s. With no drawn route, point the object at
            # the plate centre (the place the camera is looking at) at the class
            # speed; a route, when present, overrides this below with its own
            # tangent.
            _vel_obj = list(vel)
            _rot_obj = yaw
            if top_down and not _route:
                _hx, _hy = -float(_pos[0]), -float(_pos[1])
                _hn = math.hypot(_hx, _hy)
                _spd = abs(float(speed))
                if _hn > 1e-6:
                    _vel_obj = [round(_spd * _hx / _hn, 3), round(_spd * _hy / _hn, 3), 0.0]
                    _rot_obj = round(math.degrees(math.atan2(_vel_obj[1], _vel_obj[0])), 3)
                else:
                    _vel_obj = [round(_spd, 3), 0.0, 0.0]
                    _rot_obj = 0.0
            obj = dict(
                name='%s_%02d' % (spec['cls'], counter),
                cls=spec['cls'],
                shape='obj',
                model=MODEL_PREFIX + '/' + spec['asset'],
                pos=_pos,
                vel=_vel_obj,
                rot=_rot_obj,
                # the nose follows the velocity vector, which is what makes
                # head_on mean nose-toward-camera and pursuit tail-toward-it
                yaw_mode='velocity',
                scale=float(spec['scale']),
                mat=_paint_material(spec['asset'], spec['cls']),
                # No hard temperature pin. thermal.solve runs a radiative-
                # convective balance per face using the material's absorptance
                # and internal heat flux; pinning one class number here threw
                # that away and made every airframe isothermal, which is exactly
                # the flat cut-out the thermal model had to fix. The curated
                # per-asset value survives as a hint for hand-authored scenes.
                temp_c=None,
                temp_c_hint=temp_c,
                rcs_db=rcs_db,
                attitude_mode='coordinated',
                kinematics=_kin,
                visible=True,
                label=True,
            )
            if _route:
                # Anchor the route on the placement: point 0 becomes exactly the
                # clicked position and the rest keep their shape. Adding the
                # placement to raw offsets started the aircraft wherever the
                # operator's first stroke landed, one click away from the dot.
                _p0 = _route["points"][0]
                _pts = [[round(_pos[0] + float(p[0]) - float(_p0[0]), 3),
                         round(_pos[1] + float(p[1]) - float(_p0[1]), 3),
                         round(_pos[2] + float(p[2]) - float(_p0[2]), 3)]
                        for p in _route["points"]]
                # Flyability pre-shaping, before the route is published (see
                # motion.shape_route): the stroke is a sketch at plate scale, so
                # its corners are rounded to the room the drawing leaves and the
                # speed is solved down to what those arcs hold -- never below the
                # airframe's own floor, which relaxes the shape instead. The
                # verdict is measured on the flown spline, not on the polyline.
                _shaped = motion.shape_route(
                    _pts, speed=float(_route.get("speed_mps") or speed),
                    envelope=_env, dt=float(dt), frames=max(int(frames), 4))
                _inf = _shaped["info"]
                _v = float(_shaped["speed_mps"])
                obj["trajectory"] = dict(
                    mode="waypoints",
                    points=_shaped["points"],
                    speed_mps=_v,
                    loop=bool(_route.get("loop", False)),
                    # a drawn route is a cruise leg: enter at speed, only the
                    # arrival is flown down
                    ramp_in=bool(_route.get("ramp_in", False)),
                    # Record the shaping instead of implying it: a reader of the
                    # scene can see the speed the geometry allowed, the radius
                    # that was fitted, and the route length those produced.
                    preshaped=True,
                    speed_requested_mps=round(float(_inf["speed_requested_mps"]), 1),
                    fillet_radius_m=round(float(_inf["radius_room_m"]), 1),
                    length_m=round(float(_inf["length_m"]), 1))
                for _n in _inf["notes"]:
                    _route_notes.append("%s %s" % (obj["name"], _n))
                # A short stroke on the plate would otherwise stop the airframe
                # a fraction of the way through the clip: extend it along its
                # final heading so the route covers the clip at cruise speed.
                # The plate frame is span_m wide and (4:3) span_m * 0.75 tall,
                # centred on the origin; keep the extended route inside it.
                _fh = (0.5 * float(span_m), 0.5 * float(span_m) * 0.75) if top_down else None
                _added = _extend_cruise_route(obj["trajectory"],
                                              float(frames) * float(dt), _fh)
                if _added > 0.0:
                    # A 7 m drawn leg beside a 130 m cruise leg asks the spline
                    # for a 68 m knot tangent inside a 7 m cell: the same route
                    # that measured 32037 m/s^3 measured 127858 m/s^3 once the
                    # leg was appended. Cut the appended leg to the drawing's own
                    # cell size, which is geometry, not a tuning constant.
                    obj["trajectory"]["points"] = motion.densify_route(
                        obj["trajectory"]["points"],
                        cell_m=float(_inf.get("cell_m") or 0.0) or None)
                    obj["trajectory"]["length_m"] = round(float(
                        sum(math.dist(a, b) for a, b in zip(
                            obj["trajectory"]["points"], obj["trajectory"]["points"][1:]))), 1)
                    _route_notes.append(
                        "%s 航路延长 %.0f m（原航路比 %.0f s × %.0f m/s 短，"
                        "否则中途停住）" % (obj["name"], _added, float(frames) * float(dt),
                                          float(obj["trajectory"]["speed_mps"])))
                # objects[].vel used to be the view's line-of-sight velocity: on
                # a nadir plate that is [0, 0, 250], i.e. straight up at 250 m/s
                # for an airframe the same scene flies at 80 m/s toward the east.
                # Publish the route's own initial tangent instead, so vel and
                # trajectory cannot disagree and yaw_mode='velocity' starts on
                # the heading the route actually begins with.
                _v0 = motion.route_heading(obj["trajectory"]["points"]) * _v
                obj["vel"] = [round(float(c), 3) for c in _v0]
                obj["rot"] = round(math.degrees(math.atan2(float(_v0[1]),
                                                           float(_v0[0]))), 3)
            if surface:
                obj['z_offset_m'] = 0.0
            objects.append(obj)
        cursor += count * spacing + max(120.0, 2.0 * size)

    ladder_info = None
    if ladder is not None:
        # Resolvable means "the longest chain of targets whose adjacent slant
        # ranges are at least TOP_DOWN_STAGGER_CELLS cells apart". The step is
        # normally exact, so this equals the target count; a capped ladder -- or
        # one interrupted by an explicit z -- loses rungs and the summary has to
        # say the honest number instead of advertising the requested one.
        _slants = sorted(ladder['slants'])
        # Compare against the unguarded 8-cell promise, not the 1-per-mille
        # step guard, or a millimetre of z rounding reads as a lost target.
        _res = 1 + sum(1 for a, b in zip(_slants, _slants[1:])
                       if b - a >= ladder['cell'] - 1e-6)
        ladder_info = dict(count=len(_slants),
                           moved=ladder['n'],
                           step_m=round(ladder['step'], 2),
                           r_res_m=round(ladder['r_res'], 3),
                           room_m=round(ladder['room'], 1),
                           capped=bool(ladder['capped']),
                           resolvable=int(min(_res, len(_slants))))

    fov = view['fov_deg']
    px = _pixels(primary_size, range_m, fov, 960.0)
    seeker = dict(
        mode='seeker',
        range_m=[round(lo, 1), round(hi, 1)],
        # A look-down engagement closes too: the camera is above the target and
        # the range shrinks as it flies at it. "out" here used to make ground
        # scenes recede, which no operator asked for.
        closing='out' if view['aspect'] == 'pursuit' else 'in',
        closing_mps=round(speed, 3),
        target_m=round(primary_size, 3),
        target_px=[round(px * 0.7, 1), round(px * 1.4, 1)],
        az_span_deg=0.6,
        el_span_deg=0.3,
        los_jitter_deg=0.04,
        wander_deg=round(min(0.35 * fov, 4.0), 3),
        wander_hz=0.09,
        keep_z=bool(surface),
        lag_s=0.02,
        profile='linear',
        aspect=view['aspect'],
    )
    if dep is not None:
        seeker['depression_deg'] = round(float(dep), 3)
    views = [dict(name='seeker', tag='', fov_deg=round(fov, 4),
                  width=960, height=720)]
    if view['views'] == 'dual':
        views.append(dict(name='wide', tag='_wide', fov_deg=18.0,
                          width=1280, height=960))

    # Occluders around the target. Derived from the backdrop category so a city
    # plate gets buildings and a forest plate gets trees; both are real geometry,
    # which is the only way radar and lidar stop seeing through them.
    #
    # Not in a plate view. A top-down plate is a photograph of terrain that
    # already contains its own buildings; pasting flat grey boxes on top of it
    # reads as cardboard cut-outs and destroys the plate's realism, which is the
    # whole point of collecting the photographs. Occlusion in that mode is the
    # operator's job -- put the target behind a building that is already in the
    # picture -- and the non-plate scenes keep their geometry occluders.
    _cat = str(((r.get('environment') or {}).get('background') or {}).get('category') or '')         if (r.get('environment') or {}).get('background') else ''
    if env in ('ground', 'sea') and _cat and not top_down:
        _span = float(view.get('ground_span_m') or (2.0 * range_m)) or 600.0
        objects.extend(_clutter_block(env, _cat, _span, int(cap['seed'])))
    wind = float(WIND_MS.get(weather, 8.0))
    cfg = {
        'name': r['name'],
        'seed': int(cap['seed']),
        'ground': _ground_block(env, weather, site),
        'weather': dict(WEATHER_MODES[weather]),
        'sun': dict(elev_deg=38.0, az_deg=120.0,
                    irradiance=float(WEATHER_SUN[weather]['irradiance']),
                    on=bool(WEATHER_SUN[weather]['on'])),
        'air': {'temp_c': SITE[env]['air_temp_c'], 'rh': SITE[env]['rh'],
                'wind_mps': wind},
        'ir': {'bands': ['mwir', 'lwir'], 'netd_mk': 28.0, 'psf_px': 1.3,
               # Workbench switch: no engine heat means no nozzle override and no
               # exhaust cone, i.e. the same airframe rendered as a cold body.
               'engine_heat': bool((r.get('effects') or {}).get('plume', False)),
               # 0 keeps the renderer default (one sample); 2-4 antialias the
               # plume, which is only a few pixels wide at engagement range.
               **({'samples_per_axis': int((r.get('effects') or {})['supersample']),
                   'subpixel_mode': 'coverage'}
                  if int((r.get('effects') or {}).get('supersample', 0)) >= 2 else {}),
               # The preview runs an automatic gain control over the frame, not a
               # fixed 20 K window. A 20 K window around the median was chosen so
               # the 28 mK NETD survives into the 8-bit sheet, but it also clipped
               # every hot part of the target to flat white: the exhaust, the
               # nozzle and the canopy all read 255 and the image looked like a
               # cut-out. NETD is still measured on the 16-bit product, which is
               # what a detector actually consumes.
               'preview_span_c': None,
               'nuc': {'enabled': True, 'prnu': 0.008, 'dsnu_counts': 25,
                       'column_fpn_counts': 6, 'dead_px': 0.0002,
                       'seed': 12345}},
        'optical': dict(
            samples=16, exposure=-0.6, export_geometry=True,
            # Procedural skin plate: verified to render (a checkerboard plate
            # shows on the airframe). It raises target contrast 0.65x -> 0.72x
            # against the reference photos but lowers high-frequency detail
            # slightly, because the procedural pattern is low-frequency at this
            # range. Worth keeping for the contrast, not a substitute for a
            # photographed texture.
            skin=True,
            # f/2.8, not the module default 5.6. An airborne EO sensor is a fast
            # lens for light gathering, so 2.8 is the more honest figure; and at
            # this 16 um pixel pitch the diffraction OTF never reaches 0.5 before
            # Nyquist, so the aperture cannot push MTF50 out of its 0.20-0.30 band.
            # Measured over five seeds it lifts target detail 0.52x -> 0.56x.
            camera={'diffraction': {'f_number': 2.8}},
            sky_zenith_rgb=list(WEATHER_SKY[weather]['zenith']),
            sky_horizon_rgb=list(WEATHER_SKY[weather]['horizon']),
            sky_strength=float(WEATHER_SKY[weather]['strength']),
            sky_weather=weather),
        'frames': {'n': frames, 'dt': dt},
        'objects': objects,
        'sensors': _sensors(site['cam'], site['tgt'], fov, range_m, site,
                            up=((0.0, 1.0, 0.0) if top_down else None),
                                        v_req=abs(float(speed)), env=env,
                                        plate=bool(top_down)),
        'clip': dict(
            frames_per_clip=frames,
            camera='ir_camera',
            views=views,
            # A drawn route owns the clip; a seeker block beside it would make the
            # planner drag the target along the line of sight again, which is the
            # motion the operator just replaced.  Dropping the seeker also drops
            # the mount that used to follow the target, so a 250 m/s airframe
            # crossed the 307 m wide frame in twelve frames and more than half the
            # clip was empty sky -- hence camera_track, which pans the fixed mount
            # while leaving the route alone.
            **({'trajectory': dict(mode='waypoints',
                                   points=[list(p) for p in view['trajectory']['points']],
                                   speed_mps=float(view['trajectory']['speed_mps']),
                                   loop=bool(view['trajectory']['closed']),
                                   ramp_in=False),
                'camera_track': 'on'}
               if drawn else ({} if top_down else {'seeker': seeker})),
            rig={'lag_s': 0.02},
        ),
        'capture': dict(sensors=list(cap['sensors']), frames=frames, fps=fps,
                        seed=int(cap['seed'])),
        # Real datasets ship JPEG, and the 8x8 block structure is one of the
        # statistics docs/真实数据对比.md asks for (ratio 1.05-1.2). Writing a
        # .jpg beside every .png costs about 3% more disk (measured: q90 is 2.58%
        # of the PNG) while leaving the lossless PNG as the canonical product
        # every downstream script already reads.
        # quality 97, not the conventional 90: the 8x8 blocking ratio is one of
        # the acceptance statistics (docs/真实数据对比.md target 1.05-1.20), and
        # measured on our own frames q90 gives 1.79 -- MORE blocking than the
        # reference photographs (1.22), because a smooth sky shows the grid that
        # real texture hides. q97 measures 1.17, the closest match; it costs 26%
        # of the PNG instead of 6%, and the lossless PNG stays canonical.
        'output': dict(r['output'],
                       jpeg=dict(enabled=True, quality=97)),
        'recipe': r,
        'research_baseline': {
            'purpose': 'generated from a recipe by simkit.scenegen',
            'environment': '%s / %s / %s'
                           % (env, weather,
                              r['environment']['background']['mode']),
            'aspect': ('手绘轨迹 (%d 点，%.0f m/s)'
                       % (len(view['trajectory']['points']),
                          view['trajectory']['speed_mps'])
                       if drawn else
                       '%s (%s)' % (view['aspect'], seeker['closing'])),
            'sensor_calibration': 'nominal simulation settings, not measured '
                                  'sensor calibration',
        },
    }

    notes = []
    if top_down:
        # A nadir plate view looks at bare ground for the whole frame, so the
        # ground plane must be wider than what the lens covers. _ground_block
        # sizes it from the horizontal range, which is zero here, and a 200 m
        # plane under a 600 m frame showed the environment map around the edges
        # as radial streaks through the zenith.
        _gr = cfg.get('ground') or {}
        _need = float(span_m) * 2.5
        if _gr.get('enabled', True) and float(_gr.get('size') or 0.0) < _need:
            _gr['size'] = round(_need, 1)
    _apply_background(cfg, r['environment'], r['capture']['sensors'], notes)
    if not top_down and (cfg.get('ground') or {}).get('backdrop'):
        # Scale the plate so one photo covers roughly one frame. Tiling a desert
        # satellite shot every 300 m laid an obvious seam grid across the picture.
        _frame_w = 2.0 * float(range_m) * math.tan(math.radians(float(fov)) / 2.0)
        if _frame_w > 1.0:
            cfg['ground']['backdrop_span_m'] = round(_frame_w, 1)
    if top_down:
        # Placed targets: the plate becomes the map, so it is centred on the world
        # origin with no rotation and no random offset. Any of those would put the
        # vehicle somewhere other than the pixel the operator clicked.
        _g = cfg.setdefault('ground', {})
        _g['backdrop_aligned'] = True
        _g['backdrop_offset'] = [0.5, 0.5]
        _g['backdrop_span_m'] = float(span_m)
        _g['backdrop_span_y_m'] = float(span_m) * (720.0 / 960.0)
    if top_down:
        cfg['view'] = dict(aspect=aspect, ground_span_m=float(span_m),
                           height_ladder_m=[round(float(o['pos'][2]), 3)
                                            for o in objects],
                           radar_resolvable_targets=(ladder_info['resolvable']
                                                     if ladder_info else None))
    else:
        cfg['view'] = dict(aspect=aspect)
    if ladder_info is not None:
        # The operator has to be told why the formation is stacked in altitude:
        # otherwise a top_down run looks like it ignored "same height".
        _cells = (ladder_info['step_m'] / ladder_info['r_res_m']
                  if ladder_info['r_res_m'] > 0.0 else 0.0)
        notes.append('俯视模式按高度错层：相邻目标斜距相差 %.1f m（约 %.1f 个'
                     '距离单元），雷达可分辨 %d 个目标%s'
                     % (ladder_info['step_m'], _cells,
                        ladder_info['resolvable'],
                        '；错层已达画幅高度上限，部分目标仍可能落在同一距离单元'
                        if ladder_info['resolvable'] < ladder_info['count'] else ''))
    if _route_notes:
        notes.append('；'.join(_route_notes))
    if frames < 3:
        notes.append('帧数过少，轨迹只有 %d 个采样点' % frames)
    # --- co-boresight and sector fit (radar/lidar vs the imaging view) --------
    # A plate scene looks straight down, where a (0,0,1) up vector is parallel to
    # the view axis. sensor_basis then falls back to (0,1,0) for the optical
    # aperture but NOT for the radar once the radar is aimed a degree or two off
    # nadir: the two apertures ended up rolled 51 deg against each other, so an
    # azimuth in one channel could not be drawn on the other. Give the radar and
    # lidar the imaging view's own up vector, and widen the search sector to
    # cover the formation instead of leaving the outer targets outside it (the
    # radar then had to re-aim every frame, which is the same registration break
    # in another form).
    _sens = cfg.get('sensors') or {}
    _cam = _sens.get('optical') or _sens.get('ir_camera') or {}
    _cam_up = _cam.get('up')
    _rcb = _sens.get('radar')
    if isinstance(_rcb, dict):
        if _cam_up:
            _rcb['up'] = [float(v) for v in _cam_up]
            if isinstance(_sens.get('lidar'), dict):
                _sens['lidar']['up'] = [float(v) for v in _cam_up]
        # How far the formation sits off the imaging axis, as a cone angle. A
        # 2 deg search sector is sized for one target on the boresight; with two
        # or four placed objects the outer ones fell outside it, the dataset had
        # to re-aim the radar per frame, and in the multi-target demo the frames
        # where the aimed target was the weak one ended with zero detections.
        _cam_pos = [float(v) for v in (_cam.get('pos') or [0.0, 0.0, 0.0])]
        _aim_pt = [float(v) for v in (_cam.get('look_at') or [0.0, 0.0, -1.0])]
        _fwd = [(_aim_pt[k] - _cam_pos[k]) for k in range(3)]
        _fn = math.sqrt(sum(v * v for v in _fwd))
        _worst = 0.0
        if _fn > 1e-9:
            _fwd = [v / _fn for v in _fwd]
            for _o in objects:
                if not _o.get('label', True):
                    continue
                try:
                    _d = [float(_o['pos'][k]) - _cam_pos[k] for k in range(3)]
                except (KeyError, IndexError, TypeError, ValueError):
                    continue
                _dn = math.sqrt(sum(v * v for v in _d))
                if _dn <= 1e-6:
                    continue
                _cos = max(-1.0, min(1.0, sum(_d[k] * _fwd[k] for k in range(3)) / _dn))
                _worst = max(_worst, math.degrees(math.acos(_cos)))
        if _worst > 0.0:
            _need_span = round(2.0 * (1.12 * _worst + 0.1), 2)
            for _key, _label in (('az_span_deg', '方位'), ('el_span_deg', '俯仰')):
                _cur = float(_rcb.get(_key) or 0.0)
                if _need_span > _cur:
                    notes.append('雷达%s扇区由 %.1f° 扩到 %.1f° 以覆盖编队'
                                 % (_label, _cur, _need_span))
                    _rcb[_key] = _need_span
    cfg['notes'] = notes

    if r['ecm']['mode'] != 'none':
        ents = _entities_for(r, site, vel, surface,
                             parent=(objects[0]['name'] if objects else None))
        if not ents:
            raise ValueError('ecm.mode 为 %s，但没有任何实体：请提供 ecm.entities'
                             % r['ecm']['mode'])
        from . import ecm as _ecm
        block = {'enabled': True, 'entities': ents}
        try:
            _ecm.load(block)
        except ValueError as exc:
            raise ValueError('ecm.entities 不合法：%s' % exc) from None
        cfg['ecm'] = block
    return cfg


# ------------------------------------------------------------------- summaries
def env_summary(recipe):
    """'天空 · 晴 · 纯色背景' -- one line for the UI list.

    Deliberately lenient: a half-filled form still gets a readable summary
    instead of a traceback, while a complete recipe goes through the same
    normalisation build_scene uses.
    """
    try:
        r = validate_recipe(recipe)
    except ValueError:
        return _lenient_summary(recipe)
    parts = [ENV_LABELS[r['environment']['kind']],
             WEATHER_LABELS[r['environment']['weather']],
             BACKGROUND_LABELS[r['environment']['background']['mode']]]
    # A nadir multi-target radar run is separated only by the height ladder; say
    # so in the one line the operator reads before the build, rather than let a
    # single bright RD cell look like a rendering bug.
    sep = _nadir_separation_note(r)
    if sep:
        parts.append(sep)
    return ' · '.join(parts)


def _lenient_summary(recipe):
    raw = recipe if isinstance(recipe, dict) else {}
    env = raw.get('environment')
    env = env if isinstance(env, dict) else {}
    bg = env.get('background')
    bg = bg if isinstance(bg, dict) else {}
    kind, weather, mode = env.get('kind'), env.get('weather'), bg.get('mode')
    return ' · '.join([
        ENV_LABELS.get(kind, str(kind or ENV_LABELS['sky'])),
        WEATHER_LABELS.get(weather, str(weather or WEATHER_LABELS['clear'])),
        BACKGROUND_LABELS.get(mode, str(mode or BACKGROUND_LABELS['none'])),
    ])


def default_recipe():
    """A sky/clear single fighter, the numbers the curated demo scenes use."""
    return {
        'name': 'demo',
        'environment': {'kind': 'sky', 'weather': 'clear',
                        'background': {'mode': 'none'}},
        'objects': [{'asset': '军用飞机/双发战斗机.obj', 'cls': 'fighter',
                     'scale': 1.0, 'count': 1}],
        'ecm': {'mode': 'none', 'entities': []},
        'view': {'range_km': 8.0, 'aspect': 'head_on', 'fov_deg': 2.2,
                 'views': 'single'},
        'capture': {'sensors': ['optical', 'ir', 'radar'], 'frames': 20,
                    'fps': 10.0, 'seed': 7},
        'output': {'preview': True, 'coco': True, 'yolo': True, 'readme': True},
    }
