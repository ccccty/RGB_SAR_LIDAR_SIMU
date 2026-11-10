"""Cross-entity consistency checker: the automatic criteria of
docs/多目标运动仿真调研.md section 4.

Why this exists
---------------
A multi-target scene can be built from individually plausible tracks and still
look wrong: one aircraft climbs away, the flare stays frozen where it was
dispensed, the wingman sits 300 m from the leader inside a 1.2 deg seeker frame
so only one of them is ever on screen.  Every one of those defects is a
*relation* between two entities, so no per-object envelope check can see it.
This module turns the relations of section 4 into numbers.

Groups (the doc's own labels, section 4):

    A  time base, world frame and chunk invariance
    B  single-entity envelope -- NOT reimplemented here: it is judged by
       tools/motion_diagnose.py (imported and called), because two copies of a
       rule list drift apart the moment one of them learns something.
    C  cross-entity geometry: formation box, relative speed, clearance, bearing
       rate, turn-lag error and the same-frame budget
    E  dispensed-body timing: attachment, ejection, air-relative decay,
       separation rate, settling, release continuity
    F  sensor consistency: Doppler / pixel size / LOS rate from one state, the
       same body in optical vs IR / radar / lidar, and (F5) whether every
       labelled target is inside the aperture the renderer actually aims
    G  labels: stable names, dispensables never in the target classes

Group D of the doc (attitude/velocity coupling) is not duplicated here: D1-D3
are already implemented by plan.object_state, motion.coordinated_bank_deg and
motion.attitude, and the task's checklist for this module asks for A/B/C/E/F/G.
Their single-entity consequences still surface through the reused B envelope.

Every rule below is computed from the same state source the renderers read:
plan.ClipPlan for scene objects, ecm.entities_at for countermeasure bodies,
scene.build_world for the geometry the numpy sensors ray-cast, and
external.write_manifest for the per-frame pose list the optical backend
consumes.  Nothing is asserted from the config alone.

Deviations from the literal doc, stated where they happen
---------------------------------------------------------
* C3 uses the doc's bounding-sphere clearance for target entities.  Scenery
  (label: false) is deliberately excluded: scenes place a barrier or a shrub a
  few metres from a target on purpose, and calling that "overlap" would bury
  the real multi-target cases.  Parent/child pairs are excluded too: a
  dispensed body is *supposed* to start at the aircraft.
* C4 measures the relative bearing rate in the leader's body frame.  In the
  world frame a rigid formation in a coordinated turn rotates at omega and
  would always fail the doc's 3 deg/s; "does not come apart" means the offset
  is stationary relative to the leader.
* E4 evaluates the drag law at t_r + 5*tau even past the entity's life: the
  decay constant, not visibility, is what E4 judges.
* E5 substitutes a 10 % / 5 m/s tolerance (the doc gives the limit as "->" with
  no number) and evaluates at min(t_r + 5*tau, end of clip); when the clip is
  shorter than 3*tau the limit is not observable and the rule is reported as
  not applicable.
* E7 subtracts the deceleration the declared drag law itself predicts before
  applying the doc's 200 m/s^2 bound: a chaff cloud genuinely decelerates at
  hundreds of m/s^2 at release, and that is not an unexplained jump.
* F5 is new: the doc judges the scene file, but dataset.py re-aims radar and
  lidar at the primary target on every frame and copies the imaging view's
  pose into them, so the authored sensors.radar.look_at may be a pose nothing
  renders.  F5 therefore evaluates the aperture twice -- on the render pose
  (the default) and on the authored pose -- and reports both.
* Derivative checks (B3, F3) use a one-sided first sample instead of a centred
  one.  A clip starts at t=0 and the planners freeze the position for t<0, so
  a centred difference there reads half a step: it reported 40.000 m/s of
  "derivative error" on an 80 m/s track and a 50 % LOS-rate error, both pure
  artefacts.  The thresholds are untouched.

Usage
-----
    from simkit import consistency
    report = consistency.inspect("scenes/demo_all.json")        # scene config
    report = consistency.inspect(recipe_dict)                   # recipe too
    report["ok"], report["checks"], report["summary"]

The CLI wrapper is tools/check_consistency.py.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import math
import os
import shutil
import tempfile

import numpy as np

from . import ecm, external, labels, motion, plan, scene, scenegen
from .mesh import rot_x, rot_y, rot_z

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#: Generated scenes live one level below the repo root, so their
#: ../assets/models/... paths resolve.  A recipe handed to inspect() has no
#: directory of its own; _tmp is the depth studio.py writes at.
RECIPE_BASE_DIR = os.path.join(ROOT, "_tmp")

# ===========================================================================
# Thresholds.  Every number is copied from docs/多目标运动仿真调研.md section 4
# (the section is named next to each block) unless a comment says otherwise;
# the comment then states why a number had to be adopted here.
# ===========================================================================

# -- section 4 A: time base, world frame, determinism -----------------------
A_TIME_EPS_S = 1e-9          # A1: same t everywhere, to the bit
A_FROZEN_EPS_M = 0.01        # A1: below this an entity has not moved at all
A_MAX_DT_S = 60.0            # A2: a frame step above a minute is not a clip
A_MAX_RANGE_M = 1.0e6        # A2: 1000 km; beyond this a unit is probably wrong
A_MAX_SPEED_MPS = 3.0e3      # A2: Mach 9; nothing in these scenes flies faster
A_MIN_SIZE_M = 1e-3
A_MAX_SIZE_M = 1.0e3
A_CHUNK_EPS = 0.0            # A3: the doc says bit-identical, so no tolerance

# -- section 4 C: cross-entity geometry ------------------------------------
# C1: "[50, 500] m (fighter); [10, 50] m (UAV)".  The doc gives no band for
# sea/ground formations, so those are reported as not applicable instead of
# inventing one.
C1_FORMATION_GAP_M = {
    "fighter": (50.0, 500.0),
    "airliner": (50.0, 500.0),
    "uav_fixed": (10.0, 50.0),
    "uav_rotor": (10.0, 50.0),
}
C2_FORMATION_REL_SPEED_MPS = 15.0   # C2 steady
C2_BREAK_REL_SPEED_MPS = 40.0       # C2 break formation
C2_STEADY_TURN_DPS = 3.0            # frames above this are not "steady"
C3_CLEARANCE_M = 10.0               # C3: > r_i + r_j + 10 m  (airborne pair)
# A person standing three metres from a parked car is a normal scene, not a
# collision: the 10 m figure is a visual-separation budget for two aircraft
# sharing one narrow field of view. Ground scenery only has to not interpenetrate.
C3_GROUND_CLEARANCE_M = 0.5
C4_BEARING_RATE_DPS = 3.0           # C4
C5_LAG_FACTOR = 2.0                 # C5: e <= 2*R*omega*tau
C5_DEFAULT_TAU_S = 0.3              # section 2.3 worked example (tau_lag = 0.3 s)
C5_MIN_TURN_RATE_RAD_S = 0.01       # below 0.57 deg/s the lag rule says nothing
C6_VERTICAL_FILL = 0.75             # C6: vertical budget is 0.75 of the half height

# -- section 4 E: dispensed-body timing ------------------------------------
E1_ATTACH_EPS_M = 1e-6              # E1
E1_SPEED_EPS_MPS = 1e-6             # E1 velocity clause (attached cases only)
E2_EJECT_MPS = (0.0, 60.0)          # E2: doc band [0, 60] m/s
E3_DECAY_EPS = 1e-6                 # E3: d|v_rel_air|/dt must be <= 0
E4_DECAY_TAIL = 0.05                # E4: <= 5 % of |v_rel0| at 5*tau
# The doc's own A.3 example lands at 5.25 % after 5 s, and a quadratic drag law
# calibrated as tau = 3..5*tau_q (doc 3.3) lands at 4-6 % at t = 5*tau; a strict
# 5.00 % would fail the doc's own worked example.  20 % of relative slack keeps
# the rule meaningful (no-decay bodies sit at 100 %) without failing the physics
# the doc itself endorses.
E4_TAIL_SLACK = 1.20
E4_DECAY_TIME_FACTOR = 5.0
E5_RATE_TOL_REL = 0.10              # E5: doc says "->", no number; adopted
E5_RATE_TOL_MPS = 5.0
E5_ANGLE_TOL_DEG = 15.0             # E5 direction; adopted
E5_MIN_PARENT_AIRSPEED_MPS = 5.0
E6_FALL_TOL_REL = 0.05              # E6: |v_z + fall| <= 0.05*fall
E7_ACCEL_JUMP_MPS2 = 200.0          # E7
E_H = 1e-3                          # analytic derivative step (see section 5.3)

# -- section 4 F: sensor consistency ---------------------------------------
F1_SOURCE_EPS = 1e-6                # F1: world state == plan state
F1_DOPPLER_TOL_MPS = 0.05           # F1: same tolerance as the B3 derivative
F2_RANGE_TOL_REL = 0.005            # F2: doc's 0.5 % belongs to F3; adopted
F2_PX_TOL_REL = 0.01
F3_LOS_RATE_TOL_REL = 0.005         # F3: <= 0.5 %
F4_CROSS_CHANNEL_EPS_M = 0.01       # F4: <= 0.01 m
# F5: two apertures that follow the same camera are co-boresighted by
# construction, so their az/el axes must be the same axes; half the narrowest
# beam in these scenes (a 1 deg Ku azimuth beam) is already smeared by more.
F5_AXIS_ROLL_TOL_DEG = 0.5

# ===========================================================================
# Check catalogue.  The CLI and the tests read this, so a check cannot exist
# in the implementation and be missing from the report (or the reverse).
# ===========================================================================
CHECK_SPECS = [
    dict(id="A1", group="A", title="one time base everywhere",
         threshold="channel clocks equal; no entity frozen at t=0 (4 A1)"),
    dict(id="A2", group="A", title="one world frame and units",
         threshold="m / s / deg, dt>0, finite 3-vectors (4 A2)"),
    dict(id="A3", group="A", title="chunk invariant",
         threshold="same global frame bit-identical whole / 2 parts / 3 parts (4 A3)"),
    dict(id="B1", group="B", title="speed envelope",
         threshold="class Envelope, reused from motion_diagnose (4 B1)"),
    dict(id="B2", group="B", title="accel / load / bank / climb / alt / turn / jerk",
         threshold="class Envelope, reused from motion_diagnose (4 B2)"),
    dict(id="B3", group="B", title="derivative contract |v - dp/dt|",
         threshold="<= 0.05 m/s, reused from motion_diagnose (4 B3)"),
    dict(id="C1", group="C", title="formation box",
         threshold="|offset| in [50,500] m fighter, [10,50] m UAV (4 C1)"),
    dict(id="C2", group="C", title="relative speed",
         threshold="steady <= 15 m/s, break <= 40 m/s (4 C2)"),
    dict(id="C3", group="C", title="minimum separation",
         threshold="> r_i + r_j + 10 m (bounding spheres, 4 C3)"),
    dict(id="C4", group="C", title="relative bearing rate",
         threshold="<= 3 deg/s in the leader body frame (4 C4)"),
    dict(id="C5", group="C", title="turn-lag consistency",
         threshold="e <= 2*R*omega*tau_lag (4 C5)"),
    dict(id="C6", group="C", title="same-frame budget",
         threshold="|offset| < r*tan(FOV/2), vertical < 0.75*r*tan(FOV_v/2) (4 C6)"),
    dict(id="E1", group="E", title="strict attachment before release",
         threshold="|p_c - (p_p + R_p*m_b)| < 1e-6 m (4 E1)"),
    dict(id="E2", group="E", title="release velocity step = ejection vector",
         threshold="|v_c - v_p| in [0,60] m/s (4 E2)"),
    dict(id="E3", group="E", title="air-relative speed never increases",
         threshold="d|v_rel_air|/dt <= 0 (4 E3)"),
    dict(id="E4", group="E", title="5*tau decay to 5 percent",
         threshold="|v_rel_air(t_r+5*tau)| <= 0.05*|v_rel0| (4 E4)"),
    dict(id="E5", group="E", title="separation rate tends to parent-air speed",
         threshold="d|p_c-p_p|/dt -> |v_p - v_air|, not the ejection speed (4 E5)"),
    dict(id="E6", group="E", title="settling velocity",
         threshold="chaff |v_z + fall| <= 0.05*fall; flare |v_z| -> v_t (4 E6)"),
    dict(id="E7", group="E", title="release-point continuity",
         threshold="position continuous; unexplained accel step <= 200 m/s^2 (4 E7)"),
    dict(id="F1", group="F", title="Doppler from the same state",
         threshold="v_rel from the rendered state; LOS radial rate matches geometry (4 F1)"),
    dict(id="F2", group="F", title="pixel size from the same range",
         threshold="px = size*W/(2*r*tan(FOV/2)), same r as F1 (4 F2)"),
    dict(id="F3", group="F", title="LOS angular rate from the same state",
         threshold="analytic az_dot/el_dot within 0.5 percent of the geometry (4 F3)"),
    dict(id="F4", group="F", title="dispensables at the same world position",
         threshold="optical / ir / radar / lidar within 0.01 m (4 F4)"),
    dict(id="F5", group="F", title="aim: every labelled target inside the aperture",
         threshold="|az| <= span/2 + beamwidth/2 in the frame each sensor is "
                   "actually rendered with (render layer by default)"),
    dict(id="G1", group="G", title="stable entity ids across modalities",
         threshold="names unique and identical in every channel (4 G1)"),
    dict(id="G2", group="G", title="dispensables never enter target labels",
         threshold="dispensable label=False; cls absent from COCO categories (4 G2)"),
]
CHECK_IDS = [c["id"] for c in CHECK_SPECS]
_GROUP_OF_CHECK = {c["id"]: c["group"] for c in CHECK_SPECS}


# ===========================================================================
# small helpers
# ===========================================================================
def _vec3(value):
    """A finite 3-vector as np.ndarray, or None."""
    try:
        a = np.asarray(value, float).reshape(-1)
    except (TypeError, ValueError):
        return None
    return a if a.size == 3 and np.all(np.isfinite(a)) else None


def _finite(value):
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _norm(value):
    return float(np.linalg.norm(np.asarray(value, float)))


def _fmt(x, nd=3):
    if x is None:
        return "-"
    try:
        return ("%." + str(int(nd)) + "f") % float(x)
    except (TypeError, ValueError):
        return str(x)


def _result(cid, ok, detail, value=None, limit=None, applicable=True):
    """One row of the report, in the shape the task's interface promises."""
    return dict(
        id=cid, group=_GROUP_OF_CHECK.get(cid, cid[:1]), ok=bool(ok),
        detail=str(detail), value=_finite(value), limit=_finite(limit),
        applicable=bool(applicable),
    )


def _skip(cid, reason):
    return _result(cid, True, "n/a: " + reason, applicable=False)


def _sample_indices(n, frames):
    """Frame indices to evaluate.  None = every frame; int = evenly spaced."""
    n = max(int(n), 1)
    if frames is None:
        return list(range(n))
    if isinstance(frames, (int, np.integer)):
        k = max(1, min(int(frames), n))
        if k == 1:
            return [0]
        return sorted({int(round(x)) for x in np.linspace(0.0, n - 1.0, k)})
    out = sorted({int(f) for f in frames if 0 <= int(f) < n})
    return out or [0]


def _looks_like_recipe(obj):
    return (isinstance(obj, dict) and "environment" in obj and "view" in obj
            and "capture" in obj and "frames" not in obj)


def _coerce_config(src):
    """Accept a scene config dict, a recipe dict, or a path to either."""
    if isinstance(src, (str, os.PathLike)):
        path = os.path.abspath(os.fspath(src))
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        base = os.path.dirname(path)
    elif isinstance(src, dict):
        raw = src
        base = src.get("_base_dir")
    else:
        raise TypeError("inspect() takes a scene config, a recipe or a path")
    if _looks_like_recipe(raw):
        cfg = scenegen.build_scene(dict(raw))
        # build_scene writes ../assets/models/..., which only resolves from a
        # directory one level below the repo root.  The directory does not have
        # to exist: abspath() normalises "../assets" against it either way.
        base = RECIPE_BASE_DIR
    else:
        cfg = copy.deepcopy(raw)
        base = base or os.getcwd()
    cfg["_base_dir"] = os.path.abspath(base)
    return cfg


def _formation_membership(spec, primary=None):
    """(group, parent_name, tau_lag_s) for a formation member, else (None,..).

    The doc's schema (section 5.2) is link: "formation" plus a "formation"
    block; the older spellings "wingman_of" / "formation_parent" are aliases.
    A link without an explicit parent means the primary target, exactly as
    plan.ClipPlan resolves it.
    """
    link = str(spec.get("link") or "").strip().lower()
    form = spec.get("formation")
    parent = spec.get("formation_parent") or spec.get("wingman_of")
    tau = None
    group = None
    if isinstance(form, dict):
        parent = form.get("parent") or parent
        group = form.get("group") or form.get("name") or parent
        tau = form.get("tau_lag_s", form.get("tau_s"))
        link = "formation"
    elif isinstance(form, str) and form.strip():
        group = form.strip()
    if spec.get("formation_group") is not None:
        group = str(spec.get("formation_group"))
    if link == "formation" and not group:
        parent = parent or primary
        group = parent or "default"
    if group is None and parent is not None:
        group = str(parent)
    return group, parent, _finite(tau)


def _yaw_rate_dps(cp, obj, t, h=1e-2):
    """Unwrapped heading rate in deg/s from the published yaw, not from v."""
    y0 = float(cp.object_state(obj, t - h)[2])
    y1 = float(cp.object_state(obj, t + h)[2])
    d = (y1 - y0 + 180.0) % 360.0 - 180.0
    return float(d / (2.0 * h))


def _attitude_R(cp, obj, t):
    """Body -> world rotation, the same yaw/pitch/roll convention as scene.py."""
    _p, _v, yaw = cp.object_state(obj, t)
    pitch, roll = cp.object_attitude(obj, t)
    return rot_z(float(yaw)) @ rot_y(-float(pitch)) @ rot_x(float(roll))


def _fingerprint(state):
    """The chunk-invariance comparator: every state field a consumer reads."""
    poses = {}
    for name, p in (state.get("poses") or {}).items():
        poses[name] = (
            tuple(float(x) for x in p.get("pos", [])),
            float(p.get("yaw", 0.0)),
            float(p.get("pitch", 0.0)),
            float(p.get("roll", 0.0)),
            tuple(float(x) for x in p.get("velocity_mps", [])),
        )
    views = {}
    for name, v in (state.get("views") or {}).items():
        views[name] = (
            tuple(float(x) for x in v.get("pos", [])),
            tuple(float(x) for x in v.get("look_at", [])),
            float(v.get("fov_deg", 0.0)),
            int(v.get("width", 0)), int(v.get("height", 0)),
            float(v.get("roll_deg", 0.0) or 0.0),
        )
    return (int(state.get("frame", -1)), float(state.get("t", 0.0)), poses, views,
            float(state.get("range_m", 0.0)),
            tuple(float(x) for x in state.get("sensor_velocity_mps", [])),
            {k: (tuple(float(x) for x in v.get("dpos", [])),
                 float(v.get("dyaw", 0.0)),
                 tuple(float(x) for x in v.get("vel", [])))
             for k, v in (state.get("jitter") or {}).items()})


_MOTION_DIAGNOSE = None


def _load_motion_diagnose():
    """Import tools/motion_diagnose.py without making tools a package."""
    global _MOTION_DIAGNOSE
    if _MOTION_DIAGNOSE is not None:
        return _MOTION_DIAGNOSE
    path = os.path.join(ROOT, "tools", "motion_diagnose.py")
    spec = importlib.util.spec_from_file_location("simkit_consistency_motion_diagnose", path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load tools/motion_diagnose.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _MOTION_DIAGNOSE = mod
    return mod


def _materialise_scene(cfg):
    """Write a temp scene whose model paths are absolute.

    motion_diagnose.diagnose_scene() takes a path and calls scene.load_config on
    it, which overrides _base_dir with the temp directory.  Absolutising the
    model paths first keeps the B-group numbers identical to the real scene.
    """
    tmp = tempfile.mkdtemp(prefix="simkit_consistency_scene_")
    data = copy.deepcopy(cfg)
    base = data.get("_base_dir") or os.getcwd()
    for spec in data.get("objects") or []:
        for key in ("model", "optical_material_source"):
            value = spec.get(key)
            if value and not os.path.isabs(str(value)):
                spec[key] = os.path.abspath(os.path.join(base, str(value)))
    value = (data.get("optical") or {}).get("sky_image")
    if value and not os.path.isabs(str(value)):
        data["optical"]["sky_image"] = os.path.abspath(os.path.join(base, str(value)))
    path = os.path.join(tmp, "scene.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False)
    return path, tmp


# ===========================================================================
# Per-inspection context: one clip plan cache, one state source, one manifest.
# ===========================================================================
class _Context:
    def __init__(self, cfg, frames=None):
        self.cfg = cfg
        frames_cfg = cfg.get("frames") or {}
        self.n = max(int(frames_cfg.get("n", 1)), 1)
        self.dt = float(frames_cfg.get("dt", 0.2))
        self.seed = int(cfg.get("seed", 0))
        self.base = cfg.get("_base_dir") or os.getcwd()
        self.fpc = int(plan.frames_per_clip(cfg, self.n))
        self.n_clips = max(1, int(math.ceil(self.n / float(self.fpc))))
        self.indices = _sample_indices(self.n, frames)
        self.objects = [o for o in (cfg.get("objects") or []) if isinstance(o, dict)]
        if not self.objects:
            raise ValueError("a scene needs at least one object")
        self.labelled = [o for o in self.objects if o.get("label", True)]
        if not self.labelled:
            raise ValueError("a clip needs at least one labelled object")
        self._plans = {}
        self._states = {}
        self._radii = {}
        self._ecm_spec = "unset"
        self._ecm_at = {}
        self._manifest = "unset"
        self._worlds = None
        self._formations = None
        self._ground_z = None
        g = cfg.get("ground") or {}
        if g.get("enabled", True):
            self._ground_z = float(g.get("z", 0.0))
        picks = [0, self.n // 2, self.n - 1, self.indices[0], self.indices[-1]]
        self.world_frames = []
        for f in picks:
            if 0 <= f < self.n and f not in self.world_frames:
                self.world_frames.append(f)
        self.world_frames = sorted(self.world_frames)[:4]

    # -- state source -------------------------------------------------------
    def plan(self, clip):
        clip = int(clip)
        if clip not in self._plans:
            self._plans[clip] = plan.ClipPlan(self.cfg, clip, self.fpc, self.dt, self.seed)
        return self._plans[clip]

    def state(self, f):
        f = int(f)
        if f not in self._states:
            clip = f // self.fpc
            self._states[f] = self.plan(clip).state(f - clip * self.fpc, f)
        return self._states[f]

    def mode(self):
        try:
            return self.plan(0).mode
        except Exception:
            return "?"

    # -- geometry -----------------------------------------------------------
    def radius(self, spec):
        name = spec.get("name", id(spec))
        if name not in self._radii:
            try:
                _m, lo, hi = scene.prepare_mesh(spec, self.base)
                self._radii[name] = 0.5 * _norm(np.asarray(hi, float)
                                                - np.asarray(lo, float))
            except Exception:
                size = _vec3(spec.get("size", [1.0, 1.0, 1.0]))
                self._radii[name] = 0.5 * _norm(size) if size is not None else 1.0
        return self._radii[name]

    def anchors(self, f):
        """Name -> (pos, vel) of every scene object at frame f, unrounded."""
        st = self.state(f)
        t = float(st["t"])
        cp = self.plan(f // self.fpc)
        out = {}
        for spec in self.objects:
            p, v, _yaw = cp.object_state(spec, t)
            out[spec.get("name")] = (np.asarray(p, float), np.asarray(v, float))
        return out

    # -- ecm ----------------------------------------------------------------
    def ecm_spec(self):
        if self._ecm_spec == "unset":
            try:
                self._ecm_spec = ecm.load(self.cfg.get("ecm"))
            except Exception as exc:               # a broken block must be visible
                self._ecm_spec = exc
        if isinstance(self._ecm_spec, Exception):
            raise self._ecm_spec
        return self._ecm_spec

    def ecm_states(self, f):
        f = int(f)
        if f not in self._ecm_at:
            try:
                spec = self.ecm_spec()
            except Exception:
                self._ecm_at[f] = {}
            else:
                if not spec["enabled"]:
                    self._ecm_at[f] = {}
                else:
                    st = self.state(f)
                    cp = self.plan(f // self.fpc)
                    tracks = getattr(cp, "tracks", None)
                    anchors = None if tracks else self.anchors(f)
                    self._ecm_at[f] = {e["id"]: e for e in ecm.entities_at(
                        spec, float(st["t"]), anchors=anchors, tracks=tracks)}
        return self._ecm_at[f]

    # -- formations ---------------------------------------------------------
    def formations(self):
        """group -> dict(members, lead, tau); only groups with >= 2 targets."""
        if self._formations is not None:
            return self._formations
        by_name = {o.get("name"): o for o in self.objects}
        primary = self.labelled[0].get("name") if self.labelled else None
        # plan.ClipPlan's own default: a labelled non-primary object without a
        # trajectory follows the primary as a formation member, except when the
        # primary flies an authored straight line (then every absolute pos+vel
        # is the same story and two parked cars are not a formation).
        clip = self.cfg.get("clip") or {}
        traj = clip.get("trajectory") or self.labelled[0].get("trajectory")
        traj_mode = str((traj or {}).get("mode", "line")).lower()
        mode = self.mode()
        authored_line = (mode == "platform") or (
            mode != "seeker"
            and traj_mode in ("line", "straight", "static", "hover", "parked"))
        raw = {}
        for spec in self.labelled:
            group, parent, tau = _formation_membership(spec, primary)
            if (group is None and parent is None and not authored_line
                    and spec is not self.labelled[0]
                    and spec.get("motion_type") != "ground"
                    and spec.get("trajectory") is None
                    and spec.get("label", True)):
                group, parent, tau = primary, primary, C5_DEFAULT_TAU_S
            if group is None and parent is None:
                continue
            key = group or parent
            entry = raw.setdefault(key, dict(members=[], parent=None, tau=None))
            if spec not in entry["members"]:
                entry["members"].append(spec)
            if parent:
                entry["parent"] = str(parent)
                host = by_name.get(parent)
                if (host is not None and host.get("label", True)
                        and host not in entry["members"]):
                    entry["members"].append(host)
            if tau is not None:
                entry["tau"] = tau
        out = {}
        for key, entry in raw.items():
            members = [m for m in entry["members"] if m.get("label", True)]
            if len(members) < 2:
                continue
            names = {m.get("name") for m in members}
            lead = by_name.get(entry["parent"]) if entry["parent"] in names else members[0]
            out[key] = dict(members=members, lead=lead,
                            tau=entry["tau"] if entry["tau"] is not None else C5_DEFAULT_TAU_S)
        self._formations = out
        return out

    def pairs(self):
        """[(group, lead, follower, tau)] -- one pair per follower."""
        out = []
        for key, g in self.formations().items():
            for member in g["members"]:
                if member is g["lead"]:
                    continue
                out.append((key, g["lead"], member, g["tau"]))
        return out

    # -- optical channel ----------------------------------------------------
    def manifest(self):
        """frame id -> manifest entry, from the real external.write_manifest()."""
        if self._manifest == "unset":
            # shallow state copies: write_manifest() adds ECM entries to
            # state["poses"], and the cached states must stay pristine for A3.
            states = []
            for f in self.world_frames:
                st = self.state(f)
                states.append(dict(st, poses=copy.deepcopy(st.get("poses") or {})))
            tmp = tempfile.mkdtemp(prefix="simkit_consistency_manifest_")
            try:
                path = external.write_manifest(self.cfg, states, tmp)
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
            self._manifest = {int(fr.get("frame", i)): fr
                              for i, fr in enumerate(data.get("frames") or [])}
        return self._manifest

    # -- numpy sensor world -------------------------------------------------
    def worlds(self):
        """([(frame, state, world)], errors) for the expensive comparisons."""
        if self._worlds is not None:
            return self._worlds
        out, errors = [], []
        for f in self.world_frames:
            st = self.state(f)
            # state() already stamps _ecm_t and carries the planner Track set as
            # an attribute; deep-copying would drop the tracks and force the ECM
            # layer back onto a one-pose anchor.
            fr_cfg = st["cfg"]
            try:
                w = scene.build_world(fr_cfg, t=0.0, jitter=st["jitter"])
            except Exception as exc:               # missing model, bad mesh...
                errors.append("frame %d: %s: %s" % (f, type(exc).__name__, exc))
                continue
            out.append((f, st, w))
        self._worlds = (out, errors)
        return self._worlds

    def has_camera(self):
        clip = self.cfg.get("clip") or {}
        sensors = self.cfg.get("sensors") or {}
        name = clip.get("camera") or "ir_camera"
        for key in (name, "ir_camera", "optical"):
            cam = sensors.get(key)
            if isinstance(cam, dict) and cam.get("pos"):
                return True
        return bool(clip.get("views"))


# ===========================================================================
# A. time base, world frame, determinism
# ===========================================================================
def _check_A1(ctx):
    """Every channel must use the same t, and none may freeze at t=0."""
    try:
        manifest = ctx.manifest()
    except Exception as exc:
        return _result("A1", False, "optical manifest could not be built: %s: %s"
                       % (type(exc).__name__, exc), 0.0, A_TIME_EPS_S)
    problems = []
    t_err = 0.0
    frozen = []
    base = manifest.get(0)
    for f in ctx.world_frames:
        st = ctx.state(f)
        t = float(st["t"])
        entry = manifest.get(f)
        if entry is None:
            problems.append("frame %d missing from the optical manifest" % f)
            continue
        t_manifest = _finite(entry.get("t", 0.0))
        if t_manifest is None or abs(t_manifest - t) > A_TIME_EPS_S:
            problems.append("frame %d: optical timestamp %r != plan t %.6f"
                            % (f, entry.get("t"), t))
        if t_manifest is not None:
            t_err = max(t_err, abs(t_manifest - t))
        # plan.state() puts the frame clock in the config (and the planner Track
        # set beside it); any caller that drops it makes the ECM layer evaluate
        # at t=0, which is exactly the A1 failure.
        t_frame = float(ecm.frame_time(st["cfg"]))
        t_err = max(t_err, abs(t_frame - t))
        if abs(t_frame - t) > A_TIME_EPS_S:
            problems.append("frame %d: ecm.frame_time(cfg) = %.6f != plan t %.6f"
                            % (f, t_frame, t))
        cp = ctx.plan(f // ctx.fpc)
        m0 = (base.get("objects") or {}) if base else {}
        m_now = entry.get("objects") or {}
        for spec in ctx.objects:
            name = spec.get("name")
            pose = m_now.get(name)
            if pose is None:
                problems.append("frame %d: no optical pose for %s" % (f, name))
                continue
            p_now = np.asarray(pose.get("pos", [0.0, 0.0, 0.0]), float)
            p_model = np.asarray(cp.object_state(spec, t)[0], float)
            p_model0 = np.asarray(cp.object_state(spec, 0.0)[0], float)
            old = m0.get(name)
            if old is not None and _norm(p_model - p_model0) > A_FROZEN_EPS_M:
                moved = _norm(p_now - np.asarray(old.get("pos", [0.0, 0.0, 0.0]), float))
                if moved <= A_FROZEN_EPS_M:
                    frozen.append("%s frozen at t=0 (model moved %.2f m)"
                                  % (name, _norm(p_model - p_model0)))
        try:
            ecm_now = ctx.ecm_states(f)
            ecm_0 = ctx.ecm_states(0)
        except Exception:
            ecm_now, ecm_0 = {}, {}
        for eid, ent in ecm_now.items():
            if ent.get("kind") not in ecm.BODY_KINDS or not ent.get("visible", True):
                continue
            pose = m_now.get(eid)
            if pose is None:
                problems.append("frame %d: no optical pose for ECM %s" % (f, eid))
                continue
            ent0 = ecm_0.get(eid)
            if ent0 is None:
                continue                          # released after frame 0
            if _norm(np.asarray(ent["pos"]) - np.asarray(ent0["pos"])) > A_FROZEN_EPS_M:
                old = (base.get("objects") or {}).get(eid) if base else None
                if old is not None:
                    moved = _norm(np.asarray(pose["pos"], float)
                                  - np.asarray(old.get("pos", [0, 0, 0]), float))
                    if moved <= A_FROZEN_EPS_M:
                        frozen.append("ECM %s frozen at t=0" % eid)
    ok = not problems and not frozen and t_err <= A_TIME_EPS_S
    if ok:
        detail = ("clock identical across plan/manifest/frame_time (max %.1e s); "
                  "%d objects, %d ECM bodies advance with t"
                  % (t_err, len(ctx.objects), len(ctx.ecm_states(ctx.world_frames[-1]))))
        value, limit = t_err, A_TIME_EPS_S
    else:
        bits = problems[:3] + frozen[:3]
        detail = ("; ".join(bits) if bits else "time stamps differ")
        detail += " | max timestamp error %.3e s" % t_err
        if t_err > A_TIME_EPS_S:
            value, limit = t_err, A_TIME_EPS_S
        else:
            value, limit = A_FROZEN_EPS_M, A_FROZEN_EPS_M
    return _result("A1", ok, detail, value, limit)


def _check_A2(ctx):
    """One world, metres and seconds, and a clock that advances."""
    issues = []       # (kind, value, limit, text)
    dt = ctx.dt
    if not math.isfinite(dt) or dt <= 0.0:
        issues.append(("dt", dt, 0.0,
                       "frames.dt = %r: the frame clock does not advance" % dt))
    elif dt > A_MAX_DT_S:
        issues.append(("dt", dt, A_MAX_DT_S,
                       "frames.dt = %.3f s is longer than a clip step" % dt))
    if ctx.n < 1:
        issues.append(("n", ctx.n, 1.0, "frames.n must be >= 1"))
    for spec in ctx.objects:
        name = spec.get("name", "?")
        pos = spec.get("pos")
        if pos is not None:
            v = _vec3(pos)
            if v is None:
                issues.append(("pos", None, None, "%s.pos is not a finite 3-vector" % name))
            elif _norm(v) > A_MAX_RANGE_M:
                issues.append(("pos", _norm(v), A_MAX_RANGE_M,
                               "%s.pos |%.3g| m exceeds the metre-scale world bound"
                               % (name, _norm(v))))
        vel = spec.get("vel")
        if vel is not None:
            v = _vec3(vel)
            if v is None:
                issues.append(("vel", None, None, "%s.vel is not a finite 3-vector" % name))
            elif _norm(v) > A_MAX_SPEED_MPS:
                issues.append(("vel", _norm(v), A_MAX_SPEED_MPS,
                               "%s.vel %.3g m/s exceeds the unit sanity bound"
                               % (name, _norm(v))))
        size = spec.get("size")
        if size is not None:
            s = _vec3(size)
            if s is None or float(np.min(s)) <= 0.0:
                issues.append(("size", None, None,
                               "%s.size must be a positive 3-vector" % name))
            elif float(np.max(s)) > A_MAX_SIZE_M or float(np.min(s)) < A_MIN_SIZE_M:
                issues.append(("size", float(np.max(s)), A_MAX_SIZE_M,
                               "%s.size %s is not a metre-scale body"
                               % (name, [round(float(x), 3) for x in s])))
    sensors = ctx.cfg.get("sensors") or {}
    for key, sc in sensors.items():
        if not isinstance(sc, dict):
            continue
        for field in ("pos", "look_at", "up"):
            if sc.get(field) is not None and _vec3(sc.get(field)) is None:
                issues.append((field, None, None,
                               "sensors.%s.%s is not a finite 3-vector" % (key, field)))
        if sc.get("fov_deg") is not None:
            fov = _finite(sc.get("fov_deg"))
            if fov is None or not 0.0 < fov < 180.0:
                issues.append(("fov", fov, 180.0,
                               "sensors.%s.fov_deg must be in (0,180) degrees" % key))
        for field in ("width", "height"):
            if sc.get(field) is not None:
                dim = _finite(sc.get(field))
                if dim is None or not 1.0 <= dim <= 20000.0:
                    issues.append((field, dim, 20000.0,
                                   "sensors.%s.%s must be a positive pixel count"
                                   % (key, field)))
    clip = ctx.cfg.get("clip") or {}
    for view in clip.get("views") or []:
        if not isinstance(view, dict):
            continue
        fov = _finite(view.get("fov_deg"))
        if fov is None or not 0.0 < fov < 180.0:
            issues.append(("fov", fov, 180.0,
                           "clip.views[%r].fov_deg must be in (0,180)"
                           % view.get("name")))
    seeker_cfg = clip.get("seeker") or {}
    # platform mode never reads range_m (_plan_seeker is not called), so a band
    # sitting in that block is dead data, not a unit error.
    band = seeker_cfg.get("range_m") if str(seeker_cfg.get("mode", "")).lower() != "platform" else None
    if band is not None:
        try:
            arr = np.asarray(band, float).reshape(-1)
        except (TypeError, ValueError):
            arr = None
        if arr is None or arr.size != 2 or not np.all(np.isfinite(arr)) or np.min(arr) < 0:
            issues.append(("range_m", None, None,
                           "clip.seeker.range_m must be [lo, hi] metres"))
        elif arr[0] > arr[1] + 1e-9:
            issues.append(("range_m", float(arr[0]), float(arr[1]),
                           "clip.seeker.range_m is reversed"))
    if ctx._ground_z is not None and dt > 0.0:
        try:
            st = ctx.state(0)
            cp = ctx.plan(0)
            gz = ctx._ground_z
            for spec in ctx.objects:
                z = float(cp.object_state(spec, float(st["t"]))[0][2])
                if spec.get("motion_type") == "ground" and abs(z - gz) > 1.0:
                    issues.append(("ground", z, gz,
                                   "%s is a ground object %.2f m off the ground plane"
                                   % (spec.get("name"), z - gz)))
                elif z < gz - 1.0:
                    issues.append(("ground", z, gz,
                                   "%s sits %.2f m below the ground plane"
                                   % (spec.get("name"), gz - z)))
        except Exception as exc:
            issues.append(("state", None, None,
                           "could not evaluate frame 0: %s" % exc))
    decl = (ctx.cfg.get("world") or ctx.cfg.get("frame") or ctx.cfg.get("units")
            or ctx.cfg.get("conventions"))
    frame_note = ("local ENU, Z-up, metres (not declared; the observations "
                  "metadata writes the same convention)")
    if decl is not None:
        text = json.dumps(decl, ensure_ascii=False).lower()
        if not any(k in text for k in ("enu", "ecef", "z-up", "zup",
                                       "right-handed", "metre", "meter")):
            issues.append(("world", None, None,
                           "declared world frame %r is not ENU/ECEF metres" % (decl,)))
        else:
            frame_note = "declared " + text
    ok = not issues
    if ok:
        detail = ("dt=%s s, n=%d; %s; %d objects, %d sensors all finite"
                  % (_fmt(dt, 4), ctx.n, frame_note, len(ctx.objects), len(sensors)))
        value, limit = dt, A_MAX_DT_S
    else:
        value, limit, text = issues[0][1], issues[0][2], issues[0][3]
        detail = "%s (%d issue(s): %s)" % (
            text, len(issues), "; ".join(i[3] for i in issues[:4]))
    return _result("A2", ok, detail, value, limit)


def _check_A3(ctx):
    """One frame must be the same under every chunk split."""
    picks = sorted({0, ctx.n // 2, ctx.n - 1, ctx.indices[0], ctx.indices[-1]})
    picks = [f for f in picks if 0 <= f < ctx.n][:4]
    part = max(1, ctx.n // 3)
    bad = []
    worst = 0
    for f in picks:
        whole = _fingerprint(ctx.state(f))
        try:
            single = _fingerprint(next(plan.plan_frames(
                ctx.cfg, 1, ctx.seed, ctx.dt, frame_offset=f, total_frames=ctx.n)))
            start = (f // part) * part
            size = min(part, ctx.n - start)
            chunk = None
            for k, st in enumerate(plan.plan_frames(
                    ctx.cfg, size, ctx.seed, ctx.dt, frame_offset=start,
                    total_frames=ctx.n)):
                if k == f - start:
                    chunk = _fingerprint(st)
                    break
        except Exception as exc:
            bad.append("frame %d: chunked plan failed: %s" % (f, exc))
            continue
        if chunk is None:
            bad.append("frame %d: part [%d,%d) did not produce it"
                       % (f, start, start + size))
            continue
        for label, other in (("singleton", single), ("part", chunk)):
            if other == whole:
                continue
            fields = []
            if other[0] != whole[0]:
                fields.append("frame")
            if other[1] != whole[1]:
                fields.append("t")
            if other[2] != whole[2]:
                fields.append("poses")
                diffs = [abs(float(a) - float(b))
                         for name in whole[2] if name in other[2]
                         for a, b in zip(whole[2][name], other[2][name])
                         if isinstance(a, float) and isinstance(b, float)]
                if diffs:
                    worst = max(worst, max(diffs))
            if other[3] != whole[3]:
                fields.append("views")
            if other[4] != whole[4]:
                fields.append("range_m")
            if other[5] != whole[5]:
                fields.append("sensor_velocity")
            if other[6] != whole[6]:
                fields.append("jitter")
            bad.append("frame %d differs under %s chunking (%s)"
                       % (f, label, ", ".join(fields) or "unknown field"))
    ok = not bad and worst <= A_CHUNK_EPS
    if ok:
        detail = ("%d sampled frames bit-identical under whole / singleton / "
                  "%d-frame parts" % (len(picks), part))
    else:
        detail = "; ".join(bad[:4]) or "state differs between chunkings"
    return _result("A3", ok, detail, worst, A_CHUNK_EPS)


# ===========================================================================
# B. single-entity envelope -- delegated to tools/motion_diagnose.py
# ===========================================================================
def _check_B(ctx):
    """Return the three B rows, computed by motion_diagnose.diagnose_scene()."""
    try:
        md = _load_motion_diagnose()
        path, tmp = _materialise_scene(ctx.cfg)
        try:
            found = md.diagnose_scene(path)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    except Exception as exc:
        text = "motion_diagnose failed: %s: %s" % (type(exc).__name__, exc)
        return [_result(b, False, text, None, None) for b in ("B1", "B2", "B3")]

    objects = found.get("objects") or []
    buckets = {b: [] for b in ("B1", "B2", "B3")}
    enforced = 0
    for obj in objects:
        env = obj.get("envelope") or {}
        if env.get("enforce"):
            enforced += 1
        for v in obj.get("violations") or []:
            rule = str(v.get("rule"))
            target = "B3" if rule == "derivative" else (
                "B1" if rule in ("speed_max", "speed_min") else "B2")
            buckets[target].append((obj.get("name", "?"), rule, v))

    rows = []
    for cid in ("B1", "B2", "B3"):
        found_v = buckets[cid]
        if cid in ("B1", "B2") and enforced == 0:
            rows.append(_skip(cid, "no object claims a class envelope"))
            continue
        if cid == "B3" and not objects:
            rows.append(_skip(cid, "no object track to judge"))
            continue
        if not found_v:
            if cid == "B3":
                worst = max((float((o.get("summary") or {}).get(
                    "max_derivative_error_mps", 0.0)) for o in objects), default=0.0)
                rows.append(_result(cid, True,
                                    "%d object tracks honour |v - dp/dt| (worst %.2e m/s)"
                                    % (len(objects), worst), worst, 0.05))
            else:
                rows.append(_result(cid, True,
                                    "%d object tracks inside their class envelope"
                                    % enforced, 0.0, None))
            continue

        def severity(item):
            v = item[2]
            lim = abs(float(v.get("limit", 0.0)))
            if cid == "B2" or lim <= 0.0:
                return abs(float(v.get("value", 0.0))) / max(lim, 1e-9)
            return abs(float(v.get("value", 0.0))) / lim

        worst = max(found_v, key=severity)
        v = worst[2]
        detail = ("%d violation(s); worst %s %s %.4f vs limit %.4f at frame %s"
                  % (len(found_v), worst[0], v.get("rule"),
                     float(v.get("value", 0.0)), float(v.get("limit", 0.0)),
                     v.get("frame")))
        rows.append(_result(cid, False, detail, float(v.get("value", 0.0)),
                            float(v.get("limit", 0.0))))
    return rows


# ===========================================================================
# C. cross-entity geometry
# ===========================================================================
def _check_C1(ctx):
    pairs = ctx.pairs()
    if not pairs:
        return _skip("C1", "no formation declared (link/formation/wingman_of)")
    worst = None       # (excess, distance, limit, pair, frame, band)
    seen = 0
    no_band = []
    for _gname, lead, fol, _tau in pairs:
        band = C1_FORMATION_GAP_M.get(motion.Envelope.classify(lead))
        if band is None:
            band = C1_FORMATION_GAP_M.get(motion.Envelope.classify(fol))
        if band is None:
            no_band.append("%s/%s" % (lead.get("name"), fol.get("name")))
            continue
        seen += 1
        for f in ctx.indices:
            st = ctx.state(f)
            t = float(st["t"])
            cp = ctx.plan(f // ctx.fpc)
            d = _norm(np.asarray(cp.object_state(fol, t)[0], float)
                      - np.asarray(cp.object_state(lead, t)[0], float))
            if d < band[0]:
                excess, limit = d - band[0], band[0]
            elif d > band[1]:
                excess, limit = d - band[1], band[1]
            else:
                excess, limit = 0.0, band[0]
            if worst is None or excess < worst[0]:
                worst = (excess, d, limit,
                         (lead.get("name"), fol.get("name")), f, band)
    if seen == 0:
        return _skip("C1", "formation classes without a documented band: %s"
                     % ", ".join(no_band[:3]))
    ok = worst[0] >= -1e-9
    if ok:
        detail = ("%d pair(s); tightest spacing %s ~ %s %.2f m inside band "
                  "[%g, %g] m" % (seen, worst[3][0], worst[3][1], worst[1],
                                  worst[5][0], worst[5][1]))
    else:
        detail = ("%s ~ %s at frame %d: spacing %.2f m outside [%g, %g] m"
                  % (worst[3][0], worst[3][1], worst[4], worst[1],
                     worst[5][0], worst[5][1]))
    return _result("C1", ok, detail, worst[1], worst[2])


def _check_C2(ctx):
    pairs = ctx.pairs()
    if not pairs:
        return _skip("C2", "no formation declared")
    max_steady = (0.0, None)
    max_all = (0.0, None)
    steady_applied = False
    for _gname, lead, fol, _tau in pairs:
        for f in ctx.indices:
            st = ctx.state(f)
            t = float(st["t"])
            cp = ctx.plan(f // ctx.fpc)
            p1, v1, _ = cp.object_state(lead, t)
            p2, v2, _ = cp.object_state(fol, t)
            rel = _norm(np.asarray(v1, float) - np.asarray(v2, float))
            if rel > max_all[0]:
                max_all = (rel, (lead.get("name"), fol.get("name"), f))
            wl = abs(_yaw_rate_dps(cp, lead, t))
            wf = abs(_yaw_rate_dps(cp, fol, t))
            if wl <= C2_STEADY_TURN_DPS and wf <= C2_STEADY_TURN_DPS:
                steady_applied = True
                if rel > max_steady[0]:
                    max_steady = (rel, (lead.get("name"), fol.get("name"), f))
    bad_steady = steady_applied and max_steady[0] > C2_FORMATION_REL_SPEED_MPS + 1e-9
    bad_break = max_all[0] > C2_BREAK_REL_SPEED_MPS + 1e-9
    ok = not bad_steady and not bad_break
    note = ("steady frames: max %.2f m/s (limit %g); all frames: max %.2f m/s "
            "(limit %g)" % (max_steady[0], C2_FORMATION_REL_SPEED_MPS,
                            max_all[0], C2_BREAK_REL_SPEED_MPS))
    if ok:
        return _result("C2", True, note,
                       max_steady[0] if steady_applied else max_all[0],
                       C2_FORMATION_REL_SPEED_MPS if steady_applied
                       else C2_BREAK_REL_SPEED_MPS)
    if bad_steady:
        item = max_steady
        detail = ("%s ~ %s at frame %d: steady relative speed %.2f m/s > %g m/s"
                  % (item[1][0], item[1][1], item[1][2], item[0],
                     C2_FORMATION_REL_SPEED_MPS))
        return _result("C2", False, detail + " | " + note, item[0],
                       C2_FORMATION_REL_SPEED_MPS)
    item = max_all
    detail = ("%s ~ %s at frame %d: relative speed %.2f m/s > %g m/s "
              "(formation broken)" % (item[1][0], item[1][1], item[1][2],
                                      item[0], C2_BREAK_REL_SPEED_MPS))
    return _result("C2", False, detail, item[0], C2_BREAK_REL_SPEED_MPS)


def _entity_parent(ent):
    """The parent object name for any link style (release / tether / formation)."""
    fn = getattr(ecm, "_parent_name", None)
    if fn is not None:
        try:
            return fn(ent)
        except Exception:
            pass
    for key in ("formation", "release", "tether"):
        block = ent.get(key)
        if isinstance(block, dict) and block.get("parent"):
            return block["parent"]
    return ent.get("parent") or ent.get("towed_by")


def _is_ground_bound(ctx, spec, pos):
    """Is this entity sitting on the ground rather than flying?

    Scenes rarely declare motion_type, so fall back to the geometry: when the
    scene actually draws a ground plane, anything within a few metres of it is
    scenery. Without this a person standing by a parked car tripped the airborne
    10 m separation budget, which is not what that budget is for.
    """
    if spec.get("motion_type") == "ground":
        return True
    if ctx.cfg.get("ground") is None:
        return False
    gz = ctx._ground_z
    if gz is None:
        try:
            gz = float((ctx.cfg.get("ground") or {}).get("z", 0.0))
        except (TypeError, ValueError):
            gz = 0.0
    try:
        return abs(float(pos[2]) - float(gz)) <= 3.0
    except (TypeError, ValueError):
        return False


def _c3_entities(ctx, f):
    """Physical target entities at frame f: labelled objects + ECM bodies."""
    st = ctx.state(f)
    t = float(st["t"])
    cp = ctx.plan(f // ctx.fpc)
    out = []
    for spec in ctx.labelled:
        p = np.asarray(cp.object_state(spec, t)[0], float)
        out.append((spec.get("name"), p, ctx.radius(spec), None,
                    _is_ground_bound(ctx, spec, p)))
    try:
        ecm_now = ctx.ecm_states(f)
    except Exception:
        ecm_now = {}
    for eid, ent in ecm_now.items():
        if ent.get("kind") not in ecm.BODY_KINDS or not ent.get("visible", True):
            continue
        radius = float(ecm.radius_at(ent, t))
        if radius <= 1e-6:
            continue
        out.append((eid, np.asarray(ent["pos"], float), radius,
                    _entity_parent(ent), False))
    return out


def _check_C3(ctx):
    worst = None      # (distance, need, a, b, frame)
    n_pairs = 0
    for f in ctx.indices:
        ents = _c3_entities(ctx, f)
        for i in range(len(ents)):
            for j in range(i + 1, len(ents)):
                a, b = ents[i], ents[j]
                if a[3] and a[3] == b[0]:
                    continue                  # a dispensed body at its parent
                if b[3] and b[3] == a[0]:
                    continue
                d = _norm(a[1] - b[1])
                ground = bool(a[4]) or bool(b[4])
                need = a[2] + b[2] + (C3_GROUND_CLEARANCE_M if ground
                                      else C3_CLEARANCE_M)
                n_pairs += 1
                if worst is None or d - need < worst[0] - worst[1]:
                    worst = (d, need, a[0], b[0], f)
    if worst is None:
        return _skip("C3", "fewer than two physical target entities")
    ok = worst[0] - worst[1] >= -1e-9
    if ok:
        detail = ("%d entity pair(s); closest %s ~ %s: %.2f m >= %.2f m "
                  "(r_i + r_j + %g m)" % (n_pairs, worst[2], worst[3], worst[0],
                                          worst[1], C3_CLEARANCE_M))
    else:
        detail = ("%s ~ %s at frame %d: %.2f m < %.2f m required "
                  "(r_i + r_j + %g m)" % (worst[2], worst[3], worst[4], worst[0],
                                          worst[1], C3_CLEARANCE_M))
    return _result("C3", ok, detail, worst[0], worst[1])


def _check_C4(ctx):
    pairs = ctx.pairs()
    if not pairs:
        return _skip("C4", "no formation declared")
    h = min(1e-2, max(1e-4, ctx.dt * 0.25)) if ctx.dt > 0 else 1e-4
    worst = (0.0, None)
    for _gname, lead, fol, _tau in pairs:
        for f in ctx.indices:
            st = ctx.state(f)
            t = float(st["t"])
            cp = ctx.plan(f // ctx.fpc)
            p0 = np.asarray(cp.object_state(lead, t)[0], float)
            p1 = np.asarray(cp.object_state(fol, t)[0], float)
            p0b = np.asarray(cp.object_state(lead, t + h)[0], float)
            p1b = np.asarray(cp.object_state(fol, t + h)[0], float)
            d0, d1 = p1 - p0, p1b - p0b
            n0, n1 = _norm(d0), _norm(d1)
            if n0 < 1e-6 or n1 < 1e-6:
                continue
            R0 = _attitude_R(cp, lead, t)
            R1 = _attitude_R(cp, lead, t + h)
            u0 = R0.T @ (d0 / n0)
            u1 = R1.T @ (d1 / n1)
            c = float(np.clip(np.dot(u0, u1), -1.0, 1.0))
            rate = math.degrees(math.acos(c)) / h
            if rate > worst[0]:
                worst = (rate, (lead.get("name"), fol.get("name"), f))
    ok = worst[0] <= C4_BEARING_RATE_DPS + 1e-9
    if ok:
        detail = ("max body-frame bearing rate %.3f deg/s <= %g deg/s (%d pair(s))"
                  % (worst[0], C4_BEARING_RATE_DPS, len(pairs)))
    else:
        detail = ("%s ~ %s at frame %d: bearing rate %.2f deg/s > %g deg/s"
                  % (worst[1][0], worst[1][1], worst[1][2], worst[0],
                     C4_BEARING_RATE_DPS))
    return _result("C4", ok, detail, worst[0], C4_BEARING_RATE_DPS)


def _check_C5(ctx):
    pairs = ctx.pairs()
    if not pairs:
        return _skip("C5", "no formation declared")
    worst = None
    max_turn = 0.0
    for _gname, lead, fol, tau in pairs:
        form = fol.get("formation") if isinstance(fol.get("formation"), dict) else {}
        explicit = form.get("offset_body")
        by_clip = {}
        for f in ctx.indices:
            by_clip.setdefault(f // ctx.fpc, []).append(f)
        for clip, frames in by_clip.items():
            cp = ctx.plan(clip)
            # The reference offset is the DECLARED body-frame offset (or, as
            # plan.ClipPlan derives it, the nominal pos difference rotated into
            # the lead frame at clip start).  Using the observed offset at t0
            # would absorb the lag itself and hide the very error C5 judges.
            R0 = _attitude_R(cp, lead, 0.0)
            if explicit is not None:
                o_decl = np.asarray(explicit, float)
            else:
                o_decl = R0.T @ (np.asarray(fol.get("pos", [0.0, 0.0, 0.0]), float)
                                 - np.asarray(lead.get("pos", [0.0, 0.0, 0.0]), float))
            r_decl = _norm(o_decl)
            if r_decl < 1e-6:
                continue
            for f in frames:
                st = ctx.state(f)
                t = float(st["t"])
                R = _attitude_R(cp, lead, t)
                o = np.asarray(cp.object_state(fol, t)[0], float) - np.asarray(
                    cp.object_state(lead, t)[0], float)
                e = _norm(o - R @ o_decl)
                omega = abs(_yaw_rate_dps(cp, lead, t))
                max_turn = max(max_turn, omega)
                bound = C5_LAG_FACTOR * r_decl * math.radians(omega) * tau
                excess = e - bound
                if worst is None or excess > worst[0]:
                    worst = (excess, e, bound, (lead.get("name"), fol.get("name")),
                             f, omega, r_decl, tau)
    if worst is None:
        return _skip("C5", "declared formation offset is zero for every pair")
    if max_turn < math.degrees(C5_MIN_TURN_RATE_RAD_S):
        return _skip("C5", "formation never turns (max %.2f deg/s): lag bound is vacuous"
                     % max_turn)
    ok = worst[0] <= 1e-9
    if ok:
        detail = ("max formation error %.2f m <= bound %.2f m "
                  "(R=%.1f m, omega=%.2f deg/s, tau=%g s)"
                  % (worst[1], worst[2], worst[6], worst[5], worst[7]))
    else:
        detail = ("%s ~ %s at frame %d: formation error %.2f m > 2*R*omega*tau = "
                  "%.2f m (R=%.1f m, omega=%.2f deg/s, tau=%g s)"
                  % (worst[3][0], worst[3][1], worst[4], worst[1], worst[2],
                     worst[6], worst[5], worst[7]))
    return _result("C5", ok, detail, worst[1], worst[2])


def _check_C6(ctx):
    if not ctx.has_camera():
        return _skip("C6", "no imaging view configured for this clip")
    worst = (0.0, None)
    for f in ctx.indices:
        st = ctx.state(f)
        t = float(st["t"])
        cp = ctx.plan(f // ctx.fpc)
        for vname, view in (st.get("views") or {}).items():
            p0, right, up, fwd = scene.sensor_basis(
                view["pos"], view["look_at"], (0.0, 0.0, 1.0),
                view.get("roll_deg", 0.0) or 0.0)
            W = float(view.get("width") or 480)
            H = float(view.get("height") or 360)
            fov_h = math.radians(float(view.get("fov_deg") or 30.0))
            fov_v = 2.0 * math.atan(math.tan(fov_h / 2.0) * H / max(W, 1.0))
            for spec in ctx.labelled:
                p = np.asarray(cp.object_state(spec, t)[0], float)
                d = p - p0
                z = float(np.dot(d, fwd))
                if z <= 1e-3:
                    fill = 1e6
                    h_off = v_off = float("nan")
                    h_lim = v_lim = 0.0
                else:
                    h_off = float(np.dot(d, right))
                    v_off = float(np.dot(d, up))
                    h_lim = z * math.tan(fov_h / 2.0)
                    v_lim = z * math.tan(fov_v / 2.0) * C6_VERTICAL_FILL
                    fill = max(abs(h_off) / max(h_lim, 1e-9),
                               abs(v_off) / max(v_lim, 1e-9))
                if worst[1] is None or fill > worst[0]:
                    worst = (fill, (spec.get("name"), vname, f, z,
                                    h_off, v_off, h_lim, v_lim))
    item = worst[1]
    if item is None:
        return _skip("C6", "no labelled object")
    ok = worst[0] < 1.0
    detail = ("%s in view %r at frame %d: offset (%.1f, %.1f) m vs half-frame "
              "(%.1f, %.1f) m -> fill %.2f %s 1.0"
              % (item[0], item[1], item[2], item[4], item[5], item[6], item[7],
                 worst[0], "<" if ok else ">="))
    return _result("C6", ok, detail, worst[0], 1.0)


# ===========================================================================
# E. dispensed-body timing
# ===========================================================================
def _e_entities(ctx):
    """(spec, released, towed, chaff, flares) from the loaded entity schema."""
    spec = ctx.ecm_spec()
    if not spec["enabled"]:
        return spec, [], [], [], []
    ents = spec["entities"]
    released = [e for e in ents if e.get("link") == "released"]
    towed = [e for e in ents if e.get("link") == "towed"]
    chaff = [e for e in ents if e.get("kind") == "chaff_cloud"]
    flares = [e for e in ents if e.get("kind") == "infrared_flare"]
    return spec, released, towed, chaff, flares


def _parent_track(ctx, ent, cp):
    """(parent name, parent Track or None) for a linked entity."""
    parent = None
    for key in ("release", "tether", "formation"):
        block = ent.get(key)
        if isinstance(block, dict) and block.get("parent"):
            parent = block["parent"]
            break
    if parent is None:
        parent = ent.get("parent") or ent.get("towed_by")
    tracks = getattr(cp, "tracks", None) or {}
    return parent, tracks.get(parent)


def _entity_state(ent, t, cp):
    """Analytic state of one entity; the planner Track is the parent source."""
    return ecm.entity_state(ent, float(t), tracks=getattr(cp, "tracks", None))


def _air_at(release, u):
    """Air-mass velocity at age u, from the same function ecm uses."""
    fn = getattr(ecm, "_air_state", None)
    if fn is not None:
        return np.asarray(fn(release, float(u))[0], float)
    air = release.get("air") or {}
    wind = np.asarray(air.get("wind_mps") or [0.0, 0.0, 0.0], float)
    fall = float(air.get("fall_mps") or 0.0)
    if str(air.get("fall_mode", "const")).lower() == "tanh" and fall > 0.0:
        vt = fall
        return wind + np.array([0.0, 0.0,
                                -vt * math.tanh(motion.G * float(u) / vt)])
    return wind + np.array([0.0, 0.0, -fall])


def _drag_unexplained(speed0, drag):
    """The initial deceleration the declared drag law itself predicts."""
    mode = str((drag or {}).get("mode", "lag")).lower()
    if mode == "quadratic":
        L = float((drag or {}).get("L_m") or 0.0)
        return (speed0 * speed0 / L) if L > 1e-9 else 0.0
    tau = float((drag or {}).get("tau_s") or 0.0)
    return (speed0 / tau) if tau > 1e-9 else 0.0


def _check_E(ctx):
    """All seven E rules share one pass over released / towed / chaff bodies."""
    ids = ("E1", "E2", "E3", "E4", "E5", "E6", "E7")
    try:
        spec, released, towed, chaff, flares = _e_entities(ctx)
    except Exception as exc:
        return [_result(i, False, "ecm block invalid: %s: %s"
                        % (type(exc).__name__, exc), None, None) for i in ids]
    if not spec["enabled"]:
        return [_skip(i, "scene has no ecm block") for i in ids]
    if not released and not towed and not chaff and not flares:
        return [_skip(i, "no dispensed / towed / chaff body in the scene")
                for i in ids]

    trackers = {i: dict(applied=False, ok=True, worst=None, texts=[])
                for i in ids}

    def note(cid, ok, value, limit, text):
        tr = trackers[cid]
        tr["applied"] = True
        tr["ok"] = tr["ok"] and ok
        if not ok:
            tr["texts"].append(text)
        if value is None or limit in (None, 0.0):
            sev = 0.0
        else:
            sev = abs(float(value)) / max(abs(float(limit)), 1e-12)
        if tr["worst"] is None or sev >= tr["worst"][0]:
            tr["worst"] = (sev, value, limit, text)

    for clip in range(ctx.n_clips):
        cp = ctx.plan(clip)
        n_in_clip = min(ctx.fpc, max(ctx.n - clip * ctx.fpc, 0))
        t_end = max((n_in_clip - 1) * ctx.dt, 0.0) if n_in_clip else 0.0

        for ent in towed:
            parent, P = _parent_track(ctx, ent, cp)
            if P is None:
                note("E1", False, None, None,
                     "%s: tether.parent %r is not a scene object" % (ent["id"], parent))
                continue
            tw = ent.get("tether") or {}
            length = float(tw.get("length_m", 0.0))
            for t in (0.0, t_end * 0.5, t_end):
                st = _entity_state(ent, t, cp)
                d = _norm(np.asarray(st["pos"], float)
                          - np.asarray(P.pos(t), float))
                gap = abs(d - length)
                note("E1", gap <= E1_ATTACH_EPS_M, gap, E1_ATTACH_EPS_M,
                     "%s towed at t=%.2f s: tether length %.4f m vs declared %.4f m"
                     % (ent["id"], t, d, length))

        for ent in released:
            _check_one_released(ctx, ent, cp, t_end, note)

        # E6: settling is a property of the body, so an independent cloud gets
        # the same check as a released one.
        for ent in chaff:
            _check_settling(ctx, ent, cp, t_end, note)
        for ent in flares:
            _check_flare_settling(ctx, ent, cp, t_end, note)

    reasons = {
        "E1": "no released or towed body",
        "E2": "no released body",
        "E3": "no released body",
        "E4": "no released body",
        "E5": "no released body",
        "E6": "no chaff cloud with a fall speed / flare with a vertical model",
        "E7": "no released body",
    }
    rows = []
    for cid in ids:
        tr = trackers[cid]
        if not tr["applied"]:
            rows.append(_skip(cid, reasons[cid]))
            continue
        w = tr["worst"] or (0.0, None, None, "")
        detail = "; ".join(tr["texts"][:3]) if not tr["ok"] else (
            w[3] or "all dispensed bodies satisfy %s" % cid)
        rows.append(_result(cid, tr["ok"], detail, w[1], w[2]))
    return rows


def _check_settling(ctx, ent, cp, t_end, note):
    """E6 for one chaff cloud, released or independent."""
    eid = ent["id"]
    release = ent.get("release")
    if release is not None:
        air = release.get("air") or {}
        fall = float(air.get("fall_mps") or 0.0)
        tr = float(release.get("t_s") or 0.0)
        tau = float((release.get("drag") or {}).get("tau_s") or 0.0)
        # Terminal settling is only observable after the drag law has had time
        # to act; a clip that ends at 2*tau would report the ejection transient.
        if tau <= 1e-9 or (t_end - tr) < 5.0 * tau - 1e-9:
            return
        t_eval = tr + 5.0 * tau
        st = _entity_state(ent, t_eval, cp)
        v_z = float(st["vel"][2])
    else:
        fall = float(ent.get("fall_mps") or 0.0)
        t_eval = t_end
        st = _entity_state(ent, t_eval, cp)
        v_z = float(st["vel"][2])
    if fall < 0.01:
        return
    frac = abs(v_z + fall) / fall
    note("E6", frac <= E6_FALL_TOL_REL, frac, E6_FALL_TOL_REL,
         "%s at t=%.2f s: v_z %+0.2f m/s, fall %.2f -> |v_z+fall|/fall = %.3f"
         % (eid, t_eval, v_z, fall, frac))


def _check_flare_settling(ctx, ent, cp, t_end, note):
    """E6 for an infrared flare: |v_z| must tend to v_t = sqrt(g*L)."""
    eid = ent["id"]
    release = ent.get("release") or {}
    air = release.get("air") or {}
    drag = release.get("drag") or {}
    fall = float(air.get("fall_mps") or 0.0)
    L = float(drag.get("L_m") or 0.0)
    if fall > 0.01:
        v_t = fall
    elif L > 1e-6:
        v_t = math.sqrt(motion.G * L)
    else:
        note("E6", True, None, None,
             "%s: no fall speed or drag length; v_t is not defined" % eid)
        return
    tr = float(release.get("t_s") or 0.0)
    tau = float(drag.get("tau_s") or 0.0)
    if tau <= 1e-9 or (t_end - tr) < 5.0 * tau - 1e-9:
        return
    t_eval = tr + 5.0 * tau
    st = _entity_state(ent, t_eval, cp)
    v_z = float(st["vel"][2])
    frac = abs(abs(v_z) - v_t) / v_t
    note("E6", frac <= 0.10, frac, 0.10,
         "%s at t=%.2f s: |v_z| = %.2f m/s vs terminal v_t = %.2f m/s "
         "(L=%.1f m, fall=%.1f m/s)" % (eid, t_eval, abs(v_z), v_t, L, fall))


def _check_one_released(ctx, ent, cp, t_end, note):
    """E1..E7 for one released body."""
    eid = ent["id"]
    rel = ent.get("release") or {}
    parent, P = _parent_track(ctx, ent, cp)
    if P is None:
        for cid in ("E1", "E2", "E3", "E4", "E5", "E7"):
            note(cid, False, None, None,
                 "%s: release.parent %r is not a scene object" % (eid, parent))
        return
    tr = float(rel.get("t_s") or 0.0)
    if tr > t_end + 1e-9:
        for cid in ("E1", "E2", "E3", "E4", "E5", "E7"):
            note(cid, True, None, None,
                 "%s: release at %.2f s is beyond the clip (%.2f s); the body "
                 "stays attached for the whole clip" % (eid, tr, t_end))
        return
    mount = np.asarray(rel.get("mount_body") or [0.0, 0.0, 0.0], float)
    eject = np.asarray(rel.get("eject_body") or [0.0, 0.0, 0.0], float)
    drag = rel.get("drag") or {}
    tau = float(drag.get("tau_s") or 0.0)
    L = float(drag.get("L_m") or 0.0)
    h = E_H

    def attached(t):
        R = np.asarray(P.frame(t), float)
        return np.asarray(P.pos(t), float) + R @ mount

    def attached_v(t):
        return ((attached(t + h) - attached(t - h)) / (2.0 * h))

    # ---- E1: strictly attached before release -------------------------
    samples = [tr]
    if tr > 2.0 * h:
        samples.append(tr - min(0.2, 0.5 * tr))
    worst_gap = 0.0
    worst_t = tr
    for t in samples:
        st = _entity_state(ent, t, cp)
        gap = _norm(np.asarray(st["pos"], float) - attached(t))
        if gap > worst_gap:
            worst_gap, worst_t = gap, t
    st_r = _entity_state(ent, tr, cp)
    # The velocity clause belongs to the ATTACHED phase.  At exactly t_r the
    # model has already applied the ejection (E2), so sampling there would
    # report the intended impulse as an attachment error.
    vgap = 0.0
    if len(samples) > 1:
        t_pre = samples[1]
        st_pre = _entity_state(ent, t_pre, cp)
        vgap = _norm(np.asarray(st_pre["vel"], float) - attached_v(t_pre))
    note("E1", worst_gap <= E1_ATTACH_EPS_M and vgap <= E1_SPEED_EPS_MPS,
         max(worst_gap, vgap), E1_ATTACH_EPS_M,
         "%s: pre-release mount gap %.2e m at t=%.3f s, velocity gap %.2e m/s"
         % (eid, worst_gap, worst_t, vgap))

    # ---- E2: ejection vector ------------------------------------------
    dv = np.asarray(st_r["vel"], float) - np.asarray(P.vel(tr), float)
    expected = np.asarray(P.frame(tr), float) @ eject
    speed = _norm(dv)
    vec_err = _norm(dv - expected)
    band_lo, band_hi = E2_EJECT_MPS
    ok2 = band_lo - 1e-9 <= speed <= band_hi + 1e-9 and vec_err <= 1e-6
    note("E2", ok2, speed, band_hi,
         "%s: |v_c - v_p| = %.2f m/s (band [%g, %g]), ejection-vector error %.2e m/s"
         % (eid, speed, band_lo, band_hi, vec_err))

    # ---- E3/E4: air-relative decay ------------------------------------
    air0 = _air_at(rel, 0.0)
    vrel0 = np.asarray(st_r["vel"], float) - air0
    speed0 = _norm(vrel0)
    life_end = (ent.get("life") or {}).get("end_s")
    end = t_end if life_end is None else min(t_end, float(life_end))
    samples = []
    if end > tr + 2.0 * h:
        for u in np.linspace(tr + 2.0 * h, end, 9):
            st = _entity_state(ent, float(u), cp)
            samples.append((float(u), _norm(np.asarray(st["vel"], float)
                                            - _air_at(rel, float(u) - tr))))
    if len(samples) >= 2:
        deriv = [(yb - ya) / max(tb - ta, 1e-12)
                 for (ta, ya), (tb, yb) in zip(samples, samples[1:])]
        worst_d = max(deriv)
        note("E3", worst_d <= E3_DECAY_EPS, worst_d, 0.0,
             "%s: max d|v_rel_air|/dt = %.3e m/s^2 (must be <= 0)"
             % (eid, worst_d))
    else:
        note("E3", True, None, None,
             "%s: clip ends before a decay slope can be sampled" % eid)

    if speed0 <= 1e-9:
        note("E4", True, 0.0, E4_DECAY_TAIL, "%s: no air-relative velocity at release" % eid)
    elif tau <= 1e-9:
        note("E4", False, 1.0, E4_DECAY_TAIL,
             "%s: drag.tau_s = 0; there is no decay time constant" % eid)
    else:
        t4 = tr + E4_DECAY_TIME_FACTOR * tau
        st4 = _entity_state(ent, t4, cp)
        vr4 = _norm(np.asarray(st4["vel"], float) - _air_at(rel, t4 - tr))
        frac = vr4 / speed0
        limit4 = E4_DECAY_TAIL * E4_TAIL_SLACK
        note("E4", frac <= limit4, frac, limit4,
             "%s (%s drag, L=%.1f m, tau=%.2f s): |v_rel_air| at t_r+5*tau = "
             "%.3f m/s = %.2f %% of v_rel0 (limit %.1f %%)"
             % (eid, drag.get("mode", "lag"), L, tau, vr4, 100.0 * frac,
                100.0 * limit4))

    # ---- E5: separation rate tends to |v_p - v_air| -------------------
    if tau > 1e-9:
        t_eval = min(tr + E4_DECAY_TIME_FACTOR * tau, end)
        converged = (t_eval - tr) >= 3.0 * tau - 1e-9
    else:
        t_eval = end
        converged = (t_eval - tr) >= max(0.5, 3.0 * max(ctx.dt, 1e-3))
    if not converged:
        note("E5", True, None, None,
             "%s: clip ends %.2f s after release, before the rate converges"
             % (eid, max(t_eval - tr, 0.0)))
    else:
        st_e = _entity_state(ent, t_eval, cp)
        vp = np.asarray(P.vel(t_eval), float)
        air_e = _air_at(rel, t_eval - tr)
        expected_speed = _norm(vp - air_e)
        if expected_speed < E5_MIN_PARENT_AIRSPEED_MPS:
            note("E5", True, None, None,
                 "%s: parent airspeed %.2f m/s is too low for the rate to be "
                 "diagnostic" % (eid, expected_speed))
        else:
            dp = np.asarray(st_e["pos"], float) - np.asarray(P.pos(t_eval), float)
            dvel = np.asarray(st_e["vel"], float) - vp
            sep = float(np.dot(dp, dvel)) / max(_norm(dp), 1e-9)
            tol = max(E5_RATE_TOL_REL * expected_speed, E5_RATE_TOL_MPS)
            err = abs(sep - expected_speed)
            angle = None
            if _norm(dvel) > 1.0:
                ref = -(vp - air_e)
                c = float(np.clip(np.dot(dvel, ref) / max(
                    _norm(dvel) * _norm(ref), 1e-12), -1.0, 1.0))
                angle = math.degrees(math.acos(c))
            ok5 = err <= tol and (angle is None or angle <= E5_ANGLE_TOL_DEG)
            note("E5", ok5, err, tol,
                 "%s at t=%.2f s: separation rate %.2f m/s vs |v_p - v_air| "
                 "%.2f m/s (err %.2f, tol %.2f; direction %s)"
                 % (eid, t_eval, sep, expected_speed, err, tol,
                    "-" if angle is None else "%.1f deg" % angle))

    # ---- E7: release continuity / unexplained acceleration -------------
    gap_r = _norm(np.asarray(st_r["pos"], float) - attached(tr))
    eject_world = np.asarray(P.frame(tr), float) @ eject
    vel_err = _norm((np.asarray(st_r["vel"], float)
                     - np.asarray(P.vel(tr), float)) - eject_world)
    if tr > 3.0 * h and tr + 3.0 * h <= max(t_end, tr) + 1e9:
        v_m2 = np.asarray(_entity_state(ent, tr - 2.0 * h, cp)["vel"], float)
        v_m1 = np.asarray(_entity_state(ent, tr - h, cp)["vel"], float)
        v_p1 = np.asarray(_entity_state(ent, tr + 2.0 * h, cp)["vel"], float)
        v_p2 = np.asarray(_entity_state(ent, tr + 3.0 * h, cp)["vel"], float)
        a_pre = (v_m1 - v_m2) / h
        a_post = (v_p2 - v_p1) / h
        jump = _norm(a_post - a_pre)
        drag_a = _drag_unexplained(speed0, drag)
        unexplained = max(0.0, jump - 1.05 * drag_a)
        ok7 = (gap_r <= E1_ATTACH_EPS_M and vel_err <= 1e-6
               and unexplained <= E7_ACCEL_JUMP_MPS2)
        note("E7", ok7, unexplained, E7_ACCEL_JUMP_MPS2,
             "%s: release gap %.2e m, velocity-step error %.2e m/s, "
             "acceleration jump %.1f m/s^2 (drag predicts %.1f, unexplained %.1f)"
             % (eid, gap_r, vel_err, jump, drag_a, unexplained))
    else:
        ok7 = gap_r <= E1_ATTACH_EPS_M and vel_err <= 1e-6
        note("E7", ok7, max(gap_r, vel_err), E1_ATTACH_EPS_M,
             "%s: release gap %.2e m, velocity-step error %.2e m/s "
             "(release too close to t=0 for an acceleration sample)"
             % (eid, gap_r, vel_err))


# ===========================================================================
# F. sensor consistency
# ===========================================================================
def _check_F1(ctx):
    worlds, errors = ctx.worlds()
    if not worlds:
        text = "; ".join(errors[:2]) or "no frames"
        if any("FileNotFoundError" in e or "No such file" in e for e in errors):
            return _skip("F1", "geometry unavailable (model file missing): " + text)
        return _result("F1", False, "could not build the sensor world: " + text,
                       None, None)
    pos_err = vel_err = sens_err = 0.0
    proj_err = 0.0
    proj_info = None
    radar_cfg = (ctx.cfg.get("sensors") or {}).get("radar") or {}
    fc = _finite(radar_cfg.get("fc_ghz"))
    lam = 299792458.0 / (fc * 1e9) if fc else None
    for f, st, world in worlds:
        cp = ctx.plan(f // ctx.fpc)
        t = float(st["t"])
        sensor_v = np.asarray(st.get("sensor_velocity_mps") or [0.0, 0.0, 0.0], float)
        radar_block = (st["cfg"].get("sensors") or {}).get("radar") or {}
        if radar_cfg:
            used = _vec3(radar_block.get("vel"))
            if used is not None:
                sens_err = max(sens_err, _norm(used - sensor_v))
        # index, not name: a duplicated name is G1's finding, and keying the
        # world by name would make F1 judge the wrong body on top of it.
        for idx, spec in enumerate(ctx.objects):
            if not spec.get("label", True):
                continue
            o = world.objs[idx] if idx < len(world.objs) else None
            if o is None:
                continue
            name = spec.get("name")
            p_plan, v_plan, _ = cp.object_state(spec, t)
            pos_err = max(pos_err, _norm(np.asarray(o["pos"], float)
                                         - np.asarray(p_plan, float)))
            vel_err = max(vel_err, _norm(np.asarray(o["vel"], float)
                                         - np.asarray(v_plan, float)))
            if lam is None or not radar_cfg:
                continue
            # dataset.py aims radar/lidar along the primary view and writes
            # sensor_velocity_mps into their vel; use that same pose here, not
            # the sensor block's authored (often stale) base pose.
            view = next(iter((st.get("views") or {}).values()), None)
            cam = (np.asarray(view["pos"], float) if view is not None
                   else _vec3(radar_block.get("pos")))
            if cam is None:
                continue
            cam_v = sensor_v
            d = np.asarray(p_plan, float) - cam
            rng = _norm(d)
            if rng < 1e-3:
                continue
            u = d / rng
            vr = -float(np.dot(np.asarray(v_plan, float) - cam_v, u))

            def _range(tt):
                cpos = np.asarray(cp.camera(tt)[0], float)
                return _norm(np.asarray(cp.object_state(spec, tt)[0], float) - cpos)

            vr_geo = -float((_range(t + E_H) - _range(t - E_H)) / (2.0 * E_H))
            err = abs(vr - vr_geo)
            if err > proj_err:
                proj_err = err
                proj_info = (name, vr, vr_geo, -2.0 * vr / lam)
    source_err = max(pos_err, vel_err, sens_err)
    ok = source_err <= F1_SOURCE_EPS and proj_err <= F1_DOPPLER_TOL_MPS
    detail = ("world state == plan state for %d object(s) (max %.2e m, %.2e m/s); "
              "radar mount velocity error %.2e m/s; "
              "LOS radial velocity vs geometric derivative %.3e m/s"
              % (len(ctx.labelled), pos_err, vel_err, sens_err, proj_err))
    if proj_info is not None and lam is not None:
        detail += ("; e.g. %s v_r=%+.2f m/s (f_d=%+.1f Hz at %.2f mm)"
                   % (proj_info[0], proj_info[1], proj_info[3], lam * 1e3))
    if radar_cfg:
        value, limit = proj_err, F1_DOPPLER_TOL_MPS
    else:
        value, limit = source_err, F1_SOURCE_EPS
    return _result("F1", ok, detail, value, limit)


def _check_F2(ctx):
    if not ctx.has_camera():
        return _skip("F2", "no imaging view configured for this clip")
    worst_r = (0.0, None)
    worst_p = (0.0, None)
    for f in ctx.indices:
        st = ctx.state(f)
        t = float(st["t"])
        cp = ctx.plan(f // ctx.fpc)
        primary = ctx.labelled[0]
        p = np.asarray(cp.object_state(primary, t)[0], float)
        size = float(getattr(cp, "size_used", cp.size_m))
        for vname, view in (st.get("views") or {}).items():
            r_geo = _norm(p - np.asarray(view["pos"], float))
            r_plan = float(st.get("range_m") or 0.0)
            if r_plan > 1e-9:
                rrel = abs(r_geo - r_plan) / r_plan
                if rrel > worst_r[0]:
                    worst_r = (rrel, (vname, f, r_geo, r_plan))
            px_plan = float((st.get("view_px") or {}).get(vname) or 0.0)
            fov = math.radians(float(view.get("fov_deg") or 30.0))
            width = float(view.get("width") or 1.0)
            px_geo = size * width / (2.0 * max(r_geo, 1e-6) * math.tan(fov / 2.0))
            if px_plan > 1e-9:
                perr = abs(px_plan - px_geo) / px_plan
                if perr > worst_p[0]:
                    worst_p = (perr, (vname, f, px_plan, px_geo))
    ok = worst_r[0] <= F2_RANGE_TOL_REL and worst_p[0] <= F2_PX_TOL_REL
    detail = ("range (label/camera) error %.3f %% <= %g %%; px formula error "
              "%.3f %% <= %g %%" % (100.0 * worst_r[0], 100.0 * F2_RANGE_TOL_REL,
                                    100.0 * worst_p[0], 100.0 * F2_PX_TOL_REL))
    if not ok:
        bits = []
        if worst_r[0] > F2_RANGE_TOL_REL:
            bits.append("range %s frame %d: %.3f vs %.3f m"
                        % (worst_r[1][0], worst_r[1][1], worst_r[1][2], worst_r[1][3]))
        if worst_p[0] > F2_PX_TOL_REL:
            bits.append("px %s frame %d: %.2f vs %.2f"
                        % (worst_p[1][0], worst_p[1][1], worst_p[1][2], worst_p[1][3]))
        detail = "; ".join(bits) + " | " + detail
    return _result("F2", ok, detail, max(worst_r[0], worst_p[0]), F2_PX_TOL_REL)


def _los_rate(angles, t, h):
    """d(az, el)/dt at t, one-sided where the clip clock starts.

    A clip is a window onto a track that has a beginning: plan.ClipPlan's
    position functions publish the right-hand derivative at t=0 but freeze the
    position for t<0 (motion.Waypoints._distance returns the departure point for
    t<=0, measured: p(-1e-3) == p(0) to the last bit).  A centred step across
    that edge therefore differences against a frozen pose and reports exactly
    half the true line-of-sight rate -- measured 0.002451 rad/s against an
    analytic 0.004918 rad/s at frame 0, a 50 % "error" that is a property of the
    difference scheme, not of the geometry.  The threshold stays where the doc
    puts it; the sample switches to a three-point forward formula instead, whose
    truncation error is O(h^2) like the centred one, so the first frame is
    measured as accurately as the rest.

    The rule is "the left neighbour would be outside the scene clock", not
    "t == 0": a planner that does have history before the clip start (motion.Line
    and the seeker solve extend analytically) keeps the centred step through
    t=0, and both schemes agree there to O(h^2).
    """
    a_t = angles(t)
    if t - h >= 0.0:
        a_p, a_m = angles(t + h), angles(t - h)
        return ((a_p[0] - a_m[0]) / (2.0 * h),
                (a_p[1] - a_m[1]) / (2.0 * h))
    a_1, a_2 = angles(t + h), angles(t + 2.0 * h)
    return ((-3.0 * a_t[0] + 4.0 * a_1[0] - a_2[0]) / (2.0 * h),
            (-3.0 * a_t[1] + 4.0 * a_1[1] - a_2[1]) / (2.0 * h))


def _check_F3(ctx):
    if not ctx.has_camera():
        return _skip("F3", "no imaging view configured for this clip")
    h = E_H
    worst = (0.0, None)
    for f in ctx.indices:
        st = ctx.state(f)
        t = float(st["t"])
        cp = ctx.plan(f // ctx.fpc)
        primary = ctx.labelled[0]
        for vname, view in (st.get("views") or {}).items():
            p0, right, up, fwd = scene.sensor_basis(
                view["pos"], view["look_at"], (0.0, 0.0, 1.0),
                view.get("roll_deg", 0.0) or 0.0)

            def angles(tt):
                cpos = np.asarray(cp.camera(tt)[0], float)
                d = np.asarray(cp.object_state(primary, tt)[0], float) - cpos
                x = float(np.dot(d, right))
                y = float(np.dot(d, up))
                z = float(np.dot(d, fwd))
                return math.atan2(x, z), math.atan2(y, math.hypot(x, z))

            d_az, d_el = _los_rate(angles, t, h)
            cpos = np.asarray(cp.camera(t)[0], float)
            cvel = np.asarray(cp.camera_velocity(t), float)
            p = np.asarray(cp.object_state(primary, t)[0], float)
            v = np.asarray(cp.object_state(primary, t)[1], float)
            d = p - cpos
            rng = _norm(d)
            if rng < 1e-3:
                continue
            vrel = v - cvel
            x = float(np.dot(d, right))
            y = float(np.dot(d, up))
            z = float(np.dot(d, fwd))
            vx = float(np.dot(vrel, right))
            vy = float(np.dot(vrel, up))
            vz = float(np.dot(vrel, fwd))
            rng_dot = float(np.dot(d, vrel)) / rng
            den = z * z + x * x
            az_an = (vx * z - x * vz) / den if den > 1e-12 else 0.0
            # sin(el) = y/rng, so cos(el)*el_dot = vy/rng - y*rng_dot/rng^2
            # and cos(el) = hypot(x, z)/rng.  Dividing by rng instead of
            # hypot(x, z) under-counts the rate by 1/cos(el).
            el_an = ((vy - (y / rng) * rng_dot)
                     / max(math.hypot(x, z), 1e-9))
            for name, num, ana in (("az", d_az, az_an), ("el", d_el, el_an)):
                err = abs(num - ana)
                rel = err / max(abs(ana), 1e-9)
                if worst[1] is None or rel > worst[0]:
                    worst = (rel, (name, vname, f, num, ana, err))
    if worst[1] is None:
        return _skip("F3", "target is at the camera origin")
    it = worst[1]
    ok = worst[0] <= F3_LOS_RATE_TOL_REL
    detail = ("%s rate %s frame %d: numeric %.6f rad/s vs analytic %.6f rad/s "
              "(error %.4f %% <= %g %%)"
              % (it[0], it[1], it[2], it[3], it[4], 100.0 * worst[0],
                 100.0 * F3_LOS_RATE_TOL_REL))
    return _result("F3", ok, detail, worst[0], F3_LOS_RATE_TOL_REL)


def _check_F4(ctx):
    try:
        manifest = ctx.manifest()
    except Exception as exc:
        return _result("F4", False, "optical manifest unavailable: %s" % exc,
                       None, F4_CROSS_CHANNEL_EPS_M)
    worlds, errors = ctx.worlds()
    if not worlds:
        text = "; ".join(errors[:2]) or "no frames"
        if any("FileNotFoundError" in e or "No such file" in e for e in errors):
            return _skip("F4", "geometry unavailable (model file missing): " + text)
        return _result("F4", False, "sensor world unavailable: " + text, None,
                       F4_CROSS_CHANNEL_EPS_M)
    worst = (0.0, None)
    missing = []
    for f, st, world in worlds:
        entry = manifest.get(f)
        if entry is None:
            missing.append("frame %d missing from the optical manifest" % f)
            continue
        mposes = entry.get("objects") or {}
        by_name = {o.get("name"): o for o in world.objs}
        for spec in ctx.objects:
            name = spec.get("name")
            if name not in mposes or name not in by_name:
                missing.append("%s absent from optical or sensor channel" % name)
                continue
            gap = _norm(np.asarray(mposes[name].get("pos", [0, 0, 0]), float)
                        - np.asarray(by_name[name]["pos"], float))
            if gap > worst[0]:
                worst = (gap, (name, f, "optical", "sensor world"))
        for oname, obj in by_name.items():
            if not obj.get("ecm"):
                continue
            eid = obj["ecm"].get("id") or oname
            if eid not in mposes:
                missing.append("ECM %s has no optical pose at frame %d" % (eid, f))
                continue
            gap = _norm(np.asarray(mposes[eid].get("pos", [0, 0, 0]), float)
                        - np.asarray(obj["pos"], float))
            if gap > worst[0]:
                worst = (gap, (eid, f, "optical", "ir/radar/lidar"))
    ok = not missing and worst[0] <= F4_CROSS_CHANNEL_EPS_M
    if ok:
        n_ecm = sum(1 for o in worlds[0][2].objs if o.get("ecm"))
        detail = ("optical manifest vs IR/radar/lidar world: %d entity pose(s) "
                  "within %.0f mm" % (len(ctx.objects) + n_ecm,
                                      1000.0 * F4_CROSS_CHANNEL_EPS_M))
    elif missing:
        detail = "; ".join(missing[:3])
        if worst[1] is not None:
            detail += " | pose gap %.3f m" % worst[0]
    else:
        detail = ("%s at frame %d differs between %s and %s by %.4f m > %.3f m"
                  % (worst[1][0], worst[1][1], worst[1][2], worst[1][3],
                     worst[0], F4_CROSS_CHANNEL_EPS_M))
    return _result("F4", ok, detail, worst[0], F4_CROSS_CHANNEL_EPS_M)


# ===========================================================================
# F5. aim: what the aperture is pointed at when the frame is RENDERED
# ===========================================================================
def _sensor_block(ctx, name):
    """The authored config block of one scene sensor (radar / lidar), or {}."""
    blk = ((ctx.cfg.get("sensors") or {}).get(name))
    return dict(blk) if isinstance(blk, dict) else {}


def _first_labelled_pose(ctx, f):
    """World position the dataset's per-frame re-aim points at.

    dataset.py does not render radar/lidar where the scene file puts them: it
    copies the primary imaging view's pose into the sensor and re-aims it at
    "the first labelled object" (the mean of its corners).  That single line is
    invisible in the scene file, so a checker that reads sensors.radar.look_at
    is checking a pose nothing renders.  This is the same rule written once, in
    the published state every label is projected from; the mesh offset between
    an object's origin and the mean of its corners is a few metres, i.e. under
    0.05 deg at these ranges, far below the beam widths the comparison uses.
    """
    st = ctx.state(f)
    obj = ctx.labelled[0]
    return np.asarray(ctx.plan(f // ctx.fpc).object_state(obj, float(st["t"]))[0],
                      float)


def _aim_frame(ctx, name, f, layer):
    """(pos, look_at, up, re_aimed) the sensor is aimed with on this frame.

    The render layer follows dataset.run() exactly, because that is the pose
    written to disk and the one the labels describe:

      pos      the primary imaging view position (radar/lidar are body-fixed to
               the same gimbal as the camera)
      look_at  the view own aim, i.e. follow the camera, which is what keeps
               the apertures co-boresighted
      re-aim   only when the primary target would fall outside THIS sensor
               field of regard (radar: +-span/2, lidar: +-0.45*fov) is the
               frame pointed at the target instead, and the run prints a hint
               saying so.  A checker that ignores this branch is checking a
               pose nothing rendered.

    The config layer is the authored sensors.<name> block, unchanged.
    """
    blk = _sensor_block(ctx, name)
    up = tuple(float(x) for x in blk.get("up", (0.0, 0.0, 1.0)))
    st = ctx.state(f)
    views = st.get("views") or {}
    vp = next(iter(views.values()), None) if views else None
    if layer != "render" or vp is None:
        pos = [float(x) for x in (blk.get("pos") or [0.0, 0.0, 0.0])]
        look_at = [float(x) for x in (blk.get("look_at") or [0.0, 0.0, 0.0])]
        return pos, look_at, up, False
    pos = [float(x) for x in vp["pos"]]
    look_at = [float(x) for x in (vp.get("look_at") or (0.0, 0.0, 1.0))]
    target = _first_labelled_pose(ctx, f)
    _p, right, upv, fwd = scene.sensor_basis(pos, look_at, up)
    d = np.asarray(target, float) - np.asarray(pos, float)
    n = float(np.linalg.norm(d))
    if n <= 1e-6:
        return pos, look_at, up, False
    d = d / n
    az = math.degrees(math.atan2(float(np.dot(d, right)), float(np.dot(d, fwd))))
    el = math.degrees(math.atan2(float(np.dot(d, upv)), float(np.dot(d, fwd))))
    lim_az, lim_el = _field_of_regard_deg(ctx, name)
    if abs(az) > lim_az or abs(el) > lim_el:
        return pos, [float(x) for x in target], up, True
    return pos, look_at, up, False


def _field_of_regard_deg(ctx, name):
    """Half span dataset.run() uses to decide whether the frame needs a re-aim.

    Note this is narrower than _half_aperture_deg: the incident decision is
    "does the target fall inside the scanned span", the acceptance criterion is
    "does a beam still illuminate it".  Both are copied from their owner.
    """
    blk = _sensor_block(ctx, name)
    if name == "radar":
        return (0.5 * float(blk.get("az_span_deg") or 2.0),
                0.5 * float(blk.get("el_span_deg") or 1.6))
    fov = float(blk.get("fov_deg") or 4.0)
    return 0.45 * fov, 0.45 * float(blk.get("vfov_deg") or fov)


def _half_aperture_deg(ctx, name):
    """Half width of the region that still reaches the raster, per axis.

    radar.py accepts a target as long as it is inside the scanned span plus
    half a beam width (a beam illuminates a target the raster centre steps
    over), so that -- not the bare span -- is what "the aperture covers it"
    means.  The defaults and the band preset are read off radar.py itself
    instead of being copied here, because the beam width of a "Ku" antenna is
    the band's, not the module default's.
    lidar is a raster of rays with no beam width: half a pixel is the margin.
    """
    user = _sensor_block(ctx, name)
    if name == "radar":
        from . import radar
        rc = dict(radar.DEFAULTS)
        band = user.get("band")
        if band in radar.RADAR_BANDS:
            for key in ("fc_ghz", "beamwidth_az_deg", "beamwidth_el_deg"):
                if key not in user:
                    rc[key] = radar.RADAR_BANDS[band][key]
        rc.update(user)
        # the same lower bound radar.simulate applies: a beam is never narrower
        # than one raster step
        bw_az = max(float(rc["beamwidth_az_deg"]),
                    float(rc["az_span_deg"]) / max(int(rc["n_az"]), 1))
        bw_el = max(float(rc["beamwidth_el_deg"]),
                    float(rc["el_span_deg"]) / max(int(rc["n_el"]), 1))
        return (0.5 * float(rc["az_span_deg"]) + 0.5 * bw_az,
                0.5 * float(rc["el_span_deg"]) + 0.5 * bw_el)
    from . import lidar
    rc = dict(lidar.DEFAULTS)
    rc.update({k: v for k, v in user.items() if k in lidar.DEFAULTS})
    fov = float(rc["fov_deg"])
    vfov = float(rc["vfov_deg"]) if rc.get("vfov_deg") else \
        fov * float(rc["n_el"]) / float(max(int(rc["n_az"]), 1))
    return (0.5 * fov + 0.5 * fov / max(int(rc["n_az"]), 1),
            0.5 * vfov + 0.5 * vfov / max(int(rc["n_el"]), 1))


def _axis_roll_deg(ctx, name, f, up, right, fwd):
    """Roll between one sensor's az/el axes and the imaging view's, degrees.

    A sensor that follows the camera shares the view's position and aim, so its
    aperture frame can differ from the frame the labels are projected in only
    through the declared up axis (or a roll).  Such a pair is co-boresighted by
    construction, and a roll offset means every consumer that maps radar az/el
    onto the picture is rotated: on a nadir plate the same two apertures ended
    up 51 deg apart because sensor_basis() replaces a degenerate up with +Y, and
    the 2.9 deg-off radar was on the other side of that test.  Returns None when
    there is nothing to compare (no view, or the sensor was re-aimed on purpose).
    """
    st = ctx.state(f)
    views = st.get("views") or {}
    vp = next(iter(views.values()), None) if views else None
    if vp is None:
        return None
    try:
        _p, vright, _vup, vfwd = scene.sensor_basis(
            vp["pos"], vp.get("look_at") or (0.0, 0.0, 1.0),
            (0.0, 0.0, 1.0), float(vp.get("roll_deg", 0.0) or 0.0))
    except (KeyError, TypeError, ValueError):
        return None
    if abs(float(np.dot(np.asarray(vfwd, float), np.asarray(fwd, float)))) < 0.999:
        return None
    a = np.asarray(vright, float)
    b = np.asarray(right, float)
    return math.degrees(math.atan2(float(np.dot(np.asarray(fwd, float),
                                                np.cross(a, b))),
                                   float(np.dot(a, b))))


def _check_F5(ctx, render_aim=True):
    """Every labelled target has to land inside each aperture the renderer uses.

    The layer matters, and not in the direction one expects.  A look-down
    scene leaves sensors.radar.look_at at the world origin, where the target is
    2.3 deg off and the scanned span is +-1.0 deg: judged on the config the
    aperture misses everything, while the render follows the camera and
    re-aims at the target when it has to, so it sees all of them.  Both numbers
    are reported so neither answer is hidden; the verdict follows the layer the
    caller asked for (render by default, because that is what is written to
    disk)."""
    sensors = ctx.cfg.get("sensors") or {}
    candidates = [n for n in ("radar", "lidar")
                  if isinstance(sensors.get(n), dict) and sensors.get(n)]
    if not candidates:
        return _skip("F5", "no radar/lidar aperture configured for this clip")

    layers = {}
    for layer in ("render", "config"):
        worst = (0.0, None)          # (fill, (sensor, name, frame, az, el, lim))
        reached = 0
        total = 0
        reaims = 0
        roll_worst = (0.0, None)
        for f in ctx.indices:
            st = ctx.state(f)
            t = float(st["t"])
            for name in candidates:
                pos, look_at, up, re_aimed = _aim_frame(ctx, name, f, layer)
                if re_aimed:
                    reaims += 1
                p0, right, upv, fwd = scene.sensor_basis(pos, look_at, up)
                if layer == "render" and not re_aimed:
                    # A sensor that follows the camera shares its position and
                    # aim, so the two aperture frames can differ only through
                    # the declared up/roll.  A roll offset there means any
                    # consumer drawing radar az/el on the image is rotated --
                    # measured 51 deg on a nadir plate, where one aperture
                    # crossed sensor_basis()s degenerate-up fallback and the
                    # other did not.
                    roll = _axis_roll_deg(ctx, name, f, up, right, fwd)
                    if roll is not None and abs(roll) > abs(roll_worst[0]):
                        roll_worst = (roll, (name, f))
                lim_az, lim_el = _half_aperture_deg(ctx, name)
                for obj in ctx.labelled:
                    d = (np.asarray(ctx.plan(f // ctx.fpc).object_state(obj, t)[0],
                                    float) - p0)
                    z = float(np.dot(d, fwd))
                    if z <= 1e-6:
                        continue
                    az = math.degrees(math.atan2(float(np.dot(d, right)), z))
                    el = math.degrees(math.atan2(float(np.dot(d, upv)),
                                                 math.hypot(float(np.dot(d, right)),
                                                            z)))
                    total += 1
                    fill = max(abs(az) / max(lim_az, 1e-9),
                               abs(el) / max(lim_el, 1e-9))
                    if fill <= 1.0:
                        reached += 1
                    if fill > worst[0]:
                        worst = (fill, (name, obj.get("name"), f, az, el,
                                        lim_az, lim_el))
        layers[layer] = dict(reached=reached, total=total, worst=worst,
                             reaims=reaims, roll=roll_worst)

    # kept on the context so inspect() can put the layer disagreement in the
    # summary notes instead of burying it in one long detail string
    ctx.aim_layers = layers
    sel = layers["render" if render_aim else "config"]
    ok_cover = sel["total"] > 0 and sel["reached"] == sel["total"]
    ok_roll = abs(sel["roll"][0]) <= F5_AXIS_ROLL_TOL_DEG
    ok = ok_cover and ok_roll
    bits = []
    for layer in ("render", "config"):
        lay = layers[layer]
        if lay["worst"][1] is None:
            bits.append("%s layer: no target in front of the aperture" % layer)
            continue
        w = lay["worst"][1]
        bits.append("%s layer %d/%d reached (worst %s/%s f%d az %+.2f/%.1f "
                    "el %+.2f/%.1f deg, re-aimed on %d frame-sensor pair(s))" %
                    (layer, lay["reached"], lay["total"], w[0], w[1], w[2],
                     w[3], w[5], w[4], w[6], lay["reaims"]))
    if sel["roll"][1] is not None:
        bits.append("co-boresight roll %s frame %d: %+.2f deg (limit %.1f)"
                    % (sel["roll"][1][0], sel["roll"][1][1],
                       sel["roll"][0], F5_AXIS_ROLL_TOL_DEG))
    detail = "; ".join(bits)
    if not ok_roll:
        detail = ("aperture axes are rolled against the imaging view; az/el in "
                  "one channel cannot be drawn on the other") + " | " + detail
    if sel["reached"] != sel["total"] and sel["worst"][1] is not None:
        w = sel["worst"][1]
        detail = ("%s outside the %s aperture at frame %d (az %.2f/%.1f "
                  "el %.2f/%.1f deg)" % (w[1], w[0], w[2], w[3], w[5], w[4], w[6])
                  ) + " | " + detail
    # one comparable number for the row: a rolled pair has to read as a
    # violation even though its coverage alone would have passed
    value = max(sel["worst"][0],
                abs(sel["roll"][0]) / max(F5_AXIS_ROLL_TOL_DEG, 1e-9))
    return _result("F5", ok, detail, value, 1.0)


# ===========================================================================
# G. labels
# ===========================================================================
def _check_G1(ctx):
    issues = []
    names = [o.get("name") for o in ctx.objects]
    if any(not n for n in names):
        issues.append("objects with empty names: %r" % names)
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        issues.append("duplicate object names: %s" % ", ".join(dup))
    try:
        spec = ctx.ecm_spec()
        ids = [e["id"] for e in spec["entities"]]
        if len(set(ids)) != len(ids):
            issues.append("duplicate ecm ids")
        both = sorted(set(ids) & set(names))
        if both:
            issues.append("ecm id collides with an object name: %s" % ", ".join(both))
    except Exception as exc:
        issues.append("ecm block invalid: %s" % exc)
        ids = []
    try:
        render_ids = {o.get("name") for o in ecm.render_objects(ctx.cfg)}
    except Exception:
        render_ids = set()
    worlds, _errors = ctx.worlds()
    world_names = {o.get("name") for o in worlds[0][2].objs} if worlds else set()
    for f in ctx.world_frames:
        try:
            est = ctx.ecm_states(f)
        except Exception:
            est = {}
        for eid, ent in est.items():
            if ent.get("kind") not in ecm.BODY_KINDS or not ent.get("visible", True):
                continue
            # A volumetric body (chaff) is deliberately absent from the optical
            # mesh list: it is a sparse dipole field, not a surface, and drawing
            # it painted a growing opaque ball over the target (see
            # ecm.render_objects).  It must still exist in the sensor world.
            if (eid not in render_ids
                    and ent.get("kind") not in ecm.VOLUMETRIC):
                issues.append("ECM %s exists at frame %d but has no optical mesh spec "
                              "(render_objects(cfg) is evaluated at t=0)" % (eid, f))
            if worlds and eid not in world_names:
                issues.append("ECM %s missing from the sensor world at frame %d"
                              % (eid, f))
    ok = not issues
    detail = ("%d object names + %d ecm ids are unique, disjoint, and present in "
              "both optical and sensor channels" % (len(names), len(ids))
              if ok else "; ".join(issues[:4]))
    return _result("G1", ok, detail, float(len(issues)), 0.0)


def _check_G2(ctx):
    issues = []
    try:
        renders = ecm.render_objects(ctx.cfg)
    except Exception as exc:
        return _result("G2", False, "ecm block invalid: %s" % exc, 1.0, 0.0)
    for item in renders:
        if item.get("label", True):
            issues.append("dispensable %s is rendered with label=True" % item.get("name"))
    cls_names = sorted({o.get("cls", "target") for o in ctx.objects
                        if o.get("label", True)})
    if "ecm" in cls_names:
        issues.append("the COCO categories include the dispensable class 'ecm'")
    try:
        ecm_ids = {e["id"] for e in ctx.ecm_spec()["entities"]}
    except Exception as exc:
        return _result("G2", False, "ecm block invalid: %s" % exc, 1.0, 0.0)
    worlds, _errors = ctx.worlds()
    if worlds:
        f, st, world = worlds[0]
        for obj in world.objs:
            if obj.get("ecm") and obj.get("label", True):
                issues.append("sensor world body %s is labelled as a target"
                              % obj.get("name"))
        if ctx.has_camera():
            view = next(iter((st.get("views") or {}).values()), None)
            if view is not None:
                W = int(view.get("width") or 480)
                H = int(view.get("height") or 360)
                cam = dict(pos=view["pos"], look_at=view["look_at"],
                           up=[0.0, 0.0, 1.0], fov_deg=view["fov_deg"],
                           roll_deg=view.get("roll_deg", 0.0) or 0.0)
                cls_id = {c: i + 1 for i, c in enumerate(cls_names)}
                for obj in world.objs:
                    obj["cls_id"] = (cls_id.get(obj.get("cls"), 0)
                                     if obj.get("label", True) else 0)
                boxes = labels.boxes_from_corners(world, cam, W, H, min_frac=0.0)
                leaked = sorted({obj.get("name") for obj in world.objs
                                 if obj.get("ecm") and obj.get("id") in boxes})
                if leaked:
                    issues.append("dispensable(s) emitted a 2-D box: %s"
                                  % ", ".join(leaked))
                _img, anns = labels.to_coco(boxes, world, 0, "frame.png", W, H)
                labelled_names = {obj.get("name") for obj in world.objs
                                  if obj.get("label", True)}
                bad = sorted({a.get("object_name") for a in anns
                              if a.get("object_name") in ecm_ids
                              and a.get("object_name") not in labelled_names})
                if bad:
                    issues.append("dispensable(s) reached COCO: %s" % ", ".join(bad))
    ok = not issues
    detail = ("%d dispensable body/bodies carry label=False and stay out of the "
              "%d COCO target classes" % (len(renders), len(cls_names))
              if ok else "; ".join(issues[:4]))
    return _result("G2", ok, detail, float(len(issues)), 0.0)


# ===========================================================================
# public entry point
# ===========================================================================
def inspect(cfg_or_recipe, *, frames=None, render_aim=True):
    """Check one multi-target scene or recipe and return a report dict.

    Parameters
    ----------
    cfg_or_recipe : dict | str
        A scene config (the JSON shape scene.load_config returns), a scenegen
        recipe, or a path to either.
    frames : None | int | sequence
        None evaluates every frame; an int samples that many evenly spaced
        frames; a sequence names frame indices.  The B-group envelope pass is
        always the full clip (motion_diagnose reads the whole scene).
    render_aim : bool
        Judge F5 (aperture coverage) on the pose the dataset actually renders
        with -- radar/lidar re-aimed at the primary target every frame -- or,
        with False, on the pose the scene file authored.  The report always
        prints both layers; this only picks which one decides pass/fail.

    Returns
    -------
    dict
        {ok, checks: [{id, group, ok, detail, value, limit}], summary}.
        Every check in CHECK_SPECS appears exactly once, whether it passed,
        failed or was not applicable (applicable=False).
    """
    cfg = _coerce_config(cfg_or_recipe)
    ctx = _Context(cfg, frames=frames)
    checks = [
        _check_A1(ctx), _check_A2(ctx), _check_A3(ctx),
        *_check_B(ctx),
        _check_C1(ctx), _check_C2(ctx), _check_C3(ctx),
        _check_C4(ctx), _check_C5(ctx), _check_C6(ctx),
        *_check_E(ctx),
        _check_F1(ctx), _check_F2(ctx), _check_F3(ctx), _check_F4(ctx),
        _check_F5(ctx, render_aim=bool(render_aim)),
        _check_G1(ctx), _check_G2(ctx),
    ]
    by_id = {c["id"]: c for c in checks}
    checks = [by_id[i] for i in CHECK_IDS]
    failed = [c["id"] for c in checks if not c["ok"]]
    skipped = [c["id"] for c in checks if not c["applicable"]]
    notes = []
    f5 = by_id.get("F5") or {}
    if not render_aim and f5.get("applicable", True):
        notes.append("F5 judged the sensors as the scene file poses them; the "
                     "renderer re-aims radar/lidar at the primary target per "
                     "frame (inspect(..., render_aim=True) reports that layer)")
    layers = getattr(ctx, "aim_layers", None)
    if layers:
        cfg_l = layers["config"]
        ren_l = layers["render"]
        if cfg_l["reached"] != ren_l["reached"]:
            notes.append(
                "aim layers disagree: the rendered pose reaches %d of %d "
                "labelled target(s), the authored sensors pose reaches %d; the "
                "dataset re-aims radar/lidar at the primary target every frame, "
                "so the scene file is not what gets rendered"
                % (ren_l["reached"], ren_l["total"], cfg_l["reached"]))
    grounds = ctx.cfg.get("ground") or {}
    n_ent = 0
    if (ctx.cfg.get("ecm") or {}).get("enabled"):
        try:
            n_ent = len(ctx.ecm_spec()["entities"])
        except Exception:
            n_ent = 0
    summary = dict(
        scene=str(ctx.cfg.get("name") or "scene"),
        mode=ctx.mode(), frames=ctx.n, dt=ctx.dt, frames_per_clip=ctx.fpc,
        clips=ctx.n_clips, sampled_frames=len(ctx.indices),
        objects=[o.get("name") for o in ctx.objects],
        labelled=[o.get("name") for o in ctx.labelled],
        formations={k: [m.get("name") for m in g["members"]]
                    for k, g in ctx.formations().items()},
        ecm_entities=n_ent, camera=ctx.has_camera(),
        aim_layer="render" if render_aim else "config",
        ground_z=ctx._ground_z if grounds.get("enabled", True) else None,
        checks=len(checks), passed=len(checks) - len(failed) - len(skipped),
        skipped=skipped, failed=failed, ok=not failed, notes=notes,
    )
    return dict(ok=not failed, checks=checks, summary=summary)


def format_report(results):
    """Human-readable table + per-scene failure details for the CLI."""
    if isinstance(results, dict):
        results = [results]
    lines = []
    head = "%-32s %-9s %7s  %s  %s" % (
        "scene", "mode", "frames", " ".join("%-2s" % i for i in CHECK_IDS), "fails")
    lines.append(head)
    lines.append("-" * len(head))
    for res in results:
        s = res.get("summary") or {}
        cells = []
        for cid in CHECK_IDS:
            row = next((c for c in res.get("checks", []) if c["id"] == cid), None)
            if row is None:
                cells.append("??")
            elif not row.get("applicable", True):
                cells.append("..")
            elif row["ok"]:
                cells.append("++")
            else:
                cells.append("XX")
        lines.append("%-32s %-9s %7s  %s  %s" % (
            os.path.basename(str(s.get("scene", "?")))[:32],
            str(s.get("mode", "?"))[:9],
            "%s/%s" % (s.get("frames", "?"), _fmt(s.get("dt"), 3)),
            " ".join("%-2s" % c for c in cells),
            ",".join(s.get("failed") or []) or "-"))
    lines.append("")
    for res in results:
        s = res.get("summary") or {}
        bad = [c for c in res.get("checks", []) if not c["ok"]]
        notes = s.get("notes") or []
        # A note is not always a failure: the aim layers disagree on a scene
        # that renders correctly, and that is exactly the kind of "the file says
        # one thing, the renderer does another" a reader has to see even when
        # the row is green.
        if not bad and not notes:
            continue
        lines.append(os.path.basename(str(s.get("scene", "?"))))
        for c in bad:
            lines.append("  %-3s FAIL  %s" % (c["id"], c["detail"]))
        for note in notes:
            lines.append("  note  %s" % note)
        lines.append("")
    total = sum(len(r.get("checks", [])) for r in results)
    failed = sum(1 for r in results for c in r.get("checks", []) if not c["ok"])
    skipped = sum(1 for r in results for c in r.get("checks", [])
                  if not c.get("applicable", True))
    scenes_bad = sum(1 for r in results if not r.get("ok"))
    lines.append("%d scene(s), %d check(s): %d pass, %d fail, %d not applicable; "
                 "%d scene(s) with failures"
                 % (len(results), total, total - failed - skipped, failed,
                    skipped, scenes_bad))
    return "\n".join(lines)
