"""Continuous target trajectories, sensor rigs, and flight envelopes.

The first version of this generator sampled the labelled target's position
independently on every frame.  That produces a thousand unrelated photographs:
nothing in frame k predicts frame k+1, so a sequence carries no more training
signal than a single still and any temporal consumer (tracking, video detection,
3-D consistency) learns nothing.

Everything here is a pure function of clip time instead.  Frame k and frame k+1
are one integration step apart, a run split into independent chunks reproduces
exactly the same path because only the global frame index enters, and velocity
is the analytic derivative of the same expression that yields position -- so
Doppler, MTI gating and the label yaw cannot disagree with where the target is.

Derivative contract
-------------------
Every path publishes three functions of clip time which are exact derivatives
of one another:

    pos(t)      world position, metres
    vel(t)      d pos / d t, m/s
    accel(t)    d vel / d t, m/s^2

No published quantity is sampled from neighbouring render frames.  A central
difference taken across a clamped or piecewise parameter disagrees with the
analytic velocity by a factor of two at the first frame, and that half-velocity
step is exactly what a real run showed once: 76.9 g of apparent acceleration on
frame 0 of a seeker scene, decaying to 4.5 m/s^2 by frame 3.  Path.sample()
still exists as an independent yardstick for the tests; it is never the
published value.

Flight envelope
---------------
A trajectory that is differentiable is not automatically flyable: a circular
arc at 200 m/s and 200 m radius is perfectly smooth and pulls 20 g.  So this
module also owns the envelope a class of airframe can hold (Envelope) and the
checker that walks a sampled track and names the frame, the rule and the number
that broke it (check_track).  The coordinated-turn attitude solve lives here
too, so "bank angle" has exactly one definition:

    tan(bank) = a_lat / g

which is the same statement as bank = atan(v * omega / g): the vertical lift
component holds the weight, its horizontal component supplies the centripetal
force.  A scene that asks for 8 m/s^2 therefore gets 39.2 deg of bank, not a
number that was tuned to look right.

Single state source
-------------------
A Track (docs/多目标运动仿真调研.md 5.1) is t -> (pos, vel, accel, frame): the
one object every consumer reads for one body's state, so a sensor, a label and
a countermeasure cannot hold three opinions about the same aircraft.  Path
stays the authored trajectory; Track.from_path wraps one, and FormationTrack /
AnchorTrack derive a second body's absolute state from a parent Track's.
"""
import numpy as np

from .mesh import rot_x, rot_y, rot_z

# One gravity constant for every coordinated-turn and load-factor expression
# below, so "1 g of turn" means the same number in the attitude solve, in the
# envelope, and in the report.  It is 9.81 because that is the value the rest of
# the simulator already uses (plan.py's bank solve) and the value
# tests/test_motion.py pins: a "better" standard gravity here would silently
# re-scale every published bank angle by 0.03%.
G = 9.81

_EPS = 1e-12


def _unit(v, fallback=(1.0, 0.0, 0.0)):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > _EPS else np.asarray(fallback, float)


def _yaw_of(v):
    """Heading in degrees, 0 = +x, counter-clockwise. Used for the label yaw."""
    return float(np.degrees(np.arctan2(float(v[1]), float(v[0]))))


def _smoothstep(u):
    """0 -> 1 with zero slope at both ends (C1)."""
    u = float(np.clip(u, 0.0, 1.0))
    return u * u * (3.0 - 2.0 * u)


def _smoothstep_d(u):
    u = float(np.clip(u, 0.0, 1.0))
    return 6.0 * u * (1.0 - u)


def _wrap_deg(a):
    return (float(a) + 180.0) % 360.0 - 180.0


# ---------------------------------------------------------------------------
# Attitude from kinematics
# ---------------------------------------------------------------------------
# One definition of pitch/roll/load factor, shared by the paths, the planner and
# the report.  They are computed from (vel, accel), never from a rotation that
# was drawn to look plausible: if the trajectory asks for 0.8 g the bank that
# comes out is 39.2 deg, and a scene that asks for 8 g gets the 83 deg a real
# airframe would need -- which the envelope check then rejects instead of
# rendering a fighter flying sideways.

def turn_rate(v, a):
    """Signed horizontal turn rate, rad/s (positive = turning left / CCW).

    This is exactly d(yaw)/dt: the horizontal velocity is rotated by the
    horizontal lateral acceleration, so omega = (v x a)_z / |v_h|^2.
    """
    v = np.asarray(v, float)
    a = np.asarray(a, float)
    h2 = float(v[0] * v[0] + v[1] * v[1])
    if h2 < 1e-12:
        return 0.0
    return float((v[0] * a[1] - v[1] * a[0]) / h2)


def coordinated_bank_deg(v, a):
    """Bank angle in the simulator's roll convention, degrees.

    Lift holds the weight and its horizontal component supplies the centripetal
    force, so tan(bank) = a_lat / g, equivalently bank = atan(v * omega / g).

    Sign: positive roll rotates the right wing down in scene.py's
    Rz(yaw) Ry(-pitch) Rx(roll) chain, so a left turn is a negative roll.
    """
    v = np.asarray(v, float)
    a = np.asarray(a, float)
    h = float(np.hypot(v[0], v[1]))
    if h < 1e-9:
        return 0.0
    a_lat = (v[0] * a[1] - v[1] * a[0]) / h
    return float(-np.degrees(np.arctan2(a_lat, G)))


def climb_angle_deg(v):
    """Pitch that follows the velocity vector, degrees (nose-up positive)."""
    v = np.asarray(v, float)
    h = float(np.hypot(v[0], v[1]))
    return float(np.degrees(np.arctan2(v[2], h)))


def load_factor(v, a):
    """Specific force divided by g: what the airframe structure carries.

    For a level turn this reduces to 1/cos(bank); for a wings-level climb it is
    1.0, which is why climbing does not consume structural margin.
    """
    a = np.asarray(a, float)
    f = np.array([a[0], a[1], a[2] + G], float)
    return float(np.linalg.norm(f) / G)


def attitude(v, a):
    """(pitch_deg, roll_deg, load_factor) for a coordinated flight state."""
    return climb_angle_deg(v), coordinated_bank_deg(v, a), load_factor(v, a)


# ---------------------------------------------------------------------------
# Flight envelopes
# ---------------------------------------------------------------------------
# Order-of-magnitude limits per class of airframe, from open performance data
# and the certification limits that bracket them.  They are deliberately wide:
# the point is to reject a track no airframe of that class could fly (a
# multirotor at 63 m/s, a fighter in a 40 g turn), not to model one type.
#
#   fighter   : 9 g structural, ~400 m/s at altitude, 70 deg of usable bank
#   airliner  : 2.0 g manoeuvre limit (2.5 g ultimate), 30 deg of bank
#   uav_fixed : MALE fixed wing, 3 g, 20 m/s stall to 200 m/s dash
#   uav_rotor : multirotor, 25 m/s, 3 g peak, 8 m/s climb
#
# Vertical motion is bounded by the *flight-path angle*, not by a vertical speed:
# 35 m/s of descent is a gentle 13 deg glide at 150 m/s and an absurd 80 deg
# plunge at 25 m/s.  max_climb_mps is kept (validation.py measures it) but sized
# for the steepest descent the class can reach at its own cruise speed, so it
# never fires on a legal dive; max_gamma_deg is the rule that actually means
# something and is checked in check_track.
ENVELOPES = {
    "fighter": dict(max_speed_mps=400.0, min_speed_mps=90.0, max_accel_mps2=90.0,
                    max_g=9.0, max_bank_deg=70.0, max_climb_mps=250.0,
                    max_gamma_deg=60.0,
                    min_altitude_m=100.0, max_yaw_rate_dps=45.0, max_jerk_mps3=400.0),
    "airliner": dict(max_speed_mps=290.0, min_speed_mps=110.0, max_accel_mps2=10.0,
                     max_g=2.0, max_bank_deg=30.0, max_climb_mps=60.0,
                     max_gamma_deg=30.0,
                     min_altitude_m=300.0, max_yaw_rate_dps=6.0, max_jerk_mps3=40.0),
    "uav_fixed": dict(max_speed_mps=200.0, min_speed_mps=20.0, max_accel_mps2=12.0,
                      max_g=3.0, max_bank_deg=45.0, max_climb_mps=60.0,
                      max_gamma_deg=45.0,
                      min_altitude_m=60.0, max_yaw_rate_dps=12.0, max_jerk_mps3=40.0),
    "uav_rotor": dict(max_speed_mps=25.0, min_speed_mps=0.0, max_accel_mps2=12.0,
                      max_g=3.0, max_bank_deg=45.0, max_climb_mps=8.0,
                      max_gamma_deg=45.0,
                      min_altitude_m=3.0, max_yaw_rate_dps=120.0, max_jerk_mps3=45.0),
    # An object holding station at altitude -- the target of a platform-mode clip
    # (plan.py's own words: "the target sits; the platform is the mover"), a
    # balloon, an aerostat, a hovering sensor.  It is a real category and it is
    # NOT a fixed-wing airframe: judging a 7 m/s drift at 8000 m against a
    # 20 m/s stall speed reports a stall that does not exist, while the class
    # still catches the failure that does matter -- a "stationary" target that
    # is actually doing 60 m/s.
    "loiter": dict(max_speed_mps=15.0, min_speed_mps=0.0, max_accel_mps2=5.0,
                   max_g=1.5, max_bank_deg=25.0, max_climb_mps=5.0,
                   max_gamma_deg=45.0,
                   min_altitude_m=0.0, max_yaw_rate_dps=60.0, max_jerk_mps3=15.0),
    "ground": dict(max_speed_mps=60.0, min_speed_mps=0.0, max_accel_mps2=6.0,
                   max_g=1.5, max_bank_deg=15.0, max_climb_mps=0.5,
                   max_gamma_deg=10.0,
                   min_altitude_m=0.0, max_yaw_rate_dps=90.0, max_jerk_mps3=20.0),
    "ship": dict(max_speed_mps=25.0, min_speed_mps=0.0, max_accel_mps2=2.0,
                 max_g=1.2, max_bank_deg=20.0, max_climb_mps=0.2,
                 max_gamma_deg=5.0,
                 min_altitude_m=0.0, max_yaw_rate_dps=10.0, max_jerk_mps3=5.0),
    # Some objects declare a velocity and nothing else: no trajectory, no
    # kinematics block, no clip that would move them.  Applying a stall floor to
    # one of those asserts an aerodynamic claim the scene never made -- a demo
    # that drifts a model across the frame is not claiming a Global Hawk cruises
    # at 8.5 m/s.  The bounds here are sanity bounds, wide enough that any real
    # airframe clears them and any broken number does not: they catch a teleport
    # or a velocity entered in km/h, and they deliberately do not judge the class.
    "explicit": dict(max_speed_mps=400.0, min_speed_mps=0.0, max_accel_mps2=200.0,
                     max_g=20.0, max_bank_deg=180.0, max_climb_mps=400.0,
                     max_gamma_deg=90.0,
                     min_altitude_m=-1e6, max_yaw_rate_dps=1e6, max_jerk_mps3=1e4),
    # A sensor mount is not an airframe and it must not be judged as one.  An
    # eye height of 1.7 m is correct, a gimbal slews far faster than any wing
    # turns, and a deliberately vibrating mount has a velocity that does not
    # match the sampled positions.  Judging any of those against a flight
    # envelope produces violations that say nothing about the flight.  What is
    # still worth checking is the thing that would actually break a sequence:
    # the mount teleporting between frames, which the speed limit catches.
    "camera": dict(max_speed_mps=800.0, min_speed_mps=0.0, max_accel_mps2=800.0,
                   max_g=80.0, max_bank_deg=180.0, max_climb_mps=800.0,
                   max_gamma_deg=90.0,
                   min_altitude_m=-1e6, max_yaw_rate_dps=1e6, max_jerk_mps3=1e6),
}

# Classes where the wing, not a rotor, produces the turn.  For these the rate of
# turn is not a free parameter: it is exactly what the declared bank limit implies
# at the speed being flown.  Carrying a separate constant instead let the two
# rules contradict each other -- a 4 m/s aircraft at its 45 deg bank limit turns
# at 28 deg/s, so a 12 deg/s cap called a legal turn illegal while a 200 m/s one
# was never checked at all.
_AERODYNAMIC = ("fighter", "airliner", "uav_fixed")

# What a scene's "cls" usually means.  A bare "uav" maps to the fixed-wing
# entry on purpose: the scenes in this repository that say only "uav" describe a
# MALE drone, and the rotary class has to be asked for by name or by its model
# directory (see _MODEL_CLASS_HINTS).
ENVELOPE_BY_CLASS = {
    "fighter": "fighter", "interceptor": "fighter", "bomber": "fighter",
    "attack": "fighter", "jet": "fighter",
    "airliner": "airliner", "aircraft": "airliner", "transport": "airliner",
    "cargo": "airliner", "civil": "airliner",
    "uav": "uav_fixed", "drone": "uav_fixed", "uas": "uav_fixed",
    "quad": "uav_rotor", "quadrotor": "uav_rotor", "rotor": "uav_rotor",
    "multirotor": "uav_rotor", "suas": "uav_rotor",
    "ground": "ground", "vehicle": "ground", "car": "ground", "truck": "ground",
    "tank": "ground", "rover": "ground", "person": "ground", "pole": "ground",
    "ship": "ship", "boat": "ship", "vessel": "ship", "sea": "ship",
}

# Model directories that override a vague "cls": a scene that points at the
# multirotor mesh is a multirotor whatever its cls string says, and a 63 m/s
# trajectory only looks wrong once the class is right.
_MODEL_CLASS_HINTS = (
    ("旋翼无人机", "uav_rotor"), ("四旋翼", "uav_rotor"), ("旋翼", "uav_rotor"),
    ("multirotor", "uav_rotor"), ("quad", "uav_rotor"),
    ("军用飞机", "fighter"), ("固定翼无人机", "uav_fixed"),
    ("民航飞机", "airliner"), ("直升机", "uav_rotor"),
    ("军舰", "ship"), ("民用船舶", "ship"),
    ("地面车辆", "ground"), ("军用地面车辆", "ground"),
)


class Envelope:
    """The speed/acceleration/height box one class of airframe can fly in.

    A plain dict would do, but an object can answer "which rule did this track
    break" and knows how to layer a scene's own kinematics block on top of its
    class defaults, which is the difference between a number and a verdict.

    Limits use the same key names as objects[].kinematics (max_speed_mps,
    max_accel_mps2, ...) on purpose: a scene that already declares bounds feeds
    the checker without translation, and validation.py keeps working untouched.
    """

    KEYS = ("max_speed_mps", "min_speed_mps", "max_accel_mps2", "max_g",
            "max_bank_deg", "max_climb_mps", "max_gamma_deg", "min_altitude_m",
            "max_yaw_rate_dps", "max_jerk_mps3")

    # Kinds that describe how an object moves, not what airframe it is.  Their
    # limits are sanity bounds, so the report must not print them as "class" and
    # a reader must not read a pass here as an aerodynamic endorsement.
    NON_AERO = ("explicit", "camera")

    def __init__(self, kind="uav_fixed", **limits):
        self.kind = str(kind)
        self.enforce = self.kind not in self.NON_AERO
        base = dict(ENVELOPES.get(self.kind) or ENVELOPES["uav_fixed"])
        for k, v in limits.items():
            if k in self.KEYS and v is not None:
                base[k] = float(v)
        for k in self.KEYS:
            setattr(self, k, float(base.get(k, np.inf)))

    @classmethod
    def preset(cls, kind):
        return cls(kind)

    @classmethod
    def classify(cls, spec):
        """Class of an object spec: explicit hint, then model path, then cls."""
        spec = spec or {}
        explicit = spec.get("flight_class") or spec.get("envelope_class")
        if explicit and str(explicit) in ENVELOPES:
            return str(explicit)
        # A taxiing aircraft is a ground vehicle: validation_fighter_sar_ground
        # drives a fighter_01 along the apron at 3 m/s, and checking that against
        # a 90 m/s flight minimum reports a stall on an aircraft that is parked.
        if str(spec.get("motion_type", "")).lower() == "ground":
            return "ground"
        cls = str(spec.get("cls", "")).strip()
        if cls in ENVELOPES:
            return cls
        text = " ".join(str(spec.get(k, "")) for k in ("model", "asset", "name"))
        for needle, kind in _MODEL_CLASS_HINTS:
            if needle.lower() in text.lower():
                return kind
        hit = ENVELOPE_BY_CLASS.get(cls, ENVELOPE_BY_CLASS.get(cls.lower()))
        if hit:
            return hit
        # A block with declared flight limits is an airframe the scene is making a
        # claim about; anything else (a barrier, a shrub, an occluder) is scenery.
        # Defaulting scenery to the fixed-wing envelope flagged every static prop
        # in the repository for "speed below the stall speed", which is 4000
        # meaningless violations hiding the dozen real ones.
        return "uav_fixed" if (spec.get("kinematics") or spec.get("flight_class")) else "ground"

    @classmethod
    def from_spec(cls, spec=None):
        """Class defaults with the spec's own kinematics block layered on top."""
        spec = spec or {}
        env = cls(cls.classify(spec))
        kin = spec.get("kinematics") or {}
        for k in cls.KEYS:
            if k in kin:
                setattr(env, k, float(kin[k]))
        return env

    def to_dict(self):
        # Non-finite limits are written as None: an envelope that does not bound
        # something should read as "not bounded", and Infinity is not JSON.
        out = {"kind": self.kind, "enforce": bool(self.enforce)}
        for k in self.KEYS:
            v = float(getattr(self, k))
            out[k] = v if np.isfinite(v) else None
        return out

    def lateral_accel_cap(self):
        """The hardest horizontal acceleration a turn can hold, m/s^2.

        Two limits meet here and the smaller one binds.  The structure allows
        a_lat = g*sqrt(n_max^2 - 1) in a level turn (n = sqrt(1 + (a_lat/g)^2)),
        and the usable bank allows a_lat = g*tan(bank_max).  Only the first is
        an airframe number; the second is the wing's own limit, and on every
        aerodynamic class here it is the smaller one -- 65 deg of bank is 2.14 g
        of lateral acceleration, not the 8.9 g that max_g = 9 would permit.
        Reading max_g alone is how a 21 g corner was once called legal on an
        airframe that cannot exceed 65 deg of bank.

        A car or a ship does not turn on bank at all: max_bank_deg there
        describes body roll, so the along-track acceleration limit is the honest
        cap on how hard it can corner.
        """
        n = max(float(self.max_g), 1.0)
        structural = G * float(np.sqrt(max(n * n - 1.0, 0.0)))
        if self.kind in _AERODYNAMIC:
            phi = min(max(float(self.max_bank_deg), 1.0), 89.0)
            banked = G * float(np.tan(np.radians(phi)))
            return float(min(structural, banked))
        return float(min(structural, max(float(self.max_accel_mps2), 0.0)))

    def min_turn_radius_m(self, speed_mps):
        """v^2 / a_lat_max: the tightest arc this class can hold at a speed."""
        a = self.lateral_accel_cap()
        v = abs(float(speed_mps))
        if a <= 0.0 or v <= 0.0:
            return 0.0
        return float(v * v / a)

    def max_speed_for_radius(self, radius_m):
        """The fastest this class can fly a radius: sqrt(a_lat_max * R)."""
        r = max(float(radius_m), 0.0)
        return float(np.sqrt(max(self.lateral_accel_cap() * r, 0.0)))


def envelope_for(spec):
    """Shorthand for tools and tests: the envelope a config object flies in."""
    return Envelope.from_spec(spec)


# ---------------------------------------------------------------------------
# Track checker
# ---------------------------------------------------------------------------
_RULE_TEXT = {
    "speed_max": "speed above the class envelope",
    "speed_min": "speed below the class envelope (stall / not yet airborne)",
    "accel_max": "acceleration above the airframe limit",
    "g_max": "load factor above the structural limit",
    "bank_max": "bank angle above the usable bank limit",
    "climb_max": "climb or descent rate above the class limit",
    "gamma_max": "flight-path angle steeper than the airframe can hold",
    "altitude_min": "below the minimum altitude above ground",
    "yaw_rate_max": "turn rate above the airframe limit",
    "jerk_max": "jerk spike (path derivative discontinuous)",
    "derivative": "velocity disagrees with the position derivative",
}


def envelope_violations(rows, envelope, tolerance_mps=0.05, check_derivative=True):
    """Every frame that breaks its envelope, as (frame, rule, value, limit) dicts.

    This is the single implementation of "is this frame legal".  check_track and
    the scene diagnostic both call it, because two copies of the rule list drift
    apart the moment one of them learns something -- and then the tool that is
    supposed to confirm the fix reports a different answer than the test does.
    """
    if envelope is None:
        envelope = Envelope.preset("uav_fixed")

    def yaw_limit(r):
        if envelope.kind not in _AERODYNAMIC:
            return envelope.max_yaw_rate_dps
        # omega = g tan(phi) / v with phi the declared bank limit: the fastest
        # rate of turn this airframe can hold at the speed it is at right now.
        # A constant cap instead let a 4 m/s aircraft at its 45 deg bank limit
        # (28 deg/s) be called illegal while never checking a 200 m/s one.
        return float(np.degrees(G * np.tan(np.radians(envelope.max_bank_deg))
                                / max(r["speed_mps"], 1.0)))

    out = []
    for r in rows:
        if not envelope.enforce:
            # Only the published-derivative contract is checked here; there is no
            # airframe claim to hold the track against.
            if check_derivative and r["derivative_error_mps"] > tolerance_mps:
                out.append(dict(
                    frame=r["frame"], t=float(r["t"]), rule="derivative",
                    metric="derivative", value=float(r["derivative_error_mps"]),
                    limit=float(tolerance_mps),
                    message="%s: %.4f m/s at frame %d" % (
                        _RULE_TEXT["derivative"], r["derivative_error_mps"], r["frame"])))
            continue
        checks = (
            ("speed_max", r["speed_mps"], envelope.max_speed_mps, ">"),
            # A parked object is not below stall speed, it is parked, so the
            # floor only applies once the thing is actually moving.
            ("speed_min", r["speed_mps"], envelope.min_speed_mps,
             "~" if r["speed_mps"] < 1e-6 else "<"),
            ("accel_max", r["accel_mps2"], envelope.max_accel_mps2, ">"),
            ("g_max", r["g"], envelope.max_g, ">"),
            ("bank_max", abs(r["bank_deg"]), envelope.max_bank_deg, ">"),
            ("climb_max", abs(r["climb_mps"]), envelope.max_climb_mps, ">"),
            ("gamma_max", abs(r["pitch_deg"]), envelope.max_gamma_deg, ">"),
            ("altitude_min", r["altitude_m"], envelope.min_altitude_m, "<"),
            ("yaw_rate_max", abs(r.get("turn_rate_dps", r.get("yaw_rate_dps", 0.0))),
             yaw_limit(r), ">"),
            ("jerk_max", r["jerk_mps3"], envelope.max_jerk_mps3, ">"),
        )
        for rule, value, limit, cmp in checks:
            if cmp == "~":
                continue
            bad = value > limit + 1e-9 if cmp == ">" else value < limit - 1e-9
            if bad:
                out.append(dict(
                    frame=r["frame"], t=float(r["t"]), rule=rule, metric=rule,
                    value=float(value), limit=float(limit),
                    message="%s: %.3f vs limit %.3f (%s)" % (
                        _RULE_TEXT.get(rule, rule), value, limit, envelope.kind)))
        if check_derivative and r["derivative_error_mps"] > tolerance_mps:
            out.append(dict(
                frame=r["frame"], t=float(r["t"]), rule="derivative", metric="derivative",
                value=float(r["derivative_error_mps"]), limit=float(tolerance_mps),
                message="%s: %.4f m/s at frame %d" % (
                    _RULE_TEXT["derivative"], r["derivative_error_mps"], r["frame"])))
    out.sort(key=lambda d: (d["frame"], d["rule"]))
    return out


def check_track(path, envelope=None, ts=None, dt=0.05, n_frames=None, name="track",
                tolerance_mps=0.05, ref_z=0.0):
    """Walk a path and report every frame that breaks its envelope.

    The checker reads the published (pos, vel, accel) -- never the rendered
    frames -- so a run split into chunks gets the same verdict, and it reports
    the frame, the rule, the value and the limit for every violation instead of
    a single "trajectory is wrong".  That is what makes a motion bug fixable:
    "frame 0: derivative 75.3 m/s" is a clamped parameter, not a mystery.

    Returns passed, violations, summary (worst value per metric) and samples.
    """
    if envelope is None:
        envelope = Envelope.preset("uav_fixed")
    if ts is None:
        ts = np.arange(int(n_frames or 64), dtype=float) * float(dt)
    ts = [float(t) for t in ts]
    rows = []
    h = 1e-4
    for t in ts:
        p = np.asarray(path.pos(t), float)
        v = np.asarray(path.vel(t), float)
        a = np.asarray(path.accel(t), float)
        j = np.asarray(path.jerk(t), float)
        speed = float(np.linalg.norm(v))
        dpos = (np.asarray(path.pos(t + h), float) - np.asarray(path.pos(t - h), float)) / (2.0 * h)
        rows.append(dict(
            frame=int(round(t / dt)) if dt else 0, t=t, pos=p, vel=v, acc=a,
            speed_mps=speed, accel_mps2=float(np.linalg.norm(a)),
            g=load_factor(v, a), bank_deg=coordinated_bank_deg(v, a),
            pitch_deg=climb_angle_deg(v), yaw_deg=_yaw_of(v) if speed > 1e-6 else None,
            turn_rate_dps=float(np.degrees(turn_rate(v, a))),
            # Height is above the local surface, not above the sea: a scene whose
            # ground sits at 800 m must not report a 900 m drone as 900 m AGL.
            climb_mps=float(v[2]), altitude_m=float(p[2]) - float(ref_z),
            jerk_mps3=float(np.linalg.norm(j)),
            derivative_error_mps=float(np.linalg.norm(v - dpos)),
        ))
    if not rows:
        return dict(name=name, passed=True, envelope=envelope.to_dict(), summary={},
                    violations=[], samples=[])

    def worst(key):
        return float(max(abs(r[key]) for r in rows))

    summary = dict(
        frames=len(rows),
        min_speed_mps=float(min(r["speed_mps"] for r in rows)),
        max_speed_mps=float(max(r["speed_mps"] for r in rows)),
        max_accel_mps2=worst("accel_mps2"),
        max_g=worst("g"),
        max_bank_deg=worst("bank_deg"),
        max_climb_mps=worst("climb_mps"),
        max_gamma_deg=worst("pitch_deg"),
        min_altitude_m=float(min(r["altitude_m"] for r in rows)),
        max_yaw_rate_dps=worst("turn_rate_dps"),
        max_jerk_mps3=worst("jerk_mps3"),
        max_derivative_error_mps=worst("derivative_error_mps"),
    )

    violations = envelope_violations(rows, envelope, tolerance_mps=tolerance_mps)
    return dict(name=name, passed=not violations, envelope=envelope.to_dict(),
                summary=summary, violations=violations, samples=rows)


# ---------------------------------------------------------------------------
# Quintic Hermite basis: C2 interpolation of (p, v, a)
# ---------------------------------------------------------------------------
# A cubic can match position and velocity at both ends of a segment but its
# acceleration jumps at every knot, and an acceleration jump at 10 Hz is exactly
# the jitter a viewer reads as "the track is not smooth".  The quintic matches
# position, velocity *and* acceleration, so accel is continuous across knots and
# the only remaining discontinuity is in jerk, where real flight control puts it.

def _q00(s): return 1.0 - 10.0*s**3 + 15.0*s**4 - 6.0*s**5
def _q10(s): return s - 6.0*s**3 + 8.0*s**4 - 3.0*s**5
def _q20(s): return 0.5*s**2 - 1.5*s**3 + 1.5*s**4 - 0.5*s**5
def _q01(s): return 10.0*s**3 - 15.0*s**4 + 6.0*s**5
def _q11(s): return -4.0*s**3 + 7.0*s**4 - 3.0*s**5
def _q21(s): return 0.5*s**3 - s**4 + 0.5*s**5

def _q00d(s): return -30.0*s**2 + 60.0*s**3 - 30.0*s**4
def _q10d(s): return 1.0 - 18.0*s**2 + 32.0*s**3 - 15.0*s**4
def _q20d(s): return s - 4.5*s**2 + 6.0*s**3 - 2.5*s**4
def _q01d(s): return 30.0*s**2 - 60.0*s**3 + 30.0*s**4
def _q11d(s): return -12.0*s**2 + 28.0*s**3 - 15.0*s**4
def _q21d(s): return 1.5*s**2 - 4.0*s**3 + 2.5*s**4

def _q00dd(s): return -60.0*s + 180.0*s**2 - 120.0*s**3
def _q10dd(s): return -36.0*s + 96.0*s**2 - 60.0*s**3
def _q20dd(s): return 1.0 - 9.0*s + 18.0*s**2 - 10.0*s**3
def _q01dd(s): return 60.0*s - 180.0*s**2 + 120.0*s**3
def _q11dd(s): return -24.0*s + 84.0*s**2 - 60.0*s**3
def _q21dd(s): return 3.0*s - 12.0*s**2 + 10.0*s**3

def _q00ddd(s): return -60.0 + 360.0*s - 360.0*s**2
def _q10ddd(s): return -36.0 + 192.0*s - 180.0*s**2
def _q20ddd(s): return -9.0 + 36.0*s - 30.0*s**2
def _q01ddd(s): return 60.0 - 360.0*s + 360.0*s**2
def _q11ddd(s): return -24.0 + 168.0*s - 180.0*s**2
def _q21ddd(s): return 3.0 - 24.0*s + 30.0*s**2


class Path:
    """Base class: pos(t), vel(t) and accel(t) in world metres/seconds."""

    mode = "path"

    def pos(self, t):
        raise NotImplementedError

    def vel(self, t):
        raise NotImplementedError

    def accel(self, t):
        """d/dt vel.  Analytic in every built-in class; the fallback here is a
        central difference for third-party paths, and it is labelled as such
        because a difference of a difference is what produced the 76.9 g spike
        that motivated the derivative contract at the top of this file."""
        h = 1e-4
        return (np.asarray(self.vel(t + h), float) - np.asarray(self.vel(t - h), float)) / (2.0 * h)

    def jerk(self, t):
        """d/dt accel, m/s^3.  Diagnostic only: catches sawtooth tracks."""
        h = 1e-3
        return (np.asarray(self.accel(t + h), float) - np.asarray(self.accel(t - h), float)) / (2.0 * h)

    def sample(self, t, dt=1e-3):
        """Finite-difference velocity, for paths whose derivative is fiddly."""
        h = max(float(dt), 1e-4)
        return (self.pos(t + h) - self.pos(t - h)) / (2.0 * h)

    def speed(self, t):
        return float(np.linalg.norm(self.vel(t)))

    def yaw(self, t):
        v = self.vel(t)
        if float(np.hypot(v[0], v[1])) < 1e-6:
            return None            # caller keeps the previous heading
        return _yaw_of(v)

    def heading(self, t):
        """Same as yaw(); kept as a named accessor for Waypoints callers."""
        return self.yaw(t)

    def turn_rate_dps(self, t):
        return float(np.degrees(turn_rate(self.vel(t), self.accel(t))))

    def attitude(self, t):
        """(pitch_deg, roll_deg, load_factor) from this path's own analytic
        velocity and acceleration -- the same numbers the envelope check reads."""
        return attitude(self.vel(t), self.accel(t))

    def summary(self, ts, envelope=None):
        return check_track(self, envelope=envelope, ts=ts, name=self.mode)


class Static(Path):
    """Parked target. Small bob keeps it from being pixel-identical forever."""
    mode = "static"

    def __init__(self, p0, bob_m=0.0, bob_hz=0.7, phase=0.0):
        self.p0 = np.asarray(p0, float)
        self.a = float(bob_m)
        self.w = 2.0 * np.pi * float(bob_hz)
        self.ph = float(phase)
        # A bob is a platform notion (a hovering airframe or a floating hull),
        # so its peak acceleration has to stay small: 2 m/s^2 keeps a 2 m bob
        # from becoming a 3.9 g vertical oscillation, which is what a raw
        # amp/Hz pair produced.  The amplitude is reduced, not the motion
        # silently dropped, so the platform still visibly moves.
        peak = self.a * self.w * self.w
        if peak > 2.0 and self.w > 0.0:
            self.a *= 2.0 / peak
        self._dir = np.array([0.4, 0.4, 1.0])

    def _d(self, t):
        return self.a * np.sin(self.w * t + self.ph)

    def pos(self, t):
        if self.a <= 0.0:
            return self.p0.copy()
        return self.p0 + self._dir * self._d(t)

    def vel(self, t):
        if self.a <= 0.0:
            return np.zeros(3)
        return self._dir * (self.a * self.w * np.cos(self.w * t + self.ph))

    def accel(self, t):
        if self.a <= 0.0:
            return np.zeros(3)
        return self._dir * (-self.a * self.w * self.w * np.sin(self.w * t + self.ph))


class Line(Path):
    """Constant-velocity straight run."""
    mode = "line"

    def __init__(self, p0, v):
        self.p0 = np.asarray(p0, float)
        self.v = np.asarray(v, float)

    def pos(self, t):
        return self.p0 + self.v * t

    def vel(self, t):
        return self.v.copy()

    def accel(self, t):
        return np.zeros(3)


class Arc(Path):
    """Coordinated turn: constant speed, constant turn rate, optional climb.

    Speed and turn rate stay constant, so lateral acceleration is v*omega: an
    aircraft at 200 m/s on a 5000 m radius turns at 0.04 rad/s and pulls
    8 m/s^2 = 0.82 g, which is a realistic envelope rather than an arbitrary
    curve.  The constructor takes the piecewise-constant pair (radius, turn
    rate) and derives everything else, so speed, radius and turn rate cannot
    drift apart.

    A note on turning in: there is deliberately no turn-rate ramp.  An earlier
    version eased omega in from zero over entry_ramp_s, which looked reasonable
    and was wrong -- the speed of a circular arc *is* r*omega, so ramping omega
    from zero made the aircraft stand still before t=0 and accelerate out of
    the origin, which is not a flyer entering a turn but a car pulling away.
    A real roll-in on a constant-radius arc is a curvature step (the radius is
    infinite while the wings are level), and the honest fix for that is a
    clothoid, not a ramp bolted onto a circle.  A scene that needs a smooth
    entry should start on a FlightManeuver, which rolls the bank in and out.
    """
    mode = "arc"

    def __init__(self, centre, radius, omega, z0=None, vz=0.0, phase=0.0):
        self.c = np.asarray(centre, float)
        self.r = float(radius)
        self.w = float(omega)
        # z0 defaults to the centre's own height.  It used to default to 0, so
        # Arc([0, 5000, 3000], ...) silently flew at z=0 -- three kilometres below
        # the centre it was handed -- and every caller had to remember to repeat
        # the altitude in a second argument.
        self.z0 = float(self.c[2] if z0 is None else z0)
        self.vz, self.ph = float(vz), float(phase)

    def pos(self, t):
        a = self.w * t + self.ph
        return np.array([self.c[0] + self.r * np.cos(a),
                         self.c[1] + self.r * np.sin(a),
                         self.z0 + self.vz * t])

    def vel(self, t):
        a = self.w * t + self.ph
        return np.array([-self.r * self.w * np.sin(a),
                         self.r * self.w * np.cos(a), self.vz])

    def accel(self, t):
        a = self.w * t + self.ph
        return np.array([-self.r * self.w * self.w * np.cos(a),
                         -self.r * self.w * self.w * np.sin(a), 0.0])

    @property
    def omega_rad_s(self):
        return self.w

    @property
    def speed_mps(self):
        return abs(self.r * self.w)

    @property
    def turn_radius_m(self):
        return abs(self.r)

    def turn_rate_deg_s(self):
        return float(np.degrees(abs(self.w)))


class Figure8(Path):
    """Lissajous '8': smooth, bounded, and non-circular so heading keeps changing."""
    mode = "figure8"

    def __init__(self, centre, ax, ay, omega, z0=None, vz=0.0, phase=0.0):
        self.c = np.asarray(centre, float)
        self.ax, self.ay = float(ax), float(ay)
        self.w = float(omega)
        # same trap as Arc: an omitted z0 used to override the centre's height
        # with zero and floor the whole figure-eight.
        self.z0 = float(self.c[2] if z0 is None else z0)
        self.vz, self.ph = float(vz), float(phase)

    def pos(self, t):
        a = self.w * t + self.ph
        return np.array([self.c[0] + self.ax * np.sin(a),
                         self.c[1] + self.ay * np.sin(2.0 * a),
                         self.z0 + self.vz * t])

    def vel(self, t):
        a = self.w * t + self.ph
        return np.array([self.ax * self.w * np.cos(a),
                         2.0 * self.ay * self.w * np.cos(2.0 * a), self.vz])

    def accel(self, t):
        a = self.w * t + self.ph
        return np.array([-self.ax * self.w ** 2 * np.sin(a),
                         -4.0 * self.ay * self.w ** 2 * np.sin(2.0 * a), 0.0])


class Weave(Path):
    """Straight leg with a sinusoidal lateral weave -- the classic evasive run.

    Analytic and C-infinity, so a tracking model sees a real bounded lateral
    acceleration instead of the teleporting target the old generator produced.
    The lateral sinusoid is perpendicular to the velocity vector (its dot
    product with the heading is exactly zero), so the weave cannot change the
    along-track speed; it does add to the total speed, which is why the checker
    reads |vel| rather than the configured number.
    """
    mode = "weave"

    def __init__(self, p0, v, amp_m, hz=0.25, lateral=None, z_amp=0.0, phase=0.0):
        self.p0 = np.asarray(p0, float)
        self.v = np.asarray(v, float)
        self.a = float(amp_m)
        self.w = 2.0 * np.pi * float(hz)
        self.ph = float(phase)
        h = _unit(self.v)
        if lateral is None:
            lateral = np.array([-h[1], h[0], 0.0])
        self.lat = _unit(lateral, (-h[1], h[0], 0.0))
        self.za = float(z_amp)

    def pos(self, t):
        s = np.sin(self.w * t + self.ph) - np.sin(self.ph)
        return self.p0 + self.v * t + self.lat * (self.a * s) + np.array([0.0, 0.0, self.za * s])

    def vel(self, t):
        c = self.w * np.cos(self.w * t + self.ph)
        return self.v + self.lat * (self.a * c) + np.array([0.0, 0.0, self.za * c])

    def accel(self, t):
        s = self.w * self.w * np.sin(self.w * t + self.ph)
        return -self.lat * (self.a * s) - np.array([0.0, 0.0, self.za * s])

    def lateral_accel(self, t):
        return float(np.dot(self.accel(t), self.lat))

class DiveWeave(Path):
    """Diving, weaving platform track: the camera moves, the target does not.

    The old seeker mode kept the camera parked and dragged the target towards
    it, so the background never moved and every frame showed the same patch of
    ground.  A real imaging platform is the other way round: the airframe flies
    a dive at ~180 m/s while a small lateral weave keeps the line of sight from
    being a straight line.  That ground rush is exactly the cue motion-based
    detectors are trained on, so it has to come out of the geometry and not out
    of noise.

    Geometry
    --------
    Track direction u is fixed by heading and dive angle and is a unit vector,
    so the along-track speed is a true speed.  The weave is horizontal and its
    dot product with u is exactly zero, i.e. the sinusoid is perpendicular to
    the flight path rather than to its horizontal projection:

        pos(t) = p0 + u * (v t) + lat * A (sin(w t + ph) - sin(ph))
        vel(t) =      u *  v     + lat * A w cos(w t + ph)        exact d/dt
        acc(t) =                   - lat * A w^2 sin(w t + ph)

    The sin(ph) offset keeps pos(0) == p0, so a scene's stated origin is where
    the platform actually is on frame 0 instead of one amplitude away from it.

    Standoff solve
    --------------
    p0, the dive and the clip length are given, but the distance at which the
    run must end is also given.  Those over-determine the track, so the
    along-track speed is solved backwards from t_end and standoff_end_m: the
    end point is the intersection of the track line with the standoff sphere, a
    quadratic, so it stays closed form.

    Envelope-aware standoff
    -----------------------
    The solve above can demand a speed the airframe does not have: a 1 s clip
    that must close 65 m needs 65 m/s, and a scene that configured 14 m/s then
    quietly flies 4.5x its own number.  When the caller supplies the platform
    limits (bank_max_deg for the weave, max_speed_mps for the track) the weave
    amplitude and the along-track speed are clipped to what the airframe can
    hold and the shortfall is recorded in notes instead of being hidden.
    Nothing is clipped when no limits are given, so old scenes keep their
    published numbers.
    """

    mode = "dive_weave"

    def __init__(self, origin, target, speed_mps=180.0, heading_deg=90.0,
                 dive_deg=-14.0, weave_amp_m=18.0, weave_hz=0.22,
                 t_end=6.0, standoff_end_m=None, phase=0.0,
                 target_vel=None, solve="standoff", bank_max_deg=None,
                 max_speed_mps=None):
        self.p0 = np.asarray(origin, float)
        self.tgt = np.asarray(target, float)
        self.tvel = np.zeros(3) if target_vel is None else np.asarray(target_vel, float)
        self.dur = max(float(t_end), 1e-6)
        self.heading_deg = float(heading_deg)
        self.dive_deg = float(dive_deg)
        self.notes = []

        hd, dv = np.radians(self.heading_deg), np.radians(self.dive_deg)
        self.u = np.array([np.cos(dv) * np.cos(hd),
                           np.cos(dv) * np.sin(hd),
                           np.sin(dv)])
        self.lat = _unit(np.cross(self.u, np.array([0.0, 0.0, 1.0])), (1.0, 0.0, 0.0))
        self.w = 2.0 * np.pi * float(weave_hz)
        self.ph = float(phase)
        self.speed_cfg = float(speed_mps)
        self.standoff_cfg = None if standoff_end_m is None else float(standoff_end_m)
        self.solve = str(solve).lower()

        # The weave amplitude is a load-factor demand: A w^2 is the lateral
        # acceleration it asks for, and atan(A w^2 / g) is the bank the airframe
        # must hold.  A scene asking 25 m at 0.15 Hz asks for 66 deg, which no
        # MALE platform can use, so the amplitude is clipped to the bank limit
        # instead of being flown as written.
        self.A = float(weave_amp_m)
        if bank_max_deg is not None and self.w > 0.0 and self.A > 0.0:
            a_max = G * np.tan(np.radians(min(abs(float(bank_max_deg)), 80.0)))
            a_have = self.A * self.w * self.w
            if a_have > a_max:
                self.A = a_max / (self.w * self.w)
                self.notes.append(
                    "weave amplitude clipped %.2f -> %.2f m: %.1f m/s^2 would need "
                    "%.1f deg of bank, limit %.0f deg"
                    % (float(weave_amp_m), self.A, a_have,
                       np.degrees(np.arctan2(a_have, G)), float(bank_max_deg)))
        self.speed_max = None if max_speed_mps is None else float(max_speed_mps)

        self.speed_track = self._solve_track_speed()
        self.t_end = self.dur
        self.standoff_actual = float(np.linalg.norm(
            self.pos(self.dur) - self.target_at(self.dur)))

    # ---- geometry -------------------------------------------------------
    def target_at(self, t):
        return self.tgt + self.tvel * float(t)

    def _solve_track_speed(self):
        """Along-track speed: made equal to the configured *total* speed, or
        solved from the standoff sphere when a standoff is given."""
        # |vel|^2 = v^2 + (A w cos)^2, whose mean over a cycle is
        # v^2 + (A w)^2/2.  Taking this square root keeps the configured speed
        # meaning "the speed this platform flies" instead of "the speed along
        # one axis while the weave adds 3 m/s on top".
        keep = np.sqrt(max(self.speed_cfg ** 2 - 0.5 * (self.A * self.w) ** 2, 1e-6))
        speed = float(keep)
        if self.standoff_cfg is not None and self.solve not in ("speed", "keep", "rms"):
            s_end = np.sin(self.w * self.dur + self.ph) - np.sin(self.ph)
            B = (self.p0 - self.target_at(self.dur)) + self.lat * (self.A * s_end)
            along = float(np.dot(self.u, B))
            disc = along * along - float(np.dot(B, B)) + self.standoff_cfg ** 2
            x = None
            if disc >= 0.0:
                x = -along - np.sqrt(disc)
                if x <= 1e-6:
                    # already inside the standoff sphere at t=0: take the far crossing
                    x = -along + np.sqrt(disc)
            if x is not None and x > 1e-6:
                speed = float(x / self.dur)
        if self.speed_max is not None:
            # Clip against the speed the airframe actually reaches, which is the
            # along-track number combined with the weave's own lateral velocity:
            # |v|^2 = s^2 + (A w cos)^2 peaks at s^2 + (A w)^2.  Clipping only s
            # left the flown speed 0.03 m/s over the limit with the weave added.
            peak_lat = abs(self.A * self.w)
            cap = float(np.sqrt(max(self.speed_max ** 2 - peak_lat ** 2, 1e-6)))
            if abs(speed) > cap:
                self.notes.append(
                    "along-track speed clipped %.2f -> %.2f m/s so the flown speed "
                    "stays within %.2f m/s" % (speed, cap, self.speed_max))
                speed = float(cap)
        return speed

    # ---- analytic kinematics -------------------------------------------
    def pos(self, t):
        t = float(t)
        return (self.p0 + self.u * (self.speed_track * t)
                + self.lat * (self.A * (np.sin(self.w * t + self.ph) - np.sin(self.ph))))

    def vel(self, t):
        t = float(t)
        return (self.u * self.speed_track
                + self.lat * (self.A * self.w * np.cos(self.w * t + self.ph)))

    def accel(self, t):
        t = float(t)
        return self.lat * (-self.A * self.w * self.w * np.sin(self.w * t + self.ph))

    def jerk(self, t):
        t = float(t)
        return self.lat * (-self.A * self.w ** 3 * np.cos(self.w * t + self.ph))

    def lateral_accel(self, t):
        """Signed lateral acceleration (m/s^2, positive to the right of track)."""
        return float(np.dot(self.accel(t), self.lat))

    def altitude(self, t):
        return float(self.pos(t)[2])

    def range_at(self, t):
        return float(np.linalg.norm(self.target_at(t) - self.pos(t)))


class Waypoints(Path):
    """C1 cubic path with arc-length sampling and smooth open-path stops.

    Looping is explicit.  An open path brakes at its end and holds the final
    position; its endpoints never wrap to the beginning.  Whether it also
    accelerates from rest at the start is ``ramp_in``: True by default for an
    authored route that begins at a standstill, False for a drawn cruise leg
    that is already at speed on the departure point.

    The spline is C1 in arc length but only C0 in curvature, so a waypoint
    corner is a step in acceleration -- a jerk spike, not a velocity jump.  That
    is the honest description of a piecewise-cubic route, and it is why the
    class publishes accel from the exact chain rule instead of pretending the
    corners are smooth, and why the checker reports the jerk there.
    """
    mode = "waypoints"

    def __init__(self, pts, speed=8.0, samples=128, phase=0.0, loop=True,
                 ramp_in=True):
        P = np.asarray(pts, float)
        if P.ndim != 2 or P.shape[1] != 3 or len(P) < 2 or not np.isfinite(P).all():
            raise ValueError("waypoints needs at least two finite 3D points")
        if loop and np.linalg.norm(P[-1] - P[0]) < 1e-9:
            P = P[:-1]
        if len(P) < 2 or np.any(np.linalg.norm(np.diff(P, axis=0), axis=1) < 1e-6):
            raise ValueError("consecutive waypoints must be distinct")
        self.loop = bool(loop)
        self.v, self.ph = float(speed), float(phase)
        if not np.isfinite([self.v, self.ph]).all() or self.v <= 0:
            raise ValueError("waypoint speed must be finite and positive")
        # Chord-length knots and shared tangents give continuous heading at
        # every waypoint, even when the original waypoint spacing is uneven.
        Q = np.vstack([P, P[0]]) if self.loop else P
        lengths = np.linalg.norm(np.diff(Q, axis=0), axis=1)
        tangent = np.empty_like(P)
        for i in range(len(P)):
            if not self.loop and i == 0:
                tangent[i] = (P[1]-P[0])/lengths[0]
            elif not self.loop and i == len(P)-1:
                tangent[i] = (P[-1]-P[-2])/lengths[-1]
            else:
                left, right = lengths[(i-1) % len(lengths)], lengths[i]
                tangent[i] = ((P[i]-P[(i-1) % len(P)])/left*right +
                              (P[(i+1) % len(P)]-P[i])/right*left)/(left+right)
        # Knot tangents are scaled per *knot*, not per cell.  Scaling them by the
        # cell's own chord length made the two cells that meet at a waypoint
        # disagree about dp/dq by the ratio of their chords, so the closed loop
        # arrived at its first waypoint 20 % faster than it left it and the
        # velocity stepped by 0.035 m/s at the seam -- continuous position and
        # heading, discontinuous speed, which is exactly the kind of "the
        # trajectory is not coherent" a viewer notices on the loop every lap.
        unit = tangent / np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
        scale = np.empty(len(P))
        for i in range(len(P)):
            a, b = lengths[(i-1) % len(lengths)], lengths[i % len(lengths)]
            scale[i] = 0.5 * (a + b) if self.loop else (
                b if i == 0 else (a if i == len(P)-1 else 0.5*(a+b)))
        m = unit * scale[:, None]
        M = np.vstack([m, m[0]]) if self.loop else m
        self.coeff = np.stack([2*Q[:-1]-2*Q[1:]+M[:-1]+M[1:],
                               -3*Q[:-1]+3*Q[1:]-2*M[:-1]-M[1:],
                               M[:-1], Q[:-1]], axis=1)
        self.q = np.linspace(0, len(lengths), len(lengths)*max(int(samples), 32)+1)
        # Vectorised curve evaluation: the arc-length table has to be dense
        # enough that dq/darc is smooth, and a per-point Python loop is why the
        # old default could only afford 128 samples per segment.
        dense = self._curve(self.q)[0]
        # The table is a plain cumulative sum of chords, and that is deliberate:
        # _at() takes dq/darc to be the secant of exactly this table, so the
        # published velocity is the exact derivative of the published position --
        # an approximation of the true arc length, but a self-consistent one.
        # Swapping in a trapezoid rule is *more* accurate about the curve and was
        # measurably worse here: the seam step grew from 6.6e-3 to 2.9e-1 m/s,
        # because the table then no longer matched the inverse that pos() uses.
        # Do not "improve" this without changing _at() alongside it.  The residual
        # seam step falls linearly with the sampling (6.6e-3 m/s at the default,
        # 3.7e-4 at 2048 per segment); it is table resolution, not a spline defect.
        self.s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))]
        self.length = float(self.s[-1])
        # Knot slopes for the C1 inverse: dq/darc = 1/|dC/dq| is exactly the rate
        # at which the parameter advances per metre of curve, so seeding the
        # Hermite with it makes q'(arc) equal the true value at every knot and the
        # interpolation error vanish to higher order in between.
        _p, dqk = self._curve(self.q)
        slope = 1.0 / np.maximum(np.linalg.norm(dqk, axis=1), 1e-12)
        hh = np.diff(self.s)
        hh = np.maximum(hh, 1e-12)
        sec = np.diff(self.q) / hh
        m0, m1 = slope[:-1], slope[1:]
        # Fritsch-Carlson limiting: without it a knot slope far from the cell's
        # mean steepness makes q(arc) overshoot and the position runs backwards
        # along the curve inside that cell, which is a visible hesitation.
        for m, k in ((m0, sec), (m1, sec)):
            with np.errstate(divide="ignore", invalid="ignore"):
                r = np.where(k > 0, m / k, 0.0)
            np.clip(r, 0.0, 3.0, out=r)
            m[...] = r * k
        hsec = hh * sec
        self.hcoef = np.stack([2*self.q[:-1] - 2*self.q[1:] + hh*m0 + hh*m1,
                               -3*self.q[:-1] + 3*self.q[1:] - hh*(2*m0 + m1),
                               hh*m0, self.q[:-1]], axis=1)
        if not self.loop and not 0 <= self.ph <= self.length:
            raise ValueError("open waypoint phase must lie within the path length")
        remaining = self.length if self.loop else self.length-self.ph
        #: An open route has two ends and they are not the same event.  The
        #: operator draws a CRUISE route: the airframe is already doing its
        #: speed at the departure point, and only the arrival is flown down.
        #: Ramping in as well forced 0 -> v in one second, which at 250 m/s is
        #: 25 g of along-track acceleration -- a catapult, not a cruise, and on
        #: a two-second clip the whole clip was that ramp.
        self.ramp_in = bool(ramp_in)
        self.ramp_s_in = min(1.0, remaining/self.v) if self.ramp_in else 0.0
        self.ramp_s_out = min(1.0, remaining/self.v)
        self.ramp_s = self.ramp_s_out
        self.duration = remaining/self.v + self.ramp_s_out

    def _curve(self, q):
        q = np.asarray(q, float)
        i = np.minimum(q.astype(int), len(self.coeff)-1)
        # u is a column and the coefficients are (N, 3), so the Horner product
        # broadcasts to one row per sample instead of an (N, N, 3) matrix.
        u = (q - i)[:, None]
        a, b, c, d = (self.coeff[i][:, k] for k in range(4))
        pos = ((a*u+b)*u+c)*u+d
        d1 = (3*a*u+2*b)*u+c
        return pos, d1

    def _curve_at(self, q):
        """Single-point (pos, d/dq, d2/dq2)."""
        q = float(q)
        i = min(int(q), len(self.coeff)-1)
        u = q - i
        a, b, c, d = self.coeff[i]
        pos = ((a*u+b)*u+c)*u+d
        d1 = (3*a*u+2*b)*u+c
        d2 = 6*a*u+2*b
        return pos, d1, d2

    def _distance(self, t):
        """(arc length, d/dt, d2/dt2) -- all three are exact."""
        if self.loop:
            return self.v*t+self.ph, self.v, 0.0
        T_in, T_out = self.ramp_s_in, self.ramp_s_out
        if t <= 0:
            # With ramp_in the route starts from rest; without it the airframe
            # is already at cruise speed on the first frame, which is what a
            # drawn departure point means.
            return self.ph, (0.0 if T_in > 0.0 else self.v), 0.0
        if t >= self.duration:
            return self.length, 0.0, 0.0
        if T_in > 0.0 and t < T_in:
            u = t/T_in
            return (self.v*T_in*(u**3-0.5*u**4)+self.ph,
                    self.v*(3*u*u-2*u**3),
                    self.v*(6*u-6*u*u)/T_in)
        if t > self.duration-T_out:
            d, sp, ac = self._distance(self.duration-t)
            # Mirror of the run-out: arc = length - d, so its first derivative
            # keeps the sign of sp and its second flips once more.
            return self.length-d+self.ph, sp, -ac
        return self.v*(t-T_in/2)+self.ph, self.v, 0.0

    def _at(self, arc):
        """(pos, dpos/dq, d2pos/dq2, dq/darc) at an arc length.

        The inverse arc-length map q(arc) is a cubic Hermite through the table,
        with the exact derivative 1/|dC/dq| at every knot.  It has to be C1: the
        earlier version used the per-cell secant, which is the exact derivative of
        a piecewise-linear inverse and therefore *steps* at every knot, so the
        published velocity jumped between adjacent table cells -- 3e-3 m/s on an
        9 m/s route, and the loop's seam step was the same artefact seen once per
        lap.  Hermite gives the same exactness inside a cell and removes the step,
        because q'(arc) is continuous at the knots by construction.
        """
        arc = np.mod(arc, self.length) if self.loop else np.clip(arc, 0, self.length)
        i = int(np.searchsorted(self.s, arc, side="right")) - 1
        i = max(0, min(i, len(self.s)-2))
        h = max(float(self.s[i+1]) - float(self.s[i]), 1e-12)
        u = (arc - float(self.s[i])) / h
        a, b, c, d = self.hcoef[i]
        q = ((a*u + b)*u + c)*u + d
        d1 = (3*a*u + 2*b)*u + c          # dq/du
        d2 = 6*a*u + 2*b                  # d2q/du2
        p, dp, d2p = self._curve_at(q)
        return p, dp, d2p, d1 / h, d2 / (h * h)

    def pos(self, t):
        return self._at(self._distance(float(t))[0])[0]

    def vel(self, t):
        arc, speed, _acc = self._distance(float(t))
        # arc == 0 is a dead stop only when the route ramps in from rest; with
        # ramp_in off the first frame is already at cruise speed and must publish
        # the tangent velocity, not zero.
        if not self.loop and (arc >= self.length
                              or (self.ramp_in and arc <= 0)):
            return np.zeros(3)
        _p, dp, _d2p, dq, _d2q = self._at(arc)
        return dp * (dq * speed)

    def accel(self, t):
        arc, speed, sacc = self._distance(float(t))
        if not self.loop and (arc >= self.length
                              or (self.ramp_in and arc <= 0)):
            return np.zeros(3)
        _p, dp, d2p, dq, d2q = self._at(arc)
        # q is cubic in arc now, so the chain rule needs both terms:
        #   dq/dt   = q' s'
        #   d2q/dt2 = q'' s'^2 + q' s''
        # Dropping the q'' term was correct only while the inverse was
        # piecewise-linear, and it is what made accel disagree with d(vel)/dt by
        # up to 0.34 m/s^2 at the frame where the table's curvature is largest.
        qd = dq * speed
        qdd = d2q * speed * speed + dq * sacc
        return d2p * (qd * qd) + dp * qdd

    def heading(self, t):
        arc = self._distance(float(t))[0]
        return _yaw_of(self._at(arc)[1])


# ---------------------------------------------------------------------------
# Route pre-shaping: a hand-drawn stroke is a sketch, not a flight path
# ---------------------------------------------------------------------------
# A plate view is 600 m across and an operator clicks four to six points on it,
# which puts 10-30 m between them.  Flown at 80 m/s every vertex costs v^2/R of
# lateral acceleration: a 14 deg corner inside 7 m is a 29 m radius, i.e. 22 g
# and 84 deg of bank, and the measured jerk on the three-aircraft plate demo was
# 32037 m/s^3 against a 400 m/s^3 limit.  The geometry is not wrong -- the
# stroke is a gesture about where the aircraft should go -- but nothing can fly
# it as drawn, so the route is shaped before it is published:
#
#   fillet  each corner becomes a circular arc as wide as the drawing has room
#           for.  The arc is tangent to both legs, so the endpoints and the
#           directions of the stroke are untouched: only the corner is cut.
#   speed   when the room-limited arc is still too tight for the requested
#           speed, the SPEED is what gives: v = sqrt(a_lat_max * R).  That is
#           the trade the studio canvas already warns about, and it is a
#           physical result, never a hard-coded number.
#   relax   a floor stops that trade -- an airframe below its own minimum speed
#           is not flying legally.  When the floor speed still needs a wider arc
#           than the stroke can hold, the shape gives way instead, and how much
#           corner had to be thrown away is reported so the caller can print it.
#           Silently straightening an operator's route is worse than saying so.
#
# Every verdict is measured on the spline that will actually be published
# (Waypoints, differenced the way tools/motion_diagnose.py differences it) and
# not on the polyline: the promise is about the flown jerk, and only the flown
# path can answer for it.

def _route_array(points):
    """Waypoints as an (N, 3) float array, duplicates and collinear knots removed."""
    P = np.asarray(points, float)
    if P.ndim != 2 or P.shape[1] != 3 or len(P) < 2 or not np.isfinite(P).all():
        raise ValueError("route needs at least two finite 3D points")
    keep = [0]
    for i in range(1, len(P)):
        if float(np.linalg.norm(P[i] - P[keep[-1]])) > 1e-6:
            keep.append(i)
    P = P[keep]
    if len(P) < 2:
        raise ValueError("route needs at least two distinct points")
    return P


def _route_turns(P):
    """Turning angle at every vertex, radians (0 = straight on, pi = reversal)."""
    d = np.diff(P, axis=0)
    d = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
    turns = np.zeros(len(P))
    for i in range(1, len(P) - 1):
        turns[i] = float(np.arccos(min(max(float(d[i - 1] @ d[i]), -1.0), 1.0)))
    return turns


def _route_min_radius(P):
    """Smallest circumradius of any consecutive triple, metres (inf if straight).

    This is the discrete curvature of the polyline itself, so it answers "how
    tight is the drawn shape" without fitting anything: the circumradius of a
    triangle is exactly the radius of the circle through its three corners.
    """
    P = np.asarray(P, float)
    best = np.inf
    for i in range(1, len(P) - 1):
        a, b = P[i - 1] - P[i], P[i + 1] - P[i]
        la, lb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        cross = float(np.linalg.norm(np.cross(a, b)))
        if la < 1e-9 or lb < 1e-9 or cross < 1e-12:
            continue
        chord = float(np.linalg.norm(P[i + 1] - P[i - 1]))
        best = min(best, la * lb * chord / (2.0 * cross))
    return float(best)


def _fillet_room(P, turns, share=0.45):
    """Tangent length available at each vertex, metres, without crossing corners.

    A fillet eats t = r*tan(theta/2) of each adjacent leg, and two neighbouring
    fillets share the leg between them, so the room has to be booked for both.
    Three sweeps are enough for the shrink to settle: each sweep only shortens.
    """
    L = np.linalg.norm(np.diff(P, axis=0), axis=1)
    t = np.zeros(len(P))
    for i in range(1, len(P) - 1):
        t[i] = share * float(min(L[i - 1], L[i]))
    for _ in range(3):
        for j in range(len(L)):
            a, b = t[j], t[j + 1]          # the two fillets on this leg
            total = a + b
            if total > share * 2.0 * float(L[j]) and total > 1e-9:
                k = (share * 2.0 * float(L[j])) / total
                t[j], t[j + 1] = a * k, b * k
    return t


def _arc_between(centre, a_vec, b_vec, sweep, samples):
    """Points on the circle (centre, |a_vec|) from a_vec to b_vec, endpoints in."""
    n = max(int(samples), 2)
    r = float(np.linalg.norm(a_vec))
    if r < 1e-9 or sweep < 1e-6:
        return []
    ua, ub = a_vec / r, b_vec / r
    sin_s = float(np.sin(sweep))
    out = []
    for k in range(n + 1):
        u = k / float(n)
        if abs(sin_s) < 1e-9:
            w = ua * (1.0 - u) + ub * u
            w = w / max(float(np.linalg.norm(w)), 1e-12)
        else:
            w = (ua * float(np.sin((1.0 - u) * sweep))
                 + ub * float(np.sin(u * sweep))) / sin_s
        out.append(centre + w * r)
    return out


def _round_corners(P, turns, share=0.45, arc_deg=10.0, max_sweep=150.0):
    """Replace every corner with a tangent circular fillet. (points, radii).

    radii[i] is the fillet radius actually fitted at vertex i (0 where the
    corner was dropped as straight).  The largest radius the drawing allows is
    used, which for a gentle bend is far more than "half the leg": the tangent
    length is r*tan(theta/2), so a 12 deg corner turns 3 m of leg into a 28 m
    radius.  That is why the radius has to be solved from the room instead of
    pinned to a fraction of the segment.
    """
    t = _fillet_room(P, turns, share=share)
    out = [P[0].copy()]
    radii = np.zeros(len(P))
    for i in range(1, len(P) - 1):
        theta = float(turns[i])
        if theta < np.radians(0.5) or t[i] <= 1e-6:
            continue                      # straight: the knot buys nothing
        theta = min(theta, np.radians(max_sweep))
        r = t[i] / float(np.tan(theta * 0.5))
        if not np.isfinite(r) or r < 1e-3:
            out.append(P[i].copy())
            continue
        d1, d2 = P[i] - P[i - 1], P[i + 1] - P[i]
        d1 = d1 / max(float(np.linalg.norm(d1)), 1e-12)
        d2 = d2 / max(float(np.linalg.norm(d2)), 1e-12)
        a = P[i] - d1 * (r * float(np.tan(theta * 0.5)))
        b = P[i] + d2 * (r * float(np.tan(theta * 0.5)))
        bis = d2 - d1
        bis = bis / max(float(np.linalg.norm(bis)), 1e-12)
        # The centre sits r/cos(theta/2) along the internal bisector: the
        # tangent point is t = r*tan(theta/2) back along the leg and the
        # centre is r perpendicular to it, so |P-C|^2 = r^2 + t^2 and the
        # bisector distance is sqrt(r^2+t^2) = r/cos(theta/2).  r/sin(theta/2)
        # is the formula for the *inscribed* circle of the wedge, not for the
        # tangent fillet: at a 1 deg corner it put the centre 43 km away and
        # the arc 2 m off its own chord.
        centre = P[i] + bis * (r / max(float(np.cos(theta * 0.5)), 1e-9))
        arc = _arc_between(centre, a - centre, b - centre, theta,
                           max(2.0, np.degrees(theta) / arc_deg))
        for p in arc:
            if float(np.linalg.norm(p - out[-1])) > 1e-4:
                out.append(np.asarray(p, float))
        radii[i] = r
    if float(np.linalg.norm(P[-1] - out[-1])) > 1e-9:
        out.append(P[-1].copy())
    return np.asarray(out, float), radii


def _relax_route(P, target_r, sweeps=240, lam=0.5):
    """Laplacian smoothing with pinned endpoints until the polyline is gentler.

    This is the shape-sacrificing knob: the endpoints stay, the stroke's bend is
    spread until no vertex is tighter than target_r.  Repeated averaging
    converges to the straight chord, so any finite target is reachable; the
    sweep cap only bounds the work (and leaves the residue to be reported).
    """
    Q = np.array(P, float)
    if len(Q) < 3 or not np.isfinite(target_r):
        return Q
    for _ in range(int(sweeps)):
        if _route_min_radius(Q) >= target_r:
            break
        # Horizontally only: a corner is widened, a climb is not flattened.
        # Averaging z as well turned a 100 m step climb into a 1.5 deg drift,
        # i.e. it silently deleted the altitude profile the operator drew --
        # and the altitude ladder with it.  A climb that is too steep to fly is
        # reported (_see the pitch note in shape_route) instead of being
        # smoothed away.
        Q[1:-1, :2] = (1.0 - lam) * Q[1:-1, :2] + lam * 0.5 * (Q[:-2, :2] + Q[2:, :2])
    return Q


def densify_route(points, cell_m=None, max_points=192):
    """Split long legs so the C1 spline's per-knot tangents stay proportionate.

    A Waypoints knot tangent is half the sum of its two neighbouring chords, so
    a 7 m cell next to a 115 m cruise leg asks for a 61 m tangent inside a 7 m
    cell: the cubic overshoots and the same route that measured 32000 m/s^3
    measured 127858 m/s^3 after the leg was appended.  An operator can never
    draw such a pair on purpose, so the fix belongs in the geometry: every leg
    is cut to roughly the typical cell of the drawing.
    """
    P = np.asarray(points, float)
    if len(P) < 2:
        return [list(map(float, p)) for p in P]
    L = np.linalg.norm(np.diff(P, axis=0), axis=1)
    total = float(np.sum(L))
    if total <= 1e-9:
        return [list(map(float, p)) for p in P]
    # The SHORTEST drawn leg sets the cell, not the median: a 6 m cell beside a
    # 100 m one is exactly the mismatch this function exists to remove, and the
    # median of [6, 100] is 53 m, which leaves it in place.  The point budget
    # (max_points) is the floor, so a route with one 0.5 m cell and a kilometre
    # of legs cannot explode into ten thousand waypoints.
    cell = float(cell_m) if cell_m else max(total / float(max_points), 0.5)
    n = max(2, min(int(np.ceil(total / max(cell, 1e-3))), int(max_points)))
    # Uniform by ARC LENGTH, not per-leg: a cell that matches its neighbours is
    # what the spline's per-knot tangent (half the sum of the two adjacent
    # chords) is designed for.  Cutting each leg into its own number of pieces
    # leaves a 30 m cell beside a 60 m one and the cubic oscillates between
    # them -- measured on a 936 m route: the acceleration alternated 0 / -18 /
    # +36 m/s^2 along a route whose arcs are all 355 m.
    out = [P[0]]
    for k in range(1, n):
        out.append(_along(P, total * k / float(n)))
    out.append(P[-1])
    return [[round(float(c), 4) for c in p] for p in out]


def _along(P, arc):
    """The point at an arc length along a polyline."""
    acc = 0.0
    for i in range(len(P) - 1):
        seg = P[i + 1] - P[i]
        L = float(np.linalg.norm(seg))
        if acc + L >= arc:
            return P[i] + seg * ((arc - acc) / max(L, 1e-12))
        acc += L
    return P[-1]


def route_heading(points):
    """Unit tangent of a route at its first point (its initial flight direction)."""
    P = _route_array(points)
    return _unit(P[1] - P[0])


def _measure_route(path, frames=24, dt=0.1):
    """Worst per-frame accelerations of the published spline, as the checker sees them.

    The differencing scheme is motion_diagnose's on purpose: h = dt/4 for the
    acceleration (what an observer of these frames can measure) and the cruising
    part of the route only, because an open route's run-out is a different event
    from the leg being shaped.
    """
    h = max(float(dt) * 0.25, 1e-4)
    span = float(path.length) / max(float(path.v), 1e-9)
    # Cover the WHOLE route, not just the clip window.  Sampling the first 2 s
    # only left a 2 km route measured for its first 10 %: a 59 deg climb at
    # x = 500 m and any tight corner past the window went unseen, and a route can
    # only be certified if every part of it was looked at.  Inside the clip
    # window the step is the frame interval, so the numbers stay the ones the
    # checker will see.
    # Stay clear of the run-out: at arc == length the path publishes zero
    # velocity (it has arrived), and a difference taken across that clamp reads
    # 24480 m/s^3 on a route whose real worst is 130 -- the same edge artefact as
    # frame 0, at the other end of the clock.
    end = max(span - 4.0 * h, 2.0 * h)
    n = max(int(frames), 1)
    step = float(dt) if end <= n * float(dt) else end / float(n)
    ts = [k * step for k in range(int(end / step) + 1)]
    worst = dict(accel_mps2=0.0, bank_deg=0.0, jerk_mps3=0.0, g=1.0,
                 pitch_deg=0.0, turn_rate_dps=0.0)

    def accel_at(t):
        if t - h < 0.0:
            # The first sample of a clip sits on the edge of the scene clock,
            # where every planner freezes the position for t < 0: a centred
            # step there reads half the real acceleration and reports it as a
            # 200 m/s^2 "spike".  Three-point forward, same O(h^2) order.
            return (-3.0 * np.asarray(path.vel(t), float)
                    + 4.0 * np.asarray(path.vel(t + h), float)
                    - np.asarray(path.vel(t + 2.0 * h), float)) / (2.0 * h)
        return (np.asarray(path.vel(t + h), float)
                - np.asarray(path.vel(t - h), float)) / (2.0 * h)

    for t in ts:
        v = np.asarray(path.vel(t), float)
        a = accel_at(t)
        a2 = accel_at(t + h)
        worst["accel_mps2"] = max(worst["accel_mps2"], float(np.linalg.norm(a)))
        worst["jerk_mps3"] = max(worst["jerk_mps3"],
                                float(np.linalg.norm(a2 - a)) / (2.0 * h))
        worst["g"] = max(worst["g"], load_factor(v, a))
        worst["bank_deg"] = max(worst["bank_deg"], abs(coordinated_bank_deg(v, a)))
        worst["pitch_deg"] = max(worst["pitch_deg"], abs(climb_angle_deg(v)))
        worst["turn_rate_dps"] = max(worst["turn_rate_dps"],
                                     abs(float(np.degrees(turn_rate(v, a)))))
    return worst


def shape_route(points, speed=8.0, envelope=None, safety=0.85, dt=0.1,
                frames=24, max_sweeps=4):
    """Turn a drawn polyline into a route the claimed airframe can fly.

    Returns {points, speed_mps, info}.  info carries the numbers a caller needs
    to explain itself: requested vs used speed, length and radius before and
    after, the measured worst load / bank / jerk of the flown spline, and
    ready-made Chinese notes for cfg["notes"].

    The boundary between "round the corner" and "fly it slower" is the room the
    drawing leaves.  r_room is the largest tangent fillet that fits between its
    two neighbours and v_room = sqrt(a_lat_max * r_room) is the fastest that arc
    can be flown: above it nothing changes, below it the speed gives.  Below the
    airframe's own minimum speed the speed stops being a free variable -- a
    legal-looking track that is stalled is not legal -- and the route shape is
    relaxed instead, with the loss reported rather than hidden.
    """
    P = _route_array(points)
    env = envelope if envelope is not None else Envelope.preset("uav_fixed")
    a_cap = max(float(safety) * env.lateral_accel_cap(), 1e-6)
    v_floor = max(float(env.min_speed_mps), 0.0)
    # Never fly the floor itself: the C1 spline's speed ripples by a fraction of
    # a percent around the value the arc was solved for (measured: 59.9999 m/s on
    # a route commanded at exactly 60), and a 1e-4 m/s dip is a speed_min
    # violation.  2 % is smaller than any operator would notice and larger than
    # the ripple.
    v_min = v_floor * 1.02 if v_floor > 0.0 else 0.0
    v_ceil = float(env.max_speed_mps)
    v_req = float(speed)
    v = min(max(v_req, v_min), v_ceil)
    length_in = float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1)))
    r_drawn = _route_min_radius(P)
    info = dict(envelope=env.kind, speed_requested_mps=v_req, speed_mps=v,
                # The drawn stroke's own cell size, so a caller that appends a
                # cruise leg can cut it to the same scale (see densify_route).
                cell_m=1.0,
                length_in_m=length_in, length_m=length_in,
                radius_drawn_m=r_drawn, radius_room_m=r_drawn,
                accel_cap_mps2=a_cap, min_speed_mps=v_floor,
                slowed=bool(v < v_req - 1e-9), relaxed=False, shaped=False,
                measured=None, overrun=0.0, notes=[])
    turns = _route_turns(P)

    if len(P) < 3 or float(np.max(turns)) < np.radians(0.5):
        # A straight leg needs no fillet, and densifying it would only rewrite
        # the operator's two points as thirty collinear ones.
        if info["slowed"]:
            info["notes"].append("速度 %.0f m/s 低于机型最低速度，按 %.0f m/s 飞行"
                                 % (v_req, v))
        return dict(points=[[round(float(c), 3) for c in p] for p in P],
                    speed_mps=float(v), info=info)

    info["shaped"] = True
    pts, radii = _round_corners(P, turns)
    r_room = _float_min(radii, r_drawn)
    info["radius_room_m"] = float(r_room)
    v_room = float(np.sqrt(a_cap * max(r_room, 1e-6)))
    if v_room < v - 1e-6:
        if v_room >= v_min - 1e-9:
            # Keep the shape, give up speed: the trade the canvas already warns
            # about, and the one a pilot would make.
            v = v_room
            info["slowed"] = True
        else:
            # The floor binds.  No legal speed flies this stroke, so the shape is
            # what gives -- and the note says by how much.  The relaxation is
            # solved for the FLOOR speed, not for the requested one: slowing to
            # the floor and relaxing only as far as the floor needs is the least
            # shape that has to be given up, while relaxing for the request and
            # then discovering it is still illegal gives up more.
            v = v_min
            relaxed = _relax_route(P, v * v / a_cap)
            # Round the relaxed corners as well: averaging leaves a shallow
            # polygon, and a polygon is a train of small curvature steps -- a
            # 391 m arc sampled every 7 m still measured 392 m/s^3 of jerk,
            # because each vertex is a step even when the step is 1 deg.
            pts, radii = _round_corners(relaxed, _route_turns(relaxed))
            info["relaxed"] = True
            info["radius_room_m"] = _float_min(radii, _route_min_radius(relaxed))

    # Each attempt re-derives the route from the DRAWING with a single
    # relaxation, never by stacking transforms on the previous attempt: stack
    # them and the third pass has averaged and corner-cut the stroke into a
    # straight line, which measures beautifully and is not the route anyone drew.
    relax_r = None if not info["relaxed"] else max(v * v / a_cap, 1.0)
    drawn_cell = float(np.median(np.linalg.norm(np.diff(P, axis=0), axis=1)))
    pts = None
    cell = max(min(drawn_cell, max(r_room, 1.0) * float(np.radians(10.0))), 1.0)
    best = None
    for _ in range(max(int(max_sweeps), 1)):
        base = P if relax_r is None else _relax_route(P, relax_r)
        pts, radii = _round_corners(base, _route_turns(base))
        # Resample at the ARC's resolution, not at whatever the point budget
        # allows: an arc held by three points and then cut into 13 m cells
        # becomes a chain of sharp 4 deg corners, and the spline through them
        # wobbles hard enough to fail a route that is geometrically legal
        # (measured: score 0.99 at 150 m/s became 3.4 once the cell was finer
        # than the arc's own sampling).
        r_now = _float_min(radii, _route_min_radius(base))
        cell = max(min(drawn_cell, max(r_now, 1.0) * float(np.radians(10.0))), 1.0)
        cand = densify_route(pts, cell_m=cell)
        path = Waypoints([list(map(float, p)) for p in cand], speed=v,
                         loop=False, ramp_in=False)
        worst = _measure_route(path, frames=frames, dt=dt)
        score = _route_overrun(worst, env, v)
        if best is None or score < best[0]:
            best = (score, np.asarray(cand, float), v, worst,
                    _float_min(radii, _route_min_radius(base)))
        if score <= 1.0:
            break
        if v > v_min + 1e-6 and relax_r is None:
            v = max(v_min, v * float(np.sqrt(1.0 / score)))
            continue
        # The floor binds (or the speed no longer helps): widen the shape.
        cur = relax_r if relax_r is not None else max(_route_min_radius(P), 1.0)
        relax_r = cur * 1.35
        info["relaxed"] = True
    pts = best[1]

    score, pts, v, worst, r_room = best
    length_out = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
    # The resampling cell: the drawing's own cell, or the arc-sampling
    # resolution (10 deg of the fitted radius) when the drawing is coarser than
    # the arcs that now make up its corners -- whichever is finer.
    info.update(speed_mps=float(v), measured=worst, length_m=length_out,
                overrun=float(score), slowed=bool(v < v_req - 1e-9),
                radius_room_m=float(r_room),
                cell_m=float(cell))
    if info["relaxed"]:
        info["notes"].append(
            "航路太碎：原折线最小转弯半径 %.0f m，%.0f m/s（机型最低 %.0f m/s）需要 "
            "%.0f m，任何合法速度都飞不出这个形状；已把航路拉直到 %.0f m，"
            "原折线总转角 %.0f° 降到 %.0f°"
            % (min(r_drawn, 1e4) if np.isfinite(r_drawn) else 1e4,
               v, v_floor, v * v / a_cap,
               info["radius_room_m"], np.degrees(float(np.sum(turns))),
               np.degrees(float(np.sum(_route_turns(pts))))))
    elif info["slowed"]:
        info["notes"].append(
            "航路太碎：相邻点之间只放得下 %.0f m 转弯半径，%.0f m/s 下需要 %.0f m；"
            "速度由 %.0f 降到 %.0f m/s 以满足 %.1f g / %.0f° 坡度包线"
            % (r_room, v_req, v_req * v_req / a_cap, v_req, v,
               a_cap / G, env.max_bank_deg))
    else:
        info["notes"].append("航路倒角 %.0f m，%.0f m/s 下满足包线"
                             % (r_room, v))
    if worst["pitch_deg"] > float(env.max_gamma_deg) + 1e-6:
        info["notes"].append("航路爬升角 %.0f° 超过 %.0f° 包线，请减小相邻点高度差"
                             % (worst["pitch_deg"], env.max_gamma_deg))
    return dict(points=[[round(float(c), 3) for c in p] for p in pts],
                speed_mps=float(v), info=info)


def _float_min(values, fallback):
    vals = [float(v) for v in np.asarray(values, float) if float(v) > 0.0]
    return min(vals) if vals else float(fallback)


def _route_overrun(worst, env, v, margin=0.9):
    """How far into its own limits a measured route reaches, as a ratio.

    <= 1 means legal *with* that margin.  The margin is not politeness: the
    shaped polyline is flown by a C1 spline, and where a fillet meets its
    tangent the spline is measurably tighter than the arc -- 79.6 m/s on a
    355 m fillet measured 20.6 m/s^2 against the 17.9 m/s^2 the arc asks for,
    i.e. 15 % of overshoot that the design radius alone does not predict.
    """
    ratio = 0.0
    for key, limit in (("accel_mps2", env.max_accel_mps2),
                       ("bank_deg", env.max_bank_deg),
                       ("g", env.max_g),
                       ("jerk_mps3", env.max_jerk_mps3),
                       ("pitch_deg", env.max_gamma_deg)):
        limit = float(limit) * float(margin)
        if limit > 0.0 and np.isfinite(limit):
            ratio = max(ratio, float(worst.get(key, 0.0)) / limit)
    if v > 0.0 and env.kind in _AERODYNAMIC:
        omega = float(np.degrees(G * np.tan(np.radians(min(env.max_bank_deg, 89.0)))
                                 / max(v, 1.0))) * float(margin)
        ratio = max(ratio, float(worst.get("turn_rate_dps", 0.0)) / max(omega, 1e-9))
    return float(ratio)

class FlightManeuver(Path):
    """One continuous manoeuvre: roll in, turn, climb, roll out.

    Why a manoeuvre and not another shape
    -------------------------------------
    Every other path in this module is a closed-form *shape* (circle, eight,
    sinusoid).  A shape says nothing about whether an airframe could fly it: the
    eight's curvature varies with the parameter, so its load factor does too,
    and a scene that asks for a 40 m half-axis at 10 m/s asks for a 15 g turn
    without noticing.  This class starts from the airframe instead.  Its inputs
    are the ones a flight plan actually carries -- start speed, heading change,
    height change, speed change, bank limit -- and the geometry falls out of the
    point-mass equations of a coordinated turn:

        psidot = g tan(bank) / (v cos(gamma))
        (vx, vy, vz) = v (cos g cos psi, cos g sin psi, sin g)

    integrated with RK4 over the manoeuvre.  The published path is the C2
    quintic Hermite through those integrated states, so pos/vel/accel are exact
    polynomial derivatives (see the contract at the top of this file) while
    still being the trajectory the equations of motion produced.

    The bank schedule is sin^2(pi u): wings level at both ends, fully rolled in
    the middle.  That makes the roll rate zero on the first and last frame, so
    the clip neither starts nor ends with an attitude step, and the bank peak is
    solved by bisection to fly the requested heading change exactly -- which
    keeps bank = atan(v omega / g) true by construction instead of by tuning.

    A ten-frame clip of this class is one manoeuvre, not ten conventions: the
    heading, the bank and the height all move monotonically from frame 0 to
    frame 9, which is what "the ten frames are one continuous manoeuvre" means.
    """

    mode = "flight"

    def __init__(self, origin, heading_deg=0.0, speed_mps=20.0, duration_s=6.0,
                 turn_deg=0.0, climb_mps=0.0, accel_mps2=0.0, bank_max_deg=45.0,
                 gamma0_deg=0.0, knots=24, flight_class="uav_fixed"):
        self.p0 = np.asarray(origin, float)
        self.dur = max(float(duration_s), 1e-3)
        self.v0 = max(float(speed_mps), 1e-3)
        self.v1 = max(self.v0 + float(accel_mps2) * self.dur, 0.5)
        self.g0 = float(np.radians(float(gamma0_deg)))
        # The caller states the vertical speed at the end of the manoeuvre (a
        # climb rate is what a flight plan carries); it is converted here to the
        # flight-path angle the equations integrate.
        vz1 = float(climb_mps)
        self.g1 = float(np.arcsin(np.clip(vz1 / self.v1, -0.8, 0.8)))
        self.psi0 = float(np.radians(float(heading_deg)))
        self.turn_target = float(np.radians(float(turn_deg)))
        self.bank_max_rad = float(np.radians(abs(float(bank_max_deg))))
        self.flight_class = str(flight_class)
        self.notes = []

        self.bank_peak = self._solve_bank_peak()      # radians, signed
        self._states = self._integrate(int(knots))
        self.nknots = len(self._states) - 1
        self.knot_dt = self.dur / float(self.nknots)
        self.K = self._knots()

    # -- schedules --------------------------------------------------------
    def _sched(self, t):
        """(speed, gamma, bank) and their time derivatives at clip time t."""
        u = t / self.dur
        S = _smoothstep(u)
        Sd = _smoothstep_d(u) / self.dur
        v = self.v0 + (self.v1 - self.v0) * S
        gam = self.g0 + (self.g1 - self.g0) * S
        # Only the speed and the flight-path angle belong here: the bank schedule
        # is the *unknown* that _solve_bank_peak is bisecting for, so it must not
        # read self.bank_peak (which does not exist yet while solving).
        return v, gam, getattr(self, "bank_peak", 0.0) * (np.sin(np.pi * np.clip(u, 0.0, 1.0)) ** 2),             (self.v1 - self.v0) * Sd, (self.g1 - self.g0) * Sd

    def _heading_rate_at(self, t, bank):
        v, gam, _b, _vd, _gd = self._sched(t)
        return G * np.tan(bank) / (v * np.cos(gam))

    def _solve_bank_peak(self):
        """Bank peak that flies exactly the requested heading change."""
        if abs(self.turn_target) < 1e-9:
            return 0.0
        sign = float(np.sign(self.turn_target))
        target = abs(self.turn_target)

        def flown(bank_peak):
            # integrate psi only: cheap, and the result is monotone in the peak
            n = 240
            dt = self.dur / n
            psi = 0.0
            for i in range(n):
                t = (i + 0.5) * dt
                u = np.clip(t / self.dur, 0.0, 1.0)
                psi += self._heading_rate_at(t, bank_peak * np.sin(np.pi * u) ** 2) * dt
            return psi

        lo, hi = 0.0, min(self.bank_max_rad, np.radians(80.0))
        if flown(hi) < target:
            self.notes.append(
                "heading change clipped: %.1f deg needs more than the %.1f deg bank limit"
                % (np.degrees(target), np.degrees(self.bank_max_rad)))
            return sign * hi
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if flown(mid) < target:
                lo = mid
            else:
                hi = mid
        return sign * 0.5 * (lo + hi)

    # -- integration ------------------------------------------------------
    def _integrate(self, knots):
        """RK4 the point-mass model; return one state per knot."""
        n = max(64, knots * 8)
        dt = self.dur / n
        step = max(1, n // knots)
        x = np.array([self.p0[0], self.p0[1], self.p0[2], self.psi0], float)

        def deriv(state, t):
            u = np.clip(t / self.dur, 0.0, 1.0)
            bank = self.bank_peak * np.sin(np.pi * u) ** 2
            v, gam = self._sched(t)[:2]
            psi = state[3]
            return np.array([v * np.cos(gam) * np.cos(psi),
                             v * np.cos(gam) * np.sin(psi),
                             v * np.sin(gam),
                             self._heading_rate_at(t, bank)], float)

        states = [x.copy()]
        for i in range(n):
            t = i * dt
            k1 = deriv(x, t)
            k2 = deriv(x + 0.5*dt*k1, t + 0.5*dt)
            k3 = deriv(x + 0.5*dt*k2, t + 0.5*dt)
            k4 = deriv(x + dt*k3, t + dt)
            x = x + (dt/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4)
            if (i + 1) % step == 0 and len(states) < knots + 1:
                states.append(x.copy())
        while len(states) < knots + 1:
            states.append(x.copy())
        return states[:knots+1]

    def _knots(self):
        """(pos, vel, accel) at every knot, from the integrated state."""
        out = []
        for k, st in enumerate(self._states):
            t = self.knot_dt * k
            v, gam, bank, v_d, gam_d = self._sched(t)
            psi = st[3]
            cpsi, spsi = np.cos(psi), np.sin(psi)
            cg, sg = np.cos(gam), np.sin(gam)
            u_hat = np.array([cg * cpsi, cg * spsi, sg])
            n_h = np.array([-spsi, cpsi, 0.0])
            n_v = np.array([-sg * cpsi, -sg * spsi, cg])
            psi_d = self._heading_rate_at(t, bank)
            vel = v * u_hat
            # d/dt of v * u_hat with u_hat(u) fixed by (psi(t), gam(t))
            acc = v_d * u_hat + v * psi_d * cg * n_h + v * gam_d * n_v
            out.append((st[:3].copy(), vel, acc))
        return out

    # -- evaluation -------------------------------------------------------
    def _eval(self, t):
        h = self.knot_dt
        i = int(np.floor(float(t) / h))
        i = max(0, min(i, self.nknots - 1))
        s = (float(t) - i * h) / h
        p0, v0, a0 = self.K[i]
        p1, v1, a1 = self.K[i+1]
        pos = (_q00(s)*p0 + h*_q10(s)*v0 + h*h*_q20(s)*a0
               + _q01(s)*p1 + h*_q11(s)*v1 + h*h*_q21(s)*a1)
        vel = (_q00d(s)*p0/h + _q10d(s)*v0 + h*_q20d(s)*a0
               + _q01d(s)*p1/h + _q11d(s)*v1 + h*_q21d(s)*a1)
        acc = (_q00dd(s)*p0/(h*h) + _q10dd(s)*v0/h + _q20dd(s)*a0
               + _q01dd(s)*p1/(h*h) + _q11dd(s)*v1/h + _q21dd(s)*a1)
        return pos, vel, acc

    def pos(self, t):
        return self._eval(t)[0]

    def vel(self, t):
        return self._eval(t)[1]

    def accel(self, t):
        return self._eval(t)[2]

    def jerk(self, t):
        h = self.knot_dt
        i = int(np.floor(float(t) / h))
        i = max(0, min(i, self.nknots - 1))
        s = (float(t) - i * h) / h
        p0, v0, a0 = self.K[i]
        p1, v1, a1 = self.K[i+1]
        return (_q00ddd(s)*p0/(h**3) + _q10ddd(s)*v0/(h*h) + _q20ddd(s)*a0/h
                + _q01ddd(s)*p1/(h**3) + _q11ddd(s)*v1/(h*h) + _q21ddd(s)*a1/h)

    # -- reporting --------------------------------------------------------
    def profile(self):
        """What the manoeuvre actually does, for the scene report."""
        ts = np.linspace(0.0, self.dur, 241)
        vs = np.array([self.vel(t) for t in ts])
        accs = np.array([self.accel(t) for t in ts])
        banks = [coordinated_bank_deg(v, a) for v, a in zip(vs, accs)]
        gs = [load_factor(v, a) for v, a in zip(vs, accs)]
        return dict(
            duration_s=round(self.dur, 4),
            heading_change_deg=round(_wrap_deg(_yaw_of(vs[-1]) - _yaw_of(vs[0])), 3),
            turn_requested_deg=round(float(np.degrees(self.turn_target)), 3),
            bank_peak_deg=round(float(np.degrees(self.bank_peak)), 3),
            bank_max_deg=round(float(np.degrees(self.bank_max_rad)), 3),
            max_load_g=round(float(max(gs)), 3),
            speed_mps=[round(float(np.linalg.norm(vs[0])), 3),
                       round(float(np.linalg.norm(vs[-1])), 3)],
            climb_mps=[round(float(vs[0][2]), 3), round(float(vs[-1][2]), 3)],
            displacement_m=[round(float(x), 3) for x in
                            (np.asarray(self.pos(self.dur)) - self.p0)],
            notes=list(self.notes),
        )


class GroundPath(Path):
    """Flat-surface constraint applied to all path modes, including velocity."""

    def __init__(self, path, z):
        self.path, self.z, self.mode = path, float(z), path.mode

    def pos(self, t):
        p = self.path.pos(t).copy()
        p[2] = self.z
        return p

    def vel(self, t):
        v = self.path.vel(t).copy()
        v[2] = 0.0
        return v

    def accel(self, t):
        a = self.path.accel(t).copy()
        a[2] = 0.0
        return a

    def heading(self, t):
        return self.path.heading(t)


# ---------------------------------------------------------------------------
# Tracks: the single state source
# ---------------------------------------------------------------------------
# A Track is the one answer to "where is this thing at time t": position,
# velocity, acceleration and body frame, all analytic functions of clip time.
# Sensors, labels and countermeasures read the same object, so they cannot
# disagree about the same aircraft -- which is exactly how a scene ended up with
# the target on the line of sight while a decoy flew its own straight line and a
# photo showed the two hundred metres apart.
#
# The contract is docs/多目标运动仿真调研.md 5.1:
#
#     pos(t) -> (3,)      vel(t) -> (3,)
#     accel(t) -> (3,)    frame(t) -> (3,3) body -> world
#
# accel is the central difference of the analytic *velocity*, never of the
# position.  Differencing a position across a clamped or piecewise parameter is
# what produced the 76.9 g first frame documented at the top of this file; a
# centred difference of a smooth velocity has O(h^2) error, i.e. ~1e-6 m/s^2 at
# h = 1e-3, and the acceleration a sensor uses is then the derivative of the
# same function that yields Doppler.
#
# frame(t) is body -> world with exactly the rotation scene.py poses a mesh
# with, so a mount point written in body axes ([0, 18, -12] = right, up, down
# metres on the airframe) turns with the model instead of staying on a world
# azimuth.

_TRACK_H = 1e-3


def body_to_world(yaw_deg, pitch_deg=0.0, roll_deg=0.0):
    """Body -> world rotation, identical to the one scene.py builds meshes with.

    R = Rz(yaw) @ Ry(-pitch) @ Rx(roll): the same chain scene.build_world uses,
    so "body frame" here and "the drawn aircraft" cannot drift apart by a sign.
    """
    return (rot_z(float(yaw_deg)) @ rot_y(-float(pitch_deg))
            @ rot_x(float(roll_deg)))


def _rotvec(R):
    """Axis-angle vector of a rotation matrix, radians: R = exp([w]x)."""
    R = np.asarray(R, float)
    c = float(np.clip(0.5 * (float(np.trace(R)) - 1.0), -1.0, 1.0))
    th = float(np.arccos(c))
    skew = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], float)
    if th < 1e-8:
        return 0.5 * skew
    if th < np.pi - 1e-4:
        return (th / (2.0 * np.sin(th))) * skew
    # near pi the antisymmetric part vanishes; recover the axis from R + I
    A = 0.5 * (R + np.eye(3))
    i = int(np.argmax(np.diag(A)))
    axis = A[:, i] / np.sqrt(max(float(A[i, i]), 1e-12))
    return axis * th


def angular_velocity(track, t, tau=0.0):
    """World-frame angular velocity of a track's body frame, rad/s.

    tau > 0 compares the frame with the one tau seconds earlier -- the same lag
    the formation and tether rules are written in.  tau = 0 falls back to a
    centred difference of the frame.  The returned w satisfies dR/dt = [w]x R,
    so a body point r has world velocity w x (R r): that is the term which makes
    a wingman's velocity the leader's plus the rotation of the offset, instead
    of a second independent velocity.
    """
    t = float(t)
    tau = float(tau or 0.0)
    if tau > 0.0:
        R0 = np.asarray(track.frame(t - tau), float)
        R1 = np.asarray(track.frame(t), float)
        return _rotvec(R1 @ R0.T) / tau
    R0 = np.asarray(track.frame(t - _TRACK_H), float)
    R1 = np.asarray(track.frame(t + _TRACK_H), float)
    return _rotvec(R1 @ R0.T) / (2.0 * _TRACK_H)


class Track:
    """Analytic t -> (pos, vel, accel) plus the body frame.  C-infinity.

    Subclasses publish pos() and vel() (which must be exact derivatives of one
    another); accel() defaults to the centred difference of vel, which is what
    the design specifies for the seeker solve.  A subclass whose path already
    owns an analytic acceleration overrides accel() instead of paying for the
    difference (PathTrack does).
    """

    kind = "track"
    mode = "track"

    def pos(self, t):
        raise NotImplementedError

    def vel(self, t):
        raise NotImplementedError

    def accel(self, t, h=None):
        """d vel / d t: centred difference of the analytic velocity."""
        h = _TRACK_H if h is None else float(h)
        t = float(t)
        return (np.asarray(self.vel(t + h), float)
                - np.asarray(self.vel(t - h), float)) / (2.0 * h)

    def jerk(self, t, h=None):
        h = _TRACK_H if h is None else float(h)
        t = float(t)
        return (np.asarray(self.accel(t + h), float)
                - np.asarray(self.accel(t - h), float)) / (2.0 * h)

    def yaw(self, t):
        """Heading of the published velocity, degrees; None if it has none.

        None is deliberate (motion.Path.yaw does the same): a hovering body has
        no heading, and returning 0.0 would silently point it north.
        """
        v = np.asarray(self.vel(t), float)
        if float(np.hypot(v[0], v[1])) < 1e-6:
            return None
        return _yaw_of(v)

    def pitch(self, t):
        """Flight-path angle, degrees (nose-up positive)."""
        return climb_angle_deg(self.vel(t))

    def roll(self, t):
        """Coordinated-turn bank, degrees (same definition as the envelope)."""
        return coordinated_bank_deg(self.vel(t), self.accel(t))

    def frame(self, t):
        """Body -> world rotation at time t."""
        y = self.yaw(t)
        return body_to_world(0.0 if y is None else y, self.pitch(t), self.roll(t))

    def speed(self, t):
        return float(np.linalg.norm(np.asarray(self.vel(t), float)))

    def heading(self, t):
        return self.yaw(t)

    def state(self, t):
        """One dict with every published quantity, evaluated at the same t."""
        t = float(t)
        p = np.asarray(self.pos(t), float)
        v = np.asarray(self.vel(t), float)
        a = np.asarray(self.accel(t), float)
        y = self.yaw(t)
        pitch = climb_angle_deg(v)
        roll = coordinated_bank_deg(v, a)
        return dict(pos=p, vel=v, accel=a, yaw=y, pitch=pitch, roll=roll,
                    speed_mps=float(np.linalg.norm(v)),
                    frame=body_to_world(0.0 if y is None else y, pitch, roll))

    # -- constructors -----------------------------------------------------
    @classmethod
    def from_path(cls, path):
        """Wrap an existing motion.Path as a Track (pos/vel/accel as published)."""
        return PathTrack(path)

    @classmethod
    def from_functions(cls, pos, vel, accel=None, yaw=None, pitch=None, roll=None):
        return FunctionTrack(pos, vel, accel=accel, yaw=yaw, pitch=pitch, roll=roll)


class PathTrack(Track):
    """Track view of an existing motion.Path: the path stays the authority.

    pos/vel/accel are taken verbatim from the path, so a world-mode or platform
    clip keeps its published derivatives bit for bit; only the frame is new.
    """

    kind = "path"

    def __init__(self, path):
        self.path = path
        self.mode = str(getattr(path, "mode", "path"))

    def pos(self, t):
        return np.asarray(self.path.pos(float(t)), float)

    def vel(self, t):
        return np.asarray(self.path.vel(float(t)), float)

    def accel(self, t):
        return np.asarray(self.path.accel(float(t)), float)

    def jerk(self, t):
        return np.asarray(self.path.jerk(float(t)), float)

    def yaw(self, t):
        y = self.path.yaw(float(t)) if hasattr(self.path, "yaw") else None
        return None if y is None else float(y)


class FunctionTrack(Track):
    """Track built from callables -- how plan.py wraps the seeker solve.

    pos and vel are required and must be analytic derivatives of each other;
    accel is optional and defaults to the centred difference of vel (the
    design's h = 1e-3), never to a difference of pos.
    """

    kind = "function"

    def __init__(self, pos, vel, accel=None, yaw=None, pitch=None, roll=None):
        self._pos = pos
        self._vel = vel
        self._accel = accel
        self._yaw = yaw
        self._pitch = pitch
        self._roll = roll

    def pos(self, t):
        return np.asarray(self._pos(float(t)), float)

    def vel(self, t):
        return np.asarray(self._vel(float(t)), float)

    def accel(self, t, h=None):
        if self._accel is None:
            return Track.accel(self, t, h=h)
        return np.asarray(self._accel(float(t)), float)

    def yaw(self, t):
        if self._yaw is not None:
            return self._yaw(float(t))
        return Track.yaw(self, t)

    def pitch(self, t):
        if self._pitch is not None:
            return float(self._pitch(float(t)))
        return Track.pitch(self, t)

    def roll(self, t):
        if self._roll is not None:
            return float(self._roll(float(t)))
        return Track.roll(self, t)


class FormationTrack(Track):
    """A formation member: absolute state derived from a parent Track.

    offset_body is metres in the parent's body frame (x forward, y lateral,
    z up, matching scene.py's rotation): a wingman at [-120, 60, -15] sits
    120 m behind, 60 m to one side and 15 m below the leader **and rolls with
    it**.  An offset written in world axes instead stays where it was when the
    leader turns, which is how a dispensed body ends up on the aircraft's back
    instead of under its belly (docs/多目标运动仿真调研.md 1.3 c).

    modes
        rigid  the offset is carried by the current carrier frame.
        lag    the offset is carried by the carrier as it was tau_lag_s ago.
               The distance to the parent is unchanged (|R o| == |o|) but the
               member cuts the corner, which is what a real wingman does.
        free   rigid plus a bounded manoeuvre of its own inside the carrier.

    carriers
        body   the parent's full body frame: the member rolls and pitches with
               it.  Right for a dispensed body, a store or a mount, which is
               why it stays the default.
        level  yaw only, about world +Z: the member keeps a horizontal offset
               and does NOT rotate with the leader's bank.  Two aircraft an
               operator placed side by side are placed in the world, not in the
               leader's banked plane -- carrying that offset through a 77 deg
               bank swung the wingman through a couple of hundred metres of
               frame for a formation nobody asked for.

    The velocity is the parent's plus the rotation of the offset, w x (R o).
    That term is the whole difference between "flies in formation" and "two
    aircraft with the same speed that happen to be near each other".
    """

    kind = "formation"

    def __init__(self, parent, offset_body, mode="lag", tau_lag_s=0.3,
                 own_maneuver=None, max_offset_m=None, max_ang_rate_rad_s=None,
                 carrier="body"):
        self.parent = parent
        self.offset = np.asarray(offset_body, float).reshape(3)
        self.mode = str(mode or "lag").lower()
        if self.mode not in ("rigid", "lag", "free"):
            raise ValueError("formation mode must be rigid, lag or free")
        self.carrier = str(carrier or "body").lower()
        if self.carrier not in ("body", "level"):
            raise ValueError("formation carrier must be body or level")
        self.tau = max(float(tau_lag_s or 0.0), 0.0)
        #: Turn-rate ceiling the lever arm may be rotated by.  A hand-drawn
        #: waypoint route is only C1, so its curvature steps at every corner and
        #: a 250 m/s corner reports several hundred deg/s.  Multiplied by an
        #: 80 m formation offset that lever term reached 838 m/s and the follower
        #: reversed direction every frame.  None keeps the unbounded form for
        #: callers that measured their own rate.
        self.max_ang_rate = (None if max_ang_rate_rad_s is None
                             else max(float(max_ang_rate_rad_s), 0.0))
        if max_offset_m is not None and float(max_offset_m) > 0.0:
            n = float(np.linalg.norm(self.offset))
            if n > float(max_offset_m):
                self.offset = self.offset * (float(max_offset_m) / n)
        self.own = _own_maneuver(own_maneuver)

    def _carrier_at(self, tt):
        """The frame that carries the offset at absolute time tt."""
        tt = float(tt)
        if self.carrier == "level":
            yaw = self.parent.yaw(tt)
            return body_to_world(0.0 if yaw is None else float(yaw), 0.0, 0.0)
        return np.asarray(self.parent.frame(tt), float)

    def _frame_at(self, t):
        if self.mode == "lag" and self.tau > 0.0:
            return self._carrier_at(float(t) - self.tau)
        return self._carrier_at(float(t))

    def pos(self, t):
        R = self._frame_at(t)
        o = self.offset
        if self.mode == "free" and self.own is not None:
            o = o + self.own.pos(float(t))
        return np.asarray(self.parent.pos(float(t)), float) + R @ o

    def _rate_at(self, t):
        """Turn rate over the lag window, with the window's END kept at t >= 0.

        The window is anchored at the current instant and reaches back, so it
        never straddles the clip start in a way that changes its width: at
        t = -h it is [t1 - tau, 0] and at t = +h it is [t1 - tau, +h], which are
        the same measurement to first order.  Freezing the window's start at 0
        instead made the rate switch on at t = 0 (zero before, the route's real
        rate after) and the lever term stepped by 16 m/s across one centred
        difference, reading 8205 m/s^2 on the follower's first frame.
        """
        t1 = max(float(t), 0.0)
        tau = self.tau
        if tau > 0.0:
            t0 = t1 - tau
            if t0 < 0.0 and self._no_history(t0):
                # An open drawn route has no history before the clip: position at
                # negative time is frozen on the departure point.  Reading the
                # window backwards therefore reported ZERO turn rate on frame 0
                # and the route's real rate one frame later, so the follower's
                # bank swung 70 deg between the first two frames.  Where the
                # history genuinely does not exist, look forward over the same
                # width instead; the value is continuous at t = tau either way.
                t0, t1 = 0.0, tau
            R0 = self._carrier_at(t0)
            R1 = self._carrier_at(t1)
            return _rotvec(R1 @ R0.T) / tau
        return angular_velocity(self.parent, t1, 0.0)

    def _no_history(self, t):
        """True when the parent's state before the clip start is a frozen pose."""
        try:
            return bool(np.allclose(
                np.asarray(self.parent.pos(float(t)), float),
                np.asarray(self.parent.pos(0.0), float), atol=1e-6))
        except (TypeError, ValueError, AttributeError):
            return False

    def vel(self, t):
        t = float(t)
        R = self._frame_at(t)
        o = self.offset
        body_v = np.zeros(3)
        if self.mode == "free" and self.own is not None:
            o = o + self.own.pos(t)
            body_v = self.own.vel(t)
        # The rate is capped: a route corner is a curvature step, not a physical
        # turn, and the lever arm is what turned that step into an 840 m/s
        # follower that reversed every frame.
        w = self._rate_at(t)
        if self.max_ang_rate is not None:
            wn = float(np.linalg.norm(w))
            if wn > self.max_ang_rate > 0.0:
                w = w * (self.max_ang_rate / wn)
        return (np.asarray(self.parent.vel(t), float) + R @ body_v
                + np.cross(w, R @ o))

    def yaw(self, t):
        return self.parent.yaw(float(t))

    def pitch(self, t):
        return float(self.parent.pitch(float(t)))

    def roll(self, t):
        return float(self.parent.roll(float(t)))

    def frame(self, t):
        # a formation member is flown in the parent's frame; its own rendered
        # attitude comes from its own velocity (plan._ObjectTrack overrides this)
        return np.asarray(self.parent.frame(float(t)), float)


class AnchorTrack(Track):
    """Parent state reconstructed from ONE pose sample -- a compatibility path.

    The built World (and the render manifest) hand the ECM layer a parent's
    pose at the frame time.  From one sample the only honest extrapolation is a
    constant-velocity line; that is exact for a parent flying straight, and it
    is why this class is a fallback and not the source.  The planner's Track is
    the authority, because only it can answer "where was the parent when the
    flare left the rail two seconds ago".
    """

    kind = "anchor"

    def __init__(self, pos, vel, yaw=None, pitch=0.0, roll=0.0, t0=0.0):
        self.p0 = np.asarray(pos, float).reshape(3)
        self.v = np.asarray(vel, float).reshape(3)
        self.t0 = float(t0)
        self._pitch = float(pitch)
        self._roll = float(roll)
        if yaw is None:
            self._yaw = (_yaw_of(self.v)
                         if float(np.hypot(self.v[0], self.v[1])) > 1e-6 else 0.0)
        else:
            self._yaw = float(yaw)

    def pos(self, t):
        return self.p0 + self.v * (float(t) - self.t0)

    def vel(self, t):
        return self.v.copy()

    def accel(self, t):
        return np.zeros(3)

    def yaw(self, t):
        return self._yaw

    def pitch(self, t):
        return self._pitch

    def roll(self, t):
        return self._roll

    def frame(self, t):
        return body_to_world(self._yaw, self._pitch, self._roll)


class _OwnManeuver:
    """Bounded relative motion inside the body frame (formation 'free' mode).

    A lateral sinusoid is the smallest thing that is both bounded and C-inf:
    amp_m is metres either side, hz the weave frequency, phase a constant, and
    the sinusoid starts at zero offset so a scene's authored position is where
    frame 0 is.
    """

    def __init__(self, spec):
        spec = dict(spec or {})
        mode = str(spec.get("mode", "weave")).lower()
        if mode not in ("weave", "sin", "lateral"):
            raise ValueError("own_maneuver.mode must be weave")
        self.amp = float(spec.get("amp_m", 0.0))
        self.w = 2.0 * np.pi * float(spec.get("hz", 0.2))
        self.ph = float(spec.get("phase", 0.0))
        self.axis = np.asarray(spec.get("axis", [0.0, 1.0, 0.0]), float).reshape(3)

    def pos(self, t):
        return self.axis * (self.amp * (np.sin(self.w * float(t) + self.ph)
                                        - np.sin(self.ph)))

    def vel(self, t):
        return self.axis * (self.amp * self.w * np.cos(self.w * float(t) + self.ph))


def _own_maneuver(spec):
    if spec is None:
        return None
    if isinstance(spec, dict) and str(spec.get("mode", "weave")).lower() in ("none", "off"):
        return None
    return _OwnManeuver(spec)


def make_path(spec, ground_z=None):
    path = _make_path(spec, ground_z)
    return GroundPath(path, ground_z) if ground_z is not None else path


def _make_path(spec, ground_z=None):
    """Build a Path from a scene config block.

    ground_z pins the trajectory to a surface (a rover cannot fly); passing None
    leaves the z the mode itself produces.
    """
    spec = spec or {}
    mode = str(spec.get("mode", "line")).lower()
    p0 = np.array(spec.get("origin", [0.0, 0.0, 0.0]), float)
    ground = spec.get("ground_z") if ground_z is None else ground_z
    if ground is not None:
        p0[2] = float(ground)

    if mode in ("static", "hover", "parked"):
        return Static(p0, bob_m=float(spec.get("bob_m", 0.0)),
                      bob_hz=float(spec.get("bob_hz", 0.7)),
                      phase=float(spec.get("phase", 0.0)))

    if mode in ("line", "straight"):
        return Line(p0, np.array(spec.get("vel", [1.0, 0.0, 0.0]), float))

    if mode in ("weave", "evade", "maneuver"):
        return Weave(p0, np.array(spec.get("vel", [1.0, 0.0, 0.0]), float),
                     float(spec.get("amplitude_m", spec.get("weave_amp_m", 2.0))),
                     hz=float(spec.get("hz", spec.get("weave_hz", 0.25))),
                     z_amp=float(spec.get("z_amplitude_m", 0.0)),
                     phase=float(spec.get("phase", 0.0)))

    if mode in ("arc", "circle", "orbit"):
        r = float(spec.get("radius_m", 50.0))
        # Turn rate follows the speed for a fixed geometry: a scene that also
        # states omega_rad_s is describing the *same* circle through a second
        # number, so the two are cross-checked instead of silently diverging
        # (omega = v/r is what makes lateral acceleration v*omega = v^2/r).
        speed = spec.get("speed_mps")
        omega_given = spec.get("omega_rad_s")
        if omega_given is not None and speed is not None \
                and abs(float(omega_given) * r - float(speed)) > 1e-3 * max(abs(float(speed)), 1.0):
            omega = float(speed) / max(r, 1e-6)
        elif omega_given is not None:
            omega = float(omega_given)
        else:
            omega = float(speed if speed is not None else 10.0) / max(r, 1e-6)
        centre = np.array(spec.get("centre", p0), float)
        if ground is not None:
            centre[2] = float(ground)
        # A circle is under-determined by (centre, radius, phase): a scene that
        # states where the target starts must get a path that starts there, so
        # the phase is derived from the origin when it is not given.
        phase = spec.get("phase")
        if phase is None:
            origin = spec.get("origin")
            if origin is not None:
                phase = float(np.arctan2(float(origin[1]) - centre[1],
                                         float(origin[0]) - centre[0]))
            else:
                phase = 0.0
        return Arc(centre, r, omega,
                   z0=float(ground if ground is not None else centre[2]),
                   vz=float(spec.get("climb_mps", 0.0)),
                   phase=float(phase))

    if mode in ("figure8", "figure-8", "eight"):
        ax = float(spec.get("ax_m", 40.0))
        omega = float(spec.get("omega_rad_s",
                               float(spec.get("speed_mps", 10.0)) / max(ax, 1e-6)))
        centre = np.array(spec.get("centre", p0), float)
        if ground is not None:
            centre[2] = float(ground)
        return Figure8(centre, ax, float(spec.get("ay_m", ax)), omega,
                       z0=float(ground if ground is not None else centre[2]),
                       vz=float(spec.get("climb_mps", 0.0)),
                       phase=float(spec.get("phase", 0.0)))

    if mode in ("dive_weave", "diveweave", "dive"):
        return DiveWeave(p0,
                         spec.get("target", [0.0, 0.0, 0.0]),
                         speed_mps=float(spec.get("speed_mps", 180.0)),
                         heading_deg=float(spec.get("heading_deg", 90.0)),
                         dive_deg=float(spec.get("dive_deg", -14.0)),
                         weave_amp_m=float(spec.get("weave_amp_m", 0.0)),
                         weave_hz=float(spec.get("weave_hz", 0.22)),
                         t_end=float(spec.get("t_end", 6.0)),
                         standoff_end_m=spec.get("standoff_end_m"),
                         phase=float(spec.get("phase", 0.0)),
                         target_vel=spec.get("target_vel"),
                         solve=str(spec.get("solve", "standoff")),
                         bank_max_deg=spec.get("bank_max_deg"),
                         max_speed_mps=spec.get("max_speed_mps"))

    if mode in ("waypoints", "path", "spline"):
        pts = spec.get("points") or []
        if len(pts) < 2:
            raise ValueError("waypoints mode needs at least two 'points'")
        pts = [np.asarray(p, float) for p in pts]
        if ground is not None:
            for p in pts:
                p[2] = float(ground)
        return Waypoints(pts, speed=float(spec.get("speed_mps", 8.0)),
                         loop=bool(spec.get("loop", True)),
                         phase=float(spec.get("phase", 0.0)),
                         ramp_in=bool(spec.get("ramp_in", True)))

    if mode in ("flight", "maneuver", "manoeuvre", "sortie", "sequence"):
        heading = spec.get("heading_deg")
        if heading is None:
            heading = _yaw_of(spec.get("vel", (1.0, 0.0, 0.0)))
        speed = spec.get("speed_mps")
        if speed is None:
            speed = float(np.linalg.norm(np.asarray(spec.get("vel", [10.0, 0.0, 0.0]), float)))
        return FlightManeuver(
            p0,
            heading_deg=float(heading),
            speed_mps=float(speed),
            duration_s=float(spec.get("duration_s", 6.0)),
            turn_deg=float(spec.get("turn_deg", 0.0)),
            climb_mps=float(spec.get("climb_mps", 0.0)),
            accel_mps2=float(spec.get("accel_mps2", 0.0)),
            bank_max_deg=float(spec.get("bank_max_deg", 45.0)),
            gamma0_deg=float(spec.get("gamma0_deg", 0.0)),
            knots=int(spec.get("knots", 24)),
            flight_class=str(spec.get("flight_class", "uav_fixed")),
        )

    raise ValueError("unknown trajectory mode: %s" % mode)


class Rig:
    """Sensor mount: where each camera points, as a smooth function of time.

    'track' aims at the target as it was lag seconds ago.  A pure time delay is
    used rather than a stateful filter on purpose: it is causal, analytic, and
    independent of where a chunk boundary falls, so a run split into pieces
    still produces one continuous pan.  A real gimbal behaves the same way --
    the mount lags the target.

    The lag is a pure delay on an analytic path, so it is valid for negative
    time too.  Clamping the delayed time at zero (the obvious implementation)
    freezes the aim for the first lag_s of every clip and then releases it,
    which is a step in line-of-sight rate: the mount's own version of the
    76.9 g bug, and it shows up as the first ten frames panning at half speed.
    """

    def __init__(self, spec, base_pose, target_path=None, rng=None):
        spec = spec or {}
        self.mode = str(spec.get("mode", "static")).lower()
        self.base = dict(base_pose)
        self.lag = float(spec.get("lag_s", 0.35))
        self.path = target_path
        self.lead = np.array(spec.get("offset", [0.0, 0.0, 0.0]), float)
        self.orbit_r = float(spec.get("orbit_radius_m", 30.0))
        self.orbit_w = float(spec.get("orbit_speed_mps", 4.0)) / max(self.orbit_r, 1e-6)
        # smooth vibration: a sum of incommensurate sinusoids, never white noise,
        # so consecutive frames stay physically adjacent
        rng = rng if rng is not None else np.random.default_rng(0)
        self.vib_amp = float(spec.get("vibration_m", 0.0))
        self.vib_yaw = float(spec.get("vibration_deg", 0.0))
        self.vib = [(float(rng.uniform(0.4, 2.4)), float(rng.uniform(0.0, 6.283)),
                     float(rng.uniform(0.6, 1.0))) for _ in range(3)]

    def _wobble(self, t):
        if self.vib_amp <= 0.0 and self.vib_yaw <= 0.0:
            return np.zeros(3), 0.0
        s = np.zeros(3)
        for f, ph, a in self.vib:
            s += a * np.array([np.sin(2 * np.pi * f * t + ph),
                               np.cos(2 * np.pi * f * 0.87 * t + ph),
                               0.5 * np.sin(2 * np.pi * f * 1.31 * t + ph)])
        s *= 1.0 / max(len(self.vib), 1)
        return s * self.vib_amp, self.vib_yaw * float(np.sin(2 * np.pi * 0.7 * t))

    def jitter(self, t):
        """Public accessor for the smooth mount vibration (offset, yaw deg)."""
        return self._wobble(t)

    def pose(self, t, cam_spec):
        """Return (pos, look_at) for one named sensor at clip time t."""
        pos = np.array(self.base.get("pos", cam_spec.get("pos", [0, 0, 0])), float)
        look = np.array(self.base.get("look_at", cam_spec.get("look_at", [0, 0, 0])), float)
        d, dyaw = self._wobble(t)
        pos = pos + d

        if self.mode in ("track", "gimbal", "follow") and self.path is not None:
            tt = t - self.lag
            aim = self.path.pos(tt) + self.lead
            if self.mode == "follow":
                # chase camera: hang back along the target's own heading
                h = _unit(self.path.vel(tt), (1.0, 0.0, 0.0))
                pos = (aim - h * float(self.base.get("standoff_m", 25.0))
                       + np.array([0.0, 0.0, float(self.base.get("standoff_z_m", 6.0))]))
            look = aim
        elif self.mode in ("orbit", "flyby") and self.path is not None:
            aim = self.path.pos(t)
            a = self.orbit_w * t
            pos = aim + np.array([self.orbit_r * np.cos(a), self.orbit_r * np.sin(a),
                                  float(self.base.get("height_m", 6.0))])
            look = aim

        if dyaw != 0.0:
            v = look - pos
            a = np.radians(dyaw)
            ca, sa = np.cos(a), np.sin(a)
            look = pos + np.array([ca * v[0] - sa * v[1], sa * v[0] + ca * v[1], v[2]])
        return pos, look

    def aim_rate(self, t, cam_spec, h=1e-3):
        """Angular rate of the line of sight, deg/s -- what a real gimbal must
        slew at.  Reported by the diagnostic, because a mount that has to slew
        at tens of deg/s is the metric version of "the pan looks wrong"."""
        a0, a1 = self.pose(t - h, cam_spec), self.pose(t + h, cam_spec)
        d0 = a0[1] - a0[0]
        d1 = a1[1] - a1[0]
        n0 = np.linalg.norm(d0)
        n1 = np.linalg.norm(d1)
        if n0 < 1e-9 or n1 < 1e-9:
            return 0.0
        c = float(np.clip(np.dot(d0, d1) / (n0 * n1), -1.0, 1.0))
        return float(np.degrees(np.arccos(c)) / (2.0 * h))


def yaw_rate(path, t, h=1e-3):
    """Degrees/second, for reporting whether a trajectory is flyable.

    Uses the analytic turn rate where the path provides one; falling back to a
    heading difference returns 0 for a hovering mount (yaw() is undefined at
    zero horizontal speed) and would silently under-report a real spin-up.
    """
    try:
        return float(np.degrees(turn_rate(path.vel(t), path.accel(t))))
    except Exception:
        pass
    y0, y1 = path.yaw(t - h), path.yaw(t + h)
    if y0 is None or y1 is None:
        return 0.0
    d = (y1 - y0 + 180.0) % 360.0 - 180.0
    return float(d / (2.0 * h))
