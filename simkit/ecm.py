"""Scene-level countermeasure entities: geometry, motion and RF/IR effects.

simkit/interference.py perturbs rendered rasters.  It has no position, no
velocity and it never reaches a forward model, so it can imitate a bright patch
but not a jammer: the user's requirement is the opposite of that.  An entity
here is a scene object with its own position, velocity, attitude, spatial
extent, band, power and life cycle, and its effect appears **inside** the
sensor forward models:

    radar      -> extra echoes in the same (beat, Doppler) list the target
                  uses, plus an additive receiver-noise term for noise jamming
    lidar      -> the entity is real triangles in the shared World, so the same
                  rays hit it, it occludes the target and it returns a false
                  range
    optical/IR -> real triangles plus their own temperature, so the thermal
                  solver gives self-emission and the rasteriser lets the body
                  occlude the target behind it

Everything is disabled unless the scene carries an "ecm" block, so existing
datasets stay byte-identical.

Model boundaries (deliberate, and stated rather than hidden):
  * the radar link is the same radar equation radar.py uses, one-way for a
    jammer and two-way for a body; no multipath, no polarisation mismatch;
  * a chaff cloud is a resolved volume: its echo is placed at the cloud centre
    with the cloud's total cross-section, not smeared over its extent.  A real
    cloud spans several range cells, so the single-echo form understates how
    hard it is to see through; splitting the cloud into sub-scatterers is the
    next step and is not done here;
  * a noise jammer narrower than the receiver band is modelled as a flat noise
    floor instead of a few corrupted range cells;
  * nothing here has been validated against a real instrument.
"""
import zlib

import numpy as np

from . import motion
from .mesh import Mesh, box, rot_x, rot_y, rot_z

C_LIGHT = 2.99792458e8
K_BOLTZ = 1.380649e-23

_PASSIVE = ("chaff_cloud", "corner_reflector", "decoy", "infrared_flare")
_ACTIVE = ("noise_jammer", "false_target", "range_gate_pull_off",
           "velocity_gate_pull_off", "angle_deception")
GROUP_OF = {k: "passive" for k in _PASSIVE}
GROUP_OF.update({k: "active" for k in _ACTIVE})
KINDS = tuple(sorted(GROUP_OF))

# kinds that exist as triangles in the World.  The rest are pure signal
# generators (a DRFM repeater has no airframe of its own).
BODY_KINDS = ("chaff_cloud", "corner_reflector", "decoy", "infrared_flare",
              "noise_jammer")
# A chaff cloud is a volume of dipoles, not a surface: a radar ray that enters
# it must not be blocked by it, or the cloud would hide the aircraft it was
# meant to confuse. Solid bodies stay opaque.
VOLUMETRIC = ("chaff_cloud",)

# Per-kind defaults.  Every number is an engineering prior that the scene can
# override; none of them is a measured equipment figure.
DEFAULTS = {
    "chaff_cloud": dict(radius_m=2.0, growth_mps=4.0, rcs_m2=100.0, decay_s=1e9,
                        dilution_exponent=2.0, fall_mps=1.5, wind=[0.0, 0.0, 0.0],
                        aspect=[1.0, 1.0, 1.0], mat="chaff_cloud"),
    # 12*pi*a^4/lam^2 is the trihedral corner-reflector closed form; the scene
    # may override it with a measured value instead.  0.3 m is a hand-portable
    # trihedral and it is already 28.8 dBsm at Ku, i.e. it deliberately outshines
    # the airframe it is meant to decoy.  That is what the device is for, so the
    # default is not shrunk to make a scene look tidy: a smaller reflector would
    # be a weaker decoy, not a more honest one.
    "corner_reflector": dict(size_m=0.3, rcs_m2=None, mat="metal_bare"),
    "decoy": dict(size_m=0.3, rcs_m2=0.5, towed_by=None, tether_m=80.0,
                  mat="metal_bare"),
    # radius_m is the LUMINOUS ENVELOPE, not the 0.12 m pellet: a burning
    # flare is seen through its plume, and at the 8 km / 2.2 deg engagements the
    # simulator targets the pellet itself is 0.4 px -- it rendered as nothing at
    # all, which read as "the flare was not simulated".
    "infrared_flare": dict(radius_m=0.35, temp_c=2000.0, decay_s=3.0,
                           rcs_m2=0.02, mat="flare_pellet",
                           # A dispensed round is a hand-sized pellet with a
                           # few metres of luminous trail, NOT a multi-metre
                           # ball with a 20 m tail: at a 17 m wingspan the old
                           # geometry made every decoy as large as the aircraft.
                           # See flare_mesh(); per-round variation is authored
                           # in the scene.
                           trail_m=3.5),
    "noise_jammer": dict(erp_w=200.0, jam_bandwidth_mhz=200.0, mode="barrage",
                         look_at_sensor=True, sidelobe_db=-25.0,
                         off_band_factor=0.05, rcs_m2=0.2,
                         size=[0.4, 0.4, 0.4], mat="painted_metal"),
    "false_target": dict(count=5, power_w=None, power_ratio=1.0,
                         range_offset_m=0.0, range_step_m=100.0,
                         velocity_offset_mps=0.0, velocity_step_mps=0.0,
                         anchor=None),
    "range_gate_pull_off": dict(count=1, rate_mps=80.0, max_offset_m=300.0,
                                power_ratio=1.2, velocity_offset_mps=0.0,
                                anchor=None),
    "velocity_gate_pull_off": dict(count=1, rate_mps=8.0, max_offset_mps=60.0,
                                   power_ratio=1.2, range_offset_m=0.0,
                                   anchor=None),
    "angle_deception": dict(count=1, angle_offset_deg=0.5, power_ratio=1.0,
                            range_offset_m=0.0, anchor=None),
}

_COMMON = ("id", "kind", "pos", "vel", "accel", "life", "band", "visible", "note",
           # Parent-relative placement: an ejected body is positioned from the
           # aircraft it left, not from a fixed point in the world.  The flat
           # names are the legacy spelling; load() maps them onto the link
           # model below (docs/多目标运动仿真调研.md 5.4).
           "parent", "rel_pos", "rel_vel", "release_s", "rel_decay_s",
           "link", "formation", "release", "tether", "trajectory")
_LIFE = ("start_s", "end_s", "fade_s")

# ---------------------------------------------------------------------------
# Link model: who owns an entity's absolute state
# ---------------------------------------------------------------------------
# A scene object's world state is derived (seeker back-solve) or authored (a
# clipped trajectory).  An entity that is placed from an absolute pos+vel*t is
# a *second* state source, and two sources in different reference frames is
# exactly the bug this file used to have: the decoy flew 180 m/s along its own
# line while the seeker dragged the aircraft elsewhere.  So every entity now
# declares how it derives from a parent Track:
#
#   independent  absolute trajectory (authored pos + vel*t + accel) -- no parent
#   formation    parent Track + offset_body in the parent's BODY frame
#   released     attached before release.t_s, then independent of the airframe;
#                relative velocity decays toward the AIR MASS, not the aircraft
#   towed        tether: direction from the lagged parent velocity + droop
#
# docs/多目标运动仿真调研.md 5.2 (schema) and 5.3 (state) are the specification;
# the numbers below come from its section 3.2/3.3.
_LINKS = ("independent", "formation", "released", "towed")

# Kinds that are CARRIED by an airframe rather than dispensed from it: a DRFM
# repeater or a jammer pod rides with the aircraft (formation), it does not
# separate from it.  Everything else that names a parent is a dispensed body.
_CARRIED = ("false_target", "noise_jammer")

# dispensed-body physics: L is the characteristic glide distance
# L = 2m/(rho Cd A) = v_t^2/g; tau is the engineering lag calibrated to match
# it over the first five seconds (tau ~ 3-5 tau_q).  The old rel_decay_s of
# 2.5 s (chaff) / 3.0 s (flare) was an order of magnitude too slow -- it made
# the cloud keep pace with the aircraft, i.e. a formation flight instead of a
# dispense, and it bounded the separation at rel_v*tau (~15 m) where the real
# one grows without limit because the aircraft flies on.  Defaults are lag
# mode so every existing scene keeps a decay (the old default was no decay at
# all: a straight line at the inherited speed).
_DRAG = {
    "chaff_cloud": dict(mode="lag", L_m=10.0, tau_s=0.25),
    "infrared_flare": dict(mode="lag", L_m=60.0, tau_s=1.0),
    "decoy": dict(mode="lag", L_m=60.0, tau_s=1.0),
    "corner_reflector": dict(mode="lag", L_m=40.0, tau_s=0.8),
}
_DEFAULT_DRAG = dict(mode="lag", L_m=40.0, tau_s=1.0)
_FALL_MODES = ("const", "tanh")


def _number(value, name, low=None, high=None):
    """One finite float, refusing bools so 'true' cannot silently become 1."""
    if isinstance(value, bool):
        raise ValueError(name + " must be numeric")
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(name + " must be numeric") from None
    if not np.isfinite(value) or (low is not None and value < low) or (
            high is not None and value > high):
        raise ValueError(name + " is outside its finite range")
    return value


def _vec3(value, name):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(name + " must be a three-element vector")
    return [_number(v, name) for v in value]


def _life(raw, name):
    if raw is None:
        raw = {}
    if not isinstance(raw, dict) or set(raw) - set(_LIFE):
        raise ValueError(name + " accepts only start_s, end_s and fade_s")
    start = _number(raw.get("start_s", 0.0), name + ".start_s", 0.0)
    end = raw.get("end_s")
    end = None if end is None else _number(end, name + ".end_s", start)
    if end is not None and end <= start:
        raise ValueError(name + ".end_s must exceed start_s")
    # fade_s ramps in and out; a hard on/off edge flickers the RD map between
    # frames, which is exactly the artefact a temporal dataset must not learn.
    fade = _number(raw.get("fade_s", 0.0), name + ".fade_s", 0.0)
    return dict(start_s=start, end_s=end, fade_s=fade)


def _block(raw, name):
    """One nested block: a dict or None, and it must be a dict."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(name + " must be an object")
    return raw


def _only(block, name, allowed):
    unknown = set(block) - set(allowed)
    if unknown:
        raise ValueError("unknown %s field(s): %s" % (name, ", ".join(sorted(unknown))))
    return block


def _obj_name(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(name + " must name an object")
    return value


def _own_maneuver(raw, name):
    """Bounded body-frame manoeuvre of a formation member (or None)."""
    if raw is None:
        return None
    block = _only(_block(raw, name), name,
                  {"mode", "amp_m", "hz", "phase", "axis"})
    mode = str(block.get("mode", "weave")).lower()
    if mode in ("none", "off"):
        return None
    if mode not in ("weave", "sin", "lateral"):
        raise ValueError(name + ".mode must be weave or none")
    return dict(mode=mode,
                amp_m=_number(block.get("amp_m", 0.0), name + ".amp_m", 0.0),
                hz=_number(block.get("hz", 0.2), name + ".hz", 0.0),
                phase=_number(block.get("phase", 0.0), name + ".phase"),
                axis=_vec3(block.get("axis", [0.0, 1.0, 0.0]), name + ".axis"))


def _formation_spec(raw, ent, eid):
    name = eid + ".formation"
    block = _only(_block(raw, name), name,
                  {"parent", "offset_body", "mode", "tau_lag_s",
                   "own_maneuver", "max_offset_m"})
    parent = _obj_name(block.get("parent") or ent.get("parent"), name + ".parent")
    offset = _vec3(block.get("offset_body", ent.get("rel_pos") or [0.0, 0.0, 0.0]),
                   name + ".offset_body")
    mode = str(block.get("mode", "lag")).lower()
    if mode not in ("rigid", "lag", "free"):
        raise ValueError(name + ".mode must be rigid, lag or free")
    max_off = block.get("max_offset_m")
    if max_off is not None:
        max_off = _number(max_off, name + ".max_offset_m", 0.0)
    return dict(parent=parent, offset_body=offset, mode=mode,
                tau_lag_s=_number(block.get("tau_lag_s", 0.3),
                                  name + ".tau_lag_s", 0.0, 600.0),
                own_maneuver=_own_maneuver(block.get("own_maneuver"),
                                           name + ".own_maneuver"),
                max_offset_m=max_off)


def _release_spec(raw, ent, eid):
    """A dispensed body: mount + ejection vector in the parent's BODY frame.

    The mount is deliberately in body axes: written in world axes it stays on
    the azimuth it was authored at, so a 30 deg turn throws the cloud from the
    aircraft's belly to its back (docs ... 1.3 c).
    """
    name = eid + ".release"
    kind = ent["kind"]
    base = dict(_DRAG.get(kind) or _DEFAULT_DRAG)
    block = _only(_block(raw, name), name,
                  {"parent", "t_s", "mount_body", "eject_body", "drag", "air"})
    parent = _obj_name(block.get("parent") or ent.get("parent"), name + ".parent")
    mount = _vec3(block.get("mount_body", ent.get("rel_pos") or [0.0, 0.0, 0.0]),
                  name + ".mount_body")
    eject = _vec3(block.get("eject_body", ent.get("rel_vel") or [0.0, 0.0, 0.0]),
                  name + ".eject_body")

    drag = _only(_block(block.get("drag"), name + ".drag"), name + ".drag",
                 {"mode", "L_m", "tau_s"})
    mode = str(drag.get("mode", base["mode"])).lower()
    if mode not in ("quadratic", "lag"):
        raise ValueError(name + ".drag.mode must be quadratic or lag")
    L_m = drag.get("L_m", base["L_m"])
    if L_m is not None:
        L_m = _number(L_m, name + ".drag.L_m", 0.0)
    tau = drag.get("tau_s")
    if tau is None:
        # rel_decay_s is the legacy spelling of the lag time constant; when it
        # is absent the kind default applies (0.25 s chaff, 1.0 s flare).
        leg = ent.get("rel_decay_s")
        tau = base["tau_s"] if leg is None else float(leg)
    tau = _number(tau, name + ".drag.tau_s", 0.0, 1e6)

    air = _only(_block(block.get("air"), name + ".air"), name + ".air",
                {"wind_mps", "fall_mps", "fall_mode"})
    wind = air.get("wind_mps")
    if wind is None:
        wind = ent.get("wind") if kind == "chaff_cloud" else None
    wind = _vec3(wind if wind is not None else [0.0, 0.0, 0.0],
                 name + ".air.wind_mps")
    fall = air.get("fall_mps")
    if fall is None:
        fall = ent.get("fall_mps", 0.0) if kind == "chaff_cloud" else 0.0
    fall_mode = str(air.get("fall_mode", "const")).lower()
    if fall_mode not in _FALL_MODES:
        raise ValueError(name + ".air.fall_mode must be const or tanh")
    return dict(parent=parent,
                t_s=_number(block.get("t_s", ent.get("release_s") or 0.0),
                            name + ".t_s", 0.0, 1e6),
                mount_body=mount, eject_body=eject,
                drag=dict(mode=mode, L_m=L_m, tau_s=tau),
                air=dict(wind_mps=wind,
                         fall_mps=_number(fall, name + ".air.fall_mps"),
                         fall_mode=fall_mode))


def _tether_spec(raw, ent, eid):
    name = eid + ".tether"
    block = _only(_block(raw, name), name,
                  {"parent", "length_m", "tau_s", "droop_deg"})
    parent = _obj_name(block.get("parent") or ent.get("parent")
                       or ent.get("towed_by"), name + ".parent")
    length = block.get("length_m", ent.get("tether_m", 80.0))
    return dict(parent=parent,
                length_m=_number(length, name + ".length_m", 0.0, 1e5),
                tau_s=_number(block.get("tau_s", 0.2), name + ".tau_s", 0.0, 600.0),
                droop_deg=_number(block.get("droop_deg", 3.0),
                                  name + ".droop_deg", -90.0, 90.0))


def _trajectory_spec(raw, ent, eid):
    name = eid + ".trajectory"
    block = _only(_block(raw, name), name, {"mode", "origin", "vel"})
    mode = str(block.get("mode", "line")).lower()
    if mode not in ("line", "straight", "static", "hover", "parked"):
        raise ValueError(name + " must be a straight line for an ecm entity "
                         "(put a curved mover in objects[] instead)")
    origin = _vec3(block.get("origin", ent.get("pos") or [0.0, 0.0, 0.0]),
                   name + ".origin")
    vel = ([0.0, 0.0, 0.0] if mode in ("static", "hover", "parked")
           else _vec3(block.get("vel", ent.get("vel") or [0.0, 0.0, 0.0]),
                      name + ".vel"))
    return dict(origin=origin, vel=vel)


def _resolve_link(ent, item, eid):
    """Pick the link branch and map every legacy field onto it.

    Legacy mapping (docs ... 5.4):
        pos/vel, no parent                -> independent
        parent + rel_pos + rel_vel        -> released, mount_body/eject_body
        rel_decay_s                       -> drag.tau_s (lag)
        towed_by + tether_m               -> towed
        parent on a carried kind          -> formation (a pod/repeater rides)
        parent on a decoy                 -> towed (it is tied to the airframe)
    """
    kind = ent["kind"]
    declared = item.get("link")
    if declared is not None:
        if not isinstance(declared, str) or declared not in _LINKS:
            raise ValueError(eid + ".link must be one of " + ", ".join(_LINKS))
        link = declared
    elif item.get("tether") is not None or ent.get("towed_by") is not None:
        link = "towed"
    elif item.get("release") is not None:
        link = "released"
    elif item.get("formation") is not None:
        link = "formation"
    elif ent.get("parent") is not None:
        if kind in _CARRIED:
            link = "formation"
        elif kind == "decoy":
            link = "towed"
        else:
            link = "released"
    else:
        link = "independent"
    if link == "independent" and ent.get("parent") is not None:
        raise ValueError(eid + ": link=independent cannot carry a parent")
    ent["link"] = link
    if link == "formation":
        ent["formation"] = _formation_spec(item.get("formation"), ent, eid)
    elif link == "released":
        ent["release"] = _release_spec(item.get("release"), ent, eid)
    elif link == "towed":
        ent["tether"] = _tether_spec(item.get("tether"), ent, eid)
    elif item.get("trajectory") is not None:
        traj = _trajectory_spec(item.get("trajectory"), ent, eid)
        ent["pos"], ent["vel"] = traj["origin"], traj["vel"]
        ent["accel"] = [0.0, 0.0, 0.0]
    return ent


def load(spec):
    """Validate and expand an "ecm" block into entity dicts with defaults.

    Returns dict(enabled, entities).  A missing block is the disabled state and
    keeps every existing scene bit-identical.
    """
    if spec is None:
        return dict(enabled=False, entities=[])
    if not isinstance(spec, dict) or set(spec) - {"enabled", "entities", "seed"}:
        raise ValueError("ecm expects enabled, entities and seed")
    if not isinstance(spec.get("enabled", True), bool):
        raise ValueError("ecm.enabled must be boolean")
    items = spec.get("entities") or []
    if not isinstance(items, list) or len(items) > 64:
        raise ValueError("ecm.entities must be a list of at most 64 entities")
    out, seen = [], set()
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError("ecm entity must be an object")
        kind = item.get("kind")
        if kind not in GROUP_OF:
            raise ValueError("unsupported ecm kind: " + str(kind))
        allowed = set(_COMMON) | set(DEFAULTS[kind])
        unknown = set(item) - allowed
        if unknown:
            raise ValueError("unknown %s field(s): %s" % (kind, ", ".join(sorted(unknown))))
        eid = item.get("id", "ecm_%d" % index)
        if not isinstance(eid, str) or not eid.strip() or eid in seen:
            raise ValueError("ecm entity IDs must be unique non-empty strings")
        seen.add(eid)
        ent = dict(DEFAULTS[kind])
        ent.update(kind=kind, group=GROUP_OF[kind], id=eid)
        # a free-text note is carried through untouched: a scene that explains
        # why a cloud is dispensed where it is stays readable, and no sensor
        # reads this key, so it can never change a number
        ent["note"] = str(item.get("note", ""))
        ent["pos"] = _vec3(item.get("pos", [0.0, 0.0, 0.0]), eid + ".pos")
        ent["vel"] = _vec3(item.get("vel", [0.0, 0.0, 0.0]), eid + ".vel")
        ent["accel"] = _vec3(item.get("accel", [0.0, 0.0, 0.0]), eid + ".accel")
        ent["life"] = _life(item.get("life"), eid + ".life")
        ent["visible"] = bool(item.get("visible", True))
        band = item.get("band")
        if band is not None and not isinstance(band, str):
            raise ValueError(eid + ".band must be a band name such as 'Ku'")
        ent["band"] = band
        # Parent-relative placement is copied explicitly: the entity is rebuilt
        # from DEFAULTS, so anything not listed here is silently dropped -- which
        # is exactly how the first parenting attempt had no effect at all.
        for key in ("parent", "rel_pos", "rel_vel", "release_s", "rel_decay_s"):
            if item.get(key) is not None:
                ent[key] = item[key]
        for key in DEFAULTS[kind]:
            if key in item:
                ent[key] = item[key]
        if kind == "chaff_cloud":
            ent["radius_m"] = _number(ent["radius_m"], eid + ".radius_m", 1e-3)
            ent["growth_mps"] = _number(ent["growth_mps"], eid + ".growth_mps", 0.0)
            ent["rcs_m2"] = _number(ent["rcs_m2"], eid + ".rcs_m2", 0.0)
            ent["decay_s"] = _number(ent["decay_s"], eid + ".decay_s", 1e-3)
            ent["dilution_exponent"] = _number(ent["dilution_exponent"],
                                               eid + ".dilution_exponent", 0.0, 4.0)
            ent["fall_mps"] = _number(ent["fall_mps"], eid + ".fall_mps")
            ent["wind"] = _vec3(ent["wind"], eid + ".wind")
            ent["aspect"] = _vec3(ent["aspect"], eid + ".aspect")
            if min(ent["aspect"]) <= 0.0:
                raise ValueError(eid + ".aspect must be positive")
        elif kind == "corner_reflector":
            ent["size_m"] = _number(ent["size_m"], eid + ".size_m", 1e-3)
            if ent["rcs_m2"] is not None:
                ent["rcs_m2"] = _number(ent["rcs_m2"], eid + ".rcs_m2", 0.0)
        elif kind == "decoy":
            ent["size_m"] = _number(ent["size_m"], eid + ".size_m", 1e-3)
            ent["rcs_m2"] = _number(ent["rcs_m2"], eid + ".rcs_m2", 0.0)
            ent["tether_m"] = _number(ent["tether_m"], eid + ".tether_m", 0.0)
            if ent["towed_by"] is not None and not isinstance(ent["towed_by"], str):
                raise ValueError(eid + ".towed_by must be an object name")
        elif kind == "infrared_flare":
            ent["radius_m"] = _number(ent["radius_m"], eid + ".radius_m", 1e-3)
            ent["temp_c"] = _number(ent["temp_c"], eid + ".temp_c", -273.0, 3000.0)
            ent["decay_s"] = _number(ent["decay_s"], eid + ".decay_s", 1e-3)
            ent["rcs_m2"] = _number(ent["rcs_m2"], eid + ".rcs_m2", 0.0)
        elif kind == "noise_jammer":
            ent["erp_w"] = _number(ent["erp_w"], eid + ".erp_w", 0.0)
            ent["jam_bandwidth_mhz"] = _number(ent["jam_bandwidth_mhz"],
                                               eid + ".jam_bandwidth_mhz", 1e-3)
            if ent["mode"] not in ("barrage", "spot"):
                raise ValueError(eid + ".mode must be barrage or spot")
            ent["sidelobe_db"] = _number(ent["sidelobe_db"], eid + ".sidelobe_db")
            ent["off_band_factor"] = _number(ent["off_band_factor"],
                                             eid + ".off_band_factor", 0.0, 1.0)
            ent["look_at_sensor"] = bool(ent["look_at_sensor"])
            ent["rcs_m2"] = _number(ent["rcs_m2"], eid + ".rcs_m2", 0.0)
            ent["size"] = _vec3(ent["size"], eid + ".size")
        else:
            # every false-target generator needs the same four knobs
            ent["count"] = int(_number(ent["count"], eid + ".count", 1.0, 64.0))
            ent["power_ratio"] = _number(ent["power_ratio"], eid + ".power_ratio", 0.0)
            # Only a false_target ships power_w in DEFAULTS; the three gate/
            # angle deception kinds do not, and indexing the key here made
            # every one of them raise KeyError on load -- i.e. three of the five
            # active emitters could not be selected at all.
            if ent.get("power_w") is not None:
                ent["power_w"] = _number(ent["power_w"], eid + ".power_w", 0.0)
            ent["range_offset_m"] = _number(ent.get("range_offset_m", 0.0),
                                           eid + ".range_offset_m")
            ent["velocity_offset_mps"] = _number(ent.get("velocity_offset_mps", 0.0),
                                                 eid + ".velocity_offset_mps")
            if ent["anchor"] is not None and not isinstance(ent["anchor"], str):
                raise ValueError(eid + ".anchor must be an object name")
            if kind == "false_target":
                ent["range_step_m"] = _number(ent["range_step_m"], eid + ".range_step_m")
                ent["velocity_step_mps"] = _number(ent["velocity_step_mps"],
                                                   eid + ".velocity_step_mps")
            elif kind == "range_gate_pull_off":
                ent["rate_mps"] = _number(ent["rate_mps"], eid + ".rate_mps")
                ent["max_offset_m"] = _number(ent["max_offset_m"], eid + ".max_offset_m", 0.0)
            elif kind == "velocity_gate_pull_off":
                ent["rate_mps"] = _number(ent["rate_mps"], eid + ".rate_mps")
                ent["max_offset_mps"] = _number(ent["max_offset_mps"],
                                                eid + ".max_offset_mps", 0.0)
            else:
                # angle deception walks the indicated angle, not the range
                ent["angle_offset_deg"] = _number(ent["angle_offset_deg"],
                                                  eid + ".angle_offset_deg")
        if ent.get("parent") is not None and not isinstance(ent["parent"], str):
            raise ValueError(eid + ".parent must be an object name")
        for key in ("rel_pos", "rel_vel"):
            if ent.get(key) is not None:
                ent[key] = _vec3(ent[key], eid + "." + key)
        if ent.get("release_s") is not None:
            ent["release_s"] = _number(ent["release_s"], eid + ".release_s", 0.0)
        if ent.get("rel_decay_s") is not None:
            ent["rel_decay_s"] = _number(ent["rel_decay_s"], eid + ".rel_decay_s",
                                         0.0, 1e6)
        if "mat" in ent and not isinstance(ent["mat"], str):
            raise ValueError(eid + ".mat must be a material name")
        # Resolve the motion model last: every legacy alias above has already
        # been normalised, so the link branch sees one spelling of each field.
        _resolve_link(ent, item, eid)
        out.append(ent)
    return dict(enabled=bool(spec.get("enabled", True)), entities=out)


def preset(name):
    """Small explicit presets, used by the tests and the demo scene."""
    if name == "off":
        return dict(enabled=False, entities=[])
    if name not in ("noise", "false_targets", "chaff", "mixed"):
        raise ValueError("ecm preset must be off, noise, false_targets, chaff or mixed")
    entities = []
    if name in ("noise", "mixed"):
        entities.append(dict(id="jammer", kind="noise_jammer", pos=[600.0, -900.0, 2800.0],
                             erp_w=200.0, jam_bandwidth_mhz=200.0, mode="barrage",
                             band="Ku", life=dict(start_s=0.0)))
    if name in ("false_targets", "mixed"):
        entities.append(dict(id="false_targets", kind="false_target", pos=[0.0, -200.0, 2900.0],
                             count=5, range_step_m=120.0, power_ratio=1.2,
                             life=dict(start_s=0.0)))
    if name in ("chaff", "mixed"):
        entities.append(dict(id="chaff", kind="chaff_cloud", pos=[-100.0, -150.0, 2950.0],
                             vel=[180.0, 0.0, -2.0], radius_m=6.0, growth_mps=6.0,
                             rcs_m2=200.0, decay_s=4.0, band="Ku",
                             life=dict(start_s=0.1)))
    return dict(enabled=True, entities=entities)


# ---------------------------------------------------------------- life cycle
def _weight(ent, t):
    """Smooth 0..1 gate over the entity's life, zero outside it."""
    life = ent["life"]
    start, end, fade = life["start_s"], life["end_s"], life["fade_s"]
    if t < start or (end is not None and t >= end):
        return 0.0
    if not fade:
        return 1.0
    # A ramp needs a "before" to ramp from. For an entity that is already there
    # when the clip opens (start_s = 0) a fade-in would make frame 0 empty, so
    # the scene would silently lose its jammer for the one frame an author is
    # most likely to look at. The ramp therefore only truncates a window that
    # actually begins later than the clip; the tail ramp is unaffected.
    rise = (t - start) / fade if start > 0.0 else 1.0
    u = min(1.0, rise, (end - t) / fade if end is not None else 1.0)
    return u * u * (3.0 - 2.0 * u)


def chaff_rcs(ent, t):
    """Cross-section of an expanding chaff cloud, in square metres.

    The cloud holds a fixed number of dipoles, so its *total* scattering area
    is fixed; what falls with time is the density of the part a resolution cell
    sees.  Writing RCS = RCS0 * (r0/r)^k with k=2 (default) is the resolved-cell
    dilution law (k=3 would be a fully homogeneous fill); the exponential term
    is dipole settling and clumping, and the life weight switches the cloud on
    and off.  This is why a chaff cloud is not a static paint splat: its echo
    has to shrink as it blooms.
    """
    r0 = float(ent["radius_m"])
    grow = float(ent["growth_mps"]) * max(0.0, float(t) - float(ent["life"]["start_s"]))
    r = max(r0 + grow, 1e-6)
    dilution = (r0 / r) ** float(ent["dilution_exponent"])
    age = max(0.0, float(t) - float(ent["life"]["start_s"]))
    decay = float(np.exp(-age / float(ent["decay_s"])))
    return float(ent["rcs_m2"]) * _weight(ent, t) * dilution * decay


def radius_at(ent, t):
    if ent["kind"] == "chaff_cloud":
        return float(ent["radius_m"]) + float(ent["growth_mps"]) * max(
            0.0, float(t) - float(ent["life"]["start_s"]))
    if ent["kind"] == "infrared_flare":
        return float(ent["radius_m"])
    if ent["kind"] == "decoy":
        return 0.5 * float(ent["size_m"])
    if ent["kind"] == "corner_reflector":
        return 0.5 * float(ent["size_m"])
    if ent["kind"] == "noise_jammer":
        return 0.5 * float(max(ent["size"]))
    # a repeater is a rack of electronics with no extent of its own
    return 0.0


# ------------------------------------------------------------------- state
# One analytic state function per entity, the same way a scene object is a
# motion.Track.  The "link" field decides which rule runs:
#
#   independent  absolute pos + vel*t (+ accel) -- unchanged
#   formation    parent Track + offset_body in the parent's BODY frame
#   released     strictly attached before release.t_s, then air-mass relaxation
#   towed        tether along the lagged parent velocity plus droop
#
# Two mistakes this replaces, both from docs/多目标运动仿真调研.md 1.3:
#   * v = host_vel + rel_v*decay keeps the aircraft's speed forever.  Chaff then
#     flies in formation with it (wrong on the Doppler axis: a pulse-Doppler set
#     cannot separate them) and its relative displacement is bounded by
#     rel_v*tau (~15 m) when the real one grows without limit, because the
#     aircraft flies on and the cloud stops;
#   * rel_pos/rel_vel in world axes keep the mount on the world azimuth they
#     were written at, so a 30 deg turn throws the cloud from the belly to the
#     back.
# The fix is the parent Track (the planner owns it) and body-frame mount and
# ejection vectors, rotated by the same frame that draws the model.

_G = motion.G
_STATE_H = 1e-3


def _vec(value, default=(0.0, 0.0, 0.0)):
    if value is None:
        value = default
    return np.asarray(value, float).reshape(3)


def _unit(v, fallback=(1.0, 0.0, 0.0)):
    v = np.asarray(v, float).reshape(3)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else np.asarray(fallback, float)


def _yaw_of(v):
    v = np.asarray(v, float)
    if float(np.hypot(v[0], v[1])) < 1e-9:
        return 0.0
    return float(np.degrees(np.arctan2(v[1], v[0])))


def tracks_of(cfg):
    """The planner's Track set for this frame, when the config carries one.

    plan.ClipPlan hands its tracks to the frame config as an ATTRIBUTE, not as
    a mapping entry: dataset.py deep-copies the frame config with
    json.loads(json.dumps(...)) for multi-band IR, and a Track is a live
    function of time, not JSON.  A manual caller may pass the same mapping
    under "_ecm_tracks".
    """
    if cfg is None:
        return None
    tracks = getattr(cfg, "tracks", None)
    if isinstance(tracks, dict) and tracks:
        return tracks
    if isinstance(cfg, dict):
        tracks = cfg.get("_ecm_tracks")
        if isinstance(tracks, dict) and tracks:
            return tracks
    return None


def anchors_of(world):
    """Name -> pose of the objects already in the world (compatibility entry).

    The Track set is preferred whenever one is available; this exists because
    scene.py (owned elsewhere) builds the World and hands it to attach().  The
    pose now carries the body frame as well, so a mount point written in body
    axes still turns with the aircraft on this path -- position and velocity
    alone were what let the fallback treat the parent as an attitude-less point.
    """
    objs = world.get("objs", []) if isinstance(world, dict) else getattr(world, "objs", [])
    out = {}
    for o in objs or []:
        try:
            out[o["name"]] = dict(
                pos=np.asarray(o["pos"], float),
                vel=np.asarray(o.get("vel", [0.0, 0.0, 0.0]), float),
                yaw=o.get("yaw"),
                pitch=float(o.get("pitch", 0.0) or 0.0),
                roll=float(o.get("roll", 0.0) or 0.0))
        except (KeyError, TypeError):
            continue
    return out


def _parent_name(ent):
    for key in ("formation", "release", "tether"):
        block = ent.get(key)
        if isinstance(block, dict) and block.get("parent"):
            return block["parent"]
    return ent.get("parent") or ent.get("towed_by")


def _parent_track(ent, tracks, anchors, t):
    """The parent Track this entity is written against, or None if unknown."""
    parent = _parent_name(ent)
    if not parent:
        return None
    if tracks:
        hit = tracks.get(parent)
        if hit is not None:
            return hit
    if anchors:
        a = anchors.get(parent)
        if a is None:
            return None
        if isinstance(a, motion.Track):
            return a
        if isinstance(a, dict):
            return motion.AnchorTrack(_vec(a.get("pos")), _vec(a.get("vel")),
                                      yaw=a.get("yaw"),
                                      pitch=float(a.get("pitch", 0.0) or 0.0),
                                      roll=float(a.get("roll", 0.0) or 0.0), t0=t)
        try:
            pos, vel = a
        except (TypeError, ValueError):
            return None
        return motion.AnchorTrack(pos, vel, t0=t)
    return None


def _air_state(release, u):
    """(air velocity, air displacement since release) at age u >= 0.

    The air mass is wind plus a vertical fall term, and the displacement is its
    analytic integral: position and velocity have to be f and f' of the same
    expression, or the derivative contract breaks on exactly the entities whose
    Doppler the radar reads (docs ... 3.4).  fall_mode="tanh" is the quadratic
    drag terminal vertical speed -v_t tanh(g u / v_t) with its closed-form
    integral; "const" is the straight line.
    """
    air = release["air"]
    wind = _vec(air["wind_mps"])
    fall = float(air["fall_mps"])
    u = float(u)
    if str(air.get("fall_mode", "const")) == "tanh" and fall > 0.0:
        vt = fall
        a = _G * u / vt
        # ln(cosh(a)) computed without overflowing cosh for large a
        lncosh = abs(a) + float(np.log1p(np.exp(-2.0 * abs(a)))) - np.log(2.0)
        v = wind + np.array([0.0, 0.0, -vt * float(np.tanh(a))])
        disp = wind * u + np.array([0.0, 0.0, -(vt * vt / _G) * lncosh])
        return v, disp
    v = wind + np.array([0.0, 0.0, -fall])
    return v, wind * u + np.array([0.0, 0.0, -fall * u])


def _tilt_down(u, deg):
    """Rotate a unit vector downward by deg about the horizontal right axis."""
    deg = float(deg)
    if abs(deg) < 1e-9:
        return u
    axis = np.cross(u, np.array([0.0, 0.0, 1.0]))
    n = float(np.linalg.norm(axis))
    axis = axis / n if n > 1e-9 else np.array([0.0, 1.0, 0.0])
    ang = -np.radians(deg)
    c, s = float(np.cos(ang)), float(np.sin(ang))
    return u * c + np.cross(axis, u) * s + axis * float(np.dot(axis, u)) * (1.0 - c)


def _drag_state(release, vrel0, u):
    """(time-integral s, decay fraction) of the drag law at age u >= 0.

    Both modes end up in the same two lines:
        p = pr + disp + vrel0 * s         v = v_air + vrel0 * frac
    so s is a TIME (seconds), not a distance.  The quadratic law's displacement
    is L ln(1 + u/tq) along the initial relative velocity, and since
    L = |vrel0| tq that is exactly vrel0 * (tq ln(1 + u/tq)) -- docs 3.2's
    x(t) = L ln(1 + t/tau_q).  Writing vrel0 * L*ln(...) instead (the
    pseudocode's literal transpose) is dimensionally wrong and overstates the
    glide by |vrel0|.
    """
    drag = release["drag"]
    u = float(u)
    if str(drag["mode"]) == "quadratic":
        L = float(drag["L_m"] or 0.0)
        n0 = float(np.linalg.norm(vrel0))
        if L <= 1e-9 or n0 <= 1e-12:
            # L -> 0: the body stops relative to the air immediately
            return 0.0, 0.0
        tq = L / n0
        return tq * float(np.log1p(u / tq)), 1.0 / (1.0 + u / tq)
    tau = float(drag["tau_s"] or 0.0)
    if tau <= 1e-12:
        # tau -> 0: same degenerate limit
        return 0.0, 0.0
    frac = float(np.exp(-u / tau))
    return tau * (1.0 - frac), frac


def _absolute_air(ent):
    """Air mass of an un-parented body, from the legacy wind/fall fields."""
    tau = ent.get("rel_decay_s")
    tau = (_DRAG["chaff_cloud"]["tau_s"] if tau is None else float(tau))
    return dict(drag=dict(mode="lag", tau_s=tau),
                air=dict(wind_mps=_vec(ent.get("wind") or [0.0, 0.0, 0.0]),
                         fall_mps=float(ent.get("fall_mps", 0.0)),
                         fall_mode="const"))


def _absolute_chaff(ent, t):
    """A chaff cloud with no named parent is STILL a dispensed body.

    It has no aircraft to be mounted on, so its authored pos/vel is the
    dispense state at life.start_s; from there the same air-mass relaxation
    runs.  Keeping the constant speed + linear drift instead (the old model)
    double-counts the air, keeps the cloud flying in formation with the
    aircraft and, on the Doppler axis, makes it indistinguishable from it --
    docs/多目标运动仿真调研.md 1.2 and 3.3 retire exactly that model.
    """
    start = float(ent["life"]["start_s"])
    pos0 = _vec(ent["pos"])
    vel0 = _vec(ent["vel"])
    acc0 = _vec(ent["accel"])
    if t < start:
        # outside the life window anyway; keep the function defined and
        # continuous with the dispense state at t = start
        return (pos0 + vel0 * t + 0.5 * acc0 * (t * t),
                vel0 + acc0 * t)
    rel = _absolute_air(ent)
    v_at_start = vel0 + acc0 * start
    pr = pos0 + vel0 * start + 0.5 * acc0 * (start * start)
    v_air0, _ = _air_state(rel, 0.0)
    vrel0 = v_at_start - v_air0
    u = t - start
    s, frac = _drag_state(rel, vrel0, u)
    v_air, disp = _air_state(rel, u)
    return pr + disp + vrel0 * s, v_air + vrel0 * frac


def _independent(ent, t):
    """Absolute trajectory: pos + vel*t (+ 0.5 a t^2), as before.

    The one exception is a chaff cloud: it is a dispensed body with no named
    parent, so its authored state is the dispense state and it relaxes toward
    the air mass from life.start_s (see _absolute_chaff).
    """
    t = float(t)
    if ent["kind"] == "chaff_cloud":
        return _absolute_chaff(ent, t)
    p = _vec(ent["pos"]) + _vec(ent["vel"]) * t + 0.5 * _vec(ent["accel"]) * (t * t)
    v = _vec(ent["vel"]) + _vec(ent["accel"]) * t
    return p, v


def _released_state(ent, P, t):
    """Dispensed body: attached strictly, then relaxed toward the air mass."""
    r = ent["release"]
    tr = float(r["t_s"])
    mount = _vec(r["mount_body"])
    eject = _vec(r["eject_body"])
    if t < tr:
        # before release the body is bolted to the pylon: same position and
        # velocity as a formation member with an offset of mount_body
        R = np.asarray(P.frame(t), float)
        m = R @ mount
        p = np.asarray(P.pos(t), float) + m
        w = motion.angular_velocity(P, t, 0.0)
        v = np.asarray(P.vel(t), float) + np.cross(w, m)
        y = P.yaw(t)
        return p, v, (_yaw_of(v) if y is None else float(y))

    # release state: constant with respect to t, so the ejection point never
    # slides along the aircraft as the clip advances
    Rr = np.asarray(P.frame(tr), float)
    pr = np.asarray(P.pos(tr), float) + Rr @ mount
    v_air0, _ = _air_state(r, 0.0)
    vrel0 = np.asarray(P.vel(tr), float) + Rr @ eject - v_air0
    u = float(t) - tr
    s, frac = _drag_state(r, vrel0, u)
    v_air, disp = _air_state(r, u)
    p = pr + disp + vrel0 * s
    v = v_air + vrel0 * frac
    return p, v, _yaw_of(v)


def entity_state(ent, t, tracks=None, anchors=None, accel=False):
    """(pos, vel, accel?, yaw) of one entity at clip time t.

    Every branch is an analytic function of t.  The parent is the planner's
    Track when one is available; the world anchors are the compatibility
    fallback, and when neither names the parent the entity degrades to its
    absolute pos/vel -- a scene with a typo in "parent" then still renders
    instead of vanishing.
    """
    t = float(t)
    link = str(ent.get("link") or "independent")
    if link == "independent":
        p, v = _independent(ent, t)
        yaw = _yaw_of(v)
    else:
        P = _parent_track(ent, tracks, anchors, t)
        if P is None:
            p, v = _independent(ent, t)
            yaw = _yaw_of(v)
        elif link == "formation":
            f = ent["formation"]
            ft = motion.FormationTrack(
                P, f["offset_body"], mode=f["mode"],
                tau_lag_s=f["tau_lag_s"], own_maneuver=f.get("own_maneuver"),
                max_offset_m=f.get("max_offset_m"))
            p, v = ft.pos(t), ft.vel(t)
            y = P.yaw(t)
            yaw = _yaw_of(v) if y is None else float(y)
        elif link == "released":
            p, v, yaw = _released_state(ent, P, t)
        elif link == "towed":
            tw = ent["tether"]
            length = float(tw["length_m"])
            tau = float(tw["tau_s"])
            # The tether points backwards along the parent velocity AS IT WAS
            # tau seconds ago, then droops: a first-order lag is C-infinity and
            # stops the decoy teleporting the instant the aircraft turns.
            u_hat = -_unit(P.vel(t - tau))
            u_hat = _tilt_down(u_hat, tw["droop_deg"])
            p = np.asarray(P.pos(t), float) + u_hat * length
            w = motion.angular_velocity(P, t, tau)
            v = np.asarray(P.vel(t), float) + np.cross(w, u_hat * length)
            y = P.yaw(t)
            yaw = _yaw_of(v) if y is None else float(y)
        else:
            raise ValueError("unsupported ecm link: " + str(link))
    out = dict(pos=np.asarray(p, float), vel=np.asarray(v, float),
               yaw=float(yaw), link=link)
    if accel:
        # derivative of the analytic velocity, never of the position
        a0 = entity_state(ent, t - _STATE_H, tracks=tracks, anchors=anchors)["vel"]
        a1 = entity_state(ent, t + _STATE_H, tracks=tracks, anchors=anchors)["vel"]
        out["accel"] = (np.asarray(a1, float) - np.asarray(a0, float)) / (2.0 * _STATE_H)
    return out


def entities_at(spec, t, anchors=None, tracks=None):
    """Entity states at time t: pose, velocity, extent, RCS, self-emission."""
    t = float(t)
    out = []
    for ent in spec["entities"]:
        w = _weight(ent, t)
        if w <= 0.0:
            continue
        st = dict(ent)
        st["t"] = t
        st["weight"] = w
        state = entity_state(ent, t, tracks=tracks, anchors=anchors)
        st["pos"] = state["pos"]
        st["vel"] = state["vel"]
        st["yaw"] = state["yaw"]
        st["radius_m"] = radius_at(ent, t)
        # a repeater has no skin of its own: it is a signal source, not a body,
        # so it carries no cross-section and reports no extent
        if ent["kind"] == "chaff_cloud":
            st["rcs_m2"] = chaff_rcs(ent, t)
        elif ent["kind"] == "corner_reflector":
            st["rcs_m2"] = None          # solved from size and wavelength
        elif "rcs_m2" in ent:
            st["rcs_m2"] = float(ent["rcs_m2"]) * w
        else:
            st["rcs_m2"] = 0.0
        if ent["kind"] == "infrared_flare":
            # the flare cools as it burns; the rasteriser reads temp_c, so this
            # is the self-emission decay the IR frame actually shows
            age = max(0.0, t - float(ent["life"]["start_s"]))
            st["temp_c"] = -273.15 + (float(ent["temp_c"]) + 273.15) * float(
                np.exp(-age / float(ent["decay_s"])))
        elif ent["kind"] == "chaff_cloud":
            # aluminium foil strips are close to air temperature; they return
            # radar energy but are not an infrared source
            st["temp_c"] = None
        else:
            st["temp_c"] = None
        out.append(st)
    return out


# ----------------------------------------------------------------- geometry
def _subdivide(v, f):
    # v is rebuilt inside the nested helper, so it needs nonlocal: without it
    # the assignment makes v local to midpoint and every call raises
    # UnboundLocalError. The midpoint cache also has to be keyed on the
    # unordered edge so the two triangles sharing an edge reuse one vertex,
    # otherwise the surface cracks open and the rasteriser shows seams.
    cache = {}
    v = np.asarray(v, float)

    def midpoint(a, b):
        nonlocal v
        key = (min(a, b), max(a, b))
        if key not in cache:
            m = v[a] + v[b]
            m = m / np.linalg.norm(m)
            cache[key] = len(v)
            v = np.vstack([v, m])
        return cache[key]

    new = []
    for a, b, c in f:
        ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
        new += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
    return v, np.asarray(new, np.int32)


def icosphere(radius=1.0, level=1):
    """Subdivided icosahedron: no pole singularity, no degenerate triangles."""
    t = (1.0 + 5.0 ** 0.5) / 2.0
    v = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
                  [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
                  [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], float)
    f = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
                  [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
                  [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
                  [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]], np.int32)
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    for _ in range(max(0, int(level))):
        v, f = _subdivide(v, f)
    return v * float(radius), f


def flare_mesh(size=(8.0, 1.0, 0.25)):
    """One dispensed infrared decoy: a bright compact head + a long thin trail.

    Reference photos (aircraft dispensing a fan of flares) show a small
    near-white point with a narrow, tapered smoke/flare streak behind it, NOT a
    sphere. The old icosphere read as a golf ball in the optical frame and as a
    circular hot blob in LWIR. This mesh is shared by the numpy World and the
    Blender renderer so both modalities see exactly the same silhouette.

    Local frame: +X is the direction of flight, the head sits at the origin and
    the trail extends towards -X. ``size`` = (trail length, head radius,
    trail radius) in metres.
    """
    try:
        length = max(float(size[0]), 1e-3)
        head_r = max(float(size[1]), 1e-3)
        tail_r = max(float(size[2]) if len(size) > 2 else head_r * 0.25, 1e-3)
    except (TypeError, ValueError, IndexError):
        length, head_r, tail_r = 8.0, 1.0, 0.25

    # Head: an octahedron. Six vertices, eight facets -- cheap for the BVH and
    # still reads as a point source with a slight anisotropic glint.
    r = head_r
    hv = np.array([[r, 0, 0], [-r, 0, 0], [0, r, 0],
                   [0, -r, 0], [0, 0, r], [0, 0, -r]], float)
    hf = np.array([[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
                   [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]], np.int32)

    # Trail: a tapered diamond-section prism, from just behind the head to
    # -length. The trailing ring is much thinner, so the streak fades in width
    # the way a real flare trail does.
    x0, x1 = -head_r * 0.8, -length
    # Keep the far end reasonably wide: at 2.5 km a 0.03 m tip is sub-pixel and
    # the streak disappears entirely, leaving just the head (which reads as a
    # row of dots). A visible tapered trail is the whole point of this shape.
    a0, a1 = tail_r, tail_r * 0.55
    tv = np.array([
        [x0, a0, 0], [x0, 0, a0], [x0, -a0, 0], [x0, 0, -a0],
        [x1, a1, 0], [x1, 0, a1], [x1, -a1, 0], [x1, 0, -a1],
    ], float)
    tf = []
    for i in range(4):
        j = (i + 1) % 4
        tf += [[i, j, 4 + j], [i, 4 + j, 4 + i]]
    tf += [[4, 5, 6], [4, 6, 7]]          # close the far end
    tf = np.asarray(tf, np.int32)

    v = np.vstack([hv, tv])
    f = np.vstack([hf, tf + len(hv)])
    # scene.prepare_mesh recentres x/y and puts the base at z=0. Pre-centre the
    # mesh here so that function becomes a no-op: otherwise the Blender frame
    # would show the decoy 3.5 m away from where the IR/radar world put it
    # (about 28 px at 2.5 km), which would silently break registration.
    v[:, 0] += (length - head_r) * 0.5
    v[:, 2] += head_r
    return Mesh(v, f, name="flare_decoy", mat="flare_pellet")


def flare_size(ent):
    """(trail length, head radius, trail radius) shared by BOTH render paths.

    Keeping this in one place is what guarantees the numpy World and the
    Blender frame draw the identical decoy; a divergent constant here would
    show up as a cross-modal registration error that no label check can see.
    """
    r = max(float(ent.get("radius_m", 0.35)), 1e-3)
    # Scale the trail to the pellet: its radius must stay BELOW the head's,
    # otherwise a "flare" reads as a fat cone larger than the aircraft.
    return (max(float(ent.get("trail_m", 3.5)), 0.5), r,
            max(0.04, r * 0.85))


def trihedral(size=0.3):
    """Three mutually orthogonal plates: the closed-form corner reflector."""
    a = float(size)
    v = np.array([[0, 0, 0], [a, 0, 0], [0, a, 0], [0, 0, a]], float)
    f = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1]], np.int32)
    return v - a / 3.0, f


def _world_mesh(ent):
    """Triangles of one entity at its own pose, in world coordinates."""
    kind = ent["kind"]
    yaw = float(ent.get("yaw", 0.0))
    if kind == "chaff_cloud":
        r = max(ent["radius_m"], 1e-3)
        # Level 2: at 2 m radius and a few kilometres the level-1 twenty-face
        # ball reads as a visible golf ball rather than a cloud.
        v, f = icosphere(1.0, 2)
        ax = np.asarray(ent["aspect"], float)
        v = v * (np.array([r, r, r]) * ax)
        m = Mesh(v, f, name=ent["id"], mat=ent["mat"])
    elif kind == "infrared_flare":
        # A dispensed flare is a bright head with a tapered streak, not a ball.
        m = flare_mesh(flare_size(ent))
        m.name, m.mat = ent["id"], ent["mat"]
    elif kind == "corner_reflector":
        v, f = trihedral(ent["size_m"])
        m = Mesh(v, f, name=ent["id"], mat=ent["mat"])
    else:
        # a jammer pod or a decoy body: a small rigid box is enough for
        # occlusion and for the lidar return
        size = ent.get("size") if kind == "noise_jammer" else [ent["size_m"]] * 3
        m = Mesh(*[np.asarray(x) for x in (box(size).v, box(size).f)],
                 name=ent["id"], mat=ent["mat"])
    R = rot_z(yaw) @ rot_y(-float(ent.get("pitch", 0.0))) @ rot_x(float(ent.get("roll", 0.0)))
    return m.transformed(R=R, t=np.asarray(ent["pos"], float))


def render_objects(cfg, t=0.0, tracks=None):
    """ECM bodies as object specs the optical renderer can build.

    The Blender pass constructs its own scene from cfg["objects"] and never sees
    the numpy World, so a flare or a chaff cloud existed for radar, IR and lidar
    and was simply absent from the photographs. A burning flare is one of the
    brightest things in a visible-band scene, so that omission read as "the
    decoy was not simulated".

    Callers that know the clip time should pass it (and, when they have one,
    the planner's track set): dataset.py still builds this once per run at
    t=0 without the World, so the per-frame pose stream for Blender comes from
    the manifest and this only has to place the bodies on frame 0.
    """
    spec = load(cfg.get("ecm"))
    if not spec["enabled"]:
        return []
    tracks = tracks if tracks is not None else tracks_of(cfg)
    out = []
    for ent in entities_at(spec, t, tracks=tracks):
        if not ent.get("visible", True) or ent["kind"] not in BODY_KINDS:
            continue
        kind = ent["kind"]
        if kind == "chaff_cloud":
            size = [2.0 * float(ent["radius_m"])] * 3
            shape = "sphere"
        elif kind == "infrared_flare":
            # Same mesh as _world_mesh -> optical, IR, radar and lidar agree on
            # the decoy's silhouette. scene._reshape("flare") calls flare_mesh.
            size = list(flare_size(ent))
            shape = "flare"
        elif kind == "corner_reflector":
            size = [float(ent["size_m"])] * 3
            shape = "box"
        else:
            size = [float(v) for v in (ent.get("size") or [ent.get("size_m", 0.4)] * 3)]
            shape = "box"
        item = dict(name=ent["id"], cls="ecm", shape=shape,
                    size=[round(v, 4) for v in size],
                    pos=[round(float(v), 3) for v in ent["pos"]],
                    rot=float(ent.get("yaw", 0.0)),
                    pitch=float(ent.get("pitch", 0.0)),
                    roll=float(ent.get("roll", 0.0)),
                    mat=ent.get("mat", "metal_bare"), label=False,
                    ecm=True)
        if ent.get("temp_c") is not None:
            item["temp_c"] = float(ent["temp_c"])
        out.append(item)
    return out


def bodies(cfg, t, anchors=None, tracks=None):
    """[(state, world-space Mesh)] for every entity that exists as geometry."""
    spec = load(cfg.get("ecm"))
    if not spec["enabled"]:
        return []
    out = []
    tracks = tracks if tracks is not None else tracks_of(cfg)
    for ent in entities_at(spec, t, anchors=anchors, tracks=tracks):
        if not ent.get("visible", True) or ent["kind"] not in BODY_KINDS:
            continue
        out.append((ent, _world_mesh(ent)))
    return out


# ------------------------------------------------------------------- radar
def frame_time(cfg, t=0.0):
    """Scene time of the frame being simulated.

    radar.simulate is called per frame with a config, not with a clock. dataset
    stamps the clip time into "_ecm_t" before it builds the world, so entity
    motion, chaff bloom and gate walk all advance at the real frame rate;
    callers that drive a model directly (tests, tools) either set the key
    themselves or get t=0.
    """
    for key in ("_ecm_t", "t", "_t_s"):
        if key in cfg:
            try:
                return float(cfg[key])
            except (TypeError, ValueError):
                break
    return float(t)


def sigma_of(ent, lam):
    """Radar cross-section an entity presents, in square metres."""
    if ent["kind"] == "corner_reflector" and ent.get("rcs_m2") is None:
        # closed form for a trihedral corner reflector: the reason a 30 cm
        # corner can out-shine an airframe in one band and vanish in another
        return 12.0 * np.pi * float(ent["size_m"]) ** 4 / lam ** 2
    return float(ent["rcs_m2"] or 0.0)


def _band_factor(ent, radar_band):
    """Off-band penalty: dipoles and flares are cut for one band only."""
    want = ent.get("band")
    if not want or not radar_band or want == radar_band:
        return 1.0
    return float(ent.get("off_band_factor", 0.05))


def _phase(ctx, key):
    seed = int(ctx.get("frame_seed", 0)) & 0xFFFFFFFF
    salt = zlib.crc32(str(key).encode("utf-8"))
    return float(np.random.default_rng(
        np.random.SeedSequence([seed, salt])).uniform(0.0, 2 * np.pi))


def _loops(dirs, rc):
    """Beam index groups, one per receiver loop.

    A wide scan is several receiver channels, not one huge array: the angle
    error inside a loop has to stay inside the Doppler band. Splitting at the
    Doppler ambiguity strips is therefore also the natural granularity for the
    antenna pattern, and it keeps every pattern computation loop-sized.
    """
    n_az = max(1, int(rc.get("n_az", 1)))
    n_el = max(1, int(rc.get("n_el", 1)))
    if n_az * n_el >= len(dirs):
        return [(0, len(dirs))]
    return [(i, min(i + n_az * n_el, len(dirs))) for i in range(0, len(dirs), n_az * n_el)]


def _sector_basis(dirs, a, b):
    """Beam coordinates of one loop: right axis, up axis, boresight."""
    f = dirs[a:b].sum(axis=0)
    n = float(np.linalg.norm(f))
    if n < 1e-9:
        return None
    f = f / n
    ref = np.array([0.0, 0.0, 1.0])
    if abs(float(f @ ref)) > 0.98:
        ref = np.array([0.0, 1.0, 0.0])
    right = np.cross(ref, f)
    right /= max(float(np.linalg.norm(right)), 1e-12)
    up = np.cross(f, right)
    return right, up / max(float(np.linalg.norm(up)), 1e-12), f


def _sector_of(dirs, angles, rc, pos, origin):
    """(loop start, loop stop, range, unit direction) for one entity.

    The loop is chosen by the entity's angle inside each loop, not by the single
    best-matching beam: a body that sits between two beams is inside the
    scanned volume and must couple through the antenna pattern, otherwise a
    legitimate standoff geometry disappears just because the scan is coarse.
    """
    d = np.asarray(pos, float) - np.asarray(origin, float)
    R = float(np.linalg.norm(d))
    if R < 1e-6:
        return None
    u = d / R
    best = None
    for a, b in _loops(dirs, rc):
        sb = _sector_basis(dirs, a, b)
        if sb is None:
            continue
        right, up, fwd = sb
        gl = float(np.mean(dirs[a:b] @ u))
        if best is None or gl > best[0]:
            best = (gl, a, b, right, up, fwd)
    if best is None:
        return None
    _, a, b, right, up, fwd = best
    return dict(a=a, b=b, right=right, up=up, fwd=fwd, range=R, u=u)


def _angle_off(u, sec, beam):
    """Degrees between the entity and a beam inside its own loop."""
    c = float(np.clip(float(u @ beam), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def _pattern_gain(off_deg, bw_deg, floor_db):
    """Gaussian main lobe with a sidelobe floor, relative to peak gain."""
    return max(float(np.exp(-np.log(2.0) * (off_deg / max(bw_deg, 1e-6)) ** 2)),
               10.0 ** (float(floor_db) / 10.0))


def _prepare(world, cfg):
    """(spec, entity states at this frame, frame time); (None, [], t) if off.

    The planner's Tracks are preferred: they can answer for the release instant
    and for a lagged frame, which one world pose cannot. The world anchors are
    the compatibility fallback for callers that build a scene without a plan.
    """
    spec = load(cfg.get("ecm"))
    t = frame_time(cfg)
    if not spec["enabled"]:
        return None, [], t
    return spec, entities_at(spec, t, anchors=anchors_of(world),
                             tracks=tracks_of(cfg)), t


def jammer_power_w(ent, ctx, sec):
    """One-way jammer power at the radar input, in watts.

        P_j = ERP * G_r(off-boresight) * lam^2 / ((4 pi)^2 R_j^2 L)

    R_j is the JAMMER's range, not the target's. That asymmetry is the whole
    reason a standoff jammer works: the target's echo falls as R^-4 while the
    jammer's field falls as R^-2, so the J/S ratio improves as the jammer closes
    and burn-through happens at a definite range.

    Two more loss terms sit on top, both of them physical:
      * a barrage jammer spreads ERP over its own bandwidth and keeps only
        B_rx/B_jam of it inside the receiver, so it needs ~B_jam/B_rx more power
        than a spot jammer for the same J/S;
      * a jammer whose antenna does not point at the radar only couples through
        a sidelobe, which is why "look_at_sensor" changes the answer by tens of
        dB and not by a few percent.
    """
    R = float(sec["range"])
    p_rx = (float(ent["erp_w"]) * float(ctx["gain"]) * float(ctx["lam"]) ** 2
            / ((4.0 * np.pi) ** 2 * max(R, 1e-6) ** 2 * float(ctx["loss"])))
    jam_bw = float(ent["jam_bandwidth_mhz"]) * 1e6
    overlap = (min(1.0, float(ctx["fs"]) / jam_bw) if ent["mode"] == "barrage"
               else 1.0)
    off = _angle_off(sec["u"], sec, ctx["dirs"][sec["a"]])
    ant = (1.0 if ent["look_at_sensor"]
           else _pattern_gain(off, float(ctx["rc"].get("beamwidth_az_deg", 1.0)),
                              float(ent["sidelobe_db"])))
    return p_rx * overlap * _band_factor(ent, ctx.get("band")) * ant, R


def radar_noise_floor(world, cfg, rc, ctx):
    """Per-beam receiver noise power with every jammer added in.

    A noise jammer is not an echo: it enters before the matched filter and
    raises the floor of every beam that sees it, so it must be known before the
    frame's SNR is computed and before the beat signal is synthesised. Keeping
    the floor per beam (rather than one global scaler) is what makes the main
    lobe and the sidelobes differ by the antenna pattern instead of the whole
    picture simply getting darker.
    """
    spec, ents, t = _prepare(world, cfg)
    if spec is None:
        return None
    dirs = ctx["dirs"]
    pn = np.full(len(dirs), float(ctx["pn"]))
    bw = float(ctx["rc"].get("beamwidth_az_deg", 1.0))
    jams = []
    for ent in ents:
        if ent["kind"] != "noise_jammer":
            continue
        sec = _sector_of(dirs, ctx["angles"], rc, ent["pos"], ctx["origin"])
        if sec is None:
            continue
        p_peak, R = jammer_power_w(ent, ctx, sec)
        if p_peak <= 0.0:
            continue
        off = np.degrees(np.arccos(np.clip(dirs @ sec["u"], -1.0, 1.0)))
        pn += p_peak * np.maximum(
            np.exp(-np.log(2.0) * (off / max(bw, 1e-6)) ** 2),
            10.0 ** (float(ent["sidelobe_db"]) / 10.0))
        jams.append(dict(id=ent["id"], pos=[round(float(x), 3) for x in ent["pos"]],
                         range_m=round(R, 2), peak_power_w=float(p_peak),
                         loop=[int(sec["a"]), int(sec["b"])],
                         off_boresight_deg=round(float(_angle_off(
                             sec["u"], sec, dirs[sec["a"]])), 4),
                         jam_bandwidth_mhz=float(ent["jam_bandwidth_mhz"]),
                         mode=ent["mode"], band=ent.get("band")))
    if not jams:
        return None
    return dict(pn_beam=pn, jammers=jams, time_s=round(t, 4))


def _anchor_echo(world, ctx, ent):
    """The echo a repeater's false targets are copied from.

    A DRFM repeater does not invent a target: it receives the victim radar's own
    pulse and retransmits it delayed and Doppler-shifted, so the false target
    sits on the *target's* line of sight in a different range/Doppler cell.
    "anchor" names the object whose echo is copied; without it the strongest
    labelled echo is used, which is the self-screening case. When the frame has
    no target echo at all (occluded, beyond the unambiguous range, gated out by
    MTI) nothing is repeated and the entity is skipped -- a repeater with
    nothing to repeat is silent, and pretending otherwise would put ghost
    targets into frames where the radar saw nothing.
    """
    pr, obj, dist, vr = ctx["pr"], ctx["obj_of_beam"], ctx["dist"], ctx["vr"]
    mask = obj >= 0
    want = ent.get("anchor")
    if want:
        ids = [o["id"] for o in getattr(world, "objs", []) or []
               if o.get("name") == want and o.get("label", True)]
        mask = mask & np.isin(obj, ids) if ids else np.zeros_like(mask)
    if mask.any():
        b = int(np.argmax(np.where(mask, pr, -1.0)))
        return dict(beam=b, range=float(dist[b]), v_r=float(vr[b]),
                    power=float(pr[b]))
    return None


def _offset_range(ent, t, index):
    kind = ent["kind"]
    if kind == "false_target":
        return float(ent["range_offset_m"]) + float(ent["range_step_m"]) * index
    if kind == "range_gate_pull_off":
        # a range gate is *stolen* by walking the false peak away from the
        # target, so the offset has to grow during the clip instead of being a
        # single fixed ghost; it saturates at the gate's pull-off limit
        walked = float(ent["rate_mps"]) * max(0.0, t - float(ent["life"]["start_s"]))
        walked = float(np.clip(walked, -float(ent["max_offset_m"]),
                               float(ent["max_offset_m"])))
        return float(ent.get("range_offset_m", 0.0)) + walked
    return float(ent.get("range_offset_m", 0.0))


def _offset_velocity(ent, t, index):
    kind = ent["kind"]
    if kind == "false_target":
        return (float(ent["velocity_offset_mps"])
                + float(ent["velocity_step_mps"]) * index)
    if kind == "velocity_gate_pull_off":
        walked = float(ent["rate_mps"]) * max(0.0, t - float(ent["life"]["start_s"]))
        walked = float(np.clip(walked, -float(ent["max_offset_mps"]),
                               float(ent["max_offset_mps"])))
        return float(ent.get("velocity_offset_mps", 0.0)) + walked
    return float(ent.get("velocity_offset_mps", 0.0))


def radar_apply(world, cfg, rc, echoes, ctx):
    """Append ECM echoes to the radar's own echo list.

    Called after the target and clutter echoes exist and before they are
    grouped per beam, so a false target travels through exactly the same
    beat-signal / FFT / detector chain as a real echo and nothing downstream
    can tell them apart. Returns None when the scene has no ECM.

    Two physically different things live here on purpose:
      * passive bodies (chaff, corner reflector, decoy, flare) are real
        scatterers with a position of their own, so their echo is the ordinary
        two-way radar equation solved at their own range and Doppler. They
        cannot be anchored to the target: that is what "independent motion"
        means, and it is why a chaff cloud trails the aircraft instead of
        sitting on it;
      * deception jammers (false_target, *_pull_off, angle_deception) are
        transponders: their echo is a *copy* of the target's own return moved to
        another range/Doppler/angle cell. That is why they need an anchor, and
        why they fall silent when the radar sees nothing to copy.
    """
    spec, ents, t = _prepare(world, cfg)
    if spec is None or not ents:
        return None
    dirs, angles, origin = ctx["dirs"], ctx["angles"], ctx["origin"]
    lam, b, t_chirp = ctx["lam"], ctx["b"], ctx["t_chirp"]
    r_max, v_rad = ctx["r_max"], ctx["v_rad"]
    pn, tx_w, gain, loss = ctx["pn"], ctx["tx_w"], ctx["gain"], ctx["loss"]
    det_snr = float(ctx.get("det_snr_db", 12.0))
    pg_db = float(ctx.get("pg_db", 0.0))
    pn_ref = np.atleast_1d(np.asarray(pn, dtype=float))
    detections, records = [], []
    # the caller's list already holds every target and clutter echo; only the
    # ECM ones appended below may be counted as "injected", or the report would
    # grow with the scene instead of with the jamming
    n_before = len(echoes)

    for ent in ents:
        sec = _sector_of(dirs, angles, rc, ent["pos"], origin)
        if sec is None:
            continue
        R, u = float(sec["range"]), sec["u"]
        rec = dict(id=ent["id"], kind=ent["kind"], group=ent["group"],
                   pos=[round(float(x), 3) for x in ent["pos"]],
                   vel=[round(float(x), 3) for x in ent["vel"]],
                   extent_m=round(float(ent["radius_m"]), 3),
                   weight=round(float(ent["weight"]), 4),
                   loop=[int(sec["a"]), int(sec["b"])], injected=[])

        if ent["kind"] == "noise_jammer":
            # a jammer pod is a real body, but its skin return is 40+ dB below
            # the airframe carrying it and it is already in the world as
            # geometry; injecting an echo here would put a second "aircraft" on
            # the RD map. A jammer's job is the noise floor, not a skin return.
            rec["noise_only"] = True
            records.append(_jsonable(rec))
            continue

        if ent["kind"] in BODY_KINDS:
            # a real scatterer: its own position, its own velocity, its own
            # cross-section, solved with the same two-way radar equation the
            # airframe goes through, so no downstream code needs to know that
            # this echo did not come from world.faces
            sig = sigma_of(ent, lam) * _band_factor(ent, ctx.get("band"))
            rec["rcs_m2"] = round(float(sig), 4)
            if sig <= 0.0 or not (0.5 <= R < r_max):
                records.append(_jsonable(rec))
                continue
            b_i = int(np.argmax(dirs[sec["a"]:sec["b"]] @ u)) + sec["a"]
            v_r = -float(u @ (np.asarray(ent["vel"], float) - v_rad))
            p_e = (tx_w * gain ** 2 * lam ** 2 * sig
                   / ((4.0 * np.pi) ** 3 * R ** 4 * loss))
            if p_e <= 0.0:
                records.append(_jsonable(rec))
                continue
            echoes.append((b_i, 2.0 * b * R / (C_LIGHT * t_chirp),
                           2.0 * v_r / lam,
                           np.sqrt(p_e) * np.exp(1j * _phase(ctx, ent["id"]))))
            snr = 10.0 * np.log10(max(p_e / float(pn_ref[b_i]), 1e-30)) + pg_db
            rec.update(range_m=round(R, 2), v_r_mps=round(v_r, 3),
                       power_w=float(p_e), beam=b_i,
                       detected=bool(snr >= det_snr))
            if snr >= det_snr:
                detections.append(dict(
                    x=float(ent["pos"][0]), y=float(ent["pos"][1]),
                    z=float(ent["pos"][2]), range=R,
                    az=float(angles[b_i][0]), el=float(angles[b_i][1]),
                    v_r=float(v_r), snr_db=float(snr), obj=-1,
                    source="ecm_" + ent["kind"], ecm_id=ent["id"]))
            records.append(_jsonable(rec))
            continue

        anchor = _anchor_echo(world, ctx, ent)
        if anchor is None:
            rec["note"] = "no target echo this frame: repeater has nothing to copy"
            records.append(_jsonable(rec))
            continue
        base_power = float(anchor["power"])
        for k in range(int(ent["count"])):
            r_k = float(anchor["range"]) + _offset_range(ent, t, k)
            v_k = float(anchor["v_r"]) + _offset_velocity(ent, t, k)
            if not (0.5 <= r_k < r_max):
                continue
            abeam = int(anchor["beam"])
            if ent["kind"] == "angle_deception":
                # angle deception steals the *angle* track, so the copy has to
                # land in a different beam from the true echo; a copy in the
                # same beam would move nothing
                az0, el0 = angles[abeam]
                want = np.array([az0 + float(ent["angle_offset_deg"]), el0])
                abeam = int(np.argmin(((angles - want) ** 2).sum(axis=1)))
            power = (float(ent.get("power_w"))
                     if ent.get("power_w") is not None
                     else float(ent["power_ratio"]) * base_power)
            if power <= 0.0:
                continue
            echoes.append((abeam, 2.0 * b * r_k / (C_LIGHT * t_chirp),
                           2.0 * v_k / lam,
                           np.sqrt(power) * np.exp(1j * _phase(
                               ctx, "%s/%d" % (ent["id"], k)))))
            snr = 10.0 * np.log10(max(power / float(pn_ref[abeam]), 1e-30)) + pg_db
            rec["injected"].append(dict(beam=abeam, range_m=round(r_k, 3),
                                        v_r_mps=round(v_k, 3),
                                        power_w=float(power),
                                        snr_db=round(float(snr), 3)))
            if snr >= det_snr:
                detections.append(dict(
                    x=float(ent["pos"][0]), y=float(ent["pos"][1]),
                    z=float(ent["pos"][2]), range=r_k,
                    az=float(angles[abeam][0]), el=float(angles[abeam][1]),
                    v_r=float(v_k), snr_db=float(snr), obj=-1,
                    source="ecm_" + ent["kind"], ecm_id=ent["id"]))
        rec["anchor"] = dict(beam=int(anchor["beam"]),
                             range_m=round(float(anchor["range"]), 3),
                             power_w=float(base_power))
        records.append(_jsonable(rec))

    strongest = (float(np.max(np.where(ctx["obj_of_beam"] >= 0, ctx["pr"], 0.0)))
                 if len(ctx["pr"]) else 0.0)
    copies = sum(len(e["injected"]) for e in records if "injected" in e)
    report = dict(enabled=True, time_s=round(float(t), 4), entities=records,
                  injected_echoes=len(echoes) - n_before, deception_copies=copies,
                  false_detections=len(detections),
                  strongest_target_power_w=strongest)
    return dict(echoes=echoes, detections=detections, records=records,
                report=report, noise_power_w=0.0)


def _jsonable(value):
    """Round-trip through JSON types so a report can be written to a label."""
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


# ------------------------------------------------------------ see-through
def pass_through(bvh, face_obj, origin, dirs, hit, ids):
    """Let rays that stopped on an ECM body continue to what stands behind it.

    A chaff cloud is a volume of dipoles, not a wall: it must not erase the
    target's echo, or the cloud would hide the very aircraft it was released to
    protect. The cloud's own contribution is added analytically in radar_apply,
    so a geometric hit on it is dropped and the ray is re-cast behind it. Only
    rays that actually land on an ECM body are re-cast, which costs nothing when
    the scene carries no ECM.
    """
    ids = np.asarray(sorted(int(i) for i in ids), np.int64)
    if ids.size == 0:
        return hit
    t = np.array(hit["t"], float)
    tri = np.array(hit["tri"], np.int64)
    ok = np.array(hit["hit"], bool)
    point = np.array(hit["point"], float)
    origin = np.asarray(origin, float)
    for _ in range(6):
        obj = np.where(ok, face_obj[np.where(ok, tri, 0)], -1)
        blocked = ok & np.isin(obj, ids)
        if not blocked.any():
            break
        idx = np.where(blocked)[0]
        # step past the entry point; a second pass walks past the far wall of the
        # same body, so a ray crossing a whole cloud still gets through
        start = origin + dirs[idx] * (t[idx] + np.maximum(1e-3, 2e-3 * t[idx]))[:, None]
        again = bvh.ray_cast_batch(start, dirs[idx])
        t[idx] = np.where(again["hit"], t[idx] + again["t"], np.inf)
        tri[idx] = np.where(again["hit"], again["tri"], -1)
        ok[idx] = again["hit"]
        point[idx] = start + dirs[idx] * np.where(again["hit"], again["t"], 0.0)[:, None]
    return dict(t=t, tri=tri, hit=ok, point=point)


def attach(world, cfg, t, anchors=None, tracks=None):
    """Append the ECM bodies to an already built World.

    The world stays the single geometric truth for every sensor: the same
    triangles occlude in the rasteriser, return lidar pulses and appear in the
    radar ray cast (where pass_through makes the volumetric ones see-through).
    Object IDs run on from the target IDs and every entity is label=False, so no
    annotation can ever claim a jammer pod as a target class.

    scene.py hands this the built World and the frame config; a Track set on
    that config (plan.ClipPlan puts one there) takes priority, and the anchors
    argument stays as the explicit compatibility entry for everything else.
    """
    from . import materials as matlib
    from .scene import World

    spec = load(cfg.get("ecm"))
    if not spec["enabled"]:
        return world, []
    t = float(t)
    tracks = tracks if tracks is not None else tracks_of(cfg)
    anchors = anchors if anchors is not None else anchors_of(world)
    ents = entities_at(spec, t, anchors=anchors, tracks=tracks)
    if not ents:
        return world, []
    mat_names = sorted(matlib.MATERIALS)
    mat_index = {n: i for i, n in enumerate(mat_names)}
    verts, faces = [world.verts], [world.faces]
    fobj, fmat = [world.face_obj], [world.face_mat]
    src, grp = [world.face_source_material], [world.face_group]
    local = [world.face_local_centers]
    objs = list(world.objs)
    added = []
    voff = len(world.verts)
    for ent in ents:
        if not ent.get("visible", True) or ent["kind"] not in BODY_KINDS:
            continue
        if ent["mat"] not in mat_index:
            raise ValueError("ecm %s: unknown material %r" % (ent["id"], ent["mat"]))
        m = _world_mesh(ent)
        oid = len(objs)
        verts.append(m.v)
        faces.append(m.f + voff)
        fobj.append(np.full(len(m.f), oid, np.int32))
        fmat.append(np.full(len(m.f), mat_index[ent["mat"]], np.int32))
        src.append(np.full(len(m.f), "", dtype=object))
        grp.append(np.full(len(m.f), "", dtype=object))
        local.append(m.centers)
        voff += len(m.v)
        lo, hi = m.bbox
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                            for z in (lo[2], hi[2])])
        objs.append(dict(
            id=oid, name=ent["id"], cls="ecm", mat=ent["mat"],
            pos=np.asarray(ent["pos"], float), vel=np.asarray(ent["vel"], float),
            yaw=0.0, pitch=0.0, roll=0.0, size=hi - lo, corners=corners,
            temp_c=ent.get("temp_c"), internal_heat=None, rcs_db=None,
            visible=False, label=False, thermal_regions=[],
            source_material_report=dict(
                fallback=ent["mat"], source_names=[], mapped={}, unmapped=[],
                unspecified_faces=0, import_appearance=False,
                physical_properties="declared ecm material"),
            ecm=dict(kind=ent["kind"], group=ent["group"], id=ent["id"],
                     rcs_m2=ent["rcs_m2"], extent_m=ent["radius_m"])))
        added.append(objs[-1])
    if not added:
        return world, []
    out = World(np.vstack(verts), np.vstack(faces), np.concatenate(fobj),
                np.concatenate(fmat), objs,
                face_source_material=np.concatenate(src),
                face_group=np.concatenate(grp),
                face_local_centers=np.vstack(local))
    # only the volumetric bodies are see-through for the radar ray cast: a decoy
    # or a corner reflector is a solid object and must shadow what is behind it
    out.ecm_ids = tuple(o["id"] for o in added)
    out.ecm_volumetric_ids = tuple(
        o["id"] for o in added if o["ecm"]["kind"] in VOLUMETRIC)
    return out, added