"""Frame planner: one continuous clip instead of independent per-frame draws.

Why this exists
---------------
Sampling the target's pose independently on every frame yields a bag of
unrelated photographs. Nothing in frame k predicts k+1, so a tracker, a video
detector or any 3-D consistency prior has nothing to learn from, and the
labelled box jumps around the image in a way no real sensor can produce.

So a run is now built from *clips*. Each clip owns one trajectory, one camera
rig and one weather timeline, and every quantity the renderers consume is a
smooth function of clip time:

    pos(t), vel(t)        analytic, so Doppler and the label yaw agree
    camera(t)             a rig with a causal lag, never per-frame noise
    weather(t)            a preset held for the clip, or a slow smooth drift

Chunking stays exact: clip index and local time are derived from the *global*
frame id, so generating frames 0-99 in one worker and 100-199 in another
reproduces the same path no matter how the split falls.

State source
------------
Every object owns a motion.Track.  The primary target's is exposed as
track_target (the seeker back-solve wrapped as a Track, or the authored path),
and every other object is reachable through tracks / track(name).  A labelled
secondary target without an explicit per-object trajectory defaults to a
FormationTrack derived from the primary -- a body-frame offset, not a second
absolute straight line -- which is what keeps a wingman in formation when the
seeker drags the lead along its line of sight.  The ECM layer reads the same
tracks, so a dispensed body's mount is rotated by the lead's actual body frame
and its release state is evaluated at the lead's actual state.

Seeker geometry
---------------
For a seeker view the target is described in sensor-relative spherical
coordinates (range, azimuth, elevation) rather than in world coordinates, and
the range band is *solved backwards from the pixel size the target should
occupy*. That is what makes "barely visible target at a realistic imaging
range" a parameter instead of a lucky accident.

Platform (flying seeker) mode
-----------------------------
``clip.seeker.mode = "platform"`` inverts the geometry: the camera flies a real
dive-and-weave track (motion.DiveWeave) and the target stands still. The legacy
mode parked the camera and dragged the target towards it, which meant the
background never moved -- no ground rush, no platform Doppler, no MTI. Here the
LOS range is ``|target(t) - track(t)|``, the gimbal lags by ``lag_s``, the mount
banks with its lateral acceleration, and each configured view (narrow seeker,
wide search) gets its own aim, FOV and pixel span.

Both modes are pure functions of the global frame index, so a run split into
chunks reproduces frame k exactly (see selftest, check 3).
"""
import json
import os

import numpy as np

from . import motion, scene, weather


def _rng(seed):
    return np.random.default_rng(int(seed))


def range_for_px(size_m, px, fov_deg, width_px):
    """Range at which an object of size_m spans px pixels across the frame."""
    return float(size_m) * float(width_px) / (
        2.0 * float(px) * np.tan(np.radians(float(fov_deg)) / 2.0))


def px_for_range(size_m, rng_m, fov_deg, width_px):
    rng_m = max(float(rng_m), 1e-6)
    return float(size_m) * float(width_px) / (
        2.0 * rng_m * np.tan(np.radians(float(fov_deg)) / 2.0))


def object_size_m(spec, base_dir="."):
    """Characteristic size of a target: the wider of its two horizontal spans."""
    try:
        _m, lo, hi = scene.prepare_mesh(spec, base_dir)
        d = np.asarray(hi, float) - np.asarray(lo, float)
        return float(max(d[0], d[1], d[2] * 0.6, 0.2))
    except Exception:
        return 4.0


class _FormationPath:
    """A follower that mirrors the lead airframe's solved path with a fixed offset.

    Giving every object its own straight line from pos+vel made a two-ship break
    apart the instant the seeker started dragging the lead: one stayed on the
    line of sight and the other flew off at 250 m/s. A formation has to share the
    lead's motion; only its offset differs.
    """

    def __init__(self, plan, offset):
        self.plan = plan
        self.offset = np.asarray(offset, float)

    def pos(self, t):
        return np.asarray(self.plan.target_pos(t), float) + self.offset

    def vel(self, t):
        return np.asarray(self.plan.target_vel(t), float)

    def accel(self, t, h=2e-3):
        # The lead publishes no analytic acceleration, so differentiate velocity
        # the same way the rest of the planner samples it.
        t = float(t)
        return ((np.asarray(self.plan.target_vel(t + h), float)
                 - np.asarray(self.plan.target_vel(t - h), float)) / (2.0 * h))

    def yaw(self, t):
        return None


class _FrameConfig(dict):
    """One frame's config, with the planner's Track set attached as an attribute.

    dataset.py deep-copies the frame config with json.loads(json.dumps(...))
    for multi-band IR, so a live Track cannot ride inside the mapping.  An
    attribute is invisible to that copy (and to JSON), which keeps the config
    serialisable while the ECM layer can still ask "where was the parent when
    this flare left the rail" instead of guessing from one world pose.
    """

    tracks = None


class _ObjectTrack(motion.Track):
    """A scene object's published state, with its own attitude rules applied.

    This is where yaw_mode and attitude_mode are evaluated -- once.  Before,
    the label yaw came from object_state, the mesh attitude from
    object_attitude and the ECM parent pose from the built World: three places
    that could (and did) drift apart.  A Track is now the only one.
    """

    kind = "object"

    def __init__(self, base, spec):
        self.base = base
        self.spec = spec or {}
        self.name = self.spec.get("name")
        self.mode = str(getattr(base, "mode", "track"))
        self.yaw_mode = str(self.spec.get("yaw_mode", "velocity"
                                          if self.spec.get("label", True) else "fixed"))
        if self.yaw_mode not in ("velocity", "fixed"):
            raise ValueError("yaw_mode must be velocity or fixed")
        self.attitude_mode = str(self.spec.get("attitude_mode", "fixed"))

    def pos(self, t):
        return np.asarray(self.base.pos(float(t)), float)

    def vel(self, t):
        return np.asarray(self.base.vel(float(t)), float)

    def accel(self, t):
        return np.asarray(self.base.accel(float(t)), float)

    def jerk(self, t):
        base_jerk = getattr(self.base, "jerk", None)
        if base_jerk is None:
            # the legacy world-offset follower publishes no jerk; derive it the
            # way Track does everywhere else (centred difference of accel)
            return motion.Track.jerk(self, float(t))
        return np.asarray(base_jerk(float(t)), float)

    def yaw(self, t):
        t = float(t)
        rot = float(self.spec.get("rot", 0.0))
        off = float(self.spec.get("yaw_offset_deg", 0.0))
        if self.yaw_mode == "fixed":
            return rot
        v = self.vel(t)
        # 1e-6 m/s treated floating-point dust as a real heading: a formation
        # member at rest has a residual velocity from the lever term whose
        # direction is noise, and it reported 142 deg off the leader on frame 0.
        # 0.05 m/s is below any airframe's meaningful motion.
        if float(np.hypot(v[0], v[1])) > 0.05:
            return float(np.degrees(np.arctan2(v[1], v[0]))) + off
        # A stopped object keeps the heading its route ends on; a waypoint route
        # has one even at zero speed, so it is tried before the static rot.
        heading_path = self.base
        # Unwrap the track wrappers down to the path itself: PathTrack and
        # GroundPath both just hold one, and without this a route that starts
        # from rest reported the fallback rot while the next frame reported the
        # route tangent -- a 180 deg disagreement across one frame.
        while isinstance(heading_path, (motion.GroundPath, motion.PathTrack)):
            heading_path = heading_path.path
        if isinstance(heading_path, motion.Waypoints):
            h = heading_path.heading(t)
            if h is not None:
                return float(h) + off
        h = getattr(self.base, "yaw", None)
        if h is not None:
            y = h(t)
            if y is not None:
                return float(y) + off
        return rot + off

    def pitch(self, t):
        if self.attitude_mode != "coordinated":
            return float(self.spec.get("pitch", 0.0))
        return motion.climb_angle_deg(self.vel(t))

    def roll(self, t):
        if self.attitude_mode != "coordinated":
            return float(self.spec.get("roll", 0.0))
        return motion.coordinated_bank_deg(self.vel(t), self.accel(t))

    def frame(self, t):
        return motion.body_to_world(self.yaw(t), self.pitch(t), self.roll(t))


class _SeekTrack(motion.Track):
    """The seeker back-solve as a Track: position plus its analytic derivative.

    _seek_state differentiates the range/az/el sweep by hand, so the published
    velocity is the derivative of the published position and the 76.9 g first
    frame cannot come back (docs/运动修正报告.md).  The acceleration is the
    design's centred difference of that velocity (h = 1e-3).
    """

    kind = "seeker"

    def __init__(self, plan):
        self.plan = plan

    def pos(self, t):
        return np.asarray(self.plan._seek_state(float(t))[0], float)

    def vel(self, t):
        return np.asarray(self.plan._seek_state(float(t))[1], float)


class ClipPlan:
    """Trajectory + camera rig + weather for one clip. Every accessor is f(t)."""

    def __init__(self, cfg, clip, fpc, dt, base_seed):
        self.cfg = cfg
        self.clip = int(clip)
        self.fpc = max(int(fpc), 1)
        self.dt = float(dt)
        self.dur = self.fpc * self.dt
        self.base_seed = int(base_seed)
        self.rng = _rng(int(base_seed) * 104729 + self.clip * 7919 + 17)
        rng = self.rng
        self.wstates = weather.clip_states(cfg, self.fpc, self.dt, rng)

        spec = cfg.get("clip") or {}
        self.spec = spec
        sensors = cfg.get("sensors") or {}
        cam_name = spec.get("camera") or "ir_camera"
        cam0 = sensors.get(cam_name) or sensors.get("ir_camera") or sensors.get("optical") or {}
        self.cam_name = cam_name
        self.cam0 = dict(cam0)
        self.cam_pos = np.array(cam0.get("pos", [0, 0, 0]), float)
        # Pursuit mount advance (m/s along the LOS), set by _plan_seeker when
        # the scene authors seeker.advance_mps; None keeps the mount static.
        self.cam_advance = None
        self.width = int(cam0.get("width", 480))
        self.fov = float(cam0.get("fov_deg", 30.0))
        self._basis = scene.sensor_basis(cam0.get("pos", [0, 0, 0]),
                                         cam0.get("look_at", [0, 0, 1]),
                                         cam0.get("up", (0, 0, 1)))

        self.targets = [o for o in cfg.get("objects", []) if o.get("label", True)]
        if not self.targets:
            raise ValueError("a clip needs at least one labelled object")
        self.target = self.targets[0]
        self.size_m = object_size_m(self.target, cfg.get("_base_dir", "."))
        self.ground_z = spec.get("ground_z")
        if self.ground_z is None:
            g = cfg.get("ground") or {}
            self.ground_z = float(g.get("z", 0.0)) if g.get("enabled", True) else None

        seeker = spec.get("seeker") or cfg.get("seeker")
        self.mode = "seeker" if seeker else "world"
        self.path = None
        self.seek = None
        self.rig = motion.Rig(dict(spec.get("rig") or {}), cam0, None, rng)
        self.wander = dict(deg=0.0, hz=0.09, ph=0.0, ph2=0.0)

        # Field of view list. Without an explicit clip.views block this collapses
        # to the single legacy view taken from sensors[cam_name], so every old
        # scene keeps its numbers bit for bit.
        self.views = self._build_views()
        if (spec.get("views") or []) and self.views:
            first = self.views[list(self.views.keys())[0]]
            self.fov = float(first["fov_deg"])
            self.width = int(first["width"])

        observer = spec.get("observer_trajectory")
        if observer:
            if seeker:
                raise ValueError("observer_trajectory and seeker are mutually exclusive")
            self.mode = "independent_flight"
            self.path = motion.make_path(spec.get("trajectory") or self._trajectory_from_objects())
            self.observer_path = motion.make_path(observer)
            self.gimbal_lag = float((spec.get("rig") or {}).get("lag_s", 0.0))
            if self.gimbal_lag < 0:
                raise ValueError("lag_s must be non-negative")
        elif seeker:
            if str(dict(seeker).get("mode", "")).lower() == "platform":
                self.mode = "platform"
                self._plan_platform(seeker)
            else:
                self._plan_seeker(seeker)
        else:
            tspec = spec.get("trajectory") or self._trajectory_from_objects()
            self.path = motion.make_path(tspec, ground_z=(self.ground_z
                if self.target.get("motion_type") == "ground" else None))
            self.rig = motion.Rig(dict(spec.get("rig") or {}), cam0, self.path, rng)
            self.rig_lag = self.rig.lag
            # A fixed mount that pans to follow the target. Without it a long
            # clip filmed from one spot watches the target fly out of frame:
            # ds_air_uav covers 1224 m of sky through a 151 m wide frame, so 85%
            # of the frames have nothing labelled in them.
            # clip.camera is the camera NAME ("optical"), not a block, so the
            # tracking flag gets its own key.
            if str(spec.get("camera_track", "")).lower() in ("1", "true", "yes", "on"):
                self.mode = "track"

        # ---- state sources (docs/多目标运动仿真调研.md 5.1) ----------------
        # One Track per object.  The primary target's track is derived (seeker
        # back-solve) or authored (a path); every other object is an explicit
        # trajectory, a formation member derived from the primary, or the
        # legacy absolute line.  object_state, object_attitude, the labels and
        # the ECM layer all read these tracks, so no consumer gets a second
        # opinion about where a body is.
        names = [o.get("name") for o in cfg.get("objects", [])]
        if any(not n for n in names) or len(set(names)) != len(names):
            raise ValueError("objects require unique, non-empty names")
        self._objects = {o["name"]: o for o in cfg.get("objects", [])}
        self.tracks = {}
        self.object_paths = {}
        self._lead_line_is_authored = self._lead_line_authored()
        self.track_target = self._build_tracks()
        self.px0 = self.px_for_time(0.0)
        self.px1 = self.px_for_time(self.dur)
        self.r0 = self.range_for_time(0.0)
        self.r1 = self.range_for_time(self.dur)

    # ---- Track construction ---------------------------------------------
    def _lead_line_authored(self):
        """True when the primary target flies the line its own pos+vel implies.

        Then every object's absolute pos+vel is the same story in the same
        world frame -- a parking lot does not become a formation because a car
        sits beside another car.  When the primary's path is DERIVED (seeker)
        or authored as something other than a line (arc, weave, waypoints,
        drawn), an absolute line for a follower is a second reference frame,
        and that is precisely the reported bug: the follower keeps flying its
        own world line while the seeker drags the primary along another.
        """
        if self.mode == "seeker":
            return False
        if self.mode == "platform":
            return True
        traj = self.spec.get("trajectory")
        if traj is None:
            traj = self.target.get("trajectory")
        mode = str((traj or {}).get("mode", "line")).lower()
        return mode in ("line", "straight", "static", "hover", "parked")

    def _lead_kinematics(self):
        """The primary target's kinematic Track, before its attitude rules."""
        if self.mode == "seeker":
            return _SeekTrack(self)
        if self.mode == "platform":
            # the target holds station (or drives slowly); the platform moves
            return motion.PathTrack(motion.Line(self.tgt_pos, self.tgt_vel))
        return motion.PathTrack(self.path)

    def _line_path(self, obj):
        traj = dict(mode="line", vel=obj.get("vel", [0, 0, 0]))
        traj.setdefault("origin", obj.get("pos", [0, 0, 0]))
        ground = self.ground_z if obj.get("motion_type") == "ground" else None
        return motion.PathTrack(motion.make_path(traj, ground_z=ground))

    def _object_base(self, obj, lead_track, lead_nominal, build):
        """Kinematic Track of one non-primary object."""
        name = obj.get("name")
        trajectory = obj.get("trajectory")
        if trajectory is not None:
            # An explicit per-object route always wins: the scene said where
            # this one goes and it is not flying formation.
            traj = dict(trajectory)
            traj.setdefault("origin", obj.get("pos", [0, 0, 0]))
            ground = self.ground_z if obj.get("motion_type") == "ground" else None
            return motion.PathTrack(motion.make_path(traj, ground_z=ground))

        formation = obj.get("formation")
        link = str(obj.get("link") or ("formation" if formation is not None else "")).lower()
        if link == "formation":
            spec = dict(formation or {})
            parent_name = spec.get("parent") or self.target["name"]
            if parent_name == self.target["name"]:
                parent = lead_track
                parent_nominal = lead_nominal
            else:
                parent_obj = self._objects.get(parent_name)
                if parent_obj is None:
                    raise ValueError("%s: formation.parent names an unknown object %r"
                                     % (name, parent_name))
                parent = build(parent_obj)
                parent_nominal = np.asarray(parent_obj.get("pos", [0.0, 0.0, 0.0]),
                                            float)
            carrier = str(spec.get("carrier", "body")).lower()
            offset_body = spec.get("offset_body")
            if offset_body is None:
                # a scene's objects[].pos is a world coordinate; rotate it into
                # the parent's frame at t=0 so the authored placement holds
                offset_world = (np.asarray(obj.get("pos", [0.0, 0.0, 0.0]), float)
                                - parent_nominal)
                if carrier == "level":
                    yaw0 = parent.yaw(0.0)
                    frame0 = motion.body_to_world(
                        0.0 if yaw0 is None else float(yaw0), 0.0, 0.0)
                else:
                    frame0 = parent.frame(0.0)
                offset_body = np.asarray(frame0, float).T @ offset_world
            return motion.FormationTrack(
                parent, offset_body, mode=spec.get("mode", "lag"),
                tau_lag_s=spec.get("tau_lag_s", 0.3),
                own_maneuver=spec.get("own_maneuver"),
                max_offset_m=spec.get("max_offset_m"),
                carrier=carrier,
                max_ang_rate_rad_s=spec.get("max_ang_rate_rad_s"))

        if link == "independent":
            return self._line_path(obj)

        # The default: a labelled secondary target follows the primary.  This
        # replaces the "unconfigured means a straight line" fallback that let a
        # wingman fly 180 m/s while the seeker dragged the lead along a
        # different reference frame (docs ... 1.1).
        if (obj.get("label", True) and obj.get("motion_type") != "ground"
                and not self._lead_line_is_authored):
            offset_world = np.asarray(obj.get("pos", [0.0, 0.0, 0.0]), float) - lead_nominal
            # Placed side by side in the world, carried in the world's horizontal
            # plane: a yaw-only carrier keeps the wingman abeam through the
            # leader's bank.  Carrying it in the body frame put the offset into
            # the leader's banked plane at t = 0, so a 77 deg roll swung the
            # follower a couple of hundred metres of frame without anyone asking
            # for a formation manoeuvre.
            yaw0 = lead_track.yaw(0.0)
            level0 = motion.body_to_world(0.0 if yaw0 is None else float(yaw0),
                                          0.0, 0.0)
            offset_body = np.asarray(level0, float).T @ offset_world
            # A drawn route is C1 with a C0 curvature, so its corners carry
            # turn rates no airframe has.  The follower's own velocity is the
            # leader's plus w x (R o), and with an 80 m offset an unbounded w
            # produced 838 m/s and a heading that reversed every frame.  Cap the
            # rate at a 6 g lateral budget over the nominal speed, and never let
            # the lag window span more than one frame (0.3 s was three frames of
            # a 10 fps clip).
            nominal_speed = float(np.linalg.norm(
                np.asarray(self.target.get("vel", [0.0, 0.0, 0.0]), float)))
            max_rate = 9.81 * 6.0 / max(nominal_speed, 30.0)
            return motion.FormationTrack(lead_track, offset_body, mode="lag",
                                         tau_lag_s=min(0.3, self.dt),
                                         max_ang_rate_rad_s=max_rate,
                                         carrier="level")

        # Unlabelled clutter keeps the legacy world-offset follower in a seeker
        # clip: those objects were authored relative to the nominal target to
        # occlude it or to lie on the ground under it, and a body-frame offset
        # would rotate that geometry with the aircraft.
        if self.mode == "seeker" and obj.get("motion_type") != "ground":
            offset = np.asarray(obj.get("pos", [0.0, 0.0, 0.0]), float) - lead_nominal
            return _FormationPath(self, offset)

        return self._line_path(obj)

    def _build_tracks(self):
        lead = self.target
        lead_track = _ObjectTrack(self._lead_kinematics(), lead)
        self.tracks[lead["name"]] = lead_track
        lead_nominal = np.asarray(lead.get("pos", [0.0, 0.0, 0.0]), float)
        building = []

        def build(obj):
            name = obj.get("name")
            hit = self.tracks.get(name)
            if hit is not None:
                return hit
            if name in building:
                raise ValueError("formation cycle: " + " -> ".join(building + [name]))
            building.append(name)
            try:
                base = self._object_base(obj, lead_track, lead_nominal, build)
            finally:
                building.pop()
            track = _ObjectTrack(base, obj)
            self.tracks[name] = track
            # object_paths stays the public "this object has its own motion
            # model" set that tools/motion_diagnose reads; it never holds the
            # primary target, whose envelope is judged by its own mode.
            self.object_paths[name] = track
            return track

        for obj in self.cfg.get("objects", []):
            if obj is not lead:
                build(obj)
        return lead_track

    def track(self, name=None):
        """The Track of the named object (the primary target by default)."""
        if name is None:
            return self.track_target
        hit = self.tracks.get(name)
        if hit is None:
            raise KeyError("no track for object %r" % (name,))
        return hit

    def object_state(self, obj, t):
        """Single pose/velocity source for world, labels and shutter samples."""
        track = self.tracks.get(obj.get("name"))
        if track is None:
            track = self.track_target
        p, v = track.pos(t), track.vel(t)
        yaw = track.yaw(t)
        if yaw is None:
            yaw = float(obj.get("rot", 0.0)) + float(obj.get("yaw_offset_deg", 0.0))
        return np.asarray(p, float), np.asarray(v, float), float(yaw)

    def object_attitude(self, obj, t):
        """Pitch is nose-up; roll is the right-hand rotation about body +X.

        Both come from simkit.motion's single coordinated-turn definition, so the
        label orientation, the flight envelope and the motion report cannot drift
        apart: tan(bank) = a_lat / g with one g.  They are read off the object's
        own Track, which is also what the ECM mounts and the body-frame offsets
        are rotated by -- one attitude, not three.
        """
        track = self.tracks.get(obj.get("name"))
        if track is None or obj.get("attitude_mode") != "coordinated":
            return float(obj.get("pitch", 0.0)), float(obj.get("roll", 0.0))
        return float(track.pitch(t)), float(track.roll(t))

    # ---- planning -------------------------------------------------------
    def _trajectory_from_objects(self):
        t = self.target
        p = list(t.get("pos", [0.0, 0.0, 0.0]))
        v = list(t.get("vel", [0.0, 0.0, 0.0]))
        tspec = dict(t.get("trajectory") or dict(mode="line", vel=v))
        tspec.setdefault("origin", p)
        return tspec

    def _plan_seeker(self, sk):
        rng = self.rng
        px = sk.get("target_px") or [8.0, 45.0]
        px_lo, px_hi = float(min(px)), float(max(px))
        # Apparent size, not the bounding box: a 40 m wingspan seen head-on is a
        # few metres across, so the scene may state the size it wants to reason
        # about instead of inheriting the widest span of the mesh.
        size = float(sk.get("target_m") or self.size_m)
        self.size_used = size
        # range band that keeps the target inside the requested pixel band
        band = sorted([range_for_px(size, px_hi, self.fov, self.width),
                       range_for_px(size, px_lo, self.fov, self.width)])
        want = sk.get("range_m")
        if want:
            lo = max(float(want[0]), band[0])
            hi = min(float(want[1]), band[1])
            if hi <= lo:
                lo, hi = band[0], band[1]
        else:
            lo, hi = band[0], band[1]

        # The range rate is the parameter, not the end range: sampling two
        # independent ranges made the target cover a kilometre in five seconds,
        # i.e. fly at 400 m/s. A real closing rate keeps the geometry honest.
        closing = str(sk.get("closing", "in")).lower()
        dur = max(self.dur, 1e-6)
        rate = float(sk.get("closing_mps") or 0.0) or 0.25 * (hi - lo) / dur
        # A full traverse needs the start range to sit rate*dur above the band
        # floor; sampling the start uniformly across the whole band made long
        # clips clamp at lo and quietly close at a fraction of the authored
        # rate (5.4% apparent size change over a whole clip).
        amin = lo + 0.15 * (hi - lo)
        amax = hi - 0.15 * (hi - lo)
        if closing == "in":
            amin = max(amin, lo + rate * dur)
        elif closing == "out":
            amax = min(amax, hi - rate * dur)
        if amax > amin:
            a = float(rng.uniform(amin, amax))
        elif amax == amin:
            a = float(amin)
        else:
            a = float(0.5 * (lo + hi))
        if closing == "in":
            b = max(lo, a - rate * dur)
        elif closing == "out":
            b = min(hi, a + rate * dur)
        else:
            b = a

        az_span = float(sk.get("az_span_deg", 1.2))
        el_span = float(sk.get("el_span_deg", 0.5))
        # centre offsets put the engagement above the horizon (a UAV seeker looks
        # up) or below it (a ground seeker looks down) without changing the FOV
        az_c = float(sk.get("az_center_deg", 0.0))
        el_c = float(sk.get("el_center_deg", 0.0))
        self.seek = dict(
            r0=a, r1=b,
            az0=az_c + float(rng.uniform(-az_span, az_span) * 0.5),
            az1=az_c + float(rng.uniform(-az_span, az_span) * 0.5),
            el0=el_c + float(rng.uniform(-el_span, el_span) * 0.5),
            el1=el_c + float(rng.uniform(-el_span, el_span) * 0.5),
            profile=str(sk.get("profile", "linear")),
            los_jitter=float(sk.get("los_jitter_deg", 0.04)),
            # A perfectly tracking seeker puts the target on the optical axis in
            # every frame, so a detector learns "target = image centre" and
            # nothing else. Real mounts have LOS wander; this drifts the aim by a
            # few percent of the FOV so the target sweeps across the frame.
            wander_deg=float(sk.get("wander_deg", 0.0)),
            wander_hz=float(sk.get("wander_hz", 0.09)),
            wander_ph=float(rng.uniform(0.0, 6.283185)),
            wander_ph2=float(rng.uniform(0.0, 6.283185)),
            bob_hz=float(sk.get("los_jitter_hz", 0.35)),
            phase=float(rng.uniform(0.0, 6.283185)),
            keep_z=bool(sk.get("keep_z", self.ground_z is not None)),
        )
        # Pursuit mount advance.  A seeker that chases a target from behind
        # cannot be modelled with a static camera: with the target flying away
        # along the LOS (tail view, yaw_mode=velocity) a static camera can only
        # make the range GROW.  The mount therefore advances along the frame-0
        # line of sight at advance_mps; combined with closing="in" the range
        # still closes at closing_mps while the target keeps a strict tail view
        # and grows in the field of view.  Default 0 keeps the static mount.
        adv = float(sk.get("advance_mps") or 0.0)
        if adv > 0.0:
            u0, _, _ = self._seek_null(self.seek["az0"], self.seek["el0"])
            self.cam_advance = u0 * adv
        else:
            self.cam_advance = None

    def _build_views(self):
        """Field-of-view list: one entry per delivered image of the same scene.

        A narrow seeker view plus a wide search view is the normal case. Without
        a clip.views block the list holds exactly one entry taken from
        sensors[cam_name], so the legacy single-camera path is unchanged.
        """
        spec = self.cfg.get("clip") or {}
        raw = spec.get("views") or []
        out = {}
        if not raw:
            out["seeker"] = dict(name="seeker", tag="", fov_deg=float(self.fov),
                                 width=int(self.width),
                                 height=int(self.cam0.get("height", 360)),
                                 sweep_deg=0.0, sweep_hz=0.0, sensor=self.cam_name)
            return out
        for v in raw:
            name = str(v.get("name") or "view%d" % (len(out) + 1))
            out[name] = dict(
                name=name,
                tag=str(v.get("tag", "")),
                fov_deg=float(v.get("fov_deg", self.fov)),
                width=int(v.get("width", self.width)),
                height=int(v.get("height", self.cam0.get("height", 360))),
                sweep_deg=float(v.get("sweep_deg", 0.0) or 0.0),
                sweep_hz=float(v.get("sweep_hz", 0.05)),
                sensor=str(v.get("sensor") or self.cam_name),
            )
        return out

    def _plan_platform(self, sk):
        """Seeker on a flying platform: camera moves, target essentially sits.

        This is the mirror image of the legacy seeker mode. There the camera was
        parked and the target was dragged towards it, so the background never
        moved; here the airframe flies a real dive-weave and the target holds
        still, which is what gives the imagery its ground rush (and gives MTI
        something to reject).
        """
        rng = self.rng
        p = dict(sk.get("platform") or {})
        target = np.array(self.target.get("pos", [0.0, 0.0, 0.0]), float)
        tvel = np.array(self.target.get("vel", [0.0, 0.0, 0.0]), float)
        self.tgt_pos = target
        self.tgt_vel = tvel
        # Apparent size, same convention as the legacy mode: the wingspan seen
        # head-on is not the bounding box span.
        self.size_used = float(sk.get("target_m") or self.size_m)
        self.bank_max = float(sk.get("bank_max_deg", 25.0))
        self.gimbal_lag = float(sk.get("lag_s", 0.12))

        # ---- per-clip engagement geometry -------------------------------
        # Two clips of one scene must not be the same six seconds of footage,
        # otherwise the dataset is half duplicates. Everything that defines the
        # pass is drawn from the per-clip stream (seeded by the clip index), so a
        # given global frame id still reproduces exactly in any chunk worker.
        # Clips alternate sides of the nominal axis, which *guarantees* the two
        # passes differ in bearing by 2*spread (20 deg by default), and the slant
        # range is drawn from the scene's own [lo, hi] band per half, so the two
        # clips also differ in starting range and closing distance.
        var = dict(sk.get("per_clip_variation") or {})
        vary = bool(var.get("enabled", p.get("randomize", True)))
        heading0 = float(p.get("heading_deg", 90.0))
        dive0 = float(p.get("dive_deg", -14.0))
        spread = float(var.get("bearing_deg", p.get("heading_spread_deg", 10.0)))
        el_jit = float(p.get("el_jitter_deg", 0.8))

        side = 1.0 if (self.clip % 2) else -1.0
        heading = (heading0 + side * spread
                   + float(rng.uniform(-1.0, 1.0)) * 0.15 * spread)
        # Elevation of the platform above the target, signed. Positive means the
        # airframe sits above the target and dives onto it (air-to-air, air-to-
        # ground); negative means it sits BELOW and climbs at it, which is the
        # only way to describe a ground-based launcher or a low-level crossing
        # pass. Forcing |el| >= 1 deg made those two geometries impossible.
        # `el_deg` may be set on the seeker block or on its platform block; the
        # seeker block wins because that is where the engagement geometry lives.
        el_cfg = sk.get("el_deg", p.get("el_deg"))
        if el_cfg is not None:
            el = float(el_cfg) + float(rng.uniform(-1.0, 1.0)) * el_jit
        else:
            el0 = abs(float(dive0)) or 8.0
            el = max(1.0, el0 + float(rng.uniform(-1.0, 1.0)) * el_jit)
        # The track follows the line of sight it is diving along: the closing
        # rate stays ~speed instead of turning into a crossing pass.
        dive = -el + float(rng.uniform(-1.0, 1.0)) * float(
            p.get("dive_jitter_deg", var.get("dive_deg", 1.0)))
        if not vary:
            heading, dive = heading0, dive0
            el = float(el_cfg) if el_cfg is not None else -dive0

        band = sk.get("range_m") or p.get("range_m")
        r0 = None
        if band and vary:
            lo, hi = sorted(float(v) for v in list(band)[:2])
            frac = (0.5 if side > 0.0 else 0.0) + float(rng.uniform(0.12, 0.44))
            r0 = lo + (hi - lo) * frac

        if r0 is not None:
            ar, er = np.radians(heading), np.radians(el)
            to_tgt = np.array([np.cos(ar) * np.cos(er),
                               np.sin(ar) * np.cos(er),
                               -np.sin(er)])
            origin = target - to_tgt * r0
            standoff = None          # end range follows from r0 - v*t
        else:
            o = p.get("origin")
            origin = np.asarray(o if o is not None
                                else target + np.array([0.0, -5000.0, 1000.0]), float)
            standoff = p.get("standoff_end_m")

        phase = p.get("phase")
        if phase is None:
            phase = float(rng.uniform(0.0, 6.283185307179586)) if vary else 0.0
        # The platform's own airframe limits come from the scene, so the weave
        # cannot ask for a bank the aircraft does not have and the standoff solve
        # cannot invent a speed it cannot fly.  Without this the shipped scenes
        # asked for 2.5 g at 66 deg of bank (sk3_high_air2air) and for 63 m/s
        # from a platform configured at 14 m/s (ds1102_air2air_quad), both of
        # which the diagnostic flagged and neither of which is a path bug.
        plat_env = motion.Envelope.from_spec(dict(
            cls=str(p.get("flight_class") or "uav_fixed"),
            kinematics=p.get("kinematics") or {}))
        self.plat_env = plat_env
        self.path = motion.DiveWeave(
            origin, target,
            speed_mps=float(p.get("speed_mps", 180.0)),
            heading_deg=heading,
            dive_deg=dive,
            weave_amp_m=float(p.get("weave_amp_m", 0.0)),
            weave_hz=float(p.get("weave_hz", 0.22)),
            t_end=self.dur,
            standoff_end_m=standoff,
            phase=float(phase),
            target_vel=tvel,
            solve=str(p.get("solve", "standoff" if standoff is not None else "speed")).lower(),
            # bank_max_deg at seeker level overrides the class default, which is
            # how a scene says "this mount is limited" without restating the rest.
            bank_max_deg=float(sk.get("bank_max_deg") or plat_env.max_bank_deg),
            max_speed_mps=plat_env.max_speed_mps,
        )
        # NOTE: self.cam_pos is deliberately NOT moved to the track start.
        # state() translates every sensor by (camera(t) - cam_pos), so cam_pos
        # has to stay at the sensor block's own base pose; otherwise a clip whose
        # origin differs from the json pose would shift the whole turret by the
        # difference while the aim still pointed at the target.
        self._basis = scene.sensor_basis([float(v) for v in origin],
                                         [float(v) for v in target])
        self.plat_meta = dict(
            r0_m=float(np.linalg.norm(origin - target)),
            heading_deg=heading, dive_deg=dive, el_deg=el,
            speed_mps=float(p.get("speed_mps", 180.0)),
        )
        # Same wander model as the legacy seeker, so "wander_deg" means the same
        # thing in both modes: a smooth drift of a few percent of the FOV.
        self.wander = dict(
            deg=float(sk.get("wander_deg", 0.0)),
            hz=float(sk.get("wander_hz", 0.09)),
            ph=float(rng.uniform(0.0, 6.283185307179586)),
            ph2=float(rng.uniform(0.0, 6.283185307179586)),
        )


    # ---- kinematics -----------------------------------------------------
    def _seek_dir(self, az_deg, el_deg):
        return self._seek_null(az_deg, el_deg)[0]

    def _seek_null(self, az_deg, el_deg):
        """Line-of-sight unit vector and its partials w.r.t. az and el (degrees).

        The partials are what make an analytic target velocity possible.  Without
        them the only way to get a velocity is to difference two positions, and
        differencing a clamped parameter is precisely how frame 0 of a seeker
        scene came out at 75.8 m/s while frames 2+ read 150.8 m/s.
        """
        _p, right, up, fwd = self._basis
        a, e = np.radians(az_deg), np.radians(el_deg)
        d = fwd + np.tan(a) * right + np.tan(e) * up
        n = float(np.linalg.norm(d)) + 1e-12
        u = d / n
        # d(tan x)/dx = sec^2 x, with x in radians; d/d(deg) carries pi/180.
        k = np.pi / 180.0
        dd_a = (right / max(np.cos(a) ** 2, 1e-12)) * k
        dd_e = (up / max(np.cos(e) ** 2, 1e-12)) * k
        # The derivative of a normalised vector has no radial component.
        return (u,
                (dd_a - u * float(np.dot(u, dd_a))) / n,
                (dd_e - u * float(np.dot(u, dd_e))) / n)

    def _seek_sweep(self, u):
        """(w, dw/du) of the range/LOS sweep.  Exact derivative, no clamping.

        u is deliberately *not* clamped to [0, 1]: the clip is a window onto a
        longer approach, so letting the sweep continue outside the window keeps
        the track C-infinity.  Clamping froze the geometry for t < 0 and made the
        two-sided derivative at frame 0 exactly half the true one -- 75.8 m/s
        instead of 150.8 m/s, i.e. 754 m/s^2 of acceleration between frames 0
        and 1 and nothing at all afterwards.
        """
        prof = self.seek["profile"]
        if prof == "smooth":
            return u * u * (3.0 - 2.0 * u), 6.0 * u * (1.0 - u)
        if prof in ("accelerating", "accel"):
            return u * u, 2.0 * u
        if prof == "decelerating":
            return 1.0 - (1.0 - u) ** 2, 2.0 * (1.0 - u)
        return u, 1.0

    def _los_error(self, t):
        """(d_az, d_el) pointing error of the mount, degrees.

        los_jitter_deg and wander_deg are *stabilisation* errors: the gimbal does
        not hold the line of sight perfectly.  They belong to the mount, not to
        the target.  Baking them into the target's own position (the previous
        arrangement) moved an 8 km target +/-2.8 m at 0.35 Hz, which is 13.5 m/s^2
        -- 1.4 g of lateral zigzag and 54 deg of bank on a target flying straight
        and level, and it is the single largest "the trajectory looks wrong"
        contributor that is not a trajectory bug at all.  Applying the same
        angular error to the aim instead puts an identical offset on screen while
        leaving the world track of the aircraft smooth.
        """
        s = self.seek
        j = float(s.get("los_jitter", 0.0))
        ph = 2.0 * np.pi * float(s["bob_hz"]) * float(t) + s["phase"]
        d_az = j * np.sin(ph)
        d_el = 0.6 * j * np.sin(ph)
        wd = float(self.seek.get("wander_deg", 0.0))
        if wd > 0.0:
            w = 2.0 * np.pi * float(self.seek.get("wander_hz", 0.09))
            d_az += wd * np.sin(w * t + self.seek["wander_ph"])
            d_el += wd * 0.7 * np.sin(w * 1.37 * t + self.seek["wander_ph2"])
        return d_az, d_el

    def _seek_state(self, t):
        """Seeker target position and its analytic derivative.

        One expression, differentiated once by hand, so position, velocity and
        therefore Doppler cannot disagree; see the derivative contract in
        simkit/motion.py.
        """
        s = self.seek
        inv = 1.0 / self.dur if self.dur > 1e-9 else 0.0
        u = float(t) * inv
        w, wd = self._seek_sweep(u)
        wd *= inv
        r = s["r0"] + (s["r1"] - s["r0"]) * w
        rdot = (s["r1"] - s["r0"]) * wd
        az = s["az0"] + (s["az1"] - s["az0"]) * w
        el = s["el0"] + (s["el1"] - s["el0"]) * w
        az_dot = (s["az1"] - s["az0"]) * wd
        el_dot = (s["el1"] - s["el0"]) * wd
        u_hat, du_daz, du_del = self._seek_null(az, el)
        cam_p = self.cam_pos_at(t)
        cam_v = (self.cam_advance if self.cam_advance is not None
                 else np.zeros(3))
        p = cam_p + u_hat * r
        v = cam_v + u_hat * rdot + (du_daz * az_dot + du_del * el_dot) * r
        if s["keep_z"] and self.ground_z is not None:
            p[2] = float(self.ground_z) + float(self.target.get("z_offset_m", 0.0))
            # A pinned height is a sliding constraint, not a ride: whatever the
            # free solve asked for vertically is removed from the velocity too,
            # otherwise Doppler keeps reporting a climb the position does not have.
            v[2] = 0.0
        return p, v

    def target_pos(self, t):
        # One definition, read back off the primary target's Track: the seeker
        # back-solve, the platform's stationary target and the world path are
        # all Tracks, so a consumer can never be looking at a second copy.
        return self.track_target.pos(float(t))

    def target_vel(self, t, h=2e-3):
        return self.track_target.vel(float(t))

    def target_yaw(self, t):
        v = self.target_vel(t)
        if float(np.hypot(v[0], v[1])) < 1e-6:
            return float(self.target.get("rot", 0.0))
        return float(np.degrees(np.arctan2(float(v[1]), float(v[0]))))

    def cam_pos_at(self, t):
        """Line-of-sight origin. Static in the legacy modes, the track in flight."""
        if self.mode == "platform":
            return self.path.pos(float(t))
        if self.mode == "independent_flight":
            return self.observer_path.pos(float(t))
        if self.cam_advance is not None:
            # Pursuit: the mount advances along its own line of sight, so both
            # the aim point and every sensor ride the same translation and the
            # target keeps its authored tail view while the range closes.
            return self.cam_pos + self.cam_advance * float(t)
        return self.cam_pos

    def camera_velocity(self, t, h=1e-3):
        """Derivative of this clip's mount pose, never adjacent batch frames.

        In particular a singleton chunk and a clip boundary have the same
        velocity as the unsplit run. Smooth mount vibration is included.
        """
        return (np.asarray(self.camera(t + h)[0]) -
                np.asarray(self.camera(t - h)[0])) / (2.0 * h)

    def range_for_time(self, t):
        return float(np.linalg.norm(self.target_pos(t) - self.cam_pos_at(t)))

    def px_for_time(self, t):
        size = getattr(self, "size_used", self.size_m)
        return px_for_range(size, self.range_for_time(t), self.fov, self.width)

    # ---- camera ---------------------------------------------------------
    def camera(self, t):
        if self.mode == "independent_flight":
            jit, dyaw = self.rig.jitter(t)
            pos = self.observer_path.pos(t) + jit
            # Ground-truth-assisted pointing is an explicit simulation
            # assumption, not a tracker. Extend the analytic path before t=0
            # for a continuous initial delayed look direction.
            aim = self.path.pos(t - self.gimbal_lag)
            d = aim - pos
            a = np.radians(dyaw)
            ca, sa = np.cos(a), np.sin(a)
            aim = pos + np.array([ca*d[0] - sa*d[1], sa*d[0] + ca*d[1], d[2]])
            return pos, aim
        if self.mode == "seeker":
            lag = float((self.spec.get("seeker") or {}).get("lag_s", 0.12))
            pos, _dyaw = self.rig.jitter(t)
            pos = self.cam_pos_at(t) + pos
            # The lag is a pure delay of an analytic state, so negative time is
            # meaningful: clamping it at zero froze the aim for the first lag_s
            # of every clip and then released it, which is a step in line-of-sight
            # rate -- the mount's version of the frame-0 half-velocity bug.
            aim = self.target_pos(t - lag)
            d_az, d_el = self._los_error(t)
            if d_az or d_el:
                _p, right, up, _f = self._basis
                d = float(np.linalg.norm(aim - pos))
                aim = aim + (np.tan(np.radians(d_az)) * right +
                             np.tan(np.radians(d_el)) * up) * d
            return pos, aim
        if self.mode == "platform":
            # camera rides the track; the mount adds its own vibration, then the
            # gimbal looks where the target *was* lag_s ago (causal, chunk-safe)
            jit, _dyaw = self.rig.jitter(t)
            pos = self.path.pos(t) + jit
            aim = self.target_pos(t - self.gimbal_lag)
            wd = float(self.wander.get("deg", 0.0))
            if wd > 0.0:
                w = 2.0 * np.pi * float(self.wander.get("hz", 0.09))
                azw = wd * np.sin(w * t + self.wander["ph"])
                elw = wd * 0.7 * np.sin(w * 1.37 * t + self.wander["ph2"])
                _p, right, up, _f = self._basis
                d = float(np.linalg.norm(aim - pos))
                aim = aim + (np.tan(np.radians(azw)) * right +
                             np.tan(np.radians(elw)) * up) * d
            return pos, aim
        if self.mode == "track":
            # Mount stays put, gimbal follows the target.
            pos, _aim = self.rig.pose(t, self.cam0)
            return pos, self.target_pos(t)
        pos, aim = self.rig.pose(t, self.cam0)
        return pos, aim

    def roll_deg(self, t):
        """Airframe bank about the view axis, from the lateral acceleration.

        A coordinated turn satisfies a_lat = g*tan(bank), so the raw angle is
        atan(a_lat/g). The airframe cannot bank beyond bank_max_deg, and a hard
        clamp would turn the roll into a square wave, so the limit is a smooth
        saturation (tanh) instead: linear for small a_lat, asymptotic at the
        limit, and never identically zero when the track weaves.
        """
        if self.mode != "platform":
            return 0.0
        bmax = float(getattr(self, "bank_max", 0.0))
        if bmax <= 0.0:
            return 0.0
        a = float(self.path.lateral_accel(t))
        raw = float(np.degrees(np.arctan2(a, 9.81)))
        return float(bmax * np.tanh(raw / bmax))

    def _view_look(self, pos, aim, view, t):
        """Search scan for a view: yaw offset around the world vertical."""
        sw = float(view.get("sweep_deg", 0.0))
        if sw <= 0.0:
            return np.asarray(aim, float)
        ph = float(self.wander.get("ph2", 0.0))
        ang = np.radians(sw * np.sin(2.0 * np.pi * float(view.get("sweep_hz", 0.05)) * t + ph))
        v = np.asarray(aim, float) - np.asarray(pos, float)
        ca, sa = np.cos(ang), np.sin(ang)
        return np.asarray(pos, float) + np.array([ca * v[0] - sa * v[1],
                                                  sa * v[0] + ca * v[1], v[2]])

    # ---- per-frame state ------------------------------------------------
    def state(self, i, frame):
        t = float(i) * self.dt
        wst = self.wstates[min(int(i), len(self.wstates) - 1)]
        # The frame config carries this clip's Track set as an attribute (not as
        # a mapping entry): the ECM layer needs the parent's analytic state at
        # the release instant and at lagged times, and dataset.py deep-copies
        # the mapping for multi-band IR, which a live Track would break.
        fr_cfg = _FrameConfig(json.loads(json.dumps(self.cfg)))
        fr_cfg.tracks = self.tracks
        # The frame clock the ECM layer reads (ecm.frame_time).  dataset.py sets
        # the same key before building the world; setting it here as well means
        # a caller that builds the world straight from state() (tests, tools,
        # the manifest writer) still gets every entity at THIS frame's t instead
        # of at the figure-0 pose.
        fr_cfg["_ecm_t"] = t
        fr_cfg["weather"] = dict(wst)
        fr_cfg["_weather"] = weather.derive(wst)
        air = fr_cfg.setdefault("air", {})
        air["temp_c"] = float(wst["air_temp_c"])
        air["rh"] = float(wst["rh"])
        air["wind_mps"] = float(wst["wind_ms"])
        # The per-frame noise seed used to be the frame index alone, so two
        # different scenes produced byte-identical RD/IR noise on the same frame
        # (a QA pass found 2 of 40 RD images duplicated). Mix in the run seed.
        fr_cfg["_frame_seed"] = self.base_seed * 100003 + int(frame)

        pos, aim = self.camera(t)
        sensor_velocity = self.camera_velocity(t)
        delta = np.asarray(pos, float) - self.cam_pos
        for skey, sc in (fr_cfg.get("sensors") or {}).items():
            if not isinstance(sc, dict) or "pos" not in sc:
                continue
            base = np.array((self.cfg.get("sensors") or {}).get(skey, {}).get("pos",
                                                                            sc["pos"]), float)
            sc["pos"] = [float(v) for v in (base + delta)]
            # every sensor on the turret is co-boresighted with the primary
            sc["look_at"] = [float(v) for v in aim]
            sc["vel"] = [float(v) for v in sensor_velocity]

        # Per-view state. Same mount position for every view; only the optics
        # (fov / resolution) and the aim (search scan) differ.
        roll = round(self.roll_deg(t), 3)
        rng_m = self.range_for_time(t)
        size = float(getattr(self, "size_used", self.size_m))
        views, view_px = {}, {}
        for vname, vspec in self.views.items():
            vaim = self._view_look(pos, aim, vspec, t)
            views[vname] = dict(
                pos=[round(float(x), 4) for x in pos],
                look_at=[round(float(x), 4) for x in vaim],
                fov_deg=float(vspec["fov_deg"]),
                width=int(vspec["width"]),
                height=int(vspec["height"]),
                tag=str(vspec["tag"]),
                roll_deg=roll,
                sensor=str(vspec["sensor"]),
            )
            view_px[vname] = float(round(px_for_range(size, rng_m,
                                                     vspec["fov_deg"], vspec["width"]), 2))

        jit, poses = {}, {}
        for spec in self.cfg.get("objects", []):
            name = spec.get("name")
            P, V, yaw = self.object_state(spec, t)
            pitch, bank = self.object_attitude(spec, t)
            base = np.array(spec.get("pos", [0, 0, 0]), float)
            jit[name] = dict(dpos=[float(P[0] - base[0]), float(P[1] - base[1]),
                                   float(P[2] - base[2])],
                             dyaw=float(yaw - float(spec.get("rot", 0.0))),
                             vel=[float(V[0]), float(V[1]), float(V[2])],
                             pitch=pitch, roll=bank)
            poses[name] = dict(pos=[round(float(v), 4) for v in P], yaw=round(float(yaw), 3),
                               pitch=pitch, roll=bank, velocity_mps=[float(v) for v in V])
        state = dict(frame=int(frame), t=t, jitter=jit, cfg=fr_cfg, weather=dict(wst),
                    poses=poses, clip=self.clip, sequence_start=(int(i) == 0),
                    sequence_timestamp_s=int(frame)*self.dt,
                    range_m=round(rng_m, 2),
                    target_px=round(self.px_for_time(t), 1),
                    views=views, view_px=view_px,
                    sensor_velocity_mps=[float(v) for v in sensor_velocity],
                    view_range_m=round(rng_m, 2))
        if (self.cfg.get("optical") or {}).get("motion_blur"):
            exposure = float((self.cfg.get("acquisition") or {}).get("rgb_exposure_s", 0.001))
            if not np.isfinite(exposure) or not 0 < exposure <= self.dt:
                raise ValueError("RGB exposure must be positive and <= frame interval")
            samples = []
            for offset in (-exposure/2, exposure/2):
                tt = t+offset
                pp, aa = self.camera(tt)
                sample_views = {}
                for name, view in self.views.items():
                    sample_views[name] = dict(pos=pp.tolist(),
                        look_at=self._view_look(pp, aa, view, tt).tolist(), roll_deg=self.roll_deg(tt))
                sample_poses = {}
                for obj in self.cfg.get("objects", []):
                    P, _V, yaw = self.object_state(obj, tt)
                    pitch, bank = self.object_attitude(obj, tt)
                    sample_poses[obj["name"]] = dict(pos=P.tolist(), yaw=float(yaw), pitch=pitch, roll=bank)
                samples.append(dict(offset_s=offset, views=sample_views, poses=sample_poses))
            state["shutter_samples"] = samples
        return state


def clip_of(frame, fpc):
    return int(frame) // max(int(fpc), 1)


def frames_per_clip(cfg, n_frames):
    spec = cfg.get("clip") or {}
    fpc = spec.get("frames_per_clip")
    # A validation prefix must not rescale the engagement's duration/speed.
    count = int(fpc if fpc is not None else (cfg.get("frames") or {}).get("n", n_frames))
    if count < 1:
        raise ValueError("frames_per_clip must be positive")
    return count


def plan_frames(cfg, n_frames, base_seed, dt, frame_offset=0, total_frames=None):
    """Yield one state per frame.

    Exactly one ClipPlan is held at a time: a chunk that crosses a clip boundary
    rebuilds, which is required (the clip index is part of the plan) and is why
    this is a single slot rather than a growing dict. It used to be written as
    `if c not in cache: cache = {c: ...}`, which reassigned the dict on every
    build so the name promised a cache it never was. The rebuild cost is the
    real issue here (see the build_world profile in docs/架构评估.md); caching
    more clips would only hide it, so the slot is explicit about what it does.
    """
    fpc = frames_per_clip(cfg, int(total_frames or n_frames))
    current_clip = None
    current_plan = None
    for i in range(int(n_frames)):
        f = int(i) + int(frame_offset)
        c = clip_of(f, fpc)
        if c != current_clip:
            # A new clip (start of run, or a chunk straddling a boundary):
            # the old plan is meaningless because the clip index is its input.
            current_plan = ClipPlan(cfg, c, fpc, dt, base_seed)
            current_clip = c
        yield current_plan.state(f - c * fpc, f)


def summary(cfg, n_frames, base_seed, dt):
    """Per-clip facts for the UI and for QC: is this trajectory flyable?"""
    fpc = frames_per_clip(cfg, int(n_frames))
    out = []
    for c in range(max(1, int(np.ceil(n_frames / float(fpc))))):
        cp = ClipPlan(cfg, c, fpc, dt, base_seed)
        ts = np.arange(0.0, cp.dur if cp.dur > 0 else cp.dt, max(cp.dt, 1e-3))
        v = np.array([cp.target_vel(t) for t in ts]) if len(ts) else np.zeros((1, 3))
        sp = np.linalg.norm(v, axis=1)
        acc = 0.0
        if len(ts) > 1:
            acc = float(np.max(np.linalg.norm(np.diff(v, axis=0), axis=1) / max(cp.dt, 1e-6)))
        # In platform mode the body that actually moves is the camera, so the
        # headline speed/accel columns describe the platform and the target gets
        # its own pair of keys. Legacy modes are untouched.
        psp, pacc = sp, acc
        roll = [0.0, 0.0]
        if cp.mode == "platform" and cp.path is not None:
            pv = np.array([cp.path.vel(t) for t in ts]) if len(ts) else np.zeros((1, 3))
            psp = np.linalg.norm(pv, axis=1)
            if len(ts) > 1:
                pacc = float(np.max(np.linalg.norm(np.diff(pv, axis=0), axis=1)
                                    / max(cp.dt, 1e-6)))
            rs = [cp.roll_deg(t) for t in ts]
            roll = [round(min(rs), 2), round(max(rs), 2)]
        elif cp.mode == "independent_flight":
            pv = np.array([cp.camera_velocity(t) for t in ts])
            psp = np.linalg.norm(pv, axis=1)
            pacc = (float(np.max(np.linalg.norm(np.diff(pv, axis=0), axis=1) / cp.dt))
                    if len(ts) > 1 else 0.0)
        size = float(getattr(cp, "size_used", cp.size_m))
        plat = None
        meta = getattr(cp, "plat_meta", None)
        if meta:
            plat = {k: (round(float(v), 2) if isinstance(v, float) else v)
                    for k, v in meta.items()}
        views = [dict(name=n, tag=v_["tag"], fov_deg=v_["fov_deg"], width=v_["width"],
                      height=v_["height"],
                      target_px=[round(px_for_range(size, cp.r0, v_["fov_deg"], v_["width"]), 1),
                                 round(px_for_range(size, cp.r1, v_["fov_deg"], v_["width"]), 1)])
                 for n, v_ in cp.views.items()]
        out.append(dict(
            clip=c, frames=cp.fpc, mode=cp.mode,
            weather=(cp.wstates[0].get("name") if cp.wstates else None),
            range_m=[round(cp.r0, 1), round(cp.r1, 1)],
            target_px=[round(cp.px0, 1), round(cp.px1, 1)],
            speed_mps=round(float(np.median(psp)), 2),
            accel_mps2=round(pacc, 2),
            target_speed_mps=round(float(np.median(sp)), 2),
            target_accel_mps2=round(acc, 2),
            platform_speed_mps=round(float(np.median(psp)), 2)
                if cp.mode in ("platform", "independent_flight") else None,
            platform=plat,
            roll_deg=roll,
            views=views,
        ))
    return out


def _scene_path(name):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "scenes", name)


def selftest():
    """Checks that the seeker planner is self-consistent. Returns #failures.

    These are not smoke tests: each one compares numbers produced by two
    different code paths (or two different ways of slicing the same run), so a
    regression in the geometry, in the pixel law, or in chunking shows up.
    """
    fails = 0

    def report(name, ok, detail=""):
        nonlocal fails
        print("%s  %-48s %s" % ("PASS" if ok else "FAIL", name, detail))
        if not ok:
            fails += 1

    # The scene the geometry checks run on: any platform-mode scene with two
    # clips and a weave exercises every path. Picked by name so the checks fail
    # loudly rather than silently skipping if the scene set is renamed again.
    cfg = scene.load_config(_scene_path("sk3_high_air2air.json"))
    dt = float(cfg.get("frames", {}).get("dt", 0.05))
    n = int(cfg.get("frames", {}).get("n", 240))
    seed = int(cfg.get("seed", 0))
    fpc = frames_per_clip(cfg, n)
    cp = ClipPlan(cfg, 0, fpc, dt, seed)
    ts = [i * dt for i in range(fpc)]

    # 1 -- range closes monotonically ------------------------------------
    rr = [cp.range_for_time(t) for t in ts]
    steps = np.diff(np.array(rr))
    report("range closes monotonically",
           rr[0] > rr[-1] * 1.02 and float(np.max(steps)) < 0.5,
           "%.0f -> %.0f m, worst per-frame rise %+.3f m over %d frames"
           % (rr[0], rr[-1], float(np.max(steps)), fpc))

    # 2 -- pixel span follows 1/range ------------------------------------
    ta, tb = 0.0, 0.5 * (fpc - 1) * dt
    ra, rb = cp.range_for_time(ta), cp.range_for_time(tb)
    pa, pb = cp.px_for_time(ta), cp.px_for_time(tb)
    got, want = pa / pb, rb / ra
    report("target_px ~ 1/range",
           abs(got / want - 1.0) < 0.05,
           "px %.2f/%.2f = %.4f vs range %.2f/%.2f = %.4f (err %+.2f%%)"
           % (pa, pb, got, rb, ra, want, 100.0 * (got / want - 1.0)))

    # 3 -- chunking is exact ---------------------------------------------
    # A chunk that starts at frame k must reproduce frame k of the whole run,
    # including mid-clip frames where t != 0 and the geometry is not a reset.
    all_states = list(plan_frames(cfg, n, seed, dt, frame_offset=0, total_frames=n))
    chunk_ok, chunk_detail = True, []
    for k in (fpc // 2, fpc, fpc + 55):
        whole = all_states[k]
        part = list(plan_frames(cfg, 1, seed, dt, frame_offset=k, total_frames=n))[0]
        same = (whole["t"] == part["t"]
                and whole["range_m"] == part["range_m"]
                and whole["view_px"] == part["view_px"]
                and whole["frame"] == part["frame"] == k)
        chunk_ok = chunk_ok and same
        chunk_detail.append("f%d t=%.2f r=%.2f" % (k, whole["t"], whole["range_m"]))
    report("chunked run == whole run (frames %d/%d/%d)"
           % (fpc // 2, fpc, fpc + 55),
           chunk_ok, " | ".join(chunk_detail))

    # 4 -- views match the config and the optics law ---------------------
    st = all_states[0]
    want_views = {}
    for v in (cfg.get("clip") or {}).get("views") or []:
        want_views[v["name"]] = v
    names = list(st["views"].keys())
    cfg_ok = (len(names) == len(want_views) >= 2
              and all(st["views"][k]["tag"] == want_views[k]["tag"]
                      and abs(st["views"][k]["fov_deg"] - want_views[k]["fov_deg"]) < 1e-9
                      and st["views"][k]["width"] == want_views[k]["width"]
                      and st["views"][k]["height"] == want_views[k]["height"]
                      for k in names))
    nw, ww = st["views"][names[0]], st["views"][names[1]]
    # px ~ W / tan(fov/2): same mount, same range, different optics
    want_ratio = ((nw["width"] / np.tan(np.radians(nw["fov_deg"]) / 2.0))
                  / (ww["width"] / np.tan(np.radians(ww["fov_deg"]) / 2.0)))
    got_ratio = st["view_px"][names[0]] / st["view_px"][names[1]]
    # at equal width the span ratio is the inverse FOV ratio
    inv_fov = np.tan(np.radians(ww["fov_deg"]) / 2.0) / np.tan(np.radians(nw["fov_deg"]) / 2.0)
    fov_ratio = ww["fov_deg"] / nw["fov_deg"]
    report("views: count/fov/tag + px ~ W/tan(fov/2)",
           cfg_ok and abs(got_ratio / want_ratio - 1.0) < 0.02
           and abs(inv_fov / fov_ratio - 1.0) < 0.05,
           "%s px %s ratio %.3f vs %.3f | equal-width ratio %.2f vs 1/fov %.2f"
           % (names, st["view_px"], got_ratio, want_ratio, inv_fov, fov_ratio))

    # 5 -- bank stays inside its limit and actually moves ----------------
    # A scene that does not weave must bank exactly zero; one that weaves must
    # stay inside +-bank_max while crossing zero.
    rolls = [cp.roll_deg(t) for t in ts]
    bmax = float(getattr(cp, "bank_max", 0.0))
    weave_m = float((((cfg.get("clip") or {}).get("seeker") or {})
                     .get("platform") or {}).get("weave_amp_m", 0.0))
    inside = max(abs(r) for r in rolls) <= bmax + 1e-6
    if weave_m > 0.0 and bmax > 0.0:
        moves = float(np.std(rolls)) > 0.5 and max(rolls) > 0.0 > min(rolls)
    else:
        moves = max(abs(r) for r in rolls) == 0.0
    report("bank within +-%.0f deg (weave %.0f m)" % (bmax, weave_m),
           inside and moves,
           "roll %+.2f..%+.2f deg, sd %.2f, bank_max %.0f"
           % (min(rolls), max(rolls), float(np.std(rolls)), bmax))

    # 6 -- clips of one scene are different engagements ------------------
    cps = [ClipPlan(cfg, c, fpc, dt, seed) for c in (0, 1)]
    d_range = abs(cps[0].plat_meta["r0_m"] - cps[1].plat_meta["r0_m"])
    d_head = abs(cps[0].plat_meta["heading_deg"] - cps[1].plat_meta["heading_deg"])
    report("two clips differ (>=300 m slant or >=8 deg bearing)",
           d_range >= 300.0 or d_head >= 8.0,
           "r0 %.0f / %.0f (d %.0f m), heading %.1f / %.1f (d %.1f deg)"
           % (cps[0].plat_meta["r0_m"], cps[1].plat_meta["r0_m"], d_range,
              cps[0].plat_meta["heading_deg"], cps[1].plat_meta["heading_deg"], d_head))

    # 7 -- standoff_end_m is solved, not assumed -------------------------
    from .motion import DiveWeave
    dw = DiveWeave([0.0, -4400.0, 1900.0], [0.0, 0.0, 900.0], 180.0, 90.0, -14.0,
                   18.0, 0.22, 6.0, 3300.0, phase=0.3)
    got_so = float(np.linalg.norm(dw.pos(dw.t_end) - dw.target_at(dw.t_end)))
    report("standoff_end_m solved onto the sphere",
           abs(got_so - 3300.0) < 0.5,
           "asked 3300.00 m, got %.3f m at v_track %.2f m/s" % (got_so, dw.speed_track))

    # 8 -- legacy scenes keep the single-view path -----------------------
    lcfg = scene.load_config(_scene_path("sk_air_uav.json"))
    lcp = ClipPlan(lcfg, 0, frames_per_clip(lcfg, lcfg["frames"]["n"]),
                   float(lcfg["frames"]["dt"]), int(lcfg.get("seed", 0)))
    lst = lcp.state(0, 0)
    lv = lst["views"]
    lname = list(lv.keys())[0]
    report("legacy scene: one view, tag='', roll=0",
           len(lv) == 1 and lv[lname]["tag"] == "" and lv[lname]["roll_deg"] == 0.0
           and abs(lst["view_px"][lname] - lst["target_px"]) <= 0.05
           and abs(lv[lname]["fov_deg"] - lcp.fov) < 1e-9
           and lv[lname]["width"] == lcp.width,
           "%s fov %.1f w %d px %.1f (target_px %.1f)"
           % (lname, lv[lname]["fov_deg"], lv[lname]["width"],
              lst["view_px"][lname], lst["target_px"]))
    return fails
