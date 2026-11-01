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
            if len(self.targets) != 1:
                raise ValueError("independent-flight baseline requires one labelled target per scene")
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
            self.path = motion.make_path(tspec, ground_z=None)
            self.rig = motion.Rig(dict(spec.get("rig") or {}), cam0, self.path, rng)
            self.rig_lag = self.rig.lag

        self.px0 = self.px_for_time(0.0)
        self.px1 = self.px_for_time(self.dur)
        self.r0 = self.range_for_time(0.0)
        self.r1 = self.range_for_time(self.dur)

    # ---- planning -------------------------------------------------------
    def _trajectory_from_objects(self):
        t = self.target
        p = list(t.get("pos", [0.0, 0.0, 0.0]))
        v = list(t.get("vel", [1.0, 0.0, 0.0]))
        return dict(mode="line", origin=p, vel=v)

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
        a = float(rng.uniform(lo + 0.15 * (hi - lo), hi - 0.15 * (hi - lo))) if hi > lo else lo
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

        band = sk.get("range_m") or p.get("range_m")
        r0 = None
        if band and bool(var.get("enabled", p.get("randomize", True))):
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
            phase = float(rng.uniform(0.0, 6.283185307179586))
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
        _p, right, up, fwd = self._basis
        ta = np.tan(np.radians(az_deg))
        te = np.tan(np.radians(el_deg))
        d = fwd + ta * right + te * up
        return d / (np.linalg.norm(d) + 1e-12)

    def target_pos(self, t):
        t = float(t)
        if self.mode == "seeker":
            s = self.seek
            u = 0.0 if self.dur <= 1e-9 else min(max(t / self.dur, 0.0), 1.0)
            prof = s["profile"]
            if prof == "smooth":
                w = u * u * (3.0 - 2.0 * u)
            elif prof in ("accelerating", "accel"):
                w = u * u
            elif prof == "decelerating":
                w = 1.0 - (1.0 - u) ** 2
            else:
                w = u
            r = s["r0"] + (s["r1"] - s["r0"]) * w
            j = s["los_jitter"]
            wob = j * np.sin(2.0 * np.pi * s["bob_hz"] * t + s["phase"])
            az = s["az0"] + (s["az1"] - s["az0"]) * w + wob
            el = s["el0"] + (s["el1"] - s["el0"]) * w + 0.6 * wob
            p = self.cam_pos + self._seek_dir(az, el) * r
            if s["keep_z"] and self.ground_z is not None:
                p[2] = float(self.ground_z) + float(self.target.get("z_offset_m", 0.0))
            return p
        if self.mode == "platform":
            # the target sits (or drives slowly); the platform is the mover
            return self.tgt_pos + self.tgt_vel * t
        return self.path.pos(t)

    def target_vel(self, t, h=2e-3):
        t = float(t)
        if self.mode == "seeker":
            return (self.target_pos(t + h) - self.target_pos(t - h)) / (2.0 * h)
        if self.mode == "platform":
            return self.tgt_vel.copy()
        return self.path.vel(t)

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
            pos = self.cam_pos + pos
            aim = self.target_pos(max(0.0, t - lag))
            wd = float(self.seek.get("wander_deg", 0.0))
            if wd > 0.0:
                w = 2.0 * np.pi * float(self.seek.get("wander_hz", 0.09))
                azw = wd * np.sin(w * t + self.seek["wander_ph"])
                elw = wd * 0.7 * np.sin(w * 1.37 * t + self.seek["wander_ph2"])
                _p, right, up, _f = self._basis
                d = float(np.linalg.norm(aim - pos))
                aim = aim + (np.tan(np.radians(azw)) * right +
                             np.tan(np.radians(elw)) * up) * d
            return pos, aim
        if self.mode == "platform":
            # camera rides the track; the mount adds its own vibration, then the
            # gimbal looks where the target *was* lag_s ago (causal, chunk-safe)
            jit, _dyaw = self.rig.jitter(t)
            pos = self.path.pos(t) + jit
            aim = self.target_pos(max(0.0, t - self.gimbal_lag))
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
        fr_cfg = json.loads(json.dumps(self.cfg))
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
            if spec.get("label", True):
                P = self.target_pos(t)
                V = self.target_vel(t)
                yaw = self.target_yaw(t)
            else:
                P = np.array(spec.get("pos", [0, 0, 0]), float)
                V = np.array(spec.get("vel", [0, 0, 0]), float)
                yaw = float(spec.get("rot", 0.0))
            base = np.array(spec.get("pos", [0, 0, 0]), float)
            jit[name] = dict(dpos=[float(P[0] - base[0]), float(P[1] - base[1]),
                                   float(P[2] - base[2])],
                             dyaw=float(yaw - float(spec.get("rot", 0.0))),
                             vel=[float(V[0]), float(V[1]), float(V[2])])
            poses[name] = dict(pos=[round(float(v), 4) for v in P], yaw=round(float(yaw), 3),
                               velocity_mps=[float(v) for v in V])
        state = dict(frame=int(frame), t=t, jitter=jit, cfg=fr_cfg, weather=dict(wst),
                    poses=poses, clip=self.clip,
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
                    if obj.get("label", True):
                        P, yaw = self.target_pos(tt), self.target_yaw(tt)
                    else:
                        P = np.array(obj.get("pos", [0, 0, 0]))+np.array(obj.get("vel", [0, 0, 0]))*tt
                        yaw = obj.get("rot", 0.0)
                    sample_poses[obj["name"]] = dict(pos=P.tolist(), yaw=float(yaw))
                samples.append(dict(offset_s=offset, views=sample_views, poses=sample_poses))
            state["shutter_samples"] = samples
        return state


def clip_of(frame, fpc):
    return int(frame) // max(int(fpc), 1)


def frames_per_clip(cfg, n_frames):
    spec = cfg.get("clip") or {}
    fpc = spec.get("frames_per_clip")
    return int(fpc) if fpc else int(n_frames)


def plan_frames(cfg, n_frames, base_seed, dt, frame_offset=0, total_frames=None):
    """Yield one state per frame, clips cached so a chunk builds each once."""
    fpc = frames_per_clip(cfg, int(total_frames or n_frames))
    cache = {}
    for i in range(int(n_frames)):
        f = int(i) + int(frame_offset)
        c = clip_of(f, fpc)
        if c not in cache:
            cache = {c: ClipPlan(cfg, c, fpc, dt, base_seed)}
        yield cache[c].state(f - c * fpc, f)


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
