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
"""
import json

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

        if seeker:
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
        return self.path.pos(t)

    def target_vel(self, t, h=2e-3):
        t = float(t)
        if self.mode == "seeker":
            return (self.target_pos(t + h) - self.target_pos(t - h)) / (2.0 * h)
        return self.path.vel(t)

    def target_yaw(self, t):
        v = self.target_vel(t)
        if float(np.hypot(v[0], v[1])) < 1e-6:
            return float(self.target.get("rot", 0.0))
        return float(np.degrees(np.arctan2(float(v[1]), float(v[0]))))

    def range_for_time(self, t):
        return float(np.linalg.norm(self.target_pos(t) - self.cam_pos))

    def px_for_time(self, t):
        size = getattr(self, "size_used", self.size_m)
        return px_for_range(size, self.range_for_time(t), self.fov, self.width)

    # ---- camera ---------------------------------------------------------
    def camera(self, t):
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
        pos, aim = self.rig.pose(t, self.cam0)
        return pos, aim

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
        fr_cfg["_frame_seed"] = int(frame)

        pos, aim = self.camera(t)
        delta = np.asarray(pos, float) - self.cam_pos
        for skey, sc in (fr_cfg.get("sensors") or {}).items():
            if not isinstance(sc, dict) or "pos" not in sc:
                continue
            base = np.array((self.cfg.get("sensors") or {}).get(skey, {}).get("pos",
                                                                            sc["pos"]), float)
            sc["pos"] = [float(v) for v in (base + delta)]
            # every sensor on the turret is co-boresighted with the primary
            sc["look_at"] = [float(v) for v in aim]

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
            poses[name] = dict(pos=[round(float(v), 4) for v in P], yaw=round(float(yaw), 3))
        return dict(frame=int(frame), t=t, jitter=jit, cfg=fr_cfg, weather=dict(wst),
                    poses=poses, clip=self.clip,
                    range_m=round(self.range_for_time(t), 2),
                    target_px=round(self.px_for_time(t), 1))


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
        out.append(dict(
            clip=c, frames=cp.fpc, mode=cp.mode,
            weather=(cp.wstates[0].get("name") if cp.wstates else None),
            range_m=[round(cp.r0, 1), round(cp.r1, 1)],
            target_px=[round(cp.px0, 1), round(cp.px1, 1)],
            speed_mps=round(float(np.median(sp)), 2),
            accel_mps2=round(acc, 2),
        ))
    return out
