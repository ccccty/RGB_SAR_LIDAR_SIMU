"""Continuous target trajectories and sensor rigs.

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
"""
import numpy as np


def _unit(v, fallback=(1.0, 0.0, 0.0)):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else np.asarray(fallback, float)


def _yaw_of(v):
    """Heading in degrees, 0 = +x, counter-clockwise. Used for the label yaw."""
    return float(np.degrees(np.arctan2(float(v[1]), float(v[0]))))


class Path:
    """Base class: pos(t) and vel(t) in world metres/seconds."""

    mode = "path"

    def pos(self, t):
        raise NotImplementedError

    def vel(self, t):
        raise NotImplementedError

    def yaw(self, t):
        v = self.vel(t)
        if float(np.hypot(v[0], v[1])) < 1e-6:
            return None            # caller keeps the previous heading
        return _yaw_of(v)

    def sample(self, t, dt=1e-3):
        """Finite-difference velocity, for paths whose derivative is fiddly."""
        h = max(float(dt), 1e-4)
        return (self.pos(t + h) - self.pos(t - h)) / (2.0 * h)

    def speed(self, t):
        return float(np.linalg.norm(self.vel(t)))


class Static(Path):
    """Parked target. Small bob keeps it from being pixel-identical forever."""
    mode = "static"

    def __init__(self, p0, bob_m=0.0, bob_hz=0.7, phase=0.0):
        self.p0 = np.asarray(p0, float)
        self.a = float(bob_m)
        self.w = 2.0 * np.pi * float(bob_hz)
        self.ph = float(phase)

    def pos(self, t):
        if self.a <= 0.0:
            return self.p0.copy()
        d = self.a * np.sin(self.w * t + self.ph)
        return self.p0 + np.array([d * 0.4, d * 0.4, d])

    def vel(self, t):
        if self.a <= 0.0:
            return np.zeros(3)
        d = self.a * self.w * np.cos(self.w * t + self.ph)
        return np.array([d * 0.4, d * 0.4, d])


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


class Arc(Path):
    """Coordinated turn: constant speed, constant turn rate, optional climb.

    Speed and turn rate stay constant, so lateral acceleration is v*omega: an
    aircraft at 30 m/s and 15 deg/s pulls about 2.5 g, which is a realistic
    envelope rather than an arbitrary curve.
    """
    mode = "arc"

    def __init__(self, centre, radius, omega, z0=0.0, vz=0.0, phase=0.0):
        self.c = np.asarray(centre, float)
        self.r = float(radius)
        self.w = float(omega)
        self.z0, self.vz, self.ph = float(z0), float(vz), float(phase)

    def pos(self, t):
        a = self.w * t + self.ph
        return np.array([self.c[0] + self.r * np.cos(a),
                         self.c[1] + self.r * np.sin(a),
                         self.z0 + self.vz * t])

    def vel(self, t):
        a = self.w * t + self.ph
        return np.array([-self.r * self.w * np.sin(a),
                         self.r * self.w * np.cos(a), self.vz])


class Figure8(Path):
    """Lissajous '8': smooth, bounded, and non-circular so heading keeps changing."""
    mode = "figure8"

    def __init__(self, centre, ax, ay, omega, z0=0.0, vz=0.0, phase=0.0):
        self.c = np.asarray(centre, float)
        self.ax, self.ay = float(ax), float(ay)
        self.w = float(omega)
        self.z0, self.vz, self.ph = float(z0), float(vz), float(phase)

    def pos(self, t):
        a = self.w * t + self.ph
        return np.array([self.c[0] + self.ax * np.sin(a),
                         self.c[1] + self.ay * np.sin(2.0 * a),
                         self.z0 + self.vz * t])

    def vel(self, t):
        a = self.w * t + self.ph
        return np.array([self.ax * self.w * np.cos(a),
                         2.0 * self.ay * self.w * np.cos(2.0 * a), self.vz])


class Weave(Path):
    """Straight leg with a sinusoidal lateral weave -- the classic evasive run.

    Analytic and C-infinity, so a tracking model sees a real bounded lateral
    acceleration instead of the teleporting target the old generator produced.
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
        s = np.sin(self.w * t + self.ph)
        return self.p0 + self.v * t + self.lat * (self.a * s) + np.array([0.0, 0.0, self.za * s])

    def vel(self, t):
        c = self.w * np.cos(self.w * t + self.ph)
        return self.v + self.lat * (self.a * c) + np.array([0.0, 0.0, self.za * c])


class DiveWeave(Path):
    """Diving, weaving *platform* track: the camera moves, the target does not.

    The old seeker mode kept the camera parked and dragged the target towards it,
    so the background never moved and every frame showed the same patch of
    ground.  A real imaging seeker is the other way round: the airframe flies a
    dive at ~180 m/s while a small lateral weave keeps the LOS from being a
    straight line.  That ground rush is exactly the cue motion-based detectors
    are trained on, so it has to come out of the geometry, not out of noise.

    Geometry
    --------
    Track direction ``u`` is fixed by heading and dive angle, and is a unit
    vector, so the along-track speed is a true speed.  The weave is horizontal
    and ``u . lat == 0`` exactly, i.e. the sinusoid is perpendicular to the
    flight path rather than to its horizontal projection.

        pos(t) = p0 + u * (v * t) + lat * A * sin(w t + ph)
        vel(t) =        u *  v     + lat * A w cos(w t + ph)      (exact d/dt)
        acc(t) =                   - lat * A w^2 sin(w t + ph)

    ``|vel|`` is therefore ``sqrt(v^2 + (A w cos)^2)``: constant to second order
    (the RMS speed equals ``v`` when ``v`` is the configured speed), never a
    finite difference of the position.

    Standoff solve
    --------------
    ``p0``, the dive and the clip length are given, but the distance at which the
    run must end is also given.  Those over-determine the track, so the
    along-track speed is solved backwards from ``t_end`` and ``standoff_end_m``
    instead of being assumed: the end point is the intersection of the track
    line with the standoff sphere, a quadratic, so it stays closed form.
    """

    mode = "dive_weave"

    def __init__(self, origin, target, speed_mps=180.0, heading_deg=90.0,
                 dive_deg=-14.0, weave_amp_m=18.0, weave_hz=0.22,
                 t_end=6.0, standoff_end_m=None, phase=0.0,
                 target_vel=None, solve="standoff"):
        self.p0 = np.asarray(origin, float)
        self.tgt = np.asarray(target, float)
        self.tvel = np.zeros(3) if target_vel is None else np.asarray(target_vel, float)
        self.dur = max(float(t_end), 1e-6)
        self.heading_deg = float(heading_deg)
        self.dive_deg = float(dive_deg)

        hd, dv = np.radians(self.heading_deg), np.radians(self.dive_deg)
        self.u = np.array([np.cos(dv) * np.cos(hd),
                           np.cos(dv) * np.sin(hd),
                           np.sin(dv)])
        # horizontal lateral: exactly perpendicular to u, so the weave cannot
        # change the along-track speed
        self.lat = _unit(np.cross(self.u, np.array([0.0, 0.0, 1.0])), (1.0, 0.0, 0.0))
        self.A = float(weave_amp_m)
        self.w = 2.0 * np.pi * float(weave_hz)
        self.ph = float(phase)
        self.speed_cfg = float(speed_mps)
        self.standoff_cfg = None if standoff_end_m is None else float(standoff_end_m)
        self.solve = str(solve).lower()

        self.speed_track = self._solve_track_speed()
        self.t_end = self.dur
        self.standoff_actual = float(np.linalg.norm(
            self.pos(self.dur) - self.target_at(self.dur)))

    # ---- geometry -------------------------------------------------------
    def target_at(self, t):
        return self.tgt + self.tvel * float(t)

    def _solve_track_speed(self):
        """Along-track speed: RMS-equal to the configured speed, or solved from
        the standoff sphere when a standoff is given."""
        keep = np.sqrt(max(self.speed_cfg ** 2 - 0.5 * (self.A * self.w) ** 2, 1e-6))
        if self.standoff_cfg is None or self.solve in ("speed", "keep", "rms"):
            return float(keep)
        s_end = np.sin(self.w * self.dur + self.ph)
        B = (self.p0 - self.target_at(self.dur)) + self.lat * (self.A * s_end)
        along = float(np.dot(self.u, B))
        disc = along * along - float(np.dot(B, B)) + self.standoff_cfg ** 2
        if disc < 0.0:
            # the track line misses the standoff sphere: standoff below the
            # perpendicular offset, nothing to solve, keep the configured speed
            return float(keep)
        x = -along - np.sqrt(disc)
        if x <= 1e-6:
            # already inside the standoff sphere at t=0: take the far crossing
            x = -along + np.sqrt(disc)
        if x <= 1e-6:
            return float(keep)
        return float(x / self.dur)

    # ---- analytic kinematics -------------------------------------------
    def pos(self, t):
        t = float(t)
        return (self.p0 + self.u * (self.speed_track * t)
                + self.lat * (self.A * np.sin(self.w * t + self.ph)))

    def vel(self, t):
        t = float(t)
        return (self.u * self.speed_track
                + self.lat * (self.A * self.w * np.cos(self.w * t + self.ph)))

    def accel(self, t):
        t = float(t)
        return self.lat * (-self.A * self.w * self.w * np.sin(self.w * t + self.ph))

    def lateral_accel(self, t):
        """Signed lateral acceleration (m/s^2, +ve to the right of the track)."""
        return float(np.dot(self.accel(t), self.lat))

    def altitude(self, t):
        return float(self.pos(t)[2])

    def range_at(self, t):
        return float(np.linalg.norm(self.target_at(t) - self.pos(t)))


class Waypoints(Path):
    """Centripetal Catmull-Rom through hand-placed waypoints, constant speed.

    Centripetal parameterisation is used because uniform Catmull-Rom forms cusps
    and self-intersections when consecutive points are unevenly spaced, which
    shows up as a target that stutters in place.
    """

    mode = "waypoints"

    def __init__(self, pts, speed=8.0, samples=32, phase=0.0, loop=True):
        P = [np.asarray(p, float) for p in pts]
        if len(P) < 2:
            raise ValueError("waypoints needs at least two points")
        if loop:
            P = [P[-1]] + P + [P[0], P[1]]
        else:
            P = [P[0]] + P + [P[-1]]
        dense = []
        for i in range(1, len(P) - 2):
            p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
            t0 = 0.0
            t1 = t0 + max(float(np.linalg.norm(p1 - p0)), 1e-4) ** 0.5
            t2 = t1 + max(float(np.linalg.norm(p2 - p1)), 1e-4) ** 0.5
            t3 = t2 + max(float(np.linalg.norm(p3 - p2)), 1e-4) ** 0.5
            for s in np.linspace(t1, t2, int(samples), endpoint=(i == len(P) - 3)):
                dense.append(_catmull(p0, p1, p2, p3, t0, t1, t2, t3, float(s)))
        self.p = np.asarray(dense, float)
        seg = np.linalg.norm(np.diff(self.p, axis=0), axis=1)
        seg = np.maximum(seg, 1e-9)
        self.s = np.concatenate([[0.0], np.cumsum(seg)])
        self.length = float(self.s[-1])
        self.v = float(speed)
        self.ph = float(phase)

    def _at(self, arc):
        arc = np.mod(arc, self.length)
        x = np.interp(arc, self.s, self.p[:, 0])
        y = np.interp(arc, self.s, self.p[:, 1])
        z = np.interp(arc, self.s, self.p[:, 2])
        return np.array([x, y, z])

    def pos(self, t):
        return self._at(self.v * t + self.ph)

    def vel(self, t):
        return self.sample(t)


def _catmull(p0, p1, p2, p3, t0, t1, t2, t3, t):
    """Barry-Goldman form of a non-uniform Catmull-Rom segment."""
    a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
    a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
    a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
    b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
    b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
    return (t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2


def make_path(spec, ground_z=None):
    """Build a Path from a scene config block.

    ground_z pins the trajectory to a surface (a rover cannot fly); passing None
    leaves the z the mode itself produces.
    """
    spec = spec or {}
    mode = str(spec.get("mode", "line")).lower()
    p0 = np.array(spec.get("origin", [0.0, 0.0, 0.0]), float)
    if ground_z is not None:
        p0[2] = float(ground_z)

    if mode in ("static", "hover", "parked"):
        return Static(p0, bob_m=float(spec.get("bob_m", 0.0)),
                      bob_hz=float(spec.get("bob_hz", 0.7)),
                      phase=float(spec.get("phase", 0.0)))

    if mode in ("line", "straight"):
        return Line(p0, np.array(spec.get("vel", [1.0, 0.0, 0.0]), float))

    if mode in ("weave", "evade", "maneuver"):
        return Weave(p0, np.array(spec.get("vel", [1.0, 0.0, 0.0]), float),
                     float(spec.get("amplitude_m", 2.0)),
                     hz=float(spec.get("hz", 0.25)),
                     z_amp=float(spec.get("z_amplitude_m", 0.0)),
                     phase=float(spec.get("phase", 0.0)))

    if mode in ("arc", "circle", "orbit"):
        r = float(spec.get("radius_m", 50.0))
        omega = float(spec.get("speed_mps", 10.0)) / max(r, 1e-6)
        centre = np.array(spec.get("centre", p0), float)
        if ground_z is not None:
            centre[2] = float(ground_z)
        return Arc(centre, r, float(spec.get("omega_rad_s", omega)),
                   z0=float(ground_z if ground_z is not None else p0[2]),
                   vz=float(spec.get("climb_mps", 0.0)),
                   phase=float(spec.get("phase", 0.0)))

    if mode in ("figure8", "figure-8", "eight"):
        ax = float(spec.get("ax_m", 40.0))
        omega = float(spec.get("omega_rad_s",
                               float(spec.get("speed_mps", 10.0)) / max(ax, 1e-6)))
        centre = np.array(spec.get("centre", p0), float)
        if ground_z is not None:
            centre[2] = float(ground_z)
        return Figure8(centre, ax, float(spec.get("ay_m", ax)), omega,
                       z0=float(ground_z if ground_z is not None else p0[2]),
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
                         solve=str(spec.get("solve", "standoff")))

    if mode in ("waypoints", "path", "spline"):
        pts = spec.get("points") or []
        if len(pts) < 2:
            raise ValueError("waypoints mode needs at least two 'points'")
        pts = [np.asarray(p, float) for p in pts]
        if ground_z is not None:
            for p in pts:
                p[2] = float(ground_z)
        return Waypoints(pts, speed=float(spec.get("speed_mps", 8.0)),
                         loop=bool(spec.get("loop", True)),
                         phase=float(spec.get("phase", 0.0)))

    raise ValueError("unknown trajectory mode: %s" % mode)


class Rig:
    """Sensor mount: where each camera points, as a smooth function of time.

    'track' aims at the target as it was lag seconds ago.  A pure time delay is
    used rather than a stateful filter on purpose: it is causal, analytic, and
    independent of where a chunk boundary falls, so a run split into pieces still
    produces one continuous pan.  A real gimbal behaves the same way -- the mount
    lags the target.
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
            tt = max(0.0, t - self.lag)
            aim = self.path.pos(tt) + self.lead
            if self.mode == "follow":
                # chase camera: hang back along the target's own heading
                v = self.path.vel(tt)
                h = _unit(v, (1.0, 0.0, 0.0))
                pos = aim - h * float(self.base.get("standoff_m", 25.0))                     + np.array([0.0, 0.0, float(self.base.get("standoff_z_m", 6.0))])
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


def yaw_rate(path, t, h=1e-3):
    """Degrees/second, for reporting whether a trajectory is flyable."""
    y0, y1 = path.yaw(t - h), path.yaw(t + h)
    if y0 is None or y1 is None:
        return 0.0
    d = (y1 - y0 + 180.0) % 360.0 - 180.0
    return float(d / (2.0 * h))
