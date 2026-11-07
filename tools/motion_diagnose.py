"""Per-frame motion diagnostics for every scene: is this track physically possible?

Why this exists
---------------
"the trajectory looks wrong" is not a bug report anyone can act on.  This tool
turns it into numbers: for every frame of every clip it prints the position,
the published velocity, the acceleration implied by that velocity, the jerk, the
heading and its rate, the climb rate, the height above ground, the bank a
coordinated turn would need and the load factor that implies -- and then says,
per frame, which class envelope those numbers break.

Two checks matter more than the rest, and they are the ones a viewer sees:

  derivative  the published velocity must equal d(pos)/dt.  A mismatch means the
              position and the velocity are two different stories, which is
              exactly how a target acquires a 76.9 g spike on frame 0 and
              nothing on frame 3.
  continuity  the acceleration must not step.  A step is infinite jerk: the
              camera and the label both snap, and the frame looks like a cut.

The tool is importable on purpose: tests/test_motion_realism.py calls
diagnose_scene() and asserts on the returned dicts, so the numbers in
docs/运动修正报告.md and the numbers the test gates on are the same numbers.

Usage
-----
    python tools/motion_diagnose.py                       # every scene
    python tools/motion_diagnose.py scenes/demo_air.json  # one scene
    python tools/motion_diagnose.py --out out/motion.txt --json out/motion.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simkit import motion, plan, scene  # noqa: E402


def _accel_at(state_of, t, h, one_sided=False):
    """d(velocity)/dt from the *published* velocity, not from a re-derived path.

    Deliberately a finite difference: the question this tool answers is "what
    acceleration would an observer of these frames measure", so it must read the
    same velocity the renderer and the labels read.  If that velocity is not the
    derivative of the position, the error surfaces in the derivative column
    instead of being hidden by a cleverer formula.

    one_sided=True is for the first sample of a clip.  A clip is a window onto a
    track that has a beginning: the planners publish the right-hand derivative
    at t=0 (my measured case publishes 80.0 m/s there) but their position function
    is frozen for t<0.  A centred step across that edge reads
    (v(+h) - v(-h)) / 2h with v(-h) == v(0), i.e. half the real acceleration --
    it under-reported 133.4 m/s^2 as 44.9 and 80 m/s of velocity as a 40 m/s
    "derivative error".  The three-point forward formula keeps the scheme's own
    error at O(h^2), the same order the centred step has in the interior, so the
    first frame is measured as accurately as every other one.
    """
    if one_sided:
        v0 = np.asarray(state_of(t)[1], float)
        v1 = np.asarray(state_of(t + h)[1], float)
        v2 = np.asarray(state_of(t + 2.0 * h)[1], float)
        return (-3.0 * v0 + 4.0 * v1 - v2) / (2.0 * h)
    return (np.asarray(state_of(t + h)[1], float)
            - np.asarray(state_of(t - h)[1], float)) / (2.0 * h)


def _track(modes, t, dt, ground_z):
    """Sample one object over a clip and return per-frame physical quantities."""
    # Two different step sizes on purpose.
    #
    # h is what an observer of these frames can measure: accel and jerk are
    # differenced over a fraction of the frame interval, because asking "did this
    # airframe pull 6 g" means the g it pulled during the exposure.
    #
    # h_deriv answers a different question -- is the published velocity the
    # derivative of the published position -- and a coarse step answers it
    # wrongly.  At h = 0.1 s on a 60 deg turn the central difference drops an
    # O(h^2) term worth 3e-3 m/s, so the tool reported a 3e-3 "derivative error"
    # on a path whose real error is 1e-9, and a test that trusted the column
    # chased the difference scheme instead of the motion.  Small h makes the
    # scheme's own error negligible, which is the only way the column measures
    # the path.
    h = max(dt * 0.25, 1e-4)
    h_deriv = 1e-3
    rows = []
    for k in range(int(modes["frames"])):
        tt = t + k * dt
        # The first sample of the run sits on the edge of the scene clock: the
        # left neighbour t-h would be negative, where every planner here freezes
        # the position (a clip is a window onto a track that has a beginning).
        # Differencing across that edge reads a half step and reports half the
        # velocity as an "error", which is a defect of the scheme, not of the
        # motion.  Both derivatives therefore switch to a three-point forward
        # formula there; see _accel_at for the measurement behind this.
        first = (tt - h_deriv < 0.0)
        p, v, yaw = modes["state"](tt)
        a = _accel_at(modes["state"], tt, h, one_sided=first)
        a2 = _accel_at(modes["state"], tt + h, h)
        if first:
            p2 = modes["state"](tt + 2.0 * h_deriv)[0]
            p1 = modes["state"](tt + h_deriv)[0]
            # three-point forward: exact to O(h^2) on a smooth track, and it
            # never reads t < 0
            psec = (-3.0 * np.asarray(p, float) + 4.0 * np.asarray(p1, float)
                    - np.asarray(p2, float)) / (2.0 * h_deriv)
        else:
            p2 = modes["state"](tt + h_deriv)[0]
            p0 = modes["state"](tt - h_deriv)[0]
            psec = (np.asarray(p2, float) - np.asarray(p0, float)) / (2.0 * h_deriv)
        speed = float(np.linalg.norm(v))
        hspeed = float(np.hypot(v[0], v[1]))
        rows.append(dict(
            frame=k, t=round(tt, 6),
            pos=[round(float(c), 3) for c in p],
            speed_mps=round(speed, 4),
            accel_mps2=round(float(np.linalg.norm(a)), 4),
            g=round(float(np.linalg.norm([a[0], a[1], a[2] + motion.G]) / motion.G), 4),
            jerk_mps3=round(float(np.linalg.norm(a2 - a)) / (2.0 * h), 3),
            yaw_deg=None if hspeed < 1e-6 else round(float(yaw), 4),
            yaw_rate_dps=round(float(motion.turn_rate(v, a)) * 57.29577951308232, 4),
            bank_deg=round(float(motion.coordinated_bank_deg(v, a)), 4),
            pitch_deg=round(float(motion.climb_angle_deg(v)), 4),
            climb_mps=round(float(v[2]), 4),
            altitude_m=round(float(p[2]) - float(ground_z), 4),
            derivative_error_mps=round(float(np.linalg.norm(
                np.asarray(v, float) - psec)), 6),
        ))
    return rows


def _envelope_violations(rows, env):
    """Envelope verdicts, from the one implementation in simkit.motion.

    The mount is exempt from the derivative rule on purpose: a camera carrying
    rig vibration has a velocity that is sampled at a different instant from its
    positions, so the two disagree by design and the disagreement is the
    vibration, not a trajectory defect.  The position itself is still checked.
    """
    return motion.envelope_violations(
        rows, env, tolerance_mps=0.05,
        check_derivative=(env.kind != "camera"))


def _summarise(rows):
    if not rows:
        return {}
    def mx(k): return round(float(max(abs(r[k]) for r in rows)), 4)
    def mn(k): return round(float(min(r[k] for r in rows)), 4)
    return dict(
        frames=len(rows),
        speed_mps=[round(mn("speed_mps"), 3), round(max(r["speed_mps"] for r in rows), 3)],
        max_accel_mps2=mx("accel_mps2"),
        max_g=mx("g"),
        max_jerk_mps3=mx("jerk_mps3"),
        max_yaw_rate_dps=mx("yaw_rate_dps"),
        max_bank_deg=mx("bank_deg"),
        min_altitude_m=round(float(min(r["altitude_m"] for r in rows)), 3),
        max_derivative_error_mps=mx("derivative_error_mps"),
        first_frame_accel_mps2=rows[0]["accel_mps2"],
    )


def _observer_envelope(cfg):
    """Class of the camera platform itself.

    An observer that flies at 190 m/s behind a fighter is a jet, not a
    multirotor, and judging it against a 25 m/s quad envelope produced 120
    "violations" that were an artefact of the test harness rather than of the
    trajectory.  The scene gets to say what it is via observer_kinematics, and
    when it does not the declared speed decides.
    """
    clip = cfg.get("clip") or {}
    kin = dict(clip.get("observer_kinematics") or {})
    traj = clip.get("observer_trajectory") or {}
    speed = float(np.linalg.norm(np.asarray(traj.get("vel", [0.0, 0.0, 0.0]), float)))
    if not kin and not speed:
        return motion.Envelope.preset("camera")
    if max(speed, float(kin.get("max_speed_mps", 0.0))) >= 200.0:
        # A jet-rated mount still has to be a jet: the observer really is flying
        # at 190-300 m/s, so its track has to stay inside a jet envelope.
        return motion.Envelope.from_spec(dict(cls="fighter", kinematics=kin))
    # Everything below 200 m/s is a mount, not an airframe, and the camera class
    # carries the limits that mean something for one: a teleport between frames.
    return motion.Envelope.from_spec(dict(cls="camera", kinematics=kin))


def _object_envelope(cp, clip, obj):
    """The envelope an object's motion can honestly be judged against.

    Returns (envelope, source).  The order of the tests is the point: what the
    object *is* only matters once we know how it *moves*, because the same bare
    "vel" field means three different things depending on the clip.

      trajectory  the scene declares a track (objects[].trajectory, or the clip
                  trajectory when the object is flown by it) -- the author has
                  claimed an airframe can fly this, so hold it to the airframe.
      seeker      the seeker back-solves the target from range/az/el, so the
                  published speed is the real one.  A bare vel here is not a
                  placeholder, it is ignored: sk_air_uav declares vel 10 m/s and
                  actually flies 150.9 m/s, and judging the 10 would report a
                  stall on a perfectly healthy track.
      platform    the target holds station and the camera does the flying, so a
                  slow drift is the intended geometry, not a stall.
      explicit    nothing declares the motion: no trajectory, no path, no
                  seeker, so a bare vel is all there is.  That is an author's
                  stated velocity, not an aerodynamic claim, and judging it
                  against a stall floor invents an intent -- ds_air_uav drifts a
                  Global Hawk mesh across frame at 8.5 m/s and means to.
    """
    has_trajectory = bool(obj.get("trajectory"))
    has_seeker = bool((clip.get("seeker") or {}).get("profile")) or \
        cp.mode == "seeker"
    declared = bool(obj.get("kinematics") or obj.get("flight_class"))

    # An object flown by the clip's own trajectory often carries its airframe
    # class in that trajectory block rather than on itself -- demo_air_uav's
    # multirotor says "flight_class": "uav_rotor" inside clip.trajectory.  Reading
    # only the object would classify it by model name, call a 25 m/s rotor a
    # fixed-wing, and then fail the 5 m/s demo against a 20 m/s stall floor that
    # the scene never declared.
    traj_spec = dict(obj)
    if not traj_spec.get("flight_class"):
        traj = (clip.get("trajectory") or {})
        if traj.get("flight_class"):
            traj_spec["flight_class"] = traj["flight_class"]
    if obj.get("name") in getattr(cp, "object_paths", {}):
        return motion.Envelope.from_spec(traj_spec), "trajectory"
    if has_trajectory:
        return motion.Envelope.from_spec(traj_spec), "trajectory"
    if has_seeker:
        # Resolved from the seeker, so the class envelope applies to the motion
        # that is actually published, never to the vel field it overrides.
        return motion.Envelope.from_spec(traj_spec), "seeker"
    if declared:
        return motion.Envelope.from_spec(traj_spec), "declared"
    if cp.mode == "platform":
        return motion.Envelope.preset("loiter"), "platform-station"
    return motion.Envelope.preset("explicit"), "explicit"


def diagnose_scene(path, max_clips=None):
    """Diagnose every object of every clip in one scene file.

    Returns a plain dict so a test can assert on it without parsing text.
    """
    cfg = scene.load_config(path)
    clip = cfg.get("clip") or {}
    frames = cfg.get("frames") or {}
    n = int(frames.get("n", 10))
    dt = float(frames.get("dt", 0.1))
    seed = int(cfg.get("seed", 1))
    ground = cfg.get("ground") or {}
    gz = float(ground.get("z", 0.0)) if ground.get("enabled", True) else 0.0

    fpc = plan.frames_per_clip(cfg, n)
    n_clips = max(1, int(np.ceil(n / float(fpc))))
    if max_clips:
        n_clips = min(n_clips, int(max_clips))

    objects = []
    seen = set()
    for ci in range(n_clips):
        cp = plan.ClipPlan(cfg, ci, fpc, dt, seed)
        starts = ci * fpc * dt
        for obj in cfg.get("objects", []):
            name = obj.get("name", "?")
            if name in seen:
                continue
            # Platform mode means the target holds station and the camera flies
            # (plan.py says so in its own comment), so a slow drift is the
            # intended geometry and not a stall.  An object that declares its own
            # limits still gets them.
            env, source = _object_envelope(cp, clip, obj)
            rows = _track(dict(frames=fpc,
                               state=lambda t, o=obj: cp.object_state(o, t)),
                          starts, dt, gz)
            pitch, roll = cp.object_attitude(obj, starts)
            objects.append(dict(
                name=name, cls=obj.get("cls"), envelope=env.to_dict(),
                source=source, clip=ci, labelled=bool(obj.get("label", True)),
                summary=_summarise(rows),
                attitude_deg=[round(float(pitch), 3), round(float(roll), 3)],
                violations=_envelope_violations(rows, env),
                samples=rows,
            ))
            seen.add(name)
        # In platform mode the *camera* is the airframe: the ground rush, the dive
        # and the weave are the whole point of the sequence, so its own track is
        # judged against the platform envelope the scene declares.
        if cp.mode == "platform" and getattr(cp, "path", None) is not None:
            rows = _track(dict(frames=fpc,
                               state=lambda t: (np.asarray(cp.path.pos(t), float),
                                                np.asarray(cp.path.vel(t), float),
                                                cp.path.yaw(t) or 0.0)),
                          starts, dt, gz)
            # The platform's class is the planner's own decision now (it builds
            # the envelope to clip the weave and the standoff speed), so the
            # diagnostic judges the track against the limits that shaped it.
            env = getattr(cp, "plat_env", None) or motion.Envelope.preset("uav_fixed")
            objects.append(dict(
                name="platform", cls="airframe", envelope=env.to_dict(),
                source="trajectory", clip=ci, labelled=True, summary=_summarise(rows),
                attitude_deg=[round(float(motion.climb_angle_deg(cp.path.vel(starts))), 3),
                              round(float(motion.coordinated_bank_deg(
                                  cp.path.vel(starts), cp.path.accel(starts))), 3)],
                violations=_envelope_violations(rows, env), samples=rows))
        # The mount itself has to move plausibly too: a camera that jumps is a
        # broken sequence even when the target is perfect.
        if cp.mode in ("world", "independent_flight"):
            rows = _track(dict(frames=fpc,
                               state=lambda t: (np.asarray(cp.camera(t)[0], float),
                                                np.asarray(cp.camera_velocity(t), float),
                                                cp.path.yaw(t) if cp.path.yaw(t) is not None else 0.0)),
                          starts, dt, gz)
            env = _observer_envelope(cfg)
            objects.append(dict(
                name="observer", cls="camera", envelope=env.to_dict(),
                source="mount", clip=ci, labelled=False, summary=_summarise(rows),
                attitude_deg=[0.0, 0.0],
                violations=_envelope_violations(rows, env), samples=rows))

    tspec = clip.get("trajectory") or {}
    return dict(
        scene=os.path.basename(path), path=os.path.abspath(path),
        # The planner's own mode, not a guess from which config block is present:
        # a "seeker" scene whose seeker block asks for platform mode flies the
        # platform, and labelling it a seeker hides that.
        mode=cp.mode,
        trajectory=tspec.get("mode"), frames=n, dt=dt, fpc=fpc,
        seeker=clip.get("seeker"), objects=objects,
    )


def find_air_scenes(root=None):
    """Every scene that labels something airborne (or asks for a seeker view)."""
    root = root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = []
    for p in sorted(glob.glob(os.path.join(root, "scenes", "*.json"))):
        try:
            cfg = scene.load_config(p)
        except Exception:
            continue
        if not cfg.get("clip"):
            continue
        clip = cfg["clip"]
        airborne = bool(clip.get("seeker")) or any(
            str(o.get("cls", "")).lower() in motion.ENVELOPE_BY_CLASS
            and motion.ENVELOPE_BY_CLASS.get(str(o.get("cls", "")).lower(), "").startswith("uav")
            or "fighter" in str(o.get("cls", "")).lower()
            or "aircraft" in str(o.get("cls", "")).lower()
            for o in cfg.get("objects", []))
        if airborne:
            out.append(p)
    return out


def format_report(found):
    """Human-readable report: one block per scene, one line per violation."""
    lines = []
    for sc in found:
        head = "%-34s %-18s frames=%-4d dt=%-5g mode=%s" % (
            sc["scene"], str(sc["trajectory"]), sc["frames"], sc["dt"], sc["mode"])
        lines.append("=" * 100)
        lines.append(head)
        if sc.get("seeker"):
            lines.append("  seeker closing_mps=%s range_m=%s profile=%s" % (
                sc["seeker"].get("closing_mps"), sc["seeker"].get("range_m"),
                sc["seeker"].get("profile")))
        for o in sc["objects"]:
            s = o["summary"]
            bad = len(o["violations"])
            # The motion source is printed because it decides which rules ran:
            # "ok" under an explicit envelope is a self-consistency verdict, not
            # an airframe endorsement, and the two must not look alike.
            lines.append("  %-12s %-10s %-16s %s  speed %s..%s m/s  accel<=%s (%.2f g)  "
                         "jerk<=%s  yaw<=%s deg/s  bank<=%s deg  alt>=%s m  deriv=%s" % (
                             o["name"], o["envelope"]["kind"], o.get("source", "?"),
                             "FAIL" if bad else "ok  ",
                             s["speed_mps"][0], s["speed_mps"][1],
                             s["max_accel_mps2"], s["max_g"], s["max_jerk_mps3"],
                             s["max_yaw_rate_dps"], s["max_bank_deg"],
                             s["min_altitude_m"], s["max_derivative_error_mps"]))
            if s["first_frame_accel_mps2"] > 2.0 * max(s["max_accel_mps2"] * 0.5, 1.0) \
                    and s["frames"] > 2:
                lines.append("      note: frame 0 acceleration %.3f m/s^2 vs "
                             "steady-state %.3f -- boundary artefact" % (
                                 s["first_frame_accel_mps2"],
                                 float(np.median([r["accel_mps2"] for r in o["samples"][2:]]))))
            for v in o["violations"][:8]:
                lines.append("      frame %-4d %-14s %10.4f  limit %10.4f" % (
                    v["frame"], v["rule"], v["value"], v["limit"]))
            if len(o["violations"]) > 8:
                lines.append("      ... %d more violations" % (len(o["violations"]) - 8))
    total = sum(len(o["violations"]) for sc in found for o in sc["objects"])
    scenes_bad = sum(1 for sc in found if any(o["violations"] for o in sc["objects"]))
    lines.append("=" * 100)
    lines.append("%d scenes, %d with violations, %d violations total" % (
        len(found), scenes_bad, total))
    # Say out loud how many objects were judged by an airframe envelope and how
    # many only by the self-consistency rules, so "0 violations" cannot be read
    # as "every airframe was checked" when most of them were not.
    n_obj = sum(len(sc["objects"]) for sc in found)
    n_env = sum(1 for sc in found for o in sc["objects"] if o["envelope"].get("enforce"))
    lines.append("envelopes enforced on %d of %d objects; %d checked for "
                 "self-consistency only (explicit / camera)" % (n_env, n_obj, n_obj - n_env))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scenes", nargs="*", help="scene json files (default: every air scene)")
    ap.add_argument("--out", default=os.path.join("out", "motion_report.txt"))
    ap.add_argument("--json", default=os.path.join("out", "motion_report.json"))
    ap.add_argument("--all", action="store_true", help="include non-air scenes too")
    args = ap.parse_args(argv)

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    paths = args.scenes or (sorted(glob.glob(os.path.join(root, "scenes", "*.json")))
                            if args.all else find_air_scenes(root))
    found = []
    for p in paths:
        try:
            found.append(diagnose_scene(p))
        except Exception as exc:                       # a broken scene must not hide the rest
            found.append(dict(scene=os.path.basename(p), path=os.path.abspath(p),
                              error="%s: %s" % (type(exc).__name__, exc),
                              mode="error", trajectory=None, frames=0, dt=0.0,
                              seeker=None, objects=[]))
    text = format_report(found)

    def _write(path, payload):
        full = path if os.path.isabs(path) else os.path.join(root, path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as fh:
            fh.write(payload)
        return full

    # Screens and pipes mangle CJK on this platform, so the numbers always land
    # in a UTF-8 file and the console only gets the path.
    print(_write(args.out, text))
    slim = [{k: v for k, v in sc.items() if k != "objects"} | dict(objects=[
        {k: v for k, v in o.items() if k != "samples"} for o in sc["objects"]])
        for sc in found]
    print(_write(args.json, json.dumps(slim, ensure_ascii=False, indent=1)))
    bad = sum(len(o["violations"]) for sc in found for o in sc["objects"])
    print("%d scenes, %d violations" % (len(found), bad))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
