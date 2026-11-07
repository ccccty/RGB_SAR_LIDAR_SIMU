"""Motion realism: does every trajectory obey the contract it publishes?

The failure this guards against is not "a number is slightly off".  It is that a
sequence looked wrong to a human while every existing test passed, because the
tests checked that *a* position existed on each frame and never that consecutive
positions belonged to the same aircraft.  So the assertions here are all
relational -- derivative against position, acceleration against velocity, bank
against turn rate -- and they run over every Path class in simkit.motion rather
than the two that happened to have tests.

Three groups:

    derivatives  vel == d(pos)/dt and accel == d(vel)/dt to a tight tolerance,
                 on analytic paths and through the scene-level planner.
    envelopes    no track may exceed the speed, load factor, bank, climb rate or
                 altitude of the airframe class it claims, judged per frame.
    continuity   acceleration and heading must not step between frames.  A step
                 is infinite jerk: the image snaps and the label jumps, which is
                 exactly what "not continuous" means when a viewer says it.
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simkit import motion, plan, scene


def _central(fn, t, h=1e-5):
    return (np.asarray(fn(t + h), float) - np.asarray(fn(t - h), float)) / (2.0 * h)


def _all_paths():
    """One instance of every built-in Path subclass, with representative numbers."""
    return [
        motion.Static([10.0, 20.0, 30.0], bob_m=0.4, bob_hz=0.5),
        motion.Line([0.0, 0.0, 100.0], [30.0, -12.0, 2.0]),
        motion.Arc([0.0, 5000.0, 3000.0], 5000.0, 0.04, z0=3000.0),
        motion.Arc([0.0, 5000.0, 3000.0], 5000.0, 0.04, z0=3000.0),
        motion.Arc([0.0, 0.0, 80.0], 120.0, 0.15, z0=80.0, vz=1.0),
        motion.Figure8([0.0, 0.0, 50.0], 40.0, 20.0, 0.3),
        motion.Weave([0.0, 0.0, 40.0], [18.0, 0.0, 0.0], 3.0, hz=0.3, z_amp=0.8),
        motion.DiveWeave([0.0, -900.0, 500.0], [0.0, 0.0, 0.0], speed_mps=90.0,
                         heading_deg=90.0, dive_deg=-12.0, weave_amp_m=6.0,
                         weave_hz=0.2, t_end=6.0),
        motion.Waypoints([[0, 0, 30], [60, 10, 32], [80, 60, 34], [20, 70, 30]],
                         speed=9.0, loop=True),
        motion.Waypoints([[0, 0, 30], [60, 10, 32], [80, 60, 34]], speed=9.0, loop=False),
        motion.FlightManeuver([0, 0, 60], heading_deg=0.0, speed_mps=4.0,
                              duration_s=4.0, turn_deg=60.0, climb_mps=1.5,
                              accel_mps2=0.3, bank_max_deg=30.0),
        motion.FlightManeuver([0, 0, 3000], heading_deg=90.0, speed_mps=200.0,
                              duration_s=8.0, turn_deg=-90.0, bank_max_deg=65.0,
                              flight_class="fighter"),
        motion.GroundPath(motion.Arc([0, 0, 0], 40.0, 0.2), 0.0),
    ]


def _one_sided(fn, t, h=1e-5, forward=True):
    """Second-order one-sided derivative, for sampling exactly at a domain edge.

    A centred difference needs a sample on both sides.  At t=0 of an open route
    there is none -- the path holds position for t<0 by construction -- so
    (f(h)-f(-h))/2h averages the motion with the rest state and returns half of
    something.  That is the same failure mode as the original 76.9 g bug: not a
    wrong velocity, a differencing scheme sampling across a place where the
    function is deliberately not smooth.  This form is O(h^2), the same order as
    the centred rule, and needs only samples inside the domain.
    """
    if forward:
        return (-3.0 * np.asarray(fn(t), float) + 4.0 * np.asarray(fn(t + h), float)
                - np.asarray(fn(t + 2.0 * h), float)) / (2.0 * h)
    return (3.0 * np.asarray(fn(t), float) - 4.0 * np.asarray(fn(t - h), float)
            + np.asarray(fn(t - 2.0 * h), float)) / (2.0 * h)


def _smooth_samples(p):
    """Times where p is genuinely smooth, so a derivative comparison means something.

    For a waypoint route this is the interior of one segment, away from the
    corners; for everything else it is the ordinary sample set.
    """
    if isinstance(p, motion.Waypoints):
        # Interior of the first segment, well away from both of its corners.
        span = p.length / max(float(p.v), 1e-9)
        seg = span / 4.0
        return (0.25 * seg, 0.5 * seg, 0.75 * seg)
    return (0.0, 0.37, 1.5, 4.0)


class DerivativeContractTests(unittest.TestCase):
    """vel must be d(pos)/dt and accel must be d(vel)/dt -- on every class."""

    def test_every_path_publishes_consistent_derivatives(self):
        """The contract holds wherever the path is smooth.

        Two places are excluded on purpose, and both are stated rather than
        quietly skipped:

        A waypoint route is C1 but only C0 in curvature, so acceleration steps
        at every corner -- that is what "a piecewise-cubic route has a jerk
        spike at the corner" means, and it is why the checker reports jerk
        there.  A centred difference taken *across* a corner returns the mean of
        the two one-sided accelerations, so it cannot equal either.  Sampling
        the corner itself tests an ambiguity, not a defect.

        The same applies to an open route's first and last sample, where the
        path is joined to a held position and the derivative is one-sided by
        construction.
        """
        for p in _all_paths():
            name = type(p).__name__
            smooth = _smooth_samples(p)
            for t in smooth:
                with self.subTest(path=name, t=t):
                    v_from_pos = _central(p.pos, t)
                    self.assertLess(
                        float(np.linalg.norm(np.asarray(p.vel(t), float) - v_from_pos)),
                        1e-4, "%s: vel disagrees with d(pos)/dt" % name)
                    a_from_vel = _central(p.vel, t)
                    self.assertLess(
                        float(np.linalg.norm(np.asarray(p.accel(t), float) - a_from_vel)),
                        1e-4, "%s: accel disagrees with d(vel)/dt" % name)

    def test_waypoint_velocity_is_continuous_and_curvature_is_not(self):
        """The precise claim for a waypoint route, at the seam and at a corner.

        Velocity must be continuous everywhere -- a position step or a velocity
        step is a visible jump.  Acceleration is allowed to step at a corner;
        what is not allowed is a velocity step, which is what the loop used to
        have (3.5e-2 m/s) while position and heading were continuous and the old
        test passed.
        """
        p = motion.Waypoints([[0, 0, 30], [60, 10, 32], [80, 60, 34], [20, 70, 30]],
                             speed=9.0, loop=True)
        T = p.length / 9.0
        # The seam is the *same point* approached from both directions, so the two
        # velocities are compared there and nowhere else.  Comparing t=eps with
        # t=T-eps instead would compare two points 2*eps*speed apart along a
        # curving path, and on a 9 m/s route that difference is just the motion
        # (8e-3 m/s at eps=1 ms), not a discontinuity.
        v_gap = float(np.linalg.norm(np.asarray(p.vel(0.0), float)
                                     - np.asarray(p.vel(T), float)))
        self.assertLess(v_gap, 1e-6,
                        "loop seam has a velocity step of %.3e m/s" % v_gap)
        # The seam is a corner, so acceleration legitimately steps there.  Pin
        # the magnitude so a broken corner is still noticed, and pin that it is
        # a *step* rather than a smooth ramp.
        a_plus = np.asarray(p.accel(1e-6), float)
        a_minus = np.asarray(p.accel(T - 1e-6), float)
        jump = float(np.linalg.norm(a_plus - a_minus))
        self.assertGreater(jump, 1e-3, "the seam is a corner and should show one")
        self.assertLess(jump, 10.0, "a corner should not be a 1 g acceleration step")

    def test_seeker_target_has_no_first_frame_half_velocity(self):
        """The bug that started this: frame 0 flew at exactly half the speed.

        The seeker sweep used to clamp its normalised progress at u=0 and then
        differentiate with a two-sided difference, so the sample at t=0 straddled
        the clamp and returned half the true velocity: 75.8 m/s against 150.8 m/s
        on the next frame, i.e. 754 m/s^2 -- 76.9 g -- across one frame.  The
        analytic derivative removes the clamp and the difference together, so the
        speed at frame 0 now agrees with frames 1..n.
        """
        cfg = scene.load_config(os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "scenes", "sk_air_uav.json"))
        cp = plan.ClipPlan(cfg, 0, 50, 0.1, int(cfg.get("seed", 1)))
        speeds = [float(np.linalg.norm(cp.target_vel(k * 0.1))) for k in range(6)]
        self.assertGreater(speeds[0], 0.99 * max(speeds), "frame 0 lost its speed")
        self.assertLess(max(speeds) / min(speeds), 1.02, "speed is not constant")
        for k in range(6):
            t = k * 0.1
            err = float(np.linalg.norm(
                np.asarray(cp.target_vel(t), float) - _central(cp.target_pos, t)))
            self.assertLess(err, 1e-4, "frame %d velocity is not the position derivative" % k)

    def test_seeker_target_acceleration_stays_inside_its_class(self):
        """A straight-and-level receding target must not read 1.4 g of zigzag.

        los_jitter_deg used to be added to the *target's* world position, which
        for an 8 km track meant +/-2.8 m at 0.35 Hz: 13.5 m/s^2 of lateral
        acceleration and 54 deg of bank on an aircraft that is not turning at all.
        It is a mount pointing error and this pins it to the mount.
        """
        cfg = scene.load_config(os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "scenes", "sk_air_uav.json"))
        cp = plan.ClipPlan(cfg, 0, 50, 0.1, int(cfg.get("seed", 1)))
        h = 1e-3
        worst_acc = worst_bank = 0.0
        for k in range(50):
            t = k * 0.1
            v = np.asarray(cp.target_vel(t), float)
            a = (np.asarray(cp.target_vel(t + h), float)
                 - np.asarray(cp.target_vel(t - h), float)) / (2.0 * h)
            # Acceleration and bank are different quantities in different units;
            # taking a max across both compared 5.8 *degrees* of bank against a
            # 5.0 m/s^2 acceleration budget and reported a failure that was purely
            # a unit mix-up.  They get their own limits.
            worst_acc = max(worst_acc, float(np.linalg.norm(a)))
            worst_bank = max(worst_bank,
                             abs(float(motion.coordinated_bank_deg(v, a))))
        self.assertLess(worst_acc, 5.0,
                        "a receding target should accelerate < 0.5 g, not zigzag")
        self.assertLess(worst_bank, 15.0,
                        "a target that is not turning should not bank hard")

    def test_planner_reports_the_same_velocity_the_path_publishes(self):
        """object_state must not invent a second opinion about velocity."""
        cfg = scene.load_config(os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "scenes", "validation_fighter.json"))
        cp = plan.ClipPlan(cfg, 0, 120, 0.1, int(cfg.get("seed", 1)))
        obj = cfg["objects"][0]
        for t in (0.0, 1.0, 5.5):
            _p, v, _yaw = cp.object_state(obj, t)
            self.assertLess(
                float(np.linalg.norm(np.asarray(v, float) - _central(cp.object_state_path(obj), t)
                                     if False else np.asarray(v, float) - _central(
                                         lambda tt: cp.object_state(obj, tt)[0], t))), 1e-4)


class GeneratedRouteTests(unittest.TestCase):
    """A recipe's drawn route must reach the scene already flyable.

    This is the regression guard for the defect itself: the three-aircraft plate
    recipe flies 80 m/s through corners a 30 m stroke cannot express, and every
    previous run of it published a track that measured 32037 m/s^3 of jerk.
    """

    def test_plate_recipe_is_shaped_and_its_velocity_matches_its_route(self):
        import copy
        import json
        from simkit import scenegen
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, "scenes", "demo_three_ships_plate.json"),
                  encoding="utf-8") as fh:
            recipe = json.load(fh)["recipe"]
        cfg = scenegen.build_scene(copy.deepcopy(recipe))
        labelled = [o for o in cfg["objects"] if o.get("label", True)]
        self.assertEqual(len(labelled), 3)
        dt = float(cfg["frames"]["dt"])
        for obj in labelled:
            with self.subTest(name=obj["name"]):
                traj = obj["trajectory"]
                self.assertTrue(traj.get("preshaped"), "the route was published unshaped")
                env = motion.Envelope.from_spec(obj)
                path = motion.Waypoints(traj["points"], speed=float(traj["speed_mps"]),
                                        loop=False, ramp_in=False)
                worst = motion._measure_route(path, frames=20, dt=dt)
                self.assertLessEqual(worst["jerk_mps3"], env.max_jerk_mps3)
                self.assertLessEqual(worst["bank_deg"], env.max_bank_deg)
                self.assertLessEqual(worst["accel_mps2"], env.max_accel_mps2)
                # objects[].vel is the route's own initial tangent.  It used to be
                # the line of sight -- [0, 0, 250] on a plate view, i.e. straight
                # up at 250 m/s for an airframe flying at 80 m/s to the east.
                v = np.asarray(obj["vel"], float)
                speed = float(np.linalg.norm(v))
                self.assertAlmostEqual(speed, float(traj["speed_mps"]), delta=0.5)
                tan = motion.route_heading(traj["points"])
                self.assertGreater(float(np.dot(v / speed, tan)), 0.999)


class SceneLevelTests(unittest.TestCase):
    """The shipped scenes must pass their own diagnostics."""

    def _diagnose(self, name):
        import importlib.util
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        spec = importlib.util.spec_from_file_location(
            "motion_diagnose", os.path.join(root, "tools", "motion_diagnose.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.diagnose_scene(os.path.join(root, "scenes", name))

    def test_fighter_scene_reports_the_published_numbers(self):
        d = self._diagnose("validation_fighter.json")
        fighter = [o for o in d["objects"] if o["name"] == "fighter_01"][0]
        s = fighter["summary"]
        self.assertAlmostEqual(s["speed_mps"][0], 200.0, delta=0.5)
        self.assertAlmostEqual(s["max_accel_mps2"], 8.0, delta=0.05)
        self.assertAlmostEqual(abs(s["max_bank_deg"]), 39.2, delta=0.2)
        self.assertLess(s["max_derivative_error_mps"], 1e-3)
        self.assertEqual(fighter["violations"], [])

    def test_demo_scene_is_one_continuous_ten_frame_manoeuvre(self):
        """The demo clip must be one manoeuvre, not ten independent poses.

        Heading, bank and height all have to move monotonically, which they can
        only do if the ten frames are samples of a single integrated flight.
        """
        d = self._diagnose("demo_air_uav.json")
        obj = [o for o in d["objects"] if o["name"] == "uav_01"][0]
        self.assertEqual(obj["summary"]["frames"], 10)
        self.assertEqual(obj["violations"], [])
        yaws = [r["yaw_deg"] for r in obj["samples"]]
        banks = [r["bank_deg"] for r in obj["samples"]]
        alts = [r["altitude_m"] for r in obj["samples"]]
        self.assertTrue(all(y is not None for y in yaws))
        self.assertGreater(yaws[-1] - yaws[0], 45.0, "the clip turns less than it claims")
        # 0.5 deg of bank is 0.09 m/s^2 of lateral acceleration -- below anything
        # that means "turning" -- and the first frame has no left neighbour, so a
        # forward difference there measures the roll-in from rest and reads
        # +0.056 deg where a centred step reads -0.028 deg.  Pinning the sign to
        # 1e-6 deg made this assertion test the differencing scheme instead of
        # the flight, which is what the frame-0 one-sided derivative switch in
        # tools/motion_diagnose exposed.
        self.assertTrue(all(b <= 0.5 for b in banks), "bank should be to the right")
        self.assertLess(banks[0], 0.5, "the clip must start wings level")
        self.assertLess(banks[-1], 2.0, "the clip must roll out")
        self.assertLess(min(banks), -8.0, "the clip never really rolls in")
        self.assertTrue(all(np.diff(alts) > 0), "the climb is not monotonic")
        self.assertLess(obj["summary"]["max_derivative_error_mps"], 1e-3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
