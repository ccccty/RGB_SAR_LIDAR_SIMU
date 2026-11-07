"""Independent continuity, dynamics and cross-renderer pose regression checks."""
import copy
import unittest

import numpy as np

from simkit import motion, plan, scene, validation


def config():
    cam = dict(pos=[-100, -200, 100], look_at=[0, 0, 100], width=64, height=48, fov_deg=30)
    return dict(seed=3, frames=dict(n=120, dt=0.1), ground=dict(enabled=False),
                weather=dict(preset="clear", mode="fixed"), sensors=dict(optical=cam),
                objects=[dict(name="a", shape="box", pos=[0, 0, 100], vel=[10, 0, 0]),
                         dict(name="b", shape="box", pos=[20, 30, 90], vel=[0, 2, 0]),
                         dict(name="scenery", shape="box", label=False, pos=[30, 0, 0], vel=[1, 0, 0])])


class MotionTests(unittest.TestCase):
    def test_objects_and_scenery_move_independently_in_world_and_shutter(self):
        cfg = config()
        cfg["optical"] = dict(motion_blur=True)
        cfg["acquisition"] = dict(rgb_exposure_s=0.01)
        s = list(plan.plan_frames(cfg, 12, 3, 0.1))[-1]
        world = scene.build_world(s["cfg"], jitter=s["jitter"])
        for obj, spec in zip(world.objs, cfg["objects"]):
            expected = np.array(spec["pos"])+np.array(spec["vel"])*1.1
            np.testing.assert_allclose(obj["pos"], expected)
            np.testing.assert_allclose(obj["vel"], spec["vel"])
            for sample in s["shutter_samples"]:
                np.testing.assert_allclose(sample["poses"][spec["name"]]["pos"],
                    np.array(spec["pos"])+np.array(spec["vel"])*(1.1+sample["offset_s"]))

    def test_short_prefix_and_singleton_chunk_do_not_change_nominal_path(self):
        cfg = config()
        cfg["clip"] = dict(seeker=dict(target_px=[2, 15], closing_mps=5))
        full = list(plan.plan_frames(cfg, 120, 3, 0.1))
        self.assertEqual(full[:10], list(plan.plan_frames(cfg, 10, 3, 0.1)))
        self.assertEqual(full[7], next(plan.plan_frames(cfg, 1, 3, 0.1, frame_offset=7)))

    def test_open_waypoints_stop_without_teleport_and_keep_tangent(self):
        p = motion.Waypoints([[0, 0, 0], [7, 1, 0], [20, 5, 0]], speed=4, loop=False)
        np.testing.assert_allclose(p.pos(1e6), [20, 5, 0])
        np.testing.assert_allclose(p.vel(p.duration), [0, 0, 0])
        self.assertLess(np.linalg.norm(p.vel(p.duration-0.001)), 0.001)
        self.assertLess(abs(p.heading(p.duration)-p.heading(p.duration-0.001)), 0.01)
        for t in np.linspace(0.05, p.duration-0.05, 100):
            np.testing.assert_allclose(p.vel(t), p.sample(t, 1e-5), atol=0.02)

    def test_loop_waypoints_have_continuous_position_and_heading_at_seam(self):
        p = motion.Waypoints([[0, 0, 0], [10, 0, 0], [12, 20, 0], [-5, 8, 0]], speed=3)
        t = p.length/p.v
        self.assertLess(np.linalg.norm(p.pos(t-1e-4)-p.pos(t+1e-4)), 0.001)
        self.assertLess(np.linalg.norm(p.vel(t-1e-4)-p.vel(t+1e-4)), 0.001)

    def test_open_waypoint_phase_still_brakes_at_endpoint(self):
        p = motion.Waypoints([[0, 0, 0], [20, 0, 0]], speed=4, loop=False, phase=5)
        np.testing.assert_allclose(p.pos(0), [5, 0, 0])
        np.testing.assert_allclose(p.pos(p.duration), [20, 0, 0])
        self.assertLess(np.linalg.norm(p.vel(p.duration-0.001)), 0.001)

    def test_ground_constraint_removes_climb_and_bob(self):
        for spec in [dict(mode="line", vel=[1, 2, 3]), dict(mode="arc", climb_mps=2),
                     dict(mode="weave", z_amplitude_m=3), dict(mode="hover", bob_m=2)]:
            p = motion.make_path(spec, ground_z=4)
            for t in (0, 0.5, 4):
                self.assertEqual(p.pos(t)[2], 4)
                self.assertEqual(p.vel(t)[2], 0)

    def test_fixedwing_bank_and_geometry_follow_centripetal_acceleration(self):
        cfg = config()
        cfg["objects"][0]["attitude_mode"] = "coordinated"
        cfg["clip"] = dict(trajectory=dict(mode="arc", radius_m=5000, speed_mps=200, origin=[0, 0, 3000]))
        cp = plan.ClipPlan(cfg, 0, 120, 0.1, 3)
        pitch, roll = cp.object_attitude(cfg["objects"][0], 0.5)
        self.assertAlmostEqual(pitch, 0)
        self.assertAlmostEqual(np.tan(np.radians(-roll))*9.81, 200**2/5000, places=5)
        s = cp.state(5, 5)
        world = scene.build_world(s["cfg"], jitter=s["jitter"])
        np.testing.assert_allclose(world.objs[0]["corners"].mean(0),
                                   world.verts[:8].mean(0))
        self.assertAlmostEqual(world.objs[0]["roll"], roll)

    def test_disabled_platform_variation_keeps_explicit_heading_and_dive(self):
        cfg = config()
        cfg["objects"] = cfg["objects"][:1]
        cfg["clip"] = dict(seeker=dict(mode="platform", per_clip_variation=dict(enabled=False),
            platform=dict(origin=[-100, -200, 100], heading_deg=12, dive_deg=-3,
                          speed_mps=40, weave_amp_m=0.2, solve="speed")))
        for c in (0, 1):
            cp = plan.ClipPlan(cfg, c, 120, 0.1, 3)
            self.assertEqual(cp.path.heading_deg, 12)
            self.assertEqual(cp.path.dive_deg, -3)
            np.testing.assert_allclose(cp.path.pos(0), [-100, -200, 100])

    def test_preflight_rejects_unphysical_acceleration_and_missing_sensor(self):
        cfg = config()
        cfg["objects"][0]["kinematics"] = dict(max_accel_mps2=10)
        cfg["clip"] = dict(trajectory=dict(mode="weave", vel=[200, 0, 0], amplitude_m=100, hz=2))
        r = validation.inspect(cfg, sensors=("sar",))
        # Preflight reports motion_ok; "passed" belongs to the artifact check and
        # is false until rendering has been verified.
        self.assertFalse(r["motion_ok"])
        self.assertFalse(r["passed"])
        self.assertTrue(any("max_accel" in e for e in r["errors"]))
        self.assertTrue(any("not configured" in e for e in r["errors"]))


class RouteShapingTests(unittest.TestCase):
    """A drawn stroke is a sketch: shaping it must not move an endpoint or lie.

    The failure these guard against is the one measured on the three-aircraft
    plate demo: five points 10-30 m apart flown at 80 m/s put 22 g and 32037
    m/s^3 into a fighter.  Shaping has three knobs -- round the corner, slow
    down, give up shape -- and each one has to be taken in that order, be
    measured on the flown spline, and be reported.
    """

    KIN = dict(min_speed_mps=60.0, max_speed_mps=700.0, max_accel_mps2=30.0,
               max_climb_mps=250.0, min_altitude_m=50.0, max_bank_deg=65.0)

    def fighter(self):
        return motion.Envelope.from_spec(dict(cls="fighter", kinematics=self.KIN))

    def measured(self, shaped):
        path = motion.Waypoints(shaped["points"], speed=shaped["speed_mps"],
                                loop=False, ramp_in=False)
        return motion._measure_route(path, frames=24, dt=0.1)

    def test_lateral_cap_is_the_smaller_of_structure_and_bank(self):
        env = self.fighter()
        # 65 deg of bank is 2.14 g of lateral acceleration; 9 g structural would
        # need 84 deg, which this airframe does not have.
        self.assertAlmostEqual(env.lateral_accel_cap(), motion.G * np.tan(np.radians(65)),
                               places=6)
        self.assertAlmostEqual(env.min_turn_radius_m(80.0),
                               80.0 ** 2 / env.lateral_accel_cap(), places=6)
        self.assertAlmostEqual(env.max_speed_for_radius(env.min_turn_radius_m(80.0)),
                               80.0, places=6)
        # a ground vehicle turns on grip, not on bank: max_bank_deg is body roll
        car = motion.Envelope.preset("ground")
        self.assertAlmostEqual(car.lateral_accel_cap(),
                               min(motion.G * np.sqrt(car.max_g ** 2 - 1.0),
                                   car.max_accel_mps2), places=6)

    def test_straight_leg_is_left_exactly_alone(self):
        pts = [[0.0, 0.0, 100.0], [0.0, 30.0, 100.0]]
        out = motion.shape_route(pts, speed=80.0, envelope=self.fighter())
        self.assertEqual(len(out["points"]), 2)
        self.assertAlmostEqual(out["speed_mps"], 80.0)
        self.assertFalse(out["info"]["shaped"])

    def test_tight_stroke_keeps_its_endpoints_and_gets_inside_the_envelope(self):
        drawn = [[-172.818, 139.729, 0.0], [-164.193, 142.08, 0.0],
                 [-157.92, 145.214, 0.0], [-152.431, 149.915, 0.0],
                 [-146.942, 153.833, 0.0]]
        shaped = motion.shape_route(drawn, speed=80.0, envelope=self.fighter())
        self.assertLess(shaped["speed_mps"], 80.0, "a 29 m stroke at 80 m/s needs 30 g")
        for a, b in ((shaped["points"][0], drawn[0]), (shaped["points"][-1], drawn[-1])):
            for k in range(3):
                self.assertAlmostEqual(float(a[k]), float(b[k]), places=3)
        worst = self.measured(shaped)
        env = self.fighter()
        self.assertLessEqual(worst["accel_mps2"], env.max_accel_mps2)
        self.assertLessEqual(worst["bank_deg"], env.max_bank_deg)
        self.assertLessEqual(worst["jerk_mps3"], env.max_jerk_mps3)
        self.assertTrue(shaped["info"]["notes"], "shaping was not reported")

    def test_wide_stroke_keeps_all_the_shape_it_can(self):
        """A drawing with room loses no shape: it is flown as drawn."""
        drawn = [[0.0, 0.0, 1000.0], [800.0, 0.0, 1000.0],
                 [1600.0, 300.0, 1000.0], [2400.0, 900.0, 1000.0]]
        shaped = motion.shape_route(drawn, speed=150.0, envelope=self.fighter())
        self.assertAlmostEqual(shaped["speed_mps"], 150.0, delta=0.5)
        self.assertFalse(shaped["info"]["relaxed"])
        self.assertFalse(shaped["info"]["slowed"])
        self.assertGreater(shaped["info"]["radius_room_m"], 1000.0)
        worst = self.measured(shaped)
        self.assertLessEqual(worst["bank_deg"], 0.9 * self.fighter().max_bank_deg)
        self.assertLessEqual(worst["jerk_mps3"], 0.9 * self.fighter().max_jerk_mps3)

    def test_densify_splits_only_the_long_legs(self):
        out = motion.densify_route([[0.0, 0.0, 0.0], [6.0, 0.0, 0.0],
                                    [106.0, 0.0, 0.0]])
        self.assertGreater(len(out), 12, "the 100 m leg was left as one cell")
        self.assertEqual(out[0], [0.0, 0.0, 0.0])
        self.assertEqual(out[-1], [106.0, 0.0, 0.0])
        lengths = [float(np.linalg.norm(np.asarray(b, float) - np.asarray(a, float)))
                   for a, b in zip(out, out[1:])]
        self.assertLess(max(lengths), 2.0 * np.median(lengths))


class EnvelopeTests(unittest.TestCase):
    """No published track may exceed the envelope of the airframe it claims."""

    def test_every_path_stays_inside_a_sane_envelope(self):
        uav = motion.Envelope.preset("uav_fixed")
        small = motion.Envelope.preset("uav_rotor")
        cases = [
            (motion.Line([0, 0, 100], [40, 0, 0]), uav),
            (motion.Arc([0, 5000, 3000], 5000.0, 0.04, z0=3000.0), motion.Envelope.preset("fighter")),
            (motion.Arc([0, 0, 80], 120.0, 0.15, z0=80.0), small),
            (motion.Figure8([0, 0, 50], 40.0, 20.0, 0.3), small),
            (motion.Weave([0, 0, 40], [12, 0, 0], 2.0, hz=0.3), small),
            (motion.FlightManeuver([0, 0, 60], speed_mps=4.0, duration_s=4.0,
                                   turn_deg=60.0, climb_mps=1.5, bank_max_deg=30.0), small),
        ]
        for path, env in cases:
            with self.subTest(path=type(path).__name__, cls=env.kind):
                res = motion.check_track(path, env, n_frames=64, dt=0.1)
                self.assertTrue(res["passed"], "%s: %s" % (
                    type(path).__name__, res["violations"][:3]))

    def test_checker_names_the_frame_and_rule_not_just_failure(self):
        """A violation has to be actionable: frame, rule, value, limit."""
        # 200 m/s on a 200 m radius is 20 g: legal to write down, impossible to fly.
        bad = motion.Arc([0, 200, 3000], 200.0, 1.0, z0=3000.0)
        res = motion.check_track(bad, motion.Envelope.preset("fighter"),
                                 n_frames=16, dt=0.1)
        self.assertFalse(res["passed"])
        rules = {v["rule"] for v in res["violations"]}
        self.assertTrue({"g_max", "bank_max"} & rules, rules)
        first = res["violations"][0]
        for key in ("frame", "t", "rule", "value", "limit", "message"):
            self.assertIn(key, first)
        self.assertGreater(first["value"], first["limit"])

    def test_arc_turn_rate_and_bank_agree_with_the_physics(self):
        """bank = atan(v*omega/g) exactly, for the published fighter arc."""
        p = motion.Arc([0.0, 5000.0, 3000.0], 5000.0, 0.04, z0=3000.0)
        v, a = p.vel(1.1), p.accel(1.1)
        speed = float(np.linalg.norm(v))
        omega = motion.turn_rate(v, a)
        bank = motion.coordinated_bank_deg(v, a)
        self.assertAlmostEqual(speed, 200.0, places=9)
        self.assertAlmostEqual(omega, 0.04, places=12)
        self.assertAlmostEqual(np.tan(np.radians(-bank)) * motion.G,
                               speed * omega, places=8)
        self.assertAlmostEqual(float(np.linalg.norm(a)), speed * omega, places=8)
        # a level turn carries 1/cos(bank) of load, and nothing else
        self.assertAlmostEqual(motion.load_factor(v, a),
                               1.0 / np.cos(np.radians(bank)), places=9)

    def test_envelope_classification_follows_the_airframe_not_the_label(self):
        self.assertEqual(motion.Envelope.classify(dict(cls="fighter")), "fighter")
        self.assertEqual(motion.Envelope.classify(dict(cls="uav")), "uav_fixed")
        # a model directory overrides a vague cls
        self.assertEqual(motion.Envelope.classify(
            dict(cls="uav", model="../assets/models/旋翼无人机/x.obj")), "uav_rotor")
        # a taxiing aircraft is a ground vehicle
        self.assertEqual(motion.Envelope.classify(
            dict(cls="fighter", motion_type="ground")), "ground")
        # scenery with no declared limits is not judged as an aircraft
        self.assertEqual(motion.Envelope.classify(dict(cls="scenery")), "ground")
        # but anything that declares flight limits is
        self.assertEqual(motion.Envelope.classify(
            dict(cls="scenery", kinematics={"max_speed_mps": 50})), "uav_fixed")

    def test_scene_kinematics_override_the_class_defaults(self):
        env = motion.Envelope.from_spec(dict(cls="uav", kinematics={"max_speed_mps": 33.0}))
        self.assertEqual(env.kind, "uav_fixed")
        self.assertAlmostEqual(env.max_speed_mps, 33.0)
        # untouched keys keep their class value
        self.assertAlmostEqual(env.max_g, 3.0)


class ContinuityTests(unittest.TestCase):
    """Between frames the acceleration and the heading must not step."""

    def _frame_series(self, path, n=10, dt=0.1):
        return [(path.vel(k * dt), path.accel(k * dt)) for k in range(n)]

    def test_acceleration_does_not_step_between_frames(self):
        # A quintic Hermite path is C2, so accel is continuous across knots; the
        # old shape-per-frame sampling had no such guarantee at all.
        paths = [
            motion.FlightManeuver([0, 0, 60], speed_mps=4.0, duration_s=4.0,
                                  turn_deg=60.0, climb_mps=1.5, bank_max_deg=30.0),
            motion.FlightManeuver([0, 0, 3000], heading_deg=90.0, speed_mps=200.0,
                                  duration_s=8.0, turn_deg=-90.0, bank_max_deg=65.0),
        ]
        for p in paths:
            series = self._frame_series(p, n=40, dt=0.1)
            accs = np.array([a for _v, a in series])
            step = float(np.max(np.linalg.norm(np.diff(accs, axis=0), axis=1)))
            scale = float(np.max(np.linalg.norm(accs, axis=1)))
            self.assertLess(step, 0.4 * scale + 0.5,
                            "acceleration steps %.2f m/s^2 between frames" % step)

    def test_heading_changes_smoothly_and_in_one_direction_in_a_turn(self):
        p = motion.FlightManeuver([0, 0, 60], heading_deg=0.0, speed_mps=4.0,
                                  duration_s=4.0, turn_deg=60.0, bank_max_deg=30.0)
        yaws = [p.yaw(k * 0.05) for k in range(81)]
        diffs = np.diff(yaws)
        self.assertTrue(np.all(diffs > -1e-6), "heading reversed during a steady turn")
        self.assertLess(float(np.max(np.abs(np.diff(diffs)))), 0.25,
                        "heading rate is not smooth")

    def test_manoeuvre_flies_the_heading_change_it_advertises(self):
        for turn in (-90.0, 0.0, 45.0, 120.0):
            p = motion.FlightManeuver([0, 0, 500], heading_deg=30.0, speed_mps=6.0,
                                      duration_s=5.0, turn_deg=turn, bank_max_deg=40.0)
            prof = p.profile()
            with self.subTest(turn=turn):
                self.assertAlmostEqual(prof["heading_change_deg"], turn, delta=1.5)
                self.assertLessEqual(abs(prof["bank_peak_deg"]), 40.0 + 1e-6)
                self.assertLessEqual(prof["max_load_g"], 3.0 + 1e-6)

    def test_manoeuvre_clips_and_says_so_when_the_bank_limit_binds(self):
        """An impossible request must be clipped *and reported*, never silent."""
        p = motion.FlightManeuver([0, 0, 500], speed_mps=60.0, duration_s=2.0,
                                  turn_deg=170.0, bank_max_deg=25.0)
        prof = p.profile()
        self.assertLessEqual(abs(prof["bank_peak_deg"]), 25.0 + 1e-6)
        self.assertTrue(prof["notes"], "clipping was not reported")
        self.assertIn("clipped", prof["notes"][0])


if __name__ == "__main__":
    unittest.main()