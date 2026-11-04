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


if __name__ == "__main__":
    unittest.main()