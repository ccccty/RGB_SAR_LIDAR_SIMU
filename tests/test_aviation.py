"""Independent kinematic, projection, and echo-preservation regression checks."""
import copy
import unittest

import numpy as np

from simkit import labels, observations, optical, plan, radar, scene


def config():
    cam = dict(pos=[0, 0, 5500], look_at=[22000, 5000, 3500],
               fov_deg=2.2, width=640, height=480)
    return dict(frames=dict(n=8, dt=0.2), seed=7,
                ground=dict(enabled=False), weather=dict(preset="clear", mode="fixed"),
                objects=[dict(name="civil_01", shape="box", size=[30, 40, 8], cls="airplane",
                              pos=[22000, 5000, 3500], vel=[0, 180, 0], mat="painted_metal")],
                sensors=dict(optical=cam, radar=dict(cam)),
                clip=dict(frames_per_clip=4, camera="optical",
                          observer_trajectory=dict(mode="line", origin=[0, 0, 5500], vel=[220, 0, 0]),
                          trajectory=dict(mode="line", origin=[22000, 5000, 3500], vel=[0, 180, 0]),
                          rig=dict(lag_s=0.15)))


class AviationTests(unittest.TestCase):
    def test_far_plane_includes_translated_aircraft(self):
        p = np.array([0, 0, 5500])
        points = np.array([[22000, 5000, 3500], [60000, 5000, 3500]])
        self.assertGreater(optical.camera_clip_end(p, points), np.linalg.norm(points[-1]-p))

    def test_relative_motion_and_range_derivative(self):
        cfg = config()
        cp = plan.ClipPlan(cfg, 0, 4, 0.2, 7)
        t, h = 0.4, 1e-3
        np.testing.assert_allclose(cp.target_vel(t), [0, 180, 0])
        np.testing.assert_allclose(cp.camera_velocity(t), [220, 0, 0], atol=1e-7)
        d = cp.target_pos(t)-cp.cam_pos_at(t)
        expected = np.dot(d, np.array([-220, 180, 0]))/np.linalg.norm(d)
        measured = (cp.range_for_time(t+h)-cp.range_for_time(t-h))/(2*h)
        self.assertAlmostEqual(expected, measured, places=5)

    def test_singleton_chunks_and_clip_boundaries(self):
        cfg = config()
        whole = list(plan.plan_frames(cfg, 8, 7, 0.2))
        for index in (0, 3, 4, 7):
            chunk = list(plan.plan_frames(cfg, 1, 7, 0.2, frame_offset=index, total_frames=8))[0]
            self.assertEqual(whole[index], chunk)
            np.testing.assert_allclose(chunk["cfg"]["sensors"]["radar"]["vel"], [220, 0, 0], atol=1e-7)

    def test_exported_camera_matrix_reprojects_off_axis_points(self):
        cfg = config()
        state = next(plan.plan_frames(cfg, 1, 7, 0.2))
        cam = next(iter(state["views"].values()))
        cam["roll_deg"] = 23.5
        world = scene.build_world(state["cfg"], jitter=state["jitter"])
        metadata = observations.describe(state, world)
        calibration = next(iter(metadata["cameras"].values()))
        pts = world.objs[0]["corners"]
        points_camera = np.c_[pts, np.ones(len(pts))] @ np.array(calibration["world_to_camera"]).T
        image = points_camera[:, :3] @ np.array(calibration["K"]).T
        image = image[:, :2]/image[:, 2:3]
        expected = labels.project_points(pts, cam, cam["width"], cam["height"])
        np.testing.assert_allclose(image, expected[:, :2], atol=1e-7)

    def test_weak_echo_not_erased_by_detection_threshold(self):
        cfg = dict(ground=dict(enabled=False), _frame_seed=5,
                   objects=[dict(name="box", shape="box", size=[2, 2, 2],
                                 pos=[0, 15, 0], vel=[0, -10, 0], mat="painted_metal")],
                   sensors=dict(radar=dict(pos=[0, 0, 1], look_at=[0, 15, 1],
                                n_az=1, n_el=1, az_span_deg=0, el_span_deg=0,
                                n_samples=64, n_chirps=16, bandwidth_mhz=50, clutter_density=0,
                                det_snr_db=-1000, swerling="none", vel=[0, 5, 0])))
        world = scene.build_world(cfg)
        low = radar.simulate(world, cfg)
        cfg["sensors"]["radar"]["det_snr_db"] = 1000
        high = radar.simulate(world, cfg)
        self.assertGreater(len(low["detections"]), 0)
        self.assertEqual(len(high["detections"]), 0)
        np.testing.assert_array_equal(low["rd_map"], high["rd_map"])
        self.assertAlmostEqual(low["detections"][0]["v_r_true"], 15.0)

    def test_reject_duplicate_track_names(self):
        cfg = config()
        cfg["objects"].append(copy.deepcopy(cfg["objects"][0]))
        with self.assertRaisesRegex(ValueError, "unique, non-empty names"):
            plan.ClipPlan(cfg, 0, 4, 0.2, 7)

    def test_independent_flight_preserves_secondary_target_motion(self):
        cfg = config()
        cfg["objects"].append(dict(name="second", shape="box", pos=[100, 200, 3000], vel=[40, 0, 0]))
        cp = plan.ClipPlan(cfg, 0, 4, 0.2, 7)
        position, velocity, _ = cp.object_state(cfg["objects"][1], 1)
        np.testing.assert_allclose(position, [140, 200, 3000])
        np.testing.assert_allclose(velocity, [40, 0, 0])

    def test_subbeam_aircraft_does_not_inherit_ground_power(self):
        cfg = dict(ground=dict(enabled=False, size=20000, mat="concrete"), _frame_seed=3,
                   objects=[dict(name="small_aircraft", shape="box", size=[1, 1, 1],
                                 pos=[0, 1000, 10], vel=[0, -10, 0], mat="painted_metal")],
                   sensors=dict(radar=dict(pos=[0, 0, 20], look_at=[0, 1000, 10.5],
                                n_az=2, n_el=1, az_span_deg=2, el_span_deg=0,
                                bandwidth_mhz=1, n_samples=128, n_chirps=16,
                                clutter_density=0, swerling="none", det_snr_db=-1000)))
        without = radar.simulate(scene.build_world(cfg), cfg)
        cfg["ground"]["enabled"] = True
        with_ground = radar.simulate(scene.build_world(cfg), cfg)
        a = [d for d in without["detections"] if d["obj"] == 0]
        b = [d for d in with_ground["detections"] if d["obj"] == 0]
        self.assertEqual(len(a), 1)
        self.assertEqual(len(b), 1)
        self.assertAlmostEqual(a[0]["snr_db"], b[0]["snr_db"], places=6)


if __name__ == "__main__":
    unittest.main()
