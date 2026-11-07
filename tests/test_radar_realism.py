"""Measured-RD detector checks against noise theory and independently located peaks."""
import os
import unittest
from unittest.mock import patch

import numpy as np

from simkit import plan, radar, scene

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def config(target_pos=(0, 30, 0), target_velocity=(0, -8, 0), pfa=1e-6):
    return dict(
        ground=dict(enabled=False), _frame_seed=21,
        objects=[dict(name="aircraft", cls="airplane", shape="box", size=[2, 2, 2],
                      pos=list(target_pos), vel=list(target_velocity), mat="painted_metal", rcs_db=0)],
        sensors=dict(radar=dict(pos=[0, 0, 1], look_at=[0, 30, 1],
                               n_az=2, n_el=2, az_span_deg=0.5, el_span_deg=0.5,
                               n_samples=128, n_chirps=32, bandwidth_mhz=20,
                               chirp_us=50, pri_us=65, clutter_density=0,
                               swerling="none", detector="rd_threshold", pfa=pfa)))


class RadarRealismTests(unittest.TestCase):
    def test_single_beam_is_on_boresight(self):
        cfg = config()["sensors"]["radar"]
        _p, dirs, angles = radar.beam_grid(cfg, 1, 1, 40, 20)
        np.testing.assert_array_equal(angles, [[0, 0]])
        np.testing.assert_allclose(dirs[0], [0, 1, 0], atol=1e-10)

    def test_window_noise_matches_parseval_and_coherent_gain(self):
        nr, nv, pn = 64, 32, 0.7
        wr, wd = np.hanning(nr), np.hanning(nv)
        stats = radar.fft_noise_statistics(pn, wr, wd, 4, 0.01)
        rng = np.random.default_rng(71)
        # Average across independent receiver realisations, not neighbouring
        # Hann-correlated cells alone.
        means = []
        for _ in range(40):
            noise = (rng.normal(size=(nr, nv))+1j*rng.normal(size=(nr, nv)))*np.sqrt(pn/2)
            spectrum = np.fft.fft2(noise*wr[:, None]*wd[None, :])
            means.append(float(np.mean(abs(spectrum)**2)))
        self.assertAlmostEqual(np.mean(means)/stats["single_beam_fft_noise_power"], 1.0, delta=0.035)
        tone = np.ones((nr, nv), complex)
        peak = abs(np.fft.fft2(tone*wr[:, None]*wd[None, :])[0, 0])**2
        self.assertAlmostEqual(peak/stats["coherent_power_gain"], 1.0, places=12)

    def test_multi_beam_threshold_controls_marginal_false_alarm_rate(self):
        pfa, beams = 0.01, 8
        stats = radar.fft_noise_statistics(1.0, np.ones(4), np.ones(4), beams, pfa)
        noise = np.random.default_rng(66).exponential(size=(100000, beams))
        empirical = np.mean(noise.max(axis=1)>stats["threshold_noise_ratio"])
        self.assertAlmostEqual(empirical, pfa, delta=0.0012)
        single = radar.fft_noise_statistics(1.0, np.ones(4), np.ones(4), 1, pfa)
        self.assertGreater(stats["threshold_power"], single["threshold_power"])

    def test_measured_detections_are_on_the_saved_rd_peaks(self):
        cfg = config()
        out = radar.simulate(scene.build_world(cfg), cfg)
        matches = [d for d in out["detections"] if d["obj"] == 0]
        self.assertGreater(len(matches), 0)
        for det in matches:
            ir, iv = det["range_bin"], det["doppler_bin"]
            self.assertEqual(det["range"], out["range_axis"][ir])
            self.assertEqual(det["v_r"], out["doppler_axis"][iv])
            self.assertEqual(det["power"], out["rd_map"][ir, iv])
            self.assertGreater(det["power"], out["processing"]["threshold_power"])
            self.assertAlmostEqual(det["v_r_true"], 8, delta=0.02)

    def test_false_peaks_are_not_labelled_as_faint_aircraft(self):
        cfg = config(pfa=0.2)
        cfg["sensors"]["radar"]["tx_power_dbm"] = -200
        out = radar.simulate(scene.build_world(cfg), cfg)
        self.assertGreater(len(out["detections"]), 0)
        self.assertTrue(all(d["obj"] == -1 for d in out["detections"]))
        self.assertEqual(out["processing"]["unassociated_peak_count"], len(out["detections"]))
        self.assertTrue(all("v_r_true" not in d for d in out["detections"]))

    def test_noise_and_detection_independent_of_memory_chunk_size(self):
        cfg = config(pfa=0.02)
        world = scene.build_world(cfg)
        with patch.dict(os.environ, {"SIMKIT_CUBE_MB": "192"}):
            whole = radar.simulate(world, cfg)
        with patch.dict(os.environ, {"SIMKIT_CUBE_MB": "0.001"}):
            chunked = radar.simulate(world, cfg)
        np.testing.assert_array_equal(whole["rd_map"], chunked["rd_map"])
        self.assertEqual(whole["detections"], chunked["detections"])
        self.assertIsNone(chunked["cube"])

    def test_subbeam_fallback_respects_sensor_field_of_regard(self):
        for target in [(0, -30, 0), (30, 30, 0)]:
            cfg = config(target_pos=target, pfa=1e-6)
            cfg["sensors"]["radar"]["tx_power_dbm"] = 100
            out = radar.simulate(scene.build_world(cfg), cfg)
            self.assertTrue(all(d["obj"] < 0 for d in out["detections"]))
            self.assertTrue(np.all(out["sigma"] == 0))

    def test_single_channel_noise_shows_exponential_speckle(self):
        # A body behind the sensor is geometry only: nothing enters the range
        # window, so the map is pure noise. An exponential cell power has its
        # 10th percentile at 0.105*mean and its 90th at 2.303*mean, i.e. 13.45
        # dB apart -- the speckle a receiver really shows. The per-cell maximum
        # over ~800 beams this replaced is an extreme-value statistic and gave
        # 1-2 dB on the same scene, which is why the shipped image looked flat.
        cfg = config()
        cfg["objects"] = [dict(name="behind", cls="target", shape="box",
                               size=[2, 2, 2], pos=[0, -100, 0],
                               vel=[0, 0, 0], mat="painted_metal")]
        out = radar.simulate(scene.build_world(cfg), cfg)
        db = radar.to_db(out["rd_map"])
        spread = float(np.percentile(db, 90) - np.percentile(db, 10))
        self.assertGreater(spread, 12.0)
        self.assertLess(spread, 15.0)
        # A blank scene at Pfa=1e-6 must not manufacture detections.
        self.assertEqual(len(out["detections"]), 0)

    def test_three_aircraft_resolve_into_three_peaks(self):
        # Three airframes at 6.0 / 7.5 / 9.0 km with distinct closing rates,
        # i.e. three separable resolution cells. Their absolute velocities are
        # asserted on v_r_true (the associated truth surface); the measured v_r
        # is a Doppler cell centre and can only be quoted to half a cell.
        cfg = scene.load_config(os.path.join(ROOT, "scenes", "radar_tri_aircraft.json"))
        cfg.setdefault("sensors", {}).setdefault("radar", {})["detector"] = "rd_threshold"
        frames = int(cfg.get("frames", {}).get("n", 1))
        state = next(plan.plan_frames(cfg, 1, cfg.get("seed", 0),
                                      cfg.get("frames", {}).get("dt", 0.2),
                                      frame_offset=0, total_frames=frames))
        world = scene.build_world(state["cfg"], jitter=state["jitter"])
        out = radar.simulate(world, state["cfg"])
        dets = [d for d in out["detections"] if d["obj"] >= 0]
        best = {}
        for d in dets:
            cur = best.get(d["obj"])
            if cur is None or d["snr_db"] > cur["snr_db"]:
                best[d["obj"]] = d
        self.assertEqual(sorted(best), [0, 1, 2], "three bodies, three associated peaks")
        cell_r = float(np.mean(np.diff(out["range_axis"])))
        cell_v = float(np.mean(np.diff(out["doppler_axis"])))
        truth = {0: (6000.7, 119.94), 1: (7500.0, 185.00), 2: (9000.3, 240.02)}
        for oid, (rng, vr) in truth.items():
            det = best[oid]
            self.assertAlmostEqual(det["range"], rng, delta=40.0)
            self.assertAlmostEqual(det["v_r_true"], vr, delta=3.0)
            self.assertLessEqual(abs(det["v_r"] - vr), cell_v)
            self.assertGreater(det["snr_db"],
                               10.0 * np.log10(out["processing"]["threshold_noise_ratio"]))
        cells = sorted(d["range_bin"] for d in best.values())
        self.assertGreater(cells[1] - cells[0], 2)
        self.assertGreater(cells[2] - cells[1], 2)
        self.assertAlmostEqual(cell_r, 12.49, delta=0.05)

    def test_extended_body_spreads_over_several_range_cells(self):
        # A 20 m deep body at 12.49 m range resolution is not a point: its sigma
        # is shared over the cells it spans, so the peak covers more than one
        # cell. A single point echo was the other half of "one bright pixel".
        cfg = config()
        cfg["objects"] = [dict(name="body", cls="target", shape="box",
                               size=[3, 20, 3], pos=[0, 3000, 0],
                               vel=[0, -120, 0], mat="painted_metal", rcs_db=20)]
        rc = cfg["sensors"]["radar"]
        rc.update(pos=[0, 0, 0], look_at=[0, 3000, 0], n_az=8, n_el=4,
                  az_span_deg=1.0, el_span_deg=1.0, n_samples=2048, n_chirps=64,
                  bandwidth_mhz=12, chirp_us=8.192, pri_us=16.384,
                  tx_power_dbm=43, mti_min_vr_mps=1.0, swerling="none")
        out = radar.simulate(scene.build_world(cfg), cfg)
        cell = float(np.mean(np.diff(out["range_axis"])))
        self.assertAlmostEqual(cell, 12.49, delta=0.05)
        db = radar.to_db(out["rd_map"])
        floor = float(np.percentile(db, 50))
        ir, iv = np.unravel_index(int(np.argmax(out["rd_map"])), out["rd_map"].shape)
        above = db[:, iv] > floor + 6.0
        lo = hi = int(ir)
        while lo > 0 and above[lo - 1]:
            lo -= 1
        while hi + 1 < len(above) and above[hi + 1]:
            hi += 1
        self.assertGreaterEqual(hi - lo + 1, 2,
                                "contiguous range cells above floor+6 dB: %d" % (hi - lo + 1))

    def test_doppler_peak_neighbours_wrap_but_range_does_not(self):
        power = np.zeros((5, 8))
        power[2, 0], power[2, -1] = 12, 8
        power[-1, 4], power[0, 4] = 5, 9
        cells = radar.rd_peak_cells(power, 3)
        self.assertIn((2, 0), cells)
        self.assertNotIn((2, 7), cells)
        self.assertIn((0, 4), cells)
        self.assertIn((4, 4), cells)


if __name__ == "__main__":
    unittest.main()
