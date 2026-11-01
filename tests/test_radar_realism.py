"""Measured-RD detector checks against noise theory and independently located peaks."""
import os
import unittest
from unittest.mock import patch

import numpy as np

from simkit import radar, scene


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
