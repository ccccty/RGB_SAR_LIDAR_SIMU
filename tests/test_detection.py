"""Detection-theory oracles from the project's own handbook (3.2 / 3.3).

These numbers are not invented here: the threshold (11.40 dB), Albersheim's
13.11 dB, and the CA-CFAR losses (1.3 dB at Nr=24, 4.3 dB at Nr=8) are the
pinned values in docs/雷达与激光雷达仿真经验公式手册.md, so a regression in the
library shows up against a document an operator can check by hand.
"""
from __future__ import annotations

import math
import unittest

import numpy as np

from simkit import detection as det


class ThresholdTests(unittest.TestCase):

    def test_single_sample_threshold_matches_the_radar_front_end(self):
        self.assertAlmostEqual(det.threshold_db(1e-6), 11.40, delta=0.02)
        self.assertAlmostEqual(det.threshold_db(1e-4), 9.64, delta=0.03)

    def test_pfa_is_clipped_instead_of_blowing_up(self):
        self.assertTrue(math.isfinite(det.threshold_db(0.0)))
        self.assertTrue(math.isfinite(det.threshold_db(1.0)))


class AlbersheimTests(unittest.TestCase):

    def test_handbook_checkpoint(self):
        self.assertAlmostEqual(det.albersheim_snr_db(1e-6, 0.9, 1), 13.11,
                               delta=0.20)

    def test_more_pulses_need_less_snr(self):
        one = det.albersheim_snr_db(1e-6, 0.9, 1)
        many = det.albersheim_snr_db(1e-6, 0.9, 32)
        self.assertLess(many, one - 10.0)

    def test_higher_pd_needs_more_snr(self):
        self.assertLess(det.albersheim_snr_db(1e-6, 0.5, 1),
                        det.albersheim_snr_db(1e-6, 0.9, 1))


class CfarTests(unittest.TestCase):

    def test_handbook_losses(self):
        self.assertAlmostEqual(det.ca_cfar_alpha_db(24, 1e-6), 12.71, delta=0.05)
        self.assertAlmostEqual(det.cfar_loss_db(24, 1e-6), 1.3, delta=0.10)
        self.assertAlmostEqual(det.ca_cfar_alpha_db(8, 1e-6), 15.68, delta=0.05)
        self.assertAlmostEqual(det.cfar_loss_db(8, 1e-6), 4.3, delta=0.10)

    def test_loss_shrinks_as_the_reference_window_grows(self):
        losses = [det.cfar_loss_db(nr, 1e-6) for nr in (8, 16, 24, 32, 64)]
        self.assertEqual(losses, sorted(losses, reverse=True), losses)

    def test_alpha_matches_the_monte_carlo_pfa(self):
        # Exponential noise, CA-CFAR with Nr cells: the alpha from the closed
        # form must yield the requested Pfa. 3e5 trials, fixed seed.
        rng = np.random.default_rng(20261109)
        nr, pfa = 16, 1e-3
        alpha = det.ca_cfar_alpha(nr, pfa)
        trials = 300000
        noise = rng.exponential(1.0, size=(trials, nr + 1))
        threshold = alpha * noise[:, :nr].mean(axis=1)
        measured = float(np.mean(noise[:, nr] > threshold))
        self.assertGreater(measured, pfa * 0.6)
        self.assertLess(measured, pfa * 1.6)


class MarcumTests(unittest.TestCase):

    def test_pd_at_the_albersheim_operating_point(self):
        snr = 10.0 ** (13.18 / 10.0)
        pd = det.pd_noncoherent(1, snr, pfa=1e-6)
        self.assertAlmostEqual(pd, 0.90, delta=0.01)

    def test_pd_rises_with_snr(self):
        pds = [det.pd_noncoherent(1, 10.0 ** (db / 10.0), pfa=1e-6)
               for db in (10, 13, 16, 20, 25)]
        self.assertEqual(pds, sorted(pds), pds)

    def test_marcum_q_limits(self):
        self.assertAlmostEqual(det.marcum_q(1, 0.0, 0.0), 1.0, places=9)
        self.assertLess(det.marcum_q(1, 0.0, 30.0), 1e-6)


if __name__ == "__main__":
    unittest.main()
