"""Exposure contract: shutter and reference exposure describe one window.

realism.apply() used to assign reference_exposure_s unconditionally from the
*default* exposure (0.004 s), which rewrote every scene that explicitly
declared acquisition.rgb_exposure_s = 0.001 s. The optical electron model
divides by the reference, so those nine delivered scenes rendered the target
four times darker than the photographed plate behind it.
"""
from __future__ import annotations

import glob
import os
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from simkit import realism, scene  # noqa: E402


def _pair(cfg):
    acq = (cfg.get("acquisition") or {}).get("rgb_exposure_s")
    ref = ((cfg.get("optical") or {}).get("sensor") or {}).get(
        "reference_exposure_s")
    return acq, ref


class ExposureContractTests(unittest.TestCase):

    def test_every_shipped_scene_keeps_the_pair_consistent(self):
        bad = []
        for path in sorted(glob.glob(os.path.join(ROOT, "scenes", "*.json"))):
            cfg = scene.load_config(path)
            acq, ref = _pair(cfg)
            if acq is None or ref is None:
                continue
            if abs(float(acq) - float(ref)) > 1e-12:
                bad.append((os.path.basename(path), acq, ref))
        self.assertEqual(bad, [],
                         "shutter and reference exposure must agree: %r" % (bad,))

    def test_explicit_pair_is_never_rewritten(self):
        cfg = {"frames": {"dt": 0.1},
               "acquisition": {"rgb_exposure_s": 0.002},
               "optical": {"sensor": {"enabled": True,
                                      "reference_exposure_s": 0.002},
                           "motion_blur": True}}
        out = realism.apply(cfg)
        self.assertEqual(out["acquisition"]["rgb_exposure_s"], 0.002)
        self.assertEqual(
            out["optical"]["sensor"]["reference_exposure_s"], 0.002)

    def test_missing_reference_follows_the_effective_shutter(self):
        cfg = {"frames": {"dt": 0.1},
               "acquisition": {"rgb_exposure_s": 0.001},
               "optical": {"sensor": {"enabled": True}, "motion_blur": True}}
        out = realism.apply(cfg)
        self.assertEqual(
            out["optical"]["sensor"]["reference_exposure_s"], 0.001)

    def test_missing_both_falls_back_to_the_default_window(self):
        cfg = {"frames": {"dt": 0.1},
               "optical": {"sensor": {"enabled": True}, "motion_blur": True}}
        out = realism.apply(cfg)
        acq, ref = _pair(out)
        self.assertAlmostEqual(float(acq), float(ref), places=12)
        self.assertLessEqual(float(acq), realism.RGB_EXPOSURE_S + 1e-12)

    def test_disabled_sensor_is_left_alone(self):
        cfg = {"optical": {"sensor": {"enabled": False}, "motion_blur": True}}
        out = realism.apply(cfg)
        self.assertNotIn("reference_exposure_s", out["optical"]["sensor"])

    def test_motion_blur_off_is_left_alone(self):
        cfg = {"acquisition": {"rgb_exposure_s": 0.001},
               "optical": {"sensor": {"enabled": True}, "motion_blur": False}}
        out = realism.apply(cfg)
        self.assertNotIn("reference_exposure_s", out["optical"]["sensor"])


if __name__ == "__main__":
    unittest.main()
