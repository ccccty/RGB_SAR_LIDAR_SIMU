# -*- coding: utf-8 -*-
"""PLAN.md §5 items 0 and 4: the "written but switched off" realism defaults.

These tests pin the *effect*, not the implementation: after scene.load_config a
dataset must actually use the ISP stages, the HDR sensor chain, motion blur, a
capped IR preview window and the altitude-aware atmosphere; and a per-object
temp_c must no longer override the radiative balance.
"""
import copy
import os
import tempfile
import unittest

from simkit import realism, sar, scene, thermal, weather


def minimal_cfg():
    return {
        "objects": [{"name": "target", "shape": "box", "size": [2, 2, 2],
                     "pos": [0, 10, 1], "mat": "painted_metal", "temp_c": 39.0}],
        "ground": {"enabled": True, "temp_c": 31.0},
        "air": {"temp_c": 25.0},
        "sensors": {"sar": {"n_samples": 512}},
    }


class RealismDefaultsTests(unittest.TestCase):
    def test_switches_are_actually_on_after_apply(self):
        cfg = realism.apply(minimal_cfg())
        cam = cfg["optical"]["camera"]
        for stage in realism.ISP_STAGES:
            self.assertTrue(cam[stage]["enabled"], stage)
        self.assertTrue(cfg["optical"]["sensor"]["enabled"])
        self.assertTrue(cfg["optical"]["motion_blur"])
        self.assertEqual(cfg["ir"]["preview_span_c"], 20.0)
        self.assertEqual(cfg["atmosphere"]["profile"], "exponential")
        self.assertAlmostEqual(cfg["acquisition"]["rgb_exposure_s"], 0.004)
        self.assertEqual(cfg["sensors"]["sar"]["display_anchor_pct"], 1.5)

    def test_temp_c_override_is_cleared_but_ambient_is_kept(self):
        cfg = realism.apply(minimal_cfg())
        self.assertIsNone(cfg["objects"][0]["temp_c"])
        self.assertIsNone(cfg["ground"]["temp_c"])
        self.assertEqual(cfg["air"]["temp_c"], 25.0,
                         "ambient air temperature is an input, not an override")

    def test_explicit_scene_values_win_and_opt_out_works(self):
        cfg = minimal_cfg()
        cfg["optical"] = {"camera": {"vignetting": {"enabled": False}},
                          "sensor": {"enabled": False}}
        cfg["ir"] = {"preview_span_c": 12.0}
        cfg["atmosphere"] = {"profile": "homogeneous"}
        out = realism.apply(cfg)
        self.assertFalse(out["optical"]["camera"]["vignetting"]["enabled"])
        self.assertFalse(out["optical"]["sensor"]["enabled"])
        self.assertEqual(out["ir"]["preview_span_c"], 12.0)
        self.assertEqual(out["atmosphere"]["profile"], "homogeneous")

        off = minimal_cfg()
        off["realism"] = {"enabled": False}
        self.assertEqual(realism.apply(off), off)
        self.assertNotIn("camera", off.get("optical", {}))

    def test_jpeg_is_opt_in_not_a_new_default(self):
        # tests/test_jpeg_output.py pins "output.jpeg off by default".
        self.assertNotIn("jpeg", realism.apply(minimal_cfg()).get("output", {}))
        cfg = minimal_cfg()
        cfg["realism"] = {"jpeg": True}
        self.assertEqual(realism.apply(cfg)["output"]["jpeg"], 90)

    def test_plate_scene_disables_auto_exposure(self):
        cfg = minimal_cfg()
        cfg["ground"]["backdrop"] = "../assets/backdrops/desert/x.jpg"
        self.assertFalse(realism.apply(cfg)["optical"]["auto_exposure"])

    def test_load_config_applies_the_defaults(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "scene.json")
            import json
            with open(p, "w", encoding="utf-8") as fh:
                json.dump(minimal_cfg(), fh)
            loaded = scene.load_config(p)
        self.assertTrue(loaded["optical"]["sensor"]["enabled"])
        self.assertTrue(loaded["optical"]["camera"]["flare"]["enabled"])
        self.assertIsNone(loaded["objects"][0]["temp_c"])

    def test_apply_is_idempotent(self):
        once = realism.apply(minimal_cfg())
        twice = realism.apply(copy.deepcopy(once))
        self.assertEqual(once, twice)


class Item4PhysicsTests(unittest.TestCase):
    """PLAN.md §5 item 4: the last Swinbank humidity-sign bug is gone."""

    def test_sky_temperature_increases_with_humidity(self):
        dry = weather.sky_temperature_k(25.0, 0.2)
        humid = weather.sky_temperature_k(25.0, 0.8)
        self.assertGreater(humid, dry,
                           "a moister sky radiates more; (1-0.25RH) was backwards")
        self.assertGreater(dry, 270.0)
        self.assertLess(humid, 300.0)

    def test_thermal_fallback_matches_derive(self):
        # thermal.sky_temp is the no-derived-weather fallback and must agree
        # with the physics the main render loop gets from derive().
        t = thermal.sky_temp(298.15, 0.5)
        self.assertAlmostEqual(t, weather.sky_temperature_k(25.0, 0.5), places=9)

    def test_sar_display_anchor_changed(self):
        self.assertEqual(sar.DEFAULTS["display_anchor_pct"], 1.5)


if __name__ == "__main__":
    unittest.main()
