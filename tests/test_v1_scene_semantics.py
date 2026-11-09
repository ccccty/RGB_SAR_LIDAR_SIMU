# -*- coding: utf-8 -*-
"""Regression for the 2026-11-09 v1 sea/land/air scene design.

The v1109 ground-vehicle frames were unusable as a stand-in for a real aerial
vehicle dataset: a dark tank on a dune photo with diagonal tile seams, decoys
hundreds of metres outside the frame, and "static" objects that actually moved
with the seeker's formation path. These tests pin the v1 fixes without needing
the rendered dataset on disk (the generator writes scenes into the deliverable
directory, not into scenes/).
"""
import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import build_sealandair_1109v1 as gen  # noqa: E402

from simkit import plan  # noqa: E402


class V1SceneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="v1test_")
        cls.cfgs = {d["key"]: gen.build_cfg(d, cls.tmp) for d in gen.DOMAINS}

    def test_every_domain_is_five_frames(self):
        for key, cfg in self.cfgs.items():
            self.assertEqual(cfg["frames"]["n"], 5, key)
            self.assertEqual(cfg["clip"]["frames_per_clip"], 5, key)
            self.assertEqual(cfg["capture"]["frames"], 5, key)
            self.assertAlmostEqual(cfg["frames"]["dt"], 0.1)

    def test_all_asset_paths_are_absolute(self):
        for key, cfg in self.cfgs.items():
            for o in cfg.get("objects") or []:
                if o.get("model"):
                    self.assertTrue(os.path.isabs(o["model"]), (key, o["name"]))
                    self.assertTrue(os.path.isfile(o["model"]), (key, o["name"]))
            for holder, field in ((cfg.get("ground") or {}, "backdrop"),
                                  (cfg.get("optical") or {}, "sky_image"),
                                  (cfg.get("ir") or {}, "background_image")):
                p = holder.get(field)
                if p:
                    self.assertTrue(os.path.isabs(p), (key, field))
                    self.assertTrue(os.path.isfile(p), (key, field))


class GroundRealismTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="v1ground_")
        cls.cfg = gen.build_cfg(
            next(d for d in gen.DOMAINS if d["key"] == "ground_vehicle"), cls.tmp)

    def test_look_down_scene_has_no_sky_card(self):
        # A camera-parented sky card sits in front of the ground and was the
        # source of the diagonal bands in the delivered v1109 frames.
        self.assertNotIn("sky_image", self.cfg["optical"])

    def test_road_plate_and_seam_free_span(self):
        g = self.cfg["ground"]
        self.assertIn("roundabout", g["backdrop"])
        self.assertGreaterEqual(g["backdrop_span_m"], 400.0)
        self.assertTrue(g["backdrop_aligned"])

    def test_primary_is_a_normal_grounded_vehicle(self):
        o = self.cfg["objects"][0]
        self.assertIn("厢式货车", o["model"])       # van, not a tank
        self.assertEqual(o["motion_type"], "ground")
        self.assertEqual(o["vel"][2], 0.0)
        self.assertEqual(o["mat"], "paint_car_white")
        self.assertTrue(o["label"])

    def test_every_decoy_is_pinned_or_has_its_own_trajectory(self):
        for o in self.cfg["objects"][1:]:
            self.assertEqual(o["motion_type"], "ground", o["name"])
            self.assertFalse(o["label"], o["name"])
            if o["name"] in gen.STATIC_DECOYS:
                self.assertEqual(o["vel"], [0.0, 0.0, 0.0], o["name"])
            else:
                self.assertNotEqual(o["vel"], [0.0, 0.0, 0.0], o["name"])

    def test_decoys_share_the_frame_with_the_primary(self):
        # The seeker back-solves the primary hundreds of metres from the config
        # origin; decoys must be authored relative to that position.
        p0 = np.asarray(self.cfg["_v1_primary_frame0_world_m"], float)[:2]
        for o in self.cfg["objects"][1:]:
            q = np.asarray(o["pos"][:2], float)
            self.assertLess(np.linalg.norm(q - p0), 80.0, o["name"])

    def test_five_frames_have_a_smooth_closing_trajectory(self):
        states = list(plan.plan_frames(self.cfg, 5,
                                       int(self.cfg.get("seed", 0)), 0.1))
        self.assertEqual(len(states), 5)
        ranges = [s["range_m"] for s in states]
        self.assertTrue(all(b <= a + 1e-6 for a, b in zip(ranges, ranges[1:])),
                        ranges)
        pos = [np.asarray(s["poses"]["vehicle_01"]["pos"], float)
               for s in states]
        steps = [np.linalg.norm(pos[i + 1] - pos[i]) for i in range(4)]
        self.assertTrue(min(steps) > 0.0)
        self.assertLess(max(steps) - min(steps), 0.6 * max(steps) + 1e-6,
                        "target must not jump between frames: %r" % steps)


class AirFlareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="v1air_")
        cls.cfg = gen.build_cfg(
            next(d for d in gen.DOMAINS if d["key"] == "air_plane"), cls.tmp)

    def test_six_small_downward_dispensed_rounds(self):
        flares = [e for e in self.cfg["ecm"]["entities"]
                  if e["kind"] == "infrared_flare"]
        self.assertEqual(len(flares), 6)
        for f in flares:
            self.assertLess(f["vel"][2], -30.0)
            self.assertLess(f["radius_m"], 0.3)
            self.assertLess(f["trail_m"], 4.0)

    def test_rounds_are_mounted_near_the_aircraft_axis(self):
        states = list(plan.plan_frames(self.cfg, 1,
                                       int(self.cfg.get("seed", 0)), 0.1))
        air = np.asarray(states[0]["poses"]["fighter_01"]["pos"], float)
        flares = [e for e in self.cfg["ecm"]["entities"]
                  if e["kind"] == "infrared_flare"]
        for f in flares:
            p = np.asarray(f["pos"], float)
            self.assertLess(p[2], air[2])
            self.assertLess(np.linalg.norm(p[:2] - air[:2]), 25.0)


class SeaDecoyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="v1sea_")
        cls.cfg = gen.build_cfg(
            next(d for d in gen.DOMAINS if d["key"] == "sea_ship"), cls.tmp)

    def test_one_moving_and_one_anchored_craft(self):
        objs = {o["name"]: o for o in self.cfg["objects"]}
        self.assertEqual(objs["anchored_boat_static"]["vel"], [0.0, 0.0, 0.0])
        self.assertNotEqual(objs["moving_fishing_boat"]["vel"], [0.0, 0.0, 0.0])
        for name in ("anchored_boat_static", "moving_fishing_boat"):
            self.assertEqual(objs[name]["motion_type"], "ground")
            self.assertFalse(objs[name]["label"])


if __name__ == "__main__":
    unittest.main()
