# -*- coding: utf-8 -*-
"""Ground/air interference semantics that must not regress.

Three real defects found during the 2026-11-09 review:

1. the seeker planner silently turned every unmarked object into a formation
   member, so "static" barriers flew along with the target;
2. dispensed flares inherited the aircraft's full velocity and flew alongside
   it, instead of falling away below the fuselage axis;
3. each flare was a ~1 m head with a 16-22 m trail, i.e. as large as (or
   larger than) the ~17 m aircraft it was decoying.

These tests check the *generated scene semantics*, not just the code paths.
"""
import os
import sys
import unittest

import numpy as np

from simkit import ecm, plan, scene

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import build_sealandair_1109 as gen  # noqa: E402


def _cfg(key):
    dom = next(d for d in gen.DOMAINS if d["key"] == key)
    cfg = gen.build_cfg(dom)
    # Keep temporary configs out of scenes/: test_track.py pins the number of
    # shipped scene files, and a stray one would break that count.
    tmp = os.path.join(ROOT, "_tmp")
    os.makedirs(tmp, exist_ok=True)
    path = os.path.join(tmp, "_test_" + key + ".json")
    gen.jwrite(path, cfg)
    try:
        return scene.load_config(path), cfg
    finally:
        if os.path.isfile(path):
            os.remove(path)


def _states(cfg, n=3):
    return list(plan.plan_frames(cfg, n, int(cfg.get("seed", 0)),
                                 float(cfg["frames"]["dt"])))


class GroundStaticVsMovingTests(unittest.TestCase):
    def test_static_obstacles_never_move(self):
        cfg, _ = _cfg("ground_vehicle")
        names = [row[0] for row in gen.STATIC_OBSTACLES]
        for name in names:
            obj = next(o for o in cfg["objects"] if o["name"] == name)
            self.assertEqual(obj["motion_type"], "ground",
                             "without it the seeker planner drags the object along")
            self.assertEqual(obj["vel"], [0.0, 0.0, 0.0])
        frames = _states(cfg)
        for name in names:
            positions = [tuple(s["poses"][name]["pos"]) for s in frames]
            self.assertEqual(len(set(positions)), 1,
                             "%s moved across frames: %r" % (name, positions))

    def test_only_the_moving_traffic_changes_position(self):
        cfg, _ = _cfg("ground_vehicle")
        names = [row[0] for row in gen.MOVING_VEHICLES]
        frames = _states(cfg)
        for name in names:
            obj = next(o for o in cfg["objects"] if o["name"] == name)
            self.assertEqual(obj["motion_type"], "ground")
            self.assertNotEqual(obj["vel"], [0.0, 0.0, 0.0])
            positions = [tuple(s["poses"][name]["pos"]) for s in frames]
            self.assertGreater(len(set(positions)), 1,
                               "%s should be moving traffic" % name)

    def test_every_ground_decoy_uses_a_real_vehicle_or_obstacle_model(self):
        cfg, _ = _cfg("ground_vehicle")
        for obj in cfg["objects"][1:]:
            self.assertEqual(obj["shape"], "obj", obj["name"])
            self.assertNotIn("size", obj,
                             "no box/sphere stand-ins for ground interference")
            self.assertTrue(obj["model"].endswith(".obj"))


class AirFlareDispenseTests(unittest.TestCase):
    def test_rounds_are_below_and_beside_the_aircraft_axis(self):
        cfg, _ = _cfg("air_plane")
        state = _states(cfg, 1)[0]
        primary = cfg["objects"][0]
        aircraft = np.asarray(state["poses"][primary["name"]]["pos"], float)
        flares = [e for e in cfg["ecm"]["entities"]
                  if e["kind"] == "infrared_flare"]
        self.assertEqual(len(flares), 6)
        for f in flares:
            self.assertLess(f["pos"][2], aircraft[2],
                            "a downward dispense must start below the airframe")
            self.assertLess(np.linalg.norm(
                np.asarray(f["pos"][:2]) - aircraft[:2]), 25.0,
                "rounds must hang on the fuselage axis, not 100s of m away")

    def test_rounds_fall_away_and_do_not_inherit_aircraft_velocity(self):
        cfg, _ = _cfg("air_plane")
        state = _states(cfg, 1)[0]
        primary = cfg["objects"][0]
        v_air = np.asarray(state["poses"][primary["name"]]["velocity_mps"], float)
        flares = [e for e in cfg["ecm"]["entities"]
                  if e["kind"] == "infrared_flare"]
        for f in flares:
            v = np.asarray(f["vel"], float)
            self.assertLess(v[2], -30.0, "dominant downward ejection")
            self.assertLess(np.linalg.norm(v[:2]),
                            0.25 * np.linalg.norm(v_air[:2]) + 1.0,
                            "must not fly along with the aircraft")
        # Z decreases monotonically over the three delivered frames.
        spec = ecm.load(cfg["ecm"])
        zs = []
        for t in (0.0, 0.1, 0.2):
            first = next(e for e in ecm.entities_at(spec, t)
                         if e["kind"] == "infrared_flare")
            zs.append(float(first["pos"][2]))
        self.assertTrue(all(b < a for a, b in zip(zs, zs[1:])), zs)

    def test_each_round_is_much_smaller_than_the_aircraft(self):
        cfg, _ = _cfg("air_plane")
        primary = cfg["objects"][0]
        mesh, lo, hi = scene.prepare_mesh(primary, gen.SCENES)
        span = float(np.max(hi - lo))
        self.assertGreater(span, 5.0, "sanity: a real aircraft mesh was loaded")
        flares = [e for e in cfg["ecm"]["entities"]
                  if e["kind"] == "infrared_flare"]
        for f in flares:
            self.assertLess(2.0 * f["radius_m"], span * 0.1)
            self.assertLess(f["trail_m"], span * 0.35,
                            "a 20 m trail was longer than the aircraft")
        chaff = next(e for e in cfg["ecm"]["entities"]
                     if e["kind"] == "chaff_cloud")
        self.assertLess(2.0 * chaff["radius_m"], span * 0.5)

    def test_optical_manifest_and_physics_world_use_the_same_trajectory(self):
        # Blender extrapolates ECM poses linearly from the scene; the numpy
        # World evaluates ecm.entity_state. They must agree to the millimetre
        # or the flare is drawn in one place and measured in another.
        cfg, _ = _cfg("air_plane")
        spec = ecm.load(cfg["ecm"])
        raw = {e["id"]: e for e in spec["entities"]
               if e["kind"] == "infrared_flare"}
        for t in (0.0, 0.1, 0.2):
            live = {e["id"]: e for e in ecm.entities_at(spec, t)
                    if e["kind"] == "infrared_flare"}
            for eid, e in raw.items():
                expect = (np.asarray(e["pos"], float)
                          + np.asarray(e["vel"], float) * t
                          + 0.5 * np.asarray(e["accel"], float) * t * t)
                np.testing.assert_allclose(live[eid]["pos"], expect, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
