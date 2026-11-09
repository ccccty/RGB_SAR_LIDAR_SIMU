"""Six rounds dispensed from one airframe must not occupy a single point.

_complete_entity() filled every infrared_flare with the same rel_pos, so a
six-round fan rendered as one dot -- directly contradicting the recipe note and
the site caption. The spread is derived from the entity index, which is stable
for a given recipe, so the fan is reproducible.
"""
from __future__ import annotations

import os
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from simkit import ecm, scenegen  # noqa: E402


def _flare(i):
    return {"id": "flare_%02d" % (i + 1), "kind": "infrared_flare",
            "radius_m": 0.2, "trail_m": 3.0, "temp_c": 2000.0,
            "decay_s": 3.0, "rcs_m2": 0.02}


def _recipe(n):
    return {
        "name": "decoy_spread_probe",
        "environment": {"kind": "sky", "weather": "clear",
                        "background": {"mode": "library",
                                       "category": "sky"}},
        "objects": [{"asset": "军用飞机/双发战斗机.obj", "cls": "fighter",
                     "scale": 1.0, "count": 1}],
        "ecm": {"mode": "passive",
                "entities": [_flare(i) for i in range(n)]},
        "view": {"range_km": 2.8, "aspect": "pursuit", "fov_deg": 8.0},
        "capture": {"sensors": ["optical"], "frames": 5, "fps": 10.0,
                    "seed": 7},
    }


class DecoySpreadTests(unittest.TestCase):

    def test_six_rounds_get_distinct_release_times(self):
        cfg = scenegen.build_scene(_recipe(6))
        raw = [e for e in cfg["ecm"]["entities"]
               if e["kind"] == "infrared_flare"]
        self.assertEqual(len(raw), 6)
        rel = [float(e.get("release_s") or 0.0) for e in raw]
        self.assertEqual(len(set(rel)), 6, rel)
        self.assertEqual(rel, sorted(rel), rel)

    def test_six_rounds_are_spatially_separated_in_the_world(self):
        cfg = scenegen.build_scene(_recipe(6))
        spec = ecm.load(cfg["ecm"])
        t = max(float(e.get("release_s") or 0.0)
                for e in cfg["ecm"]["entities"]) + 0.5
        pts = [np.asarray(e["pos"], float)
               for e in ecm.entities_at(spec, t)
               if e["kind"] == "infrared_flare"]
        self.assertEqual(len(pts), 6)
        dmin = min(float(np.linalg.norm(a - b))
                   for i, a in enumerate(pts) for b in pts[i + 1:])
        self.assertGreaterEqual(dmin, 2.0,
                                "flares must be resolvable, min sep=%.2f m" % dmin)

    def test_a_single_round_keeps_the_legacy_position(self):
        cfg = scenegen.build_scene(_recipe(1))
        ents = [e for e in cfg["ecm"]["entities"]
                if e["kind"] == "infrared_flare"]
        self.assertEqual(len(ents), 1)
        ent = ents[0]
        # index 0 reproduces the pre-fix mount: released at t=0, 3-D offset.
        self.assertAlmostEqual(float(ent.get("release_s") or 0.0), 0.0)
        vec = ent.get("rel_pos") or ent.get("pos")
        self.assertEqual(len(vec), 3)


if __name__ == "__main__":
    unittest.main()
