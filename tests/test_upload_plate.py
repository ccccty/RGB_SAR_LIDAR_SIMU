"""An uploaded plate must become the ground material, not only a sky card.

The ground.backdrop / span / offset block used to sit inside the library
branch, so background.mode="upload" wrote only optical.sky_image -- which a
nadir camera never sees. A ground/sea recipe with an uploaded photograph
therefore rendered with no plate at all and said nothing.
"""
from __future__ import annotations

import os
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from simkit import scenegen  # noqa: E402

PLATE = os.path.join(ROOT, "assets", "backdrops", "city",
                     "06_Drone_view_of_roundabout_Unsplash.jpg")


def _recipe(kind, asset, cls, span=800.0, offset=(0.3, 0.4),
            sky_card=False):
    bg = {"mode": "upload", "path": PLATE, "span_m": span,
          "offset": list(offset)}
    if sky_card is not None:
        bg["sky_card"] = sky_card
    return {
        "name": "upload_plate_probe",
        "environment": {"kind": kind, "weather": "clear", "background": bg},
        "objects": [{"asset": asset, "cls": cls, "scale": 1.0, "count": 1}],
        "view": {"range_km": 0.65, "aspect": "look_down", "fov_deg": 10.0},
        "capture": {"sensors": ["optical", "ir"], "frames": 2, "fps": 10.0,
                    "seed": 7},
    }


class UploadPlateTests(unittest.TestCase):

    def setUp(self):
        self.assertTrue(os.path.isfile(PLATE), PLATE)

    def test_ground_upload_becomes_the_ground_plate(self):
        cfg = scenegen.build_scene(_recipe(
            "ground", "地面车辆/厢式货车.obj", "vehicle"))
        g = cfg["ground"]
        self.assertTrue(os.path.isfile(g.get("backdrop") or ""), g)
        self.assertAlmostEqual(float(g["backdrop_span_m"]), 800.0)
        self.assertEqual(g["backdrop_offset"], [0.3, 0.4])
        # The IR band reads the same file, so both modalities see one terrain.
        self.assertEqual(cfg["ir"]["background_image"], g["backdrop"])
        # sky_card=false must keep the camera-parented card out of a nadir view.
        self.assertNotIn("sky_image", cfg["optical"])

    def test_sea_upload_becomes_the_ground_plate(self):
        cfg = scenegen.build_scene(_recipe(
            "sea", "军舰/导弹艇.obj", "ship"))
        self.assertTrue(os.path.isfile(
            (cfg.get("ground") or {}).get("backdrop") or ""))

    def test_upload_missing_file_is_still_a_hard_error(self):
        r = _recipe("ground", "地面车辆/厢式货车.obj", "vehicle")
        r["environment"]["background"]["path"] = os.path.join(
            ROOT, "assets", "definitely_missing_plate.jpg")
        with self.assertRaises(ValueError) as ctx:
            scenegen.build_scene(r)
        self.assertIn("背景图不存在", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
