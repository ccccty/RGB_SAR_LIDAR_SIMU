"""The IR plate sampler must read the photograph the optical shader reads.

Two independent bugs made the IR band show different terrain from the optical
band of the same frame:

* the sampler walked PIL rows top-down while Blender's image texture puts v=0
  at the bottom row, so the plate was mirrored in y;
* the default offset was (0.5, 0.5) here and (0.0, 0.0) in the renderer, which
  shifted an un-offset scene by half a tile.

A four-quadrant plate makes both failures visible: world +y must sample the TOP
of the photograph, and an absent offset must behave exactly like (0, 0).
"""
from __future__ import annotations

import unittest

import numpy as np

from simkit import ir


def _quadrant_plate():
    """64x64 plate: TL=10, TR=20, BL=30, BR=40 (PIL row 0 is the top)."""
    g = np.zeros((64, 64), np.float32)
    g[:32, :32] = 10.0
    g[:32, 32:] = 20.0
    g[32:, :32] = 30.0
    g[32:, 32:] = 40.0
    return g


def _sample(offset, w=64, h=64, fov=30.0):
    ground = {"backdrop": "plate.png", "backdrop_span_m": 100.0,
              "backdrop_span_y_m": 100.0}
    if offset is not None:
        ground["backdrop_offset"] = list(offset)
    cfg = {"ground": ground}
    # 100 m above the plane looking straight down, +y is "up" in the frame.
    cam = {"pos": [0.0, 0.0, 100.0], "look_at": [0.0, 0.0, 0.0],
           "up": [0.0, 1.0, 0.0], "fov_deg": fov, "roll_deg": 0.0}
    return ir._plate_view_ground(_quadrant_plate(), cam, cfg, w, h)


def _corner(img, top, left):
    h, w = img.shape
    rows = slice(4, 20) if top else slice(h - 20, h - 4)
    cols = slice(4, 20) if left else slice(w - 20, w - 4)
    return float(np.mean(img[rows, cols])) * 255.0


class PlateRegistrationTests(unittest.TestCase):

    def test_world_north_samples_the_top_of_the_photograph(self):
        out = _sample((0.5, 0.5))
        self.assertIsNotNone(out)
        self.assertAlmostEqual(_corner(out, True, True), 10.0, delta=2.0)
        self.assertAlmostEqual(_corner(out, True, False), 20.0, delta=2.0)
        self.assertAlmostEqual(_corner(out, False, True), 30.0, delta=2.0)
        self.assertAlmostEqual(_corner(out, False, False), 40.0, delta=2.0)

    def test_absent_offset_matches_the_optical_default(self):
        implicit = _sample(None)
        explicit = _sample((0.0, 0.0))
        self.assertIsNotNone(implicit)
        np.testing.assert_allclose(implicit, explicit, rtol=0, atol=1e-6)

    def test_offset_shifts_the_sampled_window(self):
        # The tile repeats, so a large shift wraps; what matters is that the
        # offset actually moves the window instead of being ignored.
        base = _sample((0.5, 0.5))
        shifted = _sample((0.5, 0.55))
        self.assertIsNotNone(shifted)
        self.assertFalse(np.allclose(base, shifted),
                         "backdrop_offset must move the sampled window")


if __name__ == "__main__":
    unittest.main()
