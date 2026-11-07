"""Contract tests for the oscilloscope-style RD / RA figure renderer."""
import unittest
from unittest.mock import patch

import numpy as np

from simkit import radar_scope

W, H = 1280, 720
R_MAX = 6400.0
D_MAX = 300.0
META = dict(band="Ku", fc_ghz=15.0, range_res_m=12.5, vel_res_mps=9.5, frame=0)


def synth_rd(peaks=((0.33, 0.90, 40.0), (0.66, 0.55, 34.0), (0.88, 0.42, 24.0))):
    """Exponential noise plus injected point targets on a (range, Doppler) grid."""
    n_r, n_d = 512, 64
    rng = np.random.default_rng(11)
    power = rng.exponential(1.0, size=(n_r, n_d))
    ranges = np.linspace(0.0, R_MAX, n_r)
    speeds = np.linspace(-D_MAX, D_MAX, n_d)
    targets = []
    for k, (fr, fv, snr) in enumerate(peaks):
        i, j = int(fr * (n_r - 1)), int(fv * (n_d - 1))
        power[i, j] += 10.0 ** (snr / 10.0)
        targets.append(dict(name="fighter_%02d" % (k + 1), range_m=float(ranges[i]),
                            v_r_mps=float(speeds[j]), az_deg=-1.0 + 0.5 * k,
                            snr_db=float(snr)))
    return power, ranges, speeds, targets


def synth_ra(n_b=48, n_r=400, peak_row=10, peak_col=260, snr=38.0):
    rng = np.random.default_rng(3)
    ra = rng.exponential(1.0, size=(n_b, n_r))
    ra[peak_row, peak_col] += 10.0 ** (snr / 10.0)
    ranges = np.linspace(0.0, 6000.0, n_r)
    beams = np.linspace(-3.0, 3.0, n_b)
    targets = [dict(name="fighter_01", range_m=float(ranges[peak_col]),
                    az_deg=float(beams[peak_row]), v_r_mps=12.0, snr_db=snr)]
    return ra, ranges, beams, targets


def panel_covers(out, point):
    p0, p1, p2, p3 = out["panel"]
    return p0 <= point[0] <= p2 and p1 <= point[1] <= p3


class RadarScopeTests(unittest.TestCase):
    def test_render_rd_contract_and_mapping(self):
        power, ranges, speeds, targets = synth_rd()
        out = radar_scope.render_rd(power, ranges, speeds, targets=targets, meta=META)
        rgb = out["rgb"]
        self.assertEqual(rgb.shape, (H, W, 3))
        self.assertEqual(rgb.dtype, np.uint8)

        x0, y0, x1, y1 = out["plot"]
        for value in out["plot"]:
            self.assertIsInstance(value, int)
        self.assertGreater(x0, 0)
        self.assertLess(x1, W)
        self.assertGreater(y0, 0)
        self.assertLess(y1, H)
        self.assertGreater(x1 - x0, 400)

        # range grows to the right, positive closing speed grows upward
        self.assertLess(out["range_px"](0.0), out["range_px"](R_MAX))
        self.assertGreater(out["doppler_px"](-D_MAX), out["doppler_px"](D_MAX))
        self.assertGreaterEqual(float(out["range_px"](0.0)), float(x0))
        self.assertLessEqual(float(out["range_px"](R_MAX)), float(x1))

        lo, hi = out["db_window"]
        self.assertLess(lo, hi)
        self.assertAlmostEqual(hi - lo, 30.0, places=6)
        self.assertEqual(len(out["target_rows"]), 3)

        # the brightest target is a positive closing speed: above mid-height
        peak = out["peak_px"]
        self.assertLess(peak[1], (y0 + y1) / 2.0)

        # a pseudocolour scope, not a grey strip and not a black frame
        body = rgb[y0 + 4:y1 - 4, x0 + 4:x1 - 4].astype(int)
        self.assertGreater(float(body.std()), 12.0)
        self.assertGreater(float((body.max(axis=2) - body.min(axis=2)).mean()), 15.0)
        self.assertGreater(int(rgb.mean()), 8)

    def test_panel_is_top_right_and_never_covers_the_brightest_peak(self):
        # brightest echo sits exactly where a corner table would go
        power, ranges, speeds, targets = synth_rd(peaks=((0.86, 0.92, 45.0),
                                                        (0.66, 0.55, 30.0),
                                                        (0.88, 0.42, 24.0)))
        out = radar_scope.render_rd(power, ranges, speeds, targets=targets, meta=META)
        panel = out["panel"]
        self.assertEqual(out["panel_mode"], "gutter")
        self.assertGreater(panel[0], W / 2.0)
        self.assertLess(panel[1], H / 2.0)
        self.assertLessEqual(panel[2], W)
        self.assertLessEqual(panel[3], out["plot"][3])
        self.assertFalse(panel_covers(out, out["peak_px"]))
        self.assertEqual(len(out["target_rows"]), 3)
        # the map gave up width instead of hiding the return
        self.assertLess(out["plot"][2], out["panel"][0])

    def test_dyn_db_only_changes_brightness(self):
        power, ranges, speeds, targets = synth_rd()
        narrow = radar_scope.render_rd(power, ranges, speeds, targets, META, dyn_db=20.0)
        wide = radar_scope.render_rd(power, ranges, speeds, targets, META, dyn_db=40.0)
        self.assertEqual(narrow["plot"], wide["plot"])
        self.assertEqual(narrow["panel"], wide["panel"])
        self.assertEqual([r["name"] for r in narrow["target_rows"]],
                         [r["name"] for r in wide["target_rows"]])
        self.assertEqual(narrow["db_window"][0], wide["db_window"][0])
        self.assertNotEqual(narrow["db_window"][1], wide["db_window"][1])
        self.assertFalse(np.array_equal(narrow["rgb"], wide["rgb"]))
        self.assertGreater(float(narrow["rgb"].mean()), float(wide["rgb"].mean()))

    def test_missing_cjk_font_degrades_to_ascii(self):
        power, ranges, speeds, targets = synth_rd()
        with patch.object(radar_scope, "_font_paths", return_value=[]):
            out = radar_scope.render_rd(power, ranges, speeds, targets, META)
        self.assertFalse(out["cjk"])
        self.assertEqual(out["rgb"].shape, (H, W, 3))
        lines = list(out["title"]) + [out["y_label"]]
        for text in lines:
            for ch in text:
                self.assertLess(ord(ch), 128, "non-ASCII label survived: %r" % text)

    def test_render_ra_rotated_layout_and_angle_axis(self):
        ra, ranges, beams, targets = synth_ra()
        meta = dict(band="Ku", az_res_deg=6.0 / 48.0, frame=2)
        out = radar_scope.render_ra(ra, ranges, beams, targets, meta, size=(W, 800))
        self.assertEqual(out["rgb"].shape, (800, W, 3))
        self.assertEqual(out["y_label"], "方位 / °")
        self.assertGreater(out["doppler_px"](beams[0]), out["doppler_px"](beams[-1]))
        self.assertLess(out["range_px"](ranges[0]), out["range_px"](ranges[-1]))
        self.assertEqual(len(out["target_rows"]), 1)
        self.assertAlmostEqual(out["target_rows"][0]["az_deg"], float(beams[10]))
        lo, hi = out["db_window"]
        self.assertAlmostEqual(hi - lo, 45.0, places=6)

        rotated = radar_scope.render_ra(ra.T, ranges, beams, targets,
                                        dict(meta, ra_layout="range_beams"), size=(W, 800))
        self.assertEqual(rotated["plot"], out["plot"])
        self.assertEqual(rotated["target_rows"], out["target_rows"])
        self.assertAlmostEqual(rotated["db_window"][0], lo, places=6)

    def test_no_targets_still_renders_a_scope(self):
        power, ranges, speeds, _ = synth_rd()
        out = radar_scope.render_rd(power, ranges, speeds, targets=None, meta=META)
        self.assertEqual(out["target_rows"], [])
        self.assertEqual(out["rgb"].shape, (H, W, 3))
        self.assertLess(out["panel"][3], H)

    def test_table_truncates_after_eight_targets(self):
        power, ranges, speeds, _ = synth_rd()
        many = [dict(name="t%02d" % i, range_m=400.0 + 500.0 * i,
                     v_r_mps=-200.0 + 40.0 * i, az_deg=0.0, snr_db=20.0 + i)
                for i in range(11)]
        out = radar_scope.render_rd(power, ranges, speeds, many, META)
        self.assertEqual(len(out["target_rows"]), 8)
        self.assertEqual(out["target_rows"][-1]["index"], 7)
        self.assertGreater(out["panel"][3], out["plot"][1] + 8 * 19)
        self.assertLessEqual(out["panel"][3], H)

    def test_colour_lut_covers_all_256_levels(self):
        lut = radar_scope.cmap_lut("turbo")
        self.assertEqual(lut.shape, (256, 3))
        self.assertEqual(lut.dtype, np.uint8)
        self.assertEqual(len(np.unique(lut, axis=0)), 256)
        self.assertFalse(np.array_equal(lut[0], lut[-1]))
        self.assertIs(radar_scope.cmap_lut("turbo"), lut)   # cached, reproducible
        with self.assertRaises(ValueError):
            radar_scope.cmap_lut("no_such_map")

    def test_power_shape_must_match_axes(self):
        power, ranges, speeds, targets = synth_rd()
        with self.assertRaises(ValueError):
            radar_scope.render_rd(power, ranges[:-1], speeds, targets, META)


if __name__ == "__main__":
    unittest.main()
