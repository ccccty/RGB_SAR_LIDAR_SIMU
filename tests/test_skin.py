"""Procedural skin texture: determinism, metric scale, mean and cost.

The module this exercises replaces a flat material with a per-point albedo
factor.  For that to be safe in the renderer it has to satisfy four contracts
that are easy to break silently:

* the value depends only on the surface position (a screen-space texture would
  crawl when the camera moves),
* the same seed reproduces the same field exactly,
* the mean stays at 1.0 so the painted colour and the solar absorptivity do not
  shift, and
* a hundred thousand points evaluate in well under 100 ms.

The rest of the file pins the *physical* numbers the layer presets claim:
10-20% joint darkening, 5-10% rivets, 5-15% grime, 1-3% panel offsets, and a
1.5-3 px joint width at the render scale.
"""
import ast
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from simkit import skin

SIZES = {
    "aircraft": (19.4, 13.05, 5.64),
    "ship": (154.0, 20.0, 12.0),
    "vehicle": (4.6, 1.8, 1.5),
}
LAYERS = ("seams", "rivets", "grime", "panels", "mottle")


def cloud(n, size, seed=0):
    """Random points on a fuselage-like body: length along x, radius around it."""
    rng = np.random.default_rng(seed)
    length, width, height = size
    radius = 0.32 * min(width, height)
    angle = rng.uniform(-np.pi, np.pi, n)
    return np.stack([rng.uniform(-0.5 * length, 0.5 * length, n),
                     radius * np.sin(angle),
                     radius * np.cos(angle)], axis=1)


def only(layer, **kwargs):
    flags = {name: (name == layer) for name in LAYERS}
    flags.update(kwargs)
    return flags


def surface_patch(category, size, metres=16.0, pixels=320, **kwargs):
    """A patch sampled exactly at the pixel pitch it is given, i.e. what a
    renderer would see at that framing."""
    px = float(metres) / float(pixels)
    field = skin.SkinField(category, size=size, pixel_scale_m=px, **kwargs)
    image = field.patch(metres, metres, shape=(pixels, pixels),
                        pixel_scale_m=px)
    return field, image


class DeterminismTests(unittest.TestCase):
    def test_same_seed_same_points_is_bitwise_identical(self):
        pts = cloud(5000, SIZES["aircraft"])
        a = skin.SkinField("aircraft", size=SIZES["aircraft"], seed=11)(pts)
        b = skin.SkinField("aircraft", size=SIZES["aircraft"], seed=11)(pts)
        np.testing.assert_array_equal(a, b)
        c = skin.SkinField("aircraft", size=SIZES["aircraft"], seed=12)(pts)
        self.assertGreater(float(np.abs(a - c).max()), 1e-3,
                           "a different seed must produce a different field")

    def test_value_depends_only_on_position(self):
        # No array index, resolution or call grouping may leak into the value:
        # that is what makes the texture stick to the object instead of the view.
        base = cloud(4000, SIZES["aircraft"], seed=3)
        field = skin.SkinField("aircraft", size=SIZES["aircraft"], seed=5)
        direct = field(base)
        order = np.random.default_rng(1).permutation(len(base))
        shuffled = field(base[order])
        np.testing.assert_array_equal(shuffled, direct[order])
        # a superset must return the same values for the shared points
        extra = cloud(1000, SIZES["aircraft"], seed=4)
        bigger = field(np.concatenate([base, extra]))[:len(base)]
        np.testing.assert_array_equal(bigger, direct)

    def test_texture_is_locked_to_the_object_not_the_camera(self):
        local = cloud(3000, SIZES["aircraft"], seed=2)
        field = skin.SkinField("aircraft", size=SIZES["aircraft"], seed=7)
        want = field(local)
        angle, offset = 0.7, np.array([120.0, -40.0, 15.0])
        c, s = np.cos(angle), np.sin(angle)
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        world = local @ rot.T + offset
        got = field.modulation_world(world, inv_rotation=rot.T, origin=offset)
        np.testing.assert_allclose(got, want, rtol=0.0, atol=1e-9)
        # the same local point maps to the same value however the object is
        # placed in the world
        elsewhere = np.array([-500.0, 30.0, 2500.0])
        other = local @ rot.T + elsewhere
        np.testing.assert_allclose(
            field.modulation_world(other, inv_rotation=rot.T, origin=elsewhere),
            want, rtol=0.0, atol=1e-9)

    def test_disabled_returns_exact_unit_modulation(self):
        pts = cloud(2000, SIZES["aircraft"])
        for category in skin.PRESETS:
            off = skin.SkinField(category, size=SIZES["aircraft"], enabled=False)
            np.testing.assert_array_equal(off(pts), np.ones(len(pts)))
        # every layer off must also be exactly 1.0, not 1.0 +- rounding
        flags = {name: False for name in LAYERS}
        bare = skin.SkinField("aircraft", size=SIZES["aircraft"], **flags)(pts)
        np.testing.assert_array_equal(bare, np.ones(len(pts)))

    def test_layer_switch_removes_exactly_that_layer(self):
        # With the mean compensation out of the way, switching a layer off must
        # subtract exactly that layer's contribution and touch no other layer:
        # the salts and offsets of the others may not move.
        pts = cloud(3000, SIZES["aircraft"])
        full = skin.SkinField("aircraft", size=SIZES["aircraft"], seed=3,
                              normalize_mean=False)(pts)
        without = skin.SkinField("aircraft", size=SIZES["aircraft"], seed=3,
                                 rivets=False, normalize_mean=False)(pts)
        rivets = skin.SkinField("aircraft", size=SIZES["aircraft"], seed=3,
                                normalize_mean=False, **only("rivets"))(pts)
        self.assertTrue(np.all(full <= without + 1e-12),
                        "rivets must only ever darken the paint")
        np.testing.assert_allclose(without - full, 1.0 - rivets, atol=1e-12)


class DistributionTests(unittest.TestCase):
    def test_mean_is_preserved_within_five_percent(self):
        # 24 m at 4 cm/px: much larger than one panel, so the panel and grime
        # layers actually average out.
        for category, size in SIZES.items():
            field, image = surface_patch(category, size, metres=24.0,
                                         pixels=600)
            self.assertLess(abs(image.mean() - 1.0), 0.05,
                            "%s mean %.4f" % (category, image.mean()))
            self.assertLess(abs(image.mean() - 1.0), 0.02,
                            "%s should be closer than the 5%% contract"
                            % category)

    def test_modulation_std_is_in_the_three_to_eight_percent_band(self):
        for category, size in SIZES.items():
            _, image = surface_patch(category, size, metres=16.0, pixels=400)
            self.assertGreater(image.std(), 0.03, category)
            self.assertLess(image.std(), 0.08, category)

    def test_values_stay_in_a_physical_range(self):
        pts = cloud(20000, SIZES["ship"], seed=8)
        for category in skin.PRESETS:
            m = skin.SkinField(category, size=(154.0, 20.0, 12.0))(pts)
            self.assertTrue(np.all(np.isfinite(m)))
            self.assertGreater(float(m.min()), 0.44)
            self.assertLess(float(m.max()), 1.36)

    def test_contrast_knob_scales_the_detail(self):
        std = []
        for contrast in (0.6, 1.0, 1.4, 1.8):
            _, image = surface_patch("aircraft", SIZES["aircraft"],
                                     metres=16.0, pixels=400, contrast=contrast)
            std.append(float(image.std()))
        self.assertEqual(std, sorted(std))
        self.assertGreater(std[-1], 1.4 * std[0])


class LayerScaleTests(unittest.TestCase):
    """The presets claim physical amplitudes; measure them back out."""

    def test_seam_depth_is_ten_to_twenty_percent(self):
        field = skin.SkinField("aircraft", size=SIZES["aircraft"], seed=0,
                               **only("seams"))
        pitch, dtheta, radius = field.pitch, field.dtheta, field.radius
        # walk one pitch of x at angles that sit between two stringer lines
        xs = np.linspace(-0.5 * pitch, 0.5 * pitch, 401)
        theta = np.array([(k + 0.5) * dtheta for k in range(5)])
        points = np.stack([np.repeat(xs, len(theta)),
                           np.tile(radius * np.sin(theta), len(xs)),
                           np.tile(radius * np.cos(theta), len(xs))], axis=1)
        dark = (1.0 - field(points)).reshape(len(xs), len(theta))
        peak = dark.max(axis=0)
        self.assertGreater(float(np.median(peak)), 0.10)
        self.assertLess(float(np.median(peak)), 0.20)
        self.assertLess(float(peak.max()), 0.21)

    def test_seam_fwhm_matches_the_configured_width(self):
        field = skin.SkinField("aircraft", size=SIZES["aircraft"], seed=0,
                               **only("seams"))
        width = field.params["seam_width_m"]
        xs = np.linspace(-0.5 * field.pitch, 0.5 * field.pitch, 2001)
        theta = np.full_like(xs, 0.5 * field.dtheta + 0.13)
        pts = np.stack([xs, field.radius * np.sin(theta),
                        field.radius * np.cos(theta)], axis=1)
        dark = 1.0 - field(pts)
        above = xs[dark >= 0.5 * dark.max()]
        measured = float(above.max() - above.min())
        self.assertAlmostEqual(measured, width, delta=0.4 * width)

    def test_rivets_darken_five_to_ten_percent(self):
        field = skin.SkinField("aircraft", size=SIZES["aircraft"], seed=0,
                               **only("rivets"))
        pts = cloud(200000, SIZES["aircraft"], seed=13)
        dark = 1.0 - field(pts)
        # one rivet head is the 5-10% the preset promises; the double rows that
        # meet at a panel corner overlap and reach ~15%, which is physical
        single = float(np.percentile(dark, 99.9))
        self.assertGreater(single, 0.05)
        self.assertLess(single, 0.105)
        self.assertLess(float(dark.max()), 0.16)

    def test_grime_darkens_five_to_fifteen_percent(self):
        for category, size in SIZES.items():
            field = skin.SkinField(category, size=size, seed=0, **only("grime"))
            pts = cloud(60000, size, seed=14)
            darkest = float((1.0 - field(pts)).max())
            self.assertGreater(darkest, 0.05, category)
            self.assertLess(darkest, 0.16, category)

    def test_grime_streaks_run_downstream(self):
        # -x is the airflow.  A streak is long along x and narrow across it, so
        # the field must change much faster around the body than along it.
        # Measured on the airflow layer alone; the gravity layer is deliberately
        # anisotropic in the other direction.
        field = skin.SkinField("aircraft", size=SIZES["aircraft"], seed=0,
                               grime_gravity=0.0, **only("grime"))
        image = field.patch(4.0, 4.0, shape=(300, 300))
        along_x = np.abs(np.diff(image, axis=0)).mean()
        around = np.abs(np.diff(image, axis=1)).mean()
        self.assertGreater(around / along_x, 2.5)

    def test_panel_offsets_are_one_to_three_percent_and_blocky(self):
        field = skin.SkinField("aircraft", size=SIZES["aircraft"], seed=0,
                               **only("panels"))
        pitch, dtheta, radius = field.pitch, field.dtheta, field.radius
        base_x, base_t = 0.31 * pitch, 0.37

        def point(dx, dt):
            t = base_t + dt
            return [base_x + dx, radius * np.sin(t), radius * np.cos(t)]

        values = field(np.array([point(0.0, 0.0),
                                 point(0.05 * pitch, 0.01 * dtheta),
                                 point(-0.06 * pitch, -0.02 * dtheta)]))
        # one panel, one brightness offset
        self.assertEqual(float(values.min()), float(values.max()))
        self.assertLessEqual(abs(float(values[0]) - 1.0), 0.03)
        # across many panels the offsets are symmetric and small
        m = field(cloud(40000, SIZES["aircraft"], seed=21))
        self.assertLess(abs(float(m.mean()) - 1.0), 0.01)
        self.assertLess(0.5 * float(m.max() - m.min()), 0.031)

    def test_pixel_scale_widens_joints_to_the_pixel_footprint(self):
        for px in (0.01, 0.02, 0.04, 0.06):
            field = skin.SkinField("aircraft", size=SIZES["aircraft"], seed=0,
                                   pixel_scale_m=px)
            desc = field.describe()
            self.assertGreaterEqual(desc["seam_width_px"], 1.5)
            self.assertLessEqual(desc["seam_width_px"], 3.0)
        physical = skin.SkinField("aircraft", size=SIZES["aircraft"], seed=0)
        self.assertAlmostEqual(physical.describe()["seam_width_m"],
                               skin.PRESETS["aircraft"]["seam_width_m"])
        self.assertIsNone(physical.describe()["seam_width_px"])

    def test_joint_spacing_is_metric_and_category_dependent(self):
        aircraft = skin.SkinField("aircraft", size=SIZES["aircraft"])
        warship = skin.SkinField("warship", size=SIZES["ship"])
        uav_size = (11.0, 14.0, 4.0)
        self.assertAlmostEqual(aircraft.pitch,
                               skin.PRESETS["aircraft"]["panel_pitch_m"])
        self.assertGreater(warship.pitch, aircraft.pitch)
        # a bonded composite UAV carries weaker joints than a riveted fighter
        fighter = skin.SkinField("aircraft", size=SIZES["aircraft"],
                                 **only("seams"))
        uav = skin.SkinField("uav_fixed", size=uav_size, **only("seams"))
        fighter_joint = float((1.0 - fighter(cloud(20000, SIZES["aircraft"],
                                                   seed=9))).mean())
        uav_joint = float((1.0 - uav(cloud(20000, uav_size, seed=9))).mean())
        self.assertGreater(fighter_joint, uav_joint)


class ApiTests(unittest.TestCase):
    def test_preset_aliases_cover_categories_and_materials(self):
        for name, key in (("军用飞机", "aircraft"), ("军舰", "warship"),
                          ("民用船舶", "vessel"), ("地面车辆", "vehicle"),
                          ("paint_warship", "warship"),
                          ("paint_civil_air", "airliner"),
                          ("painted_metal", "aircraft")):
            self.assertEqual(skin.preset(name), skin.PRESETS[key])
        self.assertIn("uav_rotor", skin.PRESETS)
        with self.assertRaises(KeyError):
            skin.preset("submarine")

    def test_input_shapes_and_non_finite_points(self):
        field = skin.SkinField("aircraft", size=SIZES["aircraft"], seed=0)
        self.assertEqual(field([1.0, 0.5, 0.2]).shape, (1,))
        matrix = np.array([[1.0, 0.5, 0.2], [2.0, 0.4, 0.1]])
        self.assertEqual(field(matrix).shape, (2,))
        self.assertEqual(field(np.zeros((0, 3))).shape, (0,))
        # a non-finite shading point must not poison a whole tile
        mixed = field(np.array([[1.0, 0.5, 0.2], [np.nan, 0.0, 0.0],
                                [2.0, np.inf, 0.1]]))
        self.assertEqual(float(mixed[1]), 1.0)
        self.assertEqual(float(mixed[2]), 1.0)
        self.assertTrue(np.all(np.isfinite(mixed)))
        with self.assertRaises(ValueError):
            field(np.zeros((4, 2)))

    def test_input_array_is_not_modified(self):
        field = skin.SkinField("aircraft", size=SIZES["aircraft"], seed=0)
        pts = cloud(500, SIZES["aircraft"])
        before = pts.copy()
        field(pts)
        np.testing.assert_array_equal(pts, before)

    def test_patch_matches_direct_evaluation(self):
        field = skin.SkinField("ship", size=SIZES["ship"], seed=4)
        image = field.patch(2.0, 2.0, shape=(24, 31))
        rows, cols = np.meshgrid(np.linspace(-1.0, 1.0, 24),
                                 np.linspace(-1.0, 1.0, 31), indexing="ij")
        r = field.radius
        pts = np.stack([rows.ravel(), r * np.sin(cols.ravel() / r),
                        r * np.cos(cols.ravel() / r)], axis=1)
        np.testing.assert_allclose(field(pts).reshape(24, 31), image, atol=1e-12)

    def test_stats_and_describe_report_the_field(self):
        field = skin.SkinField("vehicle", size=SIZES["vehicle"], seed=2)
        stats = field.stats(cloud(4000, SIZES["vehicle"]))
        self.assertEqual(stats["n"], 4000)
        self.assertLess(abs(stats["mean"] - 1.0), 0.05)
        desc = field.describe()
        for key in ("panel_pitch_m", "ring_count", "arc_spacing_m",
                    "seam_width_m", "seam_depth", "layers"):
            self.assertIn(key, desc)

    def test_hf_ratio_separates_detail_from_a_smooth_field(self):
        rng = np.random.default_rng(0)
        noise = 1.0 + 0.02 * rng.normal(size=(128, 128))
        yy, xx = np.mgrid[0:128, 0:128]
        smooth = 1.0 + 0.02 * np.sin(xx / 40.0)
        self.assertGreater(skin.hf_ratio(noise), skin.hf_ratio(smooth))
        self.assertGreater(skin.hf_ratio(noise), 0.2)

    def test_preview_writes_a_readable_png(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preview.png"
            report = skin.write_preview(str(path), categories=("aircraft",),
                                        shape=(60, 80), width_m=1.0,
                                        height_m=1.0, pixel_scale_m=0.005)
            self.assertTrue(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertGreater(path.stat().st_size, 200)
            self.assertIn("aircraft", report)
            self.assertIn("hf_ratio", report["aircraft"])


class IsolationTests(unittest.TestCase):
    def test_module_imports_only_numpy_and_math(self):
        # The texture is an optical albedo modulation.  It must not reach into
        # the thermal model, the radar model, or a new dependency.
        source = Path(skin.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.extend("%s.%s" % (node.module or "", alias.name)
                                for alias in node.names)
        self.assertEqual(sorted(imported), ["math", "numpy", "simkit.png"])
        for name in ("emissivity", "rf_rho", "thermal", "radar", "heat"):
            self.assertFalse(hasattr(skin, name), name)

    def test_no_thermal_or_radar_module_is_imported_by_skin(self):
        root = Path(skin.__file__).resolve().parents[1]
        code = ("import sys; before = set(sys.modules); import simkit.skin; "
                "new = set(sys.modules) - before; "
                "print(sorted(m for m in new if m.startswith('simkit')))")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                             text=True, cwd=str(root))
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertNotIn("thermal", out.stdout)
        self.assertNotIn("radar", out.stdout)


class PerformanceTests(unittest.TestCase):
    def test_hundred_thousand_points_under_100ms(self):
        pts = cloud(100000, SIZES["aircraft"], seed=31)
        field = skin.SkinField("aircraft", size=SIZES["aircraft"], seed=0)
        field(pts[:100])                      # warm the numpy paths
        timings = []
        for _ in range(5):
            start = time.perf_counter()
            field(pts)
            timings.append(time.perf_counter() - start)
        median = sorted(timings)[2]
        self.assertLess(median, 0.100,
                        "100k points took %.1f ms" % (median * 1e3))
        self.assertLess(min(timings), 0.090)

    def test_large_evaluation_is_chunked_and_stable(self):
        # The chunked path exists so a full-frame evaluation does not allocate
        # hundreds of megabytes.  It must be bit-identical to the direct one.
        pts = cloud(300000, SIZES["aircraft"], seed=32)
        field = skin.SkinField("aircraft", size=SIZES["aircraft"], seed=0)
        start = time.perf_counter()
        chunked = field(pts)
        self.assertLess(time.perf_counter() - start, 1.5)
        pieces = np.concatenate([field(pts[i:i + 50000])
                                 for i in range(0, len(pts), 50000)])
        np.testing.assert_array_equal(chunked, pieces)


if __name__ == "__main__":
    unittest.main()
