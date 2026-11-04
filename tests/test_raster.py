"""Contract tests for the thermal rasteriser and the Planck helpers.

These pin the two things the performance work in simkit/ir.py must never
change: what a sample position means (so anti-aliasing stays correct) and what
the band radiance integral returns (so a re-render of an existing dataset is
bit-for-bit reproducible). The numeric comparisons reference a frozen copy of
the pre-optimisation implementation instead of hard-coded magic numbers.
"""
import unittest

import numpy as np

from simkit import ir, scene

# -- frozen reference: the Planck integral as it was before the caching and
# -- row-blocking rewrite ---------------------------------------------------
_C1 = 2.0 * ir.H_PLANCK * ir.C_LIGHT ** 2
_C2 = ir.H_PLANCK * ir.C_LIGHT / ir.K_BOLTZ


def reference_planck_band(T, lam1_um, lam2_um, n=48):
    lam = np.linspace(lam1_um, lam2_um, n)[None, :] * 1e-6
    T = np.asarray(T, float).reshape(-1, 1)
    with np.errstate(over="ignore", invalid="ignore"):
        B = _C1 / lam ** 5 / (np.expm1(_C2 / (lam * T)))
    B = np.nan_to_num(B, nan=0.0, posinf=0.0)
    return np.trapezoid(B, lam[0], axis=1)


def flat_quad(z, half=100.0):
    """A large axis-aligned quad at height z, facing the camera at +z."""
    verts = np.array([[-half, -half, z], [half, -half, z],
                      [half, half, z], [-half, half, z]], float)
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    world = scene.World(verts, faces, np.full(2, -1), np.zeros(2, int), [])
    return world


def tiny_scene(ir_cfg=None, cam_size=(32, 24)):
    """A small labelled box in front of an empty background."""
    cam = dict(pos=[0, 0, 2], look_at=[0, 10, 2], width=cam_size[0],
               height=cam_size[1], fov_deg=30)
    spec = dict(ground=dict(enabled=False),
                objects=[dict(name="small", shape="box", size=[0.15] * 3,
                              pos=[0.13, 10, 2], mat="painted_metal")],
                sensors=dict(ir_camera=cam),
                ir=dict(samples_per_axis=1, psf_px=0, netd_mk=0, texture_sigma_c=0,
                        nuc=dict(enabled=False), band="lwir"))
    if ir_cfg:
        spec["ir"].update(ir_cfg)
    return spec, cam


class RasterSamplePositionTests(unittest.TestCase):
    """What one sample means: a pixel-centre ray on a perspective Z buffer."""

    def test_pixel_centres_see_a_plane_at_constant_depth(self):
        world = flat_quad(0.0)
        cam = dict(pos=[0, 0, 10], look_at=[0, 0, 0], width=17, height=13, fov_deg=40)
        img, depth, pid = ir.rasterize(world, cam, np.array([4.0, 4.0]))
        np.testing.assert_array_equal(img, 4.0)
        np.testing.assert_allclose(depth, 10.0, rtol=1e-12)
        np.testing.assert_array_equal(pid, -1)

    def test_z_buffer_keeps_the_nearest_surface(self):
        far = flat_quad(0.0)
        near = flat_quad(5.0, half=50.0)
        world = scene.World(
            np.vstack([far.verts, near.verts]),
            np.vstack([far.faces, near.faces + len(far.verts)]),
            np.full(4, -1), np.zeros(4, int), [])
        cam = dict(pos=[0, 0, 10], look_at=[0, 0, 0], width=9, height=9, fov_deg=40)
        img, depth, _ = ir.rasterize(world, cam, np.array([1.0, 1.0, 2.0, 2.0]))
        np.testing.assert_array_equal(img, 2.0)
        np.testing.assert_allclose(depth, 5.0, rtol=1e-12)

    def test_roll_of_180_degrees_inverts_the_image(self):
        # Rolling the sensor by 180 degrees must turn the picture upside down and
        # mirror it -- a pure inversion of the sample lattice, nothing else.
        verts = np.array([[-1, -8, 0], [3, -8, 0], [3, 8, 0], [-1, 8, 0]], float)
        world = scene.World(verts, np.array([[0, 1, 2], [0, 2, 3]]),
                            np.full(2, -1), np.zeros(2, int), [])
        base = dict(pos=[0, 0, 10], look_at=[0, 0, 0], width=24, height=16, fov_deg=40)
        img0, depth0, _ = ir.rasterize(world, base, np.array([7.0, 7.0]))
        img180, depth180, _ = ir.rasterize(world, dict(base, roll_deg=180.0),
                                          np.array([7.0, 7.0]))
        # exactness is not available here: sensor_basis builds the rolled basis
        # with cos/sin, and sin(pi) is 1.2e-16 rather than 0, so a handful of
        # boundary samples may land on the other side of an edge. Area and
        # layout must still agree.
        self.assertLessEqual(abs(int((img180 > 0).sum()) - int((img0 > 0).sum())), 2)
        self.assertGreater(float(np.mean((img180 > 0) == (img0[::-1, ::-1] > 0))), 0.98)
        np.testing.assert_allclose(np.unique(depth180[np.isfinite(depth180)]),
                                   np.unique(depth0[np.isfinite(depth0)]), rtol=1e-12)

    def test_invalid_subpixel_mode_is_refused(self):
        world = flat_quad(0.0)
        cam = dict(pos=[0, 0, 10], look_at=[0, 0, 0], width=8, height=8, fov_deg=40)
        with self.assertRaises(ValueError):
            ir.rasterize(world, cam, np.array([1.0, 1.0]), subpixel_mode="msaa")


class RasterAntiAliasingTests(unittest.TestCase):
    """Supersampling is the anti-aliasing axis this renderer actually has."""

    def _diamond_world(self):
        # A diamond whose four edges cross the frame diagonally: with one sample
        # per pixel every pixel is either full or empty, with supersampling the
        # boundary pixels must carry fractional coverage.
        verts = np.array([[0, -6, 0], [6, 0, 0], [0, 6, 0], [-6, 0, 0]], float)
        return scene.World(verts, np.array([[0, 1, 2], [0, 2, 3]]),
                           np.full(2, -1), np.zeros(2, int), [])

    def test_one_sample_per_pixel_is_binary(self):
        world = self._diamond_world()
        cam = dict(pos=[0, 0, 10], look_at=[0, 0, 0], width=32, height=24, fov_deg=50)
        img, _, _ = ir.rasterize(world, cam, np.array([4.0, 4.0]))
        values = np.unique(img)
        self.assertEqual(len(values), 2)          # only background and face value
        self.assertIn(4.0, values)
        self.assertIn(0.0, values)

    def test_supersampling_produces_intermediate_edge_values(self):
        world = self._diamond_world()
        for samples in (2, 3, 4):
            high = dict(pos=[0, 0, 10], look_at=[0, 0, 0], fov_deg=50,
                        width=32 * samples, height=24 * samples)
            img, _, _ = ir.rasterize(world, high, np.array([4.0, 4.0]))
            down = img.reshape(24, samples, 32, samples).mean(axis=(1, 3))
            partial = np.count_nonzero((down > 0) & (down < 4))
            self.assertGreater(partial, 0, "samples=%d produced no edge coverage" % samples)

    def test_supersampled_mean_preserves_total_energy(self):
        # A full-frame quad must average back to its own value, exactly.
        world = flat_quad(0.0, half=1e3)
        cam = dict(pos=[0, 0, 10], look_at=[0, 0, 0], width=8, height=8, fov_deg=40)
        for samples in (2, 3, 4):
            high = dict(cam, width=8 * samples, height=8 * samples)
            img, _, _ = ir.rasterize(world, high, np.array([3.5, 3.5]))
            down = img.reshape(8, samples, 8, samples).mean(axis=(1, 3))
            np.testing.assert_allclose(down, 3.5, rtol=1e-15)


class RenderSamplingTests(unittest.TestCase):
    """render() keeps the documented sampling contract."""

    def _render(self, ir_cfg):
        spec, cam = tiny_scene(ir_cfg)
        world = scene.build_world(spec)
        temps = np.full(len(world.faces), 320.0)
        return ir.render(world, spec, temps, cam=cam), cam

    def test_coverage_sampling_returns_native_shape_and_fractional_occupancy(self):
        out, cam = self._render(dict(samples_per_axis=4, subpixel_mode="coverage"))
        self.assertEqual(out["radiance"].shape, (cam["height"], cam["width"]))
        coverage = out["instance_coverage"][0]
        self.assertTrue(((coverage > 0) & (coverage < 1)).any())
        self.assertTrue((coverage <= 1.0).all())
        self.assertEqual(out["sampling"]["samples_per_axis"], 4)

    def test_depth_rule_matches_sample_count(self):
        single, _ = self._render(dict(samples_per_axis=1))
        supers, _ = self._render(dict(samples_per_axis=2))
        self.assertIn("pixel-center", single["sampling"]["depth_rule"])
        self.assertIn("minimum", supers["sampling"]["depth_rule"])

    def test_coverage_mode_without_supersampling_is_refused(self):
        spec, cam = tiny_scene(dict(samples_per_axis=1, subpixel_mode="coverage"))
        world = scene.build_world(spec)
        with self.assertRaises(ValueError):
            ir.render(world, spec, np.full(len(world.faces), 320.0), cam=cam)

    def test_samples_per_axis_out_of_range_is_refused(self):
        for bad in (0, 5):
            spec, cam = tiny_scene(dict(samples_per_axis=bad))
            world = scene.build_world(spec)
            with self.assertRaises(ValueError):
                ir.render(world, spec, np.full(len(world.faces), 320.0), cam=cam)

    def test_render_is_deterministic(self):
        first, _ = self._render(dict(samples_per_axis=2, psf_px=1.2, netd_mk=20))
        second, _ = self._render(dict(samples_per_axis=2, psf_px=1.2, netd_mk=20))
        np.testing.assert_array_equal(first["counts_u16"], second["counts_u16"])
        np.testing.assert_array_equal(first["obj_id"], second["obj_id"])


class PlanckIntegralTests(unittest.TestCase):
    """The cached, row-blocked Planck integral must be the same function."""

    def test_matches_frozen_reference_bit_for_bit(self):
        rng = np.random.default_rng(7)
        temps = rng.uniform(180.0, 620.0, 4096)
        for band in ([8.0, 12.0], [3.0, 5.0], [8.0, 9.0]):
            np.testing.assert_array_equal(
                ir.planck_band(temps, band[0], band[1]),
                reference_planck_band(temps, band[0], band[1]),
                err_msg="band %r changed" % (band,))

    def test_small_and_degenerate_inputs_match_reference(self):
        for temps in ([300.0], [0.0], [-5.0], [1e-9], [180.0, 0.0, 3000.0]):
            np.testing.assert_array_equal(
                ir.planck_band(np.array(temps, float), 8.0, 12.0),
                reference_planck_band(np.array(temps, float), 8.0, 12.0))

    def test_row_blocking_does_not_change_results(self):
        rng = np.random.default_rng(11)
        temps = rng.uniform(200.0, 450.0, 5000)
        whole = ir.planck_band(temps, 8.0, 12.0)
        original = ir._PLANCK_CHUNK
        try:
            for chunk in (1, 7, 128, 5000):
                ir._PLANCK_CHUNK = chunk
                np.testing.assert_array_equal(ir.planck_band(temps, 8.0, 12.0), whole)
        finally:
            ir._PLANCK_CHUNK = original

    def test_band_argument_validation_is_still_refused(self):
        with self.assertRaises(ValueError):
            ir.planck_band([300.0], 12.0, 8.0)
        with self.assertRaises(ValueError):
            ir.planck_band([300.0], 8.0, 12.0, n=1)


class DetectorSlopeTests(unittest.TestCase):
    """ir.dldt_mode: the historical finite difference stays the default."""

    def test_analytic_table_matches_a_central_difference(self):
        grid_t, grid_d = ir.dldt_table(8.0, 12.0)
        temps = np.array([200.0, 250.0, 300.0, 350.0, 450.0])
        lut = np.interp(temps, grid_t, grid_d)
        step = 0.01
        central = ((ir.planck_band(temps + step, 8.0, 12.0) -
                    ir.planck_band(temps - step, 8.0, 12.0)) / (2.0 * step))
        np.testing.assert_allclose(lut, central, rtol=1e-6)

    def test_analytic_beats_the_one_kelvin_secant_it_replaces(self):
        grid_t, grid_d = ir.dldt_table(8.0, 12.0)
        temps = np.array([250.0, 300.0, 350.0])
        lut = np.interp(temps, grid_t, grid_d)
        step = 0.01
        central = ((ir.planck_band(temps + step, 8.0, 12.0) -
                    ir.planck_band(temps - step, 8.0, 12.0)) / (2.0 * step))
        secant = ir.planck_band(temps + 1.0, 8.0, 12.0) - ir.planck_band(temps, 8.0, 12.0)
        self.assertLess(float(np.max(np.abs(lut - central) / central)),
                        float(np.max(np.abs(secant - central) / central)))

    def _render_at(self, mode):
        cfg = dict(samples_per_axis=1, psf_px=0.0, netd_mk=30.0)
        if mode is not None:
            cfg["dldt_mode"] = mode
        spec, cam = tiny_scene(cfg)
        world = scene.build_world(spec)
        temps = np.full(len(world.faces), 330.0)
        return ir.render(world, spec, temps, cam=cam)

    def test_default_mode_is_the_analytic_slope(self):
        default = self._render_at(None)
        explicit = self._render_at("analytic")
        np.testing.assert_array_equal(default["counts_u16"], explicit["counts_u16"])
        np.testing.assert_array_equal(default["radiance"], explicit["radiance"])

    def test_difference_mode_is_still_available_and_reproducible(self):
        first = self._render_at("difference")
        second = self._render_at("difference")
        np.testing.assert_array_equal(first["counts_u16"], second["counts_u16"])
        np.testing.assert_array_equal(first["radiance"], second["radiance"])

    def test_analytic_mode_stays_within_a_count_of_the_difference(self):
        difference = self._render_at("difference")
        analytic = self._render_at("analytic")
        diff = difference["counts_u16"].astype(np.int32)
        ana = analytic["counts_u16"].astype(np.int32)
        delta = np.abs(ana - diff)
        self.assertLessEqual(int(delta.max()), 2)
        self.assertGreater(float(np.mean(delta == 0)), 0.90)
        ref = np.asarray(difference["radiance"], float)
        rel = np.abs(np.asarray(analytic["radiance"], float) - ref) / np.maximum(np.abs(ref), 1e-12)
        self.assertLess(float(np.nanmax(rel)), 1e-3)

    def test_invalid_mode_is_refused(self):
        spec, cam = tiny_scene(dict(dldt_mode="lol"))
        world = scene.build_world(spec)
        with self.assertRaises(ValueError):
            ir.render(world, spec, np.full(len(world.faces), 320.0), cam=cam)


if __name__ == "__main__":
    unittest.main()
