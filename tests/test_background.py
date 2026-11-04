# -*- coding: utf-8 -*-
"""Background-image support: IR plate compositing and the optical env map.

The optical renderer runs inside Blender and is not started here (a Cycles
render is far too slow for the unit suite); its node wiring is checked
statically instead.  The IR side is pure numpy/Pillow and is measured directly.
"""
import hashlib
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from simkit import ir, scene


ROOT = Path(__file__).resolve().parents[1]

# Captured from the implementation BEFORE background images existed: these are
# the exact bytes ir.render() produced for the scene below.  Keeping them here
# is what proves the new code path is inert when no plate is configured.
REFERENCE_COUNTS_SHA256 = "cfad5f3d710eff2c33a8ab54a3bbc87a3ef4d291e5be122e05bdc50f9548713b"
REFERENCE_PREVIEW_SHA256 = "187e3b5a5a9c19e540bf1e3fca478bac2db688492524bb4b002f60a56c629144"
REFERENCE_RADIANCE_SHA256 = "e405ffdb92bbf8c3d37f75b063a832b29bffe64c75e2c8e56ecb02a75ee86699"


def camera(width=24, height=18):
    return dict(pos=[0.0, 0.0, 2.0], look_at=[0.0, 10.0, 1.0],
                width=width, height=height, fov_deg=20.0)


def base_cfg(background=None, background_temp_k=None, top_level=False):
    """One target over empty sky: ~27 % of the frame is geometry."""
    cfg = dict(
        ground=dict(enabled=False),
        sun=dict(on=False),
        air=dict(temp_c=20.0),

        seed=7,
        sensors=dict(ir_camera=camera()),
        objects=[dict(name="target", shape="box", size=[1.5, 1.5, 1.5],
                      pos=[0.0, 10.0, 1.0], mat="painted_metal")],
        # The plate's radiometric anchor defaults to the scene's ground
        # temperature; pin it so these tests do not move when that default does.
        ir=dict(netd_mk=0.0, psf_px=0.0, texture_sigma_c=0.0, gain=1.0,
                background_temp_k=260.0,
                nuc=dict(enabled=False), preview_mode="temp"),
    )
    if background is not None:
        if top_level:
            cfg["background_image"] = background
        else:
            cfg["ir"]["background_image"] = background
    if background_temp_k is not None:
        cfg["ir"]["background_temp_k"] = background_temp_k
    return cfg


def render(cfg):
    world = scene.build_world(cfg)
    temps = np.full(len(world.faces), 300.0)
    return world, ir.render(world, cfg, temps)


class NoBackgroundRegressionTests(unittest.TestCase):
    def test_no_background_is_bit_identical_between_runs_and_to_reference(self):
        cfg = base_cfg()
        world, first = render(cfg)
        second = ir.render(world, cfg, np.full(len(world.faces), 300.0))
        for key in ("counts_u16", "preview", "apparent_temp_k", "radiance",
                    "depth", "obj_id"):
            np.testing.assert_array_equal(first[key], second[key], err_msg=key)
        self.assertEqual(
            hashlib.sha256(first["counts_u16"].tobytes()).hexdigest(),
            REFERENCE_COUNTS_SHA256)
        self.assertEqual(
            hashlib.sha256(first["preview"].tobytes()).hexdigest(),
            REFERENCE_PREVIEW_SHA256)
        self.assertEqual(
            hashlib.sha256(np.ascontiguousarray(first["radiance"], dtype=np.float64).tobytes()).hexdigest(),
            REFERENCE_RADIANCE_SHA256)


class BackgroundCompositeTests(unittest.TestCase):
    def test_only_unoccluded_pixels_change_and_the_target_is_never_covered(self):
        with tempfile.TemporaryDirectory() as temp:
            plate_path = os.path.join(temp, "white.png")
            Image.new("L", (48, 36), 255).save(plate_path)
            cfg = base_cfg(background=plate_path)
            world, plain = render(base_cfg())
            out = ir.render(world, cfg, np.full(len(world.faces), 300.0))
            void = ~np.isfinite(plain["depth"])
            hit = ~void
            self.assertTrue(void.any())
            self.assertTrue(hit.any())

            expected_radiance = float(ir.planck_band([260.0], 8.0, 12.0)[0])
            np.testing.assert_allclose(out["radiance"][void], expected_radiance,
                                       rtol=1e-12)
            np.testing.assert_allclose(out["apparent_temp_k"][void], 260.0,
                                       atol=0.5)
            self.assertEqual(np.unique(out["preview"][void]).tolist(), [255])
            self.assertNotEqual(np.unique(plain["preview"][void]).tolist(),
                                np.unique(out["preview"][void]).tolist())
            # The whole point: the hot target keeps every original value.
            for key in ("counts_u16", "preview", "apparent_temp_k", "radiance",
                        "depth", "obj_id"):
                np.testing.assert_array_equal(out[key][hit], plain[key][hit],
                                              err_msg=key)
            np.testing.assert_array_equal(np.isfinite(out["depth"]),
                                          np.isfinite(plain["depth"]))

    def test_top_level_key_and_configured_background_temperature(self):
        with tempfile.TemporaryDirectory() as temp:
            plate_path = os.path.join(temp, "plate.png")
            Image.new("L", (48, 36), 255).save(plate_path)
            cfg = base_cfg(background=plate_path, background_temp_k=300.0,
                           top_level=True)
            world, plain = render(base_cfg())
            out = ir.render(world, cfg, np.full(len(world.faces), 300.0))
            void = ~np.isfinite(out["depth"])
            expected = float(ir.planck_band([300.0], 8.0, 12.0)[0])
            np.testing.assert_allclose(out["radiance"][void], expected,
                                       rtol=1e-12)
            np.testing.assert_allclose(out["apparent_temp_k"][void], 300.0,
                                       atol=0.5)
            hit = ~void
            np.testing.assert_array_equal(out["counts_u16"][hit],
                                          plain["counts_u16"][hit])

    def test_missing_background_image_raises_chinese_value_error(self):
        cfg = base_cfg(background=os.path.join("does", "not", "exist.png"))
        world = scene.build_world(cfg)
        with self.assertRaisesRegex(ValueError, "红外背景图不存在"):
            ir.render(world, cfg, np.full(len(world.faces), 300.0))

    def test_undecodable_background_image_raises_chinese_value_error(self):
        with tempfile.TemporaryDirectory() as temp:
            plate_path = os.path.join(temp, "broken.png")
            with open(plate_path, "wb") as handle:
                handle.write(b"not a png at all")
            cfg = base_cfg(background=plate_path)
            world = scene.build_world(cfg)
            with self.assertRaisesRegex(ValueError, "红外背景图无法解码"):
                ir.render(world, cfg, np.full(len(world.faces), 300.0))

    def test_plate_is_center_cropped_and_resized_for_a_mismatched_frame(self):
        with tempfile.TemporaryDirectory() as temp:
            # 3:1 plate: a centred white band survives the crop to the 4:3 frame
            wide = np.zeros((30, 90), np.uint8)
            wide[:, 25:65] = 255
            wide_path = os.path.join(temp, "wide.png")
            Image.fromarray(wide).save(wide_path)
            plate = ir._background_plate(wide_path, 24, 18)
            self.assertEqual(plate.shape, (18, 24))
            np.testing.assert_allclose(plate, 1.0, atol=1e-12)

            # 1:3 plate: the same test through the vertical crop branch
            tall = np.zeros((90, 30), np.uint8)
            tall[25:65, :] = 255
            tall_path = os.path.join(temp, "tall.png")
            Image.fromarray(tall).save(tall_path)
            plate = ir._background_plate(tall_path, 24, 18)
            self.assertEqual(plate.shape, (18, 24))
            np.testing.assert_allclose(plate, 1.0, atol=1e-12)

            # ...and a full render with a plate whose size matches no camera
            cfg = base_cfg(background=wide_path)
            world, plain = render(base_cfg())
            out = ir.render(world, cfg, np.full(len(world.faces), 300.0))
            void = ~np.isfinite(out["depth"])
            self.assertEqual(np.unique(out["preview"][void]).tolist(), [255])
            np.testing.assert_array_equal(out["counts_u16"][~void],
                                          plain["counts_u16"][~void])

    def test_depth_zero_is_treated_as_an_unoccluded_pixel(self):
        with tempfile.TemporaryDirectory() as temp:
            plate_path = os.path.join(temp, "white.png")
            Image.new("L", (8, 8), 255).save(plate_path)
            depth = np.full((8, 8), 5.0)
            depth[0, 0] = 0.0
            img = np.full((8, 8), 1.0)
            counts = np.zeros((8, 8))
            preview = np.zeros((8, 8))
            temperature = np.full((8, 8), 300.0)
            cfg = base_cfg(background=plate_path)
            out = ir._composite_background(img, counts, preview, temperature,
                                           depth, cfg, [8.0, 12.0], 1.0,
                                           200.0, 450.0)
            expected = float(ir.planck_band([260.0], 8.0, 12.0)[0])
            self.assertEqual(float(out[0][0, 0]), expected)
            self.assertEqual(float(out[2][0, 0]), 1.0)
            # the other pixels have finite non-zero depth and are untouched
            np.testing.assert_array_equal(out[0][1:], img[1:])
            np.testing.assert_array_equal(out[3][1:], temperature[1:])

    def test_relative_path_resolves_against_scene_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = base_cfg(background=os.path.join("plates", "sky.png"))
            cfg["_base_dir"] = temp
            self.assertEqual(
                ir._background_image_path(cfg),
                os.path.normpath(os.path.join(temp, "plates", "sky.png")))


class ScenePassThroughTests(unittest.TestCase):
    def test_build_world_keeps_sensor_presentation_keys(self):
        sky = os.path.join("skies", "day.hdr")
        plate = os.path.join("plates", "ir.png")
        cfg = base_cfg(background=plate)
        cfg["optical"] = dict(sky_image=sky, sky_strength=1.4)
        world = scene.build_world(cfg)
        self.assertGreater(len(world.faces), 0)
        self.assertEqual(cfg["optical"]["sky_image"], sky)
        self.assertEqual(cfg["optical"]["sky_strength"], 1.4)
        self.assertEqual(cfg["ir"]["background_image"], plate)


class OpticalEnvironmentMapSourceTests(unittest.TestCase):
    def test_optical_renderer_wires_an_environment_texture(self):
        source = (ROOT / "blender" / "render_optical.py").read_text(encoding="utf-8")
        self.assertIn("ShaderNodeTexEnvironment", source)
        self.assertIn("ShaderNodeTexCoord", source)
        self.assertIn('opt.get("sky_image")', source)
        self.assertIn("bpy.data.images.load(path, check_existing=True)", source)
        for extension in (".hdr", ".exr", ".png", ".jpg"):
            self.assertIn(extension, source)

class RadianceHdrTests(unittest.TestCase):
    """The IR compositor must accept a Radiance plate.

    Every CC0 sky plate is .hdr and Blender takes it as an environment texture,
    but Pillow cannot open Radiance. Before this decoder, choosing the built-in
    sky and an IR band failed the whole run with "cannot identify image file".
    """

    @staticmethod
    def _write_hdr(path, width=4, height=2, rgbe=(128, 64, 32, 136)):
        """Radiance stores a flat scanline channel-planar: RRRR GGGG BBBB EEEE.

        Writing it pixel-interleaved (RGBE RGBE ...) is not a layout the format
        defines; the test data has to follow the spec, not the other way round.
        """
        r, g, b, e = rgbe
        head = (b"#?RADIANCE\n"
                b"FORMAT=32-bit_rle_rgbe\n"
                b"\n"
                b"-Y %d +X %d\n" % (height, width))
        body = (bytes([r]) * width + bytes([g]) * width
                + bytes([b]) * width + bytes([e]) * width) * height
        with open(path, "wb") as fh:
            fh.write(head + body)

    def test_flat_scanline_decodes_to_known_radiance(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "sky.hdr")
            self._write_hdr(p)
            arr = ir._read_radiance_hdr(p)
        self.assertEqual(arr.shape, (2, 4, 3))
        # e == 136 means scale 2^(136-128-8) == 1, so the bytes ARE the values
        self.assertAlmostEqual(float(arr[0, 0, 0]), 128.0, places=3)
        self.assertAlmostEqual(float(arr[0, 0, 1]), 64.0, places=3)
        self.assertAlmostEqual(float(arr[0, 0, 2]), 32.0, places=3)

    def test_rle_scanline_decodes_to_the_same_values(self):
        """Width >= 8 switches Radiance to per-channel RLE."""
        import tempfile
        width, height = 8, 2
        head = (b"#?RADIANCE\n\n-Y %d +X %d\n" % (height, width))
        body = b""
        for _ in range(height):
            body += bytes([2, 2, 0, width])          # new-style RLE marker
            for value in (128, 64, 32, 136):
                body += bytes([128 + width, value])  # one run per channel
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "rle.hdr")
            with open(p, "wb") as fh:
                fh.write(head + body)
            arr = ir._read_radiance_hdr(p)
        self.assertEqual(arr.shape, (height, width, 3))
        self.assertTrue(np.allclose(arr, np.array([128.0, 64.0, 32.0])), arr[0, 0])

    def test_plate_loads_an_hdr_file(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "sky.hdr")
            self._write_hdr(p, width=8, height=4)
            plate = ir._background_plate(p, 6, 4)
        self.assertEqual(plate.shape, (4, 6))
        self.assertGreater(float(plate.max()), 0.0)
        self.assertLessEqual(float(plate.max()), 1.0)

    def test_truncated_hdr_is_a_chinese_error(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "bad.hdr")
            with open(p, "wb") as fh:
                fh.write(b"#?RADIANCE\n\n-Y 4 +X 8\n" + bytes([2, 2, 0, 8]))
            with self.assertRaises(ValueError) as ctx:
                ir._read_radiance_hdr(p)
        self.assertIn("截断", str(ctx.exception))


class SeaClutterTests(unittest.TestCase):
    """Sea state has to reach the radar, not just the wave mesh."""

    def test_wind_raises_backscatter_monotonically(self):
        from simkit import radar
        vals = [radar.sea_clutter_db(w, 0.05) for w in (2, 6, 10, 15, 20)]
        self.assertEqual(vals, sorted(vals))
        self.assertGreater(vals[-1] - vals[0], 15.0)

    def test_grazing_angle_raises_backscatter(self):
        import math
        from simkit import radar
        low = radar.sea_clutter_db(8.0, math.sin(math.radians(1)))
        high = radar.sea_clutter_db(8.0, math.sin(math.radians(30)))
        self.assertGreater(high, low)

    def test_calm_sea_is_much_quieter_than_a_gale(self):
        from simkit import radar
        calm = radar.sea_clutter_db(3.0, 0.02)
        gale = radar.sea_clutter_db(18.0, 0.02)
        self.assertGreater(gale - calm, 20.0)


if __name__ == "__main__":
    unittest.main()
