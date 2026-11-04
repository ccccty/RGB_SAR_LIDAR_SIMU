# -*- coding: utf-8 -*-
"""Camera defects: sensor noise through a Bayer CFA, lens aberrations, MTF.

Every stage is tested twice: switched off it must leave the frame bit for bit
alone (the renderer relies on that to skip the read/modify/write cycle), and
switched on it must produce the measurable effect the module promises.  The
numerical bands are not invented here -- they are the ranges a real ISO-100
camera shows on its 8-bit output: 1.5-3 DN in the shadows, 1-2.5 DN on the sky,
MTF50 0.20-0.30 cycles/pixel for an f/8 lens behind the renderer's own
antialiasing.
"""
import json
import unittest
from pathlib import Path

import numpy as np

from simkit import camera, ir, scene


ROOT = Path(__file__).resolve().parents[1]


def all_off(**overrides):
    block = {name: dict(enabled=False) for name in camera.STAGES}
    for name, values in overrides.items():
        block.setdefault(name, dict(enabled=False))
        block[name] = dict(block[name], **values)
    return dict(optical=dict(camera=block))


def only(stage, **values):
    """Every stage off except one."""
    return all_off(**{stage: dict(enabled=True, **values)})


def grey(value, size=193):
    return np.full((size, size, 3), value, np.uint8)


def interior(image, margin=10):
    return image[margin:-margin, margin:-margin]


def autocorrelation(plane, dy=1, dx=0):
    left = plane[:-dy or None, :-dx or None].ravel()
    right = plane[dy:, dx:].ravel()
    return float(np.corrcoef(left, right)[0, 1])


class SwitchTests(unittest.TestCase):
    def test_all_switches_off_returns_the_input_object_untouched(self):
        image = grey(128)
        result = camera.apply(image, all_off(), frame_seed=3)
        self.assertIs(result["image_u8"], image)
        self.assertFalse(result["changed"])
        self.assertEqual(result["metadata"]["applied"], [])
        self.assertFalse(camera.has_effect(all_off()))

    def test_default_switches_are_documented_and_active(self):
        stages = camera.active_stages(camera.scene_settings({}))
        # Noise and the optical MTF are on by default; the three lens defects
        # that would repaint every existing frame stay off until asked for.
        self.assertIn("sensor", stages)
        self.assertIn("diffraction", stages)
        self.assertNotIn("vignetting", stages)
        self.assertNotIn("chromatic_aberration", stages)
        self.assertNotIn("flare", stages)
        self.assertNotIn("jpeg", stages)
        self.assertTrue(camera.has_effect({}))

    def test_skip_removes_a_stage(self):
        own = camera.scene_settings({})
        self.assertNotIn("sensor", camera.active_stages(own, skip=("sensor",)))
        self.assertFalse(camera.has_effect(all_off(sensor=dict(enabled=False))))

    def test_invalid_numbers_raise(self):
        cases = [
            ("sensor", dict(iso=0)),
            ("sensor", dict(full_well_e=-1)),
            ("sensor", dict(bits=9.5)),
            ("sensor", dict(cfa_pattern="RRGB")),
            ("vignetting", dict(f_number=0)),
            ("vignetting", dict(focal_mm=50)),
            ("chromatic_aberration", dict(lateral_px=-1)),
            ("diffraction", dict(f_number=0)),
            ("flare", dict(threshold=1.5)),
            ("jpeg", dict(quality=101)),
        ]
        for stage, values in cases:
            cfg = {name: dict(enabled=False) for name in camera.STAGES}
            cfg[stage] = dict(enabled=True)
            cfg[stage].update(values)
            other = "sensor" if stage != "sensor" else "jpeg"
            with self.assertRaises(ValueError):
                if stage == "vignetting":
                    camera.apply_vignetting(np.zeros((8, 8, 3)), dict(
                        camera.DEFAULTS["vignetting"], **values))
                else:
                    camera.apply(grey(32), dict(optical=dict(camera=cfg)))


class BayerSensorTests(unittest.TestCase):
    def test_mosaic_keeps_exactly_one_colour_per_photosite(self):
        rng = np.random.default_rng(0)
        frame = rng.random((16, 16, 3))
        cfa = camera.mosaic(frame, "RGGB")
        self.assertEqual(cfa.shape, (16, 16))
        np.testing.assert_array_equal(cfa[0::2, 0::2], frame[0::2, 0::2, 0])
        np.testing.assert_array_equal(cfa[0::2, 1::2], frame[0::2, 1::2, 1])
        np.testing.assert_array_equal(cfa[1::2, 0::2], frame[1::2, 0::2, 1])
        np.testing.assert_array_equal(cfa[1::2, 1::2], frame[1::2, 1::2, 2])

    def test_demosaic_reconstructs_a_flat_field_exactly(self):
        for pattern in ("RGGB", "BGGR", "GRBG", "GBRG"):
            for value in (0.04, 0.5, 0.9):
                frame = np.full((24, 32, 3), value)
                rebuilt = camera.demosaic(camera.mosaic(frame, pattern), pattern)
                np.testing.assert_allclose(rebuilt, frame, atol=1e-12,
                                           err_msg=pattern)

    def test_shadow_noise_lands_in_the_real_iso100_band(self):
        result = camera.apply(grey(8), only("sensor"))
        sigma = float(interior(result["image_u8"]).std())
        self.assertGreaterEqual(sigma, 1.5)
        self.assertLessEqual(sigma, 3.0)

    def test_sky_noise_lands_in_the_real_iso100_band(self):
        result = camera.apply(grey(111), only("sensor"))
        sigma = float(interior(result["image_u8"]).std())
        self.assertGreaterEqual(sigma, 1.0)
        self.assertLessEqual(sigma, 2.5)

    def test_pure_black_is_not_rectified_by_the_adc_clamp(self):
        # Without the output black level the read noise is half-normal at zero
        # and measures ~0.6 of its true sigma; the pedestal keeps it measurable.
        result = camera.apply(grey(0), only("sensor"))
        frame = interior(result["image_u8"])
        self.assertGreater(float(frame.mean()), 3.0)
        self.assertGreaterEqual(float(frame.std()), 1.5)

    def test_iso_scales_shot_noise(self):
        sigmas = []
        for iso in (100, 400, 1600):
            result = camera.apply(grey(128), only("sensor", iso=iso))
            sigmas.append(float(interior(result["image_u8"]).std()))
        self.assertLess(sigmas[0], sigmas[1])
        self.assertLess(sigmas[1], sigmas[2])
        # shot noise grows as sqrt(gain); allow the readout floor to soften it
        self.assertGreater(sigmas[2] / sigmas[0], 2.0)

    def test_shot_noise_preserves_the_mean(self):
        result = camera.apply(grey(160), only("sensor", read_noise_dn=0,
                                              read_noise_e=0, dsnu_dn=0,
                                              row_fpn_dn=0, column_fpn_dn=0,
                                              row_noise_dn=0, prnu=0,
                                              black_level_dn=0))
        self.assertAlmostEqual(float(interior(result["image_u8"]).mean()),
                               160.0, delta=1.0)

    def test_fixed_pattern_is_stable_and_temporal_noise_is_not(self):
        frozen = only("sensor", shot_noise=False, read_noise_dn=0, read_noise_e=0,
                      row_noise_dn=0)
        fixed = camera.apply(grey(120), frozen, frame_seed=1)["image_u8"]
        again = camera.apply(grey(120), frozen, frame_seed=999)["image_u8"]
        np.testing.assert_array_equal(fixed, again)
        self.assertGreater(float(interior(fixed).std()), 0.0)
        first = camera.apply(grey(120), only("sensor"), frame_seed=1)["image_u8"]
        second = camera.apply(grey(120), only("sensor"), frame_seed=2)["image_u8"]
        self.assertFalse(np.array_equal(first, second))

    def test_demosaic_turns_grain_into_coloured_clumps(self):
        with_cfa = camera.apply(grey(111), only("sensor"),
                                frame_seed=5)["image_u8"].astype(np.float64)
        without = camera.apply(grey(111), only("sensor", demosaic=False),
                               frame_seed=5)["image_u8"].astype(np.float64)
        chroma = interior(with_cfa[:, :, 0] - with_cfa[:, :, 1])
        chroma -= chroma.mean()
        flat = interior(without[:, :, 0] - without[:, :, 1])
        flat -= flat.mean()
        # One red sample is copied into the four green sites around it, so the
        # chroma noise is spatially correlated -- the coloured confetti of a
        # real Bayer sensor.  Per-channel noise has no such structure.
        self.assertGreater(autocorrelation(chroma), 0.25)
        self.assertLess(abs(autocorrelation(flat)), 0.1)

    def test_luma_noise_is_below_per_channel_noise(self):
        frame = interior(camera.apply(grey(111), only("sensor"),
                                      frame_seed=5)["image_u8"].astype(np.float64))
        per_channel = float(frame.std())
        luma = float(frame.mean(axis=2).std())
        self.assertLess(luma, per_channel)


def mtf50(profile, step=1.0):
    """MTF50 (cycles/pixel) of an edge spread function."""
    values = np.asarray(profile, np.float64)
    values = values - values.min()
    values = values / max(values.max(), 1e-12)
    lsf = np.gradient(values, step)
    window = 8192
    frequency = np.fft.rfftfreq(window, d=step)
    magnitude = np.abs(np.fft.rfft(lsf * np.hanning(len(lsf)), window))
    magnitude /= magnitude[0]
    below = np.where(magnitude <= 0.5)[0]
    if not len(below):
        return None
    index = int(below[0])
    if index == 0:
        return 0.0
    return float(frequency[index - 1] + (magnitude[index - 1] - 0.5)
                 * (frequency[index] - frequency[index - 1])
                 / (magnitude[index - 1] - magnitude[index]))


class LensTests(unittest.TestCase):
    def test_vignetting_follows_cos4_and_only_darkens(self):
        factor = camera.vignette_map(201, 201, fov_deg=90.0)
        self.assertAlmostEqual(float(factor[100, 100]), 1.0, delta=1e-9)
        self.assertLess(float(factor[0, 0]), 0.2)
        # independent check of the law on the 3-4-5 triangle off axis
        f_px = 0.5 * 201 / np.tan(np.radians(90.0) / 2.0)
        pixel = factor[100, 160]
        cos = f_px / np.hypot(f_px, 60.0)
        self.assertAlmostEqual(float(pixel), cos ** 4, delta=1e-9)

    def test_vignetting_off_is_identity_and_on_darkens_the_corner(self):
        image = grey(150, 129)
        off = camera.apply(image, all_off())
        np.testing.assert_array_equal(off["image_u8"], image)
        on = camera.apply(image, only("vignetting", fov_deg=60.0))["image_u8"]
        self.assertLess(float(on[0, 0].mean()), float(image[0, 0].mean()) - 10)
        self.assertAlmostEqual(float(on[64, 64].mean()), float(image[64, 64].mean()),
                               delta=1.0)

    def test_wider_aperture_vignettes_more(self):
        slow = camera.vignette_map(101, 101, fov_deg=80.0, f_number=8.0,
                                   mechanical=0.4)
        fast = camera.vignette_map(101, 101, fov_deg=80.0, f_number=2.0,
                                   mechanical=0.4)
        self.assertLess(float(fast[0, 0]), float(slow[0, 0]))

    def test_vignetting_needs_a_field_of_view(self):
        with self.assertRaises(ValueError):
            camera.vignette_map(16, 16)

    def test_chromatic_aberration_moves_red_out_and_blue_in(self):
        image = np.zeros((121, 121, 3), np.uint8)
        image[60:63, 100:103] = 255                 # spot 40 px off axis
        result = camera.apply(image, only("chromatic_aberration", lateral_px=2.0))
        self.assertTrue(result["changed"])
        frame = result["image_u8"].astype(np.float64)

        def centroid(plane):
            y, x = np.mgrid[0:plane.shape[0], 0:plane.shape[1]]
            total = plane.sum()
            return float((x * plane).sum() / total)

        red = centroid(frame[:, :, 0])
        green = centroid(frame[:, :, 1])
        blue = centroid(frame[:, :, 2])
        self.assertGreater(red, green + 0.3)
        self.assertLess(blue, green - 0.3)
        # green is the reference plane and must be untouched
        np.testing.assert_array_equal(frame[:, :, 1], image[:, :, 1].astype(np.float64))

    def test_zero_aberration_is_identity(self):
        image = grey(64, 65)
        result = camera.apply(image, only("chromatic_aberration", lateral_px=0.0))
        self.assertTrue(result["changed"])
        np.testing.assert_array_equal(result["image_u8"], image)

    def test_flare_without_a_highlight_is_bit_identical(self):
        image = grey(40, 65)
        result = camera.apply(image, only("flare"))
        self.assertTrue(result["changed"])
        np.testing.assert_array_equal(result["image_u8"], image)

    def test_flare_adds_a_halo_and_a_ghost_around_a_highlight(self):
        image = np.full((96, 96, 3), 20, np.uint8)
        image[16:21, 16:21] = 255
        result = camera.apply(image, only("flare", strength=0.5, radius_px=10.0,
                                          ghost_scale=0.8, ghost_count=1))["image_u8"]
        self.assertGreater(int(result[16, 34].max()), 20)
        # mirrored through the centre: (16,16) -> about (79,79)
        self.assertGreater(int(result[79, 79].max()), 20)

    def test_jpeg_switch_and_quality(self):
        rng = np.random.default_rng(3)
        image = np.clip(rng.normal(128, 40, (64, 64, 3)), 0, 255).astype(np.uint8)
        off = camera.apply(image, all_off())
        np.testing.assert_array_equal(off["image_u8"], image)
        high = camera.apply(image, only("jpeg", quality=95))["image_u8"].astype(float)
        low = camera.apply(image, only("jpeg", quality=40))["image_u8"].astype(float)
        self.assertFalse(np.array_equal(high.astype(np.uint8), image))
        rmse = lambda a: float(np.sqrt(((a - image.astype(float)) ** 2).mean()))
        self.assertGreater(rmse(low), rmse(high))
        with self.assertRaises(ValueError):
            camera.jpeg_roundtrip(image, 0)
        with self.assertRaises(ValueError):
            camera.active_stages(camera.settings(dict(jpeg=dict(
                enabled=True, quality_range=[90, 40]))))
        # per-frame random quality, as a phone or a scraped album produces
        drawn = {camera.apply(image, only("jpeg", quality_range=[60, 90]),
                              frame_seed=seed)["metadata"]["jpeg_quality"]
                 for seed in range(12)}
        self.assertTrue(all(60 <= q <= 90 for q in drawn))
        self.assertGreater(len(drawn), 1)


class DiffractionTests(unittest.TestCase):
    def test_otf_has_a_hard_cutoff_unlike_a_gaussian(self):
        shape = (64, 64)
        # f/22 on 6 um pixels puts the cutoff just inside Nyquist, where the
        # shape of the OTF can be sampled; the default f/8 is nearly aberration
        # free at this pitch and its cutoff sits above Nyquist.
        otf = camera.diffraction_otf(shape, f_number=22.0, wavelength_um=0.55,
                                     pixel_pitch_um=6.0, pixel_aperture=False)
        cutoff = 6.0 / (0.55 * 22.0)
        fy = np.fft.fftfreq(64)[:, None]
        fx = np.fft.rfftfreq(64)[None, :]
        frequency = np.sqrt(fx * fx + fy * fy)
        near_cutoff = otf.ravel()[np.argmin(np.abs(frequency - 0.97 * cutoff))]
        self.assertLess(float(near_cutoff), 0.15)
        # a Gaussian with the same MTF50 never reaches zero
        sigma = 0.4
        gaussian = np.exp(-2 * np.pi ** 2 * sigma ** 2 * (0.97 * cutoff) ** 2)
        self.assertGreater(float(gaussian), 0.2)

    def test_mtf50_lands_in_the_real_lens_band(self):
        # The lens stage alone on an ideal step: a perfect step has a flat
        # spectrum, so what is left is the aperture OTF times the pixel
        # aperture -- the optical MTF of an f/8 lens on 6 um pixels.  Cascaded
        # with the renderer's own pixel filter (measured MTF50 0.33 on a real
        # Cycles frame) the system lands at ~0.24, in the same band.
        step = np.zeros((48, 256, 3), np.uint8)
        step[:, 128:] = 255
        result = camera.apply(step, only("diffraction", f_number=8.0,
                                         pixel_pitch_um=6.0))["image_u8"]
        profile = result.astype(np.float64).mean(axis=2).mean(axis=0)
        value = mtf50(profile)
        self.assertIsNotNone(value)
        self.assertGreaterEqual(value, 0.20)
        self.assertLessEqual(value, 0.30)

    def test_diffraction_off_is_bit_identical(self):
        image = (np.arange(64 * 64, dtype=np.uint8).reshape(64, 64)[:, :, None]
                 * np.ones((1, 1, 3), np.uint8))
        result = camera.apply(image, all_off())
        np.testing.assert_array_equal(result["image_u8"], image)


class RendererWiringTests(unittest.TestCase):
    def test_blender_renderer_applies_the_camera_chain(self):
        source = (ROOT / "blender" / "render_optical.py").read_text(encoding="utf-8")
        self.assertIn("from simkit import camera as sk_camera", source)
        self.assertIn("sk_camera.has_effect(cfg, skip=CAMERA_SKIP)", source)
        self.assertIn("sk_camera.apply(", source)
        self.assertIn("read_rendered_rgb8(path)", source)
        self.assertIn("camera_model=camera_metadata", source)
        self.assertIn("Non-Color", source)
        # the physical electron model owns the detector when it is enabled
        self.assertIn('("sensor", "diffraction")', source)


class InfraredPreviewWindowTests(unittest.TestCase):
    """The AGC window must be a display decision, not a frame accident."""

    @staticmethod
    def scene_config(span=None):
        cfg = dict(
            ground=dict(enabled=False), sun=dict(on=False), air=dict(temp_c=20.0),
            seed=7, sensors=dict(ir_camera=dict(
                pos=[0.0, 0.0, 2.0], look_at=[0.0, 10.0, 1.0],
                width=160, height=120, fov_deg=20.0)),
            objects=[dict(name="target", shape="box", size=[1.5, 1.5, 1.5],
                          pos=[0.0, 10.0, 1.0], mat="painted_metal")],
            ir=dict(netd_mk=28.0, psf_px=1.2, texture_sigma_c=0.0, gain=1.0,
                    nuc=dict(enabled=True, prnu=0.008, dsnu_counts=25,
                             column_fpn_counts=6, seed=12345),
                    preview_mode="temp"))
        if span is not None:
            cfg["ir"]["preview_span_c"] = span
        return cfg

    @staticmethod
    def render(cfg, temperature=300.0):
        world = scene.build_world(cfg)
        return ir.render(world, cfg, np.full(len(world.faces), temperature))

    def test_default_reproduces_the_historical_percentile_stretch(self):
        t_img = np.concatenate([np.full(900, 257.0), np.full(100, 299.0)])
        low, high = ir.preview_window(t_img, {})
        self.assertAlmostEqual(low, float(np.percentile(t_img, 1.0)))
        self.assertAlmostEqual(high, float(np.percentile(t_img, 99.5)))

    def test_near_isothermal_frame_holds_six_kelvin(self):
        t_img = np.full(1000, 300.0)
        low, high = ir.preview_window(t_img, {})
        self.assertAlmostEqual(high - low, 6.0)
        self.assertAlmostEqual(0.5 * (low + high), 300.0)

    def test_span_cap_is_centred_on_the_frame_median(self):
        t_img = np.concatenate([np.full(900, 257.0), np.full(100, 299.0)])
        low, high = ir.preview_window(t_img, dict(preview_span_c=20.0))
        self.assertAlmostEqual(high - low, 20.0)
        self.assertAlmostEqual(0.5 * (low + high), float(np.median(t_img)))

    def test_cap_never_widens_a_frame_that_already_fits(self):
        t_img = np.linspace(257.0, 267.0, 1000)       # 10 K span
        base = ir.preview_window(t_img, {})
        capped = ir.preview_window(t_img, dict(preview_span_c=20.0))
        self.assertEqual(base, capped)

    def test_explicit_window_wins_over_the_cap(self):
        t_img = np.linspace(250.0, 320.0, 1000)
        low, high = ir.preview_window(t_img, dict(preview_window_c=[0.0, 30.0],
                                                  preview_span_c=10.0))
        self.assertAlmostEqual(low, 273.15)
        self.assertAlmostEqual(high, 303.15)

    def test_invalid_span_raises(self):
        with self.assertRaises(ValueError):
            ir.preview_window(np.full(100, 300.0), dict(preview_span_c=0.0))
        with self.assertRaises(ValueError):
            ir.preview_window(np.full(100, 300.0), dict(preview_span_c=-5.0))

    def test_cap_changes_the_preview_but_never_the_16_bit_product(self):
        first = self.render(self.scene_config())
        second = self.render(self.scene_config(20.0))
        # the radiometric product is the deliverable: it must not move
        np.testing.assert_array_equal(first["counts_u16"], second["counts_u16"])
        np.testing.assert_array_equal(first["radiance"], second["radiance"])
        np.testing.assert_array_equal(first["apparent_temp_k"],
                                      second["apparent_temp_k"])
        self.assertFalse(np.array_equal(first["preview"], second["preview"]))

    def test_capped_window_turns_detector_noise_into_visible_grey_steps(self):
        # A 42 K sky/target span is exactly the case that hid a 28 mK NETD on
        # a sixth of a grey level.  The window's codes-per-kelvin is the
        # transfer function between detector noise and visible grain, so take
        # it from the window the renderer actually used on this frame.
        frame = self.render(self.scene_config())
        temperature = frame["apparent_temp_k"]
        stretched = ir.preview_window(temperature, {})
        capped = ir.preview_window(temperature, dict(preview_span_c=20.0))
        stretched_slope = 255.0 / (stretched[1] - stretched[0])
        capped_slope = 255.0 / (capped[1] - capped[0])
        self.assertGreater(capped_slope, 1.8 * stretched_slope)
        # 28 mK NETD becomes 0.36 of a grey level -- quantised but not lost --
        # and the 25-count DSNU residual becomes a full grey level
        self.assertGreaterEqual(capped_slope * 0.028, 0.35)
        self.assertGreaterEqual(capped_slope * 25 * 350.0 / 65535.0, 1.0)


if __name__ == "__main__":
    unittest.main()
