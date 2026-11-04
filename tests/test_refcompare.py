"""Behavioural tests for simkit.refcompare.

These tests do not only check that functions return something: they build the
three images the task cares about (pure Gaussian noise, a smooth gradient, and a
photo-like textured/JPEG'd frame), then assert that the statistics separate them
by a wide margin.  The report tests go through the real compare() and CLI paths,
including corrupt files, mismatched sizes, missing labels/catalogues and the
"reference library is not built yet" case.

The real-photo tests at the bottom run against assets/refs when it exists and
skip otherwise, so the suite stays green before the reference library lands.
"""
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image, ImageFilter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from simkit import refcompare as rc  # noqa: E402

TOOL = os.path.join(ROOT, "tools", "compare_refs.py")
H, W = 288, 384


# ---------------------------------------------------------------------------
# Studio fixtures: images that are deliberately easy to characterise
# ---------------------------------------------------------------------------

def gradient_array(h=H, w=W):
    """A perfectly smooth linear ramp: no texture, no noise, alpha >> 2."""
    yy, xx = np.mgrid[0:h, 0:w]
    ramp = 0.10 + 0.80 * (xx / float(w - 1))
    return np.stack([ramp, ramp, ramp], axis=-1).astype(np.float32)


def gradient_gray16(h=H, w=W):
    return gradient_array(h, w).mean(axis=-1)


def noise_array(h=H, w=W, sigma=0.06, seed=3):
    """I.i.d. Gaussian noise: alpha ~ 0, noise sigma == sigma."""
    rng = np.random.default_rng(seed)
    grey = 0.5 + rng.normal(0.0, sigma, (h, w))
    grey = np.clip(grey, 0.0, 1.0)
    return np.stack([grey, grey, grey], axis=-1).astype(np.float32)


def _fbm(h, w, beta, rng):
    """Fractional Brownian field with power spectrum ~ f^-beta."""
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    radius = np.sqrt(fx * fx + fy * fy)
    radius[0, 0] = 1.0
    spectrum = radius ** (-beta / 2.0)
    phase = rng.normal(size=(h, w)) + 1j * rng.normal(size=(h, w))
    field = np.fft.ifft2(phase * spectrum).real
    return (field - field.mean()) / max(field.std(), 1e-9)


def photo_array(h=H, w=W, seed=7, quality=90):
    """A photo-like frame: 1/f ground, sky gradient, sharp textured target, grain.

    It is deliberately built from the cues the statistics must detect --
    natural 1/f slope, real edges, a little sensor grain and JPEG coding -- so a
    passing comparison here is evidence the features respond to photographic
    structure rather than to the generator's quirks.
    """
    rng = np.random.default_rng(seed)
    ny, nx = np.mgrid[0:h, 0:w]
    yn = (ny + 0.5) / h
    horizon = 0.38
    sky_t = np.clip(yn / horizon, 0.0, 1.0)
    sky = np.stack([
        0.38 + 0.30 * sky_t,
        0.55 + 0.25 * sky_t,
        0.78 + 0.15 * sky_t,
    ], axis=-1)
    coarse = _fbm(h, w, 2.1, rng)
    fine = _fbm(h, w, 1.2, rng)
    ground_t = np.clip((yn - horizon) / (1.0 - horizon), 0.0, 1.0)
    ground = np.stack([
        0.26 + 0.10 * ground_t + 0.05 * coarse,
        0.34 + 0.12 * ground_t + 0.06 * coarse,
        0.20 + 0.05 * ground_t + 0.04 * coarse,
    ], axis=-1)
    ground += 0.025 * fine[..., None]
    img = np.where((yn < horizon)[..., None], sky, ground)

    y0, y1 = int(0.26 * h), int(0.62 * h)
    x0, x1 = int(0.34 * w), int(0.66 * w)
    tx = np.linspace(0.0, 1.0, x1 - x0)[None, :]
    ty = np.linspace(0.0, 1.0, y1 - y0)[:, None]
    plate = 0.52 + 0.12 * tx + 0.05 * ty
    target = np.stack([plate * 0.98, plate * 1.00, plate * 1.03], axis=-1)
    target[(ty[..., 0] > 0.30) & (ty[..., 0] < 0.36), :, :] *= 0.55
    target[(ty[..., 0] > 0.60) & (ty[..., 0] < 0.66), :, :] *= 0.60
    target[int(0.35 * (y1 - y0)):int(0.55 * (y1 - y0)),
           int(0.10 * (x1 - x0)):int(0.28 * (x1 - x0)), :] *= 0.40
    img[y0:y1, x0:x1] = target
    img[y0:y0 + 2, x0:x1] = np.array([0.85, 0.86, 0.88])
    img = img + rng.normal(0.0, 0.006, (h, w, 1))
    arr = (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "JPEG", quality=quality)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"), dtype=np.float32) / 255.0


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------

def save_rgb(path, arr, quality=95):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr8 = (np.clip(np.asarray(arr, dtype=np.float32), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    image = Image.fromarray(arr8)
    if path.lower().endswith((".jpg", ".jpeg")):
        image.save(path, "JPEG", quality=quality)
    else:
        image.save(path)


def make_synth(root, images, labels=None, coco=None, modality="optical"):
    """images: {filename: array}; labels: {filename: doc}; coco: dict."""
    image_dir = os.path.join(root, "images", modality)
    os.makedirs(image_dir, exist_ok=True)
    for name, arr in images.items():
        save_rgb(os.path.join(image_dir, name), arr)
    label_dir = os.path.join(root, "labels", modality)
    for name, doc in (labels or {}).items():
        os.makedirs(label_dir, exist_ok=True)
        with open(os.path.join(label_dir, name), "w", encoding="utf-8") as fh:
            json.dump(doc, fh)
    if coco is not None:
        with open(os.path.join(root, "annotations_%s_coco.json" % modality),
                  "w", encoding="utf-8") as fh:
            json.dump(coco, fh)
    return root


def make_synth_gray16(root, images, modality="optical"):
    """Write a synth frame as a 16-bit grayscale PNG (precision-preserving)."""
    image_dir = os.path.join(root, "images", modality)
    os.makedirs(image_dir, exist_ok=True)
    for name, arr in images.items():
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr.mean(axis=-1)
        data = (np.clip(arr, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)
        Image.fromarray(data).save(os.path.join(image_dir, name))
    return root


def make_ref(root, images):
    os.makedirs(root, exist_ok=True)
    for name, arr in images.items():
        save_rgb(os.path.join(root, name), arr, quality=90)
    return root


def label_doc(name, width, height, boxes):
    return {"frame": 0, "image": name, "width": width, "height": height,
            "boxes": boxes}


def find_feature(report, key):
    for row in report["features"]:
        if row["key"] == key:
            return row
    raise AssertionError("feature %r missing from report" % key)


def run_cli(args, cwd=ROOT):
    env = dict(os.environ)
    env["PYTHONPATH"] = ROOT
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [sys.executable, TOOL] + list(args), cwd=cwd, env=env,
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=300)


class ImageLoadingTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def test_large_image_is_downscaled_to_analysis_canvas(self):
        path = os.path.join(self.tmp, "big.png")
        save_rgb(path, np.zeros((600, 800, 3), dtype=np.float32) + 0.5)
        record = rc.load_image(path)
        self.assertEqual(record["native_size"], (800, 600))
        self.assertEqual(record["analysis_size"], (384, 288))
        self.assertAlmostEqual(record["scale"], 0.48, places=3)
        self.assertTrue(any("resized" in note for note in record["notes"]))

    def test_small_image_is_not_upscaled(self):
        path = os.path.join(self.tmp, "small.png")
        save_rgb(path, np.zeros((80, 120, 3), dtype=np.float32) + 0.4)
        record = rc.load_image(path)
        self.assertEqual(record["analysis_size"], (120, 80))
        self.assertFalse(any("resized" in note for note in record["notes"]))

    def test_grayscale_and_rgba_sources_are_expanded(self):
        gray_path = os.path.join(self.tmp, "gray.png")
        Image.fromarray(np.full((64, 80), 128, dtype=np.uint8), mode="L").save(gray_path)
        record = rc.load_image(gray_path)
        self.assertTrue(record["grayscale"])
        self.assertEqual(record["rgb"].shape, (64, 80, 3))
        self.assertAlmostEqual(float(record["rgb"].mean()), 128.0 / 255.0, places=2)

        rgba_path = os.path.join(self.tmp, "clear.png")
        Image.fromarray(np.zeros((32, 40, 4), dtype=np.uint8), mode="RGBA").save(rgba_path)
        record = rc.load_image(rgba_path)
        # transparent pixels are composited over white, not left black
        self.assertGreater(float(record["rgb"].min()), 0.99)

    def test_16bit_source_is_normalised_and_noted(self):
        path = os.path.join(self.tmp, "depth16.png")
        arr = np.linspace(0, 65535, 200 * 120, dtype=np.float64)
        arr = arr.reshape(120, 200).astype(np.uint16)
        Image.fromarray(arr).save(path)
        record = rc.load_image(path)
        self.assertTrue(record["grayscale"])
        self.assertEqual(record["bit_depth"], 16)
        self.assertGreaterEqual(float(record["rgb"].min()), 0.0)
        self.assertLessEqual(float(record["rgb"].max()), 1.0)
        self.assertTrue(any("high-bit-depth" in note for note in record["notes"]))
        # precision is preserved, not re-quantised to 256 levels
        self.assertGreater(len(np.unique(record["rgb"])), 1000)

    def test_exif_orientation_is_applied_to_native_size(self):
        path = os.path.join(self.tmp, "rotated.jpg")
        arr = np.linspace(0, 1, 80 * 40, dtype=np.float32).reshape(40, 80)
        arr = np.stack([arr] * 3, axis=-1)
        image = Image.fromarray((arr * 255).astype(np.uint8))
        exif = Image.Exif()
        exif[274] = 6  # rotate 90 CW on display
        image.save(path, "JPEG", exif=exif, quality=95)
        record = rc.load_image(path)
        self.assertEqual(record["native_size"], (40, 80))
        self.assertEqual(record["analysis_size"], (40, 80))
        self.assertTrue(any("EXIF" in note for note in record["notes"]))

    def test_corrupt_and_undersized_files_raise_ImageLoadError(self):
        bad = os.path.join(self.tmp, "bad.png")
        with open(bad, "wb") as fh:
            fh.write(b"this is not an image")
        with self.assertRaises(rc.ImageLoadError):
            rc.load_image(bad)
        tiny = os.path.join(self.tmp, "tiny.png")
        save_rgb(tiny, np.zeros((10, 10, 3), dtype=np.float32))
        with self.assertRaises(rc.ImageLoadError):
            rc.load_image(tiny)
        with self.assertRaises(rc.ImageLoadError):
            rc.load_image(os.path.join(self.tmp, "missing.png"))


class FeatureFunctionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gradient = gradient_array()
        cls.noise = noise_array()
        cls.photo = photo_array()

    def test_noise_sigma_recovers_gaussian_sigma(self):
        for sigma in (0.02, 0.06, 0.10):
            image = noise_array(sigma=sigma, seed=int(sigma * 1000))
            estimate = rc.noise_sigma(rc.to_gray(image))
            self.assertAlmostEqual(estimate, sigma, delta=sigma * 0.08,
                                   msg="sigma %.3f estimated as %.4f" % (sigma, estimate))
        low = rc.noise_sigma(rc.to_gray(noise_array(sigma=0.02)))
        high = rc.noise_sigma(rc.to_gray(noise_array(sigma=0.10, seed=9)))
        self.assertLess(low, high)

    def test_noise_sigma_is_near_zero_for_smooth_gradient(self):
        self.assertLess(rc.noise_sigma(rc.to_gray(self.gradient)), 0.002)

    def test_psd_slope_separates_noise_photo_and_gradient(self):
        alpha_noise = rc.radial_psd(rc.to_gray(self.noise))["alpha"]
        alpha_photo = rc.radial_psd(rc.to_gray(self.photo))["alpha"]
        alpha_gradient = rc.radial_psd(rc.to_gray(self.gradient))["alpha"]
        self.assertLess(alpha_noise, 0.4)
        self.assertGreater(alpha_gradient, 4.0)
        self.assertGreater(alpha_photo, 1.0)
        self.assertLess(alpha_photo, 4.0)
        # the separations the task demands must be unmistakable, not marginal
        self.assertGreater(alpha_photo - alpha_noise, 1.5)
        self.assertGreater(alpha_gradient - alpha_photo, 1.5)

    def test_psd_band_fractions_are_a_distribution(self):
        psd = rc.radial_psd(rc.to_gray(self.photo))
        bands = np.array([psd["band_low"], psd["band_mid"],
                          psd["band_high"], psd["band_ultra"]])
        self.assertTrue(np.all(bands >= 0.0))
        self.assertAlmostEqual(float(bands.sum()), 1.0, places=2)
        # white noise has its energy at high frequencies, unlike the photo
        noise_psd = rc.radial_psd(rc.to_gray(self.noise))
        self.assertGreater(noise_psd["band_high"] + noise_psd["band_ultra"], 0.5)
        self.assertGreater(psd["band_low"], 0.5)

    def test_edge_statistics_track_blur_and_texture(self):
        sharp = rc.edge_stats(rc.to_gray(self.photo))
        blurred = rc.edge_stats(rc.to_gray(_box_blur(self.photo, 7)))
        self.assertGreater(sharp["lap_var"], blurred["lap_var"] * 3.0)
        self.assertGreater(sharp["grad_p90"], blurred["grad_p90"])
        smooth = rc.edge_stats(rc.to_gray(self.gradient))
        self.assertLess(smooth["lap_var"], sharp["lap_var"] / 10.0)
        noisy = rc.edge_stats(rc.to_gray(self.noise))
        self.assertGreater(noisy["lap_var"], smooth["lap_var"])

    def test_global_stats_histogram_and_clipping(self):
        white = np.ones((40, 60, 3), dtype=np.float32)
        black = np.zeros((40, 60, 3), dtype=np.float32)
        stats = rc.global_stats(white)
        self.assertEqual(len(stats["hist"]), rc.HIST_BINS)
        self.assertAlmostEqual(float(stats["hist"].sum()), 1.0, places=5)
        self.assertAlmostEqual(stats["sat_frac"], 1.0)
        self.assertAlmostEqual(stats["mean"], 1.0, places=5)
        dark = rc.global_stats(black)
        self.assertAlmostEqual(dark["dark_frac"], 1.0)
        self.assertAlmostEqual(dark["sat_frac"], 0.0)

    def test_color_stats_hue_and_saturation(self):
        red = np.zeros((40, 60, 3), dtype=np.float32)
        red[..., 0] = 1.0
        stats = rc.color_stats(red)
        self.assertAlmostEqual(stats["sat_mean"], 1.0, places=5)
        self.assertAlmostEqual(stats["hue_mean"], 0.0, places=3)
        self.assertEqual(len(stats["hue_hist"]), rc.HUE_BINS)
        self.assertAlmostEqual(float(stats["hue_hist"][0]), 1.0, places=4)
        self.assertGreater(stats["colorfulness"], 0.1)
        gray = np.full((40, 60, 3), 0.5, dtype=np.float32)
        gray_stats = rc.color_stats(gray)
        self.assertAlmostEqual(gray_stats["sat_mean"], 0.0, places=6)
        self.assertAlmostEqual(gray_stats["colorfulness"], 0.0, places=6)

    def test_histogram_distance_is_zero_for_identical_and_large_otherwise(self):
        a = rc.global_stats(self.photo)["hist"]
        b = rc.global_stats(self.gradient)["hist"]
        self.assertAlmostEqual(rc._hist_l1(a, a), 0.0, places=6)
        self.assertGreater(rc._hist_l1(a, b), 0.2)

    def test_individual_metric_wrappers_match_group_functions(self):
        gray = rc.to_gray(self.photo)
        self.assertAlmostEqual(rc.laplacian_variance(gray),
                               rc.edge_stats(gray)["lap_var"], places=9)
        np.testing.assert_allclose(rc.sobel_gradient_distribution(gray),
                                   rc.edge_stats(gray)["grad_hist"])
        self.assertAlmostEqual(rc.radial_psd_slope(gray),
                               rc.radial_psd(gray)["alpha"], places=9)
        np.testing.assert_allclose(
            rc.radial_band_fractions(gray),
            np.array([rc.radial_psd(gray)[key] for key in
                      ("band_low", "band_mid", "band_high", "band_ultra")]),
            rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(rc.luma_histogram(self.photo),
                                   rc.global_stats(self.photo)["hist"])
        hue = rc.hue_saturation_distributions(self.photo)
        self.assertEqual(len(hue["hue_hist"]), rc.HUE_BINS)
        self.assertAlmostEqual(hue["sat_mean"],
                               rc.color_stats(self.photo)["sat_mean"], places=6)


def _box_blur(arr, size):
    """Small separable box blur used only by the blur-tracking test."""
    image = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    image = image.filter(ImageFilter.BoxBlur(size))
    return np.asarray(image, dtype=np.float32) / 255.0


class TargetHeuristicTests(unittest.TestCase):
    def _synthetic_scene(self):
        h, w = 192, 256
        yy, xx = np.mgrid[0:h, 0:w]
        sky = np.stack([
            np.full((h, w), 0.35, np.float32),
            np.full((h, w), 0.55, np.float32),
            np.full((h, w), 0.85, np.float32),
        ], axis=-1)
        obj = np.zeros((h, w), dtype=bool)
        obj[60:150, 80:190] = True
        image = sky.copy()
        image[obj] = np.array([0.85, 0.45, 0.15])
        return image, obj

    def test_heuristic_finds_central_target_and_not_sky(self):
        image, object_mask = self._synthetic_scene()
        mask, info = rc.heuristic_target_mask(image)
        self.assertEqual(info["method"], "heuristic")
        self.assertGreater(float(mask[100:140, 100:170].mean()), 0.8)
        self.assertLess(float(mask[:15].mean()), 0.1)
        self.assertGreater(float(mask.mean()), 0.01)
        self.assertLess(float(mask.mean()), 0.6)
        self.assertEqual(info["fallback"], "")
        self.assertGreater(float(object_mask[mask].mean()), 0.7)

    def test_heuristic_is_deterministic_and_parameter_sensitive(self):
        image, _ = self._synthetic_scene()
        first, info_a = rc.heuristic_target_mask(image)
        second, info_b = rc.heuristic_target_mask(image)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(info_a["score_threshold"], info_b["score_threshold"])
        wide, info_wide = rc.heuristic_target_mask(image, center_frac=0.55)
        self.assertGreaterEqual(float(wide.mean()), 0.0)
        self.assertLessEqual(float(wide.mean()), 0.65)
        self.assertLessEqual(info_wide["params"]["center_frac"], 0.6)

    def test_flat_image_falls_back_to_central_box(self):
        flat = np.full((120, 160, 3), 0.5, dtype=np.float32)
        mask, info = rc.heuristic_target_mask(flat)
        self.assertEqual(info["fallback"], "central_box")
        self.assertAlmostEqual(float(mask.mean()), rc.DEFAULT_CENTER_FRAC ** 2, places=2)
        # the fallback box is centred
        ys, xs = np.where(mask)
        self.assertAlmostEqual(float(xs.mean()) / 160.0, 0.5, places=1)
        self.assertAlmostEqual(float(ys.mean()) / 120.0, 0.5, places=1)

    def test_region_stats_separate_target_and_background_noise(self):
        h, w = 160, 200
        rng = np.random.default_rng(4)
        clean = np.full((h, w, 3), 0.5, dtype=np.float32)
        noise = rng.normal(0.0, 0.08, (h, w, 1)).astype(np.float32)
        image = clean.copy()
        mask = np.zeros((h, w), dtype=bool)
        mask[50:110, 60:140] = True
        image[mask] = np.clip(clean[mask] + noise[mask] + 0.3, 0, 1)
        stats = rc.analyze_image(image, mask, target_source="unit")
        self.assertAlmostEqual(stats["target_area_frac"], mask.mean(), places=6)
        self.assertGreater(stats["target"]["noise_sigma"],
                           stats["background"]["noise_sigma"] * 1.5)
        self.assertNotEqual(stats["target"]["mean"], stats["background"]["mean"])
        self.assertTrue(np.isfinite(stats["full"]["alpha"]))


class AnnotationTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def _base_image(self, h=180, w=240):
        image = np.full((h, w, 3), 0.4, dtype=np.float32)
        image[:, :w // 2] = np.array([0.8, 0.7, 0.55], dtype=np.float32)
        return image

    def test_per_frame_label_json_sets_target_mask(self):
        image = self._base_image()
        synth = make_synth(
            os.path.join(self.tmp, "synth"), {"000000.png": image},
            labels={"000000.json": label_doc(
                "000000.png", 240, 180,
                {"plane": {"x0": 0, "y0": 0, "x1": 120, "y1": 180, "visible": True}})},
        )
        ref = make_ref(os.path.join(self.tmp, "ref"), {"a.jpg": photo_array(180, 240)})
        report = rc.compare(synth, ref, print_table=False, max_images=2)
        self.assertEqual(report["status"], "compared")
        self.assertEqual(report["target_regions"]["synth"]["images_with_annotation"], 1)
        self.assertIn("labels/optical", report["target_regions"]["synth"]["text"])
        area = find_feature(report, "target.area_frac")
        self.assertAlmostEqual(area["synth"], 0.5, places=2)

    def test_coco_annotations_are_a_fallback_source(self):
        image = self._base_image()
        coco = {
            "images": [{"id": 0, "file_name": "000000.png", "width": 240, "height": 180}],
            "annotations": [{"id": 0, "image_id": 0, "category_id": 1,
                             "bbox": [0.0, 0.0, 120.0, 180.0], "iscrowd": 0}],
        }
        synth = make_synth(os.path.join(self.tmp, "synth"), {"000000.png": image},
                           coco=coco)
        ref = make_ref(os.path.join(self.tmp, "ref"), {"a.jpg": photo_array(180, 240)})
        report = rc.compare(synth, ref, print_table=False)
        self.assertEqual(report["target_regions"]["synth"]["method"], "annotation")
        self.assertIn("annotations_optical_coco.json",
                      report["target_regions"]["synth"]["sources"])
        self.assertAlmostEqual(find_feature(report, "target.area_frac")["synth"],
                               0.5, places=2)

    def test_missing_labels_degrade_to_heuristic_with_warning(self):
        synth = make_synth(os.path.join(self.tmp, "synth"),
                           {"000000.png": gradient_array(180, 240)})
        ref = make_ref(os.path.join(self.tmp, "ref"), {"a.jpg": photo_array(180, 240)})
        report = rc.compare(synth, ref, print_table=False)
        self.assertEqual(report["target_regions"]["synth"]["method"], "heuristic_fallback")
        self.assertEqual(report["target_regions"]["synth"]["images_with_annotation"], 0)
        self.assertTrue(any("缺少可用标注" in warning
                            for warning in report["warnings"]))

    def test_manual_box_in_catalog_is_used_for_reference(self):
        refs_root = os.path.join(self.tmp, "refs")
        ref = make_ref(os.path.join(refs_root, "军用飞机"), {"a.jpg": photo_array(150, 200)})
        catalog = {
            "categories": {"军用飞机": {"images": [{
                "file": "军用飞机/a.jpg",
                "manual_box": [10, 20, 90, 100],
                "accepted": True,
                "width": 200,
                "height": 150,
            }]}}
        }
        with open(os.path.join(refs_root, "catalog.json"), "w", encoding="utf-8") as fh:
            json.dump(catalog, fh)
        synth = make_synth(os.path.join(self.tmp, "synth"),
                           {"000000.png": gradient_array(150, 200)})
        report = rc.compare(synth, ref, print_table=False, max_images=1)
        self.assertEqual(report["target_regions"]["ref"]["images_with_manual_box"], 1)
        self.assertEqual(report["target_regions"]["ref"]["images_with_heuristic"], 0)
        ref_area = find_feature(report, "target.area_frac")["ref"]
        self.assertAlmostEqual(ref_area, (80.0 * 80.0) / (200.0 * 150.0), places=2)

    def test_invalid_manual_box_warns_then_uses_heuristic(self):
        refs_root = os.path.join(self.tmp, "refs")
        ref = make_ref(os.path.join(refs_root, "军用飞机"), {"a.jpg": photo_array(150, 200)})
        catalog = {
            "categories": {"军用飞机": {"images": [{
                "file": "军用飞机/a.jpg",
                "manual_box": [0, 0, 0, 0],
                "accepted": True,
            }]}}
        }
        with open(os.path.join(refs_root, "catalog.json"), "w", encoding="utf-8") as fh:
            json.dump(catalog, fh)
        synth = make_synth(os.path.join(self.tmp, "synth"),
                           {"000000.png": gradient_array(150, 200)})
        report = rc.compare(synth, ref, print_table=False, max_images=1)
        self.assertEqual(report["target_regions"]["ref"]["images_with_manual_box"], 0)
        self.assertEqual(report["target_regions"]["ref"]["images_with_heuristic"], 1)
        self.assertTrue(any("manual_box" in warning for warning in report["warnings"]))

    def test_malformed_catalog_falls_back_to_heuristic(self):
        refs_root = os.path.join(self.tmp, "refs")
        ref = make_ref(os.path.join(refs_root, "军用飞机"),
                       {"a.jpg": photo_array(150, 200)})
        with open(os.path.join(refs_root, "catalog.json"), "w", encoding="utf-8") as fh:
            fh.write("{not json")
        synth = make_synth(os.path.join(self.tmp, "synth"),
                           {"000000.png": gradient_array(150, 200)})
        report = rc.compare(synth, ref, print_table=False, max_images=1)
        self.assertEqual(report["status"], "compared")
        self.assertEqual(report["target_regions"]["ref"]["images_with_heuristic"], 1)
        self.assertTrue(any("catalog.json" in warning
                            for warning in report["warnings"]))

    def test_malformed_label_json_falls_back_to_heuristic(self):
        synth = make_synth(os.path.join(self.tmp, "synth"),
                           {"000000.png": gradient_array(150, 200)})
        label_dir = os.path.join(synth, "labels", "optical")
        os.makedirs(label_dir, exist_ok=True)
        with open(os.path.join(label_dir, "000000.json"), "w", encoding="utf-8") as fh:
            fh.write("{not json")
        ref = make_ref(os.path.join(self.tmp, "ref"),
                       {"a.jpg": photo_array(150, 200)})
        report = rc.compare(synth, ref, print_table=False, max_images=1)
        self.assertEqual(report["status"], "compared")
        self.assertEqual(report["target_regions"]["synth"]["method"],
                         "heuristic_fallback")
        self.assertTrue(any("无法读取标注" in warning
                            for warning in report["warnings"]))

    def test_rejected_catalog_image_is_skipped(self):
        refs_root = os.path.join(self.tmp, "refs")
        ref = make_ref(os.path.join(refs_root, "军用飞机"), {"a.jpg": photo_array(150, 200)})
        catalog = {
            "categories": {"军用飞机": {"images": [{
                "file": "军用飞机/a.jpg", "accepted": False,
            }]}}
        }
        with open(os.path.join(refs_root, "catalog.json"), "w", encoding="utf-8") as fh:
            json.dump(catalog, fh)
        synth = make_synth(os.path.join(self.tmp, "synth"),
                           {"000000.png": gradient_array(150, 200)})
        report = rc.compare(synth, ref, print_table=False, max_images=1)
        self.assertEqual(report["status"], "no_reference")
        self.assertEqual(report["images"]["ref"]["count"], 0)
        self.assertIn("rejected", report["skipped"][0]["reason"])


class CompareReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gradient = gradient_array()
        cls.noise = noise_array()
        cls.photo = photo_array()

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def _dirs(self, synth_images, ref_images):
        synth = make_synth(os.path.join(self.tmp, "synth"), synth_images)
        ref = make_ref(os.path.join(self.tmp, "ref"), ref_images)
        return synth, ref

    def test_identical_images_pass_with_zero_deltas(self):
        synth, ref = self._dirs({"000000.png": self.photo}, {"a.png": self.photo})
        out = os.path.join(self.tmp, "report.json")
        report = rc.compare(synth, ref, out_json=out, print_table=False)
        self.assertEqual(report["status"], "compared")
        self.assertTrue(report["summary"]["pass"])
        for row in report["features"]:
            self.assertIs(row["within_tolerance"], True, row["key"])
        self.assertTrue(os.path.isfile(out))
        with open(out, encoding="utf-8") as fh:
            text = fh.read()
        self.assertNotIn("NaN", text)
        payload = json.loads(text)
        self.assertEqual(payload["status"], "compared")
        self.assertIn("features", payload)

    def test_noise_render_vs_photo_flags_noise_and_spectrum(self):
        synth, ref = self._dirs({"000000.png": self.noise}, {"a.jpg": self.photo})
        report = rc.compare(synth, ref, print_table=False)
        self.assertFalse(report["summary"]["pass"])
        noise = find_feature(report, "full.noise_sigma")
        self.assertFalse(noise["within_tolerance"])
        self.assertEqual(noise["verdict"], "偏噪")
        spectrum = find_feature(report, "full.psd_alpha")
        self.assertFalse(spectrum["within_tolerance"])
        self.assertEqual(spectrum["verdict"], "频谱偏平")
        high = find_feature(report, "full.band_high")
        self.assertFalse(high["within_tolerance"])
        self.assertEqual(high["verdict"], "高频偏多")

    def test_gradient_render_vs_photo_flags_clean_soft_and_steep_spectrum(self):
        # 16-bit so the ramp keeps its precision; an 8-bit ramp is dominated by
        # quantisation banding, a limitation covered by the next test.
        synth = make_synth_gray16(os.path.join(self.tmp, "synth"),
                                  {"000000.png": gradient_gray16(180, 240)})
        ref = make_ref(os.path.join(self.tmp, "ref"),
                       {"a.jpg": photo_array(180, 240)})
        report = rc.compare(synth, ref, print_table=False)
        self.assertFalse(report["summary"]["pass"])
        self.assertEqual(report["images"]["synth"]["bit_depths"], [16])
        noise = find_feature(report, "full.noise_sigma")
        self.assertEqual(noise["verdict"], "偏干净")
        sharpness = find_feature(report, "full.lap_var")
        self.assertFalse(sharpness["within_tolerance"])
        self.assertEqual(sharpness["verdict"], "偏软")
        spectrum = find_feature(report, "full.psd_alpha")
        self.assertFalse(spectrum["within_tolerance"])
        self.assertEqual(spectrum["verdict"], "频谱偏陡")

    def test_8bit_gradient_is_still_caught_by_noise_and_softness(self):
        # An 8-bit linear ramp is a staircase; its high-frequency banding makes
        # the PSD slope close to a photo's (documented caveat), but noise and
        # sharpness still flag it as too clean/soft.
        synth = make_synth(os.path.join(self.tmp, "synth"),
                           {"000000.png": gradient_array(180, 240)})
        ref = make_ref(os.path.join(self.tmp, "ref"),
                       {"a.jpg": photo_array(180, 240)})
        report = rc.compare(synth, ref, print_table=False)
        self.assertFalse(report["summary"]["pass"])
        self.assertEqual(find_feature(report, "full.noise_sigma")["verdict"],
                         "偏干净")
        self.assertEqual(find_feature(report, "full.lap_var")["verdict"], "偏软")

    def test_missing_reference_directory_degrades_with_report(self):
        synth = make_synth(os.path.join(self.tmp, "synth"),
                           {"000000.png": self.gradient})
        out = os.path.join(self.tmp, "no_ref.json")
        report = rc.compare(synth, os.path.join(self.tmp, "nope"), out_json=out,
                            print_table=False)
        self.assertEqual(report["status"], "no_reference")
        self.assertFalse(report["summary"]["pass"])
        self.assertIn("synth_only", report)
        self.assertTrue(os.path.isfile(out))
        self.assertIn("没有可用的参考图像", report["summary"]["reasons"])

    def test_missing_synth_directory_degrades(self):
        ref = make_ref(os.path.join(self.tmp, "ref"), {"a.jpg": self.photo})
        report = rc.compare(os.path.join(self.tmp, "nope"), ref, print_table=False)
        self.assertEqual(report["status"], "no_synth")
        self.assertFalse(report["summary"]["pass"])

    def test_corrupt_files_are_skipped_not_fatal(self):
        synth, ref = self._dirs({"000000.png": self.photo}, {"a.jpg": self.photo})
        with open(os.path.join(synth, "images", "optical", "broken.png"), "wb") as fh:
            fh.write(b"broken")
        with open(os.path.join(ref, "broken.jpg"), "wb") as fh:
            fh.write(b"broken")
        report = rc.compare(synth, ref, print_table=False)
        self.assertEqual(report["status"], "compared")
        self.assertEqual(report["images"]["synth"]["count"], 1)
        self.assertEqual(report["images"]["ref"]["count"], 1)
        self.assertGreaterEqual(len(report["skipped"]), 2)
        self.assertTrue(all(item["reason"] for item in report["skipped"]))

    def test_mixed_sizes_and_grayscale_reference_are_handled(self):
        synth = make_synth(os.path.join(self.tmp, "synth"),
                           {"000000.png": gradient_array(240, 320)})
        ref_dir = os.path.join(self.tmp, "ref")
        os.makedirs(ref_dir)
        save_rgb(os.path.join(ref_dir, "big.jpg"),
                 photo_array(480, 640), quality=90)
        gray = (np.asarray(photo_array(150, 200)) * 255).astype(np.uint8)
        Image.fromarray(gray).convert("L").save(os.path.join(ref_dir, "gray.png"))
        report = rc.compare(synth, ref_dir, print_table=False)
        self.assertEqual(report["status"], "compared")
        self.assertEqual(report["images"]["ref"]["count"], 2)
        sizes = [tuple(size) for size in report["images"]["ref"]["native_sizes"]]
        self.assertIn((640, 480), sizes)
        self.assertIn((200, 150), sizes)
        self.assertTrue(report["caveats"])

    def test_max_images_limits_work_and_is_noted(self):
        images = {"000000.png": self.gradient,
                  "000001.png": self.gradient,
                  "000002.png": self.gradient}
        synth = make_synth(os.path.join(self.tmp, "synth"), images)
        ref = make_ref(os.path.join(self.tmp, "ref"), {"a.jpg": self.photo})
        report = rc.compare(synth, ref, print_table=False, max_images=1)
        self.assertEqual(report["images"]["synth"]["count"], 1)
        self.assertTrue(any("只分析了前 1 张" in note
                            for note in report.get("notes", [])))

    def test_target_area_mismatch_produces_warning(self):
        synth = make_synth(
            os.path.join(self.tmp, "synth"), {"000000.png": self.photo},
            labels={"000000.json": label_doc(
                "000000.png", 384, 288,
                {"all": {"x0": 2, "y0": 2, "x1": 382, "y1": 286, "visible": True}})},
        )
        flat_sky = np.zeros((288, 384, 3), dtype=np.float32)
        flat_sky[:] = np.array([0.35, 0.55, 0.85], dtype=np.float32)
        ref = make_ref(os.path.join(self.tmp, "ref"), {"a.png": flat_sky})
        report = rc.compare(synth, ref, print_table=False)
        self.assertTrue(any("目标区面积占比差异较大" in warning
                            for warning in report["warnings"]))
        # a flat sky gives the heuristic no target at all -> documented fallback
        self.assertTrue(any("退化为中心框" in warning
                            for warning in report["warnings"]))
        self.assertEqual(report["target_regions"]["ref"][
            "images_with_heuristic_fallback"], 1)

    def test_report_has_expected_schema(self):
        synth, ref = self._dirs({"000000.png": self.photo}, {"a.jpg": self.photo})
        report = rc.compare(synth, ref, print_table=False)
        for key in ("tool", "version", "status", "features", "summary",
                    "thresholds", "images", "target_regions", "distributions",
                    "caveats"):
            self.assertIn(key, report)
        keys = [row["key"] for row in report["features"]]
        self.assertEqual(len(keys), len(set(keys)))
        for row in report["features"]:
            for field in ("synth", "ref", "delta", "verdict", "threshold",
                          "within_tolerance", "gate"):
                self.assertIn(field, row)
            self.assertIsInstance(row["threshold"]["text"], str)
        text = json.dumps(report, ensure_ascii=False)
        self.assertNotIn("NaN", text)
        self.assertIn("启发式", json.dumps(report["target_regions"], ensure_ascii=False))


class ParamsAndPairingTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def _write_json(self, path, payload):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False)

    def test_measurement_schema_is_read_without_metadata_classes(self):
        path = os.path.join(self.tmp, "params.json")
        self._write_json(path, {
            "generated": "2026-01-01T00:00:00Z",
            "source": "assets/refs",
            "tool": "tools/extract_refs.py",
            "method": {"target_mask": "k-means", "psd_slope": "log-log fit"},
            "uncertainty": "albedo +-35%",
            "reference_materials": {"asphalt": {"albedo": [0.16, 0.16, 0.17]}},
            "甲类": {"sky_mean": 123.4, "psd_slope": 2.4, "target_lap_var": 12.0,
                     "heuristic": {"percentile": 80}},
            "乙类": {"synth": "out/foo", "refs": "assets/refs/乙类"},
        })
        params = rc.load_ref_params(path)
        self.assertTrue(params["exists"])
        self.assertEqual(sorted(params["classes"]), ["乙类", "甲类"])
        self.assertIsNone(params["classes"]["甲类"]["synth"])
        self.assertAlmostEqual(params["classes"]["甲类"]["raw"]["psd_slope"], 2.4)
        self.assertEqual(params["classes"]["乙类"]["synth"], "out/foo")
        self.assertEqual(params["classes"]["甲类"]["heuristic"]["percentile"], 80)

    def test_list_schema_and_path_aliases(self):
        path = os.path.join(self.tmp, "params.json")
        self._write_json(path, [
            {"name": "A", "synth_dir": "out/a", "refs_dir": "refs/a"},
            {"category": "B", "heuristic": {"center_frac": 0.3}},
        ])
        params = rc.load_ref_params(path)
        self.assertEqual(sorted(params["classes"]), ["A", "B"])
        self.assertEqual(params["classes"]["A"]["synth"], "out/a")
        self.assertEqual(params["classes"]["A"]["refs"], "refs/a")
        self.assertEqual(params["classes"]["B"]["heuristic"]["center_frac"], 0.3)

    def test_missing_and_malformed_params_degrade(self):
        missing = rc.load_ref_params(os.path.join(self.tmp, "nope.json"))
        self.assertFalse(missing["exists"])
        self.assertEqual(missing["classes"], {})
        bad_path = os.path.join(self.tmp, "bad.json")
        with open(bad_path, "w", encoding="utf-8") as fh:
            fh.write("{not json")
        bad = rc.load_ref_params(bad_path)
        self.assertTrue(bad["exists"])
        self.assertEqual(bad["classes"], {})
        self.assertTrue(bad["warnings"])

    def test_compare_all_overrides_and_params(self):
        refs_root = os.path.join(self.tmp, "refs")
        for name in ("A", "B"):
            make_ref(os.path.join(refs_root, name), {"a.jpg": photo_array(180, 240)})
        synth_root = os.path.join(self.tmp, "out")
        synth_a = make_synth(os.path.join(synth_root, "run_a"),
                             {"000000.png": gradient_array(180, 240)})
        synth_b = make_synth(os.path.join(synth_root, "run_b"),
                             {"000000.png": gradient_array(180, 240)})
        params_path = os.path.join(self.tmp, "refs_params.json")
        self._write_json(params_path, {
            "classes": {"A": {"synth": synth_a, "heuristic": {"percentile": 60}}},
        })
        multi = rc.compare_all(refs_root=refs_root, synth_root=synth_root,
                               params_path=params_path, print_table=False,
                               overrides={"B": synth_b})
        self.assertEqual(multi["mode"], "all")
        self.assertEqual(len(multi["classes"]), 2)
        by_class = {report["class"]: report for report in multi["classes"]}
        self.assertEqual(by_class["A"]["pairing"], "params")
        self.assertEqual(by_class["B"]["pairing"], "override")
        self.assertEqual(by_class["A"]["status"], "compared")
        self.assertIn("heuristic", json.dumps(by_class["A"]["target_regions"],
                                              ensure_ascii=False))
        # no params entry for B -> no ref_params block, but no crash either
        self.assertNotIn("ref_params", by_class["B"])
        self.assertTrue(multi["caveats"])

    def test_compare_all_single_candidate_pairing(self):
        refs_root = os.path.join(self.tmp, "refs")
        make_ref(os.path.join(refs_root, "only"), {"a.jpg": photo_array(180, 240)})
        synth_root = os.path.join(self.tmp, "out")
        make_synth(os.path.join(synth_root, "some_run"),
                   {"000000.png": gradient_array(180, 240)})
        multi = rc.compare_all(refs_root=refs_root, synth_root=synth_root,
                               params_path=os.path.join(self.tmp, "missing.json"),
                               print_table=False)
        self.assertEqual(len(multi["classes"]), 1)
        self.assertEqual(multi["classes"][0]["class"], "only")
        self.assertEqual(multi["classes"][0]["pairing"], "single")
        self.assertTrue(any("唯一候选" in warning for warning in multi["warnings"]))

    def test_compare_all_missing_reference_root_degrades(self):
        synth_root = os.path.join(self.tmp, "out")
        make_synth(os.path.join(synth_root, "run"),
                   {"000000.png": gradient_array(180, 240)})
        multi = rc.compare_all(
            refs_root=os.path.join(self.tmp, "no_refs"),
            synth_root=synth_root,
            params_path=os.path.join(self.tmp, "missing.json"),
            print_table=False)
        self.assertEqual(multi["classes"], [])
        self.assertFalse(multi["summary"]["pass"])
        self.assertTrue(any("参考图根目录不存在" in warning
                            for warning in multi["warnings"]))

    def test_compare_all_params_raw_is_attached_to_report(self):
        refs_root = os.path.join(self.tmp, "refs")
        make_ref(os.path.join(refs_root, "A"), {"a.jpg": photo_array(180, 240)})
        synth = make_synth(os.path.join(self.tmp, "out", "run"),
                           {"000000.png": gradient_array(180, 240)})
        params_path = os.path.join(self.tmp, "refs_params.json")
        self._write_json(params_path, {
            "classes": {"A": {"synth": synth, "psd_slope": 2.3, "sky_mean": 140.0}},
        })
        multi = rc.compare_all(refs_root=refs_root,
                               synth_root=os.path.join(self.tmp, "out"),
                               params_path=params_path, print_table=False)
        report = multi["classes"][0]
        self.assertIn("ref_params", report)
        self.assertAlmostEqual(report["ref_params"]["psd_slope"], 2.3)
        self.assertIn("ref_params_note", report)


class CliTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def test_cli_single_pass_exit_zero(self):
        image = photo_array(180, 240)
        synth = make_synth(os.path.join(self.tmp, "synth"), {"000000.png": image})
        ref = make_ref(os.path.join(self.tmp, "ref"), {"a.png": image})
        out = os.path.join(self.tmp, "out", "cmp.json")
        result = run_cli(["--synth", synth, "--refs", ref, "--out", out,
                          "--quiet"])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("通过", result.stdout)
        self.assertTrue(os.path.isfile(out))
        with open(out, encoding="utf-8") as fh:
            self.assertEqual(json.load(fh)["status"], "compared")

    def test_cli_single_fail_exit_one(self):
        synth = make_synth(os.path.join(self.tmp, "synth"),
                           {"000000.png": noise_array(180, 240)})
        ref = make_ref(os.path.join(self.tmp, "ref"), {"a.jpg": photo_array(180, 240)})
        out = os.path.join(self.tmp, "cmp.json")
        result = run_cli(["--synth", synth, "--refs", ref, "--out", out,
                          "--quiet"])
        self.assertEqual(result.returncode, 1)
        self.assertIn("未通过", result.stdout)
        with open(out, encoding="utf-8") as fh:
            payload = json.load(fh)
        self.assertFalse(payload["summary"]["pass"])

    def test_cli_missing_reference_exits_one_without_traceback(self):
        synth = make_synth(os.path.join(self.tmp, "synth"),
                           {"000000.png": gradient_array(180, 240)})
        result = run_cli(["--synth", synth, "--refs",
                          os.path.join(self.tmp, "missing"), "--quiet"])
        self.assertEqual(result.returncode, 1)
        self.assertNotIn("Traceback", result.stdout + result.stderr)
        self.assertIn("参考", result.stdout)

    def test_cli_all_with_explicit_pairs(self):
        image = photo_array(180, 240)
        refs_root = os.path.join(self.tmp, "refs")
        for name in ("A", "B"):
            make_ref(os.path.join(refs_root, name), {"a.png": image})
        synth_root = os.path.join(self.tmp, "out")
        synth_a = make_synth(os.path.join(synth_root, "a"), {"000000.png": image})
        synth_b = make_synth(os.path.join(synth_root, "b"), {"000000.png": image})
        out = os.path.join(self.tmp, "all.json")
        result = run_cli([
            "--all", "--refs-root", refs_root, "--synth-root", synth_root,
            "--params", os.path.join(self.tmp, "missing.json"),
            "--pair", "A=%s" % synth_a, "--pair", "B=%s" % synth_b,
            "--out", out, "--quiet"])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("批量对比", result.stdout)
        with open(out, encoding="utf-8") as fh:
            payload = json.load(fh)
        self.assertEqual(len(payload["classes"]), 2)
        self.assertTrue(payload["summary"]["pass"])

    def test_cli_usage_error_exits_two(self):
        result = run_cli(["--synth", self.tmp])
        self.assertEqual(result.returncode, 2)

    def test_cli_list_features(self):
        result = run_cli(["--list-features"])
        self.assertEqual(result.returncode, 0)
        self.assertIn("full.noise_sigma", result.stdout)
        self.assertIn("full.psd_alpha", result.stdout)


class ModuleContractTests(unittest.TestCase):
    def test_feature_specs_and_thresholds_are_documented(self):
        keys = [spec["key"] for spec in rc.FEATURE_SPECS]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertIn("full.noise_sigma", keys)
        self.assertIn("full.psd_alpha", keys)
        for spec in rc.FEATURE_SPECS:
            self.assertGreater(spec["tol"], 0.0)
            self.assertIn(spec["kind"], ("abs", "ratio", "noise"))
            self.assertTrue(spec["label"])
        for name, value in rc.THRESHOLDS.items():
            self.assertTrue(np.isfinite(value), name)
            self.assertGreaterEqual(value, 0.0, name)

    def test_analysis_constants_are_sane(self):
        self.assertGreaterEqual(rc.ANALYSIS_MAX_SIDE, 128)
        self.assertLessEqual(rc.ANALYSIS_MAX_SIDE, 1024)
        self.assertGreater(rc.MIN_REGION_PIXELS, 0)
        self.assertGreater(rc.THRESH_NOISE_RATIO, 1.0)
        self.assertGreater(rc.THRESH_LAP_RATIO, 1.0)


class RealPhotoTests(unittest.TestCase):
    """Separation evidence against the actual reference photographs.

    These skip when assets/refs has not been populated yet, so they do not make
    the suite depend on another tool's in-progress output.
    """

    @classmethod
    def setUpClass(cls):
        cls.paths = sorted(p for p in rc.find_images(
            os.path.join(ROOT, "assets", "refs")) if p.lower().endswith((".jpg", ".jpeg")))
        cls.real_photo = None
        if cls.paths:
            try:
                cls.real_photo = rc.load_image(cls.paths[0])["rgb"]
            except rc.ImageLoadError:  # a partially written library must not break tests
                cls.paths = []
                cls.real_photo = None

    def setUp(self):
        if not self.paths:
            self.skipTest("assets/refs has no reference photos yet")

    def test_real_photo_is_separated_from_noise_and_gradient(self):
        gray = rc.to_gray(self.real_photo)
        photo_sigma = rc.noise_sigma(gray)
        photo_alpha = rc.radial_psd(gray)["alpha"]
        photo_lap = rc.edge_stats(gray)["lap_var"]

        noise = noise_array(self.real_photo.shape[0], self.real_photo.shape[1])
        noise_sigma = rc.noise_sigma(rc.to_gray(noise))
        noise_alpha = rc.radial_psd(rc.to_gray(noise))["alpha"]
        gradient = gradient_array(self.real_photo.shape[0], self.real_photo.shape[1])
        gradient_alpha = rc.radial_psd(rc.to_gray(gradient))["alpha"]
        gradient_lap = rc.edge_stats(rc.to_gray(gradient))["lap_var"]

        # Real photographs are far noisier than a synthetic gradient but far
        # cleaner than an i.i.d. Gaussian noise image at the same canvas size.
        self.assertGreater(noise_sigma, max(photo_sigma, 0.001) * 4.0)
        self.assertLess(photo_sigma, noise_sigma / 3.0)
        # And their 1/f slope sits between white noise (~0) and a smooth ramp.
        self.assertGreater(photo_alpha - noise_alpha, 1.0)
        self.assertGreater(gradient_alpha - photo_alpha, 1.0)
        self.assertGreater(photo_lap, gradient_lap * 10.0 + 1e-6)

    def test_real_photo_comparison_flags_synthetic_gradient(self):
        class_dirs = [d for d in rc.discover_ref_classes(
            os.path.join(ROOT, "assets", "refs"))]
        if not class_dirs:
            self.skipTest("no reference class directory")
        synth = make_synth_gray16(os.path.join(self.tmp_dir(), "synth"),
                                  {"000000.png": gradient_gray16(240, 320)})
        report = None
        for ref_class in class_dirs[:4]:
            candidate = rc.compare(synth, ref_class["dir"], print_table=False,
                                   max_images=2)
            if candidate["status"] == "compared":
                report = candidate
                break
        if report is None:
            # another tool may be rewriting the reference library right now
            self.skipTest("reference photos are currently unreadable")
        self.assertFalse(report["summary"]["pass"])
        self.assertFalse(find_feature(report, "full.lap_var")["within_tolerance"])
        self.assertFalse(find_feature(report, "full.psd_alpha")["within_tolerance"])
        self.assertIn("heuristic", report["target_regions"]["ref"]["method"])

    def tmp_dir(self):
        path = tempfile.mkdtemp(prefix="refcompare_real_")
        self.addCleanup(lambda: shutil.rmtree(path, ignore_errors=True))
        return path


if __name__ == "__main__":
    unittest.main()
