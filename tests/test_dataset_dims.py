# -*- coding: utf-8 -*-
"""Regression tests for dataset.py size validation, -s sar and PNG metadata.

Scenes here set ``realism.enabled=False`` so the config entry point does not
inject unrelated rendering switches: these tests only care about dimension
handling, the SAR guard and deterministic PNG bytes.
"""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from simkit import dataset, png


def scene_cfg(**overrides):
    cfg = dict(
        realism=dict(enabled=False),
        seed=1, frames=dict(n=1, dt=0.1),
        ground=dict(enabled=False), weather=dict(preset="clear"),
        objects=[dict(name="box", shape="box", size=[2, 2, 2],
                      pos=[0, 15, 0], mat="painted_metal")],
        sensors=dict(ir_camera=dict(pos=[0, 0, 1], look_at=[0, 15, 1],
                                    width=24, height=16, fov_deg=30)),
        output=dict(preview=False, readme=False, jpeg=False))
    cfg.update(overrides)
    return cfg


def write_scene(root, cfg):
    path = Path(root) / "scene.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return str(path)


class CheckDimsTests(unittest.TestCase):
    def test_valid_and_undeclared_dimensions_pass(self):
        for w, h in [(None, None), (1, 1), (480, 360), (4096, 4096)]:
            dataset._check_dims(w, h, "view 'optical'")

    def test_zero_negative_and_oversized_are_rejected(self):
        for bad in (0, -1, -480, 4097, 5000, 100000, "0", "wide"):
            with self.assertRaises(ValueError, msg=bad):
                dataset._check_dims(bad, 16, "view 'optical'")
            with self.assertRaises(ValueError, msg=bad):
                dataset._check_dims(24, bad, "view 'optical'")

    def test_error_names_the_view_and_the_value(self):
        with self.assertRaises(ValueError) as ctx:
            dataset._check_dims(0, 16, "view 'seeker'")
        text = str(ctx.exception)
        self.assertIn("seeker", text)
        self.assertIn("width", text)
        self.assertIn("0", text)

    def test_views_of_rejects_zero_and_oversized(self):
        for w, h in [(0, 16), (24, 0), (0, 0), (5000, 16)]:
            states = [dict(views=dict(optical=dict(width=w, height=h)))]
            with self.assertRaises(ValueError, msg=(w, h)):
                dataset._views_of(states, {})

    def test_views_of_accepts_undeclared(self):
        states = [dict(views=dict(optical=dict(fov_deg=30)))]
        views = dataset._views_of(states, {})
        self.assertEqual(views[0]["width"], None)
        self.assertEqual(views[0]["height"], None)


class ViewPoseTests(unittest.TestCase):
    def test_explicit_zero_is_not_swallowed_into_a_default(self):
        view = dict(name="optical", fov_deg=30, width=24, height=16)
        state = dict(views=dict(optical=dict(pos=[0, 0, 1], width=0, height=16)))
        with self.assertRaises(ValueError):
            dataset._view_pose(state, view, {})

    def test_none_falls_back_to_view_then_default(self):
        view = dict(name="optical", fov_deg=30, width=24, height=16)
        state = dict(views=dict(optical=dict(pos=[0, 0, 1],
                                             width=None, height=None)))
        self.assertEqual(dataset._view_pose(state, view, {})["width"], 24)
        self.assertEqual(dataset._view_pose(state, view, {})["height"], 16)
        view_none = dict(name="optical", fov_deg=30, width=None, height=None)
        got = dataset._view_pose(state, view_none, {})
        self.assertEqual((got["width"], got["height"]), (480, 360))

    def test_explicit_size_is_honoured(self):
        view = dict(name="optical", fov_deg=30, width=24, height=16)
        state = dict(views=dict(optical=dict(pos=[0, 0, 1],
                                             width=640, height=480)))
        got = dataset._view_pose(state, view, {})
        self.assertEqual((got["width"], got["height"]), (640, 480))

    def test_oversized_is_rejected_before_renderer(self):
        view = dict(name="optical", fov_deg=30, width=24, height=16)
        state = dict(views=dict(optical=dict(pos=[0, 0, 1],
                                             width=9999, height=16)))
        with self.assertRaises(ValueError):
            dataset._view_pose(state, view, {})


class RunGuardTests(unittest.TestCase):
    def test_run_rejects_zero_by_zero_from_scene_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = scene_cfg(sensors=dict(ir_camera=dict(
                pos=[0, 0, 1], look_at=[0, 15, 1], width=0, height=0, fov_deg=30)))
            path = write_scene(tmp, cfg)
            with self.assertRaises(ValueError) as ctx:
                dataset.run(path, str(Path(tmp) / "out"), n_frames=1,
                            sensors=("ir",), quiet=True)
            self.assertIn("width", str(ctx.exception))

    def test_sar_requested_without_sensors_sar_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_scene(tmp, scene_cfg())
            with self.assertRaises(ValueError) as ctx:
                dataset.run(path, str(Path(tmp) / "out"), n_frames=1,
                            sensors=("sar",), quiet=True)
            text = str(ctx.exception)
            self.assertIn("sensors.sar", text)
            self.assertIn("n_samples", text)   # minimal config example

    def test_ir_default_fallback_is_untouched(self):
        # The intentional "undeclared sensor -> engine defaults" behaviour for
        # ir/radar/lidar must NOT be turned into an error by the SAR guard.
        with tempfile.TemporaryDirectory() as tmp:
            path = write_scene(tmp, scene_cfg())
            dataset.run(path, str(Path(tmp) / "out"), n_frames=1,
                        sensors=("ir",), quiet=True)
            self.assertTrue((Path(tmp) / "out" / "images" / "ir" / "000000.png").exists())

    def test_legacy_optical_vs_ir_camera_size_mismatch_warns(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = scene_cfg(sensors=dict(
                optical=dict(pos=[0, 0, 1], look_at=[0, 15, 1],
                             width=96, height=72, fov_deg=30),
                ir_camera=dict(pos=[0, 0, 1], look_at=[0, 15, 1],
                               width=24, height=16, fov_deg=30)))
            path = write_scene(tmp, cfg)
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                dataset.run(path, str(Path(tmp) / "out"), n_frames=1,
                            sensors=("ir",), quiet=True)
            text = err.getvalue()
            self.assertIn("ir_camera", text)
            self.assertIn("optical", text)
            self.assertIn("忽略", text)


class PhysicalSizeHintTests(unittest.TestCase):
    def test_helper_detects_every_cfg_size_source(self):
        # The workbench writes dims into clip.views and the camera blocks;
        # neither CLI nor studio passes width/height to run(), so these cfg
        # signals are the ones that must actually fire in production.
        self.assertFalse(dataset._size_override_present({}, None, None))
        self.assertTrue(dataset._size_override_present(
            {"clip": {"views": [{"width": 320, "height": 240}]}}, None, None))
        self.assertTrue(dataset._size_override_present(
            {"clip": {"views": [{"name": "seeker", "width": 320, "height": 240},
                                {"name": "wide", "width": 640, "height": 480}]}},
            None, None))
        for name in ("optical", "eo_camera", "ir_camera", "camera"):
            self.assertTrue(dataset._size_override_present(
                {"sensors": {name: {"width": 320, "height": 240}}}, None, None),
                name)
        self.assertTrue(dataset._size_override_present(
            {"recipe": {"view": {"width": 320, "height": 240}}}, None, None))
        self.assertTrue(dataset._size_override_present({"width": 640}, None, None))
        # Compatible fallback for an explicit kwarg (never used by CLI/studio).
        self.assertTrue(dataset._size_override_present({}, 640, None))
        self.assertTrue(dataset._size_override_present({}, None, 480))

    def test_sar_radar_lidar_blocks_are_not_a_size_request(self):
        # A physics block that happens to carry a width/height is not a W/H
        # override; only camera/view blocks count.
        self.assertFalse(dataset._size_override_present(
            {"sensors": {"sar": {"width": 320, "height": 240},
                         "radar": {"width": 320, "height": 240},
                         "lidar": {"width": 320, "height": 240}}}, None, None))

    def test_clip_views_size_triggers_hint_end_to_end(self):
        # The exact production path: dims baked into clip.views, no kwargs.
        cfg = scene_cfg(
            sensors=dict(radar=dict(n_samples=64, n_chirps=16, n_az=8, n_el=2)),
            clip=dict(views=[dict(name="seeker", tag="", fov_deg=30,
                                  width=320, height=240)]))
        with tempfile.TemporaryDirectory() as tmp:
            path = write_scene(tmp, cfg)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                dataset.run(path, str(Path(tmp) / "out"), n_frames=1,
                            sensors=("radar",), quiet=False)
            text = out.getvalue()
            self.assertIn("物理参数", text)
            self.assertIn("sensors.radar.n_samples", text)
            self.assertEqual(text.count("物理参数"), 1, "hint must print once per run")

    def test_camera_block_size_triggers_hint_end_to_end(self):
        # Legacy path: dims written into a sensors camera block, no kwargs.
        cfg = scene_cfg(sensors=dict(
            ir_camera=dict(pos=[0, 0, 1], look_at=[0, 15, 1],
                           width=320, height=240, fov_deg=30),
            radar=dict(n_samples=64, n_chirps=16, n_az=8, n_el=2)))
        with tempfile.TemporaryDirectory() as tmp:
            path = write_scene(tmp, cfg)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                dataset.run(path, str(Path(tmp) / "out"), n_frames=1,
                            sensors=("radar",), quiet=False)
            self.assertIn("物理参数", out.getvalue())

    def test_non_optical_hint_goes_to_stdout_and_is_quieted(self):
        cfg = scene_cfg(sensors=dict(radar=dict(n_samples=64, n_chirps=16,
                                                n_az=8, n_el=2)))
        with tempfile.TemporaryDirectory() as tmp:
            path = write_scene(tmp, cfg)
            loud = io.StringIO()
            with contextlib.redirect_stdout(loud):
                dataset.run(path, str(Path(tmp) / "loud"), n_frames=1,
                            sensors=("radar",), width=640, height=480, quiet=False)
            text = loud.getvalue()
            self.assertIn("物理参数", text)
            self.assertIn("sensors.radar.n_samples", text)

            quiet_out = io.StringIO()
            with contextlib.redirect_stdout(quiet_out):
                dataset.run(path, str(Path(tmp) / "quiet"), n_frames=1,
                            sensors=("radar",), width=640, height=480, quiet=True)
            self.assertNotIn("物理参数", quiet_out.getvalue())

    def test_no_hint_without_a_size_override(self):
        cfg = scene_cfg(sensors=dict(radar=dict(n_samples=64, n_chirps=16,
                                                n_az=8, n_el=2)))
        with tempfile.TemporaryDirectory() as tmp:
            path = write_scene(tmp, cfg)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                dataset.run(path, str(Path(tmp) / "out"), n_frames=1,
                            sensors=("radar",), quiet=False)
            self.assertNotIn("物理参数", out.getvalue())


class MetadataDeterminismTests(unittest.TestCase):
    def test_two_identical_runs_produce_identical_png_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = write_scene(tmp, scene_cfg())
            a, b = root / "a", root / "b"
            dataset.run(path, str(a), n_frames=1, sensors=("ir",), quiet=True)
            dataset.run(path, str(b), n_frames=1, sensors=("ir",), quiet=True)
            first = {p.relative_to(a): p.read_bytes() for p in sorted(a.rglob("*.png"))}
            second = {p.relative_to(b): p.read_bytes() for p in sorted(b.rglob("*.png"))}
            self.assertEqual(set(first), set(second))
            self.assertTrue(first, "expected at least the IR PNGs")
            self.assertEqual(first, second)

    def test_ir_png_carries_deterministic_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_scene(tmp, scene_cfg())
            out = Path(tmp) / "out"
            dataset.run(path, str(out), n_frames=1, sensors=("ir",), quiet=True)
            meta = png.read_meta(str(out / "images" / "ir" / "000000.png"))
            self.assertEqual(meta["modality"], "ir")
            self.assertEqual(meta["width"], "24")
            self.assertEqual(meta["height"], "16")
            self.assertEqual(meta["bit_depth"], "16")
            self.assertEqual(meta["color_type"], "0")
            self.assertIn("16-bit", meta["radiometric_calibration"])
            self.assertIn("8-bit", meta["radiometric_calibration"])
            self.assertIn("gsd_m_per_px", meta)
            self.assertIn("fov_deg", meta)
            # No absolute path, timestamp or run id leaked into the file.
            for value in meta.values():
                if isinstance(value, str):
                    self.assertNotIn(tmp, value)
                    self.assertNotIn(":\\", value)


if __name__ == "__main__":
    unittest.main()
