# -*- coding: utf-8 -*-
"""Optional JPEG copies of the optical product, written after the render.

The renderer runs inside Blender's bundled Python, which has no Pillow, so the
camera-level JPEG switch cannot actually encode anything there.  dataset.py
runs in the project interpreter, where Pillow exists, and is therefore the
place that owns the lossy copy.  These tests pin the contract:

* output.jpeg absent/False is a no-op: the existing PNG output is byte for
  byte what the renderer wrote;
* switched on, every optical PNG gets a valid JPEG next to it with identical
  dimensions, a bounded RMSE and a smaller file;
* lower quality increases RMSE monotonically;
* COCO keeps naming the PNG, which is still on disk, so every existing
  consumer (YOLO export, contact sheets, consistency checks) keeps working.

The RMSE band is measured on real output, not invented: the 960x720 optical
frames in datasets/1103 give 0.83 DN at q=90, while synthetic high-frequency
fixtures used here stay below 2 DN.  The 6 DN ceiling is deliberately loose so
the test states the promise ("visually lossless") without overfitting one
renderer.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from simkit import dataset, png


#: RMSE ceiling for quality=90 on 8-bit DN (measured 0.83 on real frames).
RMSE_Q90_MAX = 6.0
#: A JPEG that never changes a pixel would not be a lossy codec at all.
RMSE_Q90_MIN = 0.05
#: quality_range values must stay inside this bound as the config promises.
QUALITY_RANGE = (55, 95)


def render_like_frame(seed=0, height=72, width=96):
    """A frame with gradients, edges and grain, like a rendered optical view."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width]
    base = np.stack([
        25.0 + 150.0 * x / max(width - 1, 1),
        30.0 + 130.0 * y / max(height - 1, 1),
        70.0 + 90.0 * np.sin(x / 7.0) * np.cos(y / 9.0),
    ], -1)
    base += rng.normal(0.0, 5.0, base.shape)
    base[height // 3:height // 2, width // 4:width // 2] += 80.0
    return np.clip(np.rint(base), 0, 255).astype(np.uint8)


def views():
    return [dict(name="optical", tag="", width=96, height=72),
            dict(name="wide", tag="_wide", width=96, height=72)]


def build_tree(root, frames=(0, 1), view_list=None):
    """Lay out images/optical* + labels/optical* exactly as the renderer does."""
    for view in (view_list or views()):
        tag = view["tag"]
        img_dir = Path(root) / "images" / ("optical" + tag)
        lab_dir = Path(root) / "labels" / ("optical" + tag)
        img_dir.mkdir(parents=True, exist_ok=True)
        lab_dir.mkdir(parents=True, exist_ok=True)
        for frame in frames:
            png.write_rgb8(str(img_dir / ("%06d.png" % frame)),
                           render_like_frame(seed=frame))
            (lab_dir / ("%06d.json" % frame)).write_text(
                json.dumps(dict(frame=frame, image="%06d.png" % frame,
                                width=96, height=72, boxes={})),
                encoding="utf-8")


def png_paths(root):
    return sorted(Path(root).rglob("*.png"))


def jpeg_paths(root):
    return sorted(Path(root).rglob("*.jpg"))


def read_rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), np.uint8)


def rmse(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    return float(np.sqrt(((a - b) ** 2).mean()))


class SettingsTests(unittest.TestCase):
    def test_default_is_off(self):
        off = [{}, {"output": {}}, {"output": {"jpeg": False}},
               {"output": {"jpeg": {"enabled": False}}}]
        for cfg in off:
            self.assertIsNone(dataset._jpeg_settings(cfg), cfg)

    def test_switch_forms_and_default_quality(self):
        self.assertEqual(dataset._jpeg_settings({"output": {"jpeg": True}}),
                         dict(quality=90, quality_range=None))
        self.assertEqual(dataset._jpeg_settings({"output": {"jpeg": 90}}),
                         dict(quality=90, quality_range=None))
        parsed = dataset._jpeg_settings({"output": {"jpeg": dict(
            enabled=True, quality=82, quality_range=[60, 90])}})
        self.assertEqual(parsed, dict(quality=82, quality_range=(60, 90)))
        self.assertEqual(dataset.JPEG_DEFAULT_QUALITY, 90)

    def test_invalid_numbers_raise_before_a_render(self):
        bad = [
            {"output": {"jpeg": dict(enabled=True, quality=0)}},
            {"output": {"jpeg": dict(enabled=True, quality=101)}},
            {"output": {"jpeg": dict(enabled=True, quality=None)}},
            {"output": {"jpeg": dict(enabled=True, quality="high")}},
            {"output": {"jpeg": dict(enabled=True, quality_range=[90, 40])}},
            {"output": {"jpeg": dict(enabled=True, quality_range=[0, 90])}},
            {"output": {"jpeg": dict(enabled=True, quality_range=[90, 101])}},
            {"output": {"jpeg": dict(enabled=True, quality_range=[90])}},
            {"output": {"jpeg": "high"}},
        ]
        for cfg in bad:
            with self.assertRaises(ValueError, msg=cfg):
                dataset._jpeg_settings(cfg)

    def test_range_draw_is_reproducible(self):
        settings = dataset._jpeg_settings({"output": {"jpeg": dict(
            enabled=True, quality_range=[60, 90])}})
        first = [dataset._jpeg_quality(settings, 11, frame) for frame in range(9)]
        again = [dataset._jpeg_quality(settings, 11, frame) for frame in range(9)]
        self.assertEqual(first, again)
        self.assertTrue(all(60 <= quality <= 90 for quality in first))
        self.assertGreater(len(set(first)), 1)


class PostprocessTests(unittest.TestCase):
    def test_disabled_is_byte_for_byte_noop(self):
        with tempfile.TemporaryDirectory() as tmp:
            build_tree(tmp)
            before = {path: path.read_bytes() for path in png_paths(tmp)}
            report = dataset._write_jpeg_copies(
                tmp, None, views(), [0, 1], 7)
            self.assertIsNone(report)
            after = {path: path.read_bytes() for path in png_paths(tmp)}
            self.assertEqual(before, after)
            self.assertEqual(jpeg_paths(tmp), [])

    def test_valid_jpeg_same_size_and_rmse_band(self):
        with tempfile.TemporaryDirectory() as tmp:
            build_tree(tmp)
            settings = dataset._jpeg_settings({"output": {"jpeg": dict(
                enabled=True, quality=90)}})
            report = dataset._write_jpeg_copies(tmp, settings, views(), [0, 1], 7)
            self.assertEqual(report["files"], 4)
            self.assertEqual(report["missing"], 0)
            self.assertGreater(report["png_bytes"], 0)
            self.assertLess(report["jpeg_bytes"], report["png_bytes"])
            for view in views():
                tag = view["tag"]
                for frame in (0, 1):
                    folder = Path(tmp) / "images" / ("optical" + tag)
                    source = folder / ("%06d.png" % frame)
                    target = folder / ("%06d.jpg" % frame)
                    self.assertTrue(target.exists(), target)
                    self.assertEqual(target.read_bytes()[:2], b"\xff\xd8")
                    with Image.open(target) as jpeg:
                        self.assertEqual(jpeg.format, "JPEG")
                        self.assertEqual(jpeg.size, (96, 72))
                    error = rmse(read_rgb(target), read_rgb(source))
                    self.assertGreaterEqual(error, RMSE_Q90_MIN)
                    self.assertLessEqual(error, RMSE_Q90_MAX)
                    self.assertLess(target.stat().st_size, source.stat().st_size)
                    label = json.loads((Path(tmp) / "labels" / ("optical" + tag)
                                        / ("%06d.json" % frame)).read_text(encoding="utf-8"))
                    # The geometry label still points at the PNG that exists...
                    self.assertEqual(label["image"], "%06d.png" % frame)
                    self.assertTrue((folder / label["image"]).exists())
                    # ...and the lossy copy records its own quality.
                    self.assertEqual(label["jpeg"]["image"], "%06d.jpg" % frame)
                    self.assertEqual(label["jpeg"]["quality"], 90)

    def test_lower_quality_raises_rmse(self):
        with tempfile.TemporaryDirectory() as tmp:
            build_tree(tmp, frames=(0,), view_list=views()[:1])
            source = read_rgb(Path(tmp) / "images" / "optical" / "000000.png")
            errors = {}
            for quality in (95, 90, 60, 30):
                settings = dataset._jpeg_settings({"output": {"jpeg": dict(
                    enabled=True, quality=quality)}})
                dataset._write_jpeg_copies(tmp, settings, views()[:1], [0], 7)
                decoded = read_rgb(Path(tmp) / "images" / "optical" / "000000.jpg")
                errors[quality] = rmse(decoded, source)
            self.assertLess(errors[95], errors[90])
            self.assertLess(errors[90], errors[60])
            self.assertLess(errors[60], errors[30])

    def test_quality_range_is_per_frame_and_deterministic(self):
        frames = tuple(range(8))
        labels = []
        for run in range(2):
            with tempfile.TemporaryDirectory() as tmp:
                build_tree(tmp, frames=frames, view_list=views()[:1])
                settings = dataset._jpeg_settings({"output": {"jpeg": dict(
                    enabled=True, quality_range=list(QUALITY_RANGE))}})
                report = dataset._write_jpeg_copies(
                    tmp, settings, views()[:1], frames, 7)
                self.assertEqual(report["files"], len(frames))
                self.assertEqual(sum(report["qualities"].values()), len(frames))
                drawn = []
                for frame in frames:
                    label = json.loads(
                        (Path(tmp) / "labels" / "optical" / ("%06d.json" % frame))
                        .read_text(encoding="utf-8"))
                    quality = label["jpeg"]["quality"]
                    self.assertGreaterEqual(quality, QUALITY_RANGE[0])
                    self.assertLessEqual(quality, QUALITY_RANGE[1])
                    drawn.append(quality)
                labels.append(drawn)
        self.assertEqual(labels[0], labels[1])
        self.assertGreater(len(set(labels[0])), 1)


BACKEND = r'''
import json, sys
from pathlib import Path
import numpy as np
from simkit import png

def arg(name):
    args = sys.argv[1:]
    return args[args.index(name) + 1]

manifest = json.loads(Path(arg("--manifest")).read_text(encoding="utf-8"))
out = Path(arg("--out"))
W, H = 96, 72
y, x = np.mgrid[0:H, 0:W]
base = np.stack([25 + 150 * x / (W - 1),
                 30 + 130 * y / (H - 1),
                 70 + 90 * np.sin(x / 7.0) * np.cos(y / 9.0)], -1)
img = np.clip(np.rint(base), 0, 255).astype(np.uint8)
(out / "images" / "optical").mkdir(parents=True, exist_ok=True)
(out / "labels" / "optical").mkdir(parents=True, exist_ok=True)
for fr in manifest.get("frames") or []:
    fid = int(fr["frame"])
    png.write_rgb8(str(out / "images" / "optical" / ("%06d.png" % fid)), img)
    (out / "labels" / "optical" / ("%06d.json" % fid)).write_text(
        json.dumps(dict(frame=fid, image="%06d.png" % fid, width=W, height=H,
                        boxes={"box": dict(x0=35.0, y0=23.0, x1=61.0, y1=49.0,
                                           visible=True, pixels=0)})),
        encoding="utf-8")
print("fake backend wrote %d frames" % len(manifest.get("frames") or []))
'''


def optical_config(backend_path, jpeg=None):
    cfg = dict(seed=3, frames=dict(n=2, dt=0.1), ground=dict(enabled=False),
               weather=dict(preset="clear"),
               objects=[dict(name="box", shape="box", size=[2, 2, 2],
                             pos=[0, 15, 0], mat="painted_metal")],
               sensors=dict(
                   optical=dict(pos=[0, 0, 1], look_at=[0, 15, 1],
                                width=96, height=72, fov_deg=30),
                   ir_camera=dict(pos=[0, 0, 1], look_at=[0, 15, 1],
                                  width=96, height=72, fov_deg=30)),
               optical=dict(backend="fake"),
               backends=dict(fake=dict(exe=sys.executable, args=[str(backend_path)])),
               output=dict(preview=False, readme=False))
    if jpeg is not None:
        cfg["output"]["jpeg"] = jpeg
    return cfg


class DatasetPipelineTests(unittest.TestCase):
    def test_run_outputs_default_png_and_optional_jpeg_with_valid_coco(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backend = root / "fake_backend.py"
            backend.write_text(BACKEND, encoding="utf-8")

            cfg_path = root / "scene.json"
            cfg_path.write_text(json.dumps(optical_config(backend)), encoding="utf-8")
            off_dir = root / "off"
            meta_off = dataset.run(str(cfg_path), str(off_dir), n_frames=2,
                                   sensors=("optical",), quiet=True)
            self.assertNotIn("jpeg", meta_off)
            self.assertEqual(jpeg_paths(off_dir), [])
            off_pngs = {path: path.read_bytes() for path in png_paths(off_dir)}
            self.assertEqual(len(off_pngs), 2)

            cfg_path.write_text(json.dumps(optical_config(
                backend, jpeg=dict(enabled=True, quality=90))), encoding="utf-8")
            on_dir = root / "on"
            meta_on = dataset.run(str(cfg_path), str(on_dir), n_frames=2,
                                  sensors=("optical",), quiet=True)
            self.assertTrue(meta_on["jpeg"]["enabled"])
            self.assertEqual(meta_on["jpeg"]["files"], 2)
            self.assertEqual(meta_on["jpeg"]["quality"], 90)

            # Switching JPEG on must not alter the PNG product or the COCO file.
            # Compare by file name: the two runs live in different temp roots.
            self.assertEqual({path.name: data for path, data in off_pngs.items()},
                             {path.name: path.read_bytes() for path in png_paths(on_dir)})
            self.assertEqual((off_dir / "annotations_optical_coco.json").read_bytes(),
                             (on_dir / "annotations_optical_coco.json").read_bytes())

            coco = json.loads((on_dir / "annotations_optical_coco.json")
                              .read_text(encoding="utf-8"))
            self.assertEqual(len(coco["images"]), 2)
            self.assertEqual(len(coco["annotations"]), 2)
            for image in coco["images"]:
                self.assertTrue(image["file_name"].endswith(".png"))
                self.assertTrue((on_dir / "images" / "optical"
                                 / image["file_name"]).exists())
                self.assertTrue((on_dir / "images" / "optical"
                                 / (Path(image["file_name"]).stem + ".jpg")).exists())
            for frame in (0, 1):
                source = on_dir / "images" / "optical" / ("%06d.png" % frame)
                target = on_dir / "images" / "optical" / ("%06d.jpg" % frame)
                with Image.open(target) as jpeg:
                    self.assertEqual(jpeg.size, (96, 72))
                error = rmse(read_rgb(target), read_rgb(source))
                self.assertLessEqual(error, RMSE_Q90_MAX)
                label = json.loads((on_dir / "labels" / "optical"
                                    / ("%06d.json" % frame)).read_text(encoding="utf-8"))
                self.assertEqual(label["image"], "%06d.png" % frame)
                self.assertEqual(label["jpeg"]["quality"], 90)
                self.assertTrue(label["boxes"], "geometry boxes must survive")

    def test_chunked_run_merges_jpeg_accounting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backend = root / "fake_backend.py"
            backend.write_text(BACKEND, encoding="utf-8")
            cfg_path = root / "scene.json"
            cfg_path.write_text(json.dumps(optical_config(
                backend, jpeg=dict(enabled=True, quality=90))), encoding="utf-8")
            parts = root / "parts"
            for part in (0, 1):
                dataset.run(str(cfg_path), str(parts), n_frames=1,
                            sensors=("optical",), frame_offset=part,
                            total_frames=2, part=part, quiet=True)
            self.assertEqual(len(jpeg_paths(parts)), 2)
            meta = dataset.merge_parts(str(parts), str(cfg_path), n_frames=2,
                                       sensors=("optical",), quiet=True)
            self.assertEqual(meta["jpeg"]["files"], 2)
            self.assertEqual(meta["jpeg"]["quality"], 90)
            coco = json.loads((parts / "annotations_optical_coco.json")
                              .read_text(encoding="utf-8"))
            self.assertEqual(len(coco["images"]), 2)
            for image in coco["images"]:
                self.assertTrue((parts / "images" / "optical"
                                 / image["file_name"]).exists())
                self.assertTrue((parts / "images" / "optical"
                                 / (Path(image["file_name"]).stem + ".jpg")).exists())


if __name__ == "__main__":
    unittest.main()
