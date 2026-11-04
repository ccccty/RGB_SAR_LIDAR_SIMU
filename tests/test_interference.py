"""Disturbance lifecycle, determinism, raster fidelity and export contracts."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock
from urllib.request import urlopen
from http.server import ThreadingHTTPServer

import numpy as np
from PIL import Image

from simkit import dataset, interference as effects, labels, studio
from tools import export_yolo


def write(path, value):
    labels.write_json(str(path), value)


def fixture(root, ids=(0, 1)):
    write(root/"meta.json", dict(frames=len(ids), sensors=["optical", "ir", "radar", "lidar", "sar"],
                                seed=7, ir_bands=["mwir"], categories=[dict(id=1, name="test")]))
    write(root/"frames.json", [dict(frame=i, t=i*0.1, clip=0) for i in ids])
    for stream in ("optical", "ir_mwir", "ir_preview_mwir", "radar", "radar_ra", "sar", "lidar_range", "lidar_intensity"):
        directory = (root/"lidar"/stream[6:] if stream.startswith("lidar_") else root/"images"/stream)
        directory.mkdir(parents=True, exist_ok=True)
        for i in ids:
            array = np.full((24, 32, 3) if stream == "optical" else (24, 32),
                            20000 if stream == "ir_mwir" else 100, np.uint16 if stream == "ir_mwir" else np.uint8)
            Image.fromarray(array).save(directory/("%06d.png" % i))
    for mod in ("optical", "ir_mwir", "radar", "lidar", "sar"):
        for i in ids:
            box = dict(x0=10, y0=8, x1=20, y1=16, visible=True)
            write(root/"labels"/mod/("%06d.json" % i),
                  dict(frame=i, width=32, height=24, boxes=[box] if mod == "lidar" else dict(test=box),
                       points=[dict(range=100)] if mod == "radar" else []))
        if mod != "radar":
            write(root/("annotations_%s_coco.json" % mod), dict(
                images=[dict(id=i, file_name="%06d.png" % i, width=32, height=24) for i in ids],
                annotations=[dict(id=i, image_id=i, bbox=[10, 8, 10, 8], category_id=1) for i in ids],
                categories=[dict(id=1, name="test")]))


def digest(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file()}


class InterferenceTests(unittest.TestCase):
    def test_off_is_bit_exact_for_both_depths_and_rgb(self):
        for dtype, shape in [(np.uint8, (15, 20, 3)), (np.uint16, (15, 20))]:
            a = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
            out, mask, events = effects.apply(a, effects.preset("off"), "optical", 0, 0)
            np.testing.assert_array_equal(a, out)
            self.assertEqual(out.dtype, dtype)
            self.assertTrue((mask == 1).all())
            self.assertEqual(events, [])

    def test_noise_is_reproducible_and_independent_of_other_modalities_or_order(self):
        a = np.full((32, 64), 120, np.uint8)
        spec = dict(events=[dict(id="n", kind="additive_noise", strength=0.1)], seed=9)
        whole = [effects.apply(a, spec, "radar", f, f/10, "radar")[0] for f in range(8)]
        effects.apply(a, spec, "ir", 5, 0.5)
        np.testing.assert_array_equal(whole[5], effects.apply(a, spec, "radar", 5, 0.5, "radar")[0])
        self.assertFalse(np.array_equal(whole[4], whole[5]))
        self.assertFalse(np.array_equal(whole[5], effects.apply(a, spec, "radar", 5, 0.5, "radar_ra")[0]))

    def test_schedule_fades_smoothly_and_holds_outside_event(self):
        a = np.full((16, 16), 100, np.uint8)
        spec = dict(events=[dict(kind="contrast_loss", strength=1, start_s=1, end_s=3, fade_s=0.5)])
        for t in (-1, 0, 1, 3, 4):
            np.testing.assert_array_equal(a, effects.apply(a, spec, "radar", 0, t)[0])
        self.assertEqual(effects.apply(a, spec, "radar", 0, 1.25)[2][0]["amount"], 0.5)

    def test_occlusion_and_glare_share_normalized_geometry_between_grids(self):
        spec = dict(events=[dict(kind="occlusion", center_uv=[0.5, 0.5], size_uv=[0.25, 0.5], strength=1)])
        masks = []
        for shape in [(40, 80), (80, 160)]:
            _, mask, _ = effects.apply(np.ones(shape, np.uint8), spec, "ir", 0, 0)
            masks.append(mask)
            self.assertAlmostEqual(float((mask == 0).mean()), 0.125)
        np.testing.assert_array_equal(masks[0], masks[1][::2, ::2])

    def test_clipping_preserves_uint16_counts(self):
        a = np.full((4, 4), 65530, np.uint16)
        out, _, _ = effects.apply(a, dict(events=[dict(kind="glare", strength=1, size_uv=[2, 2])]), "ir", 0, 0)
        self.assertEqual(out.dtype, np.uint16)
        self.assertEqual(int(out.min()), 65535)

    def test_bad_parameters_fail_before_rendering(self):
        bad = [dict(events=[dict(kind="unknown")]), dict(events=[dict(kind="additive_noise", strength=-1)]),
               dict(events=[dict(kind="glare", modalities=["radar"])]), dict(enabled="false"),
               dict(events=[dict(kind="occlusion", center_uv=[1])]), dict(seed=float("nan")),
               dict(events=[dict(kind="additive_noise", start_s=2, end_s=1)]),
               dict(events=[dict(id="x", kind="glare"), dict(id="x", kind="glare")])]
        for spec in bad:
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                effects.validate(spec)

    def test_export_preserves_source_updates_visibility_and_coco_and_is_repeatable(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp)/"source", Path(tmp)/"variant"
            fixture(source)
            before = digest(source)
            spec = dict(events=[dict(kind="occlusion", strength=1, size_uv=[2, 2], modalities=["optical"])])
            result = effects.export(source, target, spec)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(before, digest(source))
            label = json.loads((target/"labels/optical/000000.json").read_text())
            self.assertFalse(label["boxes"]["test"]["visible"])
            self.assertTrue(label["boxes"]["test"]["source_visible"])
            self.assertEqual(json.loads((target/"annotations_optical_coco.json").read_text())["annotations"], [])
            radar = json.loads((target/"labels/radar/000000.json").read_text())
            self.assertIn("clean reference", radar["detection_scope"])
            with Image.open(target/"images/ir_mwir/000000.png") as im:
                self.assertEqual(np.array(im).dtype, np.uint16)
            first = digest(target)
            effects.export(source, target, spec)
            self.assertEqual(first, digest(target))

    def test_split_exports_reproduce_global_frame_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/"whole"
            fixture(source, (0, 1, 2, 3))
            effects.export(source, Path(tmp)/"out_whole", effects.preset("active"))
            source2 = Path(tmp)/"part"
            fixture(source2, (2, 3))
            effects.export(source2, Path(tmp)/"out_part", effects.preset("active"))
            for i in (2, 3):
                a, b = (Path(tmp)/d/"images/radar"/("%06d.png" % i) for d in ("out_whole", "out_part"))
                self.assertEqual(a.read_bytes(), b.read_bytes())

    def test_studio_accepts_presets_and_rejects_invalid_event(self):
        body = dict(scene="validation_fighter", interference_mode="mixed")
        q = studio._parse(body)
        self.assertEqual(len(q["interference"]["events"]), 4)
        cfg = studio._load_scene(q["scene_path"])
        trajectory = copy.deepcopy(cfg["clip"]["observer_trajectory"])
        composed = studio._compose(q, cfg)
        self.assertEqual(composed["clip"]["observer_trajectory"], trajectory)
        self.assertNotIn("seeker", composed["clip"])
        self.assertEqual(len(composed["interference"]["events"]), 4)
        body["interference"] = dict(events=[dict(kind="wrong")])
        with self.assertRaises(studio.BadRequest):
            studio._parse(body)

    def test_gallery_http_serves_variant_and_png_and_handles_dict_boxes(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/"source"
            fixture(source)
            effects.export(source, source/"interfered", effects.preset("mixed"))
            with mock.patch.object(studio, "_run_dir", return_value=str(source)):
                server = ThreadingHTTPServer(("127.0.0.1", 0), studio.Handler)
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                try:
                    base = "http://127.0.0.1:%d" % server.server_port
                    with urlopen(base+"/api/dataset/test?variant=interfered") as response:
                        data = json.load(response)
                    self.assertEqual(data["variant"], "interfered")
                    self.assertEqual(data["variants"], ["clean", "interfered"])
                    self.assertTrue(any(c["mod"] == "radar" for c in data["cards"]))
                    with urlopen(base+data["cards"][0]["url"]) as response:
                        self.assertTrue(response.read().startswith(b"\x89PNG\r\n\x1a\n"))
                finally:
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=2)

    def test_fully_occluded_dataset_keeps_classes_for_negative_yolo_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp)/"source", Path(tmp)/"variant"
            fixture(source)
            effects.export(source, target, dict(events=[dict(kind="occlusion", size_uv=[2, 2], strength=1)]))
            _, names = export_yolo.class_map([("variant", str(target), 2)])
            self.assertEqual(names, {"test": 0})

    def test_unowned_or_source_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp)/"source", Path(tmp)/"existing"
            fixture(source)
            target.mkdir()
            (target/"keep.txt").write_text("keep")
            for path in (source, source.parent, target):
                with self.assertRaises(ValueError):
                    effects.export(source, path, effects.preset("active"))
            self.assertEqual((target/"keep.txt").read_text(), "keep")

    def test_generator_and_chunk_merge_emit_variant(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = dict(seed=1, frames=dict(n=2, dt=0.1), ground=dict(enabled=False),
                       weather=dict(preset="clear"),
                       objects=[dict(name="box", shape="box", size=[2, 2, 2], pos=[0, 15, 0], mat="painted_metal")],
                       sensors=dict(ir_camera=dict(pos=[0, 0, 1], look_at=[0, 15, 1], width=24, height=16, fov_deg=30)),
                       interference=effects.preset("active"))
            path = root/"scene.json"
            write(path, cfg)
            dataset.run(str(path), str(root/"whole"), sensors=("ir",), quiet=True)
            self.assertTrue((root/"whole/interfered/interference.json").exists())
            for i in (0, 1):
                dataset.run(str(path), str(root/"parts"), n_frames=1, sensors=("ir",),
                            frame_offset=i, total_frames=2, part=i, quiet=True)
            self.assertFalse((root/"parts/interfered").exists())
            dataset.merge_parts(str(root/"parts"), str(path), n_frames=2, sensors=("ir",), quiet=True)
            self.assertTrue((root/"parts/interfered/interference.json").exists())
            for i in (0, 1):
                rel = "interfered/images/ir/%06d.png" % i
                self.assertEqual((root/"whole"/rel).read_bytes(), (root/"parts"/rel).read_bytes())


if __name__ == "__main__":
    unittest.main()
