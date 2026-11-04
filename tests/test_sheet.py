"""Export-side checks: triptych tiling, contact sheet sampling, README text.

Every fixture here is a handful of 80x60 PNGs made on the fly, so the checks
never touch the multi-hundred-megabyte runs under out/.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from simkit import sheet

CELL_W, CELL_H = 640, 480
LABEL_H = 22


def _save(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return str(path)


def _rgb(path, size=(80, 60), seed=0):
    rng = np.random.default_rng(seed)
    arr = rng.integers(20, 235, (size[1], size[0], 3), dtype=np.uint8)
    return _save(path, Image.fromarray(arr, "RGB"))


def _gray(path, size=(80, 60), value=128):
    return _save(path, Image.new("L", size, value))


def _gray16(path, size=(80, 60), seed=0):
    rng = np.random.default_rng(seed)
    arr = (rng.random((size[1], size[0])) * 30000 + 1000).astype(np.uint16)
    return _save(path, Image.fromarray(arr))


def _black16(path, size=(80, 60)):
    return _save(path, Image.fromarray(np.zeros((size[1], size[0]), np.uint16)))


def _modality_frames(root, frames=2):
    for index in range(frames):
        name = "%06d.png" % index
        _rgb(root / "images" / "optical" / name, seed=index)
        _gray16(root / "images" / "ir_mwir" / name, seed=10 + index)
        _gray(root / "images" / "ir_preview_mwir" / name, value=200)
        _gray(root / "images" / "radar" / name, size=(160, 10), value=90)
        _gray(root / "lidar" / "range" / name, size=(64, 48), value=60)
        _gray(root / "images" / "sar" / name, size=(64, 64), value=30)


class TriptychTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        _modality_frames(self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def test_default_layout(self):
        made = sheet.triptych(self.root)
        self.assertEqual(len(made), 2)
        self.assertEqual(made.frames, 2)
        self.assertEqual([os.path.basename(p) for p in made], ["000000.png", "000001.png"])
        with Image.open(made[0]) as im:
            self.assertEqual(im.size, (3 * CELL_W + 2 * sheet.SEP, CELL_H + LABEL_H))
            self.assertEqual(im.mode, "RGB")

    def test_separators_are_six_px_and_202020(self):
        made = sheet.triptych(self.root)
        arr = np.asarray(Image.open(made[0]).convert("RGB"))
        for index in (1, 2):
            x = index * (CELL_W + sheet.SEP) - sheet.SEP
            band = arr[:, x:x + sheet.SEP]
            self.assertTrue(np.all(band == np.array([0x20, 0x20, 0x20])),
                            "separator %d is not #202020" % index)

    def test_radar_strip_is_letterboxed_and_centred(self):
        made = sheet.triptych(self.root)
        arr = np.asarray(Image.open(made[0]).convert("RGB"))
        x0 = 2 * (CELL_W + sheet.SEP)
        cell = arr[LABEL_H:, x0:x0 + CELL_W]
        strip_h = 40                      # a 160x10 frame fitted into 640x480
        top = (CELL_H - strip_h) // 2
        bg = np.array([0x10, 0x10, 0x10])
        self.assertTrue(np.all(cell[:top] == bg), "top letterbox is not #101010")
        self.assertTrue(np.all(cell[top + strip_h:] == bg), "bottom letterbox is not #101010")
        self.assertTrue(np.any(cell[top:top + strip_h] != bg), "radar strip is blank")

    def test_no_stretch_in_filled_cell(self):
        made = sheet.triptych(self.root)
        arr = np.asarray(Image.open(made[0]).convert("RGB"))
        with Image.open(self.root / "images" / "optical" / "000000.png") as src:
            expected = np.asarray(src.convert("RGB").resize((CELL_W, CELL_H),
                                                            sheet._RESAMPLE))
        np.testing.assert_array_equal(arr[LABEL_H:, :CELL_W], expected)

    def test_ir_preview_is_preferred_over_raw_16bit(self):
        _black16(self.root / "images" / "ir_mwir" / "000000.png")
        made = sheet.triptych(self.root, modalities=("ir_mwir",))
        arr = np.asarray(Image.open(made[0]).convert("RGB"))
        self.assertGreater(float(arr[LABEL_H:].mean()), 150.0)

    def test_16bit_is_percentile_stretched_without_preview(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _gray16(root / "images" / "ir_mwir" / "000000.png", size=(64, 48), seed=3)
            made = sheet.triptych(root, modalities=("ir_mwir",))
            arr = np.asarray(Image.open(made[0]).convert("RGB"))
            self.assertGreater(int(arr[LABEL_H:].max()), 240)
            self.assertGreater(float(arr[LABEL_H:].std()), 10.0)

    def test_radar_point_target_survives_6x_downscale(self):
        # A 4096x64 RD map shrunk into a 640x40 cell: the 2 px return at 255
        # must survive the 6.4x reduction (a mean filter would leave ~40).
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            arr = np.zeros((64, 4096), np.uint8)
            arr[47:50, 2750:2752] = 255
            _save(root / "images" / "radar" / "000000.png", Image.fromarray(arr))
            made = sheet.triptych(root, modalities=("radar",))
            full = np.asarray(Image.open(made[0]).convert("RGB"))
            cell = full[LABEL_H:]
            self.assertGreaterEqual(int(cell.max()), 200)
            self.assertGreaterEqual(int((cell > 128).sum()), 1)

    def test_optical_is_not_max_pooled(self):
        # Continuous-tone frames keep the smooth filter: a uniform optical
        # frame must stay uniform after shrinking, not fan out into blocks.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _rgb(root / "images" / "optical" / "000000.png", size=(1920, 1440), seed=5)
            made = sheet.triptych(root, modalities=("optical",))
            arr = np.asarray(Image.open(made[0]).convert("RGB"))[LABEL_H:]
            with Image.open(root / "images" / "optical" / "000000.png") as src:
                expected = np.asarray(src.convert("RGB").resize((CELL_W, CELL_H),
                                                                sheet._RESAMPLE))
            np.testing.assert_array_equal(arr, expected)

    def test_missing_modality_is_skipped_not_fatal(self):
        made = sheet.triptych(self.root, modalities=("optical", "thermal_xyz", "radar"))
        self.assertEqual(len(made), 2)
        self.assertTrue(any("thermal_xyz" in item for item in made.skipped))
        with Image.open(made[0]) as im:
            self.assertEqual(im.size, (2 * CELL_W + sheet.SEP, CELL_H + LABEL_H))

    def test_ragged_frames_keep_a_blank_cell(self):
        # One modality missing for one frame keeps every preview the same size;
        # the gap is a visible blank cell rather than a narrower tile.
        os.remove(self.root / "images" / "radar" / "000001.png")
        made = sheet.triptych(self.root, modalities=("optical", "radar"))
        self.assertEqual(len(made), 2)
        with Image.open(made[1]) as im:
            self.assertEqual(im.size, (2 * CELL_W + sheet.SEP, CELL_H + LABEL_H))
            arr = np.asarray(im.convert("RGB"))
        bg = np.array([0x10, 0x10, 0x10])
        self.assertTrue(np.all(arr[LABEL_H:, CELL_W + sheet.SEP:] == bg))

    def test_five_modalities_stay_under_width_cap(self):
        made = sheet.triptych(
            self.root,
            modalities=("optical", "ir_mwir", "radar", "sar", "lidar_range"))
        with Image.open(made[0]) as im:
            self.assertLessEqual(im.size[0], 2500)
            self.assertGreater(im.size[0], 2000)

    def test_no_labelled_text_still_renders(self):
        made = sheet.triptych(self.root, label_text=False, modalities=("optical",))
        with Image.open(made[0]) as im:
            self.assertEqual(im.size, (CELL_W, CELL_H + LABEL_H))

    def test_no_modalities_raises(self):
        with tempfile.TemporaryDirectory() as empty:
            with self.assertRaises(FileNotFoundError):
                sheet.triptych(empty)


class ContactSheetTest(unittest.TestCase):
    @staticmethod
    def _previews(root, count, size=(64, 32)):
        for index in range(count):
            _gray(root / "images" / "preview" / ("%06d.png" % index),
                  size=size, value=100 + index % 100)

    def test_caps_cells_at_grid_and_stays_small(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._previews(root, 20)
            made = sheet.contact_sheet(root, cols=4, rows=4)
            self.assertEqual(made.cells, 16)
            self.assertEqual((made.cols, made.rows), (4, 4))
            self.assertEqual(os.path.basename(made), "preview_sheet.png")
            self.assertTrue(os.path.isfile(made))
            with Image.open(made) as im:
                self.assertLessEqual(im.size[0], 2000)
                self.assertLessEqual(im.size[1], 1400)

    def test_two_frames_make_two_columns_no_empty_half(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._previews(root, 2)
            made = sheet.contact_sheet(root)
            self.assertEqual((made.cols, made.rows, made.cells), (2, 1, 2))
            with Image.open(made) as im:
                self.assertEqual(im.size[0], 2 * 480 + 10 + 2 * 16)

    def test_empty_preview_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                sheet.contact_sheet(tmp)


class ReadmeTest(unittest.TestCase):
    def _write(self, root, rel, data):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def _dataset(self, root):
        """A 2-frame, 3-sensor / 5-product dataset with jamming and labels."""
        for index in range(2):
            name = "%06d.png" % index
            _rgb(root / "images" / "optical" / name, seed=index)
            _gray16(root / "images" / "ir_mwir" / name, seed=10 + index)
            _gray(root / "images" / "ir_preview_mwir" / name, value=200)
            _gray16(root / "images" / "ir_lwir" / name, seed=20 + index)
            _gray(root / "images" / "ir_preview_lwir" / name, value=180)
            _gray(root / "images" / "radar" / name, size=(160, 10), value=90)
            _gray(root / "images" / "radar_ra" / name, size=(64, 32), value=50)
            self._write(root, "labels/optical/%s.json" % name[:-4],
                        {"frame": index, "range_m": 1000.0 + 4.0 * index,
                         "boxes": {"fighter_01": [1, 2, 3, 4]}})
            self._write(root, "labels/radar/%s.json" % name[:-4],
                        {"frame": index,
                         "boxes_rd": {"fighter_01": {"x0": 1}},
                         "boxes_3d": [{"name": "fighter_01", "cls": "fighter",
                                       "size": [10.0, 8.0, 3.0]}]})
        scene = dict(
            name="fixture_scene",
            ground=dict(enabled=True, type="sea"),
            weather=dict(mode="fixed", preset="clear"),
            frames=dict(n=2, dt=0.1),
            ir=dict(bands=["mwir", "lwir"], netd_mk=28, psf_px=1.3,
                    nuc=dict(enabled=True, prnu=0.008)),
            optical=dict(samples=16, exposure=-0.6),
            objects=[dict(name="fighter_01", cls="fighter", shape="obj",
                          model="../assets/models/x.obj", rcs_db=8, temp_c=35,
                          vel=[200, 0, 0])],
            sensors=dict(
                optical=dict(fov_deg=2.2, width=80, height=60),
                ir_camera=dict(fov_deg=2.2),
                radar=dict(band="Ku", az_span_deg=2.0, el_span_deg=1.6, n_az=40,
                           n_el=28, bandwidth_mhz=50, n_samples=2048, n_chirps=64,
                           pri_us=50, tx_power_dbm=43, noise_figure_db=8),
                lidar=dict(fov_deg=2.2, max_range_m=3000, range_spec_m=1800,
                           pulse_energy_nj=4000, aperture_mm=50, n_az=256, n_el=192)),
            ecm=dict(enabled=True, entities=[dict(id="jam_1", kind="noise_jammer",
                                                  mode="barrage", erp_w=200,
                                                  pos=[1, 2, 3], vel=[4, 5, 6],
                                                  note="fixture jammer")]))
        self._write(root, "scene.json", scene)
        meta = dict(
            name="fixture_scene", config=str(root / "scene.json"), frames=2,
            seconds=12.5, seed=7, sensors=["optical", "ir", "radar"],
            ir_bands=["mwir"], generated_by="simkit 0.2",
            views=[dict(name="seeker", fov_deg=2.2, width=80, height=60)],
            weather=[dict(name="clear", visibility_m=20000.0, rain_mmh=0.0,
                          wind_ms=5.0, air_temp_c=25.0, cloud_cover=0.05,
                          rh=0.5, sea_state=2.0)],
            categories=[dict(id=1, name="fighter", supercategory="target")],
            ecm=dict(domain="rf", entities=[
                        dict(id="jam_1", kind="noise_jammer", mode="barrage",
                             pos=[1, 2, 3], vel=[4, 5, 6])],
                    frames=2, js_db=dict(min=3.0, max=9.0),
                    noise_rise_db=dict(min=12.0, max=13.0), injected_echoes=8,
                    per_frame=[dict(frame=0,
                                    entities=[dict(id="jam_1", group="active")])]))
        self._write(root, "meta.json", meta)
        self._write(root, "frames.json", [
            dict(frame=0, t=0.0, range_m=1000.0, target_px=120.0,
                 sensor_velocity_mps=[190.0, 0.0, 0.0]),
            dict(frame=1, t=0.1, range_m=1004.0, target_px=119.0,
                 sensor_velocity_mps=[190.0, 0.0, 0.0])])
        self._write(root, "observations/000000.json", dict(
            observer=dict(position_m=[0.0, 0.0, 0.0], velocity_mps=[190.0, 0.0, 0.0]),
            objects=[dict(center_world_m=[1000.0, 0.0, 0.0],
                          velocity_mps=[200.0, 0.0, 0.0], range_m=1000.0)]))
        self._write(root, "stats.json", dict(frames=2, radar_points=18))
        self._write(root, "splits.json", dict(train=["000000"], val=["000001"]))
        self._write(root, "annotations_optical_coco.json", dict(
            info=dict(description="fixture"),
            images=[dict(id=0, file_name="000000.png", width=80, height=60),
                    dict(id=1, file_name="000001.png", width=80, height=60)],
            annotations=[dict(id=0, image_id=0, category_id=1, bbox=[1, 2, 3, 4]),
                         dict(id=1, image_id=1, category_id=1, bbox=[1, 2, 3, 4])],
            categories=[dict(id=1, name="fighter", supercategory="target")]))
        return meta

    def _assert_sections(self, text):
        for index, title in enumerate(sheet._SECTIONS, 1):
            self.assertIn("## %d. %s" % (index, title), text)
        for banned in ("TODO", "待核实", "占位"):
            self.assertNotIn(banned, text)

    def test_readme_has_all_eight_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._dataset(root)
            path = sheet.write_readme(root)
            self.assertEqual(os.path.basename(path), "数据集说明.md")
            text = Path(path).read_text(encoding="utf-8")
            self._assert_sections(text)
            self.assertIn("非实测", text)
            self.assertIn("不足以单独训练", text)
            self.assertIn("ERP 200 W", text)
            self.assertIn("3 个模态、5 类图像产品", text)
            self.assertIn("fighter 2", text)
            self.assertIn("1000.00 m", text)
            self.assertIn("距离窗", text)
            self.assertIn("2048 采样", text)
            self.assertGreater(len(text), 2000)

    def test_explicit_meta_argument(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._dataset(root)
            path = sheet.write_readme(root, {"frames": 2, "sensors": ["optical"]})
            text = Path(path).read_text(encoding="utf-8")
            self._assert_sections(text)
            self.assertIn("帧数 | 2", text)

    def test_readme_survives_a_bare_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _gray(root / "images" / "optical" / "000000.png", value=10)
            path = sheet.write_readme(root)
            text = Path(path).read_text(encoding="utf-8")
            self._assert_sections(text)
            self.assertIn("没有场景级干扰", text)


if __name__ == "__main__":
    unittest.main()
