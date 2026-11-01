"""Quality checks must reject shifted passes without penalising absent signals."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools import check_aviation as qa


def geometry(shift=0, shape=(32, 32), empty=False, depth=100.0, edge=False):
    ids = np.full(shape, -1, np.int32)
    factor = shape[0]//32
    if not empty:
        x = 0 if edge else 12+shift
        ids[12*factor:18*factor, x*factor:(x+8)*factor] = 0
    z = np.where(ids >= 0, depth, np.inf).astype(np.float32)
    return dict(instance_id=ids, depth_z_m=z)


def fixture(root, raw_geometry=True, shift=0, missing_geometry=None, bad_depth=False, true_speed=2.0):
    def write(rel, data):
        path = root/rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")
    write("frames.json", [dict(frame=0, clip=0)])
    write("_render_cfg.json", dict(optical=dict(export_geometry=raw_geometry)))
    obs = dict(sequence_timestamp_s=0.0,
               observer=dict(position_m=[0, 0, 0], velocity_mps=[0, 0, 0]),
               objects=[dict(track_id="aircraft_01", center_world_m=[0, 0, 100],
                             velocity_mps=[0, 0, -true_speed], range_m=100,
                             closing_speed_mps=true_speed, projections=dict(primary=dict(center_px=[16, 16])))],
               cameras=dict(primary=dict(K=[[10, 0, 16], [0, 10, 16], [0, 0, 1]],
                                         world_to_camera=np.eye(4).tolist())))
    write("observations/000000.json", obs)
    for mod in ("optical", "ir_mwir", "ir_lwir", "radar"):
        png = root/"images"/mod/"000000.png"
        png.parent.mkdir(parents=True, exist_ok=True)
        png.write_bytes(b"fixture-file-presence-only")
    point = dict(range=100.0, v_r=2.0, v_r_true=true_speed,
                 range_bin=10, doppler_bin=6, power=10, obj=0)
    noise = dict(range=300.0, v_r=-3.0, range_bin=30, doppler_bin=1, power=12, obj=-1)
    label = dict(points=[point], unassociated_detections=[noise],
                 boxes_3d=[dict(id=0, name="aircraft_01")],
                 processing=dict(detector="rd_threshold", threshold_power=5))
    write("labels/radar/000000.json", label)
    power = np.zeros((8, 64))
    power[6, 10], power[1, 30] = 10, 12
    (root/"raw/radar").mkdir(parents=True)
    np.savez(root/"raw/radar/000000.npz", range_m=np.arange(64)*10.0,
             closing_speed_mps=np.arange(-4, 4, dtype=float), power=power)
    for mod in ("optical", "ir_mwir", "ir_lwir"):
        write("labels/"+mod+"/000000.json", dict(width=32, height=32,
               objects=[dict(id=0, name="aircraft_01")],
               boxes=dict(aircraft_01=dict(visible=True, pixels=48))))
        if raw_geometry and mod != missing_geometry:
            (root/"raw"/mod).mkdir(parents=True)
            data = geometry(shift=shift if mod != "optical" else 0,
                            depth=np.inf if bad_depth and mod == "ir_lwir" else 100)
            np.savez(root/"raw"/mod/"000000.npz", **data)


class QualityTests(unittest.TestCase):
    def test_actual_equal_geometry_is_measured(self):
        result = qa.compare_geometry(geometry(), geometry(), 0)
        self.assertEqual(result["status"], "measured_within_tolerance")
        self.assertEqual(result["centroid_delta_px"], 0)
        self.assertEqual(result["mask_iou"], 1)
        self.assertEqual(result["median_depth_delta_m"], 0)

    def test_deliberately_shifted_image_pass_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root, shift=8)
            self.assertFalse(qa.check(root))
            report = qa.read(root/"quality.json")
            self.assertEqual(report["image_content_registration"], "failed")
            self.assertTrue(any("centroid" in e for e in report["errors"]))

    def test_missing_expected_geometry_file_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root, missing_geometry="ir_lwir")
            self.assertFalse(qa.check(root))
            self.assertTrue(any("missing raw/ir_lwir" in e for e in qa.read(root/"quality.json")["errors"]))

    def test_subpixel_or_absent_masks_never_report_zero_registration_error(self):
        result = qa.compare_geometry(geometry(), geometry(empty=True), 0)
        self.assertEqual(result["status"], "insufficient_support")
        self.assertIsNone(result["centroid_delta_px"])
        self.assertIsNone(result["mask_iou"])
        a = geometry(empty=True)
        a["instance_id"][15, 15] = 0
        a["depth_z_m"][15, 15] = 100
        self.assertEqual(qa.compare_geometry(a, a, 0)["status"], "insufficient_support")

    def test_different_resolution_metrics_are_report_only(self):
        result = qa.compare_geometry(geometry(), geometry(shape=(64, 64)), 0)
        self.assertEqual(result["status"], "different_sampling_report_only")
        self.assertAlmostEqual(result["centroid_delta_px"], 0)
        self.assertAlmostEqual(result["mask_iou"], 1)
        self.assertFalse(result["errors"])

    def test_edge_clipping_does_not_force_a_registration_pass(self):
        result = qa.compare_geometry(geometry(edge=True), geometry(), 0)
        self.assertEqual(result["status"], "edge_clipped_report_only")
        self.assertGreater(result["centroid_delta_px"], 3)

    def test_missing_depth_under_target_instance_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root, bad_depth=True)
            self.assertFalse(qa.check(root))
            self.assertTrue(any("nonpositive camera-Z" in e for e in qa.read(root/"quality.json")["errors"]))

    def test_depth_disagreement_fails_even_when_masks_align(self):
        result = qa.compare_geometry(geometry(), geometry(depth=150), 0)
        self.assertEqual(result["status"], "failed")
        self.assertTrue(any("depth" in e for e in result["errors"]))

    def test_legacy_geometry_only_remains_unverified(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root, raw_geometry=False)
            self.assertTrue(qa.check(root))
            report = qa.read(root/"quality.json")
            self.assertEqual(report["image_content_registration"], "not independently validated")
            self.assertEqual(report["radar_unassociated_detections"], 1)

    def test_single_prf_aliased_true_velocity_is_accepted(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root, true_speed=10)
            self.assertTrue(qa.check(root))
            with (root/"pairs.csv").open(newline="", encoding="utf-8") as fh:
                row = next(csv.DictReader(fh))
            self.assertEqual(row["rgb_raw"], "raw/optical/000000.npz")
            self.assertEqual(row["mwir_raw"], "raw/ir_mwir/000000.npz")
            self.assertEqual(row["radar_unassociated_detections"], "1")

    def test_wrong_rd_bin_or_alias_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root)
            path = root/"labels/radar/000000.json"
            label = qa.read(path)
            label["points"][0].update(v_r=-2, doppler_bin=2)
            path.write_text(json.dumps(label), encoding="utf-8")
            self.assertFalse(qa.check(root))
            self.assertTrue(any("truth alias" in e for e in qa.read(root/"quality.json")["errors"]))


if __name__ == "__main__":
    unittest.main()
