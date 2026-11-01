import unittest
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

from simkit import optical, optical_sensor, cli


class OpticalRealismTests(unittest.TestCase):
    def configuration(self, **sensor):
        return dict(atmosphere=dict(profile="homogeneous"),
                    acquisition=dict(rgb_exposure_s=0.001),
                    optical=dict(sensor=dict(psf_sigma_px=0, read_noise_e=0,
                                              prnu=0, shot_noise=False, **sensor)))

    def camera(self):
        return dict(pos=[0, 0, 4000], look_at=[0, 100, 4000], fov_deg=10)

    def test_psf_conserves_isolated_signal_energy(self):
        image = np.zeros((33, 33, 3))
        image[16, 16] = [1, 2, 3]
        out = optical_sensor.gaussian(image, 1.3)
        np.testing.assert_allclose(out.sum(axis=(0, 1)), [1, 2, 3], atol=1e-12)
        self.assertLess(out[16, 16, 0], 1)

    def test_exposure_doubles_unsaturated_counts(self):
        image = np.full((8, 8, 3), 0.1)
        depth = np.full((8, 8), 100.)
        a = optical_sensor.forward(image, depth, self.camera(), self.configuration(), 1)
        cfg = self.configuration()
        cfg["acquisition"]["rgb_exposure_s"] *= 2
        b = optical_sensor.forward(image, depth, self.camera(), cfg, 1)
        np.testing.assert_allclose(b["counts_u16"], a["counts_u16"].astype(float)*2, atol=1)

    def test_fixed_pattern_stable_across_frames_without_temporal_noise(self):
        cfg = self.configuration()
        cfg["optical"]["sensor"]["prnu"] = 0.01
        image, depth = np.full((16, 16, 3), 0.25), np.full((16, 16), 100.)
        a = optical_sensor.forward(image, depth, self.camera(), cfg, 1)
        b = optical_sensor.forward(image, depth, self.camera(), cfg, 999)
        np.testing.assert_array_equal(a["counts_u16"], b["counts_u16"])
        self.assertGreater(np.std(a["counts_u16"]), 0)

    def test_shot_noise_scales_with_electron_statistics(self):
        cfg = self.configuration()
        cfg["optical"]["sensor"].update(shot_noise=True, bits=16, full_well_e=10000)
        image, depth = np.full((256, 128, 3), 0.1), np.full((256, 128), 100.)
        out = optical_sensor.forward(image, depth, self.camera(), cfg, 42)
        electrons = out["counts_u16"].astype(float)*10000/65535
        self.assertAlmostEqual(float(electrons.mean()), 1000, delta=1)
        self.assertAlmostEqual(float(electrons.var()), 1000, delta=30)

    def test_dark_image_and_setting_validation(self):
        image, depth = np.zeros((4, 4, 3)), np.full((4, 4), np.inf)
        out = optical_sensor.forward(image, depth, self.camera(), self.configuration(), 0)
        self.assertEqual(int(out["counts_u16"].max()), 0)
        cfg = self.configuration()
        cfg["optical"]["sensor"]["full_well_e"] = -1
        with self.assertRaises(ValueError):
            optical_sensor.forward(image, depth, self.camera(), cfg, 0)

    def test_failed_blender_cannot_reuse_stale_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root/"manifest.json"
            manifest.write_text(json.dumps(dict(frames=[dict(frame=0)])))
            for folder, suffix in (("images/optical", ".png"), ("labels/optical", ".json")):
                path = root/folder/("000000"+suffix)
                path.parent.mkdir(parents=True)
                path.write_bytes(b"old")
                os.utime(path, ns=(0, 0))
            with patch("simkit.optical.find_blender", return_value="fake-blender"), patch(
                    "simkit.optical.subprocess.run", return_value=SimpleNamespace(returncode=0, stdout="", stderr="")):
                with self.assertRaisesRegex(RuntimeError, "produced no output"):
                    optical.render_frames({}, "config.json", str(root), str(manifest), "render.py", quiet=True)

    def test_selftest_failure_returns_nonzero_exit_status(self):
        with patch("simkit.selftest.run", return_value=False):
            self.assertEqual(cli.cmd_selftest(None), 1)


if __name__ == "__main__":
    unittest.main()
