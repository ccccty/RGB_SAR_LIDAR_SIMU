"""Failures that used to disappear without a trace.

Three places swallowed errors: the ECM manifest writer (a broken block meant the
Blender optical pass rendered no decoys while radar/IR still evaluated them),
the skin-plate bake, and the contact-sheet modality map (a scene without
ir.bands lost its IR column). Silent loss is worse than a crash: the run
succeeds and the dataset is quietly wrong.
"""
from __future__ import annotations

import os
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from simkit import dataset, external  # noqa: E402


class SkinPlateFailureTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="skin_")

    def test_failure_is_recorded_not_swallowed(self):
        cfg = {"optical": {"skin": True},
               "objects": [{"mat": "paint_definitely_missing"}]}
        out = dataset._skin_plate(cfg, self.tmp)
        self.assertIsNone(out)
        warned = cfg.get("_warnings") or []
        self.assertTrue(any("skin plate" in w for w in warned), warned)

    def test_skin_strict_promotes_it_to_an_error(self):
        cfg = {"optical": {"skin": True, "skin_strict": True},
               "objects": [{"mat": "paint_definitely_missing"}]}
        with self.assertRaises(Exception):
            dataset._skin_plate(cfg, self.tmp)

    def test_disabled_skin_returns_none_without_a_warning(self):
        cfg = {"optical": {"skin": False},
               "objects": [{"mat": "paint_definitely_missing"}]}
        self.assertIsNone(dataset._skin_plate(cfg, self.tmp))
        self.assertFalse(cfg.get("_warnings"))


class EcmManifestFailureTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="manifest_")

    def test_broken_ecm_block_fails_loudly_when_ecm_is_requested(self):
        cfg = {"ecm": {"enabled": True,
                       "entities": [{"kind": "bogus_kind"}]}}
        states = [{"frame": 0, "t": 0.0, "poses": {}}]
        with self.assertRaises(RuntimeError) as ctx:
            external.write_manifest(cfg, states, self.tmp)
        self.assertIn("ecm manifest", str(ctx.exception))
        self.assertTrue(any("ecm manifest" in w
                            for w in cfg.get("_warnings") or []))


class PreviewModalityTests(unittest.TestCase):

    def test_uses_the_ir_directory_that_actually_exists(self):
        d = tempfile.mkdtemp(prefix="prev_")
        os.makedirs(os.path.join(d, "images", "ir_preview_lwir"))
        mods = dataset._preview_modalities(["optical", "ir", "radar"], d)
        self.assertIn("ir_preview_lwir", mods)
        self.assertNotIn("ir_mwir", mods)

    def test_falls_back_to_the_legacy_name(self):
        d = tempfile.mkdtemp(prefix="prev_")
        os.makedirs(os.path.join(d, "images", "ir_mwir"))
        mods = dataset._preview_modalities(["optical", "ir"], d)
        self.assertIn("ir_mwir", mods)


if __name__ == "__main__":
    unittest.main()
