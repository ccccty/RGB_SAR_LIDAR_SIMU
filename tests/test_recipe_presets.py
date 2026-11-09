"""Curated workbench presets (2026-11-09).

18 recipes = 3 domains (air/land/sea) x 3 real-photo backdrops x 2 ECM groups
(active/passive). They live in configs/recipes/, the directory the workbench
lists via /api/recipes and loads via /api/recipe/load.

The imaging baseline is the delivered 海陆空1109v1 look: 5 frames, three
modalities, real plates, a wide plate tile with no look-down sky card, a light
civilian vehicle and a measured-scale RCS. These tests guard both the matrix and
the opt-in recipe fields that carry that baseline.
"""
from __future__ import annotations

import json
import os
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECIPES = os.path.join(ROOT, "configs", "recipes")

from simkit import ecm, scene, scenegen, studio  # noqa: E402


def _preset_paths():
    return sorted(p for p in __import__("glob").glob(
        os.path.join(RECIPES, "p_*.json")))


def _load(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _presets():
    return [(os.path.basename(p)[:-5], _load(p)) for p in _preset_paths()]


class PresetMatrixTests(unittest.TestCase):

    def test_exactly_18_presets_cover_the_full_matrix(self):
        pairs = _presets()
        self.assertEqual(len(pairs), 18)
        combos = set()
        cats = {}
        for name, rec in pairs:
            self.assertEqual(rec["name"], name)
            self.assertTrue(rec["title"].strip())
            self.assertEqual(len(rec["tags"]), 4)
            # Air presets use three DIFFERENT photographs from one category,
            # so the plate identity is the file, not the category.
            combos.add((rec["tags"][0],
                        rec["environment"]["background"]["file"],
                        rec["ecm"]["mode"]))
            cats.setdefault(rec["tags"][0], set()).add(rec["tags"][1])
        self.assertEqual(len(combos), 18)
        self.assertEqual({c[0] for c in combos}, {"air", "land", "sea"})
        self.assertEqual({c[2] for c in combos}, {"active", "passive"})
        for domain, expected in (("air", {"sky"}),
                                 ("land", {"city", "rural", "industrial"}),
                                 ("sea", {"open", "port", "island"})):
            self.assertEqual(cats[domain], expected)

    def test_three_distinct_real_plates_per_domain(self):
        seen = {}
        for _name, rec in _presets():
            bg = rec["environment"]["background"]
            self.assertEqual(bg["mode"], "library")
            plate = os.path.join(scenegen.BACKDROP_DIR, bg["category"],
                                 bg["file"])
            self.assertTrue(os.path.isfile(plate), plate)
            seen.setdefault(rec["tags"][0], set()).add(plate)
        for domain, plates in seen.items():
            self.assertEqual(len(plates), 3, domain)

    def test_every_preset_validates_and_builds_a_world(self):
        for name, rec in _presets():
            with self.subTest(preset=name):
                norm = scenegen.validate_recipe(rec)
                self.assertEqual(norm["capture"]["sensors"],
                                 ["optical", "ir", "radar"])
                self.assertEqual(norm["capture"]["frames"], 5)
                self.assertEqual(norm["ecm"]["mode"], rec["ecm"]["mode"])
                cfg = scenegen.build_scene(norm)
                self.assertGreater(len(cfg["objects"]), 0)
                self.assertGreater(len(cfg["ecm"]["entities"]), 0)
                self.assertFalse(rec["effects"]["flare"])
                self.assertFalse(cfg["optical"]["camera"]["flare"]["enabled"])

    def test_three_domains_build_real_geometry(self):
        picked = ("p_land_city_active", "p_air_skyclear_passive",
                  "p_sea_open_active")
        for name in picked:
            with self.subTest(preset=name):
                cfg = scenegen.build_scene(scenegen.validate_recipe(
                    _load(os.path.join(RECIPES, name + ".json"))))
                cfg["_base_dir"] = os.path.join(ROOT, "scenes")
                self.assertGreater(len(scene.build_world(cfg).faces), 0)

    def test_ecm_group_matches_the_declared_mode(self):
        for name, rec in _presets():
            with self.subTest(preset=name):
                want = rec["ecm"]["mode"]
                kinds = [e["kind"] for e in rec["ecm"]["entities"]]
                self.assertTrue(kinds)
                self.assertTrue(all(k in ecm.GROUP_OF for k in kinds))
                self.assertEqual({ecm.GROUP_OF[k] for k in kinds}, {want})
                ids = [e["id"] for e in rec["ecm"]["entities"]]
                self.assertEqual(len(ids), len(set(ids)))


class V1BaselineTests(unittest.TestCase):

    @staticmethod
    def _built(rec):
        return scenegen.build_scene(scenegen.validate_recipe(rec))

    def test_look_down_presets_have_wide_tile_and_no_sky_card(self):
        for name, rec in _presets():
            if rec["tags"][0] not in ("land", "sea"):
                continue
            with self.subTest(preset=name):
                cfg = self._built(rec)
                self.assertNotIn("sky_image", cfg["optical"])
                g = cfg.get("ground") or {}
                self.assertGreaterEqual(float(g.get("backdrop_span_m") or 0),
                                        400.0)
                self.assertTrue(os.path.isfile(g.get("backdrop") or ""))

    def test_air_presets_keep_the_real_sky_plate(self):
        for name, rec in _presets():
            if rec["tags"][0] != "air":
                continue
            with self.subTest(preset=name):
                cfg = self._built(rec)
                self.assertTrue(os.path.isfile(cfg["optical"]["sky_image"]))

    def test_ground_presets_use_the_light_van_and_measured_rcs(self):
        for name, rec in _presets():
            if rec["tags"][0] != "land":
                continue
            with self.subTest(preset=name):
                cfg = self._built(rec)
                obj = cfg["objects"][0]
                self.assertEqual(obj["mat"], "paint_car_white")
                self.assertEqual(float(obj["rcs_db"]), 8.0)
                self.assertEqual(rec["objects"][0]["uv"], [0.5, 0.5])
                # The plate centre is shifted onto a real multi-lane road, and
                # build_scene must carry it through its top_down branch.
                self.assertEqual(rec["environment"]["background"]["offset"],
                                 [0.20, 0.45])
                self.assertEqual(cfg["ground"]["backdrop_offset"],
                                 [0.20, 0.45])
                self.assertGreaterEqual(
                    float(cfg["ground"]["backdrop_span_m"]), 1200.0)

    def test_optin_fields_do_not_change_legacy_recipes(self):
        base = scenegen.validate_recipe(scenegen.default_recipe())
        self.assertEqual(base["environment"]["background"],
                         {"mode": "library", "category": "sky"})
        self.assertNotIn("mat", base["objects"][0])
        self.assertNotIn("rcs_db", base["objects"][0])
        self.assertTrue(base["effects"]["flare"],
                        "flare must stay ON for every existing recipe")
        cfg = self._built(scenegen.default_recipe())
        self.assertIn("sky_image", cfg["optical"])
        for stem in ("trilateral_sky_headon", "ecm_suppression_headon"):
            rec = _load(os.path.join(RECIPES, stem + ".json"))
            norm = scenegen.validate_recipe(rec)
            self.assertEqual(norm["environment"]["background"]["mode"],
                             "library")

    def test_bad_optin_values_raise_a_named_chinese_error(self):
        r = scenegen.default_recipe()
        r["objects"][0]["mat"] = "not_a_material"
        with self.assertRaises(ValueError) as ctx:
            scenegen.validate_recipe(r)
        self.assertIn("objects[0].mat", str(ctx.exception))
        r = scenegen.default_recipe()
        r["environment"]["background"]["span_m"] = 1.0
        with self.assertRaises(ValueError) as ctx:
            scenegen.validate_recipe(r)
        self.assertIn("span_m", str(ctx.exception))


class WorkbenchTests(unittest.TestCase):

    def test_api_lists_every_preset_with_a_readable_title(self):
        listed = studio._list_recipes()
        presets = [x for x in listed if x["name"].startswith("p_")]
        self.assertEqual(len(presets), 18)
        for item in presets:
            self.assertIn("三模态", item["title"])
            self.assertTrue(os.path.isfile(item["path"]))

    def test_frontend_roundtrips_the_optin_keys(self):
        with open(os.path.join(ROOT, "studio", "app.js"), "r",
                  encoding="utf-8") as fh:
            src = fh.read()
        # backgroundPayload() / buildRecipe() write them ...
        for needle in ("_bg.span_m", "_bg.sky_card", "_bg.offset",
                       "_ob.mat", "_ob.rcs_db", "flare: S.effects.flare"):
            self.assertIn(needle, src)
        # ... and applyRecipe() reads them back, resetting on a legacy recipe.
        for needle in ("S.bg.span_m", "S.bg.sky_card", "S.bg.offset",
                       "S.obj.mat", "S.obj.rcs_db", "S.effects.flare ="):
            self.assertIn(needle, src)


if __name__ == "__main__":
    unittest.main()
