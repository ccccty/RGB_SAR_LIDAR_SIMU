"""Material lineage, explicit physical mapping and OBJ import regressions."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from simkit import materials, mesh, scene


class MaterialTests(unittest.TestCase):
    def test_cleanup_drops_duplicate_and_collinear_faces_with_same_attribute_mask(self):
        obj = mesh.Mesh([[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 0, 0]],
                        [[0, 1, 2], [2, 1, 0], [0, 1, 3]],
                        face_source_material=["keep", "duplicate", "collinear"],
                        face_group=["a", "b", "c"],
                        face_uv=np.arange(18).reshape(3, 3, 2))
        clean = mesh.clean_faces(obj)
        np.testing.assert_array_equal(clean.f, [[0, 1, 2]])
        self.assertEqual(clean.face_source_material.tolist(), ["keep"])
        self.assertEqual(clean.face_group.tolist(), ["a"])
        np.testing.assert_array_equal(clean.face_uv, obj.face_uv[:1])
        self.assertEqual(clean.cleanup_report["duplicate_faces_removed"], 1)
        self.assertEqual(clean.cleanup_report["degenerate_or_invalid_faces_removed"], 1)

    def asset(self, directory):
        root = Path(directory)
        (root/"paint.mtl").write_text(
            "newmtl Coating\nKd 0.9 0.2 0.1\nPr 0.3\nmap_Kd -s 1 1 1 \"red texture.png\"\n"
            "newmtl Window\nKd 0.2 0.3 0.4\nNs 200\n", encoding="utf-8")
        (root/"plane.obj").write_text(
            "mtllib paint.mtl\no TestPlane\n"
            "v 0 0 0\nv 2 0 0\nv 2 2 0\nv 0 2 0\nv 0 0 1\n"
            "vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n"
            "usemtl Coating\ng airframe\nf 1/1 2/2 3/3 4/4\n"
            "usemtl Window\ng cockpit\nf -5/-4 -4/-3 -1/-1\n", encoding="utf-8")
        return root/"plane.obj"

    def test_polygon_triangulation_preserves_corner_uv_and_source_group(self):
        with tempfile.TemporaryDirectory() as directory:
            obj = mesh.load_obj(str(self.asset(directory)))
            self.assertEqual(obj.face_source_material.tolist(), ["Coating", "Coating", "Window"])
            self.assertEqual(obj.face_group.tolist(), ["airframe", "airframe", "cockpit"])
            np.testing.assert_allclose(obj.face_uv[1], [[0, 0], [1, 1], [0, 1]])
            np.testing.assert_array_equal(obj.f[2], [0, 1, 4])
            self.assertEqual(obj.source_materials["Coating"]["map_Kd"],
                             str(Path(directory)/"red texture.png"))

    def test_transform_prepare_and_decimation_preserve_lineage(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.asset(directory)
            source = mesh.load_obj(str(path))
            transformed = source.transformed(mesh.rot_z(45), [3, 4, 5])
            np.testing.assert_array_equal(transformed.face_source_material, source.face_source_material)
            np.testing.assert_array_equal(transformed.face_uv, source.face_uv)
            prepared, _, _ = scene.prepare_mesh(dict(shape="obj", model=str(path), scale=2, height=5,
                                                     mat="painted_metal"), directory)
            np.testing.assert_array_equal(prepared.face_uv, source.face_uv)
            np.testing.assert_array_equal(prepared.face_group, source.face_group)
            # The third triangle collapses under this grid, so attributes must
            # be selected with the SAME retained-face mask as geometry.
            reduced = mesh.decimate(source, grid=1.5)
            self.assertEqual(len(reduced.face_uv), len(reduced.f))
            self.assertEqual(reduced.face_source_material.tolist(), ["Coating", "Coating"])
            np.testing.assert_array_equal(reduced.face_uv, source.face_uv[:2])

    def test_explicit_mapping_and_fallback_not_rgb_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.asset(directory)
            spec = dict(name="airplane", shape="obj", model=str(path), mat="painted_metal",
                        import_materials=True, material_map={"Window": "glass"},
                        pos=[100, 200, 300], rot=90,
                        thermal_regions=[dict(material_names=["Window"], temp_c=40)])
            world = scene.build_world(dict(ground=dict(enabled=False), objects=[spec]))
            names = sorted(materials.MATERIALS)
            self.assertEqual([names[i] for i in world.face_mat], ["painted_metal", "painted_metal", "glass"])
            self.assertEqual(world.objs[0]["source_material_report"]["unmapped"], ["Coating"])
            self.assertEqual(world.objs[0]["thermal_regions"], spec["thermal_regions"])
            prepared, _, _ = scene.prepare_mesh(spec, directory)
            np.testing.assert_allclose(world.face_local_centers, prepared.centers)
            self.assertFalse(np.allclose(world.face_local_centers,
                                         world.verts[world.faces].mean(axis=1)))

    def test_invalid_physical_mapping_rejected(self):
        obj = mesh.box([1, 1, 1])
        with self.assertRaises(KeyError):
            materials.physical_face_materials(obj, dict(material_map={"Window": "looks_like_glass"}))
        with self.assertRaises(ValueError):
            materials.physical_face_materials(obj, dict(material_map=["glass"]))

    def test_invalid_obj_zero_index_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"bad.obj"
            path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 0 1 2\n")
            with self.assertRaisesRegex(ValueError, "vertex index"):
                mesh.load_obj(str(path))

    def test_legacy_object_without_opt_in_stays_single_material(self):
        obj = mesh.box([1, 1, 1])
        names, report = materials.physical_face_materials(obj, dict(mat="rubber"))
        self.assertEqual(set(names), {"rubber"})
        self.assertFalse(report["import_appearance"])

    def test_nonconductive_plastic_is_explicit_assumption_not_metal(self):
        obj = mesh.box([1, 1, 1])
        obj.face_source_material[:] = "PlasticM"
        names, report = materials.physical_face_materials(
            obj, dict(mat="painted_metal", material_map={"PlasticM": "plastic"}))
        self.assertEqual(set(names), {"plastic"})
        plastic = materials.get("plastic")
        epsilon = plastic["relative_permittivity_assumed"]
        fresnel_amplitude = (np.sqrt(epsilon)-1)/(np.sqrt(epsilon)+1)
        self.assertAlmostEqual(plastic["rf_rho"], fresnel_amplitude, places=3)
        self.assertLess(plastic["rf_rho"], materials.get("painted_metal")["rf_rho"])
        self.assertIn("not measured", report["material_assumptions"]["plastic"])


if __name__ == "__main__":
    unittest.main()
