# -*- coding: utf-8 -*-
"""Dispensed IR decoys are a bright head + tapered streak, never a sphere.

The same geometry must reach the numpy World (IR/radar/lidar) and the Blender
frame (optical); a centre offset between the two would silently break the
cross-modal registration this project is built on.
"""
import unittest

import numpy as np

from simkit import ecm, scene


class FlareMeshTests(unittest.TestCase):
    def test_shape_is_a_streak_not_a_ball(self):
        m = ecm.flare_mesh((10.0, 1.0, 0.25))
        lo, hi = m.bbox
        self.assertGreater(hi[0] - lo[0], 8.0, "long trail along X")
        self.assertLess(hi[1] - lo[1], 3.0)
        self.assertLess(hi[2] - lo[2], 3.0)
        self.assertEqual(m.f.shape[1], 3)
        self.assertTrue(np.isfinite(m.v).all())

    def test_varying_trail_length_changes_only_the_tail(self):
        short = ecm.flare_mesh((6.0, 1.0, 0.25))
        long = ecm.flare_mesh((14.0, 1.0, 0.25))
        self.assertLess(short.bbox[0][0], 0)
        self.assertLess(long.bbox[0][0], short.bbox[0][0])

    def test_world_and_blender_meshes_are_vertex_identical(self):
        # scene.prepare_mesh recentres x/y and puts the base at z=0. flare_mesh
        # pre-centres itself so that this becomes a no-op.
        m = ecm.flare_mesh((8.0, 1.0, 0.25))
        spec = dict(name="f", shape="flare", size=[8.0, 1.0, 0.25],
                    mat="engine_hot")
        prepared, lo, hi = scene.prepare_mesh(spec, ".")
        np.testing.assert_allclose(m.v, prepared.v, atol=0)
        np.testing.assert_array_equal(m.f, prepared.f)

    def test_render_objects_uses_the_flare_shape(self):
        cfg = dict(ecm=dict(enabled=True, entities=[dict(
            id="f1", kind="infrared_flare", pos=[1, 2, 2500],
            radius_m=1.1, trail_m=11.0, temp_c=2000,
            band="Ku", life=dict(start_s=0))]))
        objs = ecm.render_objects(cfg)
        self.assertEqual(len(objs), 1)
        self.assertEqual(objs[0]["shape"], "flare")
        self.assertEqual(objs[0]["size"][0], 11.0)
        self.assertEqual(objs[0]["mat"], "flare_pellet")
        self.assertFalse(objs[0]["label"])

    def test_trail_m_is_an_accepted_entity_field(self):
        spec = ecm.load(dict(enabled=True, entities=[dict(
            id="f1", kind="infrared_flare", pos=[0, 0, 0],
            radius_m=0.9, trail_m=9.0, life=dict(start_s=0))]))
        self.assertEqual(spec["entities"][0]["trail_m"], 9.0)
        m = ecm._world_mesh(spec["entities"][0])
        self.assertLess(m.bbox[0][0], -4.0)

    def test_chaff_cloud_is_not_rendered_as_geometry(self):
        # A chaff bloom is a sparse dipole field, not a surface: drawing it as a
        # solid sphere put a growing opaque grey ball next to the target it was
        # protecting, in every photograph.  It stays in the World the radar ray
        # cast reads (ecm_volumetric_ids), it just never becomes render
        # geometry -- optical or thermal.
        cfg = dict(ecm=dict(enabled=True, entities=[dict(
            id="c1", kind="chaff_cloud", pos=[0, 0, 0], radius_m=3.0,
            life=dict(start_s=0))]))
        self.assertEqual(ecm.render_objects(cfg), [])


if __name__ == "__main__":
    unittest.main()
