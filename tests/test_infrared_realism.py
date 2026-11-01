"""Independent physical limits for atmospheric/thermal/IR realism changes."""
import copy
import unittest

import numpy as np

from simkit import atmosphere, ir, scene, thermal


def atmosphere_config():
    return dict(weather=dict(preset="clear"), air=dict(temp_c=20),
                atmosphere=dict(profile="exponential", aerosol_scale_height_m=1200,
                                extinction_at_reference_per_km=dict(visible=0.2),
                                temperature_lapse_k_per_km=0))


class InfraredRealismTests(unittest.TestCase):
    def test_planck_band_converges_to_radiometric_anchor(self):
        coarse = float(ir.planck_band([300], 8, 12)[0])
        fine = float(ir.planck_band([300], 8, 12, n=4096)[0])
        self.assertAlmostEqual(coarse, 38.5, delta=0.1)
        self.assertAlmostEqual(coarse, fine, delta=0.005)

    def test_digital_gain_saturates_instead_of_wrapping_uint16(self):
        cam = dict(pos=[0,0,1], look_at=[0,5,1], width=8, height=8, fov_deg=15)
        cfg = dict(ground=dict(enabled=False), objects=[dict(name="hot", shape="box", size=[3,3,3],
                   pos=[0,5,0], mat="painted_metal")], sensors=dict(ir_camera=cam),
                   ir=dict(gain=10, netd_mk=0, psf_px=0, texture_sigma_c=0, nuc=dict(enabled=False)))
        world = scene.build_world(cfg)
        result = ir.render(world, cfg, np.full(len(world.faces), 450.0))
        np.testing.assert_array_equal(result["counts_u16"], np.full((8,8),65535,np.uint16))

    def test_surface_crossing_camera_plane_is_clipped_not_discarded(self):
        # Plane y=0, camera points partly downward along +Y; all four native
        # ground triangles straddle the near plane. The visible half still
        # covers every image ray that intersects the plane in front.
        vertices = np.array([[-100,-100,0], [100,-100,0], [100,100,0], [-100,100,0]], float)
        world = scene.World(vertices, np.array([[0,1,2], [0,2,3]]),
                            np.full(2,-1), np.zeros(2,int), [])
        cam = dict(pos=[0,0,10], look_at=[0,20,0], width=32, height=24, fov_deg=30)
        image, depth, _ = ir.rasterize(world, cam, np.array([4.0,4.0]))
        self.assertTrue(np.isfinite(depth).all())
        np.testing.assert_allclose(image, 4.0)

    def test_horizontal_beer_lambert_at_height(self):
        cfg = atmosphere_config()
        expected = np.exp(-0.2*20*np.exp(-5000/1200))
        actual = atmosphere.path_transmittance([0, 0, 5000], [20000, 0, 5000], cfg)
        self.assertAlmostEqual(float(actual), expected, places=13)
        sea_level = atmosphere.path_transmittance([0, 0, 0], [20000, 0, 0], cfg)
        self.assertGreater(actual, sea_level)

    def test_slanted_path_integral_matches_numerical_quadrature_and_reciprocity(self):
        cfg = atmosphere_config()
        p, q = np.array([0, 0, -100]), np.array([20000, 0, 5000])
        s = np.linspace(0, 1, 100001)
        z = np.maximum((p[None]+s[:, None]*(q-p))[..., 2], 0)
        expected = 0.2*np.linalg.norm(q-p)/1000*np.trapezoid(np.exp(-z/1200), s)
        actual = float(atmosphere.optical_depth(p, q, cfg))
        self.assertAlmostEqual(actual, expected, places=8)
        self.assertAlmostEqual(actual, float(atmosphere.optical_depth(q, p, cfg)), places=13)
        self.assertEqual(float(atmosphere.optical_depth(p, p, cfg)), 0.0)

    def test_isothermal_emission_matches_kirchhoff_limit(self):
        cfg = atmosphere_config()
        p, q = np.array([0, 0, 3500]), np.array([20000, 0, 5500])
        band = [8.0, 12.0]
        tau = atmosphere.path_transmittance(p, q, cfg, "lwir")
        expected = (1-tau)*ir.planck_band([293.15], *band)[0]
        emitted = atmosphere.path_emission(p, q, cfg, band, "lwir")
        self.assertAlmostEqual(float(emitted), float(expected), delta=0.001*float(expected))

    def test_surface_texture_moves_with_object_not_frame_seed(self):
        cfg = dict(ground=dict(enabled=False), seed=31,
                   objects=[dict(name="test", shape="box", size=[4, 3, 2],
                                 pos=[0, 0, 10], mat="painted_metal")],
                   ir=dict(texture_sigma_c=2, texture_scale_m=2))
        a = scene.build_world(cfg)
        first = ir.surface_temperature_texture(a, cfg)
        cfg["_frame_seed"] = 999
        cfg["objects"][0].update(pos=[100, -50, 3000], rot=71)
        b = scene.build_world(cfg)
        np.testing.assert_allclose(first, ir.surface_temperature_texture(b, cfg), atol=1e-12)
        self.assertGreater(float(np.std(first)), 0)

    def test_material_and_bounds_heat_regions_are_local(self):
        cfg = dict(ground=dict(enabled=False), sun=dict(on=False), air=dict(temp_c=20),
                   objects=[dict(name="test", shape="box", size=[4, 3, 2], pos=[100, 100, 3000],
                                 rot=67, mat="painted_metal", temp_c=20,
                                 thermal_regions=[dict(bounds_local_m=[[-3, -3, 1.99], [3, 3, 2.01]], temp_c=150)])])
        world = scene.build_world(cfg)
        temperature = thermal.solve(world, cfg, shadow=False)
        z = world.face_local_centers[:, 2]
        np.testing.assert_allclose(temperature[z>1.99], 423.15)
        np.testing.assert_allclose(temperature[z<1.99], 293.15)

    def test_heat_flux_region_warms_only_selected_faces(self):
        cfg = dict(ground=dict(enabled=False), sun=dict(on=False), air=dict(temp_c=20),
                   objects=[dict(name="test", shape="box", size=[4, 3, 2], mat="painted_metal",
                                 thermal_regions=[dict(bounds_local_m=[[-3, -3, 1.99], [3, 3, 2.01]], heat_flux_w_m2=1000)])])
        world = scene.build_world(cfg)
        warm = thermal.solve(world, cfg, shadow=False)
        cfg["objects"][0]["thermal_regions"] = []
        cold = thermal.solve(scene.build_world(cfg), cfg, shadow=False)
        selected = world.face_local_centers[:, 2]>1.99
        self.assertTrue((warm[selected]>cold[selected]+10).all())
        np.testing.assert_allclose(warm[~selected], cold[~selected])

    def test_no_forced_full_pixel_for_unsampled_point_target(self):
        cfg = dict(ground=dict(enabled=False), objects=[dict(name="tiny", shape="box", size=[0.001]*3,
                   pos=[0.1, 10, 0], mat="painted_metal")])
        world = scene.build_world(cfg)
        cam = dict(pos=[0, 0, 0], look_at=[0, 10, 0], width=8, height=8, fov_deg=30)
        _, _, legacy_id = ir.rasterize(world, cam, np.full(len(world.faces), 10.0))
        _, _, native_id = ir.rasterize(world, cam, np.full(len(world.faces), 10.0), subpixel_mode="none")
        self.assertTrue((legacy_id == 0).any())
        self.assertFalse((native_id == 0).any())

    def test_ground_transmission_does_not_depend_on_triangle_diagonal(self):
        vertices = np.array([[-100, -100, 0], [100, -100, 0], [100, 100, 0], [-100, 100, 0]], float)
        cam = dict(pos=[0, -20, 10], look_at=[0, 0, 0], width=40, height=30, fov_deg=30)
        source = np.array([50.0, 50.0])
        cfg = dict(air=dict(temp_c=20))
        a = scene.World(vertices, np.array([[0,1,2], [0,2,3]]), np.full(2,-1), np.zeros(2,int), [])
        b = scene.World(vertices, np.array([[0,1,3], [1,2,3]]), np.full(2,-1), np.zeros(2,int), [])
        images = []
        for world in [a,b]:
            img, depth, _ = ir.rasterize(world, cam, source)
            images.append(ir._pixel_atmosphere(img, depth, cam, cfg, [8,12], "lwir", 0.4, 0.0))
        np.testing.assert_allclose(images[0], images[1], atol=1e-12)

    def test_coverage_sensor_returns_native_shape_and_fractional_occupancy(self):
        cam = dict(pos=[0, 0, 2], look_at=[0, 10, 2], width=24, height=20, fov_deg=30)
        cfg = dict(ground=dict(enabled=False), objects=[dict(name="small", shape="box", size=[0.15]*3,
                   pos=[0.13, 10, 2], mat="painted_metal")], sensors=dict(ir_camera=cam),
                   ir=dict(samples_per_axis=4, subpixel_mode="coverage", psf_px=0, netd_mk=0,
                           texture_sigma_c=0, nuc=dict(enabled=False)))
        world = scene.build_world(cfg)
        result = ir.render(world, cfg, np.full(len(world.faces), 320.0))
        self.assertEqual(result["radiance"].shape, (20,24))
        coverage = result["instance_coverage"][0]
        self.assertTrue(((coverage>0)&(coverage<1)).any())
        self.assertTrue((coverage<=1).all())


if __name__ == "__main__":
    unittest.main()
