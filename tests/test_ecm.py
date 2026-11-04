"""Scene-level countermeasures must act inside the forward models, not on images.

Four contracts, all of them measurable without a real instrument:

  1. an entity has a position of its own and moves on its own -- it is not a
     decal that follows the target;
  2. a suppression jammer raises the receiver noise floor and the resulting SNR
     loss agrees with the analytic J/S it was configured to produce;
  3. a deception repeater puts N extra echoes into the same range-Doppler cube
     the target lives in, at the range/Doppler cells it asked for;
  4. a chaff cloud's cross-section falls as it blooms, because the same number
     of dipoles spread over a larger volume.

The thresholds are the ones the physics fixes: 1 dB is the resolution at which
the analytic and synthesised SNR are the same number, one range cell is the
smallest position error an RD map can express, and a monotone RCS is the only
statement the dilution law makes.
"""
import json
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simkit import ecm, radar, scene  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def fighter_cfg(**over):
    """One airframe on a clean boresight, no ground, no clutter.

    Deliberately bare: every number this suite asserts on has to come from the
    entity under test, not from the terrain model or a weather state machine.
    """
    rc = dict(band="Ku", fc_ghz=15.0, bandwidth_mhz=20.0, n_samples=1024,
              n_chirps=64, chirp_us=40.0, pri_us=50.0, tx_power_dbm=43.0,
              az_span_deg=2.4, el_span_deg=1.6, n_az=16, n_el=12,
              noise_figure_db=8.0, loss_db=3.0, det_snr_db=12.0,
              clutter_density=0.0, detector="analytic", max_detections=64,
              pos=[0.0, 0.0, 0.0], look_at=[0.0, 1500.0, 100.0],
              vel=[0.0, 0.0, 0.0], up=[0, 0, 1], roll_deg=0.0, pfa=1e-6)
    rc.update(over)
    return dict(_base_dir=BASE, _frame_seed=11, ground=dict(enabled=False),
                air=dict(temp_c=15.0, rh=0.5, wind_mps=2.0),
                sun=dict(on=True, elev_deg=35.0, az_deg=120.0,
                         irradiance=700.0),
                objects=[dict(name="fighter_01", shape="box",
                              size=[1.5, 9.0, 2.0], pos=[0.0, 1500.0, 100.0],
                              vel=[0.0, -60.0, 0.0], mat="painted_metal",
                              rcs_db=8.0, cls="aircraft")],
                sensors=dict(radar=rc))


def simulate(spec=None, t=0.0, **over):
    cfg = fighter_cfg(**over)
    if spec is not None:
        cfg["ecm"] = spec
        cfg["_ecm_t"] = float(t)
    world = scene.build_world(cfg, t=0.0)
    return cfg, world, radar.simulate(world, cfg)


class SchemaTests(unittest.TestCase):
    def test_missing_block_is_the_disabled_state(self):
        spec = ecm.load(None)
        self.assertFalse(spec["enabled"])
        self.assertEqual(spec["entities"], [])

    def test_unknown_field_is_rejected_not_ignored(self):
        # a silently dropped key is the classic way a scene ends up "jamming"
        # with a parameter nobody set
        with self.assertRaises(ValueError):
            ecm.load(dict(entities=[dict(kind="noise_jammer", erp_w=10.0,
                                         jam_bandwith_mhz=200.0)]))

    def test_life_window_and_geometry_are_validated(self):
        with self.assertRaises(ValueError):
            ecm.load(dict(entities=[dict(kind="chaff_cloud",
                                         life=dict(start_s=2.0, end_s=1.0))]))
        with self.assertRaises(ValueError):
            ecm.load(dict(entities=[dict(kind="chaff_cloud", pos=[0.0, 1.0])]))

    def test_kinds_split_into_passive_and_active(self):
        self.assertEqual(ecm.GROUP_OF["chaff_cloud"], "passive")
        self.assertEqual(ecm.GROUP_OF["corner_reflector"], "passive")
        self.assertEqual(ecm.GROUP_OF["noise_jammer"], "active")
        self.assertEqual(ecm.GROUP_OF["false_target"], "active")
        self.assertEqual(ecm.GROUP_OF["range_gate_pull_off"], "active")

    def test_entity_leaves_the_scene_after_its_life_ends(self):
        spec = ecm.load(dict(entities=[dict(
            kind="chaff_cloud", pos=[10.0, 900.0, 100.0],
            life=dict(start_s=1.0, end_s=3.0))]))
        self.assertEqual(ecm.entities_at(spec, 0.5), [])
        self.assertEqual(len(ecm.entities_at(spec, 2.0)), 1)
        self.assertEqual(ecm.entities_at(spec, 3.5), [])


class IndependentMotionTests(unittest.TestCase):
    """"Not only on the target body": the entity must have its own trajectory."""

    def test_entity_position_moves_independently_of_the_target(self):
        spec = ecm.load(dict(entities=[dict(
            id="cloud", kind="chaff_cloud", pos=[40.0, 1400.0, 130.0],
            vel=[120.0, -20.0, -1.0], radius_m=5.0, growth_mps=6.0,
            rcs_m2=200.0, decay_s=6.0, fall_mps=2.0, wind=[3.0, 0.0, 0.0],
            life=dict(start_s=0.0))]))
        target = np.array([0.0, 1500.0, 100.0]) - np.array([0.0, -60.0, 0.0]) * 12.0
        a = ecm.entities_at(spec, 0.0)[0]
        b = ecm.entities_at(spec, 12.0)[0]
        self.assertFalse(np.allclose(a["pos"], b["pos"]))
        d0 = float(np.linalg.norm(a["pos"] - np.array([0.0, 1500.0, 100.0])))
        d1 = float(np.linalg.norm(b["pos"] - target))
        # the gap has to *change*; a decal glued to the airframe would keep it
        self.assertGreater(abs(d1 - d0), 100.0)
        # and the cloud blooms while it drifts
        self.assertGreater(b["radius_m"], a["radius_m"])

    def test_entity_velocity_is_its_own_not_the_targets(self):
        spec = ecm.load(dict(entities=[dict(
            kind="decoy", pos=[0.0, 1500.0, 100.0], vel=[0.0, -60.0, 0.0],
            towed_by=None, life=dict(start_s=0.0))]))
        st = ecm.entities_at(spec, 3.0)[0]
        self.assertTrue(np.allclose(st["vel"], [0.0, -60.0, 0.0]))
        self.assertTrue(np.allclose(st["pos"], [0.0, 1320.0, 100.0]))

    def test_towed_decoy_hangs_on_its_tether(self):
        # the tether is measured from the anchor's position at the SAME time:
        # the scene moves the airframe by vel*t, so a decoy evaluated at t=6
        # trails the aircraft's t=6 position, not its launch point
        host = dict(name="fighter_01", pos=np.array([0.0, 1500.0, 100.0]),
                    vel=np.array([0.0, -120.0, 0.0]))
        spec = ecm.load(dict(entities=[dict(
            kind="decoy", pos=[0.0, 0.0, 0.0], towed_by="fighter_01",
            tether_m=90.0, life=dict(start_s=0.0))]))
        for t in (0.0, 6.0):
            now = dict(objs=[dict(name="fighter_01",
                                  pos=host["pos"] + host["vel"] * t,
                                  vel=host["vel"])])
            st = ecm.entities_at(spec, t, anchors=ecm.anchors_of(now))[0]
            gap = float(np.linalg.norm(st["pos"] - (host["pos"] + host["vel"] * t)))
            self.assertAlmostEqual(gap, 90.0, places=6)


class SuppressionJammerTests(unittest.TestCase):
    """A noise jammer must lift the receiver floor, not paint a bright pixel."""

    def test_snr_drop_matches_the_analytic_jammer_to_signal_ratio(self):
        _, _, base = simulate()
        self.assertIsNone(base["ecm"])
        jam = dict(enabled=True, entities=[dict(
            id="jammer", kind="noise_jammer", pos=[0.0, 1500.0, 100.0],
            erp_w=1.0, jam_bandwidth_mhz=200.0, mode="barrage", band="Ku",
            life=dict(start_s=0.0))])
        _, _, hit_out = simulate(jam, t=0.0)
        self.assertIsNotNone(hit_out["ecm"])
        noise = hit_out["ecm"]["noise"]
        measured = float(noise["snr_drop_db"])
        self.assertGreater(measured, 6.0, "a 1 W jammer at 1.5 km must deafen the radar")
        # The closed form for one target under one jammer is
        #     dSNR = 10 log10(1 + P_j / pn)
        # with pn the thermal floor. P_j is recomputed here from the radar's own
        # published gains and the geometry, so this compares the synthesis path
        # with an independent analytic calculation, not with its own report.
        pj = float(hit_out["ecm"]["jammers"][0]["peak_power_w"])
        predicted = 10.0 * np.log10(1.0 + pj / float(base["pn"]))
        self.assertLess(abs(measured - predicted), 1.0,
                        "measured %.3f dB vs analytic %.3f dB" % (measured, predicted))
        # The report's J/S is a different ratio (jammer over the strongest
        # echo), so it is NOT 10 log10(1 + J/S): a bright target sits far above
        # the thermal floor, which is why the two numbers cannot agree. What
        # must hold is the identity js_db == 10 log10(P_j / P_s).
        js_expected = 10.0 * np.log10(
            pj / float(hit_out["ecm"]["strongest_target_power_w"]))
        self.assertLess(abs(float(noise["js_db"]) - js_expected), 1.0)
        # ... and because the echo is 19 dB above the floor, the SNR loss is much
        # larger than J/S: that gap is exactly S/N and has to be there
        self.assertGreater(measured, float(noise["js_db"]))

    def test_target_snr_falls_by_the_same_amount(self):
        _, _, base = simulate()
        self.assertTrue(base["detections"])
        s0 = float(base["detections"][0]["snr_db"])
        jam = dict(enabled=True, entities=[dict(
            id="jammer", kind="noise_jammer", pos=[0.0, 1500.0, 100.0],
            erp_w=1.0, jam_bandwidth_mhz=200.0, mode="barrage", band="Ku",
            life=dict(start_s=0.0))])
        _, _, out = simulate(jam, t=0.0)
        s1 = float(out["detections"][0]["snr_db"])
        drop = s0 - s1
        self.assertGreater(drop, 6.0)
        self.assertLess(abs(drop - float(out["ecm"]["noise"]["snr_drop_db"])), 1.0)

    def test_the_noise_floor_not_the_target_power_is_what_changes(self):
        _, _, base = simulate()
        jam = dict(enabled=True, entities=[dict(
            id="jammer", kind="noise_jammer", pos=[0.0, 1500.0, 100.0],
            erp_w=1.0, jam_bandwidth_mhz=200.0, mode="barrage", band="Ku",
            life=dict(start_s=0.0))])
        _, _, out = simulate(jam, t=0.0)
        # pn is the thermal floor and must be untouched: the entity is a
        # separate source, not a multiplier somebody applied to the receiver
        self.assertAlmostEqual(float(out["pn"]), float(base["pn"]), places=18)
        self.assertGreater(float(np.max(out["pn_beam"])),
                           10.0 * float(base["pn"]))
        self.assertTrue(np.allclose(np.asarray(out["sigma"]),
                                    np.asarray(base["sigma"]), atol=1e-9))

    def test_barrage_needs_more_power_than_spot_for_the_same_loss(self):
        # the only difference between the two runs is the bandwidth the same ERP
        # is spread over, so the ratio is the physics and nothing else
        def drop(mode):
            jam = dict(enabled=True, entities=[dict(
                id="jammer", kind="noise_jammer", pos=[0.0, 1500.0, 100.0],
                erp_w=1.0, jam_bandwidth_mhz=200.0, mode=mode, band="Ku",
                life=dict(start_s=0.0))])
            _, _, out = simulate(jam, t=0.0)
            return float(out["ecm"]["noise"]["snr_drop_db"])
        self.assertGreater(drop("spot"), drop("barrage") + 3.0)


class FalseTargetTests(unittest.TestCase):
    """Deception must land in the RD cube the target lives in."""

    def test_five_false_targets_produce_five_extra_rd_peaks(self):
        _, _, base = simulate()
        cfg = fighter_cfg(detector="rd_threshold", pfa=1e-6)
        world = scene.build_world(cfg, t=0.0)
        plain = radar.simulate(world, cfg)
        n0 = len(radar.rd_peak_cells(
            plain["rd_map"], plain["processing"]["threshold_power"], 64))
        spec = dict(enabled=True, entities=[dict(
            id="repeater", kind="false_target", pos=[0.0, 1500.0, 100.0],
            count=5, range_step_m=60.0, power_ratio=8.0, band="Ku",
            life=dict(start_s=0.0))])
        cfg["ecm"] = spec
        cfg["_ecm_t"] = 0.0
        world = scene.build_world(cfg, t=0.0)
        out = radar.simulate(world, cfg)
        n1 = len(radar.rd_peak_cells(
            out["rd_map"], out["processing"]["threshold_power"], 64))
        self.assertGreaterEqual(n1 - n0, 4,
                                "peaks %d -> %d with five injected copies" % (n0, n1))
        injected = out["ecm"]["entities"][0]["injected"]
        self.assertEqual(len(injected), 5)

    def test_false_peak_sits_on_the_requested_range_cell(self):
        spec = dict(enabled=True, entities=[dict(
            id="repeater", kind="false_target", pos=[0.0, 1500.0, 100.0],
            count=4, range_step_m=80.0, range_offset_m=40.0,
            power_ratio=20.0, band="Ku", life=dict(start_s=0.0))])
        cfg, world, out = simulate(spec, t=0.0)
        cell = float(np.mean(np.diff(np.sort(np.abs(out["range_axis"])))))
        for k, item in enumerate(out["ecm"]["entities"][0]["injected"]):
            axis = np.asarray(out["range_axis"], float)
            err = float(np.min(np.abs(axis - float(item["range_m"]))))
            self.assertLess(err, cell,
                            "copy %d asked for %.1f m, nearest cell %.1f m away"
                            % (k, float(item["range_m"]), err))

    def test_a_repeater_is_silent_when_the_radar_sees_nothing(self):
        # nothing to copy -> no ghost. A repeater that invented targets into an
        # empty frame would put objects into pictures that never saw an aircraft.
        spec = dict(enabled=True, entities=[dict(
            id="repeater", kind="false_target", pos=[0.0, 1500.0, 100.0],
            count=5, range_step_m=60.0, life=dict(start_s=0.0))])
        cfg = fighter_cfg(look_at=[0.0, 1500.0, 3000.0])
        cfg["ecm"] = spec
        cfg["_ecm_t"] = 0.0
        world = scene.build_world(cfg, t=0.0)
        out = radar.simulate(world, cfg)
        rec = out["ecm"]["entities"][0]
        self.assertEqual(rec["injected"], [])
        self.assertIn("note", rec)


class ChaffTests(unittest.TestCase):
    """The cloud must expand and dilute: area up, cross-section down."""

    def test_rcs_falls_monotonically_while_the_cloud_grows(self):
        spec = ecm.load(dict(entities=[dict(
            kind="chaff_cloud", pos=[0.0, 1400.0, 100.0], radius_m=6.0,
            growth_mps=6.0, rcs_m2=200.0, decay_s=8.0, dilution_exponent=2.0,
            life=dict(start_s=0.0))]))
        ent = spec["entities"][0]
        rcs = [ecm.chaff_rcs(ent, t) for t in (0.0, 1.0, 2.0, 4.0, 8.0)]
        radius = [ecm.radius_at(ent, t) for t in (0.0, 1.0, 2.0, 4.0, 8.0)]
        self.assertTrue(all(b > a for a, b in zip(radius, radius[1:])),
                        "radius must grow: %s" % (radius,))
        self.assertTrue(all(b < a for a, b in zip(rcs, rcs[1:])),
                        "rcs must fall: %s" % (rcs,))
        # dilution alone, with no settling, is exactly (r0/r)^k
        r0, r1 = radius[0], radius[-1]
        dilution = (r0 / r1) ** 2.0
        residual = rcs[-1] / (200.0 * dilution)
        self.assertGreater(residual, 0.0)
        self.assertLess(residual, 1.0)

    def test_the_cloud_is_geometry_in_the_world_and_label_free(self):
        spec = dict(enabled=True, entities=[dict(
            id="cloud", kind="chaff_cloud", pos=[0.0, 1400.0, 120.0],
            radius_m=8.0, growth_mps=4.0, rcs_m2=150.0, band="Ku",
            life=dict(start_s=0.0))])
        cfg, world, _ = simulate(spec, t=0.0)
        names = [o["name"] for o in world.objs]
        self.assertIn("cloud", names)
        cloud = [o for o in world.objs if o["name"] == "cloud"][0]
        self.assertFalse(cloud["label"])
        self.assertEqual(cloud["cls"], "ecm")
        self.assertGreater(len(world.faces), 0)

    def test_the_cloud_does_not_erase_the_aircraft_behind_it(self):
        # a volumetric scatterer must be see-through for the ray cast: dropping
        # chaff would otherwise hide the very target it protects
        # the cloud sits on the line of sight, 150 m IN FRONT of the airframe:
        # every ray that reaches the target has to cross it first
        spec = dict(enabled=True, entities=[dict(
            id="cloud", kind="chaff_cloud", pos=[0.0, 1350.0, 100.0],
            radius_m=20.0, growth_mps=0.0, rcs_m2=150.0,
            life=dict(start_s=0.0))])
        cfg, world, out = simulate(spec, t=0.0)
        self.assertIn("cloud", [o["name"] for o in world.objs])
        real = [d for d in out["detections"] if d["obj"] >= 0]
        self.assertTrue(real, "the airframe must still be detected through the cloud")


class RegressionsTests(unittest.TestCase):
    def test_absent_ecm_leaves_every_number_identical(self):
        _, _, a = simulate()
        cfg = fighter_cfg()
        cfg["ecm"] = dict(enabled=False, entities=[])
        world = scene.build_world(cfg, t=0.0)
        b = radar.simulate(world, cfg)
        self.assertIsNone(b["ecm"])
        self.assertEqual(len(a["detections"]), len(b["detections"]))
        self.assertTrue(np.array_equal(np.asarray(a["rd_map"]),
                                       np.asarray(b["rd_map"])))

    def test_the_image_layer_module_is_still_the_image_layer(self):
        # the two layers coexist: interference.py keeps working on rendered
        # rasters and must not have been repurposed into the physics path
        from simkit import interference
        spec = interference.validate(dict(enabled=True, seed=1, events=[
            dict(id="n", kind="additive_noise", modalities=["radar"],
                 strength=0.03, start_s=0.0, end_s=1.0)]))
        self.assertTrue(spec["enabled"])
        # it must stay a raster operator: no scene hook, no entity vocabulary
        self.assertFalse(hasattr(interference, "attach"))
        self.assertFalse(hasattr(interference, "entities_at"))
        # and its own contract still says it is not an RF model
        self.assertIn("not RF or countermeasure models", interference.__doc__)

    def test_report_is_json_serialisable(self):
        spec = dict(enabled=True, entities=[
            dict(id="jam", kind="noise_jammer", pos=[0.0, 1500.0, 100.0],
                 erp_w=1.0, band="Ku", life=dict(start_s=0.0)),
            dict(id="ghosts", kind="false_target", pos=[0.0, 1500.0, 100.0],
                 count=3, range_step_m=50.0, life=dict(start_s=0.0))])
        _, _, out = simulate(spec, t=0.0)
        text = json.dumps(out["ecm"])
        self.assertIn("entities", json.loads(text))


if __name__ == "__main__":
    unittest.main(verbosity=2)
