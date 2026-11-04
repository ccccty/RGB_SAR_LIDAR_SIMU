"""One state source for every entity: Track, the four link branches, degenerate limits.

What is being pinned here is the design in docs/多目标运动仿真调研.md section 5:

  5.1  a Track is the only state source: pos/vel/accel/frame as analytic
       functions of clip time, accel differentiated from the analytic velocity;
  5.2  four entity links (independent, formation, released, towed) sharing one
       interface, with mount and ejection vectors in the PARENT BODY frame;
  5.3  the state rules, including the three degenerate limits the document
       requires as unit tests:
         * L -> 0 (or tau -> 0): the dispensed body stops in the air at once,
           p is the release point and v is the air velocity;
         * wind = 0, fall = 0: only the ejection displacement remains;
         * t < t_release: identically equal to a formation member with offset 0.

The numeric acceptance cases the task asks for are here too, because they are
the ones that used to fail by eye:

  * a two-ship formation keeps its spacing (< 1% relative deviation) head-on
    and in pursuit, because the follower's state is DERIVED from the leader's;
  * after a chaff release the relative speed grows toward the aircraft's own
    speed and the distance grows without limit (the aircraft flies on, the
    cloud stops against the air) -- the reported ranges are in the assertion
    messages so a regression is a number, not a picture;
  * a 90 deg parent yaw carries the mount around the body, because the mount is
    written in body axes and rotated by Track.frame(t).
"""
import glob
import json
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simkit import ecm, motion, plan, scene  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _central(fn, t, h=1e-5):
    return (np.asarray(fn(t + h), float) - np.asarray(fn(t - h), float)) / (2.0 * h)


def load_entity(**ent):
    """One entity through the real validator, with the schema spelled out."""
    return ecm.load(dict(entities=[dict(ent)]))["entities"][0]


def straight_host(pos=(0.0, 0.0, 3000.0), vel=(200.0, 0.0, 0.0)):
    """A parent Track flying a straight line: the exact parent of every case."""
    return motion.PathTrack(motion.Line(list(pos), list(vel)))


def released_spec(**over):
    """A chaff body written with the new schema (5.2)."""
    ent = dict(id="chaff", kind="chaff_cloud", life=dict(start_s=0.0),
               release=dict(parent="host", t_s=2.0,
                            mount_body=[-8.0, 1.5, -1.0],
                            eject_body=[0.0, 18.0, -12.0],
                            drag=dict(mode="lag", tau_s=0.25),
                            air=dict(wind_mps=[5.0, 0.0, 0.0], fall_mps=0.3)))
    ent.update(over)
    return ecm.load(dict(entities=[ent]))["entities"][0]


class TrackContractTests(unittest.TestCase):
    """5.1: pos/vel/accel/frame, analytic, and independent of any chunking."""

    def test_from_path_is_the_path_and_its_exact_derivatives(self):
        path = motion.Arc([0.0, 5000.0, 3000.0], 5000.0, 0.04, z0=3000.0)
        track = motion.Track.from_path(path)
        self.assertIsInstance(track, motion.Track)
        for t in (0.0, 1.1, 4.0):
            np.testing.assert_allclose(track.pos(t), path.pos(t))
            np.testing.assert_allclose(track.vel(t), path.vel(t))
            np.testing.assert_allclose(track.accel(t), path.accel(t))
            # and the published velocity really is the position derivative
            self.assertLess(
                float(np.linalg.norm(np.asarray(track.vel(t), float)
                                     - _central(track.pos, t))), 1e-4)

    def test_accel_comes_from_velocity_not_from_position(self):
        # A track whose position has curvature: d(vel)/dt and the analytic
        # acceleration must agree, and it must be the velocity that is
        # differenced (the design's h = 1e-3 rule).
        w = 0.7

        def pos(t):
            return np.array([10.0 + 3.0 * t, 4.0 + 2.0 * np.sin(w * t),
                             3000.0 + 0.5 * t * t])

        def vel(t):
            return np.array([3.0, 2.0 * w * np.cos(w * t), t])

        def accel(t):
            return np.array([0.0, -2.0 * w * w * np.sin(w * t), 1.0])

        track = motion.Track.from_functions(pos, vel)
        t = 0.9
        # the default accel IS the centred difference of vel, so it matches the
        # closed-form d(vel)/dt to the O(h^2) truncation error of h = 1e-3
        self.assertLess(float(np.linalg.norm(np.asarray(track.accel(t), float)
                                             - np.asarray(accel(t), float))), 1e-6)
        # the published accel is exactly the centred difference of the analytic
        # velocity at h = 1e-3 -- not a difference of positions
        h = 1e-3
        want = (np.asarray(vel(t + h), float) - np.asarray(vel(t - h), float)) / (2.0 * h)
        np.testing.assert_allclose(track.accel(t), want, atol=1e-9)

    def test_frame_is_the_scene_rotation_of_yaw_pitch_roll(self):
        yaw, pitch, roll = 37.0, 6.0, -21.0
        track = motion.Track.from_functions(
            lambda t: [0.0, 0.0, 3000.0], lambda t: [200.0, 0.0, 0.0],
            yaw=lambda t: yaw, pitch=lambda t: pitch, roll=lambda t: roll)
        from simkit.mesh import rot_x, rot_y, rot_z
        want = rot_z(yaw) @ rot_y(-pitch) @ rot_x(roll)
        np.testing.assert_allclose(track.frame(0.0), want, atol=1e-12)
        # body +x maps to the heading: a mount point at [1, 0, 0] is forward
        forward = np.asarray(track.frame(0.0), float) @ np.array([1.0, 0.0, 0.0])
        self.assertAlmostEqual(math.degrees(math.atan2(forward[1], forward[0])),
                               yaw, places=9)

    def test_angular_velocity_is_the_frame_rate(self):
        track = motion.AnchorTrack([0.0, 0.0, 3000.0], [200.0, 0.0, 0.0], yaw=0.0)
        np.testing.assert_allclose(motion.angular_velocity(track, 0.0, 0.0),
                                   [0.0, 0.0, 0.0], atol=1e-9)
        # a frame yawing at 20 deg/s reads back 0.349 rad/s about +z
        turn = motion.Track.from_functions(
            lambda tt: [0.0, 0.0, 3000.0], lambda tt: [200.0, 0.0, 0.0],
            yaw=lambda tt: 20.0 * tt)
        w = motion.angular_velocity(turn, 0.0, 0.5)
        self.assertAlmostEqual(float(w[2]), math.radians(20.0), places=9)


class LinkContractTests(unittest.TestCase):
    """5.2/5.3: four link branches, one interface, derivative contract each."""

    def _four(self):
        host = straight_host()
        ents = {
            "independent": load_entity(
                id="free", kind="corner_reflector", pos=[40.0, 1500.0, 130.0],
                vel=[120.0, -20.0, -1.0], life=dict(start_s=0.0)),
            "formation": load_entity(
                id="wing", kind="corner_reflector", life=dict(start_s=0.0),
                formation=dict(parent="host", offset_body=[-120.0, 60.0, -15.0],
                               mode="lag", tau_lag_s=0.3)),
            "released": released_spec(),
            "towed": load_entity(
                id="decoy", kind="decoy", life=dict(start_s=0.0),
                tether=dict(parent="host", length_m=90.0, tau_s=0.2,
                            droop_deg=3.0)),
        }
        return host, ents

    def test_every_link_satisfies_the_derivative_contract(self):
        host, ents = self._four()
        tracks = {"host": host}
        for name, ent in ents.items():
            # sample away from the release instant: across it the velocity jumps
            # by the ejection impulse, and a centred difference there has no
            # meaning (E7 allows the jump, it is the ejection)
            if name == "released":
                # away from the ejection impulse
                times = (2.2, 3.5, 5.0)
            elif name == "independent":
                # away from life.start_s, where an absolute chaff cloud starts
                times = (1.1, 2.0, 3.0)
            else:
                times = (0.0, 0.7, 1.9)
            for t in times:
                with self.subTest(link=name, t=t):
                    st = ecm.entity_state(ent, t, tracks=tracks)
                    err = float(np.linalg.norm(np.asarray(st["vel"], float)
                                               - _central(
                                                   lambda tt: ecm.entity_state(
                                                       ent, tt, tracks=tracks)["pos"], t)))
                    self.assertLess(err, 1e-3,
                                    "%s: v disagrees with dp/dt at t=%s" % (name, t))

    def test_independent_is_the_absolute_line_it_always_was(self):
        _host, ents = self._four()
        st = ecm.entity_state(ents["independent"], 3.0, tracks={})
        np.testing.assert_allclose(st["pos"], [40.0 + 360.0, 1500.0 - 60.0, 130.0 - 3.0])
        np.testing.assert_allclose(st["vel"], [120.0, -20.0, -1.0])

    def test_an_absolute_chaff_relaxes_to_the_air_mass(self):
        # a twin-fighter style cloud: absolute pos/vel + wind/fall, no parent.
        # Keeping the authored 190 m/s forever is what made chaff Doppler-shift
        # exactly like the aircraft it was dispensed from.
        ent = load_entity(
            id="cloud", kind="chaff_cloud", pos=[-70.0, -50.0, 2994.0],
            vel=[190.0, 0.0, -2.0], wind=[6.0, 0.0, 0.0], fall_mps=1.2,
            rel_decay_s=0.25, life=dict(start_s=0.0))
        st0 = ecm.entity_state(ent, 0.0, tracks={})
        np.testing.assert_allclose(st0["pos"], [-70.0, -50.0, 2994.0], atol=1e-9)
        np.testing.assert_allclose(st0["vel"], [190.0, 0.0, -2.0], atol=1e-9)
        late = ecm.entity_state(ent, 5.0, tracks={})
        # terminal state is the air mass: wind 6 m/s, fall 1.2 m/s
        np.testing.assert_allclose(late["vel"], [6.0, 0.0, -1.2], atol=1e-6)
        self.assertLess(abs(float(late["vel"][0])), 10.0,
                        "the cloud still carries the aircraft's speed")
        moved = float(np.linalg.norm(np.asarray(late["pos"], float)
                                     - np.asarray(st0["pos"], float)))
        self.assertLess(moved, 200.0)

    def test_formation_offset_is_in_the_parents_body_frame(self):
        host, ents = self._four()
        ent = ents["formation"]
        tracks = {"host": host}
        mount = np.array([-120.0, 60.0, -15.0])
        for t in (0.0, 1.5, 4.0):
            st = ecm.entity_state(ent, t, tracks=tracks)
            body = np.asarray(host.frame(t - 0.3), float).T @ (
                np.asarray(st["pos"], float) - np.asarray(host.pos(t), float))
            np.testing.assert_allclose(body, mount, atol=1e-9)
        # and the distance never drifts: |offset| is a formation constant
        dist = [float(np.linalg.norm(
            np.asarray(ecm.entity_state(ent, t, tracks=tracks)["pos"], float)
            - np.asarray(host.pos(t), float))) for t in np.linspace(0.0, 8.0, 33)]
        self.assertLess(max(abs(d - float(np.linalg.norm(mount))) for d in dist), 1e-9)

    def test_formation_free_mode_manoeuvre_is_bounded_and_still_c1(self):
        host = straight_host()
        track = motion.FormationTrack(
            host, [-100.0, 40.0, -10.0], mode="free", tau_lag_s=0.3,
            own_maneuver=dict(mode="weave", amp_m=8.0, hz=0.3))
        # the manoeuvre stays inside the authored box: never further than the
        # offset plus its amplitude, and the base offset is always in it
        limit = float(np.linalg.norm([-100.0, 40.0, -10.0])) + 8.0 + 1e-9
        for t in np.linspace(0.0, 6.0, 61):
            gap = float(np.linalg.norm(np.asarray(track.pos(t), float)
                                       - np.asarray(host.pos(t), float)))
            self.assertLessEqual(gap, limit)
            err = float(np.linalg.norm(np.asarray(track.vel(t), float)
                                       - _central(track.pos, t)))
            self.assertLess(err, 1e-3)

    def test_formation_max_offset_clamps_a_scene_that_writes_nonsense(self):
        host = straight_host()
        track = motion.FormationTrack(host, [0.0, 5000.0, 0.0], mode="rigid",
                                      max_offset_m=500.0)
        gap = float(np.linalg.norm(np.asarray(track.pos(1.0), float)
                                   - np.asarray(host.pos(1.0), float)))
        self.assertAlmostEqual(gap, 500.0, places=9)

    def test_released_is_strictly_attached_before_the_release(self):
        """E1 / the third required degenerate check."""
        host, ents = self._four()
        rel = ents["released"]
        mount = np.array(rel["release"]["mount_body"], float)
        wing = load_entity(
            id="wing", kind="corner_reflector", life=dict(start_s=0.0),
            formation=dict(parent="host", offset_body=list(mount),
                           mode="rigid", tau_lag_s=0.0))
        tracks = {"host": host}
        for t in (0.0, 0.5, 1.999):
            a = ecm.entity_state(rel, t, tracks=tracks)
            b = ecm.entity_state(wing, t, tracks=tracks)
            np.testing.assert_allclose(a["pos"], b["pos"], atol=1e-9)
            np.testing.assert_allclose(a["vel"], b["vel"], atol=1e-9)

    def test_release_point_is_continuous_and_the_velocity_jumps_by_the_ejection(self):
        host, ents = self._four()
        rel = ents["released"]
        tracks = {"host": host}
        tr = float(rel["release"]["t_s"])
        before = ecm.entity_state(rel, tr - 1e-6, tracks=tracks)
        after = ecm.entity_state(rel, tr + 1e-6, tracks=tracks)
        self.assertLess(float(np.linalg.norm(np.asarray(after["pos"], float)
                                             - np.asarray(before["pos"], float))), 1e-3)
        eject = np.array(rel["release"]["eject_body"], float)
        dv = np.asarray(after["vel"], float) - np.asarray(before["vel"], float)
        np.testing.assert_allclose(dv, eject, atol=1e-2)

    def test_towed_decoy_hangs_at_the_tether_and_turns_with_the_aircraft(self):
        host = straight_host()
        ent = load_entity(
            id="decoy", kind="decoy", life=dict(start_s=0.0),
            tether=dict(parent="host", length_m=90.0, tau_s=0.2, droop_deg=0.0))
        tracks = {"host": host}
        for t in (0.0, 3.0):
            st = ecm.entity_state(ent, t, tracks=tracks)
            gap = float(np.linalg.norm(np.asarray(st["pos"], float)
                                       - np.asarray(host.pos(t), float)))
            self.assertAlmostEqual(gap, 90.0, places=9)
            # behind the aircraft, not beside it
            self.assertLess(float(np.asarray(st["pos"], float)[0]),
                            float(np.asarray(host.pos(t), float)[0]))
        # droop lowers the body below the tether line
        droop = load_entity(
            id="decoy", kind="decoy", life=dict(start_s=0.0),
            tether=dict(parent="host", length_m=90.0, tau_s=0.2, droop_deg=10.0))
        st0 = ecm.entity_state(ent, 1.0, tracks=tracks)
        st1 = ecm.entity_state(droop, 1.0, tracks=tracks)
        self.assertLess(float(st1["pos"][2]), float(st0["pos"][2]))

    def test_the_mount_turns_with_a_ninety_degree_parent_yaw(self):
        """Acceptance 3: the body-frame mount rotates with the airframe."""
        mount = [-8.0, 1.5, -1.0]
        ent = load_entity(
            id="chaff", kind="chaff_cloud", life=dict(start_s=0.0),
            release=dict(parent="host", t_s=5.0, mount_body=mount,
                         eject_body=[0.0, 0.0, 0.0],
                         drag=dict(mode="lag", tau_s=0.25)))
        host0 = straight_host(pos=(0.0, 0.0, 3000.0), vel=(200.0, 0.0, 0.0))
        host90 = straight_host(pos=(0.0, 0.0, 3000.0), vel=(0.0, 200.0, 0.0))
        p0 = np.asarray(ecm.entity_state(ent, 1.0, tracks={"host": host0})["pos"], float)
        p90 = np.asarray(ecm.entity_state(ent, 1.0, tracks={"host": host90})["pos"], float)
        m0 = p0 - np.asarray(host0.pos(1.0), float)
        m90 = p90 - np.asarray(host90.pos(1.0), float)
        # the world offset changed (it is NOT pinned to the old world azimuth)...
        self.assertGreater(float(np.linalg.norm(m90 - m0)), 1.0)
        # ...by exactly the 90 deg body rotation, and the lever arm is unchanged
        r90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        np.testing.assert_allclose(m90, r90 @ m0, atol=1e-9)
        self.assertAlmostEqual(float(np.linalg.norm(m0)),
                               float(np.linalg.norm(mount)), places=9)

    def test_a_turning_aircraft_carries_the_mount_around_its_body(self):
        """A yaw rate must move the mount in the world, the body offset constant."""
        ent = load_entity(
            id="chaff", kind="chaff_cloud", life=dict(start_s=0.0),
            release=dict(parent="host", t_s=100.0, mount_body=[-8.0, 1.5, -1.0],
                         eject_body=[0.0, 0.0, 0.0],
                         drag=dict(mode="lag", tau_s=0.25)))
        turn = motion.PathTrack(motion.Arc([0.0, 8000.0, 3000.0], 8000.0, 0.035,
                                           z0=3000.0))
        mount = np.array([-8.0, 1.5, -1.0])
        world = []
        for t in (0.0, 1.0, 2.0, 3.0):
            st = ecm.entity_state(ent, t, tracks={"host": turn})
            body = np.asarray(turn.frame(t), float).T @ (
                np.asarray(st["pos"], float) - np.asarray(turn.pos(t), float))
            np.testing.assert_allclose(body, mount, atol=1e-9)
            world.append(np.asarray(st["pos"], float))
        # and it really moved: a mount pinned to the world would not
        self.assertGreater(float(np.linalg.norm(world[-1] - world[0])), 1.0)


class ReleasedPhysicsTests(unittest.TestCase):
    """The physics that made chaff look like it was glued to the wing."""

    def test_chaff_table_distance_and_relative_speed(self):
        """Acceptance 2: separation grows without limit, rel speed -> aircraft."""
        host = straight_host(vel=(200.0, 0.0, 0.0))
        ent = released_spec(release=dict(
            parent="host", t_s=0.0, mount_body=[-8.0, 1.5, -1.0],
            eject_body=[0.0, 18.0, -12.0],
            drag=dict(mode="lag", tau_s=0.25),
            air=dict(wind_mps=[0.0, 0.0, 0.0], fall_mps=0.0)))
        tracks = {"host": host}
        table = []
        for t in (0.0, 1.0, 2.0, 5.0):
            st = ecm.entity_state(ent, t, tracks=tracks)
            d = float(np.linalg.norm(np.asarray(st["pos"], float)
                                     - np.asarray(host.pos(t), float)))
            vr = float(np.linalg.norm(np.asarray(st["vel"], float)
                                      - np.asarray(host.vel(t), float)))
            table.append((t, d, vr))
        msg = "t/d/vrel = " + ", ".join("%.1f s: %.1f m, %.2f m/s" % row for row in table)
        # the cloud stops against the air, the aircraft keeps flying: distance
        # grows without bound (a bounded rel_v*tau model saturates near 5 m)
        self.assertGreater(table[-1][1], 500.0, msg)
        self.assertGreater(table[-1][1], table[2][1], msg)
        for (_t0, _d0, v0), (_t1, _d1, v1) in zip(table, table[1:]):
            self.assertGreaterEqual(v1, v0 - 1e-9, msg)
        # at the end the relative speed IS the aircraft's speed: it left the
        # airframe behind, it did not fly along with it
        self.assertGreater(table[-1][2], 0.95 * 200.0, msg)
        self.assertLess(table[-1][2], 200.0 + 1e-6, msg)

    def test_quadratic_drag_mode_approaches_the_same_limit(self):
        host = straight_host()
        ent = released_spec(release=dict(
            parent="host", t_s=0.0, mount_body=[0.0, 0.0, 0.0],
            eject_body=[0.0, 20.0, 0.0],
            drag=dict(mode="quadratic", L_m=10.0),
            air=dict(wind_mps=[0.0, 0.0, 0.0], fall_mps=0.0)))
        tracks = {"host": host}
        d0 = float(np.linalg.norm(
            np.asarray(ecm.entity_state(ent, 0.0, tracks=tracks)["pos"], float)
            - np.asarray(host.pos(0.0), float)))
        d5 = float(np.linalg.norm(
            np.asarray(ecm.entity_state(ent, 5.0, tracks=tracks)["pos"], float)
            - np.asarray(host.pos(5.0), float)))
        self.assertGreater(d5, d0 + 500.0)

    def test_zero_length_or_zero_tau_stops_the_body_in_the_air(self):
        """First required degenerate check: L -> 0 and tau -> 0."""
        host = straight_host()
        tracks = {"host": host}
        pr = np.asarray(host.pos(2.0), float) + np.array([-8.0, 1.5, -1.0])
        air = np.array([5.0, 0.0, -0.3])
        cases = [
            ("tau", released_spec(release=dict(
                parent="host", t_s=2.0, mount_body=[-8.0, 1.5, -1.0],
                eject_body=[0.0, 18.0, -12.0], drag=dict(mode="lag", tau_s=0.0),
                air=dict(wind_mps=[5.0, 0.0, 0.0], fall_mps=0.3)))),
            ("L", released_spec(release=dict(
                parent="host", t_s=2.0, mount_body=[-8.0, 1.5, -1.0],
                eject_body=[0.0, 18.0, -12.0], drag=dict(mode="quadratic", L_m=0.0),
                air=dict(wind_mps=[5.0, 0.0, 0.0], fall_mps=0.3)))),
        ]
        for name, ent in cases:
            for u in (0.01, 0.5, 3.0):
                st = ecm.entity_state(ent, 2.0 + u, tracks=tracks)
                np.testing.assert_allclose(st["pos"], pr + air * u, atol=1e-9,
                                           err_msg="%s=0: the body did not stop" % name)
                np.testing.assert_allclose(st["vel"], air, atol=1e-12,
                                           err_msg="%s=0: velocity is not the air mass" % name)

    def test_zero_wind_and_zero_fall_leaves_only_the_ejection(self):
        """Second required degenerate check: A == 0, so only the ejection moves."""
        host = straight_host()
        tracks = {"host": host}
        ent = released_spec(release=dict(
            parent="host", t_s=1.0, mount_body=[0.0, 0.0, 0.0],
            eject_body=[0.0, 20.0, 0.0],
            drag=dict(mode="lag", tau_s=5.0),
            air=dict(wind_mps=[0.0, 0.0, 0.0], fall_mps=0.0)))
        pr = np.asarray(host.pos(1.0), float)
        # A == 0 (no wind, no fall): the only displacement is the release state
        # relaxing toward the air, i.e. vrel0 * s with vrel0 = v_host + eject
        vrel0 = np.asarray(host.vel(1.0), float) + np.array([0.0, 20.0, 0.0])
        for u in (0.0, 0.5, 2.0):
            st = ecm.entity_state(ent, 1.0 + u, tracks=tracks)
            frac = math.exp(-u / 5.0)
            s = 5.0 * (1.0 - frac)
            np.testing.assert_allclose(st["pos"], pr + vrel0 * s, atol=1e-9)
            np.testing.assert_allclose(st["vel"], vrel0 * frac, atol=1e-9)

    def test_wind_drift_is_measured_from_the_release_not_from_zero(self):
        """The drift bug in 1.2: at t < t_release the body must not have drifted."""
        host = straight_host(pos=(0.0, 0.0, 3000.0))
        ent = released_spec(release=dict(
            parent="host", t_s=2.0, mount_body=[-8.0, 1.5, -1.0],
            eject_body=[0.0, 18.0, -12.0],
            drag=dict(mode="lag", tau_s=0.25),
            air=dict(wind_mps=[5.0, 0.0, 0.0], fall_mps=0.3)))
        tracks = {"host": host}
        # attached at t=1: the position is the pylon, untouched by the wind
        st = ecm.entity_state(ent, 1.0, tracks=tracks)
        np.testing.assert_allclose(
            np.asarray(st["pos"], float),
            np.asarray(host.pos(1.0), float) + np.array([-8.0, 1.5, -1.0]), atol=1e-9)
        np.testing.assert_allclose(st["vel"], host.vel(1.0), atol=1e-9)
        # and just after release the drift starts from the release point, not
        # from t=0: at u=0 there is no wind displacement yet
        st0 = ecm.entity_state(ent, 2.0, tracks=tracks)
        self.assertLess(float(st0["pos"][2]), 3000.0 - 0.9)

    def test_life_start_and_release_time_are_independent(self):
        """5.2: life.start_s = when it is visible, release.t_s = when it lets go."""
        host = straight_host()
        ent = released_spec(life=dict(start_s=3.0, end_s=5.0),
                            release=dict(parent="host", t_s=1.0,
                                         mount_body=[-8.0, 1.5, -1.0],
                                         eject_body=[0.0, 18.0, -12.0],
                                         drag=dict(mode="lag", tau_s=0.25)))
        spec = dict(enabled=True, entities=[ent])
        tracks = {"host": host}
        self.assertEqual(ecm.entities_at(spec, 0.5, tracks=tracks), [])
        self.assertEqual(ecm.entities_at(spec, 4.0, tracks=tracks)[0]["id"], "chaff")
        self.assertEqual(ecm.entities_at(spec, 6.0, tracks=tracks), [])
        # visible from 3.0 yet released at 1.0: it is already flying free at 4.0
        st = ecm.entity_state(ent, 4.0, tracks=tracks)
        self.assertGreater(float(np.linalg.norm(
            np.asarray(st["vel"], float) - np.asarray(host.vel(4.0), float))), 100.0)


class PlanTrackTests(unittest.TestCase):
    """plan side: the primary target is a Track, followers are derived."""

    def _config(self, seeker=True, closing="in", objects=None):
        cam = dict(pos=[0, 0, 0], look_at=[0, 8000, 0], width=320, height=240,
                   fov_deg=2.2)
        objs = objects
        if objs is None:
            objs = [dict(name="lead", shape="box", size=[10.0, 4.0, 3.0],
                         pos=[0.0, 6000.0, 300.0], vel=[0.0, 180.0, 0.0],
                         cls="fighter"),
                    dict(name="wing", shape="box", size=[10.0, 4.0, 3.0],
                         pos=[60.0, 6000.0, 300.0], vel=[0.0, 180.0, 0.0],
                         cls="fighter")]
        cfg = dict(seed=5, frames=dict(n=20, dt=0.1), ground=dict(enabled=False),
                   weather=dict(preset="clear", mode="fixed"),
                   sensors=dict(optical=cam), objects=objs)
        if seeker:
            cfg["clip"] = dict(seeker=dict(closing=closing, target_px=[8, 45]))
        return cfg

    def test_the_primary_target_is_exposed_as_a_track(self):
        cfg = self._config()
        cp = plan.ClipPlan(cfg, 0, 20, 0.1, 5)
        self.assertIsInstance(cp.track_target, motion.Track)
        self.assertIs(cp.track(), cp.track_target)
        for t in (0.0, 0.7, 1.5):
            np.testing.assert_allclose(cp.target_pos(t), cp.track_target.pos(t))
            np.testing.assert_allclose(cp.target_vel(t), cp.track_target.vel(t))
            self.assertLess(float(np.linalg.norm(
                np.asarray(cp.target_vel(t), float)
                - _central(cp.target_pos, t, 1e-5))), 1e-6)
        # the seeker's acceleration is the centred difference of its velocity
        t = 0.8
        h = 1e-3
        want = (np.asarray(cp.target_vel(t + h), float)
                - np.asarray(cp.target_vel(t - h), float)) / (2.0 * h)
        np.testing.assert_allclose(cp.track_target.accel(t), want, atol=1e-9)

    def test_secondary_target_defaults_to_formation_and_keeps_the_spacing(self):
        """Acceptance 1: two-ship spacing is constant head-on and in pursuit."""
        for closing in ("in", "out"):
            with self.subTest(closing=closing):
                cfg = self._config(closing=closing)
                cp = plan.ClipPlan(cfg, 0, 20, 0.1, 5)
                wing = cp.tracks["wing"]
                self.assertIsInstance(wing.base, motion.FormationTrack)
                gaps = [float(np.linalg.norm(
                    np.asarray(wing.pos(t), float)
                    - np.asarray(cp.track_target.pos(t), float)))
                        for t in np.linspace(0.0, cp.dur, 41)]
                ref = gaps[0]
                rel = max(abs(g - ref) / ref for g in gaps)
                self.assertLess(rel, 0.01,
                                "spacing deviates by %.4f%%" % (100.0 * rel))
                # the wingman really moves with the lead (it is not parked)
                self.assertGreater(float(np.linalg.norm(
                    np.asarray(wing.pos(cp.dur), float)
                    - np.asarray(wing.pos(0.0), float))), 100.0)

    def test_explicit_object_trajectory_still_wins(self):
        cfg = self._config()
        cfg["objects"][1]["trajectory"] = dict(mode="line",
                                               origin=[60.0, 6000.0, 300.0],
                                               vel=[40.0, 0.0, 0.0])
        cp = plan.ClipPlan(cfg, 0, 20, 0.1, 5)
        self.assertIsInstance(cp.track("wing").base, motion.PathTrack)
        p, v, _yaw = cp.object_state(cfg["objects"][1], 1.0)
        np.testing.assert_allclose(p, [100.0, 6000.0, 300.0], atol=1e-9)
        np.testing.assert_allclose(v, [40.0, 0.0, 0.0], atol=1e-9)

    def test_independent_link_keeps_the_absolute_line(self):
        cfg = self._config()
        cfg["objects"][1]["link"] = "independent"
        cp = plan.ClipPlan(cfg, 0, 20, 0.1, 5)
        self.assertIsInstance(cp.track("wing").base, motion.PathTrack)
        p, _v, _y = cp.object_state(cfg["objects"][1], 2.0)
        np.testing.assert_allclose(p, [60.0, 6360.0, 300.0], atol=1e-9)

    def test_a_world_scene_that_authors_its_own_lines_keeps_them(self):
        """A parking lot is not a formation: an authored line stays authored."""
        cfg = self._config(seeker=False)
        cfg["clip"] = dict(trajectory=dict(mode="line",
                                           origin=[0.0, 6000.0, 300.0],
                                           vel=[0.0, 180.0, 0.0]))
        cp = plan.ClipPlan(cfg, 0, 20, 0.1, 5)
        self.assertIsInstance(cp.track("wing").base, motion.PathTrack)

    def test_an_explicit_formation_block_places_the_offset_in_the_body_frame(self):
        cfg = self._config()
        cfg["objects"][1].pop("vel", None)
        cfg["objects"][1]["formation"] = dict(parent="lead",
                                             offset_body=[0.0, 80.0, 0.0],
                                             mode="rigid", tau_lag_s=0.0)
        cp = plan.ClipPlan(cfg, 0, 20, 0.1, 5)
        lead = cp.track("lead")
        wing = cp.track("wing")
        for t in (0.0, 1.0, 1.8):
            body = np.asarray(lead.frame(t), float).T @ (
                np.asarray(wing.pos(t), float) - np.asarray(lead.pos(t), float))
            np.testing.assert_allclose(body, [0.0, 80.0, 0.0], atol=1e-9)

    def test_a_curved_lead_makes_the_default_a_formation(self):
        cfg = self._config(seeker=False)
        cfg["clip"] = dict(trajectory=dict(mode="arc", centre=[0.0, 8000.0, 300.0],
                                           radius_m=2000.0, speed_mps=180.0,
                                           origin=[0.0, 6000.0, 300.0]))
        cfg["objects"][1].pop("vel", None)
        cp = plan.ClipPlan(cfg, 0, 20, 0.1, 5)
        self.assertIsInstance(cp.track("wing").base, motion.FormationTrack)

    def test_the_frame_config_carries_the_tracks_and_stays_json_safe(self):
        cfg = self._config()
        cfg["ecm"] = dict(enabled=True, entities=[dict(
            id="chaff_01", kind="chaff_cloud", life=dict(start_s=0.0),
            release=dict(parent="lead", t_s=0.5, mount_body=[-8.0, 1.5, -1.0],
                         eject_body=[0.0, 18.0, -12.0],
                         drag=dict(mode="lag", tau_s=0.25),
                         air=dict(wind_mps=[5.0, 0.0, 0.0], fall_mps=0.3)))])
        cp = plan.ClipPlan(cfg, 0, 20, 0.1, 5)
        st = cp.state(6, 6)
        # the ECM layer sees exactly the planner's tracks...
        self.assertIs(ecm.tracks_of(st["cfg"]), cp.tracks)
        # ...and the config stays JSON-serialisable (dataset.py deep-copies it)
        json.dumps(st["cfg"])
        # the parent state comes from the Track, so a body still on the pylon
        # sits exactly at the mount point of the lead at that same instant
        ent = ecm.load(cfg["ecm"])["entities"][0]
        t_attach = 0.3
        body = ecm.entity_state(ent, t_attach, tracks=ecm.tracks_of(st["cfg"]))
        want = (np.asarray(cp.track_target.pos(t_attach), float)
                + np.asarray(cp.track_target.frame(t_attach), float)
                @ np.array([-8.0, 1.5, -1.0]))
        np.testing.assert_allclose(body["pos"], want, atol=1e-9)
        # and the built World agrees with the Track for this frame (one source)
        t_frame = float(st["t"])
        body_frame = ecm.entity_state(ent, t_frame, tracks=ecm.tracks_of(st["cfg"]))
        world = scene.build_world(st["cfg"], jitter=st["jitter"])
        chaff = [o for o in world.objs if o["name"] == "chaff_01"][0]
        self.assertLess(float(np.linalg.norm(np.asarray(chaff["pos"], float)
                                             - np.asarray(body_frame["pos"], float))), 1e-9)

    def test_both_targets_stay_in_the_same_frame(self):
        """The symptom: the lead on the LOS, the wingman flying out of frame."""
        from simkit import labels
        for closing in ("in", "out"):
            with self.subTest(closing=closing):
                cfg = self._config(closing=closing)
                cp = plan.ClipPlan(cfg, 0, 20, 0.1, 5)
                offsets = []
                width = 0.0
                for i in range(0, 20, 4):
                    st = cp.state(i, i)
                    cam = next(iter(st["views"].values()))
                    world = scene.build_world(st["cfg"], jitter=st["jitter"])
                    wing = [o for o in world.objs if o["name"] == "wing"][0]
                    px = labels.project_points(wing["corners"], cam,
                                               cam["width"], cam["height"])
                    self.assertTrue((px[:, 2] > 1.0).all(), "wing is behind the camera")
                    cx = float(np.mean(px[:, 0]))
                    cy = float(np.mean(px[:, 1]))
                    self.assertTrue(0.0 <= cx < cam["width"],
                                    "wing left the frame at x=%.1f" % cx)
                    self.assertTrue(0.0 <= cy < cam["height"],
                                    "wing left the frame at y=%.1f" % cy)
                    offsets.append(cx - cam["width"] / 2.0)
                    width = float(cam["width"])
                # and the offset on screen does not run away: it is a formation
                self.assertLess(max(offsets) - min(offsets), width)


class CompatibilityTests(unittest.TestCase):
    """5.4: the old field names map onto the link model, scene by scene."""

    def test_parent_rel_pos_rel_vel_maps_to_a_released_body(self):
        ent = load_entity(
            id="chaff", kind="chaff_cloud", parent="lead",
            rel_pos=[-60.0, 20.0, 0.0], rel_vel=[-6.0, 3.0, -1.5],
            release_s=1.0, rel_decay_s=0.3, wind=[5.0, 0.0, 0.0],
            fall_mps=0.5, life=dict(start_s=0.0))
        self.assertEqual(ent["link"], "released")
        rel = ent["release"]
        self.assertEqual(rel["parent"], "lead")
        self.assertEqual(rel["t_s"], 1.0)
        np.testing.assert_allclose(rel["mount_body"], [-60.0, 20.0, 0.0])
        np.testing.assert_allclose(rel["eject_body"], [-6.0, 3.0, -1.5])
        self.assertEqual(rel["drag"]["mode"], "lag")
        self.assertAlmostEqual(rel["drag"]["tau_s"], 0.3)
        np.testing.assert_allclose(rel["air"]["wind_mps"], [5.0, 0.0, 0.0])
        self.assertAlmostEqual(rel["air"]["fall_mps"], 0.5)

    def test_a_displaced_body_defaults_to_a_fast_lag(self):
        ent = load_entity(
            id="chaff", kind="chaff_cloud", parent="lead",
            rel_pos=[-60.0, 20.0, 0.0], life=dict(start_s=0.0))
        # chaff cloud: tau_q 0.03-0.2 s, so tau ~ 3-5 tau_q is 0.2-0.3 s
        self.assertLess(ent["release"]["drag"]["tau_s"], 0.5)
        flare = load_entity(
            id="flare", kind="infrared_flare", parent="lead",
            rel_pos=[-10.0, 4.0, -2.0], life=dict(start_s=0.0))
        self.assertLess(flare["release"]["drag"]["tau_s"], 1.6)
        self.assertGreater(flare["release"]["drag"]["tau_s"], 0.8)

    def test_towed_by_and_tether_map_to_the_towed_link(self):
        ent = load_entity(
            id="decoy", kind="decoy", towed_by="lead", tether_m=90.0,
            life=dict(start_s=0.0))
        self.assertEqual(ent["link"], "towed")
        self.assertEqual(ent["tether"]["parent"], "lead")
        self.assertAlmostEqual(ent["tether"]["length_m"], 90.0)

    def test_pos_and_vel_without_a_parent_stay_independent(self):
        ent = load_entity(
            id="jammer", kind="noise_jammer", pos=[1.0, 2.0, 3.0],
            vel=[4.0, 5.0, 6.0], life=dict(start_s=0.0))
        self.assertEqual(ent["link"], "independent")
        st = ecm.entity_state(ent, 2.0, tracks={})
        np.testing.assert_allclose(st["pos"], [9.0, 12.0, 15.0])

    def test_an_unknown_field_in_a_link_block_is_rejected(self):
        with self.assertRaises(ValueError):
            ecm.load(dict(entities=[dict(
                id="chaff", kind="chaff_cloud", parent="lead",
                release=dict(parent="lead", t_s=1.0, mount=[0.0, 0.0, 0.0]))]))

    def test_every_shipped_scene_still_plans_and_renders_a_world(self):
        root = os.path.join(BASE, "scenes")
        paths = sorted(glob.glob(os.path.join(root, "*.json")))
        self.assertEqual(len(paths), 32)
        for path in paths:
            with self.subTest(scene=os.path.basename(path)):
                cfg = scene.load_config(path)
                n = int((cfg.get("frames") or {}).get("n", 2))
                dt = float((cfg.get("frames") or {}).get("dt", 0.2))
                rows = plan.summary(cfg, n, cfg.get("seed", 0), dt)
                self.assertTrue(rows)
                state = next(plan.plan_frames(cfg, 1, cfg.get("seed", 0), dt))
                # scenes/template_object.json points at a model that is not in
                # the repository (it is the fill-in template); geometry cannot be
                # built from a missing mesh, so only planning is asserted there
                base = cfg.get("_base_dir", root)
                models = [o.get("model") for o in cfg.get("objects", []) if o.get("model")]
                if any(not os.path.exists(os.path.join(base, m)) for m in models):
                    continue
                world = scene.build_world(state["cfg"], jitter=state["jitter"])
                self.assertTrue(world.objs)


if __name__ == "__main__":
    unittest.main(verbosity=2)