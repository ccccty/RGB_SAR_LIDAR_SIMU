"""Cross-entity consistency checker: does it actually catch the failures?

The checker exists because a multi-target scene can pass every per-object test
and still be wrong.  So the important tests here are the NEGATIVE ones: a pair
5 m apart must make C1 and C3 fail, a 300 m formation inside a 1.2 deg seeker
must make C6 fail, a 180 m/s ejection must make E2 fail.  A test suite that only
feeds clean scenes would prove nothing about the detector.

Positive controls sit next to each negative one so a rule cannot be "passed" by
being permanently red: the clean two-ship, the lagged turn formation and the
properly released chaff all have to come out clean.
"""
import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from simkit import consistency, plan, scenegen  # noqa: E402

CLI = os.path.join(BASE, "tools", "check_consistency.py")


# ---------------------------------------------------------------------------
# fixtures: box geometry, no asset files, so the tests never depend on the
# model repository being complete
# ---------------------------------------------------------------------------
def fighter(name, pos, vel, **over):
    spec = dict(name=name, cls="fighter", shape="box", size=[12.0, 9.0, 3.0],
                pos=[float(x) for x in pos], vel=[float(x) for x in vel],
                yaw_mode="velocity", attitude_mode="coordinated")
    spec.update(over)
    return spec


def make_scene(objects, *, n=20, dt=0.1, trajectory=None, fov=6.0,
               cam_pos=(-8000.0, 0.0, 4000.0), look_at=(0.0, 0.0, 4000.0),
               ecm_entities=None, ground=False, seed=3):
    traj = trajectory or dict(mode="line", origin=[0.0, 0.0, 4000.0],
                              vel=[200.0, 0.0, 0.0])
    cfg = dict(
        name="consistency_fixture", seed=seed,
        ground=dict(enabled=bool(ground)),
        frames=dict(n=int(n), dt=float(dt)),
        objects=objects,
        clip=dict(frames_per_clip=int(n), camera="ir_camera",
                  trajectory=traj, rig=dict(mode="track", lag_s=0.0)),
        sensors={"ir_camera": dict(pos=[float(x) for x in cam_pos],
                                   look_at=[float(x) for x in look_at],
                                   fov_deg=float(fov), width=960, height=720)},
    )
    if ecm_entities is not None:
        cfg["ecm"] = dict(enabled=True, entities=ecm_entities)
    return cfg


def formation_scene(gap, *, fov=6.0, n=20, wing_traj=None, formation=None,
                    lead=None, wing_extra=None):
    lead = lead or fighter("lead", [0.0, 0.0, 4000.0], [200.0, 0.0, 0.0])
    wing = fighter("wing", [0.0, float(gap), 4000.0], [200.0, 0.0, 0.0],
                   link="formation",
                   formation=formation or dict(parent="lead", mode="lag",
                                               tau_lag_s=0.3))
    if wing_traj is not None:
        wing["trajectory"] = wing_traj
    if wing_extra:
        wing.update(wing_extra)
    traj = (lead.get("trajectory") if isinstance(lead, dict)
            else None) or dict(mode="line", origin=[0.0, 0.0, 4000.0],
                               vel=[200.0, 0.0, 0.0])
    return make_scene([lead, wing], n=n, fov=fov, trajectory=traj,
                      cam_pos=(-8000.0, 0.0, 4000.0),
                      look_at=(0.0, 0.0, 4000.0))


ARC = dict(mode="arc", centre=[0.0, 0.0, 4000.0], radius_m=2000.0,
           speed_mps=200.0, phase=0.0)


def arc_formation_scene(*, wing_traj=None, n=30):
    lead = fighter("lead", [2000.0, 0.0, 4000.0], [0.0, 200.0, 0.0])
    wing = fighter("wing", [2000.0, 100.0, 4000.0], [0.0, 200.0, 0.0],
                   link="formation",
                   formation=dict(parent="lead", mode="lag", tau_lag_s=0.3))
    if wing_traj is not None:
        wing["trajectory"] = wing_traj
    return make_scene([lead, wing], n=n, fov=10.0, trajectory=dict(ARC),
                      cam_pos=(-6000.0, 0.0, 4000.0),
                      look_at=(2000.0, 0.0, 4000.0))


def release_scene(*, eject=(0.0, 6.0, -1.0), tau=0.25, drag=None, n=30,
                  parent="lead", kind="chaff_cloud", lead_traj=None,
                  lead_pos=(0.0, 0.0, 4000.0), lead_vel=(200.0, 0.0, 0.0),
                  fall=1.5, wind=(5.0, 0.0, 0.0)):
    lead = fighter("lead", lead_pos, lead_vel)
    ent = dict(id="chaff_1", kind=kind, life=dict(start_s=0.0),
               release=dict(parent=parent, t_s=0.5, mount_body=[0.0, 0.0, -2.0],
                            eject_body=list(eject),
                            drag=drag or dict(mode="lag", tau_s=tau),
                            air=dict(wind_mps=list(wind), fall_mps=float(fall))))
    return make_scene([lead], n=n, ecm_entities=[ent],
                      trajectory=lead_traj or dict(
                          mode="line", origin=[0.0, 0.0, 4000.0],
                          vel=[200.0, 0.0, 0.0]))


def rows(result):
    return {c["id"]: c for c in result["checks"]}


def check(result, cid):
    return rows(result)[cid]


def _manifest_patch(mutator):
    """Patch the optical manifest writer with a deliberately wrong one.

    The F4/A1 rules must be able to catch the historical failure (a frozen or
    displaced optical channel), so these tests inject that failure at the exact
    contract the checker reads.  They do not depend on how the renderer is wired
    today.
    """
    real = consistency.external.write_manifest

    def fake(cfg, states, out_dir, part=None):
        path = real(cfg, states, out_dir, part=part)
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        mutator(data)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False)
        return path

    return mock.patch.object(consistency.external, "write_manifest", fake)


# ---------------------------------------------------------------------------
# contract and positive controls
# ---------------------------------------------------------------------------
class ContractTests(unittest.TestCase):
    def test_every_catalogue_id_appears_exactly_once(self):
        result = consistency.inspect(formation_scene(100.0), frames=3)
        ids = [c["id"] for c in result["checks"]]
        self.assertEqual(ids, consistency.CHECK_IDS)
        self.assertEqual(len(ids), len(set(ids)))
        for c in result["checks"]:
            for key in ("id", "group", "ok", "detail", "value", "limit"):
                self.assertIn(key, c)
            self.assertIn(c["group"], "ABCEFG")
            self.assertIsInstance(c["detail"], str)
        self.assertEqual(result["summary"]["checks"], len(consistency.CHECK_IDS))
        self.assertIn("ok", result)

    def test_clean_formation_passes(self):
        result = consistency.inspect(formation_scene(100.0), frames=6)
        self.assertTrue(result["ok"], result["summary"]["failed"])
        self.assertTrue(check(result, "C1")["ok"])
        self.assertTrue(check(result, "C2")["ok"])
        self.assertTrue(check(result, "C3")["ok"])
        self.assertTrue(check(result, "C6")["ok"])
        # no turn: the lag bound is vacuous, and the report says so instead of
        # faking a pass
        self.assertFalse(check(result, "C5")["applicable"])

    def test_lagged_turn_formation_is_inside_the_bound(self):
        result = consistency.inspect(arc_formation_scene(), frames=10)
        c5 = check(result, "C5")
        self.assertTrue(result["ok"], result["summary"]["failed"])
        self.assertTrue(c5["applicable"])
        self.assertTrue(c5["ok"])
        # e = R*omega*tau = 100 m * 0.1 rad/s * 0.3 s = 3 m, bound = 6 m
        self.assertAlmostEqual(c5["value"], 3.0, delta=0.2)
        self.assertAlmostEqual(c5["limit"], 6.0, delta=0.2)
        self.assertTrue(check(result, "C4")["ok"])

    def test_frame_sampling_limits_the_sampled_set(self):
        result = consistency.inspect(formation_scene(100.0), frames=4)
        self.assertEqual(result["summary"]["sampled_frames"], 4)
        self.assertEqual(result["summary"]["frames"], 20)

    def test_input_config_is_not_mutated(self):
        cfg = formation_scene(100.0)
        before = copy.deepcopy(cfg)
        consistency.inspect(cfg, frames=2)
        self.assertEqual(cfg, before)

    def test_chunk_invariance_across_three_clips(self):
        cfg = formation_scene(100.0, n=25)
        cfg["clip"]["frames_per_clip"] = 10
        result = consistency.inspect(cfg, frames=5)
        a3 = check(result, "A3")
        self.assertTrue(a3["ok"], a3["detail"])
        self.assertIn("bit-identical", a3["detail"])

    def test_time_clock_must_advance(self):
        cfg = formation_scene(100.0, n=6)
        cfg["frames"]["dt"] = 0.0
        result = consistency.inspect(cfg, frames=3)
        a2 = check(result, "A2")
        self.assertFalse(a2["ok"])
        self.assertIn("clock does not advance", a2["detail"])


# ---------------------------------------------------------------------------
# C: the deliberately broken geometries
# ---------------------------------------------------------------------------
class GeometryViolationTests(unittest.TestCase):
    def test_five_metre_pair_fails_c1_and_c3(self):
        result = consistency.inspect(formation_scene(5.0), frames=2)
        c1, c3 = check(result, "C1"), check(result, "C3")
        self.assertFalse(c1["ok"])
        self.assertFalse(c3["ok"])
        self.assertIn("5.00", c1["detail"])
        self.assertIn("outside [50, 500]", c1["detail"])
        self.assertIn("5.00", c3["detail"])
        self.assertIn("r_i + r_j + 10", c3["detail"])
        self.assertFalse(result["ok"])

    def test_300_metres_at_8km_12deg_leaves_the_frame(self):
        # the doc's headline case: 8 km range and 1.2 deg give an 83.8 m half
        # width, so a 300 m formation is out of frame even though it is inside
        # the 50-500 m formation box.
        result = consistency.inspect(formation_scene(300.0, fov=1.2), frames=2)
        c1, c6 = check(result, "C1"), check(result, "C6")
        self.assertTrue(c1["ok"], c1["detail"])
        self.assertFalse(c6["ok"])
        self.assertIn("fill 3.5", c6["detail"])
        self.assertAlmostEqual(c6["limit"], 1.0)
        self.assertGreater(c6["value"], 3.0)

    def test_wide_fov_keeps_the_same_300m_formation_in_frame(self):
        result = consistency.inspect(formation_scene(300.0, fov=6.0), frames=2)
        self.assertTrue(check(result, "C6")["ok"], check(result, "C6")["detail"])

    def test_straight_wing_breaks_the_turn_lag(self):
        line = dict(mode="line", origin=[2000.0, 100.0, 4000.0],
                    vel=[0.0, 200.0, 0.0])
        result = consistency.inspect(arc_formation_scene(wing_traj=line), frames=10)
        c4, c5 = check(result, "C4"), check(result, "C5")
        self.assertFalse(c4["ok"])
        self.assertFalse(c5["ok"])
        self.assertIn("formation error", c5["detail"])
        self.assertGreater(c5["value"], c5["limit"])

    def test_five_metre_pair_is_detected_with_a_wide_view_too(self):
        # C6 must not be the thing that "catches" C1/C3: with a wide lens the
        # pair is well inside the frame, so only the spacing rules fire.
        result = consistency.inspect(formation_scene(5.0, fov=20.0), frames=2)
        self.assertFalse(check(result, "C1")["ok"])
        self.assertFalse(check(result, "C3")["ok"])
        self.assertTrue(check(result, "C6")["ok"])


# ---------------------------------------------------------------------------
# E: dispensed-body timing
# ---------------------------------------------------------------------------
class ReleaseViolationTests(unittest.TestCase):
    def test_proper_release_passes_all_seven(self):
        result = consistency.inspect(release_scene(), frames=8)
        self.assertTrue(result["ok"], result["summary"]["failed"])
        for cid in ("E1", "E2", "E3", "E4", "E5", "E6", "E7"):
            self.assertTrue(check(result, cid)["applicable"], cid)
            self.assertTrue(check(result, cid)["ok"], check(result, cid)["detail"])

    def test_ejection_above_sixty_metres_per_second_is_caught(self):
        result = consistency.inspect(release_scene(eject=(180.0, 0.0, 0.0)),
                                     frames=8)
        e2 = check(result, "E2")
        self.assertFalse(e2["ok"])
        self.assertAlmostEqual(e2["value"], 180.0, delta=0.1)
        self.assertAlmostEqual(e2["limit"], 60.0)
        self.assertIn("180.00", e2["detail"])

    def test_release_without_a_parent_is_caught(self):
        result = consistency.inspect(release_scene(parent="nobody"), frames=4)
        e1 = check(result, "E1")
        self.assertFalse(e1["ok"])
        self.assertIn("not a scene object", e1["detail"])

    def test_settling_rule_catches_a_cloud_that_keeps_rising(self):
        # E6 is a relation between the published v_z and the declared fall
        # speed.  Inject a cloud that still rises at +3 m/s while the air mass
        # says -1.5 m/s: the rule must report |v_z + fall| / fall > 1.
        cfg = release_scene()
        real = consistency._entity_state

        def shifted(ent, t, cp):
            state = dict(real(ent, t, cp))
            state["vel"] = np.asarray(state["vel"], float) + np.array([0.0, 0.0, 3.0])
            return state

        with mock.patch.object(consistency, "_entity_state", shifted):
            result = consistency.inspect(cfg, frames=8)
        e6 = check(result, "E6")
        self.assertFalse(e6["ok"])
        self.assertGreater(e6["value"], 1.0)


# ---------------------------------------------------------------------------
# F: sensor-channel consistency
# ---------------------------------------------------------------------------
class SensorViolationTests(unittest.TestCase):
    def test_displaced_optical_pose_is_caught(self):
        cfg = release_scene()
        result_clean = consistency.inspect(cfg, frames=4)
        self.assertTrue(result_clean["ok"], result_clean["summary"]["failed"])

        def shift(data):
            for fr in data["frames"]:
                for pose in (fr.get("objects") or {}).values():
                    pose["pos"][0] += 5.0

        with _manifest_patch(shift):
            result = consistency.inspect(cfg, frames=4)
        f4 = check(result, "F4")
        self.assertFalse(f4["ok"])
        self.assertGreater(f4["value"], 4.9)
        self.assertIn("5.", f4["detail"])

    def test_frozen_optical_channel_is_caught(self):
        cfg = release_scene(n=12)

        def freeze(data):
            frames = data["frames"]
            if not frames:
                return
            first = frames[0].get("objects") or {}
            for fr in frames[1:]:
                for eid, pose in (fr.get("objects") or {}).items():
                    if eid in first:
                        pose["pos"] = list(first[eid]["pos"])

        with _manifest_patch(freeze):
            result = consistency.inspect(cfg, frames=4)
        a1 = check(result, "A1")
        self.assertFalse(a1["ok"])
        self.assertIn("frozen", a1["detail"])


# ---------------------------------------------------------------------------
# G: labels
# ---------------------------------------------------------------------------
class LabelViolationTests(unittest.TestCase):
    def test_dispensable_id_colliding_with_a_target_name_is_caught(self):
        cfg = make_scene([fighter("chaff_01", [0, 0, 4000], [200, 0, 0])], n=8)
        cfg["ecm"] = dict(enabled=True, entities=[dict(
            id="chaff_01", kind="chaff_cloud", pos=[-30.0, 10.0, 3990.0],
            vel=[190.0, 0.0, 0.0], life=dict(start_s=0.0))])
        result = consistency.inspect(cfg, frames=2)
        g1 = check(result, "G1")
        self.assertFalse(g1["ok"])
        self.assertIn("collides", g1["detail"])

    def test_dispensable_class_in_coco_categories_is_caught(self):
        cfg = make_scene([fighter("lead", [0, 0, 4000], [200, 0, 0],
                                  cls="ecm")], n=8)
        cfg["ecm"] = dict(enabled=True, entities=[dict(
            id="chaff_01", kind="chaff_cloud", pos=[-30.0, 10.0, 3990.0],
            vel=[190.0, 0.0, 0.0], life=dict(start_s=0.0))])
        result = consistency.inspect(cfg, frames=2)
        g2 = check(result, "G2")
        self.assertFalse(g2["ok"])
        self.assertIn("'ecm'", g2["detail"])


# ---------------------------------------------------------------------------
# B is borrowed, not re-implemented
# ---------------------------------------------------------------------------
class BorrowedEnvelopeTests(unittest.TestCase):
    def test_motion_diagnose_is_imported(self):
        md = consistency._load_motion_diagnose()
        self.assertTrue(callable(md.diagnose_scene))

    def test_speed_above_the_declared_kinematics_is_caught_by_b1(self):
        cfg = make_scene([fighter("lead", [0, 0, 4000], [500, 0, 0],
                                  kinematics=dict(max_speed_mps=300.0))], n=8)
        # the clip trajectory is what actually flies the primary target
        cfg["clip"]["trajectory"]["vel"] = [500.0, 0.0, 0.0]
        result = consistency.inspect(cfg, frames=3)
        b1 = check(result, "B1")
        self.assertFalse(b1["ok"])
        self.assertIn("speed_max", b1["detail"])


# ---------------------------------------------------------------------------
# recipe input and the CLI wrapper
# ---------------------------------------------------------------------------
class RecipeAndCliTests(unittest.TestCase):
    def test_recipe_is_built_and_inspected(self):
        recipe = scenegen.default_recipe()
        recipe["capture"]["frames"] = 4
        recipe["capture"]["fps"] = 10.0
        result = consistency.inspect(recipe)
        self.assertEqual(result["summary"]["mode"], "seeker")
        self.assertEqual(result["summary"]["frames"], 4)
        self.assertEqual([c["id"] for c in result["checks"]],
                         consistency.CHECK_IDS)

    def _run_cli(self, cfg, frames="2"):
        tmp = tempfile.mkdtemp(prefix="consistency_cli_")
        try:
            path = os.path.join(tmp, "scene.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(cfg, fh, ensure_ascii=False)
            proc = subprocess.run(
                [sys.executable, CLI, path, "--frames", frames],
                cwd=BASE, capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=300)
        finally:
            for name in os.listdir(tmp):
                os.remove(os.path.join(tmp, name))
            os.rmdir(tmp)
        return proc

    def test_cli_exits_nonzero_on_a_violation(self):
        proc = self._run_cli(formation_scene(5.0, n=4))
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        self.assertIn("C1", proc.stdout)
        self.assertIn("FAIL", proc.stdout)

    def test_cli_exits_zero_on_a_clean_scene(self):
        proc = self._run_cli(formation_scene(100.0, n=4))
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("1 scene(s)", proc.stdout)




# ---------------------------------------------------------------------------
# the clip-start boundary: a centred difference cannot be taken off the edge
# ---------------------------------------------------------------------------
WAYPOINT_ROUTE = [[0.0, 0.0, 4000.0], [8.0, 2.4, 4000.0],
                  [14.0, 5.5, 4000.0], [20.0, 10.2, 4000.0]]


def waypoint_scene(n=20, speed=80.0):
    """A drawn route: the planner freezes the position before t=0.

    The object has to fly its own waypoints, so the clip must not carry a
    trajectory of its own: a clip trajectory is what flies the primary target,
    and a straight line extends backwards analytically (motion.Line has no
    history problem, which is the control case in this class)."""
    lead = fighter("lead", WAYPOINT_ROUTE[0], [speed, 0.0, 0.0],
                   trajectory=dict(mode="waypoints", points=WAYPOINT_ROUTE,
                                   speed_mps=float(speed), loop=False,
                                   ramp_in=False))
    cfg = make_scene([lead], n=n)
    cfg["clip"].pop("trajectory", None)
    return cfg


class ClipStartBoundaryTests(unittest.TestCase):
    def test_centred_difference_at_frame_zero_reports_half_the_speed(self):
        """Prove the artefact the checker must not report as a motion defect.

        motion.Waypoints freezes the position for t <= 0 (an open drawn route
        has no history before its departure point) while it publishes the
        right-hand velocity there.  A centred step across that edge therefore
        differences against a frozen pose and returns exactly half the
        velocity -- the number this test pins down."""
        cfg = waypoint_scene()
        dt = float(cfg["frames"]["dt"])
        cp = plan.ClipPlan(cfg, 0, int(cfg["clip"]["frames_per_clip"]), dt,
                           int(cfg["seed"]))
        obj = cfg["objects"][0]
        h = 1e-3

        def p(t):
            return np.asarray(cp.object_state(obj, t)[0], float)

        v0 = np.asarray(cp.object_state(obj, 0.0)[1], float)
        self.assertGreater(float(np.linalg.norm(v0)), 70.0)
        self.assertLess(float(np.linalg.norm(p(-h) - p(0.0))), 1e-9)
        central = float(np.linalg.norm(v0 - (p(h) - p(-h)) / (2.0 * h)))
        one_sided = float(np.linalg.norm(
            v0 - (-3.0 * p(0.0) + 4.0 * p(h) - p(2.0 * h)) / (2.0 * h)))
        self.assertAlmostEqual(central, 0.5 * float(np.linalg.norm(v0)),
                               delta=0.2)
        self.assertLess(one_sided, 0.01)

    def test_frame_zero_is_not_reported_as_a_derivative_error(self):
        result = consistency.inspect(waypoint_scene(), frames=1)
        b3, f3 = check(result, "B3"), check(result, "F3")
        self.assertTrue(b3["ok"], b3["detail"])
        self.assertTrue(f3["ok"], f3["detail"])
        self.assertLess(b3["value"], 0.05)

    def test_a_velocity_that_is_not_the_path_derivative_is_still_caught(self):
        """The boundary fix must not blunt the rule it was applied to."""
        cfg = waypoint_scene()
        real = plan.ClipPlan.object_state

        def shifted(self, obj, t):
            pos, vel, yaw = real(self, obj, t)
            return pos, np.asarray(vel, float) + np.array([4.0, 0.0, 0.0]), yaw

        with mock.patch.object(plan.ClipPlan, "object_state", shifted):
            result = consistency.inspect(cfg, frames=3)
        b3 = check(result, "B3")
        self.assertFalse(b3["ok"])
        self.assertGreater(b3["value"], 3.5)

    def test_a_line_route_keeps_the_centred_step_and_still_passes(self):
        cfg = make_scene([fighter("lead", [0.0, 0.0, 4000.0],
                                   [200.0, 0.0, 0.0])], n=8)
        result = consistency.inspect(cfg, frames=1)
        self.assertTrue(check(result, "F3")["ok"],
                        check(result, "F3")["detail"])
        self.assertTrue(check(result, "B3")["ok"],
                        check(result, "B3")["detail"])


# ---------------------------------------------------------------------------
# F5: the aperture the renderer actually uses, not the one the file claims
# ---------------------------------------------------------------------------
def radar_scene(*, radar_look_at=(0.0, 0.0, 0.0), gap=0.0, n=8):
    """Two parked-in-the-file targets, at the centre of the imaging lens."""
    lead = fighter("lead", [0.0, 0.0, 4000.0], [0.0, 200.0, 0.0])
    objs = [lead]
    if gap:
        objs.append(fighter("wing", [0.0, float(gap), 4000.0],
                            [0.0, 200.0, 0.0]))
    cfg = make_scene(objs, n=n, trajectory=dict(mode="line",
                                                origin=[0.0, 0.0, 4000.0],
                                                vel=[0.0, 200.0, 0.0]))
    cfg["sensors"]["radar"] = dict(
        pos=[-8000.0, 0.0, 4000.0], look_at=[float(x) for x in radar_look_at],
        up=[0.0, 0.0, 1.0], az_span_deg=2.0, el_span_deg=1.6,
        n_az=40, n_el=20, beamwidth_el_deg=6.0)
    return cfg


class RenderAimTests(unittest.TestCase):
    def test_the_report_always_states_both_layers(self):
        f5 = check(consistency.inspect(radar_scene(), frames=1), "F5")
        self.assertIn("render layer", f5["detail"])
        self.assertIn("config layer", f5["detail"])

    def test_render_layer_covers_targets_the_authored_pose_misses(self):
        cfg = radar_scene(radar_look_at=(0.0, 0.0, 0.0))
        report = consistency.inspect(cfg, frames=2)
        f5 = check(report, "F5")
        self.assertTrue(f5["applicable"])
        self.assertTrue(f5["ok"], f5["detail"])
        self.assertEqual(report["summary"]["aim_layer"], "render")
        # the authored pose really does miss: the target sits 26.6 deg above
        # the origin-looking boresight, the render aim points straight at it
        config_only = consistency.inspect(cfg, frames=2, render_aim=False)
        f5c = check(config_only, "F5")
        self.assertFalse(f5c["ok"])
        self.assertIn("outside the radar aperture", f5c["detail"])
        self.assertEqual(config_only["summary"]["aim_layer"], "config")

    def test_render_aim_still_fails_when_the_formation_does_not_fit(self):
        # 300 m of lateral offset at 8 km is 2.15 deg, just past the 2.0 deg
        # half-aperture: re-aiming at the first target cannot save the second
        cfg = radar_scene(gap=300.0)
        f5 = check(consistency.inspect(cfg, frames=2), "F5")
        self.assertFalse(f5["ok"])
        self.assertIn("wing", f5["detail"])

    def test_aim_note_surfaces_the_layer_disagreement(self):
        cfg = radar_scene(radar_look_at=(0.0, 0.0, 0.0))
        report = consistency.inspect(cfg, frames=2)
        notes = " ".join(report["summary"]["notes"])
        self.assertIn("aim layers disagree", notes)

if __name__ == "__main__":
    unittest.main(verbosity=2)
