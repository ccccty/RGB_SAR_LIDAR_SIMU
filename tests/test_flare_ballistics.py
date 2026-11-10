"""Flare ballistics: mount on the airframe, then a ballistic arc.

Three contracts, in the order a viewer notices them:

1. at its release instant a round sits ON the airframe (metres, not tens of
   metres behind it);
2. after release the round falls behind on a vertical arc -- z strictly
   descending, vertical speed converging to the drag terminal speed
   sqrt(g*L) that the E6 settling rule checks;
3. the forward glide is bounded by the drag time constant, so the aircraft
   pulls away and the round drops behind instead of cruising alongside.
"""
import json
import math
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from simkit import ecm, plan as planlib, scenegen  # noqa: E402

G = 9.81
RECIPE = os.path.join(ROOT, "configs", "recipes", "p_air_skyclear_passive.json")


def _run(seconds=6.0, dt=0.1):
    recipe = json.load(open(RECIPE, encoding="utf-8"))
    n = int(round(seconds / dt))
    # frame count BEFORE build_scene: the seeker band is sized from
    # capture.frames * dt, exactly as the dataset builder does
    recipe.setdefault("capture", {})["frames"] = n
    cfg = scenegen.build_scene(recipe)
    cfg["frames"]["n"] = n
    cfg["clip"]["frames_per_clip"] = n
    cfg["capture"]["frames"] = n
    prim = cfg["objects"][0]["name"]
    states = list(planlib.plan_frames(cfg, n, int(cfg.get("seed") or 0), dt))
    spec = ecm.load(cfg.get("ecm"))
    ent_at = []
    for st in states:
        anchors = {nm: (np.asarray(ps["pos"], float),
                        np.asarray(ps.get("velocity_mps") or [0.0, 0.0, 0.0], float))
                   for nm, ps in (st.get("poses") or {}).items()}
        ent_at.append({e["id"]: e for e in ecm.entities_at(
            spec, st["t"], anchors=anchors, tracks=ecm.tracks_of(st.get("cfg")))})
    flare_ids = [k for k, e in ent_at[0].items()
                 if e.get("kind") == "infrared_flare"]
    release_s = {fid: float(f["release"]["t_s"])
                 for fid, f in ((fid, [x for x in spec["entities"]
                                      if x["id"] == fid][0])
                                for fid in flare_ids)}
    return states, ent_at, prim, flare_ids, release_s


class FlareBallisticsTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        (cls.states, cls.ent_at, cls.prim,
         cls.flare_ids, cls.release_s) = _run()

    def test_rounds_start_on_the_airframe(self):
        # At each round's own release instant the mount is a wing-root
        # dispenser: metres from the aircraft, not 45 m behind it on an
        # already-detached glide path.
        worst = 0.0
        for fid in self.flare_ids:
            i_rel = int(round(self.release_s[fid] / 0.1))
            pp = np.asarray(self.states[i_rel]["poses"][self.prim]["pos"], float)
            fp = np.asarray(self.ent_at[i_rel][fid]["pos"], float)
            worst = max(worst, float(np.linalg.norm(pp - fp)))
        # 15 m is the wing root of a ~17 m aircraft: a dispenser sits on the
        # airframe, whereas the old default put rounds 45-65 m aft / 25-37 m
        # below -- already a full airframe away at birth.
        self.assertLess(worst, 15.0,
                        "released rounds must sit on the airframe: %.1f m" % worst)

    def test_height_strictly_descends_after_release(self):
        for fid in self.flare_ids:
            i_rel = int(round(self.release_s[fid] / 0.1))
            z = [self.ent_at[i][fid]["pos"][2]
                 for i in range(i_rel, len(self.states))]
            drops = [b - a for a, b in zip(z, z[1:])]
            self.assertTrue(all(d < 0.0 for d in drops),
                            "%s must descend monotonically once free: %s"
                            % (fid, drops[:6]))

    def test_vertical_speed_converges_to_the_terminal_speed(self):
        for fid in self.flare_ids:
            i_rel = int(round(self.release_s[fid] / 0.1))
            vt = math.sqrt(G * 60.0)        # drag default L_m = 60 for flares
            tail = [self.ent_at[i][fid]["vel"][2]
                    for i in range(max(i_rel + 20, len(self.states) - 8),
                                   len(self.states))]
            mean_vz = float(np.mean(tail))
            self.assertLess(mean_vz, 0.0)
            self.assertLess(abs(abs(mean_vz) - vt) / vt, 0.20,
                            "%s v_z must converge to the terminal speed %.1f: %s"
                            % (fid, vt, tail))

    def test_forward_glide_is_bounded_so_the_aircraft_pulls_away(self):
        # The drag time constant sheds the airframe's speed: the round covers a
        # bounded forward distance and then simply falls, while the aircraft
        # keeps flying -- the separation must grow well past the release
        # distance instead of the fan cruising in formation.
        n0 = self.flare_ids[0]
        first = np.asarray(self.ent_at[0][n0]["pos"], float)
        last = np.asarray(self.ent_at[-1][n0]["pos"], float)
        p_first = np.asarray(self.states[0]["poses"][self.prim]["pos"], float)
        p_last = np.asarray(self.states[-1]["poses"][self.prim]["pos"], float)
        sep0 = float(np.linalg.norm(first - p_first))
        sepN = float(np.linalg.norm(last - p_last))
        self.assertGreater(sepN, 3.0 * sep0,
                           "aircraft must pull away from its flares")
        # once free, the round's own horizontal speed must be drag-shed to
        # near zero: it falls, it does not cruise
        tail_h = [float(np.hypot(self.ent_at[i][n0]["vel"][0],
                                 self.ent_at[i][n0]["vel"][1]))
                  for i in range(len(self.states) - 8, len(self.states))]
        self.assertLess(max(tail_h), 8.0,
                        "forward speed must be shed by drag: %s" % tail_h)


if __name__ == "__main__":
    unittest.main()
