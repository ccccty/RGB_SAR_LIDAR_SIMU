"""The recipe -> scene generator has to produce scenes the pipeline can eat.

A generator that only validates its own recipe would be theatre: the acceptance
bar is that scene.load_config + scene.build_world succeed on the emitted file,
that the three environment states produce three different worlds, that the
aspect rule survives into the seeker plan, and that the ECM defaults keep the
calibration that was actually measured instead of a token jammer.

Every test here is a contract with a consumer:
  * scene.py      -- load_config / build_world accept the dict as written;
  * plan.py       -- clip.seeker turns head_on into closing and pursuit into
                     opening, with the target's nose toward the camera in the
                     first case and its tail in the second;
  * ecm.py        -- the entity block passes load() and carries 200 W / Ku /
                     barrage / 500 m, not a few watts;
  * studio.py     -- validate_recipe / build_scene / list_assets are callable
                     from the web layer with no extra arguments.
"""
import contextlib
import json
import math
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simkit import ecm, plan, scene, scenegen  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def has_chinese(text):
    return any(0x4e00 <= ord(ch) <= 0x9fff for ch in str(text))


@contextlib.contextmanager
def written(cfg):
    """Materialise a built scene one level below the repo root.

    That depth is not cosmetic: build_scene writes "../assets/models/..." into
    each object, exactly like the curated files in scenes/, so the config has to
    be loaded from a directory one level down or the model paths cannot resolve.
    studio.py writes its generated scenes into _tmp/, which has the same depth.
    """
    directory = tempfile.mkdtemp(prefix='.scenegen_test_', dir=BASE)
    try:
        path = os.path.join(directory, '%s.json' % (cfg.get('name') or 'scene'))
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(cfg, fh, ensure_ascii=False)
        yield path
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def recipe(env='sky', **over):
    r = scenegen.default_recipe()
    r['name'] = 'test_' + env
    r['environment']['kind'] = env
    r.update(over)
    return r


def world_of(cfg):
    with written(cfg) as path:
        return scene.build_world(scene.load_config(path))


class RecipeValidationTests(unittest.TestCase):
    def test_default_recipe_is_valid_and_complete(self):
        r = scenegen.default_recipe()
        norm = scenegen.validate_recipe(r)
        self.assertEqual(norm['name'], 'demo')
        self.assertEqual(norm['environment']['kind'], 'sky')
        self.assertEqual(norm['environment']['weather'], 'clear')
        self.assertEqual(norm['view']['aspect'], 'head_on')
        self.assertEqual(norm['capture']['sensors'], ['optical', 'ir', 'radar'])
        self.assertTrue(norm['objects'][0]['asset'].endswith('.obj'))

    def test_default_recipe_is_a_fresh_dict(self):
        first = scenegen.default_recipe()
        first['objects'][0]['asset'] = 'nothing.obj'
        first['capture']['frames'] = 999
        self.assertNotEqual(scenegen.default_recipe()['objects'][0]['asset'],
                            'nothing.obj')
        self.assertEqual(scenegen.default_recipe()['capture']['frames'], 20)

    def test_validate_does_not_mutate_the_argument(self):
        r = scenegen.default_recipe()
        before = json.dumps(r, sort_keys=True, ensure_ascii=False)
        scenegen.validate_recipe(r)
        self.assertEqual(json.dumps(r, sort_keys=True, ensure_ascii=False), before)
        first = scenegen.validate_recipe(r)['objects']
        first.append({'asset': 'junk'})
        self.assertEqual(len(scenegen.validate_recipe(r)['objects']), 1)

    def test_form_strings_are_coerced_to_numbers(self):
        r = scenegen.default_recipe()
        r['view']['range_km'] = '6.5'
        r['capture']['fps'] = '5'
        r['capture']['frames'] = '12'
        r['capture']['seed'] = '9'
        norm = scenegen.validate_recipe(r)
        self.assertEqual(norm['view']['range_km'], 6.5)
        self.assertEqual(norm['capture']['fps'], 5.0)
        self.assertEqual(norm['capture']['frames'], 12)
        self.assertEqual(norm['capture']['seed'], 9)

    def test_missing_blocks_fall_back_to_defaults(self):
        norm = scenegen.validate_recipe({
            'objects': [{'asset': '军用飞机/双发战斗机.obj'}],
            'capture': {'sensors': ['ir']}})
        self.assertEqual(norm['environment']['kind'], 'sky')
        # The default is "auto": the environment picks the engagement (air is
        # met head-on or chased, ground and sea are looked down on). Pinning
        # head_on here would have made every surface scene a grazing shot.
        self.assertEqual(norm['view']['aspect'], 'auto')
        self.assertEqual(norm['capture']['frames'], 20)
        self.assertEqual(norm['ecm']['mode'], 'none')
        self.assertEqual(norm['objects'][0]['cls'], 'fighter')
        self.assertEqual(norm['objects'][0]['count'], 1)

    def test_auto_aspect_resolves_per_environment(self):
        """auto must pick an air engagement for sky and a look-down for surface.

        This is the contract the workbench relies on: one "auto" value so an
        operator does not have to know that a tank cannot be met head-on.
        """
        for env, cls, asset, want_dep in (
                ('sky', 'fighter', '军用飞机/双发战斗机.obj', False),
                ('ground', 'vehicle', '军用地面车辆/主战坦克.obj', True),
                ('sea', 'ship', '军舰/导弹艇.obj', True)):
            r = scenegen.default_recipe()
            r['environment']['kind'] = env
            r['objects'] = [{'asset': asset, 'cls': cls, 'scale': 1.0, 'count': 1}]
            r['view'].update(aspect='auto', range_km=1.5)
            cfg = scenegen.build_scene(scenegen.validate_recipe(r))
            sk = cfg['clip']['seeker']
            if want_dep:
                self.assertEqual(sk['aspect'], 'look_down', env)
                self.assertGreater(float(sk.get('depression_deg') or 0), 10.0, env)
                cam = cfg['sensors']['optical']
                self.assertGreater(cam['pos'][2], 100.0, env)
            else:
                self.assertIn(sk['aspect'], ('head_on', 'pursuit'), env)
                self.assertNotIn('depression_deg', sk)

class PlacementTests(unittest.TestCase):
    """Clicking a pixel on a plate must put the object at that pixel.

    The mapping is the contract the interactive editor rests on: image u,v ->
    world x,y with the plate centred on the origin and no rotation. It was
    measured wrong once (a 4:3 frame covers span*3/4 vertically, not span), so
    the arithmetic is pinned here rather than re-derived by eye.
    """

    def _scene(self, uv, span=600.0):
        from simkit import scenegen
        r = scenegen.default_recipe()
        r['name'] = 'place_test'
        r['environment'].update(kind='ground', weather='clear')
        r['objects'] = [dict(asset='军用地面车辆/主战坦克.obj', cls='vehicle',
                             scale=1.0, count=1, uv=list(uv))]
        r['view'].update(aspect='top_down', ground_span_m=span, fov_deg=10.0)
        r['capture'].update(sensors=['optical'], frames=1)
        return scenegen.build_scene(scenegen.validate_recipe(r))

    def test_uv_maps_to_the_expected_world_xy(self):
        cfg = self._scene([0.35, 0.72], span=600.0)
        x, y, z = cfg['objects'][0]['pos']
        self.assertAlmostEqual(x, (0.35 - 0.5) * 600.0, places=1)
        self.assertAlmostEqual(y, (0.5 - 0.72) * 600.0 * 0.75, places=1)
        self.assertEqual(z, 0.0)

    def test_top_down_is_a_static_nadir_camera(self):
        cfg = self._scene([0.5, 0.5])
        self.assertNotIn('seeker', cfg['clip'])
        cam = cfg['sensors']['optical']
        self.assertAlmostEqual(cam['pos'][0], 0.0, places=3)
        self.assertAlmostEqual(cam['pos'][1], 0.0, places=3)
        self.assertGreater(cam['pos'][2], 100.0)
        self.assertEqual(cam.get('up'), [0.0, 1.0, 0.0])
        # No seeker means plan.ClipPlan.mode == "world", i.e. the rig holds the
        # authored pose instead of tracking anything.
        self.assertNotIn('seeker', cfg['clip'])

    def test_plate_is_aligned_so_the_click_is_reproducible(self):
        cfg = self._scene([0.5, 0.5])
        g = cfg['ground']
        self.assertTrue(g['backdrop_aligned'])
        self.assertEqual(g['backdrop_offset'], [0.5, 0.5])
        self.assertAlmostEqual(g['backdrop_span_y_m'], g['backdrop_span_m'] * 0.75, places=2)

class BackdropLibraryTests(unittest.TestCase):
    """The collected plate library and its nadir filter."""

    def test_library_has_the_five_categories(self):
        from simkit import scenegen
        lib = scenegen.backdrop_library()
        self.assertEqual(set(lib), set(scenegen.BACKDROP_CATEGORIES))
        for cat in scenegen.BACKDROP_CATEGORIES:
            self.assertTrue(lib[cat], cat)

    def test_top_down_prefers_a_near_nadir_plate(self):
        """An oblique plate under a straight-down camera reads wrong."""
        from simkit import scenegen
        meta = scenegen.backdrop_meta()
        aerial = [k for k, v in meta.items() if v.get("aerial")]
        if not aerial:
            self.skipTest("no plate is tagged aerial")
        for cat in ("city", "port"):
            files = [k for k in aerial if k.startswith(cat + "/")]
            if not files:
                continue
            picked = scenegen._pick_backdrop(cat, seed=0, prefer_aerial=True)
            rel = cat + "/" + os.path.basename(picked)
            self.assertIn(rel, aerial, rel)

    def test_a_chosen_file_beats_the_seed_rotation(self):
        """Picking a specific plate must survive any seed."""
        from simkit import scenegen
        lib = scenegen.backdrop_library()
        files = lib.get("sky") or []
        if len(files) < 2:
            self.skipTest("need at least two sky plates")
        want = files[1]
        for seed in (0, 3, 99):
            r = scenegen.default_recipe()
            r['environment'].update(kind='sky', weather='clear')
            r['environment']['background'] = {'mode': 'library', 'category': 'sky',
                                              'file': want}
            r['objects'] = [dict(asset='军用飞机/双发战斗机.obj', cls='fighter',
                                 scale=1.0, count=1)]
            r['view'].update(aspect='head_on', range_km=6.0, fov_deg=2.2)
            r['capture'].update(sensors=['optical'], frames=1, seed=seed)
            cfg = scenegen.build_scene(scenegen.validate_recipe(r))
            used = os.path.basename(cfg['optical'].get('sky_image') or '')
            self.assertEqual(used, want)

    def test_manual_placement_turns_the_plate_into_a_map(self):
        """top_down must centre the plate and disable the tiling tricks."""
        from simkit import scenegen
        r = scenegen.default_recipe()
        r['environment'].update(kind='ground', weather='clear',
                                background={'mode': 'library', 'category': 'forest'})
        r['objects'] = [dict(asset='军用地面车辆/主战坦克.obj', cls='vehicle',
                             scale=1.0, count=1, uv=[0.5, 0.5])]
        r['view'].update(aspect='top_down', ground_span_m=600.0, fov_deg=10.0)
        r['capture'].update(sensors=['optical'], frames=1)
        cfg = scenegen.build_scene(scenegen.validate_recipe(r))
        self.assertTrue(cfg['ground']['backdrop_aligned'])
        self.assertEqual(cfg['ground']['backdrop_offset'], [0.5, 0.5])

    def test_unknown_keys_are_dropped_not_carried_into_the_scene(self):
        r = scenegen.default_recipe()
        r['ui_only'] = {'slider': 3}
        r['view']['unused'] = 1
        norm = scenegen.validate_recipe(r)
        self.assertNotIn('ui_only', norm)
        self.assertNotIn('unused', norm['view'])

    def test_background_needs_its_payload(self):
        r = scenegen.default_recipe()
        r['environment']['background'] = {'mode': 'hdri', 'id': 'sky_4k'}
        self.assertEqual(
            scenegen.validate_recipe(r)['environment']['background'],
            {'mode': 'hdri', 'id': 'sky_4k'})
        r['environment']['background'] = {'mode': 'upload', 'path': 'u/bg.png'}
        self.assertEqual(
            scenegen.validate_recipe(r)['environment']['background']['path'],
            'u/bg.png')


class RecipeRejectionTests(unittest.TestCase):
    """Illegal recipes raise a Chinese ValueError that names the bad field."""

    def cases(self):
        out = []

        def add(field, mutate):
            out.append((field, mutate))

        def missing_objects(r):
            r['objects'] = []

        def bad_env(r):
            r['environment']['kind'] = 'underwater'

        def bad_range(r):
            r['view']['range_km'] = 500.0

        def no_sensors(r):
            r['capture']['sensors'] = []

        def bad_aspect(r):
            r['view']['aspect'] = 'side_on'

        def bad_weather(r):
            r['environment']['weather'] = 'snow'

        def bad_asset(r):
            r['objects'][0]['asset'] = '军用飞机/不存在的飞机.obj'

        def escape_asset(r):
            r['objects'][0]['asset'] = '../outside.obj'

        def bad_sensor(r):
            r['capture']['sensors'] = ['optical', 'sonar']

        def bad_frames(r):
            r['capture']['frames'] = 0

        def bad_fps(r):
            r['capture']['fps'] = 500.0

        def bad_seed(r):
            r['capture']['seed'] = -1

        def bad_scale(r):
            r['objects'][0]['scale'] = 0.0

        def bad_count(r):
            r['objects'][0]['count'] = 0

        def entities_without_mode(r):
            r['ecm'] = {'mode': 'none', 'entities': [
                {'id': 'x', 'kind': 'noise_jammer'}]}

        def bad_entity(r):
            r['ecm'] = {'mode': 'active',
                        'entities': [{'id': 'x', 'kind': 'laser_gun'}]}

        def hdri_without_id(r):
            r['environment']['background'] = {'mode': 'hdri'}

        for field, fn in (('objects', missing_objects),
                          ('environment.kind', bad_env),
                          ('view.range_km', bad_range),
                          ('capture.sensors', no_sensors),
                          ('view.aspect', bad_aspect),
                          ('environment.weather', bad_weather),
                          ('objects[0].asset', bad_asset),
                          ('objects[0].asset', escape_asset),
                          ('capture.sensors', bad_sensor),
                          ('capture.frames', bad_frames),
                          ('capture.fps', bad_fps),
                          ('capture.seed', bad_seed),
                          ('objects[0].scale', bad_scale),
                          ('objects[0].count', bad_count),
                          ('ecm.mode', entities_without_mode),
                          ('ecm.entities', bad_entity),
                          ('environment.background', hdri_without_id)):
            add(field, fn)
        return out

    def test_each_bad_field_raises_chinese_value_error(self):
        for field, mutate in self.cases():
            with self.subTest(field=field):
                r = scenegen.default_recipe()
                mutate(r)
                with self.assertRaises(ValueError) as caught:
                    scenegen.validate_recipe(r)
                msg = str(caught.exception)
                self.assertTrue(has_chinese(msg),
                                'message must be Chinese: ' + msg)
                self.assertIn(field, msg,
                              'message must name the field %r: %s' % (field, msg))

    def test_non_dict_recipe_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            scenegen.validate_recipe(['not', 'a', 'recipe'])
        self.assertIn('配方', str(caught.exception))

    def test_build_scene_rejects_the_same_recipes(self):
        r = scenegen.default_recipe()
        r['view']['range_km'] = 500.0
        with self.assertRaises(ValueError):
            scenegen.build_scene(r)


class SceneContractTests(unittest.TestCase):
    def test_required_top_level_keys(self):
        cfg = scenegen.build_scene(scenegen.default_recipe())
        for key in ('name', 'seed', 'frames', 'objects', 'ground', 'sensors',
                    'clip', 'weather', 'ir', 'optical', 'air', 'sun'):
            self.assertIn(key, cfg)
        self.assertNotIn('ecm', cfg)          # mode none writes no ecm block

    def test_frames_follow_fps(self):
        r = scenegen.default_recipe()
        r['capture']['fps'] = 20.0
        r['capture']['frames'] = 8
        cfg = scenegen.build_scene(r)
        self.assertEqual(cfg['frames'], {'n': 8, 'dt': 0.05})
        self.assertEqual(cfg['clip']['frames_per_clip'], 8)
        self.assertEqual(plan.frames_per_clip(cfg, 8), 8)

    def test_objects_carry_every_required_field(self):
        cfg = scenegen.build_scene(scenegen.default_recipe())
        self.assertTrue(cfg['objects'])
        names = [o['name'] for o in cfg['objects']]
        self.assertEqual(len(names), len(set(names)))
        for obj in cfg['objects']:
            for key in ('name', 'cls', 'model', 'shape', 'scale', 'pos', 'rot',
                        'rcs_db', 'temp_c', 'mat'):
                self.assertIn(key, obj)
            self.assertEqual(obj['shape'], 'obj')
            self.assertTrue(obj['model'].startswith('../assets/models/'),
                            obj['model'])
            self.assertEqual(len(obj['pos']), 3)
            self.assertIsInstance(obj['rcs_db'], float)
            self.assertIsInstance(obj['temp_c'], float)

    def test_count_expands_into_a_formation(self):
        r = scenegen.default_recipe()
        r['objects'][0]['count'] = 3
        cfg = scenegen.build_scene(r)
        self.assertEqual(len(cfg['objects']), 3)
        self.assertEqual(len({o['name'] for o in cfg['objects']}), 3)
        z = [o['pos'][2] for o in cfg['objects']]
        self.assertEqual(len(set(z)), 1)
        lateral = [o['pos'][1] for o in cfg['objects']]
        self.assertEqual(len(set(lateral)), 3, 'objects must not sit on top')

    def test_scene_is_plain_json(self):
        cfg = scenegen.build_scene(scenegen.default_recipe())
        text = json.dumps(cfg, ensure_ascii=False)
        self.assertEqual(json.loads(text)['name'], cfg['name'])
        self.assertNotIn('numpy', text.lower())

    def test_all_five_sensor_blocks_always_exist(self):
        cfg = scenegen.build_scene(scenegen.default_recipe())
        for key in ('optical', 'ir_camera', 'radar', 'lidar', 'sar'):
            self.assertIn(key, cfg['sensors'])
        self.assertIn('mwir', cfg['ir']['bands'])
        self.assertEqual(cfg['sensors']['radar']['band'], 'Ku')
        # the clip camera must be a real sensor block or plan.py cannot read it
        self.assertIn(cfg['clip']['camera'], cfg['sensors'])
        # lidar must at least reach the target it is pointed at
        self.assertGreaterEqual(cfg['sensors']['lidar']['max_range_m'], 8000.0)
        anchor = cfg['sensors']['sar']['anchor']
        self.assertAlmostEqual(anchor[2], cfg['objects'][0]['pos'][2], places=3)

    def test_dual_view_adds_a_wider_lens(self):
        r = scenegen.default_recipe()
        r['view']['views'] = 'dual'
        cfg = scenegen.build_scene(r)
        views = cfg['clip']['views']
        self.assertEqual(len(views), 2)
        self.assertGreater(views[1]['fov_deg'], views[0]['fov_deg'])

    def test_observation_geometry_is_consistent(self):
        cfg = scenegen.build_scene(scenegen.default_recipe())
        cam = cfg['sensors']['optical']['pos']
        target = cfg['objects'][0]['pos']
        slant = math.dist(cam, target)
        # build_scene places the camera at view.range_km, so the world at frame 0
        # already sits where the plan says it does
        self.assertAlmostEqual(slant, 8000.0, delta=5.0)
        self.assertEqual(cfg['sensors']['radar']['look_at'],
                         cfg['sensors']['optical']['look_at'])


class EnvironmentTests(unittest.TestCase):
    def test_three_environments_build_three_different_worlds(self):
        faces = {}
        for env in scenegen.ENV_KINDS:
            cfg = scenegen.build_scene(recipe(env))
            world = world_of(cfg)
            self.assertEqual(len(world.objs), 1)
            self.assertGreater(len(world.faces), 0)
            faces[env] = len(world.faces)
        self.assertEqual(len(set(faces.values())), 3,
                         'sky/ground/sea must not build the same world: %r' % faces)
        self.assertLess(faces['sky'], faces['ground'])
        self.assertLess(faces['ground'], faces['sea'])

    def test_sky_has_no_ground_block(self):
        cfg = scenegen.build_scene(recipe('sky'))
        self.assertEqual(cfg['ground'], {'enabled': False})

    def test_ground_is_a_plane_and_sea_is_a_wave_mesh(self):
        ground = scenegen.build_scene(recipe('ground'))['ground']
        sea = scenegen.build_scene(recipe('sea'))['ground']
        self.assertTrue(ground['enabled'])
        self.assertEqual(ground['type'], 'ground')
        self.assertTrue(sea['enabled'])
        self.assertEqual(sea['type'], 'sea')
        self.assertIn('wind_ms', sea)
        self.assertEqual(sea['mat'], 'seawater')
        self.assertNotEqual(ground['mat'], sea['mat'])

    def test_weather_modes_reach_the_scene(self):
        clear = scenegen.build_scene(recipe('ground'))['weather']
        self.assertEqual(clear, {'mode': 'fixed', 'preset': 'clear'})
        r = recipe('sea')
        r['environment']['weather'] = 'evolve'
        self.assertEqual(scenegen.build_scene(r)['weather']['mode'], 'evolve')
        r['environment']['weather'] = 'mixed'
        self.assertEqual(scenegen.build_scene(r)['weather']['mode'], 'per_clip')

    def test_ground_vehicle_sits_on_the_ground(self):
        r = recipe('ground')
        r['objects'] = [{'asset': '军用地面车辆/主战坦克.obj', 'cls': 'vehicle',
                         'scale': 1.0, 'count': 1}]
        cfg = scenegen.build_scene(r)
        self.assertEqual(cfg['objects'][0]['pos'][2], 0.0)
        self.assertTrue(cfg['clip']['seeker']['keep_z'])
        self.assertGreater(len(world_of(cfg).faces), 0)


class AspectTests(unittest.TestCase):
    def test_head_on_flies_at_the_camera(self):
        cfg = scenegen.build_scene(scenegen.default_recipe())
        seeker = cfg['clip']['seeker']
        self.assertEqual(seeker['closing'], 'in')
        self.assertEqual(seeker['range_m'], [6800.0, 9200.0])
        obj = cfg['objects'][0]
        cam = cfg['sensors']['optical']['pos']
        toward = sum((cam[i] - obj['pos'][i]) * obj['vel'][i] for i in range(3))
        self.assertGreater(toward, 0.0,
                           'head_on must fly the nose toward the camera')
        # rot is written rounded to a milli-degree
        self.assertAlmostEqual(
            obj['rot'], math.degrees(math.atan2(obj['vel'][1], obj['vel'][0])),
            delta=0.01)

    def test_pursuit_shows_the_tail(self):
        r = scenegen.default_recipe()
        r['view']['aspect'] = 'pursuit'
        cfg = scenegen.build_scene(r)
        self.assertEqual(cfg['clip']['seeker']['closing'], 'out')
        obj = cfg['objects'][0]
        cam = cfg['sensors']['optical']['pos']
        away = sum((cam[i] - obj['pos'][i]) * obj['vel'][i] for i in range(3))
        self.assertLess(away, 0.0, 'pursuit must fly the tail toward the camera')

    def test_planner_turns_aspect_into_a_closing_trajectory(self):
        for aspect, closing in (('head_on', 'in'), ('pursuit', 'out')):
            with self.subTest(aspect=aspect):
                r = scenegen.default_recipe()
                r['view']['aspect'] = aspect
                cfg = scenegen.build_scene(r)
                with written(cfg) as path:
                    loaded = scene.load_config(path)
                    n = loaded['frames']['n']
                    cp = plan.ClipPlan(loaded, 0, plan.frames_per_clip(loaded, n),
                                       loaded['frames']['dt'], loaded['seed'])
                    self.assertEqual(cp.mode, 'seeker')
                    self.assertEqual(cp.seek is not None, True)
                    self.assertFalse(cp.seek['keep_z'],
                                     'an airframe must not be pinned to ground_z')
                    r0, r1 = cp.range_for_time(0.0), cp.range_for_time(cp.dur)
                    if closing == 'in':
                        self.assertGreater(r0, r1)
                    else:
                        self.assertLess(r0, r1)
                    # and the aspect still holds halfway through the clip
                    mid = cp.dur * 0.5
                    p = cp.target_pos(mid)
                    v = cp.target_vel(mid)
                    los = np_sub(cp.cam_pos_at(mid), p)     # camera minus target
                    dot = sum(v[i] * los[i] for i in range(3))
                    if closing == 'in':
                        self.assertGreater(dot, 0.0, 'nose must face the camera')
                    else:
                        self.assertLess(dot, 0.0, 'tail must face the camera')
                    self.assertEqual(cp.targets[0]['cls'], 'fighter')


def np_sub(a, b):
    return [float(a[i]) - float(b[i]) for i in range(3)]


class EcmTests(unittest.TestCase):
    def test_none_writes_no_ecm_block(self):
        cfg = scenegen.build_scene(scenegen.default_recipe())
        self.assertNotIn('ecm', cfg)

    def assert_calibrated(self, cfg):
        self.assertIn('ecm', cfg)
        self.assertTrue(cfg['ecm']['enabled'])
        self.assertTrue(cfg['ecm']['entities'])
        spec = ecm.load(cfg['ecm'])            # must pass the real validator
        jams = [e for e in spec['entities'] if e['kind'] == 'noise_jammer']
        self.assertEqual(len(jams), 1)
        jam = jams[0]
        # the measured calibration, not a token jammer
        self.assertEqual(jam['erp_w'], scenegen.ECM_JAMMER_ERP_W)
        self.assertEqual(jam['erp_w'], 200.0)
        self.assertGreaterEqual(jam['erp_w'], 100.0,
                                'a 2 W jammer would be indistinguishable from off')
        self.assertEqual(jam['band'], 'Ku')
        self.assertEqual(jam['mode'], 'barrage')
        self.assertEqual(jam['jam_bandwidth_mhz'], 200.0)
        self.assertEqual(jam['sidelobe_db'], -25.0)
        self.assertIs(jam['look_at_sensor'], True)
        for key in ('erp_w', 'band', 'mode', 'sidelobe_db', 'off_band_factor',
                    'jam_bandwidth_mhz', 'rcs_m2', 'size', 'mat', 'life'):
            self.assertIn(key, jam)
        cam = cfg['sensors']['optical']['pos']
        self.assertAlmostEqual(math.dist(jam['pos'], cam), 500.0, delta=1.0)
        return jam

    def test_active_mode_emits_the_calibrated_jammer(self):
        r = scenegen.default_recipe()
        r['ecm'] = {'mode': 'active', 'entities': []}
        cfg = scenegen.build_scene(r)
        self.assert_calibrated(cfg)
        kinds = {e['kind'] for e in cfg['ecm']['entities']}
        self.assertIn('false_target', kinds)
        self.assertGreater(len(world_of(cfg).faces), 0)

    def test_passive_mode_emits_chaff(self):
        r = scenegen.default_recipe()
        r['ecm'] = {'mode': 'passive', 'entities': []}
        cfg = scenegen.build_scene(r)
        kinds = {e['kind'] for e in cfg['ecm']['entities']}
        self.assertIn('chaff_cloud', kinds)
        chaff = [e for e in cfg['ecm']['entities']
                 if e['kind'] == 'chaff_cloud'][0]
        self.assertGreater(chaff['rcs_m2'], 10.0)
        self.assertGreater(chaff['growth_mps'], 0.0)
        ecm.load(cfg['ecm'])

    def test_mixed_mode_is_the_union(self):
        r = scenegen.default_recipe()
        r['ecm'] = {'mode': 'mixed', 'entities': []}
        cfg = scenegen.build_scene(r)
        self.assert_calibrated(cfg)
        kinds = {e['kind'] for e in cfg['ecm']['entities']}
        self.assertTrue({'noise_jammer', 'false_target', 'chaff_cloud'} <= kinds)
        self.assertGreater(len(world_of(cfg).objs), 1,
                           'ECM bodies must reach the world, not just the JSON')

    def test_user_entities_are_used_verbatim(self):
        given = [{'id': 'my_jammer', 'kind': 'noise_jammer', 'erp_w': 350.0,
                  'band': 'Ku', 'mode': 'spot', 'jam_bandwidth_mhz': 50.0,
                  'pos': [10.0, 20.0, 30.0], 'life': {'start_s': 0.0}}]
        r = scenegen.default_recipe()
        r['ecm'] = {'mode': 'active', 'entities': given}
        cfg = scenegen.build_scene(r)
        self.assertEqual(len(cfg['ecm']['entities']), 1)
        self.assertEqual(cfg['ecm']['entities'][0]['id'], 'my_jammer')
        self.assertEqual(cfg['ecm']['entities'][0]['mode'], 'spot')
        ecm.load(cfg['ecm'])

    def test_defaults_of_an_entity_survive_ecm_load(self):
        """Every field ecm.py needs has a value, so nothing is left implicit."""
        r = scenegen.default_recipe()
        r['ecm'] = {'mode': 'mixed', 'entities': []}
        ents = scenegen.build_scene(r)['ecm']['entities']
        spec = ecm.load({'enabled': True, 'entities': ents})
        for ent in spec['entities']:
            self.assertIn('pos', ent)
            self.assertIn('vel', ent)
            self.assertIn('life', ent)
            self.assertIsInstance(ent['life']['start_s'], float)


class ClosingRateTests(unittest.TestCase):
    """Frames decide clip length; they must never decide the aircraft's speed."""

    def seeker_speed(self, frames, asset='军用飞机/双发战斗机.obj', cls='fighter'):
        r = scenegen.default_recipe()
        r['capture']['frames'] = frames
        r['objects'] = [{'asset': asset, 'cls': cls, 'scale': 1.0, 'count': 1}]
        cfg = scenegen.build_scene(r)
        vel = cfg['objects'][0]['vel']
        return cfg, math.sqrt(sum(v * v for v in vel))

    def test_speed_is_independent_of_the_frame_count(self):
        speeds = {}
        for frames in (1, 2, 20, 60):
            cfg, speed = self.seeker_speed(frames)
            speeds[frames] = cfg['clip']['seeker']['closing_mps']
            # the vector and the declared closing rate agree, always
            self.assertAlmostEqual(speed, 250.0, delta=0.5, msg=frames)
        self.assertEqual(len(set(speeds.values())), 1,
                         'frame count changed the physics: %r' % speeds)
        self.assertEqual(speeds[2], 250.0)

    def test_a_short_clip_crosses_less_distance_not_faster(self):
        spans = {}
        rates = {}
        for frames in (2, 20):
            cfg, _speed = self.seeker_speed(frames)
            with written(cfg) as path:
                loaded = scene.load_config(path)
                cp = plan.ClipPlan(loaded, 0,
                                   plan.frames_per_clip(loaded, frames),
                                   loaded['frames']['dt'], loaded['seed'])
                spans[frames] = abs(cp.r1 - cp.r0)
                rates[frames] = abs((cp.r1 - cp.r0) / cp.dur)
        self.assertLess(spans[2], spans[20],
                        'the 2-frame clip must travel less, not fly faster')
        self.assertAlmostEqual(spans[2], 50.0, delta=1.0)     # 250 m/s * 0.2 s
        self.assertAlmostEqual(spans[20], 500.0, delta=1.0)   # 250 m/s * 2.0 s
        for frames, rate in rates.items():
            self.assertLessEqual(rate, 300.0, frames)

    def test_the_class_table_decides_the_speed(self):
        cases = [('军用飞机/双发战斗机.obj', 'fighter', 250.0),
                 ('旋翼无人机/四旋翼无人机.obj', 'uav_rotor', 25.0),
                 ('军用地面车辆/主战坦克.obj', 'vehicle', 15.0)]
        for asset, cls, expected in cases:
            with self.subTest(cls=cls):
                self.assertEqual(scenegen.CLOSING_MPS[cls], expected)
                cfg, speed = self.seeker_speed(20, asset, cls)
                self.assertEqual(cfg['clip']['seeker']['closing_mps'], expected)
                self.assertAlmostEqual(speed, expected, delta=0.5)

    def test_recipe_can_override_the_closing_speed(self):
        r = scenegen.default_recipe()
        r['view']['closing_mps'] = 120.0
        cfg = scenegen.build_scene(r)
        self.assertEqual(cfg['clip']['seeker']['closing_mps'], 120.0)
        vel = cfg['objects'][0]['vel']
        self.assertAlmostEqual(math.sqrt(sum(v * v for v in vel)), 120.0,
                               delta=0.5)

    def test_absurd_closing_speeds_are_rejected_in_chinese(self):
        for bad in (0.0, 0.05, 900.0, -250.0):
            with self.subTest(cls=bad):
                r = scenegen.default_recipe()
                r['view']['closing_mps'] = bad
                with self.assertRaises(ValueError) as caught:
                    scenegen.validate_recipe(r)
                msg = str(caught.exception)
                self.assertIn('view.closing_mps', msg)
                self.assertTrue(has_chinese(msg))

    def test_too_few_frames_is_noted_not_an_error(self):
        cfg, _ = self.seeker_speed(2)
        self.assertEqual(len(cfg['objects']), 1)
        self.assertTrue(any('帧数过少' in note for note in cfg['notes']),
                        cfg['notes'])
        cfg, _ = self.seeker_speed(20)
        self.assertEqual(cfg['notes'], [])


class BackgroundTests(unittest.TestCase):
    """A chosen background must reach the renderer keys, not just the recipe."""

    PNG = os.path.join(scenegen.MODELS_DIR, '旋翼无人机', '四旋翼无人机.png')

    def test_none_writes_no_background_key(self):
        cfg = scenegen.build_scene(scenegen.default_recipe())
        self.assertNotIn('sky_image', cfg['optical'])
        self.assertNotIn('background_image', cfg['ir'])
        self.assertNotIn('ir', cfg['sensors'])
        self.assertEqual(cfg['notes'], [])

    def test_upload_reaches_both_renderers_and_no_radar(self):
        self.assertTrue(os.path.isfile(self.PNG))
        r = scenegen.default_recipe()
        r['environment']['background'] = {'mode': 'upload', 'path': self.PNG}
        cfg = scenegen.build_scene(r)
        # render_optical.py reads cfg['optical']['sky_image'] and ir.py reads
        # cfg['ir']['background_image']; those two keys are the whole contract.
        self.assertEqual(cfg['optical']['sky_image'], os.path.abspath(self.PNG))
        self.assertEqual(cfg['ir']['background_image'], os.path.abspath(self.PNG))
        # sensors.optical is the geometry block: a background there is ignored
        # by both renderers (measured: bit-identical to the gradient sky), so the
        # generator must not put one there and pretend it works.
        self.assertNotIn('sky_image', cfg['sensors']['optical'])
        self.assertNotIn('ir', cfg['sensors'])
        # a sky behind a range-Doppler map is not a thing
        for key in ('radar', 'lidar', 'sar'):
            self.assertNotIn('sky_image', cfg['sensors'][key])
            self.assertNotIn('background_image', cfg['sensors'][key])
        self.assertGreater(len(world_of(cfg).faces), 0)

    def test_only_the_requested_modalities_get_a_background(self):
        r = scenegen.default_recipe()
        r['environment']['background'] = {'mode': 'upload', 'path': self.PNG}
        r['capture']['sensors'] = ['optical']
        cfg = scenegen.build_scene(r)
        self.assertIn('sky_image', cfg['optical'])
        self.assertNotIn('background_image', cfg['ir'])
        r['capture']['sensors'] = ['radar']
        cfg = scenegen.build_scene(r)
        self.assertNotIn('sky_image', cfg['optical'])
        self.assertNotIn('background_image', cfg['ir'])

    def test_relative_upload_is_written_absolute(self):
        rel = os.path.relpath(self.PNG, scenegen.ROOT).replace(os.sep, '/')
        r = scenegen.default_recipe()
        r['environment']['background'] = {'mode': 'upload', 'path': rel}
        cfg = scenegen.build_scene(r)
        self.assertTrue(os.path.isabs(cfg['optical']['sky_image']))
        self.assertEqual(cfg['optical']['sky_image'], os.path.abspath(self.PNG))

    def test_missing_upload_raises_chinese_value_error(self):
        r = scenegen.default_recipe()
        r['environment']['background'] = {'mode': 'upload',
                                          'path': 'no/such/plate.png'}
        with self.assertRaises(ValueError) as caught:
            scenegen.build_scene(r)
        msg = str(caught.exception)
        self.assertIn('背景图', msg)
        self.assertTrue(has_chinese(msg))

    def test_hdri_uses_the_local_cache(self):
        try:
            from simkit import assets_fetch
        except Exception as exc:                        # pragma: no cover
            self.skipTest('assets_fetch not importable: %s' % exc)
        cached = [e for e in assets_fetch.list_cached()
                  if e.get('kind') == 'hdri' and os.path.isfile(e.get('path') or '')]
        if not cached:
            self.skipTest('no cached HDRI on this machine')
        # Weather decides which plate is used, not the id the form happened to
        # send: the UI always sends whatever it last used, so honouring the id
        # first made "cloudy" keep rendering the clear sky.
        r = scenegen.default_recipe()
        weather = r['environment']['weather']
        r['environment']['background'] = {'mode': 'hdri', 'id': cached[0]['id']}
        cfg = scenegen.build_scene(r)
        chosen = cfg['optical']['sky_image']
        self.assertTrue(os.path.isfile(chosen), chosen)
        want = [i for i in scenegen.WEATHER_HDRI.get(weather, ())
                if any(e.get('id') == i for e in cached)]
        if want:
            by_id = {e['id']: e['path'] for e in cached}
            self.assertEqual(chosen, by_id[want[0]],
                             'weather %r should pick %r' % (weather, want[0]))
        self.assertEqual(cfg['ir']['background_image'], chosen)
        self.assertEqual(cfg['notes'], [])
        self.assertGreater(len(world_of(cfg).faces), 0)

    def test_hdri_offline_falls_back_to_the_gradient_sky(self):
        from unittest import mock
        try:
            from simkit import assets_fetch
        except Exception as exc:                        # pragma: no cover
            self.skipTest('assets_fetch not importable: %s' % exc)

        def offline(*_args, **_kwargs):
            raise assets_fetch.AssetsUnavailable('simulated offline')

        r = scenegen.default_recipe()
        r['environment']['background'] = {'mode': 'hdri',
                                          'id': 'definitely_not_cached_sky'}
        # A machine with no cached plates at all, and no network: the weather
        # table cannot be satisfied either, so the gradient has to take over.
        with mock.patch.object(assets_fetch, 'list_cached', lambda *a, **k: []), \
                mock.patch.object(assets_fetch, 'hdri_for_weather', offline):
            cfg = scenegen.build_scene(r)               # must not raise
        self.assertNotIn('sky_image', cfg['optical'])
        self.assertNotIn('background_image', cfg['ir'])
        self.assertTrue(any('内置天空不可用' in note for note in cfg['notes']),
                        cfg['notes'])
        # the scene is still a legal scene
        self.assertGreater(len(world_of(cfg).faces), 0)

    def test_background_blocks_only_touch_optical_and_ir(self):
        r = scenegen.default_recipe()
        r['environment']['background'] = {'mode': 'upload', 'path': self.PNG}
        cfg = scenegen.build_scene(r)
        for key, block in cfg['sensors'].items():
            if key in ('optical', 'ir'):
                continue
            self.assertNotIn('sky_image', block, key)
            self.assertNotIn('background_image', block, key)


class RadarWindowTests(unittest.TestCase):
    """The radar's unambiguous range window must contain the target.

    radar.py gates every echo with r_max = c*n_samples/(4*B). A constant 2048 at
    50 MHz sees 3.07 km, so at 8 km every frame reported boxes_rd = 0 while the
    run itself looked successful -- an empty modality is the failure mode this
    class exists to prevent.
    """

    def r_max(self, radar):
        return (scenegen.C_LIGHT * radar['n_samples']
                / (4.0 * float(radar['bandwidth_mhz']) * 1e6))

    def test_sample_count_stays_a_power_of_two_that_covers_the_range(self):
        """Samples are no longer dictated by range alone -- speed competes too.

        v_nyq = lam*duty*fs/(4*n) does not involve bandwidth, so a fast closure
        forces n down; the range window is then held by narrowing the band.  The
        invariant that must survive is the one radar.py gates on: a power-of-two
        count whose c*n/(4B) still contains the target.
        """
        for km in (2, 4, 8, 16, 32):
            with self.subTest(range_km=km):
                r = scenegen.default_recipe()
                r['view']['range_km'] = km
                radar = scenegen.build_scene(r)['sensors']['radar']
                self.assertTrue(2048 <= radar['n_samples'] <= 32768)
                self.assertEqual(radar['n_samples'] & (radar['n_samples'] - 1), 0,
                                 'n_samples must stay a power of two for the FFT')
                self.assertGreaterEqual(self.r_max(radar), 1.2 * km * 1000.0)

    def test_doppler_window_covers_the_closing_speed(self):
        """An unambiguous Doppler axis is the point of deriving the schedule.

        Before this, chirp/pri were constants: a 250 m/s fighter folded to
        50 m/s at Ku and the RD image taught the wrong velocity.
        """
        for km in (2, 8, 16):
            with self.subTest(range_km=km):
                r = scenegen.default_recipe()
                r['view']['range_km'] = km
                cfg = scenegen.build_scene(r)
                radar = cfg['sensors']['radar']
                v = abs(float((cfg['clip'].get('seeker') or {}).get('closing_mps') or 0.0))
                self.assertGreater(v, 0.0)
                self.assertGreaterEqual(radar['v_nyq_mps'], v,
                                        '%g km: v_nyq %.1f < closing %.1f'
                                        % (km, radar['v_nyq_mps'], v))
                self.assertTrue(radar['unambiguous'])
                # the window and the velocity budget are paid for out of range
                # resolution, so the floor has to hold too
                self.assertLessEqual(radar['range_res_m'],
                                     scenegen.RADAR_RES_FLOOR_M + 1e-6)

    def test_window_always_covers_the_planned_target(self):
        for km in (2, 8, 20, 40, 100):
            with self.subTest(range_km=km):
                r = scenegen.default_recipe()
                r['view']['range_km'] = km
                r['capture']['frames'] = 20
                cfg = scenegen.build_scene(r)
                radar = cfg['sensors']['radar']
                self.assertTrue(2048 <= radar['n_samples'] <= 32768)
                self.assertEqual(radar['n_samples'] & (radar['n_samples'] - 1), 0,
                                 'n_samples must stay a power of two for the FFT')
                with written(cfg) as path:
                    loaded = scene.load_config(path)
                    cp = plan.ClipPlan(loaded, 0, plan.frames_per_clip(loaded, 20),
                                       loaded['frames']['dt'], loaded['seed'])
                    # the whole clip lives inside the sampling window
                    self.assertLess(max(cp.r0, cp.r1), self.r_max(radar),
                                    '%g km scene at %.0f m, window %.0f m'
                                    % (km, max(cp.r0, cp.r1), self.r_max(radar)))

    def test_long_range_narrows_the_band_instead_of_shipping_a_blind_radar(self):
        r = scenegen.default_recipe()
        r['view']['range_km'] = 100.0
        radar = scenegen.build_scene(r)['sensors']['radar']
        self.assertLess(radar['bandwidth_mhz'], 50.0)
        self.assertGreaterEqual(self.r_max(radar), 100000.0)

    def test_impossible_geometry_reports_ambiguity_instead_of_hiding_it(self):
        """At 100 km a 250 m/s closure cannot fit: say so, keep the number."""
        r = scenegen.default_recipe()
        r['view']['range_km'] = 100.0
        radar = scenegen.build_scene(r)['sensors']['radar']
        self.assertFalse(radar['unambiguous'])
        self.assertGreater(radar['v_nyq_mps'], 0.0)
        self.assertIn('v_nyq_mps', radar)
        self.assertGreaterEqual(self.r_max(radar), 100000.0)

    def test_radar_and_lidar_ranges_agree_with_the_scene(self):
        r = scenegen.default_recipe()
        r['view']['range_km'] = 8.0
        sensors = scenegen.build_scene(r)['sensors']
        self.assertGreaterEqual(self.r_max(sensors['radar']), 8000.0)
        self.assertGreater(sensors['lidar']['max_range_m'], 8000.0)
        self.assertGreater(sensors['lidar']['max_range_m'],
                           sensors['lidar']['range_spec_m'])
        self.assertAlmostEqual(sensors['lidar']['range_spec_m'], 8800.0, delta=2.0)


class AssetListTests(unittest.TestCase):
    def test_every_obj_on_disk_is_listed(self):
        on_disk = 0
        for _base, _dirs, files in os.walk(scenegen.MODELS_DIR):
            on_disk += sum(1 for f in files if f.lower().endswith('.obj'))
        rows = scenegen.list_assets()
        self.assertEqual(len(rows), on_disk)
        self.assertGreaterEqual(len(rows), 60)

    def test_rows_have_the_documented_shape(self):
        for row in scenegen.list_assets():
            self.assertEqual(set(row), {'category', 'name', 'path', 'tris',
                                        'size_m', 'cls'})
            self.assertGreater(row['tris'], 0, row['path'])
            self.assertEqual(len(row['size_m']), 3)
            self.assertTrue(all(v > 0 for v in row['size_m']), row['path'])
            self.assertFalse(row['path'].startswith('assets/'))
            self.assertTrue(os.path.isfile(
                os.path.join(scenegen.MODELS_DIR, *row['path'].split('/'))))
            self.assertTrue(row['cls'])
            self.assertNotEqual(row['cls'], 'target', row['path'])

    def test_directory_to_class_table(self):
        rows = {r['path']: r for r in scenegen.list_assets()}
        self.assertEqual(rows['军用飞机/双发战斗机.obj']['cls'], 'fighter')
        self.assertEqual(rows['军舰/导弹艇.obj']['cls'], 'ship')
        self.assertEqual(rows['军用地面车辆/主战坦克.obj']['cls'], 'vehicle')
        self.assertEqual(rows['地面车辆/皮卡.obj']['cls'], 'vehicle')
        self.assertTrue(any(v['cls'] == 'uav_rotor' for v in rows.values()))
        self.assertTrue(any(v['cls'] == 'helicopter' for v in rows.values()))
        self.assertTrue(any(v['cls'] == 'airliner' for v in rows.values()))

    def test_geometry_comes_from_the_asset_index(self):
        rows = {r['path']: r for r in scenegen.list_assets()}
        with open(scenegen.INDEX_PATH, 'r', encoding='utf-8') as fh:
            index = json.load(fh)['models']
        checked = 0
        for entry in index:
            key = scenegen._index_key(entry.get('path'))
            if key not in rows:
                continue
            size = entry['geometry']['bbox_extent']
            self.assertEqual(rows[key]['tris'], int(entry['geometry']['faces']))
            for a, b in zip(rows[key]['size_m'], size):
                self.assertAlmostEqual(a, float(b), places=3)
            checked += 1
        self.assertGreater(checked, 40)

    def test_every_asset_is_usable_as_a_recipe_object(self):
        for row in scenegen.list_assets():
            with self.subTest(asset=row['path']):
                r = scenegen.default_recipe()
                r['objects'] = [{'asset': row['path'], 'cls': row['cls']}]
                norm = scenegen.validate_recipe(r)
                self.assertEqual(norm['objects'][0]['cls'], row['cls'])


class SummaryTests(unittest.TestCase):
    def test_summary_reads_like_the_ui_wants(self):
        r = scenegen.default_recipe()
        self.assertEqual(scenegen.env_summary(r), '天空 · 晴 · 纯色背景')
        r['environment']['background'] = {'mode': 'hdri', 'id': 'dawn'}
        self.assertEqual(scenegen.env_summary(r), '天空 · 晴 · HDRI')
        r['environment']['background'] = {'mode': 'upload',
                                          'path': 'u/bg.png'}
        self.assertEqual(scenegen.env_summary(r), '天空 · 晴 · 上传图')

    def test_summary_covers_all_three_environments(self):
        labels = {'sky': '天空', 'ground': '地面', 'sea': '海面'}
        for env in scenegen.ENV_KINDS:
            self.assertTrue(scenegen.env_summary(recipe(env)).startswith(
                labels[env]))

    def test_summary_is_lenient_while_the_form_is_empty(self):
        for partial in ({}, {'environment': {}},
                        {'environment': {'kind': 'sea', 'weather': 'rain'}},
                        {'environment': 'broken'}):
            text = scenegen.env_summary(partial)
            self.assertIsInstance(text, str)
            self.assertIn('·', text)


class StudioIntegrationTests(unittest.TestCase):
    def test_tables_match_studio(self):
        try:
            from simkit import studio
        except Exception as exc:                       # pragma: no cover
            self.skipTest('studio not importable: %s' % exc)
        self.assertEqual(list(studio.WEATHER_MODES), list(scenegen.WEATHER_MODES))
        for key, block in studio.WEATHER_MODES.items():
            self.assertEqual(dict(block), dict(scenegen.WEATHER_MODES[key]))
            self.assertEqual(studio.WEATHER_LABELS[key],
                             scenegen.WEATHER_LABELS[key])
        self.assertEqual([s['key'] for s in studio.SENSORS],
                         list(scenegen.SENSOR_KEYS))

    def test_studio_can_build_and_compose_a_recipe_scene(self):
        try:
            from simkit import studio
        except Exception as exc:                       # pragma: no cover
            self.skipTest('studio not importable: %s' % exc)
        r = scenegen.default_recipe()
        r['environment']['kind'] = 'ground'
        r['environment']['background'] = {'mode': 'upload',
                                          'path': BackgroundTests.PNG}
        r['ecm'] = {'mode': 'mixed', 'entities': []}
        path = None
        try:
            q = studio._parse({'recipe': r, 'frames': 10, 'fps': 10})
            path = q['scene_path']
            self.assertEqual(q['scene'], 'demo')
            cfg = studio._compose(q, studio._load_scene(path))
            world = scene.build_world(cfg)
            self.assertGreater(len(world.faces), 0)
            self.assertEqual(cfg['objects'][0]['shape'], 'obj')
            # the studio overlay must not drop the render-block keys, or a
            # recipe-originated run would render without its chosen background
            self.assertEqual(cfg['optical']['sky_image'],
                             os.path.abspath(BackgroundTests.PNG))
            self.assertEqual(cfg['ir']['background_image'],
                             os.path.abspath(BackgroundTests.PNG))
        finally:
            if path and os.path.isfile(path):
                os.remove(path)


if __name__ == '__main__':
    unittest.main()
