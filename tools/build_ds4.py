
"""Rebuild the YOLO training set with the current kernel.

4 seeker scenes x 240 frames (two 120-frame clips each), all five sensors,
6-way chunked render per scene, then merge. Resumable: a scene whose
frames.json already reports the target length is skipped.
"""
import json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

ROOT = r'D:\work\多模态数据集生成\RGB_SAR_LIDAR_SIMU'
PY = os.path.join(ROOT, 'env', 'Scripts', 'python.exe')
DEST = os.path.join(ROOT, 'out', 'ds4')
LOG = os.path.join(ROOT, 'out', 'ds4_build.log')
SCENES = [
    ('seeker_air2ground', 'scenes/sk3_low_air2ground_clutter.json', 240),
    ('seeker_air2air_high', 'scenes/sk3_high_air2air.json', 240),
    ('seeker_surface2air', 'scenes/sk3_low_surface2air.json', 240),
    ('seeker_air2air_occluded', 'scenes/sk3_low_air2air_occluded.json', 240),
]
SENSORS = 'ir,radar,lidar,optical,sar'
CHUNKS = 6


def log(msg):
    line = '[%s] %s' % (time.strftime('%H:%M:%S'), msg)
    print(line, flush=True)
    with open(LOG, 'a', encoding='utf-8') as fh:
        fh.write(line + '\n')


def done(name, frames):
    p = os.path.join(DEST, name, 'frames.json')
    if not os.path.exists(p):
        return False
    try:
        d = json.load(open(p, encoding='utf-8'))
        n = len(d if isinstance(d, list) else d.get('frames') or [])
        return n >= frames
    except Exception:
        return False


def run_scene(name, cfg, frames):
    out = os.path.join('out', 'ds4', name)
    env = dict(os.environ)
    env['PYTHONPATH'] = ROOT
    env['SIMKIT_BLENDER_THREADS'] = '3'
    log('start %s (%d frames, %d chunks)' % (name, frames, CHUNKS))
    t0 = time.time()

    def one(i):
        args = [PY, '-m', 'simkit', 'gen', '-c', cfg, '-o', out, '-n', str(frames),
                '-s', SENSORS, '--part', '%d/%d' % (i, CHUNKS), '-q']
        p = subprocess.Popen(args, cwd=ROOT, env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        o, _ = p.communicate()
        return i, p.returncode, o.decode('utf-8', 'replace')[-800:]

    bad = []
    with ThreadPoolExecutor(max_workers=CHUNKS) as ex:
        for i, rc, tail in ex.map(one, range(CHUNKS)):
            if rc != 0:
                bad.append((i, tail))
    if bad:
        for i, tail in bad:
            log('  chunk %d FAILED: %s' % (i, tail.replace('\n', ' | ')[-300:]))
        return False
    dt = time.time() - t0
    log('  render done %.1f min (%5.1f s/frame)' % (dt / 60, dt / frames))
    p = subprocess.run([PY, '-m', 'simkit', 'merge', '-c', cfg, '-o', out,
                        '-n', str(frames), '-s', SENSORS],
                       cwd=ROOT, env=env, capture_output=True, text=True)
    if p.returncode != 0:
        log('  MERGE FAILED: %s' % (p.stdout + p.stderr)[-400:])
        return False
    log('  merged ok, total %.1f min' % ((time.time() - t0) / 60))
    return True


def main():
    os.makedirs(DEST, exist_ok=True)
    log('build start: %d scenes x 240 frames, sensors=%s' % (len(SCENES), SENSORS))
    for name, cfg, frames in SCENES:
        if done(name, frames):
            log('skip %s (already %d frames)' % (name, frames))
            continue
        if not run_scene(name, cfg, frames):
            log('ABORT at %s' % name)
            sys.exit(1)
    log('all scenes done')


if __name__ == '__main__':
    main()
