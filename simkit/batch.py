"""Parallel batch generation.

Each job is one independent "simkit gen" process. Fanning out at the process
level rather than with threads is deliberate: the numpy pipelines hold the GIL
for most of their inner loops and the optical pass blocks on an external Blender
process, so threads would mostly wait.
"""
import os
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def run_batch(jobs, workers=1, on_event=None, cancel=None):
    """Run jobs [(scene_path, out_dir, frames, sensors)] with a worker pool.

    Returns (results, stats). results keeps submission order. on_event is called
    from the scheduling thread with progress dicts so a UI can poll it.
    """
    jobs = list(jobs)
    workers = max(1, int(workers))
    results = [None] * len(jobs)
    lock = threading.Lock()
    queue = list(range(len(jobs)))
    started = [0]
    t_all = time.perf_counter()

    def emit(kind, **kw):
        if on_event:
            try:
                on_event(dict(kind=kind, total=len(jobs), workers=workers, **kw))
            except Exception:
                pass

    def worker(slot):
        env = dict(os.environ, PYTHONPATH=ROOT, PYTHONDONTWRITEBYTECODE="1")
        while True:
            with lock:
                if cancel is not None and cancel.is_set():
                    return
                if not queue:
                    return
                idx = queue.pop(0)
                started[0] += 1
            scene, out_dir, frames, sensors = jobs[idx]
            t0 = time.perf_counter()
            emit("start", index=idx, slot=slot, scene=scene, out=out_dir,
                 done=started[0] - 1)
            cmd = [sys.executable, "-m", "simkit", "gen", "-c", scene,
                   "-o", out_dir, "-n", str(int(frames)),
                   "-s", ",".join(sensors)]
            proc = subprocess.run(cmd, cwd=ROOT, env=env,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            ok = proc.returncode == 0
            tail = ""
            if not ok:
                lines = (proc.stderr or b"").decode("utf-8", "replace").strip().splitlines()
                tail = " | ".join(lines[-3:])
            res = dict(index=idx, scene=scene, out=out_dir, ok=ok,
                       seconds=round(time.perf_counter() - t0, 2), error=tail)
            with lock:
                results[idx] = res
            emit("done", index=idx, slot=slot, ok=ok, seconds=res["seconds"])

    threads = [threading.Thread(target=worker, args=(i,), daemon=True)
               for i in range(min(workers, max(1, len(jobs))))]
    emit("begin")
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - t_all
    done = [r for r in results if r is not None]
    ok_n = sum(1 for r in done if r["ok"])
    emit("end", ok=ok_n, seconds=round(elapsed, 2))
    return done, dict(elapsed=round(elapsed, 2), ok=ok_n,
                      failed=len(jobs) - ok_n, workers=workers,
                      total=len(jobs))

def run_parallel_frames(scene_path, out_dir, frames, sensors, workers=1,
                        seed=None, on_event=None):
    """Render ONE dataset by splitting its frame range across workers.

    The engine already supports "gen --part i/K" plus "merge", so a dataset can
    be cut into K independent frame ranges without touching the renderer. Each
    range writes its own pass files into the same output directory and merge
    stitches them back into one dataset.
    """
    frames = int(frames)
    workers = max(1, min(int(workers), max(1, frames)))
    env = dict(os.environ, PYTHONPATH=ROOT, PYTHONDONTWRITEBYTECODE="1")
    base = [sys.executable, "-m", "simkit"]
    common = ["-c", scene_path, "-o", out_dir, "-n", str(frames),
              "-s", ",".join(sensors)]
    if seed is not None:
        common += ["--seed", str(int(seed))]
    if workers == 1:
        t0 = time.perf_counter()
        p = subprocess.run(base + ["gen"] + common, cwd=ROOT, env=env,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        ok = p.returncode == 0
        return ok, dict(workers=1, elapsed=round(time.perf_counter() - t0, 2),
                        error="" if ok else (p.stderr or b"").decode("utf-8", "replace")[-400:])
    t0 = time.perf_counter()
    procs = []
    for i in range(workers):
        if on_event:
            try:
                on_event(dict(kind="part", index=i, total=workers))
            except Exception:
                pass
        procs.append(subprocess.Popen(
            base + ["gen"] + common + ["--part", "%d/%d" % (i, workers)],
            cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE))
    errors = []
    for i, p in enumerate(procs):
        p.wait()
        if p.returncode != 0:
            errors.append("part %d: %s" % (i, (p.stderr.read() or b"").decode("utf-8", "replace")[-200:]))
    if errors:
        return False, dict(workers=workers, elapsed=round(time.perf_counter() - t0, 2),
                           error=" | ".join(errors)[:600])
    merge = subprocess.run(base + ["merge"] + common, cwd=ROOT, env=env,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    ok = merge.returncode == 0
    return ok, dict(workers=workers, elapsed=round(time.perf_counter() - t0, 2),
                    error="" if ok else (merge.stderr or b"").decode("utf-8", "replace")[-400:])