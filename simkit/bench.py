"""Machine capability probe and parallel scaling benchmark.

The point of this module is to answer one question before a long batch starts:
how many frames can THIS machine render at once without thrashing?  It answers
it by measuring, not by guessing from os.cpu_count(): the optical pass spends
most of its wall time inside a Blender subprocess, so the useful parallelism is
decided by memory and by how well the Python side scales, not by the core count
alone.

Design notes
------------
* One "unit" is one rendered frame of the engine's cheapest sensor, because that
  is a few seconds instead of a minute and the scaling curve still shows whether
  the machine is CPU-bound or memory-bound.
* Every measurement is wall-clock over one run, so the caller can re-run it on a
  busy machine and get a different (correct) recommendation.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

#: Workers are capped so the desktop stays usable and the measurement itself
#: does not take longer than the batch it is advising.
MAX_WORKERS = 16
#: Leave this many logical cores free for the OS, Blender helpers and the user.
CORE_RESERVE = 2
#: Rough resident set per worker. Measured on this engine at 960x720 with the
#: wide view; over-estimating is the safe direction.
RSS_PER_WORKER_MB = 1100


def _logical_cores():
    return max(1, int(os.cpu_count() or 1))


def _total_ram_mb():
    """Total and available physical memory in MB, without psutil."""
    if os.name == "nt":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            st = MEMORYSTATUSEX()
            st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            return int(st.ullTotalPhys // (1024 * 1024)), int(st.ullAvailPhys // (1024 * 1024))
        except Exception:
            pass
    try:
        with open("/proc/meminfo", "r") as fh:
            info = dict(line.split(":", 1) for line in fh)
        total = int(info["MemTotal"].split()[0]) // 1024
        avail = int(info["MemAvailable"].split()[0]) // 1024
        return total, avail
    except Exception:
        return 0, 0


def probe():
    """Static facts about the machine, no rendering."""
    total_mb, avail_mb = _total_ram_mb()
    cores = _logical_cores()
    return dict(
        logical_cores=cores,
        physical_cores=_physical_cores(cores),
        ram_total_mb=total_mb,
        ram_available_mb=avail_mb,
        platform=sys.platform,
    )


def _physical_cores(logical):
    """Best-effort physical core count; falls back to logical/2."""
    if os.name == "nt":
        try:
            out = subprocess.run(
                ["wmic", "cpu", "get", "NumberOfCores"], capture_output=True,
                text=True, timeout=10).stdout
            nums = [int(x) for x in out.split() if x.isdigit()]
            if nums:
                return sum(nums)
        except Exception:
            pass
    try:
        with open("/proc/cpuinfo", "r") as fh:
            ids = {line.split(":")[1].strip()
                   for line in fh if line.startswith("core id")}
        if ids:
            return len(ids) * 1
    except Exception:
        pass
    return max(1, logical // 2)


def _bench_scene(name, sensor="ir", frames=1):
    """Write a small fixed scene so the measurement is comparable run to run.

    The scene must sit EXACTLY one level below the repository root: a recipe
    writes model paths as "../assets/models/...", which resolves against the
    scene's own directory, so a nested folder silently breaks every render.
    """
    from . import scenegen
    base = json.load(open(os.path.join(ROOT, "configs", "recipes",
                                       "trilateral_sky_headon.json"), encoding="utf-8"))
    rec = json.loads(json.dumps(base))
    rec["name"] = "bench"
    rec["capture"].update(sensors=[sensor], frames=int(frames), seed=3)
    rec["view"].update(range_km=6.0, aspect="head_on", fov_deg=2.2)
    cfg = scenegen.build_scene(scenegen.validate_recipe(rec))
    d = os.path.join(ROOT, "_tmp")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "_bench_%s.json" % name)
    json.dump(cfg, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return path


def _run_one(scene, out_dir, sensor):
    env = dict(os.environ, PYTHONPATH=ROOT, PYTHONDONTWRITEBYTECODE="1")
    t0 = time.perf_counter()
    p = subprocess.run([sys.executable, "-m", "simkit", "gen", "-c", scene,
                        "-o", out_dir, "-n", "1", "-s", sensor],
                       cwd=ROOT, env=env, stdout=subprocess.DEVNULL,
                       stderr=subprocess.PIPE)
    return (time.perf_counter() - t0), p.returncode == 0


def measure(workers=(1, 2, 4, 8, 12), sensor="ir", allow_shrink=True):
    """Wall-clock time for N simultaneous single-frame renders.

    Returns a list of {workers, seconds, per_frame, ok}.  The caller compares
    per_frame across rows: the worker count with the smallest per_frame is the
    one to recommend, and a flat or rising curve is the honest signal that the
    machine is already saturated.
    """
    tmp = os.path.join(ROOT, "_tmp", "bench_out")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)
    rows = []
    try:
        scenes = {}
        for w in workers:
            for k in range(int(w)):
                d = os.path.join(tmp, "w%d_%d" % (int(w), k))
                scenes[(int(w), k)] = (_bench_scene("w%d_%d" % (int(w), k), sensor, 1), d)
        for w in workers:
            w = int(w)
            t0 = time.perf_counter()
            procs = []
            env = dict(os.environ, PYTHONPATH=ROOT, PYTHONDONTWRITEBYTECODE="1")
            for k in range(w):
                scene, d = scenes[(w, k)]
                procs.append(subprocess.Popen(
                    [sys.executable, "-m", "simkit", "gen", "-c", scene,
                     "-o", d, "-n", "1", "-s", sensor],
                    cwd=ROOT, env=env, stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL))
            oks = [p.wait() == 0 for p in procs]
            span = time.perf_counter() - t0
            rows.append(dict(workers=w, seconds=round(span, 2),
                             per_frame=round(span / max(w, 1), 3),
                             ok=all(oks)))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return rows


def knee(rows, min_gain=0.08):
    """The smallest worker count past which more workers stop paying.

    The measured curve is the point: the optical pass saturates near 2x because
    each Blender instance already fills the CPU, while a pure IR batch keeps
    scaling. Recommending the last row of the ladder would triple the memory for
    a few percent, so the recommendation is the knee, not the maximum.
    """
    usable = [r for r in rows if r.get("ok")]
    if len(usable) < 2:
        return (usable[0]["workers"] if usable else 1)
    for i in range(1, len(usable)):
        prev, cur = usable[i - 1], usable[i]
        if not prev["per_frame"]:
            continue
        gain = (prev["per_frame"] - cur["per_frame"]) / prev["per_frame"]
        if gain < min_gain:
            return prev["workers"]
    return usable[-1]["workers"]


def recommend(bench=None, probe_info=None, sensor="ir"):
    """Suggested worker count, with the reason spelled out."""
    info = probe_info or probe()
    rows = bench if bench is not None else measure(sensor=sensor)
    usable = [r for r in rows if r.get("ok")]
    best = min(usable, key=lambda r: r["per_frame"]) if usable else None
    knee_w = knee(rows) if usable else 1
    cores = info["logical_cores"]
    by_core = max(1, min(MAX_WORKERS, cores - CORE_RESERVE))
    by_ram = max(1, (info["ram_available_mb"] // RSS_PER_WORKER_MB)) if info["ram_available_mb"] else MAX_WORKERS
    ceiling = max(1, min(MAX_WORKERS, by_core, by_ram))
    chosen = max(1, min(ceiling, knee_w))
    reason = []
    reason.append("逻辑核 %d，留 %d 核给系统 → 上限 %d" % (cores, CORE_RESERVE, by_core))
    if by_ram < by_core:
        reason.append("可用内存 %.1f GB，每 worker 约 %.1f GB → 上限 %d"
                      % (info["ram_available_mb"] / 1024.0, RSS_PER_WORKER_MB / 1024.0, by_ram))
    if usable:
        first = usable[0]
        at = next((r for r in usable if r["workers"] == chosen), None)
        reason.append("实测拐点 %d 路（每帧 %.2f 秒，1 路 %.2f 秒，加速 %.2fx）"
                      % (chosen, at["per_frame"] if at else first["per_frame"],
                         first["per_frame"],
                         (first["per_frame"] / at["per_frame"]) if at and at["per_frame"] else 1.0))
        tail = usable[-1]
        if tail["workers"] > chosen:
            reason.append("再加到 %d 路只快 %.0f%%，不值得多占内存"
                          % (tail["workers"],
                             100.0 * (at["per_frame"] - tail["per_frame"])
                             / max(at["per_frame"], 1e-9) if at else 0))
    return dict(
        recommended=chosen,
        ceiling=ceiling,
        sensor=sensor,
        probe=info,
        bench=rows,
        reason=reason,
        speedup=(round(usable[0]["per_frame"] / best["per_frame"], 2)
                 if best and best["per_frame"] else 1.0),
    )


if __name__ == "__main__":
    print(json.dumps(recommend(), ensure_ascii=False, indent=1))
