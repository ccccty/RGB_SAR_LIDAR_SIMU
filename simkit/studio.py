"""Local dataset studio: click parameters, get a dataset.

A single-process, dependency-free HTTP server that wraps the generator. It
exists because the command line needs seven flags and a JSON edit to answer a
question the user asks in one sentence ("give me rain, small targets, Ka band,
500 frames"), and because a dataset you cannot look at before committing an
hour of compute is a dataset you generate three times.

Endpoints
    GET  /                      the UI
    GET  /api/presets           built-in scene presets + option lists
    POST /api/generate          start a run  (JSON body)
    POST /api/preview           render a few frames into a preview run
    GET  /api/jobs              every run this server has started
    GET  /api/job/<id>          one run: status, progress, tail of the log
    POST /api/job/<id>/cancel   stop a run
    GET  /api/img/<id>/<path>   serve a generated image or label
    GET  /api/plan              per-clip trajectory facts (ranges, pixel size)

Nothing here touches the network, and the UI is plain ES5 so it works from a
USB stick on an air-gapped machine.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENES = os.path.join(ROOT, "scenes")
RUNS = os.path.join(ROOT, "out", "studio_runs")
UI = os.path.join(ROOT, "studio")

PRESETS = {
    "air_uav": dict(
        title="空中 · 无人机", scene="sk_air_uav.json", cls="uav",
        note="导引头仰视，目标 10-55 px，2.2 度视场，闭合 150 m/s",
        fov=2.2, sizes=dict(small=[8, 20], medium=[18, 42], large=[35, 80])),
    "ground_vehicle": dict(
        title="地面 · 车辆", scene="sk_ground_vehicle.json", cls="vehicle",
        note="地面视角，目标 12-70 px，6 度视场，闭合 8 m/s",
        fov=6.0, sizes=dict(small=[8, 18], medium=[16, 40], large=[35, 80])),
    "sea_boat": dict(
        title="海面 · 船只", scene="sk_sea_boat.json", cls="boat",
        note="海面视角，目标 10-60 px，3 度视场，雷达 12 m/s 远离",
        fov=3.0, sizes=dict(small=[8, 20], medium=[18, 45], large=[40, 85])),
}

WEATHER_MODES = {
    "clear": dict(mode="fixed", preset="clear"),
    "cloudy": dict(mode="fixed", preset="cloudy"),
    "rain": dict(mode="fixed", preset="rain"),
    "fog": dict(mode="fixed", preset="fog"),
    "mixed": dict(mode="per_clip", pool=["clear", "cloudy", "rain", "fog"]),
    "evolve": dict(mode="evolve", pool=["clear", "cloudy", "rain", "fog"]),
}

SENSOR_KEYS = {"optical": "optical", "ir": "ir", "radar": "radar", "lidar": "lidar"}
RADAR_BANDS = ["X", "Ku", "K", "Ka", "W", "W94"]

_JOBS = {}
_LOCK = threading.Lock()
_RE_PROG = re.compile(r"frame\s+(\d+)/(\d+)")


def _now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


class Job:
    def __init__(self, jid, name, out_dir, cfg_path, args):
        self.id = jid
        self.name = name
        self.out = out_dir
        self.cfg = cfg_path
        self.args = args
        self.state = "queued"
        self.started = time.time()
        self.finished = None
        self.done = 0
        self.total = 0
        self.log = []
        self.error = None
        self.proc = None

    def push(self, line):
        line = line.rstrip()
        if not line:
            return
        self.log.append(line)
        if len(self.log) > 400:
            self.log = self.log[-400:]
        m = _RE_PROG.search(line)
        if m:
            self.done, self.total = int(m.group(1)), int(m.group(2))

    def as_dict(self):
        return dict(id=self.id, name=self.name, state=self.state, out=self.out,
                    done=self.done, total=self.total,
                    seconds=round((self.finished or time.time()) - self.started, 1),
                    error=self.error, log=self.log[-40:], config=self.cfg)


def _spawn(job, argv):
    def run():
        job.state = "running"
        try:
            env = dict(os.environ)
            env["PYTHONPATH"] = ROOT
            env.setdefault("PYTHONIOENCODING", "utf-8")
            p = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                 errors="replace", bufsize=1)
            job.proc = p
            for line in p.stdout:
                job.push(line)
            rc = p.wait()
            job.state = "done" if rc == 0 else "failed"
            if rc != 0:
                job.error = "exit code %d" % rc
        except Exception as exc:                    # pragma: no cover
            job.state = "failed"
            job.error = "%s: %s" % (type(exc).__name__, exc)
            job.log.append(traceback.format_exc()[-800:])
        finally:
            job.finished = time.time()

    t = threading.Thread(target=run, daemon=True)
    t.start()


def _load_preset(key):
    spec = PRESETS[key]
    path = os.path.join(SCENES, spec["scene"])
    with open(path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    return cfg


def _apply(cfg, body):
    """Overlay the UI choices onto a preset config."""
    size = str(body.get("size") or "medium")
    px = (body.get("target_px") or PRESETS[body["preset"]]["sizes"].get(size)
          or [10, 45])
    sk = cfg.setdefault("clip", {}).setdefault("seeker", {})
    sk["target_px"] = [float(min(px)), float(max(px))]
    sk["closing"] = str(body.get("closing") or "in")

    wmode = str(body.get("weather") or "clear")
    cfg["weather"] = dict(WEATHER_MODES.get(wmode, WEATHER_MODES["clear"]))

    fps = float(body.get("fps") or 10.0)
    frames = int(body.get("frames") or 300)
    cfg.setdefault("frames", {})["n"] = frames
    cfg["frames"]["dt"] = round(1.0 / max(fps, 0.1), 4)
    fpc = int(body.get("frames_per_clip") or 50)
    cfg["clip"]["frames_per_clip"] = max(2, min(fpc, frames))

    band = body.get("radar_band")
    if band in RADAR_BANDS:
        cfg["sensors"]["radar"]["band"] = band
    seed = body.get("seed")
    if seed not in (None, ""):
        cfg["seed"] = int(seed)

    # The config is written into the run directory, so a relative mesh path like
    # ../assets/models/rover.obj no longer resolves. Absolute paths keep a run
    # reproducible no matter where its config is copied.
    for o in cfg.get("objects", []):
        m = o.get("model")
        if m and not os.path.isabs(m):
            o["model"] = os.path.normpath(os.path.join(SCENES, m))
    return cfg


def _sensor_arg(body):
    want = body.get("sensors") or ["ir"]
    out = []
    if "optical" in want:
        out.append("optical")
    if "ir" in want:
        out.append("ir")
    if "radar" in want:
        out.append("radar")
    if "lidar" in want:
        out.append("lidar")
    return ",".join(out) or "ir"


def start_job(body, preview=False):
    key = str(body.get("preset") or "air_uav")
    if key not in PRESETS:
        raise ValueError("unknown preset: " + key)
    cfg = _apply(_load_preset(key), body)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    tag = "preview_" if preview else "ds_"
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(body.get("name") or ""))[:40]
    jid = "%s%s_%s" % (tag, stamp, key)
    if name:
        jid += "_" + name
    out_dir = os.path.join(RUNS, jid)
    os.makedirs(out_dir, exist_ok=True)
    cfg_path = os.path.join(out_dir, "scene.json")
    with open(cfg_path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, ensure_ascii=False, indent=1)

    frames = int(body.get("frames") or 300)
    if preview:
        frames = max(4, min(frames, 8))
    sensors = _sensor_arg(body)
    if preview and "ir" not in sensors:
        sensors = "ir" if sensors == "" else sensors
    argv = [sys.executable, "-m", "simkit", "gen", "-c", cfg_path, "-o", out_dir,
            "-n", str(frames), "-s", sensors]
    if cfg.get("seed") is not None:
        argv += ["--seed", str(int(cfg["seed"]))]
    job = Job(jid, key + ("" if not name else "/" + name), out_dir, cfg_path, argv)
    job.total = frames
    with _LOCK:
        _JOBS[jid] = job
    _spawn(job, argv)
    return job


def _run_dir(jid):
    j = _JOBS.get(jid)
    if j:
        return j.out
    cand = os.path.join(RUNS, jid)
    return cand if os.path.isdir(cand) else None


def _safe_join(base, rel):
    rel = unquote(rel).replace("\\", "/").lstrip("/")
    p = os.path.normpath(os.path.join(base, rel))
    if not p.startswith(os.path.normpath(base)):
        return None
    return p


def _list_datasets():
    out = []
    if not os.path.isdir(RUNS):
        return out
    for nm in sorted(os.listdir(RUNS), reverse=True):
        d = os.path.join(RUNS, nm)
        st_p = os.path.join(d, "stats.json")
        if not os.path.isdir(d):
            continue
        st = {}
        if os.path.exists(st_p):
            try:
                st = json.load(open(st_p, encoding="utf-8"))
            except (OSError, ValueError):
                st = {}
        mods = sorted(f for f in os.listdir(d)
                      if f.startswith("annotations_") and f.endswith("_coco.json"))
        out.append(dict(id=nm, dir=d, frames=st.get("frames"),
                        modalities=[m[len("annotations_"):-len("_coco.json")] for m in mods],
                        optical=st.get("optical_images"),
                        ir=st.get("ir_images"), lidar=st.get("lidar_images"),
                        radar=st.get("radar_frames"),
                        size_mb=round(sum(os.path.getsize(os.path.join(dp, f))
                                          for dp, _s, fs in os.walk(d) for f in fs) / 1048576.0, 1)))
    return out


class Handler(BaseHTTPRequestHandler):
    server_version = "simkit-studio"

    def log_message(self, fmt, *a):        # keep the console for the run log
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        raw = self.rfile.read(n)
        try:
            return json.loads(raw.decode("utf-8"))
        except ValueError:
            return {}

    def do_GET(self):
        u = urlparse(self.path)
        p = u.path
        if p == "/favicon.ico":
            return self._send(204, b"", "image/x-icon")
        if p in ("/", "/index.html"):
            return self._static("index.html", "text/html; charset=utf-8")
        if p.startswith("/api/img/"):
            rest = p[len("/api/img/"):]
            jid, _, rel = rest.partition("/")
            base = _run_dir(jid)
            if not base:
                return self._send(404, dict(error="no such run"))
            fp = _safe_join(base, rel)
            if not fp or not os.path.isfile(fp):
                return self._send(404, dict(error="no such file"))
            ctype = "image/png" if fp.lower().endswith(".png") else "application/json"
            with open(fp, "rb") as fh:
                return self._send(200, fh.read(), ctype)
        if p == "/api/presets":
            return self._send(200, dict(
                presets=[dict(key=k, title=v["title"], note=v["note"],
                              cls=v["cls"], fov=v["fov"],
                              sizes=v["sizes"]) for k, v in PRESETS.items()],
                weather=list(WEATHER_MODES.keys()),
                sensors=list(SENSOR_KEYS.keys()),
                radar_bands=RADAR_BANDS,
                runs=RUNS))
        if p == "/api/jobs":
            with _LOCK:
                js = [j.as_dict() for j in _JOBS.values()]
            return self._send(200, dict(jobs=sorted(js, key=lambda j: j["id"], reverse=True)))
        if p.startswith("/api/job/"):
            jid = p[len("/api/job/"):].strip("/")
            j = _JOBS.get(jid)
            if not j:
                return self._send(404, dict(error="no such job"))
            return self._send(200, j.as_dict())
        if p == "/api/datasets":
            return self._send(200, dict(datasets=_list_datasets()))
        if p == "/api/plan":
            q = parse_qs(u.query)
            key = (q.get("preset") or ["air_uav"])[0]
            body = dict(preset=key, frames=int((q.get("frames") or ["100"])[0]))
            try:
                from simkit import plan, scene as sk_scene
                cfg = _apply(_load_preset(key), body)
                tmp = os.path.join(ROOT, "out", "_studio_plan.json")
                os.makedirs(os.path.dirname(tmp), exist_ok=True)
                with open(tmp, "w", encoding="utf-8") as fh:
                    json.dump(cfg, fh, ensure_ascii=False)
                loaded = sk_scene.load_config(tmp)
                rows = plan.summary(loaded, int(body["frames"]), loaded.get("seed", 0),
                                    loaded["frames"]["dt"])
                return self._send(200, dict(clips=rows))
            except Exception as exc:
                return self._send(200, dict(clips=[], error=str(exc)))
        if p.startswith("/api/dataset/"):
            jid = p[len("/api/dataset/"):].strip("/")
            base = _run_dir(jid)
            if not base:
                return self._send(404, dict(error="no such run"))
            info = dict(id=jid, dir=base, stats={}, meta={}, samples={})
            for nm, key in (("stats.json", "stats"), ("meta.json", "meta")):
                fp = os.path.join(base, nm)
                if os.path.exists(fp):
                    try:
                        info[key] = json.load(open(fp, encoding="utf-8"))
                    except (OSError, ValueError):
                        pass
            for mod, rel in (("optical", "images/optical"), ("mwir", "images/ir_preview_mwir"),
                             ("lwir", "images/ir_preview_lwir"), ("radar_ra", "images/radar_ra"),
                             ("lidar", "lidar/range")):
                d = os.path.join(base, rel)
                if os.path.isdir(d):
                    files = sorted(f for f in os.listdir(d) if f.endswith(".png"))
                    info["samples"][mod] = files[:6]
            return self._send(200, info)
        return self._static(p.lstrip("/"), None)

    def _static(self, rel, ctype):
        rel = rel or "index.html"
        fp = _safe_join(UI, rel)
        if not fp or not os.path.isfile(fp):
            return self._send(404, dict(error="not found: " + rel))
        ext = os.path.splitext(fp)[1].lower()
        mime = {".html": "text/html; charset=utf-8", ".js": "application/javascript; charset=utf-8",
                ".css": "text/css; charset=utf-8", ".png": "image/png",
                ".json": "application/json; charset=utf-8"}.get(ext, "application/octet-stream")
        with open(fp, "rb") as fh:
            return self._send(200, fh.read(), mime)

    def do_POST(self):
        u = urlparse(self.path)
        if u.path == "/api/generate":
            try:
                job = start_job(self._body(), preview=False)
            except Exception as exc:
                return self._send(400, dict(error=str(exc)))
            return self._send(200, job.as_dict())
        if u.path == "/api/preview":
            try:
                job = start_job(self._body(), preview=True)
            except Exception as exc:
                return self._send(400, dict(error=str(exc)))
            return self._send(200, job.as_dict())
        if u.path.startswith("/api/job/") and u.path.endswith("/cancel"):
            jid = u.path[len("/api/job/"):-len("/cancel")].strip("/")
            j = _JOBS.get(jid)
            if not j:
                return self._send(404, dict(error="no such job"))
            if j.proc and j.proc.poll() is None:
                j.proc.terminate()
                j.state = "cancelled"
            return self._send(200, j.as_dict())
        return self._send(404, dict(error="unknown endpoint"))


def serve(port=8808, host="127.0.0.1"):
    os.makedirs(RUNS, exist_ok=True)
    httpd = ThreadingHTTPServer((host, int(port)), Handler)
    url = "http://%s:%d/" % (host, int(port))
    print("simkit studio -> %s" % url)
    print("runs land in %s" % RUNS)
    print("press Ctrl+C to stop")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()
