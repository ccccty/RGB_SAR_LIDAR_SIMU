"""Scene assembly: JSON config -> triangle world with per-face object/material ids."""
import json
import os

import numpy as np

from . import materials as matlib
from .bvh import BVH
from .mesh import (Mesh, box, cylinder, decimate, load_obj, normalize_to_height,
                   plane, rot_z, sea)

SIGMA = 5.670374419e-8


class World:
    """Flattened triangle soup plus bookkeeping for labels."""

    def __init__(self, verts, faces, face_obj, face_mat, objs):
        self.verts = verts
        self.faces = faces
        self.face_obj = face_obj      # (F,) int, -1 = background
        self.face_mat = face_mat      # (F,) int index into MATERIALS list
        self.objs = objs              # list of dicts
        self._bvh = None

    @property
    def bvh(self):
        if self._bvh is None:
            self._bvh = BVH(self.verts, self.faces)
        return self._bvh


def _reshape(shape, size, center):
    if shape == "box":
        return box(size, (0, 0, 0))
    if shape == "cylinder":
        return cylinder(size[0], size[2], (0, 0, 0))
    raise ValueError("unknown shape " + shape)


def load_object_mesh(spec, base_dir):
    shape = spec.get("shape", "box")
    if shape == "obj":
        path = spec["model"]
        if not os.path.isabs(path):
            path = os.path.join(base_dir, path)
        m = load_obj(path)
        m.mat = spec.get("mat", "painted_metal")
        return m
    return _reshape(shape, spec.get("size", [1, 1, 1]), (0, 0, 0))


def prepare_mesh(spec, base_dir):
    """Load a spec's mesh with decimation, scaling and height normalisation applied.

    Both the numpy world builder and the Blender renderer go through this, so the
    optical pass and the physics passes can never disagree about an object's size.
    Returns a mesh recentred on x/y with its base at z=0, plus its local bbox.
    """
    m = load_object_mesh(spec, base_dir)
    if spec.get("decimate"):
        m = decimate(m, target_tris=int(spec["decimate"]))
    if "scale" in spec:
        s = spec["scale"]
        s = [float(s)] * 3 if isinstance(s, (int, float)) else [float(v) for v in s]
        m = Mesh(m.v * np.array(s), m.f, m.name, m.mat)
    if "height" in spec:
        m = normalize_to_height(m, float(spec["height"]))
    lo, hi = m.bbox
    c_local = (lo + hi) / 2.0
    shift = np.array([c_local[0], c_local[1], lo[2]])
    m = Mesh(m.v - shift, m.f, m.name, m.mat)
    return m, (lo - shift), (hi - shift)


def build_world(cfg, t=0.0, jitter=None):
    """Assemble the world at time t. jitter: optional dict of per-object pose offsets."""
    mat_names = sorted(matlib.MATERIALS)
    mat_index = {n: i for i, n in enumerate(mat_names)}

    verts_all, faces_all, fobj, fmat = [], [], [], []
    objs = []
    voff = 0
    fo = 0

    def add(mesh, obj_id, mat_name):
        nonlocal voff, fo
        verts_all.append(mesh.v)
        faces_all.append(mesh.f + voff)
        fobj.append(np.full(len(mesh.f), obj_id, np.int32))
        fmat.append(np.full(len(mesh.f), mat_index[mat_name], np.int32))
        voff += len(mesh.v)
        fo += len(mesh.f)

    # ground / sea (optional: air scenes may disable it or push it far below)
    g = cfg.get("ground", {})
    if g.get("enabled", True):
        gsize = float(g.get("size", 120.0))
        gz = float(g.get("z", 0.0))
        if str(g.get("type", "ground")).lower() in ("sea", "ocean", "water"):
            w = cfg.get("_weather") or {}
            wind = float(g.get("wind_ms", w.get("wind_ms", 8.0)))
            gmesh = sea(gsize, gz, int(g.get("nx", 48)), int(g.get("ny", 48)),
                        wind_ms=wind, seed=int(g.get("seed", 7)))
            add(gmesh, -1, g.get("mat", "seawater"))
        else:
            gmesh = plane(gsize, gz, 12, 12)
            add(gmesh, -1, g.get("mat", "asphalt"))

    for i, spec in enumerate(cfg.get("objects", [])):
        m, lo, hi = prepare_mesh(spec, cfg.get("_base_dir", "."))
        size_local = hi - lo

        pos = np.array(spec.get("pos", [0, 0, 0]), float)
        vel = np.array(spec.get("vel", [0, 0, 0]), float)
        yaw = float(spec.get("rot", 0.0))
        pos = pos + vel * t
        if jitter and spec.get("name") in jitter:
            j = jitter[spec["name"]]
            pos = pos + np.array(j.get("dpos", [0, 0, 0]), float)
            yaw += float(j.get("dyaw", 0.0))

        R = rot_z(yaw)
        wm = Mesh(m.v @ R.T + pos, m.f, spec.get("name", "obj%d" % i),
                  spec.get("mat", "painted_metal"))
        add(wm, i, spec.get("mat", "painted_metal"))

        corners = np.array([[x, y, z] for x in (lo[0], hi[0])
                            for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        corners = corners @ R.T + pos
        objs.append(dict(
            id=i, name=spec.get("name", "obj%d" % i), cls=spec.get("cls", "target"),
            mat=spec.get("mat", "painted_metal"), pos=pos, vel=vel, yaw=yaw,
            size=size_local, corners=corners,
            temp_c=spec.get("temp_c"), internal_heat=spec.get("internal_heat"),
            rcs_db=spec.get("rcs_db"), visible=bool(spec.get("visible", True)),
            # scenery objects are geometry only: they cast shadows, occlude and
            # return radar energy, but never appear in a label or a dataset class
            label=bool(spec.get("label", True)),
        ))

    verts = np.vstack(verts_all)
    faces = np.vstack(faces_all)
    return World(verts, faces, np.concatenate(fobj), np.concatenate(fmat), objs)


def load_config(path):
    with open(path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    cfg["_base_dir"] = os.path.dirname(os.path.abspath(path))
    return cfg


def sensor_basis(pos, look_at, up=(0, 0, 1)):
    p = np.asarray(pos, float)
    f = np.asarray(look_at, float) - p
    f = f / (np.linalg.norm(f) + 1e-12)
    u = np.asarray(up, float)
    if abs(float(np.dot(f, u))) > 0.999:
        u = np.array([0.0, 1.0, 0.0])
    r = np.cross(f, u)
    r = r / (np.linalg.norm(r) + 1e-12)
    u2 = np.cross(r, f)
    return p, r, u2, f
