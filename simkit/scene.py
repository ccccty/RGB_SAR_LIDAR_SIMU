"""Scene assembly: JSON config -> triangle world with per-face object/material ids."""
import json
import os

import numpy as np

from . import materials as matlib
from .bvh import BVH
from .mesh import (Mesh, box, cylinder, clean_faces, decimate, load_obj, normalize_to_height,
                   plane, rot_x, rot_y, rot_z, sea)

SIGMA = 5.670374419e-8


class World:
    """Flattened triangle soup plus bookkeeping for labels."""

    def __init__(self, verts, faces, face_obj, face_mat, objs, *, face_source_material=None,
                 face_group=None, face_local_centers=None):
        self.verts = verts
        self.faces = faces
        self.face_obj = face_obj      # (F,) int, -1 = background
        self.face_mat = face_mat      # (F,) int index into MATERIALS list
        self.objs = objs              # list of dicts
        self.face_source_material = (np.full(len(faces), "", dtype=object) if face_source_material is None
                                     else np.asarray(face_source_material, dtype=object))
        self.face_group = (np.full(len(faces), "", dtype=object) if face_group is None
                           else np.asarray(face_group, dtype=object))
        self.face_local_centers = (np.asarray(verts)[np.asarray(faces)].mean(axis=1)
                                   if face_local_centers is None else np.asarray(face_local_centers, float))
        self._bvh = None
        self._face_normals = None
        self._face_areas = None

    @property
    def bvh(self):
        if self._bvh is None:
            self._bvh = BVH(self.verts, self.faces)
        return self._bvh

    # Every sensor that needs geometry recomputed unit normals and triangle
    # areas for itself, and radar/sar already probe for these methods with
    # hasattr -- which could never succeed, so the same cross product ran once
    # per sensor per frame.  Caching them here makes those probes work and
    # makes the private copies redundant.
    def face_normals(self):
        """Unit normal of every triangle, cached for the world's lifetime."""
        if self._face_normals is None:
            tri = np.asarray(self.verts)[np.asarray(self.faces)]
            n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            length = np.linalg.norm(n, axis=1)
            self._face_normals = n / np.maximum(length, 1e-12)[:, None]
        return self._face_normals

    def face_areas(self):
        """Area of every triangle, cached; half the cross-product magnitude."""
        if self._face_areas is None:
            tri = np.asarray(self.verts)[np.asarray(self.faces)]
            n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            self._face_areas = 0.5 * np.linalg.norm(n, axis=1)
        return self._face_areas


def _reshape(shape, size, center):
    if shape == "box":
        return box(size, (0, 0, 0))
    if shape == "cylinder":
        return cylinder(size[0], size[2], (0, 0, 0))
    if shape == "sphere":
        # ECM bodies (a chaff bloom, a burning flare) are spheres. Imported here
        # rather than at module scope because ecm itself imports scene.
        from . import ecm as _ecmlib
        radius = max(0.5 * float(min(size)), 1e-3)
        # Level 2: a 2 m radius cloud rendered with the twenty-face level-1 ball
        # reads as a golf ball, not a chaff bloom.
        verts, faces = _ecmlib.icosphere(radius, 2)
        return Mesh(verts, faces, name="sphere")
    raise ValueError("unknown shape " + shape)


def load_object_mesh(spec, base_dir):
    shape = spec.get("shape", "box")
    # A spec that names a model file but not shape="obj" used to fall through to
    # the box branch, rendering a bounding box with the right size and the wrong
    # radar return. That failure is silent, so it is refused here instead.
    if "model" in spec and str(shape).lower() != "obj":
        raise ValueError(
            "object %r declares a model but shape is not 'obj' (got %r); "
            "refusing to substitute a box" % (spec.get("name", "?"), shape))
    if shape == "obj":
        path = spec["model"]
        if not os.path.isabs(path):
            path = os.path.join(base_dir, path)
        m = load_obj(path)
        m.mat = spec.get("mat", "painted_metal")
        return m
    return _reshape(shape, spec.get("size", [1, 1, 1]), (0, 0, 0))


# Prepared meshes keyed by everything that changes the result.  A scene that
# jitters an object's pose reuse the same entry because pose is applied later,
# in build_world, not here.
_PREPARED_MESHES = {}


def _prepare_key(spec, base_dir):
    """Identity of a prepared mesh: the file plus every transform applied to it."""
    model = spec.get("model")
    if model and not os.path.isabs(str(model)):
        model = os.path.normpath(os.path.join(base_dir, str(model)))
    scale = spec.get("scale")
    if isinstance(scale, (list, tuple)):
        scale = tuple(float(v) for v in scale)
    elif scale is not None:
        scale = float(scale)
    height = spec.get("height")
    return (str(spec.get("shape", "box")), model,
            int(spec["decimate"]) if spec.get("decimate") else None,
            scale, float(height) if height is not None else None,
            tuple(float(v) for v in spec.get("size", [1, 1, 1])),
            str(spec.get("mat", "painted_metal")))


def prepare_mesh(spec, base_dir):
    """Load a spec's mesh with decimation, scaling and height normalisation applied.

    Both the numpy world builder and the Blender renderer go through this, so the
    optical pass and the physics passes can never disagree about an object's size.
    Returns a mesh recentred on x/y with its base at z=0, plus its local bbox.

    The result depends only on the model file plus decimate/scale/height, all of
    which are fixed per scene, so it is cached per process.  Without the cache a
    340-frame clip reloads and decimates every OBJ once per frame; a 20k-triangle
    model made that the dominant cost of a render.
    """
    key = _prepare_key(spec, base_dir)
    hit = _PREPARED_MESHES.get(key)
    if hit is not None:
        return hit
    m = load_object_mesh(spec, base_dir)
    if spec.get("decimate"):
        m = decimate(m, target_tris=int(spec["decimate"]))
    if "scale" in spec:
        s = spec["scale"]
        s = [float(s)] * 3 if isinstance(s, (int, float)) else [float(v) for v in s]
        m = m.with_geometry(verts=m.v * np.array(s))
    if "height" in spec:
        m = normalize_to_height(m, float(spec["height"]))
    lo, hi = m.bbox
    c_local = (lo + hi) / 2.0
    shift = np.array([c_local[0], c_local[1], lo[2]])
    m = m.with_geometry(verts=m.v - shift)
    m = clean_faces(m)
    out = (m, (lo - shift), (hi - shift))
    _PREPARED_MESHES[key] = out
    return out


def build_world(cfg, t=0.0, jitter=None):
    """Assemble the world at time t. jitter: optional dict of per-object pose offsets.

    Only geometry is built here. Sensor-side presentation keys -- notably
    optical.sky_image and ir.background_image -- are never read, rewritten or
    whitelisted away, so the renderers receive them unchanged in the same cfg
    (path resolution lives in simkit/ir.py and blender/render_optical.py).
    """
    mat_names = sorted(matlib.MATERIALS)
    mat_index = {n: i for i, n in enumerate(mat_names)}

    verts_all, faces_all, fobj, fmat = [], [], [], []
    source_mats, source_groups, local_centers = [], [], []
    objs = []
    voff = 0
    fo = 0

    def add(mesh, obj_id, mat_name, spec=None, local=None):
        nonlocal voff, fo
        verts_all.append(mesh.v)
        faces_all.append(mesh.f + voff)
        fobj.append(np.full(len(mesh.f), obj_id, np.int32))
        physical, report = matlib.physical_face_materials(mesh, spec or {"mat": mat_name})
        fmat.append(np.array([mat_index[name] for name in physical], np.int32))
        source_mats.append(mesh.face_source_material)
        source_groups.append(mesh.face_group)
        local_centers.append(mesh.centers if local is None else local)
        voff += len(mesh.v)
        fo += len(mesh.f)
        return report

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
        pitch, roll = float(spec.get("pitch", 0.0)), float(spec.get("roll", 0.0))
        pos = pos + vel * t
        if jitter and spec.get("name") in jitter:
            j = jitter[spec["name"]]
            pos = pos + np.array(j.get("dpos", [0, 0, 0]), float)
            yaw += float(j.get("dyaw", 0.0))
            pitch, roll = float(j.get("pitch", pitch)), float(j.get("roll", roll))
            # A trajectory supplies the instantaneous velocity; without this the
            # Doppler shift and the MTI gate would use the static "vel" from the
            # config and contradict the path the target is actually flying.
            if "vel" in j:
                vel = np.array(j["vel"], float)

        R = rot_z(yaw) @ rot_y(-pitch) @ rot_x(roll)
        wm = m.transformed(R=R, t=pos)
        wm.name = spec.get("name", "obj%d" % i)
        wm.mat = spec.get("mat", "painted_metal")
        material_report = add(wm, i, wm.mat, spec=spec, local=m.centers)
        material_report["geometry_cleanup"] = m.cleanup_report

        corners = np.array([[x, y, z] for x in (lo[0], hi[0])
                            for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        corners = corners @ R.T + pos
        objs.append(dict(
            id=i, name=spec.get("name", "obj%d" % i), cls=spec.get("cls", "target"),
            mat=spec.get("mat", "painted_metal"), pos=pos, vel=vel, yaw=yaw, pitch=pitch, roll=roll,
            size=size_local, corners=corners,
            temp_c=spec.get("temp_c"), internal_heat=spec.get("internal_heat"),
            rcs_db=spec.get("rcs_db"), visible=bool(spec.get("visible", True)),
            # scenery objects are geometry only: they cast shadows, occlude and
            # return radar energy, but never appear in a label or a dataset class
            label=bool(spec.get("label", True)),
            thermal_regions=spec.get("thermal_regions", []),
            source_material_report=material_report,
        ))

    verts = np.vstack(verts_all)
    faces = np.vstack(faces_all)
    world = World(verts, faces, np.concatenate(fobj), np.concatenate(fmat), objs,
                  face_source_material=np.concatenate(source_mats),
                  face_group=np.concatenate(source_groups),
                  face_local_centers=np.vstack(local_centers))
    # Scene-level countermeasure bodies are geometry, not a post-process: the
    # chaff cloud, decoy and corner reflector have to occlude in the rasteriser,
    # return lidar pulses and appear in the radar ray cast. They are appended
    # here so every sensor sees one consistent world. Off unless the scene
    # carries an "ecm" block, so existing scenes are untouched.
    if (cfg.get("ecm") or {}).get("enabled"):
        from . import ecm as _ecmlib
        world, _added = _ecmlib.attach(world, cfg, float(cfg.get("_ecm_t", t)))
    return world


def load_config(path):
    with open(path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    # dataset._render_config stamps the ORIGINAL scene directory into the
    # side-car config before writing it to out/<run>/. Keep it: overwriting it
    # with this file's own directory makes "../assets/..." resolve to
    # out/<run>/../assets, so relative background plates silently vanish on the
    # Blender side while absolute paths keep working.
    if not cfg.get("_base_dir"):
        cfg["_base_dir"] = os.path.dirname(os.path.abspath(path))
    return cfg


def sensor_basis(pos, look_at, up=(0, 0, 1), roll_deg=0.0):
    p = np.asarray(pos, float)
    f = np.asarray(look_at, float) - p
    f = f / (np.linalg.norm(f) + 1e-12)
    u = np.asarray(up, float)
    if abs(float(np.dot(f, u))) > 0.999:
        u = np.array([0.0, 1.0, 0.0])
    r = np.cross(f, u)
    r = r / (np.linalg.norm(r) + 1e-12)
    u2 = np.cross(r, f)
    # A banked seeker rolls the image about the optical axis. Blender applies
    # roll to the optical camera, so every numpy renderer and every projection
    # must apply it too or the modalities stop being registered: at 24.7 deg of
    # bank the same target sat 40 px apart in the optical and the thermal frame
    # (measured), which silently poisons any fused training set.
    roll = float(roll_deg or 0.0)
    if abs(roll) > 1e-9:
        t = np.radians(roll)
        c, s_ = np.cos(t), np.sin(t)
        r, u2 = r * c + u2 * s_, -r * s_ + u2 * c
    return p, r, u2, f