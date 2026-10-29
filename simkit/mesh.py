"""Triangle meshes: OBJ loading, primitives, transforms."""
import math

import numpy as np


def rot_z(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_x(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


class Mesh:
    """Triangle soup. Faces are (F,3) indices into verts."""

    def __init__(self, verts, faces, name="mesh", mat="concrete"):
        self.v = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
        self.f = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
        self.name = name
        self.mat = mat

    # ---- geometry ----
    def tri(self):
        return self.v[self.f]

    @property
    def centers(self):
        return self.v[self.f].mean(axis=1)

    @property
    def areas(self):
        t = self.v[self.f]
        return 0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)

    @property
    def normals(self):
        t = self.v[self.f]
        n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        ln = np.linalg.norm(n, axis=1, keepdims=True)
        ln[ln == 0] = 1.0
        return n / ln

    @property
    def bbox(self):
        return self.v.min(axis=0), self.v.max(axis=0)

    def transformed(self, R=None, t=None):
        R = np.eye(3) if R is None else np.asarray(R, float)
        t = np.zeros(3) if t is None else np.asarray(t, float)
        return Mesh(self.v @ R.T + t, self.f, self.name, self.mat)

    def copy(self):
        return Mesh(self.v.copy(), self.f.copy(), self.name, self.mat)


def box(size, center=(0, 0, 0)):
    sx, sy, sz = (np.asarray(size, float) / 2.0)
    cx, cy, cz = center
    v = np.array([
        [-sx, -sy, -sz], [sx, -sy, -sz], [sx, sy, -sz], [-sx, sy, -sz],
        [-sx, -sy, sz], [sx, -sy, sz], [sx, sy, sz], [-sx, sy, sz],
    ], float) + np.array([cx, cy, cz])
    f = np.array([
        [0, 2, 1], [0, 3, 2],   # bottom (normal -z)
        [4, 5, 6], [4, 6, 7],   # top
        [0, 1, 5], [0, 5, 4],   # -y
        [1, 2, 6], [1, 6, 5],   # +x
        [2, 3, 7], [2, 7, 6],   # +y
        [3, 0, 4], [3, 4, 7],   # -x
    ], np.int32)
    return Mesh(v, f)


def cylinder(radius, height, center=(0, 0, 0), n=24):
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    cx, cy, cz = center
    top = np.stack([radius * np.cos(ang) + cx, radius * np.sin(ang) + cy,
                    np.full(n, cz + height / 2)], axis=1)
    bot = top.copy()
    bot[:, 2] = cz - height / 2
    v = np.vstack([top, bot, [[cx, cy, cz + height / 2]], [[cx, cy, cz - height / 2]]])
    f = []
    for i in range(n):
        j = (i + 1) % n
        f.append([i, j, n + j])
        f.append([i, n + j, n + i])
        f.append([2 * n, i, j])
        f.append([2 * n + 1, n + j, n + i])
    return Mesh(v, np.array(f, np.int32))


def plane(size=200.0, z=0.0, nx=10, ny=10):
    xs = np.linspace(-size / 2, size / 2, nx + 1)
    ys = np.linspace(-size / 2, size / 2, ny + 1)
    v = np.array([[x, y, z] for y in ys for x in xs], float)
    f = []
    for iy in range(ny):
        for ix in range(nx):
            a = iy * (nx + 1) + ix
            b, c, d = a + 1, a + nx + 2, a + nx + 1
            f.append([a, b, c])
            f.append([a, c, d])
    return Mesh(v, np.array(f, np.int32))


def sea(size=400.0, z=0.0, nx=48, ny=48, wind_ms=8.0, seed=1, chop=0.6):
    """Wind-driven sea height field.

    Not a spectral wave model: a handful of sinusoids whose peak wavelength and
    significant height follow the fully-developed-sea scaling
        Hs ~ 0.021 * U^2      (m),   lam_p ~ 0.6 * U^2  (m)
    which is the same approximation Cox-Munk slope statistics are quoted with.
    It is enough for (a) a believable horizon and glints in the visible band,
    (b) tilted facets that give the radar wind-dependent backscatter, and
    (c) slope/sky-view geometry for the IR band.
    """
    rng = np.random.default_rng(int(seed))
    u = max(float(wind_ms), 0.5)
    hs = min(0.021 * u * u, 0.25 * size / max(nx, ny))
    lam_p = max(0.6 * u * u, 4.0)
    xs = np.linspace(-size / 2, size / 2, nx + 1)
    ys = np.linspace(-size / 2, size / 2, ny + 1)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    H = np.zeros_like(X)
    base_dir = rng.uniform(0.0, 2.0 * math.pi)
    n_comp = 6
    for i in range(n_comp):
        lam = lam_p * (0.62 ** i)
        k = 2.0 * math.pi / max(lam, 1e-3)
        theta = base_dir + float(rng.normal(0.0, 0.6))
        amp = (hs / 2.0) * (0.55 ** i) * (1.0 + 0.3 * float(rng.normal()))
        H += amp * np.sin(k * (X * math.cos(theta) + Y * math.sin(theta))
                          + float(rng.uniform(0, 2 * math.pi)))
    # sharpen crests a little (real waves are not sinusoids)
    H = H * (1.0 + chop * 0.5 * np.tanh(H / max(hs / 2.0, 1e-6)))
    V = np.array([[X[i, j], Y[i, j], z + H[i, j]]
                  for j in range(ny + 1) for i in range(nx + 1)], float)
    f = []
    for j in range(ny):
        for i in range(nx):
            a = j * (nx + 1) + i
            b, c, d = a + 1, a + nx + 2, a + nx + 1
            f.append([a, b, c])
            f.append([a, c, d])
    return Mesh(V, np.array(f, np.int32))


def decimate(mesh, target_tris=None, grid=None):
    """Vertex-clustering decimation: weld vertices inside a uniform grid.

    Cheap (one pass, no topology) and adequate for scenery and for imported
    high-poly CAD models: the silhouette is preserved to the grid size, which is
    what matters for occlusion, shadows and radar facets. Triangles that
    collapse to a line or a point are dropped.
    """
    if target_tris is None and grid is None:
        return mesh
    lo, hi = mesh.bbox
    diag = max(float(np.linalg.norm(hi - lo)), 1e-9)

    def cluster(cell):
        key = np.floor((mesh.v - lo) / cell).astype(np.int64)
        _, inv = np.unique(key, axis=0, return_inverse=True)
        nv = int(inv.max()) + 1
        cnt = np.bincount(inv).astype(np.float64)
        verts = np.zeros((nv, 3))
        for k in range(3):
            verts[:, k] = np.bincount(inv, weights=mesh.v[:, k]) / cnt
        f = inv[mesh.f]
        keep = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
        return verts, f[keep]

    if grid is None:
        # bisect the cell size so the surviving face count lands near the target
        target = max(int(target_tris), 4)
        a, b = diag / 400.0, diag / 2.0
        for _ in range(9):
            c = 0.5 * (a + b)
            _, f = cluster(c)
            if len(f) > target:
                a = c
            else:
                b = c
        verts, f = cluster(b)
    else:
        verts, f = cluster(max(float(grid), 1e-9))
    if len(f) == 0:
        return mesh
    out = Mesh(verts, f, mesh.name, mesh.mat)
    out.mtl_normals = getattr(mesh, "mtl_normals", None)
    return out


def load_obj(path, name=None):
    verts, faces = [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if line.startswith("v "):
                p = line.split()
                verts.append([float(p[1]), float(p[2]), float(p[3])])
            elif line.startswith("f "):
                idx = []
                for tok in line.split()[1:]:
                    i = int(tok.split("/")[0])
                    idx.append(i - 1 if i > 0 else len(verts) + i)
                for k in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[k], idx[k + 1]])
    if not verts or not faces:
        raise ValueError("empty or unsupported OBJ: " + str(path))
    return Mesh(np.array(verts, float), np.array(faces, np.int32),
                name or path.split("\\")[-1].replace(".obj", ""))


def normalize_to_height(mesh, height, ground_z=0.0):
    """Scale mesh so its Z extent equals height, put its base at ground_z, centered in XY."""
    lo, hi = mesh.bbox
    ext = hi - lo
    if ext[2] <= 0:
        return mesh
    s = height / ext[2]
    v = mesh.v * s
    lo, hi = v.min(axis=0), v.max(axis=0)
    c = (lo + hi) / 2.0
    v = v - np.array([c[0], c[1], lo[2] - ground_z])
    return Mesh(v, mesh.f, mesh.name, mesh.mat)
