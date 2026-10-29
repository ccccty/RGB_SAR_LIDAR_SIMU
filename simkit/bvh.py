"""Axis-aligned BVH with vectorised ray casting (numpy only)."""
import numpy as np


class BVH:
    def __init__(self, verts, faces, leaf_size=8):
        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        self.v0, self.v1, self.v2 = v0, v1, v2
        self.faces = faces
        self.cent = (v0 + v1 + v2) / 3.0
        tri_min = np.minimum(np.minimum(v0, v1), v2)
        tri_max = np.maximum(np.maximum(v0, v1), v2)
        self.tri_min, self.tri_max = tri_min, tri_max

        n = len(faces)
        self.order = np.arange(n)
        self.bmin, self.bmax = [], []
        self.left, self.right, self.start, self.count = [], [], [], []
        stack = [(0, n)]
        # iterative build: create node placeholders then fill children by index
        self._build(0, n, tri_min, tri_max, leaf_size)

        self.bmin = np.array(self.bmin, float).reshape(-1, 3)
        self.bmax = np.array(self.bmax, float).reshape(-1, 3)
        self.left = np.array(self.left, np.int32)
        self.right = np.array(self.right, np.int32)
        self.start = np.array(self.start, np.int32)
        self.count = np.array(self.count, np.int32)

    def _node_bounds(self, lo, hi):
        idx = self.order[lo:hi]
        return self.tri_min[idx].min(axis=0), self.tri_max[idx].max(axis=0)

    def _build(self, lo, hi, tmin, tmax, leaf_size):
        node = len(self.bmin)
        bmin, bmax = self._node_bounds(lo, hi)
        self.bmin.append(bmin)
        self.bmax.append(bmax)
        self.left.append(-1)
        self.right.append(-1)
        self.start.append(-1)
        self.count.append(0)
        n = hi - lo
        if n <= leaf_size:
            self.start[node] = lo
            self.count[node] = n
            return node
        ext = bmax - bmin
        axis = int(np.argmax(ext))
        seg = self.cent[self.order[lo:hi], axis]
        mid = lo + n // 2
        part = np.argpartition(seg, n // 2)
        self.order[lo:hi] = self.order[lo:hi][part]
        l = self._build(lo, mid, tmin, tmax, leaf_size)
        r = self._build(mid, hi, tmin, tmax, leaf_size)
        self.left[node] = l
        self.right[node] = r
        return node

    # ---------- traversal ----------
    def ray_cast(self, origin, dirs):
        """origin (3,), dirs (N,3) unit vectors. Returns dict of (N,) arrays."""
        origin = np.asarray(origin, float)
        dirs = np.asarray(dirs, float)
        N = len(dirs)
        t_hit = np.full(N, np.inf)
        tri_hit = np.full(N, -1, np.int64)
        bmin, bmax = self.bmin, self.bmax
        left, right, start, count = self.left, self.right, self.start, self.count
        # inverse direction for slab test, tolerate zeros
        inv = 1.0 / np.where(np.abs(dirs) < 1e-12, 1e-12, dirs)
        for i in range(N):
            d = dirs[i]
            iv = inv[i]
            o = origin
            stack = [0]
            best = np.inf
            best_tri = -1
            while stack:
                node = stack.pop()
                # slab test
                t1 = (bmin[node] - o) * iv
                t2 = (bmax[node] - o) * iv
                tn = np.minimum(t1, t2).max()
                tf = np.maximum(t1, t2).min()
                if tf < max(tn, 0.0) or tn > best:
                    continue
                if self.count[node] > 0:
                    lo = start[node]
                    hi = lo + count[node]
                    idx = self.order[lo:hi]
                    th, ok = self._hit_tris(idx, o, d)
                    if ok.any():
                        j = int(np.argmin(th))
                        if th[j] < best:
                            best = float(th[j])
                            best_tri = int(idx[j])
                    continue
                l = left[node]
                r = right[node]
                if l >= 0:
                    stack.append(l)
                if r >= 0:
                    stack.append(r)
            t_hit[i] = best
            tri_hit[i] = best_tri
        hit = tri_hit >= 0
        tt = np.where(hit, t_hit, 0.0)
        pts = origin + dirs * tt[:, None]
        return {"t": t_hit, "tri": tri_hit, "hit": hit, "point": pts}

    def _hit_tris(self, idx, o, d):
        v0 = self.v0[idx]
        e1 = self.v1[idx] - v0
        e2 = self.v2[idx] - v0
        p = np.cross(d, e2)
        det = (e1 * p).sum(axis=1)
        ok = np.abs(det) > 1e-12
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        tv = o - v0
        u = (tv * p).sum(axis=1) * inv
        q = np.cross(tv, e1)
        v = (q * d).sum(axis=1) * inv
        t = (e2 * q).sum(axis=1) * inv
        ok &= (u >= 0) & (u <= 1) & (v >= 0) & (u + v <= 1) & (t > 1e-9)
        t = np.where(ok, t, np.inf)
        return t, ok

    def normal_at(self, tri):
        v0, v1, v2 = self.v0[tri], self.v1[tri], self.v2[tri]
        nrm = np.cross(v1 - v0, v2 - v0)
        ln = np.linalg.norm(nrm)
        return nrm / ln if ln > 0 else np.array([0.0, 0.0, 1.0])
