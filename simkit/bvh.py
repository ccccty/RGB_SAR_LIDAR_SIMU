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
        origin = np.broadcast_to(origin, dirs.shape).copy()
        return self.ray_cast_batch(origin, dirs)

    def ray_cast_batch(self, origins, dirs, leaf_size=8, max_stack=128):
        """origins (N,3), dirs (N,3). One ray per row, all traversed together.

        The scalar implementation below cost one Python-level traversal per ray,
        which dominated every timing in this project (a single IR frame spent
        45 s in the thermal shadow pass alone). Traversing every ray at once,
        with a per-ray explicit stack, removes that constant factor without
        changing a single hit.
        """
        origins = np.ascontiguousarray(origins, dtype=float)
        dirs = np.ascontiguousarray(dirs, dtype=float)
        N = len(dirs)
        if N == 0:
            z = np.zeros(0)
            return {"t": z, "tri": np.zeros(0, np.int64), "hit": np.zeros(0, bool),
                    "point": np.zeros((0, 3))}
        origins = np.broadcast_to(origins, dirs.shape).copy()

        bmin, bmax = self.bmin, self.bmax
        left, right, start, count = self.left, self.right, self.start, self.count

        inv = 1.0 / np.where(np.abs(dirs) < 1e-12, 1e-12, dirs)
        best = np.full(N, np.inf)
        best_tri = np.full(N, -1, np.int64)

        stack = np.zeros((max_stack, N), np.int32)
        sp = np.zeros(N, np.int32)
        node = np.zeros(N, np.int32)
        alive = np.ones(N, bool)
        K = leaf_size

        while True:
            idx = np.nonzero(alive)[0]
            if idx.size == 0:
                break
            nd = node[idx]
            o = origins[idx]
            iv = inv[idx]
            bb_lo = bmin[nd]
            bb_hi = bmax[nd]
            t1 = (bb_lo - o) * iv
            t2 = (bb_hi - o) * iv
            tn = np.minimum(t1, t2).max(axis=1)
            tf = np.maximum(t1, t2).min(axis=1)
            bnd = best[idx]
            miss = (tf < np.maximum(tn, 0.0)) | (tn > bnd)

            leaf = (~miss) & (count[nd] > 0)
            if leaf.any():
                li = np.nonzero(leaf)[0]
                lo = start[nd[li]].astype(np.int64)
                nleaf = count[nd[li]].astype(np.int64)
                kin = np.arange(K)
                tri_idx = self.order[np.minimum(lo[:, None] + kin[None, :], len(self.order) - 1)]
                valid = kin[None, :] < nleaf[:, None]
                th, ok = self._hit_tris_rays(tri_idx, o[li], dirs[idx[li]])
                ok &= valid
                th = np.where(ok, th, np.inf)
                j = np.argmin(th, axis=1)
                tmin = th[np.arange(li.size), j]
                better = tmin < best[idx[li]]
                if better.any():
                    bi = li[better]
                    best[idx[bi]] = tmin[better]
                    best_tri[idx[bi]] = tri_idx[np.arange(li.size)[better], j[better]]

            # everything that missed or was a leaf pops; internal nodes descend
            pop = miss | leaf
            if pop.any():
                pi = idx[pop]
                sp[pi] -= 1
                ok_pop = sp[pi] >= 0
                pi2 = pi[ok_pop]
                node[pi2] = stack[sp[pi2], pi2]
                dead = pi[~ok_pop]
                if dead.size:
                    alive[dead] = False

            desc = (~pop)
            if desc.any():
                di = idx[desc]
                dn = node[di]
                st = sp[di]
                stack[np.minimum(st, max_stack - 1), di] = right[dn]
                sp[di] = st + 1
                node[di] = left[dn]

        hit = best_tri >= 0
        tt = np.where(hit, best, 0.0)
        pts = origins + dirs * tt[:, None]
        return {"t": best, "tri": best_tri, "hit": hit, "point": pts}

    def _hit_tris_rays(self, idx, o, d):
        """idx (M,K), o (M,3), d (M,3) -> (M,K) distances and hits."""
        v0 = self.v0[idx]
        e1 = self.v1[idx] - v0
        e2 = self.v2[idx] - v0
        oo = o[:, None, :]
        dd = d[:, None, :]
        p = np.cross(dd, e2)
        det = (e1 * p).sum(axis=2)
        ok = np.abs(det) > 1e-12
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        tv = oo - v0
        u = (tv * p).sum(axis=2) * inv
        qv = np.cross(tv, e1)
        v = (qv * dd).sum(axis=2) * inv
        tt = (e2 * qv).sum(axis=2) * inv
        ok &= (u >= 0) & (u <= 1) & (v >= 0) & (u + v <= 1) & (tt > 1e-9)
        return np.where(ok, tt, np.inf), ok

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
