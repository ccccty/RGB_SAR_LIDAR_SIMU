"""Axis-aligned BVH with vectorised ray casting (numpy only)."""
import numpy as np


class BVH:
    # Build leaf size. 16 measured faster end-to-end than 8 or 4 on a 38k-face
    # scene in both ray workloads tried (thermal shadow rays and isotropic
    # lidar-style rays): the shallower tree builds in 100 ms instead of 165 ms
    # and casts faster, and the larger leaves only add triangle tests that the
    # 10-wide gather absorbs. The hit/miss answer is identical for every leaf
    # size (verified on three scenes), because the leaf size only changes the
    # acceleration structure.
    def __init__(self, verts, faces, leaf_size=16):
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
        self.leaf_size = int(leaf_size)
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

        # Edge vectors are per-triangle constants: _hit_tris_rays rebuilt
        # v1-v0 and v2-v0 for every gathered triangle of every ray. They cost
        # 2*F*3 doubles (1.8 MB at 38k faces) and hold exactly the doubles the
        # old expressions produced, so gathering them changes nothing.
        self.e1 = self.v1 - self.v0
        self.e2 = self.v2 - self.v0
        # A leaf never holds more than leaf_size triangles, and the split rule
        # can leave fewer (8 requested -> 5 observed on a 38k-face scene).
        # Traversal must gather at least the real maximum or it would silently
        # skip the tail of a leaf, but widening to leaf_size also tests slots
        # that are only padding. max_leaf is both correct and smaller.
        self.max_leaf = int(self.count.max()) if len(self.count) else 0

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

    def ray_cast_batch(self, origins, dirs, leaf_size=None, max_stack=128, chunk=None):
        """origins (N,3), dirs (N,3). One ray per row, all traversed together.

        The scalar implementation below cost one Python-level traversal per ray,
        which dominated every timing in this project (a single IR frame spent
        45 s in the thermal shadow pass alone). Traversing every ray at once,
        with a per-ray explicit stack, removes that constant factor without
        changing a single hit.

        leaf_size defaults to None, meaning "the tree's own maximum leaf
        occupancy". Passing a number is still accepted, but the gather is never
        allowed to be narrower than the real maximum (that would silently drop
        triangles from the tail of a leaf).

        Splitting the batch into cache-sized blocks was measured on the 38k-face
        thermal shadow pass and is kept OUT: at 19k rays it made the cast 25%
        slower (each block re-pays the per-iteration numpy call overhead while
        the traversal only runs a few dozen iterations), and bigger blocks were
        monotonically better. A single batch is both simpler and faster here.
        """
        origins = np.ascontiguousarray(origins, dtype=float)
        dirs = np.ascontiguousarray(dirs, dtype=float)
        N = len(dirs)
        if N == 0:
            z = np.zeros(0)
            return {"t": z, "tri": np.zeros(0, np.int64), "hit": np.zeros(0, bool),
                    "point": np.zeros((0, 3))}
        width = self.max_leaf if leaf_size is None else max(int(leaf_size), self.max_leaf)
        return self._cast_block(origins, dirs, width, max_stack)

    def _cast_block(self, origins, dirs, K, max_stack):
        """Traverse the whole ray batch; the traversal stack lives here."""
        if origins.shape != dirs.shape:
            origins = np.broadcast_to(origins, dirs.shape).copy()
        N = len(dirs)

        bmin, bmax = self.bmin, self.bmax
        left, right, start, count = self.left, self.right, self.start, self.count

        inv = 1.0 / np.where(np.abs(dirs) < 1e-12, 1e-12, dirs)
        best = np.full(N, np.inf)
        best_tri = np.full(N, -1, np.int64)

        stack = np.zeros((max_stack, N), np.int32)
        sp = np.zeros(N, np.int32)
        node = np.zeros(N, np.int32)
        alive = np.ones(N, bool)

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
        """idx (M,K), o (M,3), d (M,3) -> (M,K) distances and hits.

        Written one vector component at a time. np.cross builds each component
        as a1*b2 - a2*b1 and (x*y).sum(axis=2) reduces three terms left to
        right, so this is the same sequence of IEEE operations as the previous
        version -- it just no longer materialises four (M,K,3) temporaries,
        which measured 1.5x faster at the leaf widths this tree produces.
        """
        v0 = self.v0[idx]
        e1 = self.e1[idx]
        e2 = self.e2[idx]
        dx = d[:, None, 0]
        dy = d[:, None, 1]
        dz = d[:, None, 2]
        px = dy * e2[..., 2] - dz * e2[..., 1]
        py = dz * e2[..., 0] - dx * e2[..., 2]
        pz = dx * e2[..., 1] - dy * e2[..., 0]
        det = e1[..., 0] * px + e1[..., 1] * py + e1[..., 2] * pz
        ok = np.abs(det) > 1e-12
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        t0 = o[:, None, 0] - v0[..., 0]
        t1 = o[:, None, 1] - v0[..., 1]
        t2 = o[:, None, 2] - v0[..., 2]
        u = (t0 * px + t1 * py + t2 * pz) * inv
        qx = t1 * e1[..., 2] - t2 * e1[..., 1]
        qy = t2 * e1[..., 0] - t0 * e1[..., 2]
        qz = t0 * e1[..., 1] - t1 * e1[..., 0]
        v = (qx * dx + qy * dy + qz * dz) * inv
        tt = (e2[..., 0] * qx + e2[..., 1] * qy + e2[..., 2] * qz) * inv
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
