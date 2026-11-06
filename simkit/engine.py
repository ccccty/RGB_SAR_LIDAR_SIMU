"""Engine exhaust as geometry and a temperature field.

A jet in the thermal infrared is not an isothermal airframe. The order of
brightness is: exhaust plume, then the nozzle, then the skin. The plume is
often the strongest feature in a mid-wave frame, and it was completely absent:
every face of an aircraft carried one class temperature, so the target
rendered as a flat white cut-out.

This module adds both pieces as real geometry, so they occlude, get occluded
and rasterise through the same path as everything else:

  * a hot nozzle ring, selected from the rear of the airframe along its own
    velocity vector rather than from a per-file model convention;
  * an exhaust plume cone behind it whose temperature falls to ambient on two
    scales and whose emissivity is a gas path, not a surface.

Numbers are order-of-magnitude for a turbofan at military power: about 900 K at
the nozzle exit, a bright core that breaks up within a few metres, and a wake
that fades over tens of metres. An afterburner would be brighter still and is
not modelled.
"""

import numpy as np


#: A free jet cools on two very different scales. One exponential either made
#: the core vanish too fast to read as an exhaust or dragged the whole trail
#: out bright; the sum gives a bright core and a dissolving tail.
#:
#: Both scales were shortened by roughly a factor of three. At 8 km through a
#: 2.2 deg frame the old pair kept a 46 m cone emitting above NETD, and 46 m is
#: 144 px against a 63 px airframe: the target rendered as a bright rod with the
#: aircraft as a speck at its tip. A turbofan's visible core is two or three
#: nozzle diameters; everything past that is below the sensor's noise floor.
CORE_FRACTION = 0.70
CORE_DECAY_M = 2.5
WAKE_FRACTION = 0.22
WAKE_DECAY_M = 6.0

#: Per-class engine profile: nozzle and exit temperature in kelvin, exit
#: diameter and visible plume length in metres, and the speed at which that
#: plume length is the one for this class (military power / cruise).  Slower
#: means less thrust, a shorter and cooler plume.
ENGINE_PROFILES = {
    "fighter": dict(t_nozzle=880.0, t_exit=940.0, dia_m=0.85, len_m=14.0, v_ref=250.0),
    "airliner": dict(t_nozzle=730.0, t_exit=790.0, dia_m=1.50, len_m=24.0, v_ref=240.0),
    "helicopter": dict(t_nozzle=620.0, t_exit=680.0, dia_m=0.32, len_m=7.0, v_ref=60.0),
    "uav_fixed": dict(t_nozzle=560.0, t_exit=620.0, dia_m=0.26, len_m=6.0, v_ref=35.0),
    "uav_rotor": dict(t_nozzle=430.0, t_exit=480.0, dia_m=0.16, len_m=3.0, v_ref=15.0),
    "vehicle": dict(t_nozzle=392.0, t_exit=430.0, dia_m=0.09, len_m=2.0,
                     hot_frac=0.26, v_ref=20.0),
    # A ship shows its funnel and engine casing, warm rather than hot.
    "ship": dict(t_nozzle=358.0, t_exit=392.0, dia_m=0.32, len_m=5.0,
                 hot_frac=0.18, v_ref=12.0),
    "tank": dict(t_nozzle=400.0, t_exit=440.0, dia_m=0.09, len_m=1.8,
                 hot_frac=0.28, v_ref=12.0),
}
DEFAULT_PROFILE = dict(t_nozzle=520.0, t_exit=590.0, dia_m=0.18, len_m=5.0,
                       v_ref=60.0)

#: Effective emissivity of the plume through its own gas path. A volume of hot
#: CO2/H2O is not a surface; this is the path-averaged value that lands the
#: plume at the right brightness without pretending it is opaque.
PLUME_EMISSIVITY = 0.20
PLUME_MATERIAL = "exhaust_plume"
#: Transmission buckets, faintest first.  A plume face is put in the bucket that
#: matches how much gas the camera looks through, so the silhouette ramps into
#: the background instead of ending on one emissivity.
PLUME_BUCKETS = ("exhaust_plume_q1", "exhaust_plume_q2", "exhaust_plume_q3",
                 "exhaust_plume_q4", "exhaust_plume_q5")

#: Rear fraction of the fuselage treated as the nozzle, and the radius limit
#: around the axis (as a fraction of length) so tail fins are not painted 900 K.
NOZZLE_LENGTH_FRAC = 0.035
NOZZLE_RADIUS_FRAC = 0.06

#: Facet count. 14x10 put facets wider than a pixel at engagement range, so the
#: cone rasterised as a silhouette of quads rather than a plume; at 28x16 the
#: facets fall below a pixel and the flat shading reads as a gradient.
PLUME_RINGS = 28
PLUME_SEGMENTS = 16


def _decay_temp(t_amb, t_exit, distance_m):
    """Excess temperature down the jet axis, on two scales."""
    excess = (t_exit - t_amb)
    return t_amb + excess * (CORE_FRACTION * np.exp(-distance_m / CORE_DECAY_M)
                             + WAKE_FRACTION * np.exp(-distance_m / WAKE_DECAY_M))


def _unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])


def _basis(axis):
    a = _unit(axis)
    ref = np.array([0.0, 0.0, 1.0])
    if abs(float(a @ ref)) > 0.95:
        ref = np.array([1.0, 0.0, 0.0])
    u = _unit(np.cross(a, ref))
    v = np.cross(a, u)
    return a, u, v


def profile_for(cls):
    return dict(ENGINE_PROFILES.get(cls) or DEFAULT_PROFILE)


def _throttle(obj, prof):
    """0..1 thrust proxy from the object's own speed.

    Floored at 0.12 so a parked airframe still shows a warm nozzle instead of a
    perfectly cold one -- a jet on the ground is not shut down, and a zero here
    would make the exhaust vanish the moment a scene stopped moving.
    """
    v = float(np.linalg.norm(np.asarray(obj.get("vel", [0.0, 0.0, 0.0]), float)))
    v_ref = max(float(prof.get("v_ref", 250.0)), 1.0)
    return min(max(v / v_ref, 0.12), 1.0)


def _pose(world, obj):
    """(centre, forward unit) for one object in world coordinates."""
    pos = np.asarray(obj.get("pos", [0.0, 0.0, 0.0]), float)
    vel = np.asarray(obj.get("vel", [0.0, 0.0, 0.0]), float)
    if float(np.linalg.norm(vel)) > 1e-6:
        return pos, _unit(vel)
    yaw = np.radians(float(obj.get("yaw", 0.0)))
    return pos, _unit(np.array([np.cos(yaw), np.sin(yaw), 0.0]))


def _hot_frac(cls):
    """Rear fraction of the body treated as the engine hot region.

    A jet's nozzle is a couple of percent of the airframe; a tank's engine bay
    and a ship's funnel are a large plate of the hull. One constant could not
    describe both -- at 3.5 % a tank had no hot region at all and rendered as a
    flat patch the same temperature as the road.
    """
    p = profile_for(cls)
    return float(p.get("hot_frac", NOZZLE_LENGTH_FRAC))


def _rear_span(world, obj, cent):
    """(forward unit, tail coordinate, head coordinate, length, rear mask)."""
    pos, fwd = _pose(world, obj)
    rel = cent - pos
    along = rel @ fwd
    lo, hi = float(along.min()), float(along.max())
    length = max(hi - lo, 1e-6)
    rear = along <= (lo + _hot_frac(obj.get("cls")) * length)
    if rear.any():
        radial = np.linalg.norm(rel - np.outer(along, fwd), axis=1)
        rear = rear & (radial <= NOZZLE_RADIUS_FRAC * length)
    return fwd, lo, hi, length, rear


def nozzle_faces(world, cfg):
    """Indices and kelvin of the faces that form each exhaust nozzle.

    Selected in world space from the rear of the airframe along its velocity,
    so a model whose nose points -X and one whose nose points +Y behave alike.
    """
    if not cfg.get("ir", {}).get("engine_heat", True):
        return np.zeros(0, np.int64), np.zeros(0)
    cent = world.verts[world.faces].mean(axis=1)
    idx, temps = [], []
    for obj in world.objs:
        if not obj.get("label", True):
            continue
        mask = world.face_obj == obj["id"]
        if not mask.any():
            continue
        prof = profile_for(obj.get("cls"))
        _fwd, _lo, _hi, _len, rear = _rear_span(world, obj, cent[mask])
        if not rear.any():
            continue
        sel = np.flatnonzero(mask)[rear]
        idx.append(sel)
        # The nozzle cools with thrust too: at idle it is a warm ring, not the
        # 880 K exit a fighter shows at military power.
        t_amb = ambient_k(cfg)
        thr = _throttle(obj, prof)
        t_noz = t_amb + (float(prof["t_nozzle"]) - t_amb) * (0.45 + 0.55 * thr)
        temps.append(np.full(len(sel), t_noz))
    if not idx:
        return np.zeros(0, np.int64), np.zeros(0)
    return np.concatenate(idx), np.concatenate(temps)


def plume_geometry(world, cfg):
    """Build the exhaust plume behind every labelled object.

    Returns (verts, faces, temps). The cone starts at the rear of the airframe,
    widens, then closes and fades; temperature follows a two-scale decay to
    ambient along the axis and carries a turbulent jitter, so the surface is
    mottled instead of a flat-shaded slab.
    """
    empty = (np.zeros((0, 3)), np.zeros((0, 3), np.int64), np.zeros(0),
             np.zeros(0))
    if not cfg.get("ir", {}).get("engine_heat", True):
        return empty
    t_amb = float(cfg.get("air", {}).get("temp_c", 20.0)) + 273.15
    view_point = _view_point(cfg)
    cent_all = world.verts[world.faces].mean(axis=1)
    verts_all, faces_all, temps_all, weights_all = [], [], [], []
    voff = 0
    for obj in world.objs:
        if not obj.get("label", True):
            continue
        mask = world.face_obj == obj["id"]
        if not mask.any():
            continue
        prof = profile_for(obj.get("cls"))
        pos, fwd = _pose(world, obj)
        _f, lo, _hi, _len, _rear = _rear_span(world, obj, cent_all[mask])
        exit_c = pos + fwd * lo
        r0 = max(prof["dia_m"] * 0.5, 0.05)
        # Thrust, not the airframe, sets the plume.  A per-class length drew the
        # same 14 m cone at 80 m/s cruise as at military power, which reads as a
        # solid rod bolted to a jet that is barely working.
        thr = _throttle(obj, prof)
        L = float(prof["len_m"]) * (0.35 + 0.65 * thr)
        t_exit = t_amb + (float(prof["t_exit"]) - t_amb) * (0.45 + 0.55 * thr)
        grow = max(2.2, L / max(r0 * 7.0, 1e-6))
        a, u, v = _basis(fwd)
        pts, tps = [], []
        for i in range(PLUME_RINGS + 1):
            s = i / float(PLUME_RINGS)
            d = s * L
            # A real exhaust opens to three or four exit diameters before it
            # breaks up.  Capping the growth at 2.2 r0 left a cone 1-2 px wide at
            # range, and point-sampled rasterisation turned that into a ragged
            # dashed hairline instead of a plume.
            r = r0 * (1.0 + min(grow, 3.6) * s) * (1.0 - 0.55 * s * s * s)
            base_t = _decay_temp(t_amb, t_exit, d)
            c = exit_c - a * d
            for j in range(PLUME_SEGMENTS):
                th = 2.0 * np.pi * j / PLUME_SEGMENTS
                jit = 1.0 + 0.16 * np.sin(3.1 * i + 1.9 * j) * (0.35 + 0.65 * s)
                pts.append(c + u * (r * np.cos(th)) + v * (r * np.sin(th)))
                tps.append(base_t * jit)
        V = np.array(pts, float)
        VT = np.array(tps, float)
        F = []
        for i in range(PLUME_RINGS):
            for j in range(PLUME_SEGMENTS):
                j2 = (j + 1) % PLUME_SEGMENTS
                p0 = i * PLUME_SEGMENTS + j
                p1 = i * PLUME_SEGMENTS + j2
                p2 = (i + 1) * PLUME_SEGMENTS + j
                p3 = (i + 1) * PLUME_SEGMENTS + j2
                F.append([p0, p1, p2])
                F.append([p1, p3, p2])
        F = np.array(F, np.int64)
        ftemp = VT[F].mean(axis=1)
        # Path length through the gas, as seen from the camera.  The shell is a
        # stand-in for a volume: a face pointing at the camera is traversed end
        # to end, a face at the silhouette rim is grazed and contributes almost
        # nothing.  That cosine is what gives the plume a soft rim instead of a
        # facet edge, without needing a volume renderer.
        fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
        fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
        if view_point is None:
            weight = np.ones(len(F))
        else:
            rays = view_point - V[F].mean(axis=1)
            rays /= np.maximum(np.linalg.norm(rays, axis=1, keepdims=True), 1e-12)
            weight = np.abs(np.einsum("ij,ij->i", fn, rays)) ** 1.2
        verts_all.append(V)
        faces_all.append(F + voff)
        temps_all.append(ftemp)
        weights_all.append(np.clip(weight, 0.0, 1.0))
        voff += len(V)
    if not verts_all:
        return empty
    return (np.vstack(verts_all), np.vstack(faces_all),
            np.concatenate(temps_all), np.concatenate(weights_all))


def attach(world, cfg):
    """Return a render world that includes the plumes, plus their temperatures.

    The plume is appended, never substituted: the caller keeps the original
    world for radar, lidar and the shadow solve, so the exhaust cannot block the
    sun or answer a radar pulse.
    """
    V, F, temps, weights = plume_geometry(world, cfg)
    if len(F) == 0:
        return world, temps
    from . import materials as matlib
    from .scene import World
    names = sorted(matlib.MATERIALS)
    for bucket in PLUME_BUCKETS:
        if bucket not in matlib.MATERIALS:
            raise RuntimeError("material %r missing from matlib" % bucket)
    # Each face is drawn with the transmission its own path through the gas
    # justifies.  Without this the cone ended on a single 0.28 emissivity and
    # read as a solid bar one pixel wide.
    span = float(len(PLUME_BUCKETS) - 1)
    pick = np.rint(np.clip(weights, 0.0, 1.0) * span).astype(int)
    bucket_mi = np.array([names.index(b) for b in PLUME_BUCKETS], np.int32)
    fmat_plume = bucket_mi[np.clip(pick, 0, len(PLUME_BUCKETS) - 1)]
    # plume_geometry numbers its faces against its OWN vertex block, so shift
    # them past the airframe before merging. Without this the cone indexes the
    # aircraft vertices and rasterises as a patch on the fuselage.
    verts = np.vstack([world.verts, V])
    faces = np.vstack([world.faces, F + len(world.verts)])
    n = len(F)
    fobj = np.concatenate([world.face_obj, np.full(n, -2, np.int32)])
    fmat = np.concatenate([world.face_mat, fmat_plume])
    fs = np.concatenate([world.face_source_material,
                         np.full(n, PLUME_MATERIAL, dtype=object)])
    fg = np.concatenate([world.face_group, np.full(n, "", dtype=object)])
    lc = np.vstack([world.face_local_centers, V[F].mean(axis=1)])
    out = World(verts, faces, fobj, fmat, list(world.objs),
                face_source_material=fs, face_group=fg, face_local_centers=lc)
    return out, temps


def ambient_k(cfg):
    return float(cfg.get("air", {}).get("temp_c", 20.0)) + 273.15


def _view_point(cfg):
    """Camera position the plume's transmission weighting is measured from.

    Returns None when the config names no usable camera, and the caller then
    weights every face equally (the pre-existing look).
    """
    sensors = cfg.get("sensors") or {}
    name = (cfg.get("clip") or {}).get("camera") or "ir_camera"
    cam = (sensors.get(name) or sensors.get("ir_camera")
           or sensors.get("optical") or {})
    pos = cam.get("pos")
    try:
        p = np.asarray(pos, float).reshape(3)
    except (TypeError, ValueError):
        return None
    return p if np.isfinite(p).all() else None