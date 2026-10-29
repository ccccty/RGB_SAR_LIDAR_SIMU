"""LiDAR: pulsed time-of-flight range imaging and point cloud (numpy only).

One pulse per beam on an az/el raster; each beam is a single exact ray into the
same BVH the radar uses, so geometry cannot drift between modalities.  Radiometry
is the standard lidar link equation

    N = E_t * (rho/pi) * (A_r / R^2) * cos(inc) * exp(-2 beta R) * eta / E_ph

counted in detected photons.  The one absolute number an instrument datasheet
actually gives you -- its range for a 10 % reflector -- is what calibrates the
background noise floor, so atmosphere, reflectivity and range all keep following
the physics while sensitivity stays a single checkable constant.
"""
import numpy as np

from .scene import sensor_basis

C = 2.99792458e8
H_PLANCK = 6.62607015e-34

DEFAULTS = dict(
    pos=[0.0, -80.0, 5.0],
    look_at=[0.0, 0.0, 0.0],
    up=[0.0, 0.0, 1.0],
    fov_deg=30.0,
    vfov_deg=None,
    n_az=360,
    n_el=64,
    max_range_m=400.0,
    wavelength_nm=905.0,
    pulse_energy_nj=1000.0,      # 1 uJ per pulse
    aperture_mm=25.0,
    optics_eff=0.60,
    range_spec_m=200.0,          # datasheet range for a 10 % target, clear noon
    det_snr_db=11.0,             # SNR the datasheet range is quoted at
    range_noise_m=0.02,
    pulse_width_ns=4.0,          # sets the range resolution, c*tau/2
    speckle_looks=4.0,
    min_snr_db=3.0,
    photon_floor=50.0,
)


def _cfg(user):
    rc = dict(DEFAULTS)
    rc.update({k: v for k, v in (user or {}).items() if k in DEFAULTS})
    return rc


def beam_grid(rc):
    p, r, u, f = sensor_basis(rc["pos"], rc["look_at"], rc.get("up", (0, 0, 1)))
    n_az, n_el = int(rc["n_az"]), int(rc["n_el"])
    haz = np.radians(float(rc["fov_deg"]))
    if rc.get("vfov_deg"):
        hve = np.radians(float(rc["vfov_deg"]))
    else:
        hve = haz * n_el / float(max(n_az, 1))     # square pixels
    az = np.linspace(-haz / 2.0, haz / 2.0, n_az)
    el = np.linspace(hve / 2.0, -hve / 2.0, n_el)
    A, E = np.meshgrid(az, el)
    ca, sa, ce, se = np.cos(A), np.sin(A), np.cos(E), np.sin(E)
    d = (ce[..., None] * ca[..., None] * f
         + ce[..., None] * sa[..., None] * r
         + se[..., None] * u)
    return p, d.reshape(-1, 3) / np.linalg.norm(d.reshape(-1, 3), axis=1, keepdims=True)


def _face_normals(w):
    t = w.verts[w.faces]
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    ln[ln == 0] = 1.0
    return n / ln


def simulate(world, cfg):
    from . import materials as matlib

    rc = _cfg(cfg.get("sensors", {}).get("lidar"))
    p, dirs = beam_grid(rc)
    n_el, n_az = int(rc["n_el"]), int(rc["n_az"])
    max_range = float(rc["max_range_m"])

    hit = world.bvh.ray_cast(p, dirs)
    tri, ok, t = hit["tri"], hit["hit"].copy(), np.asarray(hit["t"], float)
    ok &= np.isfinite(t) & (t > 1e-3) & (t <= max_range)

    lam = float(rc["wavelength_nm"]) * 1e-9
    e_ph = H_PLANCK * C / lam
    e_t = float(rc["pulse_energy_nj"]) * 1e-9
    a_r = np.pi * (float(rc["aperture_mm"]) * 1e-3 / 2.0) ** 2
    eta = float(rc["optics_eff"])

    # material lookup
    names = sorted(matlib.MATERIALS)
    alb = np.array([sum(matlib.MATERIALS[n]["albedo"]) / 3.0 for n in names])
    normals = _face_normals(world)

    # atmosphere: one-way extinction is beta per metre, two-way doubles it
    w = cfg.get("_weather") or {}
    beta = float(w.get("beta_lidar_per_km", 0.24)) / 1000.0
    solar = float(w.get("solar_w_m2", 700.0))

    # noise floor calibrated from the datasheet range at rho = 0.10
    r_ref = float(rc["range_spec_m"])
    n_ref = e_t * (0.10 / np.pi) * (a_r / r_ref ** 2) * 1.0 * eta / e_ph
    s = 10.0 ** (float(rc["det_snr_db"]) / 20.0)
    n_bg = max(n_ref ** 2 / s ** 2 - n_ref, 0.0)
    # solar background photons: night is much quieter than high noon
    n_bg *= 0.15 + 0.85 * min(1.0, max(solar, 0.0) / 700.0)
    n_bg = max(n_bg, float(rc["photon_floor"]))

    rng = np.random.default_rng(int(cfg.get("_frame_seed", 0)))
    rng_l = np.random.default_rng(int(cfg.get("_frame_seed", 0)) + 977)

    r = np.full(len(dirs), np.inf)
    n_ph = np.zeros(len(dirs))
    refl = np.zeros(len(dirs))
    obj = np.full(len(dirs), -1, np.int64)
    snr_db = np.full(len(dirs), -60.0)

    idx = np.where(ok)[0]
    if len(idx):
        t_i = tri[idx]
        rng_i = t[idx]
        n = normals[t_i]
        d = dirs[idx]
        cos_i = np.abs((n * (-d)).sum(axis=1))
        cos_i = np.clip(cos_i, 1e-3, 1.0)
        rho = alb[world.face_mat[t_i]]
        tau = np.exp(-2.0 * beta * rng_i)
        n_p = (e_t * (rho / np.pi) * (a_r / rng_i ** 2) * cos_i * tau * eta / e_ph)
        n_p = np.nan_to_num(n_p, nan=0.0, posinf=0.0)
        # speckle: a single pulse sees a random realisation of the return
        looks = max(float(rc["speckle_looks"]), 1.0)
        n_p = n_p * rng_l.gamma(looks, 1.0 / looks, len(n_p))

        snr = n_p / np.sqrt(n_p + n_bg + 1e-12)
        keep = snr >= 10.0 ** (float(rc["min_snr_db"]) / 20.0)

        r[idx] = np.where(keep, rng_i, np.inf)
        n_ph[idx] = n_p
        refl[idx] = np.where(keep, rho * cos_i * tau, 0.0)
        obj[idx] = np.where(keep, world.face_obj[t_i], -1)
        snr_db[idx] = 10.0 * np.log10(np.maximum(snr, 1e-6))

    # range measurement noise on returned points
    ret = np.isfinite(r)
    r_noisy = r.copy()
    r_noisy[ret] = r[ret] + rng.normal(0.0, float(rc["range_noise_m"]), int(ret.sum()))

    range_img = np.where(ret, r_noisy, 0.0).reshape(n_el, n_az)
    inten_img = np.clip(refl, 0.0, 1.0).reshape(n_el, n_az)
    obj_img = obj.reshape(n_el, n_az)
    snr_img = snr_db.reshape(n_el, n_az)

    # calibrated intensity for display: reflectance dominates, plus a mild
    # range falloff so a distant low-reflector does not look like a near one
    disp = np.zeros_like(inten_img)
    if ret.any():
        rr = np.where(ret, r, 1.0)
        near = np.exp(-(rr - r_ref * 0.25) / max(r_ref, 1.0))
        d = np.clip(inten_img * (0.35 + 0.65 * near.reshape(n_el, n_az)), 0.0, 1.0)
        disp = d ** 0.8

    pts = np.where(ret)[0]
    cloud = dict(
        xyz=(p + dirs[pts] * r_noisy[pts, None]) if len(pts) else np.zeros((0, 3)),
        range=r_noisy[pts],
        intensity=np.clip(refl[pts], 0, 1),
        snr_db=snr_db[pts],
        obj=obj[pts],
    )

    return dict(range_m=range_img, intensity=disp, reflectance=inten_img,
                obj_id=obj_img, snr_db=snr_img, cloud=cloud,
                hit_ratio=float(ret.mean()), max_range_m=max_range,
                n_beams=len(dirs), n_points=int(ret.sum()),
                n_bg=n_bg, rays=dirs, origin=p,
                range_res_m=C * float(rc["pulse_width_ns"]) * 1e-9 / 2.0,
                pixel_deg=float(rc["fov_deg"]) / max(n_az, 1))


def write_ply(path, cloud, origin):
    """Binary little-endian PLY so MeshLab / CloudCompare / Open3D can read it."""
    import os
    xyz = cloud["xyz"]
    n = len(xyz)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    hdr = (
        "ply\nformat binary_little_endian 1.0\n"
        "comment simkit lidar point cloud (metres, world frame)\n"
        "element vertex %d\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float range\nproperty float intensity\n"
        "property float snr_db\nproperty int object_id\n"
        "end_header\n" % n
    )
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                   ("range", "<f4"), ("intensity", "<f4"),
                   ("snr_db", "<f4"), ("object_id", "<i4")])
    arr = np.zeros(n, dt)
    if n:
        arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        arr["range"] = cloud["range"]
        arr["intensity"] = cloud["intensity"]
        arr["snr_db"] = cloud["snr_db"]
        arr["object_id"] = cloud["obj"]
    with open(path, "wb") as fh:
        fh.write(hdr.encode("ascii"))
        fh.write(arr.tobytes())
    return n
