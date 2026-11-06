"""Generate the four seeker-view scenes.

    python tools\\make_seeker_scenes.py

Writes into scenes/:

    sk3_high_air2air.json         高空空中目标  (sky background, 7.1-9.5 km head-on)
    sk3_low_surface2air.json      低空上升接近  (ground launcher climbing at a UAV)
    sk3_low_air2air_occluded.json 低空空中目标  (urban high-rise occlusion)
    sk3_low_air2ground_clutter.json 低空下视 (ground target among clutter + SAR)

Every scene is two 120-frame clips at 20 fps, rendered through a narrow seeker
lens and a wide search lens at the same instant, with the same weather state
driving all modalities. Numbers here are the geometry; the per-clip jitter that
makes clip 0 and clip 1 different passes lives in simkit/plan.py.

Why a generator and not four hand-edited JSONs: the sensor blocks are 90% the
same, and the pixel-size band is the one thing that must stay consistent
(15-70 px in the narrow lens). Deriving the range from the pixel band here means
a change to target size or FOV cannot silently push the target out of band.
"""
import json
import math
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENES = os.path.join(ROOT, "scenes")

FPS = 20.0
DT = 1.0 / FPS
FPC = 120            # frames per clip
NFRAMES = 2 * FPC    # two clips per scene


def px_for(size_m, rng_m, fov_deg, width):
    """Pixel span of a size_m object at rng_m through a fov_deg lens."""
    return size_m * width / (2.0 * rng_m * math.tan(math.radians(fov_deg) / 2.0))


def rng_for(size_m, px, fov_deg, width):
    return size_m * width / (2.0 * px * math.tan(math.radians(fov_deg) / 2.0))


def mesh_obj(name, cls, model, pos, rot=0.0, mat="painted_metal", decimate=None,
             scale=None, temp_c=None, rcs_db=None, vel=(0.0, 0.0, 0.0), label=True):
    o = dict(name=name, cls=cls, shape="obj", model="../assets/models/" + model,
             pos=list(pos), vel=list(vel), rot=rot, mat=mat, label=label)
    if decimate:
        o["decimate"] = int(decimate)
    if scale is not None:
        o["scale"] = scale
    if temp_c is not None:
        o["temp_c"] = temp_c
    if rcs_db is not None:
        o["rcs_db"] = rcs_db
    return o


def box_obj(name, size, pos, mat="building_wall", rot=0.0, label=False, temp_c=None,
            rcs_db=None):
    o = dict(name=name, cls="scenery", shape="box", size=list(size), pos=list(pos),
             rot=rot, mat=mat, label=label)
    if temp_c is not None:
        o["temp_c"] = temp_c
    if rcs_db is not None:
        o["rcs_db"] = rcs_db
    return o


def sensors(eye, fov, width, height, radar_band="Ka", radar_span=(2.0, 1.5),
            lidar_max=6000, lidar_fov=None, band_note=None):
    """The five physical sensors, all co-boresighted with the seeker turret."""
    lidar_fov = lidar_fov or fov * 1.8
    return {
        "ir_camera": dict(pos=list(eye), look_at=[0.0, 0.0, 0.0], fov_deg=fov,
                          width=width, height=height),
        "optical": dict(pos=list(eye), look_at=[0.0, 0.0, 0.0], fov_deg=fov,
                        width=width, height=height),
        # A seeker's radar has to see the same targets the optics do, so the three
        # numbers that decide whether it can are set from the pass range, not
        # from the defaults:
        #   n_samples x c/2B = unambiguous range. 512 x 7.5 MHz = 10.2 km, which
        #     covers the pass; the old 512 x 30 MHz was 2.6 km against 6-10 km
        #     targets, so nothing was ever in range. Keeping the sample count low
        #     also keeps the noise bandwidth down: the same range window with 1024
        #     samples cost 3 dB of SNR for resolution this target does not need.
        #   chirp 15 us -> +/-333 m/s unambiguous closing velocity. A weaving dive
        #     closes at ~250 m/s, which folds onto zero Doppler at 50 us.
        #   beamwidth_el 6 deg instead of the 12 deg default: a 2x12 fan beam is a
        #     search radar, not a seeker, and it cost 6 dB of gain. With a 6 deg
        #     elevation beam, 10 W (40 dBm) gives ~18-25 dB on both the 0.5 m^2
        #     rover at 2.6 km and the aircraft at 9.6 km -- a plausible seeker
        #     instead of a 1.3 kW one with a beam that is wrong in shape.
        "radar": dict(band=radar_band, az_span_deg=radar_span[0], el_span_deg=radar_span[1],
                      n_az=40, n_el=28, bandwidth_mhz=7.5, n_samples=1024, chirp_us=15, pri_us=15,
                      beamwidth_el_deg=6.0,
                      # 16 chirps was a 0.8 ms dwell: 12.5 m/s Doppler bins, i.e. a range-Doppler
                # image with no usable velocity axis. 128 chirps is 6.4 ms -- a
                # realistic seeker CPI -- and costs ~1 s/frame, giving 1.6 m/s bins.
                n_chirps=256, tx_power_dbm=40, noise_figure_db=8,
                      clutter_density=0.04, mti_min_vr_mps=0.8, pfa=1e-06, swerling="I",
                      clutter_db=-12.0, ground_clutter_db=-35.0,
                      pos=list(eye), look_at=[0.0, 0.0, 0.0], up=[0, 0, 1]),
        "lidar": dict(pos=list(eye), look_at=[0.0, 0.0, 0.0], fov_deg=lidar_fov,
                      n_az=480, n_el=120, max_range_m=lidar_max,
                      range_spec_m=max(lidar_max * 0.5, 500.0), pulse_energy_nj=4000,
                      aperture_mm=50, wavelength_nm=905, vfov_deg=lidar_fov * 0.75),
    }


def sar_block(anchor, altitude, swath, width=1200.0, bw=150.0, heading=90.0):
    return dict(band="X", fc_ghz=9.6, bandwidth_mhz=bw, prf_hz=1200,
                n_pulses=512, n_cross=320, n_samples=768,
                altitude_m=altitude, speed_mps=180, heading_deg=heading,
                squint_deg=90, side="right",
                swath_center_m=swath, swath_width_m=width,
                tx_power_w=200, antenna_len_m=1.5, antenna_h_m=0.4,
                anchor=list(anchor))


def clip(target_m, fov, width, height, origin, speed, el_deg, band, weave,
         weave_hz, wander=0.35, wide_fov=18.0, wide_w=640,
         wide_h=480, sweep=1.2, sweep_hz=0.06, variation=True):
    """One clip template.

    `el_deg` is the signed elevation of the airframe above the target: positive
    for a dive from above (air-to-air, air-to-ground), negative for a launcher
    below the target climbing at it (surface-to-air). `band` is the slant-range
    window; the platform is placed inside it per clip and the closing distance
    is `speed * clip_length`, so the pixel size at both ends is predictable.
    """
    return dict(
        frames_per_clip=FPC,
        camera="ir_camera",
        views=[
            dict(name="seeker", tag="", fov_deg=fov, width=width, height=height),
            dict(name="wide", tag="_wide", fov_deg=wide_fov, width=wide_w,
                 height=wide_h, sweep_deg=sweep, sweep_hz=sweep_hz),
        ],
        seeker=dict(
            mode="platform",
            platform=dict(origin=list(origin), speed_mps=speed, heading_deg=90.0,
                          dive_deg=(-abs(el_deg) if el_deg >= 0 else abs(el_deg)),
                          weave_amp_m=weave, weave_hz=weave_hz,
                          standoff_end_m=float(band[0]), solve="standoff"),
            target_m=target_m,
            el_deg=float(el_deg),
            range_m=[float(band[0]), float(band[1])],
            los_jitter_deg=0.02, wander_deg=wander, wander_hz=0.09,
            lag_s=0.12, bank_max_deg=25,
            per_clip_variation=dict(enabled=bool(variation), bearing_deg=22.0,
                                    dive_deg=1.5),
        ),
        rig=dict(vibration_m=0.02, vibration_deg=0.05),
    )


def common(ground, weather, sun, air, ir_netd=30, samples=16, exposure=-0.8):
    return dict(
        ground=ground,
        weather=dict(mode="per_clip", pool=weather),
        sun=sun, air=air,
        ir=dict(bands=["mwir", "lwir"], netd_mk=ir_netd, psf_px=1.3,
                nuc=dict(enabled=True, prnu=0.008, dsnu_counts=25,
                         column_fpn_counts=6, dead_px=0.0002, seed=12345)),
        optical=dict(samples=samples, exposure=exposure),
        frames=dict(n=NFRAMES, dt=DT),
    )


# ---------------------------------------------------------------------------
# 1. 高空空中目标: nothing but sky, 7.1-9.5 km head-on
# ---------------------------------------------------------------------------
def high_air2air():
    target_m, fov, w, h = 10.0, 2.2, 768, 576
    band = [rng_for(target_m, 28.0, fov, w), rng_for(target_m, 21.0, fov, w)]  # near, far
    s = common(
        ground=dict(enabled=False),
        weather=["clear", "cloudy"],
        sun=dict(elev_deg=38, az_deg=120, irradiance=900, on=True),
        air=dict(temp_c=-22, rh=0.15, wind_mps=18),
        ir_netd=28, samples=16, exposure=-1.4,
    )
    tgt = mesh_obj("uav_high", "uav", "global_hawk.obj", [0, 0, 8000],
                   rot=200, mat="painted_metal", decimate=12000, scale=0.026,
                   temp_c=6, rcs_db=1.5, vel=[6, -4, 0])
    # 7.2 deg of elevation puts the airframe 1.5 km above the target at the far
    # end: a shallow stern conversion, not a vertical drop.
    s.update(seed=101, objects=[tgt],
             sensors=sensors([0, -band[1], 9500], fov, w, h, radar_band="Ku",
                             radar_span=(1.6, 1.2), lidar_max=20000),
             clip=clip(target_m, fov, w, h, [0, -band[1], 9500], 250, 7.2, band,
                       25.0, 0.15, wide_fov=18.0))
    return "sk3_high_air2air.json", s


# ---------------------------------------------------------------------------
# 2. 低空上升接近: launcher on the ground, climbing at a UAV 5.7-8.0 km out
# ---------------------------------------------------------------------------
def low_surface2air():
    target_m, fov, w, h = 8.0, 2.2, 768, 576
    band = [rng_for(target_m, 28.0, fov, w), rng_for(target_m, 20.0, fov, w)]
    s = common(
        ground=dict(enabled=True, size=14000, mat="asphalt", z=0),
        weather=["clear", "cloudy", "rain"],
        sun=dict(elev_deg=28, az_deg=210, irradiance=560, on=True),
        air=dict(temp_c=14, rh=0.62, wind_mps=5),
        ir_netd=30, samples=16, exposure=-0.6,
    )
    tgt = mesh_obj("uav_intruder", "uav", "global_hawk.obj", [0, 0, 900],
                   rot=110, mat="painted_metal", decimate=12000, scale=0.0192,
                   temp_c=18, rcs_db=0.5, vel=[4, 10, 0])
    # Negative elevation: the launcher starts 25 m above the ground, 900 m below
    # the target, and the track climbs. This is the geometry a ground-based
    # imaging seeker actually flies, and it cannot be expressed with a dive.
    s.update(seed=211, objects=[tgt],
             sensors=sensors([0, -band[1], 25], fov, w, h, radar_band="Ka",
                             radar_span=(2.0, 2.0), lidar_max=18000),
             clip=clip(target_m, fov, w, h, [0, -band[1], 25], 235, -5.0, band,
                       12.0, 0.20, wide_fov=18.0))
    return "sk3_low_surface2air.json", s


# ---------------------------------------------------------------------------
# 3. 低空空中目标 + 城市高楼遮挡: the target disappears behind towers mid-clip
# ---------------------------------------------------------------------------
def occluder_walls(cfg, seed, target_name="uav_low"):
    """Walls anchored on each clip's *actual* start pose, not on a grid.

    Whether a building hides the target depends on where the sight line happens
    to cross it, and the two clips are drawn +/-22 deg apart in bearing -- a
    fixed grid therefore blocked one clip completely and missed the other
    entirely (measured). Placing the walls at a fixed distance along each clip's
    own line of sight makes the occlusion a property of the geometry we control:
    the first wall hides the target during the initial approach, the second one
    hides it again once the run has advanced, and the gap between them is the
    window where the seeker re-acquires.
    """
    from simkit import plan as pl
    walls = []
    dt = float(cfg["frames"]["dt"])
    fpc = int(cfg["clip"]["frames_per_clip"])
    for c in (0, 1):
        cp = pl.ClipPlan(cfg, c, fpc, dt, seed)
        cam = np.asarray(cp.path.pos(0.0), float)
        tgt = np.asarray(cp.target_pos(0.0), float)
        d = tgt - cam
        d = d / (np.linalg.norm(d) + 1e-12)
        # One wall, placed at ~45% of this clip's closing distance. A wall ahead
        # of a straight run blocks the sight line until the airframe flies past
        # it and nothing after that -- which is exactly the story we want:
        # blocked on entry, re-acquisition past the block. Two walls simply
        # blocked the whole clip (measured: 100%).
        dist = 0.22 * 100.0 * dt * fpc
        for k, (width, hgt) in enumerate(((190.0, 235.0),)):
            p = cam + d * dist
            ang = math.degrees(math.atan2(d[1], d[0])) + 90.0
            walls.append(box_obj("occ_c%d_%d" % (c, k), [width, 26.0, hgt],
                                 [float(p[0]), float(p[1]), 0.0], rot=ang,
                                 mat="building_wall", temp_c=24.0))
    return walls


def occlusion_report(cfg, seed, target_name="uav_low", lift=3.0):
    """Fraction of frames whose sight line is blocked, per clip."""
    from simkit import plan as pl, scene as sk
    n = int(cfg["frames"]["n"])
    dt = float(cfg["frames"]["dt"])
    blocked, per_clip = [], {}
    for s in pl.plan_frames(cfg, n, seed, dt):
        w = sk.build_world(s["cfg"], t=0.0, jitter=s["jitter"])
        cam = np.asarray(s["views"]["seeker"]["pos"], float)
        tgt = np.asarray(s["poses"][target_name]["pos"], float) + np.array([0, 0, lift])
        d = tgt - cam
        L = float(np.linalg.norm(d))
        d = d / (L + 1e-12)
        hit = w.bvh.ray_cast(cam, d[None, :])
        b = bool(hit["hit"][0] and float(hit["t"][0]) < L - 6.0)
        blocked.append(b)
        per_clip.setdefault(s["clip"], []).append(b)
    return blocked, {c: (sum(v) / float(len(v)) if v else 0.0)
                     for c, v in per_clip.items()}


def low_air2air_occluded():
    target_m, fov, w, h = 5.0, 2.2, 768, 576
    band = [2100.0, 3210.0]
    s = common(
        ground=dict(enabled=True, size=8000, mat="asphalt", z=0),
        weather=["clear", "cloudy", "rain"],
        sun=dict(elev_deg=46, az_deg=150, irradiance=700, on=True),
        air=dict(temp_c=21, rh=0.5, wind_mps=4),
        ir_netd=30, samples=16, exposure=-0.9,
    )
    tgt = mesh_obj("uav_low", "uav", "uav_demo.obj", [0, 0, 140],
                   rot=180, mat="painted_metal", decimate=4000, scale=3.5,
                   temp_c=25, rcs_db=-2.0, vel=[3, 2, 0])
    s.update(seed=317, objects=[tgt],
             sensors=sensors([0, -band[1], 84], fov, w, h, radar_band="Ka",
                             radar_span=(2.0, 2.0), lidar_max=5000),
             clip=clip(target_m, fov, w, h, [0, -band[1], 84], 100, -1.0, band,
                       8.0, 0.25, wide_fov=18.0, sweep=0.8))
    # Walls are computed from the clip's own geometry, so they hide the target on
    # purpose rather than by luck: both aircraft are below the rooftops and the
    # sight line has to pass a 235 m block 260 m ahead.
    s["objects"] = s["objects"] + occluder_walls(s, s["seed"])
    return "sk3_low_air2air_occluded.json", s


# ---------------------------------------------------------------------------
# 4. 低空下视 + 地面杂物: one vehicle among barriers, fences and shrubs, with SAR
# ---------------------------------------------------------------------------
def low_air2ground_clutter():
    target_m, fov, w, h = 3.1, 2.2, 768, 576
    band = [rng_for(target_m, 30.0, fov, w), rng_for(target_m, 17.0, fov, w)]
    s = common(
        ground=dict(enabled=True, size=8000, mat="asphalt", z=0),
        weather=["clear", "cloudy", "rain"],
        sun=dict(elev_deg=34, az_deg=170, irradiance=640, on=True),
        air=dict(temp_c=17, rh=0.55, wind_mps=5),
        ir_netd=30, samples=16, exposure=-0.7,
    )
    tgt = mesh_obj("vehicle_rover", "vehicle", "rover.obj", [0, 300, 0],
                   rot=90, mat="painted_metal", decimate=9000, scale=1.0,
                   temp_c=41, rcs_db=-3.0, vel=[0, 3.0, 0])
    clutter = [
        # a decoy that is deliberately NOT labelled: same class of object, so a
        # detector that keys on "vehicle-like blob" pays for it in false positives
        mesh_obj("decoy_rover", "vehicle", "rover.obj", [38, 430, 0],
                 rot=75, mat="painted_metal", decimate=6000, scale=1.0,
                 temp_c=39, rcs_db=-4.0, label=False),
        mesh_obj("barrier_1", "scenery", "concrete_road_barrier.obj", [-26, 250, 0],
                 rot=95, mat="concrete", decimate=2500, temp_c=36, label=False),
        mesh_obj("barrier_2", "scenery", "concrete_road_barrier.obj", [-26, 268, 0],
                 rot=95, mat="concrete", decimate=2500, temp_c=36, label=False),
        mesh_obj("barrier_3", "scenery", "concrete_road_barrier.obj", [24, 336, 0],
                 rot=90, mat="concrete", decimate=2500, temp_c=36, label=False),
        mesh_obj("fence_1", "scenery", "modular_chainlink_fence.obj", [-60, 300, 0],
                 rot=90, mat="metal_bare", decimate=3000, temp_c=34, label=False),
        mesh_obj("fence_2", "scenery", "modular_chainlink_fence.obj", [70, 300, 0],
                 rot=90, mat="metal_bare", decimate=3000, temp_c=34, label=False),
        mesh_obj("shrub_1", "scenery", "shrub_01.obj", [-45, 210, 0],
                 mat="foliage", decimate=2500, temp_c=29, label=False),
        mesh_obj("shrub_2", "scenery", "shrub_01.obj", [52, 230, 0],
                 mat="foliage", decimate=2500, temp_c=29, label=False),
        mesh_obj("shrub_3", "scenery", "shrub_01.obj", [-70, 420, 0],
                 mat="foliage", decimate=2500, temp_c=29, label=False),
        box_obj("shed_a", [18, 12, 7], [-95, 520, 0], mat="building_wall", temp_c=28),
        box_obj("shed_b", [24, 14, 9], [105, 260, 0], mat="building_wall", temp_c=28),
    ]
    start = [0, -band[1], 210.0]
    s.update(seed=419, objects=[tgt] + clutter,
             sensors=sensors([0, -band[1], 210.0], fov, w, h, radar_band="W",
                             radar_span=(2.5, 2.5), lidar_max=5000),
             clip=clip(target_m, fov, w, h, start, 167, 2.6, band, 5.0, 0.20,
                       wide_fov=18.0, sweep=1.0))
    s["sensors"]["sar"] = sar_block([0, 300, 0], altitude=3000, swath=5000,
                                    width=1200, bw=150)
    return "sk3_low_air2ground_clutter.json", s


def main():
    builders = [high_air2air, low_surface2air, low_air2air_occluded,
                low_air2ground_clutter]
    for fn in builders:
        name, cfg = fn()
        path = os.path.join(SCENES, name)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, ensure_ascii=False, indent=1)
            fh.write("\n")
        # report the pixel band this scene implies, so a wrong number is caught
        # here instead of after a two-hour render
        c = cfg["clip"]["seeker"]
        views = {v["name"]: v for v in cfg["clip"]["views"]}
        tgt_m = c["target_m"]
        lo, hi = sorted(float(v) for v in c.get("range_m") or [0.0, 0.0])
        v = views["seeker"]
        print("%-36s el %+5.1f deg  range %5.0f..%-5.0f m  narrow %4.1f..%-4.1f px "
              "(%.1f%%..%.1f%%)  wide %.1f..%.1f px"
              % (name, c.get("el_deg", 0.0), lo, hi,
                 px_for(tgt_m, hi, v["fov_deg"], v["width"]),
                 px_for(tgt_m, lo, v["fov_deg"], v["width"]),
                 100.0 * px_for(tgt_m, hi, v["fov_deg"], v["width"]) / v["width"],
                 100.0 * px_for(tgt_m, lo, v["fov_deg"], v["width"]) / v["width"],
                 px_for(tgt_m, hi, views["wide"]["fov_deg"], views["wide"]["width"]),
                 px_for(tgt_m, lo, views["wide"]["fov_deg"], views["wide"]["width"])))
    # The occlusion scene is the one with a hard claim to defend, so the builder
    # measures it instead of trusting the layout.
    from simkit import scene as _sk
    occ_cfg = _sk.load_config(os.path.join(SCENES, "sk3_low_air2air_occluded.json"))
    blocked, per_clip = occlusion_report(occ_cfg, occ_cfg["seed"])
    runs, cur = [], 0
    for b in blocked:
        if b:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    print("occlusion: %.0f%% of frames blocked, per clip %s, runs %s"
          % (100.0 * sum(blocked) / max(len(blocked), 1),
             {k: "%.0f%%" % (100 * v) for k, v in sorted(per_clip.items())}, runs))
    print("wrote %d scenes into %s" % (len(builders), SCENES))


if __name__ == "__main__":
    main()
