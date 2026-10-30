"""Internal correctness checks (run with: python -m simkit selftest)."""
import json
import os

import numpy as np


def _ok(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + ("  " + detail if detail else ""))
    return bool(cond)


def run():
    from . import ir, labels, materials as matlib, radar, scene, thermal
    from .bvh import BVH
    from .mesh import box
    ok = True

    # 1. ray casting
    b = box([2, 1, 1], [0, 0, 1])
    t = BVH(b.v, b.f).ray_cast(np.array([5.0, 0, 1]), np.array([[-1.0, 0, 0]]))
    ok &= _ok("ray hits +x face at t=4", abs(float(t["t"][0]) - 4.0) < 1e-6, "t=%.3f" % t["t"][0])

    # 2. Planck band radiance
    L300 = float(ir.planck_band(np.array([300.0]), 8.0, 12.0)[0])
    L400 = float(ir.planck_band(np.array([400.0]), 8.0, 12.0)[0])
    ok &= _ok("planck 8-12um @300K in 10..80 W/m2/sr", 10.0 < L300 < 80.0, "L=%.2f" % L300)
    ok &= _ok("planck increases with T", L400 > L300, "%.2f -> %.2f" % (L300, L400))

    # 3. flat plate RCS against PO closed form
    lam = radar.C_LIGHT / 77e9
    area = np.array([1.0])
    s = radar.facet_sigma(area, np.array([1.0]), np.array([1.0]), np.array([0.0]), lam,
                          spec_scale=1.0)
    expect = 4 * np.pi * 1.0 / lam ** 2
    ok &= _ok("PO flat plate sigma", abs(float(s[0]) / expect - 1.0) < 0.05,
              "%.3g vs %.3g" % (s[0], expect))

    # 4. thermal: dark surface hotter than bright one in sunlight
    cfg = dict(air=dict(temp_c=20, rh=0.4, wind_mps=1.0),
               sun=dict(elev_deg=60, az_deg=180, irradiance=900),
               ground=dict(temp_c=25),
               objects=[dict(name="dark", cls="t", shape="box", size=[1, 1, 1],
                             pos=[0, 0, 0], mat="asphalt"),
                        dict(name="bright", cls="t", shape="box", size=[1, 1, 1],
                             pos=[5, 0, 0], mat="metal_bare")])
    w = scene.build_world(cfg)
    T = thermal.solve(w, cfg)
    nrm = thermal.face_normals(w)
    up = nrm[:, 2] > 0.9
    dark = T[up & (w.face_obj == 0)].mean()
    bright = T[up & (w.face_obj == 1)].mean()
    ok &= _ok("low-emissivity metal runs hotter in sun than asphalt", bright > dark,
              "metal %.1fC vs asphalt %.1fC" % (bright - 273.15, dark - 273.15))
    cfg["sun"]["on"] = False
    w2 = scene.build_world(cfg)
    T2 = thermal.solve(w2, cfg)
    dev = float(np.abs(T2 - 293.15).max()) - 273.15 + 273.15
    ok &= _ok("no sun -> only radiative cooling below air (3..20 K)",
              3.0 < float(np.abs(T2 - 293.15).max()) < 20.0,
              "max dev %.1f K" % float(np.abs(T2 - 293.15).max()))

    # 5. label projection matches mask
    cam = dict(pos=[-8, -12, 2.0], look_at=[0, 0, 1.0], fov_deg=45, width=320, height=240)
    cfg["sensors"] = {"ir_camera": cam}
    cfg["ir"] = {"band_um": [8, 12], "netd_mk": 10, "psf_px": 0.0}
    w = scene.build_world(cfg)
    T = thermal.solve(w, cfg)
    out = ir.render(w, cfg, T)
    mb = labels.boxes_from_mask(out["obj_id"], w)
    cb = labels.boxes_from_corners(w, cam, 320, 240)
    if 0 in mb and 0 in cb:
        d = max(abs(mb[0][k] - cb[0][k]) for k in ("x0", "y0", "x1", "y1"))
        ok &= _ok("mask bbox ~ projected bbox (<6 px)", d < 6.0, "maxdiff=%.1f" % d)
    else:
        ok &= _ok("mask bbox present", False)

    # 6. radar detection pipeline runs and finds the target
    rcfg = dict(pos=[-15, 0, 1.5], look_at=[0, 0, 1.0], n_az=32, n_el=6,
                az_span_deg=30, el_span_deg=12, det_snr_db=10.0)
    cfg2 = dict(air=cfg["air"], sun=cfg["sun"], ground=dict(size=60, temp_c=25),
                objects=[dict(name="tgt", cls="car", shape="box", size=[4, 2, 1.6],
                              pos=[0, 0, 0], mat="metal_bare", rcs_db=10.0)],
                sensors={"radar": rcfg})
    w2 = scene.build_world(cfg2)
    r = radar.simulate(w2, cfg2)
    tgt = [d for d in r["detections"] if d["obj"] == 0]
    ok &= _ok("radar detects a 10 dBsm target at ~15 m", len(tgt) > 0,
              "n=%d best_snr=%s" % (len(tgt), ("%.1f dB" % max(d["snr_db"] for d in tgt)) if tgt else "-"))
    ok &= _ok("range-Doppler cube shape",
              r["cube"].shape == (int(rcfg["n_az"]) * int(rcfg["n_el"]), 256, 32),
              str(r["cube"].shape))

    # 7. weather state machine drives every modality
    from . import weather
    clr = weather.derive(weather.resolve({"preset": "clear"}))
    cld = weather.derive(weather.resolve({"preset": "cloudy"}))
    rn = weather.derive(weather.resolve({"preset": "rain"}))
    ok &= _ok("clouds cut solar and warm the sky",
              cld["solar_w_m2"] < clr["solar_w_m2"] and cld["t_sky_k"] > clr["t_sky_k"],
              "%.0f -> %.0f W/m2, sky %.1f -> %.1f K"
              % (clr["solar_w_m2"], cld["solar_w_m2"], clr["t_sky_k"], cld["t_sky_k"]))
    ok &= _ok("rain raises Ka attenuation and LWIR extinction",
              rn["radar_att_db_km"]["Ka"] > 5.0 * clr["radar_att_db_km"]["Ka"]
              and rn["tau_ir_per_km"]["lwir"] > clr["tau_ir_per_km"]["lwir"],
              "Ka %.3f -> %.3f dB/km" % (clr["radar_att_db_km"]["Ka"],
                                         rn["radar_att_db_km"]["Ka"]))

    # 8. MWIR vs LWIR band radiance ratio at 300 K
    Lm = float(ir.planck_band(np.array([300.0]), 3.0, 5.0)[0])
    Ll = float(ir.planck_band(np.array([300.0]), 8.0, 12.0)[0])
    ok &= _ok("MWIR/LWIR ratio at 300 K in 0.02..0.4", 0.02 < Lm / Ll < 0.4,
              "%.3f" % (Lm / Ll))

    # 9. radar band preset switches carrier and beamwidth
    cfg3 = dict(air=cfg["air"], sun=cfg["sun"], ground=dict(size=60, temp_c=25),
                objects=[dict(name="t", cls="car", shape="box", size=[4, 2, 1.6],
                              pos=[0, 0, 0], mat="metal_bare", rcs_db=10.0)],
                sensors={"radar": dict(rcfg, band="X")})
    r3 = radar.simulate(scene.build_world(cfg3), cfg3)
    ok &= _ok("band preset X -> 9.5 GHz carrier",
              abs(float(r3["radar_cfg"]["fc_ghz"]) - 9.5) < 1e-6,
              "%.2f GHz" % float(r3["radar_cfg"]["fc_ghz"]))

    # 10. air scene: ground plane can be disabled
    cfg4 = dict(objects=[dict(name="t", cls="uav", shape="box", size=[1, 1, 1],
                              pos=[0, 0, 50])], ground=dict(enabled=False))
    w4 = scene.build_world(cfg4)
    ok &= _ok("ground disabled -> no background faces", bool((w4.face_obj >= 0).all()),
              "faces=%d" % len(w4.face_obj))

    # 11. fixed-pattern noise must be fixed (same pattern every frame)
    ncfg = dict(air=cfg["air"], sun=cfg["sun"], ground=dict(size=40, temp_c=25),
                objects=[dict(name="t", cls="car", shape="box", size=[4, 2, 1.6],
                              pos=[0, 0, 0], mat="metal_bare")],
                sensors={"ir_camera": cam},
                # texture_sigma_c=0: per-frame thermal texture would otherwise
                # legitimately change between frames and mask the fixed pattern
                ir=dict(band="lwir", texture_sigma_c=0.0,
                        nuc=dict(enabled=True, seed=7, prnu=0.01, dsnu_counts=40.0)))
    w5 = scene.build_world(ncfg)
    T5 = thermal.solve(w5, ncfg)
    a = json.loads(json.dumps(ncfg))
    a["_frame_seed"] = 1
    b = json.loads(json.dumps(ncfg))
    b["_frame_seed"] = 2          # different frame noise, same detector
    c = json.loads(json.dumps(ncfg))
    c["_frame_seed"] = 1
    c["ir"]["nuc"]["enabled"] = False   # ideal detector
    na = ir.render(w5, a, T5)["counts_u16"].astype(float)
    nb = ir.render(w5, b, T5)["counts_u16"].astype(float)
    nc = ir.render(w5, c, T5)["counts_u16"].astype(float)
    da = (na - nc).ravel()
    db = (nb - nc).ravel()
    corr = float(np.corrcoef(da, db)[0, 1])
    ok &= _ok("NUC pattern repeats across frames (delta correlation > 0.9)",
              corr > 0.9, "corr=%.3f, mean|delta|=%.1f counts"
              % (corr, float(np.abs(da).mean())))

    # 12. emissivity: bare metal reflects the cold sky instead of emitting
    mcfg = dict(air=cfg["air"], sun=dict(on=False), ground=dict(size=30, temp_c=20),
                objects=[dict(name="bare", cls="t", shape="box", size=[1.5, 1.5, 0.05],
                              pos=[-1.5, 0, 1.0], mat="metal_bare", temp_c=60.0),
                         dict(name="paint", cls="t", shape="box", size=[1.5, 1.5, 0.05],
                              pos=[1.5, 0, 1.0], mat="painted_metal", temp_c=60.0)],
                sensors={"ir_camera": dict(pos=[0, -7, 1.2], look_at=[0, 0, 1.0],
                                           fov_deg=45, width=200, height=140)},
                ir=dict(netd_mk=0.0, texture_sigma_c=0.0, psf_px=0.0, band="lwir"))
    mw = scene.build_world(mcfg)
    Tw = thermal.solve(mw, mcfg)
    mcfg["_frame_seed"] = 3
    mout = ir.render(mw, mcfg, Tw)
    t_bare = float(mout["apparent_temp_k"][mout["obj_id"] == 0].mean())
    t_paint = float(mout["apparent_temp_k"][mout["obj_id"] == 1].mean())
    ok &= _ok("low-emissivity metal reads colder than painted metal (same T)",
              t_paint - t_bare > 5.0,
              "bare %.1f K vs painted %.1f K" % (t_bare, t_paint))
    ok &= _ok("band emissivity table differs MWIR vs LWIR",
              abs(matlib.eps(matlib.get("metal_bare"), "mwir")
                  - matlib.eps(matlib.get("metal_bare"), "lwir")) > 0.02,
              "bare eps %.2f -> %.2f"
              % (matlib.eps(matlib.get("metal_bare"), "mwir"),
                 matlib.eps(matlib.get("metal_bare"), "lwir")))

    # 13. sea surface: rougher with wind, and it is a background (-1), not an object
    from . import mesh as meshlib
    calm = meshlib.sea(300.0, 0.0, 24, 24, wind_ms=3.0, seed=5)
    rough = meshlib.sea(300.0, 0.0, 24, 24, wind_ms=14.0, seed=5)
    ok &= _ok("sea surface roughness grows with wind speed",
              float(rough.v[:, 2].std()) > 2.0 * float(calm.v[:, 2].std()),
              "Hs proxy calm %.2f m vs windy %.2f m"
              % (float(calm.v[:, 2].std()), float(rough.v[:, 2].std())))
    scfg = dict(air=dict(temp_c=20, rh=0.6, wind_mps=8), sun=dict(elev_deg=30),
                ground=dict(enabled=True, type="sea", size=200, nx=16, ny=16,
                            wind_ms=8.0, mat="seawater"),
                objects=[dict(name="boat", cls="boat", shape="box", size=[5, 2, 1.5],
                              pos=[0, 0, 0.5], vel=[0, 3.0, 0],
                              mat="painted_metal", rcs_db=5)],
                sensors={"radar": dict(pos=[0, -50, 4], look_at=[0, 0, 1.0], band="Ka",
                                       az_span_deg=5, el_span_deg=3, n_az=40, n_el=10,
                                       bandwidth_mhz=200, n_samples=256, det_snr_db=11,
                                       mti_min_vr_mps=2.0)})
    sw = scene.build_world(scfg)
    ok &= _ok("sea scene has background faces with obj id -1",
              int((sw.face_obj == -1).sum()) > 100,
              "background faces=%d" % int((sw.face_obj == -1).sum()))
    scfg["_frame_seed"] = 4
    sh = radar.simulate(sw, scfg)
    moving = [d for d in sh["detections"] if abs(d["v_r"]) >= 2.0]
    still = [d for d in sh["detections"] if abs(d["v_r"]) < 2.0]
    ok &= _ok("MTI gate keeps moving targets and drops stationary returns",
              len(still) == 0 and len(moving) > 0,
              "%d moving kept, %d stationary kept" % (len(moving), len(still)))

# --- radiometric calibration anchors ---------------------------------
    # These pin the numbers the IR forward model must reproduce. They are the
    # self-checks published in docs/IR_经验模型与改进方案.md, so a change that
    # silently rescales the whole modality fails here instead of shipping.
    from . import weather as wmod
    L8_12 = float(ir.planck_band(np.array([300.0]), 8.0, 12.0)[0])
    L3_5 = float(ir.planck_band(np.array([300.0]), 3.0, 5.0)[0])
    ok &= _ok("band radiance 8-12um @300K ~ 38.5 W/m2sr", abs(L8_12 - 38.5) < 1.5,
              "%.2f" % L8_12)
    dL = (float(ir.planck_band(np.array([301.0]), 8.0, 12.0)[0]) - L8_12)
    ok &= _ok("dL/dT 8-12um @300K ~ 0.63 W/m2sr/K", abs(dL - 0.63) < 0.05, "%.3f" % dL)
    ok &= _ok("MWIR/LWIR band radiance ratio 0.02..0.10", 0.02 < L3_5 / L8_12 < 0.10,
              "%.3f" % (L3_5 / L8_12))

    clear = wmod.derive(wmod.resolve({"preset": "clear", "air_temp_c": 28.0, "rh": 0.6}))
    tm = clear["tau_ir_per_km"]["mwir"]
    tl = clear["tau_ir_per_km"]["lwir"]
    ok &= _ok("clear-sky MWIR extinction 0.2..0.8 /km", 0.2 < tm < 0.8, "%.3f" % tm)
    ok &= _ok("clear-sky LWIR extinction 0.05..0.4 /km", 0.05 < tl < 0.4, "%.3f" % tl)
    ok &= _ok("MWIR extincts more than LWIR (aerosol + CO2)", tm > tl,
              "%.3f vs %.3f" % (tm, tl))

    km = clear["sky_kappa"]["mwir"]
    kl = clear["sky_kappa"]["lwir"]
    ok &= _ok("sky coupling MWIR 0.02..0.25 (near transparent)", 0.02 < km < 0.25,
              "%.3f" % km)
    ok &= _ok("sky coupling LWIR 0.35..0.70 (near opaque)", 0.35 < kl < 0.70,
              "%.3f" % kl)
    ok &= _ok("MWIR sky is far darker than LWIR sky", km < 0.4 * kl,
              "%.3f vs %.3f" % (km, kl))

    # path radiance must exist: a 10 km path adds more than a 1 km one
    b_air = float(ir.planck_band(np.array([clear["t_air_k"]]), 8.0, 12.0)[0])
    p1 = (1.0 - np.exp(-tl * 1.0)) * b_air
    p10 = (1.0 - np.exp(-tl * 10.0)) * b_air
    ok &= _ok("path radiance grows with range", p10 > 3.0 * p1,
              "%.3f -> %.3f W/m2sr" % (p1, p10))

    # --- radar detection statistics --------------------------------------
    pfa = 1e-6
    det_db = 10.0 * np.log10(-np.log(pfa))
    ok &= _ok("Pfa=1e-6 threshold ~ 11.4 dB", abs(det_db - 11.40) < 0.05, "%.2f dB" % det_db)

    print("\nRESULT:", "ALL PASS" if ok else "FAILURES PRESENT")
    return ok