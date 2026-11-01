"""Civil airborne-observation baseline: 2 official models x 3 flight geometries.

One labelled aircraft per run. Research assumptions, not calibrated equipment.
Use a new output directory on every build to prevent mixed-generation datasets.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import platform
import shutil

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def config(role, geometry, frames=12, dt=0.2, realism=False):
    target_speed = 200.0 if role == "airliner" else 40.0
    target_vel = {"parallel": [target_speed, 0, 0],
                  "crossing": [0, target_speed, 0],
                  "opposite": [-target_speed, 0, 0]}[geometry]
    observer = [0, 0, 5500]
    target = [22000, 5000, 3500]
    cam = dict(pos=observer, look_at=target, fov_deg=2.2, width=640, height=480)
    cfg = dict(
        name="civil_%s_%s" % (role, geometry), seed=20261101,
        frames=dict(n=frames, dt=dt),
        ground=dict(enabled=True, size=160000, z=0, mat="foliage"),
        weather=dict(mode="fixed", preset="clear", air_temp_c=10, rh=0.4),
        air=dict(temp_c=10, rh=0.4, wind_mps=3),
        sun=dict(on=True, elev_deg=35, az_deg=130),
        objects=[dict(name=role+"_01", cls="airplane" if role == "airliner" else "uav",
                      shape="obj", model=str(ROOT / "assets" / "aviation" / (role+".obj")),
                      mat="painted_metal", pos=target, vel=target_vel, rot=0,
                      decimate=5000, label=True)],
        sensors=dict(optical=dict(cam), ir_camera=dict(cam), radar=dict(
            pos=observer, look_at=target, band="X", bandwidth_mhz=2.5,
            n_samples=1024, n_chirps=32, chirp_us=100, pri_us=125,
            az_span_deg=2.2, el_span_deg=1.65, n_az=8, n_el=6,
            beamwidth_az_deg=2, beamwidth_el_deg=6,
            tx_power_dbm=40, noise_figure_db=8, pfa=1e-6,
            mti_min_vr_mps=0, clutter_density=0.02, swerling="I")),
        ir=dict(bands=["mwir", "lwir"], netd_mk=30, psf_px=1.2,
                texture_sigma_c=0, nuc=dict(enabled=True, seed=12345)),
        optical=dict(samples=8, ground_detail=True),
        acquisition=dict(rgb_exposure_s=0.001, ir_exposure_s=0.001),
        clip=dict(frames_per_clip=frames, camera="optical",
                  observer_trajectory=dict(mode="line", origin=observer, vel=[220, 0, 0]),
                  trajectory=dict(mode="line", origin=target, vel=target_vel),
                  rig=dict(lag_s=0.15, vibration_m=0, vibration_deg=0),
                  views=[dict(name="primary", tag="", **{k: cam[k] for k in ("fov_deg", "width", "height")})]),
        research_baseline=dict(
            purpose="civil airborne aircraft detection and tracking research",
            status="geometry and data-contract baseline; not validated against real sensors",
            pointing="ground-truth target position with pure delay; not a working tracker",
            classes="model-instance identity; not proof that crew status is visually observable",
            attitude="level-flight yaw only; stabilized camera roll; no full aircraft dynamics",
            atmosphere="existing homogeneous empirical atmosphere; no altitude-profile integration",
            materials="single painted-metal preset; source GLB textures not yet used by renderer",
            thermal="steady-state surface balance; no engine segmentation or heat inertia",
            radar="uncalibrated research FMCW preset; frozen geometry; no range migration",
            environment="procedural flat ground, fixed clear weather",
            split="QA only; no independent train/val/test population"))
    if realism:
        obj = cfg["objects"][0]
        obj.update(import_materials=True,
                   optical_material_source=str(ROOT/"assets/aviation"/(role+".glb")),
                   material_map={"PaintM": "painted_metal", "PaintPM": "painted_metal",
                                 "MetalM": "metal_bare", "MetalPM": "metal_bare",
                                 "GlassM": "glass", "GlassPM": "glass",
                                 "RubberPM": "rubber", "PlasticM": "plastic", "PlasticPM": "plastic"})
        # Explicit research heat-source regions, not measured engine fields.
        # Bounds are in normalized OBJ local coordinates, prior to world pose.
        obj["thermal_regions"] = ([
            dict(name="left_nozzle_research", material_names=["MetalM"], bounds_local_m=[[2, -11.5, 0], [5, -8.5, 3]],
                 heat_flux_w_m2=500),
            dict(name="right_nozzle_research", material_names=["MetalM"], bounds_local_m=[[2, 8.5, 0], [5, 11.5, 3]],
                 heat_flux_w_m2=500)] if role == "airliner" else [
            dict(name="engine_research", material_names=["MetalM"], bounds_local_m=[[-0.8, -0.2, 0.4], [-0.4, 0.2, 1.0]],
                 heat_flux_w_m2=150)])
        cfg["atmosphere"] = dict(profile="exponential", reference_altitude_m=0,
                                  aerosol_scale_height_m=1200, molecular_scale_height_m=8000,
                                  water_scale_height_m=2000, temperature_at_reference_c=20)
        cfg["optical"].update(samples=16, export_geometry=True, motion_blur=True,
            sensor=dict(enabled=True, psf_sigma_px=0.6, full_well_e=30000,
                        read_noise_e=3, prnu=0.001, bits=12, reference_exposure_s=0.001,
                        airlight_linear_rgb=[0.35, 0.45, 0.60], seed=12345))
        cfg["ir"].update(samples_per_axis=2, subpixel_mode="coverage", texture_sigma_c=0.3,
                         texture_seed=12345)
        cfg["sensors"]["radar"].update(detector="rd_threshold", max_detections=256)
        cfg["research_baseline"].update(
            status="realism v3; empirical parameters, no real-sensor validation",
            atmosphere="altitude-profile Beer-Lambert and band path emission; empirical broadband coefficients",
            materials="source PBR and UV retained; explicit source-name physical-material mapping",
            thermal="local research heat-flux regions; steady state, no measured engine temperatures",
            radar="measured RD peak detector with window/noise calibration; frozen CPI, approximate RCS",
            rgb="scene-linear atmosphere/PSF/electron response; endpoint motion exposure; no absolute radiometric calibration")
    return cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "out" / "aviation_baseline_v1")
    ap.add_argument("-n", "--frames", type=int, default=12)
    ap.add_argument("--dt", type=float, default=0.2)
    ap.add_argument("--sensors", default="optical,ir,radar")
    ap.add_argument("--config-only", action="store_true")
    ap.add_argument("--realism", action="store_true", help="enable source PBR, altitude atmosphere and sensor models")
    ap.add_argument("--only", help="one run, e.g. airliner_parallel")
    args = ap.parse_args()
    if args.frames < 2 or args.dt <= 0:
        ap.error("need >=2 frames and dt>0")
    # Never blend a new parameter set with existing images.
    if args.out.exists():
        ap.error("output already exists; choose a fresh --out")
    for role in ("airliner", "uav"):
        if not (ROOT / "assets" / "aviation" / (role+".obj")).exists():
            ap.error("download/prepare models first; see docs/民航机载基线.md")
    chosen = [(r, g) for r in ("airliner", "uav")
              for g in ("parallel", "crossing", "opposite")
              if args.only is None or args.only == r+"_"+g]
    if not chosen:
        ap.error("unknown --only run")
    args.out.mkdir(parents=True)
    source_paths = (list((ROOT / "simkit").glob("*.py")) + list((ROOT / "blender").glob("*.py"))+
                    list((ROOT / "tests").glob("*.py")) + [ROOT / "requirements.txt"]+
                    [Path(__file__),
                     ROOT / "tools/check_aviation.py", ROOT / "tools/prepare_aviation_models.py",
                     ROOT / "tools/fetch_aviation_models.py"])
    def git(*argv):
        return subprocess.run(["git", *argv], cwd=ROOT, capture_output=True,
                              text=True, encoding="utf-8", errors="replace", check=True).stdout.strip()
    manifest = dict(schema_version=1, command=sys.argv, python=sys.version,
                    platform=platform.platform(), numpy=np.__version__,
                    git_head=git("rev-parse", "HEAD"), git_worktree_dirty=bool(git("status", "--porcelain")),
                    source_sha256={str(p.relative_to(ROOT)).replace("\\", "/"):
                                   hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
                    assets=read_json(ROOT / "assets/aviation/sources.json"),
                    geometry=read_json(ROOT / "assets/aviation/geometry.json"),
                    configs={}, runs={})
    for p in source_paths:
        snapshot = args.out / "_source_snapshot" / p.relative_to(ROOT)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, snapshot)
    for role, geometry in chosen:
        cfg = config(role, geometry, args.frames, args.dt, realism=args.realism)
        path = args.out / (role+"_"+geometry+".json")
        path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest["configs"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest["runs"][role+"_"+geometry] = "configured" if args.config_only else "running"
        (args.out / "build_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if not args.config_only:
            try:
                subprocess.run([sys.executable, "-m", "simkit", "gen", "-c", str(path),
                                "-o", str(args.out / (role+"_"+geometry)),
                                "-s", args.sensors], cwd=ROOT, check=True)
            except subprocess.CalledProcessError:
                manifest["runs"][role+"_"+geometry] = "failed"
                (args.out / "build_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                raise
            manifest["runs"][role+"_"+geometry] = "complete"
            (args.out / "build_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Baseline output:", args.out)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
