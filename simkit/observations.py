"""Explicit geometry and timing for research datasets (not detectability labels)."""
import numpy as np

from . import labels, scene, thermal


def describe(state, world):
    cfg = state["cfg"]
    t = float(state["t"])
    platform_v = np.asarray(state["sensor_velocity_mps"], float)
    cameras = {}
    for name, cam in state["views"].items():
        W, H = int(cam["width"]), int(cam["height"])
        p, r, u, f = scene.sensor_basis(cam["pos"], cam["look_at"],
                                      roll_deg=cam.get("roll_deg", 0.0))
        fx = W / (2.0 * np.tan(np.radians(cam["fov_deg"]) / 2.0))
        R = np.array([r, -u, f])  # CV camera: right, down, forward
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, -R @ p
        cameras[name] = dict(
            K=[[fx, 0.0, W/2.0], [0.0, fx, H/2.0], [0.0, 0.0, 1.0]],
            world_to_camera=T.tolist(), width=W, height=H,
            pose=cam, distortion_model="none", pixel_coordinates="continuous image-edge coordinates")
    primary = next(iter(state["views"].values()))
    p = np.asarray(primary["pos"], float)
    objects = []
    override_temperature, override_flux = thermal.region_overrides(world)
    triangles = world.verts[world.faces]
    areas = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]), axis=1)/2
    for obj in world.objs:
        if not obj.get("label", True):
            continue
        center = np.asarray(obj["corners"]).mean(axis=0)
        offset = center - p
        distance = float(np.linalg.norm(offset))
        relative_v = np.asarray(obj["vel"]) - platform_v
        closing = -float(offset @ relative_v) / max(distance, 1e-12)
        projections = {}
        for name, cam in state["views"].items():
            projected = labels.project_points(obj["corners"], cam, cam["width"], cam["height"])
            center_px = labels.project_points(center[None], cam, cam["width"], cam["height"])[0]
            projections[name] = dict(center_px=center_px[:2].tolist(),
                                     depth_m=float(center_px[2]),
                                     extent_px=np.ptp(projected[:, :2], axis=0).tolist(),
                                     geometry_only=True)
        objects.append(dict(track_id=obj["name"], cls=obj["cls"],
                            center_world_m=center.tolist(), origin_world_m=np.asarray(obj["pos"]).tolist(),
                            velocity_mps=np.asarray(obj["vel"]).tolist(), yaw_deg=float(obj["yaw"]),
                            range_m=distance, relative_velocity_mps=relative_v.tolist(),
                            closing_speed_mps=closing, projections=projections,
                            source_material_report=obj.get("source_material_report"),
                            heat_annotations=dict(
                                regions=obj.get("thermal_regions", []),
                                matched_flux_faces=int(np.count_nonzero((world.face_obj == obj["id"]) & np.isfinite(override_flux))),
                                matched_flux_area_m2=float(areas[(world.face_obj == obj["id"]) & np.isfinite(override_flux)].sum()),
                                matched_temperature_faces=int(np.count_nonzero((world.face_obj == obj["id"]) & np.isfinite(override_temperature))))))
    rc = cfg.get("sensors", {}).get("radar", {})
    pri = float(rc.get("pri_us", 65.0))*1e-6
    duration = ((int(rc.get("n_chirps", 32))-1)*pri +
                float(rc.get("chirp_us", 50.0))*1e-6)
    exposure = cfg.get("acquisition", {})
    rgb_motion = bool((cfg.get("optical") or {}).get("motion_blur") and state.get("shutter_samples"))
    return dict(schema_version=1, frame=state["frame"], clip=state["clip"],
                timestamp_s=t, sequence_timestamp_s=state["frame"]*float(cfg["frames"]["dt"]),
                conventions=dict(world="right-handed Z-up, metres", velocity="m/s",
                                 radar_velocity="positive closing; stored RD axis is aliased"),
                observer=dict(position_m=p.tolist(), velocity_mps=platform_v.tolist()),
                cameras=cameras, objects=objects,
                acquisition=dict(
                    image_reference_time_s=t,
                    rgb_exposure_s=exposure.get("rgb_exposure_s"),
                    ir_exposure_s=exposure.get("ir_exposure_s"),
                    exposure_integration_simulated=rgb_motion,
                    rgb_motion_integration="Cycles two endpoint rigid transforms" if rgb_motion else "none",
                    rgb_interval_s=[t-float(exposure.get("rgb_exposure_s", 0))/2,
                                    t+float(exposure.get("rgb_exposure_s", 0))/2],
                    ir_exposure_integration_simulated=False,
                    radar_interval_s=[t-duration/2, t+duration/2],
                    radar_duration_s=duration, radar_geometry_model="frozen_at_reference_time",
                    motion_during_radar_cpi_simulated=False),
                assumptions=cfg.get("research_baseline", {}))
