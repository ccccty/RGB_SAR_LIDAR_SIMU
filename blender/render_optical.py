# -*- coding: utf-8 -*-
"""Runs INSIDE Blender: render the visible-band frames and export boxes.

  blender -b --factory-startup -P render_optical.py -- cfg.json out_dir manifest.json project_root

Multi-view
----------
A clip may carry several fields of view at once (a narrow seeker camera and a
wide search camera on the same mount).  Views are declared in

    optical.views = [{"name": "seeker", "tag": "",      "fov_deg": 2.2,  "width": 960,  "height": 720},
                     {"name": "wide",   "tag": "_wide", "fov_deg": 18.0, "width": 1280, "height": 960}]

and the per-frame poses arrive in the manifest as

    frame["views"][name] = {"pos": [...], "look_at": [...], "fov_deg": ..,
                            "width": .., "height": .., "roll_deg": ..}

The first view is the primary one and keeps writing to images/optical for
backwards compatibility; every other view writes to images/optical<TAG>.
When the config declares no views the renderer behaves exactly as before: one
camera taken from sensors[<CAM_KEY>] and the per-frame "camera" block.
"""
import json
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Quaternion, Vector

argv = sys.argv[sys.argv.index("--") + 1:]
CFG_PATH, OUT_DIR, MANIFEST, PROJECT_ROOT = argv[0], argv[1], argv[2], argv[3]
OUT_DIR = os.path.abspath(OUT_DIR)
MANIFEST = os.path.abspath(MANIFEST)
CFG_PATH = os.path.abspath(CFG_PATH)
sys.path.insert(0, PROJECT_ROOT)

from simkit import scene as sk_scene  # noqa: E402

cfg = sk_scene.load_config(CFG_PATH)
opt = cfg.get("optical", {})
CAM_KEY = opt.get("camera_key", "optical")          # the sensor this renderer draws


def _views():
    """Declared views, or the legacy single camera from sensors[CAM_KEY]."""
    vs = opt.get("views")
    if not vs:
        c = cfg["sensors"][CAM_KEY]
        return [dict(name=CAM_KEY, tag="", fov_deg=float(c.get("fov_deg", 45.0)),
                     width=int(c.get("width", 960)), height=int(c.get("height", 720)))]
    out = []
    for i, v in enumerate(vs):
        c = cfg["sensors"].get(CAM_KEY) or {}
        out.append(dict(name=str(v.get("name") or ("view%d" % i)),
                        tag=str(v.get("tag", "") if i else ""),
                        fov_deg=float(v.get("fov_deg", c.get("fov_deg", 45.0))),
                        width=int(v.get("width", c.get("width", 960))),
                        height=int(v.get("height", c.get("height", 720)))))
    # the primary view must land on the legacy path, whatever tag it was given
    out[0]["tag"] = ""
    return out


VIEWS = _views()
PRIMARY = VIEWS[0]
SAMPLES = int(opt.get("samples", 64))
ENGINE = opt.get("engine", "CYCLES")


def log(msg):
    print("[simkit] " + str(msg), flush=True)


def purge():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def make_material(mat_name):
    from simkit import materials as matlib
    spec = matlib.get(mat_name)
    m = bpy.data.materials.new(mat_name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes.get("Principled BSDF")
    rgb = spec["albedo"]
    bsdf.inputs["Base Color"].default_value = (rgb[0], rgb[1], rgb[2], 1.0)
    bsdf.inputs["Roughness"].default_value = 0.25 if mat_name in ("painted_metal", "glass") else 0.65
    bsdf.inputs["Metallic"].default_value = 0.85 if mat_name == "metal_bare" else 0.1
    if mat_name == "glass":
        bsdf.inputs["Roughness"].default_value = 0.05
        bsdf.inputs["Metallic"].default_value = 0.0
    return m


def gradient_sky(sc, bg, nt, opt, sun_elev_deg, cloud):
    """Analytic zenith->horizon gradient, used when the ground is not in frame.

    Blender's Nishita sky is tuned for an observer near sea level: asked for a
    seeker at 9.5 km it returned a flat warm grey (measured RGB 153/144/118)
    instead of the deep blue the same physics gives. Rather than fight the
    model, air-only scenes get an explicit gradient whose two endpoints are
    stated in the config, so the look is a parameter and not an accident of a
    tonemapper.
    """
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    coord = nt.nodes.new("ShaderNodeTexCoord")
    coord.location = (-900, 0)
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    sep.location = (-700, 0)
    rng = nt.nodes.new("ShaderNodeMapRange")
    rng.location = (-520, 0)
    rng.inputs["From Min"].default_value = -0.15
    rng.inputs["From Max"].default_value = 0.85
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.location = (-340, 0)
    zen = opt.get("sky_zenith_rgb", [0.030, 0.075, 0.220])
    hor = opt.get("sky_horizon_rgb", [0.52, 0.60, 0.72])
    ramp.color_ramp.elements[0].position = 0.0
    ramp.color_ramp.elements[0].color = (zen[0], zen[1], zen[2], 1.0)
    ramp.color_ramp.elements[1].position = 1.0
    ramp.color_ramp.elements[1].color = (hor[0], hor[1], hor[2], 1.0)
    bg2 = nt.nodes.new("ShaderNodeBackground")
    bg2.location = (-100, 0)
    out = nt.nodes.new("ShaderNodeOutputWorld")
    out.location = (120, 0)
    nt.links.new(coord.outputs["Generated"], sep.inputs["Vector"])
    nt.links.new(sep.outputs["Z"], rng.inputs["Value"])
    nt.links.new(rng.outputs["Result"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bg2.inputs["Color"])
    # a thin sun elevation term keeps a low sun from being over-brightened; the
    # disc itself is not drawn, a seeker's exposure is set by the sky, not by it
    gain = float(opt.get("sky_strength", 1.0)) * (1.0 - 0.45 * min(max(cloud, 0.0), 1.0))
    gain *= 0.55 + 0.45 * max(0.0, math.sin(math.radians(max(sun_elev_deg, 0.0))))
    bg2.inputs["Strength"].default_value = float(gain)
    nt.links.new(bg2.outputs["Background"], out.inputs["Surface"])
    return bg2


def add_ground_detail(m, mat_name, scale_m=45.0, contrast=0.22, bump=0.25):
    """Procedural albedo/bump variation for the ground plane.

    A single flat facet spanning 20 km renders as one uniform grey field: from a
    seeker at 2 km the whole frame is the same pixel value, so the incoming
    ground gives no texture, no scale cue and no motion parallax -- the exact
    things a detection/tracking model is supposed to learn from. Two noise
    octaves (a ~45 m patchiness plus fine grain) restore that without any
    external texture files, which keeps the build offline and reproducible.
    """
    from simkit import materials as matlib
    nt = m.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    if bsdf is None:
        return m
    base = list(matlib.get(mat_name)["albedo"])
    coord = nt.nodes.new("ShaderNodeTexCoord")
    coord.location = (-1000, 0)
    # "Object" coordinates: the ground object sits at the world origin, so the
    # pattern is world-fixed and consecutive frames see the same field moving
    # under the airframe instead of a boiling texture.
    tex = nt.nodes.new("ShaderNodeTexNoise")
    tex.location = (-800, 0)
    tex.inputs["Scale"].default_value = 1.0 / max(float(scale_m), 0.5)
    tex.inputs["Detail"].default_value = 9.0
    tex.inputs["Roughness"].default_value = 0.62
    fine = nt.nodes.new("ShaderNodeTexNoise")
    fine.location = (-800, -300)
    fine.inputs["Scale"].default_value = 1.0 / max(float(scale_m) * 0.08, 0.05)
    fine.inputs["Detail"].default_value = 4.0
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "FLOAT"
    mix.location = (-600, -150)
    mix.inputs["Factor"].default_value = 0.35
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.location = (-420, 0)
    lo = [max(0.0, min(1.0, c * (1.0 - contrast))) for c in base]
    hi = [max(0.0, min(1.0, c * (1.0 + contrast))) for c in base]
    ramp.color_ramp.elements[0].position = 0.35
    ramp.color_ramp.elements[0].color = (lo[0], lo[1], lo[2], 1.0)
    ramp.color_ramp.elements[1].position = 0.65
    ramp.color_ramp.elements[1].color = (hi[0], hi[1], hi[2], 1.0)
    bmp = nt.nodes.new("ShaderNodeBump")
    bmp.location = (-200, -260)
    bmp.inputs["Strength"].default_value = float(bump)
    bmp.inputs["Distance"].default_value = 0.06
    nt.links.new(coord.outputs["Object"], tex.inputs["Vector"])
    nt.links.new(coord.outputs["Object"], fine.inputs["Vector"])
    nt.links.new(tex.outputs["Fac"], mix.inputs[2])
    nt.links.new(fine.outputs["Fac"], mix.inputs[3])
    nt.links.new(mix.outputs[0], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(mix.outputs[0], bmp.inputs["Height"])
    nt.links.new(bmp.outputs["Normal"], bsdf.inputs["Normal"])
    return m


def add_mesh_object(name, mesh, mat_name, location, yaw_deg):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in mesh.v], [], [tuple(f) for f in mesh.f])
    me.validate()
    me.update()
    ob = bpy.data.objects.new(name, me)
    ob.location = Vector(tuple(float(x) for x in location))
    ob.rotation_euler = (0.0, 0.0, math.radians(float(yaw_deg)))
    bpy.context.scene.collection.objects.link(ob)
    ob.data.materials.append(make_material(mat_name))
    return ob


def build_scene():
    purge()
    sc = bpy.context.scene
    sc.render.engine = ENGINE
    if ENGINE == "CYCLES":
        sc.cycles.samples = SAMPLES
        sc.cycles.use_denoising = True
        sc.cycles.device = "CPU"
    sc.render.resolution_x = PRIMARY["width"]
    sc.render.resolution_y = PRIMARY["height"]
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGB"
    sc.render.image_settings.color_depth = "8"
    wanted = opt.get("view_transform", "Standard")
    for vt in ([wanted] if wanted else []) + ["AgX", "Filmic", "Standard"]:
        try:
            sc.view_settings.view_transform = vt
            break
        except Exception:
            continue
    sc.view_settings.exposure = float(opt.get("exposure", -2.5))

    # world / sky
    world = bpy.data.worlds.new("simkit_world")
    sc.world = world
    world.use_nodes = True
    nt = world.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    bg = nt.nodes.new("ShaderNodeBackground")
    sky = nt.nodes.new("ShaderNodeTexSky")
    sky.sky_type = "NISHITA"
    try:
        sky.turbidity = float(opt.get("turbidity", 2.5))
    except Exception:
        pass
    sky.sun_elevation = math.radians(float(cfg["sun"].get("elev_deg", 35.0)))
    sky.sun_rotation = math.radians(float(cfg["sun"].get("az_deg", 120.0)))
    sky.sun_intensity = 0.5
    # Blender's Nishita sky defaults to sea level, which is why a seeker at 9 km
    # rendered an olive-grey haze instead of the deep blue you actually see up
    # there: the sky model has to be told the altitude it is being viewed from.
    try:
        alt_km = opt.get("sky_altitude_km")
        if alt_km is None:
            eye = (cfg.get("sensors") or {}).get(CAM_KEY) or {}
            alt_km = max(0.0, float((eye.get("pos") or [0, 0, 0])[2]) / 1000.0)
        sky.altitude = float(alt_km)
        # Blender ships dust_density = 1.0, i.e. a hazy desert haze. Rendered from
        # 9 km that turned the whole sky olive-brown; at altitude the air is
        # clean, so both densities are set explicitly from the weather state.
        sky.dust_density = float(opt.get("sky_dust", 0.15))
        sky.air_density = float(opt.get("sky_air", 1.0))
        sky.sun_disc = bool(opt.get("sun_disc", False))
    except Exception:
        pass
    out = nt.nodes.new("ShaderNodeOutputWorld")
    nt.links.new(sky.outputs[0], bg.inputs[0])
    nt.links.new(bg.outputs[0], out.inputs[0])
    bg.inputs[1].default_value = float(opt.get("sky_strength", 0.35)) if cfg["sun"].get("on", True) else 0.03

    sun_obj = None
    sun_energy = 0.0
    # sun lamp (energy in W/m^2)
    if cfg["sun"].get("on", True):
        ld = bpy.data.lights.new("sun", type="SUN")
        # physical W/m^2 -> Blender sun strength (800 W/m^2 ~ 2.5, a normal clear sky)
        ld.energy = float(cfg["sun"].get("irradiance", 800.0)) * float(opt.get("sun_scale", 1.0 / 320.0))
        ld.angle = math.radians(0.53)
        lo = bpy.data.objects.new("sun", ld)
        el = math.radians(float(cfg["sun"].get("elev_deg", 35.0)))
        az = math.radians(float(cfg["sun"].get("az_deg", 120.0)))
        to_sun = Vector((math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)))
        lo.rotation_euler = to_sun.to_track_quat("Z", "Y").to_euler()
        sc.collection.objects.link(lo)
        sun_obj = lo
        sun_energy = float(ld.energy)

    # ground (optional: air scenes disable it)
    from simkit.mesh import plane
    g = cfg.get("ground", {})
    if g.get("enabled", True):
        gm = plane(float(g.get("size", 120.0)), float(g.get("z", 0.0)), 1, 1)
        gmat = g.get("mat", "asphalt")
        gob = add_mesh_object("ground", gm, gmat, (0, 0, 0), 0.0)
        if opt.get("ground_detail", True) and gob.data.materials:
            is_sea = str(g.get("type", "ground")).lower() in ("sea", "ocean", "water")
            # Feature size has to be small compared with what the lens resolves:
            # a 2.2 deg seeker covers ~76 m at 2 km, so 45 m "patches" read as
            # four grey bands. 3-6 m gives patchiness plus grain and still
            # survives downsampling in the gallery.
            add_ground_detail(gob.data.materials[0], gmat,
                              scale_m=float(opt.get("ground_scale_m",
                                                    3.0 if is_sea else 6.0)),
                              contrast=float(opt.get("ground_contrast",
                                                     0.45 if is_sea else 0.32)),
                              bump=float(opt.get("ground_bump", 0.45)))

    # objects
    meshes = {}
    for object_index, spec in enumerate(cfg.get("objects", [])):
        # prepare_mesh applies decimate/scale/height exactly as the numpy world
        # builder does, and returns the recentred bbox used for projections.
        m, lo, hi = sk_scene.prepare_mesh(spec, cfg.get("_base_dir", "."))
        sizes = hi - lo
        ob = add_mesh_object(spec.get("name", "obj"), m, spec.get("mat", "painted_metal"),
                             spec.get("pos", (0, 0, 0)), spec.get("rot", 0.0))
        ob.pass_index = object_index+1
        from blender.material_helpers import apply_source_materials
        material_report = apply_source_materials(ob, m, spec, cfg.get("_base_dir", "."))
        meshes[spec.get("name", "obj")] = dict(ob=ob, size=sizes, lo=lo, hi=hi,
                                               label=bool(spec.get("label", True)),
                                               material_report=material_report)

    # camera
    cd = bpy.data.cameras.new("cam")
    cd.lens_unit = "FOV"
    cd.sensor_fit = "HORIZONTAL"
    cd.angle = math.radians(float(PRIMARY["fov_deg"]))
    # Blender's default clip range is 0.1..100 m, which silently clips any
    # target farther away than 100 m. Size the frustum to the actual scene, and
    # include the wide view, which reaches farther off-axis than the narrow one.
    cam0 = cfg["sensors"][CAM_KEY]
    reach = 1.0
    for _n, _m in meshes.items():
        for _c in (_m["lo"], _m["hi"]):
            reach = max(reach, (Vector(tuple(float(x) for x in _c))
                                - Vector(tuple(float(x) for x in cam0["pos"]))).length)
    # a narrow seeker sits far from its target; leave headroom for the wide
    # lens so a single clip range serves every view
    reach *= max(1.0, float(opt.get("reach_scale", 1.0)))
    cd.clip_start = max(0.01, min(1.0, reach * 1e-4))
    cd.clip_end = reach * 4.0 + 100.0
    cam = bpy.data.objects.new("cam", cd)
    pos = Vector(tuple(float(x) for x in cam0["pos"]))
    cam.location = pos
    look = Vector(tuple(float(x) for x in cam0.get("look_at", [0, 0, 0])))
    cam.rotation_euler = (look - pos).to_track_quat("-Z", "Y").to_euler()
    sc.collection.objects.link(cam)
    sc.camera = cam
    env = dict(sky=sky, bg=bg, sun_obj=sun_obj, sun_energy=sun_energy,
               base_bg=float(bg.inputs[1].default_value),
               base_turb=float(getattr(sky, "turbidity", 2.5)),
               grad_bg=None, grad_base=0.0)
    # An air-only frame has nothing but sky in it, so the sky is the picture. See
    # gradient_sky() for why the Nishita model is not used there.
    use_grad = (str(opt.get("sky_mode", "")).lower() == "gradient"
                or not cfg.get("ground", {}).get("enabled", True))
    if use_grad:
        g = gradient_sky(sc, bg, nt, opt, float(cfg["sun"].get("elev_deg", 35.0)), 0.2)
        env["grad_bg"] = g
        env["grad_base"] = float(g.inputs["Strength"].default_value)
        # the Nishita nodes were removed to make room for the gradient: holding a
        # stale reference to a freed node is a segfault, not an exception
        env["sky"] = None
        env["bg"] = g
    return sc, cam, meshes, env


def aim_camera(sc, cam, pose, view, cam_key):
    """Point the single Blender camera at one view's pose, roll included.

    A banked seeker rolls the horizon; refusing to model it would teach a
    detector that the world is always level, which is exactly the invariance a
    real gimbal destroys on a weaving dive.
    """
    w = int(view["width"])
    h = int(view["height"])
    sc.render.resolution_x = w
    sc.render.resolution_y = h
    fov = float(pose.get("fov_deg") or view["fov_deg"])
    cam.data.angle = math.radians(fov)
    cpos = Vector(tuple(float(x) for x in pose["pos"]))
    cam.location = cpos
    look = Vector(tuple(float(x) for x in pose.get("look_at", [0, 0, 0])))
    if (look - cpos).length > 1e-6:
        q = (look - cpos).to_track_quat("-Z", "Y")
        roll = float(pose.get("roll_deg", 0.0) or 0.0)
        if roll:
            q = q @ Quaternion((0.0, 0.0, 1.0), math.radians(roll))
        cam.rotation_mode = "QUATERNION"
        cam.rotation_quaternion = q
    bpy.context.view_layer.update()
    # Model-local bounds omit object translation. The old far plane could clip
    # a 22 km aircraft while still exporting its projected label. Recompute
    # after every world/camera pose update, including the exposure probe.
    from simkit.optical import camera_clip_end
    world_corners = [tuple(ob.matrix_world @ Vector(c))
                     for ob in sc.objects if ob.type == "MESH" for c in ob.bound_box]
    cam.data.clip_end = camera_clip_end(tuple(cam.location), world_corners)


def pose_for(fr, view):
    """Resolve the pose of one view in one frame, tolerant of both layouts."""
    vs = fr.get("views") or {}
    p = vs.get(view["name"])
    if p:
        return p
    # legacy: a single "camera" block keyed by sensor name
    camf = fr.get("camera") or {}
    p = camf.get(CAM_KEY) or (list(camf.values())[0] if camf else None)
    return p or {}


def auto_exposure(sc, view, base_bias):
    """Iterate the film exposure until frame 0 meters to a normal brightness.

    A single exposure number cannot serve both ends of this dataset: a seeker at
    9 km against sky needs several stops more than a launcher 800 m from sunlit
    asphalt, and the fixed values I tried produced a near-black sky in one scene
    and blown-white ground in the other (both measured).

    The measured value is the *display* mean of a 24x18 one-sample probe. Blender
    hands back byte-image pixels already encoded, so a target of 0.48 here means
    "a normally exposed frame", not a linear-light radiance. A saturated probe
    (1.0) carries no gradient information, which is why this iterates: the first
    pass brackets, the second lands, the third is only for pathological scenes.
    Costs about a second per scene.
    """
    if not opt.get("auto_exposure", True):
        return float(opt.get("exposure", -2.5))
    target = float(opt.get("auto_target", 0.48))
    rx, ry = sc.render.resolution_x, sc.render.resolution_y
    samples = sc.cycles.samples if ENGINE == "CYCLES" else None
    denoise = sc.cycles.use_denoising if ENGINE == "CYCLES" else None
    old_exp = sc.view_settings.exposure
    probe = os.path.join(OUT_DIR, "_exposure_probe.png")
    exp = float(opt.get("exposure", 0.0))
    try:
        # each lens meters its own field of view: a 2.2 deg seeker and an 18 deg
        # search lens see wildly different amounts of sky, and one shared
        # exposure blows the wide one out (measured on the air-to-ground scene)
        sc.render.resolution_x, sc.render.resolution_y = 24, 18
        if ENGINE == "CYCLES":
            sc.cycles.samples = 1
            sc.cycles.use_denoising = False
        for i in range(5):
            sc.view_settings.exposure = exp
            sc.render.filepath = probe
            bpy.ops.render.render(write_still=True)
            img = bpy.data.images.load(probe)
            px = list(img.pixels)
            n = max(len(px) // 4, 1)
            lum = sum(0.2126 * px[j] + 0.7152 * px[j + 1] + 0.0722 * px[j + 2]
                      for j in range(0, len(px), 4)) / n
            bpy.data.images.remove(img)
            step = math.log(max(target, 1e-6) / max(lum, 1e-6), 2.0)
            log("auto exposure [%s] pass %d: mean %.3f -> %+.2f stops (at %+.2f)"
                % (view["name"], i + 1, lum, step, exp))
            if abs(step) < 0.12:
                break
            exp = max(-8.0, min(10.0, exp + step))
        return exp + float(base_bias or 0.0)
    except Exception as exc:
        log("auto exposure failed (%s), using configured %.2f"
            % (exc, float(opt.get("exposure", -2.5))))
        return float(opt.get("exposure", -2.5))
    finally:
        sc.render.resolution_x, sc.render.resolution_y = rx, ry
        if ENGINE == "CYCLES":
            sc.cycles.samples = samples
            sc.cycles.use_denoising = denoise
        sc.view_settings.exposure = old_exp
        try:
            os.remove(probe)
        except OSError:
            pass


def main():
    from bpy_extras.object_utils import world_to_camera_view
    sc, cam, meshes, env = build_scene()
    manifest = json.load(open(MANIFEST, "r", encoding="utf-8"))
    # Meter through the lens the clip actually looks through: the config pose is
    # a placeholder (a seeker's config camera points at the world origin, i.e.
    # straight down, while every real frame looks nearly level). Probing the
    # placeholder measured a different sky than the one being rendered and
    # underexposed the whole scene by more than a stop.
    exposures = {}
    if manifest["frames"]:
        # meter the MIDDLE frame of the clip, not the first: the urban scene opens
        # with a building filling the lens (a dark wall face), and metering that
        # raised the exposure until every later sky frame blew out to white. The
        # frame is chosen from the GLOBAL clip position so all chunk workers agree
        # -- metering each chunk's own middle frame puts a brightness step at
        # every chunk boundary.
        fpc = int((cfg.get("clip") or {}).get("frames_per_clip")
                  or (cfg.get("frames") or {}).get("n") or 0)
        mid = fpc // 2
        fr0 = min(manifest["frames"],
                  key=lambda f: abs((int(f.get("frame", 0)) % fpc) - mid) if fpc else 0)
        for nm, pose in (fr0.get("objects") or {}).items():
            if nm in meshes:
                ob = meshes[nm]["ob"]
                ob.location = Vector(tuple(float(x) for x in pose["pos"]))
                ob.rotation_euler = (0.0, 0.0, math.radians(float(pose["yaw"])))
        for v in VIEWS:
            pv = pose_for(fr0, v)
            if pv:
                aim_camera(sc, cam, pv, v, CAM_KEY)
            bpy.context.view_layer.update()
            exposures[v["name"]] = auto_exposure(
                sc, v, opt.get("exposure_bias", 0.0))
    base_exp = opt.get("exposure_bias", 0.0)
    export_geometry = bool(opt.get("export_geometry") or (opt.get("sensor") or {}).get("enabled"))
    pass_output = None
    if export_geometry:
        from blender.sensor_helpers import setup_passes
        pass_output = setup_passes(sc, os.path.join(OUT_DIR, "_passes"))
    for v in VIEWS:
        os.makedirs(os.path.join(OUT_DIR, "images", "optical" + v["tag"]), exist_ok=True)
        os.makedirs(os.path.join(OUT_DIR, "labels", "optical" + v["tag"]), exist_ok=True)
    deps = bpy.context.evaluated_depsgraph_get()
    n_frames = len(manifest["frames"])
    total = n_frames * len(VIEWS)

    done = 0
    for fi, fr in enumerate(manifest["frames"]):
        fid = int(fr.get("frame", fi))
        # per-frame weather -> sun strength, sky turbidity, ambient level
        w = fr.get("weather") or {}
        if w:
            sf = float(w.get("solar_factor", 1.0))
            if env["sun_obj"] is not None:
                env["sun_obj"].data.energy = env["sun_energy"] * max(sf, 0.0)
            cloud = float(w.get("cloud_cover", 0.1))
            if env.get("grad_bg") is not None:
                # overcast flattens and dims the gradient instead of tinting it
                env["grad_bg"].inputs["Strength"].default_value = (
                    env["grad_base"] * (1.0 - 0.40 * min(max(cloud, 0.0), 1.0)))
            else:
                env["bg"].inputs[1].default_value = env["base_bg"] * (0.55 + 1.2 * cloud)
            if env.get("sky") is not None:
                try:
                    env["sky"].turbidity = float(w.get("turbidity", env["base_turb"]))
                except Exception:
                    pass
            # With auto exposure the probe already accounts for the sky state,
            # so the per-frame correction only tracks the sun, not the cloud.
            wet = 0.25 * min(1.0, float(w.get("rain_mmh", 0.0)) / 16.0)
        for name, pose in fr["objects"].items():
            if name in meshes:
                ob = meshes[name]["ob"]
                ob.location = Vector(tuple(float(x) for x in pose["pos"]))
                ob.rotation_euler = (0.0, 0.0, math.radians(float(pose["yaw"])))
        bpy.context.view_layer.update()

        for v in VIEWS:
            # per-frame camera pose (seeker / gimbal): the physics passes track,
            # so the optical pass must track with exactly the same pose
            pose = pose_for(fr, v)
            if pose:
                aim_camera(sc, cam, pose, v, CAM_KEY)
            else:
                sc.render.resolution_x = v["width"]
                sc.render.resolution_y = v["height"]
                cam.data.angle = math.radians(v["fov_deg"])
                bpy.context.view_layer.update()
            # per-view exposure (see exposures[] above), then the per-frame
            # weather trim: overcast darkens slightly, rain a touch more
            if w:
                cl = float(w.get("cloud_cover", 0.1))
                rain = 0.25 * min(1.0, float(w.get("rain_mmh", 0.0)) / 16.0)
                sc.view_settings.exposure = (
                    exposures.get(v["name"], base_exp)
                    - (0.35 if opt.get("auto_exposure", True) else 0.6) * cl - rain)
            W, H = int(v["width"]), int(v["height"])
            path = os.path.join(OUT_DIR, "images", "optical" + v["tag"], "%06d.png" % fid)
            from blender.sensor_helpers import configure_exposure
            configure_exposure(sc, cam, meshes, fr, v,
                               float(cfg.get("frames", {}).get("dt", 0.2)),
                               bool(opt.get("motion_blur", False)))
            if pass_output is not None:
                from blender.sensor_helpers import pass_paths
                pass_paths(pass_output, "%06d%s" % (fid, v["tag"]))
            sc.render.filepath = path
            try:
                bpy.ops.render.render(write_still=True)
            except RuntimeError as exc:
                # OpenImageDenoise allocates its own buffers and dies with "out of
                # memory" when several Blender instances share a busy machine. A
                # denoise-free frame is slightly grainier; a missing frame is a hole
                # in the dataset, so fall back instead of aborting the chunk.
                if "denois" not in str(exc).lower() and "memory" not in str(exc).lower():
                    raise
                print("WARN denoiser failed for frame %d, re-rendering without it" % fid,
                      flush=True)
                sc.cycles.use_denoising = False
                bpy.ops.render.render(write_still=True)
            if not os.path.exists(path):
                raise RuntimeError("render did not produce %s (frame_path=%s)"
                                   % (path, sc.render.frame_path()))

            measured_ids, sensor_metadata = None, None
            if pass_output is not None:
                from blender.sensor_helpers import read_passes
                raw = read_passes(pass_output, "%06d%s" % (fid, v["tag"]), W, H)
                measured_ids = np.rint(raw["instance"]).astype(np.int32)-1
                depth = raw["depth"]
                valid_depth = np.isfinite(depth) & (depth > 0) & (depth < 1e9)
                depth = np.where(valid_depth, depth, np.inf).astype(np.float32)
                raw_data = dict(linear_scene=raw["linear"], depth_z_m=depth,
                                instance_id=measured_ids, valid_depth=valid_depth)
                if (opt.get("sensor") or {}).get("enabled"):
                    from simkit.optical_sensor import forward
                    from simkit import png as png_writer
                    sensor_cfg = dict(cfg, weather=w)
                    sensor = forward(raw["linear"], depth, pose, sensor_cfg,
                                     int(cfg.get("seed", 0))*100003+fid,
                                     exposure_stops=sc.view_settings.exposure)
                    png_writer.write_rgb8(path, sensor["image_u8"])
                    raw_data.update(counts_u16=sensor["counts_u16"],
                                    linear_received=sensor["linear_received"],
                                    transmittance=sensor["transmittance"])
                    sensor_metadata = sensor["metadata"]
                raw_dir = os.path.join(OUT_DIR, "raw", "optical"+v["tag"])
                os.makedirs(raw_dir, exist_ok=True)
                np.savez_compressed(os.path.join(raw_dir, "%06d.npz" % fid), **raw_data)

            boxes = {}
            for name, info in meshes.items():
                # scenery (buildings, barriers, decoys) is drawn but never
                # labelled: a label file that boxes the whole city teaches a
                # detector that everything is a target
                if not info.get("label", True):
                    continue
                ob = info["ob"]
                corners = [ob.matrix_world @ Vector(tuple(c)) for c in
                           np.array([[x, y, z] for x in (info["lo"][0], info["hi"][0])
                                     for y in (info["lo"][1], info["hi"][1])
                                     for z in (info["lo"][2], info["hi"][2])])]
                pts = [world_to_camera_view(sc, cam, c) for c in corners]
                if all(p.z <= 0.05 for p in pts):
                    continue
                xs = [p.x * W for p in pts]
                ys = [(1.0 - p.y) * H for p in pts]
                x0, x1 = max(min(xs), 0.0), min(max(xs), float(W))
                y0, y1 = max(min(ys), 0.0), min(max(ys), float(H))
                if x1 <= x0 or y1 <= y0:
                    continue
                # Sub-pixel targets: a 3 px box rounded to 0 width is a lost
                # label, so keep at least one pixel of extent in each axis.
                if x1 - x0 < 1.0:
                    cx = 0.5 * (x0 + x1)
                    x0, x1 = max(0.0, cx - 0.5), min(float(W), cx + 0.5)
                if y1 - y0 < 1.0:
                    cy = 0.5 * (y0 + y1)
                    y0, y1 = max(0.0, cy - 0.5), min(float(H), cy + 0.5)
                # coarse occlusion test through the object centre
                center = ob.matrix_world @ Vector((0.0, 0.0, float(info["size"][2]) * 0.5))
                d = center - cam.location
                hit, loc, nrm, idx, hob, mat = sc.ray_cast(deps, cam.location,
                                                           d.normalized(), distance=d.length)
                visible = (not hit) or (hob is not None and hob.name == name)
                raster_pixels = None
                if measured_ids is not None:
                    raster_pixels = int(np.count_nonzero(measured_ids == ob.pass_index-1))
                    visible = raster_pixels > 0
                boxes[name] = dict(x0=round(x0, 1), y0=round(y0, 1),
                                   x1=round(x1, 1), y1=round(y1, 1),
                                   visible=bool(visible), source="corners",
                                   pixels=raster_pixels,
                                   visibility_source="rendered_instance_pass" if measured_ids is not None else "center_ray")
            json.dump(dict(frame=fid, image="%06d.png" % fid, width=W, height=H,
                           view=v["name"], fov_deg=float(v["fov_deg"]),
                           boxes=boxes, weather=dict(w or {}),
                           sensor_model=sensor_metadata,
                           motion_blur_simulated=bool(sc.render.use_motion_blur),
                           material_reports={name: info["material_report"] for name, info in meshes.items()},
                           geometry_passes=(dict(depth="camera_forward_z_m", instance="non-antialiased reference geometry")
                                            if measured_ids is not None else None)),
                      open(os.path.join(OUT_DIR, "labels", "optical" + v["tag"],
                                        "%06d.json" % fid), "w", encoding="utf-8"), indent=1)
            done += 1
            log("frame %d/%d view %s done (%d/%d)"
                % (fi + 1, n_frames, v["name"], done, total))


main()
