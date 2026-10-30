# -*- coding: utf-8 -*-
"""Runs INSIDE Blender: render the visible-band frames and export boxes.

  blender -b --factory-startup -P render_optical.py -- cfg.json out_dir manifest.json project_root
"""
import json
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
CFG_PATH, OUT_DIR, MANIFEST, PROJECT_ROOT = argv[0], argv[1], argv[2], argv[3]
OUT_DIR = os.path.abspath(OUT_DIR)
MANIFEST = os.path.abspath(MANIFEST)
CFG_PATH = os.path.abspath(CFG_PATH)
sys.path.insert(0, PROJECT_ROOT)

from simkit import scene as sk_scene  # noqa: E402
from simkit.mesh import Mesh  # noqa: E402

cfg = sk_scene.load_config(CFG_PATH)
opt = cfg.get("optical", {})
CAM_KEY = "optical"          # the sensor this renderer draws
cam_cfg = cfg["sensors"][CAM_KEY]
W = int(cam_cfg.get("width", 960))
H = int(cam_cfg.get("height", 720))
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
    sc.render.resolution_x = W
    sc.render.resolution_y = H
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
        add_mesh_object("ground", gm, g.get("mat", "asphalt"), (0, 0, 0), 0.0)

    # objects
    meshes = {}
    for spec in cfg.get("objects", []):
        # prepare_mesh applies decimate/scale/height exactly as the numpy world
        # builder does, and returns the recentred bbox used for projections.
        m, lo, hi = sk_scene.prepare_mesh(spec, cfg.get("_base_dir", "."))
        sizes = hi - lo
        ob = add_mesh_object(spec.get("name", "obj"), m, spec.get("mat", "painted_metal"),
                             spec.get("pos", (0, 0, 0)), spec.get("rot", 0.0))
        meshes[spec.get("name", "obj")] = dict(ob=ob, size=sizes, lo=lo, hi=hi)

    # camera
    cd = bpy.data.cameras.new("cam")
    cd.lens_unit = "FOV"
    cd.sensor_fit = "HORIZONTAL"
    cd.angle = math.radians(float(cam_cfg.get("fov_deg", 45.0)))
    # Blender's default clip range is 0.1..100 m, which silently clips any
    # target farther away than 100 m. Size the frustum to the actual scene.
    reach = 1.0
    for _n, _m in meshes.items():
        for _c in (_m["lo"], _m["hi"]):
            reach = max(reach, (Vector(tuple(float(x) for x in _c))
                                - Vector(tuple(float(x) for x in cam_cfg["pos"]))).length)
    cd.clip_start = max(0.01, min(1.0, reach * 1e-4))
    cd.clip_end = reach * 4.0 + 100.0
    cam = bpy.data.objects.new("cam", cd)
    pos = Vector(tuple(float(x) for x in cam_cfg["pos"]))
    cam.location = pos
    look = Vector(tuple(float(x) for x in cam_cfg["look_at"]))
    cam.rotation_euler = (look - pos).to_track_quat("-Z", "Y").to_euler()
    sc.collection.objects.link(cam)
    sc.camera = cam
    env = dict(sky=sky, bg=bg, sun_obj=sun_obj, sun_energy=sun_energy,
               base_bg=float(bg.inputs[1].default_value),
               base_turb=float(getattr(sky, "turbidity", 2.5)))
    return sc, cam, meshes, env


def main():
    from bpy_extras.object_utils import world_to_camera_view
    sc, cam, meshes, env = build_scene()
    manifest = json.load(open(MANIFEST, "r", encoding="utf-8"))
    os.makedirs(os.path.join(OUT_DIR, "images", "optical"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "labels", "optical"), exist_ok=True)
    deps = bpy.context.evaluated_depsgraph_get()

    for fi, fr in enumerate(manifest["frames"]):
        fid = int(fr.get("frame", fi))
        # per-frame weather -> sun strength, sky turbidity, ambient level
        w = fr.get("weather") or {}
        if w:
            sf = float(w.get("solar_factor", 1.0))
            if env["sun_obj"] is not None:
                env["sun_obj"].data.energy = env["sun_energy"] * max(sf, 0.0)
            cloud = float(w.get("cloud_cover", 0.1))
            env["bg"].inputs[1].default_value = env["base_bg"] * (0.55 + 1.2 * cloud)
            try:
                env["sky"].turbidity = float(w.get("turbidity", env["base_turb"]))
            except Exception:
                pass
            sc.view_settings.exposure = float(opt.get("exposure", -2.5)) + (
                -0.6 * cloud - 0.25 * min(1.0, float(w.get("rain_mmh", 0.0)) / 16.0))
        # per-frame camera pose (seeker / gimbal): the physics passes track, so
        # the optical pass must track with exactly the same pose
        camf = (fr.get("camera") or {})
        cpose = camf.get(CAM_KEY) or (list(camf.values())[0] if camf else None)
        if cpose:
            cpos = Vector(tuple(float(x) for x in cpose["pos"]))
            cam.location = cpos
            look = Vector(tuple(float(x) for x in cpose.get("look_at", [0, 0, 0])))
            if (look - cpos).length > 1e-6:
                cam.rotation_euler = (look - cpos).to_track_quat("-Z", "Y").to_euler()
            fv = cpose.get("fov_deg")
            if fv:
                cam.data.angle = math.radians(float(fv))
        for name, pose in fr["objects"].items():
            if name in meshes:
                ob = meshes[name]["ob"]
                ob.location = Vector(tuple(float(x) for x in pose["pos"]))
                ob.rotation_euler = (0.0, 0.0, math.radians(float(pose["yaw"])))
        bpy.context.view_layer.update()
        path = os.path.join(OUT_DIR, "images", "optical", "%06d.png" % fid)
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

        boxes = {}
        for name, info in meshes.items():
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
            # coarse occlusion test through the object centre
            center = ob.matrix_world @ Vector((0.0, 0.0, float(info["size"][2]) * 0.5))
            d = center - cam.location
            hit, loc, nrm, idx, hob, mat = sc.ray_cast(deps, cam.location, d.normalized(), distance=d.length)
            visible = (not hit) or (hob is not None and hob.name == name)
            boxes[name] = dict(x0=round(x0, 1), y0=round(y0, 1), x1=round(x1, 1), y1=round(y1, 1),
                               visible=bool(visible), source="corners")
        json.dump(dict(frame=fid, image="%06d.png" % fid, width=W, height=H, boxes=boxes,
                       weather=dict(w or {})),
                  open(os.path.join(OUT_DIR, "labels", "optical", "%06d.json" % fid), "w",
                       encoding="utf-8"), indent=1)
        log("frame %d/%d rendered" % (fi + 1, len(manifest["frames"])))


main()