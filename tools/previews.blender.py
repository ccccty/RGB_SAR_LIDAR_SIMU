"""Runs INSIDE Blender: render one preview PNG per model in the asset library.

  blender -b --factory-startup -P tools/previews.blender.py -- catalog.json out_dir

Every OBJ gets a three-quarter view with a transparent film, so the host-side
compositor can drop it onto a white card without halos. Workbench is used on
purpose: it is the only engine that renders reliably on a headless machine
without a GPU, and for a solid-shaded asset preview it is also the fastest
(roughly a second a model against minutes for Cycles).

The models are already normalised by tools/prepare_vehicles.blender.py: Z up,
base on z=0, +X along the body. So the camera only has to look at the bbox
centre from a fixed direction, and a ship and a quadcopter frame identically.
"""
import json
import math
import os
import sys

import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
CATALOG, OUT_DIR = argv[0], argv[1]
CATALOG = os.path.abspath(CATALOG)
OUT_DIR = os.path.abspath(OUT_DIR)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(OUT_DIR, exist_ok=True)

RES_X, RES_Y = 512, 384
FOV_DEG = 34.0
# Three-quarter view: to the front-right and above, the angle every product
# shot of a vehicle uses because it shows length, width and height at once.
VIEW_DIR = Vector((0.86, -0.94, 0.52)).normalized()


def log(msg):
    print("[preview] " + str(msg), flush=True)


def setup_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_WORKBENCH"
    sc.render.resolution_x = RES_X
    sc.render.resolution_y = RES_Y
    sc.render.resolution_percentage = 100
    sc.render.film_transparent = True
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGBA"

    sh = sc.display.shading
    sh.light = "STUDIO"
    sh.color_type = "SINGLE"
    sh.single_color = (0.62, 0.66, 0.72)
    sh.show_cavity = True
    sh.cavity_type = "BOTH"
    sh.curvature_ridge_factor = 0.9
    sh.curvature_valley_factor = 0.9
    sh.show_shadows = False
    sc.display.render_aa = "8"
    return sc


def make_camera(sc):
    cam_data = bpy.data.cameras.new("preview_cam")
    cam_data.angle = math.radians(FOV_DEG)
    cam_data.sensor_fit = "HORIZONTAL"
    cam = bpy.data.objects.new("preview_cam", cam_data)
    sc.collection.objects.link(cam)
    sc.camera = cam
    return cam


def frame_camera(cam, lo, hi):
    """Place the camera so the bbox corners just fill the frame.

    Fitting a bounding sphere is much easier but wastes the frame badly for
    long thin subjects: an airliner ends up occupying a fifth of the width.
    Solving the perspective constraint per corner instead is exact --
    for every corner, depth = d + dot(p - centre, forward) and the corner
    stays inside the frustum when |x| <= tan(h) * depth. Rearranged, that is a
    lower bound on the camera distance; the largest bound over all corners is
    the tight fit.
    """
    centre = (lo + hi) / 2.0
    forward = -VIEW_DIR
    world_up = Vector((0.0, 0.0, 1.0))
    right = forward.cross(world_up).normalized()
    up = right.cross(forward).normalized()

    tan_h = math.tan(math.radians(FOV_DEG) / 2.0)
    tan_v = tan_h * RES_Y / RES_X
    margin = 1.10                      # 8% clear space on the binding axis

    corners = [Vector((x, y, z)) for x in (lo.x, hi.x)
               for y in (lo.y, hi.y) for z in (lo.z, hi.z)]
    dist = 0.0
    for p in corners:
        rel = p - centre
        depth_term = rel.dot(forward)
        x = rel.dot(right)
        y = rel.dot(up)
        dist = max(dist,
                   abs(x) * margin / tan_h - depth_term,
                   abs(y) * margin / tan_v - depth_term)
    dist = max(dist, (hi - lo).length * 0.05)

    cam.location = centre + VIEW_DIR * dist
    cam.rotation_euler = (-VIEW_DIR).to_track_quat("-Z", "Y").to_euler()
    cam.data.clip_start = max(dist * 0.01, 0.01)
    cam.data.clip_end = dist * 12.0 + (hi - lo).length * 4.0


def import_obj(path):
    before = set(bpy.context.scene.objects)
    bpy.ops.wm.obj_import(filepath=path, forward_axis="Y", up_axis="Z")
    new = [o for o in bpy.context.scene.objects if o not in before and o.type == "MESH"]
    if not new:
        return None
    return new


def bbox_of(objs):
    pts = [o.matrix_world @ Vector(c) for o in objs for c in o.bound_box]
    lo = Vector([min(p[k] for p in pts) for k in range(3)])
    hi = Vector([max(p[k] for p in pts) for k in range(3)])
    return lo, hi


def main():
    with open(CATALOG, encoding="utf-8") as fh:
        records = json.load(fh)

    done, failed = [], []
    for rec in records:
        src = os.path.join(ROOT, rec["file"])
        name = rec["name_cn"]
        out = os.path.join(OUT_DIR, name + ".png")
        if not os.path.exists(src):
            failed.append((name, "missing obj"))
            continue
        try:
            sc = setup_scene()
            cam = make_camera(sc)
            objs = import_obj(src)
            if not objs:
                failed.append((name, "no mesh"))
                continue
            for o in objs:
                o.select_set(True)
            lo, hi = bbox_of(objs)
            frame_camera(cam, lo, hi)
            sc.render.filepath = out
            bpy.ops.render.render(write_still=True)
            done.append((name, os.path.getsize(out)))
        except Exception as exc:
            failed.append((name, str(exc)[:90]))

    log("rendered %d, failed %d" % (len(done), len(failed)))
    for n, why in failed[:12]:
        log("  FAILED %s: %s" % (n, why))
    with open(os.path.join(OUT_DIR, "_report.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(ok=len(done), failed=failed), fh, ensure_ascii=False, indent=1)


main()