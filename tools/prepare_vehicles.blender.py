"""Blender stage 2 for the vehicle asset library: GLB -> normalized OBJ.

Run: third_party/blender/blender.exe -b --factory-startup \
       -P tools/prepare_vehicles.blender.py -- assets/_cache/vehicles/manifest.json

Reads the manifest written by tools/fetch_vehicle_models.py, then for every
entry: clears the imported glTF parent empties, scales the body axis to the
declared real length, aligns it to +X, decimates to a triangle budget, drops
the model on z=0 centred in XY, and exports OBJ into assets/models/<category>/.

Outputs (next to the manifest):
  convert_report.json  per-model triangles, dimensions and ground offset
  catalog.json         the library catalogue written into assets/models/
"""
import json
import os
import sys

import bpy
from mathutils import Matrix, Vector

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(ROOT, "assets", "models")

CATEGORY_EN = {
    "旋翼无人机": "Rotary-wing UAV",
    "固定翼无人机": "Fixed-wing UAV",
    "直升机": "Helicopter",
    "民航飞机": "Civil aircraft",
    "军用飞机": "Military aircraft",
    "地面车辆": "Ground vehicle",
    "军用地面车辆": "Military ground vehicle",
    "军舰": "Naval vessel",
    "民用船舶": "Civil vessel",
}
LICENSE_URL = {"CC0": "https://creativecommons.org/publicdomain/zero/1.0/",
               "CC-BY": "https://creativecommons.org/licenses/by/4.0/"}


def select_only(objs, active=None):
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = active or objs[0]


def bbox(objs):
    pts = [o.matrix_world @ v.co for o in objs for v in o.data.vertices]
    lo = Vector([min(p[k] for p in pts) for k in range(3)])
    hi = Vector([max(p[k] for p in pts) for k in range(3)])
    return lo, hi


def main():
    argv = sys.argv
    manifest_path = argv[argv.index("--") + 1]
    with open(manifest_path, encoding="utf-8") as fh:
        entries = json.load(fh)
    meta_path = os.path.join(os.path.dirname(manifest_path), "meta.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as fh:
            meta = {m["id"]: m for m in json.load(fh)}

    reports, records = [], []
    for item in entries:
        src = item["glb"]
        mid = os.path.basename(src).replace(".glb", "")
        info = meta.get(mid, {})
        cate = info.get("category", "未分类")
        name_cn = info.get("name_cn", mid)

        bpy.ops.wm.read_factory_settings(use_empty=True)
        try:
            bpy.ops.import_scene.gltf(filepath=src)
        except Exception as exc:
            reports.append(dict(id=mid, ok=False, why="import: %s" % exc))
            continue
        meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
        if not meshes:
            reports.append(dict(id=mid, ok=False, why="no mesh"))
            continue

        # glTF nests meshes under rotated empties; detach before transforming.
        select_only(meshes)
        bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

        in_tris = sum(len(o.data.polygons) for o in meshes)
        lo, hi = bbox(meshes)
        dims = hi - lo
        if min(dims) <= 0:
            reports.append(dict(id=mid, ok=False, why="degenerate"))
            continue

        # Body axis -> +X. Aircraft use the fuselage (minor horizontal extent),
        # ground vehicles and ships use the long axis.
        axis = item.get("axis", "auto")
        if axis in ("major", "minor"):
            big = "y" if dims.y > dims.x else "x"
            axis = big if axis == "major" else ("x" if big == "y" else "y")
        elif axis == "auto":
            axis = "y" if dims.y > dims.x else "x"
        length = dims.y if axis == "y" else dims.x
        if length <= 0:
            reports.append(dict(id=mid, ok=False, why="flat"))
            continue

        scale = float(item["length_m"]) / length
        rot = Matrix.Rotation(1.5707963267948966, 4, "Z") if axis == "y" else Matrix.Identity(4)
        xf = Matrix.Scale(scale, 4) @ rot
        for o in meshes:
            o.matrix_world = xf @ o.matrix_world
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

        # One mesh makes the triangle budget exact.
        select_only(meshes, active=meshes[0])
        if len(meshes) > 1:
            bpy.ops.object.join()
        obj = bpy.context.view_layer.objects.active

        target = int(item.get("tri_target", 3500))
        polys = len(obj.data.polygons)
        if polys > target:
            mod = obj.modifiers.new("dec", "DECIMATE")
            mod.ratio = max(0.02, target / float(polys))
            select_only([obj], active=obj)
            bpy.ops.object.modifier_apply(modifier=mod.name)

        lo, hi = bbox([obj])
        shift = Vector([-((lo.x + hi.x) / 2.0), -((lo.y + hi.y) / 2.0), -lo.z])
        obj.matrix_world = Matrix.Translation(shift) @ obj.matrix_world
        select_only([obj], active=obj)
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

        out_dir = os.path.join(DEST, cate)
        os.makedirs(out_dir, exist_ok=True)
        dst = os.path.join(out_dir, name_cn + ".obj")
        bpy.ops.wm.obj_export(filepath=dst, export_selected_objects=True,
                              forward_axis="Y", up_axis="Z", apply_modifiers=True,
                              export_triangulated_mesh=True, export_materials=False)

        lo, hi = bbox([obj])
        fin = [round(v, 3) for v in (hi - lo)]
        reports.append(dict(id=mid, ok=True, in_tris=in_tris,
                            out_tris=len(obj.data.polygons), dims_m=fin,
                            base_z=round(lo.z, 5), bytes=os.path.getsize(dst)))
        records.append(dict(
            id=mid, name_cn=name_cn, category=cate,
            category_en=CATEGORY_EN.get(cate, ""),
            file="assets/models/%s/%s.obj" % (cate, name_cn),
            source_title=info.get("source_title", ""),
            author=info.get("author", ""),
            license=info.get("license", ""),
            license_url=LICENSE_URL.get(info.get("license", ""), ""),
            source_page=info.get("source_page", ""),
            length_m=item["length_m"], dims_m=fin,
            tris=len(obj.data.polygons), bytes=os.path.getsize(dst)))
        print("[vehicles] %-12s %-14s %6d -> %5d tris %s" % (
            mid, name_cn, in_tris, len(obj.data.polygons), fin), flush=True)

    order = list(CATEGORY_EN)
    records.sort(key=lambda r: (order.index(r["category"]) if r["category"] in order else 99,
                                r["name_cn"]))
    with open(os.path.join(DEST, "catalog.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(records, fh, ensure_ascii=False, indent=1)
    with open(os.path.join(os.path.dirname(manifest_path), "convert_report.json"),
              "w", encoding="utf-8") as fh:
        json.dump(reports, fh, ensure_ascii=False, indent=1)
    bad = [r for r in reports if not r.get("ok")]
    print("[vehicles] %d ok, %d failed, catalog -> assets/models/catalog.json"
          % (len(reports) - len(bad), len(bad)))


main()