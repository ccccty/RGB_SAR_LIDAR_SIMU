"""Convert .glb/.gltf meshes to .obj via headless Blender (also reports bbox)."""
import os
import sys

import bpy

argv = sys.argv[sys.argv.index("--") + 1:]
jobs = []
for i in range(0, len(argv) - 1, 2):
    jobs.append((argv[i], argv[i + 1]))

for src, dst in jobs:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=src)
    objs = [o for o in bpy.data.objects if o.type == "MESH"]
    if not objs:
        print("[glb2obj] no mesh in", src)
        continue
    # bake transforms, then report a world-space bbox
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    mins = [min(min((o.matrix_world @ v.co)[k] for v in o.data.vertices) for o in objs)
            for k in range(3)]
    maxs = [max(max((o.matrix_world @ v.co)[k] for v in o.data.vertices) for o in objs)
            for k in range(3)]
    dims = [maxs[k] - mins[k] for k in range(3)]
    tri = sum(len(o.data.polygons) for o in objs)
    bpy.ops.wm.obj_export(filepath=dst, export_selected_objects=True,
                          forward_axis="Y", up_axis="Z", apply_modifiers=True)
    print("[glb2obj] %s -> %s  tris=%d  dims=%.3f x %.3f x %.3f m"
          % (os.path.basename(src), os.path.basename(dst), tri, dims[0], dims[1], dims[2]))
