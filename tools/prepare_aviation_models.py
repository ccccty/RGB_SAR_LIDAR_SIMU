"""Blender script: official GLBs -> shared, metre-scale OBJ geometry.

Run: third_party/blender/blender.exe -b --factory-startup -P tools/prepare_aviation_models.py
Preserves a material-bearing OBJ and the original GLB. The current simkit
rendering path still uses its coarse material preset; source textures are not
claimed to survive that rendering path.
"""
import json
import hashlib
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "assets" / "aviation"
reports = []
for role, span in [("airliner", 60.9), ("uav", 6.1)]:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(DEST / (role + ".glb")))
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    coords = [o.matrix_world @ v.co for o in meshes for v in o.data.vertices]
    lo = Vector([min(p[k] for p in coords) for k in range(3)])
    hi = Vector([max(p[k] for p in coords) for k in range(3)])
    dims = hi-lo
    print("[aviation]", role, "source dimensions", list(dims), "objects", len(meshes), flush=True)
    # NASA viewer assets have the fuselage on Y, wings across X. Rotate about
    # world Z so the flight-model forward axis is +X and wings are across Y.
    # Verified from plan-view silhouettes: NASA imports have the nose on -Y.
    # +90 degrees maps -Y to +X. The previous -90 made both models fly tail-first.
    rotation = Matrix.Rotation(1.5707963267948966, 4, "Z")
    scale = span / dims.x
    transform = Matrix.Scale(scale, 4) @ rotation
    for o in meshes:
        o.matrix_world = transform @ o.matrix_world
    bpy.ops.object.select_all(action="DESELECT")
    for o in meshes:
        o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    dst = DEST / (role + ".obj")
    bpy.ops.wm.obj_export(filepath=str(dst), export_selected_objects=True,
                          forward_axis="Y", up_axis="Z", apply_modifiers=True,
                          export_triangulated_mesh=True)
    coords = [o.matrix_world @ v.co for o in meshes for v in o.data.vertices]
    final = [max(p[k] for p in coords)-min(p[k] for p in coords) for k in range(3)]
    reports.append(dict(role=role, source_dimensions=dims[:], scale=scale,
                        dimensions_m=final, wingspan_m=span,
                        obj_sha256=hashlib.sha256(dst.read_bytes()).hexdigest(),
                        blender_version=bpy.app.version_string,
                        forward_axis="+X nose; verified plan-view silhouette",
                        normalization="declared research scale; verify variant and axis before production",
                        source_meshes=len(meshes),
                        materials=sorted({m.name for o in meshes for m in o.data.materials if m})))
    print("[aviation] exported", dst.name, final, flush=True)
(DEST / "geometry.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
