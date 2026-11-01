"""Headless Blender integration check for downloaded NASA source PBR assets.

    blender -b --factory-startup -P tests/blender_material_probe.py

Requires the two downloaded/converted aviation assets. Geometry is not rendered;
checks prove UV/material linkage and source-object cleanup, not visual fidelity.
"""
import json
from pathlib import Path
import sys

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/"blender"))
from simkit import scene
from material_helpers import apply_source_materials

bpy.ops.wm.read_factory_settings(use_empty=True)
reports = []
for role in ("airliner", "uav"):
    spec = dict(shape="obj", model=str(ROOT/"assets/aviation"/(role+".obj")),
                mat="painted_metal", decimate=5000, import_materials=True,
                optical_material_source=str(ROOT/"assets/aviation"/(role+".glb")))
    mesh, _, _ = scene.prepare_mesh(spec, str(ROOT))
    data = bpy.data.meshes.new(role)
    data.from_pydata(mesh.v.tolist(), [], mesh.f.tolist())
    before_faces = np.array([p.vertices[:] for p in data.polygons], dtype=np.int32)
    data.validate()
    after_faces = np.array([p.vertices[:] for p in data.polygons], dtype=np.int32)
    np.testing.assert_array_equal(before_faces, after_faces,
                                  err_msg="Blender validation changed shared prepared geometry")
    data.update()
    ob = bpy.data.objects.new(role, data)
    bpy.context.scene.collection.objects.link(ob)
    data.materials.append(bpy.data.materials.new("fallback_"+role))
    before = set(bpy.data.objects)
    report = apply_source_materials(ob, mesh, spec, str(ROOT))
    assert set(bpy.data.objects) == before, "source objects leaked into the scene"
    assert report["faces_with_source_material"] == len(mesh.f), report
    assert not report["missing_materials"] and not report["missing_uv_faces"], report
    assert len(data.uv_layers.active.data) == 3*len(mesh.f)
    textures = [n for mat in data.materials if mat and mat.use_nodes
                for n in mat.node_tree.nodes if n.type == "TEX_IMAGE"]
    assert textures and all(n.image is not None and len(n.image.pixels)>0 for n in textures)
    # Cached imports may not add geometry on repeated calls either.
    assert apply_source_materials(ob, mesh, dict(spec, import_materials=False), str(ROOT))["enabled"] is False
    reports.append(dict(role=role, triangles=len(mesh.f), texture_nodes=len(textures), report=report))
    print("PASS source PBR", role, len(mesh.f), "triangles", len(textures), "texture nodes", flush=True)
out = ROOT/"out/_tmp/materials"
out.mkdir(parents=True, exist_ok=True)
(out/"probe_results.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
