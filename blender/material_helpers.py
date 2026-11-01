"""Apply source appearance to the SAME prepared triangle mesh used by physics.

This module runs inside Blender. It does not infer emissivity/RCS from colours.
Original glTF PBR materials may be reused without importing a second geometry
into the rendered scene; all imported objects are removed immediately.
"""
import math
import os
import re

import bpy
import numpy as np

_SOURCE_CACHE = {}


def _source_materials(path):
    path = os.path.abspath(path)
    if path in _SOURCE_CACHE:
        try:
            if all(mat.name in bpy.data.materials for mat in _SOURCE_CACHE[path].values()):
                return _SOURCE_CACHE[path]
        except ReferenceError:
            pass  # A factory reset invalidated datablocks between renders.
        del _SOURCE_CACHE[path]
    if not os.path.isfile(path):
        raise FileNotFoundError("optical_material_source not found: " + path)
    if os.path.splitext(path)[1].lower() not in (".glb", ".gltf"):
        raise ValueError("optical_material_source currently supports GLB/glTF")
    before = set(bpy.data.objects)
    before_meshes = set(bpy.data.meshes)
    selected = list(bpy.context.selected_objects)
    active = bpy.context.view_layer.objects.active
    try:
        bpy.ops.import_scene.gltf(filepath=path)
        imported = set(bpy.data.objects)-before
        materials = {}
        for ob in imported:
            if ob.type != "MESH":
                continue
            for mat in ob.data.materials:
                if mat is None:
                    continue
                # Blender appends .001 on name collisions; glTF source material
                # names survive as the base name and match OBJ's usemtl names.
                name = re.sub(r"\.\d{3}$", "", mat.name)
                materials.setdefault(mat.name, mat)
                materials.setdefault(name, mat)
        if not materials:
            raise ValueError("source GLB/glTF has no material slots")
        _SOURCE_CACHE[path] = materials
        return materials
    finally:
        # Keep material/image datablocks, remove every imported object (including
        # cameras/lights/parents) so source geometry can never render twice.
        for ob in list(set(bpy.data.objects)-before):
            bpy.data.objects.remove(ob, do_unlink=True)
        for data in list(set(bpy.data.meshes)-before_meshes):
            if data.users == 0:
                bpy.data.meshes.remove(data)
        for ob in selected:
            if ob.name in bpy.context.view_layer.objects:
                ob.select_set(True)
        if active is not None and active.name in bpy.context.view_layer.objects:
            bpy.context.view_layer.objects.active = active


def _image_texture(tree, path, non_color=False):
    node = tree.nodes.new("ShaderNodeTexImage")
    image = bpy.data.images.load(path, check_existing=True)
    if non_color:
        image.colorspace_settings.name = "Non-Color"
    node.image = image
    return node


def _mtl_material(name, data):
    mat = bpy.data.materials.new("source_"+name)
    mat.use_nodes = True
    tree = mat.node_tree
    bsdf = tree.nodes.get("Principled BSDF")
    kd = data.get("Kd", [0.8, 0.8, 0.8])
    bsdf.inputs["Base Color"].default_value = (*kd, 1.0)
    roughness = data.get("Pr", math.sqrt(2.0/(max(data.get("Ns", 32.0), 0.0)+2.0)))
    bsdf.inputs["Roughness"].default_value = min(max(float(roughness), 0.0), 1.0)
    bsdf.inputs["Metallic"].default_value = min(max(float(data.get("Pm", 0.0)), 0.0), 1.0)
    # MTL d is dissolve/opacity, not glass transmission. Preserve opacity as
    # metadata without inventing a transmissive optical material.
    mat["source_opacity"] = float(data.get("d", 1.0-data.get("Tr", 0.0)))
    missing = []
    for key, socket in [("map_Kd", "Base Color"), ("map_Pr", "Roughness"), ("map_Pm", "Metallic")]:
        path = data.get(key)
        if not path:
            continue
        if not os.path.isfile(path):
            missing.append(path)
            continue
        texture = _image_texture(tree, path, non_color=(key != "map_Kd"))
        tree.links.new(texture.outputs["Color"], bsdf.inputs[socket])
    normal_path = data.get("norm")
    if normal_path:
        if os.path.isfile(normal_path):
            texture = _image_texture(tree, normal_path, non_color=True)
            normal = tree.nodes.new("ShaderNodeNormalMap")
            tree.links.new(texture.outputs["Color"], normal.inputs["Color"])
            tree.links.new(normal.outputs["Normal"], bsdf.inputs["Normal"])
        else:
            missing.append(normal_path)
    bump_path = data.get("map_Bump", data.get("bump"))
    if bump_path and not normal_path:
        if os.path.isfile(bump_path):
            texture = _image_texture(tree, bump_path, non_color=True)
            bump = tree.nodes.new("ShaderNodeBump")
            tree.links.new(texture.outputs["Color"], bump.inputs["Height"])
            tree.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
        else:
            missing.append(bump_path)
    return mat, missing


def apply_source_materials(ob, mesh, spec, base_dir="."):
    """Return a JSON-safe appearance report; legacy single material is opt-in safe.

    Call immediately AFTER add_mesh_object(name, prepared_mesh, ...) so triangle
    order equals mesh.f. Use spec.import_materials=true to enable OBJ/MTL slots,
    optionally spec.optical_material_source='...glb' for original PBR nodes.
    """
    report = dict(enabled=bool(spec.get("import_materials", False)), source="object_preset",
                  material_slots=[], missing_materials=[], missing_textures=[], missing_uv_faces=0)
    if not report["enabled"]:
        return report
    if len(ob.data.polygons) != len(mesh.f):
        raise ValueError("prepared mesh polygon order/count changed before source-material assignment")
    original = spec.get("optical_material_source")
    if original:
        if not os.path.isabs(original):
            original = os.path.join(base_dir, original)
        source = _source_materials(original)
        report["source"] = os.path.abspath(original)
    else:
        source = {}
        for name, data in mesh.source_materials.items():
            source[name], missing = _mtl_material(name, data)
            report["missing_textures"].extend(missing)
        report["source"] = "OBJ/MTL"
    # Slot zero is the existing explicitly declared fallback material.
    if not ob.data.materials:
        raise ValueError("add_mesh_object must provide a fallback material")
    slots = {}
    names = sorted(set(str(n) for n in mesh.face_source_material if n))
    for name in names:
        material = source.get(name)
        if material is None:
            report["missing_materials"].append(name)
            continue
        slots[name] = len(ob.data.materials)
        ob.data.materials.append(material)
    valid_uv = np.isfinite(mesh.face_uv).all(axis=(1, 2))
    report["missing_uv_faces"] = int(np.count_nonzero(~valid_uv))
    if valid_uv.any():
        layer = ob.data.uv_layers.new(name="UVMap")
        uv = np.nan_to_num(mesh.face_uv, nan=0.0).reshape(-1, 2)
        layer.data.foreach_set("uv", uv.ravel())
        ob.data.uv_layers.active = layer
    for i, polygon in enumerate(ob.data.polygons):
        name = str(mesh.face_source_material[i])
        slot = slots.get(name, 0)
        if slot and not valid_uv[i]:
            material = ob.data.materials[slot]
            textured = material.use_nodes and any(n.type == "TEX_IMAGE" for n in material.node_tree.nodes)
            if textured:
                slot = 0  # no valid UV: explicit fallback, not an arbitrary texture corner
        polygon.material_index = slot
    report["material_slots"] = sorted(slots)
    report["faces_with_source_material"] = sum(p.material_index != 0 for p in ob.data.polygons)
    return report
