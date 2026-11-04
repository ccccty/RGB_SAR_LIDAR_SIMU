"""Cycles passes and motion exposure for the shared geometry (Blender only)."""
import os
from pathlib import Path

import bpy
import numpy as np
from mathutils import Quaternion, Vector


def setup_passes(scene, directory):
    scene.use_nodes = True
    nodes = scene.node_tree.nodes
    nodes.clear()
    layer = nodes.new("CompositorNodeRLayers")
    composite = nodes.new("CompositorNodeComposite")
    scene.node_tree.links.new(layer.outputs["Image"], composite.inputs["Image"])
    scene.view_layers[0].use_pass_z = True
    scene.view_layers[0].use_pass_object_index = True
    scene.view_layers[0].update_render_passes()
    output = nodes.new("CompositorNodeOutputFile")
    output.base_path = str(directory)
    output.format.file_format = "OPEN_EXR"
    output.format.color_depth = "32"
    output.format.color_mode = "RGBA"
    for slot in list(output.file_slots):
        output.file_slots.remove(output.inputs[0])
    for name, socket in (("linear", "Image"), ("depth", "Depth"), ("instance", "IndexOB")):
        output.file_slots.new(name)
        scene.node_tree.links.new(layer.outputs[socket], output.inputs[name])
    return output


def pass_paths(output, stem):
    # Render frame remains controlled by the motion model. Capture the actual
    # file names after rendering instead of assuming a fixed frame suffix.
    for name, slot in zip(("linear", "depth", "instance"), output.file_slots):
        slot.path = stem+"_"+name+"_"


def read_passes(output, stem, width, height):
    result = {}
    for name in ("linear", "depth", "instance"):
        paths = list(Path(output.base_path).glob(stem+"_"+name+"_*.exr"))
        if len(paths) != 1:
            raise RuntimeError("expected one EXR for %s: %s" % (name, paths))
        path = paths[0]
        img = bpy.data.images.load(str(path), check_existing=False)
        img.colorspace_settings.name = "Non-Color"
        pixels = np.empty(width*height*4, np.float32)
        img.pixels.foreach_get(pixels)
        arr = pixels.reshape(height, width, 4)[::-1].copy()
        result[name] = arr[:, :, :3] if name == "linear" else arr[:, :, 0]
        bpy.data.images.remove(img)
        path.unlink()
    return result


def set_pose(ob, pose):
    ob.location = Vector(pose["pos"])
    q = (Vector(pose["look_at"])-ob.location).to_track_quat("-Z", "Y")
    q = q @ Quaternion((0, 0, 1), np.radians(pose.get("roll_deg", 0.0)))
    ob.rotation_mode = "QUATERNION"
    ob.rotation_quaternion = q


def configure_exposure(scene, camera, meshes, frame, view, dt, enabled):
    samples = frame.get("shutter_samples") or []
    camera.animation_data_clear()
    for info in meshes.values():
        info["ob"].animation_data_clear()
    scene.render.use_motion_blur = bool(enabled and len(samples) == 2)
    if not scene.render.use_motion_blur:
        return
    duration = samples[1]["offset_s"]-samples[0]["offset_s"]
    if dt <= 0 or duration <= 0 or duration > dt:
        raise ValueError("motion exposure must be positive and <= frame interval")
    scene.render.motion_blur_shutter = duration/dt
    if hasattr(scene.render, "motion_blur_position"):
        scene.render.motion_blur_position = "CENTER"
    reference = int(frame["frame"])+1
    for sample in samples:
        key = reference+sample["offset_s"]/dt
        set_pose(camera, sample["views"][view["name"]])
        camera.keyframe_insert("location", frame=key)
        camera.keyframe_insert("rotation_quaternion", frame=key)
        for name, pose in sample["poses"].items():
            if name not in meshes:
                continue
            ob = meshes[name]["ob"]
            ob.location = Vector(pose["pos"])
            ob.rotation_euler = (np.radians(pose.get("roll", 0)),
                                 -np.radians(pose.get("pitch", 0)), np.radians(pose["yaw"]))
            ob.keyframe_insert("location", frame=key)
            ob.keyframe_insert("rotation_euler", frame=key)
    for ob in [camera]+[info["ob"] for info in meshes.values()]:
        if ob.animation_data and ob.animation_data.action:
            # Blender 4.5 supports legacy fcurves through the action API.
            for curve in getattr(ob.animation_data.action, "fcurves", []):
                for key in curve.keyframe_points:
                    key.interpolation = "LINEAR"
    scene.frame_set(reference)
