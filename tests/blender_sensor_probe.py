"""Blender integration test: Z-depth convention and actual exposure motion blur.
Run from project root with blender -b --factory-startup -P tests/blender_sensor_probe.py.
"""
import sys,os
sys.path.insert(0,os.getcwd())
import bpy,numpy as np
from mathutils import Vector
from blender.sensor_helpers import setup_passes,pass_paths,read_passes,configure_exposure
bpy.ops.wm.read_factory_settings(use_empty=True)
s=bpy.context.scene
s.render.engine='CYCLES'
s.cycles.samples=8
s.render.resolution_x=96
s.render.resolution_y=64
s.render.resolution_percentage=100
bpy.ops.mesh.primitive_plane_add(size=100,location=(0,0,0))
bpy.context.object.pass_index=1
bpy.ops.object.camera_add(location=(0,0,10))
c=bpy.context.object
c.rotation_euler=(0,0,0)
c.data.angle=1.2
s.camera=c
o=setup_passes(s,os.path.abspath('out/_tmp/optical_pass_probe'))
pass_paths(o,'plane')
bpy.ops.render.render()
a=read_passes(o,'plane',96,64)
print('PLANE_DEPTH',float(a['depth'].min()),float(a['depth'].max()),float(a['depth'][32,48]),float(a['depth'][32,10]))
assert np.max(np.abs(a['depth']-10))<0.01
# Motion: emissive cube translating laterally during 0.2s exposure.
bpy.data.objects.remove(bpy.data.objects.get('Plane'),do_unlink=True)
bpy.ops.mesh.primitive_cube_add(location=(0,0,0))
ob=bpy.context.object
m=bpy.data.materials.new('emission');m.use_nodes=True
nt=m.node_tree;nt.nodes.clear();em=nt.nodes.new('ShaderNodeEmission');out=nt.nodes.new('ShaderNodeOutputMaterial');nt.links.new(em.outputs[0],out.inputs[0]);ob.data.materials.append(m)
ob.pass_index=1
cam_pose={'pos':[0,0,10],'look_at':[0,0,0]}
fr={'frame':0,'shutter_samples':[{'offset_s':-0.1,'views':{'primary':cam_pose},'poses':{'box':{'pos':[-2,0,0],'yaw':0}}},{'offset_s':0.1,'views':{'primary':cam_pose},'poses':{'box':{'pos':[2,0,0],'yaw':0}}}]}
configure_exposure(s,c,{'box':{'ob':ob}},fr,{'name':'primary'},1,True)
pass_paths(o,'motion')
bpy.ops.render.render()
a=read_passes(o,'motion',96,64)
print('MOTION',s.render.use_motion_blur,s.render.motion_blur_shutter, np.where(a['linear'][32,:,0]>0.05)[0].tolist())
np.savez_compressed('out/_tmp/optical_pass_probe/motion.npz',**a)

# Compare actual motion-blurred irradiance with a static reference.
blurred_width=int(np.count_nonzero(a['linear'][32,:,0]>0.05))
configure_exposure(s,c,{'box':{'ob':ob}},fr,{'name':'primary'},1,False)
ob.location=(0,0,0)
pass_paths(o,'static')
bpy.ops.render.render()
static=read_passes(o,'static',96,64)
static_width=int(np.count_nonzero(static['linear'][32,:,0]>0.05))
assert blurred_width>static_width+5,(blurred_width,static_width)
import json
report={'depth_convention':'camera_forward_z_m','flat_plane_depth_m':10,'motion_blur_width_px':blurred_width,'static_width_px':static_width,'passed':True}
open('out/_tmp/optical_pass_probe/results.json','w').write(json.dumps(report,indent=2))
print('SENSOR_PROBE_PASS',report)
