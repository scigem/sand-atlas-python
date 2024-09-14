import bpy
import sys
import numpy
from math import radians

# run with:
# blender --background --python mesh_to_blender.py -- /path/to/file.stl


filename = sys.argv[-1]
if filename.isdigit():
    frame_end = int(sys.argv[-1])
    filename = sys.argv[-2]
else:
    frame_end = 60
# Deselect all objects
bpy.ops.object.select_all(action="DESELECT")

# Select all objects in the scene
bpy.ops.object.select_all(action="SELECT")

# Delete all selected objects
bpy.ops.object.delete()

scene = bpy.context.scene

# make collection
new_collection = bpy.data.collections.new("Meshes")
scene.collection.children.link(new_collection)

scene.transform_orientation_slots[0].type = "LOCAL"
scene.frame_start = 1
scene.frame_end = frame_end

scene.render.resolution_x = 1000
scene.render.resolution_y = 1000
scene.render.resolution_percentage = 100

bpy.context.scene.render.film_transparent = True
# bpy.data.worlds['World'].color = [16/255, 16/255, 16/255]
# bpy.ops.world.new()
# bpy.context.scene.world.color = [16/255, 16/255, 16/255]


cam_data = bpy.data.cameras.new("Cam")
cam_ob = bpy.data.objects.new(name="Cam", object_data=cam_data)
scene.collection.objects.link(cam_ob)
scene.camera = cam_ob  # set the active camera
cam_data.type = "ORTHO"
# cam_data.clip_end = numpy.amax(dimensions*2)
# cam_data.lens = 18  # zoom
cam_data.ortho_scale = 1.15


cam_ob.rotation_euler = (1.570797, 0, 1.570797)
# (dimensions[0], dimensions[1]/2, dimensions[2]/2)
cam_ob.location = (1, 0, 0)

bpy.ops.object.light_add(
    type="AREA",
    radius=1,
    align="WORLD",
    location=(1.01, 0, 0),
    #  1.01*dimensions[0],
    #  dimensions[1]/2,
    #  dimensions[2]/2),
    rotation=(0, 1.5708, 0),
)
bpy.data.objects["Area"].data.energy = 10

# with bpy.context.temp_override(area="VIEW_3D"):
bpy.ops.wm.stl_import(filepath=filename)
ob = bpy.context.selected_objects[-1]


# data = numpy.load(filename, allow_pickle=True)
# edges = []
# new_mesh = bpy.data.meshes.new('mesh')
# new_mesh.from_pydata(data['vertices'], edges, data['faces'])
# new_mesh.update(calc_edges=True)
# # make object from mesh
# ob = bpy.data.objects.new('object', new_mesh)


# bpy.context.view_layer.objects.active = bpy.data.objects["object"]
# bpy.data.objects["object"].select_set(True)


# shade smooth
# for poly in ob.data.polygons:
#     poly.use_smooth = True


# scale object to be unit size
dim = ob.dimensions
dim_max = numpy.amax([dim.x, dim.y, dim.z])

ob.scale = (1.0 / dim_max, 1.0 / dim_max, 1.0 / dim_max)

# add object to scene collection
new_collection.objects.link(ob)

ob.keyframe_insert("rotation_euler", frame=1)
ob.rotation_euler.z = radians(360 * 1 / frame_end)
ob.keyframe_insert("rotation_euler", frame=2)
ob.rotation_euler.z = radians(360 * (frame_end - 1) / frame_end)
ob.keyframe_insert("rotation_euler", frame=frame_end - 1)
ob.rotation_euler.z = radians(360)
ob.keyframe_insert("rotation_euler", frame=frame_end)

# make everything rotate around its local axis
mesh_obs = [o for o in scene.objects if o.type == "MESH"]
with bpy.context.temp_override(selected_objects=mesh_obs):
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="MEDIAN")
ob.matrix_world.translation -= ob.location

# Now lets fill in any holes in the mesh left from the marching cubes
# bpy.context.view_layer.objects.active = ob
# bpy.ops.object.editmode_toggle()
# # bpy.ops.mesh.fill()
# # bpy.ops.mesh.fill_holes()
# bpy.ops.mesh.fill_grid()
# bpy.ops.object.editmode_toggle()


scene.render.filepath = filename[:-4] + "/frame_"  # save here
bpy.ops.render.render(animation=True)
