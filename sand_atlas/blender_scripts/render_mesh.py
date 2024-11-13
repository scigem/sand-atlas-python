import bpy
import argparse
import sys
import numpy
from math import radians

# Function to convert hex color to Blender color
def hex_to_blender_color(hex_code, alpha=1.0):
    # Remove the '#' if present
    hex_code = hex_code.lstrip('#')
    
    # Convert hex to RGB, and then to Blender's 0-1 scale
    r = int(hex_code[0:2], 16) / 255.0
    g = int(hex_code[2:4], 16) / 255.0
    b = int(hex_code[4:6], 16) / 255.0
    
    return (r, g, b, alpha)

# run with:
# blender --background --python mesh_to_blender.py -- /path/to/file.stl

# print(sys.argv)
# filename = sys.argv[-1]
# if filename.isdigit():
#     frame_end = int(sys.argv[-1])
#     filename = sys.argv[-2]
# else:
#     frame_end = 60

# Find the index of '--' to isolate your script's arguments
try:
    idx = sys.argv.index('--') + 1
    # Pass only the arguments after '--' to argparse
    script_args = sys.argv[idx:]
except ValueError:
    script_args = []  # Default to empty list if '--' is not found

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description="Process input for a Blender script.")

# Add positional argument for the filename
parser.add_argument("--filename", type=str, help="Path to the STL file")

# Add optional argument for the end frame
parser.add_argument("--frame_end", type=int, default=60, help="End frame (optional, default is 60)")

# Add optional argument for the end frame
parser.add_argument("--bg_colour", type=str, help="Background colour as a hex string, e.g. 00ff00 (optional)")

parser.add_argument("--fg_colour", type=str, help="Foreground colour as a hex string e.g. ff0000 (optional)")

# Parse arguments, ignoring Blender's own arguments
args = parser.parse_args(script_args)

# Access filename and frame_end from args
filename = args.filename
frame_end = args.frame_end
bg_colour = args.bg_colour
fg_colour = args.fg_colour

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

if bg_colour == 'None':
    bpy.context.scene.render.film_transparent = True
else:
    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = hex_to_blender_color(bg_colour)
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
    rotation=(0, 1.5708, 0),
)
bpy.data.objects["Area"].data.energy = 10

# with bpy.context.temp_override(area="VIEW_3D"):

bpy.ops.wm.stl_import(filepath=filename)
ob = bpy.context.selected_objects[-1]
if fg_colour != 'None':
    # Create a new material
    material = bpy.data.materials.new(name="ImportedObjectMaterial")
    material.use_nodes = True  # Enable nodes to access Principled BSDF

    # Get the Principled BSDF node and set the color
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        # Set color using RGB values (example: red color)
        bsdf.inputs["Base Color"].default_value = hex_to_blender_color(fg_colour)

    # Assign the material to the object
    if ob.data.materials:
        # Replace the existing material
        ob.data.materials[0] = material
    else:
        # Add new material
        ob.data.materials.append(material)


# shade smooth
# for poly in ob.data.polygons:
#     poly.use_smooth = True


# scale object to be unit size
dim = ob.dimensions
dim_max = numpy.amax([dim.x, dim.y, dim.z])
# dim_max = numpy.sqrt(dim.x**2 + dim.y**2 + dim.z**2)

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

scene.render.filepath = filename[:-4] + "/frame_"  # save here
bpy.ops.render.render(animation=True)
