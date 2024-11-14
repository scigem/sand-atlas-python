import sys
import bpy
import pyopenvdb
import numpy

# import tifffile

# data = numpy.random.rand(50, 50, 50)
# data = tifffile.imread("SHAPE/particle_10.tiff")
input_file = sys.argv[-2]  # File path is second-to-last argument
voxel_size_m = float(sys.argv[-1])  # Voxel size is the last argument

# Ensure voxel_size_m is positive and non-zero
if voxel_size_m <= 0:
    raise ValueError("Voxel size must be a positive, non-zero value.")
else:
    print("Voxel size [m]=", voxel_size_m)

data = numpy.load(input_file)

# outname = ".".join(input_file.split(".")[:-1])
input_folder = "/".join(input_file.split("/")[:-2])  # Get the folder name stripping the "npy" part too
particle_name = input_file.split("/")[-1].split(".")[0]

grid = pyopenvdb.FloatGrid()
grid.copyFromArray(data.astype(float))

grid.background = 0.0
grid.gridClass = pyopenvdb.GridClass.FOG_VOLUME
grid.name = "density"

# Write the OpenVDB grid to a file
vdb_file = "/tmp/volume.vdb"
pyopenvdb.write(vdb_file, grids=[grid])

# Import the OpenVDB file into Blender
bpy.ops.object.volume_import(filepath=vdb_file)

# Deselect all objects
bpy.ops.object.select_all(action="DESELECT")

# Select the default cube (usually named "Cube" by default)
cube = bpy.data.objects.get("Cube")

# Select the cube
cube.select_set(True)

# Set the cube as the active object
bpy.context.view_layer.objects.active = cube

# Add the "Volume to Mesh" modifier
volume_to_mesh_modifier = cube.modifiers.new(name="VolumeToMesh", type="VOLUME_TO_MESH")
bpy.context.object.modifiers["VolumeToMesh"].object = bpy.data.objects["volume"]

# Set the parameters for the "Volume to Mesh" modifier
volume_to_mesh_modifier.resolution_mode = "VOXEL_SIZE"  # Options: 'GRID', 'VOXEL_AMOUNT', 'VOXEL_SIZE'
volume_to_mesh_modifier.threshold = 0.5  # Set the threshold for the volume to mesh conversion

# Adjust the voxel size for fineness
volume_to_mesh_modifier.voxel_size = 1.0
# Set adaptivity to control mesh simplification
volume_to_mesh_modifier.adaptivity = 0.0
# Apply the "Volume to Mesh" modifier
bpy.ops.object.modifier_apply(modifier="VolumeToMesh")

# Get the dimensions of the cube
dim = cube.dimensions
dim_min = numpy.amin([dim.x, dim.y, dim.z])

# Define voxel sizes for different qualities
voxel_sizes = [1, dim_min / 100, dim_min / 30, dim_min / 10, dim_min / 3]
print(voxel_sizes)

for quality in ["ORIGINAL", "100", "30", "10", "3"]:
    if quality == "ORIGINAL":
        pass
    elif quality == "100":
        remesh_modifier = cube.modifiers.new(name="Remesh", type="REMESH")
        remesh_modifier.mode = "VOXEL"
        remesh_modifier.voxel_size = voxel_sizes[1]
    elif quality == "30":
        remesh_modifier.voxel_size = voxel_sizes[2]
    elif quality == "10":
        remesh_modifier.voxel_size = voxel_sizes[3]
    elif quality == "3":
        remesh_modifier.voxel_size = voxel_sizes[4]

    # The volume is now a mesh object
    mesh_obj = bpy.context.active_object

    # Scale the mesh based on the voxel size
    mesh_obj.scale = (voxel_size_m, voxel_size_m, voxel_size_m)

    # Set the origin of the mesh to the center of the volume
    bpy.ops.object.origin_set(type="ORIGIN_CENTER_OF_VOLUME", center="MEDIAN")

    # Export the mesh as an STL file
    output_path = f"{input_folder}/stl_{quality}/{particle_name}.stl"  # Set the output file path
    bpy.ops.wm.stl_export(filepath=output_path, export_selected_objects=True, apply_modifiers=True)

# Set the grid as a level set
grid.transform = pyopenvdb.createLinearTransform(voxelSize=1.0)

# Write the grid to a VDB file
output_path = f"{input_folder}/vdb/{particle_name}.vdb"
pyopenvdb.write(output_path, grids=[grid])


# Write the grid to a NPY file
bbox = grid.evalActiveVoxelBoundingBox()
output_path = f"{input_folder}/npy/{particle_name}.npy"
array = numpy.zeros([bbox[1][0] - bbox[0][0], bbox[1][1] - bbox[0][1], bbox[1][2] - bbox[0][2]])
grid.copyToArray(array)

numpy.save(output_path, array)
