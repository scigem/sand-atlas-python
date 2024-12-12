import sys
import bpy
import pyopenvdb
import numpy
import bmesh
import tifffile

def moment_of_inertia_tensor(voxel_grid):
  """
  Calculates the moment of inertia tensor for a 3D voxellated binary particle.

  Args:
    voxel_grid: A 3D numpy array representing the particle. 
                 1 represents the particle, 0 represents empty space.

  Returns:
    A 3x3 numpy array representing the moment of inertia tensor.
  """

  # Calculate the center of mass
  indices = numpy.argwhere(voxel_grid == 1)
  center_of_mass = numpy.mean(indices, axis=0)

  # Initialize the inertia tensor
  I = numpy.zeros((3, 3))

  # Calculate the inertia tensor elements
  for index in indices:
    x, y, z = index - center_of_mass
    I[0, 0] += y**2 + z**2
    I[1, 1] += x**2 + z**2
    I[2, 2] += x**2 + y**2
    I[0, 1] -= x * y
    I[0, 2] -= x * z
    I[1, 2] -= y * z

  # Make the tensor symmetric
  I[1, 0] = I[0, 1]
  I[2, 0] = I[0, 2]
  I[2, 1] = I[1, 2]

  return I

def ellipse_axes(voxel_grid):
    M = numpy.sum(voxel_grid)
    I = moment_of_inertia_tensor(voxel_grid)
    eig_values, eig_vectors = numpy.linalg.eig(I)

    a = numpy.sqrt(5 * (eig_values[0] + eig_values[1] - eig_values[2]) / (2*M))
    b = numpy.sqrt(5 * (eig_values[1] + eig_values[2] - eig_values[0]) / (2*M))
    c = numpy.sqrt(5 * (eig_values[2] + eig_values[0] - eig_values[1]) / (2*M))
    
    return a, b, c

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

data = data[::8, ::8, ::8]  # Downsample the data by a factor of 8 in each dimension

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

axes = ellipse_axes(data)
dim_min = 2*min(axes)

# print('min_eig_value:', dim_min)
# min_eig_vector = eig_vectors[:, min_eig_value_index]
# theta = numpy.arccos(min_eig_vector[2])  # Angle between the z-axis and the smallest eigenvector

# Get the dimensions of the cube
dim = cube.dimensions
dim_min_ortho = numpy.amin([dim.x, dim.y, dim.z])
# print('COMPARE WITH PREVIOUS RESULT:', dim_min_ortho)

original_volume = numpy.sum(data)*voxel_size_m**3

# Define voxel sizes for different qualities
voxel_sizes = [1, dim_min / 100, dim_min / 30, dim_min / 10, dim_min / 3]

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

    obj = bpy.context.active_object

    # Set the origin of the mesh to the center of the volume
    bpy.ops.object.origin_set(type="ORIGIN_CENTER_OF_VOLUME", center="MEDIAN")

    depsgraph = bpy.context.evaluated_depsgraph_get()

    # Get the evaluated object with all modifiers applied
    eval_obj = obj.evaluated_get(depsgraph)
    eval_mesh = eval_obj.to_mesh()

    # Calculate volume using the evaluated mesh
    bm = bmesh.new()
    bm.from_mesh(eval_mesh)
    volume = bm.calc_volume()

    bm.free()

    # Free the temporary mesh
    eval_obj.to_mesh_clear()
    
    radius_offset = (original_volume / volume)**(1/3)

    # Scale the mesh based on the voxel size and preserving the volume
    obj.scale = (voxel_size_m*radius_offset, voxel_size_m*radius_offset, voxel_size_m*radius_offset)

    # Export the mesh as an STL file
    output_path = f"{input_folder}/stl_{quality}/{particle_name}.stl"  # Set the output file path
    bpy.ops.wm.stl_export(filepath=output_path, export_selected_objects=True, apply_modifiers=True)

# Set the grid as a level set
grid.transform = pyopenvdb.createLinearTransform(voxelSize=1.0)

# Write the grid to a VDB file
output_path = f"{input_folder}/vdb/{particle_name}.vdb"
pyopenvdb.write(output_path, grids=[grid])


# Write the grid to a NPY file
# bbox = grid.evalActiveVoxelBoundingBox()
# output_path = f"{input_folder}/yade/{particle_name}.npy"
# array = numpy.zeros([bbox[1][0] - bbox[0][0], bbox[1][1] - bbox[0][1], bbox[1][2] - bbox[0][2]])
# grid.copyToArray(array)

# numpy.save(output_path, array)
# tifffile.imwrite(output_path[:-4] + ".tiff",
#                  array.astype(numpy.uint16),
#                  resolution=(1/voxel_size_m, 1/voxel_size_m),
#                  resolutionunit='none',
#                  metadata={
#                     "unit": "m",
#                     "spacing": voxel_size_m,  # For ImageJ, this sets the z-spacing if a stack
#                     "axes": "ZYX"
#                  }
#                 )

# output_path = f"{input_folder}/yade/voxel_size_m.txt"
# numpy.savetxt(output_path, [voxel_size_m], fmt="%f")
