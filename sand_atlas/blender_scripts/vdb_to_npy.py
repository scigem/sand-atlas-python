import bpy
import numpy as np
import sys
import pyopenvdb as vdb

# Ensure the filepath is passed as an argument
if len(sys.argv) < 6:
    print("Error: No .vdb file path provided.")
    sys.exit(1)

# Get the .vdb file path from command-line arguments
vdb_file_path = sys.argv[5]
output_file_name = vdb_file_path.split("/")[-1].split(".")[0] + ".npy"

# Load the .vdb file
volume_data = vdb.read(vdb_file_path, "density")
bbox = volume_data.evalActiveVoxelBoundingBox()

array = np.zeros([bbox[1][0] - bbox[0][0], bbox[1][1] - bbox[0][1], bbox[1][2] - bbox[0][2]])
volume_data.copyToArray(array)

np.save(output_file_name, array)
