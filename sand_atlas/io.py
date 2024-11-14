import os
import sys
import shutil
import subprocess
import platform
import numpy
import tifffile
import nrrd
import json
import h5py


def load_data(filename):
    """
    Load data from a file based on its extension.
    Parameters:
    filename (str): The path to the file to be loaded.
    Returns:
    data: The data loaded from the file. The type of data returned depends on the file extension:
    - For '.tif' or '.tiff' files, returns a memmap or an array from tifffile.
    - For '.raw' files, returns a memmap from numpy.
    - For '.npz' files, returns an array from numpy.
    - For '.nrrd' files, returns the data and header from nrrd.
    """
    extension = filename.split(".")[-1]
    if (extension.lower() == "tif") or (extension.lower() == "tiff"):
        try:
            data = tifffile.memmap(filename)
        except Exception:
            data = tifffile.imread(filename)
    elif extension.lower() == "raw":
        data = numpy.memmap(filename)
    elif extension.lower() == "npz":
        data = numpy.load(filename, allow_pickle=True)["arr_0"]
    elif extension.lower() == "nrrd":
        data, header = nrrd.read(filename)
    elif extension.lower() == "h5":
        with h5py.File(filename, "r") as f:
            keys = list(f.keys())
            data = f[keys[0]]
    else:
        raise ValueError("Unsupported file extension")

    return data


def save_data(data, filename):
    """
    Save data to a file with the specified filename and extension.

    Parameters:
    data (array-like): The data to be saved.
    filename (str): The name of the file to save the data to. The extension of the filename determines the format in which the data will be saved.

    Supported file extensions:
    - 'tif' or 'tiff': Save data as a TIFF file using tifffile.imsave.
    - 'raw': Save data as a raw binary file using data.tofile.
    - 'npz': Save data as a NumPy compressed file using numpy.savez.
    - 'nrrd': Save data as an NRRD file using nrrd.write.

    Raises:
    ValueError: If the file extension is not supported.
    """
    root_dir = "/".join(filename.split("/")[:-1])
    extension = filename.split(".")[-1]

    if (extension.lower() == "tif") or (extension.lower() == "tiff"):
        tifffile.imsave(filename, data)
    elif extension.lower() == "raw":
        data.tofile(filename)
    elif extension.lower() == "npz":
        numpy.savez(filename, data)
    elif extension.lower() == "nrrd":
        nrrd.write(filename, data)
    elif extension.lower() == "h5":
        with h5py.File(filename, "w") as f:
            f.create_dataset("arr_0", data=data)
    elif extension.lower() == "neuroglancer":
        try:
            from cloudvolume import CloudVolume
        except ImportError:
            raise ImportError("To save data in Neuroglancer format, please install the 'cloud-volume' package")
        # Create the directory for Neuroglancer Precomputed output
        output_path = root_dir + "/neuroglancer"
        os.makedirs(output_path, exist_ok=True)

        # Specify the layer properties
        info = CloudVolume.create_new_info(
            num_channels=1,  # or more if it's a multi-channel TIFF
            layer_type="segmentation",  # or 'image' if you're working with an image layer
            data_type=data.dtype,  # or match your data type (e.g., uint8, float32)
            encoding="raw",  # raw encoding for segmentation (use 'jpeg' or 'compressed_segmentation' for image data)
            resolution=[1, 1, 1],  # voxel resolution, set appropriately
            voxel_offset=[0, 0, 0],  # starting point in your coordinate system
            chunk_size=[64, 64, 64],  # size of chunks (tune for performance)
            volume_size=data.shape,  # shape in (X, Y, Z)
        )

        # Create the CloudVolume and add the image data
        vol = CloudVolume(f"file://{output_path}", info=info, compress=False)
        vol.commit_info()

        # Add the tiff data to the volume
        vol[:, :, :] = data
        # vol[:, :, :] = numpy.expand_dims(data, axis=-1)  # Expand dimensions if necessary
    else:
        raise ValueError("Unsupported file extension")


def convert(input_filename, output_filename):
    """
    Converts data from an input file and saves it to an output file.

    Args:
        input_filename (str): The path to the input file containing the data to be converted.
        output_filename (str): The path to the output file where the converted data will be saved.

    Returns:
        None
    """
    data = load_data(input_filename)
    save_data(data, output_filename)


def load_json(filename):
    with open(filename) as f:
        return json.load(f)


def find_blender():
    # Try finding blender in PATH using shutil.which
    blender_path = shutil.which("blender")

    if blender_path:
        return blender_path

    # If not found in PATH, manually check typical installation locations
    system_platform = platform.system()

    if system_platform == "Darwin":  # macOS
        possible_paths = [
            "/Applications/Blender.app/Contents/MacOS/Blender",  # Typical macOS location
            "/usr/local/bin/blender",
        ]
    elif system_platform == "Linux":  # Linux
        possible_paths = ["/usr/bin/blender", "/usr/local/bin/blender"]
    elif system_platform == "Windows":  # Windows
        possible_paths = [
            "C:\\Program Files\\Blender Foundation\\Blender\\blender.exe",
            "C:\\Program Files (x86)\\Blender Foundation\\Blender\\blender.exe",
        ]
    else:
        possible_paths = []

    # Check each possible path
    for path in possible_paths:
        if os.path.exists(path):
            return path

    return None


def add_to_path(blender_path):
    # Get the directory of the blender executable
    blender_dir = os.path.dirname(blender_path)

    # Add this directory to the PATH environment variable
    os.environ["PATH"] += os.pathsep + blender_dir


def check_blender_command():
    # Find Blender path
    blender_path = find_blender()

    if blender_path is None:
        print("Error: `blender` command not found.")
        print("Make sure Blender is installed and accessible in your PATH.")
        sys.exit(1)

    # Add the found blender path to the PATH if necessary
    add_to_path(blender_path)

    # Try running blender after adding to PATH
    try:
        subprocess.run(["blender", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"CalledProcessError: {e}")
        print("Error: Blender is installed but couldn't be executed.")
        sys.exit(1)


def check_ffmpeg_command():
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"CalledProcessError: {e}")
        print("Error: ffmpeg is not installed or not accessible in your PATH.")
        sys.exit(1)


def make_zips(data_foldername, output_foldername):
    print("Making zips...")

    if os.path.exists(output_foldername):
        os.system(f"rm -rf {output_foldername}/*.zip")
    else:
        os.makedirs(output_foldername)

    for quality in ["ORIGINAL", "100", "30", "10", "3"]:
        os.system(f"zip -j {output_foldername}/meshes_{quality}.zip {data_foldername}/stl_{quality}/*.stl")
        os.system(
            f"cp {data_foldername}/stl_{quality}/particle_00001.stl {output_foldername}/ref_particle_{quality}.stl"
        )
    os.system(f"zip -j {output_foldername}/level_sets.zip {data_foldername}/vdb/*.vdb")
