import os
import numpy
import argparse
import subprocess
import pandas as pd
from tqdm import tqdm
from skimage.measure import label, regionprops, regionprops_table
from skimage.morphology import closing
from skimage.filters import threshold_otsu, gaussian
import sand_atlas.io
import sand_atlas.video


def gray_to_bw(data, threshold=None, blur=None):
    """
    Apply a threshold to the data loaded from a file.

    Parameters:
    filename (str): The path to the file containing the data.
    threshold (float, optional): The threshold value to apply. If None, Otsu's method is used to determine the threshold.
    blur (float, optional): The sigma value for Gaussian blur. If provided, the data will be blurred before thresholding.

    Returns:
    numpy.ndarray: A binary array where values above the threshold are True and values below are False.
    """

    if threshold is None:
        threshold = threshold_otsu(data)

    if blur is not None:
        data = gaussian(data, sigma=blur)

    print(f"Thresholding at {threshold}")
    binary = data > threshold

    return binary


def label_binary_data(binary_data, minimum_voxels=100):
    """
    Labels connected components in a binary data array.

    This function takes a binary data array, performs a morphological closing operation
    using a 3x3x3 structuring element, and then labels the connected components in the
    closed binary data. Labelled areas smaller than 100 voxels are then removed and the
    array is relabelled sequentially.

    Parameters:
    binary_data (numpy.ndarray): A 3D binary data array where the connected components are to be labeled.

    Returns:
    numpy.ndarray: A 3D array with the same shape as `binary_data`, where each connected component is assigned a unique integer label.
    """

    cube = numpy.ones((3, 3, 3), dtype="uint8")
    closed = closing(binary_data, cube)

    labelled = label(closed)
    print(f"Labelled {labelled.max()} regions. Filtering out regions less than {minimum_voxels} voxels.")

    # Step 2: Count the size of each label
    label_sizes = numpy.bincount(labelled.ravel())

    # Step 3: Create a mask to remove small labels
    large_labels = numpy.where(label_sizes >= minimum_voxels)[0]

    # Step 4: Create a mask of the valid labels
    filtered_image = numpy.isin(labelled, large_labels.astype(int)) * labelled
    filtered_image = filtered_image.astype(int)

    unique_labels = numpy.unique(filtered_image)

    relabelled = numpy.zeros_like(labelled)
    for i in range(1, len(unique_labels)):  # skip 0
        relabelled[filtered_image == unique_labels[i]] = i
    print(f"Relabelled image with {relabelled.max()} labels.")

    return relabelled


def labelled_image_to_mesh(labelled_data, sand_type, microns_per_voxel, output_dir, debug=False):
    """
    Converts labelled image data to 3D mesh files using Blender.

    Parameters:
    labelled_data (numpy.ndarray): 3D array where each voxel is labelled with an integer indicating the particle it belongs to.
    sand_type (str): Type of sand being processed.
    microns_per_voxel (float): Size of each voxel in microns.
    output_dir (str): Directory where output files will be saved.
    debug (bool, optional): If True, enables debug mode. Default is False.

    Returns:
    None

    This function processes the labelled image data to identify individual particles, filters out particles touching the edges,
    and saves each particle as a .npy file. It then calls a Blender script to convert these .npy files into 3D mesh files.
    """
    current_file_path = os.path.abspath(__file__)
    blender_script_path = os.path.join(os.path.dirname(current_file_path), "blender_scripts", "vdb.py")

    voxel_size_m = microns_per_voxel * 1e-6
    if not os.path.exists(output_dir):
        os.mkdir({output_dir})

    for subfolder in ["npy", "stl_3", "stl_10", "stl_30", "stl_100", "stl_ORIGINAL", "vdb"]:
        if not os.path.exists(f"{output_dir}/{subfolder}/"):
            os.mkdir(f"{output_dir}/{subfolder}/")

    num_particles = numpy.amax(labelled_data)
    print(f"Found {num_particles} labels")

    spacing = (voxel_size_m, voxel_size_m, voxel_size_m)
    props = regionprops(labelled_data, spacing=spacing)
    print("Calculated region properties")

    # # filter out small particles
    # j = 0
    # for i in tqdm(range(num_particles)):
    #     if props[i].area > 100 * (voxel_size_m**3):
    #         j += 1

    # print(f"Only {j} particles are larger than 100 voxels")

    nx, ny, nz = labelled_data.shape
    j = 0

    for i in tqdm(range(num_particles)):
        # if props[i].area > 100 * (voxel_size_m**3):

        x_min, y_min, z_min, x_max, y_max, z_max = props[i].bbox

        if x_min == 0 or y_min == 0 or z_min == 0 or x_max == nx or y_max == ny or z_max == nz:
            print(f"Particle {i} touching edge of the box, skipping")
        else:
            crop = labelled_data[x_min:x_max, y_min:y_max, z_min:z_max]

            this_particle = crop == props[i].label

            this_particle = numpy.pad(this_particle, 1, mode="constant")

            outname = f"{output_dir}/npy/particle_{props[i].label:05}.npy"
            numpy.save(outname, this_particle)

            print(str(voxel_size_m))

            subprocess.run(
                [
                    "blender",
                    "--background",
                    "--python",
                    blender_script_path,
                    "--",
                    outname,
                    str(voxel_size_m),  # Pass voxel_size as an argument
                ]
            )

            j += 1
    print(f"{j} out of {num_particles} particles saved to disk")


def get_particle_properties(labelled_data, raw_data, microns_per_voxel):
    """
    Calculate particle properties from labelled and raw image data.

    Parameters:
    labelled_data (ndarray): The labelled image data where each particle is assigned a unique label.
    raw_data (ndarray): The raw intensity image data.
    microns_per_voxel (float): The size of each voxel in microns.

    Returns:
    DataFrame: A pandas DataFrame containing the calculated properties for each particle:
    - Volume (µm³): The volume of each particle.
    - Equivalent Diameter (µm): The diameter of a sphere with the same volume as the particle.
    - Major Axis Length (µm): The length of the major axis of the particle.
    - Minor Axis Length (µm): The length of the minor axis of the particle.
    - Aspect Ratio: The ratio of the major axis length to the minor axis length.
    """

    props = regionprops_table(
        labelled_data,
        intensity_image=raw_data,
        spacing=(microns_per_voxel, microns_per_voxel, microns_per_voxel),
        properties=("area", "equivalent_diameter", "major_axis_length", "minor_axis_length"),
    )

    df = pd.DataFrame(props)
    df.index = df.index + 1  # Start indexing at 1
    df["Aspect Ratio"] = df["major_axis_length"] / df["minor_axis_length"]

    return df.rename(
        columns={
            "area": "Volume (µm³)",
            "equivalent_diameter": "Equivalent Diameter (µm)",
            "major_axis_length": "Major Axis Length (µm)",
            "minor_axis_length": "Minor Axis Length (µm)",
        }
    )


def full_analysis_script():
    """
    Perform a full analysis of a sand sample based on provided arguments.

    This function sets up an argument parser to handle command-line arguments for analyzing a sand sample.
    It requires a JSON file describing the data and optionally accepts paths to raw and labelled data files,
    a threshold value, a sigma value for Gaussian blur, and a binning factor.

    Command-line Arguments:
    - json (str): The path to the JSON file containing the description of the data.
    - --raw (str, optional): The path to the file containing the raw data. Default is None.
    - --label (str, optional): The path to the file containing the labelled data. Default is None.
    - --threshold (int, optional): The threshold value to use. Default is None.
    - --blur (float, optional): The sigma value for Gaussian blur. Default is None.
    - --binning (int, optional): The binning factor to use. Default is None.

    Returns:
    None

    Raises:
    SystemExit: If neither raw data file nor labelled data file is provided.
    """
    parser = argparse.ArgumentParser(description="Perform a full analysis of a sand sample.")
    parser.add_argument("json", type=str, help="The path to the json file containing the description of the data.")
    parser.add_argument("--raw", type=str, help="The path to the file containing the raw data.", default=None)
    parser.add_argument("--label", type=str, help="The path to the file containing the labelled data.", default=None)
    parser.add_argument("--threshold", type=int, help="The threshold value to use.", default=None)
    parser.add_argument("--blur", type=float, help="The sigma value for Gaussian blur.", default=None)
    parser.add_argument("--binning", type=int, help="The binning factor to use.", default=None)

    args = parser.parse_args()

    if args.raw is None and args.label is None:
        print("You must provide a raw data file, a labelled data file or both. Try `sand_atlas_process --help`.")
        return

    full_analysis(
        args.json,
        raw_data_filename=args.raw,
        labelled_data_filename=args.label,
        threshold=args.threshold,
        blur=args.blur,
        binning=args.binning,
    )


def properties_script():
    """
    Perform a full analysis of a sand sample based on provided JSON and label data files.

    This script processes the input JSON file to extract metadata and uses the label data file
    to compute particle properties. Optionally, raw data and binning factor can be provided
    to adjust the analysis.

    Command-line Arguments:
    - json (str): The path to the JSON file containing the description of the data.
    - label (str): The path to the file containing the labelled data.
    - raw (str, optional): The path to the file containing the raw data. Default is None.
    - binning (int, optional): The binning factor to use. Default is None.

    The script performs the following steps:
    1. Parses command-line arguments.
    2. Validates the existence of the JSON file.
    3. Loads and processes the JSON data to extract the microns per voxel.
    4. Loads and optionally bins the label data.
    5. Loads and optionally bins the raw data if provided.
    6. Computes particle properties using the label and raw data.
    7. Saves the computed properties to a CSV file named "summary.csv".

    Returns:
    None
    """
    parser = argparse.ArgumentParser(description="Perform a full analysis of a sand sample.")
    parser.add_argument("json", type=str, help="The path to the json file containing the description of the data.")
    parser.add_argument("label", type=str, help="The path to the file containing the labelled data.", default=None)
    parser.add_argument("--raw", type=str, help="The path to the file containing the raw data.", default=None)
    parser.add_argument("--binning", type=int, help="The binning factor to use.", default=None)

    args = parser.parse_args()

    if not os.path.exists(args.json):
        print("The JSON file does not exist.")
        return
    else:
        json_data = sand_atlas.io.load_json(args.json)
        microns_per_voxel = float(json_data["microns_per_pixel"])
        if args.binning is not None:
            microns_per_voxel *= float(args.binning)

    label_data = sand_atlas.io.load_data(args.label)
    if args.binning is not None:
        label_data = bin_data(label_data, args.binning)
    if args.raw is not None:
        raw_data = sand_atlas.io.load_data(args.raw)
        if args.binning is not None:
            raw_data = bin_data(raw_data, args.binning)
    else:
        raw_data = None

    df = get_particle_properties(label_data, raw_data, microns_per_voxel)
    df.to_csv("summary.csv", index_label="Particle ID")


def vdb_to_npy():
    """
    Converts a VDB file to a NumPy .npy file using a Blender script.

    This function checks if the Blender command is available, parses the command-line arguments to get the VDB filename,
    constructs the path to the Blender script, and runs the Blender script in the background to perform the conversion.

    Args:
        None

    Command-line Arguments:
        vdb_filename (str): The path to the VDB file to be converted.

    Raises:
        SystemExit: If the Blender command is not available or if there are issues with the command-line arguments.
    """
    sand_atlas.io.check_blender_command()

    parser = argparse.ArgumentParser(description="Convert a vbd file to a numpy npy file.")
    parser.add_argument("vdb_filename", type=str, help="The path to the VDB file.")

    args = parser.parse_args()
    current_file_path = os.path.abspath(__file__)
    blender_script_path = os.path.join(os.path.dirname(current_file_path), "blender_scripts", "vdb_to_npy.py")

    subprocess.run(["blender", "--background", "--python", blender_script_path, "--", args.vdb_filename])


def full_analysis(
    json_filename, raw_data_filename=None, labelled_data_filename=None, threshold=None, blur=None, binning=None
):
    """
    Perform a full analysis of sand particle data.

    This function performs a comprehensive analysis of sand particle data, including loading raw and labelled data,
    processing the data, generating 3D meshes, creating videos, and saving results.

    Parameters:
    -----------
    json_filename : str
        Path to the JSON file containing metadata and configuration.
    raw_data_filename : str, optional
        Path to the raw data file. If None, the function will skip loading raw data.
    labelled_data_filename : str, optional
        Path to the labelled data file. If None, the function will generate labelled data from raw data.
    threshold : float, optional
        Threshold value for converting grayscale images to binary images. Used if labelled data needs to be generated.
    blur : float, optional
        Blur value for preprocessing the raw data before thresholding. Used if labelled data needs to be generated.
    binning : int, optional
        Binning factor to downsample the data. If provided, the data will be downsampled by this factor.

    Returns:
    --------
    None
    """
    sand_atlas.io.check_blender_command()
    sand_atlas.io.check_ffmpeg_command()

    if not os.path.exists(json_filename):
        print("The JSON file does not exist.")
        return
    else:
        sand_type = json_filename.split("/")[-1].split(".")[0]
        json_data = sand_atlas.io.load_json(json_filename)
        microns_per_voxel = float(json_data["microns_per_pixel"])
        if binning is not None:
            microns_per_voxel *= float(binning)

    output_dir = f"output/{sand_type}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(f"{output_dir}/upload"):
        os.makedirs(f"{output_dir}/upload")

    if raw_data_filename is not None:
        raw_data = sand_atlas.io.load_data(raw_data_filename)
        if binning is not None:
            raw_data = bin_data(raw_data, binning)

    if labelled_data_filename is None:
        labelled_data_filename = f"{output_dir}/upload/{sand_type}-labelled.tif"

    properties_filename = f"{output_dir}/summary.csv"

    if os.path.exists(labelled_data_filename):
        labelled_data = sand_atlas.io.load_data(labelled_data_filename)
        if binning is not None:
            labelled_data = bin_data(labelled_data, binning)
    else:
        binary_data = gray_to_bw(raw_data, threshold, blur)
        labelled_data = label_binary_data(binary_data)
        sand_atlas.io.save_data(labelled_data, labelled_data_filename)

    labelled_image_to_mesh(labelled_data, sand_type, microns_per_voxel, output_dir, debug=False)

    sand_atlas.io.make_zips(output_dir, output_dir + "/upload/")

    if not os.path.exists(properties_filename):
        df = get_particle_properties(labelled_data, raw_data, microns_per_voxel)
        df.to_csv(properties_filename, index_label="Particle ID")

    stl_foldername = f"{output_dir}/stl_ORIGINAL"
    print("Making website videos")
    sand_atlas.video.make_website_video(stl_foldername, f"{output_dir}/upload/")
    print("Making individual videos")
    sand_atlas.video.make_individual_videos(stl_foldername, f"{output_dir}/media/")

    if not os.path.exists(f"{output_dir}/upload/{sand_type}-raw.tif"):
        os.system(f"cp {raw_data_filename} {output_dir}/upload/{sand_type}-raw.tif")
    if not os.path.exists(f"{output_dir}/upload/{sand_type}-labelled.tif"):
        os.system(f"cp {labelled_data_filename} {output_dir}/upload/{sand_type}-labelled.tif")


def bin_data(data, factor):
    """
    Downsample a 3D data array by a given factor.

    Parameters:
    data (numpy.ndarray): The 3D data array to be downsampled.
    factor (int): The downsampling factor.

    Returns:
    numpy.ndarray: The downsampled 3D data array.
    """

    # return data[::factor, ::factor, ::factor]

    # Trim the array to make each dimension divisible by the factor
    trimmed_shape = (
        data.shape[0] - data.shape[0] % factor,
        data.shape[1] - data.shape[1] % factor,
        data.shape[2] - data.shape[2] % factor,
    )

    trimmed_array = data[: trimmed_shape[0], : trimmed_shape[1], : trimmed_shape[2]]

    # Calculate the shape of the downscaled array
    new_shape = (trimmed_shape[0] // factor, trimmed_shape[1] // factor, trimmed_shape[2] // factor)

    # Reshape and compute the median for each block
    downscaled_array = numpy.median(
        trimmed_array.reshape(new_shape[0], factor, new_shape[1], factor, new_shape[2], factor), axis=(1, 3, 5)
    )

    return downscaled_array
