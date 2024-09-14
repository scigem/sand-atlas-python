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


def threshold(data, threshold=None, blur=None):
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


def label_binary_data(binary_data):
    """
    Labels connected components in a binary data array.

    This function takes a binary data array, performs a morphological closing operation
    using a 5x5x5 structuring element, and then labels the connected components in the
    closed binary data.

    Parameters:
    binary_data (numpy.ndarray): A 3D binary data array where the connected components are to be labeled.

    Returns:
    numpy.ndarray: A 3D array with the same shape as `binary_data`, where each connected component is assigned a unique integer label.
    """

    cube = numpy.ones((5, 5, 5), dtype="uint8")
    closed = closing(binary_data, cube)

    labelled = label(closed)

    return labelled


def labelled_image_to_mesh(labelled_data, sand_type, microns_per_voxel, debug=False):
    current_file_path = os.path.abspath(__file__)
    blender_script_path = os.path.join(os.path.dirname(current_file_path), "blender_scripts", "vdb.py")

    voxel_size_m = microns_per_voxel * 1e-6
    if not os.path.exists(f"output/{sand_type}"):
        os.mkdir(f"output/{sand_type}")

    for subfolder in ["npy", "stl_3", "stl_10", "stl_30", "stl_100", "stl_ORIGINAL", "vdb"]:
        if not os.path.exists(f"output/{sand_type}/{subfolder}/"):
            os.mkdir(f"output/{sand_type}/{subfolder}/")

    num_particles = numpy.amax(labelled_data)
    print(f"Found {num_particles} labels")

    spacing = (voxel_size_m, voxel_size_m, voxel_size_m)
    props = regionprops(labelled_data, spacing=spacing)
    print("Calculated region properties")

    # filter out small particles
    j = 0
    for i in tqdm(range(num_particles)):
        if props[i].area > 100 * (voxel_size_m**3):
            j += 1

    print(f"Only {j} particles are larger than 100 voxels")

    nx, ny, nz = labelled_data.shape
    j = 0

    for i in tqdm(range(num_particles)):
        if props[i].area > 100 * (voxel_size_m**3):

            x_min, y_min, z_min, x_max, y_max, z_max = props[i].bbox

            if x_min == 0 or y_min == 0 or z_min == 0 or x_max == nx or y_max == ny or z_max == nz:
                print(f"Particle {i} touching edge of the box, skipping")
            else:
                crop = labelled_data[x_min:x_max, y_min:y_max, z_min:z_max]

                this_particle = crop == props[i].label

                this_particle = numpy.pad(this_particle, 1, mode="constant")

                outname = f"output/{sand_type}/npy/particle_{props[i].label:05}.npy"
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
    props = regionprops_table(
        labelled_data,
        intensity_image=raw_data,
        spacing=(microns_per_voxel, microns_per_voxel, microns_per_voxel),
        properties=("area", "equivalent_diameter", "major_axis_length", "minor_axis_length"),
    )
    df = pd.DataFrame(props)
    df = df.drop(index=0)  # drop the background
    df["Aspect Ratio"] = df["major_axis_length"] / df["minor_axis_length"]

    return df.rename(
        columns={
            "area": "Area (micron<sup>2</sup>)",
            "equivalent_diameter": "Equivalent Diameter (micron)",
            "major_axis_length": "Major Axis Length (micron)",
            "minor_axis_length": "Minor Axis Length (micron)",
        }
    )


def script():
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


def full_analysis(
    json_filename, raw_data_filename=None, labelled_data_filename=None, threshold=None, blur=None, binning=None
):
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

    if raw_data_filename is not None:
        raw_data = sand_atlas.io.load_data(raw_data_filename)
        if binning is not None:
            raw_data = raw_data[::binning, ::binning, ::binning]

    if labelled_data_filename is None:
        labelled_data_filename = f"{output_dir}/labelled_data.tif"

    properties_filename = f"{output_dir}/summary.csv"

    if os.path.exists(labelled_data_filename):
        labelled_data = sand_atlas.io.load_data(labelled_data_filename)
        if binning is not None:
            labelled_data = labelled_data[::binning, ::binning, ::binning]
    else:
        binary_data = threshold(raw_data, threshold, blur)
        labelled_data = label_binary_data(binary_data)
        sand_atlas.io.save_data(labelled_data, labelled_data_filename)

    labelled_image_to_mesh(labelled_data, sand_type, microns_per_voxel, debug=False)

    sand_atlas.io.make_zips(output_dir, output_dir + "/upload/")

    if not os.path.exists(properties_filename):
        df = get_particle_properties(labelled_data, raw_data, microns_per_voxel)
        df.to_csv(properties_filename, index_label="Particle ID")

    stl_foldername = f"output/{sand_type}/stl_ORIGINAL"
    sand_atlas.video.make_website_video(stl_foldername, f"output/{sand_type}/upload/")
    sand_atlas.video.make_instagram_videos(stl_foldername, f"output/{sand_type}/media/")

    if not os.path.exists(f"output/{sand_type}/upload/{sand_type}-raw.tif"):
        os.system(f"cp {raw_data_filename} output/{sand_type}/upload/{sand_type}-raw.tif")
    if not os.path.exists(f"output/{sand_type}/upload/{sand_type}-labelled.tif"):
        os.system(f"cp {labelled_data_filename} output/{sand_type}/upload/{sand_type}-labelled.tif")
