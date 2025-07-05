import os
import sys
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

from glob import glob

# from CLUMP import GenerateClump_Euclidean_3D
from scipy.ndimage import distance_transform_edt, maximum_filter
from skimage.measure import label, regionprops
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.morphology import remove_small_objects
import spam.label
from mpl_toolkits.mplot3d import Axes3D


def labelled_image_to_multipheres(labelled_data, sand_type, microns_per_voxel, output_dir, debug=False):
    """Convert a labelled image to a multisphere representation.
    Args:
        labelled_data: A 3D numpy array representing the labelled image.
        sand_type: The type of sand (not used in this function).
        microns_per_voxel: The size of a voxel in microns (not used in this function).
        output_dir: Directory to save the output files.
        debug: If True, will generate debug plots.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    boundingBoxes = spam.label.boundingBoxes(labelled_data)
    for l in np.unique(labelled_data):
        if l == 0:
            continue

        # get the cropped binary mask for this label
        binary_mask = labelled_data == l

        # remove small objects from the binary mask
        binary_mask = remove_small_objects(binary_mask)

        # get the bounding box for this label
        bbox = boundingBoxes[l]
        x_min, x_max, y_min, y_max, z_min, z_max = bbox

        # crop the binary mask
        cropped_mask = binary_mask[x_min:x_max, y_min:y_max, z_min:z_max]

        # generate the multisphere representation
        filenamePrefix = f"particle_{l:06d}"
        binary_to_clump(
            cropped_mask, microns_per_voxel, output_dir, filenamePrefix, numpasses=5, debug=debug, save_all_passes=True
        )


def binary_to_clump(
    img, microns_per_voxel, output_dir, filenamePrefix, numpasses=1, debug=False, save_all_passes=False
):
    """
    Turn an image of a particle into a multisphere clump representation. Algorithm based on Felix Buchele's DEM10 presentation.

    This function performs watershed transformation on the input image and returns a list of spheres at the centroids of the labels.

        img (numpy.ndarray): A 3D numpy array representing the input image.
        microns_per_voxel (float): The size of each voxel in microns.
        output_dir (str): The directory where the output files will be saved.
        filenamePrefix (str): The prefix for the output file names.
        numpasses (int, optional): The number of passes to perform. Defaults to 1.
        debug (bool, optional): Whether to display debugging plots. Defaults to False.
        save_all_passes (bool, optional): Whether to save all passes in a CSV file. Defaults to False.

        tuple: A list of tuples containing the x, y, z coordinates and the distance transform value at each centroid, and the final projected image.
    """

    max_dim = max(img.shape)
    min_distance = max_dim // 2
    true_volume = np.sum(img > 0) * microns_per_voxel**3

    coords = []
    while len(coords) == 0:
        if min_distance < 2:
            sys.exit("Did not find any coordinates for multispheres in the image.")
        min_distance = min_distance // 2
        dt = distance_transform_edt(img)
        coords = peak_local_max(dt, min_distance=min_distance, labels=img)

    clump = []
    for coord in coords:
        clump.append(
            (
                coord[0],
                coord[1],
                coord[2],
                dt[int(coord[0]), int(coord[1]), int(coord[2])],
            )
        )

    # now create an artificial image from the clump
    projected_img = np.zeros_like(img, dtype=np.uint8)
    x = np.arange(img.shape[0])
    y = np.arange(img.shape[1])
    z = np.arange(img.shape[2])
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    for sphere in clump:
        xp, yp, zp, radius = sphere
        this_projected_img = (X - xp) ** 2 + (Y - yp) ** 2 + (Z - zp) ** 2 <= radius**2
        projected_img[this_projected_img] = 1

    for i in range(numpasses):
        min_distance = min_distance // 2 if min_distance > 1 else 1
        # take a difference between the original image and the projected image
        # Only consider areas where original has content but projected doesn't
        diff_img = np.logical_and(img > 0, projected_img == 0).astype(np.uint8)

        # use this as seeds for the clump generation
        dt_diff = distance_transform_edt(diff_img)
        coords = peak_local_max(dt_diff, min_distance=min_distance, labels=diff_img)

        for coord in coords:
            radius = dt_diff[int(coord[0]), int(coord[1]), int(coord[2])]

            # Only add spheres with meaningful radius
            if radius > 0.5:
                clump.append((coord[0], coord[1], coord[2], radius))

        # update the projected image with the new spheres
        for sphere in clump:
            xp, yp, zp, radius = sphere
            distances = (X - xp) ** 2 + (Y - yp) ** 2 + (Z - zp) ** 2
            this_projected_img = distances <= radius**2
            projected_img[this_projected_img] = 1

        diff_img = np.logical_and(img > 0, projected_img == 0).astype(np.uint8)
        error = np.sum(diff_img) / np.sum(img > 0)
        print(f"Pass {i+1}/{numpasses} completed. Current clump size: " f"{len(clump)} spheres. Error: {error:.4f}")

        if save_all_passes or debug:
            os.makedirs(f"{output_dir}/multisphere/quality_{i+1}", exist_ok=True)

        if debug:
            fig = plt.figure()
            ax = fig.add_subplot(131, projection="3d")
            ax.voxels(img, alpha=0.5)
            ax = fig.add_subplot(132, projection="3d")
            ax.voxels(projected_img, alpha=0.5)
            # show the spheres
            ax = fig.add_subplot(133, projection="3d")
            ax.voxels(diff_img, alpha=0.5)
            # for sphere in clump:
            #     xp, yp, zp, radius = sphere
            #     # calculate the projected radius to get the size of the sphere in the plot
            #     ax.scatter(xp, yp, zp, s=radius * 100, alpha=0.5)
            plt.savefig(
                f"{output_dir}/multisphere/quality_{i+1}/{filenamePrefix}.png",
            )

        to_save = np.array(clump)
        # scale the final clump to have the same volume as the original image
        clump_volume = np.sum([4 / 3 * np.pi * (radius**3) for _, _, _, radius in clump])
        vol_scaling = (true_volume / clump_volume) ** (1 / 3)  # Scale factor to match the volume
        to_save[:, :3] *= vol_scaling  # Scale the coordinates by the volume scaling factor
        to_save[:, 3] *= vol_scaling  # Scale the radius by the volume scaling factor
        # to_save *= microns_per_voxel  # Scale the coordinates by microns per voxel

        if save_all_passes:
            np.savetxt(
                f"{output_dir}/multisphere/quality_{i+1}/{filenamePrefix}.csv",
                to_save,
                delimiter=",",
                header="x (microns),y (microns),z (microns),radius (microns)",
                comments="",
            )
    return np.array(clump), projected_img


def stl_to_clump(inputFolder, N, rMin=0, div=102, overlap=0.6):
    """
    Convert an STL file to a clump representation.

    This function uses the CLUMP library to generate a clump from a given STL file.
    It allows for customization of parameters such as the number of spheres, minimum radius,
    division, overlap, and output options.
    """

    rMin = 0  # Minimum radius of spheres in the clump
    div = 102  # Division parameter for clump generation
    overlap = 0.6  # Overlap parameter for clump generation

    rootDir = "/".join(inputFolder.split("/")[:-1])  # Get the root directory of the input folder
    stlFiles = glob(inputFolder + "/*.stl")
    if len(stlFiles) == 0:
        raise ValueError("No STL files found in the specified folder.")

    outputDir = rootDir + "/clumps"
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    for stlFile in stlFiles:
        thisFileName = stlFile.split("/")[-1].split(".")[0]

        outputFileTxt = outputDir + "/" + thisFileName + ".txt"
        outputFileVTK = outputDir + "/" + thisFileName + ".vtk"

        GenerateClump_Euclidean_3D(
            stlFile, N, rMin, div, overlap, output=outputFileTxt, outputVTK=outputFileVTK, visualise=False
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert STL files to clump representation.")
    parser.add_argument("inputFolder", type=str, help="Path to the folder containing STL files.")
    parser.add_argument("N", type=int, help="Number of spheres in the clump.")
    args = parser.parse_args()

    stl_to_clump(args.inputFolder, args.N)

    # fake_image = np.zeros((100, 100, 100), dtype=np.uint8)
    # fake_image[30:70, 30:70, 30:70] = 1

    # clump, projected_img = binary_to_clump(fake_image, numpasses=7, debug=False, save_all_passes=True)
