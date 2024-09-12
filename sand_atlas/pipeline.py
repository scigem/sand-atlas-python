import os
import numpy
from skimage.measure import label, regionprops_table
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


def get_particle_properties(labelled_data, raw_data):
    regions = regionprops_table(
        labelled_data,
        intensity_image=raw_data,
        properties=("area", "equivalent_diameter", "minor_axis_length", "major_axis_length"),
    )
    return regions


def full_analysis(sand_type, raw_data_filename, labelled_data_filename=None, threshold=None, blur=None):
    raw_data = sand_atlas.io.load_data(raw_data_filename)

    if labelled_data_filename == None:
        labelled_data_filename = "labelled_data.tif"

    properties_filename = "properties.csv"

    if os.path.exists(labelled_data_filename):
        labelled_data = sand_atlas.io.load_data(labelled_data_filename)
    else:
        binary_data = threshold(raw_data, threshold, blur)
        labelled_data = label_binary_data(binary_data)
        sand_atlas.io.save_data(labelled_data, "labelled_data.tif")

    if not os.path.exists(properties_filename):
        properties = get_particle_properties(labelled_data, raw_data)
        numpy.savetxt("summary.csv", properties, delimiter=",")

    sand_atlas.video.make_website_video(labelled_data_filename, sand_type)
    sand_atlas.video.make_instagram_videos(labelled_data_filename, sand_type)
