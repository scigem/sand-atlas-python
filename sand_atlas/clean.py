import sys, numpy, os, tifffile
import scipy.ndimage
import spam.label
import multiprocessing
from tqdm import tqdm
import sand_atlas.io


def clean_subvolume(label, labelRegion, z_slice, y_slice, x_slice):
    """
    Cleans a subvolume by retaining only the largest connected component associated with a given label.

    Parameters:
    - label (int): The label of the region being processed.
    - labelRegion (numpy.ndarray): A subvolume of the labeled image corresponding to the bounding box.
    - z_slice, y_slice, x_slice (slice): The slices defining the bounding box of the region in the 3D image.

    Returns:
    - tuple: Contains the original label, cleaned subvolume (with only the largest component), and the slices.
             Returns None if the input region is invalid or empty.
    """
    if labelRegion is None or labelRegion.size == 0:
        return None  # Skip invalid or empty regions

    # Ensure the labelRegion is a binary mask where the label matches the input label
    labelRegion_binary = labelRegion == label

    # Label connected components in the binary region
    re_labelRegion, _ = scipy.ndimage.label(labelRegion_binary)
    # tifffile.imwrite(f'check_{label}.tif', re_labelRegion.astype(labelRegion.dtype))

    # Calculate the volume (number of voxels) for each connected component
    list_of_subLabels = numpy.unique(re_labelRegion)  # List of unique sublabel IDs
    sublabel_volumes = {
        subLabel: numpy.sum(re_labelRegion == subLabel) for subLabel in list_of_subLabels if subLabel != 0
    }

    if not sublabel_volumes:
        return None  # No valid sublabels found

    # Identify the sublabel with the largest volume
    largest_sublabel = max(sublabel_volumes, key=sublabel_volumes.get)

    # Retain only the largest sublabel in the cleaned region (binary array)
    labelRegion_clean_binary = re_labelRegion == largest_sublabel

    return label, labelRegion_clean_binary, z_slice, y_slice, x_slice


def parallel_clean_subvolumes(lab, bounding_boxes, num_processors, removeEdgeLabels=True, makeLabelsSequential=True):
    """
    Cleans subvolumes of a labeled 3D image in parallel by retaining only the largest connected component for each label.

    Parameters:
    - lab (numpy.ndarray): The labeled 3D image to process.
    - bounding_boxes (list of tuple of slices): List of slices defining bounding boxes for each labeled region.
    - num_processors (int): Number of processors to use for parallel processing.
    - removeEdgeLabels (bool): If True, removes labels that touch the edges of the volume.
    - makeLabelsSequential (bool): If True, ensures the labels are sequential starting from 1.

    Returns:
    - numpy.ndarray: A cleaned labeled 3D image where only the largest component for each label is retained.
    """

    # Remove labels that touch the edges of the volume, if specified
    if removeEdgeLabels:
        print(f"\tRemoving labels that touch the edges...", end="")
        labels_on_edge = spam.label.labelsOnEdges(lab)  # Identify labels on the edges
        lab = spam.label.removeLabels(lab, labels_on_edge)  # Remove edge labels
        print("Done.")

    # Ensure that the labels are sequential starting from 1, if specified
    if makeLabelsSequential:
        print(f"\tMaking labels sequential...", end="")
        lab = spam.label.makeLabelsSequential(lab)
        print("Done.")

    # Initialize a new labeled image with the same shape as the input
    new_lab = numpy.zeros_like(lab, dtype=lab.dtype)

    # Get a list of all unique labels in the image, excluding 0 (background)
    list_of_labels = numpy.unique(lab)
    list_of_labels = list_of_labels[list_of_labels != 0]

    print(f"Starting parallel processing with {num_processors} processors.")
    # Prepare tasks for parallel processing
    tasks = []
    for label in tqdm(list_of_labels, desc="\tIsolating subvolumes"):
        # Find the bounding box slices for the current label
        z_slice, y_slice, x_slice = bounding_boxes[label - 1]
        # Extract the subvolume corresponding to the bounding box
        labelRegion = lab[z_slice, y_slice, x_slice]
        # Add the task for this label and its subvolume
        tasks.append([label, labelRegion, z_slice, y_slice, x_slice])

    # Use multiprocessing to process the tasks in parallel
    with multiprocessing.Pool(num_processors) as pool:
        # Execute the tasks in parallel and track progress with a progress bar
        pool_results = pool.starmap(clean_subvolume, tqdm(tasks, desc="\tProcessing subvolumes"))

    # Filter out invalid results (e.g., None) from the processing results
    pool_results = [result for result in pool_results if result is not None]

    # Update the cleaned labeled image with the results
    for label, labelRegion_return, z_slice, y_slice, x_slice in tqdm(
        pool_results, desc="\tRe-assembling labelled array"
    ):
        # Merge the cleaned subvolume back into the main volume
        new_lab[z_slice, y_slice, x_slice] = numpy.where(labelRegion_return, label, new_lab[z_slice, y_slice, x_slice])

    # Return the cleaned labeled image
    return new_lab


def clean_labels(file_path, num_processors=4, verbosity=0):
    # Step 1: Get the file path and the number of processors from command line arguments
    # file_path = sys.argv[1]  # First command-line argument: path to the labeled 3D image file
    # num_processors = int(input("\nEnter the number of processors to be used for parallel processing: "))

    # Step 2: Load the already labeled 3D tif file
    if verbosity > 0:
        print(f"\nLoading labelled 3D image from {file_path}... ", end="")
    lab = sand_atlas.io.load_data(file_path)  # Load the labeled 3D image from the specified file
    if verbosity > 0:
        print(f"Done. \n\tShape of the labelled image: {lab.shape}")

    # Step 3: Compute the bounding boxes for the labeled components using scipy
    if verbosity > 0:
        print("Computing bounding boxes for labelled components... ", end="")
    bounding_boxes = scipy.ndimage.find_objects(lab)  # Get bounding box slices for each labeled region
    if verbosity > 0:
        print(f"Done.\n\tFound {len(bounding_boxes)} bounding boxes.")

    # Step 4: Run the parallel processing
    clean_lab = parallel_clean_subvolumes(lab, bounding_boxes, num_processors)  # Clean the labeled image in parallel
    del lab  # Free memory by deleting the original labeled image

    # Step 5: Save the cleaned labels back to a new tif file
    # Generate the output file path based on the input file name
    output_file_path = os.path.join(
        os.path.dirname(file_path),  # Get the directory of the input file
        f"{os.path.splitext(os.path.basename(file_path))[0]}_clean.tif",  # Add "_clean" to the file name
    )
    if verbosity > 0:
        print(f"Saving cleaned 3D labels...", end="")

    # Determine the appropriate data type for the output file based on the maximum label value
    if clean_lab.max() < 256:
        data_type = numpy.uint8  # Use 8-bit unsigned integer for small label ranges
    elif clean_lab.max() < 65536:
        data_type = numpy.uint16  # Use 16-bit unsigned integer for moderate label ranges
    else:
        data_type = numpy.uint32  # Use 32-bit unsigned integer for large label ranges

    # Save the cleaned labeled image to the output file
    tifffile.imwrite(output_file_path, clean_lab.astype(data_type))
    if verbosity > 0:
        print(f"Done.\n\tCleaned image saved as {output_file_path}")
