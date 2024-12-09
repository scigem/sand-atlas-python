import numpy
import spam.label
import skimage.measure
import scipy.ndimage
import scipy.spatial
import progressbar
import multiprocessing

# try:
#     multiprocessing.set_start_method("fork")
# except RuntimeError:
#     pass

# Global number of processes
nProcessesDefault = multiprocessing.cpu_count()

labelType = "<u4"


def computeConvexVolume(args):
    label, lab, boundingBoxes, centresOfMass = args

    # Extract subvolume for this label
    labelI = spam.label.getLabel(lab, label, boundingBoxes=boundingBoxes, centresOfMass=centresOfMass)
    subvol = labelI["subvol"]
    points = numpy.transpose(numpy.where(subvol))

    # If fewer than 4 points, we cannot form a 3D convex hull
    if points.shape[0] < 4:
        return label, 0

    try:
        hull = scipy.spatial.ConvexHull(points)
        # Use the hull vertices for a Delaunay triangulation
        deln = scipy.spatial.Delaunay(points[hull.vertices])

        # Create a coordinate grid covering the subvolume
        # coords shape: (N, 3), where N = Z*Y*X
        coords = numpy.indices(subvol.shape).transpose(1, 2, 3, 0).reshape(-1, 3)

        # Find which points are inside the hull
        simplex_ids = deln.find_simplex(coords)
        simplex_ids = simplex_ids.reshape(subvol.shape)

        # Inside points have simplex_ids >= 0
        hullIm = numpy.zeros(subvol.shape, dtype=numpy.uint8)
        hullIm[simplex_ids >= 0] = 1

        # Compute the volume of the hull
        # volumes(hullIm) returns an array of length hullIm.max()+1, i.e. 2 elements (0 and 1)
        hullVol = spam.label.volumes(hullIm)

        # hullVol[0] = background volume, hullVol[1] = hull volume
        return label, hullVol[1] if len(hullVol) > 1 else 0

    except Exception:
        # If convex hull or Delaunay fails, return 0
        return label, 0


def convexVolume(
    lab,
    boundingBoxes=None,
    centresOfMass=None,
    volumes=None,
    nProcesses=nProcessesDefault,
    verbose=True,
):
    """
    This function computes the convex hull of each label of the labeled image and returns an array
    with the convex volume of each particle.

    Parameters
    ----------
        lab : 3D array of integers
            Labeled volume, with lab.max() labels.

        boundingBoxes : array, optional
            Bounding boxes as returned by `boundingBoxes(lab)`.
            If not defined (Default = None), it is recomputed by running `boundingBoxes(lab)`.

        centresOfMass : array, optional
            Centers of mass as returned by `centresOfMass(lab)`.
            If not defined (Default = None), it is recomputed by running `centresOfMass(lab)`.

        volumes : array, optional
            Volumes as returned by `volumes(lab)`.
            If not defined (Default = None), it is recomputed by running `volumes(lab)`.

        nProcesses : int, optional
            Number of processes for multiprocessing.
            Default = number of CPUs in the system.

        verbose : bool, optional
            True for printing the evolution of the process.
            Default = True.

    Returns
    -------
        convexVolume : array of floats
            An array of length `lab.max() + 1` with the convex volume of each label.
            The background (label 0) will have a volume of 0.

    Note
    ----
    convexVolume can only be computed for particles with volume greater than 3 voxels.
    If this is not the case, it returns 0.
    """

    lab = lab.astype(labelType)

    # Compute boundingBoxes if needed
    if boundingBoxes is None:
        boundingBoxes = spam.label.boundingBoxes(lab)
    # Compute centresOfMass if needed
    if centresOfMass is None:
        centresOfMass = spam.label.centresOfMass(lab)
    # Compute volumes if needed
    if volumes is None:
        volumes = spam.label.volumes(lab)

    nLabels = lab.max()
    convexVolume = numpy.zeros(nLabels + 1, dtype="float")

    if verbose:
        widgets = [
            progressbar.FormatLabel(" "),
            " ",
            progressbar.Bar(),
            "",
            progressbar.AdaptiveETA(),
        ]
        pbar = progressbar.ProgressBar(widgets=widgets, maxval=nLabels)
        pbar.start()
        finishedNodes = 0

    tasks = [(label, lab, boundingBoxes, centresOfMass) for label in range(1, nLabels + 1)]

    # Run multiprocessing
    with multiprocessing.Pool(processes=nProcesses) as pool:
        for returns in pool.imap_unordered(computeConvexVolume, tasks):
            if verbose:
                finishedNodes += 1
                pbar.update(finishedNodes)
            convexVolume[returns[0]] = returns[1]
        pool.close()
        pool.join()

    if verbose:
        pbar.finish()

    return convexVolume


def compactness(lab, volumes=None, boundingBoxes=None, centresOfMass=None, gaussianFilterSigma=0.75, minVol=256):
    """
    Calculate the compactness of labeled particles in a 3D image. Compactness is defined as
    C = 36 * pi * V^2 / A^3, where V is the volume of the particle and A is the surface area. For a sphere,
    the compactness is 1.0. For other shapes, the compactness is less than 1.0. A value of 0.0 indicates an
    infinitely thin particle.

    Parameters:
    lab (numpy.ndarray): Labeled 3D image where each particle has a unique label.
    boundingBoxes (list, optional): List of bounding boxes for each label. If None, they will be computed.
    centresOfMass (list, optional): List of centers of mass for each label. If None, they will be computed.
    gaussianFilterSigma (float, optional): Sigma value for the Gaussian filter applied to the extracted grain. Default is 0.75.
    minVol (int, optional): Minimum volume threshold for considering a label. Default is 256.

    Returns:
    numpy.ndarray: Array of compactness values for each label.
    """

    if boundingBoxes is None:
        boundingBoxes = spam.label.boundingBoxes(lab)
    if centresOfMass is None:
        centresOfMass = spam.label.centresOfMass(lab, boundingBoxes=boundingBoxes, minVol=minVol)
    if volumes is None:
        volumes = spam.label.volumes(lab)

    compactness = numpy.zeros((lab.max() + 1), dtype="<f4")

    for label in range(1, lab.max() + 1):
        if not (centresOfMass[label] == numpy.array([0.0, 0.0, 0.0])).all():
            # Extract grain
            GL = spam.label.getLabel(
                lab,
                label,
                boundingBoxes=boundingBoxes,
                centresOfMass=centresOfMass,
                extractCube=True,
                margin=2,
                maskOtherLabels=True,
            )
            # Gaussian smooth
            grainCubeFiltered = scipy.ndimage.gaussian_filter(GL["subvol"].astype("<f4"), sigma=gaussianFilterSigma)
            # mesh edge
            grains, faces, _, _ = skimage.measure.marching_cubes(grainCubeFiltered, level=0.5)
            # compute surface
            surfaceArea = skimage.measure.mesh_surface_area(grains, faces)
            # compute psi
            compactness[label] = 36 * numpy.pi * volumes[label] ** 2 / surfaceArea**3

    return compactness