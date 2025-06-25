import numpy as np
from scipy import ndimage
from scipy.fft import fftn, fftshift
from scipy.signal import fftconvolve
from skimage import filters, feature
from skimage.util import img_as_float
from skimage.measure import shannon_entropy


def global_snr(volume, debug=False):
    """
    The global SNR is calculated as the mean of all voxel intensities divided by the standard deviation
    of all voxel intensities in the input volume.

    Parameters
    ----------
    volume : np.ndarray
        A 3D NumPy array representing the volume data.

    Returns
    -------
    float
        The global signal-to-noise ratio (SNR) of the input volume.
    """
    SNR = np.mean(volume) / np.std(volume)
    if debug:
        print(f"SNR = {SNR}")
    return SNR


def modality_index(volume, debug=False):
    """
    Estimate the modality index of the histogram of a 3D volume.
    Approximates phase separation quality using multi-Otsu thresholds.
    Optimized for speed by using histogram-based calculations.
    """
    # Compute thresholds using multi-Otsu on the raw volume
    thresholds = filters.threshold_multiotsu(volume, classes=3)
    print(f"Thresholds = {thresholds}")
    d = thresholds[1] - thresholds[0]

    # Assign voxels to phases
    phase_masks = [
        (volume >= volume.min()) & (volume < thresholds[0]),
        (volume >= thresholds[0]) & (volume < thresholds[1]),
        (volume >= thresholds[1]) & (volume <= volume.max()),
    ]

    # Compute stds for each phase
    stds = []
    for mask in phase_masks:
        vals = volume[mask]
        if vals.size == 0:
            stds.append(0)
            continue
        stds.append(np.std(vals))

    modality_index = d / (np.mean(stds) if np.mean(stds) > 0 else 1)
    if debug:
        print(f"Modality Index = {modality_index}")
    return modality_index


def image_entropy(volume, debug=False):
    """
    Compute the Shannon entropy of the voxel intensity distribution.
    """
    entropy = shannon_entropy(volume)
    if debug:
        print(f"Image Entropy = {entropy}")
    return entropy


def fft_peak_frequency(volume, debug=False):
    """
    Estimate the dominant spatial frequency in a 3D volume using FFT.
    Returns normalized spatial frequency (0–1 scale).
    """
    fshift = fftshift(fftn(volume))
    power_spectrum = np.abs(fshift) ** 2
    proj = np.mean(power_spectrum, axis=(0, 1))
    peak_idx = np.argmax(proj)
    entropy = peak_idx / len(proj)
    if debug:
        print(f"FFT Peak Frequency = {entropy}")
    return entropy


def autocorrelation_range(volume, debug=False):
    """
    Estimate spatial correlation length from the autocorrelation function of the volume.
    Returns the voxel lag where correlation drops below 0.5.
    """
    norm_volume = volume - np.mean(volume)
    corr = fftconvolve(norm_volume, norm_volume[::-1, ::-1, ::-1], mode="same")
    center = tuple(s // 2 for s in corr.shape)
    profile = corr[center[0], center[1], :]
    profile /= profile[center[2]]
    below = np.where(profile < 0.5)[0]
    spatial_correlation_length = below[0] if len(below) else len(profile)
    if debug:
        print(f"Autocorrelation Range = {spatial_correlation_length}")
    return spatial_correlation_length


def edge_density(volume, debug=False):
    """
    Compute the normalized count of edge-like voxels in a 3D volume.
    Uses gradient magnitude to approximate edges.
    """
    gx = ndimage.sobel(volume, axis=0)
    gy = ndimage.sobel(volume, axis=1)
    gz = ndimage.sobel(volume, axis=2)
    gmag = np.sqrt(gx**2 + gy**2 + gz**2)
    edges = gmag > np.percentile(gmag, 90)
    density = np.sum(edges) / volume.size
    if debug:
        print(f"Edge Density = {density}")
    return density


def fractal_dimension(volume, threshold=0.5, debug=False):
    """
    Estimate the 3D box-counting fractal dimension of a thresholded binary volume.
    """
    binary = volume > threshold
    sizes = 2 ** np.arange(1, int(np.log2(min(volume.shape))) - 1)
    counts = []

    for size in sizes:
        count = 0
        for z in range(0, volume.shape[0], size):
            for y in range(0, volume.shape[1], size):
                for x in range(0, volume.shape[2], size):
                    box = binary[z : z + size, y : y + size, x : x + size]
                    if np.any(box):
                        count += 1
        counts.append(count)

    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    if debug:
        print(f"Fractal Dimension = {-coeffs[0]}")
    return -coeffs[0]


def gradient_std(volume, debug=False):
    """
    Compute the standard deviation of 3D gradient magnitude.
    Captures edge sharpness and texture variation.
    """
    gx = ndimage.sobel(volume, axis=0)
    gy = ndimage.sobel(volume, axis=1)
    gz = ndimage.sobel(volume, axis=2)
    gmag = np.sqrt(gx**2 + gy**2 + gz**2)
    gradient_std = np.std(gmag)
    if debug:
        print(f"Gradient Std = {gradient_std}")
    return gradient_std


def otsu_porosity(volume, debug=False):
    """
    Estimate porosity from a global Otsu threshold applied to a 3D volume.
    """
    thresh = filters.threshold_otsu(volume)
    binary = volume < thresh
    porosity = np.sum(binary) / volume.size
    if debug:
        print(f"Otsu Porosity = {porosity}")
    return porosity


def local_porosity_std(volume, window_size=10, debug=False):
    """
    Estimate spatial homogeneity by computing standard deviation of porosity in subvolumes.
    """
    shape = volume.shape
    porosities = []

    for z in range(0, shape[0] - window_size + 1, window_size):
        for y in range(0, shape[1] - window_size + 1, window_size):
            for x in range(0, shape[2] - window_size + 1, window_size):
                subvol = volume[z : z + window_size, y : y + window_size, x : x + window_size]
                porosities.append(otsu_porosity(subvol))
    local_std = np.std(porosities)
    if debug:
        print(f"Local Porosity Std = {local_std}")
    return local_std
