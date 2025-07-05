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


def image_entropy(volume, window_size=16, debug=False):
    """
    Compute local entropy variation in a 3D volume.
    High entropy variation indicates regions with different texture complexity,
    which can indicate noise or varying image quality across the volume.

    Parameters
    ----------
    volume : np.ndarray
        A 3D NumPy array representing the volume data.
    window_size : int
        Size of the local window for entropy calculation.
    debug : bool
        Whether to print debug information.

    Returns
    -------
    float
        Standard deviation of local entropy values across the volume.
        Higher values indicate inconsistent texture/noise distribution.
    """
    shape = volume.shape
    local_entropies = []

    # Calculate stride to ensure reasonable sampling
    stride = max(1, window_size // 2)

    for z in range(0, shape[0] - window_size + 1, stride):
        for y in range(0, shape[1] - window_size + 1, stride):
            for x in range(0, shape[2] - window_size + 1, stride):
                # Extract local window
                window = volume[z : z + window_size, y : y + window_size, x : x + window_size]

                # Skip windows with very low variance (uniform regions)
                if np.std(window) > 1e-10:
                    local_entropy = shannon_entropy(window)
                    local_entropies.append(local_entropy)

    if len(local_entropies) == 0:
        entropy_std = 0.0
    else:
        # Return standard deviation of local entropies
        entropy_std = np.std(local_entropies)

    if debug:
        global_entropy = shannon_entropy(volume)
        mean_local_entropy = np.mean(local_entropies) if local_entropies else 0
        print(f"Global Entropy = {global_entropy:.3f}")
        print(f"Mean Local Entropy = {mean_local_entropy:.3f}")
        print(f"Local Entropy Std = {entropy_std:.3f}")

    return entropy_std


def fft_peak_frequency(volume, debug=False):
    """
    Estimate the dominant spatial frequency in a 3D volume using FFT.
    Computes the radial frequency distribution and finds the peak.
    Returns normalized spatial frequency (0-1 scale).
    """
    # Compute 3D FFT and shift zero frequency to center
    fshift = fftshift(fftn(volume))
    power_spectrum = np.abs(fshift) ** 2

    # Get volume center and dimensions
    center = np.array(volume.shape) // 2
    max_radius = min(center)  # Maximum meaningful radius

    # Create coordinate grids
    z, y, x = np.ogrid[: volume.shape[0], : volume.shape[1], : volume.shape[2]]

    # Calculate radial distance from center
    radial_dist = np.sqrt((z - center[0]) ** 2 + (y - center[1]) ** 2 + (x - center[2]) ** 2)

    # Compute radial power spectrum
    radial_bins = np.arange(0, max_radius + 1)
    radial_power = np.zeros(len(radial_bins) - 1)

    for i in range(len(radial_bins) - 1):
        mask = (radial_dist >= radial_bins[i]) & (radial_dist < radial_bins[i + 1])
        if np.any(mask):
            radial_power[i] = np.mean(power_spectrum[mask])

    # Find peak frequency (excluding DC component at index 0)
    if len(radial_power) > 1:
        peak_idx = np.argmax(radial_power[1:]) + 1
        normalized_frequency = peak_idx / len(radial_power)
    else:
        normalized_frequency = 0.0

    if debug:
        print(f"FFT Peak Frequency = {normalized_frequency}")
    return normalized_frequency


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
    Compute a normalized gradient quality metric.
    Combines gradient standard deviation with signal-to-noise characteristics
    to produce a scale-invariant quality measure.

    Returns
    -------
    float
        Normalized gradient quality metric. LOWER values indicate better edge definition.
        Inverted so that good quality = low values, consistent with error-like metrics.
    """
    gx = ndimage.sobel(volume, axis=0)
    gy = ndimage.sobel(volume, axis=1)
    gz = ndimage.sobel(volume, axis=2)
    gmag = np.sqrt(gx**2 + gy**2 + gz**2)

    grad_std = np.std(gmag)
    grad_mean = np.mean(gmag)

    # Normalized metrics
    coefficient_of_variation = grad_std / grad_mean if grad_mean > 0 else 0
    p25 = np.percentile(gmag, 25)  # Noise level estimate
    p90 = np.percentile(gmag, 90)  # Signal level estimate
    gradient_snr = p90 / p25 if p25 > 0 else 0

    # Combined normalized metric: geometric mean of CoV and normalized SNR
    # SNR is log-scaled and normalized to similar range as CoV
    normalized_snr = np.log10(gradient_snr + 1)  # +1 to avoid log(0)
    gradient_quality_metric = np.sqrt(coefficient_of_variation * normalized_snr)

    # Invert so that LOWER values = BETTER quality
    # Add small constant to avoid division issues, then invert
    inverted_metric = 1.0 / (gradient_quality_metric + 0.1)

    if debug:
        print(f"Gradient Std = {grad_std:.3f}")
        print(f"CoV = {coefficient_of_variation:.3f}")
        print(f"Gradient SNR = {gradient_snr:.3f}")
        print(f"Normalized SNR = {normalized_snr:.3f}")
        print(f"Raw Quality Metric = {gradient_quality_metric:.3f}")
        print(f"Inverted Metric (lower=better) = {inverted_metric:.3f}")

    return inverted_metric


def otsu_solid_fraction(volume, debug=False, maskValue=0):
    """
    Estimate solid fraction from a global Otsu threshold applied to a 3D volume. High solid fraction means not many voids, lots of particles in contact and difficult to segment.
    """
    # Otsu is computed ignoring the mask
    thresh = filters.threshold_otsu(volume[volume!=maskValue])
    binary = volume > thresh
    solid_fraction = np.sum(binary) / volume.size
    if debug:
        print(f"Otsu Solid Fraction = {solid_fraction}")
    return solid_fraction


def local_solid_fraction_std(volume, window_size=10, debug=False):
    """
    Estimate spatial homogeneity by computing standard deviation of solid fraction in subvolumes.
    """
    shape = volume.shape
    solid_fractions = []

    for z in range(0, shape[0] - window_size + 1, window_size):
        for y in range(0, shape[1] - window_size + 1, window_size):
            for x in range(0, shape[2] - window_size + 1, window_size):
                subvol = volume[z : z + window_size, y : y + window_size, x : x + window_size]
                solid_fractions.append(otsu_solid_fraction(subvol))
    local_std = np.std(solid_fractions)
    if debug:
        print(f"Local Solid Fraction Std = {local_std}")
    return local_std
