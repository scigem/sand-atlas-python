import argparse
import concurrent.futures
from colorama import Fore, Style
from tqdm import tqdm
import sand_atlas.pipeline
import sand_atlas.quality


# Thresholds: (green, yellow)
thresholds = {
    "Global SNR": (3, 5),
    "Modality Index": (1.5, 2.5),
    "Entropy": (6, 7.5),
    "FFT Peak Frequency": (0.05, 0.2),  # interpret loosely
    "Autocorrelation Range": (3, 8),
    "Edge Density": (0.2, 0.4),
    "Fractal Dimension": (1.5, 1.7),
    "Gradient Std": (10, 25),
    "Otsu Porosity": (0.2, 0.5),
    "Slice-to-slice Variation": (2, 5),
}

# Suggestions
suggestions = {
    "Global SNR": "Increase X-ray current or reduce scan speed",
    "Modality Index": "Histogram not clearly bimodal; improve contrast or material difference",
    "Entropy": "Too much texture/noise; reduce scan noise or smooth sample",
    "FFT Peak Frequency": "Low frequency = large features; consider voxel size",
    "Autocorrelation Range": "Low range = noisy image; try filtering",
    "Edge Density": "High density suggests complex shapes or oversegmentation risk",
    "Fractal Dimension": "High = rough shapes; may impact watershed algorithms",
    "Gradient Std": "High variation may cause thresholding instability",
    "Otsu Porosity": "Very low or very high values may indicate poor separability",
    "Slice-to-slice Variation": "Non-uniform sample or scan instability",
}


def rate(value, thresholds):
    low, high = thresholds
    if value < low:
        return "🟢", "Good"
    elif value < high:
        return "🟡", "Moderate"
    else:
        return "🔴", "Poor"


def print_report(metrics):
    print("\n" + "=" * 40)
    print(f"{Style.BRIGHT}CT Scan Quality Report Card{Style.RESET_ALL}")
    print("=" * 40)

    issues = []

    for category, items in metrics.items():
        print(f"\n{Style.BRIGHT}{category}:{Style.RESET_ALL}")
        for name, value in items.items():
            symbol, rating = rate(value, thresholds[name])
            color = {"Good": Fore.GREEN, "Moderate": Fore.YELLOW, "Poor": Fore.RED}[rating]
            print(f"  {symbol} {name:<25}: {value:.2f}   {color}{rating}{Style.RESET_ALL}")
            if rating == "Poor":
                issues.append(f"- {name}: {suggestions[name]}")

    print("\n" + "-" * 40)
    if issues:
        print(f"{Fore.RED}{Style.BRIGHT}Potential Issues Detected:{Style.RESET_ALL}")
        for issue in issues:
            print(f"  {Fore.RED}{issue}{Style.RESET_ALL}")
    else:
        print(f"{Fore.GREEN}No major issues detected. Scan quality likely suitable for segmentation.{Style.RESET_ALL}")
    print("=" * 40)


def compute_all_metrics(volume, debug=False):
    """
    Compute a full suite of segmentation-relevant quality metrics from a 3D CT volume,
    showing progress with tqdm.

    Parameters:
        volume (ndarray): 3D NumPy array (should be float in [0, 1] range)

    Returns:
        dict: Nested dictionary grouping computed metrics by category
    """
    metric_funcs = [
        ("Contrast and Noise", "Global SNR", sand_atlas.quality.global_snr),
        # ("Contrast and Noise", "Modality Index", sand_atlas.quality.modality_index),
        ("Contrast and Noise", "Entropy", sand_atlas.quality.image_entropy),
        ("Structure and Texture", "FFT Peak Frequency", sand_atlas.quality.fft_peak_frequency),
        ("Structure and Texture", "Autocorrelation Range", sand_atlas.quality.autocorrelation_range),
        ("Edge and Shape Complexity", "Edge Density", sand_atlas.quality.edge_density),
        ("Edge and Shape Complexity", "Fractal Dimension", sand_atlas.quality.fractal_dimension),
        ("Edge and Shape Complexity", "Gradient Std", sand_atlas.quality.gradient_std),
        ("3D Homogeneity", "Otsu Porosity", sand_atlas.quality.otsu_porosity),
        ("3D Homogeneity", "Slice-to-slice Variation", sand_atlas.quality.local_porosity_std),
    ]

    results = {
        "Contrast and Noise": {},
        "Structure and Texture": {},
        "Edge and Shape Complexity": {},
        "3D Homogeneity": {},
    }

    def run_metric(args):
        category, name, func = args
        return (category, name, func(volume, debug=debug))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(run_metric, args) for args in metric_funcs]
        for f in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures), desc="Computing metrics", unit="metric"
        ):
            category, name, value = f.result()
            results[category][name] = value

    return results


def preflight_script():
    parser = argparse.ArgumentParser(
        description="Perform a preflight analysis of a 3D data set to understand its quality."
    )
    parser.add_argument("--raw", type=str, help="The path to the file containing the raw data.", default=None)
    parser.add_argument("--binning", type=int, help="The binning factor to use.", default=None)
    # parser.add_argument("--output", type=str, help="The path to the output file.", default="summary.csv")

    args = parser.parse_args()

    data = sand_atlas.io.load_data(args.raw)

    if args.binning is not None:
        data = sand_atlas.pipeline.bin_data(data, args.binning)

    metrics = compute_all_metrics(volume=data, debug=False)
    print_report(metrics)
