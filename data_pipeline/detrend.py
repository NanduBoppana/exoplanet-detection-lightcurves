"""
Removes instrumental noise and stellar variability from a light curve
using flattening (detrending).
"""

import lightkurve as lk
import matplotlib.pyplot as plt
import argparse


def detrend_light_curve(fits_path: str, window_length: int = 401, plot: bool = True):
    """
    Loads a light curve FITS file, removes NaNs, and flattens it to
    remove long-term trends and stellar variability.

    Args:
        fits_path: Path to the input FITS file
        window_length: Window length for the Savitzky-Golay flattening filter
        plot: Whether to display before/after plots

    Returns:
        Flattened lk.LightCurve object
    """
    print(f"Loading light curve from {fits_path}...")
    lc = lk.read(fits_path)

    lc = lc.remove_nans().remove_outliers(sigma=5)

    print(f"Flattening light curve (window_length={window_length})...")
    flat_lc = lc.flatten(window_length=window_length)

    if plot:
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        lc.plot(ax=axes[0], label="Raw")
        flat_lc.plot(ax=axes[1], label="Detrended", color="orange")
        axes[0].set_title("Before Detrending")
        axes[1].set_title("After Detrending")
        plt.tight_layout()
        plt.show()

    return flat_lc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detrend a light curve FITS file")
    parser.add_argument("--input", type=str, required=True, help="Path to input FITS file")
    parser.add_argument("--window", type=int, default=401, help="Flattening window length")
    args = parser.parse_args()

    detrend_light_curve(args.input, args.window)
