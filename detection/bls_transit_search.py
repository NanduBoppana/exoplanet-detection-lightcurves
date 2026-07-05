"""
Detects candidate exoplanet transit signals in a light curve using
the Box Least Squares (BLS) algorithm.
"""

import lightkurve as lk
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import os

# Allow importing detrend_light_curve from data_pipeline
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data_pipeline"))
from detrend import detrend_light_curve


def run_bls_search(fits_path: str, min_period: float = 0.5, max_period: float = 20.0,
                    plot: bool = True):
    """
    Runs a Box Least Squares search on a light curve to detect
    candidate periodic transit signals.

    Args:
        fits_path: Path to the input FITS file
        min_period: Minimum orbital period to search (days)
        max_period: Maximum orbital period to search (days)
        plot: Whether to display the BLS periodogram and folded light curve

    Returns:
        dict with best period, transit time, duration, and depth
    """
    print(f"Loading and detrending light curve from {fits_path}...")
    flat_lc = detrend_light_curve(fits_path, plot=False)

    print(f"Running BLS search (period range: {min_period}-{max_period} days)...")
    periodogram = flat_lc.to_periodogram(
        method="bls",
        period=np.linspace(min_period, max_period, 10000),
        frequency_factor=500,
    )

    best_period = periodogram.period_at_max_power
    best_t0 = periodogram.transit_time_at_max_power
    best_duration = periodogram.duration_at_max_power

    print(f"Best period found: {best_period:.4f} days")
    print(f"Transit epoch (t0): {best_t0}")
    print(f"Transit duration: {best_duration}")

    folded_lc = flat_lc.fold(period=best_period, epoch_time=best_t0)

    if plot:
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        periodogram.plot(ax=axes[0])
        axes[0].axvline(best_period.value, color="red", linestyle="--", alpha=0.6,
                         label=f"Best period: {best_period:.2f}d")
        axes[0].set_title("BLS Periodogram")
        axes[0].legend()

        folded_lc.scatter(ax=axes[1], s=2)
        axes[1].set_title(f"Phase-Folded Light Curve (Period = {best_period:.4f}d)")

        plt.tight_layout()
        fig.savefig("bls_output.png", dpi=150)
        plt.show()

    return {
        "period": best_period,
        "t0": best_t0,
        "duration": best_duration,
        "periodogram": periodogram,
        "folded_lc": folded_lc,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BLS transit search on a light curve")
    parser.add_argument("--input", type=str, required=True, help="Path to input FITS file")
    parser.add_argument("--min_period", type=float, default=0.5, help="Minimum period to search (days)")
    parser.add_argument("--max_period", type=float, default=20.0, help="Maximum period to search (days)")
    args = parser.parse_args()

    run_bls_search(args.input, args.min_period, args.max_period)