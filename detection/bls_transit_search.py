"""
BLS Transit Search
------------------
Runs Box Least Squares (BLS) period detection on a detrended TESS light curve
to find candidate transit events.

Usage:
    python detection/bls_transit_search.py
"""

import numpy as np
import matplotlib.pyplot as plt
import lightkurve as lk
from astropy.timeseries import BoxLeastSquares
from astropy import units as u
import warnings
warnings.filterwarnings("ignore")


# ── CONFIG ────────────────────────────────────────────────────────────────────
# Known exoplanet host stars (good for testing — real answers are published)
TARGET_ID   = "TIC 261136679"   # TESS ID for a known exoplanet host
SECTOR      = None               # None = auto-pick first available sector
MISSION     = "TESS"

# BLS search parameters
MIN_PERIOD  = 0.5   # days (minimum orbital period to search)
MAX_PERIOD  = 20.0  # days (maximum orbital period to search)
DURATION_GRID = np.array([0.05, 0.1, 0.15, 0.2, 0.3])  # transit durations to try (in days)

# Output
SAVE_PLOT   = True
PLOT_PATH   = "bls_output.png"
# ─────────────────────────────────────────────────────────────────────────────


def fetch_and_detrend(target_id, sector=None, mission="TESS"):
    """Fetch light curve and apply spline-based detrending."""
    print(f"Fetching light curve for {target_id}...")

    search = lk.search_lightcurve(target_id, mission=mission, sector=sector)
    if len(search) == 0:
        raise ValueError(f"No light curves found for {target_id}")

    # Download the first available result
    lc = search[0].download()
    print(f"  Downloaded sector {lc.sector}, {len(lc)} data points")

    # Remove NaNs and normalize
    lc = lc.remove_nans().normalize()

    # Detrend: flatten removes long-term stellar variability
    lc_flat = lc.flatten(window_length=401)

    # Sigma clip: remove outliers beyond 3 sigma
    lc_clean = lc_flat.remove_outliers(sigma=3.0)

    print(f"  After cleaning: {len(lc_clean)} data points remain")
    return lc_clean


def run_bls(lc, min_period, max_period, duration_grid):
    """Run BLS period search on a cleaned light curve."""
    print(f"\nRunning BLS search (period range: {min_period}–{max_period} days)...")

    time  = lc.time.value
    flux  = lc.flux.value
    flux_err = lc.flux_err.value if lc.flux_err is not None else np.ones_like(flux) * 0.001

    # Build period grid (log-spaced for better coverage)
    periods = np.exp(np.linspace(np.log(min_period), np.log(max_period), 5000))

    # Run BLS
    bls = BoxLeastSquares(time * u.day, flux, dy=flux_err)
    result = bls.power(periods * u.day, duration_grid * u.day, method="fast")

    # Find best period
    best_idx    = np.argmax(result.power)
    best_period = result.period[best_idx].value
    best_t0     = result.transit_time[best_idx].value
    best_dur    = result.duration[best_idx].value
    best_depth  = result.depth[best_idx]
    best_power  = result.power[best_idx]

    # Signal-to-noise estimate: depth / scatter of out-of-transit flux
    snr = best_depth / np.std(flux)

    print(f"\n  ✓ Best period    : {best_period:.4f} days")
    print(f"  ✓ Transit epoch  : {best_t0:.4f} BTJD")
    print(f"  ✓ Duration       : {best_dur * 24:.2f} hours")
    print(f"  ✓ Depth          : {best_depth * 100:.4f}%  ({best_depth * 1e6:.1f} ppm)")
    print(f"  ✓ BLS power      : {best_power:.4f}")
    print(f"  ✓ Estimated SNR  : {snr:.2f}")

    return result, {
        "period":  best_period,
        "t0":      best_t0,
        "duration": best_dur,
        "depth":   best_depth,
        "power":   best_power,
        "snr":     snr,
    }


def plot_results(lc, bls_result, best_params, save_path=None):
    """Plot: raw light curve, BLS periodogram, phase-folded transit."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle(f"BLS Transit Search Results\nBest Period: {best_params['period']:.4f} days  |  SNR: {best_params['snr']:.2f}", fontsize=13)

    # ── Panel 1: Cleaned light curve ─────────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(lc.time.value, lc.flux.value, "k.", ms=1.5, alpha=0.5)
    ax1.set_xlabel("Time (BTJD days)")
    ax1.set_ylabel("Normalized Flux")
    ax1.set_title("Cleaned Light Curve")

    # ── Panel 2: BLS Periodogram ─────────────────────────────────────────────
    ax2 = axes[1]
    ax2.plot(bls_result.period.value, bls_result.power, "b-", lw=0.8, alpha=0.7)
    ax2.axvline(best_params["period"], color="red", lw=1.5, ls="--", label=f"Best period: {best_params['period']:.4f} d")
    ax2.set_xlabel("Period (days)")
    ax2.set_ylabel("BLS Power")
    ax2.set_title("BLS Periodogram")
    ax2.legend()

    # ── Panel 3: Phase-folded light curve ────────────────────────────────────
    ax3 = axes[2]
    lc_folded = lc.fold(period=best_params["period"], epoch_time=best_params["t0"])
    lc_binned = lc_folded.bin(time_bin_size=0.01)

    ax3.plot(lc_folded.phase.value, lc_folded.flux.value, "k.", ms=1.5, alpha=0.3, label="Data")
    ax3.plot(lc_binned.phase.value, lc_binned.flux.value, "r-", lw=2, label="Binned")
    ax3.set_xlabel("Phase")
    ax3.set_ylabel("Normalized Flux")
    ax3.set_title(f"Phase-folded Transit  (depth: {best_params['depth']*100:.4f}%,  duration: {best_params['duration']*24:.2f} hr)")
    ax3.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n  Plot saved to {save_path}")

    plt.show()
    return fig


def assess_candidate(best_params):
    """Simple rule-based significance check on the detected candidate."""
    print("\n── Candidate Assessment ─────────────────────────────────────────")

    flags = []

    if best_params["snr"] >= 7.1:
        flags.append(("✓", "SNR above 7.1 threshold (standard transit detection criterion)"))
    else:
        flags.append(("✗", f"SNR = {best_params['snr']:.2f} below 7.1 threshold — likely noise"))

    if best_params["depth"] > 0.0001:  # > 100 ppm
        flags.append(("✓", f"Depth {best_params['depth']*1e6:.0f} ppm — detectable transit"))
    else:
        flags.append(("✗", "Depth < 100 ppm — possibly noise"))

    if 0.5 <= best_params["period"] <= 20:
        flags.append(("✓", f"Period {best_params['period']:.3f} d — plausible orbital range"))

    if best_params["duration"] * 24 <= 0.6 * 24 * (best_params["period"] / (2 * np.pi)) ** (1/3):
        flags.append(("✓", "Duration consistent with planetary transit geometry"))

    for icon, msg in flags:
        print(f"  {icon} {msg}")

    passed = sum(1 for icon, _ in flags if icon == "✓")
    verdict = "CANDIDATE" if passed >= 3 else "LIKELY FALSE POSITIVE"
    print(f"\n  Verdict: {verdict} ({passed}/{len(flags)} checks passed)")
    return verdict


def main():
    # 1. Fetch and detrend
    lc = fetch_and_detrend(TARGET_ID, sector=SECTOR, mission=MISSION)

    # 2. Run BLS
    bls_result, best_params = run_bls(lc, MIN_PERIOD, MAX_PERIOD, DURATION_GRID)

    # 3. Assess candidate
    verdict = assess_candidate(best_params)

    # 4. Plot
    plot_results(lc, bls_result, best_params, save_path=PLOT_PATH if SAVE_PLOT else None)

    # 5. Return params for downstream use (classification step)
    return {**best_params, "verdict": verdict}


if __name__ == "__main__":
    params = main()