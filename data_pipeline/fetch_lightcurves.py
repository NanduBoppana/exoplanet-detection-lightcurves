"""
Fetches stellar light curve data for a given target using Lightkurve.
"""

import lightkurve as lk
import argparse
import os

def fetch_light_curve(target_name: str, mission: str = "Kepler", output_dir: str = "data"):
    """
    Search for and download light curve data for a given target star.

    Args:
        target_name: Name or ID of the target star (e.g., "Kepler-10")
        mission: Mission to search (Kepler, K2, TESS)
        output_dir: Directory to save the downloaded FITS file

    Returns:
        lk.LightCurve object, or None if no data found
    """
    print(f"Searching for light curve data: {target_name} ({mission})...")

    search_result = lk.search_lightcurve(target_name, mission=mission)

    if len(search_result) == 0:
        print(f"No light curve data found for {target_name}.")
        return None

    print(f"Found {len(search_result)} result(s). Downloading first entry...")
    lc = search_result[0].download()

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{target_name.replace(' ', '_')}_lc.fits")
    lc.to_fits(save_path, overwrite=True)
    print(f"Saved light curve to {save_path}")

    return lc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch stellar light curve data")
    parser.add_argument("--target", type=str, default="Kepler-10", help="Target star name")
    parser.add_argument("--mission", type=str, default="Kepler", help="Mission (Kepler, K2, TESS)")
    args = parser.parse_args()

    fetch_light_curve(args.target, args.mission)
