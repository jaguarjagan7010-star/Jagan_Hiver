"""
download_data.py — Downloads the Customer Support on Twitter dataset.

Two modes:
  1. Automatic: uses the Kaggle API (requires ~/.kaggle/kaggle.json)
  2. Manual:    prints instructions if credentials are missing

Usage:
    python scripts/download_data.py
"""

import os
import sys
import zipfile
from pathlib import Path

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import DATA_RAW

KAGGLE_DATASET = "thoughtvector/customer-support-on-twitter"


def kaggle_credentials_exist() -> bool:
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    # Also check environment variables (set in .env)
    has_env = os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY")
    return kaggle_json.exists() or bool(has_env)


def download_via_api():
    """Download using the kaggle Python package."""
    import kaggle  # noqa: imported here so missing install gives a clear error
    print(f"Downloading dataset: {KAGGLE_DATASET}")
    print(f"Destination: {DATA_RAW}")
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        KAGGLE_DATASET,
        path=str(DATA_RAW),
        unzip=True,
        quiet=False,
    )
    print("Download complete.")


def list_downloaded_files():
    files = list(DATA_RAW.glob("*"))
    if files:
        print("\nFiles in data/raw/:")
        for f in files:
            size_mb = f.stat().st_size / 1_048_576
            print(f"  {f.name}  ({size_mb:.1f} MB)")
    else:
        print("data/raw/ is still empty.")


def print_manual_instructions():
    print("\n" + "=" * 60)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("=" * 60)
    print("No Kaggle credentials found.")
    print()
    print("Option A — Manual browser download:")
    print("  1. Go to: https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter")
    print("  2. Click the Download button (top right)")
    print("  3. Unzip the downloaded file")
    print(f"  4. Copy the CSV file(s) into:  {DATA_RAW}")
    print("  5. Re-run:  python scripts/explore_data.py")
    print()
    print("Option B — Kaggle API (faster for future use):")
    print("  1. Go to https://www.kaggle.com/account")
    print("  2. Scroll to 'API' section -> click 'Create New Token'")
    print("  3. This downloads kaggle.json")
    print(f"  4. Move it to:  {Path.home() / '.kaggle' / 'kaggle.json'}")
    print("  5. Re-run this script")
    print("=" * 60)


def main():
    DATA_RAW.mkdir(parents=True, exist_ok=True)

    # Check if data already exists
    existing_csvs = list(DATA_RAW.glob("*.csv"))
    if existing_csvs:
        print("Dataset already present:")
        list_downloaded_files()
        return

    if kaggle_credentials_exist():
        try:
            download_via_api()
            list_downloaded_files()
        except Exception as e:
            print(f"Kaggle API download failed: {e}")
            print_manual_instructions()
    else:
        print_manual_instructions()


if __name__ == "__main__":
    main()
