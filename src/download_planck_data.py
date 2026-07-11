"""Download the official Planck 2018 PR3 unbinned TT power spectrum."""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path


PLANCK_TT_URL = (
    "https://irsa.ipac.caltech.edu/data/Planck/release_3/ancillary-data/"
    "cosmoparams/COM_PowerSpect_CMB-TT-full_R3.01.txt"
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_PLANCK_FILE = RAW_DIR / "COM_PowerSpect_CMB-TT-full_R3.01.txt"
EXPECTED_HEADER_FIELDS = ["l", "Dl", "-dDl", "+dDl"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Download the official Planck PR3 unbinned CMB TT spectrum."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload the official file even if it already exists.",
    )
    return parser.parse_args()


def first_nonempty_line(path: Path) -> str:
    """Return the first nonempty line from a text file."""

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                return stripped
    raise ValueError(f"The downloaded file is empty: {path}")


def validate_downloaded_file(path: Path) -> None:
    """Confirm that the raw file exists, is nonempty, and has the expected header."""

    if not path.exists():
        raise FileNotFoundError(f"Expected downloaded file was not found: {path}")

    file_size = path.stat().st_size
    if file_size <= 0:
        raise ValueError(f"Downloaded file is empty: {path}")

    header = first_nonempty_line(path)
    missing_fields = [field for field in EXPECTED_HEADER_FIELDS if field not in header.split()]
    if missing_fields:
        raise ValueError(
            "The first nonempty line does not contain the expected Planck header "
            f"fields {EXPECTED_HEADER_FIELDS}. Found: {header!r}"
        )


def download_planck_file(output_path: Path, force: bool = False) -> None:
    """Download the official Planck file unless it already exists."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not force:
        validate_downloaded_file(output_path)
        print("Official Planck file already exists; use --force to redownload.")
        print(f"Source URL: {PLANCK_TT_URL}")
        print(f"Output location: {output_path}")
        print(f"File size: {output_path.stat().st_size} bytes")
        return

    try:
        with urllib.request.urlopen(PLANCK_TT_URL, timeout=60) as response:
            data = response.read()
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Could not download the official Planck file. If internet access is "
            "unavailable, manually place COM_PowerSpect_CMB-TT-full_R3.01.txt "
            f"in {RAW_DIR} and rerun this script."
        ) from exc

    if not data:
        raise ValueError("The official Planck server returned an empty file.")

    output_path.write_bytes(data)
    validate_downloaded_file(output_path)

    print(f"Source URL: {PLANCK_TT_URL}")
    print(f"Output location: {output_path}")
    print(f"File size: {output_path.stat().st_size} bytes")


def main() -> None:
    """Download and validate the official raw Planck TT spectrum file."""

    args = parse_args()
    try:
        download_planck_file(RAW_PLANCK_FILE, force=args.force)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
