"""One-time script to extract .mat reference data to .npz files.

Run this script to convert the Matlab .mat test data files from the tdcalc
matlab_source directory into numpy .npz files in tests/reference_data/.
This removes the scipy dependency for running tests.

Usage:
    uv run python tests/extract_mat_data.py
"""

from pathlib import Path

import numpy as np
import scipy.io


def extract_mat_to_npz(mat_path: Path, npz_path: Path) -> None:
    """Convert a .mat file to .npz, preserving all numeric arrays.

    Args:
        mat_path: Path to source .mat file.
        npz_path: Path to output .npz file.
    """
    data = scipy.io.loadmat(str(mat_path), squeeze_me=True)

    arrays = {}
    for key, value in data.items():
        if key.startswith("_"):
            continue
        if isinstance(value, np.ndarray):
            arrays[key] = value
        elif np.isscalar(value):
            arrays[key] = np.array(value)

    np.savez(str(npz_path), **arrays)
    print(f"  {mat_path.name} -> {npz_path.name}: {len(arrays)} arrays")


def main() -> None:
    """Extract all .mat files to .npz format."""
    project_root = Path(__file__).parent.parent
    mat_dir = project_root / "geometry" / "tdcalc" / "matlab_source"
    out_dir = project_root / "tests" / "reference_data"
    out_dir.mkdir(parents=True, exist_ok=True)

    mat_files = sorted(mat_dir.glob("*.mat"))
    if not mat_files:
        print(f"No .mat files found in {mat_dir}")
        return

    print(f"Extracting {len(mat_files)} .mat files:")
    for mat_path in mat_files:
        npz_path = out_dir / mat_path.with_suffix(".npz").name
        extract_mat_to_npz(mat_path, npz_path)

    print("Done.")


if __name__ == "__main__":
    main()
