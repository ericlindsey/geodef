"""Shared fixtures and path configuration for geodef tests."""

from pathlib import Path

import numpy as np
import pytest

from geodef import cache

REFERENCE_DATA_DIR = Path(__file__).parent / "reference_data"


@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path: Path) -> None:
    """Redirect the geodef cache to a temp directory for every test."""
    cache.set_dir(tmp_path / "geodef_cache")
    cache.enable()


@pytest.fixture
def nu() -> float:
    """Standard Poisson's ratio used across all tests."""
    return 0.25


@pytest.fixture
def reference_data_dir() -> Path:
    """Path to the directory containing reference .npz data files."""
    return REFERENCE_DATA_DIR


def load_reference_data(name: str) -> dict[str, np.ndarray]:
    """Load a reference data .npz file by name (without extension).

    Args:
        name: Filename stem, e.g. 'HS_simple'.

    Returns:
        Dictionary of numpy arrays from the .npz file.
    """
    path = REFERENCE_DATA_DIR / f"{name}.npz"
    data = np.load(str(path))
    return dict(data)
