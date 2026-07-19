"""Execution checks for tutorial notebooks."""

from __future__ import annotations

import os
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]

# The canonical tutorial sequence (see ``tutorials/README.md``).
TUTORIAL_NOTEBOOKS = (
    ROOT / "tutorials" / "00_preflight.ipynb",
    ROOT / "tutorials" / "01_forward_model.ipynb",
    ROOT / "tutorials" / "02_discretization_and_g_matrix.ipynb",
    ROOT / "tutorials" / "03_unregularized_inversion.ipynb",
    ROOT / "tutorials" / "04_regularization.ipynb",
    ROOT / "tutorials" / "05_multiple_datasets.ipynb",
    ROOT / "tutorials" / "06_correlated_noise.ipynb",
    ROOT / "tutorials" / "07_bounds_and_constraints.ipynb",
    ROOT / "tutorials" / "08_uncertainty_and_resolution.ipynb",
    ROOT / "tutorials" / "09_nonlinear_geometry.ipynb",
    ROOT / "tutorials" / "10_gradient_geometry.ipynb",
    ROOT / "tutorials" / "11_triangular_faults.ipynb",
    ROOT / "tutorials" / "12_interseismic_coupling.ipynb",
    ROOT / "tutorials" / "13_model_misspecification.ipynb",
    ROOT / "tutorials" / "14_bayesian_inversion.ipynb",
)

# Optional imports needed by individual chapters.
_OPTIONAL_IMPORTS = {
    "10_gradient_geometry.ipynb": ("jax",),
    "14_bayesian_inversion.ipynb": ("jax", "blackjax"),
}


def _notebook_id(path: Path) -> str:
    """Return a compact pytest id for a notebook path."""
    return path.name


@pytest.mark.parametrize("notebook_path", TUTORIAL_NOTEBOOKS, ids=_notebook_id)
def test_tutorial_notebook_executes(
    notebook_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Execute a tutorial notebook from its own directory."""
    for module_name in _OPTIONAL_IMPORTS.get(notebook_path.name, ()):
        pytest.importorskip(module_name)

    src_path = ROOT / "src"
    existing_pythonpath = os.environ.get("PYTHONPATH")
    pythonpath = str(src_path)
    if existing_pythonpath:
        pythonpath = f"{pythonpath}{os.pathsep}{existing_pythonpath}"

    monkeypatch.setenv("PYTHONPATH", pythonpath)
    monkeypatch.setenv("MPLBACKEND", "Agg")

    notebook = nbformat.read(notebook_path, as_version=4)
    client = NotebookClient(
        notebook,
        kernel_name="python3",
        timeout=600,
        resources={"metadata": {"path": str(notebook_path.parent)}},
    )
    client.execute()
