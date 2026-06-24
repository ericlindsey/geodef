"""Execution checks for tutorial notebooks."""

from __future__ import annotations

import os
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]

# The live, progressively built tutorial sequence (see PLAN.md). Previous-
# generation notebooks are retained as ``old_*`` reference copies and are not
# executed here; new notebooks are added to this tuple as they are written.
TUTORIAL_NOTEBOOKS = (
    ROOT / "tutorials" / "01_forward_model.ipynb",
    ROOT / "tutorials" / "02_discretization_and_g_matrix.ipynb",
)


def _notebook_id(path: Path) -> str:
    """Return a compact pytest id for a notebook path."""
    return path.name


@pytest.mark.parametrize("notebook_path", TUTORIAL_NOTEBOOKS, ids=_notebook_id)
def test_tutorial_notebook_executes(
    notebook_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Execute a tutorial notebook from its own directory."""
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
