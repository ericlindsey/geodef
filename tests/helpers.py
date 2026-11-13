"""Function-style shortcuts over the inversion API, shared by the test suite."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from geodef.data import DataSet
from geodef.fault import Fault
from geodef.invert import (
    ABICCurveResult,
    DatasetDiagnostics,
    InversionResult,
    LCurveResult,
    LinearSystem,
)

Datasets = DataSet | list[DataSet]


def lcurve(
    fault: Fault,
    datasets: Datasets,
    regularization: str | np.ndarray = "laplacian",
    regularization_range: tuple[float, float] = (1e-2, 1e6),
    n: int = 50,
    bounds=None,
    method: str | None = None,
    **system_kwargs,
) -> LCurveResult:
    """Run an L-curve sweep on a freshly assembled system."""
    system = LinearSystem(fault, datasets, regularization, **system_kwargs)
    return system.lcurve(regularization_range, n, bounds, method)


def abic_curve(
    fault: Fault,
    datasets: Datasets,
    regularization: str | np.ndarray = "laplacian",
    regularization_range: tuple[float, float] = (1e-2, 1e6),
    n: int = 50,
    **system_kwargs,
) -> ABICCurveResult:
    """Run an ABIC sweep on a freshly assembled system."""
    system = LinearSystem(fault, datasets, regularization, **system_kwargs)
    return system.abic_curve(regularization_range, n)


def model_covariance(
    result: InversionResult, fault: Fault, datasets: Datasets, kind: str = "posterior"
) -> np.ndarray:
    """Model covariance of a result re-attached to ``fault`` and ``datasets``."""
    return result.attach(fault, datasets).covariance(kind=kind)


def model_resolution(
    result: InversionResult, fault: Fault, datasets: Datasets
) -> np.ndarray:
    """Model resolution of a result re-attached to ``fault`` and ``datasets``."""
    return result.attach(fault, datasets).resolution()


def model_uncertainty(
    result: InversionResult, fault: Fault, datasets: Datasets, kind: str = "posterior"
) -> np.ndarray:
    """Model uncertainty of a result re-attached to ``fault`` and ``datasets``."""
    return result.attach(fault, datasets).uncertainty(kind=kind)


def diagnostics(result: InversionResult) -> dict[str, DatasetDiagnostics]:
    """Per-dataset diagnostics of a result."""
    return result.diagnostics()


def save(result: InversionResult, fname: str | Path) -> None:
    """Save a result."""
    result.save(fname)


def load(fname: str | Path) -> InversionResult:
    """Load a result."""
    return InversionResult.load(fname)


def save_table(result: InversionResult, fname: str | Path, fault: Fault) -> None:
    """Save a result's slip table."""
    result.save_table(fname, fault)
