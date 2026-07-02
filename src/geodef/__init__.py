"""GeoDef: forward and inverse modeling of fault slip in elastic half-spaces."""

__version__ = "0.1.0"

from geodef import (
    cache,
    euler,
    greens,
    mesh,
    okada,
    okada85,
    okada92,
    plot,
    transforms,
    tri,
)
from geodef.data import GNSS, DataSet, InSAR, Vertical, spatial_covariance
from geodef.fault import Fault, magnitude_to_moment, moment_to_magnitude
from geodef.greens import select_slip_columns, stack_obs, stack_weights
from geodef.invert import (
    ABICCurveResult,
    DatasetDiagnostics,
    InversionResult,
    LCurveResult,
    LinearSystem,
    abic_curve,
    compute_abic,
    dataset_diagnostics,
    invert,
    lcurve,
    model_covariance,
    model_resolution,
    model_uncertainty,
)

__all__ = [
    # Submodules
    "cache",
    "euler",
    "greens",
    "mesh",
    "okada",
    "okada85",
    "okada92",
    "plot",
    "transforms",
    "tri",
    # Data types
    "GNSS",
    "InSAR",
    "Vertical",
    "DataSet",
    "spatial_covariance",
    # Fault geometry and moment
    "Fault",
    "magnitude_to_moment",
    "moment_to_magnitude",
    # Green's matrix assembly
    "select_slip_columns",
    "stack_obs",
    "stack_weights",
    # Inversion and assessment
    "ABICCurveResult",
    "DatasetDiagnostics",
    "InversionResult",
    "LCurveResult",
    "LinearSystem",
    "abic_curve",
    "compute_abic",
    "dataset_diagnostics",
    "invert",
    "lcurve",
    "model_covariance",
    "model_resolution",
    "model_uncertainty",
]
