"""GeoDef: forward and inverse modeling of fault slip in elastic half-spaces."""

__version__ = "1.1.0"

from geodef import (
    backend,
    bayes,
    cache,
    euler,
    geomap,
    geometry,
    gradients,
    greens,
    invert,
    medium,
    mesh,
    okada,
    okada85,
    okada92,
    plot,
    slip,
    transforms,
    tri,
    validation,
)
from geodef.data import GNSS, DataSet, InSAR, Vertical, spatial_covariance
from geodef.fault import Fault, magnitude_to_moment, moment_to_magnitude
from geodef.geometry import LocalFrame
from geodef.greens import select_slip_columns, stack_obs, stack_weights
from geodef.invert import (
    ABICCurveResult,
    DatasetDiagnostics,
    GeometrySearchResult,
    InversionResult,
    LCurveResult,
    LinearSystem,
    abic_curve,
    compute_abic,
    dataset_diagnostics,
    geometry_search,
    lcurve,
    model_covariance,
    model_resolution,
    model_uncertainty,
    solve,
)
from geodef.medium import DEFAULT_MEDIUM, ElasticMedium

__all__ = [
    # Submodules
    "backend",
    "bayes",
    "cache",
    "euler",
    "geomap",
    "geometry",
    "gradients",
    "greens",
    "invert",
    "medium",
    "mesh",
    "okada",
    "okada85",
    "okada92",
    "plot",
    "slip",
    "transforms",
    "tri",
    "validation",
    # Data types
    "GNSS",
    "InSAR",
    "Vertical",
    "DataSet",
    "spatial_covariance",
    # Fault geometry, medium, and moment
    "Fault",
    "LocalFrame",
    "ElasticMedium",
    "DEFAULT_MEDIUM",
    "magnitude_to_moment",
    "moment_to_magnitude",
    # Green's matrix assembly
    "select_slip_columns",
    "stack_obs",
    "stack_weights",
    # Inversion and assessment
    "ABICCurveResult",
    "DatasetDiagnostics",
    "GeometrySearchResult",
    "InversionResult",
    "LCurveResult",
    "LinearSystem",
    "abic_curve",
    "compute_abic",
    "dataset_diagnostics",
    "geometry_search",
    "solve",
    "lcurve",
    "model_covariance",
    "model_resolution",
    "model_uncertainty",
]
