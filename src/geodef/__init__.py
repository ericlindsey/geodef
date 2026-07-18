"""GeoDef: forward and inverse modeling of fault slip in elastic half-spaces."""

__version__ = "0.1.0"

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
from geodef.data import GNSS, DataSet, InSAR, Vertical

# Beginner-public names (Fault, GNSS/InSAR/Vertical, DataSet, LocalFrame,
# ElasticMedium, DEFAULT_MEDIUM, solve, InversionResult) are imported plainly
# and listed in ``__all__`` below. Expert-public names use a redundant ``as``
# alias: they remain reachable as ``geodef.<name>`` for backward compatibility
# but are kept out of ``__all__`` and reached through their module in new code
# (``geodef.invert.lcurve``, ``geodef.greens.stack_obs``, ...), pending removal
# from the top level over the roadmap 3.1 deprecation cycle.
from geodef.data import spatial_covariance as spatial_covariance
from geodef.fault import Fault
from geodef.fault import magnitude_to_moment as magnitude_to_moment
from geodef.fault import moment_to_magnitude as moment_to_magnitude
from geodef.geometry import LocalFrame
from geodef.greens import select_slip_columns as select_slip_columns
from geodef.greens import stack_obs as stack_obs
from geodef.greens import stack_weights as stack_weights
from geodef.invert import ABICCurveResult as ABICCurveResult
from geodef.invert import DatasetDiagnostics as DatasetDiagnostics
from geodef.invert import GeometrySearchResult as GeometrySearchResult
from geodef.invert import InversionResult, solve
from geodef.invert import LCurveResult as LCurveResult
from geodef.invert import LinearSystem as LinearSystem
from geodef.invert import abic_curve as abic_curve
from geodef.invert import compute_abic as compute_abic
from geodef.invert import geometry_search as geometry_search
from geodef.invert import lcurve as lcurve
from geodef.invert import model_covariance as model_covariance
from geodef.invert import model_resolution as model_resolution
from geodef.invert import model_uncertainty as model_uncertainty
from geodef.medium import DEFAULT_MEDIUM, ElasticMedium

# ----------------------------------------------------------------------
# Public API tiers
#
# ``__all__`` is the beginner-public vocabulary: the small, stable set a
# novice needs for the everyday forward-modeling and inversion path, plus
# the submodules that are the discovery surface for everything else.
#
# The expert-public API lives under those submodule paths -- e.g.
# ``geodef.invert.lcurve``, ``geodef.greens.matrix``, ``geodef.slip.pack``.
# Several expert names are still re-exported at the top level (imported
# above) for backward compatibility; they are intentionally kept out of
# ``__all__`` and will be removed from the top level over a deprecation
# cycle (roadmap 3.1). Reach them through their module in new code.
#
# Private names are underscore-prefixed and are not part of any tier.
# ----------------------------------------------------------------------
__all__ = [
    # Submodules (discovery surface for the expert-public API)
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
    # Beginner-public: domain objects
    "Fault",
    "GNSS",
    "InSAR",
    "Vertical",
    "DataSet",
    "LocalFrame",
    "ElasticMedium",
    "DEFAULT_MEDIUM",
    # Beginner-public: the one-shot inversion and its result record
    "solve",
    "InversionResult",
]
