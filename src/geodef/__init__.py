"""GeoDef: forward and inverse modeling of fault slip in elastic half-spaces."""

__version__ = "0.1.0"

from geodef import (
    backend,
    bayes,
    cache,
    data,
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
from geodef.fault import Fault
from geodef.geometry import LocalFrame
from geodef.invert import InversionResult, LinearSystem, solve
from geodef.medium import DEFAULT_MEDIUM, ElasticMedium

# The everyday vocabulary is importable from the top level; everything else
# is reached through its module (``geodef.plot.slip``, ``geodef.mesh.from_trace``).
__all__ = [
    # Core objects
    "Fault",
    "GNSS",
    "InSAR",
    "Vertical",
    "DataSet",
    "LocalFrame",
    "ElasticMedium",
    "DEFAULT_MEDIUM",
    # Inversion
    "solve",
    "LinearSystem",
    "InversionResult",
    # Modules
    "backend",
    "bayes",
    "cache",
    "data",
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
]
