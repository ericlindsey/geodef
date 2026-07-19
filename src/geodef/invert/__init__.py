"""One-call inversion for fault slip from geodetic data.

Solves d = Gm for slip m with optional regularization and bounds.
Supports weighted least-squares, non-negative least-squares,
bounded least-squares, and constrained (QP) solvers.
Automatic hyperparameter tuning via ABIC or cross-validation.

This package re-exports its full public surface (see
``docs/api_stability.md``); the implementation lives in private sibling
submodules: results (``_results``), file I/O (``_io``), regularization
(``_regularization``), solver dispatch (``_solvers``), the prepared
``LinearSystem`` (``_system``), hyperparameter selection (``_selection``),
assessment (``_assessment``), and nonlinear geometry search
(``_geometry``).
"""

import numpy as np

from geodef.data import DataSet
from geodef.fault import Fault
from geodef.invert._assessment import (
    diagnostics as diagnostics,
)
from geodef.invert._assessment import (
    model_covariance as model_covariance,
)
from geodef.invert._assessment import (
    model_resolution as model_resolution,
)
from geodef.invert._assessment import (
    model_uncertainty as model_uncertainty,
)
from geodef.invert._assessment import (
    prediction as prediction,
)
from geodef.invert._assessment import (
    residual as residual,
)
from geodef.invert._assessment import (
    summary as summary,
)
from geodef.invert._geometry import (
    geometry_search as geometry_search,
)
from geodef.invert._io import (
    RESULT_SCHEMA_VERSION as RESULT_SCHEMA_VERSION,
)
from geodef.invert._io import (
    load as load,
)
from geodef.invert._io import (
    save as save,
)
from geodef.invert._io import (
    save_table as save_table,
)
from geodef.invert._results import (
    ABICCurveResult as ABICCurveResult,
)
from geodef.invert._results import (
    DatasetDiagnostics as DatasetDiagnostics,
)
from geodef.invert._results import (
    GeometrySearchResult as GeometrySearchResult,
)
from geodef.invert._results import (
    InversionResult as InversionResult,
)
from geodef.invert._results import (
    LCurveResult as LCurveResult,
)
from geodef.invert._selection import (
    abic_curve as abic_curve,
)
from geodef.invert._selection import (
    compute_abic as compute_abic,
)
from geodef.invert._selection import (
    lcurve as lcurve,
)
from geodef.invert._solvers import (
    BoundsSpec as BoundsSpec,
)
from geodef.invert._system import (
    LinearSystem as LinearSystem,
)


def solve(
    fault: Fault,
    datasets: DataSet | list[DataSet],
    *,
    regularization: str | np.ndarray | None = None,
    regularization_strength: float | str = 0.0,
    bounds: BoundsSpec = None,
    method: str | None = None,
    regularization_target: np.ndarray | None = None,
    components: str = "both",
    rake: float | None = None,
    slip_azimuth: float | None = None,
    constraints: tuple[np.ndarray, np.ndarray] | None = None,
    cv_folds: int = 5,
    plate_rake: float | np.ndarray | None = None,
) -> InversionResult:
    """Invert geodetic data for fault slip.

    Args:
        fault: Fault geometry.
        datasets: One or more geodetic datasets.
        regularization: Regularization type. One of ``'laplacian'``,
            ``'damping'``, ``'stresskernel'``, a custom matrix, or None.
        regularization_strength: Scalar weight on the regularization term,
            or ``'abic'`` / ``'cv'`` for automatic tuning.
        bounds: Per-component slip bounds ``(lower, upper)``.
            Use None for unbounded side, e.g. ``(0, None)``.
        method: Solver — ``'wls'``, ``'nnls'``, ``'bounded_ls'``, or
            ``'constrained'``. Auto-selected from bounds if None.
        regularization_target: Reference model vector, shape
            (n_components * N,). Regularizes toward this target instead
            of zero: minimizes ``||L(m - m_ref)||^2``. Only valid when
            regularization is set.
        components: Which slip components to solve for. One of
            ``'both'`` (default), ``'strike'``, ``'dip'``, ``'rake'``,
            or ``'azimuth'``.
        rake: Fixed rake angle in degrees (same for all patches, in each
            patch's local strike-dip frame), required when
            ``components='rake'``. Only physically meaningful for planar
            faults; use ``slip_azimuth`` for curved meshes.
        slip_azimuth: Geographic slip azimuth in degrees CW from North,
            required when ``components='azimuth'``. Each patch's
            effective local rake is ``slip_azimuth - strike_i``,
            so this correctly handles faults with varying strike.
        plate_rake: Large-scale direction as a local rake angle, scalar or
            shape (N,), required when ``components='plate'``. The solved
            blocks are rake-parallel and rake-perpendicular.
        constraints: Inequality constraints ``(C, d_ineq)`` such that
            ``C @ m <= d_ineq``. Only used with ``method='constrained'``.
        cv_folds: Number of folds for cross-validation (default 5).

    Returns:
        InversionResult with slip, residuals, and fit statistics.

    Raises:
        ValueError: For invalid arguments.
    """
    sys = LinearSystem(
        fault,
        datasets,
        regularization,
        components,
        rake,
        slip_azimuth,
        plate_rake,
    )
    return sys.invert(
        regularization_strength,
        bounds,
        method,
        regularization_target,
        constraints,
        cv_folds,
    )


# ======================================================================
# Private helpers
# ======================================================================
