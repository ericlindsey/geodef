"""Linear slip inversion: ``d = G m`` with weights, regularization, and bounds.

Two entry points share one implementation:

- :func:`solve` inverts a fault and its datasets in one call.
- :class:`LinearSystem` holds the assembled ``G``, ``d``, ``W``, and ``L``
  so they can be inspected or reused for L-curve and ABIC sweeps and for
  solves at several regularization strengths.

Both return an :class:`InversionResult`, which carries named slip views,
fit statistics, and the assessment methods ``covariance()``,
``uncertainty()``, and ``resolution()``.

The implementation lives in private sibling modules: results
(``_results``), result files (``_io``), regularization operators
(``_regularization``), solver dispatch (``_solvers``), the linear system
(``_system``), and nonlinear geometry search (``_geometry``).
"""

import numpy as np

from geodef.data import DataSet
from geodef.fault import Fault
from geodef.invert._geometry import geometry_search as geometry_search
from geodef.invert._results import ABICCurveResult as ABICCurveResult
from geodef.invert._results import DatasetDiagnostics as DatasetDiagnostics
from geodef.invert._results import GeometrySearchResult as GeometrySearchResult
from geodef.invert._results import InversionResult as InversionResult
from geodef.invert._results import LCurveResult as LCurveResult
from geodef.invert._results import SolveDetails as SolveDetails
from geodef.invert._selection import compute_abic as compute_abic
from geodef.invert._solvers import BoundsSpec as BoundsSpec
from geodef.invert._system import LinearSystem as LinearSystem


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
    return sys.solve(
        regularization_strength,
        bounds,
        method,
        regularization_target,
        constraints,
        cv_folds,
    )
