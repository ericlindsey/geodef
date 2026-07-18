"""One-call inversion for fault slip from geodetic data.

Solves d = Gm for slip m with optional regularization and bounds.
Supports weighted least-squares, non-negative least-squares,
bounded least-squares, and constrained (QP) solvers.
Automatic hyperparameter tuning via ABIC or cross-validation.
"""

from collections.abc import Mapping

import numpy as np
import scipy.optimize

from geodef import backend
from geodef.data import DataSet
from geodef.fault import Fault
from geodef.geometry import (
    LocalFrame,
    _resolve_frame,
    as_planar_vector,
)
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
    GeometrySearchResult,
    InversionResult,
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
    BoundsSpec,
)
from geodef.invert._system import (
    LinearSystem,
)
from geodef.invert._system import (
    _validate_args as _validate_args,
)

_THETA_NAMES = ("e0", "n0", "depth", "strike", "dip", "length", "width")


# ======================================================================
# LinearSystem: persistent prepared system with cached matrix products
# ======================================================================


# ======================================================================
# Module-level convenience functions (backward-compatible wrappers)
# ======================================================================


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


def _projection_matrix(datasets: list[DataSet]) -> np.ndarray:
    """Build the linear map from stacked [E, N, U] displacements to data.

    Every displacement dataset's ``project()`` is linear, so the exact
    operator is recovered by probing it with unit basis fields. Column
    ``3*k + c`` corresponds to component ``c`` of station ``k`` within its
    dataset block, matching the row layout of ``gradients.rect_greens``.

    Args:
        datasets: Datasets in the same order used to stack observations.

    Returns:
        Block-diagonal projection matrix, shape (M_total, 3*nobs_total).
    """
    blocks = []
    for ds in datasets:
        n = ds.n_stations
        zero = np.zeros(n)
        cols = []
        for k in range(n):
            for c in range(3):
                unit = [zero, zero, zero]
                probe = np.zeros(n)
                probe[k] = 1.0
                unit[c] = probe
                cols.append(ds.project(*unit))
        blocks.append(np.column_stack(cols))
    return scipy.linalg.block_diag(*blocks)


def _vp_residual(
    x,
    theta_base,
    free_idx,
    e_obs,
    n_obs,
    P,
    W_half,
    d_w,
    LtL,
    n_length,
    n_width,
    col_start,
    col_stop,
    nu,
):
    """Weighted variable-projection residual and inner slip (traceable).

    Assembles G(theta) with the differentiable ``gradients.rect_greens``,
    projects into data space, solves the regularized least-squares slip,
    and returns the weighted residual. Pure function of its arguments so
    the JIT compilation is shared across calls with the same shapes.
    """
    import jax.numpy as jnp

    from geodef.gradients import rect_greens

    theta = theta_base.at[free_idx].set(x)
    G3 = rect_greens(theta, e_obs, n_obs, n_length, n_width, nu)
    G_w = W_half @ (P @ G3)[:, col_start:col_stop]
    H = G_w.T @ G_w + LtL
    m = jnp.linalg.solve(H, G_w.T @ d_w)
    return d_w - G_w @ m, m


def _vp_residual_and_jacobian(x, *args):
    """Residual, inner slip, and forward-mode residual Jacobian.

    Everything the optimizer needs comes from this one function: the
    objective is ``r @ r``, its exact gradient is ``2 J.T @ r``, and
    ``J`` at the optimum is the Gauss-Newton covariance Jacobian.
    Forward-mode only — reverse-mode differentiation through the kernel
    compiles far more slowly for no benefit at this parameter count.
    """
    import jax

    r_w, m = _vp_residual(x, *args)
    jac = jax.jacfwd(lambda xx: _vp_residual(xx, *args)[0])(x)
    return r_w, m, jac


_VP_STATIC_ARGNUMS = (9, 10, 11, 12, 13)
_vp_jitted: dict = {}


def _vp_kernel():
    """The JIT-compiled variable-projection kernel, cached at module level.

    Module-level caching means repeated ``geometry_search`` calls with
    the same problem shapes (multi-start, repeated studies) reuse the
    compilation instead of retracing per call.
    """
    if "kernel" not in _vp_jitted:
        import jax

        _vp_jitted["kernel"] = jax.jit(
            _vp_residual_and_jacobian, static_argnums=_VP_STATIC_ARGNUMS
        )
    return _vp_jitted["kernel"]


def _fault_from_planar_vector(
    theta: np.ndarray,
    frame: LocalFrame,
    n_length: int,
    n_width: int,
) -> Fault:
    """Construct a planar fault from the local expert parameter vector."""
    geographic = frame.to_geographic(east=theta[0], north=theta[1], up=0.0)
    return Fault.planar(
        lat=float(geographic[1]),
        lon=float(geographic[0]),
        depth=float(theta[2]),
        strike=float(theta[3]),
        dip=float(theta[4]),
        length=float(theta[5]),
        width=float(theta[6]),
        n_length=n_length,
        n_width=n_width,
        frame=frame,
    )


def geometry_search(
    theta0: np.ndarray | Mapping[str, float],
    datasets: DataSet | list[DataSet],
    *,
    ref_lat: float | None = None,
    ref_lon: float | None = None,
    frame: LocalFrame | None = None,
    free: list[str] | None = None,
    bounds: dict[str, tuple[float, float]] | None = None,
    n_length: int = 1,
    n_width: int = 1,
    components: str = "both",
    regularization: str | np.ndarray | None = None,
    regularization_strength: float = 0.0,
    nu: float = 0.25,
) -> GeometrySearchResult:
    """Gradient-based nonlinear inversion for planar fault geometry.

    Minimizes the weighted data misfit over selected geometry parameters
    with the slip distribution solved linearly inside (variable
    projection): at each trial geometry, ``G(theta)`` is assembled with
    the differentiable ``gradients.rect_greens`` and the regularized
    least-squares slip is computed, and JAX differentiates the whole
    pipeline so the optimizer (L-BFGS-B) follows exact gradients. This
    replaces the grid-then-``minimize_scalar`` recipe of tutorial 10 and
    scales to several simultaneous geometry parameters.

    Requires the JAX backend (``geodef.backend.set_backend('jax')``).

    Args:
        theta0: Starting parameter mapping, or expert array
            ``[east, north, depth, strike, dip, length, width]``. Requires
            ``frame`` or ``ref_lat``/``ref_lon``.
        datasets: One or more displacement datasets (GNSS, InSAR,
            Vertical).
        ref_lat: Latitude anchoring the local Cartesian frame.
        ref_lon: Longitude anchoring the local Cartesian frame.
        frame: Explicit local frame for array ``theta0``. Mutually exclusive
            with an incompatible legacy ``ref_lat``/``ref_lon`` origin.
        free: Names of parameters to optimize (subset of ``e0, n0,
            depth, strike, dip, length, width``). Default: all seven.
        bounds: Optional per-parameter ``(lower, upper)`` bounds, keyed
            by parameter name.
        n_length: Number of patches along strike.
        n_width: Number of patches down dip.
        components: Slip components for the inner solve: ``'both'``,
            ``'strike'``, or ``'dip'``.
        regularization: Regularization type for the inner solve (as in
            ``invert()``), or None for no regularization.
        regularization_strength: Regularization weight lambda for the inner
            solve (held fixed during the search).
        nu: Poisson's ratio.

    Returns:
        GeometrySearchResult with optimal ``fault``, expert ``theta``, frame,
        inner slip, misfit, and a Gauss-Newton covariance.

    Raises:
        RuntimeError: If the JAX backend is not active.
        ValueError: If ``free`` contains an unknown parameter name or
            ``components`` is not supported here.
    """
    if backend.get_backend() != "jax":
        raise RuntimeError(
            "geometry_search requires the JAX backend; "
            "call geodef.backend.set_backend('jax') first."
        )
    import jax.numpy as jnp

    if isinstance(datasets, DataSet):
        datasets = [datasets]
    if free is None:
        free = list(_THETA_NAMES)
    unknown = [name for name in free if name not in _THETA_NAMES]
    if unknown:
        raise ValueError(
            f"Unknown free parameter(s) {unknown}; expected names from {_THETA_NAMES}."
        )
    if components not in ("both", "strike", "dip"):
        raise ValueError(
            "geometry_search supports components 'both', 'strike', or "
            f"'dip', got {components!r}"
        )

    frame = _resolve_frame(frame, ref_lat, ref_lon)
    theta0 = as_planar_vector(theta0)
    free_idx = np.array([_THETA_NAMES.index(name) for name in free])

    # Template system provides the stacked data, weights, and (fixed)
    # regularization operator; its Green's matrix is not used.
    template = _fault_from_planar_vector(theta0, frame, n_length, n_width)
    sys = LinearSystem(template, datasets, regularization, components)
    n_patches = n_length * n_width
    col_start, col_stop = {
        "both": (0, 2 * n_patches),
        "strike": (0, n_patches),
        "dip": (n_patches, 2 * n_patches),
    }[components]

    e_parts, n_parts = [], []
    for ds in datasets:
        enu = frame.to_enu(
            lon=ds.lon,
            lat=ds.lat,
            alt=np.full(ds.n_stations, frame.origin_alt),
        )
        e_parts.append(enu[:, 0])
        n_parts.append(enu[:, 1])
    e_obs = np.concatenate(e_parts)
    n_obs = np.concatenate(n_parts)

    P = jnp.asarray(_projection_matrix(datasets))
    W_half = jnp.asarray(scipy.linalg.cholesky(sys.W, lower=False))
    d_w = jnp.asarray(sys.d_w)
    theta_base = jnp.asarray(theta0)
    free_j = jnp.asarray(free_idx)
    if regularization_strength > 0.0:
        if sys.L is None:
            raise ValueError(
                "regularization_strength > 0 requires a regularization operator"
            )
        LtL = jnp.asarray(sys.LtL) * regularization_strength
    else:
        LtL = jnp.zeros((sys.G.shape[1], sys.G.shape[1]))

    vp_args = (
        theta_base,
        free_j,
        jnp.asarray(e_obs),
        jnp.asarray(n_obs),
        P,
        W_half,
        d_w,
        LtL,
    )
    vp_static = (n_length, n_width, col_start, col_stop, float(nu))
    kernel = _vp_kernel()

    def scipy_objective(x: np.ndarray) -> tuple[float, np.ndarray]:
        r_w, _, jac = kernel(jnp.asarray(x), *vp_args, *vp_static)
        value = float(backend.to_numpy(r_w @ r_w))
        grad = 2.0 * backend.to_numpy(jac.T @ r_w)
        return value, np.asarray(grad, dtype=float)

    scipy_bounds = None
    if bounds is not None:
        scipy_bounds = [bounds.get(name, (None, None)) for name in free]

    opt = scipy.optimize.minimize(
        scipy_objective,
        theta0[free_idx],
        jac=True,
        method="L-BFGS-B",
        bounds=scipy_bounds,
    )

    x_opt = jnp.asarray(opt.x)
    r_w, m, jac = kernel(x_opt, *vp_args, *vp_static)
    chi2 = float(backend.to_numpy(r_w @ r_w))
    n_data = len(sys.d)
    k = len(free)
    dof = max(n_data - k, 1)
    reduced_chi2 = chi2 / dof

    jtj = backend.to_numpy(jac.T @ jac)
    theta_cov = reduced_chi2 * np.linalg.inv(jtj)

    theta_opt = theta0.copy()
    theta_opt[free_idx] = np.asarray(opt.x, dtype=float)
    fault_opt = _fault_from_planar_vector(theta_opt, frame, n_length, n_width)

    return GeometrySearchResult(
        fault=fault_opt,
        frame=frame,
        theta=theta_opt,
        free=list(free),
        slip=backend.to_numpy(m),
        chi2=chi2,
        reduced_chi2=reduced_chi2,
        theta_cov=theta_cov,
        success=bool(opt.success),
        message=str(opt.message),
        n_iterations=int(opt.nit),
    )


# ======================================================================
# Private helpers
# ======================================================================


