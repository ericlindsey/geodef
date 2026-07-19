"""Result records for slip inversion, hyperparameter, and geometry analyses.

Private submodule of :mod:`geodef.invert`; the public names defined here
are re-exported by the package ``__init__`` and are stable under
``geodef.invert.<name>`` (see ``docs/api_stability.md``).
"""

import dataclasses
from typing import TYPE_CHECKING

import numpy as np

from geodef.fault import Fault
from geodef.geometry import LocalFrame
from geodef.slip import from_plate, from_rake, magnitude, unpack
from geodef.slip import rake as slip_rake

if TYPE_CHECKING:
    import matplotlib


@dataclasses.dataclass(frozen=True)
class InversionResult:
    """Result of a fault slip inversion.

    Attributes:
        slip: Slip per patch, shape (N, n_components). Columns ordered
            as [strike-slip, dip-slip] for ``components='both'``, or
            a single column for ``'strike'``, ``'dip'``, ``'rake'``, or
            ``'azimuth'``.
        slip_vector: Blocked solution vector, shape (n_components * N,).
        residuals: Observation minus prediction, shape (M,).
        predicted: Forward-modeled observations, shape (M,).
        reduced_chi2: Reduced chi-squared misfit, r^T W r / (M - P).
        rms: Root-mean-square of residuals.
        moment: Scalar seismic moment in N-m.
        Mw: Moment magnitude.
        regularization: Regularization type used, or None if unregularized.
        regularization_strength: Regularization weight used, or None if unregularized.
        components: Which slip components were solved for. One of
            ``'both'``, ``'strike'``, ``'dip'``, ``'rake'``, or
            ``'azimuth'``.
        rake: Fixed rake angle in degrees (in each patch's local
            strike-dip frame) when ``components='rake'``, else ``None``.
            Only physically meaningful for planar faults with uniform
            strike; use ``slip_azimuth`` for curved meshes.
        slip_azimuth: Fixed geographic slip azimuth in degrees CW from
            North when ``components='azimuth'``, else ``None``. Each
            patch's effective local rake is ``slip_azimuth - strike_i``,
            so this correctly handles faults with varying strike.
        plate_rake: Large-scale plate direction expressed as local rake per
            patch when ``components='plate'``. The two solution blocks are
            rake-parallel and rake-perpendicular.
        dataset_names: Stable dataset identifiers in stacked-row order.
        dataset_slices: Corresponding slices into ``predicted`` and
            ``residuals``.
        solver: Solver selected for the completed inversion.
        success: Whether the solver completed successfully.
        message: Solver completion message.
        regularization_selection: ``'abic'`` or ``'cv'`` when selected
            automatically, otherwise ``None``.
        backend: Array backend active during the solve.
        precision: Floating-point precision active during the solve.
        warnings: Interpretation warnings retained with the result.
        quantity: ``'displacement'`` or ``'velocity'``.
        units: Units inherited from the input datasets.
        system_hash: SHA-256 fingerprint of G, d, W, and L.
        lower_bounds: Expanded lower parameter bounds, if any.
        upper_bounds: Expanded upper parameter bounds, if any.
        regularization_target: Reference model used by regularization, if any.
        constraint_matrix: Linear inequality matrix, if any.
        constraint_bounds: Linear inequality right-hand side, if any.
        dataset_diagnostics: Solve-time diagnostics in dataset order.
    """

    slip: np.ndarray
    slip_vector: np.ndarray
    residuals: np.ndarray
    predicted: np.ndarray
    reduced_chi2: float
    rms: float
    moment: float
    Mw: float
    regularization: str | np.ndarray | None
    regularization_strength: float | None
    components: str
    rake: float | None = None
    slip_azimuth: float | None = None
    plate_rake: np.ndarray | None = None
    local_rake: np.ndarray | None = None
    dataset_names: tuple[str, ...] = ()
    dataset_slices: tuple[slice, ...] = ()
    solver: str = "unknown"
    success: bool = True
    message: str = ""
    regularization_selection: str | None = None
    backend: str = "numpy"
    precision: str = "float64"
    warnings: tuple[str, ...] = ()
    quantity: str = "displacement"
    units: str = "m"
    system_hash: str = ""
    lower_bounds: np.ndarray | None = None
    upper_bounds: np.ndarray | None = None
    regularization_target: np.ndarray | None = None
    constraint_matrix: np.ndarray | None = None
    constraint_bounds: np.ndarray | None = None
    dataset_diagnostics: tuple["DatasetDiagnostics", ...] = ()

    @property
    def n_patches(self) -> int:
        """Number of fault patches represented by the result."""
        divisor = 2 if self.components in {"both", "plate"} else 1
        return self.slip_vector.size // divisor

    @property
    def strike_slip(self) -> np.ndarray:
        """Physical strike-slip component per patch."""
        return self._physical_components()[0]

    @property
    def dip_slip(self) -> np.ndarray:
        """Physical dip-slip component per patch."""
        return self._physical_components()[1]

    @property
    def slip_magnitude(self) -> np.ndarray:
        """Unsigned physical slip magnitude per patch."""
        return magnitude(self.strike_slip, self.dip_slip)

    @property
    def slip_rake(self) -> np.ndarray:
        """Physical local rake in degrees per patch."""
        return slip_rake(self.strike_slip, self.dip_slip)

    @property
    def rake_parallel(self) -> np.ndarray:
        """Plate-rake-parallel solution component per patch."""
        if self.components != "plate":
            raise AttributeError("rake_parallel requires components='plate'")
        return self.slip_vector[: self.n_patches]

    @property
    def rake_perpendicular(self) -> np.ndarray:
        """Plate-rake-perpendicular solution component per patch."""
        if self.components != "plate":
            raise AttributeError("rake_perpendicular requires components='plate'")
        return self.slip_vector[self.n_patches :]

    def _physical_components(self) -> tuple[np.ndarray, np.ndarray]:
        """Convert the solved basis to physical strike/dip components."""
        angle: float | np.ndarray | None
        if self.components == "rake":
            angle = self.rake
        elif self.components == "azimuth":
            angle = self.local_rake
        elif self.components == "plate":
            angle = self.plate_rake
        else:
            angle = None
        return _physical_components(self.slip_vector, self.components, angle)


@dataclasses.dataclass(frozen=True)
class DatasetDiagnostics:
    """Per-dataset fit diagnostics.

    Attributes:
        chi2: Weighted sum of squared residuals for this dataset.
        reduced_chi2: chi2 / effective DOF.
        wrms: Weighted root-mean-square residual.
        rms: Unweighted root-mean-square residual.
        n_obs: Number of observations in this dataset.
        dof: Effective degrees of freedom (n_obs - leverage).
        leverage: Sum of hat-matrix diagonal entries for this dataset
            (effective number of parameters consumed).
    """

    chi2: float
    reduced_chi2: float
    wrms: float
    rms: float
    n_obs: int
    dof: float
    leverage: float


@dataclasses.dataclass(frozen=True)
class LCurveResult:
    """Result of an L-curve analysis.

    Attributes:
        regularization_values: Array of lambda values swept.
        misfits: Data misfit norm ||Gm - d|| at each lambda.
        model_norms: Regularized model norm ||Lm|| at each lambda.
        optimal: Lambda at the maximum-curvature corner.
    """

    regularization_values: np.ndarray
    misfits: np.ndarray
    model_norms: np.ndarray
    optimal: float

    def plot(
        self,
        *,
        ax: "matplotlib.axes.Axes | None" = None,
        line_kwargs: dict | None = None,
        marker_kwargs: dict | None = None,
        annotate: bool = True,
    ) -> "matplotlib.axes.Axes":
        """Plot the L-curve with the optimal point marked.

        Args:
            ax: Axes to plot on. Creates a new figure if ``None``.
            line_kwargs: Extra kwargs for the curve line.
            marker_kwargs: Extra kwargs for the optimal-point marker.
            annotate: Whether to label the optimal point with its
                regularization-strength value (default ``True``).

        Returns:
            The axes used for plotting.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()

        lkw = {"color": "b", "marker": ".", "linestyle": "-"}
        if line_kwargs:
            lkw.update(line_kwargs)
        ax.loglog(self.misfits, self.model_norms, **lkw)

        mkw: dict = {"color": "r", "marker": "o", "markersize": 10, "linestyle": "none"}
        if marker_kwargs:
            mkw.update(marker_kwargs)
        idx = np.argmin(np.abs(self.regularization_values - self.optimal))
        ax.loglog(self.misfits[idx], self.model_norms[idx], **mkw)

        if annotate:
            ax.annotate(
                f"λ = {self.optimal:.3g}",
                xy=(self.misfits[idx], self.model_norms[idx]),
                xytext=(10, 10),
                textcoords="offset points",
                fontsize=9,
                color=mkw.get("color", "r"),
                arrowprops={"arrowstyle": "->", "color": mkw.get("color", "r")},
            )

        ax.set_xlabel("Data misfit ||Gm - d||")
        ax.set_ylabel("Model norm ||Lm||")
        ax.set_title("L-curve")
        return ax


@dataclasses.dataclass(frozen=True)
class GeometrySearchResult:
    """Result of a gradient-based nonlinear geometry search.

    Attributes:
        fault: Optimal fault geometry.
        frame: Local frame defining ``theta``.
        theta: Optimal geometry, full 7-vector
            ``[e0, n0, depth, strike, dip, length, width]`` in the local
            Cartesian :attr:`geometry.frame`.
        free: Names of the parameters that were optimized.
        slip: Slip solved linearly at the optimal geometry (inner solve).
        chi2: Weighted misfit ``r^T W r`` at the optimum.
        reduced_chi2: ``chi2 / (n_data - n_free)``.
        theta_cov: Gauss-Newton covariance of the free parameters,
            shape (k, k), scaled by the reduced chi-squared.
        success: Whether the optimizer reported convergence.
        message: Optimizer status message.
        n_iterations: Number of optimizer iterations.
    """

    fault: Fault
    frame: LocalFrame
    theta: np.ndarray
    free: list[str]
    slip: np.ndarray
    chi2: float
    reduced_chi2: float
    theta_cov: np.ndarray
    success: bool
    message: str
    n_iterations: int


@dataclasses.dataclass(frozen=True)
class ABICCurveResult:
    """Result of an ABIC curve analysis.

    Attributes:
        regularization_values: Array of lambda values swept.
        abic_values: ABIC value at each lambda (lower is better).
        misfits: Data misfit norm ||Gm - d|| at each lambda.
        model_norms: Regularized model norm ||Lm|| at each lambda.
        optimal: Lambda at the minimum ABIC.
    """

    regularization_values: np.ndarray
    abic_values: np.ndarray
    misfits: np.ndarray
    model_norms: np.ndarray
    optimal: float

    def plot(
        self,
        *,
        ax: "matplotlib.axes.Axes | None" = None,
        line_kwargs: dict | None = None,
        marker_kwargs: dict | None = None,
        annotate: bool = True,
    ) -> "matplotlib.axes.Axes":
        """Plot ABIC vs regularization strength with the optimal point marked.

        Args:
            ax: Axes to plot on. Creates a new figure if ``None``.
            line_kwargs: Extra kwargs for the curve line.
            marker_kwargs: Extra kwargs for the optimal-point marker.
            annotate: Whether to label the optimal point with its
                regularization-strength value (default ``True``).

        Returns:
            The axes used for plotting.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()

        lkw = {"color": "b", "marker": ".", "linestyle": "-"}
        if line_kwargs:
            lkw.update(line_kwargs)
        ax.semilogx(self.regularization_values, self.abic_values, **lkw)

        mkw: dict = {"color": "r", "marker": "o", "markersize": 10, "linestyle": "none"}
        if marker_kwargs:
            mkw.update(marker_kwargs)
        idx = np.argmin(np.abs(self.regularization_values - self.optimal))
        ax.semilogx(self.regularization_values[idx], self.abic_values[idx], **mkw)

        if annotate:
            ax.annotate(
                f"λ = {self.optimal:.3g}",
                xy=(self.regularization_values[idx], self.abic_values[idx]),
                xytext=(0, 20),
                textcoords="offset points",
                fontsize=9,
                color=mkw.get("color", "r"),
                arrowprops={"arrowstyle": "->", "color": mkw.get("color", "r")},
            )

        ax.set_xlabel("Regularization strength (lambda)")
        ax.set_ylabel("ABIC")
        ax.set_title("ABIC curve")
        return ax


def _physical_components(
    vector: np.ndarray,
    components: str,
    basis_angle: float | np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a solved basis vector to physical strike/dip components."""
    if components == "both":
        return unpack(vector)
    if components == "strike":
        return vector, np.zeros_like(vector)
    if components == "dip":
        return np.zeros_like(vector), vector
    if components in {"rake", "azimuth"}:
        if basis_angle is None:
            raise ValueError(f"{components} result is missing angle metadata")
        return from_rake(vector, basis_angle)
    if components == "plate":
        if basis_angle is None:
            raise ValueError("plate result is missing plate_rake metadata")
        parallel, perpendicular = unpack(vector)
        return from_plate(parallel, perpendicular, basis_angle)
    raise ValueError(f"Unknown slip components {components!r}")
