"""Result records for slip inversion, hyperparameter, and geometry analyses.

Private submodule of :mod:`geodef.invert`; the public names defined here
are re-exported by the package ``__init__`` as ``geodef.invert.<name>``.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING, overload

import numpy as np

from geodef.fault import Fault
from geodef.slip import from_plate, from_rake, magnitude, unpack
from geodef.slip import rake as slip_rake
from geodef.transforms import LocalFrame

if TYPE_CHECKING:
    import matplotlib

    from geodef.data import DataSet
    from geodef.invert._system import LinearSystem


@dataclasses.dataclass(frozen=True)
class SolveDetails:
    """How an inversion was solved: the bookkeeping behind a result.

    Everything here is recorded so a result can be reproduced and audited;
    none of it is needed to read the slip or the fit.

    Attributes:
        solver: Solver used: ``'wls'``, ``'nnls'``, ``'bounded_ls'``, or
            ``'constrained'``.
        success: Whether the solver reported success.
        message: Solver completion message.
        regularization_selection: ``'abic'`` or ``'cv'`` when the
            regularization strength was chosen automatically, else ``None``.
        backend: Array backend active during the solve.
        precision: Floating-point precision active during the solve.
        system_hash: SHA-256 fingerprint of ``G``, ``d``, ``W``, and ``L``.
        dataset_slices: Rows of ``predicted`` and ``residuals`` belonging to
            each dataset, in ``dataset_names`` order.
        dataset_diagnostics: Per-dataset fit statistics, same order.
        local_rake: Per-patch rake of the ``'azimuth'`` basis, else ``None``.
        lower_bounds: Lower parameter bounds after expansion, if any.
        upper_bounds: Upper parameter bounds after expansion, if any.
        regularization_target: Reference model of the regularization, if any.
        constraint_matrix: Linear inequality matrix ``C``, if any.
        constraint_bounds: Right-hand side of ``C m <= d``, if any.
    """

    solver: str = "unknown"
    success: bool = True
    message: str = ""
    regularization_selection: str | None = None
    backend: str = "numpy"
    precision: str = "float64"
    system_hash: str = ""
    dataset_slices: tuple[slice, ...] = ()
    dataset_diagnostics: tuple[DatasetDiagnostics, ...] = ()
    local_rake: np.ndarray | None = None
    lower_bounds: np.ndarray | None = None
    upper_bounds: np.ndarray | None = None
    regularization_target: np.ndarray | None = None
    constraint_matrix: np.ndarray | None = None
    constraint_bounds: np.ndarray | None = None


@dataclasses.dataclass(frozen=True)
class InversionResult:
    """The answer to an inversion: slip, fit, and the problem it solves.

    Read slip through the named per-patch views (``strike_slip``,
    ``dip_slip``, ``slip_magnitude``, ``slip_rake``) and the fit through
    ``reduced_chi2``, ``prediction()``, and ``diagnostics()``. Assessment
    (``covariance()``, ``uncertainty()``, ``resolution()``) uses the fault
    and datasets the result remembers. How it was solved is in ``details``.

    Attributes:
        slip_vector: Solution in the solved basis, blocked by component:
            shape (N,) for one-component bases, (2N,) for ``'both'`` and
            ``'plate'``.
        residuals: Observed minus predicted data, stacked over datasets.
        predicted: Predicted data, stacked over datasets.
        reduced_chi2: Weighted misfit per degree of freedom,
            ``r^T W r / (M - P)``.
        rms: Root-mean-square residual, in ``units``.
        moment: Scalar seismic moment in N m (NaN for velocity data).
        Mw: Moment magnitude (NaN for velocity data).
        regularization: Regularization operator used, or ``None``.
        regularization_strength: Lambda used, or ``None``.
        components: Slip basis: ``'both'``, ``'strike'``, ``'dip'``,
            ``'rake'``, ``'azimuth'``, or ``'plate'``.
        rake: Fixed rake in degrees for ``components='rake'``.
        slip_azimuth: Geographic slip azimuth for ``components='azimuth'``.
        plate_rake: Per-patch plate direction for ``components='plate'``.
        dataset_names: Dataset names in stacked-row order.
        warnings: Interpretation warnings.
        quantity: ``'displacement'`` or ``'velocity'``.
        units: Units of the data, residuals, and RMS.
        details: Solver bookkeeping; see :class:`SolveDetails`.
        system: The :class:`LinearSystem` that produced this result. It
            gives the assessment methods access to the fault and data.
            ``None`` for a result loaded from a file until :meth:`attach`
            is called.
    """

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
    dataset_names: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    quantity: str = "displacement"
    units: str = "m"
    details: SolveDetails = dataclasses.field(default_factory=SolveDetails)
    system: LinearSystem | None = dataclasses.field(
        default=None, repr=False, compare=False
    )

    # ------------------------------------------------------------------
    # The problem this result solves
    # ------------------------------------------------------------------

    def _system(self) -> LinearSystem:
        if self.system is None:
            raise ValueError(
                "this result is not attached to a fault and datasets (it was "
                "loaded from a file); call result.attach(fault, datasets) first"
            )
        return self.system

    @property
    def fault(self) -> Fault:
        """The fault this result was solved on."""
        return self._system().fault

    @property
    def datasets(self) -> list[DataSet]:
        """The datasets this result was fit to, in stacked-row order."""
        return self._system().datasets

    def attach(
        self, fault: Fault, datasets: DataSet | list[DataSet]
    ) -> InversionResult:
        """Return a copy of this result attached to its fault and datasets.

        Needed only for results loaded from a file, which store the
        solution but not the problem.

        Args:
            fault: Fault geometry used for the inversion.
            datasets: Dataset(s) used for the inversion.

        Returns:
            A result whose assessment methods are available.
        """
        from geodef.invert._system import LinearSystem

        system = LinearSystem(
            fault,
            datasets,
            self.regularization,
            self.components,
            self.rake,
            self.slip_azimuth,
            self.plate_rake,
        )
        return dataclasses.replace(self, system=system)

    # ------------------------------------------------------------------
    # Fit: predictions, residuals, diagnostics
    # ------------------------------------------------------------------

    @overload
    def prediction(self, dataset: None = None) -> dict[str, np.ndarray]: ...
    @overload
    def prediction(self, dataset: DataSet | str) -> np.ndarray: ...
    def prediction(
        self, dataset: DataSet | str | None = None
    ) -> dict[str, np.ndarray] | np.ndarray:
        """Model predictions, for every dataset or for one.

        Args:
            dataset: A dataset that was inverted (the object itself or its
                name in ``dataset_names``). ``None`` returns every dataset.

        Returns:
            The predicted observation vector for ``dataset``, or a
            name-keyed mapping of all of them in solve order.
        """
        return self._by_dataset(self.predicted, dataset)

    @overload
    def residual(self, dataset: None = None) -> dict[str, np.ndarray]: ...
    @overload
    def residual(self, dataset: DataSet | str) -> np.ndarray: ...
    def residual(
        self, dataset: DataSet | str | None = None
    ) -> dict[str, np.ndarray] | np.ndarray:
        """Observed-minus-predicted residuals, for every dataset or for one.

        Args:
            dataset: A dataset that was inverted (the object itself or its
                name in ``dataset_names``). ``None`` returns every dataset.

        Returns:
            The residual vector for ``dataset``, or a name-keyed mapping of
            all of them in solve order.
        """
        return self._by_dataset(self.residuals, dataset)

    def diagnostics(self) -> dict[str, DatasetDiagnostics]:
        """Fit diagnostics keyed by dataset name, in solve order."""
        names = self.dataset_names or tuple(
            f"data_{index + 1}"
            for index in range(len(self.details.dataset_diagnostics))
        )
        return dict(zip(names, self.details.dataset_diagnostics))

    def _by_dataset(
        self, stacked: np.ndarray, dataset: DataSet | str | None
    ) -> dict[str, np.ndarray] | np.ndarray:
        """Split a stacked row vector by dataset, or select one dataset's rows."""
        names = self.dataset_names or ("data",)
        slices = self.details.dataset_slices or (slice(None),)
        if dataset is None:
            return {name: stacked[rows] for name, rows in zip(names, slices)}
        if isinstance(dataset, str):
            if dataset not in names:
                raise KeyError(f"no dataset named {dataset!r}; names are {names}")
            return stacked[slices[names.index(dataset)]]
        for candidate, rows in zip(self.datasets, slices):
            if candidate is dataset:
                return stacked[rows]
        raise ValueError(
            "that dataset is not one of the datasets this result was fit to"
        )

    def summary(self) -> str:
        """Format the essential assumptions and fit statistics as text."""
        regularization = "none"
        if self.regularization is not None:
            kind = (
                self.regularization
                if isinstance(self.regularization, str)
                else "custom"
            )
            regularization = f"{kind} (lambda={self.regularization_strength:.6g})"
            if self.details.regularization_selection is not None:
                selection = self.details.regularization_selection
                regularization += f", selected by {selection}"
        lines = [
            f"solver: {self.details.solver} "
            f"({'success' if self.details.success else 'failed'})",
            f"datasets: {', '.join(self.dataset_names) or 'data'}",
            f"quantity: {self.quantity} [{self.units}]",
            f"components: {self.components}",
            f"regularization: {regularization}",
            f"reduced chi-squared: {self.reduced_chi2:.6g}",
            f"RMS: {self.rms:.6g} {self.units}",
            f"backend: {self.details.backend}/{self.details.precision}",
        ]
        for name, values in self.diagnostics().items():
            lines.append(
                f"{name}: n={values.n_obs}, reduced chi-squared="
                f"{values.reduced_chi2:.6g}, RMS={values.rms:.6g} {self.units}"
            )
        lines.extend(f"warning: {warning}" for warning in self.warnings)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Assessment: covariance, uncertainty, resolution
    # ------------------------------------------------------------------

    def covariance(self, kind: str = "posterior") -> np.ndarray:
        """Model covariance matrix ``C_m``, shape (n_params, n_params).

        Without regularization both kinds reduce to ``(G^T W G)^{-1}``.
        With regularization, ``H = G^T W G + lambda L^T L`` and:

        - ``kind='posterior'``: the linear-Gaussian posterior covariance
          ``H^{-1}``.
        - ``kind='estimator'``: the covariance of the penalized estimate
          under data noise alone, ``H^{-1} G^T W G H^{-1}``.

        Args:
            kind: ``'posterior'`` (default) or ``'estimator'``.
        """
        return self._system().model_covariance(self, kind=kind)

    def uncertainty(self, kind: str = "posterior") -> np.ndarray:
        """One-sigma uncertainty per parameter: ``sqrt(diag(covariance))``.

        Args:
            kind: Covariance kind; see :meth:`covariance`.
        """
        return self._system().model_uncertainty(self, kind=kind)

    def resolution(self) -> np.ndarray:
        """Model resolution matrix ``R = H^{-1} G^T W G``.

        ``R`` is the identity for a perfectly resolved model; diagonal
        values below one mark parameters that regularization smooths or
        damps, and each row is that parameter's averaging kernel.
        """
        return self._system().model_resolution(self)

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------

    def save(self, fname: str | Path) -> None:
        """Save the solution arrays and metadata to a ``.npz`` file.

        Args:
            fname: Output path.
        """
        from geodef.invert import _io

        _io.save(self, fname)

    @classmethod
    def load(
        cls,
        fname: str | Path,
        *,
        fault: Fault | None = None,
        datasets: DataSet | list[DataSet] | None = None,
    ) -> InversionResult:
        """Load a result saved by :meth:`save`.

        Args:
            fname: Path to the ``.npz`` file.
            fault: Optional fault geometry; together with ``datasets`` it
                re-attaches the result so assessment methods work.
            datasets: Optional dataset(s) used for the inversion.

        Returns:
            The loaded result.
        """
        from geodef.invert import _io

        result = _io.load(fname)
        if fault is not None and datasets is not None:
            return result.attach(fault, datasets)
        return result

    def save_table(self, fname: str | Path, fault: Fault | None = None) -> None:
        """Save the slip distribution as a human-readable per-patch table.

        Args:
            fname: Output text path.
            fault: Fault geometry; defaults to the attached fault.
        """
        from geodef.invert import _io

        _io.save_table(self, fname, self.fault if fault is None else fault)

    # ------------------------------------------------------------------
    # Slip views
    # ------------------------------------------------------------------

    @property
    def slip(self) -> np.ndarray:
        """Solution in the solved basis as one column per component, (N, P/N)."""
        n_columns = 2 if self.components in {"both", "plate"} else 1
        return self.slip_vector.reshape(n_columns, -1).T

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
            angle = self.details.local_rake
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
