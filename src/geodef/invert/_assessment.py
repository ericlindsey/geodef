"""Post-inversion assessment: predictions, diagnostics, and model metrics.

Private submodule of :mod:`geodef.invert`. These free functions consume a
compact ``InversionResult`` (or the raw system matrices) so results stay
serializable records rather than workflow facades.
"""

import numpy as np

from geodef.data import DataSet
from geodef.fault import Fault
from geodef.invert._results import DatasetDiagnostics, InversionResult
from geodef.invert._system import LinearSystem


def prediction(result: InversionResult) -> dict[str, np.ndarray]:
    """Split stacked model predictions by dataset name.

    Args:
        result: Inversion result from :func:`solve`.

    Returns:
        Name-keyed prediction arrays in solve order.
    """
    if not result.dataset_names:
        return {"data": result.predicted}
    return {
        name: result.predicted[row_slice]
        for name, row_slice in zip(result.dataset_names, result.dataset_slices)
    }


def residual(result: InversionResult) -> dict[str, np.ndarray]:
    """Split stacked observation-minus-prediction residuals by dataset name.

    Args:
        result: Inversion result from :func:`solve`.

    Returns:
        Name-keyed residual arrays in solve order.
    """
    if not result.dataset_names:
        return {"data": result.residuals}
    return {
        name: result.residuals[row_slice]
        for name, row_slice in zip(result.dataset_names, result.dataset_slices)
    }


def diagnostics(result: InversionResult) -> dict[str, DatasetDiagnostics]:
    """Return stored fit diagnostics keyed by dataset name.

    Args:
        result: Inversion result from :func:`solve`.

    Returns:
        Name-keyed per-dataset diagnostics in solve order.
    """
    names = result.dataset_names or tuple(
        f"data_{index + 1}" for index in range(len(result.dataset_diagnostics))
    )
    return dict(zip(names, result.dataset_diagnostics))


def summary(result: InversionResult) -> str:
    """Format the essential assumptions and fit statistics as plain text.

    Args:
        result: Inversion result from :func:`solve`.

    Returns:
        Multi-line human-readable summary.
    """
    regularization = "none"
    if result.regularization is not None:
        regularization_name = (
            result.regularization
            if isinstance(result.regularization, str)
            else "custom"
        )
        regularization = (
            f"{regularization_name} (lambda={result.regularization_strength:.6g})"
        )
        if result.regularization_selection is not None:
            regularization += f", selected by {result.regularization_selection}"
    lines = [
        f"solver: {result.solver} ({'success' if result.success else 'failed'})",
        f"datasets: {', '.join(result.dataset_names) or 'data'}",
        f"quantity: {result.quantity} [{result.units}]",
        f"components: {result.components}",
        f"regularization: {regularization}",
        f"reduced chi-squared: {result.reduced_chi2:.6g}",
        f"RMS: {result.rms:.6g} {result.units}",
        f"backend: {result.backend}/{result.precision}",
    ]
    for name, values in diagnostics(result).items():
        lines.append(
            f"{name}: n={values.n_obs}, reduced chi-squared="
            f"{values.reduced_chi2:.6g}, RMS={values.rms:.6g} {result.units}"
        )
    lines.extend(f"warning: {warning}" for warning in result.warnings)
    return "\n".join(lines)


def model_covariance(
    result: InversionResult,
    fault: Fault,
    datasets: DataSet | list[DataSet],
    kind: str = "posterior",
) -> np.ndarray:
    """Compute the model covariance matrix.

    For the unregularized case both kinds reduce to
    ``Cm = (G^T W G)^{-1}``. For the regularized case, with
    ``H = G^T W G + lambda L^T L`` (see docs/conventions.md):

    - ``kind='posterior'`` (default) — the linear-Gaussian posterior
      covariance ``Cm = H^{-1}``.
    - ``kind='estimator'`` — the frequentist covariance of the penalized
      estimator under data noise alone (Tarantola, 2005),
      ``Cm = H^{-1} G^T W G H^{-1}``.

    Args:
        result: Output from ``invert()``.
        fault: Fault geometry.
        datasets: Dataset(s) used in the inversion.
        kind: ``'posterior'`` (default) or ``'estimator'``.

    Returns:
        Model covariance matrix, shape (n_params, n_params).
    """
    sys = LinearSystem(
        fault,
        datasets,
        result.regularization,
        result.components,
        result.rake,
        result.slip_azimuth,
        result.plate_rake,
    )
    return sys.model_covariance(result, kind=kind)


def model_resolution(
    result: InversionResult,
    fault: Fault,
    datasets: DataSet | list[DataSet],
) -> np.ndarray:
    """Compute the model resolution matrix.

    ``R = (G^T W G + lambda L^T L)^{-1} G^T W G``

    For perfect resolution (overdetermined, no regularization), R = I.
    With regularization, diagonal values < 1 indicate smoothed/damped
    parameters.

    Args:
        result: Output from ``invert()``.
        fault: Fault geometry.
        datasets: Dataset(s) used in the inversion.

    Returns:
        Resolution matrix, shape (n_params, n_params).
    """
    sys = LinearSystem(
        fault,
        datasets,
        result.regularization,
        result.components,
        result.rake,
        result.slip_azimuth,
        result.plate_rake,
    )
    return sys.model_resolution(result)


def model_uncertainty(
    result: InversionResult,
    fault: Fault,
    datasets: DataSet | list[DataSet],
    kind: str = "posterior",
) -> np.ndarray:
    """Compute per-parameter 1-sigma uncertainty from model covariance.

    Equivalent to ``np.sqrt(np.diag(model_covariance(...)))``.

    Args:
        result: Output from ``invert()``.
        fault: Fault geometry.
        datasets: Dataset(s) used in the inversion.
        kind: Covariance kind, ``'posterior'`` (default) or
            ``'estimator'``; see :func:`model_covariance`.

    Returns:
        Uncertainty array, shape (n_params,).
    """
    sys = LinearSystem(
        fault,
        datasets,
        result.regularization,
        result.components,
        result.rake,
        result.slip_azimuth,
        result.plate_rake,
    )
    return sys.model_uncertainty(result, kind=kind)
