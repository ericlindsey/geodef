"""Fit and diagnostic plots for inversion results.

Private submodule of :mod:`geodef.plot`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from geodef.plot._shared import (
    _ensure_axes,
)

if TYPE_CHECKING:
    import matplotlib.axes

    from geodef.invert import InversionResult


def fit(
    observed: np.ndarray,
    predicted: np.ndarray,
    *,
    ax: matplotlib.axes.Axes | None = None,
    style: str = "scatter",
    scatter_kwargs: dict | None = None,
    title: str | None = None,
) -> matplotlib.axes.Axes:
    """Plot observed vs. predicted values.

    Args:
        observed: Observed data vector.
        predicted: Predicted data vector (same length).
        ax: Axes to plot on. Creates a new figure if ``None``.
        style: ``'scatter'`` for obs-vs-pred with 1:1 line, or
            ``'residual_histogram'`` for a histogram of residuals.
        scatter_kwargs: Extra kwargs passed to ``ax.scatter()`` (scatter
            style only).
        title: Axes title.

    Returns:
        The axes used for plotting.

    Raises:
        ValueError: If *style* is invalid.
    """
    ax = _ensure_axes(ax)

    if style == "scatter":
        skw: dict[str, Any] = {"s": 10, "alpha": 0.7, "edgecolors": "none"}
        if scatter_kwargs:
            skw.update(scatter_kwargs)
        ax.scatter(observed, predicted, **skw)
        lo = min(np.min(observed), np.min(predicted))
        hi = max(np.max(observed), np.max(predicted))
        margin = (hi - lo) * 0.05
        ax.plot(
            [lo - margin, hi + margin],
            [lo - margin, hi + margin],
            "k--",
            linewidth=0.8,
            label="1:1",
        )
        ax.set_xlabel("Observed")
        ax.set_ylabel("Predicted")
        ax.set_title(title or "Observed vs. Predicted")

    elif style == "residual_histogram":
        residuals = observed - predicted
        ax.hist(residuals, bins="auto", edgecolor="black", linewidth=0.5)
        ax.set_xlabel("Residual (obs - pred)")
        ax.set_ylabel("Count")
        ax.set_title(title or "Residual histogram")

    else:
        raise ValueError(
            f"Unknown style {style!r}. Use 'scatter' or 'residual_histogram'."
        )

    return ax


def prediction(
    result: InversionResult,
    *,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.axes.Axes:
    """Plot observed values against predictions for every named dataset.

    Args:
        result: Inversion result containing named dataset partitions.
        ax: Axes to plot on. Creates a new figure if ``None``.

    Returns:
        The axes used for plotting.
    """
    from geodef.invert import prediction as split_prediction
    from geodef.invert import residual as split_residual

    ax = _ensure_axes(ax)
    predicted = split_prediction(result)
    residuals = split_residual(result)
    for name in predicted:
        observed = predicted[name] + residuals[name]
        ax.scatter(observed, predicted[name], s=12, alpha=0.7, label=name)
    all_observed = result.predicted + result.residuals
    lower = float(min(np.min(all_observed), np.min(result.predicted)))
    upper = float(max(np.max(all_observed), np.max(result.predicted)))
    margin = 0.05 * (upper - lower) if upper > lower else 1.0
    ax.plot(
        [lower - margin, upper + margin],
        [lower - margin, upper + margin],
        "k--",
        linewidth=0.8,
    )
    ax.set_xlabel(f"Observed [{result.units}]")
    ax.set_ylabel(f"Predicted [{result.units}]")
    ax.set_title("Observed vs. predicted")
    ax.legend()
    return ax


def residual(
    result: InversionResult,
    *,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.axes.Axes:
    """Plot residual distributions for every named dataset.

    Args:
        result: Inversion result containing named dataset partitions.
        ax: Axes to plot on. Creates a new figure if ``None``.

    Returns:
        The axes used for plotting.
    """
    from geodef.invert import residual as split_residual

    ax = _ensure_axes(ax)
    for name, values in split_residual(result).items():
        ax.hist(values, bins="auto", alpha=0.55, label=name)
    ax.set_xlabel(f"Observation - prediction [{result.units}]")
    ax.set_ylabel("Count")
    ax.set_title("Residuals")
    ax.legend()
    return ax


def diagnostics(
    result: InversionResult,
    *,
    ax: matplotlib.axes.Axes | None = None,
    metric: str = "reduced_chi2",
) -> matplotlib.axes.Axes:
    """Compare one stored fit diagnostic across named datasets.

    Args:
        result: Inversion result containing per-dataset diagnostics.
        ax: Axes to plot on. Creates a new figure if ``None``.
        metric: ``'reduced_chi2'``, ``'chi2'``, ``'rms'``, or ``'wrms'``.

    Returns:
        The axes used for plotting.

    Raises:
        ValueError: If ``metric`` is not supported.
    """
    from geodef.invert import diagnostics as result_diagnostics

    labels = {
        "reduced_chi2": "Reduced chi-squared",
        "chi2": "Chi-squared",
        "rms": f"RMS [{result.units}]",
        "wrms": "Weighted RMS",
    }
    if metric not in labels:
        raise ValueError(f"metric must be one of {sorted(labels)}, got {metric!r}")
    values = result_diagnostics(result)
    ax = _ensure_axes(ax)
    ax.bar(list(values), [getattr(value, metric) for value in values.values()])
    ax.set_ylabel(labels[metric])
    ax.set_title("Dataset diagnostics")
    return ax


def summary(
    result: InversionResult,
    *,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.axes.Axes:
    """Render the plain-text inversion summary on an axes.

    Args:
        result: Inversion result to summarize.
        ax: Axes to plot on. Creates a new figure if ``None``.

    Returns:
        The axes used for plotting.
    """
    from geodef.invert import summary as format_summary

    ax = _ensure_axes(ax)
    ax.text(
        0.02,
        0.98,
        format_summary(result),
        ha="left",
        va="top",
        family="monospace",
        transform=ax.transAxes,
    )
    ax.set_axis_off()
    ax.set_title("Inversion summary")
    return ax
