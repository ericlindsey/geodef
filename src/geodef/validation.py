"""Physical input validation: fail-early helpers and interactive reports.

Two layers:

- **Constructor checks** (``as_1d_floats``, ``check_range``,
  ``check_positive``, ``check_covariance``) raise ``ValueError`` naming the
  offending argument, its received shape or range, and the expected unit.
  Public constructors use these so invalid physics fails at creation, not
  as a mysterious downstream result.
- **Interactive reports**: ``Fault.validate()``, ``DataSet.validate()``,
  and ``Mesh.validate()`` return a :class:`ValidationReport` of
  :class:`ValidationIssue` entries — errors for physically invalid setups
  and warnings for suspicious-but-legal ones — for notebook workflows.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import numpy.typing as npt

__all__ = [
    "ValidationIssue",
    "ValidationReport",
    "as_1d_floats",
    "check_finite_scalar",
    "check_covariance",
    "check_positive",
    "check_range",
]


def _unit_suffix(unit: str | None) -> str:
    return f" (expected unit: {unit})" if unit else ""


def as_1d_floats(
    name: str,
    value: npt.ArrayLike,
    *,
    n: int | None = None,
    unit: str | None = None,
) -> np.ndarray:
    """Coerce to a finite 1-D float array or raise a precise ``ValueError``.

    Args:
        name: Argument name used in error messages.
        value: Array-like input.
        n: Required length, if known.
        unit: Physical unit named in error messages.

    Returns:
        The validated ``float64`` array.

    Raises:
        ValueError: If the input is not 1-D, has the wrong length, or
            contains NaN/inf values.
    """
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 1:
        raise ValueError(
            f"{name} must be a 1-D array, got shape {arr.shape}{_unit_suffix(unit)}"
        )
    if n is not None and arr.shape[0] != n:
        raise ValueError(
            f"{name} must be 1-D with the same length as the other inputs: "
            f"expected length {n}, got {arr.shape[0]}"
        )
    if not np.all(np.isfinite(arr)):
        bad = np.flatnonzero(~np.isfinite(arr))
        raise ValueError(
            f"{name} contains non-finite values at indices "
            f"{bad[:5].tolist()}{'...' if bad.size > 5 else ''}"
            f"{_unit_suffix(unit)}"
        )
    return arr


def check_range(
    name: str,
    value: npt.ArrayLike,
    lo: float,
    hi: float,
    *,
    unit: str | None = None,
) -> None:
    """Require every element of ``value`` to lie in ``[lo, hi]``.

    Raises:
        ValueError: Naming the argument, offending range, and unit.
    """
    arr = np.asarray(value, dtype=float)
    if arr.size and (np.nanmin(arr) < lo or np.nanmax(arr) > hi):
        raise ValueError(
            f"{name} must lie in [{lo:g}, {hi:g}]"
            f"{f' {unit}' if unit else ''}; got values in "
            f"[{np.nanmin(arr):g}, {np.nanmax(arr):g}]"
        )


def check_positive(
    name: str,
    value: npt.ArrayLike,
    *,
    unit: str | None = None,
) -> None:
    """Require every element of ``value`` to be strictly positive and finite.

    Raises:
        ValueError: Naming the argument and unit.
    """
    arr = np.asarray(value, dtype=float)
    if arr.size and (not np.all(np.isfinite(arr)) or np.min(arr) <= 0):
        raise ValueError(
            f"{name} must be positive and finite{_unit_suffix(unit)}; "
            f"got minimum {np.min(arr):g}"
        )


def check_covariance(
    cov: npt.ArrayLike,
    n: int,
    *,
    name: str = "covariance",
    require_positive_definite: bool = True,
) -> np.ndarray:
    """Validate a covariance matrix: shape, symmetry, positive definiteness.

    Args:
        cov: Candidate covariance matrix.
        n: Required dimension (``n_obs``).
        name: Argument name used in error messages.
        require_positive_definite: Set ``False`` only for advanced
            semidefinite/operator cases; symmetry and shape are still
            enforced.

    Returns:
        The matrix as a float array.

    Raises:
        ValueError: With the failed property and a remediation hint.
    """
    arr = np.asarray(cov, dtype=float)
    if arr.shape != (n, n):
        raise ValueError(
            f"{name} shape {arr.shape} does not match expected ({n}, {n}) "
            "(n_obs by n_obs)"
        )
    asym = float(np.max(np.abs(arr - arr.T))) if n else 0.0
    scale = float(np.max(np.abs(arr))) if n else 0.0
    if asym > 1e-10 * max(scale, 1e-300):
        raise ValueError(
            f"{name} is not symmetric: max |C - C^T| = {asym:.3g}. "
            "Symmetrize with 0.5 * (C + C.T) if the asymmetry is roundoff."
        )
    if require_positive_definite and n:
        try:
            np.linalg.cholesky(arr + 0.0)
        except np.linalg.LinAlgError:
            # Smooth spatial kernels are often PSD within roundoff; only a
            # genuinely negative spectrum is a user error.
            eigs = np.linalg.eigvalsh(arr)
            lo, hi = float(eigs.min()), float(eigs.max())
            if lo < -1e-10 * max(hi, 1e-300):
                raise ValueError(
                    f"{name} is not positive (semi)definite: smallest "
                    f"eigenvalue {lo:.3g}. Check for sign errors or "
                    "duplicated observations; add a small diagonal "
                    "(nugget) term, or pass validate_covariance=False for "
                    "an advanced operator model."
                ) from None
    return arr


@dataclasses.dataclass(frozen=True)
class ValidationIssue:
    """One finding from an interactive ``validate()`` call.

    Attributes:
        severity: ``"error"`` (physically invalid) or ``"warning"``
            (legal but suspicious).
        field: Name of the input or property the issue concerns.
        message: Human-readable description with values and units.
    """

    severity: str
    field: str
    message: str

    def __str__(self) -> str:
        return f"{self.severity.upper():7s} {self.field}: {self.message}"


@dataclasses.dataclass(frozen=True)
class ValidationReport:
    """Collected findings from a ``validate()`` call.

    Attributes:
        issues: All findings, errors first.
    """

    issues: tuple[ValidationIssue, ...]

    @property
    def ok(self) -> bool:
        """True when no error-severity issues were found."""
        return self.n_errors == 0

    @property
    def n_errors(self) -> int:
        """Number of error-severity issues."""
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def n_warnings(self) -> int:
        """Number of warning-severity issues."""
        return sum(1 for i in self.issues if i.severity == "warning")

    def raise_if_errors(self) -> None:
        """Raise ``ValueError`` summarizing all error-severity issues."""
        if not self.ok:
            errors = "; ".join(
                f"{i.field}: {i.message}" for i in self.issues if i.severity == "error"
            )
            raise ValueError(f"validation failed: {errors}")

    def __str__(self) -> str:
        if not self.issues:
            return "ValidationReport: ok (no issues)"
        lines = [
            f"ValidationReport: {self.n_errors} error(s), {self.n_warnings} warning(s)"
        ]
        lines += [f"  {issue}" for issue in self.issues]
        return "\n".join(lines)


class _ReportBuilder:
    """Internal accumulator used by the validate() implementations."""

    def __init__(self) -> None:
        self._issues: list[ValidationIssue] = []

    def error(self, field: str, message: str) -> None:
        self._issues.append(ValidationIssue("error", field, message))

    def warning(self, field: str, message: str) -> None:
        self._issues.append(ValidationIssue("warning", field, message))

    def report(self) -> ValidationReport:
        ordered = sorted(self._issues, key=lambda i: i.severity != "error")
        return ValidationReport(issues=tuple(ordered))


def check_finite_scalar(name: str, value: float, *, unit: str | None = None) -> float:
    """Require a finite scalar; return it as ``float``.

    Raises:
        ValueError: Naming the argument and unit.
    """
    x = float(value)
    if not math.isfinite(x):
        raise ValueError(f"{name} must be finite, got {value!r}{_unit_suffix(unit)}")
    return x
