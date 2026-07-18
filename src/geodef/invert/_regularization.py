"""Regularization operators for the augmented least-squares system.

Private submodule of :mod:`geodef.invert`. Builds the matrix ``L`` and
the right-hand side of the ``sqrt(lambda) * L`` rows appended to the
whitened system (convention: ``Phi = r^T W r + lambda ||L(m - m_ref)||^2``).
"""

import numpy as np
import scipy.linalg

from geodef.fault import Fault
from geodef.greens import select_slip_columns

_VALID_REGULARIZATION_STRINGS = {"laplacian", "damping", "stresskernel"}


def _build_regularization_matrix(
    fault: Fault,
    regularization: str | np.ndarray,
    n_params: int,
    n_components: int,
    components: str,
    rake: float | None = None,
    slip_azimuth: float | None = None,
    plate_rake: np.ndarray | None = None,
) -> np.ndarray:
    """Build the regularization matrix L.

    Args:
        fault: Fault geometry.
        regularization: Regularization type or custom matrix.
        n_params: Number of model parameters (n_components * n_patches).
        n_components: Number of slip components (1 or 2).
        components: Active slip parameterization.
        rake: Fixed rake angle, used when ``components='rake'``.
        slip_azimuth: Fixed geographic slip azimuth, used when
            ``components='azimuth'``.
        plate_rake: Per-patch plate rake, used when ``components='plate'``.

    Returns:
        Regularization matrix with n_params columns.
    """
    if isinstance(regularization, np.ndarray):
        return regularization

    if regularization == "damping":
        return np.eye(n_params)

    if regularization == "laplacian":
        L_patch = fault.laplacian
        if n_components == 1:
            return L_patch
        return scipy.linalg.block_diag(L_patch, L_patch)

    if regularization == "stresskernel":
        K = fault.stress_kernel()
        return select_slip_columns(
            K,
            fault.n_patches,
            components,
            rake,
            fault_strike=fault.strike,
            slip_azimuth=slip_azimuth,
            plate_rake=plate_rake,
        )

    raise ValueError(f"Unknown regularization type: {regularization!r}")


def _build_reg_rhs(
    L: np.ndarray,
    regularization_strength: float,
    regularization_target: np.ndarray | None,
) -> np.ndarray:
    """Build the right-hand side for the regularization rows.

    For standard regularization (target=None): zeros.
    For target regularization: sqrt(lambda) * L @ m_ref.
    """
    if regularization_target is None:
        return np.zeros(L.shape[0])
    return np.sqrt(regularization_strength) * (L @ regularization_target)
