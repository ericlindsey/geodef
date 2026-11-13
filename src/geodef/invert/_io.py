"""Inversion-result files: one ``.npz`` holding arrays and JSON metadata.

Private submodule of :mod:`geodef.invert`, reached through
``InversionResult.save`` / ``InversionResult.load`` / ``save_table``.
Arrays are stored as plain numeric arrays (never pickled); everything else
is one JSON string under the ``metadata`` key, readable with
``json.loads(str(np.load(path)["metadata"]))``.
"""

import dataclasses
from pathlib import Path

import numpy as np
import orjson

from geodef.fault import Fault
from geodef.invert._results import DatasetDiagnostics, InversionResult, SolveDetails

_ARRAYS = ("slip_vector", "residuals", "predicted", "plate_rake")
_DETAIL_ARRAYS = (
    "local_rake",
    "lower_bounds",
    "upper_bounds",
    "regularization_target",
    "constraint_matrix",
    "constraint_bounds",
)


def _number(value: float | None) -> float | str | None:
    """Encode a float so NaN and infinity survive JSON."""
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else str(value)


def save(result: InversionResult, fname: str | Path) -> None:
    """Save a result to one ``.npz`` file.

    Args:
        result: Inversion result to write.
        fname: Output path; ``.npz`` is appended by NumPy if missing.
    """
    details = result.details
    arrays: dict[str, np.ndarray] = {
        name: np.asarray(getattr(result, name), dtype=float)
        for name in _ARRAYS
        if getattr(result, name) is not None
    }
    arrays.update(
        {
            f"details.{name}": np.asarray(getattr(details, name), dtype=float)
            for name in _DETAIL_ARRAYS
            if getattr(details, name) is not None
        }
    )
    regularization = result.regularization
    if isinstance(regularization, np.ndarray):
        arrays["regularization"] = np.asarray(regularization, dtype=float)
        regularization = "custom"
    metadata = {
        "regularization": regularization,
        "regularization_strength": _number(result.regularization_strength),
        "reduced_chi2": _number(result.reduced_chi2),
        "rms": _number(result.rms),
        "moment": _number(result.moment),
        "Mw": _number(result.Mw),
        "components": result.components,
        "rake": _number(result.rake),
        "slip_azimuth": _number(result.slip_azimuth),
        "dataset_names": list(result.dataset_names),
        "warnings": list(result.warnings),
        "quantity": result.quantity,
        "units": result.units,
        "details": {
            "solver": details.solver,
            "success": details.success,
            "message": details.message,
            "regularization_selection": details.regularization_selection,
            "backend": details.backend,
            "precision": details.precision,
            "system_hash": details.system_hash,
            "dataset_slices": [[r.start, r.stop] for r in details.dataset_slices],
            "dataset_diagnostics": [
                {k: _number(v) for k, v in dataclasses.asdict(d).items()}
                for d in details.dataset_diagnostics
            ],
        },
    }
    arrays["metadata"] = np.array(orjson.dumps(metadata).decode())
    np.savez(Path(fname), allow_pickle=False, **arrays)


def _float(value: float | str | None) -> float | None:
    return None if value is None else float(value)


def load(fname: str | Path) -> InversionResult:
    """Load a result written by :func:`save`.

    The result is not attached to a fault or datasets; see
    ``InversionResult.load`` and ``InversionResult.attach``.

    Args:
        fname: Path to the ``.npz`` file.

    Returns:
        The loaded result.
    """
    with np.load(Path(fname), allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    metadata = orjson.loads(str(arrays.pop("metadata")))
    detail_metadata = metadata.pop("details")
    details = SolveDetails(
        solver=detail_metadata["solver"],
        success=detail_metadata["success"],
        message=detail_metadata["message"],
        regularization_selection=detail_metadata["regularization_selection"],
        backend=detail_metadata["backend"],
        precision=detail_metadata["precision"],
        system_hash=detail_metadata["system_hash"],
        dataset_slices=tuple(
            slice(start, stop) for start, stop in detail_metadata["dataset_slices"]
        ),
        dataset_diagnostics=tuple(
            DatasetDiagnostics(
                chi2=float(values["chi2"]),
                reduced_chi2=float(values["reduced_chi2"]),
                wrms=float(values["wrms"]),
                rms=float(values["rms"]),
                n_obs=int(values["n_obs"]),
                dof=float(values["dof"]),
                leverage=float(values["leverage"]),
            )
            for values in detail_metadata["dataset_diagnostics"]
        ),
        **{name: arrays.get(f"details.{name}") for name in _DETAIL_ARRAYS},
    )
    regularization = metadata["regularization"]
    if regularization == "custom":
        regularization = arrays["regularization"]
    return InversionResult(
        slip_vector=arrays["slip_vector"],
        residuals=arrays["residuals"],
        predicted=arrays["predicted"],
        plate_rake=arrays.get("plate_rake"),
        reduced_chi2=float(metadata["reduced_chi2"]),
        rms=float(metadata["rms"]),
        moment=float(metadata["moment"]),
        Mw=float(metadata["Mw"]),
        regularization=regularization,
        regularization_strength=_float(metadata["regularization_strength"]),
        components=metadata["components"],
        rake=_float(metadata["rake"]),
        slip_azimuth=_float(metadata["slip_azimuth"]),
        dataset_names=tuple(metadata["dataset_names"]),
        warnings=tuple(metadata["warnings"]),
        quantity=metadata["quantity"],
        units=metadata["units"],
        details=details,
    )


def save_table(result: InversionResult, fname: str | Path, fault: Fault) -> None:
    """Save a slip distribution as a human-readable per-patch table.

    Args:
        result: Inversion result to write.
        fname: Output text path.
        fault: Matching fault geometry.

    Raises:
        ValueError: If result and fault patch counts differ.
    """
    if result.n_patches != fault.n_patches:
        raise ValueError(
            f"result has {result.n_patches} patches but fault has {fault.n_patches}"
        )
    slip_2d = result.slip
    regularization_desc = (
        "none"
        if result.regularization is None
        else (
            result.regularization
            if isinstance(result.regularization, str)
            else "custom"
        )
    )
    strength_desc = (
        "N/A"
        if result.regularization_strength is None
        else f"{result.regularization_strength:.6g}"
    )
    header_lines = [
        "geodef InversionResult",
        f"components: {result.components}",
        f"regularization: {regularization_desc}, strength: {strength_desc}",
        f"reduced_chi2: {result.reduced_chi2:.6g}",
        f"rms: {result.rms:.6g} {result.units}",
        f"moment: {result.moment:.6g} N-m",
        f"Mw: {result.Mw:.4f}",
    ]
    if result.rake is not None:
        header_lines.append(f"rake_deg: {result.rake:.6g}")
    if result.slip_azimuth is not None:
        header_lines.append(f"slip_azimuth_deg: {result.slip_azimuth:.6g}")

    if fault.engine == "okada":
        if fault.length is None or fault.width is None:
            raise ValueError("rectangular fault is missing patch dimensions")
        column_names = "lon lat depth_m strike dip length_m width_m"
        geometry = np.column_stack(
            [
                fault.lon,
                fault.lat,
                fault.depth,
                fault.geographic_strike,
                fault.dip,
                fault.length,
                fault.width,
            ]
        )
    else:
        column_names = "lon lat depth_m strike dip area_m2"
        geometry = np.column_stack(
            [
                fault.lon,
                fault.lat,
                fault.depth,
                fault.geographic_strike,
                fault.dip,
                fault.areas,
            ]
        )

    if result.components == "both":
        slip_columns = ["slip_strike_m", "slip_dip_m"]
    elif result.components == "strike":
        slip_columns = ["slip_strike_m"]
    elif result.components == "dip":
        slip_columns = ["slip_dip_m"]
    elif result.components in {"rake", "azimuth"}:
        slip_columns = ["slip_amplitude_m"]
    else:
        slip_columns = ["slip_rake_parallel_m", "slip_rake_perpendicular_m"]
    header_lines.append(f"{column_names}  {'  '.join(slip_columns)}")
    np.savetxt(
        Path(fname),
        np.column_stack([geometry, slip_2d]),
        header="\n".join(header_lines),
        fmt="%.6f",
    )
