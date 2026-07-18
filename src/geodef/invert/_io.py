"""Versioned result-file I/O: safe npz archives plus JSON manifests.

Private submodule of :mod:`geodef.invert`. ``save``/``load`` write and
read the versioned archive format (schema history in
``_SUPPORTED_SCHEMA_VERSIONS``); ``save_table`` writes the human-readable
per-patch table. Numeric arrays never use pickle or object dtypes.
"""

import hashlib
import importlib.metadata
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import numpy as np
import orjson

from geodef.fault import Fault
from geodef.invert._results import DatasetDiagnostics, InversionResult

RESULT_SCHEMA_VERSION = 3


_RESULT_SCHEMA = "geodef.inversion_result"


# Schema versions this build can still read. Version 2 named the
# regularization fields ``smoothing*``; version 3 renamed them to
# ``regularization*`` and its keys are migrated on load.
_SUPPORTED_SCHEMA_VERSIONS = frozenset({2, 3})


def _manifest_path(path: Path) -> Path:
    """Return the readable sidecar path for a result archive."""
    return path.with_suffix(".manifest.json")


def _json_float(value: float) -> float | str:
    """Encode non-finite floats without non-standard JSON numbers."""
    value = float(value)
    if np.isnan(value):
        return "nan"
    if np.isposinf(value):
        return "inf"
    if np.isneginf(value):
        return "-inf"
    return value


def _manifest_float(value: object) -> float:
    """Decode a finite or explicitly tagged manifest float."""
    if value == "nan":
        return float("nan")
    if value == "inf":
        return float("inf")
    if value == "-inf":
        return float("-inf")
    if not isinstance(value, (int, float)):
        raise ValueError(f"expected a numeric manifest value, got {value!r}")
    return float(value)


def _array_checksum(array: np.ndarray) -> str:
    """Return a SHA-256 checksum over an array's stored bytes."""
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _array_manifest(array: np.ndarray) -> dict[str, object]:
    """Describe one numeric archive array for safe loading."""
    return {
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "sha256": _array_checksum(array),
    }


def _result_arrays(result: InversionResult) -> dict[str, np.ndarray]:
    """Collect numeric result arrays without object dtypes."""
    arrays = {
        "slip": np.asarray(result.slip),
        "slip_vector": np.asarray(result.slip_vector),
        "residuals": np.asarray(result.residuals),
        "predicted": np.asarray(result.predicted),
    }
    optional = {
        "plate_rake": result.plate_rake,
        "local_rake": result.local_rake,
        "lower_bounds": result.lower_bounds,
        "upper_bounds": result.upper_bounds,
        "regularization_target": result.regularization_target,
        "constraint_matrix": result.constraint_matrix,
        "constraint_bounds": result.constraint_bounds,
    }
    if isinstance(result.regularization, np.ndarray):
        optional["regularization"] = result.regularization
    for name, value in optional.items():
        if value is not None:
            arrays[name] = np.asarray(value)
    for name, array in arrays.items():
        if array.dtype.hasobject:
            raise TypeError(f"result array {name!r} must not have object dtype")
    return arrays


def _diagnostic_manifest(values: DatasetDiagnostics) -> dict[str, object]:
    """Encode one diagnostics record for JSON."""
    return {
        "chi2": _json_float(values.chi2),
        "reduced_chi2": _json_float(values.reduced_chi2),
        "wrms": _json_float(values.wrms),
        "rms": _json_float(values.rms),
        "n_obs": values.n_obs,
        "dof": _json_float(values.dof),
        "leverage": _json_float(values.leverage),
    }


def _build_manifest(
    result: InversionResult,
    arrays: Mapping[str, np.ndarray],
) -> dict[str, object]:
    """Build the versioned result manifest."""
    regularization: str | None
    if isinstance(result.regularization, np.ndarray):
        regularization = "__array__"
    else:
        regularization = result.regularization
    result_metadata: dict[str, object] = {
        "reduced_chi2": _json_float(result.reduced_chi2),
        "rms": _json_float(result.rms),
        "moment": _json_float(result.moment),
        "Mw": _json_float(result.Mw),
        "regularization": regularization,
        "regularization_strength": (
            None
            if result.regularization_strength is None
            else _json_float(result.regularization_strength)
        ),
        "components": result.components,
        "rake": None if result.rake is None else _json_float(result.rake),
        "slip_azimuth": (
            None if result.slip_azimuth is None else _json_float(result.slip_azimuth)
        ),
        "dataset_names": list(result.dataset_names),
        "dataset_slices": [
            [row_slice.start, row_slice.stop] for row_slice in result.dataset_slices
        ],
        "solver": result.solver,
        "success": result.success,
        "message": result.message,
        "regularization_selection": result.regularization_selection,
        "backend": result.backend,
        "precision": result.precision,
        "warnings": list(result.warnings),
        "quantity": result.quantity,
        "units": result.units,
        "system_hash": result.system_hash,
        "dataset_diagnostics": [
            _diagnostic_manifest(values) for values in result.dataset_diagnostics
        ],
    }
    return {
        "schema": _RESULT_SCHEMA,
        "schema_version": RESULT_SCHEMA_VERSION,
        "geodef_version": importlib.metadata.version("geodef"),
        "result": result_metadata,
        "arrays": {name: _array_manifest(array) for name, array in arrays.items()},
    }


def save(result: InversionResult, fname: str | Path) -> None:
    """Save a result as a safe versioned NumPy archive plus JSON manifest.

    The manifest is embedded in the ``.npz`` for single-file portability and
    written beside it as ``<stem>.manifest.json`` for direct inspection.
    Numeric arrays never use pickle or object dtypes.

    Args:
        result: Inversion result to save.
        fname: Destination ``.npz`` path.

    Raises:
        ValueError: If ``fname`` does not end in ``.npz``.
        TypeError: If a result array has object dtype.
    """
    path = Path(fname)
    if path.suffix != ".npz":
        raise ValueError("result archive filename must end in '.npz'")
    arrays = _result_arrays(result)
    manifest = _build_manifest(result, arrays)
    manifest_bytes = orjson.dumps(manifest, option=orjson.OPT_INDENT_2)
    archive_arrays = dict(arrays)
    archive_arrays["__manifest__"] = np.frombuffer(manifest_bytes, dtype=np.uint8)
    np.savez_compressed(file=path, **archive_arrays)  # type: ignore[arg-type]
    with open(_manifest_path(path), "wb") as file:
        file.write(manifest_bytes)
        file.write(b"\n")


def _as_mapping(value: object, label: str) -> Mapping[str, object]:
    """Validate a JSON object loaded from a manifest."""
    if not isinstance(value, dict):
        raise ValueError(f"manifest {label} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"manifest {label} keys must be strings")
    return cast(Mapping[str, object], value)


def _read_manifest(archive: Mapping[str, np.ndarray], path: Path) -> dict[str, object]:
    """Read and cross-check embedded and sidecar manifests."""
    embedded = bytes(np.asarray(archive["__manifest__"], dtype=np.uint8))
    sidecar_path = _manifest_path(path)
    if sidecar_path.exists():
        with open(sidecar_path, "rb") as file:
            sidecar = file.read().rstrip(b"\n")
        if sidecar != embedded:
            raise ValueError("result sidecar manifest does not match the archive")
    loaded = orjson.loads(embedded)
    mapping = _as_mapping(loaded, "root")
    return dict(mapping)


def _validated_arrays(
    archive: Mapping[str, np.ndarray],
    manifest: Mapping[str, object],
) -> dict[str, np.ndarray]:
    """Validate every declared array before constructing a result."""
    array_metadata = _as_mapping(manifest.get("arrays"), "arrays")
    arrays: dict[str, np.ndarray] = {}
    for name, raw_metadata in array_metadata.items():
        if name not in archive:
            raise ValueError(f"result archive is missing declared array {name!r}")
        metadata = _as_mapping(raw_metadata, f"arrays.{name}")
        array = np.asarray(archive[name])
        if array.dtype.hasobject:
            raise ValueError(f"result array {name!r} has unsafe object dtype")
        shape = metadata.get("shape")
        if not isinstance(shape, list) or list(array.shape) != shape:
            raise ValueError(f"result array {name!r} shape does not match manifest")
        if metadata.get("dtype") != array.dtype.str:
            raise ValueError(f"result array {name!r} dtype does not match manifest")
        if metadata.get("sha256") != _array_checksum(array):
            raise ValueError(f"result array {name!r} checksum does not match manifest")
        arrays[name] = array.copy()
    for required in {"slip", "slip_vector", "residuals", "predicted"}:
        if required not in arrays:
            raise ValueError(f"result manifest is missing required array {required!r}")
    return arrays


def _required_string(metadata: Mapping[str, object], name: str) -> str:
    """Read a required string manifest field."""
    value = metadata.get(name)
    if not isinstance(value, str):
        raise ValueError(f"manifest result.{name} must be a string")
    return value


def _optional_float(metadata: Mapping[str, object], name: str) -> float | None:
    """Read an optional float manifest field."""
    value = metadata.get(name)
    return None if value is None else _manifest_float(value)


def _optional_array(arrays: Mapping[str, np.ndarray], name: str) -> np.ndarray | None:
    """Return an optional copied archive array."""
    array = arrays.get(name)
    return None if array is None else array.copy()


def _load_diagnostics(value: object) -> tuple[DatasetDiagnostics, ...]:
    """Decode diagnostics records from a manifest."""
    if not isinstance(value, list):
        raise ValueError("manifest result.dataset_diagnostics must be a list")
    records = []
    for index, raw_record in enumerate(value):
        record = _as_mapping(raw_record, f"dataset_diagnostics[{index}]")
        n_obs = record.get("n_obs")
        if not isinstance(n_obs, int):
            raise ValueError("diagnostic n_obs must be an integer")
        records.append(
            DatasetDiagnostics(
                chi2=_manifest_float(record.get("chi2")),
                reduced_chi2=_manifest_float(record.get("reduced_chi2")),
                wrms=_manifest_float(record.get("wrms")),
                rms=_manifest_float(record.get("rms")),
                n_obs=n_obs,
                dof=_manifest_float(record.get("dof")),
                leverage=_manifest_float(record.get("leverage")),
            )
        )
    return tuple(records)


def _load_partitions(
    metadata: Mapping[str, object], n_rows: int
) -> tuple[tuple[str, ...], tuple[slice, ...]]:
    """Decode and validate named dataset row partitions."""
    raw_names = metadata.get("dataset_names")
    raw_slices = metadata.get("dataset_slices")
    if not isinstance(raw_names, list) or not all(
        isinstance(name, str) for name in raw_names
    ):
        raise ValueError("manifest result.dataset_names must be a string list")
    if len(raw_names) != len(set(raw_names)):
        raise ValueError("manifest dataset names must be unique")
    if not isinstance(raw_slices, list) or len(raw_slices) != len(raw_names):
        raise ValueError("manifest dataset slices must match dataset names")
    slices = []
    expected_start = 0
    for raw_slice in raw_slices:
        if (
            not isinstance(raw_slice, list)
            or len(raw_slice) != 2
            or not all(isinstance(bound, int) for bound in raw_slice)
        ):
            raise ValueError("manifest dataset slices must be [start, stop] pairs")
        start, stop = cast(list[int], raw_slice)
        if start != expected_start or stop < start:
            raise ValueError("manifest dataset slices must be contiguous and ordered")
        slices.append(slice(start, stop))
        expected_start = stop
    if slices and expected_start != n_rows:
        raise ValueError("manifest dataset slices do not cover all observation rows")
    return tuple(cast(list[str], raw_names)), tuple(slices)


def _migrate_v2_regularization_keys(
    metadata: Mapping[str, object], arrays: Mapping[str, np.ndarray]
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Rename schema-version-2 ``smoothing*`` keys to ``regularization*``.

    Version 2 stored the regularization type, strength, and selection under
    ``smoothing`` names, with the custom operator and target arrays keyed by
    ``smoothing`` and ``smoothing_target``. Version 3 renamed them; this
    remaps a version-2 manifest so the shared loader can read it unchanged.

    Args:
        metadata: The version-2 ``result`` manifest mapping.
        arrays: The version-2 archive arrays.

    Returns:
        The metadata and arrays with version-3 key names.
    """
    metadata = dict(metadata)
    arrays = dict(arrays)
    for old, new in (
        ("smoothing", "regularization"),
        ("smoothing_strength", "regularization_strength"),
        ("smoothing_selection", "regularization_selection"),
    ):
        if old in metadata and new not in metadata:
            metadata[new] = metadata.pop(old)
    for old, new in (
        ("smoothing", "regularization"),
        ("smoothing_target", "regularization_target"),
    ):
        if old in arrays and new not in arrays:
            arrays[new] = arrays.pop(old)
    return metadata, arrays


def _load_versioned(
    archive: Mapping[str, np.ndarray], manifest: Mapping[str, object]
) -> InversionResult:
    """Construct a result after validating the current schema."""
    if manifest.get("schema") != _RESULT_SCHEMA:
        raise ValueError(f"unknown result schema {manifest.get('schema')!r}")
    version = manifest.get("schema_version")
    if version not in _SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"unsupported result schema version {version!r}; "
            f"this GeoDef reads versions {sorted(_SUPPORTED_SCHEMA_VERSIONS)}"
        )
    arrays = _validated_arrays(archive, manifest)
    metadata = _as_mapping(manifest.get("result"), "result")
    if version == 2:
        metadata, arrays = _migrate_v2_regularization_keys(metadata, arrays)
    dataset_names, dataset_slices = _load_partitions(metadata, arrays["predicted"].size)

    raw_regularization = metadata.get("regularization")
    if raw_regularization == "__array__":
        regularization: str | np.ndarray | None = _optional_array(
            arrays, "regularization"
        )
        if regularization is None:
            raise ValueError("custom regularization is missing its archive array")
    elif raw_regularization is None or isinstance(raw_regularization, str):
        regularization = raw_regularization
    else:
        raise ValueError("manifest result.regularization must be a string or null")

    success = metadata.get("success")
    if not isinstance(success, bool):
        raise ValueError("manifest result.success must be boolean")
    raw_warnings = metadata.get("warnings")
    if not isinstance(raw_warnings, list) or not all(
        isinstance(warning, str) for warning in raw_warnings
    ):
        raise ValueError("manifest result.warnings must be a string list")
    regularization_selection = metadata.get("regularization_selection")
    if regularization_selection is not None and not isinstance(
        regularization_selection, str
    ):
        raise ValueError("manifest regularization_selection must be a string or null")

    return InversionResult(
        slip=arrays["slip"],
        slip_vector=arrays["slip_vector"],
        residuals=arrays["residuals"],
        predicted=arrays["predicted"],
        reduced_chi2=_manifest_float(metadata.get("reduced_chi2")),
        rms=_manifest_float(metadata.get("rms")),
        moment=_manifest_float(metadata.get("moment")),
        Mw=_manifest_float(metadata.get("Mw")),
        regularization=regularization,
        regularization_strength=_optional_float(metadata, "regularization_strength"),
        components=_required_string(metadata, "components"),
        rake=_optional_float(metadata, "rake"),
        slip_azimuth=_optional_float(metadata, "slip_azimuth"),
        plate_rake=_optional_array(arrays, "plate_rake"),
        local_rake=_optional_array(arrays, "local_rake"),
        dataset_names=dataset_names,
        dataset_slices=dataset_slices,
        solver=_required_string(metadata, "solver"),
        success=success,
        message=_required_string(metadata, "message"),
        regularization_selection=regularization_selection,
        backend=_required_string(metadata, "backend"),
        precision=_required_string(metadata, "precision"),
        warnings=tuple(cast(list[str], raw_warnings)),
        quantity=_required_string(metadata, "quantity"),
        units=_required_string(metadata, "units"),
        system_hash=_required_string(metadata, "system_hash"),
        lower_bounds=_optional_array(arrays, "lower_bounds"),
        upper_bounds=_optional_array(arrays, "upper_bounds"),
        regularization_target=_optional_array(arrays, "regularization_target"),
        constraint_matrix=_optional_array(arrays, "constraint_matrix"),
        constraint_bounds=_optional_array(arrays, "constraint_bounds"),
        dataset_diagnostics=_load_diagnostics(metadata.get("dataset_diagnostics")),
    )


def _legacy_optional_scalar(
    archive: Mapping[str, np.ndarray], name: str
) -> float | None:
    """Read a NaN-sentinel scalar from the unversioned schema."""
    if name not in archive:
        return None
    value = float(archive[name][0])
    return None if np.isnan(value) else value


def _load_legacy(archive: Mapping[str, np.ndarray]) -> InversionResult:
    """Migrate the branch's unversioned result archive in memory."""
    required = {
        "slip",
        "slip_vector",
        "residuals",
        "predicted",
        "reduced_chi2",
        "rms",
        "moment",
        "Mw",
        "smoothing_str",
        "smoothing_strength",
        "components",
    }
    missing = required - set(archive)
    if missing:
        raise ValueError(f"legacy result archive is missing {sorted(missing)}")
    regularization_name = str(archive["smoothing_str"][0])
    if regularization_name == "__none__":
        regularization: str | np.ndarray | None = None
    elif regularization_name == "__array__":
        if "smoothing_arr" not in archive:
            raise ValueError("legacy custom regularization array is missing")
        regularization = archive["smoothing_arr"].copy()
    else:
        regularization = regularization_name
    plate_rake = archive.get("plate_rake")
    local_rake = archive.get("local_rake")
    n_rows = archive["predicted"].size
    return InversionResult(
        slip=archive["slip"].copy(),
        slip_vector=archive["slip_vector"].copy(),
        residuals=archive["residuals"].copy(),
        predicted=archive["predicted"].copy(),
        reduced_chi2=float(archive["reduced_chi2"][0]),
        rms=float(archive["rms"][0]),
        moment=float(archive["moment"][0]),
        Mw=float(archive["Mw"][0]),
        regularization=regularization,
        regularization_strength=_legacy_optional_scalar(archive, "smoothing_strength"),
        components=str(archive["components"][0]),
        rake=_legacy_optional_scalar(archive, "rake"),
        slip_azimuth=_legacy_optional_scalar(archive, "slip_azimuth"),
        plate_rake=(
            None if plate_rake is None or plate_rake.size == 0 else plate_rake.copy()
        ),
        local_rake=(
            None if local_rake is None or local_rake.size == 0 else local_rake.copy()
        ),
        dataset_names=("data",),
        dataset_slices=(slice(0, n_rows),),
        warnings=("loaded and migrated a legacy unversioned result archive",),
    )


def load(fname: str | Path) -> InversionResult:
    """Load and validate a versioned result, or migrate an old archive.

    Args:
        fname: Result ``.npz`` path.

    Returns:
        Reconstructed compact result record.

    Raises:
        FileNotFoundError: If the archive does not exist.
        ValueError: If its schema, manifest, or arrays fail validation.
    """
    path = Path(fname)
    if not path.exists():
        raise FileNotFoundError(f"result archive not found: {path}")
    try:
        with np.load(path, allow_pickle=False) as loaded:
            archive = {name: loaded[name].copy() for name in loaded.files}
    except ValueError as error:
        raise ValueError(
            "result archive contains an unsafe or unreadable array"
        ) from error
    if "__manifest__" not in archive:
        return _load_legacy(archive)
    manifest = _read_manifest(archive, path)
    return _load_versioned(archive, manifest)


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
    slip_2d = result.slip if result.slip.ndim == 2 else result.slip[:, np.newaxis]
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
        if fault._length is None or fault._width is None:
            raise ValueError("rectangular fault is missing patch dimensions")
        column_names = "lon lat depth_m strike dip length_m width_m"
        geometry = np.column_stack(
            [
                fault._lon,
                fault._lat,
                fault._depth,
                fault.strike,
                fault.dip,
                fault._length,
                fault._width,
            ]
        )
    else:
        column_names = "lon lat depth_m strike dip area_m2"
        geometry = np.column_stack(
            [
                fault._lon,
                fault._lat,
                fault._depth,
                fault.strike,
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
