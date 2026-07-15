"""One-call inversion for fault slip from geodetic data.

Solves d = Gm for slip m with optional regularization and bounds.
Supports weighted least-squares, non-negative least-squares,
bounded least-squares, and constrained (QP) solvers.
Automatic hyperparameter tuning via ABIC or cross-validation.
"""

import dataclasses
import functools
import hashlib
import importlib.metadata
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import orjson
import scipy.linalg
import scipy.optimize

if TYPE_CHECKING:
    import matplotlib

from geodef import backend
from geodef.data import DataSet
from geodef.fault import Fault, moment_to_magnitude
from geodef.geometry import (
    LocalFrame,
    _resolve_frame,
    as_planar_vector,
)
from geodef.greens import matrix, select_slip_columns, stack_obs, stack_weights
from geodef.slip import from_plate, from_rake, magnitude, unpack
from geodef.slip import rake as slip_rake

_VALID_METHODS = {"wls", "nnls", "bounded_ls", "constrained"}
_VALID_SMOOTHING_STRINGS = {"laplacian", "damping", "stresskernel"}
_VALID_STRENGTH_STRINGS = {"abic", "cv"}
_VALID_COMPONENTS = {"both", "strike", "dip", "rake", "azimuth", "plate"}
RESULT_SCHEMA_VERSION = 2
_RESULT_SCHEMA = "geodef.inversion_result"

# A bound may be a scalar (all parameters), an array of length n_components
# (one value per slip component, broadcast over patches), or an array of
# length n_params (one value per parameter). ``None`` means unbounded.
_BoundValue = float | np.ndarray | None
BoundsSpec = tuple[_BoundValue, _BoundValue] | None
# Internal fully-expanded form: per-parameter lower/upper arrays.
_ExpandedBounds = tuple[np.ndarray, np.ndarray] | None


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
        smoothing: Regularization type used, or None if unregularized.
        smoothing_strength: Regularization weight used, or None if unregularized.
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
        smoothing_selection: ``'abic'`` or ``'cv'`` when selected
            automatically, otherwise ``None``.
        backend: Array backend active during the solve.
        precision: Floating-point precision active during the solve.
        warnings: Interpretation warnings retained with the result.
        quantity: ``'displacement'`` or ``'velocity'``.
        units: Units inherited from the input datasets.
        system_hash: SHA-256 fingerprint of G, d, W, and L.
        lower_bounds: Expanded lower parameter bounds, if any.
        upper_bounds: Expanded upper parameter bounds, if any.
        smoothing_target: Reference model used by regularization, if any.
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
    smoothing: str | np.ndarray | None
    smoothing_strength: float | None
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
    smoothing_selection: str | None = None
    backend: str = "numpy"
    precision: str = "float64"
    warnings: tuple[str, ...] = ()
    quantity: str = "displacement"
    units: str = "m"
    system_hash: str = ""
    lower_bounds: np.ndarray | None = None
    upper_bounds: np.ndarray | None = None
    smoothing_target: np.ndarray | None = None
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
        "smoothing_target": result.smoothing_target,
        "constraint_matrix": result.constraint_matrix,
        "constraint_bounds": result.constraint_bounds,
    }
    if isinstance(result.smoothing, np.ndarray):
        optional["smoothing"] = result.smoothing
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
    smoothing: str | None
    if isinstance(result.smoothing, np.ndarray):
        smoothing = "__array__"
    else:
        smoothing = result.smoothing
    result_metadata: dict[str, object] = {
        "reduced_chi2": _json_float(result.reduced_chi2),
        "rms": _json_float(result.rms),
        "moment": _json_float(result.moment),
        "Mw": _json_float(result.Mw),
        "smoothing": smoothing,
        "smoothing_strength": (
            None
            if result.smoothing_strength is None
            else _json_float(result.smoothing_strength)
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
        "smoothing_selection": result.smoothing_selection,
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


def _load_versioned(
    archive: Mapping[str, np.ndarray], manifest: Mapping[str, object]
) -> InversionResult:
    """Construct a result after validating the current schema."""
    if manifest.get("schema") != _RESULT_SCHEMA:
        raise ValueError(f"unknown result schema {manifest.get('schema')!r}")
    version = manifest.get("schema_version")
    if version != RESULT_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported result schema version {version!r}; "
            f"this GeoDef supports version {RESULT_SCHEMA_VERSION}"
        )
    arrays = _validated_arrays(archive, manifest)
    metadata = _as_mapping(manifest.get("result"), "result")
    dataset_names, dataset_slices = _load_partitions(metadata, arrays["predicted"].size)

    raw_smoothing = metadata.get("smoothing")
    if raw_smoothing == "__array__":
        smoothing: str | np.ndarray | None = _optional_array(arrays, "smoothing")
        if smoothing is None:
            raise ValueError("custom smoothing is missing its archive array")
    elif raw_smoothing is None or isinstance(raw_smoothing, str):
        smoothing = raw_smoothing
    else:
        raise ValueError("manifest result.smoothing must be a string or null")

    success = metadata.get("success")
    if not isinstance(success, bool):
        raise ValueError("manifest result.success must be boolean")
    raw_warnings = metadata.get("warnings")
    if not isinstance(raw_warnings, list) or not all(
        isinstance(warning, str) for warning in raw_warnings
    ):
        raise ValueError("manifest result.warnings must be a string list")
    smoothing_selection = metadata.get("smoothing_selection")
    if smoothing_selection is not None and not isinstance(smoothing_selection, str):
        raise ValueError("manifest smoothing_selection must be a string or null")

    return InversionResult(
        slip=arrays["slip"],
        slip_vector=arrays["slip_vector"],
        residuals=arrays["residuals"],
        predicted=arrays["predicted"],
        reduced_chi2=_manifest_float(metadata.get("reduced_chi2")),
        rms=_manifest_float(metadata.get("rms")),
        moment=_manifest_float(metadata.get("moment")),
        Mw=_manifest_float(metadata.get("Mw")),
        smoothing=smoothing,
        smoothing_strength=_optional_float(metadata, "smoothing_strength"),
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
        smoothing_selection=smoothing_selection,
        backend=_required_string(metadata, "backend"),
        precision=_required_string(metadata, "precision"),
        warnings=tuple(cast(list[str], raw_warnings)),
        quantity=_required_string(metadata, "quantity"),
        units=_required_string(metadata, "units"),
        system_hash=_required_string(metadata, "system_hash"),
        lower_bounds=_optional_array(arrays, "lower_bounds"),
        upper_bounds=_optional_array(arrays, "upper_bounds"),
        smoothing_target=_optional_array(arrays, "smoothing_target"),
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
    smoothing_name = str(archive["smoothing_str"][0])
    if smoothing_name == "__none__":
        smoothing: str | np.ndarray | None = None
    elif smoothing_name == "__array__":
        if "smoothing_arr" not in archive:
            raise ValueError("legacy custom smoothing array is missing")
        smoothing = archive["smoothing_arr"].copy()
    else:
        smoothing = smoothing_name
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
        smoothing=smoothing,
        smoothing_strength=_legacy_optional_scalar(archive, "smoothing_strength"),
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
    smoothing_desc = (
        "none"
        if result.smoothing is None
        else (result.smoothing if isinstance(result.smoothing, str) else "custom")
    )
    strength_desc = (
        "N/A"
        if result.smoothing_strength is None
        else f"{result.smoothing_strength:.6g}"
    )
    header_lines = [
        "geodef InversionResult",
        f"components: {result.components}",
        f"smoothing: {smoothing_desc}, strength: {strength_desc}",
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


@dataclasses.dataclass(frozen=True)
class LCurveResult:
    """Result of an L-curve analysis.

    Attributes:
        smoothing_values: Array of lambda values swept.
        misfits: Data misfit norm ||Gm - d|| at each lambda.
        model_norms: Regularized model norm ||Lm|| at each lambda.
        optimal: Lambda at the maximum-curvature corner.
    """

    smoothing_values: np.ndarray
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
                smoothing-strength value (default ``True``).

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
        idx = np.argmin(np.abs(self.smoothing_values - self.optimal))
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


_THETA_NAMES = ("e0", "n0", "depth", "strike", "dip", "length", "width")


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
        smoothing_values: Array of lambda values swept.
        abic_values: ABIC value at each lambda (lower is better).
        misfits: Data misfit norm ||Gm - d|| at each lambda.
        model_norms: Regularized model norm ||Lm|| at each lambda.
        optimal: Lambda at the minimum ABIC.
    """

    smoothing_values: np.ndarray
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
        """Plot ABIC vs smoothing strength with the optimal point marked.

        Args:
            ax: Axes to plot on. Creates a new figure if ``None``.
            line_kwargs: Extra kwargs for the curve line.
            marker_kwargs: Extra kwargs for the optimal-point marker.
            annotate: Whether to label the optimal point with its
                smoothing-strength value (default ``True``).

        Returns:
            The axes used for plotting.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()

        lkw = {"color": "b", "marker": ".", "linestyle": "-"}
        if line_kwargs:
            lkw.update(line_kwargs)
        ax.semilogx(self.smoothing_values, self.abic_values, **lkw)

        mkw: dict = {"color": "r", "marker": "o", "markersize": 10, "linestyle": "none"}
        if marker_kwargs:
            mkw.update(marker_kwargs)
        idx = np.argmin(np.abs(self.smoothing_values - self.optimal))
        ax.semilogx(self.smoothing_values[idx], self.abic_values[idx], **mkw)

        if annotate:
            ax.annotate(
                f"λ = {self.optimal:.3g}",
                xy=(self.smoothing_values[idx], self.abic_values[idx]),
                xytext=(0, 20),
                textcoords="offset points",
                fontsize=9,
                color=mkw.get("color", "r"),
                arrowprops={"arrowstyle": "->", "color": mkw.get("color", "r")},
            )

        ax.set_xlabel("Smoothing strength (lambda)")
        ax.set_ylabel("ABIC")
        ax.set_title("ABIC curve")
        return ax


# ======================================================================
# LinearSystem: persistent prepared system with cached matrix products
# ======================================================================


class LinearSystem:
    """Prepared linear system for fault slip inversion.

    Encapsulates the Green's matrix, data vector, weight matrix, and
    smoothing matrix for a given fault-dataset pair.  Expensive derived
    products (G^T W G, L^T L, G^T W d) are computed on first access and
    cached, so they are shared across ``invert``, ``lcurve``,
    ``abic_curve``, and the post-inversion analysis methods.

    Use this class directly when performing multiple analyses on the same
    fault and datasets (e.g. comparing L-curve and ABIC, then running
    diagnostics after inversion).  The module-level convenience functions
    (``invert``, ``lcurve``, etc.) create a ``LinearSystem`` internally
    and are fully backward-compatible.

    Args:
        fault: Fault geometry.
        datasets: One or more geodetic datasets.
        smoothing: Regularization type — ``'laplacian'``, ``'damping'``,
            ``'stresskernel'``, a custom matrix, or ``None``.
        components: Slip components to solve for: ``'both'`` (default),
            ``'strike'``, ``'dip'``, ``'rake'``, ``'azimuth'``, or ``'plate'``.
        rake: Constant local rake for ``components='rake'``.
        slip_azimuth: Constant geographic direction for
            ``components='azimuth'``.
        plate_rake: Scalar or per-patch large-scale direction in local rake
            coordinates for ``components='plate'``.

    Examples:
        >>> sys = LinearSystem(fault, [gnss, insar], smoothing='laplacian')
        >>> lc = sys.lcurve()
        >>> result = sys.invert(smoothing_strength=lc.optimal)
        >>> diag = sys.dataset_diagnostics(result)
    """

    def __init__(
        self,
        fault: Fault,
        datasets: DataSet | list[DataSet],
        smoothing: str | np.ndarray | None = None,
        components: str = "both",
        rake: float | None = None,
        slip_azimuth: float | None = None,
        plate_rake: float | np.ndarray | None = None,
    ) -> None:
        if isinstance(datasets, DataSet):
            datasets = [datasets]
        for ds in datasets:
            if not isinstance(ds, DataSet):
                raise TypeError(
                    f"datasets must contain DataSet instances, got {type(ds).__name__}"
                )
        if not datasets:
            raise ValueError("datasets must contain at least one DataSet")
        semantics = {(dataset.quantity, dataset.units) for dataset in datasets}
        if len(semantics) != 1:
            raise ValueError(
                "joint datasets must use the same quantity and units; "
                f"received {sorted(semantics)}"
            )

        explicit_names = [
            dataset.dataset_name
            for dataset in datasets
            if dataset.dataset_name is not None
        ]
        if len(explicit_names) != len(set(explicit_names)):
            raise ValueError("explicit dataset names must be unique")
        used_names = set(explicit_names)
        generated_counts: dict[str, int] = {}
        dataset_names: list[str] = []
        for dataset in datasets:
            if dataset.dataset_name is not None:
                dataset_names.append(dataset.dataset_name)
                continue
            base = type(dataset).__name__.lower()
            count = generated_counts.get(base, 0) + 1
            candidate = base if count == 1 else f"{base}_{count}"
            while candidate in used_names:
                count += 1
                candidate = f"{base}_{count}"
            generated_counts[base] = count
            used_names.add(candidate)
            dataset_names.append(candidate)

        offset = 0
        dataset_slices = []
        for dataset in datasets:
            dataset_slices.append(slice(offset, offset + dataset.n_obs))
            offset += dataset.n_obs
        if components not in _VALID_COMPONENTS:
            raise ValueError(
                f"components must be one of {_VALID_COMPONENTS}, got {components!r}"
            )
        if components == "rake" and rake is None:
            raise ValueError("components='rake' requires a rake angle in degrees")
        if rake is not None and components != "rake":
            raise ValueError(
                f"rake angle is only used with components='rake', "
                f"got components={components!r}"
            )
        if components == "azimuth" and slip_azimuth is None:
            raise ValueError("components='azimuth' requires a slip_azimuth in degrees")
        if slip_azimuth is not None and components != "azimuth":
            raise ValueError(
                f"slip_azimuth is only used with components='azimuth', "
                f"got components={components!r}"
            )
        if components == "plate" and plate_rake is None:
            raise ValueError("components='plate' requires plate_rake")
        if plate_rake is not None and components != "plate":
            raise ValueError(
                "plate_rake is only used with components='plate', "
                f"got components={components!r}"
            )

        self.fault = fault
        self.datasets = datasets
        self.dataset_names = tuple(dataset_names)
        self.dataset_slices = tuple(dataset_slices)
        self.quantity, self.units = next(iter(semantics))
        self.smoothing = smoothing
        self.components = components
        self.rake = rake
        self.slip_azimuth = slip_azimuth
        self.plate_rake = (
            None
            if plate_rake is None
            else np.broadcast_to(
                np.asarray(plate_rake, dtype=float), (fault.n_patches,)
            ).copy()
        )

        n_patches = fault.n_patches
        n_components = 2 if components in {"both", "plate"} else 1
        self._n_patches = n_patches
        self._n_params = n_components * n_patches

        G_full = matrix(fault, datasets)
        self.d = stack_obs(datasets)
        self.W = stack_weights(datasets)
        self.G = select_slip_columns(
            G_full,
            n_patches,
            components,
            rake,
            fault_strike=fault.strike,
            slip_azimuth=slip_azimuth,
            plate_rake=self.plate_rake,
        )
        self.G_w, self.d_w = _apply_weights(self.G, self.d, self.W)
        self.L: np.ndarray | None = (
            _build_smoothing_matrix(
                fault,
                smoothing,
                self._n_params,
                n_components,
                components,
                rake,
                slip_azimuth,
                self.plate_rake,
            )
            if smoothing is not None
            else None
        )

    @functools.cached_property
    def GtWG(self) -> np.ndarray:
        """G^T W G — normal equations matrix (without regularization)."""
        return self.G_w.T @ self.G_w

    @functools.cached_property
    def LtL(self) -> np.ndarray:
        """L^T L — regularization normal equations matrix.

        Raises:
            AttributeError: If the system was constructed without smoothing.
        """
        if self.L is None:
            raise AttributeError(
                "LtL is not available: LinearSystem has no smoothing matrix"
            )
        return self.L.T @ self.L

    @functools.cached_property
    def Gtwd(self) -> np.ndarray:
        """G^T W d — normal equations right-hand side."""
        return self.G_w.T @ self.d_w

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _abic_value(
        self,
        smoothing_strength: float,
    ) -> tuple[float, float, float]:
        """ABIC, misfit norm, and model norm at a given smoothing strength.

        Uses cached GtWG, LtL, and Gtwd.  ``eig_LtL`` (lambda-independent)
        is computed on the first call and cached in ``self.__dict__``.

        The ABIC formula (Fukuda & Johnson 2008, 2010) requires the weighted
        misfit ``r^T W r`` internally.  The returned ``misfit_norm`` is the
        unweighted ``||Gm - d||`` for consistent plotting against lcurve.

        Args:
            smoothing_strength: Regularization weight lambda.

        Returns:
            (abic, misfit_norm, model_norm) where misfit_norm = ||Gm - d||
            and model_norm = ||Lm||.
        """
        alpha2 = smoothing_strength
        n_data = len(self.d)

        H = self.GtWG + alpha2 * self.LtL
        m = np.linalg.solve(H, self.Gtwd)

        residuals = self.d - self.G @ m
        misfit_weighted = float(residuals @ self.W @ residuals)
        penalty = alpha2 * float(m @ self.LtL @ m)
        total = max(misfit_weighted + penalty, 1e-300)
        abic1 = n_data * np.log(total)

        # eig_LtL is lambda-independent — compute once and cache
        eig_LtL: np.ndarray | None = self.__dict__.get("_eig_LtL")
        if eig_LtL is None:
            eig_LtL = np.linalg.eigvalsh(self.LtL)
            self.__dict__["_eig_LtL"] = eig_LtL

        eig_prior = alpha2 * _rank_positive_eigs(eig_LtL)
        abic2 = float(np.sum(np.log(eig_prior)))

        eig_post = _rank_positive_eigs(np.linalg.eigvalsh(H))
        abic3 = float(np.sum(np.log(eig_post)))

        abic = abic1 - abic2 + abic3
        misfit_norm = float(np.sqrt(residuals @ residuals))
        model_norm = float(np.sqrt((self.L @ m) @ (self.L @ m)))
        return abic, misfit_norm, model_norm

    def _abic_sweep_jax(
        self,
        lambdas: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate the ABIC sweep as one batched JAX computation.

        Computes the same quantities as ``_abic_value`` at every lambda,
        but the linear solves, log-determinants, and norms are batched
        across the lambda axis so XLA fuses the whole sweep. The
        posterior log-determinant uses ``slogdet(H)`` instead of filtered
        eigenvalues; for the positive-definite systems swept here the two
        agree.

        Args:
            lambdas: Regularization weights to sweep, shape (n,).

        Returns:
            Arrays ``(abic_values, misfits, model_norms)``, each (n,).
        """
        import jax.numpy as jnp

        assert self.L is not None
        n_data = len(self.d)
        lam = jnp.asarray(lambdas)
        GtWG = jnp.asarray(self.GtWG)
        LtL = jnp.asarray(self.LtL)
        Gtwd = jnp.asarray(self.Gtwd)
        G = jnp.asarray(self.G)
        d = jnp.asarray(self.d)
        W = jnp.asarray(self.W)
        L = jnp.asarray(self.L)

        H = GtWG[None, :, :] + lam[:, None, None] * LtL[None, :, :]
        rhs = jnp.broadcast_to(Gtwd, (len(lambdas), len(Gtwd)))
        m = jnp.linalg.solve(H, rhs[..., None]).squeeze(-1)

        r = d[None, :] - m @ G.T
        misfit_weighted = jnp.einsum("nm,mk,nk->n", r, W, r)
        penalty = lam * jnp.einsum("np,pq,nq->n", m, LtL, m)
        total = misfit_weighted + penalty
        total = jnp.maximum(total, jnp.finfo(total.dtype).tiny)
        abic1 = n_data * backend.to_numpy(jnp.log(total))

        # eig_LtL is lambda-independent — compute once and cache, and
        # split sum(log(lam*|e|)) into k*log(lam) + sum(log|e|)
        eig_LtL: np.ndarray | None = self.__dict__.get("_eig_LtL")
        if eig_LtL is None:
            eig_LtL = np.linalg.eigvalsh(self.LtL)
            self.__dict__["_eig_LtL"] = eig_LtL
        eig_pos = _rank_positive_eigs(eig_LtL)
        abic2 = len(eig_pos) * np.log(lambdas) + np.sum(np.log(eig_pos))

        _, abic3 = jnp.linalg.slogdet(H)

        abic = abic1 - abic2 + backend.to_numpy(abic3)
        misfits = np.sqrt(np.sum(backend.to_numpy(r) ** 2, axis=1))
        Lm = backend.to_numpy(m @ L.T)
        model_norms = np.sqrt(np.sum(Lm**2, axis=1))
        return abic, misfits, model_norms

    def _optimal_abic(self) -> float:
        """Find optimal smoothing strength by minimizing ABIC.

        Returns:
            Optimal lambda.
        """
        if self.L is None:
            raise ValueError("ABIC requires a smoothing matrix")

        def objective(log10_lam: float) -> float:
            return self._abic_value(10.0**log10_lam)[0]

        result = scipy.optimize.minimize_scalar(
            objective,
            bounds=(-6, 10),
            method="bounded",
        )
        return 10.0**result.x

    def _optimal_cv(
        self,
        bounds: _ExpandedBounds,
        method: str | None,
        cv_folds: int,
    ) -> float:
        """Find optimal smoothing strength by K-fold cross-validation.

        Args:
            bounds: Expanded per-parameter slip bounds.
            method: Solver method.
            cv_folds: Number of folds.

        Returns:
            Optimal lambda.
        """
        if self.L is None:
            raise ValueError("Cross-validation requires a smoothing matrix")

        n_obs = self.G_w.shape[0]
        solve_method = method if method is not None else _auto_select_method(bounds)

        rng = np.random.default_rng(0)
        perm = rng.permutation(n_obs)
        fold_sizes = np.full(cv_folds, n_obs // cv_folds)
        fold_sizes[: n_obs % cv_folds] += 1
        folds = np.split(perm, np.cumsum(fold_sizes[:-1]))

        lambdas = np.geomspace(1e-4, 1e8, 50)
        cv_errors = np.zeros(len(lambdas))

        for i, lam in enumerate(lambdas):
            fold_errors = 0.0
            for fold in folds:
                mask = np.ones(n_obs, dtype=bool)
                mask[fold] = False
                G_aug = np.vstack([self.G_w[mask], np.sqrt(lam) * self.L])
                d_aug = np.concatenate([self.d_w[mask], np.zeros(self.L.shape[0])])
                m = _solve(G_aug, d_aug, bounds, solve_method, None)
                pred_test = self.G_w[fold] @ m
                fold_errors += float(np.sum((self.d_w[fold] - pred_test) ** 2))
            cv_errors[i] = fold_errors / n_obs

        return float(lambdas[np.argmin(cv_errors)])

    def _hat_diagonal(self, smoothing_strength: float | None) -> np.ndarray:
        """Diagonal of the hat matrix H = G_w (G_w^T G_w + λ L^T L)^{-1} G_w^T.

        Args:
            smoothing_strength: Regularization weight, or None.

        Returns:
            Leverage vector, shape (M,).
        """
        H = self.GtWG.copy()
        if (
            self.L is not None
            and smoothing_strength is not None
            and smoothing_strength > 0
        ):
            H += smoothing_strength * self.LtL
        A = np.linalg.solve(H.T, self.G_w.T).T
        return np.sum(A * self.G_w, axis=1)

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def invert(
        self,
        smoothing_strength: float | str = 0.0,
        bounds: BoundsSpec = None,
        method: str | None = None,
        smoothing_target: np.ndarray | None = None,
        constraints: tuple[np.ndarray, np.ndarray] | None = None,
        cv_folds: int = 5,
    ) -> InversionResult:
        """Invert for fault slip using this prepared system.

        Args:
            smoothing_strength: Scalar regularization weight, or
                ``'abic'`` / ``'cv'`` for automatic tuning.
            bounds: Per-component slip bounds ``(lower, upper)``.
            method: Solver — ``'wls'``, ``'nnls'``, ``'bounded_ls'``,
                or ``'constrained'``. Auto-selected from bounds if None.
            smoothing_target: Reference vector, shape ``(n_params,)``.
                Regularizes toward this target instead of zero.
            constraints: Inequality constraints ``(C, d_ineq)`` such
                that ``C @ m <= d_ineq``.
            cv_folds: Number of folds for cross-validation (default 5).

        Returns:
            InversionResult with slip, residuals, and fit statistics.

        Raises:
            ValueError: For invalid arguments.
        """
        _validate_args(
            self.datasets,
            self.components,
            self.smoothing,
            smoothing_strength,
            bounds,
            method,
            smoothing_target,
            self._n_params,
            self.rake,
            self.slip_azimuth,
            self.plate_rake,
        )

        exp_bounds = _expand_bounds(
            bounds, self._n_patches, self._n_params // self._n_patches
        )

        smoothing_selection = (
            smoothing_strength if isinstance(smoothing_strength, str) else None
        )
        if isinstance(smoothing_strength, str):
            if smoothing_strength == "abic":
                strength = self._optimal_abic()
            elif smoothing_strength == "cv":
                strength = self._optimal_cv(exp_bounds, method, cv_folds)
            else:
                raise ValueError(
                    "smoothing_strength string must be 'abic' or 'cv', "
                    f"got {smoothing_strength!r}"
                )
        else:
            strength = float(smoothing_strength)

        if self.L is not None and strength > 0:
            d_reg = _build_reg_rhs(self.L, strength, smoothing_target)
            G_aug = np.vstack([self.G_w, np.sqrt(strength) * self.L])
            d_aug = np.concatenate([self.d_w, d_reg])
            reg_strength: float | None = strength
        else:
            G_aug = self.G_w
            d_aug = self.d_w
            reg_strength = None if strength == 0.0 else strength

        if method is None:
            method = _auto_select_method(exp_bounds)

        m = _solve(G_aug, d_aug, exp_bounds, method, constraints)

        predicted = self.G @ m
        residuals = self.d - predicted
        reduced_chi2 = _compute_reduced_chi2(residuals, self.W, self._n_params)
        rms = float(np.sqrt(np.mean(residuals**2)))

        if self.components in {"both", "plate"}:
            slip = np.column_stack([m[: self._n_patches], m[self._n_patches :]])
        else:
            slip = m.reshape(-1, 1)
        basis_angle: float | np.ndarray | None
        if self.components == "rake":
            basis_angle = self.rake
        elif self.components == "azimuth":
            assert self.slip_azimuth is not None
            basis_angle = self.slip_azimuth - self.fault.strike
        elif self.components == "plate":
            basis_angle = self.plate_rake
        else:
            basis_angle = None
        strike_slip, dip_slip = _physical_components(m, self.components, basis_angle)
        result_warnings: list[str] = []
        if self.d.size <= self._n_params:
            result_warnings.append(
                "the inversion has no positive nominal degrees of freedom"
            )
        if self.quantity == "velocity":
            moment = float("nan")
            mw = float("nan")
            result_warnings.append(
                "moment and Mw are undefined for velocity data; slip is a slip rate"
            )
        else:
            moment = self.fault.moment(magnitude(strike_slip, dip_slip))
            mw = moment_to_magnitude(moment)

        constraint_matrix: np.ndarray | None = None
        constraint_bounds: np.ndarray | None = None
        if constraints is not None:
            constraint_matrix = np.asarray(constraints[0], dtype=float).copy()
            constraint_bounds = np.asarray(constraints[1], dtype=float).copy()
        lower_bounds: np.ndarray | None = None
        upper_bounds: np.ndarray | None = None
        if exp_bounds is not None:
            lower_bounds = exp_bounds[0].copy()
            upper_bounds = exp_bounds[1].copy()
        system_hash = _system_hash(self.G, self.d, self.W, self.L)
        fit_diagnostics = tuple(
            self._compute_dataset_diagnostics(residuals, reg_strength)
        )

        return InversionResult(
            slip=slip,
            slip_vector=m,
            residuals=residuals,
            predicted=predicted,
            reduced_chi2=reduced_chi2,
            rms=rms,
            moment=moment,
            Mw=mw,
            smoothing=self.smoothing if reg_strength is not None else None,
            smoothing_strength=reg_strength,
            components=self.components,
            rake=self.rake,
            slip_azimuth=self.slip_azimuth,
            plate_rake=self.plate_rake,
            local_rake=(
                self.slip_azimuth - self.fault.strike
                if self.slip_azimuth is not None
                else None
            ),
            dataset_names=self.dataset_names,
            dataset_slices=self.dataset_slices,
            solver=method,
            success=True,
            message=f"{method} completed",
            smoothing_selection=smoothing_selection,
            backend=backend.get_backend(),
            precision=backend.get_precision(),
            warnings=tuple(result_warnings),
            quantity=self.quantity,
            units=self.units,
            system_hash=system_hash,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            smoothing_target=(
                None
                if smoothing_target is None
                else np.asarray(smoothing_target, dtype=float).copy()
            ),
            constraint_matrix=constraint_matrix,
            constraint_bounds=constraint_bounds,
            dataset_diagnostics=fit_diagnostics,
        )

    def lcurve(
        self,
        smoothing_range: tuple[float, float] = (1e-2, 1e6),
        n: int = 50,
        bounds: BoundsSpec = None,
        method: str | None = None,
    ) -> LCurveResult:
        """Sweep smoothing strength and compute the L-curve.

        For unconstrained (``wls``) solves, GtWG, LtL, and Gtwd are used
        directly so each iteration is a single linear solve with no matrix
        assembly.  For constrained solves the augmented system is used.

        Misfits are the unweighted norm ``||Gm - d||``.

        Args:
            smoothing_range: ``(min_lambda, max_lambda)`` range to sweep.
            n: Number of lambda values to evaluate.
            bounds: Per-component slip bounds.
            method: Solver method.

        Returns:
            LCurveResult with sweep arrays and optimal lambda.

        Raises:
            ValueError: If the system has no smoothing matrix.
        """
        if self.L is None:
            raise ValueError("lcurve requires a smoothing matrix")

        lambdas = np.geomspace(smoothing_range[0], smoothing_range[1], n)
        misfits = np.empty(n)
        model_norms = np.empty(n)

        exp_bounds = _expand_bounds(
            bounds, self._n_patches, self._n_params // self._n_patches
        )
        solve_method = method if method is not None else _auto_select_method(exp_bounds)

        if solve_method == "wls":
            for i, lam in enumerate(lambdas):
                H = self.GtWG + lam * self.LtL
                m = np.linalg.solve(H, self.Gtwd)
                residuals = self.d - self.G @ m
                misfits[i] = float(np.sqrt(residuals @ residuals))
                model_norms[i] = float(np.sqrt((self.L @ m) @ (self.L @ m)))
        else:
            for i, lam in enumerate(lambdas):
                G_aug = np.vstack([self.G_w, np.sqrt(lam) * self.L])
                d_aug = np.concatenate([self.d_w, np.zeros(self.L.shape[0])])
                m = _solve(G_aug, d_aug, exp_bounds, solve_method, None)
                residuals = self.d - self.G @ m
                misfits[i] = float(np.sqrt(residuals @ residuals))
                model_norms[i] = float(np.sqrt((self.L @ m) @ (self.L @ m)))

        optimal = _lcurve_corner(lambdas, misfits, model_norms)
        return LCurveResult(
            smoothing_values=lambdas,
            misfits=misfits,
            model_norms=model_norms,
            optimal=optimal,
        )

    def abic_curve(
        self,
        smoothing_range: tuple[float, float] = (1e-2, 1e6),
        n: int = 50,
    ) -> ABICCurveResult:
        """Sweep smoothing strength and compute the ABIC at each value.

        GtWG, LtL, Gtwd, and eig_LtL are all computed once and reused
        across all iterations.  Misfits are the unweighted norm ``||Gm - d||``,
        consistent with ``lcurve``.

        Args:
            smoothing_range: ``(min_lambda, max_lambda)`` range to sweep.
            n: Number of lambda values to evaluate.

        Returns:
            ABICCurveResult with sweep arrays and optimal lambda.

        Raises:
            ValueError: If the system has no smoothing matrix.
        """
        if self.L is None:
            raise ValueError("abic_curve requires a smoothing matrix")

        lambdas = np.geomspace(smoothing_range[0], smoothing_range[1], n)

        if backend.get_backend() == "jax":
            abic_values, misfits, model_norms = self._abic_sweep_jax(lambdas)
        else:
            abic_values = np.empty(n)
            misfits = np.empty(n)
            model_norms = np.empty(n)
            for i, lam in enumerate(lambdas):
                abic_values[i], misfits[i], model_norms[i] = self._abic_value(lam)

        optimal = float(lambdas[np.argmin(abic_values)])
        return ABICCurveResult(
            smoothing_values=lambdas,
            abic_values=abic_values,
            misfits=misfits,
            model_norms=model_norms,
            optimal=optimal,
        )

    def dataset_diagnostics(
        self,
        result: InversionResult,
    ) -> list[DatasetDiagnostics]:
        """Compute per-dataset fit diagnostics using the hat matrix.

        Args:
            result: Output from ``invert()``.

        Returns:
            List of ``DatasetDiagnostics``, one per dataset.
        """
        return self._compute_dataset_diagnostics(
            result.residuals, result.smoothing_strength
        )

    def _compute_dataset_diagnostics(
        self,
        residuals: np.ndarray,
        smoothing_strength: float | None,
    ) -> list[DatasetDiagnostics]:
        """Compute named-fit statistics from solve-time arrays."""
        lev = self._hat_diagonal(smoothing_strength)

        diags = []
        for ds, idx in zip(self.datasets, self.dataset_slices):
            n = ds.n_obs
            r_k = residuals[idx]
            W_k = self.W[idx, idx]

            chi2_k = float(r_k @ W_k @ r_k)
            lev_k = float(np.sum(lev[idx]))
            dof_k = n - lev_k
            reduced_chi2_k = chi2_k / dof_k if dof_k > 0 else float("nan")
            wrms_k = float(np.sqrt(chi2_k / n))
            rms_k = float(np.sqrt(np.mean(r_k**2)))

            diags.append(
                DatasetDiagnostics(
                    chi2=chi2_k,
                    reduced_chi2=reduced_chi2_k,
                    wrms=wrms_k,
                    rms=rms_k,
                    n_obs=n,
                    dof=dof_k,
                    leverage=lev_k,
                )
            )

        return diags

    def model_covariance(
        self, result: InversionResult, kind: str = "posterior"
    ) -> np.ndarray:
        """Compute the model covariance matrix.

        For the unregularized case both kinds reduce to::

            Cm = (G^T W G)^{-1}

        For the regularized case, with ``H = G^T W G + lambda L^T L``
        (see docs/conventions.md):

        - ``kind='posterior'`` (default) — the linear-Gaussian posterior
          covariance ``Cm = H^{-1}``, treating ``lambda L^T L`` as a prior
          precision. This is the quantity taught in Tutorial 09 and the one
          consistent with the Bayesian slip draws in ``geodef.bayes``.
        - ``kind='estimator'`` — the frequentist covariance of the penalized
          estimator under data noise alone (Tarantola, 2005)::

              Cm = H^{-1} @ G^T W G @ H^{-1}

          It excludes the bias the regularization introduces, so it shrinks
          to zero as ``lambda`` grows; interpret it together with the
          resolution matrix.

        Args:
            result: Output from ``invert()``.
            kind: ``'posterior'`` (default) or ``'estimator'``.

        Returns:
            Model covariance matrix, shape (n_params, n_params).
        """
        if kind not in ("posterior", "estimator"):
            raise ValueError(f"kind must be 'posterior' or 'estimator', got {kind!r}")
        if self.L is not None and result.smoothing_strength is not None:
            H = self.GtWG + result.smoothing_strength * self.LtL
            if kind == "posterior":
                return np.linalg.inv(H)
            H_inv = np.linalg.inv(H)
            return H_inv @ self.GtWG @ H_inv
        return np.linalg.inv(self.GtWG)

    def model_resolution(self, result: InversionResult) -> np.ndarray:
        """Compute the model resolution matrix.

        ``R = (G^T W G + lambda L^T L)^{-1} G^T W G``

        Args:
            result: Output from ``invert()``.

        Returns:
            Resolution matrix, shape (n_params, n_params).
        """
        if self.L is not None and result.smoothing_strength is not None:
            H = self.GtWG + result.smoothing_strength * self.LtL
            return np.linalg.solve(H, self.GtWG)
        return np.linalg.solve(self.GtWG, self.GtWG)

    def model_uncertainty(
        self, result: InversionResult, kind: str = "posterior"
    ) -> np.ndarray:
        """Compute per-parameter 1-sigma uncertainty from model covariance.

        Args:
            result: Output from ``invert()``.
            kind: Covariance kind, ``'posterior'`` (default) or
                ``'estimator'``; see :meth:`model_covariance`.

        Returns:
            Uncertainty array, shape (n_params,).
        """
        Cm = self.model_covariance(result, kind=kind)
        return np.sqrt(np.maximum(np.diag(Cm), 0.0))


# ======================================================================
# Module-level convenience functions (backward-compatible wrappers)
# ======================================================================


def solve(
    fault: Fault,
    datasets: DataSet | list[DataSet],
    smoothing: str | np.ndarray | None = None,
    smoothing_strength: float | str = 0.0,
    bounds: BoundsSpec = None,
    method: str | None = None,
    smoothing_target: np.ndarray | None = None,
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
        smoothing: Regularization type. One of ``'laplacian'``,
            ``'damping'``, ``'stresskernel'``, a custom matrix, or None.
        smoothing_strength: Scalar weight on the regularization term,
            or ``'abic'`` / ``'cv'`` for automatic tuning.
        bounds: Per-component slip bounds ``(lower, upper)``.
            Use None for unbounded side, e.g. ``(0, None)``.
        method: Solver — ``'wls'``, ``'nnls'``, ``'bounded_ls'``, or
            ``'constrained'``. Auto-selected from bounds if None.
        smoothing_target: Reference model vector, shape
            (n_components * N,). Regularizes toward this target instead
            of zero: minimizes ``||L(m - m_ref)||^2``. Only valid when
            smoothing is set.
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
        smoothing,
        components,
        rake,
        slip_azimuth,
        plate_rake,
    )
    return sys.invert(
        smoothing_strength, bounds, method, smoothing_target, constraints, cv_folds
    )


def compute_abic(
    G: np.ndarray,
    d: np.ndarray,
    W: np.ndarray,
    L: np.ndarray,
    smoothing_strength: float,
) -> float:
    """Compute the ABIC value for a given smoothing strength.

    Implements the Akaike Bayesian Information Criterion following
    Fukuda & Johnson (2008, 2010).

    Args:
        G: Green's matrix, shape (M, P).
        d: Data vector, shape (M,).
        W: Weight matrix, shape (M, M).
        L: Regularization matrix, shape (K, P).
        smoothing_strength: Regularization weight (lambda = alpha^2).

    Returns:
        ABIC scalar value (lower is better).
    """
    alpha2 = smoothing_strength
    n_data = len(d)

    GtWG = G.T @ W @ G
    LtL = L.T @ L
    H = GtWG + alpha2 * LtL
    m = np.linalg.solve(H, G.T @ W @ d)

    residuals = d - G @ m
    misfit = float(residuals @ W @ residuals)
    penalty = alpha2 * float(m @ LtL @ m)
    total = max(misfit + penalty, 1e-300)
    abic1 = n_data * np.log(total)

    eig_prior = alpha2 * _rank_positive_eigs(np.linalg.eigvalsh(LtL))
    abic2 = np.sum(np.log(eig_prior))

    eig_post = _rank_positive_eigs(np.linalg.eigvalsh(H))
    abic3 = np.sum(np.log(eig_post))

    return float(abic1 - abic2 + abic3)


def lcurve(
    fault: Fault,
    datasets: DataSet | list[DataSet],
    smoothing: str | np.ndarray = "laplacian",
    smoothing_range: tuple[float, float] = (1e-2, 1e6),
    n: int = 50,
    bounds: BoundsSpec = None,
    method: str | None = None,
    components: str = "both",
    rake: float | None = None,
    slip_azimuth: float | None = None,
    plate_rake: float | np.ndarray | None = None,
) -> LCurveResult:
    """Sweep smoothing strength and compute the L-curve.

    Args:
        fault: Fault geometry.
        datasets: One or more geodetic datasets.
        smoothing: Regularization type.
        smoothing_range: ``(min_lambda, max_lambda)`` range to sweep.
        n: Number of lambda values to evaluate.
        bounds: Per-component slip bounds.
        method: Solver method.
        components: Which slip components to solve for.
        rake: Fixed rake angle in degrees, required when
            ``components='rake'``.
        slip_azimuth: Geographic slip azimuth in degrees, required when
            ``components='azimuth'``.
        plate_rake: Local plate-rake direction, required when
            ``components='plate'``.

    Returns:
        LCurveResult with sweep arrays and optimal lambda.
    """
    sys = LinearSystem(
        fault, datasets, smoothing, components, rake, slip_azimuth, plate_rake
    )
    return sys.lcurve(smoothing_range, n, bounds, method)


def abic_curve(
    fault: Fault,
    datasets: DataSet | list[DataSet],
    smoothing: str | np.ndarray = "laplacian",
    smoothing_range: tuple[float, float] = (1e-2, 1e6),
    n: int = 50,
    components: str = "both",
    rake: float | None = None,
    slip_azimuth: float | None = None,
    plate_rake: float | np.ndarray | None = None,
) -> ABICCurveResult:
    """Sweep smoothing strength and compute the ABIC at each value.

    Also records misfit and model norm for context. The optimal lambda
    is the one that minimizes ABIC.

    Args:
        fault: Fault geometry.
        datasets: One or more geodetic datasets.
        smoothing: Regularization type.
        smoothing_range: ``(min_lambda, max_lambda)`` range to sweep.
        n: Number of lambda values to evaluate.
        components: Which slip components to solve for.
        rake: Fixed rake angle in degrees, required when
            ``components='rake'``.
        slip_azimuth: Geographic slip azimuth in degrees, required when
            ``components='azimuth'``.
        plate_rake: Local plate-rake direction, required when
            ``components='plate'``.

    Returns:
        ABICCurveResult with sweep arrays and optimal lambda.
    """
    sys = LinearSystem(
        fault, datasets, smoothing, components, rake, slip_azimuth, plate_rake
    )
    return sys.abic_curve(smoothing_range, n)


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
    smoothing: str | np.ndarray | None = None,
    smoothing_strength: float = 0.0,
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
        smoothing: Regularization type for the inner solve (as in
            ``invert()``), or None for no regularization.
        smoothing_strength: Regularization weight lambda for the inner
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
    sys = LinearSystem(template, datasets, smoothing, components)
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
    if smoothing_strength > 0.0:
        if sys.L is None:
            raise ValueError("smoothing_strength > 0 requires a smoothing operator")
        LtL = jnp.asarray(sys.LtL) * smoothing_strength
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
    smoothing = "none"
    if result.smoothing is not None:
        smoothing_name = (
            result.smoothing if isinstance(result.smoothing, str) else "custom"
        )
        smoothing = f"{smoothing_name} (lambda={result.smoothing_strength:.6g})"
        if result.smoothing_selection is not None:
            smoothing += f", selected by {result.smoothing_selection}"
    lines = [
        f"solver: {result.solver} ({'success' if result.success else 'failed'})",
        f"datasets: {', '.join(result.dataset_names) or 'data'}",
        f"quantity: {result.quantity} [{result.units}]",
        f"components: {result.components}",
        f"regularization: {smoothing}",
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
        result.smoothing,
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
        result.smoothing,
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
        result.smoothing,
        result.components,
        result.rake,
        result.slip_azimuth,
        result.plate_rake,
    )
    return sys.model_uncertainty(result, kind=kind)


# ======================================================================
# Private helpers
# ======================================================================


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


def _validate_args(
    datasets: list[DataSet],
    components: str,
    smoothing: str | np.ndarray | None,
    smoothing_strength: float | str,
    bounds: BoundsSpec,
    method: str | None,
    smoothing_target: np.ndarray | None,
    n_params: int,
    rake: float | None = None,
    slip_azimuth: float | None = None,
    plate_rake: np.ndarray | None = None,
) -> None:
    """Validate invert() arguments."""
    for ds in datasets:
        if not isinstance(ds, DataSet):
            raise TypeError(
                f"datasets must contain DataSet instances, got {type(ds).__name__}"
            )

    if components not in _VALID_COMPONENTS:
        raise ValueError(
            f"components must be one of {_VALID_COMPONENTS}, got {components!r}"
        )

    if components == "rake" and rake is None:
        raise ValueError("components='rake' requires a rake angle in degrees")
    if rake is not None and components != "rake":
        raise ValueError(
            f"rake angle is only used with components='rake', "
            f"got components={components!r}"
        )
    if components == "azimuth" and slip_azimuth is None:
        raise ValueError("components='azimuth' requires a slip_azimuth in degrees")
    if slip_azimuth is not None and components != "azimuth":
        raise ValueError(
            f"slip_azimuth is only used with components='azimuth', "
            f"got components={components!r}"
        )
    if components == "plate" and plate_rake is None:
        raise ValueError("components='plate' requires plate_rake")
    if plate_rake is not None and components != "plate":
        raise ValueError(
            "plate_rake is only used with components='plate', "
            f"got components={components!r}"
        )

    if method is not None and method not in _VALID_METHODS:
        raise ValueError(f"method must be one of {_VALID_METHODS}, got {method!r}")

    if isinstance(smoothing, str) and smoothing not in _VALID_SMOOTHING_STRINGS:
        raise ValueError(
            f"smoothing must be one of {_VALID_SMOOTHING_STRINGS} "
            f"or a numpy array, got {smoothing!r}"
        )

    if isinstance(smoothing, np.ndarray) and smoothing.shape[1] != n_params:
        raise ValueError(
            f"smoothing matrix must have {n_params} columns, got {smoothing.shape[1]}"
        )

    if isinstance(smoothing_strength, str):
        if smoothing_strength not in _VALID_STRENGTH_STRINGS:
            raise ValueError(
                f"smoothing_strength must be a float or one of "
                f"{_VALID_STRENGTH_STRINGS}, got {smoothing_strength!r}"
            )
        if smoothing is None:
            raise ValueError(
                f"smoothing_strength='{smoothing_strength}' requires "
                f"smoothing to be set"
            )

    if smoothing_target is not None:
        if smoothing is None and smoothing_strength == 0.0:
            raise ValueError("smoothing_target requires smoothing to be set")
        target = np.asarray(smoothing_target)
        if target.shape != (n_params,):
            raise ValueError(
                f"smoothing_target must have shape ({n_params},), got {target.shape}"
            )


def _apply_weights(
    G: np.ndarray,
    d: np.ndarray,
    W: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply data weights via W^(1/2).

    For diagonal W, uses efficient element-wise scaling.
    For full W, uses Cholesky decomposition.

    Returns:
        (G_weighted, d_weighted).
    """
    off_diag = W - np.diag(np.diag(W))
    if np.allclose(off_diag, 0):
        w_half = np.sqrt(np.diag(W))
        return w_half[:, np.newaxis] * G, w_half * d

    W_half = scipy.linalg.cholesky(W, lower=False)
    return W_half @ G, W_half @ d


def _build_smoothing_matrix(
    fault: Fault,
    smoothing: str | np.ndarray,
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
        smoothing: Smoothing type or custom matrix.
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
    if isinstance(smoothing, np.ndarray):
        return smoothing

    if smoothing == "damping":
        return np.eye(n_params)

    if smoothing == "laplacian":
        L_patch = fault.laplacian
        if n_components == 1:
            return L_patch
        return scipy.linalg.block_diag(L_patch, L_patch)

    if smoothing == "stresskernel":
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

    raise ValueError(f"Unknown smoothing type: {smoothing!r}")


def _build_reg_rhs(
    L: np.ndarray,
    smoothing_strength: float,
    smoothing_target: np.ndarray | None,
) -> np.ndarray:
    """Build the right-hand side for the regularization rows.

    For standard regularization (target=None): zeros.
    For target regularization: sqrt(lambda) * L @ m_ref.
    """
    if smoothing_target is None:
        return np.zeros(L.shape[0])
    return np.sqrt(smoothing_strength) * (L @ smoothing_target)


def _expand_bounds(
    bounds: BoundsSpec,
    n_patches: int,
    n_components: int,
) -> _ExpandedBounds:
    """Expand bounds to per-parameter lower/upper arrays.

    Each of ``(lower, upper)`` may be ``None`` (unbounded), a scalar (applied
    to every parameter), an array of length ``n_components`` (one value per
    slip component, broadcast across all patches), or an array of length
    ``n_params = n_patches * n_components`` (one value per parameter).

    Args:
        bounds: The user bounds specification, or None.
        n_patches: Number of patches N.
        n_components: Number of slip components solved for (1 or 2).

    Returns:
        ``(lower, upper)`` per-parameter arrays with ``-inf``/``+inf`` for
        unbounded entries, or None if ``bounds`` is None.

    Raises:
        ValueError: If an array bound has an unsupported length.
    """
    if bounds is None:
        return None
    n_params = n_patches * n_components

    def _expand(val: _BoundValue, fill: float) -> np.ndarray:
        if val is None:
            return np.full(n_params, fill)
        arr = np.asarray(val, dtype=float)
        if arr.ndim == 0:
            return np.full(n_params, float(arr))
        if arr.shape == (n_params,):
            return arr
        if n_components > 1 and arr.shape == (n_components,):
            return np.repeat(arr, n_patches)
        raise ValueError(
            "bounds array must be a scalar, length n_components "
            f"({n_components}), or length n_params ({n_params}); "
            f"got shape {arr.shape}"
        )

    lower_raw, upper_raw = bounds
    return _expand(lower_raw, -np.inf), _expand(upper_raw, np.inf)


def _auto_select_method(bounds: _ExpandedBounds) -> str:
    """Choose solver based on expanded per-parameter bounds."""
    if bounds is None:
        return "wls"
    lower, upper = bounds
    if np.all(lower == 0.0) and np.all(np.isposinf(upper)):
        return "nnls"
    return "bounded_ls"


def _solve(
    G: np.ndarray,
    d: np.ndarray,
    bounds: _ExpandedBounds,
    method: str,
    constraints: tuple[np.ndarray, np.ndarray] | None,
) -> np.ndarray:
    """Dispatch to the appropriate solver.

    Returns:
        Solution vector m, shape (n_params,).
    """
    if method == "wls":
        m_rows, n_cols = G.shape
        if m_rows > n_cols:
            # Overdetermined: normal equations are faster than lstsq (SVD).
            return np.linalg.solve(G.T @ G, G.T @ d)
        # Underdetermined or square: lstsq gives the minimum-norm solution.
        m, _, _, _ = np.linalg.lstsq(G, d, rcond=None)
        return m

    if method == "nnls":
        m, _ = scipy.optimize.nnls(G, d)
        return m

    if method == "bounded_ls":
        lower, upper = (-np.inf, np.inf) if bounds is None else bounds
        result = scipy.optimize.lsq_linear(G, d, bounds=(lower, upper))
        return result.x

    if method == "constrained":
        return _solve_constrained(G, d, bounds, constraints)

    raise ValueError(f"Unknown method: {method!r}")


def _solve_constrained(
    G: np.ndarray,
    d: np.ndarray,
    bounds: _ExpandedBounds,
    constraints: tuple[np.ndarray, np.ndarray] | None,
) -> np.ndarray:
    """Solve via quadratic programming (minimize ||Gm - d||^2 subject to constraints).

    Uses scipy.optimize.minimize with SLSQP, which supports both
    bounds and linear inequality constraints.

    Args:
        G: Design matrix (possibly augmented with regularization).
        d: Data vector (possibly augmented).
        bounds: Per-component (lower, upper) bounds, or None.
        constraints: ``(C, d_ineq)`` such that ``C @ m <= d_ineq``, or None.

    Returns:
        Solution vector m.

    Raises:
        RuntimeError: If SLSQP fails to converge to a feasible solution.
    """
    objective_scale = max(float(np.linalg.norm(G)), float(np.linalg.norm(d)), 1.0)
    G_scaled = G / objective_scale
    d_scaled = d / objective_scale
    GtG = G_scaled.T @ G_scaled
    Gtd = G_scaled.T @ d_scaled

    def objective(m: np.ndarray) -> float:
        r = G_scaled @ m - d_scaled
        return 0.5 * float(r @ r)

    def gradient(m: np.ndarray) -> np.ndarray:
        return GtG @ m - Gtd

    if bounds is not None:
        lower, upper = bounds
        scipy_bounds = list(zip(lower, upper))
    else:
        scipy_bounds = None

    scipy_constraints = []
    if constraints is not None:
        C, d_ineq = constraints
        scipy_constraints.append(
            {
                "type": "ineq",
                "fun": lambda m, C=C, d_ineq=d_ineq: d_ineq - C @ m,
                "jac": lambda m, C=C: -C,
            }
        )

    m0, _, _, _ = np.linalg.lstsq(G, d, rcond=None)
    if bounds is not None:
        m0 = np.clip(m0, bounds[0], bounds[1])

    result = scipy.optimize.minimize(
        objective,
        m0,
        jac=gradient,
        method="SLSQP",
        bounds=scipy_bounds,
        constraints=scipy_constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"Constrained solver failed: {result.message}")
    if constraints is not None:
        C, d_ineq = constraints
        feasibility_tolerance = 1e-8 * max(float(np.max(np.abs(d_ineq))), 1.0)
        max_violation = float(np.max(C @ result.x - d_ineq))
        if max_violation > feasibility_tolerance:
            raise RuntimeError(
                "Constrained solver returned an infeasible solution: "
                f"maximum inequality violation is {max_violation:.3g}"
            )
    return result.x


def _rank_positive_eigs(eigs: np.ndarray) -> np.ndarray:
    """Eigenvalues above the numerical-rank cutoff (as in ``matrix_rank``).

    A graph Laplacian's zero modes come back from ``eigvalsh`` as values of
    order 1e-15 with either sign; a plain ``> 0`` filter keeps them, which
    injects a spurious ``n0 * log(lambda)`` term into ABIC and biases the
    selected smoothing strength.
    """
    eigs = np.abs(np.asarray(eigs, dtype=float))
    if eigs.size == 0:
        return eigs
    tol = eigs.max() * eigs.size * np.finfo(float).eps
    return eigs[eigs > tol]


def _compute_reduced_chi2(
    residuals: np.ndarray,
    W: np.ndarray,
    n_params: int,
) -> float:
    """Compute reduced chi-squared: r^T W r / (M - n_params)."""
    n_obs = len(residuals)
    dof = n_obs - n_params
    if dof <= 0:
        return float("nan")
    weighted_ssr = residuals @ W @ residuals
    return float(weighted_ssr / dof)


def _system_hash(
    greens_matrix: np.ndarray,
    observations: np.ndarray,
    weights: np.ndarray,
    regularizer: np.ndarray | None,
) -> str:
    """Fingerprint the numerical system needed to verify a reproduced solve."""
    digest = hashlib.sha256()
    for label, array in (
        ("G", greens_matrix),
        ("d", observations),
        ("W", weights),
        ("L", regularizer),
    ):
        digest.update(label.encode())
        if array is None:
            digest.update(b"none")
            continue
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.shape).encode())
        digest.update(contiguous.dtype.str.encode())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _lcurve_corner(
    lambdas: np.ndarray,
    misfits: np.ndarray,
    model_norms: np.ndarray,
) -> float:
    """Find the L-curve corner (maximum curvature point).

    Computes curvature of the parametric curve (log misfit, log model_norm)
    and returns the lambda at maximum curvature.

    Returns:
        Optimal lambda at the corner.
    """
    x = np.log(np.maximum(misfits, 1e-300))
    y = np.log(np.maximum(model_norms, 1e-300))

    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    curvature = (dx * ddy - dy * ddx) / (dx**2 + dy**2) ** 1.5

    curvature[0] = -np.inf
    curvature[-1] = -np.inf

    return float(lambdas[np.argmax(curvature)])
