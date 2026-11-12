"""Geodetic datasets: GNSS, InSAR, and vertical observations.

A dataset pairs observation locations with measured values and their
uncertainties. Every dataset exposes the same linear-algebra view:

- ``obs``: the observation vector ``d``
- ``sigma``: one-sigma uncertainties in the same layout
- ``covariance``: the data covariance ``C_d``
- ``project(east, north, up)``: how a displacement field maps into ``obs``

so the inversion code never needs to know which kind of data it holds.

Classes:
    DataSet: Abstract base shared by every data type.
    GNSS: East/North(/Up) displacements or velocities at stations.
    InSAR: Line-of-sight displacements with look vectors.
    Vertical: Single-component vertical observations (leveling, coral uplift).
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import numpy.typing as npt

from geodef.validation import (
    ValidationReport,
    _ReportBuilder,
    as_1d_floats,
    check_covariance,
    check_positive,
    check_range,
)

if TYPE_CHECKING:
    from typing_extensions import Self

    from geodef.fault import Fault

_DISPLACEMENT_UNITS = frozenset({"m", "cm", "mm"})
_VELOCITY_UNITS = frozenset({"m/s", "m/yr", "cm/yr", "mm/yr"})
_FILE_METADATA_KEYS = frozenset(
    {"dataset_name", "quantity", "units", "epoch", "time_span"}
)


def _readonly(array: np.ndarray) -> np.ndarray:
    """Mark an array read-only and return it."""
    array.flags.writeable = False
    return array


def _per_station(
    name: str, values: npt.ArrayLike, n: int, *, unit: str | None = None
) -> np.ndarray:
    """Return a read-only length-``n`` float array, broadcasting a scalar."""
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        array = np.full(n, float(array))
    return _readonly(as_1d_floats(name, array, n=n, unit=unit).copy())


def _table_column(table: object, name: str) -> np.ndarray:
    """Read one named column from a mapping, structured array, or DataFrame."""
    names: object
    if isinstance(table, np.ndarray):
        names = table.dtype.names or ()
    elif isinstance(table, Mapping):
        names = table.keys()
    else:
        names = getattr(table, "columns", ())
    if name not in names:  # type: ignore[operator]
        raise KeyError(name)
    return np.asarray(table[name])  # type: ignore[index]


def _missing_mask(values: np.ndarray) -> np.ndarray:
    """Return a boolean mask of missing (NaN, NaT, or None) entries."""
    if np.issubdtype(values.dtype, np.floating):
        return np.isnan(values)
    if np.issubdtype(values.dtype, np.datetime64):
        return np.isnat(values)
    if values.dtype.kind == "O":
        return np.array(
            [
                value is None
                or (isinstance(value, (float, np.floating)) and np.isnan(value))
                for value in values
            ],
            dtype=bool,
        )
    return np.zeros(values.size, dtype=bool)


def _read_header(path: Path) -> dict[str, str]:
    """Read the ``# key: value`` comment lines at the top of a ``.dat`` file."""
    header: dict[str, str] = {}
    with open(path) as file:
        for line in file:
            if not line.startswith("#"):
                break
            key, separator, value = line.lstrip("#").strip().partition(":")
            if separator:
                header[key.strip()] = value.strip()
    return header


class DataSet(ABC):
    """Abstract base class for geodetic data types.

    Subclasses store their observation arrays and implement ``obs``,
    ``sigma``, ``project``, and ``with_obs``. The base class owns the
    station coordinates, naming, measurement metadata, and covariance.

    Args:
        lon: Observation longitudes in degrees, shape (n_stations,).
        lat: Observation latitudes in degrees, shape (n_stations,).
        name: Dataset name, used to label joint-inversion results and plots.
        station_names: Optional per-station labels, shape (n_stations,).
        quantity: ``'displacement'`` or ``'velocity'``.
        units: Units shared by observations and uncertainties.
        epoch: Optional representative observation epoch.
        time_span: Optional ``(start, end)`` epoch labels.
        covariance: Optional full data covariance, shape (n_obs, n_obs).
            Defaults to ``diag(sigma**2)``.
        validate_covariance: Set ``False`` to skip the positive-definiteness
            check on an explicit covariance.
    """

    greens_type: ClassVar[str] = "displacement"
    #: Array-valued constructor arguments, in the column order of ``.dat`` files.
    _fields: ClassVar[tuple[str, ...]] = ()
    #: The subset of ``_fields`` that ``from_table`` cannot do without.
    _required_fields: ClassVar[tuple[str, ...]] = ()
    _file_header: ClassVar[str] = ""

    def __init__(
        self,
        *,
        lon: npt.ArrayLike,
        lat: npt.ArrayLike,
        name: str | None = None,
        station_names: Sequence[str] | np.ndarray | None = None,
        quantity: str = "displacement",
        units: str = "m",
        epoch: str | None = None,
        time_span: tuple[str, str] | None = None,
        covariance: npt.ArrayLike | None = None,
        validate_covariance: bool = True,
    ) -> None:
        lon_array = _readonly(as_1d_floats("lon", lon, unit="degrees").copy())
        n = lon_array.shape[0]
        if n == 0:
            raise ValueError("a dataset requires at least one station")
        lat_array = _per_station("lat", np.atleast_1d(lat), n, unit="degrees")
        check_range("lat", lat_array, -90.0, 90.0, unit="degrees")
        check_range("lon", lon_array, -360.0, 360.0, unit="degrees")

        if name is not None and (not isinstance(name, str) or not name.strip()):
            raise ValueError(
                "name is the dataset's name and must be a non-empty string; "
                "pass per-station labels as station_names"
            )
        labels: np.ndarray | None = None
        if station_names is not None:
            labels = np.asarray(station_names, dtype=str)
            if labels.shape != (n,):
                raise ValueError(
                    "station_names must be a 1-D sequence of length n_stations"
                )
            labels = _readonly(labels)

        if quantity not in {"displacement", "velocity"}:
            raise ValueError("quantity must be 'displacement' or 'velocity'")
        allowed = _DISPLACEMENT_UNITS if quantity == "displacement" else _VELOCITY_UNITS
        if units not in allowed:
            raise ValueError(
                f"{quantity} units must be one of {sorted(allowed)}, got {units!r}"
            )
        if time_span is not None:
            if len(time_span) != 2 or not all(str(value) for value in time_span):
                raise ValueError("time_span must contain two non-empty epoch labels")
            time_span = (str(time_span[0]), str(time_span[1]))

        self._lon = lon_array
        self._lat = lat_array
        self._name = name
        self._station_names = labels
        self._quantity = quantity
        self._units = units
        self._epoch = epoch
        self._time_span = time_span
        self._covariance: np.ndarray | None = (
            None if covariance is None else np.asarray(covariance, dtype=float)
        )
        self._validate_covariance = validate_covariance
        self._diagonal_covariance: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Identity and metadata
    # ------------------------------------------------------------------

    @property
    def name(self) -> str | None:
        """Dataset name used to label joint-inversion results and plots."""
        return self._name

    @property
    def station_names(self) -> np.ndarray | None:
        """Optional per-station labels, shape (n_stations,)."""
        return self._station_names

    @property
    def quantity(self) -> str:
        """Measurement quantity: ``'displacement'`` or ``'velocity'``."""
        return self._quantity

    @property
    def units(self) -> str:
        """Units of observations and uncertainties."""
        return self._units

    @property
    def epoch(self) -> str | None:
        """Optional representative observation epoch."""
        return self._epoch

    @property
    def time_span(self) -> tuple[str, str] | None:
        """Optional start and end epochs used to estimate the measurement."""
        return self._time_span

    @property
    def lon(self) -> np.ndarray:
        """Observation longitudes in degrees, shape (n_stations,)."""
        return self._lon

    @property
    def lat(self) -> np.ndarray:
        """Observation latitudes in degrees, shape (n_stations,)."""
        return self._lat

    @property
    def n_stations(self) -> int:
        """Number of physical observation locations."""
        return self._lon.shape[0]

    # ------------------------------------------------------------------
    # The linear-algebra view
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def obs(self) -> np.ndarray:
        """Observation vector ``d``, shape (n_obs,)."""

    @property
    @abstractmethod
    def sigma(self) -> np.ndarray:
        """One-sigma uncertainties in the layout of ``obs``."""

    @property
    def n_obs(self) -> int:
        """Length of the observation vector."""
        return self.obs.shape[0]

    @abstractmethod
    def project(
        self, east: np.ndarray, north: np.ndarray, up: np.ndarray
    ) -> np.ndarray:
        """Map a displacement field at the stations into observation space.

        Args:
            east: East displacement at each station, shape (n_stations,).
            north: North displacement at each station, shape (n_stations,).
            up: Up displacement at each station, shape (n_stations,).

        Returns:
            Predicted observations in the layout of ``obs``.
        """

    @abstractmethod
    def with_obs(self, obs: npt.ArrayLike) -> Self:
        """Return a copy holding new observation values.

        Station locations, uncertainties, and metadata are unchanged. Use it
        to wrap a model prediction or a residual as a dataset.

        Args:
            obs: New observation vector in the layout of ``obs``.

        Returns:
            A new dataset of the same type.

        Raises:
            ValueError: If ``obs`` does not have shape (n_obs,).
        """

    @property
    def covariance(self) -> np.ndarray:
        """Data covariance matrix ``C_d``, shape (n_obs, n_obs).

        The explicit covariance when one was supplied, otherwise
        ``diag(sigma**2)``.
        """
        if self._covariance is not None:
            return self._covariance
        if self._diagonal_covariance is None:
            self._diagonal_covariance = _readonly(np.diag(self.sigma**2))
        return self._diagonal_covariance

    def _checked_obs(self, obs: npt.ArrayLike) -> np.ndarray:
        """Validate a replacement observation vector for ``with_obs``."""
        return as_1d_floats("obs", obs, n=self.n_obs)

    def _finish_init(self) -> None:
        """Validate and freeze the explicit covariance once ``n_obs`` is known."""
        if self._covariance is not None:
            check_covariance(
                self._covariance,
                self.n_obs,
                require_positive_definite=self._validate_covariance,
            )
            self._covariance = _readonly(self._covariance.copy())

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def synthetic(
        cls,
        fault: Fault,
        *,
        strike_slip: float | npt.ArrayLike = 0.0,
        dip_slip: float | npt.ArrayLike = 0.0,
        rng: np.random.Generator | int | None = None,
        **kwargs: Any,
    ) -> Self:
        """Forward-model slip on a fault into a dataset of this type.

        The values the fault would produce at the given locations replace
        the observation arguments of the constructor; every other
        constructor argument (``lon``, ``lat``, uncertainties, look
        vectors, names) is passed as a keyword.

        Args:
            fault: Source fault.
            strike_slip: Strike-slip per patch in meters, scalar or (N,).
            dip_slip: Dip-slip per patch in meters, scalar or (N,).
            rng: Random generator or integer seed used to add noise drawn
                from the dataset's own covariance. ``None`` adds no noise.
            **kwargs: Constructor arguments other than the observations.

        Returns:
            A dataset whose ``obs`` is the predicted (optionally noisy) data.

        Examples:
            >>> gnss = GNSS.synthetic(
            ...     fault, lon=lon, lat=lat, dip_slip=1.0, sigma=0.005, rng=0
            ... )
        """
        lon = np.atleast_1d(np.asarray(kwargs["lon"], dtype=float))
        lat = np.atleast_1d(np.asarray(kwargs["lat"], dtype=float))
        east, north, up = fault.displacement(
            lon=lon, lat=lat, strike_slip=strike_slip, dip_slip=dip_slip
        )
        clean = cls._from_displacement(east, north, up, **kwargs)
        if rng is None:
            return clean
        generator = np.random.default_rng(rng)
        if clean._covariance is None:
            noise = generator.normal(0.0, clean.sigma)
        else:
            noise = generator.multivariate_normal(
                np.zeros(clean.n_obs), clean._covariance
            )
        return clean.with_obs(clean.obs + noise)

    @classmethod
    @abstractmethod
    def _from_displacement(
        cls, east: np.ndarray, north: np.ndarray, up: np.ndarray, **kwargs: Any
    ) -> Self:
        """Build an instance whose observations are a displacement field."""

    @classmethod
    def from_table(
        cls,
        table: object,
        *,
        columns: Mapping[str, str] | None = None,
        missing: str = "raise",
        **kwargs: Any,
    ) -> Self:
        """Build a dataset from named table columns.

        Args:
            table: A mapping of column name to values, a structured NumPy
                array, or a pandas/polars ``DataFrame``.
            columns: Mapping from constructor argument (``'lon'``,
                ``'east'``, ``'sigma'``, ``'station_names'``, ...) to the
                table's column name. Defaults to columns named exactly like
                the arguments.
            missing: ``'raise'`` to reject missing values or ``'drop'`` to
                remove every row that contains one.
            **kwargs: Remaining constructor arguments, such as ``name``,
                ``units``, or a scalar ``sigma``.

        Returns:
            A validated dataset of this type.

        Raises:
            KeyError: If a mapped column is absent from the table.
            ValueError: If fields are unknown, lengths differ, or missing
                values are present and ``missing='raise'``.
        """
        if missing not in {"raise", "drop"}:
            raise ValueError("missing must be 'raise' or 'drop'")
        allowed = (*cls._fields, "station_names")
        if columns is None:
            columns = {field: field for field in cls._required_fields}
        unknown = sorted(set(columns) - set(allowed))
        if unknown:
            raise ValueError(
                f"columns contains unknown fields {unknown}; "
                f"{cls.__name__} accepts {list(allowed)}"
            )

        arrays: dict[str, Any] = {}
        for field, source in columns.items():
            try:
                values = _table_column(table, source)
            except KeyError:
                raise KeyError(
                    f"source column {source!r} mapped to {field!r} was not found"
                ) from None
            if values.ndim != 1:
                raise ValueError(
                    f"column {field!r} ({source!r}) must be one-dimensional, "
                    f"got shape {values.shape}"
                )
            arrays[field] = values
        lengths = {field: values.size for field, values in arrays.items()}
        if len(set(lengths.values())) > 1:
            raise ValueError(
                f"mapped table columns must have matching lengths: {lengths}"
            )

        incomplete = np.zeros(next(iter(lengths.values())), dtype=bool)
        for field, values in arrays.items():
            field_missing = _missing_mask(values)
            if field_missing.any() and missing == "raise":
                raise ValueError(
                    f"column {field!r} ({columns[field]!r}) contains missing "
                    "values; pass missing='drop' to remove incomplete rows"
                )
            incomplete |= field_missing
        arrays = {field: values[~incomplete] for field, values in arrays.items()}
        return cls(**arrays, **kwargs)

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------

    def _columns(self) -> list[np.ndarray]:
        """Numeric columns written by ``save``, after lon and lat."""
        raise NotImplementedError

    def save(self, fname: str | Path) -> None:
        """Save to a whitespace-delimited ``.dat`` file readable by ``load``.

        Names and measurement metadata are stored in ``#`` comment lines, so
        the numeric block still loads with ``np.loadtxt``.

        Args:
            fname: Output file path.
        """
        lines = []
        if self._station_names is not None:
            lines.append("names: " + " ".join(self._station_names))
        if self._name is not None:
            lines.append(f"dataset_name: {self._name}")
        lines += [f"quantity: {self._quantity}", f"units: {self._units}"]
        if self._epoch is not None:
            lines.append(f"epoch: {self._epoch}")
        if self._time_span is not None:
            lines.append(f"time_span: {self._time_span[0]} | {self._time_span[1]}")
        lines.append(self._file_header)
        np.savetxt(
            Path(fname),
            np.column_stack([self._lon, self._lat, *self._columns()]),
            header="\n".join(lines),
            fmt="%.8f",
        )

    @classmethod
    def load(cls, fname: str | Path, **kwargs: Any) -> Self:
        """Load a ``.dat`` file written by ``save``.

        Columns follow the constructor's array arguments; see each
        subclass for its layout. Lines starting with ``#`` are comments.

        Args:
            fname: Path to the data file.
            **kwargs: Constructor arguments that override the file contents.

        Returns:
            A new dataset of this type.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        path = Path(fname)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        raw = np.loadtxt(path, comments="#", ndmin=2)
        header = _read_header(path)
        arguments: dict[str, Any] = dict(zip(cls._fields, raw.T))
        if "names" in header:
            arguments["station_names"] = header["names"].split()
        if "dataset_name" in header:
            arguments["name"] = header["dataset_name"]
        for key in ("quantity", "units", "epoch"):
            if key in header:
                arguments[key] = header[key]
        start, separator, end = header.get("time_span", "").partition("|")
        if separator:
            arguments["time_span"] = (start.strip(), end.strip())
        arguments.update(kwargs)
        return cls(**arguments)

    # ------------------------------------------------------------------
    # Interactive checks
    # ------------------------------------------------------------------

    def validate(self) -> ValidationReport:
        """Check this dataset for suspicious-but-legal configurations.

        The constructor already rejects invalid inputs; this report flags
        things worth a look in interactive work: duplicated station
        coordinates, extreme uncertainty spreads, and type-specific checks
        such as the InSAR look-vector sign.

        Returns:
            A :class:`geodef.validation.ValidationReport`.
        """
        builder = _ReportBuilder()
        coordinates = np.column_stack([self._lon, self._lat])
        n_duplicates = coordinates.shape[0] - np.unique(coordinates, axis=0).shape[0]
        if n_duplicates:
            builder.warning(
                "lon/lat",
                f"{n_duplicates} duplicated station coordinate(s); duplicated "
                "rows double-weight those points",
            )
        spread = self.sigma.max() / self.sigma.min()
        if spread > 1e6:
            builder.warning(
                "sigma",
                f"uncertainty spread {spread:.2g}x between largest and "
                "smallest; check units and zero placeholders",
            )
        self._validate_extra(builder)
        return builder.report()

    def _validate_extra(self, builder: _ReportBuilder) -> None:
        """Subclass hook for dataset-specific ``validate`` checks."""

    def __repr__(self) -> str:
        label = "" if self._name is None else f"{self._name!r}, "
        return (
            f"{type(self).__name__}({label}n_stations={self.n_stations}, "
            f"n_obs={self.n_obs}, units={self._units!r})"
        )


class GNSS(DataSet):
    """Displacements or velocities at GNSS stations.

    Three-component (East, North, Up) by default; leave ``up`` as ``None``
    for horizontal-only data. Values and uncertainties may be scalars,
    which are repeated for every station.

    Args:
        lon: Station longitudes in degrees, shape (n_stations,).
        lat: Station latitudes in degrees, shape (n_stations,).
        east: East component values.
        north: North component values.
        up: Up component values, or ``None`` for horizontal-only data.
        sigma: One-sigma uncertainty applied to every component that has no
            component-specific value below.
        sigma_east: East-component uncertainty.
        sigma_north: North-component uncertainty.
        sigma_up: Up-component uncertainty.
        rho: Optional East-North correlation coefficient in [-1, 1], scalar
            or per station. Builds the matching block covariance; mutually
            exclusive with ``covariance``.
        **kwargs: Common dataset arguments (``name``, ``station_names``,
            ``quantity``, ``units``, ``epoch``, ``time_span``,
            ``covariance``, ``validate_covariance``); see :class:`DataSet`.

    File layout for ``save``/``load``: ``lon lat east north up sigma_east
    sigma_north sigma_up``. Horizontal-only data writes ``0``/``1``
    placeholders; reload with ``GNSS.load(fname, components='en')``.

    Examples:
        >>> gnss = GNSS(lon=lon, lat=lat, east=e, north=n, up=u, sigma=0.005)
    """

    _fields = (
        "lon",
        "lat",
        "east",
        "north",
        "up",
        "sigma_east",
        "sigma_north",
        "sigma_up",
    )
    _required_fields = ("lon", "lat", "east", "north")
    _file_header = "lon lat uE uN uZ sigE sigN sigZ"

    def __init__(
        self,
        *,
        lon: npt.ArrayLike,
        lat: npt.ArrayLike,
        east: npt.ArrayLike,
        north: npt.ArrayLike,
        up: npt.ArrayLike | None = None,
        sigma: npt.ArrayLike | None = None,
        sigma_east: npt.ArrayLike | None = None,
        sigma_north: npt.ArrayLike | None = None,
        sigma_up: npt.ArrayLike | None = None,
        rho: npt.ArrayLike | None = None,
        **kwargs: Any,
    ) -> None:
        if rho is not None and kwargs.get("covariance") is not None:
            raise ValueError("Provide either rho or covariance, not both")
        super().__init__(lon=lon, lat=lat, **kwargs)
        n = self.n_stations
        if up is None and sigma_up is not None:
            raise ValueError("sigma_up was given without up values")

        values = {"east": east, "north": north, "up": up}
        sigmas = {"east": sigma_east, "north": sigma_north, "up": sigma_up}
        self._values: dict[str, np.ndarray] = {}
        self._sigmas: dict[str, np.ndarray] = {}
        for component, value in values.items():
            if value is None:
                continue
            component_sigma = sigmas[component]
            if component_sigma is None:
                component_sigma = sigma
            if component_sigma is None:
                raise ValueError(
                    f"sigma_{component} is required (or pass sigma= to use "
                    "one uncertainty for every component)"
                )
            self._values[component] = _per_station(component, value, n, unit=self.units)
            self._sigmas[component] = _per_station(
                f"sigma_{component}", component_sigma, n, unit=self.units
            )
            check_positive(
                f"sigma_{component}", self._sigmas[component], unit=self.units
            )

        if rho is not None:
            self._covariance = self._east_north_covariance(rho)
        self._finish_init()

    def _east_north_covariance(self, rho: npt.ArrayLike) -> np.ndarray:
        """Build the covariance with a per-station East-North correlation."""
        n = self.n_stations
        rho_array = np.broadcast_to(np.asarray(rho, dtype=float), (n,))
        if np.any(np.abs(rho_array) > 1.0):
            raise ValueError("rho must lie in [-1, 1]")
        covariance = np.diag(self.sigma**2)
        east_rows = np.arange(n) * len(self._values)
        off_diagonal = rho_array * self.sigma_east * self.sigma_north
        covariance[east_rows, east_rows + 1] = off_diagonal
        covariance[east_rows + 1, east_rows] = off_diagonal
        return covariance

    @property
    def components(self) -> str:
        """Observed components: ``'enu'`` or ``'en'``."""
        return "enu" if "up" in self._values else "en"

    @property
    def east(self) -> np.ndarray:
        """East component values, shape (n_stations,)."""
        return self._values["east"]

    @property
    def north(self) -> np.ndarray:
        """North component values, shape (n_stations,)."""
        return self._values["north"]

    @property
    def up(self) -> np.ndarray | None:
        """Up component values, or ``None`` for horizontal-only data."""
        return self._values.get("up")

    @property
    def sigma_east(self) -> np.ndarray:
        """East-component one-sigma uncertainties."""
        return self._sigmas["east"]

    @property
    def sigma_north(self) -> np.ndarray:
        """North-component one-sigma uncertainties."""
        return self._sigmas["north"]

    @property
    def sigma_up(self) -> np.ndarray | None:
        """Up-component one-sigma uncertainties, or ``None``."""
        return self._sigmas.get("up")

    @property
    def obs(self) -> np.ndarray:
        """Observations interleaved by station: ``[e1, n1, u1, e2, n2, u2, ...]``.

        Horizontal-only data omits the ``u`` entries.
        """
        return _readonly(np.column_stack(list(self._values.values())).ravel())

    @property
    def sigma(self) -> np.ndarray:
        """One-sigma uncertainties interleaved to match ``obs``."""
        return _readonly(np.column_stack(list(self._sigmas.values())).ravel())

    @property
    def n_obs(self) -> int:
        """Length of the observation vector (components times stations)."""
        return len(self._values) * self.n_stations

    def project(
        self, east: np.ndarray, north: np.ndarray, up: np.ndarray
    ) -> np.ndarray:
        """Interleave displacement components into the layout of ``obs``."""
        fields = {"east": east, "north": north, "up": up}
        return np.column_stack([fields[c] for c in self._values]).ravel()

    def with_obs(self, obs: npt.ArrayLike) -> Self:
        """Return a copy with new values; see :meth:`DataSet.with_obs`."""
        columns = self._checked_obs(obs).reshape(self.n_stations, -1).T
        new = copy.copy(self)
        new._values = {
            component: _readonly(column.copy())
            for component, column in zip(self._values, columns)
        }
        return new

    @classmethod
    def _from_displacement(
        cls,
        east: np.ndarray,
        north: np.ndarray,
        up: np.ndarray,
        *,
        components: str = "enu",
        **kwargs: Any,
    ) -> Self:
        if components not in {"enu", "en"}:
            raise ValueError("components must be 'enu' or 'en'")
        return cls(
            east=east, north=north, up=up if components == "enu" else None, **kwargs
        )

    @classmethod
    def load(cls, fname: str | Path, *, components: str = "enu", **kwargs: Any) -> Self:
        """Load a GNSS ``.dat`` file.

        Expected columns: ``lon lat east north up sigma_east sigma_north
        sigma_up``.

        Args:
            fname: Path to the data file.
            components: ``'enu'`` to keep all components or ``'en'`` to
                discard the vertical.
            **kwargs: Constructor arguments that override the file contents.

        Returns:
            A new ``GNSS`` instance.
        """
        if components not in {"enu", "en"}:
            raise ValueError("components must be 'enu' or 'en'")
        if components == "en":
            kwargs = {"up": None, "sigma_up": None, **kwargs}
        return super().load(fname, **kwargs)

    def _columns(self) -> list[np.ndarray]:
        n = self.n_stations
        up = self.up if self.up is not None else np.zeros(n)
        sigma_up = self.sigma_up if self.sigma_up is not None else np.ones(n)
        return [
            self.east,
            self.north,
            up,
            self.sigma_east,
            self.sigma_north,
            sigma_up,
        ]

    def to_gmt(self, fname: str | Path) -> None:
        """Save horizontal vectors as ``lon lat uE uN sigE sigN`` for GMT.

        Args:
            fname: Output file path.
        """
        np.savetxt(
            Path(fname),
            np.column_stack(
                [
                    self.lon,
                    self.lat,
                    self.east,
                    self.north,
                    self.sigma_east,
                    self.sigma_north,
                ]
            ),
            header="lon lat uE uN sigE sigN",
            fmt="%.8f",
        )


class InSAR(DataSet):
    """Line-of-sight displacements from SAR interferometry.

    Each pixel has a scalar LOS value and a **unit** look vector pointing
    from the ground **to the satellite**, so ``look_u`` is positive and
    uplift produces positive LOS motion. If your processing chain provides
    satellite-to-ground vectors, negate all three components;
    ``validate()`` flags the likely mix-up. Scalars are repeated for every
    pixel.

    Args:
        lon: Pixel longitudes in degrees, shape (n_stations,).
        lat: Pixel latitudes in degrees, shape (n_stations,).
        los: Line-of-sight values.
        sigma: One-sigma uncertainties.
        look_e: East component of the look vector.
        look_n: North component of the look vector.
        look_u: Up component of the look vector.
        normalize_look: Renormalize the look vectors to unit length instead
            of requiring them to arrive normalized.
        **kwargs: Common dataset arguments; see :class:`DataSet`.

    File layout for ``save``/``load``: ``lon lat los sigma look_e look_n
    look_u``.
    """

    _fields = ("lon", "lat", "los", "sigma", "look_e", "look_n", "look_u")
    _required_fields = _fields
    _file_header = "lon lat uLOS sigLOS losE losN losU"

    def __init__(
        self,
        *,
        lon: npt.ArrayLike,
        lat: npt.ArrayLike,
        los: npt.ArrayLike,
        sigma: npt.ArrayLike,
        look_e: npt.ArrayLike,
        look_n: npt.ArrayLike,
        look_u: npt.ArrayLike,
        normalize_look: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(lon=lon, lat=lat, **kwargs)
        n = self.n_stations
        self._los = _per_station("los", los, n, unit=self.units)
        self._sigma = _per_station("sigma", sigma, n, unit=self.units)
        check_positive("sigma", self._sigma, unit=self.units)

        look = np.column_stack(
            [
                _per_station(f"look_{axis}", value, n, unit="unit vector component")
                for axis, value in (("e", look_e), ("n", look_n), ("u", look_u))
            ]
        )
        norms = np.linalg.norm(look, axis=1)
        if normalize_look:
            check_positive("look vector norm", norms)
            look = look / norms[:, None]
        elif np.any(np.abs(norms - 1.0) > 1e-3):
            worst = float(norms[np.argmax(np.abs(norms - 1.0))])
            raise ValueError(
                "look vectors must be unit length: worst norm is "
                f"{worst:.6g}. Pass normalize_look=True to renormalize, "
                "or normalize (look_e, look_n, look_u) yourself."
            )
        self._look = _readonly(look)
        self._finish_init()

    def _validate_extra(self, builder: _ReportBuilder) -> None:
        if np.any(self.look_u < 0):
            builder.warning(
                "look_u",
                "negative up-components: these look vectors appear to point "
                "satellite-to-ground, but GeoDef expects ground-to-satellite "
                "(look_u > 0). If so, negate all three components or the "
                "predicted LOS sign will be reversed.",
            )

    @property
    def look_e(self) -> np.ndarray:
        """East components of the unit look vectors, shape (n_stations,)."""
        return self._look[:, 0]

    @property
    def look_n(self) -> np.ndarray:
        """North components of the unit look vectors, shape (n_stations,)."""
        return self._look[:, 1]

    @property
    def look_u(self) -> np.ndarray:
        """Up components of the unit look vectors, shape (n_stations,)."""
        return self._look[:, 2]

    @property
    def los(self) -> np.ndarray:
        """Line-of-sight values, shape (n_stations,)."""
        return self._los

    @property
    def obs(self) -> np.ndarray:
        """Line-of-sight observations, shape (n_stations,)."""
        return self._los

    @property
    def sigma(self) -> np.ndarray:
        """One-sigma uncertainties, shape (n_stations,)."""
        return self._sigma

    def project(
        self, east: np.ndarray, north: np.ndarray, up: np.ndarray
    ) -> np.ndarray:
        """Project displacement onto the line of sight."""
        return self.look_e * east + self.look_n * north + self.look_u * up

    def with_obs(self, obs: npt.ArrayLike) -> Self:
        """Return a copy with new values; see :meth:`DataSet.with_obs`."""
        new = copy.copy(self)
        new._los = _readonly(self._checked_obs(obs).copy())
        return new

    @classmethod
    def _from_displacement(
        cls,
        east: np.ndarray,
        north: np.ndarray,
        up: np.ndarray,
        **kwargs: Any,
    ) -> Self:
        template = cls(los=0.0, **kwargs)
        return template.with_obs(template.project(east, north, up))

    def _columns(self) -> list[np.ndarray]:
        return [self._los, self._sigma, self.look_e, self.look_n, self.look_u]

    def to_gmt(self, fname: str | Path) -> None:
        """Save ``lon lat uLOS`` for GMT gridding commands.

        Args:
            fname: Output file path.
        """
        np.savetxt(
            Path(fname),
            np.column_stack([self.lon, self.lat, self._los]),
            header="lon lat uLOS",
            fmt="%.8f",
        )


class Vertical(DataSet):
    """Single-component vertical observations.

    Suitable for leveling, coral uplift, tide gauges, or any data that
    measures only the vertical. Scalars are repeated for every point.

    Args:
        lon: Observation longitudes in degrees, shape (n_stations,).
        lat: Observation latitudes in degrees, shape (n_stations,).
        displacement: Vertical values, positive up.
        sigma: One-sigma uncertainties.
        **kwargs: Common dataset arguments; see :class:`DataSet`.

    File layout for ``save``/``load``: ``lon lat displacement sigma``.
    """

    _fields = ("lon", "lat", "displacement", "sigma")
    _required_fields = _fields
    _file_header = "lon lat uZ sigZ"

    def __init__(
        self,
        *,
        lon: npt.ArrayLike,
        lat: npt.ArrayLike,
        displacement: npt.ArrayLike,
        sigma: npt.ArrayLike,
        **kwargs: Any,
    ) -> None:
        super().__init__(lon=lon, lat=lat, **kwargs)
        n = self.n_stations
        self._displacement = _per_station("displacement", displacement, n)
        self._sigma = _per_station("sigma", sigma, n, unit=self.units)
        check_positive("sigma", self._sigma, unit=self.units)
        self._finish_init()

    @property
    def displacement(self) -> np.ndarray:
        """Vertical values, positive up, shape (n_stations,)."""
        return self._displacement

    @property
    def obs(self) -> np.ndarray:
        """Vertical observations, shape (n_stations,)."""
        return self._displacement

    @property
    def sigma(self) -> np.ndarray:
        """One-sigma uncertainties, shape (n_stations,)."""
        return self._sigma

    def project(
        self, east: np.ndarray, north: np.ndarray, up: np.ndarray
    ) -> np.ndarray:
        """Keep the vertical component."""
        return up

    def with_obs(self, obs: npt.ArrayLike) -> Self:
        """Return a copy with new values; see :meth:`DataSet.with_obs`."""
        new = copy.copy(self)
        new._displacement = _readonly(self._checked_obs(obs).copy())
        return new

    @classmethod
    def _from_displacement(
        cls,
        east: np.ndarray,
        north: np.ndarray,
        up: np.ndarray,
        **kwargs: Any,
    ) -> Self:
        return cls(displacement=up, **kwargs)

    def _columns(self) -> list[np.ndarray]:
        return [self._displacement, self._sigma]

    def to_gmt(self, fname: str | Path) -> None:
        """Save ``lon lat uZ`` for GMT.

        Args:
            fname: Output file path.
        """
        np.savetxt(
            Path(fname),
            np.column_stack([self.lon, self.lat, self._displacement]),
            header="lon lat uZ",
            fmt="%.8f",
        )


def spatial_covariance(
    lon: np.ndarray,
    lat: np.ndarray,
    sill: float,
    correlation_length: float,
    *,
    model: str = "exponential",
    nugget: float = 0.0,
) -> np.ndarray:
    """Build a spatially-correlated data covariance matrix ``C_d``.

    Off-diagonal correlation decays with great-circle distance between
    observation points, following a stationary, isotropic covariance model.
    This is the standard way to represent spatially-correlated InSAR noise
    (atmosphere, orbits) instead of assuming diagonal ``C_d``. Pass the result
    as ``covariance=`` to a single-component-per-station dataset (``InSAR`` or
    ``Vertical``), or thread it through ``geodef.invert.solve()``.

    The covariance is ``C_ij = sill * rho(d_ij) + nugget * delta_ij`` where
    ``rho`` is the correlation function:

    - ``'exponential'``: ``rho(d) = exp(-d / L)``
    - ``'gaussian'``:    ``rho(d) = exp(-(d / L)^2)``

    Args:
        lon: Observation longitudes in degrees, shape (n,).
        lat: Observation latitudes in degrees, shape (n,).
        sill: Correlated variance (the ``d -> 0`` covariance), in the squared
            units of the observations (e.g. m^2).
        correlation_length: Decorrelation length ``L`` in meters.
        model: Correlation function, ``'exponential'`` or ``'gaussian'``.
        nugget: Uncorrelated (white-noise) variance added to the diagonal,
            in the same squared units as ``sill``.

    Returns:
        Symmetric positive-definite covariance matrix, shape (n, n).

    Raises:
        ValueError: If ``sill`` or ``correlation_length`` is not positive,
            ``nugget`` is negative, or ``model`` is unknown.
    """
    if sill <= 0:
        raise ValueError("sill must be positive")
    if correlation_length <= 0:
        raise ValueError("correlation_length must be positive")
    if nugget < 0:
        raise ValueError("nugget must be non-negative")

    lat_r = np.radians(np.asarray(lat, dtype=float))
    lon_r = np.radians(np.asarray(lon, dtype=float))
    dlat = lat_r[:, None] - lat_r[None, :]
    dlon = lon_r[:, None] - lon_r[None, :]
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat_r)[:, None] * np.cos(lat_r)[None, :] * np.sin(dlon / 2.0) ** 2
    )
    dist = 2.0 * 6371000.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))

    ratio = dist / correlation_length
    if model == "exponential":
        rho = np.exp(-ratio)
    elif model == "gaussian":
        rho = np.exp(-(ratio**2))
    else:
        raise ValueError(f"model must be 'exponential' or 'gaussian', got {model!r}")

    cov = sill * rho
    cov[np.diag_indices_from(cov)] += nugget
    return cov
