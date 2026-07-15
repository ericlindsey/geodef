"""Geodetic dataset classes for inversion and forward modeling.

Provides a ``DataSet`` base class and concrete subclasses for common geodetic
data types. Each subclass specifies what Green's function output it needs
(``greens_type``) and how to map raw components to observations (``project()``).

Classes:
    DataSet: Abstract base for all geodetic data types.
    GNSS: Three-component (or horizontal-only) displacement/velocity data.
    InSAR: Line-of-sight displacement data with look vectors.
    Vertical: Single-component vertical displacement (e.g. coral uplift).
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path

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


def _make_readonly(arr: np.ndarray) -> np.ndarray:
    """Set an array's writeable flag to False and return it."""
    arr.flags.writeable = False
    return arr


def _broadcast_1d(name: str, values: npt.ArrayLike, n: int) -> np.ndarray:
    """Broadcast a scalar or validate a one-dimensional array."""
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        array = np.full(n, float(array))
    return as_1d_floats(name, array, n=n)


def _validate_measurement_metadata(quantity: str, units: str) -> None:
    """Validate observation semantics without converting user values."""
    if quantity not in {"displacement", "velocity"}:
        raise ValueError("quantity must be 'displacement' or 'velocity'")
    displacement_units = {"m", "cm", "mm"}
    velocity_units = {"m/s", "m/yr", "cm/yr", "mm/yr"}
    allowed = displacement_units if quantity == "displacement" else velocity_units
    if units not in allowed:
        raise ValueError(
            f"{quantity} units must be one of {sorted(allowed)}, got {units!r}"
        )


def _names_header(name: np.ndarray | None) -> str:
    """Encode site names as a leading ``savetxt`` header line, or ``''``.

    The numeric block stays purely numeric; names ride along in a ``# names:``
    comment so the file still round-trips through ``np.loadtxt``.
    """
    if name is None:
        return ""
    return "names: " + " ".join(str(x) for x in name) + "\n"


def _metadata_header(dataset: "DataSet") -> str:
    """Encode common dataset metadata as leading comment lines."""
    lines = [_names_header(dataset.station_names)]
    if dataset.dataset_name is not None:
        lines.append(f"dataset_name: {dataset.dataset_name}\n")
    lines.extend([f"quantity: {dataset.quantity}\n", f"units: {dataset.units}\n"])
    if dataset.epoch is not None:
        lines.append(f"epoch: {dataset.epoch}\n")
    if dataset.time_span is not None:
        lines.append(f"time_span: {dataset.time_span[0]} | {dataset.time_span[1]}\n")
    return "".join(lines)


def _read_names(path: Path) -> np.ndarray | None:
    """Read the ``# names:`` comment line from a ``.dat`` file, or ``None``."""
    with open(path) as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            stripped = line.lstrip("#").strip()
            if stripped.startswith("names:"):
                tokens = stripped[len("names:") :].split()
                return np.asarray(tokens, dtype=str)
    return None


def _read_metadata(
    path: Path,
) -> tuple[str | None, str, str, str | None, tuple[str, str] | None]:
    """Read common metadata comments from a dataset text file."""
    values: dict[str, str] = {}
    with open(path) as file:
        for line in file:
            if not line.startswith("#"):
                break
            stripped = line.lstrip("#").strip()
            key, separator, value = stripped.partition(":")
            if separator and key in {
                "dataset_name",
                "quantity",
                "units",
                "epoch",
                "time_span",
            }:
                values[key] = value.strip()
    raw_span = values.get("time_span")
    time_span = None
    if raw_span is not None:
        start, separator, end = raw_span.partition("|")
        if separator:
            time_span = (start.strip(), end.strip())
    return (
        values.get("dataset_name"),
        values.get("quantity", "displacement"),
        values.get("units", "m"),
        values.get("epoch"),
        time_span,
    )


class DataSet(ABC):
    """Abstract base class for geodetic data types.

    Subclasses must implement ``project()`` and set ``greens_type``.
    The base class provides common infrastructure for coordinates,
    observations, uncertainties, and covariance.

    Args:
        lat: Observation point latitudes, shape (n_stations,).
        lon: Observation point longitudes, shape (n_stations,).
    """

    greens_type: str = "displacement"

    def __init__(
        self,
        *,
        lon: np.ndarray,
        lat: np.ndarray,
        name: np.ndarray | None = None,
        dataset_name: str | None = None,
        quantity: str = "displacement",
        units: str = "m",
        epoch: str | None = None,
        time_span: tuple[str, str] | None = None,
        covariance: np.ndarray | None = None,
        validate_covariance: bool = True,
    ) -> None:
        lon = _make_readonly(as_1d_floats("lon", lon, unit="degrees"))
        lat = _make_readonly(as_1d_floats("lat", lat, n=lon.shape[0], unit="degrees"))

        n = lat.shape[0]
        if n == 0:
            raise ValueError("a dataset requires at least one station")
        check_range("lat", lat, -90.0, 90.0, unit="degrees")
        check_range("lon", lon, -360.0, 360.0, unit="degrees")

        if name is not None:
            name = np.asarray(name, dtype=str)
            if name.shape != (n,):
                raise ValueError("name must be a 1-D array of length n_stations")
            name = _make_readonly(name)

        if dataset_name is not None and not dataset_name.strip():
            raise ValueError("dataset_name must not be empty")
        _validate_measurement_metadata(quantity, units)
        if time_span is not None:
            if len(time_span) != 2 or not all(str(value) for value in time_span):
                raise ValueError("time_span must contain two non-empty epoch labels")
            time_span = (str(time_span[0]), str(time_span[1]))

        self._lon = lon
        self._lat = lat
        self._name = name
        self._dataset_name = dataset_name
        self._quantity = quantity
        self._units = units
        self._epoch = epoch
        self._time_span = time_span
        self._covariance_explicit = covariance
        self._validate_covariance = validate_covariance
        self._covariance_cache: np.ndarray | None = None

    @property
    def name(self) -> np.ndarray | None:
        """Optional per-station site names, shape (n_stations,), or None."""
        return self._name

    @property
    def station_names(self) -> np.ndarray | None:
        """Optional per-station labels, shape ``(n_stations,)``."""
        return self._name

    @property
    def dataset_name(self) -> str | None:
        """Stable identifier used for joint results and plots."""
        return self._dataset_name

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
        """Observation point longitudes, shape (n_stations,)."""
        return self._lon

    @property
    def lat(self) -> np.ndarray:
        """Observation point latitudes, shape (n_stations,)."""
        return self._lat

    @property
    def n_stations(self) -> int:
        """Number of physical observation locations."""
        return self._lat.shape[0]

    @property
    @abstractmethod
    def obs(self) -> np.ndarray:
        """Observation vector."""
        ...

    @property
    @abstractmethod
    def sigma(self) -> np.ndarray:
        """1-sigma uncertainties, same shape as ``obs``."""
        ...

    @property
    @abstractmethod
    def n_obs(self) -> int:
        """Length of the observation vector."""
        ...

    @abstractmethod
    def project(self, *components: np.ndarray) -> np.ndarray:
        """Map Green's function components to this data type's observation space.

        For displacement data types, receives three arrays (ue, un, uz).
        For strain data types, receives six arrays (exx, eyy, ezz, exy, exz, eyz).

        Args:
            *components: Component arrays, each shape (n_stations,).

        Returns:
            Projected observations, shape (n_obs,).
        """
        ...

    @property
    def covariance(self) -> np.ndarray:
        """Data covariance matrix, shape (n_obs, n_obs).

        Returns the user-provided covariance if set, otherwise builds a
        diagonal matrix from ``sigma``.
        """
        if self._covariance_cache is not None:
            return self._covariance_cache

        if self._covariance_explicit is not None:
            cov = np.asarray(self._covariance_explicit, dtype=float)
        else:
            cov = np.diag(self.sigma**2)

        self._covariance_cache = _make_readonly(cov)
        return self._covariance_cache

    def _validate_covariance_shape(self) -> None:
        """Validate explicit covariance. Call after n_obs is known.

        Checks shape, symmetry, and (unless the constructor was passed
        ``validate_covariance=False``) positive definiteness.
        """
        if self._covariance_explicit is not None:
            check_covariance(
                self._covariance_explicit,
                self.n_obs,
                require_positive_definite=self._validate_covariance,
            )

    def validate(self) -> ValidationReport:
        """Check this dataset for suspicious-but-legal configurations.

        Constructor validation already rejects invalid inputs; this report
        flags things worth a look in interactive work: duplicated station
        coordinates, extreme uncertainty spreads, and subclass-specific
        checks (e.g. InSAR look-vector sign).

        Returns:
            A :class:`geodef.validation.ValidationReport`.
        """
        b = _ReportBuilder()
        coords = np.column_stack([self._lon, self._lat])
        uniq = np.unique(coords, axis=0)
        if uniq.shape[0] < coords.shape[0]:
            b.warning(
                "lon/lat",
                f"{coords.shape[0] - uniq.shape[0]} duplicated station "
                "coordinate(s); duplicated rows double-weight those points",
            )
        sigma = np.asarray(self.sigma, dtype=float)
        if sigma.size and sigma.max() / sigma.min() > 1e6:
            b.warning(
                "sigma",
                f"uncertainty spread {sigma.max() / sigma.min():.2g}x between "
                "largest and smallest; check units and zero placeholders",
            )
        self._validate_extra(b)
        return b.report()

    def _validate_extra(self, builder: "_ReportBuilder") -> None:
        """Subclass hook for dataset-specific validate() checks."""


class GNSS(DataSet):
    """Three-component displacement or velocity observations at GNSS stations.

    Supports full 3-component (E, N, U) or horizontal-only (E, N) data.
    Pass ``vu=None, su=None`` for horizontal-only.

    Args:
        lon: Station longitudes, shape (n_stations,).
        lat: Station latitudes, shape (n_stations,).
        ve: East component values, shape (n_stations,).
        vn: North component values, shape (n_stations,).
        vu: Up component values, shape (n_stations,), or None for horizontal-only.
        se: East component 1-sigma, shape (n_stations,).
        sn: North component 1-sigma, shape (n_stations,).
        su: Up component 1-sigma, shape (n_stations,), or None for horizontal-only.
        rho: Optional East-North correlation coefficient in [-1, 1], scalar
            or shape (n_stations,). When given, a block covariance with the
            correct per-station E-N correlation is built automatically.
            Mutually exclusive with ``covariance``.
        name: Optional per-station site names, shape (n_stations,).
        covariance: Optional full covariance matrix, shape (n_obs, n_obs).
    """

    greens_type = "displacement"

    def __init__(
        self,
        *,
        lon: np.ndarray,
        lat: np.ndarray,
        ve: np.ndarray,
        vn: np.ndarray,
        vu: np.ndarray | None = None,
        se: np.ndarray,
        sn: np.ndarray,
        su: np.ndarray | None = None,
        rho: np.ndarray | float | None = None,
        name: np.ndarray | None = None,
        dataset_name: str | None = None,
        quantity: str = "displacement",
        units: str = "m",
        epoch: str | None = None,
        time_span: tuple[str, str] | None = None,
        covariance: np.ndarray | None = None,
        validate_covariance: bool = True,
    ) -> None:
        if rho is not None and covariance is not None:
            raise ValueError("Provide either rho or covariance, not both")
        super().__init__(
            lon=lon,
            lat=lat,
            name=name,
            dataset_name=dataset_name,
            quantity=quantity,
            units=units,
            epoch=epoch,
            time_span=time_span,
            covariance=covariance,
            validate_covariance=validate_covariance,
        )

        n = self.n_stations
        ve = as_1d_floats("ve", ve, n=n, unit="meters or meters/time")
        vn = as_1d_floats("vn", vn, n=n, unit="meters or meters/time")
        se = as_1d_floats("se", se, n=n, unit="same as ve")
        sn = as_1d_floats("sn", sn, n=n, unit="same as vn")

        # Validate vertical component consistency
        if (vu is None) != (su is None):
            raise ValueError("vu and su must both be None or both provided")

        if vu is not None:
            assert su is not None  # enforced by the both-or-neither check
            vu = as_1d_floats("vu", vu, n=n, unit="meters or meters/time")
            su = as_1d_floats("su", su, n=n, unit="same as vu")

        check_positive("se", se, unit="same as ve")
        check_positive("sn", sn, unit="same as vn")
        if su is not None:
            check_positive("su", su, unit="same as vu")

        self._ve = _make_readonly(ve)
        self._vn = _make_readonly(vn)
        self._vu = _make_readonly(vu) if vu is not None else None
        self._se = _make_readonly(se)
        self._sn = _make_readonly(sn)
        self._su = _make_readonly(su) if su is not None else None

        if rho is not None:
            self._covariance_explicit = self._build_en_covariance(rho)

        self._validate_covariance_shape()

    def _build_en_covariance(self, rho: np.ndarray | float) -> np.ndarray:
        """Build a block covariance with per-station East-North correlation.

        Diagonal entries are the component variances; each station's E-N pair
        gets an off-diagonal ``rho * se * sn``. The Up component (if present)
        stays uncorrelated.

        Args:
            rho: East-North correlation coefficient in [-1, 1], scalar or
                shape (n_stations,).

        Returns:
            Covariance matrix, shape (n_obs, n_obs).

        Raises:
            ValueError: If ``rho`` has the wrong shape or lies outside [-1, 1].
        """
        n = self.n_stations
        rho_arr = np.broadcast_to(np.asarray(rho, dtype=float), (n,))
        if np.any(np.abs(rho_arr) > 1.0):
            raise ValueError("rho must lie in [-1, 1]")

        n_comp = 3 if self._vu is not None else 2
        cov = np.diag(self.sigma**2)
        e_idx = np.arange(n) * n_comp
        n_idx = e_idx + 1
        off = rho_arr * self._se * self._sn
        cov[e_idx, n_idx] = off
        cov[n_idx, e_idx] = off
        return cov

    @property
    def components(self) -> str:
        """Active component string: ``'enu'`` or ``'en'``."""
        return "enu" if self._vu is not None else "en"

    @property
    def east(self) -> np.ndarray:
        """East component values, shape ``(n_stations,)``."""
        return self._ve

    @property
    def north(self) -> np.ndarray:
        """North component values, shape ``(n_stations,)``."""
        return self._vn

    @property
    def up(self) -> np.ndarray | None:
        """Up component values, or ``None`` for horizontal data."""
        return self._vu

    @property
    def sigma_east(self) -> np.ndarray:
        """East-component standard deviations."""
        return self._se

    @property
    def sigma_north(self) -> np.ndarray:
        """North-component standard deviations."""
        return self._sn

    @property
    def sigma_up(self) -> np.ndarray | None:
        """Up-component standard deviations, or ``None``."""
        return self._su

    @property
    def n_obs(self) -> int:
        """Length of observation vector (n_components * n_stations)."""
        n_comp = 3 if self._vu is not None else 2
        return n_comp * self.n_stations

    @property
    def obs(self) -> np.ndarray:
        """Observation vector, interleaved by station.

        Returns:
            For 3-component: ``[e1, n1, u1, e2, n2, u2, ...]``.
            For 2-component: ``[e1, n1, e2, n2, ...]``.
        """
        if self._vu is not None:
            return _make_readonly(
                np.column_stack([self._ve, self._vn, self._vu]).ravel()
            )
        return _make_readonly(np.column_stack([self._ve, self._vn]).ravel())

    @property
    def sigma(self) -> np.ndarray:
        """1-sigma uncertainties, interleaved to match ``obs``."""
        if self._su is not None:
            return _make_readonly(
                np.column_stack([self._se, self._sn, self._su]).ravel()
            )
        return _make_readonly(np.column_stack([self._se, self._sn]).ravel())

    def project(self, *components: np.ndarray) -> np.ndarray:
        """Project displacement components into GNSS observation space.

        Args:
            *components: (ue, un, uz) displacement arrays, each (n_stations,).

        Returns:
            Interleaved components, shape (n_obs,).
        """
        ue, un, uz = components
        if self._vu is not None:
            return np.column_stack([ue, un, uz]).ravel()
        return np.column_stack([ue, un]).ravel()

    def save(self, fname: str | Path, *, format: str = "dat") -> None:
        """Save GNSS data to a whitespace-delimited file.

        Always writes 8 columns (``lon lat uE uN uZ sigE sigN sigZ``) so
        the file can always be read back with ``load()``.  For horizontal-only
        instances, ``uZ`` and ``sigZ`` are written as ``0.0`` / ``1.0``
        placeholders; reload with ``components='en'`` to discard them.

        Args:
            fname: Output file path.
            format: File format.  Only ``"dat"`` is supported.

        Raises:
            ValueError: If ``format`` is not ``"dat"``.
        """
        if format != "dat":
            raise ValueError(f"Unknown format {format!r}; use 'dat'")

        vu = self._vu if self._vu is not None else np.zeros(self.n_stations)
        su = self._su if self._su is not None else np.ones(self.n_stations)
        data = np.column_stack(
            [
                self._lon,
                self._lat,
                self._ve,
                self._vn,
                vu,
                self._se,
                self._sn,
                su,
            ]
        )
        header = _metadata_header(self) + "lon lat uE uN uZ sigE sigN sigZ"
        np.savetxt(Path(fname), data, header=header, fmt="%.8f")

    def to_gmt(self, fname: str | Path) -> None:
        """Save GNSS data in GMT-compatible format.

        Writes ``lon lat uE uN sigE sigN`` (horizontal) with a ``#``-prefixed
        header, suitable for ``psvelo`` or ``psxy`` GMT commands.

        Args:
            fname: Output file path.
        """
        header = "lon lat uE uN sigE sigN"
        data = np.column_stack(
            [
                self._lon,
                self._lat,
                self._ve,
                self._vn,
                self._se,
                self._sn,
            ]
        )
        np.savetxt(Path(fname), data, header=header, fmt="%.8f")

    @classmethod
    def load(
        cls,
        fname: str | Path,
        *,
        components: str = "enu",
    ) -> "GNSS":
        """Load GNSS data from a whitespace-delimited .dat file.

        Expected columns: ``lon lat uE uN uZ sigE sigN sigZ``

        Lines starting with ``#`` are treated as comments.

        Args:
            fname: Path to the data file.
            components: Which components to keep: ``'enu'`` or ``'en'``.

        Returns:
            A new ``GNSS`` instance.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        path = Path(fname)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        raw = np.loadtxt(path, comments="#", ndmin=2)
        lon, lat = raw[:, 0], raw[:, 1]
        ve, vn = raw[:, 2], raw[:, 3]
        se, sn = raw[:, 5], raw[:, 6]
        vu: np.ndarray | None = raw[:, 4]
        su: np.ndarray | None = raw[:, 7]

        if components == "en":
            vu, su = None, None

        dataset_name, quantity, units, epoch, time_span = _read_metadata(path)

        return cls(
            lon=lon,
            lat=lat,
            ve=ve,
            vn=vn,
            vu=vu,
            se=se,
            sn=sn,
            su=su,
            name=_read_names(path),
            dataset_name=dataset_name,
            quantity=quantity,
            units=units,
            epoch=epoch,
            time_span=time_span,
        )


class InSAR(DataSet):
    """Line-of-sight displacement observations (e.g. from SAR interferometry).

    Each pixel has a scalar LOS measurement and a 3-component **unit** look
    vector pointing from the ground **to the satellite** (so ``look_u`` is
    positive and uplift produces positive LOS motion toward the satellite).
    If your processing chain provides satellite-to-ground vectors, negate
    all three components; ``validate()`` flags the likely mix-up.

    Args:
        lon: Pixel longitudes, shape (n_stations,).
        lat: Pixel latitudes, shape (n_stations,).
        los: LOS displacement values, shape (n_stations,).
        sigma: 1-sigma uncertainties, shape (n_stations,).
        look_e: East component of look vector, shape (n_stations,).
        look_n: North component of look vector, shape (n_stations,).
        look_u: Up component of look vector, shape (n_stations,).
        covariance: Optional full covariance matrix, shape (n_obs, n_obs).
        validate_covariance: Set ``False`` to skip the positive-definiteness
            check on an explicit covariance (advanced semidefinite cases).
        normalize_look: Renormalize the look vectors to unit length instead
            of requiring them to arrive normalized.
    """

    greens_type = "displacement"

    def __init__(
        self,
        *,
        lon: np.ndarray,
        lat: np.ndarray,
        los: np.ndarray,
        sigma: np.ndarray,
        look_e: np.ndarray,
        look_n: np.ndarray,
        look_u: np.ndarray,
        name: np.ndarray | None = None,
        dataset_name: str | None = None,
        quantity: str = "displacement",
        units: str = "m",
        epoch: str | None = None,
        time_span: tuple[str, str] | None = None,
        covariance: np.ndarray | None = None,
        validate_covariance: bool = True,
        normalize_look: bool = False,
    ) -> None:
        super().__init__(
            lon=lon,
            lat=lat,
            name=name,
            dataset_name=dataset_name,
            quantity=quantity,
            units=units,
            epoch=epoch,
            time_span=time_span,
            covariance=covariance,
            validate_covariance=validate_covariance,
        )

        n = self.n_stations
        los = as_1d_floats("los", los, n=n, unit="meters")
        sigma = as_1d_floats("sigma", sigma, n=n, unit="meters")
        look_e = as_1d_floats("look_e", look_e, n=n, unit="unit vector component")
        look_n = as_1d_floats("look_n", look_n, n=n, unit="unit vector component")
        look_u = as_1d_floats("look_u", look_u, n=n, unit="unit vector component")

        check_positive("sigma", sigma, unit="meters")

        norms = np.sqrt(look_e**2 + look_n**2 + look_u**2)
        if normalize_look:
            check_positive("look vector norm", norms)
            look_e = look_e / norms
            look_n = look_n / norms
            look_u = look_u / norms
        elif np.any(np.abs(norms - 1.0) > 1e-3):
            worst = float(norms[np.argmax(np.abs(norms - 1.0))])
            raise ValueError(
                "look vectors must be unit length: worst norm is "
                f"{worst:.6g}. Pass normalize_look=True to renormalize, "
                "or normalize (look_e, look_n, look_u) yourself."
            )

        self._los = _make_readonly(los)
        self._sigma = _make_readonly(sigma)
        self._look_e = _make_readonly(np.asarray(look_e))
        self._look_n = _make_readonly(np.asarray(look_n))
        self._look_u = _make_readonly(np.asarray(look_u))

        self._validate_covariance_shape()

    def _validate_extra(self, builder) -> None:
        if np.any(self._look_u < 0):
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
        return self._look_e

    @property
    def look_n(self) -> np.ndarray:
        """North components of the unit look vectors, shape (n_stations,)."""
        return self._look_n

    @property
    def look_u(self) -> np.ndarray:
        """Up components of the unit look vectors, shape (n_stations,)."""
        return self._look_u

    @property
    def n_obs(self) -> int:
        """Length of observation vector (one per pixel)."""
        return self.n_stations

    @property
    def obs(self) -> np.ndarray:
        """LOS displacement observations, shape (n_stations,)."""
        return self._los

    @property
    def sigma(self) -> np.ndarray:
        """1-sigma uncertainties, shape (n_stations,)."""
        return self._sigma

    def project(self, *components: np.ndarray) -> np.ndarray:
        """Project displacement components onto the line-of-sight direction.

        Args:
            *components: (ue, un, uz) displacement arrays, each (n_stations,).

        Returns:
            LOS-projected displacements, shape (n_stations,).
        """
        ue, un, uz = components
        return self._look_e * ue + self._look_n * un + self._look_u * uz

    def save(self, fname: str | Path, *, format: str = "dat") -> None:
        """Save InSAR data to a whitespace-delimited file.

        Writes the same column layout expected by ``load()``.

        Args:
            fname: Output file path.
            format: File format.  Only ``"dat"`` is supported.

        Raises:
            ValueError: If ``format`` is not ``"dat"``.
        """
        if format != "dat":
            raise ValueError(f"Unknown format {format!r}; use 'dat'")
        data = np.column_stack(
            [
                self._lon,
                self._lat,
                self._los,
                self._sigma,
                self._look_e,
                self._look_n,
                self._look_u,
            ]
        )
        header = _metadata_header(self) + "lon lat uLOS sigLOS losE losN losU"
        np.savetxt(Path(fname), data, header=header, fmt="%.8f")

    def to_gmt(self, fname: str | Path) -> None:
        """Save InSAR data in GMT-compatible format.

        Writes ``lon lat uLOS`` with a ``#``-prefixed header, suitable for
        ``xyz2grd`` or ``surface`` GMT commands.

        Args:
            fname: Output file path.
        """
        data = np.column_stack([self._lon, self._lat, self._los])
        np.savetxt(Path(fname), data, header="lon lat uLOS", fmt="%.8f")

    @classmethod
    def load(
        cls,
        fname: str | Path,
    ) -> "InSAR":
        """Load InSAR data from a whitespace-delimited .dat file.

        Expected columns: ``lon lat uLOS sigLOS losE losN losU``

        Lines starting with ``#`` are treated as comments.

        Args:
            fname: Path to the data file.

        Returns:
            A new ``InSAR`` instance.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        path = Path(fname)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        raw = np.loadtxt(path, comments="#", ndmin=2)
        lon, lat = raw[:, 0], raw[:, 1]
        los, sigma = raw[:, 2], raw[:, 3]
        look_e, look_n, look_u = raw[:, 4], raw[:, 5], raw[:, 6]
        dataset_name, quantity, units, epoch, time_span = _read_metadata(path)

        return cls(
            lon=lon,
            lat=lat,
            los=los,
            sigma=sigma,
            look_e=look_e,
            look_n=look_n,
            look_u=look_u,
            name=_read_names(path),
            dataset_name=dataset_name,
            quantity=quantity,
            units=units,
            epoch=epoch,
            time_span=time_span,
        )


class Vertical(DataSet):
    """Single-component vertical displacement observations.

    Suitable for coral uplift rates, tide gauge data, or any dataset
    that measures only the vertical component.

    Args:
        lon: Observation longitudes, shape (n_stations,).
        lat: Observation latitudes, shape (n_stations,).
        displacement: Vertical displacement values, shape (n_stations,).
        sigma: 1-sigma uncertainties, shape (n_stations,).
        name: Optional per-station site names, shape (n_stations,).
        covariance: Optional full covariance matrix, shape (n_obs, n_obs).
    """

    greens_type = "displacement"

    def __init__(
        self,
        *,
        lon: np.ndarray,
        lat: np.ndarray,
        displacement: np.ndarray,
        sigma: np.ndarray,
        name: np.ndarray | None = None,
        dataset_name: str | None = None,
        quantity: str = "displacement",
        units: str = "m",
        epoch: str | None = None,
        time_span: tuple[str, str] | None = None,
        covariance: np.ndarray | None = None,
        validate_covariance: bool = True,
    ) -> None:
        super().__init__(
            lon=lon,
            lat=lat,
            name=name,
            dataset_name=dataset_name,
            quantity=quantity,
            units=units,
            epoch=epoch,
            time_span=time_span,
            covariance=covariance,
            validate_covariance=validate_covariance,
        )

        n = self.n_stations
        displacement = as_1d_floats("displacement", displacement, n=n, unit="meters")
        sigma = as_1d_floats("sigma", sigma, n=n, unit="meters")
        check_positive("sigma", sigma, unit="meters")

        self._displacement = _make_readonly(displacement)
        self._sigma = _make_readonly(sigma)

        self._validate_covariance_shape()

    @property
    def n_obs(self) -> int:
        """Length of observation vector (one per point)."""
        return self.n_stations

    @property
    def obs(self) -> np.ndarray:
        """Vertical displacement observations, shape (n_stations,)."""
        return self._displacement

    @property
    def sigma(self) -> np.ndarray:
        """1-sigma uncertainties, shape (n_stations,)."""
        return self._sigma

    def project(self, *components: np.ndarray) -> np.ndarray:
        """Extract the vertical component.

        Args:
            *components: (ue, un, uz) displacement arrays, each (n_stations,).

        Returns:
            Vertical displacements, shape (n_stations,).
        """
        _ue, _un, uz = components
        return uz

    def save(self, fname: str | Path, *, format: str = "dat") -> None:
        """Save vertical data to a whitespace-delimited file.

        Writes the same column layout expected by ``load()``.

        Args:
            fname: Output file path.
            format: File format.  Only ``"dat"`` is supported.

        Raises:
            ValueError: If ``format`` is not ``"dat"``.
        """
        if format != "dat":
            raise ValueError(f"Unknown format {format!r}; use 'dat'")
        data = np.column_stack(
            [
                self._lon,
                self._lat,
                self._displacement,
                self._sigma,
            ]
        )
        header = _metadata_header(self) + "lon lat uZ sigZ"
        np.savetxt(Path(fname), data, header=header, fmt="%.8f")

    def to_gmt(self, fname: str | Path) -> None:
        """Save vertical data in GMT-compatible format.

        Writes ``lon lat uZ`` with a ``#``-prefixed header.

        Args:
            fname: Output file path.
        """
        data = np.column_stack([self._lon, self._lat, self._displacement])
        np.savetxt(Path(fname), data, header="lon lat uZ", fmt="%.8f")

    @classmethod
    def load(
        cls,
        fname: str | Path,
    ) -> "Vertical":
        """Load vertical data from a whitespace-delimited .dat file.

        Expected columns: ``lon lat uZ sigZ``

        Lines starting with ``#`` are treated as comments.

        Args:
            fname: Path to the data file.

        Returns:
            A new ``Vertical`` instance.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        path = Path(fname)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        raw = np.loadtxt(path, comments="#", ndmin=2)
        lon, lat = raw[:, 0], raw[:, 1]
        displacement, sigma = raw[:, 2], raw[:, 3]
        dataset_name, quantity, units, epoch, time_span = _read_metadata(path)

        return cls(
            lon=lon,
            lat=lat,
            displacement=displacement,
            sigma=sigma,
            name=_read_names(path),
            dataset_name=dataset_name,
            quantity=quantity,
            units=units,
            epoch=epoch,
            time_span=time_span,
        )


def gnss(
    *,
    lon: npt.ArrayLike,
    lat: npt.ArrayLike,
    east: npt.ArrayLike,
    north: npt.ArrayLike,
    up: npt.ArrayLike,
    sigma_east: npt.ArrayLike,
    sigma_north: npt.ArrayLike,
    sigma_up: npt.ArrayLike,
    name: str | None = None,
    station_names: Sequence[str] | None = None,
    quantity: str = "displacement",
    units: str = "m",
    epoch: str | None = None,
    time_span: tuple[str, str] | None = None,
    covariance: np.ndarray | None = None,
) -> GNSS:
    """Build a validated three-component GNSS dataset.

    Scalar uncertainties are broadcast over stations. Component names are
    geographic East, North, and Up; values and uncertainties use ``units``.

    Args:
        lon: Station longitudes in degrees.
        lat: Station latitudes in degrees.
        east: East component values.
        north: North component values.
        up: Up component values.
        sigma_east: East-component standard deviations.
        sigma_north: North-component standard deviations.
        sigma_up: Up-component standard deviations.
        name: Stable dataset identifier.
        station_names: Optional station labels.
        quantity: ``'displacement'`` or ``'velocity'``.
        units: Units shared by values and uncertainties.
        epoch: Optional representative epoch.
        time_span: Optional start and end epoch labels.
        covariance: Optional full observation covariance.

    Returns:
        An existing :class:`GNSS` dataset object.
    """
    lon_array = np.asarray(lon, dtype=float)
    n = lon_array.size
    return GNSS(
        lon=lon_array,
        lat=np.asarray(lat, dtype=float),
        ve=_broadcast_1d("east", east, n),
        vn=_broadcast_1d("north", north, n),
        vu=_broadcast_1d("up", up, n),
        se=_broadcast_1d("sigma_east", sigma_east, n),
        sn=_broadcast_1d("sigma_north", sigma_north, n),
        su=_broadcast_1d("sigma_up", sigma_up, n),
        name=None if station_names is None else np.asarray(station_names, dtype=str),
        dataset_name=name,
        quantity=quantity,
        units=units,
        epoch=epoch,
        time_span=time_span,
        covariance=covariance,
    )


def horizontal_gnss(
    *,
    lon: npt.ArrayLike,
    lat: npt.ArrayLike,
    east: npt.ArrayLike,
    north: npt.ArrayLike,
    sigma_east: npt.ArrayLike,
    sigma_north: npt.ArrayLike,
    name: str | None = None,
    station_names: Sequence[str] | None = None,
    quantity: str = "displacement",
    units: str = "m",
    epoch: str | None = None,
    time_span: tuple[str, str] | None = None,
    covariance: np.ndarray | None = None,
) -> GNSS:
    """Build a validated horizontal GNSS dataset.

    Args:
        lon: Station longitudes in degrees.
        lat: Station latitudes in degrees.
        east: East component values.
        north: North component values.
        sigma_east: East-component standard deviations.
        sigma_north: North-component standard deviations.
        name: Stable dataset identifier.
        station_names: Optional station labels.
        quantity: ``'displacement'`` or ``'velocity'``.
        units: Units shared by values and uncertainties.
        epoch: Optional representative epoch.
        time_span: Optional start and end epoch labels.
        covariance: Optional full observation covariance.

    Returns:
        An existing horizontal :class:`GNSS` dataset object.
    """
    lon_array = np.asarray(lon, dtype=float)
    n = lon_array.size
    return GNSS(
        lon=lon_array,
        lat=np.asarray(lat, dtype=float),
        ve=_broadcast_1d("east", east, n),
        vn=_broadcast_1d("north", north, n),
        se=_broadcast_1d("sigma_east", sigma_east, n),
        sn=_broadcast_1d("sigma_north", sigma_north, n),
        name=None if station_names is None else np.asarray(station_names, dtype=str),
        dataset_name=name,
        quantity=quantity,
        units=units,
        epoch=epoch,
        time_span=time_span,
        covariance=covariance,
    )


def insar(
    *,
    lon: npt.ArrayLike,
    lat: npt.ArrayLike,
    los: npt.ArrayLike,
    sigma: npt.ArrayLike,
    look_e: npt.ArrayLike,
    look_n: npt.ArrayLike,
    look_u: npt.ArrayLike,
    name: str | None = None,
    quantity: str = "displacement",
    units: str = "m",
    epoch: str | None = None,
    time_span: tuple[str, str] | None = None,
    covariance: np.ndarray | None = None,
    normalize_look: bool = False,
) -> InSAR:
    """Build a validated InSAR line-of-sight dataset.

    Args:
        lon: Pixel longitudes in degrees.
        lat: Pixel latitudes in degrees.
        los: Line-of-sight values.
        sigma: Line-of-sight standard deviations.
        look_e: East look-vector components.
        look_n: North look-vector components.
        look_u: Up look-vector components.
        name: Stable dataset identifier.
        quantity: ``'displacement'`` or ``'velocity'``.
        units: Units shared by values and uncertainties.
        epoch: Optional representative epoch.
        time_span: Optional start and end epoch labels.
        covariance: Optional full observation covariance.
        normalize_look: Normalize supplied look vectors when true.

    Returns:
        An existing :class:`InSAR` dataset object.
    """
    lon_array = np.asarray(lon, dtype=float)
    n = lon_array.size
    return InSAR(
        lon=lon_array,
        lat=np.asarray(lat, dtype=float),
        los=_broadcast_1d("los", los, n),
        sigma=_broadcast_1d("sigma", sigma, n),
        look_e=_broadcast_1d("look_e", look_e, n),
        look_n=_broadcast_1d("look_n", look_n, n),
        look_u=_broadcast_1d("look_u", look_u, n),
        dataset_name=name,
        quantity=quantity,
        units=units,
        epoch=epoch,
        time_span=time_span,
        covariance=covariance,
        normalize_look=normalize_look,
    )


def vertical(
    *,
    lon: npt.ArrayLike,
    lat: npt.ArrayLike,
    displacement: npt.ArrayLike,
    sigma: npt.ArrayLike,
    name: str | None = None,
    station_names: Sequence[str] | None = None,
    quantity: str = "displacement",
    units: str = "m",
    epoch: str | None = None,
    time_span: tuple[str, str] | None = None,
    covariance: np.ndarray | None = None,
) -> Vertical:
    """Build a validated vertical-observation dataset.

    Args:
        lon: Observation longitudes in degrees.
        lat: Observation latitudes in degrees.
        displacement: Vertical displacement or velocity values.
        sigma: Standard deviations.
        name: Stable dataset identifier.
        station_names: Optional point labels.
        quantity: ``'displacement'`` or ``'velocity'``.
        units: Units shared by values and uncertainties.
        epoch: Optional representative epoch.
        time_span: Optional start and end epoch labels.
        covariance: Optional full observation covariance.

    Returns:
        An existing :class:`Vertical` dataset object.
    """
    lon_array = np.asarray(lon, dtype=float)
    n = lon_array.size
    return Vertical(
        lon=lon_array,
        lat=np.asarray(lat, dtype=float),
        displacement=_broadcast_1d("displacement", displacement, n),
        sigma=_broadcast_1d("sigma", sigma, n),
        name=None if station_names is None else np.asarray(station_names, dtype=str),
        dataset_name=name,
        quantity=quantity,
        units=units,
        epoch=epoch,
        time_span=time_span,
        covariance=covariance,
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
