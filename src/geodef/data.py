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
from pathlib import Path

import numpy as np


def _make_readonly(arr: np.ndarray) -> np.ndarray:
    """Set an array's writeable flag to False and return it."""
    arr.flags.writeable = False
    return arr


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
        lat: np.ndarray,
        lon: np.ndarray,
        *,
        covariance: np.ndarray | None = None,
    ) -> None:
        lat = _make_readonly(np.asarray(lat, dtype=float))
        lon = _make_readonly(np.asarray(lon, dtype=float))

        n = lat.shape[0]
        if lat.ndim != 1 or lon.shape != (n,):
            raise ValueError("lat and lon must be 1-D arrays of the same length")

        self._lat = lat
        self._lon = lon
        self._covariance_explicit = covariance
        self._covariance_cache: np.ndarray | None = None

    @property
    def lat(self) -> np.ndarray:
        """Observation point latitudes, shape (n_stations,)."""
        return self._lat

    @property
    def lon(self) -> np.ndarray:
        """Observation point longitudes, shape (n_stations,)."""
        return self._lon

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
            cov = np.diag(self.sigma ** 2)

        self._covariance_cache = _make_readonly(cov)
        return self._covariance_cache

    def _validate_covariance_shape(self) -> None:
        """Check explicit covariance shape against n_obs. Call after n_obs is known."""
        if self._covariance_explicit is not None:
            n = self.n_obs
            cov = np.asarray(self._covariance_explicit, dtype=float)
            if cov.shape != (n, n):
                raise ValueError(
                    f"covariance shape {cov.shape} does not match "
                    f"expected ({n}, {n})"
                )


class GNSS(DataSet):
    """Three-component displacement or velocity observations at GNSS stations.

    Supports full 3-component (E, N, U) or horizontal-only (E, N) data.
    Pass ``vu=None, su=None`` for horizontal-only.

    Args:
        lat: Station latitudes, shape (n_stations,).
        lon: Station longitudes, shape (n_stations,).
        ve: East component values, shape (n_stations,).
        vn: North component values, shape (n_stations,).
        vu: Up component values, shape (n_stations,), or None for horizontal-only.
        se: East component 1-sigma, shape (n_stations,).
        sn: North component 1-sigma, shape (n_stations,).
        su: Up component 1-sigma, shape (n_stations,), or None for horizontal-only.
        covariance: Optional full covariance matrix, shape (n_obs, n_obs).
    """

    greens_type = "displacement"

    def __init__(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        ve: np.ndarray,
        vn: np.ndarray,
        vu: np.ndarray | None,
        se: np.ndarray,
        sn: np.ndarray,
        su: np.ndarray | None,
        *,
        covariance: np.ndarray | None = None,
    ) -> None:
        super().__init__(lat, lon, covariance=covariance)

        ve = np.asarray(ve, dtype=float)
        vn = np.asarray(vn, dtype=float)
        se = np.asarray(se, dtype=float)
        sn = np.asarray(sn, dtype=float)

        n = self.n_stations
        if ve.shape != (n,) or vn.shape != (n,):
            raise ValueError(
                "ve, vn must be 1-D arrays of the same length as lat/lon"
            )
        if se.shape != (n,) or sn.shape != (n,):
            raise ValueError(
                "se, sn must be 1-D arrays of the same length as lat/lon"
            )

        # Validate vertical component consistency
        if (vu is None) != (su is None):
            raise ValueError("vu and su must both be None or both provided")

        if vu is not None:
            vu = np.asarray(vu, dtype=float)
            su = np.asarray(su, dtype=float)
            if vu.shape != (n,) or su.shape != (n,):
                raise ValueError(
                    "vu, su must be 1-D arrays of the same length as lat/lon"
                )

        # Validate all sigmas are positive
        if np.any(se <= 0) or np.any(sn <= 0):
            raise ValueError("All sigma values must be positive")
        if su is not None and np.any(su <= 0):
            raise ValueError("All sigma values must be positive")

        self._ve = _make_readonly(ve)
        self._vn = _make_readonly(vn)
        self._vu = _make_readonly(vu) if vu is not None else None
        self._se = _make_readonly(se)
        self._sn = _make_readonly(sn)
        self._su = _make_readonly(su) if su is not None else None

        self._validate_covariance_shape()

    @property
    def components(self) -> str:
        """Active component string: ``'enu'`` or ``'en'``."""
        return "enu" if self._vu is not None else "en"

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
        return _make_readonly(
            np.column_stack([self._ve, self._vn]).ravel()
        )

    @property
    def sigma(self) -> np.ndarray:
        """1-sigma uncertainties, interleaved to match ``obs``."""
        if self._su is not None:
            return _make_readonly(
                np.column_stack([self._se, self._sn, self._su]).ravel()
            )
        return _make_readonly(
            np.column_stack([self._se, self._sn]).ravel()
        )

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

        Always writes 9 columns (``lon lat hgt uE uN uZ sigE sigN sigZ``) so
        the file can always be read back with ``load()``.  For horizontal-only
        instances, ``uZ`` and ``sigZ`` are written as ``0.0`` / ``1.0``
        placeholders; reload with ``components='en'`` to discard them.
        Height (``hgt``) is written as zeros because the class does not store
        elevation.

        Args:
            fname: Output file path.
            format: File format.  Only ``"dat"`` is supported.

        Raises:
            ValueError: If ``format`` is not ``"dat"``.
        """
        if format != "dat":
            raise ValueError(f"Unknown format {format!r}; use 'dat'")
        hgt = np.zeros(self.n_stations)
        vu = self._vu if self._vu is not None else np.zeros(self.n_stations)
        su = self._su if self._su is not None else np.ones(self.n_stations)
        data = np.column_stack([
            self._lon, self._lat, hgt,
            self._ve, self._vn, vu,
            self._se, self._sn, su,
        ])
        np.savetxt(Path(fname), data,
                   header="lon lat hgt uE uN uZ sigE sigN sigZ",
                   fmt="%.8f")

    def to_gmt(self, fname: str | Path) -> None:
        """Save GNSS data in GMT-compatible format.

        Writes ``lon lat uE uN sigE sigN`` (horizontal) with a ``#``-prefixed
        header, suitable for ``psvelo`` or ``psxy`` GMT commands.

        Args:
            fname: Output file path.
        """
        header = "lon lat uE uN sigE sigN"
        data = np.column_stack([
            self._lon, self._lat,
            self._ve, self._vn,
            self._se, self._sn,
        ])
        np.savetxt(Path(fname), data, header=header, fmt="%.8f")

    @classmethod
    def load(
        cls,
        fname: str | Path,
        *,
        components: str = "enu",
    ) -> "GNSS":
        """Load GNSS data from a whitespace-delimited .dat file.

        Expected columns: ``lon lat hgt uE uN uZ sigE sigN sigZ``

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
        ve, vn, vu = raw[:, 3], raw[:, 4], raw[:, 5]
        se, sn, su = raw[:, 6], raw[:, 7], raw[:, 8]

        if components == "en":
            vu, su = None, None

        return cls(lat, lon, ve, vn, vu, se, sn, su)


class InSAR(DataSet):
    """Line-of-sight displacement observations (e.g. from SAR interferometry).

    Each pixel has a scalar LOS measurement and a 3-component unit look vector
    defining the satellite-to-ground direction.

    Args:
        lat: Pixel latitudes, shape (n_stations,).
        lon: Pixel longitudes, shape (n_stations,).
        los: LOS displacement values, shape (n_stations,).
        sigma: 1-sigma uncertainties, shape (n_stations,).
        look_e: East component of look vector, shape (n_stations,).
        look_n: North component of look vector, shape (n_stations,).
        look_u: Up component of look vector, shape (n_stations,).
        covariance: Optional full covariance matrix, shape (n_obs, n_obs).
    """

    greens_type = "displacement"

    def __init__(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        los: np.ndarray,
        sigma: np.ndarray,
        look_e: np.ndarray,
        look_n: np.ndarray,
        look_u: np.ndarray,
        *,
        covariance: np.ndarray | None = None,
    ) -> None:
        super().__init__(lat, lon, covariance=covariance)

        los = np.asarray(los, dtype=float)
        sigma = np.asarray(sigma, dtype=float)
        look_e = np.asarray(look_e, dtype=float)
        look_n = np.asarray(look_n, dtype=float)
        look_u = np.asarray(look_u, dtype=float)

        n = self.n_stations
        if not all(arr.shape == (n,) for arr in (los, sigma, look_e, look_n, look_u)):
            raise ValueError(
                "los, sigma, look_e, look_n, look_u must be 1-D arrays of the "
                "same length as lat/lon"
            )

        if np.any(sigma <= 0):
            raise ValueError("All sigma values must be positive")

        self._los = _make_readonly(los)
        self._sigma = _make_readonly(sigma)
        self._look_e = _make_readonly(look_e)
        self._look_n = _make_readonly(look_n)
        self._look_u = _make_readonly(look_u)

        self._validate_covariance_shape()

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

        Writes the same column layout expected by ``load()``.  Height is
        written as zeros.

        Args:
            fname: Output file path.
            format: File format.  Only ``"dat"`` is supported.

        Raises:
            ValueError: If ``format`` is not ``"dat"``.
        """
        if format != "dat":
            raise ValueError(f"Unknown format {format!r}; use 'dat'")
        hgt = np.zeros(self.n_stations)
        data = np.column_stack([
            self._lon, self._lat, hgt,
            self._los, self._sigma,
            self._look_e, self._look_n, self._look_u,
        ])
        np.savetxt(Path(fname), data,
                   header="lon lat hgt uLOS sigLOS losE losN losU",
                   fmt="%.8f")

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

        Expected columns: ``lon lat hgt uLOS sigLOS losE losN losU``

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
        los, sigma = raw[:, 3], raw[:, 4]
        look_e, look_n, look_u = raw[:, 5], raw[:, 6], raw[:, 7]

        return cls(lat, lon, los, sigma, look_e, look_n, look_u)


class Vertical(DataSet):
    """Single-component vertical displacement observations.

    Suitable for coral uplift rates, tide gauge data, or any dataset
    that measures only the vertical component.

    Args:
        lat: Observation latitudes, shape (n_stations,).
        lon: Observation longitudes, shape (n_stations,).
        displacement: Vertical displacement values, shape (n_stations,).
        sigma: 1-sigma uncertainties, shape (n_stations,).
        covariance: Optional full covariance matrix, shape (n_obs, n_obs).
    """

    greens_type = "displacement"

    def __init__(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        displacement: np.ndarray,
        sigma: np.ndarray,
        *,
        covariance: np.ndarray | None = None,
    ) -> None:
        super().__init__(lat, lon, covariance=covariance)

        displacement = np.asarray(displacement, dtype=float)
        sigma = np.asarray(sigma, dtype=float)

        n = self.n_stations
        if displacement.shape != (n,) or sigma.shape != (n,):
            raise ValueError(
                "displacement and sigma must be 1-D arrays of the same length "
                "as lat/lon"
            )

        if np.any(sigma <= 0):
            raise ValueError("All sigma values must be positive")

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

        Writes the same column layout expected by ``load()``.  Height is
        written as zeros.

        Args:
            fname: Output file path.
            format: File format.  Only ``"dat"`` is supported.

        Raises:
            ValueError: If ``format`` is not ``"dat"``.
        """
        if format != "dat":
            raise ValueError(f"Unknown format {format!r}; use 'dat'")
        hgt = np.zeros(self.n_stations)
        data = np.column_stack([
            self._lon, self._lat, hgt,
            self._displacement, self._sigma,
        ])
        np.savetxt(Path(fname), data,
                   header="lon lat hgt uZ sigZ",
                   fmt="%.8f")

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

        Expected columns: ``lon lat hgt uZ sigZ``

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
        displacement, sigma = raw[:, 3], raw[:, 4]

        return cls(lat, lon, displacement, sigma)
