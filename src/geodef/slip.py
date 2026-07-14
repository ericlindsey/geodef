"""Named slip and displacement value objects.

The low-level numerical convention remains a blocked NumPy vector.  These
objects add names and basis metadata so that a one-component amplitude is not
mistaken for strike slip and a plate-motion basis is not mistaken for each
triangle's local strike/dip basis.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from typing import TYPE_CHECKING, Literal

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from geodef.fault import Fault

SlipBasisName = Literal["strike_dip", "strike", "dip", "rake", "azimuth", "plate"]


def _readonly_1d(values: npt.ArrayLike, name: str) -> np.ndarray:
    """Return a copied, read-only one-dimensional float array."""
    array = np.atleast_1d(np.asarray(values, dtype=float)).copy()
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    array.flags.writeable = False
    return array


def _broadcast_angle(values: npt.ArrayLike, size: int, name: str) -> np.ndarray:
    """Broadcast a scalar or validate a per-patch angle array."""
    angle = np.asarray(values, dtype=float)
    try:
        broadcast = np.broadcast_to(angle, (size,)).copy()
    except ValueError as exc:
        raise ValueError(f"{name} must be scalar or have shape ({size},)") from exc
    if not np.all(np.isfinite(broadcast)):
        raise ValueError(f"{name} must contain only finite values")
    broadcast.flags.writeable = False
    return broadcast


@dataclasses.dataclass(frozen=True, init=False)
class SlipModel:
    """Immutable per-patch slip with an explicit linear-algebra basis.

    Construct ordinary two-component slip with ``SlipModel(strike, dip)``.
    The class methods construct one-component fixed-direction models or the
    two-component plate-motion coordinates used for smoothly varying meshes.

    ``vector`` always represents the coordinates actually stored or solved:
    one amplitude per patch for fixed rake/azimuth, or two blocked components
    for strike/dip and plate-parallel/plate-perpendicular models.  ``strike``
    and ``dip`` are the derived physical components in each patch's local
    frame.
    """

    _vector: np.ndarray
    basis: SlipBasisName
    _basis_rake: np.ndarray | None
    slip_azimuth: float | None

    def __init__(self, strike: npt.ArrayLike, dip: npt.ArrayLike) -> None:
        """Create a model from local strike-slip and dip-slip components."""
        strike_array = _readonly_1d(strike, "strike")
        dip_array = _readonly_1d(dip, "dip")
        if strike_array.shape != dip_array.shape:
            raise ValueError(
                "strike and dip must have the same shape, got "
                f"{strike_array.shape} and {dip_array.shape}"
            )
        vector = np.concatenate([strike_array, dip_array])
        vector.flags.writeable = False
        self._initialize(vector, "strike_dip")

    def _initialize(
        self,
        vector: np.ndarray,
        basis: SlipBasisName,
        *,
        basis_rake: np.ndarray | None = None,
        slip_azimuth: float | None = None,
    ) -> None:
        """Initialize frozen storage for public constructors."""
        object.__setattr__(self, "_vector", vector)
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "_basis_rake", basis_rake)
        object.__setattr__(self, "slip_azimuth", slip_azimuth)

    @classmethod
    def _from_storage(
        cls,
        vector: npt.ArrayLike,
        basis: SlipBasisName,
        *,
        basis_rake: npt.ArrayLike | None = None,
        slip_azimuth: float | None = None,
    ) -> SlipModel:
        """Build a model after a public constructor defines its semantics."""
        model = cls.__new__(cls)
        vector_array = _readonly_1d(vector, "vector")
        if basis in {"strike_dip", "plate"}:
            if vector_array.size == 0 or vector_array.size % 2:
                raise ValueError(f"basis={basis!r} requires a non-empty 2*N vector")
            n_patches = vector_array.size // 2
        else:
            if vector_array.size == 0:
                raise ValueError("vector must contain at least one patch")
            n_patches = vector_array.size

        rake_array = (
            None
            if basis_rake is None
            else _broadcast_angle(basis_rake, n_patches, "rake")
        )
        model._initialize(
            vector_array,
            basis,
            basis_rake=rake_array,
            slip_azimuth=slip_azimuth,
        )
        return model

    @classmethod
    def from_strike(cls, amplitude: npt.ArrayLike) -> SlipModel:
        """Create a one-component strike-slip model."""
        return cls._from_storage(amplitude, "strike")

    @classmethod
    def from_dip(cls, amplitude: npt.ArrayLike) -> SlipModel:
        """Create a one-component dip-slip model."""
        return cls._from_storage(amplitude, "dip")

    @classmethod
    def from_rake(cls, amplitude: npt.ArrayLike, rake: float) -> SlipModel:
        """Create one amplitude per patch along a constant local rake."""
        return cls._from_storage(amplitude, "rake", basis_rake=rake)

    @classmethod
    def from_azimuth(
        cls,
        amplitude: npt.ArrayLike,
        *,
        azimuth: float,
        fault_strike: npt.ArrayLike,
    ) -> SlipModel:
        """Create amplitudes along a fixed geographic slip azimuth.

        Args:
            amplitude: One signed amplitude per patch.
            azimuth: Geographic direction in degrees clockwise from North.
            fault_strike: Local strike of every patch in degrees.
        """
        amplitude_array = _readonly_1d(amplitude, "amplitude")
        strike = _broadcast_angle(fault_strike, amplitude_array.size, "fault_strike")
        return cls._from_storage(
            amplitude_array,
            "azimuth",
            basis_rake=azimuth - strike,
            slip_azimuth=float(azimuth),
        )

    @classmethod
    def from_plate_rake(
        cls,
        parallel: npt.ArrayLike,
        perpendicular: npt.ArrayLike,
        *,
        plate_rake: npt.ArrayLike,
    ) -> SlipModel:
        """Create two-component plate-rake coordinates.

        ``parallel`` follows the large-scale plate-motion rake and
        ``perpendicular`` is 90 degrees counter-clockwise from it in the
        local strike/dip component plane.  Both are blocked in ``vector``.
        """
        parallel_array = _readonly_1d(parallel, "parallel")
        perpendicular_array = _readonly_1d(perpendicular, "perpendicular")
        if parallel_array.shape != perpendicular_array.shape:
            raise ValueError(
                "parallel and perpendicular must have the same shape, got "
                f"{parallel_array.shape} and {perpendicular_array.shape}"
            )
        vector = np.concatenate([parallel_array, perpendicular_array])
        return cls._from_storage(vector, "plate", basis_rake=plate_rake)

    @classmethod
    def from_euler_pole(
        cls,
        parallel: npt.ArrayLike,
        perpendicular: npt.ArrayLike,
        *,
        fault: Fault,
        pole: tuple[float, float, float],
    ) -> SlipModel:
        """Create plate coordinates whose direction follows an Euler pole.

        The pole supplies a smooth geographic horizontal velocity direction.
        Its local rake on each patch is ``azimuth - patch_strike``; the rate
        controls direction/sign only, while ``parallel`` and ``perpendicular``
        retain the model's requested units.

        Args:
            parallel: Plate-rake-parallel component per patch.
            perpendicular: Plate-rake-perpendicular component per patch.
            fault: Fault whose patch centers and local strikes define the basis.
            pole: ``(latitude, longitude, rate)`` in degrees, degrees, deg/Myr.
        """
        from geodef.euler import pole_velocity

        centers = fault.centers_geo
        east, north = pole_velocity(
            centers[:, 1], centers[:, 0], pole[0], pole[1], pole[2]
        )
        speed = np.hypot(east, north)
        if np.any(speed == 0.0):
            raise ValueError("Euler pole produces zero velocity at a patch center")
        azimuth = np.degrees(np.arctan2(east, north))
        return cls.from_plate_rake(
            parallel,
            perpendicular,
            plate_rake=azimuth - fault.strike,
        )

    @property
    def n_patches(self) -> int:
        """Number of fault patches represented."""
        return self._vector.size // self.n_components

    @property
    def n_components(self) -> int:
        """Number of components in the stored linear-algebra basis."""
        return 2 if self.basis in {"strike_dip", "plate"} else 1

    @property
    def vector(self) -> np.ndarray:
        """Read-only blocked vector in this model's declared basis."""
        return self._vector

    @property
    def amplitude(self) -> np.ndarray:
        """Signed amplitude for a one-component model.

        Raises:
            AttributeError: If the model has two stored components.
        """
        if self.n_components != 1:
            raise AttributeError("amplitude is only defined for one-component slip")
        return self._vector

    @property
    def strike(self) -> np.ndarray:
        """Physical strike-slip component in every patch's local frame."""
        n = self.n_patches
        if self.basis == "strike_dip":
            return self._vector[:n]
        if self.basis == "strike":
            return self._vector
        if self.basis == "dip":
            return np.zeros(n)
        assert self._basis_rake is not None
        theta = np.deg2rad(self._basis_rake)
        if self.basis in {"rake", "azimuth"}:
            return self._vector * np.cos(theta)
        return self._vector[:n] * np.cos(theta) - self._vector[n:] * np.sin(theta)

    @property
    def dip(self) -> np.ndarray:
        """Physical dip-slip component in every patch's local frame."""
        n = self.n_patches
        if self.basis == "strike_dip":
            return self._vector[n:]
        if self.basis == "strike":
            return np.zeros(n)
        if self.basis == "dip":
            return self._vector
        assert self._basis_rake is not None
        theta = np.deg2rad(self._basis_rake)
        if self.basis in {"rake", "azimuth"}:
            return self._vector * np.sin(theta)
        return self._vector[:n] * np.sin(theta) + self._vector[n:] * np.cos(theta)

    @property
    def magnitude(self) -> np.ndarray:
        """Unsigned physical slip magnitude per patch."""
        return np.hypot(self.strike, self.dip)

    @property
    def rake(self) -> np.ndarray:
        """Physical local rake in degrees, computed from strike/dip slip."""
        return np.degrees(np.arctan2(self.dip, self.strike))

    @property
    def plate_rake(self) -> np.ndarray | None:
        """Large-scale plate direction in local rake coordinates, if present."""
        return self._basis_rake if self.basis == "plate" else None

    @property
    def rake_parallel(self) -> np.ndarray:
        """Plate-rake-parallel component.

        Raises:
            AttributeError: If this is not a plate-coordinate model.
        """
        if self.basis != "plate":
            raise AttributeError("rake_parallel requires basis='plate'")
        return self._vector[: self.n_patches]

    @property
    def rake_perpendicular(self) -> np.ndarray:
        """Plate-rake-perpendicular component.

        Raises:
            AttributeError: If this is not a plate-coordinate model.
        """
        if self.basis != "plate":
            raise AttributeError("rake_perpendicular requires basis='plate'")
        return self._vector[self.n_patches :]


@dataclasses.dataclass(frozen=True)
class Displacement:
    """Named East, North, Up displacement arrays with tuple unpacking."""

    east: np.ndarray
    north: np.ndarray
    up: np.ndarray

    def __post_init__(self) -> None:
        """Copy and validate component arrays."""
        east = _readonly_1d(self.east, "east")
        north = _readonly_1d(self.north, "north")
        up = _readonly_1d(self.up, "up")
        if east.shape != north.shape or east.shape != up.shape:
            raise ValueError(
                "east, north, and up must have the same shape, got "
                f"{east.shape}, {north.shape}, and {up.shape}"
            )
        object.__setattr__(self, "east", east)
        object.__setattr__(self, "north", north)
        object.__setattr__(self, "up", up)

    def __iter__(self) -> Iterator[np.ndarray]:
        """Yield East, North, Up arrays for tuple unpacking."""
        yield self.east
        yield self.north
        yield self.up

    @property
    def vector(self) -> np.ndarray:
        """Read-only observation-interleaved ``[E, N, U, ...]`` vector."""
        vector = np.column_stack([self.east, self.north, self.up]).ravel()
        vector.flags.writeable = False
        return vector
