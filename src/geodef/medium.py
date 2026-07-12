"""Elastic medium parameters for the half-space Green's functions.

Every dislocation engine in GeoDef assumes a homogeneous isotropic elastic
half-space. :class:`ElasticMedium` is the single declared home for that
medium's parameters: construct one and pass it to ``Fault`` (or rely on
:data:`DEFAULT_MEDIUM`, a 30 GPa Poisson solid), and the same values flow
through Green's matrices, stress kernels, and moment calculations.

Usage::

    import geodef

    medium = geodef.ElasticMedium(shear_modulus=35e9, poisson_ratio=0.27)
    fault = geodef.Fault.planar(..., medium=medium)
"""

from __future__ import annotations

import dataclasses
import math

__all__ = ["DEFAULT_MEDIUM", "ElasticMedium"]


@dataclasses.dataclass(frozen=True)
class ElasticMedium:
    """Homogeneous isotropic elastic half-space parameters.

    Attributes:
        shear_modulus: Shear modulus (rigidity) mu in Pa. Must be positive
            and finite. Default 30 GPa, a common crustal value.
        poisson_ratio: Poisson's ratio nu, dimensionless, in ``[0, 0.5)``
            (crustal rocks are typically 0.1-0.35). Default 0.25 (a Poisson
            solid, where the Lame parameters are equal).
    """

    shear_modulus: float = 30e9
    poisson_ratio: float = 0.25

    def __post_init__(self) -> None:
        if not (math.isfinite(self.shear_modulus) and self.shear_modulus > 0):
            raise ValueError(
                f"shear_modulus must be positive and finite (Pa), "
                f"got {self.shear_modulus!r}"
            )
        if not (math.isfinite(self.poisson_ratio) and 0.0 <= self.poisson_ratio < 0.5):
            raise ValueError(
                f"poisson_ratio must lie in [0, 0.5), got {self.poisson_ratio!r}"
            )

    @property
    def mu(self) -> float:
        """Shear modulus in Pa (alias for ``shear_modulus``)."""
        return self.shear_modulus

    @property
    def nu(self) -> float:
        """Poisson's ratio (alias for ``poisson_ratio``)."""
        return self.poisson_ratio

    @property
    def lame_lambda(self) -> float:
        """First Lame parameter ``lambda = 2 mu nu / (1 - 2 nu)`` in Pa."""
        return (
            2.0
            * self.shear_modulus
            * self.poisson_ratio
            / (1.0 - 2.0 * self.poisson_ratio)
        )

    @property
    def young_modulus(self) -> float:
        """Young's modulus ``E = 2 mu (1 + nu)`` in Pa."""
        return 2.0 * self.shear_modulus * (1.0 + self.poisson_ratio)


DEFAULT_MEDIUM = ElasticMedium()
"""The default medium: a 30 GPa Poisson solid (nu = 0.25)."""
