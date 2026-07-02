from dataclasses import dataclass
import numpy as np

@dataclass
class Ellipsoid:
    a: float
    f: float

    @property
    def finv(self) -> float:
        return 1.0 / self.f if self.f != 0 else 0.0

    @property
    def e2(self) -> float:
        f = self.f
        return 2.0 * f - f * f

WGS84 = Ellipsoid(a=6378137.0, f=1./298.257223563)
print(WGS84.a, WGS84.e2)
