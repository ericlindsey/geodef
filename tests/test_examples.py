"""Lightweight smoke tests for the real-data examples.

Full example notebooks (e.g. the Gorkha inversion) are heavy — the 2841-patch
triangular mesh makes every Green's-matrix assembly take tens of seconds — so
they are run manually, not in CI. These tests instead load the bundled fault
and datasets through the public API and check their basic structure, which is
cheap and still catches format or API drift.
"""

from pathlib import Path

import pytest

import geodef

ROOT = Path(__file__).resolve().parents[1]
GORKHA = ROOT / "examples" / "gorkha_earthquake"

pytestmark = pytest.mark.skipif(
    not (GORKHA / "fault" / "qiu+15_geo.ned").exists(),
    reason="Gorkha example data not present",
)


def test_gorkha_fault_loads():
    fault = geodef.fault.Fault.load(str(GORKHA / "fault" / "qiu+15_geo"), format="ned")
    assert fault.engine == "tri"
    assert fault.n_patches > 1000
    assert fault.areas.min() > 0


def test_gorkha_datasets_load():
    gnss = geodef.data.GNSS.load(str(GORKHA / "data" / "aria_offsets_for_geodef.dat"))
    insar = geodef.data.InSAR.load(str(GORKHA / "data" / "t048_insar_for_geodef.dat"))
    assert gnss.n_stations > 0
    assert gnss.n_obs == 3 * gnss.n_stations
    assert insar.n_obs == insar.n_stations
    # look vectors are unit-length to within rounding
    norm = insar._look_e**2 + insar._look_n**2 + insar._look_u**2
    assert abs(float(norm.mean()) - 1.0) < 0.05
