"""Property-based numerical contracts (roadmap 3.4).

Round trips, linearity, and basis invariants checked over generated
inputs with hypothesis. The dependency is dev-only: these tests skip
cleanly when hypothesis is absent so the base CI tier stays honest.
"""

import numpy as np
import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402
from hypothesis.extra import numpy as hnp  # noqa: E402

from geodef import slip, transforms  # noqa: E402

# Bounded, finite, well-scaled floats: slip in meters.
finite_slip = hnp.arrays(
    dtype=np.float64,
    shape=st.integers(min_value=1, max_value=32),
    elements=st.floats(min_value=-100.0, max_value=100.0),
)

angles_deg = st.floats(min_value=-360.0, max_value=720.0)


@given(finite_slip, st.data())
def test_pack_unpack_round_trip(strike_slip, data):
    dip_slip = data.draw(
        hnp.arrays(
            dtype=np.float64,
            shape=strike_slip.shape,
            elements=st.floats(min_value=-100.0, max_value=100.0),
        )
    )
    vector = slip.pack(strike_slip, dip_slip)
    ss, ds = slip.unpack(vector)
    np.testing.assert_array_equal(ss, strike_slip)
    np.testing.assert_array_equal(ds, dip_slip)
    assert vector.shape == (2 * strike_slip.size,)


@given(finite_slip, angles_deg)
def test_from_rake_preserves_magnitude(amplitude, rake_deg):
    ss, ds = slip.from_rake(amplitude, rake_deg)
    np.testing.assert_allclose(
        slip.magnitude(ss, ds), np.abs(amplitude), rtol=0, atol=1e-9
    )


@given(finite_slip, angles_deg)
def test_plate_basis_round_trip(parallel, plate_rake):
    perpendicular = np.zeros_like(parallel)
    ss, ds = slip.from_plate(parallel, perpendicular, plate_rake)
    back_par, back_perp = slip.to_plate(ss, ds, plate_rake)
    np.testing.assert_allclose(back_par, parallel, atol=1e-9)
    np.testing.assert_allclose(back_perp, perpendicular, atol=1e-9)


@given(
    st.floats(min_value=-89.0, max_value=89.0),
    st.floats(min_value=-179.0, max_value=179.0),
    st.floats(min_value=-1000.0, max_value=5000.0),
)
@settings(max_examples=50)
def test_geodetic_ecef_round_trip(lat, lon, alt):
    x, y, z = transforms.geod2ecef(lat, lon, alt)
    lat2, lon2, alt2 = transforms.ecef2geod(x, y, z)
    np.testing.assert_allclose([lat2, lon2], [lat, lon], atol=1e-9)
    np.testing.assert_allclose(alt2, alt, atol=1e-6)


@given(
    st.floats(min_value=-60.0, max_value=60.0),
    st.floats(min_value=-179.0, max_value=179.0),
    st.floats(min_value=-50e3, max_value=50e3),
    st.floats(min_value=-50e3, max_value=50e3),
)
@settings(max_examples=50)
def test_enu_geodetic_round_trip(lat0, lon0, east, north):
    lat, lon, alt = transforms.enu2geod(east, north, 0.0, lat0, lon0, 0.0)
    e2, n2, u2 = transforms.geod2enu(lat, lon, alt, lat0, lon0, 0.0)
    np.testing.assert_allclose([e2, n2], [east, north], atol=1e-6)
    np.testing.assert_allclose(u2, 0.0, atol=1e-6)


@given(st.data())
@settings(max_examples=20, deadline=None)
def test_reshape_flatten_round_trip(data):
    import geodef

    fault = geodef.Fault.planar(
        lat=0.0,
        lon=0.0,
        depth=12e3,
        strike=30.0,
        dip=45.0,
        length=30e3,
        width=15e3,
        n_length=3,
        n_width=2,
    )
    values = data.draw(
        hnp.arrays(
            dtype=np.float64,
            shape=(fault.n_patches,),
            elements=st.floats(min_value=-10.0, max_value=10.0),
        )
    )
    grid = fault.reshape_patches(values)
    back = fault.flatten_patches(grid)
    np.testing.assert_array_equal(back, values)


@given(st.data())
@settings(max_examples=10, deadline=None)
def test_forward_model_is_linear_in_slip(data):
    """G @ (a x + b y) == a G x + b G y through fault.displacement."""
    import geodef

    fault = geodef.Fault.planar(
        lat=0.0,
        lon=0.0,
        depth=12e3,
        strike=30.0,
        dip=45.0,
        length=30e3,
        width=15e3,
        n_length=2,
        n_width=2,
    )
    n = fault.n_patches
    draw_slip = lambda: data.draw(  # noqa: E731
        hnp.arrays(
            dtype=np.float64,
            shape=(n,),
            elements=st.floats(min_value=-5.0, max_value=5.0),
        )
    )
    sx, dx = draw_slip(), draw_slip()
    sy, dy = draw_slip(), draw_slip()
    a = data.draw(st.floats(min_value=-3.0, max_value=3.0))
    b = data.draw(st.floats(min_value=-3.0, max_value=3.0))
    obs_lat = np.array([0.05, -0.1])
    obs_lon = np.array([0.1, 0.2])

    combined = fault.displacement(obs_lat, obs_lon, a * sx + b * sy, a * dx + b * dy)
    ex, nx, ux = fault.displacement(obs_lat, obs_lon, sx, dx)
    ey, ny, uy = fault.displacement(obs_lat, obs_lon, sy, dy)

    np.testing.assert_allclose(combined[0], a * ex + b * ey, atol=1e-9)
    np.testing.assert_allclose(combined[1], a * nx + b * ny, atol=1e-9)
    np.testing.assert_allclose(combined[2], a * ux + b * uy, atol=1e-9)
