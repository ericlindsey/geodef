"""Tests for the unified dataset constructors, ``synthetic``, and ``from_table``."""

import numpy as np
import pytest

import geodef
from geodef import GNSS, InSAR, Vertical


@pytest.fixture
def fault():
    return geodef.Fault.planar(
        lon=100.0,
        lat=0.0,
        depth=15e3,
        strike=0.0,
        dip=30.0,
        length=40e3,
        width=20e3,
        n_length=4,
        n_width=2,
    )


# ----------------------------------------------------------------------
# One constructor per dataset type
# ----------------------------------------------------------------------


def test_gnss_constructor_carries_names_and_measurement_metadata():
    dataset = GNSS(
        lon=np.array([100.0, 100.1]),
        lat=np.array([0.0, 0.1]),
        east=np.array([0.01, 0.02]),
        north=np.array([0.03, 0.04]),
        up=np.array([0.005, 0.006]),
        sigma_east=0.001,
        sigma_north=0.001,
        sigma_up=0.002,
        name="campaign_gnss",
        station_names=["A", "B"],
        quantity="velocity",
        units="mm/yr",
        epoch="2025.0",
        time_span=("2020.0", "2025.0"),
    )

    assert dataset.name == "campaign_gnss"
    np.testing.assert_array_equal(dataset.station_names, ["A", "B"])
    assert dataset.quantity == "velocity"
    assert dataset.units == "mm/yr"
    assert dataset.epoch == "2025.0"
    assert dataset.time_span == ("2020.0", "2025.0")
    np.testing.assert_array_equal(dataset.east, [0.01, 0.02])
    np.testing.assert_array_equal(dataset.sigma_up, [0.002, 0.002])


def test_gnss_single_sigma_applies_to_every_component():
    dataset = GNSS(
        lon=[100.0, 100.1], lat=[0.0, 0.1], east=0.0, north=0.0, up=0.0, sigma=0.01
    )

    np.testing.assert_array_equal(dataset.sigma, np.full(6, 0.01))


def test_gnss_component_sigma_overrides_the_shared_sigma():
    dataset = GNSS(
        lon=[100.0],
        lat=[0.0],
        east=[0.01],
        north=[0.02],
        up=[0.03],
        sigma=0.004,
        sigma_up=0.008,
    )

    np.testing.assert_array_equal(dataset.sigma, [0.004, 0.004, 0.008])


def test_gnss_requires_an_uncertainty_for_each_component():
    with pytest.raises(ValueError, match="sigma_north"):
        GNSS(lon=[100.0], lat=[0.0], east=[0.01], north=[0.02], sigma_east=0.001)


def test_gnss_without_up_is_horizontal_only():
    dataset = GNSS(lon=[100.0], lat=[0.0], east=[0.01], north=[0.02], sigma=0.001)

    assert dataset.components == "en"
    assert dataset.up is None
    assert dataset.sigma_up is None
    assert dataset.n_obs == 2


def test_gnss_rejects_vertical_sigma_without_vertical_values():
    with pytest.raises(ValueError, match="sigma_up"):
        GNSS(
            lon=[100.0],
            lat=[0.0],
            east=[0.01],
            north=[0.02],
            sigma=0.001,
            sigma_up=0.002,
        )


def test_insar_broadcasts_scalar_look_and_uncertainty():
    dataset = InSAR(
        lon=[100.0, 100.1],
        lat=[0.0, 0.1],
        los=[0.01, 0.02],
        sigma=0.003,
        look_e=0.0,
        look_n=0.0,
        look_u=1.0,
        name="ascending",
    )

    assert dataset.name == "ascending"
    np.testing.assert_array_equal(dataset.sigma, [0.003, 0.003])
    np.testing.assert_array_equal(dataset.look_u, [1.0, 1.0])


def test_vertical_constructor_names_dataset_and_stations():
    dataset = Vertical(
        lon=[100.0, 100.1],
        lat=[0.0, 0.1],
        displacement=[0.01, 0.02],
        sigma=0.004,
        name="uplift",
        station_names=["reef_a", "reef_b"],
    )

    assert dataset.name == "uplift"
    np.testing.assert_array_equal(dataset.station_names, ["reef_a", "reef_b"])


def test_dataset_metadata_rejects_inconsistent_quantity_and_units():
    with pytest.raises(ValueError, match="velocity units"):
        GNSS(
            lon=[100.0],
            lat=[0.0],
            east=[0.01],
            north=[0.02],
            sigma=0.001,
            quantity="velocity",
            units="m",
        )


def test_dataset_identity_and_semantics_roundtrip_through_dat(tmp_path):
    dataset = Vertical(
        lon=[100.0],
        lat=[0.0],
        displacement=[2.0],
        sigma=0.1,
        name="leveling",
        station_names=["benchmark_a"],
        quantity="velocity",
        units="mm/yr",
        epoch="2025.0",
        time_span=("2020.0", "2025.0"),
    )
    path = tmp_path / "vertical.dat"

    dataset.save(path)
    loaded = Vertical.load(path)

    assert loaded.name == "leveling"
    assert loaded.quantity == "velocity"
    assert loaded.units == "mm/yr"
    assert loaded.epoch == "2025.0"
    assert loaded.time_span == ("2020.0", "2025.0")
    np.testing.assert_array_equal(loaded.station_names, ["benchmark_a"])


# ----------------------------------------------------------------------
# with_obs
# ----------------------------------------------------------------------


def test_with_obs_replaces_values_and_keeps_everything_else():
    dataset = GNSS(
        lon=[100.0, 100.1],
        lat=[0.0, 0.1],
        east=[1.0, 2.0],
        north=[3.0, 4.0],
        up=[5.0, 6.0],
        sigma=0.01,
        name="network",
    )

    replaced = dataset.with_obs(np.arange(6.0))

    np.testing.assert_array_equal(replaced.east, [0.0, 3.0])
    np.testing.assert_array_equal(replaced.north, [1.0, 4.0])
    np.testing.assert_array_equal(replaced.up, [2.0, 5.0])
    np.testing.assert_array_equal(replaced.sigma, dataset.sigma)
    assert replaced.name == "network"
    np.testing.assert_array_equal(dataset.east, [1.0, 2.0])


def test_with_obs_rejects_wrong_length():
    dataset = Vertical(lon=[100.0], lat=[0.0], displacement=[1.0], sigma=0.1)

    with pytest.raises(ValueError, match="obs"):
        dataset.with_obs([1.0, 2.0])


# ----------------------------------------------------------------------
# synthetic
# ----------------------------------------------------------------------


def test_synthetic_gnss_without_noise_is_the_forward_model(fault):
    lon = np.array([99.9, 100.0, 100.2])
    lat = np.array([-0.1, 0.05, 0.1])
    dip_slip = np.linspace(0.5, 1.5, fault.n_patches)

    dataset = GNSS.synthetic(fault, lon=lon, lat=lat, dip_slip=dip_slip, sigma=0.01)

    east, north, up = fault.displacement(lon=lon, lat=lat, dip_slip=dip_slip)
    np.testing.assert_allclose(dataset.east, east)
    np.testing.assert_allclose(dataset.north, north)
    np.testing.assert_allclose(dataset.up, up)
    np.testing.assert_array_equal(dataset.sigma, np.full(9, 0.01))


def test_synthetic_noise_matches_the_declared_sigma_stream(fault):
    lon = np.array([99.9, 100.0, 100.2])
    lat = np.array([-0.1, 0.05, 0.1])
    clean = GNSS.synthetic(fault, lon=lon, lat=lat, dip_slip=1.0, sigma=0.01)

    noisy = GNSS.synthetic(
        fault, lon=lon, lat=lat, dip_slip=1.0, sigma=0.01, rng=np.random.default_rng(3)
    )

    expected = clean.obs + np.random.default_rng(3).normal(0.0, 0.01, 9)
    np.testing.assert_allclose(noisy.obs, expected)


def test_synthetic_accepts_an_integer_seed(fault):
    kwargs = dict(lon=[100.0, 100.1], lat=[0.0, 0.1], dip_slip=1.0, sigma=0.01)

    first = GNSS.synthetic(fault, rng=7, **kwargs)
    second = GNSS.synthetic(fault, rng=7, **kwargs)

    np.testing.assert_array_equal(first.obs, second.obs)


def test_synthetic_horizontal_gnss(fault):
    dataset = GNSS.synthetic(
        fault, lon=[100.0], lat=[0.1], strike_slip=1.0, sigma=0.01, components="en"
    )

    assert dataset.components == "en"
    assert dataset.n_obs == 2


def test_synthetic_insar_projects_onto_the_look_vector(fault):
    lon = np.array([99.9, 100.1])
    lat = np.array([0.0, 0.1])

    dataset = InSAR.synthetic(
        fault,
        lon=lon,
        lat=lat,
        dip_slip=1.0,
        sigma=0.005,
        look_e=0.6,
        look_n=0.0,
        look_u=0.8,
    )

    east, _north, up = fault.displacement(lon=lon, lat=lat, dip_slip=1.0)
    np.testing.assert_allclose(dataset.obs, 0.6 * east + 0.8 * up)


def test_synthetic_vertical_keeps_the_up_component(fault):
    lon = np.array([99.9, 100.1])
    lat = np.array([0.0, 0.1])

    dataset = Vertical.synthetic(fault, lon=lon, lat=lat, dip_slip=1.0, sigma=0.01)

    _east, _north, up = fault.displacement(lon=lon, lat=lat, dip_slip=1.0)
    np.testing.assert_allclose(dataset.obs, up)


def test_synthetic_draws_correlated_noise_from_a_full_covariance(fault):
    lon = np.array([99.9, 100.1])
    lat = np.array([0.0, 0.1])
    covariance = np.array([[4e-4, 3.9e-4], [3.9e-4, 4e-4]])
    clean = Vertical.synthetic(
        fault, lon=lon, lat=lat, dip_slip=1.0, sigma=0.02, covariance=covariance
    )

    draws = np.array(
        [
            Vertical.synthetic(
                fault,
                lon=lon,
                lat=lat,
                dip_slip=1.0,
                sigma=0.02,
                covariance=covariance,
                rng=seed,
            ).obs
            - clean.obs
            for seed in range(400)
        ]
    )

    assert np.corrcoef(draws.T)[0, 1] > 0.9


# ----------------------------------------------------------------------
# from_table
# ----------------------------------------------------------------------


def test_from_table_uses_explicit_columns_and_station_names():
    table = {
        "longitude": [100.0, 100.1],
        "latitude": [0.0, 0.1],
        "uplift_mm": [1.0, 2.0],
        "sigma_mm": [0.1, 0.2],
        "benchmark": ["A", "B"],
    }

    dataset = Vertical.from_table(
        table,
        columns={
            "lon": "longitude",
            "lat": "latitude",
            "displacement": "uplift_mm",
            "sigma": "sigma_mm",
            "station_names": "benchmark",
        },
        units="mm",
        name="leveling",
    )

    assert isinstance(dataset, Vertical)
    assert dataset.units == "mm"
    assert dataset.name == "leveling"
    np.testing.assert_array_equal(dataset.station_names, ["A", "B"])


def test_from_table_defaults_to_columns_named_like_the_fields():
    table = {
        "lon": [100.0, 100.1],
        "lat": [0.0, 0.1],
        "displacement": [1.0, 2.0],
        "sigma": [0.1, 0.2],
    }

    dataset = Vertical.from_table(table)

    np.testing.assert_array_equal(dataset.obs, [1.0, 2.0])


def test_from_table_reads_structured_arrays():
    table = np.array(
        [(100.0, 0.0, 0.01, 0.001), (100.1, 0.1, 0.02, 0.001)],
        dtype=[("x", float), ("y", float), ("z", float), ("sz", float)],
    )

    dataset = Vertical.from_table(
        table, columns={"lon": "x", "lat": "y", "displacement": "z", "sigma": "sz"}
    )

    np.testing.assert_array_equal(dataset.obs, [0.01, 0.02])


def test_from_table_drops_rows_with_missing_values():
    table = {
        "lon": [100.0, 100.1, 100.2],
        "lat": [0.0, 0.1, 0.2],
        "east": [1.0, np.nan, 3.0],
        "north": [4.0, 5.0, 6.0],
        "se": [0.1, 0.1, 0.1],
        "sn": [0.2, 0.2, 0.2],
    }

    dataset = GNSS.from_table(
        table,
        columns={
            "lon": "lon",
            "lat": "lat",
            "east": "east",
            "north": "north",
            "sigma_east": "se",
            "sigma_north": "sn",
        },
        missing="drop",
    )

    np.testing.assert_array_equal(dataset.east, [1.0, 3.0])
    assert dataset.components == "en"


def test_from_table_reports_missing_values_by_field_and_column():
    table = {
        "lon": [100.0],
        "lat": [0.0],
        "east_mm": [np.nan],
        "north_mm": [1.0],
    }

    with pytest.raises(ValueError, match="east.*east_mm"):
        GNSS.from_table(
            table,
            columns={
                "lon": "lon",
                "lat": "lat",
                "east": "east_mm",
                "north": "north_mm",
            },
            sigma=0.1,
        )


def test_from_table_reports_absent_source_column():
    with pytest.raises(KeyError, match="uplift"):
        Vertical.from_table(
            {"lon": [100.0], "lat": [0.0], "sigma": [0.1]},
            columns={
                "lon": "lon",
                "lat": "lat",
                "displacement": "uplift",
                "sigma": "sigma",
            },
        )


def test_from_table_rejects_unknown_fields():
    with pytest.raises(ValueError, match="unknown fields"):
        Vertical.from_table(
            {"lon": [100.0], "lat": [0.0], "d": [1.0], "s": [0.1]},
            columns={"lon": "lon", "lat": "lat", "uplift": "d", "sigma": "s"},
        )
