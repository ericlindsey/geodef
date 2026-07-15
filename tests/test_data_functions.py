"""Tests for the function-oriented data construction API."""

import numpy as np
import pytest

from geodef import data
from geodef.data import GNSS, InSAR, Vertical


def test_gnss_returns_named_dataset_with_measurement_metadata():
    dataset = data.gnss(
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

    assert isinstance(dataset, GNSS)
    assert dataset.dataset_name == "campaign_gnss"
    np.testing.assert_array_equal(dataset.station_names, ["A", "B"])
    assert dataset.quantity == "velocity"
    assert dataset.units == "mm/yr"
    assert dataset.epoch == "2025.0"
    assert dataset.time_span == ("2020.0", "2025.0")
    np.testing.assert_array_equal(dataset.east, [0.01, 0.02])
    np.testing.assert_array_equal(dataset.sigma_up, [0.002, 0.002])


def test_horizontal_gnss_has_only_named_horizontal_components():
    dataset = data.horizontal_gnss(
        lon=[100.0],
        lat=[0.0],
        east=[0.01],
        north=[0.02],
        sigma_east=0.001,
        sigma_north=0.002,
    )

    assert isinstance(dataset, GNSS)
    assert dataset.components == "en"
    assert dataset.up is None
    assert dataset.sigma_up is None


def test_insar_broadcasts_scalar_look_and_uncertainty():
    dataset = data.insar(
        lon=[100.0, 100.1],
        lat=[0.0, 0.1],
        los=[0.01, 0.02],
        sigma=0.003,
        look_e=0.0,
        look_n=0.0,
        look_u=1.0,
        name="ascending",
    )

    assert isinstance(dataset, InSAR)
    assert dataset.dataset_name == "ascending"
    np.testing.assert_array_equal(dataset.sigma, [0.003, 0.003])
    np.testing.assert_array_equal(dataset.look_u, [1.0, 1.0])


def test_vertical_returns_existing_dataset_class():
    dataset = data.vertical(
        lon=[100.0, 100.1],
        lat=[0.0, 0.1],
        displacement=[0.01, 0.02],
        sigma=0.004,
        name="uplift",
        station_names=["reef_a", "reef_b"],
    )

    assert isinstance(dataset, Vertical)
    assert dataset.dataset_name == "uplift"
    np.testing.assert_array_equal(dataset.station_names, ["reef_a", "reef_b"])


def test_dataset_metadata_rejects_inconsistent_quantity_and_units():
    with pytest.raises(ValueError, match="velocity units"):
        data.horizontal_gnss(
            lon=[100.0],
            lat=[0.0],
            east=[0.01],
            north=[0.02],
            sigma_east=0.001,
            sigma_north=0.001,
            quantity="velocity",
            units="m",
        )


def test_class_constructor_keeps_legacy_station_name_keyword():
    dataset = Vertical(
        lon=np.array([100.0]),
        lat=np.array([0.0]),
        displacement=np.array([0.01]),
        sigma=np.array([0.001]),
        name=np.array(["A"]),
    )

    np.testing.assert_array_equal(dataset.station_names, ["A"])
    assert dataset.dataset_name is None


def test_dataset_identity_and_semantics_roundtrip_through_dat(tmp_path):
    dataset = data.vertical(
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

    assert loaded.dataset_name == "leveling"
    assert loaded.quantity == "velocity"
    assert loaded.units == "mm/yr"
    assert loaded.epoch == "2025.0"
    assert loaded.time_span == ("2020.0", "2025.0")
    np.testing.assert_array_equal(loaded.station_names, ["benchmark_a"])
