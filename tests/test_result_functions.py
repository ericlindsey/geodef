"""Tests for self-interpreting inversion results and assessment functions."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from geodef import backend, data, invert, plot
from geodef.fault import Fault


@pytest.fixture
def fault():
    """Return a small overdetermined planar fault problem."""
    return Fault.planar(
        lat=0.0,
        lon=100.0,
        depth=8_000.0,
        strike=0.0,
        dip=45.0,
        length=10_000.0,
        width=10_000.0,
        n_length=1,
        n_width=1,
    )


def _datasets(fault):
    lon = np.array([99.9, 100.0, 100.1])
    lat = np.array([-0.1, 0.0, 0.1])
    east, north, up = fault.displacement(lat, lon, [1.0], [0.25])
    gnss = data.gnss(
        lon=lon,
        lat=lat,
        east=east,
        north=north,
        up=up,
        sigma_east=0.001,
        sigma_north=0.001,
        sigma_up=0.002,
        name="gnss",
    )
    vertical = data.vertical(
        lon=lon,
        lat=lat,
        displacement=up,
        sigma=0.002,
        name="leveling",
    )
    return gnss, vertical


def test_result_records_named_partitions_and_solver_provenance(fault):
    gnss, vertical = _datasets(fault)

    result = invert.solve(
        fault,
        [gnss, vertical],
        regularization="damping",
        regularization_strength=10.0,
        bounds=(0.0, None),
    )

    assert result.dataset_names == ("gnss", "leveling")
    assert result.dataset_slices == (slice(0, 9), slice(9, 12))
    assert result.solver == "nnls"
    assert result.success is True
    assert result.regularization_selection is None
    assert result.backend == backend.get_backend()
    assert result.precision == backend.get_precision()
    assert result.quantity == "displacement"
    assert result.units == "m"
    assert len(result.system_hash) == 64
    np.testing.assert_array_equal(result.lower_bounds, [0.0, 0.0])
    assert result.upper_bounds is not None
    assert np.all(np.isposinf(result.upper_bounds))


def test_automatic_regularization_selection_is_recorded(fault):
    gnss, _ = _datasets(fault)

    result = invert.solve(
        fault,
        gnss,
        regularization="damping",
        regularization_strength="abic",
    )

    assert result.regularization_selection == "abic"
    assert result.regularization_strength is not None


def test_explicit_duplicate_dataset_names_raise(fault):
    gnss, _ = _datasets(fault)
    duplicate = data.horizontal_gnss(
        lon=gnss.lon,
        lat=gnss.lat,
        east=gnss.east,
        north=gnss.north,
        sigma_east=gnss.sigma_east,
        sigma_north=gnss.sigma_north,
        name="gnss",
    )

    with pytest.raises(ValueError, match="dataset names.*unique"):
        invert.solve(fault, [gnss, duplicate])


def test_unnamed_datasets_receive_stable_readable_names(fault):
    gnss, _ = _datasets(fault)
    first = data.horizontal_gnss(
        lon=gnss.lon,
        lat=gnss.lat,
        east=gnss.east,
        north=gnss.north,
        sigma_east=gnss.sigma_east,
        sigma_north=gnss.sigma_north,
    )
    second = data.horizontal_gnss(
        lon=gnss.lon,
        lat=gnss.lat,
        east=gnss.east,
        north=gnss.north,
        sigma_east=gnss.sigma_east,
        sigma_north=gnss.sigma_north,
    )

    result = invert.solve(fault, [first, second])

    assert result.dataset_names == ("gnss", "gnss_2")


def test_joint_inversion_rejects_mixed_measurement_semantics(fault):
    gnss, vertical = _datasets(fault)
    velocity = data.vertical(
        lon=vertical.lon,
        lat=vertical.lat,
        displacement=vertical.obs,
        sigma=vertical.sigma,
        quantity="velocity",
        units="mm/yr",
    )

    with pytest.raises(ValueError, match="same quantity and units"):
        invert.solve(fault, [gnss, velocity])


def test_assessment_functions_return_name_keyed_views(fault):
    gnss, vertical = _datasets(fault)
    result = invert.solve(fault, [gnss, vertical])

    predictions = invert.prediction(result)
    residuals = invert.residual(result)
    diagnostics = invert.diagnostics(result)

    assert tuple(predictions) == result.dataset_names
    assert tuple(residuals) == result.dataset_names
    assert tuple(diagnostics) == result.dataset_names
    assert predictions["gnss"].shape == (gnss.n_obs,)
    assert predictions["leveling"].shape == (vertical.n_obs,)
    np.testing.assert_allclose(
        predictions["gnss"] + residuals["gnss"],
        gnss.obs,
    )
    assert diagnostics["leveling"].n_obs == vertical.n_obs


def test_summary_is_human_readable_and_named(fault):
    result = invert.solve(fault, list(_datasets(fault)))

    text = invert.summary(result)

    assert "gnss" in text
    assert "leveling" in text
    assert "reduced chi-squared" in text
    assert result.solver in text


@pytest.mark.parametrize(
    ("function", "expected_title"),
    [
        (plot.prediction, "Observed vs. predicted"),
        (plot.residual, "Residuals"),
        (plot.diagnostics, "Dataset diagnostics"),
        (plot.summary, "Inversion summary"),
    ],
)
def test_result_plot_functions_need_no_manual_slicing(fault, function, expected_title):
    result = invert.solve(fault, list(_datasets(fault)))

    axes = function(result)

    assert axes.get_title() == expected_title
    plt.close(axes.figure)
