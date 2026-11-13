"""Tests for self-interpreting inversion results and assessment functions."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from geodef import GNSS, Vertical, backend, invert, plot
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
    east, north, up = fault.displacement(
        lat=lat, lon=lon, strike_slip=[1.0], dip_slip=[0.25]
    )
    gnss = GNSS(
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
    vertical = Vertical(
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
    assert result.details.dataset_slices == (slice(0, 9), slice(9, 12))
    assert result.details.solver == "nnls"
    assert result.details.success is True
    assert result.details.regularization_selection is None
    assert result.details.backend == backend.get_backend()
    assert result.details.precision == backend.get_precision()
    assert result.quantity == "displacement"
    assert result.units == "m"
    assert len(result.details.system_hash) == 64
    np.testing.assert_array_equal(result.details.lower_bounds, [0.0, 0.0])
    assert result.details.upper_bounds is not None
    assert np.all(np.isposinf(result.details.upper_bounds))


def test_automatic_regularization_selection_is_recorded(fault):
    gnss, _ = _datasets(fault)

    result = invert.solve(
        fault,
        gnss,
        regularization="damping",
        regularization_strength="abic",
    )

    assert result.details.regularization_selection == "abic"
    assert result.regularization_strength is not None


def test_explicit_duplicate_dataset_names_raise(fault):
    gnss, _ = _datasets(fault)
    duplicate = GNSS(
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
    first = GNSS(
        lon=gnss.lon,
        lat=gnss.lat,
        east=gnss.east,
        north=gnss.north,
        sigma_east=gnss.sigma_east,
        sigma_north=gnss.sigma_north,
    )
    second = GNSS(
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
    velocity = Vertical(
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

    predictions = result.prediction()
    residuals = result.residual()
    diagnostics = result.diagnostics()

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

    text = result.summary()

    assert "gnss" in text
    assert "leveling" in text
    assert "reduced chi-squared" in text
    assert result.details.solver in text


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


def test_result_remembers_its_problem_for_assessment(fault):
    gnss, vertical = _datasets(fault)

    result = invert.solve(fault, [gnss, vertical])

    assert result.fault is fault
    assert result.datasets == [gnss, vertical]
    covariance = result.covariance()
    assert covariance.shape == (2, 2)
    np.testing.assert_allclose(result.uncertainty(), np.sqrt(np.diag(covariance)))
    np.testing.assert_allclose(result.resolution(), np.eye(2), atol=1e-8)


def test_attach_restores_assessment_on_a_loaded_result(fault, tmp_path):
    gnss, vertical = _datasets(fault)
    result = invert.solve(fault, [gnss, vertical])
    path = tmp_path / "result.npz"
    result.save(path)

    loaded = invert.InversionResult.load(path, fault=fault, datasets=[gnss, vertical])

    np.testing.assert_allclose(loaded.uncertainty(), result.uncertainty())


def test_linear_system_solves_repeatedly_without_reassembly(fault):
    gnss, vertical = _datasets(fault)
    system = invert.LinearSystem(fault, [gnss, vertical], regularization="damping")

    weak = system.solve(regularization_strength=1e-6)
    strong = system.solve(regularization_strength=1e6)

    assert weak.system is system and strong.system is system
    assert np.linalg.norm(strong.slip_vector) < np.linalg.norm(weak.slip_vector)


def test_prediction_and_residual_select_one_dataset_by_object_or_name(fault):
    gnss, vertical = _datasets(fault)
    result = invert.solve(fault, [gnss, vertical])

    np.testing.assert_array_equal(
        result.prediction(vertical), result.prediction()["leveling"]
    )
    np.testing.assert_array_equal(result.prediction("gnss"), result.predicted[:9])
    np.testing.assert_allclose(
        result.residual(gnss), gnss.obs - result.prediction(gnss)
    )


def test_prediction_rejects_a_dataset_that_was_not_inverted(fault):
    gnss, vertical = _datasets(fault)
    result = invert.solve(fault, gnss)

    with pytest.raises(ValueError, match="not one of the datasets"):
        result.prediction(vertical)
    with pytest.raises(KeyError, match="leveling"):
        result.prediction("leveling")
