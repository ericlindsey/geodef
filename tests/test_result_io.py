"""Tests for inversion-result files: one .npz with arrays and JSON metadata."""

import json

import numpy as np
import pytest

from geodef.invert import DatasetDiagnostics, InversionResult, SolveDetails


def _result() -> InversionResult:
    diagnostics = DatasetDiagnostics(
        chi2=1.5,
        reduced_chi2=0.5,
        wrms=0.7,
        rms=0.2,
        n_obs=3,
        dof=2.5,
        leverage=0.5,
    )
    return InversionResult(
        slip_vector=np.array([1.0, 0.25]),
        residuals=np.array([0.1, -0.2, 0.3]),
        predicted=np.array([1.0, 2.0, 3.0]),
        reduced_chi2=0.5,
        rms=0.2,
        moment=1e18,
        Mw=5.9,
        regularization="damping",
        regularization_strength=10.0,
        components="both",
        dataset_names=("gnss",),
        warnings=("example warning",),
        details=SolveDetails(
            solver="bounded_ls",
            message="bounded_ls completed",
            regularization_selection="abic",
            system_hash="a" * 64,
            dataset_slices=(slice(0, 3),),
            dataset_diagnostics=(diagnostics,),
            lower_bounds=np.array([0.0, -1.0]),
            upper_bounds=np.array([2.0, 1.0]),
        ),
    )


def test_save_writes_one_file_with_readable_metadata(tmp_path):
    path = tmp_path / "result.npz"

    _result().save(path)

    assert sorted(p.name for p in tmp_path.iterdir()) == ["result.npz"]
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
    assert metadata["components"] == "both"
    assert metadata["details"]["solver"] == "bounded_ls"


def test_round_trip_preserves_the_result(tmp_path):
    path = tmp_path / "result.npz"
    result = _result()

    result.save(path)
    loaded = InversionResult.load(path)

    np.testing.assert_array_equal(loaded.slip_vector, result.slip_vector)
    np.testing.assert_array_equal(loaded.strike_slip, result.strike_slip)
    np.testing.assert_array_equal(loaded.predicted, result.predicted)
    assert loaded.reduced_chi2 == result.reduced_chi2
    assert loaded.regularization == "damping"
    assert loaded.dataset_names == ("gnss",)
    assert loaded.warnings == ("example warning",)
    assert loaded.details.solver == "bounded_ls"
    assert loaded.details.regularization_selection == "abic"
    assert loaded.details.dataset_slices == (slice(0, 3),)
    assert loaded.details.dataset_diagnostics == result.details.dataset_diagnostics
    np.testing.assert_array_equal(loaded.details.lower_bounds, [0.0, -1.0])
    assert loaded.details.constraint_matrix is None
    assert loaded.prediction()["gnss"].tolist() == [1.0, 2.0, 3.0]


def test_custom_regularization_and_nonfinite_values_round_trip(tmp_path):
    path = tmp_path / "result.npz"
    operator = np.array([[1.0, -1.0], [0.0, 1.0]])
    result = InversionResult(
        slip_vector=np.array([0.5]),
        residuals=np.array([0.0]),
        predicted=np.array([1.0]),
        reduced_chi2=float("inf"),
        rms=0.0,
        moment=float("nan"),
        Mw=float("nan"),
        regularization=operator,
        regularization_strength=2.0,
        components="dip",
        quantity="velocity",
        units="mm/yr",
    )

    result.save(path)
    loaded = InversionResult.load(path)

    np.testing.assert_array_equal(loaded.regularization, operator)
    assert np.isinf(loaded.reduced_chi2)
    assert np.isnan(loaded.Mw)
    assert loaded.units == "mm/yr"


def test_loaded_result_needs_attach_before_assessment(tmp_path):
    path = tmp_path / "result.npz"
    _result().save(path)
    loaded = InversionResult.load(path)

    with pytest.raises(ValueError, match="attach"):
        loaded.uncertainty()
