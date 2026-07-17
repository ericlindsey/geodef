"""Tests for safe, versioned inversion-result files."""

import json

import numpy as np

from geodef import invert
from geodef.invert import DatasetDiagnostics, InversionResult


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
        slip=np.array([[1.0, 0.25]]),
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
        dataset_slices=(slice(0, 3),),
        solver="bounded_ls",
        success=True,
        message="bounded_ls completed",
        regularization_selection="abic",
        backend="numpy",
        precision="float64",
        warnings=("example warning",),
        quantity="displacement",
        units="m",
        system_hash="a" * 64,
        lower_bounds=np.array([0.0, -1.0]),
        upper_bounds=np.array([2.0, 1.0]),
        dataset_diagnostics=(diagnostics,),
    )


def test_save_writes_npz_and_human_readable_manifest(tmp_path):
    path = tmp_path / "result.npz"

    invert.save(_result(), path)

    manifest_path = tmp_path / "result.manifest.json"
    assert path.exists()
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema"] == "geodef.inversion_result"
    assert manifest["schema_version"] == invert.RESULT_SCHEMA_VERSION
    assert manifest["result"]["dataset_names"] == ["gnss"]
    assert manifest["arrays"]["slip"]["shape"] == [1, 2]


def test_save_load_roundtrips_result_metadata_and_arrays(tmp_path):
    result = _result()
    path = tmp_path / "result.npz"

    invert.save(result, path)
    loaded = invert.load(path)

    np.testing.assert_array_equal(loaded.slip, result.slip)
    np.testing.assert_array_equal(loaded.lower_bounds, result.lower_bounds)
    assert loaded.dataset_names == result.dataset_names
    assert loaded.dataset_slices == result.dataset_slices
    assert loaded.dataset_diagnostics == result.dataset_diagnostics
    assert loaded.regularization_selection == "abic"
    assert loaded.warnings == ("example warning",)
    assert loaded.system_hash == "a" * 64


def test_embedded_manifest_keeps_npz_portable_without_sidecar(tmp_path):
    path = tmp_path / "result.npz"
    invert.save(_result(), path)
    (tmp_path / "result.manifest.json").unlink()

    loaded = invert.load(path)

    assert loaded.dataset_names == ("gnss",)


def test_load_migrates_legacy_unversioned_archive(tmp_path):
    path = tmp_path / "legacy.npz"
    np.savez_compressed(
        path,
        slip=np.array([[1.0, 0.0]]),
        slip_vector=np.array([1.0, 0.0]),
        residuals=np.array([0.0]),
        predicted=np.array([1.0]),
        reduced_chi2=np.array([0.0]),
        rms=np.array([0.0]),
        moment=np.array([1e18]),
        Mw=np.array([5.9]),
        smoothing_str=np.array(["__none__"]),
        smoothing_strength=np.array([np.nan]),
        components=np.array(["both"]),
    )

    loaded = invert.load(path)

    assert loaded.dataset_names == ("data",)
    assert loaded.dataset_slices == (slice(0, 1),)
    assert "legacy" in loaded.warnings[0]


def test_load_migrates_v2_smoothing_keys(tmp_path):
    """A version-2 file keyed by ``smoothing*`` loads into the new fields."""
    path = tmp_path / "v2.npz"
    invert.save(_result(), path)
    (tmp_path / "v2.manifest.json").unlink()

    with np.load(path, allow_pickle=False) as loaded:
        arrays = {name: loaded[name].copy() for name in loaded.files}
    manifest = json.loads(bytes(arrays.pop("__manifest__").tolist()).decode())
    manifest["schema_version"] = 2
    rename = {
        "regularization": "smoothing",
        "regularization_strength": "smoothing_strength",
        "regularization_selection": "smoothing_selection",
    }
    manifest["result"] = {
        rename.get(key, key): value for key, value in manifest["result"].items()
    }
    arrays["__manifest__"] = np.frombuffer(
        json.dumps(manifest).encode(), dtype=np.uint8
    )
    np.savez_compressed(path, **arrays)

    loaded_result = invert.load(path)

    assert loaded_result.regularization == "damping"
    assert loaded_result.regularization_strength == 10.0
    assert loaded_result.regularization_selection == "abic"


def test_result_record_has_no_io_workflow_methods():
    assert not hasattr(InversionResult, "save")
    assert not hasattr(InversionResult, "load")
    assert not hasattr(InversionResult, "save_table")
