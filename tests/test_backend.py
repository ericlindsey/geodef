"""Tests for the geodef.backend array-namespace shim."""

import importlib

import numpy as np
import pytest

from geodef import backend

try:
    import jax  # noqa: F401

    HAVE_JAX = True
except ImportError:
    HAVE_JAX = False

requires_jax = pytest.mark.skipif(not HAVE_JAX, reason="jax is not installed")


@pytest.fixture(autouse=True)
def restore_backend():
    """Restore the default backend and precision after each test."""
    yield
    backend.set_backend("numpy")
    backend.set_precision("float64")


# ======================================================================
# Defaults and selection
# ======================================================================


class TestSelection:
    def test_default_backend_is_numpy(self):
        assert backend.get_backend() == "numpy"

    def test_default_namespace_is_numpy(self):
        assert backend.namespace() is np

    def test_set_backend_numpy_roundtrip(self):
        backend.set_backend("numpy")
        assert backend.get_backend() == "numpy"
        assert backend.namespace() is np

    def test_set_backend_invalid_name_raises(self):
        with pytest.raises(ValueError, match="jax"):
            backend.set_backend("tensorflow")

    def test_set_backend_jax_missing_raises_importerror(self):
        if HAVE_JAX:
            pytest.skip("jax is installed")
        with pytest.raises(ImportError, match="geodef\\[jax\\]"):
            backend.set_backend("jax")

    @requires_jax
    def test_set_backend_jax(self):
        import jax.numpy as jnp

        backend.set_backend("jax")
        assert backend.get_backend() == "jax"
        assert backend.namespace() is jnp


# ======================================================================
# Precision
# ======================================================================


class TestPrecision:
    def test_default_precision_is_float64(self):
        assert backend.get_precision() == "float64"
        assert backend.default_dtype() == np.float64

    def test_set_precision_float32(self):
        backend.set_precision("float32")
        assert backend.get_precision() == "float32"
        assert backend.default_dtype() == np.float32

    def test_set_precision_invalid_raises(self):
        with pytest.raises(ValueError, match="float16"):
            backend.set_precision("float16")

    @requires_jax
    def test_jax_float64_default(self):
        backend.set_backend("jax")
        xp = backend.namespace()
        assert xp.asarray(1.0, dtype=backend.default_dtype()).dtype == np.float64

    @requires_jax
    def test_jax_float32_opt_in(self):
        backend.set_backend("jax")
        backend.set_precision("float32")
        xp = backend.namespace()
        assert xp.asarray(1.0, dtype=backend.default_dtype()).dtype == np.float32


# ======================================================================
# Array conversion at module boundaries
# ======================================================================


class TestToNumpy:
    def test_numpy_array_passes_through(self):
        a = np.arange(3.0)
        out = backend.to_numpy(a)
        assert isinstance(out, np.ndarray)
        np.testing.assert_array_equal(out, a)

    @requires_jax
    def test_jax_array_converts(self):
        backend.set_backend("jax")
        xp = backend.namespace()
        a = xp.arange(3.0)
        out = backend.to_numpy(a)
        assert isinstance(out, np.ndarray)
        np.testing.assert_array_equal(out, np.arange(3.0))


# ======================================================================
# Environment variable
# ======================================================================


class TestEnvVar:
    def test_env_var_selects_backend_at_import(self, monkeypatch):
        monkeypatch.setenv("GEODEF_BACKEND", "numpy")
        importlib.reload(backend)
        assert backend.get_backend() == "numpy"

    @requires_jax
    def test_env_var_jax(self, monkeypatch):
        monkeypatch.setenv("GEODEF_BACKEND", "jax")
        importlib.reload(backend)
        assert backend.get_backend() == "jax"
        importlib.reload(backend)

    def test_env_var_invalid_falls_back_to_numpy(self, monkeypatch, caplog):
        monkeypatch.setenv("GEODEF_BACKEND", "not-a-backend")
        importlib.reload(backend)
        assert backend.get_backend() == "numpy"

    def test_env_var_unset_defaults_to_numpy(self, monkeypatch):
        monkeypatch.delenv("GEODEF_BACKEND", raising=False)
        importlib.reload(backend)
        assert backend.get_backend() == "numpy"
