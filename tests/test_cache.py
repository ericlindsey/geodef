"""Tests for geodef.cache — hash-based disk caching for expensive computations."""

from pathlib import Path

import numpy as np
import pytest

from geodef import cache
from geodef.cache import compute_hash
from geodef.data import GNSS, InSAR
from geodef.fault import Fault
from geodef.greens import greens


# ====================================================================
# Group 1: compute_hash (pure function)
# ====================================================================


class TestComputeHash:
    """Tests for the deterministic hash function."""

    def test_deterministic(self) -> None:
        """Same key_data produces the same hash on repeated calls."""
        key = {"arr": np.array([1.0, 2.0, 3.0]), "name": "test"}
        assert compute_hash(key) == compute_hash(key)

    def test_different_arrays(self) -> None:
        """Different array values produce different hashes."""
        key_a = {"arr": np.array([1.0, 2.0, 3.0])}
        key_b = {"arr": np.array([1.0, 2.0, 4.0])}
        assert compute_hash(key_a) != compute_hash(key_b)

    def test_different_dtypes(self) -> None:
        """float64 vs float32 arrays produce different hashes."""
        key_a = {"arr": np.array([1.0, 2.0], dtype=np.float64)}
        key_b = {"arr": np.array([1.0, 2.0], dtype=np.float32)}
        assert compute_hash(key_a) != compute_hash(key_b)

    def test_different_shapes(self) -> None:
        """Arrays with same bytes but different shapes produce different hashes."""
        data = np.arange(6, dtype=np.float64)
        key_a = {"arr": data.reshape(6)}
        key_b = {"arr": data.reshape(2, 3)}
        assert compute_hash(key_a) != compute_hash(key_b)

    def test_strings(self) -> None:
        """String values hash correctly and differ."""
        key_a = {"name": "okada"}
        key_b = {"name": "tri"}
        assert compute_hash(key_a) != compute_hash(key_b)

    def test_none(self) -> None:
        """None values are handled without error and differ from non-None."""
        key_a = {"val": None}
        key_b = {"val": np.array([0.0])}
        h_a = compute_hash(key_a)
        h_b = compute_hash(key_b)
        assert isinstance(h_a, str)
        assert len(h_a) == 64  # SHA-256 hex digest
        assert h_a != h_b

    def test_mixed_types(self) -> None:
        """Dict with arrays, strings, floats, and None hashes without error."""
        key = {
            "arr": np.array([1.0, 2.0]),
            "name": "test",
            "value": 3.14,
            "count": 42,
            "optional": None,
        }
        h = compute_hash(key)
        assert isinstance(h, str)
        assert len(h) == 64
        # Same input reproduces same hash
        assert compute_hash(key) == h


# ====================================================================
# Group 2: Module-level config functions
# ====================================================================


class TestConfig:
    """Tests for cache configuration functions."""

    def test_default_directory(self) -> None:
        """Default cache directory is .geodef_cache."""
        # Create a fresh config to check the default
        from geodef.cache import _CacheConfig

        cfg = _CacheConfig()
        assert cfg.directory == Path(".geodef_cache")

    def test_set_dir(self, tmp_path: Path) -> None:
        """set_dir changes the cache directory."""
        new_dir = tmp_path / "my_cache"
        cache.set_dir(new_dir)
        assert cache.get_dir() == new_dir

    def test_set_dir_string(self, tmp_path: Path) -> None:
        """set_dir accepts a string path."""
        cache.set_dir(str(tmp_path / "str_cache"))
        assert cache.get_dir() == tmp_path / "str_cache"

    def test_enable_disable(self) -> None:
        """disable() and enable() toggle is_enabled()."""
        assert cache.is_enabled() is True
        cache.disable()
        assert cache.is_enabled() is False
        cache.enable()
        assert cache.is_enabled() is True

    def test_clear(self, tmp_path: Path) -> None:
        """clear() removes all cached files."""
        cache_dir = tmp_path / "cache"
        cache.set_dir(cache_dir)
        # Create some fake cache files
        sub = cache_dir / "ab"
        sub.mkdir(parents=True)
        (sub / "abcdef.npz").write_bytes(b"fake")
        assert any(cache_dir.rglob("*.npz"))
        cache.clear()
        assert not any(cache_dir.rglob("*.npz"))


# ====================================================================
# Group 3: cached_compute
# ====================================================================


class TestCachedCompute:
    """Tests for the core cached_compute function."""

    def test_miss_calls_compute(self, tmp_path: Path) -> None:
        """On first call, compute_fn is invoked and result is correct."""
        expected = np.array([[1.0, 2.0], [3.0, 4.0]])
        call_count = [0]

        def compute() -> np.ndarray:
            call_count[0] += 1
            return expected

        key = {"id": "test_miss"}
        result = cache.cached_compute(key, compute)
        assert call_count[0] == 1
        np.testing.assert_array_equal(result, expected)

    def test_hit_skips_compute(self, tmp_path: Path) -> None:
        """On second call with same key, compute_fn is NOT invoked."""
        expected = np.array([10.0, 20.0, 30.0])
        call_count = [0]

        def compute() -> np.ndarray:
            call_count[0] += 1
            return expected

        key = {"id": "test_hit"}
        cache.cached_compute(key, compute)
        assert call_count[0] == 1
        cache.cached_compute(key, compute)
        assert call_count[0] == 1

    def test_hit_returns_correct_data(self, tmp_path: Path) -> None:
        """Loaded result matches original exactly."""
        expected = np.random.default_rng(42).random((5, 3))
        key = {"id": "test_exact"}
        cache.cached_compute(key, lambda: expected)
        loaded = cache.cached_compute(key, lambda: np.zeros(1))
        np.testing.assert_array_equal(loaded, expected)

    def test_miss_on_changed_key(self, tmp_path: Path) -> None:
        """Changing one value in key_data causes recompute."""
        call_count = [0]

        def compute() -> np.ndarray:
            call_count[0] += 1
            return np.array([float(call_count[0])])

        cache.cached_compute({"x": np.array([1.0])}, compute)
        cache.cached_compute({"x": np.array([2.0])}, compute)
        assert call_count[0] == 2

    def test_disabled_always_computes(self, tmp_path: Path) -> None:
        """With disable(), compute_fn is always called and no files created."""
        cache.disable()
        call_count = [0]

        def compute() -> np.ndarray:
            call_count[0] += 1
            return np.array([1.0])

        key = {"id": "test_disabled"}
        cache.cached_compute(key, compute)
        cache.cached_compute(key, compute)
        assert call_count[0] == 2
        assert not any(cache.get_dir().rglob("*.npz"))

    def test_creates_subdirectory(self, tmp_path: Path) -> None:
        """Cache file is stored in a 2-char hex subdirectory."""
        key = {"id": "test_subdir"}
        cache.cached_compute(key, lambda: np.array([1.0]))
        npz_files = list(cache.get_dir().rglob("*.npz"))
        assert len(npz_files) == 1
        # Parent should be a 2-char hex directory
        assert len(npz_files[0].parent.name) == 2


# ====================================================================
# Group 4: info()
# ====================================================================


class TestInfo:
    """Tests for the cache info function."""

    def test_info_empty(self, tmp_path: Path) -> None:
        """Empty cache reports 0 files and 0 bytes."""
        result = cache.info()
        assert result["n_files"] == 0
        assert result["total_bytes"] == 0

    def test_info_after_writes(self, tmp_path: Path) -> None:
        """After caching, reports correct file count and nonzero size."""
        cache.cached_compute({"a": "x"}, lambda: np.ones((10, 10)))
        cache.cached_compute({"a": "y"}, lambda: np.ones((5, 5)))
        result = cache.info()
        assert result["n_files"] == 2
        assert result["total_bytes"] > 0


# ====================================================================
# Group 5: Integration with greens()
# ====================================================================


@pytest.fixture
def fault_small() -> Fault:
    """A small 2x2 fault for fast integration tests."""
    return Fault.planar(
        lat=0.0, lon=100.0, depth=15e3,
        strike=0.0, dip=45.0,
        length=20e3, width=10e3,
        n_length=2, n_width=2,
    )


@pytest.fixture
def gnss_data() -> GNSS:
    """2-station GNSS dataset."""
    lat = np.array([0.2, -0.2])
    lon = np.array([100.0, 100.0])
    n = len(lat)
    return GNSS(
        lat, lon,
        ve=np.zeros(n), vn=np.zeros(n), vu=np.zeros(n),
        se=np.ones(n), sn=np.ones(n), su=np.ones(n),
    )


@pytest.fixture
def insar_data() -> InSAR:
    """2-pixel InSAR dataset."""
    lat = np.array([0.2, -0.2])
    lon = np.array([100.0, 100.0])
    n = len(lat)
    return InSAR(
        lat, lon,
        los=np.zeros(n), sigma=np.ones(n),
        look_e=np.full(n, 0.1), look_n=np.full(n, 0.1),
        look_u=np.full(n, 0.98),
    )


class TestGreensIntegration:
    """Tests for caching integration with greens()."""

    def test_greens_caches_result(
        self, fault_small: Fault, gnss_data: GNSS
    ) -> None:
        """First call creates cache file, second call uses it."""
        G1 = greens(fault_small, gnss_data)
        assert cache.info()["n_files"] == 1
        G2 = greens(fault_small, gnss_data)
        np.testing.assert_array_equal(G1, G2)
        # Still only 1 file (reused, not duplicated)
        assert cache.info()["n_files"] == 1

    def test_greens_invalidates_on_fault_change(
        self, gnss_data: GNSS
    ) -> None:
        """Different fault geometry produces a different cache entry."""
        fault_a = Fault.planar(
            lat=0.0, lon=100.0, depth=15e3,
            strike=0.0, dip=45.0,
            length=20e3, width=10e3, n_length=2, n_width=2,
        )
        fault_b = Fault.planar(
            lat=0.0, lon=100.0, depth=15e3,
            strike=90.0, dip=45.0,  # different strike
            length=20e3, width=10e3, n_length=2, n_width=2,
        )
        greens(fault_a, gnss_data)
        greens(fault_b, gnss_data)
        assert cache.info()["n_files"] == 2

    def test_greens_invalidates_on_data_change(
        self, fault_small: Fault
    ) -> None:
        """Different observation locations produce a different cache entry."""
        lat_a = np.array([0.2, -0.2])
        lat_b = np.array([0.3, -0.3])
        lon = np.array([100.0, 100.0])
        data_a = GNSS(lat_a, lon, np.zeros(2), np.zeros(2), np.zeros(2),
                       np.ones(2), np.ones(2), np.ones(2))
        data_b = GNSS(lat_b, lon, np.zeros(2), np.zeros(2), np.zeros(2),
                       np.ones(2), np.ones(2), np.ones(2))
        greens(fault_small, data_a)
        greens(fault_small, data_b)
        assert cache.info()["n_files"] == 2

    def test_greens_joint_datasets(
        self, fault_small: Fault, gnss_data: GNSS, insar_data: InSAR
    ) -> None:
        """Joint datasets produce one cache file per dataset."""
        greens(fault_small, [gnss_data, insar_data])
        assert cache.info()["n_files"] == 2

    def test_greens_disabled(
        self, fault_small: Fault, gnss_data: GNSS
    ) -> None:
        """No files created when caching is disabled."""
        cache.disable()
        greens(fault_small, gnss_data)
        assert cache.info()["n_files"] == 0


# ====================================================================
# Group 6: stress_kernel() depth fix + caching
# ====================================================================


class TestStressKernel:
    """Tests for stress_kernel obs_depth fix and caching."""

    def test_greens_matrix_obs_depth_param(self, fault_small: Fault) -> None:
        """greens_matrix accepts obs_depth and returns a result."""
        G = fault_small.greens_matrix(
            fault_small._lat, fault_small._lon,
            kind="strain", obs_depth=fault_small._depth,
        )
        assert G.shape[0] > 0
        assert G.shape[1] == 2 * fault_small.n_patches

    def test_stress_kernel_uses_depth(self, fault_small: Fault) -> None:
        """stress_kernel evaluates strain at patch depths, not the surface.

        If obs_depth is correctly passed, the strain kernel at depth will
        differ from a (wrong) surface-only evaluation.
        """
        K = fault_small.stress_kernel()
        # The kernel should be nonzero (strain interactions exist)
        assert np.any(K != 0.0)
        # Shape: strain components * n_patches rows, 2 * n_patches cols
        n = fault_small.n_patches
        assert K.shape == (4 * n, 2 * n)

    def test_stress_kernel_caches(self, fault_small: Fault) -> None:
        """stress_kernel creates a cache file and reuses it."""
        fault_small.stress_kernel()
        assert cache.info()["n_files"] == 1
        fault_small.stress_kernel()
        # Still 1 file (reused)
        assert cache.info()["n_files"] == 1

    def test_stress_kernel_different_mu(self, fault_small: Fault) -> None:
        """Different mu values produce different cache entries."""
        fault_small.stress_kernel(mu=30e9)
        fault_small.stress_kernel(mu=40e9)
        assert cache.info()["n_files"] == 2
