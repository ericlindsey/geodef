# `geodef.cache` — Disk caching

Hash-based caching of Green's matrices and stress kernels. Identical inputs always find the cache; changed inputs always recompute. Cache is enabled by default and stores `.npz` files keyed by SHA-256 hash.

Caching changes runtime, not numerical results. It is especially helpful while
trying different regularization strengths on fixed geometry and data, because
the expensive elastic response is unchanged. Disable it for benchmarks so a
disk hit is not mistaken for faster computation.

---

## Configuration

```python
import geodef

geodef.cache.set_dir("my_cache/")   # default: .geodef_cache/
geodef.cache.get_dir()               # → Path('.geodef_cache')

geodef.cache.enable()                # on by default
geodef.cache.disable()               # disable for this session
geodef.cache.is_enabled()            # → True/False
```

---

## Inspection and cleanup

```python
geodef.cache.info()
# → {'n_files': 3, 'total_bytes': 45678}

geodef.cache.clear()    # delete all cached files
```

---

## How caching works

`geodef.greens.greens()` and `Fault.stress_kernel()` automatically cache their
results. The cache key is computed from all input arrays and parameters (fault
geometry, observation coordinates, data class, active GNSS components, look
vectors, etc.) using SHA-256. If the key matches an existing `.npz` file, the
result is loaded from disk; otherwise it is computed and saved.

The hash identifies values and configuration, not scientific provenance. Keep
raw-data processing, coordinate conventions, and software versions in your
research metadata; the cache is disposable and can always be rebuilt.

To bypass caching entirely for a session:

```python
geodef.cache.disable()
G = geodef.greens.greens(fault, gnss)  # always recomputes
geodef.cache.enable()
```

---

## Advanced: `cached_compute`

```python
from geodef.cache import cached_compute

result = cached_compute(key_dict, compute_fn)
```

`key_dict` is any dict of arrays and scalars; `compute_fn` is a zero-argument callable. If `key_dict` hashes to a known cache file, the saved result is returned; otherwise `compute_fn()` is called and the result is saved. Use this to cache custom expensive computations.
