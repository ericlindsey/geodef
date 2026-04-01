# `geodef.cache` — Disk caching

Hash-based caching of Green's matrices and stress kernels. Identical inputs always find the cache; changed inputs always recompute. Cache is enabled by default and stores `.npz` files keyed by SHA-256 hash.

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

`geodef.greens.greens()` and `Fault.stress_kernel()` automatically cache their results. The cache key is computed from all input arrays and parameters (fault geometry, observation coordinates, data class, etc.) using SHA-256. If the key matches an existing `.npz` file, the result is loaded from disk; otherwise it is computed and saved.

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
