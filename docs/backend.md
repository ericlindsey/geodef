# `geodef.backend` — Array backend selection

Selects the array library used by the compute kernels: **NumPy** (the default)
or **JAX**. The JAX backend JIT-compiles the Green's-function kernels through
XLA on ordinary CPUs and offloads to a GPU when one is available, and it
enables automatic differentiation of the forward model.

NumPy stays the default everywhere — nothing changes for existing users unless
a backend is explicitly selected.

---

## Installation

The JAX backend is an optional extra:

```bash
pip install geodef[jax]
```

On a machine with an NVIDIA GPU, install JAX's CUDA build instead (see the
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html)):

```bash
pip install geodef "jax[cuda12]"
```

**What to expect on a laptop:** JAX's Metal backend for Apple GPUs is
experimental and does not support float64, so on Apple-silicon laptops the
practical speedup comes from XLA JIT compilation and vectorization on the CPU,
not GPU offload. That is still a substantial win for large Green's matrices.

---

## Configuration

```python
import geodef

geodef.backend.set_backend("jax")     # or "numpy" (default)
geodef.backend.get_backend()          # → 'jax'

geodef.backend.namespace()            # → the active array module
                                      #   (numpy or jax.numpy)
```

The backend can also be chosen at import time with an environment variable:

```bash
GEODEF_BACKEND=jax python my_script.py
```

An unknown or uninstalled backend named in `GEODEF_BACKEND` is ignored with a
logged warning, and GeoDef falls back to NumPy.

---

## Precision

Computations default to **float64**. GPU hardware is typically far faster in
float32, so a lower-precision mode is available as an explicit opt-in:

```python
geodef.backend.set_precision("float32")   # opt-in, GPU-friendly
geodef.backend.get_precision()            # → 'float32'
geodef.backend.default_dtype()            # → dtype('float32')
```

The dislocation kernels are sensitive near the fault surface: expect reduced
accuracy for observation points close to patch edges in float32. Keep the
default float64 unless throughput matters more than near-field precision.

With the JAX backend, precision is synced to JAX's `jax_enable_x64` flag,
which is process-global — enabling float32 here affects other JAX code in the
same process.

---

## Engine coverage

The `okada85` (surface deformation) and `tri` (triangular dislocation)
engines run fully on the selected backend. The `okada92` (internal
deformation, DC3D) engine is a faithful scalar port of the Fortran
reference and always runs on NumPy, regardless of the selected backend.

On the JAX backend, `greens.displacement_greens` (rectangular patches,
surface data) evaluates all patches in one JIT-compiled batched kernel
call instead of looping — typically 10-50x faster than the NumPy loop even
on a plain CPU, after a one-time JIT compilation per problem shape. Strain
and triangular Green's assembly currently still loop per patch and see no
speedup yet. Benchmark with:

```bash
uv run python benchmarks/bench_greens.py
```

---

## Module boundaries

Backend arrays are converted back to NumPy at public API boundaries:

```python
geodef.backend.to_numpy(arr)   # → np.ndarray, zero-copy where possible
```

User-facing functions accept and return `np.ndarray` regardless of the active
backend; JAX arrays only exist inside the compute kernels.
