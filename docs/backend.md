# `geodef.backend` — Array backend selection

> Conventions — axes, depth sign, angles, units, array ordering, regularization: see [`conventions.md`](conventions.md).

Selects the array library used by the compute kernels: **NumPy** (the default)
or **JAX**. The JAX backend JIT-compiles the Green's-function kernels through
XLA on ordinary CPUs and offloads to a GPU when one is available, and it
enables automatic differentiation of the forward model.

NumPy stays the default everywhere — nothing changes for existing users unless
a backend is explicitly selected.

## Why use JAX?

JAX is not a different elastic model. GeoDef evaluates the same Okada and
triangular-dislocation equations with a different array engine. It adds two
capabilities:

1. **Just-in-time (JIT) compilation:** the first call for a new array shape is
   compiled by XLA and may be slow; later calls with that shape reuse the
   compiled program and are usually much faster.
2. **Automatic differentiation (autodiff):** JAX applies the chain rule to the
   implemented forward model and returns derivatives such as
   `partial displacement / partial dip`. These are derivatives of the actual
   numerical code, not finite-difference approximations.

The [JAX documentation](https://docs.jax.dev/en/latest/) introduces JIT and
autodiff in more depth. Use NumPy for ordinary forward models and linear
inversions; select JAX when repeated Green's-matrix assembly or geometry
derivatives justify the compilation cost.

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
experimental, does not support float64, and may lag current JAX/JAXlib releases.
See [Apple's Metal plug-in page](https://developer.apple.com/metal/jax/) before
trying it. The supported GeoDef laptop path is the normal JAX CPU build: XLA
JIT compilation and vectorization can still substantially accelerate large
Green's matrices on Apple silicon.

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
accuracy for observation points close to patch edges in float32, and
~1%-of-scale errors in far-field Green's coefficients of small patches
(the Chinnery corner differences nearly cancel). A good laptop workflow is
to explore in float32 — hyperparameter sweeps, coarse geometry searches —
and rerun the final inversion in float64:

```python
geodef.backend.set_precision("float32")
ac = geodef.invert.abic_curve(fault, data, regularization="laplacian")   # fast sweep
geodef.backend.set_precision("float64")
result = geodef.invert.solve(fault, data, regularization="laplacian",
                             regularization_strength=ac.optimal)   # final solve
```

With the JAX backend, precision is synced to JAX's `jax_enable_x64` flag,
which is process-global — enabling float32 here affects other JAX code in the
same process.

For teaching and research, distinguish **numerical precision** from
**parameter uncertainty**. Float32/float64 control roundoff in a calculation;
they do not describe uncertainty in fault geometry, data, or slip. A result can
be numerically stable to many digits and still be geophysically uncertain.

---

## Engine coverage

All three engines — `okada85` (surface deformation), `okada92` (internal
deformation, DC3D), and `tri` (triangular dislocation) — run fully on the
selected backend. `okada92` is vectorized over observation points, so
strain Green's functions and fault self-stress kernels evaluate all
observation points per patch in a single call on either backend.

On the JAX backend, rectangular Green's assembly — `displacement_greens`,
`strain_greens` at the surface, and `strain_greens` at depth (the
`Fault.stress_kernel` / stress-shadows path) — evaluates all patches in
one JIT-compiled batched kernel call instead of looping, typically
10-50x faster than the NumPy loop even on a plain CPU after a one-time
JIT compilation per problem shape. Differentiable triangular Green's assembly
uses `jax.vmap` over triangles, so it is also compatible with JIT compilation;
its larger kernel can have a substantial first-call compilation cost. Benchmark
with:

```bash
uv run python benchmarks/bench_greens.py
```

---

## Module boundaries

Backend arrays are converted back to NumPy at public API boundaries:

```python
geodef.backend.to_numpy(arr)   # → np.ndarray, zero-copy where possible
```

The kernels use `backend.masked_eval(func, mask, args, n_out)` to evaluate a
vectorized function only where a boolean mask is true: on NumPy the true lanes
are gathered and scattered back (no wasted work), while on JAX the function
runs on full arrays under `where` because traced shapes cannot depend on data.
It is public so custom engines can reuse the same backend-portable pattern.

Most high-level user-facing functions convert results to `np.ndarray`
regardless of the active backend. The low-level traceable functions in
`geodef.gradients` and `geodef.bayes.logpdf` intentionally preserve JAX arrays
so they can be composed inside `jax.jit`, `jax.jacfwd`, and `jax.grad`. Use
`backend.to_numpy` when bringing one of those results back to plotting, file
I/O, or other ordinary NumPy code.
