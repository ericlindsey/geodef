"""Benchmark NumPy vs JAX Green's-matrix assembly.

Times ``geodef.greens.displacement_greens`` (rectangular Okada patches,
surface observations) on the NumPy backend and, when JAX is installed, on
the JAX backend, across a range of patch and observation counts.

Methodology:
- ``displacement_greens`` is called directly, so the disk cache is never
  involved; ``geodef.cache`` is disabled anyway for belt and braces.
- The JAX path is JIT-compiled; the first call at each problem size pays
  the compilation cost and is reported separately from the steady-state
  time (the compile is paid once per problem shape, the steady state is
  what repeated assemblies during hyperparameter sweeps or nonlinear
  geometry searches would see).
- Backends run sequentially, never in parallel. Note that XLA may use
  multiple CPU threads by default while the NumPy kernels here are
  single-threaded elementwise math; on GPU machines the JAX numbers
  reflect the accelerator.

Usage::

    uv run python benchmarks/bench_greens.py
    uv run python benchmarks/bench_greens.py --npatch 100 400 --nobs 500 --repeats 5
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import geodef
from geodef import backend, greens


def make_problem(
    npatch: int, nobs: int, seed: int = 0
) -> tuple[np.ndarray, ...]:
    """Build a synthetic dipping-fault grid and surface observation set.

    Args:
        npatch: Number of rectangular fault patches.
        nobs: Number of surface observation points.
        seed: RNG seed for reproducible geometry.

    Returns:
        Argument tuple for ``greens.displacement_greens``.
    """
    rng = np.random.default_rng(seed)
    lat = rng.uniform(-0.5, 0.5, nobs)
    lon = rng.uniform(-0.5, 0.5, nobs)
    lat0 = rng.uniform(-0.25, 0.25, npatch)
    lon0 = rng.uniform(-0.25, 0.25, npatch)
    depth = rng.uniform(5e3, 3e4, npatch)
    strike = rng.uniform(0.0, 360.0, npatch)
    dip = rng.uniform(10.0, 90.0, npatch)
    L = np.full(npatch, 5e3)
    W = np.full(npatch, 5e3)
    return lat, lon, lat0, lon0, depth, strike, dip, L, W


def time_assembly(args: tuple[np.ndarray, ...], repeats: int) -> tuple[float, float]:
    """Time displacement_greens on the active backend.

    Args:
        args: Problem arguments from ``make_problem``.
        repeats: Number of timed steady-state calls.

    Returns:
        Tuple of (first-call seconds, best steady-state seconds). On JAX
        the first call includes JIT compilation.
    """
    t0 = time.perf_counter()
    greens.displacement_greens(*args)
    first = time.perf_counter() - t0

    best = np.inf
    for _ in range(repeats):
        t0 = time.perf_counter()
        greens.displacement_greens(*args)
        best = min(best, time.perf_counter() - t0)
    return first, best


def main() -> None:
    """Run the benchmark sweep and print a comparison table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--npatch", type=int, nargs="+", default=[50, 200, 800],
        help="Patch counts to sweep",
    )
    parser.add_argument(
        "--nobs", type=int, nargs="+", default=[200, 1000],
        help="Observation counts to sweep",
    )
    parser.add_argument(
        "--repeats", type=int, default=3, help="Timed repeats per case"
    )
    cli = parser.parse_args()

    geodef.cache.disable()

    try:
        import jax

        have_jax = True
        devices = ", ".join(str(d) for d in jax.devices())
    except ImportError:
        have_jax = False
        devices = "n/a (jax not installed)"

    print(f"JAX devices: {devices}")
    header = (
        f"{'npatch':>7} {'nobs':>6} {'numpy (s)':>10} "
        f"{'jax 1st (s)':>12} {'jax best (s)':>13} {'speedup':>8}"
    )
    print(header)
    print("-" * len(header))

    for npatch in cli.npatch:
        for nobs in cli.nobs:
            args = make_problem(npatch, nobs)

            backend.set_backend("numpy")
            _, t_np = time_assembly(args, cli.repeats)

            if have_jax:
                backend.set_backend("jax")
                t_first, t_jax = time_assembly(args, cli.repeats)
                backend.set_backend("numpy")
                print(
                    f"{npatch:>7} {nobs:>6} {t_np:>10.4f} "
                    f"{t_first:>12.4f} {t_jax:>13.4f} {t_np / t_jax:>7.1f}x"
                )
            else:
                print(
                    f"{npatch:>7} {nobs:>6} {t_np:>10.4f} "
                    f"{'-':>12} {'-':>13} {'-':>8}"
                )


if __name__ == "__main__":
    main()
