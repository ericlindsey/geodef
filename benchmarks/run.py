"""GeoDef benchmark harness (roadmap 3.4).

Times Green's assembly, system preparation, and the regularized solve on
declared seed problems, recording every qualifier needed to interpret
the numbers. See benchmarks/README.md.
"""

import argparse
import json
import platform
import time
import tracemalloc
from datetime import datetime, timezone

import numpy as np

import geodef
from geodef import backend, cache, greens
from geodef.invert import LinearSystem

PROBLEMS = {
    "smoke": {"n_length": 4, "n_width": 3, "n_stations": 20},
    "teaching": {"n_length": 10, "n_width": 8, "n_stations": 100},
    "realistic": {"n_length": 50, "n_width": 40, "n_stations": 5000},
}


def build_problem(n_length: int, n_width: int, n_stations: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    fault = geodef.Fault.planar(
        lat=0.0,
        lon=0.0,
        depth=15e3,
        strike=30.0,
        dip=20.0,
        length=100e3,
        width=60e3,
        n_length=n_length,
        n_width=n_width,
    )
    lat = rng.uniform(-1.0, 1.0, n_stations)
    lon = rng.uniform(-1.0, 1.0, n_stations)
    true = np.zeros(2 * fault.n_patches)
    true[fault.n_patches :] = 1.0
    ue, un, uz = fault.displacement(lat, lon, true[: fault.n_patches], 1.0)
    sigma = 0.002
    noise = rng.normal(0.0, sigma, (3, n_stations))
    gnss = geodef.data.gnss(
        lon=lon,
        lat=lat,
        east=ue + noise[0],
        north=un + noise[1],
        up=uz + noise[2],
        sigma_east=np.full(n_stations, sigma),
        sigma_north=np.full(n_stations, sigma),
        sigma_up=np.full(n_stations, sigma),
    )
    return fault, gnss


def timed(fn):
    tracemalloc.start()
    t0 = time.perf_counter()
    fn()
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {"seconds": elapsed, "peak_bytes": peak}


def measure(scale: str, spec: dict) -> dict:
    fault, gnss = build_problem(**spec)
    cache.disable()  # measure real assembly, not disk reads

    record: dict = {
        "scale": scale,
        "problem": {
            "n_patches": fault.n_patches,
            "n_params": 2 * fault.n_patches,
            "n_obs": gnss.n_obs,
            **spec,
        },
        "stages": {},
    }
    record["stages"]["greens_assembly_cold"] = timed(
        lambda: greens.matrix(fault, gnss)
    )
    cache.enable()
    greens.matrix(fault, gnss)  # populate the disk cache
    record["stages"]["greens_assembly_cached"] = timed(
        lambda: greens.matrix(fault, gnss)
    )

    holder: dict = {}

    def prepare():
        holder["system"] = LinearSystem(fault, [gnss], regularization="laplacian")
        holder["system"].GtWG  # force the cached products

    record["stages"]["system_preparation"] = timed(prepare)
    record["stages"]["regularized_solve"] = timed(
        lambda: holder["system"].invert(regularization_strength=1.0)
    )
    return record


def environment() -> dict:
    import scipy

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "geodef": geodef.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "python": platform.python_version(),
        "backend": backend.get_backend(),
        "precision": backend.get_precision(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "system": f"{platform.system()} {platform.release()}",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=sorted(PROBLEMS), default=None)
    parser.add_argument("--out", default=None, help="Write JSON here")
    args = parser.parse_args()

    import tempfile

    scratch = tempfile.mkdtemp(prefix="geodef-bench-cache-")
    cache.set_dir(scratch)

    scales = [args.scale] if args.scale else sorted(set(PROBLEMS) - {"smoke"})
    report = {"environment": environment(), "results": []}
    for scale in scales:
        print(f"measuring {scale} ...", flush=True)
        report["results"].append(measure(scale, PROBLEMS[scale]))

    text = json.dumps(report, indent=1)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text + "\n")
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
