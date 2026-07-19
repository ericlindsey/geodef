# GeoDef benchmarks

An on-demand measurement harness (roadmap 3.4) — **not** part of the test
suite or the wheel. Every report records the problem definition, compile
vs steady-state timing, peak memory, backend, precision, library
versions, and a hardware stamp, so no speedup number ever circulates
without its qualifiers.

Run:

```bash
uv run python benchmarks/run.py                 # all seed problems
uv run python benchmarks/run.py --scale teaching
uv run python benchmarks/run.py --out results.json
```

Seed problems:

- `teaching`: ~80 patches x ~100 observations — the tutorial scale.
- `realistic`: ~2 000 patches x ~5 000 observations — a small real study.

Measured stages: Green's-matrix assembly (cold = first call, warm =
cache hit), `LinearSystem` preparation, and the regularized solve.
Add problems or stages by extending `PROBLEMS`/`measure` in `run.py`;
keep the metadata fields intact. Phase 4.2 uses these numbers to
document the scale boundary where users should downsample, switch to
operators, or move to JAX/GPU.
