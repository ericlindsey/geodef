# Compatibility and deprecation policy

This document defines what GeoDef users can rely on across releases. It is
part of the public contract: changes to this policy are themselves recorded
in [CHANGELOG.md](CHANGELOG.md).

## What is public API

- Everything importable from the top-level `geodef` namespace and listed in
  its `__all__`.
- Every function, class, method, and attribute documented in `docs/*.md`.
- Documented array conventions: East/North/Up axes, depth positive down,
  blocked slip vectors (`[:N]` strike-slip, `[N:]` dip-slip), row ordering of
  stacked observations, SI units with degree angles.

Names with a leading underscore, module internals not mentioned in the
documentation, and the exact text of error and warning messages are private
and may change at any time.

## Versioning

GeoDef versions are `MAJOR.MINOR.PATCH`:

- **Patch** releases fix bugs and documentation. They may change numerical
  output only to correct a demonstrated error, and any such change is tagged
  **numerical** in the changelog.
- **Minor** releases add functionality and may introduce deprecations, but
  do not remove or repurpose public names.
- **Major** releases may remove names whose deprecation period has elapsed.

## Deprecation

After the first public release, a public name or behavior is removed only
through this sequence:

1. A release notes the deprecation in the changelog, emits
   `DeprecationWarning` at the old call site, and documents the migration
   with a copy-pasteable example.
2. The old behavior keeps working for at least one minor release.
3. The removal lands in the next major release.

A name is never silently reused for a different meaning: if semantics must
change, the old name warns or errors rather than returning something new.

## Pre-release window (current status)

GeoDef has not yet had a public release. Until the first announced release,
cleanup identified by the design roadmap (`PLAN.md`) may land as direct
breaking changes without a deprecation period; every such break is recorded
in the changelog with its migration. If you are already using GeoDef in this
window, pin a git commit and read `CHANGELOG.md` when updating.

## Numerical stability

- Results carry documented physical conventions; a change that alters
  numerical output beyond documented test tolerances is tagged
  **numerical** in the changelog, with the reason (bug fix, algorithm
  change, dependency change).
- Caching never changes results: cached and uncached paths must agree, and
  cache keys include every input that affects the output.
- The NumPy backend is the reference implementation; the JAX backend is
  validated against it and documented divergences (e.g. float32 opt-in) are
  explicit.

## Dependencies

- `numpy`, `scipy`, and `matplotlib` are the only required dependencies.
- Optional stacks (JAX, blackjax, meshpy, netCDF4, pyproj, cartopy) are
  never imported at `import geodef` time and never required by the base
  install; features that need them raise a clear `ImportError` naming the
  extra to install.
- Supported Python and dependency ranges are declared in `pyproject.toml`
  and enforced in CI, including an oldest-supported-versions job.
