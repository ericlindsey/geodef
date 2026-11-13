"""Elastic dislocation kernels: the published solutions GeoDef is built on.

- :mod:`okada85`: rectangular dislocation, surface displacement, tilt, and
  strain (Okada, 1985).
- :mod:`okada92`: rectangular dislocation, displacement and strain at depth
  (Okada, 1992; the DC3D routine).
- :mod:`tri`: triangular dislocation, displacement and strain (Nikkhoo &
  Walter, 2015).
- :mod:`okada`: one ``displacement`` function that picks ``okada85`` at the
  surface and ``okada92`` at depth.

These are line-by-line ports kept close to their sources so they can be
checked against the papers. They work in local Cartesian coordinates relative
to the source; :class:`geodef.Fault` handles geography, patches, and slip.
"""

from geodef.kernels import okada, okada85, okada92, tri

__all__ = ["okada", "okada85", "okada92", "tri"]
