"""Fault geometry and slip model classes.

FaultModel manages collections of rectangular fault patches with geographic
coordinates. SlipModel assigns slip distributions and computes forward models.
These are the existing implementations from shakeout_v2, migrated into the
geodef package. They will be redesigned in Phase 3.
"""

import logging

import numpy as np
from matplotlib.collections import LineCollection

from geodef import greens as _greens
from geodef import transforms

logger = logging.getLogger(__name__)


class FaultModel:
    """Collection of rectangular fault patches in geographic coordinates."""

    def __init__(self) -> None:
        self.latc = np.array([])
        self.lonc = np.array([])
        self.depth = np.array([])
        self.strike = np.array([])
        self.dip = np.array([])
        self.L = np.array([])
        self.W = np.array([])
        self.strikeid = np.array([])
        self.dipid = np.array([])
        self.npatches = 0

    def add_patch(
        self,
        latc: float | np.ndarray,
        lonc: float | np.ndarray,
        depth: float | np.ndarray,
        strike: float | np.ndarray,
        dip: float | np.ndarray,
        L: float | np.ndarray,
        W: float | np.ndarray,
        strikeid: float | np.ndarray,
        dipid: float | np.ndarray,
    ) -> None:
        """Add one or more patches to the fault model.

        Args:
            latc: Patch center latitude(s).
            lonc: Patch center longitude(s).
            depth: Patch centroid depth(s).
            strike: Strike angle(s) in degrees.
            dip: Dip angle(s) in degrees.
            L: Along-strike length(s).
            W: Down-dip width(s).
            strikeid: Along-strike index/indices.
            dipid: Down-dip index/indices.
        """
        self.latc = np.append(self.latc, latc)
        self.lonc = np.append(self.lonc, lonc)
        self.depth = np.append(self.depth, depth)
        self.strike = np.append(self.strike, strike)
        self.dip = np.append(self.dip, dip)
        self.L = np.append(self.L, L)
        self.W = np.append(self.W, W)
        self.strikeid = np.append(self.strikeid, strikeid)
        self.dipid = np.append(self.dipid, dipid)
        self.npatches = len(self.W)

    def load_patches_topleft(self, fname: str) -> None:
        """Load patches defined by top-left corner from a text file.

        Expected columns: [id, dipid, strikeid, lon, lat, depth, L, W, strike, dip].

        Args:
            fname: Path to the patch file.
        """
        filedata = np.loadtxt(fname, ndmin=2)

        dipid = filedata[:, 1]
        strikeid = filedata[:, 2]
        lon = filedata[:, 3]
        lat = filedata[:, 4]
        depth = filedata[:, 5]
        L = filedata[:, 6]
        W = filedata[:, 7]
        strike = filedata[:, 8]
        dip = filedata[:, 9]

        eoffset = (L / 2.0) * np.sin(np.radians(strike)) + (W / 2.0) * np.cos(np.radians(dip)) * np.cos(np.radians(strike))
        noffset = (L / 2.0) * np.cos(np.radians(strike)) + (W / 2.0) * np.cos(np.radians(dip)) * np.sin(np.radians(strike))
        uoffset = (W / 2.0) * np.sin(np.radians(dip))

        latc = []
        lonc = []
        depthc = []
        for i in range(len(lat)):
            latci, lonci, depthci = transforms.translate_flat(
                lat[i], lon[i], depth[i], eoffset[i], noffset[i], uoffset[i]
            )
            latc.append(latci)
            lonc.append(lonci)
            depthc.append(depthci)

        self.add_patch(latc, lonc, depthc, strike, dip, L, W, strikeid, dipid)

    def load_patches_center(self, fname: str) -> None:
        """Load patches defined by center coordinates from a text file.

        Expected columns: [id, dipid, strikeid, lon, lat, depth, L, W, strike, dip].

        Args:
            fname: Path to the patch file.
        """
        filedata = np.loadtxt(fname, ndmin=2)

        dipid = filedata[:, 1]
        strikeid = filedata[:, 2]
        lonc = filedata[:, 3]
        latc = filedata[:, 4]
        depth = filedata[:, 5]
        L = filedata[:, 6]
        W = filedata[:, 7]
        strike = filedata[:, 8]
        dip = filedata[:, 9]

        self.add_patch(latc, lonc, depth, strike, dip, L, W, strikeid, dipid)

    def load_patches_comsol(self, fname: str) -> None:
        """Load patches from COMSOL-format text file.

        Args:
            fname: Path to the patch file.
        """
        filedata = np.loadtxt(fname, ndmin=2)

        lonc = filedata[:, 1]
        latc = filedata[:, 2]
        depth = filedata[:, 5]
        L = filedata[:, 3]
        W = filedata[:, 4]
        strike = filedata[:, 6]
        dip = filedata[:, 7]
        dipid = filedata[:, 8]
        strikeid = filedata[:, 9]

        self.add_patch(latc, lonc, depth, strike, dip, L, W, strikeid, dipid)

    def create_planar_model(
        self,
        latcorner: float,
        loncorner: float,
        depthcorner: float,
        strike: float,
        dip: float,
        L: float,
        W: float,
        nL: int,
        nW: int,
    ) -> None:
        """Create a planar fault from its top-left corner.

        Args:
            latcorner: Latitude of top-left corner.
            loncorner: Longitude of top-left corner.
            depthcorner: Depth of top-left corner.
            strike: Strike angle in degrees.
            dip: Dip angle in degrees.
            L: Total along-strike length.
            W: Total down-dip width.
            nL: Number of patches along strike.
            nW: Number of patches down dip.
        """
        patchL = L / nL
        patchW = W / nW
        sindip = np.sin(np.radians(dip))
        cosdip = np.cos(np.radians(dip))
        sinstr = np.sin(np.radians(strike))
        cosstr = np.cos(np.radians(strike))
        for i in range(nL):
            for j in range(nW):
                tot_eoffset = (i + 0.5) * patchL * sinstr + (j + 0.5) * patchW * cosdip * cosstr
                tot_noffset = (i + 0.5) * patchL * cosstr - (j + 0.5) * patchW * cosdip * sinstr
                tot_uoffset = (j + 0.5) * patchW * sindip

                latcij, loncij, depthcij = transforms.translate_flat(
                    latcorner, loncorner, depthcorner,
                    tot_eoffset, tot_noffset, tot_uoffset,
                )
                self.add_patch(latcij, loncij, depthcij, strike, dip, patchL, patchW, i, j)

    def create_planar_model_centered(
        self,
        latc: float,
        lonc: float,
        depthc: float,
        strike: float,
        dip: float,
        L: float,
        W: float,
        nL: int,
        nW: int,
    ) -> None:
        """Create a planar fault from its center.

        Args:
            latc: Latitude of fault center.
            lonc: Longitude of fault center.
            depthc: Depth of fault center.
            strike: Strike angle in degrees.
            dip: Dip angle in degrees.
            L: Total along-strike length.
            W: Total down-dip width.
            nL: Number of patches along strike.
            nW: Number of patches down dip.
        """
        patchL = L / nL
        patchW = W / nW
        sindip = np.sin(np.radians(dip))
        cosdip = np.cos(np.radians(dip))
        sinstr = np.sin(np.radians(strike))
        cosstr = np.cos(np.radians(strike))

        fault_eoffset = -0.5 * L * sinstr - 0.5 * W * cosdip * cosstr
        fault_noffset = -0.5 * L * cosstr + 0.5 * W * cosdip * sinstr
        fault_uoffset = -0.5 * W * sindip
        for i in range(nL):
            for j in range(nW):
                tot_eoffset = fault_eoffset + (i + 0.5) * patchL * sinstr + (j + 0.5) * patchW * cosdip * cosstr
                tot_noffset = fault_noffset + (i + 0.5) * patchL * cosstr - (j + 0.5) * patchW * cosdip * sinstr
                tot_uoffset = fault_uoffset + (j + 0.5) * patchW * sindip

                latcij, loncij, depthcij = transforms.translate_flat(
                    latc, lonc, depthc,
                    tot_eoffset, tot_noffset, tot_uoffset,
                )
                self.add_patch(latcij, loncij, depthcij, strike, dip, patchL, patchW, i, j)

    def find_patch(self, patchid: int, strikeoffset: int, dipoffset: int) -> int:
        """Find the index of a neighboring patch.

        Args:
            patchid: Index of the reference patch.
            strikeoffset: Offset in along-strike direction.
            dipoffset: Offset in down-dip direction.

        Returns:
            Index of the target patch. Clamps to fault edges.
        """
        strid = min(max(self.strikeid), max(0, self.strikeid[patchid] + strikeoffset))
        did = min(max(self.dipid), max(0, self.dipid[patchid] + dipoffset))
        try:
            return np.logical_and(self.strikeid == strid, self.dipid == did).nonzero()[0][0]
        except IndexError:
            logger.error("find_patch: element not found")
            return patchid

    def get_greens(self, lat: np.ndarray, lon: np.ndarray, kind: str = 'displacement') -> np.ndarray:
        """Compute Green's function matrix at observation points.

        Args:
            lat: Observation latitudes.
            lon: Observation longitudes.
            kind: 'displacement' or 'strain'.

        Returns:
            Green's matrix G.

        Raises:
            ValueError: If kind is not recognized.
        """
        if kind == 'displacement':
            return _greens.displacement_greens(
                lat, lon, self.latc, self.lonc, self.depth,
                self.strike, self.dip, self.L, self.W,
            )
        elif kind == 'strain':
            return _greens.strain_greens(
                lat, lon, self.latc, self.lonc, self.depth,
                self.strike, self.dip, self.L, self.W,
            )
        raise ValueError(f"Unknown kind: {kind!r}. Use 'displacement' or 'strain'.")

    def get_selfstress(self, mu: float = 30e3) -> np.ndarray:
        """Compute the stress interaction kernel for the fault.

        Args:
            mu: Shear modulus (default 30 GPa in kPa units).

        Returns:
            Stress kernel matrix.
        """
        K = _greens.strain_greens(
            self.latc, self.lonc, self.latc, self.lonc,
            self.depth, self.strike, self.dip, self.L, self.W,
        )
        return mu * K

    def load_pickle(self, fname: str) -> None:
        """Load fault model from a numpy .npy file.

        Args:
            fname: Path to the .npy file.
        """
        savearray = np.load(fname)
        self.latc = np.append(self.latc, savearray[:, 0])
        self.lonc = np.append(self.lonc, savearray[:, 1])
        self.depth = np.append(self.depth, savearray[:, 2])
        self.strike = np.append(self.strike, savearray[:, 3])
        self.dip = np.append(self.dip, savearray[:, 4])
        self.L = np.append(self.L, savearray[:, 5])
        self.W = np.append(self.W, savearray[:, 6])
        self.strikeid = np.append(self.strikeid, savearray[:, 7])
        self.dipid = np.append(self.dipid, savearray[:, 8])
        self.npatches = len(self.W)

    def save_pickle(self, fname: str) -> None:
        """Save fault model to a numpy .npy file.

        Args:
            fname: Path for the output .npy file.
        """
        savearray = np.column_stack((
            self.latc, self.lonc, self.depth, self.strike, self.dip,
            self.L, self.W, self.strikeid, self.dipid,
        ))
        np.save(fname, savearray)

    def get_patch_verts_center_3d(self) -> list:
        """Get 3-D patch vertices (lon, lat, depth_km).

        Returns:
            List of 4-vertex polygon tuples.
        """
        verts3d, _ = self._get_patch_verts_center_both()
        return verts3d

    def get_patch_verts_center_2d(self) -> list:
        """Get 2-D patch vertices (lon, lat).

        Returns:
            List of 4-vertex polygon tuples.
        """
        _, verts2d = self._get_patch_verts_center_both()
        return verts2d

    def _get_patch_verts_center_both(self) -> tuple[list, list]:
        """Compute both 2-D and 3-D patch vertices.

        Returns:
            Tuple (verts3d, verts2d).
        """
        verts3d = []
        verts2d = []
        for i in range(len(self.latc)):
            sindip = np.sin(np.radians(self.dip[i]))
            cosdip = np.cos(np.radians(self.dip[i]))
            sinstr = np.sin(np.radians(self.strike[i]))
            cosstr = np.cos(np.radians(self.strike[i]))
            ztop = 1.0e-3 * (self.depth[i] + (self.W[i] / 2.0) * sindip)
            zbot = 1.0e-3 * (self.depth[i] - (self.W[i] / 2.0) * sindip)
            zs = [ztop, ztop, zbot, zbot]

            offsets = [
                (-(self.L[i] / 2.0) * sinstr + (self.W[i] / 2.0) * cosdip * cosstr,
                 -(self.L[i] / 2.0) * cosstr - (self.W[i] / 2.0) * cosdip * sinstr),
                (+(self.L[i] / 2.0) * sinstr + (self.W[i] / 2.0) * cosdip * cosstr,
                 +(self.L[i] / 2.0) * cosstr - (self.W[i] / 2.0) * cosdip * sinstr),
                (+(self.L[i] / 2.0) * sinstr - (self.W[i] / 2.0) * cosdip * cosstr,
                 +(self.L[i] / 2.0) * cosstr + (self.W[i] / 2.0) * cosdip * sinstr),
                (-(self.L[i] / 2.0) * sinstr - (self.W[i] / 2.0) * cosdip * cosstr,
                 -(self.L[i] / 2.0) * cosstr + (self.W[i] / 2.0) * cosdip * sinstr),
            ]
            xs, ys = [], []
            for eoff, noff in offsets:
                y, x, _ = transforms.translate_flat(
                    self.latc[i], self.lonc[i], 0, eoff, noff, 0,
                )
                xs.append(x)
                ys.append(y)

            verts3d.append(list(zip(xs, ys, zs)))
            verts2d.append(list(zip(xs, ys)))
        return verts3d, verts2d

    def plot_patch_outlines(
        self,
        ax: object | None = None,
        indices: list[int] | None = None,
        outline_color: str = 'k',
        outline_lw: float = 1.2,
        outline_ls: str = '-',
        top_color: str = 'tab:red',
        top_lw: float = 2.5,
        top_ls: str = '-',
        **kwargs: object,
    ) -> tuple[LineCollection, LineCollection]:
        """Draw rectangular patch outlines, emphasizing the shallow (top) edge.

        Args:
            ax: Matplotlib axes (default: current axes).
            indices: Which patches to draw (default: all).
            outline_color: Color for non-top edges.
            outline_lw: Line width for non-top edges.
            outline_ls: Line style for non-top edges.
            top_color: Color for top edges.
            top_lw: Line width for top edges.
            top_ls: Line style for top edges.
            **kwargs: Passed to LineCollection.

        Returns:
            Tuple (outline_lc, top_lc) of LineCollection objects.
        """
        import matplotlib.pyplot as plt
        if ax is None:
            ax = plt.gca()

        _, verts2d = self._get_patch_verts_center_both()
        verts3d = self.get_patch_verts_center_3d()
        if indices is None:
            indices = range(len(verts2d))

        outline_segments = []
        top_segments = []

        for i in indices:
            v2 = verts2d[i]
            v3 = verts3d[i]

            zs = np.array([p[2] for p in v3])
            zmin = zs.min()
            top_idx = np.where(np.isclose(zs, zmin))[0]

            edges = [
                np.array([v2[0], v2[1]]),
                np.array([v2[1], v2[2]]),
                np.array([v2[2], v2[3]]),
                np.array([v2[3], v2[0]]),
            ]

            top_edge = None
            if set(top_idx.tolist()) == {0, 1}:
                top_edge = edges[0]
            elif set(top_idx.tolist()) == {1, 2}:
                top_edge = edges[1]
            elif set(top_idx.tolist()) == {2, 3}:
                top_edge = edges[2]
            elif set(top_idx.tolist()) == {3, 0}:
                top_edge = edges[3]
            else:
                lens = [np.hypot(*(e[1] - e[0])) for e in edges]
                top_edge = edges[int(np.argmin(lens))]

            top_segments.append(top_edge)
            for e in edges:
                if not np.allclose(e, top_edge):
                    outline_segments.append(e)

        outline_kwargs = dict(colors=outline_color, linewidths=outline_lw, linestyles=outline_ls)
        outline_kwargs.update({k: v for k, v in kwargs.items() if k not in outline_kwargs})
        outline_lc = LineCollection(outline_segments, **outline_kwargs)
        ax.add_collection(outline_lc)

        top_kwargs = dict(colors=top_color, linewidths=top_lw, linestyles=top_ls)
        top_kwargs.update({k: v for k, v in kwargs.items() if k not in top_kwargs})
        top_lc = LineCollection(top_segments, **top_kwargs)
        ax.add_collection(top_lc)

        return outline_lc, top_lc


class SlipModel:
    """Slip distribution on a FaultModel, with forward modeling."""

    def __init__(self) -> None:
        self.F: FaultModel | None = None
        self.slip = np.array([])
        self.rake = np.array([])
        self.risetime = np.array([])
        self.onsettime = np.array([])

    def set_fault_model(self, F: FaultModel) -> None:
        """Attach a FaultModel to this slip model.

        Args:
            F: FaultModel instance.
        """
        self.F = F

    def random_slip(
        self,
        bound_cond: list[int],
        magnitude: float,
        rake: float,
        risetime: float,
        hypolon: float,
        hypolat: float,
        hypodepth: float,
        rupvel: float,
    ) -> None:
        """Generate a random slip distribution scaled to a target magnitude.

        Args:
            bound_cond: 4-element list of boundary conditions (0=zero slip).
            magnitude: Target moment magnitude.
            rake: Rake angle in degrees.
            risetime: Rise time in seconds.
            hypolon: Hypocenter longitude.
            hypolat: Hypocenter latitude.
            hypodepth: Hypocenter depth.
            rupvel: Rupture velocity.
        """
        self.slip = np.random.rand(self.F.npatches)
        self.set_bc_slip(bound_cond)
        self.scale_to_magnitude(magnitude)
        self.rake = rake * np.ones(self.F.npatches)
        self.risetime = risetime * np.ones(self.F.npatches)
        self.onsettime = self.slip_time(hypolat, hypolon, hypodepth, rupvel)

    def smooth_slip(
        self,
        bound_cond: list[int],
        magnitude: float,
        smoothing_iter: int,
    ) -> None:
        """Smooth existing slip by iterative neighbor averaging.

        Args:
            bound_cond: 4-element list of boundary conditions.
            magnitude: Target moment magnitude after smoothing.
            smoothing_iter: Number of smoothing iterations.
        """
        for _ in range(smoothing_iter):
            for i in range(self.F.npatches):
                rti = self.F.find_patch(i, 1, 0)
                lti = self.F.find_patch(i, -1, 0)
                upi = self.F.find_patch(i, 0, -1)
                dni = self.F.find_patch(i, 0, 1)
                self.slip[i] = (
                    2 * self.slip[i]
                    + self.slip[rti] + self.slip[lti]
                    + self.slip[upi] + self.slip[dni]
                ) / 6
            self.slip = self.slip - min(self.slip)
            self.set_bc_slip(bound_cond)
        self.scale_to_magnitude(magnitude)

    def load_slip_timedep(self, fname: str) -> None:
        """Load time-dependent slip from file.

        Args:
            fname: Path to the slip file.
        """
        self.F.load_patches_center(fname)
        indata = np.loadtxt(fname, ndmin=2)
        self.rake = indata[:, 10]
        self.slip = indata[:, 11]
        self.risetime = indata[:, 12]
        self.onsettime = indata[:, 13]

    def load_slip_static(self, fname: str) -> None:
        """Load static slip from file.

        Args:
            fname: Path to the slip file.
        """
        self.F.load_patches_center(fname)
        indata = np.loadtxt(fname, ndmin=2)
        self.rake = indata[:, 10]
        self.slip = indata[:, 11]

    def save_slip(self, fname: str) -> None:
        """Save slip model to text file.

        Args:
            fname: Path for the output file.
        """
        outdata = np.column_stack((
            range(self.F.npatches), self.F.dipid, self.F.strikeid,
            self.F.lonc, self.F.latc, self.F.depth, self.F.L, self.F.W,
            self.F.strike, self.F.dip, self.rake, self.slip,
            self.risetime, self.onsettime,
        ))
        np.savetxt(fname, outdata, fmt='%10.5f')

    def set_bc_slip(self, bound_cond: list[int]) -> None:
        """Apply boundary conditions (zero slip at edges).

        Args:
            bound_cond: 4-element list [top, right, bottom, left].
                0 means zero slip on that edge.
        """
        if bound_cond[0] == 0:
            self.slip[self.F.dipid == min(self.F.dipid)] = 0
        if bound_cond[1] == 0:
            self.slip[self.F.strikeid == max(self.F.strikeid)] = 0
        if bound_cond[2] == 0:
            self.slip[self.F.dipid == max(self.F.dipid)] = 0
        if bound_cond[3] == 0:
            self.slip[self.F.strikeid == min(self.F.strikeid)] = 0

    def scale_to_magnitude(self, magnitude: float) -> None:
        """Rescale slip to achieve a target moment magnitude.

        Args:
            magnitude: Target Mw.
        """
        rescale = self.get_moment_from_mag(magnitude) / self.get_moment()
        self.slip = rescale * self.slip

    def get_magnitude(self) -> float:
        """Compute moment magnitude from current slip.

        Returns:
            Moment magnitude Mw.
        """
        mo = self.get_moment()
        return (2.0 / 3.0) * np.log10(mo) - 10.7

    def get_moment(self) -> float:
        """Compute scalar seismic moment from current slip.

        Returns:
            Seismic moment in N-m (assumes mu=30 GPa and dimensions in meters).
        """
        return 1e7 * 30e9 * np.sum(self.slip * self.F.L * self.F.W)

    def get_moment_from_mag(self, mag: float) -> float:
        """Convert moment magnitude to seismic moment.

        Args:
            mag: Moment magnitude Mw.

        Returns:
            Seismic moment in N-m.
        """
        return 10 ** ((3.0 / 2.0) * (mag + 10.7))

    def slip_time(
        self, hypolat: float, hypolon: float, hypodepth: float, rupvel: float,
    ) -> np.ndarray:
        """Compute onset times from rupture propagation.

        Args:
            hypolat: Hypocenter latitude.
            hypolon: Hypocenter longitude.
            hypodepth: Hypocenter depth.
            rupvel: Rupture velocity.

        Returns:
            Array of onset times for each patch.
        """
        patchdist = np.sqrt(
            (self.F.depth - hypodepth) ** 2
            + transforms.haversine(hypolat, hypolon, self.F.latc, self.F.lonc) ** 2
        )
        return patchdist / rupvel

    def forward_model_static(
        self, latobs: np.ndarray, lonobs: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute static displacements from the current slip model.

        Args:
            latobs: Observation latitudes.
            lonobs: Observation longitudes.

        Returns:
            Tuple (edisp, ndisp, udisp) displacement arrays.
        """
        G = self.F.get_greens(latobs, lonobs)

        nobs = len(latobs)
        edisp = np.zeros(nobs)
        ndisp = np.zeros(nobs)
        udisp = np.zeros(nobs)

        for i in range(nobs):
            for j in range(self.F.npatches):
                sinrake = np.sin(np.radians(self.rake[j]))
                cosrake = np.cos(np.radians(self.rake[j]))
                edisp[i] += self.slip[j] * (cosrake * G[3 * i, 2 * j] + sinrake * G[3 * i, 2 * j + 1])
                ndisp[i] += self.slip[j] * (cosrake * G[3 * i + 1, 2 * j] + sinrake * G[3 * i + 1, 2 * j + 1])
                udisp[i] += self.slip[j] * (cosrake * G[3 * i + 2, 2 * j] + sinrake * G[3 * i + 2, 2 * j + 1])

        return edisp, ndisp, udisp
