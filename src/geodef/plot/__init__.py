"""Visualization functions for geodef.

Every public function follows a consistent pattern:

- Accepts an optional ``ax`` parameter; creates a new figure if ``None``.
- Returns the ``matplotlib.axes.Axes`` used for plotting.
- Never calls ``plt.show()``.
- Passes ``**kwargs`` through to the underlying matplotlib artist.
"""

from geodef.plot._assessment_plots import (
    resolution as resolution,
)
from geodef.plot._assessment_plots import (
    uncertainty as uncertainty,
)
from geodef.plot._data_plots import (
    insar as insar,
)
from geodef.plot._data_plots import (
    map_view as map_view,
)
from geodef.plot._data_plots import (
    vectors as vectors,
)
from geodef.plot._fault_plots import (
    fault3d as fault3d,
)
from geodef.plot._fault_plots import (
    patches as patches,
)
from geodef.plot._fault_plots import (
    slip as slip,
)
from geodef.plot._fault_plots import (
    slip_interpolated as slip_interpolated,
)
from geodef.plot._fit_plots import (
    diagnostics as diagnostics,
)
from geodef.plot._fit_plots import (
    fit as fit,
)
from geodef.plot._fit_plots import (
    prediction as prediction,
)
from geodef.plot._fit_plots import (
    residual as residual,
)
from geodef.plot._fit_plots import (
    summary as summary,
)



# ======================================================================
# Internal helpers
# ======================================================================


# ======================================================================
# Public API — Fault patch plots
# ======================================================================


# ======================================================================
# Public API — Data observation plots
# ======================================================================


# ======================================================================
# Public API — Fault geometry visualization
# ======================================================================


