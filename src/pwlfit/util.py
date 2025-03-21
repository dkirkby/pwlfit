from typing import NamedTuple

import numpy as np

import pwlfit.grid
from plwfit.fit import Float64NDArray, Int64NDArray


class GeneratedData(NamedTuple):
    xdata: Float64NDArray
    ydata: Float64NDArray
    ivar: Int64NDArray
    iknots: Float64NDArray
    xknots: Float64NDArray
    yknots: Float64NDArray
    grid: pwlfit.grid.Grid


def generate_data(ndata: int, ngrid: int, nknots: int,
                  noise: float = 0.01, missing_frac: float = 0,
                  xlo: float = 1, xhi: float = 2, ylo: float = -1, yhi: float = 1,
                  transform: str = "identity", seed: int = 123):
    """
    Generate random data for testing piecewise linear fitting.
    """
    rng = np.random.default_rng(seed)

    # Generate the grid to use
    xdata = np.linspace(xlo, xhi, ndata)
    grid = pwlfit.grid.Grid(xdata, ngrid, transform=transform)

    # Pick a random subset of interior grid points to be knots
    iknots = rng.choice(np.arange(1, ngrid - 1), nknots - 2, replace=False)
    iknots.sort()
    iknots = np.insert(iknots, 0, 0)
    iknots = np.append(iknots, ngrid - 1)

    # Generate random y values at the knots
    xknots = grid.xgrid[iknots]
    yknots = rng.uniform(ylo, yhi, nknots)
    # Use linear interpolation (in the transformed x space) to get ydata
    ydata = np.interp(grid.sdata, grid.sgrid[iknots], yknots)

    if noise > 0:
        # Add Gaussian noise to the ydata and set ivar accordingly
        ydata += rng.normal(0, noise, ndata)
        ivar = np.full(ndata, noise ** -2)

    if missing_frac > 0:
        # Set a random fraction of the data to be missing: y=NaN, ivar=0
        nmissing = int(ndata * missing_frac)
        missing_indices = rng.choice(np.arange(ndata), nmissing, replace=False)
        ydata[missing_indices] = np.nan
        ivar[missing_indices] = 0

    return GeneratedData(xdata=xdata, ydata=ydata, ivar=ivar,
                         iknots=iknots, xknots=xknots, yknots=yknots, grid=grid)
