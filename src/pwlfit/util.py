from typing import NamedTuple

import numpy as np

import pwlfit.grid


class GeneratedData(NamedTuple):
    xdata: np.ndarray
    ydata: np.ndarray
    ivar: np.ndarray
    iknots: np.ndarray
    xknots: np.ndarray
    yknots: np.ndarray
    grid: pwlfit.grid.Grid


def generate_data(ndata: int, ngrid: int, nknots: int,
                  noise: float = 0.01, missing_frac: float = 0,
                  xlo: float = 1, xhi: float = 2, ylo: float = -1, yhi: float = 1,
                  seed: int = 123):
    """
    Generate random data for testing piecewise linear fitting."
    """
    rng = np.random.default_rng(seed)

    # Generate the grid to use
    xdata = np.linspace(xlo, xhi, ndata)
    grid = pwlfit.grid.Grid(xdata, ngrid)

    # Pick a random subset of interior grid points to be knots
    iknots = rng.choice(np.arange(1, ngrid - 1), nknots - 2, replace=False)
    iknots.sort()
    iknots = np.insert(iknots, 0, 0)
    iknots = np.append(iknots, ngrid - 1)

    xknots = grid.xgrid[iknots]
    yknots = rng.uniform(ylo, yhi, nknots)
    ydata = np.interp(xdata, xknots, yknots) + rng.normal(0, noise, ndata)
    ivar = np.full(ndata, noise ** -2)

    if missing_frac > 0:
        nmissing = int(ndata * missing_frac)
        missing_indices = rng.choice(np.arange(ndata), nmissing, replace=False)
        ydata[missing_indices] = np.nan
        ivar[missing_indices] = 0

    return GeneratedData(xdata=xdata, ydata=ydata, ivar=ivar,
                         iknots=iknots, xknots=xknots, yknots=yknots, grid=grid)
