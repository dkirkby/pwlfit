from typing import NamedTuple

import numpy as np

import pwlfit.grid


class GeneratedData(NamedTuple):
    x_data: np.ndarray
    y_data: np.ndarray
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
    x_data = np.linspace(xlo, xhi, ndata)
    grid = pwlfit.grid.Grid(x_data, ngrid)

    # Pick a random subset of interior grid points to be knots
    iknots = rng.choice(np.arange(1, ngrid - 1), nknots - 2, replace=False)
    iknots.sort()
    iknots = np.insert(iknots, 0, 0)
    iknots = np.append(iknots, ngrid - 1)

    xknots = grid.x_grid[iknots]
    yknots = rng.uniform(ylo, yhi, nknots)
    y_data = np.interp(x_data, xknots, yknots) + rng.normal(0, noise, ndata)
    ivar = np.full(ndata, noise ** -2)

    if missing_frac > 0:
        nmissing = int(ndata * missing_frac)
        missing_indices = rng.choice(np.arange(ndata), nmissing, replace=False)
        y_data[missing_indices] = np.nan
        ivar[missing_indices] = 0

    return GeneratedData(x_data=x_data, y_data=y_data, ivar=ivar,
                         iknots=iknots, xknots=xknots, yknots=yknots, grid=grid)
