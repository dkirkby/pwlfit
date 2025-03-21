from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

import pwlfit.grid
from pwlfit.fit import Float64NDArray, Int64NDArray


class GeneratedData(NamedTuple):
    xdata: Float64NDArray
    ydata: Float64NDArray
    ivar: Int64NDArray
    iknots: Float64NDArray
    xknots: Float64NDArray
    yknots: Float64NDArray
    y1knots: Float64NDArray
    y2knots: Float64NDArray
    grid: pwlfit.grid.Grid


def generate_data(ndata: int, ngrid: int, nknots: int,
                  noise: float = 0.01, missing_frac: float = 0,
                  xlo: float = 1, xhi: float = 2, ylo: float = -1, yhi: float = 1,
                  continuous: bool = True, transform: str = "identity", seed: int = 123):
    """
    Generate random data for testing piecewise linear fitting.

    Parameters
    ----------
    ndata : int
        Number of data points to generate.
    ngrid : int
        Number of equally spaced grid points to use for possible knot locations.
    nknots : int
        Number of knots to use for the true piecewise linear model.
    noise : float, optional
        Standard deviation of Gaussian noise to add to the data (default is 0.01).
    missing_frac : float, optional
        Fraction of data points to set as missing (default is 0). Missing data points
        have ydata=NaN and ivar=0.
    xlo : float, optional
        Lower bound for the x data (default is 1).
    xhi : float, optional
        Upper bound for the x data (default is 2).
    ylo : float, optional
        Lower bound for the y data (default is -1).
    yhi : float, optional
        Upper bound for the y data (default is 1).
    continuous : bool, optional
        If True, use a continuous piecewise linear model to generate data.
        If False, use a discontinuous piecewise linear model (default is True).
    transform : str, optional
        Transformation to apply to the x data before generating the grid.
        Options are 'identity' (default) or 'log'.
    seed : int, optional
        Random seed for reproducibility (default is 123).

    Returns
    -------
    GeneratedData
        A NamedTuple containing the generated data:
        - xdata: x values of the data points
        - ydata: y values of the data points
        - ivar: inverse variance of the data points (1/noise^2)
        - iknots: indices of the knots used in the piecewise linear model
        - xknots: x values of the knots
        - yknots: y values of the knots (None if discontinuous)
        - y1knots: y values at the left ends of the intervals between knots
        - y2knots: y values at the right ends of the intervals between knots
        - grid: Grid object containing information about the grid used
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
    if continuous:
        yknots = rng.uniform(ylo, yhi, nknots)
        y1knots = yknots[:-1]
        y2knots = yknots[1:]
        ydata = np.interp(grid.sdata, grid.sgrid[iknots], yknots)
    else:
        y1knots = rng.uniform(ylo, yhi, nknots - 1)
        y2knots = rng.uniform(ylo, yhi, nknots - 1)
        yknots = None
        ydata = np.empty(ndata)
        for i in range(nknots - 1):
            k1, k2 = grid.breaks[iknots[i]], grid.breaks[iknots[i + 1]]
            ydata[k1:k2] = np.interp(grid.sdata[k1:k2], grid.sgrid[iknots[i:i+2]], [y1knots[i], y2knots[i]])

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

    return GeneratedData(xdata=xdata, ydata=ydata, ivar=ivar, iknots=iknots, xknots=xknots,
                         yknots=yknots, y1knots=y1knots, y2knots=y2knots, grid=grid)


def smooth_weighted_data(y: ArrayLike, ivar: ArrayLike, iknots: ArrayLike, grid: pwlfit.grid.Grid,
                         window_size: int, poly_order: int = 3, transformed: bool = True) -> Float64NDArray:
    """Smooth the data using a weighted Savitsky-Golay polynomial fit around each knot.

    Parameters
    ----------
    y : ArrayLike
        The y values of the data points. Must have the same length as grid.xdata.
        Values where ivar==0 are ignored.
    ivar : ArrayLike
        The inverse variance of the data points. Must have the same length as grid.xdata.
    iknots : ArrayLike
        The indices of the knots in the grid where smoothed values are required.
    grid : pwlfit.grid.Grid
        The grid object containing information about the grid to use.
    window_size : int
        The size of the smoothing window. Must be an odd integer. The size is in index space,
        not the physical (x or s) grid space, i.e. it determines how many data points close
        to each knot are used in the polynomial fit.
    poly_order : int
        The order of the polynomial to fit within the smoothing window. Default is 3.
    transformed : bool
        If True, the smoothing is done in the grid's transformed space (sdata).
        If False, it is done in the grid's original space (xdata). Default is True.
        This has no effect for the default grid transformation (identity).

    Returns
    -------
    Float64NDArray
        An array of smoothed values at the knot locations. Has the same length as iknots.
        If all data points within the window for a knot are invalid (ivar==0), the smoothed value is NaN.
    """
    if not window_size % 2 == 1:
        raise ValueError("window_size must be an odd integer")
    if not len(y) == len(grid.xdata):
        raise ValueError("y must have the same length as grid.xdata")
    if not len(ivar) == len(grid.xdata):
        raise ValueError("ivar must have the same length as grid.xdata")

    half_window = window_size // 2
    n = len(iknots)
    ysmooth = np.full(n, np.nan)
    for i, iknot in enumerate(iknots):
        k0 = grid.breaks[iknot]
        k1 = max(0, k0 - half_window)
        k2 = min(len(y), k0 + half_window + 1)

        if transformed:
            x_window = grid.sdata[k1:k2] - grid.sgrid[iknot]
        else:
            x_window = grid.xdata[k1:k2] - grid.xgrid[iknot]
        y_window = np.array(y[k1:k2]) # copy to avoid modifying original when ivar==0
        w_window = ivar[k1:k2]
        if np.all(w_window <= 0):
            # No valid data in this window, so smoothed value is NaN
            continue

        # build the weighted design (Vandermonde) matrix
        # rows are [ sqrt(w) * 1, sqrt(w) * x, sqrt(w) * x**2, ... ]
        X = np.vander(x_window, poly_order + 1, increasing=True)

        # Form the diagonal weight matrix
        W = np.diag(w_window)

        # Zero any NaN values in y_window when ivar == 0
        y_window[w_window == 0] = 0

        # Solve the weighted least squares problem:
        # (X^T W X) a = X^T W y  =>  a = (X^T W X)^{-1} (X^T W y)
        try:
            coeffs = np.linalg.pinv(X.T @ W @ X) @ (X.T @ W @ y_window)
        except np.linalg.LinAlgError:
            Wsqrt = np.sqrt(w_window)
            coeffs = np.linalg.lstsq(X * Wsqrt[:, None], y_window * Wsqrt, rcond=None)[0]

        # For an "increasing" Vandermonde matrix, evaluating at x=0 gives: p(0) = coeffs[0]
        ysmooth[i] = coeffs[0]

    return ysmooth
