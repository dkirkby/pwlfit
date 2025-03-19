from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike

import pwlfit.grid

class FitResult(NamedTuple):
    iknots: np.ndarray
    xknots: np.ndarray
    yknots: np.ndarray
    xfit: np.ndarray = None
    yfit: np.ndarray = None
    chisq: np.ndarray = None


def fitFixedKnotsContinuous(y: ArrayLike, ivar: ArrayLike, iknots: ArrayLike,
                            grid: pwlfit.grid.Grid, fit: bool = False) -> FitResult:
    """
    Fit a continuous piecewise linear function to noisy data with fixed knots.
    The free parameters are the values of the piecewise linear function at the knots.

    Parameters:
    y (np.ndarray): The y values of the data to fit.
    ivar (np.ndarray): The inverse variance of the data (1/sigma^2).
    iknots (np.ndarray): The indices of the knots in the grid.
    grid (Grid): The grid object containing the x_data and s_data.
    fit (bool): If True, return the fitted values and chi-squared for each data point.
        Default is False.

    Returns:
    FitResult
    -----------
    A named tuple containing the results of the fit. The fields are:
    - iknots: The indices of the knots.
    - yknots: The fitted values at the knots.
    - xfit: The x values corresponding to the fitted data (if fit=True).
    - yfit: The fitted y values (if fit=True).
    - chisq: The chi-squared values for each data point (if fit=True).
    """
    n = len(iknots)
    Adiag = np.zeros(n)
    Aupper = np.zeros(n)
    # matrix is symmetric with Alower[i] = Aupper[i-1]
    b = np.zeros(n)
    k0 = grid.breaks[iknots[0]]
    ndata = grid.breaks[iknots[-1]] - k0
    tsave = np.zeros(ndata)
    # Loop over knots
    for i in range(n - 1):
        slo = grid.s_grid[iknots[i]]
        shi = grid.s_grid[iknots[i + 1]]
        k1, k2 = grid.breaks[iknots[i]], grid.breaks[iknots[i + 1]]
        t = (grid.s_data[k1:k2] - slo) / (shi - slo)
        wgt = ivar[k1:k2]
        wgtY = wgt * y[k1:k2]
        wgtY[wgt == 0] = 0 # y = Nan for ivar = 0 is ok
        Adiag[i] += np.sum(wgt * (1 - t)**2)
        Aupper[i] += np.sum(wgt * t * (1 - t))
        b[i] += np.sum(wgtY * (1 - t))
        Adiag[i+1] += np.sum(wgt * t**2)
        b[i+1] += np.sum(wgtY * t)
        tsave[k1-k0:k2-k0] = t
    # Solve the tridiagonal system using the Thomas algorithm.
    for i in range(1, n):
        Adiag[i] -= Aupper[i-1] ** 2 / Adiag[i-1]
        b[i] -= Aupper[i-1] * b[i-1] / Adiag[i-1]
    yknots = np.zeros(n)
    yknots[-1] = b[-1] / Adiag[-1]
    for i in range(n - 2, -1, -1):
        yknots[i] = (b[i] - Aupper[i] * yknots[i+1]) / Adiag[i]
    xknots = grid.x_grid[iknots]

    xfit, yfit, chisq = None, None, None
    if fit:
        # Calculate corresponding chisq for each data point covered by the knots
        xfit = np.empty(ndata)
        yfit = np.zeros(ndata)
        chisq = np.zeros(ndata)
        for i in range(n - 1):
            k1, k2 = grid.breaks[iknots[i]], grid.breaks[iknots[i + 1]]
            xfit[k1-k0:k2-k0] = grid.x_data[k1:k2]
            t = tsave[k1-k0:k2-k0]
            yfit[k1-k0:k2-k0] = yknots[i] + t * (yknots[i + 1] - yknots[i])
            wgt = ivar[k1:k2]
            chisq[k1-k0:k2-k0] = wgt * (y[k1:k2] - yfit[k1-k0:k2-k0]) ** 2
            chisq[k1-k0:k2-k0][wgt == 0] = 0

    return FitResult(iknots=iknots, xknots=xknots, yknots=yknots,
                     xfit=xfit, yfit=yfit, chisq=chisq)
