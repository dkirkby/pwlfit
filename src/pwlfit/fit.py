from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike

import pwlfit.grid

class FitResult(NamedTuple):
    iknots: np.ndarray
    xknots: np.ndarray
    yknots: np.ndarray
    y1knots: np.ndarray
    y2knots: np.ndarray
    xfit: np.ndarray = None
    yfit: np.ndarray = None
    chisq: np.ndarray = None


class CummulativeSums(NamedTuple):
    Sw: np.ndarray
    Sx: np.ndarray
    Sy: np.ndarray
    Sxx: np.ndarray
    Sxy: np.ndarray
    Syy: np.ndarray


def calculateCumulativeSums(y: ArrayLike, ivar: ArrayLike, iknots: ArrayLike,
                            grid: pwlfit.grid.Grid) -> CummulativeSums:
    """
    Calculate cumulative sums for the piecewise linear fit.

    Parameters:
    y (np.ndarray): The y values of the data to fit. Ignored when corresponding ivar=0.
    ivar (np.ndarray): The inverse variance of the data (1/sigma^2).
    iknots (np.ndarray): The indices of the knots in the grid.
    grid (Grid): The grid object containing the xdata and sdata.

    Returns:
    CummulativeSums
    """
    n = len(iknots)
    S = np.zeros((6, n))
    for i in range(1, n):
        k1, k2 = grid.breaks[iknots[i-1]], grid.breaks[iknots[i]]
        W = ivar[k1:k2]
        X = grid.xdata[k1:k2]
        Y = y[k1:k2]
        WY = W * y[k1:k2]
        WY[W == 0] = 0  # Handle NaN values for ivar = 0
        WYY = WY * Y
        WYY[W == 0] = 0  # Handle NaN values for ivar = 0
        S[0, i] = np.sum(W)  # Sw
        S[1, i] = np.sum(W * X)  # Sx
        S[2, i] = np.sum(WY)  # Sy
        S[3, i] = np.sum(W * X**2)  # Sxx
        S[4, i] = np.sum(WY * X)  # Sxy
        S[5, i] = np.sum(WYY)  # Syy
    # Cumulative sums
    S = np.cumsum(S, axis=1)
    return CummulativeSums(Sw=S[0], Sx=S[1], Sy=S[2], Sxx=S[3], Sxy=S[4], Syy=S[5])


class SegmentFit(NamedTuple):
    a: float
    b: float
    chisq: float


def segmentFit(i1: int, i2: int, cumSums: CummulativeSums, eps: float = 1e-10) -> NamedTuple:
    """
    Fit data between knots i1 and i2 to a linear function a + b * s.

    Parameters:
    i1 (int): The index of the first knot.
    i2 (int): The index of the second knot.
    cumSums (CummulativeSums): The cumulative sums of the data.
    eps (float): A small value to avoid division by zero.

    Returns:
    SegmentFit
    """
    # Calculate cumulative sums for this segment
    Sw = cumSums.Sw[i2] - cumSums.Sw[i1]
    Sx = cumSums.Sx[i2] - cumSums.Sx[i1]
    Sy = cumSums.Sy[i2] - cumSums.Sy[i1]
    Sxx = cumSums.Sxx[i2] - cumSums.Sxx[i1]
    Sxy = cumSums.Sxy[i2] - cumSums.Sxy[i1]
    Syy = cumSums.Syy[i2] - cumSums.Syy[i1]
    # Calculate linear coefficients (a,b) and chisq for this segment
    denom = Sw * Sxx - Sx ** 2
    if denom > eps:
        b = (Sw * Sxy - Sx * Sy) / denom
        a = (Sy - b * Sx) / Sw
        chisq = Syy - a * Sy - b * Sxy
    else:
        a = Sy / Sw
        b = 0
        chisq = Syy - a * Sy
    return SegmentFit(a=a, b=b, chisq=chisq)


def fitPrunedKnotsDiscontinuous(y: ArrayLike, ivar: ArrayLike, iknots: ArrayLike,
                                grid: pwlfit.grid.Grid, mu: float = 2,
                                fit: bool = False) -> FitResult:
    """
    Fit a discontinuous piecewise linear function to noisy data with pruned knots.
    The free parameters are the parameters (a,b) of a linear fit a + b*x for each
    segment between the knots.

    Parameters:
    y (np.ndarray): The y values of the data to fit. Ignored when corresponding ivar=0.
    ivar (np.ndarray): The inverse variance of the data (1/sigma^2).
    iknots (np.ndarray): The indices of the knots in the grid.
    grid (Grid): The grid object containing the xdata and sdata.
    mu (float): A penalty term for the number of knots. Larger values will favor fewer knots.
    fit (bool): If True, return the fitted values and chi-squared for each data point.
        Default is False.

    Returns:
    FitResult
    -----------
    A named tuple containing the results of the fit. The fields are:
    - iknots: The indices of the knots.
    - xknots: The x values corresponding to the knots.
    - yknots: Set to None for discontinuous fits.
    - y1knots: The fitted values at the left side of each segment.
    - y2knots: The fitted values at the right side of each segment.
    - xfit: The x values corresponding to the fitted data (if fit=True).
    - yfit: The fitted y values (if fit=True).
    - chisq: The chi-squared values for each data point (if fit=True).
    """
    cumSums = calculateCumulativeSums(y, ivar, iknots, grid)

    n = len(iknots)
    ndata = grid.breaks[iknots[-1]] - grid.breaks[iknots[0]]

    backptr = np.full(n, -1, dtype=int)
    min_cost = np.full(n, np.inf)
    min_cost[0] = 0
    for i2 in range(1, n):
        for i1 in range(i2):
            sfit = segmentFit(i1, i2, cumSums)
            cost = sfit.chisq
            if i1 > 0:
                cost += min_cost[i1] + mu * ndata / n
            if cost < min_cost[i2]:
                min_cost[i2] = cost
                backptr[i2] = i1
    pruned = [ ]
    j = n - 1
    while j != -1:
        pruned.append(iknots[j])
        j = backptr[j]
    pruned.reverse()

    xknots = grid.xgrid[pruned]
    y1knots = np.zeros(len(pruned) - 1)
    y2knots = np.zeros(len(pruned) - 1)
    for j in range(len(pruned) - 1):
        sfit = segmentFit(pruned[j], pruned[j + 1], cumSums)
        y1knots[j] = sfit.a + sfit.b * xknots[j]
        y2knots[j] = sfit.a + sfit.b * xknots[j + 1]

    xfit, yfit, chisq = None, None, None

    return FitResult(iknots=pruned, xknots=xknots,
                     yknots=None, y1knots=y1knots, y2knots=y2knots,
                     xfit=xfit, yfit=yfit, chisq=chisq)


def fitFixedKnotsContinuous(y: ArrayLike, ivar: ArrayLike, iknots: ArrayLike,
                            grid: pwlfit.grid.Grid, fit: bool = False) -> FitResult:
    """
    Fit a continuous piecewise linear function to noisy data with fixed knots.
    The free parameters are the values of the piecewise linear function at the knots.

    Parameters:
    y (np.ndarray): The y values of the data to fit. Ignored when corresponding ivar=0.
    ivar (np.ndarray): The inverse variance of the data (1/sigma^2).
    iknots (np.ndarray): The indices of the knots in the grid.
    grid (Grid): The grid object containing the xdata and sdata.
    fit (bool): If True, return the fitted values and chi-squared for each data point.
        Default is False.

    Returns:
    FitResult
    -----------
    A named tuple containing the results of the fit. The fields are:
    - iknots: The indices of the knots.
    - yknots: The fitted values at the knots.
    - y1knots: The fitted values at the left side of each segment.
    - y2knots: The fitted values at the right side of each segment.
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
        slo = grid.sgrid[iknots[i]]
        shi = grid.sgrid[iknots[i + 1]]
        k1, k2 = grid.breaks[iknots[i]], grid.breaks[iknots[i + 1]]
        t = (grid.sdata[k1:k2] - slo) / (shi - slo)
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
    if np.any(Adiag == 0):
        raise ValueError("Matrix is singular, cannot solve for knots.")
    for i in range(1, n):
        Adiag[i] -= Aupper[i-1] ** 2 / Adiag[i-1]
        b[i] -= Aupper[i-1] * b[i-1] / Adiag[i-1]
    yknots = np.zeros(n)
    yknots[-1] = b[-1] / Adiag[-1]
    for i in range(n - 2, -1, -1):
        yknots[i] = (b[i] - Aupper[i] * yknots[i+1]) / Adiag[i]

    xknots = grid.xgrid[iknots]
    y1knots = yknots[:-1]
    y2knots = yknots[1:]

    xfit, yfit, chisq = None, None, None
    if fit:
        # Calculate corresponding chisq for each data point covered by the knots
        xfit = np.empty(ndata)
        yfit = np.zeros(ndata)
        chisq = np.zeros(ndata)
        for i in range(n - 1):
            k1, k2 = grid.breaks[iknots[i]], grid.breaks[iknots[i + 1]]
            xfit[k1-k0:k2-k0] = grid.xdata[k1:k2]
            t = tsave[k1-k0:k2-k0]
            yfit[k1-k0:k2-k0] = y1knots[i] + t * (y2knots[i] - y1knots[i])
            wgt = ivar[k1:k2]
            chisq[k1-k0:k2-k0] = wgt * (y[k1:k2] - yfit[k1-k0:k2-k0]) ** 2
            chisq[k1-k0:k2-k0][wgt == 0] = 0

    return FitResult(iknots=iknots, xknots=xknots,
                     yknots=yknots, y1knots=y1knots, y2knots=y2knots,
                     xfit=xfit, yfit=yfit, chisq=chisq)
