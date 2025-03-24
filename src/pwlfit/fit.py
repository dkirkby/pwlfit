from typing import NamedTuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

import pwlfit.grid


Float64NDArray = NDArray[np.float64]  # np.ndarray[float] in python 3.10+
Int64NDArray = NDArray[np.int64]      # np.ndarray[int] in python 3.10+


class FitResult(NamedTuple):
    """A named tuple to hold the results of the piecewise linear fit.

    The fields are:
    - iknots: The indices of the knots in a grid that define the piecewise segments.
    - xknots: The x values corresponding to the knots.
    - yknots: The y values at the knots (None for discontinuous fits).
    - y1knots: The fitted values at the left side of each segment.
    - y2knots: The fitted values at the right side of each segment.
    - ndof: The number of degrees of freedom for the fit, equal to the number of
        data points in the fit region minus the number of free parameters in the fit.
        Note that data points with ivar==0 are included in this count.
    - xfit: The x values corresponding to the fitted data (if fit=True).
    - yfit: The fitted y values (if fit=True).
    - chisq: The chi-squared values for each data point (if fit=True).

    The number of segments is len(iknots) - 1. All arrays above have length len(iknots),
    except for y1knots and y2knots which have length len(iknots) - 1.

    The xfit, yfit, and chisq arrays are only populated when `fit=True` is passed to one
    of the fitting functions below.

    A continuous piecewise fit has y1knots[i] = yknots[i] and y2knots[i] = yknots[i+1] for all i.
    A discontinuous piecewise fit has yknots set to None, and y1knots and y2knots
    represent the fitted values at the left and right side of each segment respectively.

    To plot a continuous piecewise fit, use:

    >>> plt.plot(xknots, yknots, 'o-')

    To plot a discontinuous piecewise fit, use:

    >>> for i in range(len(y1knots)):
    >>>     plt.plot([xknots[i], xknots[i+1]], [y1knots[i], y2knots[i]], 'o-')
    """
    iknots: Int64NDArray
    xknots: Float64NDArray
    yknots: Float64NDArray
    y1knots: Float64NDArray
    y2knots: Float64NDArray
    ndof: int
    xfit: Union[None, Float64NDArray] = None
    yfit: Union[None, Float64NDArray] = None
    chisq: Union[None, Float64NDArray] = None


class CummulativeSums(NamedTuple):
    Sw: Float64NDArray
    Sx: Float64NDArray
    Sy: Float64NDArray
    Sxx: Float64NDArray
    Sxy: Float64NDArray
    Syy: Float64NDArray


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
        X = grid.sdata[k1:k2]
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
    pruned = np.array(pruned)

    xknots = grid.xgrid[pruned]
    sknots = grid.sgrid[pruned]
    y1knots = np.zeros(len(pruned) - 1)
    y2knots = np.zeros(len(pruned) - 1)
    for j in range(len(pruned) - 1):
        sfit = segmentFit(pruned[j], pruned[j + 1], cumSums)
        y1knots[j] = sfit.a + sfit.b * sknots[j]
        y2knots[j] = sfit.a + sfit.b * sknots[j + 1]
    ndof = ndata - 2 * (len(pruned) - 1) # 2 parameters per segment

    xfit, yfit, chisq = evaluateFit(y, ivar, pruned, y1knots, y2knots, grid) if fit else (None, None, None)

    return FitResult(iknots=pruned, xknots=xknots,
                     yknots=None, y1knots=y1knots, y2knots=y2knots,
                     xfit=xfit, yfit=yfit, chisq=chisq, ndof=ndof)


def fitPrunedKnotsContinuous(y: ArrayLike, ivar: ArrayLike, iknots: ArrayLike, yknots: ArrayLike,
                             grid: pwlfit.grid.Grid, mu: float = 2, fit: bool = False) -> FitResult:
    """
    Fit a continuous piecewise linear function to noisy data with pruned knots.
    Since the y values to use at each knot are provided, the linear segment between
    any two knots is already defined and there are no free parameters to fit.
    Instead, this function performs an optimization to determine the best pruned
    subset of the input iknots to use.

    Parameters:
    y (np.ndarray): The y values of the data to fit. Ignored when corresponding ivar=0.
    ivar (np.ndarray): The inverse variance of the data (1/sigma^2).
    iknots (np.ndarray): The indices of the knots in the grid.
    yknots (np.ndarray): The y values at the knots to use for the fit.
    grid (Grid): The grid object containing the xdata and sdata.
    mu (float): A penalty term for the number of knots. Larger values will favor fewer knots.
    fit (bool): If True, return the fitted values and chi-squared for each data point.
        Default is False.

    Returns:
    FitResult
    """
    n = len(iknots)
    ndata = grid.breaks[iknots[-1]] - grid.breaks[iknots[0]]

    # Precompute the error matrix E[i1,i2] for i1 < i2
    E = np.full((n, n), np.inf)
    for i1 in range(n):
        slo = grid.sgrid[iknots[i1]]
        k1 = grid.breaks[iknots[i1]]
        for i2 in range(i1 + 1, n):
            shi = grid.sgrid[iknots[i2]]
            k2 = grid.breaks[iknots[i2]]
            t = (grid.sdata[k1:k2] - slo) / (shi - slo)
            linear = yknots[i1] * (1 - t) + yknots[i2] * t
            wgt = ivar[k1:k2]
            chisq = wgt * (y[k1:k2] - linear) ** 2
            chisq[wgt == 0] = 0  # Ignore any y=NaN values when ivar==0
            E[i1, i2] = np.sum(chisq)

    # Dynamic programming to find the best subset of iknots to use.
    backptr = np.full(n, -1, dtype=int)
    min_cost = np.full(n, np.inf)
    min_cost[0] = 0
    for i2 in range(1, n):
        for i1 in range(i2):
            cost = E[i1, i2]
            if i1 > 0:
                cost += min_cost[i1] + mu * ndata / n
            if cost < min_cost[i2]:
                min_cost[i2] = cost
                backptr[i2] = i1

    # Reconstruct the best subset of iknots
    pruned = [ ]
    j = n - 1
    while j != -1:
        pruned.append(j)
        j = backptr[j]
    pruned.reverse()
    pruned = np.array(pruned)

    xknots = grid.xgrid[iknots[pruned]]
    y1knots = yknots[pruned[:-1]]
    y2knots = yknots[pruned[1:]]
    yknots = yknots[pruned]
    ndof = ndata - len(pruned)  # 1 parameter per knot

    xfit, yfit, chisq = evaluateFit(y, ivar, pruned, y1knots, y2knots, grid) if fit else (None, None, None)

    return FitResult(iknots=pruned, xknots=xknots,
                     yknots=yknots, y1knots=y1knots, y2knots=y2knots,
                     xfit=xfit, yfit=yfit, chisq=chisq, ndof=ndof)


def fitFixedKnotsContinuous(y: ArrayLike, ivar: ArrayLike, grid: pwlfit.grid.Grid,
                            iknots: Union[None, ArrayLike] = None, fit: bool = False) -> FitResult:
    """
    Fit a continuous piecewise linear function to noisy data with fixed knots.
    The free parameters are the values of the piecewise linear function at the knots.

    Parameters:
    y (np.ndarray): The y values of the data to fit. Ignored when corresponding ivar=0.
    ivar (np.ndarray): The inverse variance of the data (1/sigma^2).
    grid (Grid): The grid object defining the x coordinates of the data and linear segment
        boundaries to use.
    iknots (np.ndarray): The indices of the knots in the grid or use all available grid
        points if None. Default is None.
    fit (bool): If True, return the fitted values and chi-squared for each data point.
        Default is False.

    Returns:
    FitResult
    """
    iknots = checkIKnots(iknots, grid)
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
    ndof = ndata - n

    xfit, yfit, chisq = evaluateFit(y, ivar, iknots, y1knots, y2knots, grid) if fit else (None, None, None)

    return FitResult(iknots=iknots, xknots=xknots,
                     yknots=yknots, y1knots=y1knots, y2knots=y2knots,
                     xfit=xfit, yfit=yfit, chisq=chisq, ndof=ndof)


def evaluateFit(y: Float64NDArray, ivar: Float64NDArray, iknots: np.ndarray,
                y1knots: Float64NDArray, y2knots: Float64NDArray, grid: pwlfit.grid.Grid) -> tuple:
    """
    Evaluate the piecewise linear fit at the grid points defined by iknots.

    Parameters:
    y (np.ndarray): The y values of the data to fit. Ignored when corresponding ivar=0.
    ivar (np.ndarray): The inverse variance of the data (1/sigma^2).
    iknots (np.ndarray): The indices of the knots in the grid.
    y1knots (np.ndarray): The y values at the left side of each segment.
    y2knots (np.ndarray): The y values at the right side of each segment.
    grid (Grid): The grid object containing the xdata and sdata.

    Returns:
    tuple: A tuple (xfit, yfit, chisq) to include in a FitResult
    """
    n = len(iknots)
    k0 = grid.breaks[iknots[0]]
    ndata = grid.breaks[iknots[-1]] - k0
    xfit = np.zeros(ndata)
    yfit = np.zeros(ndata)
    chisq = np.zeros(ndata)
    for i in range(n - 1):
        k1, k2 = grid.breaks[iknots[i]], grid.breaks[iknots[i + 1]]
        xfit[k1-k0:k2-k0] = grid.xdata[k1:k2]
        t = (grid.sdata[k1:k2] - grid.sgrid[iknots[i]]) / (grid.sgrid[iknots[i + 1]] - grid.sgrid[iknots[i]])
        yfit[k1-k0:k2-k0] = y1knots[i] + t * (y2knots[i] - y1knots[i])
        wgt = ivar[k1:k2]
        chisq[k1-k0:k2-k0] = wgt * (y[k1:k2] - yfit[k1-k0:k2-k0]) ** 2
        chisq[k1-k0:k2-k0][wgt == 0] = 0  # Ignore any y=NaN values when ivar==0

    return xfit, yfit, chisq


def checkIKnots(iknots, grid):
    """Check that the provided iknots are valid for the given grid."""
    if iknots is None:
        # Use all knots by default
        return np.arange(grid.ngrid)
    if not np.all(np.diff(iknots) > 0):
        raise ValueError("iknots must be strictly increasing.")
    if iknots[0] < 0 or iknots[-1] >= grid.ngrid:
        raise ValueError("iknots must be within the range of the grid.")
    if len(iknots) < 2:
        raise ValueError("At least two knots are required.")
    return iknots
