import numpy as np


def fitFixedKnotsContinuous(y, ivar, iknots, grid, fit=False):

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
        Y = y[k1:k2]
        Adiag[i] += np.sum(wgt * (1 - t)**2)
        Aupper[i] += np.sum(wgt * t * (1 - t))
        b[i] += np.sum(wgt * Y * (1 - t))
        Adiag[i+1] += np.sum(wgt * t**2)
        b[i+1] += np.sum(wgt * Y * t)
        tsave[k1-k0:k2-k0] = t
    # Solve the tridiagonal system using the Thomas algorithm.
    for i in range(1, n):
        Adiag[i] -= Aupper[i-1] ** 2 / Adiag[i-1]
        b[i] -= Aupper[i-1] * b[i-1] / Adiag[i-1]
    Y = np.zeros(n)
    Y[-1] = b[-1] / Adiag[-1]
    for i in range(n - 2, -1, -1):
        Y[i] = (b[i] - Aupper[i] * Y[i+1]) / Adiag[i]

    result = dict(iknots=iknots, Y=Y)

    if fit:
        # Calculate corresponding chisq for each data point covered by the knots
        x_fit = np.empty(ndata)
        y_fit = np.zeros(ndata)
        chisq = np.zeros(ndata)
        for i in range(n - 1):
            k1, k2 = grid.breaks[iknots[i]], grid.breaks[iknots[i + 1]]
            x_fit[k1-k0:k2-k0] = grid.x_data[k1:k2]
            t = tsave[k1-k0:k2-k0]
            y_fit[k1-k0:k2-k0] = Y[i] + t * (Y[i + 1] - Y[i])
            chisq[k1-k0:k2-k0] = ivar[k1:k2] * (y[k1:k2] - y_fit[k1-k0:k2-k0]) ** 2
        result.update(x_fit=x_fit, y_fit=y_fit, chisq=chisq)

    return result
