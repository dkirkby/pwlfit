from typing import Callable

import numpy as np
from numpy.typing import ArrayLike


class Grid:
    """A class to represent a grid for piecewise linear fitting."""

    def __init__(self, xdata: ArrayLike, ngrid: int,
                 transform: Callable[[float], float] = lambda x: x,
                 inverse: Callable[[float], float] = lambda x: x) -> None:
        """
        Initialize the grid of possible breakpoints for piecewise linear fitting.

        Parameters:
        xdata (np.ndarray): The x values used to tabulate the data.
        ngrid (int): The number of points to use in the grid.
        transform (callable): Transform from xdata to the space in which the model is linear.
            Default is identity.
        inverse (callable): Inverse of the transform to map back to xdata space.
            Default is identity.
        Raises ValueError if:
         - ngrid < 2
         - xdata is not strictly increasing
         - transformed xdata is not strictly increasing
         - transform and inverse are not consistent with each other
         - there is not at least one data point between grid points
        """
        if not np.all(np.diff(xdata) > 0):
            raise ValueError("xdata must be strictly increasing.")
        if ngrid < 2:
            raise ValueError("ngrid must be at least 2.")
        self.ngrid = ngrid
        self.xdata = xdata
        self.sdata = transform(xdata)
        if not np.all(np.diff(self.sdata) > 0):
            raise ValueError("Transformed xdata must be strictly increasing.")
        if not np.allclose(inverse(self.sdata), self.xdata):
            raise ValueError("Transform and inverse must be consistent.")
        self.sgrid = np.linspace(self.sdata[0], self.sdata[-1], ngrid)
        self.xgrid = inverse(self.sgrid)
        # Tabulate how xgrid and xdata are interleaved.
        self.breaks = np.searchsorted(self.xdata, self.xgrid)
        self.breaks[-1] += 1
        if not np.all(np.diff(self.breaks) > 0):
            raise ValueError("Must be at least one data point between grid points.")
