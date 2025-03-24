from typing import Callable, Union

import numpy as np
from numpy.typing import ArrayLike


class Grid:
    """A class to represent a grid for piecewise linear fitting."""

    named_transforms = {
        "identity": (lambda x: x, lambda x: x),
        "log": (np.log, np.exp),
    }

    def __init__(self, xdata: ArrayLike, ngrid: int = 0, xgrid: Union[None, ArrayLike] = None,
                 transform: Union[str, Callable[[float], float]] = "identity",
                 inverse: Union[None, Callable[[float], float]] = None) -> None:
        """
        Initialize the grid of possible breakpoints for piecewise linear fitting.

        Parameters:
        xdata (np.ndarray): The x values used to tabulate the data.
        ngrid (int): The number of points to use in the grid. Default is 0, which means
            that xgrid must be provided.
        xgrid (np.ndarray or None): The x values for the grid. If None, it will be
            calculated from xdata and ngrid. Default is None.
        transform (callable or str): Transform from xdata to the space in which the model is linear.
            If a string, must be one of the named transforms: 'identity' or 'log'.
            Default is 'identity'. If not a string, inverse must also be provided.
        inverse (callable or None): Inverse of the transform to map back to xdata space.
            If transform is a string, this must be None. Otherwise, it must be provided.
        Raises ValueError if:
         - ngrid < 2
         - xdata is not strictly increasing
         - xgrid is provided but not strictly increasing or does not cover the range of xdata
         - transformed xdata is not strictly increasing
         - transform and inverse are not consistent with each other
         - there is not at least one data point between grid points
        """
        if not np.all(np.diff(xdata) > 0):
            raise ValueError("xdata must be strictly increasing.")
        self.xdata = xdata

        # Check transform,inverse args.
        if isinstance(transform, str):
            if transform not in self.named_transforms:
                raise ValueError(f"Transform '{transform}' is not recognized.")
            transform, inverse = self.named_transforms[transform]
        elif inverse is None:
            raise ValueError("If transform is not a string, inverse must be provided.")
        # Apply transform and check for strictly increasing.
        self.sdata = transform(xdata)
        if not np.all(np.diff(self.sdata) > 0):
            raise ValueError("Transformed xdata must be strictly increasing.")
        if not np.allclose(inverse(self.sdata), self.xdata):
            raise ValueError("Transform and inverse must be consistent.")

        # Check ngrid, xgrid args.
        if ngrid == 0 and xgrid is None:
            raise ValueError("Either ngrid or xgrid must be provided.")
        if xgrid is not None and ngrid not in (0,  len(xgrid)):
            raise ValueError("ngrid must match the length of xgrid if provided.")
        if xgrid is not None:
            if np.any(np.diff(xgrid) <= 0):
                raise ValueError("xgrid must be strictly increasing.")
            if xgrid[0] > xdata[0] or xgrid[-1] < xdata[-1]:
                raise ValueError("xgrid must cover the range of xdata.")
            self.xgrid = np.asarray(xgrid)
            self.sgrid = transform(xgrid)
            self.ngrid = len(xgrid)
        else:
            if ngrid < 2:
                raise ValueError("ngrid must be at least 2.")
            self.ngrid = ngrid
            # Calculate grid points in the original and transformed spaces.
            self.sgrid = np.linspace(self.sdata[0], self.sdata[-1], ngrid)
            self.xgrid = inverse(self.sgrid)

        # Tabulate how xgrid and xdata are interleaved.
        self.breaks = np.searchsorted(self.xdata, self.xgrid)
        self.breaks[-1] = len(self.xdata)
        if not np.all(np.diff(self.breaks) > 0):
            raise ValueError("Must be at least one data point between grid points.")
