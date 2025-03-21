from typing import Callable, Union

import numpy as np
from numpy.typing import ArrayLike


class Grid:
    """A class to represent a grid for piecewise linear fitting."""

    named_transforms = {
        "identity": (lambda x: x, lambda x: x),
        "log": (np.log, np.exp),
    }

    def __init__(self, xdata: ArrayLike, ngrid: int,
                 transform: Union[str, Callable[[float], float]] = "identity",
                 inverse: Union[None, Callable[[float], float]] = None) -> None:
        """
        Initialize the grid of possible breakpoints for piecewise linear fitting.

        Parameters:
        xdata (np.ndarray): The x values used to tabulate the data.
        ngrid (int): The number of points to use in the grid.
        transform (callable or str): Transform from xdata to the space in which the model is linear.
            If a string, must be one of the named transforms: 'identity' or 'log'.
            Default is 'identity'. If not a string, inverse must also be provided.
        inverse (callable or None): Inverse of the transform to map back to xdata space.
            If transform is a string, this must be None. Otherwise, it must be provided.
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
        # Calculate grid points in the original and transformed spaces.
        self.sgrid = np.linspace(self.sdata[0], self.sdata[-1], ngrid)
        self.xgrid = inverse(self.sgrid)
        # Tabulate how xgrid and xdata are interleaved.
        self.breaks = np.searchsorted(self.xdata, self.xgrid)
        self.breaks[-1] = len(self.xdata)
        if not np.all(np.diff(self.breaks) > 0):
            raise ValueError("Must be at least one data point between grid points.")
