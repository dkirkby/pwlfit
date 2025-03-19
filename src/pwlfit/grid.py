from typing import Callable

import numpy as np


class Grid:
    """A class to represent a grid for piecewise linear fitting."""

    def __init__(self, x_data: np.ndarray, n_points: int,
                 transform: Callable[[float], float] = lambda x: x,
                 inverse: Callable[[float], float] = lambda x: x):
        """
        Initialize the grid of possible breakpoints for piecewise linear fitting.

        Parameters:
        x_data (np.ndarray): The x values used to tabulate the data.
        n_points (int): The number of points to use in the grid.
        transform (callable): Transform from x_data to the space in which the model is linear.
            Default is identity.
        inverse (callable): Inverse of the transform to map back to x_data space.
            Default is identity.
        Raises ValueError if:
         - x_data is not strictly increasing
         - transformed x_data is not strictly increasing
         - n_points < 2
         - transform and inverse are not consistent with each other
        """
        if not np.all(np.diff(x_data) > 0):
            raise ValueError("x_data must be strictly increasing.")
        if n_points < 2:
            raise ValueError("n_points must be at least 2.")
        self.x_data = x_data
        self.s_data = transform(x_data)
        if not np.all(np.diff(self.s_data) > 0):
            raise ValueError("Transformed x_data must be strictly increasing.")
        if not np.allclose(inverse(self.s_data), self.x_data):
            raise ValueError("Transform and inverse must be consistent.")
        self.s_grid = np.linspace(self.s_data[0], self.s_data[-1], n_points)
        self.x_grid = inverse(self.s_grid)
        # Tabulate how x_grid and x_data are interleaved.
        self.breaks = np.searchsorted(self.x_data, self.x_grid)
        self.breaks[-1] += 1
