import unittest

import numpy as np

from pwlfit.grid import Grid


class TestGrid(unittest.TestCase):

    def verify_breaks(self, grid):
        # Check that the breaks are in increasing order
        self.assertTrue(np.all(np.diff(grid.breaks) >= 0))
        # Check that the breaks are consistent with the x_data and x_grid
        for i in range(grid.n_grid - 1):
            k1, k2 = grid.breaks[i], grid.breaks[i + 1]
            self.assertTrue(np.all(grid.x_data[k1:k2] >= grid.x_grid[i]))
            self.assertTrue(np.all(grid.x_data[k1:k2] <= grid.x_grid[i + 1]))

    def test_equal(self):
        x_data = np.linspace(0, 10, 101)
        grid = Grid(x_data, n_grid=101)
        self.verify_breaks(grid)

    def test_uniform_dense(self):
        x_data = np.linspace(0, 10, 1001)
        grid = Grid(x_data, n_grid=101)
        self.verify_breaks(grid)

    def test_nonuniform_dense(self):
        rng = np.random.default_rng(42)
        x_data = rng.uniform(0, 10, 1000)
        x_data.sort()
        grid = Grid(x_data, n_grid=100)
        self.verify_breaks(grid)

    def test_uniform_sparse(self):
        x_data = np.linspace(0, 10, 11)
        self.assertRaises(ValueError, Grid, x_data, n_grid=25)

    def test_log_transform(self):
        x_data = np.linspace(1, 10, 101)
        grid = Grid(x_data, n_grid=25, transform=np.log, inverse=np.exp)
        self.verify_breaks(grid)


if __name__ == "__main__":
    unittest.main()
