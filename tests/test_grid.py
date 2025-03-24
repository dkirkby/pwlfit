import unittest

import numpy as np

from pwlfit.grid import Grid


class TestGrid(unittest.TestCase):

    def verify_breaks(self, grid):
        # Check that the breaks are in increasing order
        self.assertTrue(np.all(np.diff(grid.breaks) >= 0))
        # Check that the breaks are consistent with the xdata and xgrid
        for i in range(grid.ngrid - 1):
            k1, k2 = grid.breaks[i], grid.breaks[i + 1]
            self.assertTrue(np.all(grid.xdata[k1:k2] >= grid.xgrid[i]))
            self.assertTrue(np.all(grid.xdata[k1:k2] <= grid.xgrid[i + 1]))

    def test_equal(self):
        xdata = np.linspace(0, 10, 101)
        grid = Grid(xdata, ngrid=101)
        self.verify_breaks(grid)

    def test_uniform_dense(self):
        xdata = np.linspace(0, 10, 1001)
        grid = Grid(xdata, ngrid=101)
        self.verify_breaks(grid)

    def test_nonuniform_dense(self):
        rng = np.random.default_rng(42)
        xdata = rng.uniform(0, 10, 1000)
        xdata.sort()
        grid = Grid(xdata, ngrid=100)
        self.verify_breaks(grid)

    def test_uniform_sparse(self):
        xdata = np.linspace(0, 10, 11)
        self.assertRaises(ValueError, Grid, xdata, ngrid=25)

    def test_log_transform(self):
        xdata = np.linspace(1, 10, 101)
        grid = Grid(xdata, ngrid=25, transform=np.log, inverse=np.exp)
        self.verify_breaks(grid)

    def test_xgrid_ok(self):
        xdata = np.linspace(0, 10, 101)
        xgrid = np.linspace(0, 10, 11)
        grid = Grid(xdata, xgrid=xgrid)
        self.verify_breaks(grid)

    def test_xgrid_ngrid_ok(self):
        xdata = np.linspace(0, 10, 101)
        xgrid = np.linspace(0, 10, 11)
        grid = Grid(xdata, xgrid=xgrid, ngrid=len(xgrid))
        self.verify_breaks(grid)


if __name__ == "__main__":
    unittest.main()
