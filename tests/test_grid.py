import unittest

import numpy as np

from pwlfit.grid import Grid


class TestGrid(unittest.TestCase):

    def setUp(self):
        # Create a simple grid for testing
        rng = np.random.default_rng(42)
        x_data = rng.uniform(0, 10, 1000)
        x_data.sort()
        self.grid = Grid(x_data, n_points=100)

    def test_initialization(self):
        # Test if the grid is initialized correctly
        self.assertEqual(len(self.grid.x_grid), 100)


if __name__ == "__main__":
    unittest.main()
