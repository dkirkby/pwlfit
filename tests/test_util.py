import unittest

import numpy as np

from pwlfit.grid import Grid
from pwlfit.util import generate_data


class TestGenerateData(unittest.TestCase):

    def test_generate_data(self):
        # Test the basic functionality of generate_data
        ndata = 100
        ngrid = 10
        nknots = 5
        data = generate_data(ndata, ngrid, nknots)

        # Check the shapes of the returned data
        self.assertEqual(data.xdata.shape, (ndata,))
        self.assertEqual(data.ydata.shape, (ndata,))
        self.assertEqual(data.ivar.shape, (ndata,))
        self.assertEqual(data.iknots.shape, (nknots,))
        self.assertIsInstance(data.grid, Grid)


if __name__ == "__main__":
    unittest.main()
