import unittest

import numpy as np

from pwlfit.grid import Grid
from pwlfit.util import generate_data, smooth_weighted_data, read_sample_data


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


class TestSmoothData(unittest.TestCase):

    def testTransformed(self):

        D = generate_data(
            2000, 20, 5, noise=0.01, missing_frac=0.05,
            xlo=0.1, xhi=10, transform='log')

        for transformed in (True, False):
            ysmooth = smooth_weighted_data(
                D.ydata, D.ivar, D.iknots, D.grid, window_size=31,
                poly_order=3, transformed=transformed)
            self.assertTrue(np.allclose(D.yknots, ysmooth, atol=0.01, rtol=0.02))


class TestSampleData(unittest.TestCase):

    def testReadABC(self):

        for sampleID in 'ABC':
            xdata, ydata, ivar = read_sample_data(sampleID)
            self.assertTrue(xdata.size == ydata.size)
            self.assertTrue(xdata.size == ivar.size)
            self.assertTrue(np.all(ivar >= 0))
            self.assertTrue(np.all(np.diff(xdata[ivar > 0]) > 0))

    def testReadX(self):
        # Test that ValueError is raised for invalid sampleID
        with self.assertRaises(ValueError):
            read_sample_data('X')
        with self.assertRaises(ValueError):
            read_sample_data('')


if __name__ == "__main__":
    unittest.main()
