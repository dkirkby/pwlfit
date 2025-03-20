import unittest

import numpy as np

from pwlfit.fit import fitFixedKnotsContinuous, fitPrunedKnotsDiscontinuous, fitPrunedKnotsContinuous
from pwlfit.util import generate_data


class TestPrunedKnotsContinuous(unittest.TestCase):

    def test(self):
        D = generate_data(ndata=1000, ngrid=50, nknots=10, noise=0.05, missing_frac=0.05)
        # Interpolate the true model to each grid point
        yknots = np.interp(D.grid.xgrid, D.xknots, D.yknots)
        # Fit using all possible knots and verify that they are pruned to the correct subset
        fit = fitPrunedKnotsContinuous(D.ydata, D.ivar, np.arange(D.grid.ngrid), yknots, D.grid, mu=2)
        self.assertTrue(np.array_equal(D.iknots, fit.iknots))


class TestPrunedKnotsDiscontinuous(unittest.TestCase):

    def testContinuous(self):
        D = generate_data(ndata=1000, ngrid=50, nknots=10, noise=0.05, missing_frac=0.05)
        # Fit using all possible knots and verify that they are pruned to the correct subset
        fit = fitPrunedKnotsDiscontinuous(D.ydata, D.ivar, np.arange(D.grid.ngrid), D.grid, mu=2)
        self.assertTrue(np.array_equal(D.iknots, fit.iknots))


class TestFixedKnotsContinuous(unittest.TestCase):

    def setUp(self):
        self.D = generate_data(ndata=1000, ngrid=50, nknots=10, noise=0.05, missing_frac=0.05)

    def testFullRange(self):
        iknots = self.D.iknots
        ndata = self.D.grid.breaks[iknots[-1]] - self.D.grid.breaks[iknots[0]]
        fit = fitFixedKnotsContinuous(self.D.ydata, self.D.ivar, iknots, self.D.grid, fit=True)
        # Check that the fit results are of the expected shape
        self.assertTrue(np.array_equal(fit.iknots, iknots))
        self.assertEqual(fit.xknots.shape, (len(iknots),))
        self.assertEqual(fit.y1knots.shape, (len(iknots) - 1,))
        self.assertEqual(fit.y2knots.shape, (len(iknots) - 1,))
        self.assertTrue(np.array_equal(fit.y1knots[1:], fit.y2knots[:-1]))
        self.assertEqual(fit.xfit.shape, (ndata,))
        self.assertEqual(fit.yfit.shape, (ndata,))
        self.assertEqual(fit.chisq.shape, (ndata,))
        # Check that the fitted values are within the expected range
        self.assertTrue((np.mean(fit.chisq) > 0.9) and (np.mean(fit.chisq) < 1.1))

    def testPartialRange(self):
        iknots = self.D.iknots[2:-1]
        ndata = self.D.grid.breaks[iknots[-1]] - self.D.grid.breaks[iknots[0]]
        fit = fitFixedKnotsContinuous(self.D.ydata, self.D.ivar, iknots, self.D.grid, fit=True)
        # Check that the fit results are of the expected shape
        self.assertTrue(np.array_equal(fit.iknots, iknots))
        self.assertEqual(fit.xknots.shape, (len(iknots),))
        self.assertEqual(fit.y1knots.shape, (len(iknots) - 1,))
        self.assertEqual(fit.y2knots.shape, (len(iknots) - 1,))
        self.assertTrue(np.array_equal(fit.y1knots[1:], fit.y2knots[:-1]))
        self.assertEqual(fit.xfit.shape, (ndata,))
        self.assertEqual(fit.yfit.shape, (ndata,))
        self.assertEqual(fit.chisq.shape, (ndata,))
        # Check that the fitted values are within the expected range
        self.assertTrue((np.mean(fit.chisq) > 0.9) and (np.mean(fit.chisq) < 1.1))

    def testSubRange(self):
        iknots = self.D.iknots[2::2]
        ndata = self.D.grid.breaks[iknots[-1]] - self.D.grid.breaks[iknots[0]]
        fit = fitFixedKnotsContinuous(self.D.ydata, self.D.ivar, iknots, self.D.grid, fit=True)
        # Check that the fit results are of the expected shape
        self.assertTrue(np.array_equal(fit.iknots, iknots))
        self.assertEqual(fit.xknots.shape, (len(iknots),))
        self.assertEqual(fit.y1knots.shape, (len(iknots) - 1,))
        self.assertEqual(fit.y2knots.shape, (len(iknots) - 1,))
        self.assertTrue(np.array_equal(fit.y1knots[1:], fit.y2knots[:-1]))
        self.assertEqual(fit.xfit.shape, (ndata,))
        self.assertEqual(fit.yfit.shape, (ndata,))
        self.assertEqual(fit.chisq.shape, (ndata,))
        # Check that the fitted values are within the expected range
        self.assertTrue(np.mean(fit.chisq) > 10)

    def testZeroIvar(self):
        ivar = np.zeros_like(self.D.ivar)
        with self.assertRaises(ValueError):
            fitFixedKnotsContinuous(self.D.ydata, ivar, self.D.iknots, self.D.grid, fit=True)
