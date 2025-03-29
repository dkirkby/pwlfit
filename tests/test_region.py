import unittest

import numpy as np

from pwlfit.util import read_sample_data
from pwlfit.grid import Grid
from pwlfit.fit import fitFixedKnotsContinuous
from pwlfit.region import findRegions, insertKnots, Region


class TestRegions(unittest.TestCase):

    def testInsertKnots(self):
        self.assertEqual(insertKnots(i1=5, i2=10, max_span=7), [ ])
        self.assertEqual(insertKnots(i1=5, i2=10, max_span=5), [ ])
        self.assertEqual(insertKnots(i1=5, i2=10, max_span=3), [ 8 ])
        self.assertEqual(insertKnots(i1=5, i2=10, max_span=2), [ 7, 9 ])

    def testFindRegions(self):
        xdata, ydata, ivar = read_sample_data('C')
        grid = Grid(xdata, ngrid=2049, transform='log')
        iknots = np.arange(0, grid.ngrid, 256)
        fit = fitFixedKnotsContinuous(ydata, ivar, grid, iknots, fit=True)
        chisq_median, chisq_smooth, regions = findRegions(
            fit, grid, inset=4, pad=3, chisq_cut=4, window_size=19, poly_order=1)
        self.assertEqual(len(regions), 3)
        self.assertEqual(regions[0], Region(lo=353, hi=409))
        self.assertEqual(regions[1], Region(lo=790, hi=806))
        self.assertEqual(regions[2], Region(lo=1573, hi=1595))
        self.assertTrue(np.allclose(np.median(chisq_smooth), chisq_median))
        self.assertTrue(np.allclose(chisq_median, 1.066, atol=1e-3, rtol=1e-4))


if __name__ == "__main__":
    unittest.main()
