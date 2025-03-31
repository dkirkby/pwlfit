import unittest

import numpy as np

from pwlfit.util import read_sample_data
from pwlfit.grid import Grid
from pwlfit.fit import fitFixedKnotsContinuous
from pwlfit.region import findRegions, insertKnots, Region


class TestRegions(unittest.TestCase):

    def testInsertMaxSpan(self):
        i1 = 5
        for max_span in range(2, 10):
            for i2 in range(i1+1, 100):
                inserted = insertKnots(i1, i2, max_span=max_span)
                all = [i1] + inserted + [i2]
                diff = np.diff(all)
                self.assertTrue(np.all((diff > 0) & (diff <= max_span)))

    def testInsertNInsert(self):
        i1 = 5
        for i2 in range(i1+2, 100):
            for ninsert in range(1, i2 - i1):
                inserted = insertKnots(i1, i2, ninsert=ninsert)
                self.assertEqual(len(inserted), ninsert)
                diff = np.diff([i1] + inserted + [i2])
                self.assertTrue(np.all(diff > 0))

    def testInsertKnots(self):
        self.assertEqual(insertKnots(i1=5, i2=10, max_span=7), [ ])
        self.assertEqual(insertKnots(i1=5, i2=10, max_span=5), [ ])
        self.assertEqual(insertKnots(i1=5, i2=10, max_span=3), [ 7 ])
        self.assertEqual(insertKnots(i1=5, i2=10, max_span=2), [ 7, 8 ])

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
