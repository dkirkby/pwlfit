import unittest
import tempfile
import os

import numpy as np

from pwlfit.driver import PWLinearFitConfig, PWLinearFitter
from pwlfit.util import read_sample_data
from pwlfit.grid import Grid


class TestDriver(unittest.TestCase):

    def testSaveLoadConfig(self):

        conf1 = PWLinearFitConfig()
        conf1.options.verbose = True
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmpfile:
            filename = tmpfile.name
            conf1.save(filename)
        conf2 = PWLinearFitConfig.load(filename)
        self.assertEqual(conf1, conf2)
        os.remove(filename)

    def testDriverWithoutRegions(self):
        x, y, ivar = read_sample_data('A')
        grid = Grid(x, ngrid=100)
        conf = PWLinearFitConfig()
        conf.options.find_regions = False
        fitter = PWLinearFitter(grid, conf)
        result = fitter(y, ivar)
        self.assertEqual(type(result.iknots), np.ndarray)
        self.assertEqual(result.iknots.dtype, np.int64)

    def testDriverWithRegions(self):
        x, y, ivar = read_sample_data('C')
        grid = Grid(x, ngrid=2049)
        conf = PWLinearFitConfig()
        conf.options.find_regions = True
        fitter = PWLinearFitter(grid, conf)
        result = fitter(y, ivar)
        self.assertEqual(type(result.iknots), np.ndarray)
        self.assertEqual(result.iknots.dtype, np.int64)

    def testDriverNoRegions(self):
        # Test the case where there are no regions of large smoothed chisq
        xdata = np.linspace(0, 10, 100)
        ydata = np.zeros_like(xdata)
        ivar = np.full(xdata.shape, 0.1)
        grid = Grid(xdata, ngrid=20)

        config = PWLinearFitConfig()
        config.options.find_regions = True
        config.regions.verbose = True
        fitter = PWLinearFitter(grid, config)

        result = fitter(ydata, ivar)
        self.assertEqual(len(result.iknots), config.final.min_total_knots)
        self.assertTrue(np.all(result.yknots == 0))
        self.assertEqual(type(result.iknots), np.ndarray)
        self.assertEqual(result.iknots.dtype, np.int64)
