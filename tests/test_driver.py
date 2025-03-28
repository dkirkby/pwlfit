import unittest
import tempfile
import os

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

    def testDriverWithRegions(self):
        x, y, ivar = read_sample_data('C')
        grid = Grid(x, ngrid=2049)
        conf = PWLinearFitConfig()
        conf.options.find_regions = True
        fitter = PWLinearFitter(grid, conf)
        result = fitter(y, ivar)
