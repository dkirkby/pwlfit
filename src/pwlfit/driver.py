from dataclasses import dataclass, asdict, field
from typing import Callable, Union

import numpy as np
from numpy.typing import ArrayLike

import pwlfit.grid
import pwlfit.fit
import pwlfit.region
import pwlfit.util


@dataclass
class PWLinearFitOptions:
    verbose: bool = False
    find_regions: bool = False
    use_continuous_pruned_fit: bool = True

@dataclass
class FindRegionsConfig:
    verbose: bool = False
    num_coarse_knots: int = 9
    region_inset: int = 4
    region_pad: int = 3
    chisq_window_size: int = 19
    chisq_poly_order: int = 1
    smooth_chisq_cut: float = 4.0

@dataclass
class PrunedFitContinuousConfig:
    verbose: bool = False
    smoothing_window_size: int = 9
    smoothing_poly_order: int = 3
    smoothing_transformed: bool = True
    penalty: float = 2

@dataclass
class PrunedFitDiscontinuousConfig:
    verbose: bool = False
    penalty: float = 2

@dataclass
class PWLinearFitConfig:
    options: PWLinearFitOptions = field(default_factory=PWLinearFitOptions)
    regions: FindRegionsConfig = field(default_factory=FindRegionsConfig)
    continuous: PrunedFitContinuousConfig = field(default_factory=PrunedFitContinuousConfig)
    discontinuous: PrunedFitDiscontinuousConfig = field(default_factory=PrunedFitDiscontinuousConfig)


class PWLinearFitter:

    def __init__(self, grid: pwlfit.grid.Grid, config: PWLinearFitConfig):
        self.grid = grid
        self.config = config
        if self.config.options.find_regions:
            # Initialize the knots to use for the coarse fit
            ncoarse = self.config.regions.num_coarse_knots
            spacing = int(round(grid.ngrid / (ncoarse - 1)))
            self.coarse_iknots = [ k * spacing for k in range(ncoarse - 1) ] + [grid.ngrid - 1]
        else:
            self.coarse_fit = None
            self.chisq_median = np.nan
            self.chisq_smooth = None

    def __call__(self, y: ArrayLike, ivar: ArrayLike) -> Union[None, pwlfit.fit.FitResult]:

        opts = self.config.options
        rconf = self.config.regions
        if opts.find_regions:
            # Perform an initial coarse fit to the smooth trend of the data
            self.coarse_fit = pwlfit.fit.fitFixedKnotsContinuous(
                y, ivar, self.grid, iknots=self.coarse_iknots, fit=True)
            # Find regions of the data that deviate significantly from the smooth trend
            self.chisq_median, self.chisq_smooth, self.regions = pwlfit.region.findRegions(
                self.coarse_fit, self.grid, inset=rconf.region_inset,
                pad=rconf.region_pad, chisq_cut=rconf.smooth_chisq_cut,
                window_size=rconf.chisq_window_size, poly_order=rconf.chisq_poly_order)
            # Verbose reporting
            if opts.verbose or rconf.verbose:
                print(f'Found {len(self.regions)} regions with ' +
                      f'median smoothed chisq {self.chisq_median:.3f}')
            if rconf.verbose:
                for i, region in enumerate(self.regions):
                    print(f'  region[{i}] lo={region.lo} hi={region.hi}')
        else:
            # Create a single region covering the entire grid
            self.regions = [ pwlfit.region.Region(lo=0, hi=self.grid.ngrid) ]

        # Enforce min/max region sizes.

        # Prune each region independently.
        pconf = self.config.continuous if opts.use_continuous_pruned_fit else self.config.discontinuous
        for i, region in enumerate(self.regions):
            region_iknots = np.arange(region.lo, region.hi + 1)
            if opts.use_continuous_pruned_fit:
                region_yknots = pwlfit.util.smooth_weighted_data(
                    y, ivar, self.grid, iknots=region_iknots, window_size=pconf.smoothing_window_size,
                    poly_order=pconf.smoothing_poly_order, transformed=pconf.smoothing_transformed)
                region.fit = pwlfit.fit.fitPrunedKnotsContinuous(
                    y, ivar, self.grid, yknots=region_yknots,
                    iknots=region_iknots, mu=pconf.penalty, fit=False)
            else:
                region.fit = pwlfit.fit.fitPrunedKnotsDiscontinuous(
                    y, ivar, self.grid, iknots=region_iknots, mu=pconf.penalty, fit=False)
            if pconf.verbose:
                print(f'Pruned region {i} from {region.hi-region.lo+1} to {len(region.fit.iknots)} knots')

        # Combined pruned regions into a global list of knots to use for the final fit

        # Perform the final fit

        # Prune the final fit to remove the least significant knots if necessary

        return None
