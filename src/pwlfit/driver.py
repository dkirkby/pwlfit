from dataclasses import dataclass, asdict, field
from typing import Callable, Union
from pathlib import Path
import time

import yaml

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
    max_region_knots: int = 64

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
class FinalFitConfig:
    verbose: bool = False
    max_spacing_factor: int = 9
    min_region_knots: int = 3
    eval_fit: bool = False

@dataclass
class PWLinearFitConfig:
    options: PWLinearFitOptions = field(default_factory=PWLinearFitOptions)
    regions: FindRegionsConfig = field(default_factory=FindRegionsConfig)
    continuous: PrunedFitContinuousConfig = field(default_factory=PrunedFitContinuousConfig)
    discontinuous: PrunedFitDiscontinuousConfig = field(default_factory=PrunedFitDiscontinuousConfig)
    final: FinalFitConfig = field(default_factory=FinalFitConfig)

    def save(self, filename: Union[str,Path]) -> None:

        with open(filename, 'w') as f:
            yaml.dump(asdict(self), f, sort_keys=False)

    @classmethod
    def load(cls, filename: Union[str,Path]) -> "PWLinearFitConfig":

        with open(filename, 'r') as f:
            config = yaml.safe_load(f)
        return cls(
            options=PWLinearFitOptions(**config.get('options', {})),
            regions=FindRegionsConfig(**config.get('regions', {})),
            continuous=PrunedFitContinuousConfig(**config.get('continuous', {})),
            discontinuous=PrunedFitDiscontinuousConfig(**config.get('discontinuous', {})),
            final=FinalFitConfig(**config.get('final', {})))


class PWLinearFitter:

    def __init__(self, grid: pwlfit.grid.Grid, config: PWLinearFitConfig = PWLinearFitConfig()) -> None:
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

        start = time.time()
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
                window_size=rconf.chisq_window_size, poly_order=rconf.chisq_poly_order,
                verbose=rconf.verbose)
            # Verbose reporting
            if opts.verbose or rconf.verbose:
                print(f'Found {len(self.regions)} regions with ' +
                      f'median smoothed chisq {self.chisq_median:.3f}')
            if rconf.verbose:
                for i, region in enumerate(self.regions):
                    print(f'  region[{i}] lo={region.lo} hi={region.hi}')
        else:
            # Create a single region covering the entire grid
            self.regions = [ pwlfit.region.Region(lo=0, hi=self.grid.ngrid-1) ]

        # Split large regions in order to limit quadratic running time
        if rconf.max_region_knots > 0:
            self.regions = pwlfit.region.splitRegions(
                self.regions, rconf.max_region_knots, rconf.verbose)

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
        fconf = self.config.final
        iknots = pwlfit.region.combineRegions(
            self.regions, self.grid, max_spacing_factor=fconf.max_spacing_factor,
            min_knots=fconf.min_region_knots, verbose=fconf.verbose)
        if opts.verbose or fconf.verbose:
            print(f'Final fit uses {len(iknots)} knots: {iknots}')

        # Perform the final fit
        self.fit = pwlfit.fit.fitFixedKnotsContinuous(y, ivar, self.grid, iknots=iknots, fit=fconf.eval_fit)

        # Prune the final fit to remove the least significant knots if necessary

        self.elapsed = time.time() - start
        if opts.verbose:
            print(f'PWLinearFitter completed in {1e3*self.elapsed:.1f} ms')

        return self.fit
