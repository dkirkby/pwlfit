from typing import Tuple,List
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

import pwlfit.fit
import pwlfit.grid

import scipy.signal


@dataclass
class Region:
    lo: int
    hi: int


def findRegions(fit: pwlfit.fit.FitResult, grid: pwlfit.grid.Grid,
                inset: int = 4, pad: int = 3, chisq_cut: float = 4,
                window_size: int = 19, poly_order: int = 1) -> Tuple[float,List[Region]]:

    # Smooth chisq with Savitzky-Golay filter
    chisq_smooth = scipy.signal.savgol_filter(fit.chisq, window_size, poly_order)

    # Normalize the cut to the median smooth chisq
    chisq_median = np.median(chisq_smooth)
    chisq_cut *= chisq_median

    regions = []
    currentRegion = None
    for i in range(inset, grid.ngrid - 1 - inset):
        above = False
        for k in range(grid.breaks[i], grid.breaks[i+1]):
            if chisq_smooth[k] > chisq_cut:
                above = True
                break
        if above:
            if currentRegion is None:
                currentRegion = Region(lo=i - pad, hi=i + pad)
            else:
                currentRegion.hi = i + pad
        else:
            if currentRegion is not None:
                currentRegion.hi = i + pad
                regions.append(currentRegion)
                currentRegion = None

    # merge overlapping regions
    merged = regions[:1]
    for i in range(1, len(regions)):
        latest = merged[-1]
        if regions[i].lo - latest.hi <= 1:
            latest.hi = regions[i].hi
        else:
            merged.append(regions[i])

    return chisq_median, merged
