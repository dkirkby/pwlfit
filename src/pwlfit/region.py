from typing import Tuple, List, Union
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

import pwlfit.fit
import pwlfit.grid

import scipy.signal


@dataclass
class Region:
    lo: int
    hi: int
    fit: Union[None, pwlfit.fit.FitResult] = None


def findRegions(fit: pwlfit.fit.FitResult, grid: pwlfit.grid.Grid,
                inset: int = 4, pad: int = 3, chisq_cut: float = 4,
                window_size: int = 19, poly_order: int = 1
                ) -> Tuple[float, NDArray[np.float64], List[Region]]:
    """
    Find regions of the fit where the chisq is above a threshold that can be analyzed independently.

    Parameters
    ----------
    fit : FitResult
        The result of a coarse fit to the data that captures the overall smooth structure.
        Must have a valid chisq attribute.
    grid : Grid
        The grid of possible knot locations to use for finding regions.
    inset : int
        The number of grid points to ignore at the beginning and end of the grid.
    pad : int
        The number of grid points to add to the beginning and end of each region.
    chisq_cut : float
        The threshold for the chisq above which a region is considered significant.
        Will be scaled by the median of the smoothed chisq.
    window_size : int
        The size of the Savitzky-Golay filter window to smooth the chisq. Must be odd.
    poly_order : int
        The order of the Savitzky-Golay filter polynomial to smooth the chisq.

    Returns
    -------
    float
        The median of the smoothed chisq.
    NDArray[np.float64]
        The smoothed array of chisq values.
    List[Region]
        The list of regions where the chisq is above the threshold.
    """
    if fit.chisq is None:
        raise ValueError('FitResult must have a valid chisq. Did you forget fit=True?')
    if 2 * inset >= grid.ngrid:
        raise ValueError(f'Invalid inset value {inset}')
    if pad < 0 or 2 * pad >= grid.ngrid:
        raise ValueError(f'Invalid pad value {pad}')
    if window_size % 2 == 0 or window_size < 0:
        raise ValueError(f'Invalid window size {window_size} (should be an odd integer > 0)')

    # Inset must be at least big enough for the padding
    inset = max(inset, pad)

    # Smooth chisq with Savitzky-Golay filter
    chisq_smooth = scipy.signal.savgol_filter(fit.chisq, window_size, poly_order)

    # Normalize the cut to the median smooth chisq
    chisq_median = np.median(chisq_smooth)
    chisq_cut *= chisq_median

    # Build list of consecutive knots where the smooth chisq exceeds the cut,
    # with padding added
    regions = []
    currentRegion = None
    for i in range(inset, grid.ngrid - 1 - inset):
        if np.any(chisq_smooth[grid.breaks[i]:grid.breaks[i+1]] > chisq_cut):
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

    return chisq_median, chisq_smooth, merged


def insertKnots(i1: int, i2: int, max_span: int, verbose: bool = False) -> List[int]:
    """Insert knots to fit the smooth trend outside of any region.
    """
    iknots = [ ]
    ninsert = int(np.floor((i2 - i1 + 1) / max_span))
    if ninsert > 0:
        spacing = int(np.round((i2 - i1 + 1) / (ninsert + 1)))
        #iknots = spacing * (1 + np.range(ninsert)) + i1
        iknots = [ i1 + (j + 1) * spacing for j in range(ninsert) ]
        if verbose:
            print(f'Inserting {len(iknots)} knots {iknots} to fit continuum over [{i1},{i2}]')
        return iknots
    else:
        return [ ]


def combineRegions(regions: List[Region], grid: pwlfit.grid.Grid,
                   max_spacing_factor: int = 9, verbose: bool = False) -> List[int]:
    """Combine regions into a list of knots for the final fit.
    """
    iknots = [ 0 ]
    iprev = 0
    max_span = int(np.floor((grid.ngrid - 1 ) / (max_spacing_factor - 1)))
    if verbose:
        print(f'Combining {len(regions)} regions with max_span {max_span}')
    for iregion, region in enumerate(regions):
        ilo, ihi = region.lo, region.hi
        # Insert knots for continuum fit before this region, if necessary
        iknots.extend(insertKnots(iprev, ilo, max_span, verbose))
        # Add the pruned knots for this region
        region_knots = region.fit.iknots
        if verbose:
            print(f'Adding {len(region_knots)} knots {region_knots} for region {iregion} [{ilo},{ihi}]')
        iknots.extend(region_knots)
        iprev = ihi
    # Insert knots for continuum fit after the last region, if necessary
    ilast = grid.ngrid - 1
    iknots.extend(insertKnots(iprev, ilast, max_span, verbose))
    iknots.append(ilast)

    return iknots
