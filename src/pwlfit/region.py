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
                window_size: int = 19, poly_order: int = 1, verbose: bool = False
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
    verbose : bool
        If True, print additional information about the regions found.

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
        k1, k2 = grid.breaks[i], grid.breaks[i + 1]
        if np.any(chisq_smooth[k1:k2] > chisq_cut):
            if currentRegion is None:
                currentRegion = Region(lo=i - pad, hi=i + pad)
                if verbose:
                    print(f'Start {currentRegion} at i={i} with smooth chisq ' +
                          f'{np.max(chisq_smooth[k1:k2]):.3f} > {chisq_cut:.3f}')
            else:
                currentRegion.hi = i + pad
        else:
            if currentRegion is not None:
                currentRegion.hi = i + pad
                regions.append(currentRegion)
                if verbose:
                    print(f'End {currentRegion}')
                currentRegion = None

    # merge overlapping regions
    merged = regions[:1]
    for i in range(1, len(regions)):
        latest = merged[-1]
        if regions[i].lo - latest.hi <= 1:
            latest.hi = regions[i].hi
            if verbose:
                print(f'Merging {latest} with {regions[i]}')
        else:
            merged.append(regions[i])

    return chisq_median, chisq_smooth, merged


def splitRegions(regions_in: List[Region], max_knots: int, verbose: bool = False) -> List[Region]:
    """
    Split regions that exceed the maximum number of knots into smaller regions.

    Parameters
    ----------
    regions : List[Region]
        The list of regions to be processed.
    max_knots : int
        The maximum number of knots that can be used in a region. Used to limit the
        time taken to perform a pruned fit, which grows quadratically with the number
        of knots. Regions larger than this size will be split into smaller
        consecutive regions.
    """
    # split any regions with a span > max_knots
    regions = [ ]
    for region in regions_in:
        nknots = region.hi - region.lo + 1
        if nknots <= max_knots:
            regions.append(region)
        else:
            nsplit = int(np.ceil(nknots / max_knots))
            rsize = int(np.round(nknots / nsplit))
            if verbose:
                print(f'Splitting [{region.lo},{region.hi}] into {nsplit} regions of size {rsize}')
            new_regions = [ ]
            for i in range(nsplit - 1):
                ilo = region.lo + i * rsize
                ihi = region.lo + (i + 1) * rsize
                new_regions.append(Region(lo=ilo, hi=ihi))
            # Ensure the last region captures all remaining points
            new_regions.append(Region(lo=new_regions[-1].hi, hi=region.hi))
            # Replace the original region with the new regions
            regions.extend(new_regions)
    return regions


def insertKnots(i1: int, i2: int, max_span: int, verbose: bool = False) -> List[int]:
    """Insert knots to fit the smooth trend outside of any region.
    """
    iknots = [ ]
    ninsert = int(np.ceil((i2 - i1) / max_span)) - 1
    if ninsert > 0:
        iknots = [ i1 + (j + 1) * max_span for j in range(ninsert) ]
        if verbose:
            print(f'Inserting {ninsert} knots {iknots} into [{i1},{i2}] with max_span {max_span}')
        return iknots
    else:
        return [ ]


def combineRegions(regions: List[Region], grid: pwlfit.grid.Grid,
                   max_spacing_factor: int = 9, verbose: bool = False) -> List[int]:
    """Combine regions into a list of knots for the final fit.
    """
    max_span = int(np.floor((grid.ngrid - 1 ) / (max_spacing_factor - 1)))
    if verbose:
        print(f'Combining {len(regions)} regions with max_span {max_span}')

    iknots = [ ]
    if regions[0].lo > 0:
        iknots.append(0)  # Always start with the first knot at the beginning of the grid
        iknots.extend(insertKnots(0, regions[0].lo, max_span, verbose))

    for iregion, region in enumerate(regions):
        ilo, ihi = region.lo, region.hi
        if iregion > 0:
            # Insert knots for continuum fit before this region, if necessary
            iknots.extend(insertKnots(iprev, ilo, max_span, verbose))
        # Add the pruned knots for this region
        region_knots = region.fit.iknots if region.fit is not None else np.arange(ilo, ihi + 1)
        if verbose:
            print(f'Adding {len(region_knots)} knots {region_knots} for region {iregion} [{ilo},{ihi}]')
        if len(iknots) > 0 and region_knots[0] == iknots[-1]:
            # First knot might be duplicated if this region was split earlier
            region_knots = region_knots[1:]
        iknots.extend(region_knots)
        iprev = ihi
    # Insert knots for continuum fit after the last region, if necessary
    ilast = grid.ngrid - 1
    if iprev < ilast:
        iknots.extend(insertKnots(iprev, ilast, max_span, verbose))
        iknots.append(ilast) # Always end with the final knot at the end of the grid

    return iknots
