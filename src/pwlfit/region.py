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


def insertKnots(i1: int, i2: int, ninsert: int = 0, max_span: int = 0,
                verbose: bool = False) -> List[int]:
    """Insert knots into [i1,i2] that are equally spaced. The number of knots to
    insert is either specified by ninsert or calculated from the max_span.

    Parameters
    ----------
    i1 : int
        The starting index of the region.
    i2 : int
        The ending index of the region.
    ninsert : int
        The number of knots to insert. If 0, the number is calculated from the max_span.
    max_span : int
        The maximum span between knots. Ignored if ninsert is specified.
    verbose : bool
        If True, print additional information about the knots being inserted.

    Returns
    -------
    List[int]
        A list of knot indices to be inserted into the region (not including i1 and i2).
    """
    iknots = [ ]
    if ninsert <= 0 and max_span <= 0:
        raise ValueError('Either ninsert or max_span must be > 0')
    if ninsert == 0 and i2 > i1:
        # Calculate the number of knots to insert based on the max_span
        ninsert = int(np.ceil((i2 - i1) / max_span)) - 1
    if ninsert > 0:
        # Calculate the floating point spacing between knots
        delta = (i2 - i1) / (ninsert + 1)
        # Round each inserted knot to its nearest integer
        iknots = [ i1 + int(np.round((j + 1) * delta)) for j in range(ninsert) ]
        if verbose:
            print(f'Inserting {ninsert} knots {iknots} into [{i1},{i2}] with max_span {max_span}')
    return iknots


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
            breaks = insertKnots(region.lo, region.hi, max_span=max_knots, verbose=verbose)
            new_regions = [ ]
            iprev = region.lo
            for i in breaks:
                new_regions.append(Region(lo=iprev, hi=i))
                iprev = i
            # Ensure the last region captures all remaining points
            new_regions.append(Region(lo=new_regions[-1].hi, hi=region.hi))
            if verbose:
                print(f'Split [{region.lo},{region.hi}] at {breaks} into {new_regions}')
            # Replace the original region with the new regions
            regions.extend(new_regions)
    return regions


def combineRegions(regions: List[Region], grid: pwlfit.grid.Grid,
                   max_spacing_factor: int = 9, min_knots: int = 3,
                   verbose: bool = False) -> NDArray[np.int64]:
    """Combine regions into a list of knots for the final fit.

    Parameters
    ----------
    regions : List[Region]
        The list of regions to be combined. If a region has a fit attribute, only
        the knots from that fit will be used. Otherwise, all knots in the region
        will be used.
    grid : Grid
        The grid of possible knot locations to use for the final fit. Only the ngrid
        attribute is used.
    max_spacing_factor : int
        The maximum spacing factor between knots. This is used to determine the
        maximum span between knots when inserting new knots, calcualted as
        int(floor((ngrid - 1 ) / (max_spacing_factor - 1))). For example, ngrid=101
        and max_spacing_factor=11 gives a max_span of 10.
    min_knots : int
        The minimum number of knots to use in a region. If the number of knots
        in a region is less than this value, additional knots will be inserted.
    verbose : bool
        If True, print additional information about the regions being combined.

    Returns
    -------
    NDArray[np.int64]
        A list of knot indices for the final fit. The first and last knots of
        the grid are always included.
    """
    max_span = int(np.floor((grid.ngrid - 1 ) / (max_spacing_factor - 1)))
    if verbose:
        print(f'Combining {len(regions)} regions with max_span {max_span}')

    if len(regions) == 0:
        return np.array(
            insertKnots(0, grid.ngrid - 1, max_span=max_span, verbose=verbose),
            dtype=np.int64)

    iknots = [ ]
    if regions[0].lo > 0:
        iknots.append(0)  # Always start with the first knot at the beginning of the grid
        iknots.extend(insertKnots(0, regions[0].lo, max_span=max_span, verbose=verbose))

    for iregion, region in enumerate(regions):
        ilo, ihi = region.lo, region.hi
        if iregion > 0:
            # Insert knots for continuum fit before this region, if necessary
            iknots.extend(insertKnots(iprev, ilo, max_span=max_span, verbose=verbose))
        # Add the pruned knots for this region
        region_knots = region.fit.iknots if region.fit is not None else np.arange(ilo, ihi + 1)
        # Ensure at least min_knots are used in the region if possible
        if len(region_knots) < min_knots and ihi - ilo >= min_knots:
            added_knots = insertKnots(ilo, ihi, ninsert=min_knots - 2, verbose=verbose)
            region_knots = np.array([ilo] + added_knots + [ihi])
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
        iknots.extend(insertKnots(iprev, ilast, max_span=max_span, verbose=verbose))
        iknots.append(ilast) # Always end with the final knot at the end of the grid

    return np.array(iknots, dtype=np.int64)
