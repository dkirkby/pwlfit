import pathlib
import json
import argparse
import dataclasses
from typing import Generator, Union, Tuple

import numpy as np
from numpy.typing import NDArray

import fitsio

from pwlfit.grid import Grid
from pwlfit.driver import PWLinearFitConfig, PWLinearFitter

# Define types used below
FloatArray = NDArray[Union[np.float32, np.float64]]
GenTuple = Tuple[FloatArray, FloatArray, dict]
GenType = Generator[GenTuple, None, None]

# Define the wavelength grid used for DESI spectral reductions
wmin, wmax, wdelta = 3600, 9824, 0.8
fullwave = np.round(np.arange(wmin, wmax + wdelta, wdelta), 1)
cslice = {'b': slice(0, 2751), 'r': slice(2700, 5026), 'z': slice(4900, 7781)}


def getReduxGrid(ngrid: int = 2049, transform='log') -> Grid:
    return Grid(fullwave, ngrid=ngrid, transform=transform)


def coaddedExposureGenerator(
        night: int, expid: int,
        path: Union[str, pathlib.Path],
        max_zero_run: int = 400, verbose: bool = False
        ) -> GenType:
    """
    Generator for coadded DESI per-exposure spectra.

    Parameters
    ----------
    night : int
        DESI night number in the format YYYYMMDD.
    expid : int
        DESI exposure ID.
    path : str or Path
        Path to the coadded spectra with a {night}/{expid} structure.
        Use /global/cfs/cdirs/desi/spectro/redux/{RELEASE}/exposures
        for release data at NERSC.
    max_zero_run : int
        Maximum length of consecutive zero ivars allowed before spectrum
        is rejected.
    verbose : bool
        Print debug information when True.

    Yields
    ------
    flux : FloatArray
        Coadded spectrum flux.
    ivar : FloatArray
        Coadded spectrum inverse variance.
    meta : dict
        Metadata for the spectrum.
    """
    try:
        import fitsio
    except ImportError:
        raise RuntimeError("fitsio is required by coaddedExposureGenerator()")

    # Initialize coaddition result
    nwave = len(fullwave)
    wsum = np.zeros((500, nwave))
    wflux = np.zeros((500, nwave))

    if not path.exists():
        raise RuntimeError(f'Invalid path: {path}')
    exptag = str(expid).zfill(8)
    path = path / str(night) / exptag
    if not path.exists():
        raise RuntimeError(f'invalid night/expid path: {path}')

    # Loop over spectrographs
    for spec in range(10):
        wsum.fill(0)
        wflux.fill(0)
        fibermap = None
        # Loop over camera bands to coadd
        for band in 'brz':
            name = path / f'cframe-{band}{spec}-{exptag}.fits'
            if not name.exists():
                if verbose:
                    print(f'Ignoring missing {name}')
                continue
            with fitsio.FITS(name) as hdus:
                if fibermap is None:
                    fibermap = hdus['FIBERMAP'].read()
                flux = hdus['FLUX'].read()
                ivar = hdus['IVAR'].read()
            wflux[:, cslice[band]] += flux * ivar
            wsum[:, cslice[band]] += ivar

        if fibermap is None:
            continue

        # Perform coaddition over bands
        flux = np.divide(wflux, wsum, out=np.zeros_like(wflux), where=wsum > 0)

        objtype = fibermap['OBJTYPE']
        fiber = fibermap['FIBER']
        tgtid = fibermap['TARGETID']
        nzero = np.array([ longest_zero_run(wsum[i]) for i in range(len(fiber)) ])

        sel = (objtype == 'TGT') & (nzero < max_zero_run)
        if verbose:
            print(f'Found {np.sum(sel)} targets from {night}/{exptag} spec {spec}')
        if np.any(sel):
            for idx in np.where(sel)[0]:
                meta = dict(night=night, expid=expid,
                            spec=spec, tgtid=tgtid[idx], fiber=fiber[idx])
                yield flux[idx], wsum[idx], meta


def fitSpectra(generator: GenType, config: PWLinearFitConfig, grid: Grid,
               savepath: Union[str, pathlib.Path],
               metadata: dict = { }, overwrite: bool = False) -> int:
    """
    Fit DESI spectra using the provided generator and metadata.
    """
    savepath = pathlib.Path(savepath)
    if savepath.exists() and not overwrite:
        return 0

    fitter = PWLinearFitter(grid, config)

    nfit = 0
    result = metadata.extend(dict(
        # record the grid used
        grid=grid.asdict(),
        # record the config used
        config=dataclasses.asdict(config),
        # empty array for individual fit results
        fits=[ ],
    ))

    for flux, ivar, spec_meta in generator:
        if flux.ndim != 1 or flux.shape != ivar.shape:
            raise ValueError("flux and ivar must be 1D arrays of the same shape")
        if config.options.verbose:
            print(f'Fitting spectrum [{idx}] with {spec_meta}')
        try:
            fit = fitter(flux[idx], ivar[idx])
            # Calculate the mean flux from the coarse fit
            F0 = np.mean(fitter.coarse_fit.yknots)
            # Normalize the final fluxes to F0
            Fratio = fit.yknots / F0
            # Save the fit result
            result['fits'].append(dict(
                meta=spec_meta,
                F0=round(F0, 5),
                elapsed=round(1e3*fitter.elapsed,1),
                chisq_median=round(fitter.chisq_median,3),
                nregions=len(fitter.regions),
                iknots=fit.iknots.tolist(),
                F=np.round(1000 * Fratio, 0).astype(int).tolist(),
            ))
        except Exception as e:
            print(f"Error fitting {spec_meta}: {e}")

    with open(savepath, 'w') as f:
        if config.options.verbose:
            print(f"Writing {len(result['fits'])} fits to {savepath}")
        json.dump(result, f, separators=(',', ':'))

    return len(result['fits'])
