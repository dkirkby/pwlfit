# Piecewise Linear Fits to Noisy Data

[![PyPI package](https://img.shields.io/badge/pip%20install-pwlfit-brightgreen)](https://pypi.org/project/pwlfit/) [![GitHub Release](https://img.shields.io/github/v/release/dkirkby/pwlfit?color=green)](https://github.com/dkirkby/pwlfit/releases) [![Actions Status](https://github.com/dkirkby/pwlfit/workflows/Test/badge.svg)](https://github.com/dkirkby/pwlfit/actions) [![License](https://img.shields.io/github/license/dkirkby/pwlfit)](https://github.com/dkirkby/pwlfit/blob/main/LICENSE)

Use this package to perform efficient piecewise linear fits to noisy data.

## Installation

Install the [latest released version](https://github.com/dkirkby/pwlfit/releases/latest) from [pypi](https://pypi.org/project/pwlfit/) using:
```
pip install pwlfit
```
The required dependencies are numpy, scipy, yaml.

The changes with each version are documented [here](CHANGELOG.md).

## Quick Start

Fit some sample data included in this package using:
```
from pwlfit.util import read_sample_data
from pwlfit.grid import Grid
from pwlfit.driver import PWLinearFitter

x, y, ivar = read_sample_data('A')
grid = Grid(x, ngrid=100)

fitter = PWLinearFitter(grid)

fit = fitter(y, ivar)

plt.plot(x, y, '.')
plt.plot(fit.xknots, fit.yknots, 'o-');
```
to produce this plot:

![Fit to sample A data](https://github.com/dkirkby/pwlfit/blob/main/examples/Quickstart-sampleA.png?raw=true)

For more details see the Quickstart notebook: [readonly](https://github.com/dkirkby/pwlfit/blob/main/examples/Quickstart.ipynb) or [live (via google colab)](https://colab.research.google.com/github/dkirkby/pwlfit/blob/main/examples/Quickstart.ipynb).
