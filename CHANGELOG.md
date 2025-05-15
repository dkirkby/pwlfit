# CHANGELOG for pwlfit

## Introduction

This is the log of changes to the [pwlfit package](https://github.com/dkirkby/pwlfit).

All notable changes to this project will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - Unreleased

### Fixed
- Fixed problem where a region extending to right edge is missed in findRegions.a

### Changed
- Use sigma clipped mean of (un-smoothed) coarse fit chisq instead of median of smoothed chisq in findRegions.

### Added
- Added regions config params scaled_cut and clip_nsigma.

## [0.2.0] - 2025-04-05

### Added
- Release workflow
- asdict() methods to Grid and FitResult
### Changed
- Add link to README for running Quickstart notebook via colab
### Fixed
- Several edge case bugs

## [0.1.0] - 2025-04-01

Initial migration from a jupyter notebook to a repo.
