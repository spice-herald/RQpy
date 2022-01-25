# RQpy Legacy Package

This package has been archived and its constituent parts have been moved to various new homes:
 - Event triggering and IV/dIdV processing: [**`pytesdaq`**](https://github.com/berkeleytes/pytesdaq)
 - Feature extraction and post-analysis: [**`detprocess`**](https://github.com/slwatkins/detprocess)
 - Dark matter limit setting and sensitivity: [**DarkLim**](https://github.com/slwatkins/DarkLim)

Development has ceased on this repository, please check out above repositories for any future versions this package's functionality. The old README text is shown below.

-----------

RQpy (Reduced Quantities) provides helpful tools for dark matter search related analysis. It contains submodules for processing, energy calibration, plotting, automated cut routines, and many other useful DM analysis tools. This repository is tailored specifically towards creating an efficient workflow for dark matter search analysis using detectors studied by the Pyle Group, and it may not generalize well to research using any detector.

To install the most recent development version of RQpy, clone this repo, then from the top-level directory of the repo, type the following line into your command line

`pip install .`

If using a shared Python installation, you may want to add the `--user` flag to the above line.

This package requires python 3.6 or greater. Use of the much of the functionality in the `io` and `process` submodules requires an installation of the CDMS internal IO package `rawio`.

This package also requires a Fortran compiler for installation. We recommend installing `gfortran`, which can be done via `sudo apt-get install gfortran` on Linux.
