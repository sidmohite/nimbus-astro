# nimbus - A Bayesian inference framework to constrain kilonovae models
nimbus is a hierarchical Bayesian framework to infer the intrinsic luminosity parameters
of kilonovae (KNe) associated with gravitational-wave (GW) events, based purely on non-detections.
This framework makes use of GW 3-D distance information and electromagnetic upper limits from
a given survey for multiple events, and self-consistently accounts for finite sky-coverage and prob-
ability of astrophysical origin.

## Installation
nimbus can be installed by cloning this repo:

    git clone git@github.com:sidmohite/nimbus-astro.git
    
    cd nimbus-astro
    
    python setup.py install
    
Note : For best results, installation should be done into a virtual Python/Anaconda environment.

## Data Inputs
In order to use nimbus to constrain kilonova models we first need to ensure we have the relevant
data files:

* A `survey file` containing field, pixel and extinction specific information for the survey.
    * Currently the code expects this file to exist as a Python pickle  - `.pkl` file.
      The file should contain 3 attributes/columns at the very least - `field_ID` (ID specifying which field),
      `ebv` (E(B-V) extinction value along the line-of-sight of the field) and `A_lambda` (the total extinction
      in a given passband lambda)
* A `data file` containing all observational data for the event(s) from the survey including upper limits for each 
observed field and passband filter as well as associated observation times.
    * The code expects this file to be in `csv` or `txt` format.
* A `skymap file` containing the 3-D GW skymap localization information.

