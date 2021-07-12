#!/usr/bin/env python
import numpy as np
import pandas as pd
from glob import glob

def combine_infield_likelihoods(infield_files):
    infield_likelihoods = np.array([np.loadtxt(field_file) for field_file
                                  in infield_files])
    likelihoods = np.sum(infield_likelihoods, axis=0)
    return likelihoods

def lnposterior_allsky(likelihoods, P_f, P_A, P_T):
    return np.log((likelihoods + (1 - np.sum(P_f)))*P_A + P_T)

def lnposterior_covfrac(likelihoods, P_f, cov_frac, P_A, P_T):
    return np.log((likelihoods*cov_frac/np.sum(P_f) + (1 - cov_frac))*P_A +\
                 P_T)

def combine_fields(infield_files, samples, P_f, P_A, coverage_fraction,
                  output_file):
    likelihoods = combine_infield_likelihoods(infield_files)

    if coverage_fraction:
        ln_posterior = lnposterior_covfrac(likelihoods, P_f, coverage_fraction,
                                          P_A, 1.- P_A)
    else:
        ln_posterior = lnposterior_allsky(likelihoods, P_f, P_A, 1.- P_A)
    np.savetxt(output_file,np.c_[samples,ln_posterior])
