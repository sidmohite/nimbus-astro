#!/usr/bin/env python
import numpy as np
import pandas as pd
import argparse
from glob import glob

parser = argparse.ArgumentParser(description = 'Combine the in-field\
                                likelihoods to construct the final posterior',
                                formatter_class =\
                                argparse.ArgumentDefaultsHelpFormatter,
                                fromfile_prefix_chars='@')
parser.add_argument('--sample_file', required = True, metavar = 'SAMPLE_FILE',
                   help = 'File containing sample points')
parser.add_argument('--field_probs_file', required = True,
                   metavar = 'FIELD_PROB_FILE',
                   help = 'File containing the total sky probability for each\
                   field.')
parser.add_argument('--infield_likelihoods_str', required = True,
                   metavar = 'INFIELD_LIKELIHOODS_STR',
                   help = 'Common string for files containing sample\
                   likelihood values for each field. The code expects files to\
                   be named using a common string pattern (see below) with the\
                   field number appended at the end.')
parser.add_argument('--coverage_fraction', type = float, default=0,
                   metavar = 'COV_FRAC', help = 'Assumed pseudo fraction of\
                   the event skymap that is surveyed by the telescope.\
                   Range(0-1)')
parser.add_argument('--P_A', required = True, type = float,
                   metavar = 'P_ASTRO', help = 'Probability of the event being\
                   astrophysical.Range(0-1)')
parser.add_argument('--output_file', required = True, metavar = 'OUTPUT_FILE',
                   help = 'Output file.')

args = parser.parse_args()
P_A = args.P_A
coverage_fraction = args.coverage_fraction
output_file = args.output_file

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

def main(likelihoods, P_f, P_A, coverage_fraction, samples, output_file):
    if coverage_fraction:
        ln_posterior = lnposterior_covfrac(likelihoods, P_f, coverage_fraction,
                                          P_A, 1.- P_A)
    else:
        ln_posterior = lnposterior_allsky(likelihoods, P_f, P_A, 1.- P_A)
    np.savetxt(output_file,np.c_[samples,ln_posterior])

if __name__ == "__main__":
    infield_files = np.sort(glob(args.infield_likelihoods_str+'*.txt'))
    samples = np.loadtxt(args.sample_file)
    likelihoods = combine_infield_likelihoods(infield_files)
    P_f = np.loadtxt(args.field_probs_file)
    main(likelihoods, P_f, P_A, coverage_fraction, samples, output_file)
