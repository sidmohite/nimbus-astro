#!/usr/bin/env python
import numpy as np
from astropy.time import Time
import pandas as pd
from nimbus import skymap_utils as sky_utils
import argparse
from glob import glob

def main(survey_file, skymap_file, field_probs_file, infield_likelihoods_path,
        common_str):
    df_survey = pd.read_pickle(survey_file)

    infield_likelihood_files = np.sort(glob(infield_likelihoods_path))
    field_probs = np.zeros(len(infield_likelihood_files))
    skymap_prob = sky_utils.Skymap_Probability(skymap_fits_file=skymap_file)
    for i,field_file in enumerate(infield_likelihood_files):
        field_num = int(field_file.split(common_str)[1].split('.')[0])
        ipix_field = df_survey.ipix.values[df_survey.field_ID==field_num][0]
        field_probs[i] = skymap_prob.calculate_field_prob(ipix_field)

    np.savetxt(field_probs_file,field_probs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Calculate the field\
                                    probabilities and store them in file.',
                                    formatter_class =\
                                    argparse.ArgumentDefaultsHelpFormatter,
                                    fromfile_prefix_chars='@')
    parser.add_argument('--field_probs_file', required = True,
                       metavar = 'FIELD_PROB_FILE',
                       help = 'File to save the field probabilities in.')
    parser.add_argument('--survey_file', required = True,
                       metavar = 'SURVEY_FILE', help = 'File containing field,\
                       pixel and extinction specific information for the\
                       survey.')
    parser.add_argument('--skymap_file', required = True,
                       metavar = 'SKYMAP_FILE', help = 'Skymap file for the\
                       event.')
    parser.add_argument('--infield_likelihoods_path', required = True,
                       metavar = 'INFIELD_LIKELIHOODS_PATH',
                       help = 'Path to files containing sample likelihood\
                       values for each field. The code expects files to be\
                       named using a common string pattern (see below) with\
                       the field number appended at the end.')
    parser.add_argument('--common_str', required = True,
                       metavar = 'COMMON_STR',
                       help = 'The common string pattern (see below) the code\
                       expects the files to be named with.')
    args = parser.parse_args()
    survey_file = args.survey_file
    infield_likelihoods_path = args.infield_likelihoods_path
    skymap_file = args.skymap_file
    common_str = args.common_str
    field_probs_file = args.field_probs_file
    main(survey_file, skymap_file, field_probs_file, infield_likelihoods_path,
        common_str)
