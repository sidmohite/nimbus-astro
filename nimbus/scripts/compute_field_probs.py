#!/usr/bin/env python
import numpy as np
import pandas as pd
from nimbus import skymap_utils as sky_utils
from glob import glob

def compute_field_probs(survey_file, skymap_file, field_probs_file,
                       infield_likelihoods_path, common_str):
    df_survey = pd.read_pickle(survey_file)
    infield_likelihood_files = np.sort(glob(infield_likelihoods_path))
    field_probs = np.zeros(len(infield_likelihood_files))
    skymap_prob = sky_utils.Skymap_Probability(skymap_fits_file=skymap_file)
    for i,field_file in enumerate(infield_likelihood_files):
        field_num = int(field_file.split(common_str)[1].split('.')[0])
        ipix_field = df_survey.ipix.values[df_survey.field_ID==field_num][0]
        field_probs[i] = skymap_prob.calculate_field_prob(ipix_field)

    np.savetxt(field_probs_file,field_probs)
