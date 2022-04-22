"""
A module for handling skymaps and associated utilities.

Classes:

    Skymap_Probability
"""

__author__ = 'Siddharth Mohite'

import numpy as np
from scipy.stats import norm, truncnorm
from scipy.integrate import quad
import healpy as hp


class Skymap_Probability():
    """
    Ingests skymaps to acquire marginal distance distributions.

    Attributes
    ----------
    skymap_file : str
        Path to the fits.gz skymap file for an event.
    nside : int
        Number (a power of 2) representing the resolution of the skymap.
    prob : array
        Array of probabilities for every pixels in the skymap.
    distmu : array
        Array of mean distances (Mpc) of the marginalised distance
        distribution for each pixel.
    distsigma : array
        Array of standard deviations (Mpc) of the marginalised distance
        distribution for each pixel.
    distnorm : array
        Array of normalization factors for the marginalised distance
        distribution for each pixel.

    Usage
    -----
    skymap_prob = Skymap_Probability(skymap_fits_file)
    """

    def __init__(self, skymap_fits_file):
        """
        Instantiates class that handles skymap probability.

        Parameters:
        -----------
        skymap_fits_file : str
            Path to the fits.gz skymap file for an event.
        """
        print("Ingesting skymap:"+skymap_fits_file)
        self.skymap_file = skymap_fits_file
        prob, distmu, distsigma, distnorm = hp.read_map(self.skymap_file,
                                                       field=range(4))
        npix = len(prob)
        self.nside = hp.npix2nside(npix)
        self.prob = hp.ud_grade(prob, 256)
        self.distmu = hp.ud_grade(distmu, 256)
        self.distsigma = hp.ud_grade(distsigma, 256)
        self.distnorm = hp.ud_grade(distnorm, 256)

    def calculate_field_prob(self, ipix_field):
        """
        Returns the total probability contained within each field.

        Parameters:
        ----------
        ipix_field : array
            Array of pixel indices contained in field.
        """
        return self.prob[ipix_field].sum()

    def construct_margdist_distribution(self, ipix_field, field_prob):
        """
        Returns the approximate probability density for distance marginalised
        over pixels in a field.

        Parameters
        ----------
        ipix_field : array
            Array of pixel indices contributing to each field.
        field_prob : array
            Array of total field probabilites.

        Returns
        -------
        approx_dist_pdf : scipy.stats.rv_continuous.pdf object
            The probability density function (pdf) of the distance over
            the given field, approximated as a normal distribution.
        """
        dp_dr = lambda r: np.sum(
            self.prob[ipix_field] * r**2 * self.distnorm[ipix_field] *
            norm(self.distmu[ipix_field], self.distsigma[ipix_field]).pdf(r))\
            / field_prob
        mean = quad(lambda x: x * dp_dr(x), 0, np.inf)[0]
        sd = np.sqrt(quad(lambda x: x**2 * dp_dr(x), 0, np.inf)[0] - mean**2)
        dmin = np.maximum(1,mean - 5*sd)
        dmax = mean + 5*sd
        approx_dist_pdf = truncnorm((dmin-mean)/sd, (dmax-mean)/sd, mean,
                                   sd).pdf
        return approx_dist_pdf
