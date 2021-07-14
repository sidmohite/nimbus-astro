"""
The main module of nimbus that sets up the Bayesian formalism.

Classes:

    Kilonova_Inference
"""

__author__ = 'Siddharth Mohite'

import numpy as np
from scipy.stats import norm, truncnorm
from scipy.integrate import quad
from scipy.special import expit
from multiprocessing import Pool
from functools import partial

class Kilonova_Inference():
    """
    Initializes utility functions for inference and defines the model.

    Attributes
    ----------
    lc_model_funcs : array-like
        The array whose elements are band-specific functions that define the
	light-curve evolution as a function of time.
    nullevent_mlim_pdf : func
        The function that evaluates the pdf for the observed upper limits when
        the event is either not in the observed fields or is terrestrial.

    Usage
    -----
    kne_inf = Kilonova_Inference(lc_model_func)
    """

    def __init__(self, lc_model_funcs, nullevent_mlim_pdf):
        print("Initializing inference framework...")
        self.lc_model_funcs = lc_model_funcs
        self.nbands = len(lc_model_funcs)
        self.nullevent_mlim_pdf = nullevent_mlim_pdf

    def lc_model_powerlaw(self, M_0, gamma, t_0, t):
        """
        Returns the absolute magnitude evolution as a power law.

        Parameters
        ----------
        M_0 : float
            The peak absolute magnitude of the light curve.
        gamma : float
            Power law index for the light curve decay.
        t_0 : float
            Initial time of the event.
        t : float or array
            Array of observation times.

        Returns
        -------
        M : float or array
            Absolute magnitude light curve as a function of time (same shape as
            t).
        """
        return (M_0 * pow(t_0/t, gamma))

    def lc_model_linear(self, M_0, alpha, t_0, t):
        """
        Returns the absolute magnitude evolution as a linear decay/rise.

        Parameters
        ----------
        M_0 : float
            The peak absolute magnitude of the light curve.
        alpha : float
            Linear decay/rise index for the light curve.
        t_0 : float
            Initial time of the event.
        t : float or array
            Array of observation times.

        Returns
        -------
        M : float or array
            Absolute magnitude light curve as a function of time (same shape as
            t).
        """
        return M_0 + alpha*(t-t_0)

    def M_to_m(self, M, distance):
        """
        Returns the apparent magnitude using a distance and absolute
        magnitude.

        Parameters
        ----------
        M : float or array
            Absolute magnitude of object.
        distance : float or array
            Distance of the object (must have same size as M).

        Returns
        -------
        m : float or array
            Apparent magnitude of the object (same size as M or distance).
        """
        return (M + 5 * np.log10(distance * 1e6) - 5)

    def dlim(self, mlim, M):
        """
        Returns the limiting distance for a model with absolute magnitude M
        and limiting magnitude mlim.

        Parameters
        ----------
        mlim : float or array
            Limitng magnitude from observations.
        M : float or array
            Absolute magnitude from model (must have same shape as mlim).

        Returns
        -------
        dlim : float or array (same shape as mlim)
            Limiting distance for given parameters.
        """
        return 10**((mlim - M)/5.) * 10 * 1e-6

    def create_distance_dist(self, mu_f, sigma_f):
        """
        Returns a truncated normal distribution as the distance distribution.

        Parameters
        ----------
        mu_f : float
            Mean of the distance distribution.
        sigma_f : float
            Standard deviation of the distance distribution.

        Returns
        -------
        distance_dist : scipy.stats.rv_continuous.pdf object
            The probability density function of the truncated normal
            distribution.
        """
        #set min,max distances as 0 Mpc, 4000 Mpc
        a = (0. - mu_f)/sigma_f
        b = (4000. - mu_f)/sigma_f
        return truncnorm(a, b, mu_f, sigma_f)

    def calc_expit_argument(self,d_lim,maglim_err=0.1):
        """
        Returns a logistic/expit function that accounts for errors in the
        measurement of limiting magnitudes.

        Parameters
        ----------
        d_lim : float
            Limiting distance corresponding to the observed limiting
            magnitude.
        maglim_err : float
            Error in the limiting magnitude measurement (default=0.1 mag).

        Returns
        -------
        expit_func : func
            Logitic function based on errors in the limiting magnitude.
        """
        if maglim_err==0.:
            maglim_err = 0.1
        dlow = d_lim*10**-(3*maglim_err/5) # set dlow at 3-sigma
        dmid = d_lim*10**-(maglim_err/5) # set dmid at 1-sigma
        a = np.log(0.021/0.979)/(dlow - dmid)
        b = -1.0*dmid
        return lambda x : expit(a*(x + b))

    def calc_likelihood_integral(self, M, expit_func, dist_samples,
                                mlow, mhigh):
        """
        Returns the single observation likelihood integral evaluated using
        posterior samples drawn from the distance distribution.
        """
        dist_samples_survey = dist_samples[(dist_samples>self.dlim(mlow,M))
                            &(dist_samples<=self.dlim(mhigh,M))]
        dist_samples_high = dist_samples[dist_samples>self.dlim(mhigh,M)]
        N_samples_survey = len(dist_samples_survey)
        N_samples_high = len(dist_samples_high)
        N_total = N_samples_survey + N_samples_high
        if (N_samples_survey==0)&(N_samples_high!=0):
            return 1./(mhigh-mlow)
        elif (N_samples_survey!=0)&(N_samples_high==0):
            return np.sum((1./(
                         np.vectorize(
                         self.M_to_m)(M, dist_samples_survey) -mlow))*\
                         np.vectorize(expit_func)(dist_samples_survey))/\
                         N_samples_survey
        elif (N_samples_survey!=0)&(N_samples_high!=0):
            return N_samples_survey/N_total * np.sum((1./(
                   np.vectorize(self.M_to_m)(M, dist_samples_survey) - mlow))*\
                   np.vectorize(expit_func)(dist_samples_survey)) +\
                   (N_samples_high/N_total) * (1./(mhigh-mlow))
        return 0.

    def create_dlim_pdf(self, M, d_lim, maglim_err, norm_factor, p_d, d_min,
                       d_max):
        """
        Returns the likelihood of a single observation for a given field and
        model, under the astrophysical hypothesis and using distance limits.

        Parameters
        ----------
        M : float
            The absolute magnitude of the model.
        d_lim : float
            The observed limiting distance below which non-detection is
            invalid.
        maglim_err : float
            Error in limiting magnitude measurement.
        norm_factor : float
            Pre-computed normalization factor for the likelihood.
        p_d : func
            The probability density function (pdf) of the distance.
        d_min : float
            Lower limit of the distance distribution.
        d_max : float
            Upper limit of the distance distribution.

        Returns
        -------
        dlim_pdf : float
            The likelihood of obtaining the observed limiting magnitude
            given model absolute magnitude M.
        """
        expit_num = self.calc_expit_argument(d_lim, maglim_err)
        num = quad(lambda d : (1./(self.M_to_m(M,d)-self.M_to_m(M,d_min)))*\
                  expit_num(d)*p_d(d), d_min+0.1, d_max)[0]
        return num/norm_factor

    def create_mlim_pdf(self, M, d_lim, maglim_err, p_d, m_low, m_high,
                       eps=0.1, dmax=3000):
        """
        Returns the likelihood of a single observation for a given field and
        model, under the astrophysical hypothesis and using survey upper limits.

        Parameters
        ----------
        M : float
            The absolute magnitude of the model.
        d_lim : float
            The limiting distance below which non-detection is invalid.
        maglim_err : float
            Error in limiting magnitude measurement.
        p_d : func
            The probability density function (pdf) of the distance.
        m_low : float
            Lower limit of the limiting magnitude distribution.
        m_high : float
            Upper limit of the limiting magnitude distribution.

        Returns
        -------
        mlim_pdf : float
            The likelihood of obtaining the observed limiting magnitude
            given model absolute magnitude M.
        """
        expit_num = self.calc_expit_argument(d_lim, maglim_err)
        num = quad(
            lambda d : (1./(self.M_to_m(M, d) - m_low))*expit_num(d)*p_d(d),
            self.dlim(m_low,M)+eps, self.dlim(m_high,M))[0] +\
            quad(lambda d : (1./(m_high - m_low))*p_d(d), self.dlim(m_high,M),
            dmax)[0]
        den = quad(
            lambda m: quad(
            lambda d : (1./(self.M_to_m(M, d) - m_low))*\
            self.calc_expit_argument(self.dlim(m,M),maglim_err)(d)*\
            p_d(d), self.dlim(m_low,M)+eps,
            self.dlim(m_high,M))[0], m_low, m_high)[0] +\
            quad(lambda d : p_d(d), self.dlim(m_high,M), dmax)[0]
        if den==0.:
            return 0.
        return num/den

    def create_mlim_pdf_fromsamples(self, M, d_lim, maglim_err, dist_samples,
                                   m_low, m_high):
        """
        Returns the likelihood of a single observation for a given field and
        model, under the astrophysical hypothesis using survey limits and
        distance posterior samples.
        """
        expit_num = self.calc_expit_argument(d_lim, maglim_err)
        num = self.calc_likelihood_integral(M, expit_num, dist_samples,
                                           m_low, m_high)
        den = quad(
            lambda m: self.calc_likelihood_integral(M,
            self.calc_expit_argument(self.dlim(m,M), maglim_err), dist_samples,
            m_low, m_high), m_low, m_high)[0]
        if den==0.:
            return 0.
        return num/den

    def calc_infield_filter_dlim_likelihood(
                                           self, params, fid, mlims, t0, T,
                                           p_d_f, maglimerrs, dmin, dmax,
                                           mlow_t, mhigh_t, norm_factors):
        """
        Returns the likelihood of observations for a single field in a single
        filter (filter ID : fid) using distance limits, under the astrophysical
        hypothesis.

        Parameters
        ----------
        params : array-like
            List or array of model parameters for the kilonova light-curve.
        fid : integer
            Filter ID number of the corresponding filter.
        mlims : array
            Array of observed filter-specific limiting magnitudes
            corresponding to the time array T.
        t0 : float
            Initial time of the event.
        T : array
            Array of observation times corrsponding to the array of upper
            limits mlims.
        p_d_f : func
            The field-specific probability density function (pdf) of the
            distance.
        maglimerrs : array
            Array of measurement errors in the limiting magnitudes (mlims).
        dmin : float
            Lower limit of the distance distribution.
        dmax : float
            Upper limit of the distance distribution.
        mlow_t : float
            Lower limit of the limiting magnitude distribution in the
            null-event hypothesis.
        mhigh_t : float
            Upper limit of the limiting magnitude distribution in the
            null-event hypothesis.
        norm_factors : array
            Array of normalization factors for each observation.

        Returns
        -------
        plims_f : float
            Likelihood of observations for a single field in a single
            filter (filter ID : fid) using distance limits, under the
            astrophysical hypothesis.
        """
        M = np.array([self.lc_model_funcs[fid-1](*params, t_0=t0, t=t)\
                    for t in T])
        dlims = np.array(list(map(self.dlim, mlims, M)))
        pool = Pool(processes=2)
        plims_f_t = pool.starmap(partial(self.create_dlim_pdf, p_d=p_d_f,
                                d_min=dmin, d_max=dmax),
                                np.c_[M,dlims,maglimerrs,norm_factors])
        pool.close()
        plims_f_t_nondet = np.array([self.nullevent_mlim_pdf(mlow_t,mhigh_t)\
                                   for m in mlims])
        plims_f = np.product(plims_f_t/plims_f_t_nondet)
        return plims_f

    def calc_infield_dlim_likelihood(
                                    self, params, filter_ids, mlims_array, t0,
                                    filter_obs_times, p_d_f, P_f,
                                    maglimerr_array, dmin, dmax, m_low_t,
                                    m_high_t, norm_factor_array):
        """
        Returns the overall likelihood of observations for a single field using
        distance limits, under the astrophysical hypothesis.

        Parameters
        ----------
        params : array-like
            List of lists or array of arrays of model parameters for the
            kilonova light-curve corresponding to each filter.
        filter_ids : array-like
            Array of Filter ID numbers (integers) as per the survey.
        mlims_array : array-like
            List/Array of arrays with observed filter-specific limiting
            magnitudes corresponding to the time array filter_obs_times.
        t0 : float
            Initial time of the event.
        filter_obs_times : array-like
            List/Array of arrays of observation times corrsponding to the
            array of upper limits mlims (shape same as mlims_array).
        p_d_f : func
            The field-specific probability density function (pdf) of the
            distance.
        P_f : float
            Sky probability of the event being in the given field (f).
        maglimerrs_array : array
            List/Array of arrays with measurement errors in the limiting
            magnitudes (shape same as mlims_array).
        dmin : float
            Lower limit of the distance distribution.
        dmax : float
            Upper limit of the distance distribution.
        m_low_t : float
            Lower limit of the limiting magnitude distribution in the
            null-event hypothesis.
        m_high_t : float
            Upper limit of the limiting magnitude distribution in the
            null-event hypothesis.
        norm_factor_array : array
            List/Array of arrays with normalization factors for each
            observation (shape same as mlims_array).

        Returns
        -------
        likelihood : float
            Overall likelihood of observations for a single field using
            distance limits, under the astrophysical hypothesis.
        """
        plims = np.array([self.calc_infield_filter_dlim_likelihood(
                        params[2*(fid-1):2*fid], fid, mlims_array[i], t0,
                        filter_obs_times[i], p_d_f, maglimerr_array[i], dmin,
                        dmax, m_low_t, m_high_t, norm_factor_array[i])\
                        for i,fid in enumerate(filter_ids)])
        return np.product(plims)*P_f

    def calc_infield_filter_mlim_likelihood(
                                           self, params, fid, mlims, t0, T,
                                           p_d_f, maglimerrs, mlow_a, mhigh_a,
                                           mlow_t, mhigh_t):
        """
        Returns the likelihood of observations for a single field in a single
        filter (filter ID : fid) using survey limits, under the astrophysical
        hypothesis.

        Parameters
        ----------
        params : array-like
            List or array of model parameters for the kilonova light-curve.
        fid : integer
            Filter ID number of the corresponding filter.
        mlims : array
            Array of observed filter-specific limiting magnitudes
            corresponding to the time array T.
        t0 : float
            Initial time of the event.
        T : array
            Array of observation times corrsponding to the array of upper
            limits mlims.
        p_d_f : func
            The field-specific probability density function (pdf) of the
            distance.
        maglimerrs : array
            Array of measurement errors in the limiting magnitudes (mlims).
        mlow_a : float
            Lower limit of the limiting magnitude distribution in the
            astrophysical hypothesis.
        mhigh_a : float
            Upper limit of the limiting magnitude distribution in the
            astrophysical hypothesis.
        mlow_t : float
            Lower limit of the limiting magnitude distribution in the
            null-event hypothesis.
        mhigh_t : float
            Upper limit of the limiting magnitude distribution in the
            null-event hypothesis.

        Returns
        -------
        plims_f : float
            Likelihood of observations for a single field in a single
            filter (filter ID : fid) using survey limits, under the
            astrophysical hypothesis.
        """
        M = np.array([self.lc_model_funcs[fid-1](*params, t_0=t0, t=t)\
                    for t in T])
        dlims = np.array(list(map(self.dlim, mlims, M)))
        pool = Pool(processes=2)
        plims_f_t = pool.starmap(partial(self.create_mlim_pdf, p_d=p_d_f,
                                m_low=mlow_a,m_high=mhigh_a),
                                np.c_[M,dlims,maglimerrs])
        pool.close()
        plims_f_t_nondet = np.array([self.nullevent_mlim_pdf(mlow_t,mhigh_t)\
                                   for m in mlims])
        plims_f = np.product(plims_f_t/plims_f_t_nondet)
        return plims_f

    def calc_infield_mlim_likelihood(
                                    self, params, filter_ids, mlims_array, t0,
                                    filter_obs_times, p_d_f, P_f,
                                    maglimerr_array, m_low_a, m_high_a,
                                    m_low_t, m_high_t):
        """
        Returns the overall likelihood of observations for a single field using
        survey limits, under the astrophysical hypothesis.

        Parameters
        ----------
        params : array-like
            List of lists or array of arrays of model parameters for the
            kilonova light-curve corresponding to each filter.
        filter_ids : array-like
            Array of Filter ID numbers (integers) as per the survey.
        mlims_array : array-like
            List/Array of arrays with observed filter-specific limiting
            magnitudes corresponding to the time array filter_obs_times.
        t0 : float
            Initial time of the event.
        filter_obs_times : array-like
            List/Array of arrays of observation times corrsponding to the
            array of upper limits mlims (shape same as mlims_array).
        p_d_f : func
            The field-specific probability density function (pdf) of the
            distance.
        P_f : float
            Sky probability of the event being in the given field (f).
        maglimerrs_array : array
            List/Array of arrays with measurement errors in the limiting
            magnitudes (shape same as mlims_array).
        m_low_a : float
            Lower limit of the limiting magnitude distribution in the
            astrophysical hypothesis.
        m_high_a : float
            Upper limit of the limiting magnitude distribution in the
            astrophysical hypothesis.
        m_low_t : float
            Lower limit of the limiting magnitude distribution in the
            null-event hypothesis.
        m_high_t : float
            Upper limit of the limiting magnitude distribution in the
            null-event hypothesis.

        Returns
        -------
        likelihood : float
            Overall likelihood of observations for a single field using
            survey limits, under the astrophysical hypothesis.
        """
        plims = np.array([self.calc_infield_filter_mlim_likelihood(
                        params[2*(fid-1):2*fid], fid, mlims_array[i], t0,
                        filter_obs_times[i], p_d_f, maglimerr_array[i],
                        m_low_a, m_high_a, m_low_t, m_high_t)\
                        for i,fid in enumerate(filter_ids)])
        return np.product(plims)*P_f

    def calc_infield_filter_mlim_likelihood_fromsamples(
                                                       self, params, fid,
                                                       mlims, t0, T, d_samples,
                                                       maglimerrs, mlow_a,
                                                       mhigh_a, mlow_t,
                                                       mhigh_t):
        """
        Returns the likelihood of observations for a single field in a single
        filter (filter ID : fid) using survey limits and distance posterior
        samples, under the astrophysical hypothesis.
        """
        M = np.array([self.lc_model_funcs[fid-1](*params, t_0=t0, t=t)\
                    for t in T])
        dlims = np.array(list(map(self.dlim, mlims, M)))
        pool = Pool(processes=2)
        plims_f_t = pool.starmap(partial(self.create_mlim_pdf_fromsamples,
                                dist_samples=d_samples, m_low=mlow_a,
                                m_high=mhigh_a), np.c_[M,dlims,maglimerrs])
        pool.close()
        plims_f_t_nondet = np.array([self.nullevent_mlim_pdf(mlow_t,mhigh_t)\
                                   for m in mlims])
        plims_f = np.product(plims_f_t/plims_f_t_nondet)
        return plims_f

    def calc_infield_mlim_likelihood_fromsamples(
                                                self, params, filter_ids,
                                                mlims_array, t0,
                                                filter_obs_times, d_samples,
                                                P_f, maglimerr_array, m_low_a,
                                                m_high_a, m_low_t, m_high_t):
        """
        Returns the overall likelihood of observations for a single field using
        survey limits and distance posterior samples, under the astrophysical
        hypothesis.
        """
        plims = np.array([self.calc_infield_filter_mlim_likelihood_fromsamples(
                        params[2*(fid-1):2*fid], fid, mlims_array[i], t0,
                        filter_obs_times[i], d_samples, maglimerr_array[i],
                        m_low_a, m_high_a, m_low_t, m_high_t)\
                        for i,fid in enumerate(filter_ids)])
        return np.product(plims)*P_f
