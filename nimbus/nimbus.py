#!/usr/bin/env python
"""
The main module of kNe-inference that sets up the Bayesian formalism.

Classes:

    Kilonova_Inference
    Sampler
"""

__author__ = 'Siddharth Mohite'

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, truncnorm
from scipy.integrate import quad
from scipy.special import expit
import emcee
from multiprocessing import Pool, Process
from functools import partial
import corner
import healpy as hp


class Kilonova_Inference():
    """
    Initializes utility functions for inference and defines the MCMC model.

    Attributes
    ----------
    lc_model_g : func
        The function that defines the g-band light-curve evolution as a
        function of time.
    lc_model_r : func
        The function that defines the r-band light-curve evolution as a
        function of time.

    Methods
    -------
    lc_model_powerlaw(M_0,gamma,t_0,t):
        Returns the absolute magnitude evolution as a power law.
    M_to_m(M,distance):
        Returns the apparent magnitude using a distance and absolute
        magnitude.
    dlim(mlim,M):
        Returns the limiting distance for a model with absolute magnitude M
        and limiting magnitude mlim.
    create_distance_dist(mu_f,sigma_f):
        Returns a truncated normal distribution as the distance distribution.
    create_mlim_pdf(M,d_lim,p_d,m_low,m_high):
        Returns the probability density for the likelihood.
    compute_f_bar_sum(mlims):
        Returns the complementary number of observations for every field.
    ln_prior_uniform(ndim,params,limits):
        Returns a boolean indicating if the sample parameters are within prior
        bounds.
    ln_likelihood(params,nparams_g,mlims_g,mlims_r,T_g,T_r,t0,p_d_g,p_d_r,
    P_f_g,P_f_r,P_f_bar_g,P_f_bar_r,P_tot_gbar,P_tot_rbar,mask_g,mask_r,P_A,
    P_T,plims_f_bar,plims_T,mlow_g,mhigh_g,mlow_r,mhigh_r):
        Returns the single event log-likelihood for a single model.
    ln_likelihood_events(params,nevents,nparams_g,mlims_g,mlims_r,T_g,T_r,t0,
    p_d_g,p_d_r,P_f_g,P_f_r,P_f_bar_g,P_f_bar_r,P_tot_gbar,P_tot_rbar,mask_g,
    mask_r,P_A,P_T,plims_f_bar,plims_T,mlow_g,mhigh_g,mlow_r,mhigh_r):
        Returns the multiple event log-likelihood.
    ln_prob(params,ndim,nevents,limits,nparams_g,mlims_g,mlims_r,T_g,T_r,t0,
    p_d_g,p_d_r,P_f_g,P_f_r,P_f_bar_g,P_f_bar_r,P_tot_gbar,P_tot_rbar,mask_g,
    mask_r,P_A,P_T,plims_f_bar,plims_T,mlow_g,mhigh_g,mlow_r,mhigh_r):
        Returns the log-posterior.

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
                Time of the event.
            t : float or array
                Array of observation times.

        Returns
        -------
            Absolute magnitude light curve as a function of time (same shape
            as t).
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
                Time of the event.
            t : float or array
                Array of observation times.

        Returns
        -------
            Absolute magnitude light curve as a function of time (same shape
            as t).
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
                Apparent magnitude of the object (same size as M or d).
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
        a = (0. - mu_f)/sigma_f
        b = (4000. - mu_f)/sigma_f
        return truncnorm(a, b, mu_f, sigma_f)

    def calc_expit_argument(self,d_lim,maglim_err=0.1):
        if maglim_err==0.:
            maglim_err = 0.1
        dlow = d_lim*10**-(3*maglim_err/5) # set dlow at 3-sigma
        dmid = d_lim*10**-(maglim_err/5) # set dmid at 1-sigma
        a = np.log(0.021/0.979)/(dlow - dmid)
        b = -1.0*dmid
        return lambda x : expit(a*(x + b))

    def calc_likelihood_integral(self, M, expit_func, dist_samples, mlow, mhigh):
        dist_samples_survey = dist_samples[(dist_samples>self.dlim(mlow,M))&(dist_samples<=self.dlim(mhigh,M))]
        dist_samples_high = dist_samples[dist_samples>self.dlim(mhigh,M)]
        N_samples_survey = len(dist_samples_survey)
        N_samples_high = len(dist_samples_high)
        N_total = N_samples_survey + N_samples_high
        if (N_samples_survey==0)&(N_samples_high!=0):
            return 1./(mhigh-mlow)
        elif (N_samples_survey!=0)&(N_samples_high==0):
            return np.sum((1./(np.vectorize(self.M_to_m)(M, dist_samples_survey) - mlow))*np.vectorize(expit_func)(dist_samples_survey))/N_samples_survey
        elif (N_samples_survey!=0)&(N_samples_high!=0):
            return N_samples_survey/N_total * np.sum((1./(np.vectorize(self.M_to_m)(M, dist_samples_survey) - mlow))*\
                                                    np.vectorize(expit_func)(dist_samples_survey)) + (N_samples_high/N_total) * (1./(mhigh-mlow))
        return 0.

    def create_dlim_pdf(self, M, d_lim, maglim_err, norm_factor, p_d, d_min, d_max):
        """
        Returns the likelihood of the observations given a model, under the
        astrophysical hypothesis.

        Parameters
        ----------
            M : float
                The absolute magnitude of the model.
            d_lim : float
                The observed limiting distance below which non-detection is invalid.
            p_d : func
                The probability density function (pdf) of the distance.
            d_low : float
                Lower limit of the distance distribution.
            d_max : float
                Upper limit of the distance distribution.
            norm_factor : float
                Normalization factor for the likelihood calculation.

        Returns
        -------
            dlim_pdf : float
                The likelihood of obtaining the observed limiting magnitude
                given model absolute magnitude M.
        """
        expit_num = self.calc_expit_argument(d_lim, maglim_err)
        num = quad(lambda d : (1./(self.M_to_m(M,d)-self.M_to_m(M,d_min)))*expit_num(d)*p_d(d), d_min+0.1, d_max)[0]
        return num/norm_factor

    def create_mlim_pdf(self, M, d_lim, maglim_err, p_d, m_low, m_high, eps=0.1, dmax=3000):
        """
        Returns the likelihood of the observations given a model, under the
        astrophysical hypothesis.

        Parameters
        ----------
            M : float
                The absolute magnitude of the model.
            d_lim : float
                The limiting distance below which non-detection is invalid.
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
        num = quad(lambda d : (1./(self.M_to_m(M, d) - m_low))*expit_num(d)*p_d(d), self.dlim(m_low,M)+eps, self.dlim(m_high,M))[0] + quad(lambda d : (1./(m_high - m_low))*p_d(d), self.dlim(m_high,M), dmax)[0]
        den = quad(
            lambda m: quad(lambda d : (1./(self.M_to_m(M, d) - m_low))*self.calc_expit_argument(self.dlim(m,M),maglim_err)(d)*p_d(d), self.dlim(m_low,M)+eps, self.dlim(m_high,M))[0], m_low, m_high)[0] + quad(lambda d : p_d(d), self.dlim(m_high,M), dmax)[0]
        if den==0.:
            return 0.
        return num/den

    def create_mlim_pdf_fromsamples(self, M, d_lim, maglim_err, dist_samples, m_low, m_high):
        expit_num = self.calc_expit_argument(d_lim, maglim_err)
        num = self.calc_likelihood_integral(M, expit_num, dist_samples, m_low, m_high)
        den = quad(lambda m: self.calc_likelihood_integral(M, self.calc_expit_argument(self.dlim(m,M), maglim_err), dist_samples, m_low, m_high), m_low, m_high)[0]
        if den==0.:
            return 0.
        return num/den

    def calc_infield_filter_dlim_likelihood(
                                      self, params, fid, mlims, t0, T, p_d_f,
                                      maglimerrs, dmin, dmax, mlow_t, mhigh_t, norm_factors):
        M = np.array([self.lc_model_funcs[fid-1](*params, t_0=t0, t=t)\
                    for t in T])
        dlims = np.array(list(map(self.dlim, mlims, M)))
        pool = Pool(processes=2)
        plims_f_t = pool.starmap(partial(self.create_dlim_pdf, p_d=p_d_f,
                                d_min=dmin, d_max=dmax),np.c_[M,dlims,maglimerrs,norm_factors])
        pool.close()
        plims_f_t_nondet = np.array([self.nullevent_mlim_pdf(mlow_t,mhigh_t)\
                                   for m in mlims])
        plims_f = np.product(plims_f_t/plims_f_t_nondet)
        return plims_f

    def calc_infield_dlim_likelihood(
                               self, params, filter_ids, mlims_array, t0,
                               filter_obs_times, p_d_f, P_f, maglimerr_array,
                               dmin, dmax, m_low_t, m_high_t, norm_factor_array):
        plims = np.array([self.calc_infield_filter_dlim_likelihood(
                        params[2*(fid-1):2*fid], fid, mlims_array[i], t0,
                        filter_obs_times[i], p_d_f, maglimerr_array[i], dmin,
                        dmax, m_low_t, m_high_t, norm_factor_array[i])\
                        for i,fid in enumerate(filter_ids)])
        return np.product(plims)*P_f

    def calc_infield_filter_mlim_likelihood(
                                      self, params, fid, mlims, t0, T, p_d_f,
                                      maglimerrs, mlow_a, mhigh_a, mlow_t, mhigh_t):
        M = np.array([self.lc_model_funcs[fid-1](*params, t_0=t0, t=t)\
                    for t in T])
        dlims = np.array(list(map(self.dlim, mlims, M)))
        pool = Pool(processes=2)
        plims_f_t = pool.starmap(partial(self.create_mlim_pdf, p_d=p_d_f,
                                m_low=mlow_a,m_high=mhigh_a),np.c_[M,dlims,maglimerrs])
        pool.close()
        plims_f_t_nondet = np.array([self.nullevent_mlim_pdf(mlow_t,mhigh_t)\
                                   for m in mlims])
        plims_f = np.product(plims_f_t/plims_f_t_nondet)
        return plims_f

    def calc_infield_mlim_likelihood(
                               self, params, filter_ids, mlims_array, t0,
                               filter_obs_times, p_d_f, P_f, maglimerr_array,
                               m_low_a, m_high_a, m_low_t, m_high_t):
        plims = np.array([self.calc_infield_filter_mlim_likelihood(
                        params[2*(fid-1):2*fid], fid, mlims_array[i], t0,
                        filter_obs_times[i], p_d_f, maglimerr_array[i], m_low_a,
                        m_high_a, m_low_t, m_high_t)\
                        for i,fid in enumerate(filter_ids)])
        return np.product(plims)*P_f

    def calc_infield_filter_mlim_likelihood_fromsamples(
                                      self, params, fid, mlims, t0, T, d_samples,
                                      maglimerrs, mlow_a, mhigh_a, mlow_t, mhigh_t):
        M = np.array([self.lc_model_funcs[fid-1](*params, t_0=t0, t=t)\
                    for t in T])
        dlims = np.array(list(map(self.dlim, mlims, M)))
        pool = Pool(processes=2)
        plims_f_t = pool.starmap(partial(self.create_mlim_pdf_fromsamples, dist_samples=d_samples,
                                m_low=mlow_a,m_high=mhigh_a),np.c_[M,dlims,maglimerrs])
        pool.close()
        plims_f_t_nondet = np.array([self.nullevent_mlim_pdf(mlow_t,mhigh_t)\
                                   for m in mlims])
        plims_f = np.product(plims_f_t/plims_f_t_nondet)
        return plims_f

    def calc_infield_mlim_likelihood_fromsamples(
                               self, params, filter_ids, mlims_array, t0,
                               filter_obs_times, d_samples, P_f, maglimerr_array,
                               m_low_a, m_high_a, m_low_t, m_high_t):
        plims = np.array([self.calc_infield_filter_mlim_likelihood_fromsamples(
                        params[2*(fid-1):2*fid], fid, mlims_array[i], t0,
                        filter_obs_times[i], d_samples, maglimerr_array[i], m_low_a,
                        m_high_a, m_low_t, m_high_t)\
                        for i,fid in enumerate(filter_ids)])
        return np.product(plims)*P_f

    def compute_f_bar_sum(self, mlims):
        """
        Returns the complementary number of observations for every field.

        Parameters
        ----------
            mlims : array
                Array of observed limiting magnitudes.

        Returns
        -------
            f_bar_sum : array
                Array of the complementary number of observations in all
                fields except the current one.
        """
        total_len = sum(len(m) for m in mlims)
        return np.array([total_len - len(mlim) for mlim in mlims])

    def ln_prior_uniform(self, ndim, params, limits):
        """
        Returns a boolean indicating if the sample parameters are within prior
        bounds.

        Parameters
        ----------
            ndim : int
                Number of dimensions or parameters in the model.
            params : list
                Set of model parameters for a given sample.
            limits : array
                Array of prior bounds for each parameter. Shape = (nparams,2)

        Returns
        -------
            ln_prior_uniform : bool
                True if sample parameters are within the bounds. False
                otherwise.
        """
        return (sum(list(map(
               lambda x, lim: lim[0] <= x <= lim[1], params,
               limits))) == ndim)

    def ln_likelihood_generic(self, params, field_args, P_f, P_A, P_T):
        pool = Pool()
        field_likelihoods = pool.starmap(partial(self.calc_infield_likelihood,
                                        params=params,
                                        lc_model=self.lc_model_g),
                                        field_args)
        pool.close()
        pool.join()
        return np.log((np.sum(field_likelihoods) + 1-np.sum(P_f))*P_A + P_T)

    def ln_likelihood(
                     self, params, nparams_g, mlims_g, mlims_r, T_g, T_r, t0,
                     p_d_g, p_d_r, P_f_g, P_f_r, P_f_bar_g, P_f_bar_r,
                     P_tot_gbar, P_tot_rbar, mask_g, mask_r, P_A, P_T,
                     plims_f_bar, plims_T, mlow_g, mhigh_g, mlow_r, mhigh_r):
        """
        Returns the single event log-likelihood for a model draw.

        Parameters
        ----------
            params : list
                Set of model parameters for a given sample. Assume g-band
                parameters are assigned first in the array.
            nparams_g : int
                Number of parameters for g_band evolution.
            mlims_g : array
                Array of observed g-band limiting magnitudes (corrected for
                extinction). Shape = (nfields, nobs)
            mlims_r : array
                Array of observed r-band limiting magnitudes (corrected for
                extinction). Shape = (nfields, nobs)
            T_g : array
                Array of observation times in g-band (shape same as mlims_g).
            T_r : array
                Array of observation times in r-band (shape same as mlims_r).
            t0 : float
                Time of the event.
            p_d_g : array
                Array of probability density functions of the distance for
                each field in the g-band. Shape = (nfields,)
            p_d_r : array
                Array of probability density functions of the distance for
                each field in the r-band. Shape = (nfields,)
            P_f_g : array
                Array of total enclosed probabilities for the event to be in a
                field in the g-band. Shape = (nfields,)
            P_f_r : array
                Array of total enclosed probabilities for the event to be in a
                field in the r-band. Shape = (nfields,)
            P_f_bar_g : array
                Array of the complementary number of observations in the
                g-band in all fields except the current one.
                Shape = (nfields,)
            P_f_bar_r : array
                Array of the complementary number of observations in the
                r-band in all fields except the current one.
                Shape = (nfields,)
            P_tot_gbar : float
                Total probability of observations in the g-band being
                unassociated with the kilonova.
            P_tot_rbar : float
                Total probability of observations in the r-band being
                unassociated with the kilonova.
            mask_g : array
                Array of booleans indicating which observations in the g-band
                have corresponding r-band observations, in the same field.
                Shape = (nfields,)
            mask_r : array
                Array of booleans indicating which observations in the r-band
                have corresponding g-band observations, in the same field.
                Shape = (nfields,)
            P_A : float
                Probability of the event being astrophysical.
            P_T : float
                Probability of the event being terrestrial.
            plims_f_bar : float
                The likelihood of observations under the hypothesis that the
                event is outside the searched area.
            plims_T : float
                The likelihood of observations under the hypothesis that the
                event is terrestrial.
            mlow_g : array
                Array containing the field-specific lower limit of the
                limiting magnitude distribution for the g-band.
                Shape = (nfields,)
            mhigh_g : array
                Array containing the field-specific upper limit of the
                limiting magnitude distribution for the g-band.
                Shape = (nfields,)
            mlow_r : array
                Array containing the field-specific lower limit of the
                limiting magnitude distribution for the r-band.
                Shape = (nfields,)
            mhigh_r : array
                Array containing the field-specific upper limit of the
                limiting magnitude distribution for the r-band.
                Shape = (nfields,)

        Returns
        -------
            ln_likelihood : float
                The log-likelihood for the given model draw.
        """
        g_params = params[:nparams_g]
        r_params = params[nparams_g:]
        M_g = np.array([
                      self.lc_model_g(*g_params, t_0=t0, t=t) for t in
                      T_g])
        M_r = np.array([
                      self.lc_model_r(*r_params, t_0=t0, t=t) for t in
                      T_r])
        dlims_g = np.array(list(map(self.dlim, mlims_g, M_g)))
        dlims_r = np.array(list(map(self.dlim, mlims_r, M_r)))
        plims_f_t_g = np.array(list(map(np.vectorize(
                               self.create_mlim_pdf), M_g, dlims_g, p_d_g,
                               mlow_g, mhigh_g)))
        plims_f_t_r = np.array(list(map(np.vectorize(
                               self.create_mlim_pdf), M_r, dlims_r, p_d_r,
                               mlow_r, mhigh_r)))
        plims_f_g = np.array([np.product(p) for p in plims_f_t_g])
        plims_f_r = np.array([np.product(p) for p in plims_f_t_r])
        plims_r_g = (plims_f_g * P_f_g * P_f_bar_g)[mask_g] \
            * (plims_f_r * P_f_bar_r)[mask_r]
        plims_g = (plims_f_g * P_f_g * P_f_bar_g)[~mask_g] * P_tot_rbar
        plims_r = (plims_f_r * P_f_r * P_f_bar_r)[~mask_r] * P_tot_gbar
        return np.log((
                     np.sum(plims_r_g) + np.sum(plims_g) + np.sum(plims_r)
                     + plims_f_bar) * P_A + plims_T * P_T)

    def ln_likelihood_events(
                            self, params, nevents, nparams_g, mlims_g,
                            mlims_r, T_g, T_r, t0, p_d_g, p_d_r, P_f_g, P_f_r,
                            P_f_bar_g, P_f_bar_r, P_tot_gbar, P_tot_rbar,
                            mask_g, mask_r, P_A, P_T, plims_f_bar, plims_T,
                            mlow_g, mhigh_g, mlow_r, mhigh_r):
        """
        Returns the multiple event log-likelihood.

        Parameters
        ----------
            params : list
                Set of model parameters for a given sample.
            nevents : int
                Total number of events being considered in the inference.

            Refer to the method ln_likelihood(**args) for a description other
            parameters. Note that the shape of all these parameters needs to
            be one dimension more than those in that method, to account for
            the separate events. Shape = (nevents, nfields, nobs) or
            (nevents, nfields) wherever applicable.

        Returns
        -------
            ln_likelihood_events : float
                The multiple event log-likelihood for the given model draw.
        """
        params_arr = np.ones(nevents)[:, None] * params
        ln_likelihood_arr = np.array(list(map(
                                    self.ln_likelihood, params_arr, nparams_g,
                                    mlims_g, mlims_r, T_g, T_r, t0, p_d_g,
                                    p_d_r, P_f_g, P_f_r, P_f_bar_g, P_f_bar_r,
                                    P_tot_gbar, P_tot_rbar, mask_g, mask_r,
                                    P_A, P_T, plims_f_bar, plims_T, mlow_g,
                                    mhigh_g, mlow_r, mhigh_r)))
        return np.sum(ln_likelihood_arr)

    def ln_prob(
               self, params, ndim, nevents, limits, nparams_g, mlims_g,
               mlims_r, T_g, T_r, t0, p_d_g, p_d_r, P_f_g, P_f_r,
               P_f_bar_g, P_f_bar_r, P_tot_gbar, P_tot_rbar, mask_g, mask_r,
               P_A, P_T, plims_f_bar, plims_T, mlow_g, mhigh_g, mlow_r,
               mhigh_r):
        """
        Returns the log-posterior.

        Parameters
        ----------
            params : list
                Set of model parameters for a given sample.
            ndim : int
                Number of dimensions or parameters in the model.
            nevents : int
                Total number of events being considered in the inference.
            limits : array
                Array of prior bounds for each parameter.

            Refer to the method ln_likelihood_events(**args) for a description
            other parameters.

        Returns
        -------
            ln_prob : float
                The log-posterior for the given model draw.
        """
        if self.ln_prior_uniform(ndim, params, limits):
            return self.ln_likelihood_events(params, nevents, nparams_g,
                                            mlims_g, mlims_r, T_g, T_r, t0,
                                            p_d_g, p_d_r, P_f_g, P_f_r,
                                            P_f_bar_g, P_f_bar_r, P_tot_gbar,
                                            P_tot_rbar, mask_g, mask_r, P_A,
                                            P_T, plims_f_bar, plims_T, mlow_g,
                                            mhigh_g, mlow_r, mhigh_r)
        return -np.inf


class Sampler(Kilonova_Inference):
    """
    A class to set up the emcee (https://pypi.org/project/emcee/) sampler.

    Attributes
    ----------
    ndim : int
        Number of dimensions or parameters in the model.
    nwalkers : int
        Number of walkers in the ensemble to sample the posterior.
    nevents : int
        Total number of events being considered in the inference.
    limits : array
        Array of prior bounds for each parameter.
    data : tuple
        Tuple combining all the additional arguments to be passed to the
        ln_prob function.

    Methods
    -------
    sample(nburn,nsteps,pool=False):
        Returns the emcee sampler object that contains posterior samples.
    corner_plot(sampler,fname,labels):
        Returns and optionally saves the corner plot of the posterior samples.

    Usage
    -----
    sampler =
    Sampler(ndim,nwalkers,nevents,limits,nparams_g,mlims_g,mlims_r,T_g,T_r,t0,
    p_d_g,p_d_r,P_f_g,P_f_r,P_f_bar_g,P_f_bar_r,P_tot_gbar,P_tot_rbar,mask_g,
    mask_r,P_A,P_T,plims_f_bar,plims_T,mlow_g,mhigh_g,mlow_r,mhigh_r)
    """

    def __init__(
                self, ndim, nwalkers, nevents, limits, nparams_g, mlims_g,
                mlims_r, T_g, T_r, t0, p_d_g, p_d_r, P_f_g, P_f_r, P_f_bar_g,
                P_f_bar_r, P_tot_gbar, P_tot_rbar, mask_g, mask_r, P_A, P_T,
                plims_f_bar, plims_T, mlow_g, mhigh_g, mlow_r, mhigh_r):
        Kilonova_Inference.__init__(self)
        print("Initializing sampler...")
        self.ndim = ndim
        self.nwalkers = nwalkers
        self.nevents = nevents
        self.limits = limits
        self.data = (
                    ndim, nevents, limits, nparams_g, mlims_g, mlims_r, T_g,
                    T_r, t0, p_d_g, p_d_r, P_f_g, P_f_r, P_f_bar_g, P_f_bar_r,
                    P_tot_gbar, P_tot_rbar, mask_g, mask_r, P_A, P_T,
                    plims_f_bar, plims_T, mlow_g, mhigh_g, mlow_r, mhigh_r)

    def sample(self, nburn, nsteps, pool=False):
        """
        Returns the emcee sampler object that contains posterior samples.

        Parameters
        ----------
            nburn : int
                Number of burn-in steps for the MCMC sampler.
            nsteps : int
                Number of steps, post burn-in, for the MCMC sampler.
            pool : bool , optional
                Option to use python multiprocessing to speed up expensive
                likelihood calculations(default=False).

        Returns
        -------
            sampler : emcee.EnsembleSampler object
                Sampler object containing posterior samples.
        """
        pos0 = [[np.random.uniform(
               low=lim[0], high=lim[1]) for lim in self.limits] for i in
               range(self.nwalkers)]
        if pool:
            pool_inst = Pool()
            sampler = emcee.EnsembleSampler(
                                           self.nwalkers, self.ndim,
                                           self.ln_prob, args=self.data,
                                           pool=pool_inst)
            pos, prob, state = sampler.run_mcmc(pos0, nburn, progress=True)
            sampler.reset()
            sampler.run_mcmc(pos, nsteps, progress=True)
            return sampler
        sampler = emcee.EnsembleSampler(
                                       self.nwalkers, self.ndim, self.ln_prob,
                                       args=self.data)
        pos, prob, state = sampler.run_mcmc(pos0, nburn, progress=True)
        sampler.reset()
        sampler.run_mcmc(pos, nsteps, progress=True)
        return sampler

    def corner_plot(self, sampler, fname=None, labels=[r"$M0$", r"$\gamma$"]):
        """
        Returns and optionally saves the corner plot of the posterior samples.

        Parameters
        ----------
        sampler : emcee.EnsembleSampler object
            Sampler object containing posterior samples.
        fname : str , optional
            Filename/path to save the corner plot (default format is 'png').
            Default=None
        labels : list
            List of strings specifying the names of parameters to plot.

        Returns
        -------
            Corner plot of posterior samples for the model parameters.
        """
        fig = corner.corner(
                           sampler.flatchain, labels=labels,
                           quantiles=[0.02, 0.5, 0.98], show_titles=True,
                           title_kwargs={"fontsize": 14})
        if fname is not None:
            plt.savefig(fname=fname, format='png')
