"""
Class to estimate prior expectations of target rotation period based on
select input data.
"""

import pandas as pd
import numpy as np
import emcee
import statsmodels.api as sm
from statsmodels.nonparametric.bandwidths import select_bandwidth

from .utils import _random_seed

class priorclass():
    """ Class managing the prior expectations on rotation period.

    Examples
    --------

    Parameters
    ----------


    Attributes
    ----------
    """

    def __init__(self, obs, verbose = False):
        self.obs = obs
        self.verbose = verbose

    def load_prior_data(self):
        """
        This will randomly shuffle the data following the set random seed.
        """
        df = pd.read_csv('../michael/data/prior_data.csv', index_col = None)
        self.train = df.sample(len(df), ignore_index=True).reset_index(drop=True)

    def build_kde(self):
        """
        Build a KDE based on the prior data from the Santos et al. (2019, 2020)
        catalogues. The KDE is based on a subselection of the full data set
        based on the uncertainties on the input observables to save on
        computation time. IF there are fewer than 2000 stars in a 2 sigma
        region in temperature, it is extended to 3 sigma.

        It is always recommended to check the number of stars included in the
        KDE when using this function. You can do this by setting `verbose=True`
        when initialising the `janet` class.
        """

        self.sel_train = self.train.loc[np.abs(self.train.logT - self.obs['logT'][0])
                                    < 2*self.obs['logT'][1]]
        if len(self.sel_train) < 2000:
            self.sel_train = self.train.loc[np.abs(self.train.logT - self.obs['logT'][0])
                                    < 3*self.obs['logT'][1]]

        self.bw = select_bandwidth(self.sel_train.values,
                      bw = 'scott', kernel=None)

        self.kde = sm.nonparametric.KDEMultivariate(data = self.sel_train.values,
                                               var_type = 'c'*len(self.sel_train.columns),
                                               bw = self.bw)

        if self.verbose:
            print(f'KDE built on {len(self.sel_train)} values.')

    def ln_normal(self, x, mu, sigma):
        """
        A normal distribution in log space.
        """
        return 0.5 * np.abs(x - mu)**2 / sigma**2

    def prior_pdf(self, p):
        """
        Returns the prior pdf for a given data input.
        """
        return self.kde.pdf(p)

    def likelihood(self, p):
        """
        Returns likelihood function as a sum of normal distributions
        of the form Normal(parameter - observed, uncertainty), and the
        prior probility resulting from the trained KDE.
        """
        like = np.log(1e-30 + self.prior_pdf(p))
        like += self.ln_normal(p[0], *self.obs['logT'])
        like += self.ln_normal(p[1], *self.obs['logg'])
        like += self.ln_normal(p[3], *self.obs['MG'])
        like += self.ln_normal(p[4], *self.obs['logbp_rp'])

        return like

    def sample(self, nwalkers = 32, nsteps = 1000):
        """
        Draw samples from the KDE distribution given the observations in
        logT, logg, MG and log(bp_rp).
        """
        ndim = 5
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.likelihood)

        start =  [self.obs['logT'][0], self.obs['logg'][0], 1.3, self.obs['MG'][0], self.obs['logbp_rp'][0]]
        p0 =  [start + np.random.rand(ndim) * [0.005, 0.001, 0.2, 0.2, 0.001] for n in range(nwalkers)]

        sampler.run_mcmc(p0, nsteps, progress=self.verbose)

        frac_acc = np.mean(sampler.acceptance_fraction)
        if frac_acc < 0.2:
            warnings.warn(f'Sampler acceptance fraction is low: {frac_acc}')

        self.samples = sampler.get_chain(flat=True)
        self.prot_prior = np.nanpercentile(self.samples[:,2], [16, 50, 84])

        if self.verbose:
            print('Done sampling prior!')

        return self.samples, self.prot_prior

    def __call__(self):
        self.load_prior_data()
        self.build_kde()
        samples, prot_prior = self.sample()
        return samples, prot_prior
