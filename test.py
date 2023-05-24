import matplotlib.pyplot as plt
import numpy as np
import cpnest.model

from pyfieldsim.field import Field
from pyfieldsim.observation import Observation


def powerlaw_b_fixed(L, C, a):
    return C * L ** (-(a + 3))


class PowerLawBFixed(cpnest.model.Model):
    def __init__(self, lum):
        self.names = ['C', 'a']
        self.bounds = [[0.01, 1000], [-3, 10]]
        self.L = lum

    def log_prior(self, param):
        logP = super(PowerLawBFixed, self).log_prior(param)
        logP += -np.log(param['C'])
        return logP

    def log_likelihood(self, param):
        model = powerlaw_b_fixed(self.L, param['C'], param['a'])
        return np.log(model).sum()


if __name__ == "__main__":
    fld = Field((200, 200))
    fld.initialize_field(density=0.02, cst_lm=1000,
                         datatype='luminosity')

    fld.show_field(field='true')

    obs = Observation(fld)

    print("datatype  : {}".format(obs.datatype))
    print("obs_status: {}".format(obs.status))

    stars, coords = obs.count_single_stars()

    hist, edges = np.histogram(stars, bins='sqrt')
    bin_edges = np.logspace(np.log10(edges[0]), np.log10(edges[-1]), len(edges))

    ModelInference = PowerLawBFixed(stars)
    work = cpnest.CPNest(ModelInference, verbose=2, nensemble=0, nlive=100, maxmcmc=500, nslice=4, nhamiltonian=0,
                         resume=0)
    work.run()

    posteriors = work.get_posterior_samples()

    fig_post, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    ax1.hist(posteriors['C'], density=True, bins='auto')
    ax1.set_xlabel('Factor')
    ax2.hist(posteriors['a'], density=True, bins='auto')
    ax1.set_xlabel('Exponent')

    models = [powerlaw_b_fixed(stars, posteriors['C'][i], posteriors['a'][i] + 1) for i in range(posteriors.shape[0])]
    models = np.array(models)

    l, m, h = np.percentile(models[31.7, 50, 68.3], axis=0)

    fig = plt.figure()
    ax = fig.gca()
    hist, edges, patch = ax.hist(stars, bins=bin_edges, log=True)
    ax.plot(stars, m, linewidth=0.5, color='k')
    ax.fill_between(stars, l, h, facecolor='grey', alpha=0.5)
    ax.grid()

    ax.set_xscale('log')
    plt.show()

    # -------------------------------------------------------------------------
    # Adding photon noise to sources.
    # -------------------------------------------------------------------------
    # fld.add_photon_noise()
    # obs.update_image()
    #
    # stars, coords = obs.count_single_stars()
    #
    # hist, edges = np.histogram(stars, bins='sqrt')
    # bin_edges = np.logspace(np.log10(edges[0]), np.log10(edges[-1]), len(edges))
    #
    # fig = plt.figure()
    # ax = fig.gca()
    # ax.grid()
    # hist, edges, patch = ax.hist(stars, bins=bin_edges, log=True)
    # ax.set_xscale('log')

    # Creating gain map.
    # fld.create_gain_map()

    # Adding background.
    # fld.add_background()

    # Convolving with gaussian psf kernel
    # psf = GaussKernel(3, size=2.5)
    # fld.apply_psf(psf)

    # Showing fields
    # fld.show_field(field='ph_noise')
    # fld.show_field(field='background')
    # fld.show_field(field='psf')
    # fld.show_field(field='gain_map')

    # -------------------------------------------------------------------------
    # Complete operation
    # -------------------------------------------------------------------------
    # field = Field((200, 200))
    # field.initialize_field(density=0.02, datatype='luminosity')
    #
    # field.show_field('true')
    #
    # observation = Observation(field)
    #
    # # counting single stars
    # stars, coords = obs.count_single_stars()
    #
    # hist, edges = np.histogram(stars, bins='sqrt')
    # bin_edges = np.logspace(np.log10(edges[0]), np.log10(edges[-1]), len(edges))
    # fig = plt.figure()
    # ax = fig.gca()
    # ax.grid()
    # hist, edges, patch = ax.hist(stars, bins=bin_edges, log=True)
    # ax.set_xscale('log')
    #
    # psf = GaussKernel(sigma=3)
    #
    # field.record_field(kernel=psf, delta_time=1000, snr=10, bgnd_rel_var=0.05, gain_mean=1, gain_rel_var=0.01,
    #                    dk_c_fraction=0.1, dk_c_rel_var=0.01, dk_c=1, force=True)
    #
    # field.show_field('exposure')
