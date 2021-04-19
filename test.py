import matplotlib.pyplot as plt
import numpy as np
import cpnest
import cpnest.model

from fieldsim.field import Field
from fieldsim.observation import Observation
from fieldsim.psf.kernels import GaussKernel


def powerlaw_b_fixed_3(x, C, a):
    return C * x ** (-(a + 3))


class PowerLawBFixed3(cpnest.model.Model):
    def __init__(self, data, x_data, noise_mean=0.):
        self.names = ['C', 'a']
        self.bounds = [[0.01, 1000], [0.01, 10]]
        self.data = data
        self.noise_mean = noise_mean
        self.x = x_data


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

    fig = plt.figure()
    ax = fig.gca()
    ax.grid()
    hist, edges, patch = ax.hist(stars, bins=bin_edges, log=True)
    ax.set_xscale('log')

    # Adding photon noise to sources.
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

    plt.show()

    # -------------------------------------------------------------------------
    # Complete operation
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
