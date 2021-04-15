import matplotlib.pyplot as plt
import numpy as np

from fieldsim.field import Field
from fieldsim.observation import Observation

if __name__ == "__main__":
    fld = Field((200, 200))
    fld.initialize_field(density=0.02, cst_lm=100,
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
    fld.add_photon_noise()
    fld.show_field(field='ph_noise')
    obs.update_image()

    stars, coords = obs.count_single_stars()

    hist, edges = np.histogram(stars, bins='sqrt')
    bin_edges = np.logspace(np.log10(edges[0]), np.log10(edges[-1]), len(edges))

    fig = plt.figure()
    ax = fig.gca()
    ax.grid()
    hist, edges, patch = ax.hist(stars, bins=bin_edges, log=True)
    ax.set_xscale('log')

    # Creating gain map.
    fld.create_gain_map()
    fld.show_field(field='gain_map')

    plt.show()
