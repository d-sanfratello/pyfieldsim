import matplotlib.pyplot as plt
import numpy as np

from fieldsim.field import Field
from fieldsim.observation import Observation

if __name__ == "__main__":
    fld = Field((200, 200))
    fld.initialize_field(density=0.05, datatype='mass')

    fld.show_field(field='true')

    obs = Observation(fld)

    print(obs.datatype, obs.status)

    stars, coords = obs.count_single_stars()

    hist, edges = np.histogram(stars, bins='sqrt')
    bin_edges = np.logspace(0, np.log10(edges[-1]), len(edges))

    fig = plt.figure()
    ax = fig.gca()
    ax.grid()
    hist, edges, patch = ax.hist(stars, bins=bin_edges, log=True)
    ax.set_xscale('log')

    plt.show()
