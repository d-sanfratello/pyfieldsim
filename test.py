import matplotlib.pyplot as plt

from fieldsim.field import Field
from fieldsim.observation import Observation

if __name__ == "__main__":
    fld = Field((200, 200))
    fld.initialize_field(density=0., datatype='mass')

    print(fld.true_field)

    fld.show_field(field='true')

    obs = Observation(fld)

    print(obs.datatype, obs.status)

    stars, coords = obs.count_single_stars()

    fig = plt.figure()
    ax = fig.gca()
    ax.grid()
    hist, edges, patch = ax.hist(stars, bins='auto', log=True)

    plt.show()
