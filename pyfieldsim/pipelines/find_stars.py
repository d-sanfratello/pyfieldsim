import argparse as ag
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os

from pathlib import Path

from pyfieldsim.core.fieldtypes.field import Field
from pyfieldsim.utils.metadata import read_metadata

from pyfieldsim.errors.exceptions import WrongDataFileError


def main():
    parser = ag.ArgumentParser(
        prog='fs-find-stars',
        description='',
    )
    parser.add_argument('sources')
    parser.add_argument('data_file')
    parser.add_argument("-b", "--b-mean", type=float,
                        dest='b_mean', default=None, required=True,
                        help="")
    parser.add_argument("-s", "--b-std", type=float,
                        dest='b_std', default=None, required=True,
                        help="")
    parser.add_argument("-o", "--output",
                        dest='out_folder', default=None,
                        help="")

    args = parser.parse_args()

    if args.out_folder is None:
        out_folder = Path(os.getcwd())
    else:
        out_folder = Path(args.out_folder)

    sources_file = Path(args.sources)
    data_file = Path(args.data_file)

    sources_metadata = read_metadata(sources_file)
    data_field = Field.from_field(data_file)

    b_mean = args.b_mean
    b_std = args.b_std

    #  TODO: Complete star recognition from here. Add pipeline to setup and
    #   helper.

    s_stars, s_coords = find_stars(sources_field, sources_metadata)
    p_metadata = Path(data_file.stem + '_meta').with_suffix('.h5')
    p_metadata = read_metadata(p_metadata)

    s_stars = s_stars * float(p_metadata['delta_time'])

    fig = plt.figure()
    ax = fig.gca()

    hist_s = ax.hist(
        s_stars, histtype='step', color='black',
        label="Distribution of sources."
    )

    if not sources:
        ph_field = Field.from_field(data_file)

        p_stars, p_coords = find_stars(ph_field, sources_metadata)

        hist_p = ax.hist(
            p_stars, histtype='step', color='red',
            label="Distribution of noise-contaminated sources."
        )

        plt.legend(loc='best')

        with h5py.File(out_folder.joinpath('P_recovered.h5'), 'w') as f:
            l_dset = f.create_dataset('luminosity',
                                      shape=p_stars.shape,
                                      dtype=float)
            l_dset[0:] = p_stars

            c_dset = f.create_dataset('coords',
                                      shape=p_coords.shape,
                                      dtype=int)
            c_dset[0:] = p_coords

    fig.savefig(out_folder.joinpath("hist.pdf"))
    plt.show()


def find_stars(ext_field, metadata):
    shape = metadata['shape']
    pad = metadata['pad']

    field = np.copy(
        ext_field.field[pad[0]: -pad[0], pad[1]: -pad[1]]
    )

    recorded_stars = []
    recorded_coords = []

    brightest = field.max()
    while brightest > 0:
        coords = np.unravel_index(np.argmax(field), shape)

        recorded_stars.append(brightest)
        recorded_coords.append(coords)

        # once the limit value has been found and stored, the
        # corresponding pixel on the CCD is set to 0.
        field[coords] = 0
        brightest = field.max()

    recorded_stars = np.asarray(recorded_stars)
    recorded_coords = np.asarray(recorded_coords)

    return recorded_stars, recorded_coords


if __name__ == "__main__":
    main()
