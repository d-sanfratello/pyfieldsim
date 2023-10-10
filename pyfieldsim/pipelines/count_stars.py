import argparse as ag
import matplotlib.pyplot as plt
import numpy as np
import os

from pathlib import Path

from pyfieldsim.core.fieldtypes.field import Field
from pyfieldsim.core.stars import new_point_star
from pyfieldsim.utils.metadata import read_metadata
from pyfieldsim.utils.save_stars import save_point_stars

from pyfieldsim.errors.exceptions import WrongDataFileError


def main():
    """
    Pipeline that counts the stars in a given simulation.
    """
    parser = ag.ArgumentParser(
        prog='fs-count-stars',
        usage=__doc__,
    )
    parser.add_argument('data_file',
                        help="Sources file to count.")
    parser.add_argument("--no-log", action='store_true', default=False,
                        dest='no_log',
                        help="Wether to show the counts or the decimal "
                             "logarithm of the counts in the plot")
    parser.add_argument("-o", "--output",
                        dest='out_folder', default=None,
                        help="The folder where to save the output of this "
                             "pipeline.")

    args = parser.parse_args()

    if args.out_folder is None:
        out_folder = Path(os.getcwd())
    else:
        out_folder = Path(args.out_folder)

    data_file = Path(args.data_file)

    if data_file.name.startswith('S'):
        sources_metadata = read_metadata(data_file)
        sources_field = Field.from_sources(data_file)
    else:
        raise WrongDataFileError(
            "File must be sources ('S')."
        )
    stars = find_stars(sources_field, sources_metadata)

    fig = plt.figure()
    ax = fig.gca()

    counts, bins = ax.hist(
        [s.A for s in stars],
        histtype='step', color='black',
        label="Distribution of sources.",
        bins=int(np.round(np.sqrt(len(stars))))
    )[:-1]

    centers = 0.5 * (bins[:-1] + bins[1:])
    errors = np.sqrt(counts)

    ax.errorbar(centers, counts, errors, linestyle='none', marker='.',
                capsize=2)

    if not args.no_log:
        ax.set_xscale('log')
        ax.set_yscale('log')

    save_point_stars(stars, data_file, options='countS')

    fig.savefig(out_folder.joinpath("hist.pdf"))


def find_stars(ext_field, metadata):
    shape = metadata['shape']
    pad = metadata['pad']

    field = np.copy(
        ext_field.field[pad[0]: -pad[0], pad[1]: -pad[1]]
    )

    stars = []
    brightest = field.max()
    while brightest > 0:
        coords = np.unravel_index(np.argmax(field), shape)

        stars.append(
            new_point_star(
                A=brightest,
                mu=[coords[1], coords[0]]
            )
        )

        # once the limit value has been found and stored, the
        # corresponding pixel on the CCD is set to 0.
        field[coords] = 0
        brightest = field.max()

    return stars


if __name__ == "__main__":
    main()
