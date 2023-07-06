import argparse as ag
import matplotlib.pyplot as plt
import numpy as np
import os

from pathlib import Path

from pyfieldsim.core.fieldtypes.field import Field
from pyfieldsim.core.stars import new_point_star
from pyfieldsim.utils.metadata import read_metadata
from pyfieldsim.utils.save_stars import save_stars

from pyfieldsim.errors.exceptions import WrongDataFileError


def main():
    parser = ag.ArgumentParser(
        prog='fs-count-stars',
        description='',
    )
    parser.add_argument('data_file')
    parser.add_argument("-o", "--output",
                        dest='out_folder', default=None,
                        help="")

    args = parser.parse_args()

    if args.out_folder is None:
        out_folder = Path(os.getcwd())
    else:
        out_folder = Path(args.out_folder)

    data_file = Path(args.data_file)

    """
    Method that counts the stars in a given field, if the `ImageStatus` of
    the image is `SINGLESTARS` or `PH_NOISE`.

    Since any other step in the simulation adds background or spreads the
    values over different pixels, usage of this method is forbidden in those
    cases.

    This method stores a copy of the image extracted at initialization and
    finds the highest value (if `field.Field.datatype` is `LUMINOSITY` or
    `MASS`) or the lowest (if `MAGNITUDE`). It also stores its coordinates.
    So the pixel of the image copy at those coordinates is set to zero. This
    process is, then, reiterated until the field is filled with zeros and
    the values and the corrisponding coordinates are returned.

    Returns
    -------
    recorded_stars: `numpy.ndarray`
        Returns the array of measured values for the stars in the field.
        Those values are in the unit defined by the field datatype. See also
        `utils.DataType`.
    recorded_coords: `numpy.ndarray` of numpy arrays of length 2.
        Returns, for each value returned in `recorded_stars`, the
        coordinates at which that values has been found.

    Raises
    ------
    `IncompatibleStatusError`:
        If `field.Field.status` is not `SINGLESTARS` or `PH_NOISE`. As
        explained, this procedure wouldn't make sense with other steps of
        the simulation.

    See Also
    --------
    `field.Field`, `utils.DataType` and `utils.ImageStatus`.
    """
    if data_file.name.startswith('S'):
        sources = True

        sources_metadata = read_metadata(data_file)
        sources_field = Field.from_sources(data_file)
    elif data_file.name.startswith('P'):
        sources = False

        sources_data_file = Field.from_sources(
            Path('S' + data_file.stem[1:]).with_suffix('.h5')
        )
        sources_metadata = read_metadata(sources_data_file)
        sources_field = Field.from_sources(data_file)

        p_metadata = Path(data_file.stem + '_meta').with_suffix('.h5')
        p_metadata = read_metadata(p_metadata)
    else:
        raise WrongDataFileError(
            "File must be either sources ('S') or photon noise contaminated "
            "('P')."
        )
    stars = find_stars(sources_field, sources_metadata)

    if not sources:
        for s in stars:
            s.A = s.A * float(p_metadata['delta_time'])

    fig = plt.figure()
    ax = fig.gca()

    hist_s = ax.hist(
        [s.A for s in stars],
        histtype='step', color='black',
        label="Distribution of sources."
    )

    if not sources:
        ph_field = Field.from_field(data_file)

        p_stars = find_stars(ph_field, sources_metadata)

        hist_p = ax.hist(
            [p.A for p in p_stars],
            histtype='step', color='red',
            label="Distribution of noise-contaminated sources."
        )

        plt.legend(loc='best')

        save_stars(p_stars, data_file, options='countP')

    save_stars(stars, data_file, options='countS')

    fig.savefig(out_folder.joinpath("hist.pdf"))
    plt.show()


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
