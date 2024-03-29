import argparse as ag
import numpy as np
import os
import scipy.signal as scipysig

from pathlib import Path

from pyfieldsim.core.fieldtypes.field import Field
from pyfieldsim.core.psf import GaussKernel
from pyfieldsim.utils.metadata import read_metadata, save_metadata


def main():
    """
    Pipeline to simulate a simulated observation of a field.
    """
    parser = ag.ArgumentParser(
        prog='fs-observation',
        usage=__doc__,
    )
    parser.add_argument('sources',
                        help="The name of the simulated sources to observe.")
    parser.add_argument('-t', '--delta-time', type=float,
                        dest='delta_time', default=1,
                        help="Time integration constant.")
    parser.add_argument('-b', action='store_true',
                        dest='background', default=False,
                        help="Flag to add a background or to keep the field "
                             "flat.")
    parser.add_argument('-k', '--psf-width', type=float,
                        dest='psf_width', default=None,
                        help="With of the PSF")
    parser.add_argument('-g', action='store_true',
                        dest='gain_map', default=False,
                        help="Flag to add the gain map of the CCD.")
    parser.add_argument('-c', action='store_true',
                        dest='dark_current', default=False,
                        help="Flag to add the dark current of the CCD.")
    parser.add_argument("-o", "--output",
                        dest='out_folder', default=None,
                        help="The folder where to save the output of this "
                             "pipeline.")

    args = parser.parse_args()

    if args.out_folder is None:
        out_folder = Path(os.getcwd())
    else:
        out_folder = Path(args.out_folder)

    sources_file = Path(args.sources)
    metadata = read_metadata(sources_file)

    field = Field.from_sources(sources_file).field

    if args.background:
        background_filename = 'B' + sources_file.name[1:]
        background = Field.from_field(background_filename).field

        field = field + background
    else:
        background_filename = None

    if args.psf_width is not None:
        psf = GaussKernel(sigma=args.psf_width, size=5)
        field = scipysig.convolve2d(
            field,
            psf.kernel,
            mode='same'
        )

        field = np.where(field < 0, 0, field)

    if args.gain_map:
        gain_map_filename = 'G' + sources_file.name[1:]
        gain_map = Field.from_field(gain_map_filename).field

        field = gain_map * field
    else:
        gain_map_filename = None

    rng = np.random.default_rng(seed=metadata['seed'])

    field = field * args.delta_time

    field = rng.poisson(field)
    field = np.where(
        field < 0, 0, field
    )

    if args.dark_current:
        dark_current_filename = 'C' + sources_file.name[1:]
        dark_current = Field.from_field(dark_current_filename).field

        field = field + dark_current
    else:
        dark_current_filename = None

    pad = metadata['pad']
    field = field[
            pad[0]:-pad[0],
            pad[1]:-pad[1]
    ]

    field = Field(
        field,
        seed=metadata['seed'],
        sources_file=sources_file,
        integration_time=args.delta_time,
        bkgnd_file=background_filename,
        gain_map_file=gain_map_filename,
        dk_c_file=dark_current_filename
    )

    observation_filename = 'O' + sources_file.name[1:]
    observation_filename = out_folder.joinpath(observation_filename)

    field.export_field(filename=observation_filename)

    metadata = {
        'delta_time': args.delta_time,
        'psf_width': args.psf_width,
        'has_background': args.background,
        'has_gain_map': args.gain_map,
        'has_dark_current': args.dark_current,
    }

    save_metadata(
        metadata=metadata,
        filename=observation_filename
    )


if __name__ == "__main__":
    main()
