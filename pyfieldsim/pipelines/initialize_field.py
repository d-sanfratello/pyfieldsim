import argparse as ag
import os

from pathlib import Path

# from pyfieldsim.core.psf import GaussKernel
from pyfieldsim.field import Field
# from pyfieldsim.observation import Observation
from pyfieldsim.utils.parse import (get_filename,
                                    parse_dtype)


def main():
    parser = ag.ArgumentParser(
        prog='fs-initialize',
        description='This script shows a fits or h5 image of a spectrum.',
    )
    parser.add_argument("-f", "--field-size", type=int,
                        dest='field_size', default=100,
                        help="")
    parser.add_argument("-d", "--density", type=float,
                        dest='density', default=2e-3,
                        help="")
    parser.add_argument("-i", "--imf-exponent", type=float,
                        dest='e_imf', default=2.4,
                        help="")
    parser.add_argument("-l", "--lm-exponent", type=float,
                        dest='e_lm', default=3,
                        help="")
    parser.add_argument("-c", "--lm-const", type=float,
                        dest='cst_lm', default=1,
                        help="")
    parser.add_argument("-o", "--output",
                        dest='out_folder', default=None,
                        help="")
    parser.add_argument("--data-type",
                        dest='datatype', default='luminosity',
                        help="")
    parser.add_argument("--seed", type=int,
                        dest='seed', default=None,
                        help="")

    args = parser.parse_args()

    field_size = (args.field_size, args.field_size)

    if args.out_folder is None:
        out_folder = Path(os.getcwd())
    else:
        out_folder = Path(args.out_folder)

    dtype = parse_dtype(args)

    filename = get_filename(args, out_folder=out_folder, dtype=dtype)

    field = Field(field_size)
    field.initialize_field(
        density=args.density,
        e_imf=args.e_imf,
        e_lm=args.e_lm,
        cst_lm=args.cst_lm,
        seed=args.seed,
        datatype=args.datatype,
        force=False
    )

    # FIXME: output files should be hdf5 with one dataset for data and one
    #  dataset for stored simulation parameters.
    field.export_field(filename=filename)

    # -------------------------------------------------------------------------
    # Complete operation
    # -------------------------------------------------------------------------
    # field = Field((100, 100))
    # field.initialize_field(density=0.002, datatype='luminosity')
    #
    # field.show_field('true')
    #
    # observation = Observation(field)
    #
    # # counting single stars
    # stars, coords = observation.count_single_stars()
    # print(f'{len(stars)} single stars')
    #
    # psf = GaussKernel(sigma=3)
    #
    # field.record_field(kernel=psf,
    #                    delta_time=1000, snr=10, bgnd_rel_var=0.05,
    #                    gain_mean=1, gain_rel_var=0.01,
    #                    dk_c_fraction=0.1, dk_c_rel_var=0.01, dk_c=1,
    #                    force=True)
    #
    # field.save_field(name='stars')
    # field.show_field('exposure')


if __name__ == "__main__":
    main()
