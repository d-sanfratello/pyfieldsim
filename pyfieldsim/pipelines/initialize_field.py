import argparse as ag
import os

from pathlib import Path

from pyfieldsim.core.fieldtypes.sources import Sources
from pyfieldsim.utils.parse import (get_filename,)


def main():
    """
    Pipeline to initialize the field of stars.
    """
    parser = ag.ArgumentParser(
        prog='fs-initialize',
        usage=__doc__
    )
    parser.add_argument("-f", "--field-size", type=int,
                        dest='field_size', default=100,
                        help="Size of one side of the final field.")
    parser.add_argument("-m", "--min-mass", type=float,
                        dest='m_min', default=1,
                        help="Lower bound on mass for the Salpeter-like IMF.")
    parser.add_argument("-M", "--max-mass", type=float,
                        dest='m_max', default=350,
                        help="Upper bound on mass for the Salpeter-like IMF.")
    parser.add_argument("-d", "--density", type=float,
                        dest='density', default=2e-3,
                        help="Density of stars in the field.")
    parser.add_argument("-i", "--imf-exponent", type=float,
                        dest='e_imf', default=2.4,
                        help="Exponent of the IMF power-law.")
    parser.add_argument("-l", "--lm-exponent", type=float,
                        dest='e_lm', default=3,
                        help="Exponent in the mass-luminosity relation.")
    parser.add_argument("-c", "--lm-const", type=float,
                        dest='cst_lm', default=1,
                        help="Multiplication constant in the mass-luminosity "
                             "relation.")
    parser.add_argument("-o", "--output",
                        dest='out_folder', default=None,
                        help="The folder where to save the output of this "
                             "pipeline.")
    parser.add_argument("--seed", type=int,
                        dest='seed', default=None,
                        help="The seed for the simulation process.")

    args = parser.parse_args()

    field_size = (args.field_size, args.field_size)

    if args.out_folder is None:
        out_folder = Path(os.getcwd())
    else:
        out_folder = Path(args.out_folder)

    filename = get_filename(out_folder=out_folder)

    source_list = Sources(field_size)
    source_list.initialize_field(
        m_min=args.m_min,
        m_max=args.m_max,
        density=args.density,
        e_imf=args.e_imf,
        e_lm=args.e_lm,
        cst_lm=args.cst_lm,
        seed=args.seed,
    )
    source_list.export_sources(filename=filename)


if __name__ == "__main__":
    main()
