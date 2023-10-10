import argparse as ag
import os

from pathlib import Path

from pyfieldsim.core.fieldtypes.noise import background


def main():
    """
    Pipeline to generate a background field for the simulation.
    """
    parser = ag.ArgumentParser(
        prog='fs-generate-background',
        usage=__doc__,
    )
    parser.add_argument('sources')
    parser.add_argument('--snr', type=float,
                        dest='snr', default=10,
                        help="Signal to Noise Ratio of the brightest source "
                             "in the field.")
    parser.add_argument('--sigma', type=float,
                        dest='sigma', default=None,
                        help="Sigma of the normal distribution of the "
                             "bacground values.")
    parser.add_argument('-r', '--relative-sigma', type=float,
                        dest='rel_var', default=0.1,
                        help="Relative sigma of the normal distribution of "
                             "the background values.")
    parser.add_argument("-o", "--output",
                        dest='out_folder', default=None,
                        help="The folder where to save the output of this "
                             "pipeline.")

    args = parser.parse_args()

    if args.out_folder is None:
        out_folder = Path(os.getcwd())
    else:
        out_folder = Path(args.out_folder)

    sigma = rel_sigma = None
    if args.sigma is not None:
        sigma = args.sigma
    else:
        rel_sigma = args.rel_var

    sources_file = Path(args.sources)
    background_field = background(
        sources_file,
        snr=args.snr, sigma=sigma,
        rel_var=rel_sigma
    )

    filename = 'B' + sources_file.name[1:]
    filename = out_folder.joinpath(filename)

    background_field.export_field(filename=filename)


if __name__ == "__main__":
    main()
