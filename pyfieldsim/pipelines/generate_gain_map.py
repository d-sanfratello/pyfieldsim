import argparse as ag
import os

from pathlib import Path

from pyfieldsim.core.fieldtypes.sensor import create_gain_map


def main():
    """
    Pipeline to generate a gain map for the CCD.
    """
    parser = ag.ArgumentParser(
        prog='fs-generate-gain_map',
        usage=__doc__,
    )
    parser.add_argument('sources')
    parser.add_argument('-m', '--mean-gain', type=float,
                        dest='mean_gain', default=1,
                        help='Mean of the gain map of the sensor.')
    parser.add_argument('-r', '--relative-sigma', type=float,
                        dest='rel_var', default=0.05,
                        help='Relative sigma for the distribution of gain '
                             'values.')
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
    gain_map = create_gain_map(
        sources_file,
        mean_gain=args.mean_gain,
        rel_var=args.rel_var
    )

    filename = 'G' + sources_file.name[1:]
    filename = out_folder.joinpath(filename)

    gain_map.export_field(filename=filename)


if __name__ == "__main__":
    main()
