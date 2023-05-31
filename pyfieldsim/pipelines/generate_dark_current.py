import argparse as ag
import os

from pathlib import Path

from pyfieldsim.core.fieldtypes.sensor import create_dark_current


def main():
    parser = ag.ArgumentParser(
        prog='fs-generate-dark_current',
        description='',
    )
    parser.add_argument('sources')
    parser.add_argument('-b', '--background_fraction', type=float,
                        dest='bgnd_fraction', default=0.1,
                        help='')
    parser.add_argument('-r', '--relative-sigma', type=float,
                        dest='rel_var', default=0.05,
                        help='')
    parser.add_argument("-o", "--output",
                        dest='out_folder', default=None,
                        help="")

    args = parser.parse_args()

    if args.out_folder is None:
        out_folder = Path(os.getcwd())
    else:
        out_folder = Path(args.out_folder)

    sources_file = Path(args.sources)
    dark_current = create_dark_current(
        sources_file,
        b_fraction=args.bgnd_fraction,
        rel_var=args.rel_var
    )

    filename = 'C' + sources_file.name[1:]
    filename = out_folder.joinpath(filename)

    dark_current.export_field(filename=filename)


if __name__ == "__main__":
    main()
