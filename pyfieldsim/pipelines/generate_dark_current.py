import argparse as ag
import os

from pathlib import Path

from pyfieldsim.core.fieldtypes.sensor import (
    create_dark_current,
    create_bad_pixels
)


def main():
    parser = ag.ArgumentParser(
        prog='fs-generate-dark_current',
        description='',
    )
    parser.add_argument('sources')
    parser.add_argument('-b', '--background_fraction', type=float,
                        dest='bgnd_fraction', default=0.1,
                        help='')
    parser.add_argument('-t', '--delta-time', type=float,
                        dest='delta_time', default=1,
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
    dark_current = create_dark_current(
        sources_file,
        b_fraction=args.bgnd_fraction,
        delta_time=args.delta_time
    )

    dark_current = create_bad_pixels(
        sources_file,
        dark_current
    )

    filename = 'C' + sources_file.name[1:]
    filename = out_folder.joinpath(filename)

    dark_current.export_field(filename=filename)


if __name__ == "__main__":
    main()
