import argparse as ag
import os

from pathlib import Path

from pyfieldsim.core.fieldtypes.noise import ph_noise


def main():
    parser = ag.ArgumentParser(
        prog='fs-generate-ph-noise',
        description='',
    )
    parser.add_argument('sources')
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
    w_ph_noise = ph_noise(
        sources_file=sources_file,
        delta_time=float(args.delta_time)
    )

    filename = 'P' + sources_file.name[1:]
    filename = out_folder.joinpath(filename)

    w_ph_noise.export_field(filename=filename)


if __name__ == "__main__":
    main()
