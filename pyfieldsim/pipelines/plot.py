import argparse as ag
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

from pyfieldsim.core.fieldtypes.field import Field
from pyfieldsim.utils.metadata import read_metadata


def main():
    parser = ag.ArgumentParser(
        prog='fs-plot',
        description='',
    )
    parser.add_argument('data')
    parser.add_argument("-o", "--output",
                        dest='out_folder', default=None,
                        help="")
    parser.add_argument("--crop", action='store_true',
                        dest='crop', default=False,
                        help="")
    parser.add_argument("--no-log", action='store_true',
                        dest='no_log', default=False,
                        help="")

    args = parser.parse_args()

    out_folder = None
    if args.out_folder is not None:
        out_folder = Path(args.out_folder)

    data_file = Path(args.data)

    if data_file.name.startswith('S'):
        sources = True
        metadata = read_metadata(
            Path(data_file.stem + '_meta').with_suffix('.h5')
        )
    else:
        sources = False
        metadata = read_metadata(
            Path('S' + data_file.name[1:])
        )

    if sources:
        field = Field.from_sources(data_file)
    else:
        field = Field.from_field(data_file)

    fig = plt.figure()
    ax = fig.gca()

    plot_field = field.field
    if not args.no_log:
        plot_field = np.log10(plot_field)

    img = ax.imshow(plot_field, origin='lower', cmap='Greys')
    plt.colorbar(img)

    if args.crop and not (data_file.stem.startswith('O') or
                          data_file.stem.startswith('L')):
        pad = metadata['pad']
        ext_shape = metadata['ext_shape']

        crop_coords = [
            [pad[0], ext_shape[0] - pad[0]],
            [pad[1], ext_shape[1] - pad[1]]
        ]

        ax.plot([crop_coords[0][0], crop_coords[0][0]],
                [crop_coords[1][0], crop_coords[1][1]],
                linestyle='solid', color='red')
        ax.plot([crop_coords[0][1], crop_coords[0][1]],
                [crop_coords[1][0], crop_coords[1][1]],
                linestyle='solid', color='red')
        ax.plot([crop_coords[1][0], crop_coords[1][1]],
                [crop_coords[0][0], crop_coords[0][0]],
                linestyle='solid', color='red')
        ax.plot([crop_coords[1][0], crop_coords[1][1]],
                [crop_coords[0][1], crop_coords[0][1]],
                linestyle='solid', color='red')

    if out_folder is not None:
        fig.savefig(out_folder.joinpath(f"plot_{data_file.name[0]}.pdf"))

    plt.show()


if __name__ == "__main__":
    main()
