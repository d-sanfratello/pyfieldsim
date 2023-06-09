import argparse as ag

import h5py
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

from pyfieldsim.core.fieldtypes.field import Field
from pyfieldsim.utils.metadata import read_metadata


def main():
    parser = ag.ArgumentParser(
        prog='fs-eval-background',
        description='',
    )
    parser.add_argument('data')
    parser.add_argument("-o", "--output",
                        dest='out_folder', default=None,
                        help="")
    parser.add_argument('-a', '--areas', required=True,
                        dest='areas',
                        help="")

    args = parser.parse_args()

    out_folder = None
    if args.out_folder is not None:
        out_folder = Path(args.out_folder)

    data_file = Path(args.data)
    field = Field.from_field(data_file)
    metadata = read_metadata(
        Path("S" + data_file.stem[1:] + "_meta").with_suffix('.h5')
    )

    areas = eval(args.areas)
    shape = metadata['shape']
    grid = np.asarray([x for x in np.ndindex(shape)])

    b_mean = np.inf
    b_std = None
    region = None
    for a in areas:
        c, r = a
        c = np.array([c[1], c[0]])

        valid_coords = np.asarray([
            x for x in grid if np.linalg.norm([c, x]) <= r
        ])

        b_values = np.asarray([field.field[x[0], x[1]] for x in valid_coords])

        if b_values.mean() < b_mean:
            b_mean = b_values.mean()
            b_std = b_values.std(ddof=1)
            region = (c, r)
        elif b_values.mean() == b_mean and b_values.std(ddof=1) >= b_std:
            b_mean = b_values.mean()
            b_std = b_values.std(ddof=1)
            region = ((c[0], c[1]), r)

    with h5py.File(
            out_folder.joinpath(data_file.stem[1:] + '_bgnd_mean.h5')
    ) as f:
        region_dset = f.create_dataset('region',
                                       dtype=int, shape=(2,))
        region_dset[0:] = np.array(region[0])

        radius_dset = f.create_dataset('radius',
                                       dtype=float,
                                       shape=(1,))
        radius_dset[0:] = region[1]

        bgnd_dset = f.create_dataset('background',
                                     dtype=float,
                                     shape=(2,))
        bgnd_dset[0:] = np.array([b_mean, b_std])

    print(f"background from region: c = {region[0]}, r = {region[1]}")
    print(f"=====")
    print(f"b = {b_mean} +- {b_std}")

    return


if __name__ == "__main__":
    main()
