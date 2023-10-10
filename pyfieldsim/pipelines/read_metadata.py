import argparse as ag
import h5py

from pathlib import Path

from pyfieldsim.utils.metadata import read_metadata


def main():
    """
    Pipeline to show metadata from a file.
    """
    parser = ag.ArgumentParser(
        prog='fs-show-meta',
        usage=__doc__,
    )
    parser.add_argument('file',
                        help="The file whose metadata are shown.")

    args = parser.parse_args()

    data_file = Path(args.file).with_suffix('')

    print(f'=====')
    if data_file.stem.startswith('O'):
        print(f'Metadata and background analysis for file {data_file.name}')
        metadata = read_metadata(data_file)
        bkg_analysis = Path(
            data_file.name + '_bkg_analysis.h5'
        )

        bkg_metadata = read_metadata(bkg_analysis)
    elif data_file.stem.startswith('R'):
        print(f'Recovered stars from file {data_file.name}')

        if not data_file.stem.endswith('_meta'):
            data_file = Path(
                data_file.name + '_meta.h5'
            )

        with h5py.File(data_file, 'r') as f:
            recovered_stars = {
                k: v for k, v in f.attrs.items()
            }

        n_stars = int(recovered_stars['n_stars'])
        mu_x = eval(recovered_stars['mu_x'])
        mu_y = eval(recovered_stars['mu_y'])
        A = eval(recovered_stars['A'])
        ids = eval(recovered_stars['id'])

        metadata = {
            'N stars': n_stars
        }
        for _, idx in enumerate(ids):
            metadata[f's_{idx}'] = f'({mu_x[_]}, {mu_y[_]})'.ljust(20) \
                                   + f'A = {A[_]}'
    else:
        print(f'Metadata for file {data_file.name}')
        metadata = read_metadata(data_file)

    print(f'-----')
    for k, v in metadata.items():
        print(f'{k}\t:\t{v}')
    if data_file.stem.startswith('O'):
        print('-----')
        for k, v in bkg_metadata.items():
            print(f'{k}\t:\t{v}')
    print('=====')


if __name__ == "__main__":
    main()
