import argparse as ag
import h5py

from pathlib import Path

from pyfieldsim.utils.metadata import read_metadata


def main():
    parser = ag.ArgumentParser(
        prog='fs-show-meta',
        description='',
    )
    parser.add_argument('file')

    args = parser.parse_args()

    data_file = Path(args.file)

    print(f'=====')
    if data_file.stem.startswith('O'):
        print(f'Background analysis for file {data_file.stem}')
        bkg_analysis = Path(
            data_file.stem + '_bkg_analysis'
        ).with_suffix('.h5')

        metadata = read_metadata(bkg_analysis)
    elif data_file.stem.startswith('R'):
        print(f'Recovered stars from file {data_file.stem}')

        if not data_file.stem.endswith('_meta'):
            data_file = Path(
                data_file.stem + '_meta.h5'
            )

        with h5py.File(data_file, 'r') as f:
            recovered_stars = {
                k: v for k, v in f.attrs.items()
            }

        n_stars = int(recovered_stars['n_stars'])
        mu_x = eval(recovered_stars['mu_x'])
        mu_y = eval(recovered_stars['mu_y'])
        A = eval(recovered_stars['A'])

        metadata = {
            'N stars': n_stars
        }
        for _ in range(metadata['N stars']):
            metadata[f's_{_}'] = f'({mu_x[_]}, {mu_y[_]}), A = {A[_]}'
    else:
        print(f'Metadata for file {data_file.stem}')
        metadata = read_metadata(data_file)

    print(f'-----')
    for k, v in metadata.items():
        print(f'{k}\t:\t{v}')
    print('=====')


if __name__ == "__main__":
    main()
