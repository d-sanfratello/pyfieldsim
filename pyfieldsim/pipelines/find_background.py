import argparse as ag
import matplotlib.pyplot as plt
import numpy as np
import os

from scipy.stats import norm

from pathlib import Path

from pyfieldsim.core.fieldtypes.field import Field
from pyfieldsim.utils.metadata import save_metadata


def main():
    """
    Pipeline to evaluate the background over a field.
    """
    parser = ag.ArgumentParser(
        prog='fs-eval-background',
        usage=__doc__,
    )
    parser.add_argument('data',
                        help="The data over which the background is "
                             "evaluated.")
    parser.add_argument("-o", "--output",
                        dest='out_folder', default=None,
                        help="The folder where to save the output of this "
                             "pipeline.")

    args = parser.parse_args()

    data_file = Path(args.data)
    field = Field.from_field(data_file)

    out_folder = Path(os.getcwd())
    if args.out_folder is not None:
        out_folder = Path(args.out_folder)
    data_file = data_file.with_suffix('')
    out_stem = out_folder.joinpath(data_file.name + '_bkg_analysis.h5')

    fig, axs = plt.subplots(nrows=2)
    ax_0, ax_1 = axs.flat
    ax_0.grid()
    ax_1.grid()

    px = field.field.flatten()
    n, bins, patches = ax_0.hist(
        px,
        histtype='step',
        bins=int(np.sqrt(len(px))) // 4,
    )
    centers = (bins[:-1] + bins[1:]) / 2
    dist_norm = (np.diff(bins) * n).sum()

    mean = np.mean(px)
    std = np.std(px, ddof=1)

    x = np.arange(bins.min(), bins.max())
    pdf = norm.pdf(x, mean, std) * dist_norm
    ax_0.plot(x, pdf)

    ax_0.errorbar(
        centers, n, np.sqrt(n),
        linestyle='', capsize=2, color='blue', marker='.'
    )
    ax_0.set_yscale('log')
    ax_0.set_ylim(1e-1, 10 ** np.ceil(np.log10(n.max())))
    ax_0.set_xticklabels([])

    ax_0.axvline(mean, 0, 1, color='red', ls='solid')
    ax_0.axvline(mean + std, 0, 1, color='grey', ls='-',
                 label=r'1$\sigma$')
    ax_0.axvline(mean + 2.5 * std, 0, 1, color='grey', ls='--', alpha=0.75,
                 label=r'2.5$\sigma$')
    ax_0.axvline(mean + 3 * std, 0, 1, color='grey', ls='-.', alpha=0.5,
                 label=r'3$\sigma$')

    ax_1.axvline(mean, 0, 1, color='red', ls='solid')
    ax_1.axvline(mean + std, 0, 1, color='grey', ls='-')
    ax_1.axvline(mean + 2.5 * std, 0, 1, color='grey', ls='--', alpha=0.75)
    ax_1.axvline(mean + 3 * std, 0, 1, color='grey', ls='-.', alpha=0.5)

    expected = norm.pdf(centers, mean, std) * dist_norm
    err_diffs = (n - expected) / np.sqrt(n)
    ax_1.scatter(centers, err_diffs, marker='.', color='blue')

    ax_0.legend(loc='best')
    ax_0.set_xlim(bins.min(), bins.max())
    ax_1.set_xlim(bins.min(), bins.max())

    ax_0.set_ylabel('Occurrences')
    ax_1.set_ylabel('Norm. res.')
    ax_1.set_xlabel('Counts on the sensor')

    plt.tight_layout()
    fig.savefig(out_stem.with_suffix('.pdf'))

    metadata = {
        'mean': mean,
        'std': std,
        '2.5s': mean + 2.5 * std,
        '3s': mean + 3 * std
    }

    save_metadata(
        metadata,
        out_stem.with_suffix('.h5')
    )

    print('Background analysis')
    print('=====')
    for k, v in metadata.items():
        print(f'{k}  \t{v}')
    print('=====')


if __name__ == "__main__":
    main()
