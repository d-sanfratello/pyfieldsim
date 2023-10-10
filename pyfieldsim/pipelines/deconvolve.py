import argparse as ag

import matplotlib.pyplot as plt

from pathlib import Path
from skimage import restoration

from pyfieldsim.core.psf import GaussKernel

from pyfieldsim.core.fieldtypes.field import Field
from pyfieldsim.utils.metadata import read_metadata, save_metadata


# noinspection PyArgumentList,PyUnboundLocalVariable,PyTypeChecker
def main():
    """
    Pipeline that applies the Richardson-Lucy deconvolution algorithm to a
    field of stars.
    """
    parser = ag.ArgumentParser(
        prog='fs-deconvolve',
        usage=__doc__,
    )
    parser.add_argument('data_file',
                        help="The field to apply te RL deconvolution to.")
    parser.add_argument('-k', '--psf-width', type=float, required=True,
                        dest='psf_width', default=None,
                        help="The width of the PSF.")
    parser.add_argument("-n", "--iterations", type=int,
                        dest='iterations', default=30,
                        help="Number of iterations for the Richardson-Lucy "
                             "algorithm.")

    args = parser.parse_args()

    data_file = Path(args.data_file)
    data_field = Field.from_field(data_file)

    psf_k = GaussKernel(sigma=args.psf_width, size=5)

    rl_deconv = restoration.richardson_lucy(
        data_field.field, psf_k.kernel,
        num_iter=args.iterations,
        clip=False
    )
    s_metadata = read_metadata(
        'S' + data_file.name[1:]
    )

    rl_field = Field(
        rl_deconv,
        seed=s_metadata['seed']
    )

    observation_filename = Path('L' + data_file.name[1:])
    rl_field.export_field(filename=observation_filename)

    metadata = {
        'psf_width': args.psf_width,
        'iterations': args.iterations,
    }

    save_metadata(
        metadata=metadata,
        filename=observation_filename
    )

    fig = plt.figure()
    ax = fig.gca()

    img = ax.imshow(rl_field.field, origin='lower',
                    cmap='Greys',
                    interpolation='none')
    plt.colorbar(img)

    out_path = observation_filename.parent.joinpath(
        f'RL_k{args.psf_width}_n{args.iterations}.pdf'
    )

    fig.savefig(out_path)

    plt.close(fig)


if __name__ == "__main__":
    main()
