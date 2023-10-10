import argparse as ag

import h5py
import numpy as np

from pathlib import Path

from pyfieldsim.core.stars.find_utils import (
    dist,
    plot_recovered_stars,
    select_hypothesis,
)

from pyfieldsim.core.fieldtypes.field import Field
from pyfieldsim.utils.metadata import read_metadata
from pyfieldsim.utils.save_stars import save_stars


# noinspection PyArgumentList,PyUnboundLocalVariable,PyTypeChecker
def main():
    """
    Pipeline to export a set of identified stars from a folder structure to
    a single metadata file.
    """
    parser = ag.ArgumentParser(
        prog='fs-export-stars',
        usage=__doc__,
    )
    parser.add_argument('main_folder', nargs='?', default='./')
    parser.add_argument('-r', required=True,
                        dest='initial_radius', type=float,
                        help="The initial search radius for the algorithm.")
    parser.add_argument('-f', action='store_true',
                        dest='is_flat', default=False,
                        help="Wether the field is flat or not.")
    parser.add_argument("-s", "--sources", action='store_true',
                        dest='show_sources', default=False,
                        help="Wheter to show the true sources in the field "
                             "plot or not.")
    parser.add_argument("--options",
                        dest='options', default=None,
                        help="String to be added to the name of the output.")

    args = parser.parse_args()

    if not isinstance(args.main_folder, str):
        raise ValueError("Too many paths to unpack.")

    main_folder = Path(args.main_folder)

    samplings_folder = main_folder.joinpath('samplings')
    plot_folder = main_folder.joinpath('plots')

    initial_radius = args.initial_radius

    psf_samplings = {}
    for psf in samplings_folder.glob('sampling_output_psf*/'):
        psf_samplings[psf.name[-1]] = psf

    if len(psf_keys := list(psf_samplings.keys())) != 2:
        raise ValueError("You have not run the two psfs, yet.")
    psf_keys = np.sort(np.asarray(psf_keys, dtype=int)).astype(str).tolist()

    post_psf = {}
    logZ_psf = {}
    for psf_k in psf_keys:
        print(f"-- Loading psf posterior {psf_k}")
        path = samplings_folder.joinpath(f'sampling_output_psf{psf_k}')
        post_psf[psf_k], logZ_psf[psf_k] = load_data(path)

    # If the two stars lies farther than the initial radius from each other,
    # the two stars hypothesis is discarded.
    median_0 = (
        np.median(post_psf['2']['mu_x0']),
        np.median(post_psf['2']['mu_y0'])
    )
    median_1 = (
        np.median(post_psf['2']['mu_x1']),
        np.median(post_psf['2']['mu_y1'])
    )
    s0_s1_dist = dist(median_0, median_1)
    if s0_s1_dist > initial_radius:
        logZ_psf['2'] = -np.inf

    stars = []
    pos_errors = []

    # 10^0.5 -> substantial evidence
    # 10^1 -> strong evidence
    # 10^1.5 -> very strong evidence
    # 10^2 -> decisive evidence
    sigma, b_u, hyp_psf = select_hypothesis(
        hyp_1='1', hyp_2='2',
        logZ=logZ_psf,
        logB_lim_hyp2=-1 * np.log(10),
        stars=stars,
        pos_errors=pos_errors,
        post_1=post_psf['1'],
        post_2=post_psf['2'],
        is_flat=args.is_flat,
    )

    saved_ids = [_ + 1 for _ in range(len(stars))]

    stars_path = samplings_folder.glob('sampling_output_star_*/')

    stars_samplings = {}
    for star in stars_path:
        stars_samplings[star.name[21:]] = star
    star_keys = list(stars_samplings.keys())
    star_keys = np.sort(np.asarray(star_keys, dtype=int)).astype(str).tolist()

    post_star = {}
    logZ_star = {}
    post_bgnd = {}
    logZ_bgnd = {}

    if not args.is_flat:
        bgnd_path = samplings_folder.glob('sampling_output_bgnd_*/')

        bgnd_samplings = {}
        for bgnd in bgnd_path:
            bgnd_samplings[bgnd.name[21:]] = bgnd
        bgnd_keys = list(bgnd_samplings.keys())
        bgnd_keys = np.sort(np.asarray(bgnd_keys, dtype=int)).astype(str).tolist()

        for star_k in bgnd_keys:
            print(f"-- Loading star and background posterior {star_k}")

            path_s = samplings_folder.joinpath(f'sampling_output_star_{star_k}')
            post_star[star_k], logZ_star[star_k] = load_data(path_s)

            path_b = samplings_folder.joinpath(f'sampling_output_bgnd_{star_k}')
            post_bgnd[star_k], logZ_bgnd[star_k] = load_data(path_b)
    else:
        bgnd_keys = star_keys
        for star_k in bgnd_keys:
            print(f"-- Loading star posterior {star_k}")

            path_s = samplings_folder.joinpath(f'sampling_output_star_{star_k}')
            post_star[star_k], logZ_star[star_k] = load_data(path_s)

            post_bgnd[star_k], logZ_bgnd[star_k] = (None, -np.inf)

    for star_k in bgnd_keys:
        sigma, b_u, hyp_s_b = select_hypothesis(
            hyp_1='s', hyp_2='b',
            logZ={
                's': logZ_star[star_k],
                'b': logZ_bgnd[star_k]
            },
            logB_lim_hyp2=0.5 * np.log(10),
            stars=stars,
            pos_errors=pos_errors,
            post_1=post_star[star_k],
            post_2=post_bgnd[star_k],
            is_flat=args.is_flat,
            sigma=sigma
        )

        if hyp_s_b == 's':
            saved_ids.append(int(star_k))

    if args.options is None:
        options = 'noAA'
    else:
        options = [args.options, 'noAA']

    data_field_path = main_folder.glob('L*.h5')
    data_field_path = [p for p in data_field_path
                       if p.name.find('meta') < 0]
    if len(data_field_path) == 0:
        data_field_path = main_folder.glob('O*.h5')
        data_field_path = [p for p in data_field_path
                           if p.name.find('meta') < 0]

    data_field_path = data_field_path[0]

    sources_file = Path('S' + data_field_path.name[1:])
    sources_metadata = read_metadata(sources_file)

    shape = sources_metadata['shape']
    pad = sources_metadata['pad']

    with h5py.File(sources_file, 'r') as f:
        coords = np.asarray(f['coords'])

    for s in coords:
        s[0] -= pad[0]
        s[1] -= pad[1]

    data_field = Field.from_field(data_field_path)

    plot_recovered_stars(
        data_field.field,
        stars=stars,
        pos_errors=pos_errors,
        shape=shape,
        show_sources=args.show_sources,
        is_flat=args.is_flat,
        sources=coords,
        out_path=plot_folder.joinpath(f'recovered_stars_all_noAA.pdf')
    )

    save_stars(
        stars,
        main_folder.joinpath('R_stars'),
        saved_ids,
        hyp_psf,
        options=options
    )


def load_data(path):
    path = Path(path)

    with h5py.File(path.joinpath('cpnest.h5'), 'r') as f:
        post = np.asarray(f['combined']['posterior_samples'])
        logZ = np.asarray(f['combined']['logZ']).reshape((1,))[0]

    return post, logZ


if __name__ == "__main__":
    main()
