import argparse as ag

import h5py
import numpy as np

from pathlib import Path

from pyfieldsim.core.stars.find_utils import (
    dist,
    select_hypothesis,
)

from pyfieldsim.utils.save_stars import load_stars, save_stars

# TODO: Add plot


# noinspection PyArgumentList,PyUnboundLocalVariable,PyTypeChecker
def main():
    parser = ag.ArgumentParser(
        prog='fs-anti-alias',
        description='',
    )
    parser.add_argument('stars')
    parser.add_argument('-f', action='store_true',
                        dest='is_flat', default=False,
                        help="")
    parser.add_argument("--options",
                        dest='options', default=None,
                        help="")

    args = parser.parse_args()

    stars, pos_errors, saved_ids, hyp_psf = load_stars(Path(args.stars))
    sigma = stars[0].sigma[0, 0]

    print("- Removing aliases")
    if args.is_flat:
        n_limit = 3
    else:
        n_limit = 2

    if hyp_psf == '1':
        psf_stars = stars[:1]

        analized_stars_idx = [
            _ + 1 for _, s in enumerate(stars[1:])
            if dist(s.mu, psf_stars[0].mu) <= n_limit * sigma
        ]
    else:
        psf_stars = stars[:2]

        analized_stars_idx = [
            _ + 2 for _, s in enumerate(stars[2:])
            if dist(s.mu, psf_stars[0].mu) <= n_limit * sigma
            or dist(s.mu, psf_stars[1].mu) <= n_limit * sigma
        ]

    if len(analized_stars_idx) > 1:
        psf_stars = np.array([
            p.mu for p in psf_stars
        ])
        analized_stars = np.array([
            stars[_].mu for _ in analized_stars_idx
        ])

        mean_analized_stars = np.mean(analized_stars, axis=0)

        for p in psf_stars:
            if dist(p, mean_analized_stars) <= sigma:
                for idx in sorted(analized_stars_idx, reverse=True):
                    del stars[idx]
                    del saved_ids[idx]
                    del pos_errors[idx]
                break

    ctr = len(psf_stars)
    remaining_stars = stars[ctr:]
    while len(remaining_stars) > 0:
        ref_star = remaining_stars[0]
        ctr += 1

        analized_stars_idx = [
            _ + ctr for _, s in enumerate(stars[ctr:])
            if dist(s.mu, ref_star.mu) <= n_limit * sigma
        ]
        if len(analized_stars_idx) > 1:
            analized_stars = np.array([
                stars[_].mu for _ in analized_stars_idx
            ])
            mean_analized_stars = np.mean(analized_stars, axis=0)

            if dist(ref_star.mu, mean_analized_stars) <= sigma:
                for idx in sorted(analized_stars_idx, reverse=True):
                    del stars[idx]
                    del saved_ids[idx]
                    del pos_errors[idx]

        remaining_stars = stars[ctr:]

    if args.options is None:
        options = 'AA'
    else:
        options = [args.options, 'AA']

    save_stars(
        stars,
        Path('./R_stars'),
        saved_ids,
        hyp_psf,
        options=options
    )


if __name__ == "__main__":
    main()
