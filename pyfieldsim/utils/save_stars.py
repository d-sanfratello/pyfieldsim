import numpy as np

from pathlib import Path

from .metadata import save_metadata


def save_stars(stars, data_file, saved_ids, options=None):
    stars = np.asarray(stars)
    n_stars = stars.shape[0]

    output = Path('R' + data_file.stem[1:]).with_suffix('.h5')
    suffix = output.suffix
    if options is not None:
        output = Path(output.stem + f"_{''.join(options)}").with_suffix(suffix)

    mu_x = np.zeros(n_stars, dtype=float)
    mu_y = np.zeros(n_stars, dtype=float)
    A = np.zeros(n_stars, dtype=float)

    for _, s in enumerate(stars):
        mu_x[_] = np.round(
            s.mu[0],
            decimals=find_rounding_decimals(s.fmt_mu_x(s.mu[0]))
        )
        mu_y[_] = np.round(
            s.mu[1],
            decimals=find_rounding_decimals(s.fmt_mu_y(s.mu[1]))
        )
        A[_] = np.round(
            s.A,
            decimals=find_rounding_decimals(s.fmt_A(s.A))
        )

    metadata = {
        'n_stars': n_stars,
        'mu_x': mu_x.tolist(),
        'mu_y': mu_y.tolist(),
        'A': A.tolist(),
        'id': saved_ids
    }

    save_metadata(metadata, output)


def find_rounding_decimals(str_):
    mantissa = str_.split('e')[0]
    exponent = str_.split('e')[1]
    decimals = len(mantissa.split('.')[1]) - int(exponent) - 1

    return decimals
