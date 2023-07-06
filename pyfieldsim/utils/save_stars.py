import numpy as np

from pathlib import Path

from metadata import save_metadata


def save_stars(stars, data_file, options=None):
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
        mu_x[_] = np.round(s.mu[0], decimals=1)
        mu_y[_] = np.round(s.mu[1], decimals=1)
        A[_] = np.round(s.A, decimals=1)

    metadata = {
        'n_stars': n_stars,
        'mu_x': mu_x.tolist(),
        'mu_y': mu_y.tolist(),
        'A': A.tolist()
    }

    save_metadata(metadata, output)
