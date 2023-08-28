import numpy as np

from pathlib import Path

from .metadata import save_metadata, read_metadata
from pyfieldsim.core.stars.star import new_star


def save_point_stars(stars, data_file, options=None):
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
        mu_x[_] = s.mu[0]
        mu_y[_] = s.mu[1]

        A[_] = s.A

    metadata = {
        'n_stars': n_stars,
        'mu_x': mu_x.tolist(),
        'mu_y': mu_y.tolist(),
        'A': A.tolist(),
        'id': np.linspace(1, n_stars, n_stars, dtype=int),
    }

    save_metadata(metadata, output)


def save_stars(stars, data_file, saved_ids, hyp_psf, options=None):
    stars = np.asarray(stars)
    n_stars = stars.shape[0]

    output = Path('R' + data_file.stem[1:]).with_suffix('.h5')
    suffix = output.suffix
    if options is not None:
        output = Path(output.stem + f"_{''.join(options)}").with_suffix(suffix)

    mu_x = np.zeros(n_stars, dtype=float)
    mu_y = np.zeros(n_stars, dtype=float)
    A = np.zeros(n_stars, dtype=float)
    pos_errors = np.zeros((n_stars, 2, 2), dtype=float)
    A_errors = np.zeros((n_stars, 2), dtype=float)

    sigma = stars[0].sigma[0, 0]

    for _, s in enumerate(stars):
        try:
            decimals = find_rounding_decimals(s.fmt_mu_x(s.mu[0]))
            value = np.round(s.mu[0], decimals=decimals)
        except TypeError:
            value_str = find_rounding_from_errs(s.mu[0], s.pos_error[0])
            value = float(value_str)
        finally:
            # noinspection PyUnboundLocalVariable
            mu_x[_] = value

        try:
            decimals = find_rounding_decimals(s.fmt_mu_y(s.mu[1]))
            value = np.round(s.mu[1], decimals=decimals)
        except TypeError:
            value_str = find_rounding_from_errs(s.mu[1], s.pos_error[1])
            value = float(value_str)
        finally:
            # noinspection PyUnboundLocalVariable
            mu_y[_] = value

        try:
            decimals = find_rounding_decimals(s.fmt_A(s.A))
            value = np.round(s.A, decimals=decimals)
        except TypeError:
            value_str = find_rounding_from_errs(s.A, s.A_error)
            value = float(value_str)
        finally:
            # noinspection PyUnboundLocalVariable
            A[_] = value

        pos_errors[_] = np.asarray(s.pos_error)
        A_errors[_] = np.asarray(s.A_error)

    metadata = {
        'n_stars': n_stars,
        'mu_x': mu_x.tolist(),
        'mu_y': mu_y.tolist(),
        'A': A.tolist(),
        'sigma': (sigma * np.ones(n_stars)).tolist(),
        'id': saved_ids,
        'pos_errors': pos_errors.tolist(),
        'A_errors': A_errors.tolist(),
        'hyp_psf': hyp_psf
    }

    save_metadata(metadata, output)


def load_stars(data_file, options=None):
    file = Path('R' + data_file.stem[1:]).with_suffix('.h5')
    suffix = file.suffix
    if options is not None:
        file = Path(file.stem + f"_{''.join(options)}").with_suffix(suffix)

    metadata = read_metadata(file)

    star_obj = zip(
        metadata['A'],
        metadata['mu_x'],
        metadata['mu_y'],
        metadata['sigma'],
        metadata['pos_errors'],
        metadata['A_errors']
    )

    stars = [
        new_star(
            A=float(A),
            mu=(float(mu_x), float(mu_y)),
            sigma=float(sigma) * np.eye(2),
            pos_error=np.asarray(p_err).flatten(),
            A_error=np.asarray(A_err)
        )
        for A, mu_x, mu_y, sigma, p_err, A_err in star_obj
    ]

    return stars, metadata['pos_errors'], metadata['id'], metadata['hyp_psf']


def find_rounding_decimals(str_):
    mantissa = str_.split('e')[0]
    exponent = str_.split('e')[1]

    if mantissa.find('.') > -1:
        decimals = len(mantissa.split('.')[1]) - int(exponent)
    else:
        decimals = 0 - int(exponent)

    return decimals


def find_rounding_from_errs(value, errors):
    min_error = min(errors[0], errors[1])
    odg_err = np.floor(np.log10(min_error)).astype(int)
    odg_meas = np.floor(np.log10(value)).astype(int)

    odg = odg_meas - odg_err
    fmt = f"{{0:{odg}}}".format

    return fmt(value)
