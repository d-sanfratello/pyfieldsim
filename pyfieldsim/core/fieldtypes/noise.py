import numpy as np

from pyfieldsim.core.fieldtypes.field import Field


def ph_noise(sources_file, delta_time):
    sources_field = Field.from_sources(sources_file)

    rng = np.random.default_rng(seed=sources_field.metadata['_seed'])

    # Generating the auxiliary fields and simulating exposure time
    # variation with `delta_time`
    exposed_field = sources_field * delta_time

    # Poisson noise generation of photon noise around the (integer)
    # luminosity of the star as mean
    w_ph_noise = rng.poisson(exposed_field)
    w_ph_noise = np.where(
        w_ph_noise < 0, 0, w_ph_noise
    )

    return Field(
        w_ph_noise,
        seed=sources_field.metadata['_seed'],
        sources_file=str(sources_file),
        ph_noise_file=None,
        bkgnd_file=None,
        psf_file=None,
        gain_map_file=None,
        dk_c_file=None
    )


def background(sources_file,
               snr,
               sigma=None, rel_var=None):
    ph_noise_field_filename = 'P' + sources_file[1:]
    ph_noise_field_file = Field.from_field(ph_noise_field_filename)

    bgst_star = np.max(sources_file.field)

    mean_bgnd = bgst_star / snr
    if rel_var is not None:
        sigma = rel_var * mean_bgnd
    elif sigma is None:
        raise ValueError("Must pass a sigma or relative sigma value.")

    rng = np.random.default_rng(seed=ph_noise_field_file.metadata['_seed'])

    # Generating the background field to be added to the photon noise
    # contaminated field.
    bgnd = rng.normal(
        loc=mean_bgnd, scale=sigma,
        size=ph_noise_field_file.metadata['ext_shape']
    )
    bgnd = np.where(
        bgnd < 0, 0, bgnd
    )

    return Field(
        bgnd,
        seed=ph_noise_field_file.metadata['_seed'],
        sources_file=str(sources_file),
        ph_noise_file=str(ph_noise_field_file),
        bkgnd_file=None,
        psf_file=None,
        gain_map_file=None,
        dk_c_file=None
    )
