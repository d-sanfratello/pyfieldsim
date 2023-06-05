import numpy as np

from pathlib import Path

from pyfieldsim.core.fieldtypes.field import Field
from pyfieldsim.utils.metadata import read_metadata, save_metadata


def ph_noise(sources_file, delta_time):
    sources_file = Path(sources_file)
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

    metadata = {
        "delta_time": delta_time
    }

    save_metadata(
        metadata=metadata,
        filename='P' + sources_file.name[1:]
    )

    return Field(
        w_ph_noise,
        seed=sources_field.metadata['_seed'],
        sources_file=str(sources_file),
        ph_noise_file=None,
        bkgnd_file=None,
        gain_map_file=None,
        dk_c_file=None
    )


def background(sources_file,
               snr,
               sigma=None, rel_var=None):
    sources_file = Path(sources_file)
    ph_noise_field_filename = 'P' + sources_file.name[1:]
    ph_noise_field_file = Field.from_field(ph_noise_field_filename)

    sim_meta = read_metadata(sources_file)

    bgst_star = np.max(ph_noise_field_file.field)

    mean_bgnd = bgst_star / snr
    if rel_var is not None:
        sigma = rel_var * mean_bgnd
    elif sigma is None:
        raise ValueError("Must pass a sigma or relative sigma value.")

    metadata = {
        "mean": mean_bgnd,
        "sigma": sigma,
        "rel_var": rel_var
    }

    save_metadata(
        metadata=metadata,
        filename='B' + sources_file.name[1:]
    )

    rng = np.random.default_rng(seed=sim_meta['_seed'])

    # Generating the background field to be added to the photon noise
    # contaminated field.
    bgnd = rng.normal(
        loc=mean_bgnd, scale=sigma,
        size=sim_meta['ext_shape'],
    )
    bgnd = np.where(
        bgnd < 0, 0, bgnd
    )

    return Field(
        bgnd,
        seed=sim_meta['_seed'],
        sources_file=str(sources_file),
        ph_noise_file=ph_noise_field_filename,
        bkgnd_file=None,
        gain_map_file=None,
        dk_c_file=None
    )
