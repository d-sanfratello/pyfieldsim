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
    w_ph_noise = rng.poisson(exposed_field.field)
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
