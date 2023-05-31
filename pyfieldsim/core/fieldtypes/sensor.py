import numpy as np

from pyfieldsim.core.fieldtypes.field import Field


def create_gain_map(sources_file, mean_gain, rel_var):
    sources_file = Field.from_field(sources_file)

    rng = np.random.default_rng(seed=sources_file.metadata['_seed'])

    # Generating the gain map field to be multiply the read data.
    gain_map = rng.normal(
        loc=mean_gain, scale=rel_var * mean_gain,
        size=sources_file.metadata['ext_shape']
    )

    gain_map = np.where(gain_map < 0, 0, gain_map)

    return Field(
        gain_map,
        seed=sources_file.metadata['_seed'],
        sources_file=str(sources_file),
        ph_noise_file='P' + str(sources_file)[1:],
        bkgnd_file='B' + str(sources_file)[1:],
        psf_file=None,
        gain_map_file=None,
        dk_c_file=None
    )
