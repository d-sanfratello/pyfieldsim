import numpy as np

from pathlib import Path

from pyfieldsim.core.fieldtypes.field import Field
from pyfieldsim.utils.metadata import read_metadata, save_metadata


def create_gain_map(sources_file, mean_gain, rel_var):
    sources_file = Path(sources_file)
    sim_meta = read_metadata(sources_file)

    rng = np.random.default_rng(seed=sim_meta['seed'])

    # Generating the gain map field to be multiply the read data.
    gain_map = rng.normal(
        loc=mean_gain, scale=rel_var * mean_gain,
        size=sim_meta['ext_shape']
    )

    metadata = {
        "mean": mean_gain,
        "rel_var": rel_var
    }

    save_metadata(
        metadata=metadata,
        filename='G' + sources_file.name[1:]
    )

    gain_map = np.where(gain_map < 0, 0, gain_map)

    return Field(
        gain_map,
        seed=sim_meta['seed'],
        sources_file=str(sources_file),
        ph_noise_file='P' + str(sources_file)[1:],
        bkgnd_file='B' + str(sources_file)[1:],
        psf_file=None,
        gain_map_file=None,
        dk_c_file=None
    )


def create_dark_current(sources_file,
                        b_fraction=0.1,
                        rel_var=0.01):
    sources_file = Path(sources_file)
    background_filename = 'B' + sources_file.name[1:]
    background_file = Field.from_field(background_filename)

    sim_meta = read_metadata(sources_file)

    rng = np.random.default_rng(seed=sim_meta['seed'])

    b_mean = np.mean(background_file.field)
    dk_c_mean = b_mean * b_fraction

    # Generating the dark current field for the simulated CCD.
    dk_c_field = rng.normal(
        loc=dk_c_mean, scale=rel_var * dk_c_mean,
        size=sim_meta['ext_shape']
    )

    metadata = {
        "mean": dk_c_mean,
        "rel_var": rel_var
    }

    save_metadata(
        metadata=metadata,
        filename='C' + sources_file.name[1:]
    )

    dk_c_field = np.where(dk_c_field < 0, 0, dk_c_field)
    dk_c_field = np.round(dk_c_field, dtype=int)

    return Field(
        dk_c_field,
        seed=sim_meta['seed'],
        sources_file=str(sources_file),
        ph_noise_file='P' + str(sources_file)[1:],
        bkgnd_file='B' + str(sources_file)[1:],
        psf_file=None,
        gain_map_file=None,
        dk_c_file=None
    )
