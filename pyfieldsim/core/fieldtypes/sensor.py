import numpy as np

from pathlib import Path

from pyfieldsim.core.fieldtypes.field import Field
from pyfieldsim.utils.metadata import read_metadata, save_metadata


def create_gain_map(sources_file, mean_gain, rel_var):
    """
    Function that creates a gain map Field for a set of sources, with a
    given mean gain and relative variation.

    Parameters
    ----------
    sources_file: `string` or `Path`
        The HDF5 file containing the generated sources.
    mean_gain: `number`
        The mean of the distribution from which to extract the gain values
        for each pixel.
    rel_var: `number`
        The relative standard deviation to create a gaussian gain map.

    Returns
    -------
    `Field` instance
        A `Field` object containing the generated gain map of the sensor for
        the set of sources.
    """
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
    gain_map = np.where(gain_map > 1, 1, gain_map)

    return Field(
        gain_map,
        seed=sim_meta['seed'],
        sources_file=str(sources_file),
        bkgnd_file='B' + str(sources_file)[1:],
        psf_file=None,
        gain_map_file=None,
        dk_c_file=None
    )


def create_dark_current(sources_file,
                        b_fraction=0.1,
                        delta_time=1):
    """
    Function that creates a dark current Field for a set of sources, with a
    given fraction of the background image and the integration time
    parameter.

    Parameters
    ----------
    sources_file: `string` or `Path`
        The HDF5 file containing the generated sources.
    b_fraction: `number`
        The fraction of the background mean to use as a mean for the
        generation of the dark current counts.
    delta_time: `number`
        The integration time parameter to convert the values from readings
        per unit time in actual counts.

    Returns
    -------
    `Field` instance
        A `Field` object containing the generated dark current field of the
        sensor.
    """
    sources_file = Path(sources_file)
    background_filename = 'B' + sources_file.name[1:]
    background_file = Field.from_field(background_filename)

    sim_meta = read_metadata(sources_file)

    rng = np.random.default_rng(seed=sim_meta['seed'])

    # Generation of the dark current mean from the b_fraction of the
    # background mean.
    b_mean = np.mean(background_file.field)
    dk_c_mean = b_mean * b_fraction * delta_time

    # Generating the dark current field for the simulated CCD.
    dk_c_field = rng.poisson(
        lam=dk_c_mean,
        size=sim_meta['ext_shape']
    )

    metadata = {
        "mean": dk_c_mean,
    }

    save_metadata(
        metadata=metadata,
        filename='C' + sources_file.name[1:]
    )

    dk_c_field = np.where(dk_c_field < 0, 0, dk_c_field).astype(int)

    return Field(
        dk_c_field,
        seed=sim_meta['seed'],
        sources_file=str(sources_file),
        bkgnd_file='B' + str(sources_file)[1:],
        psf_file=None,
        gain_map_file=None,
        dk_c_file=None
    )


def create_bad_pixels(sources_file, dc_field):
    """
    Function that modifies a dark current Field for a set of sources,
    to include a handful of hot pixels.

    Parameters
    ----------
    sources_file: `string` or `Path`
        The HDF5 file containing the generated sources.
    dc_field: `Field`
        A `Field` instance containing the simulated dark current of the sensor.

    Returns
    -------
    `Field` instance
        A `Field` object containing the dark current field but updated with
        some hot pixels.
    """
    sim_meta = read_metadata(sources_file)
    rng = np.random.default_rng(seed=sim_meta['seed'])
    shape = dc_field.field.shape

    n_bad_pixels = rng.integers(low=0, high=shape[0]*shape[1] // 500,
                                endpoint=False)

    idx_bad_pixels = rng.integers(low=0, high=shape[0]*shape[1],
                                  endpoint=False, size=n_bad_pixels)
    coords_bad_pixels = np.unravel_index(idx_bad_pixels, shape=shape)

    dc_field.field[coords_bad_pixels] *= 10

    return dc_field
