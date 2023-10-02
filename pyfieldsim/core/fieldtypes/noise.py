import warnings

import h5py.h5f
import numpy as np

from pathlib import Path

from pyfieldsim.core.fieldtypes.field import Field
from pyfieldsim.utils.metadata import read_metadata, save_metadata


def background(sources_file,
               snr,
               sigma=None, rel_var=None):
    """
    Function that creates a background Field for a set of sources, with a
    given SNR and relative variation (or sigma).

    Parameters
    ----------
    sources_file: `string` or `Path`
        The HDF5 file containing the generated sources.
    snr: `number`
        The S/N between the mean of the background and the brightest source
        in the field.
    sigma: `number`
        The standard deviation for generating a gaussian background. If
        rel_var is set, this is ignored.
    rel_var: `number`
        The relative standard deviation to create a gaussian background.

    Returns
    -------
    `Field` instance
        A `Field` object containing the generated background for the set of
        sources.
    """
    sources_file = Path(sources_file).with_suffix('.h5')
    field_file = Field.from_sources(sources_file)

    sim_meta = read_metadata(sources_file)

    bgst_star = np.max(field_file.field)

    mean_bgnd = bgst_star / snr
    if rel_var is not None:
        sigma = rel_var * mean_bgnd
    elif sigma is None:
        raise ValueError("Must pass a sigma or relative sigma value.")

    metadata = {
        "snr": snr,
        "mean": mean_bgnd,
        "sigma": sigma,
        "rel_var": rel_var
    }

    save_metadata(
        metadata=metadata,
        filename='B' + sources_file.name[1:]
    )

    rng = np.random.default_rng(seed=sim_meta['seed'])

    # Generating the background field to be added to the photon noise
    # contaminated field.
    bgnd = rng.normal(
        loc=mean_bgnd, scale=sigma,
        size=sim_meta['ext_shape'],
    )
    # If the generated background is negative on a pixel, it is set to 0 for
    # the given pixel.
    bgnd = np.where(
        bgnd < 0, 0, bgnd
    )

    return Field(
        bgnd,
        seed=sim_meta['seed'],
        sources_file=str(sources_file),
        bkgnd_file=None,
        gain_map_file=None,
        dk_c_file=None
    )
