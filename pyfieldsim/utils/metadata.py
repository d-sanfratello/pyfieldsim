import h5py

from pathlib import Path


def read_metadata(sources_file):
    """
    Function to read the metadata of a file.

    Parameters
    ----------
    sources_file: `string` or `Path`-like object
        File from which metadata are read.

    Returns
    -------
    metadata: `dict`
        A dictionary containing the recovered metadata.

    """
    filename = Path(sources_file).with_suffix('')
    if not filename.name.endswith('_meta'):
        filename = filename.parent.joinpath(
            filename.name + '_meta.h5'
        )
    else:
        filename = filename.parent.joinpath(
            filename.name + '.h5'
        )

    with h5py.File(filename, 'r') as f:
        metadata = {
            k: v for k, v in f.attrs.items()
        }

    for k, v in metadata.items():
        if k in ['seed']:
            metadata[k] = int(v)
        elif k in ['ext_shape', 'pad', 'shape',
                   'A', 'id', 'mu_x', 'mu_y', 'pos_errors', 'sigma',
                   'A_errors']:
            metadata[k] = eval(v)
        elif k.startswith('has'):
            metadata[k] = bool(v)
        elif k == 'hyp_psf':
            metadata[k] = v
        else:
            metadata[k] = float(v)

    return metadata


def save_metadata(metadata, filename):
    """
    Function to save the metadata of an object in a file.

    Parameters
    ----------
    metadata: `dict`
        The dictionary containing the metadata to be saved.
    filename: `string` or `Path`-like object
        The path to the metadata file.
    """
    filename = Path(filename).with_suffix('')
    filename = Path(filename.name + '_meta.h5')
    if filename.suffix.lower() != '.h5':
        filename = filename.with_suffix('.h5')

    with h5py.File(filename, "w") as file:
        for k, v in metadata.items():
            file.attrs[k] = str(v)
