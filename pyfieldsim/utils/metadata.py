import h5py

from pathlib import Path


def read_metadata(sources_file):
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
    filename = Path(filename).with_suffix('')
    filename = Path(filename.name + '_meta.h5')
    if filename.suffix.lower() != '.h5':
        filename = filename.with_suffix('.h5')

    with h5py.File(filename, "w") as file:
        for k, v in metadata.items():
            file.attrs[k] = str(v)
