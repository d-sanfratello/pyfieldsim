import h5py

from pathlib import Path


def read_metadata(sources_file):
    filename = Path(sources_file)

    if not filename.stem.endswith('_meta'):
        filename = Path(
            filename.stem + '_meta.h5'
        )

    with h5py.File(filename, 'r') as f:
        metadata = {
            k: v for k, v in f.attrs.items()
        }

    for k, v in metadata.items():
        if k in ['seed']:
            metadata[k] = int(v)
        elif k in ['ext_shape', 'pad', 'shape']:
            metadata[k] = eval(v)
        else:
            metadata[k] = float(v)

    return metadata


def save_metadata(metadata, filename):
    filename = Path(filename)
    filename = Path(filename.stem + '_meta').with_suffix('.h5')
    if filename.suffix.lower() != '.h5':
        filename = filename.with_suffix('.h5')

    with h5py.File(filename, "w") as file:
        for k, v in metadata.items():
            file.attrs[k] = str(v)
