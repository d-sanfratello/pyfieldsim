import h5py

from pathlib import Path


def read_metadata(sources_file):
    sources_file = Path(sources_file)

    with h5py.File(sources_file, 'r') as f:
        metadata = {
            k: v for k, v in f.attrs.items()
        }

    return metadata


def save_metadata(metadata, filename):
    filename = Path(filename)
    filename.rename(filename.name + '_meta')
    if filename.suffix.lower() != '.h5':
        filename = filename.with_suffix('.h5')

    with h5py.File(filename, "w") as file:
        for k, v in metadata.items():
            file.attrs[k] = str(v)
