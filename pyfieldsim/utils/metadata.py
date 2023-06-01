import h5py

from pathlib import Path


def read_metadata(sources_file):
    sources_file = Path(sources_file)

    with h5py.File(sources_file, 'r') as f:
        metadata = {
            k: v for k, v in f.attrs.items()
        }

    return metadata
