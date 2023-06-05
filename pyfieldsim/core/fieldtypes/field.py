import h5py
import numpy as np

from pathlib import Path

from pyfieldsim.utils.metadata import read_metadata


class Field:
    @classmethod
    def from_sources(cls, sources_file):
        sources_file = Path(sources_file)

        with h5py.File(sources_file, 'r') as f:
            coords = np.asarray(f['coords'])
            lum = np.asarray(f['luminosity'])

        metadata = read_metadata(
            Path(sources_file.stem + '_meta').with_suffix('.h5')
        )

        field = np.zeros(
            shape=metadata['ext_shape']
        )
        for c, l in zip(coords, lum):
            field[c[0], c[1]] = l

        return Field(field, sources_file=sources_file, seed=metadata['seed'])

    @classmethod
    def from_field(cls, field_file):
        field_file = Path(field_file)

        with h5py.File(field_file, 'r') as f:
            field = np.asarray(f['field'])

            metadata = {
                k: v for k, v in f.attrs.items()
            }

        return Field(
            field, **{k[1:]: v for k, v in metadata.items() if v is not None}
        )

    def __init__(self, field, *,
                 seed=None,
                 sources_file=None,
                 ph_noise_file=None,
                 bkgnd_file=None,
                 psf_file=None,
                 gain_map_file=None,
                 dk_c_file=None):
        self.field = field
        self._seed = seed

        for k in self.__init__.__kwdefaults__:
            setattr(self, f'_{k}', locals()[k])

    def export_field(self, filename):
        filename = Path(filename)
        if filename.suffix.lower() != '.h5':
            filename = filename.with_suffix('.h5')

        with h5py.File(filename, "w") as file:
            field = file.create_dataset(
                'field',
                shape=self.field.shape,
                dtype=np.float
            )
            field[0:] = self.field

            for k, v in self.metadata.items():
                file.attrs[k] = str(v)

    def __mul__(self, other):
        if isinstance(other, Field):
            new_field = other.field * self.field
        else:
            new_field = self.field * other

        return new_field

    def __add__(self, other):
        new_field = self.field + other.field

        return new_field

    @property
    def metadata(self):
        meta = {
            k: str(v) for k, v in self.__dict__.items() if k != 'field'
        }
        for k, v in meta.items():
            if k == '_seed':
                meta[k] = int(v)
            elif v == 'None':
                meta[k] = None

        return meta
