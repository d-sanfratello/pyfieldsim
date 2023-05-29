import h5py
import numpy as np

from pathlib import Path


class Field:
    @classmethod
    def from_sources(cls, sources_file):
        sources_file = Path(sources_file)

        with h5py.File(sources_file, 'r') as f:
            coords = np.asarray(f['coords'])
            lum = np.asarray(f['luminosity'])

            metadata = {
                k: v for k, v in f.attrs.items()
            }

        field = np.zeros(
            shape=(metadata['ext_shape'], metadata['ext_shape'])
        )
        field[coords] = lum

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
            field, *{k: v for k, v in metadata.items() if v is not None}
        )

    def __init__(self, field,
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
                file.attrs['k'] = v

    @property
    def metadata(self):
        meta = {
            k: str(v) for k, v in self.__dict__ if k != 'field'
        }

        return meta
