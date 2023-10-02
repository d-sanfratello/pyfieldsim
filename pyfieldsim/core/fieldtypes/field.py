import h5py
import numpy as np

from pathlib import Path

from pyfieldsim.utils.metadata import read_metadata


class Field:
    @classmethod
    def from_sources(cls, sources_file, datatype='mass'):
        """
        Class method that creates a Field object from a sources' file.

        Parameters
        ----------
        sources_file: `string` or `Path`
            path to the file containing the sources.

        datatype: `string`
            type of data to be extracted, wether ``mass``, ``luminosity`` or
            ``magnitude``.

        Returns
        -------
        Field object
            a Field object containing the sources from the file.
        """
        sources_file = Path(sources_file).with_suffix('.h5')

        with h5py.File(sources_file, 'r') as f:
            coords = np.asarray(f['coords'])
            data = np.asarray(f[datatype])

        metadata = read_metadata(sources_file)

        field = np.zeros(
            shape=metadata['ext_shape']
        )
        for c, l in zip(coords, data):
            field[c[0], c[1]] = l

        return Field(field, sources_file=sources_file, seed=metadata['seed'])

    @classmethod
    def from_field(cls, field_file):
        """
        Class method that creates a Field object from a field file.

        Parameters
        ----------
        field_file: `string` or `Path`
            path to the file containing the existing field.

        Returns
        -------
        Field object
            a Field object containing the saved field from the file.
        """
        field_file = Path(field_file).with_suffix('.h5')

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
                 integration_time=None,
                 bkgnd_file=None,
                 psf_file=None,
                 gain_map_file=None,
                 dk_c_file=None):
        self.field = field
        self._seed = seed

        for k in self.__init__.__kwdefaults__:
            setattr(self, f'_{k}', locals()[k])

    def export_field(self, filename):
        """
        Method that exports the Field into an HDF5 file, for later analysis.

        Parameters
        ----------
        filename: `string` of `Path`
            The name of the destination file.
        """
        filename = Path(filename)
        if filename.suffixes[-1].lower() != '.h5':
            filename = filename.with_suffix('.h5')

        # TODO: export metadata as in other files.
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
        """
        Multiplication of two Field instances.
        """
        if isinstance(other, Field):
            new_field = other.field * self.field
        else:
            new_field = self.field * other

        return new_field

    def __add__(self, other):
        """
        Sum of two Field instances.
        """
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
            elif k == '_integration_time':
                meta[k] = float(v)

        return meta
