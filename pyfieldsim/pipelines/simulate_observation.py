#         # Simulation advancement at each step after initialization.
#         self.add_photon_noise(
#             delta_time=delta_time,
#             force=True,
#             multiply=False
#         )
#         self.add_background(
#             fluct=background_fluct,
#             snr=snr,
#             rel_var=bgnd_rel_var,
#             force=True
#         )
#         self.apply_psf(kernel=kernel, force=True)
#
#         self.create_gain_map(
#             mean_gain=gain_mean,
#             rel_var=gain_rel_var,
#             force=True
#         )
#         self.create_dark_current(
#             b_fraction=dk_c_fraction,
#             rel_var=dk_c_rel_var,
#             dk_c=dk_c,
#             force=True
#         )
#
#         self.recorded_field = self.gain_map \
#           * self.w_psf_field \
#           + self.dark_current

import argparse as ag
import os
import scipy.signal as scipysig

from pathlib import Path

from pyfieldsim.core.fieldtypes.field import Field
from pyfieldsim.core.psf import GaussKernel


def main():
    parser = ag.ArgumentParser(
        prog='fs-simulate-observation',
        description='',
    )
    parser.add_argument('sources')
    parser.add_argument('-k', '--psf-width', type=float,
                        dest='psf_width', default=3,
                        help='')
    parser.add_argument("-o", "--output",
                        dest='out_folder', default=None,
                        help="")

    args = parser.parse_args()

    if args.out_folder is None:
        out_folder = Path(os.getcwd())
    else:
        out_folder = Path(args.out_folder)

    sources_file = Path(args.sources)
    ph_noise_contaminated_filename = 'P' + sources_file.name[1:]
    background_filename = 'B' + sources_file.name[1:]
    gain_map_filename = 'G' + sources_file.name[1:]
    dark_current_filename = 'C' + sources_file.name[1:]

    observation_filename = 'O' + sources_file.name[1:]
    observation_filename = out_folder.joinpath(observation_filename)

    ph_noise_contaminated = Field.from_field(ph_noise_contaminated_filename)
    background = Field.from_field(background_filename)
    gain_map = Field.from_field(gain_map_filename)
    dark_current = Field.from_field(dark_current_filename)

    psf = GaussKernel(sigma=args.psf_width)

    noise_contaminated = ph_noise_contaminated + background
    field_w_psf = scipysig.convolve2d(
        noise_contaminated,
        psf.kernel,
        mode='same'
    )
    field_w_psf = Field(
        field=field_w_psf,
        seed=ph_noise_contaminated.metadata['_seed'],
        sources_file=sources_file,
        ph_noise_file=ph_noise_contaminated_filename,
        bkgnd_file=background_filename,
    )
    field_w_gain_map = Field(
        gain_map * field_w_psf,
        seed=ph_noise_contaminated.metadata['_seed'],
        sources_file=sources_file,
        ph_noise_file=ph_noise_contaminated_filename,
        bkgnd_file=background_filename,
        gain_map_file=gain_map_filename,
    )
    field_w_dk_current = Field(
        field_w_gain_map + dark_current,
        seed=ph_noise_contaminated.metadata['_seed'],
        sources_file=sources_file,
        ph_noise_file=ph_noise_contaminated_filename,
        bkgnd_file=background_filename,
        gain_map_file=gain_map_filename,
        dk_c_file=dark_current_filename
    )

    field_w_dk_current.export_field(filename=observation_filename)


if __name__ == "__main__":
    main()


# if __name__ == "__main__":
#     # -------------------------------------------------------------------------
#     # Complete operation
#     # -------------------------------------------------------------------------
#
#     # counting single stars
#     stars, coords = observation.count_single_stars()
#     print(f'{len(stars)} single stars')
#
#     psf = GaussKernel(sigma=3)
#
#     field.record_field(kernel=psf,
#                        delta_time=1000, snr=10, bgnd_rel_var=0.05,
#                        gain_mean=1, gain_rel_var=0.01,
#                        dk_c_fraction=0.1, dk_c_rel_var=0.01, dk_c=1,
#                        force=True)
#
#     field.save_field(name='stars')
#     field.show_field('exposure')
